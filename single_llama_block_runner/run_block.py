#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Standalone synthetic single Llama block benchmark.

This runner intentionally avoids the vLLM engine, scheduler, tokenizer, model
loader, and KV cache. It uses vLLM's memory-efficient Triton prefill attention
kernel by default, with optional FlashAttention varlen and PyTorch SDPA paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


D = 4096
N_Q_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
FFN_INTER = 14336
EPS = 1e-5

@dataclass(frozen=True)
class LlamaCase:
    num_seqs: int
    seq_len: int
    strategy: str


CASES: dict[str, LlamaCase] = {
    "1k": LlamaCase(num_seqs=1, seq_len=1024, strategy="sequence_sharded"),
    "8k": LlamaCase(num_seqs=1, seq_len=8192, strategy="sequence_sharded"),
    "1kx8": LlamaCase(num_seqs=8, seq_len=1024, strategy="batch_sharded"),
    "1kx128": LlamaCase(num_seqs=128, seq_len=1024, strategy="batch_sharded"),
}


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    tp_size: int
    is_distributed: bool
    gpu_name: str
    capability: tuple[int, int]


@dataclass
class CaseBenchmarkResult:
    case: str
    strategy: str
    num_seqs: int
    seq_len: int
    local_num_seqs: int
    local_seq_len: int
    total_tokens: int
    local_tokens: int
    attention_backend: str
    forward_ms: list[float]
    mean_forward_ms: float
    std_forward_ms: float
    min_forward_ms: float
    max_forward_ms: float
    tokens_per_second: float
    peak_memory_gib: float
    output_shape: tuple[int, ...]
    reference_check: str | None = None


@dataclass
class BlockWeights:
    gam_a: torch.Tensor
    gam_f: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w_gu: torch.Tensor
    w_down: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one synthetic Llama block forward pass."
    )
    parser.add_argument("--case",
                        choices=[*CASES.keys(), "all"],
                        default="all")
    parser.add_argument("--tp-size", type=int, choices=[1, 4, 8], default=1)
    parser.add_argument(
        "--parallel-strategy",
        choices=["benchmark", "tensor-parallel"],
        default="benchmark",
        help=(
            "benchmark matches benchmarks/{jax,pytorch}-llama: replicated "
            "weights with batch- or sequence-sharded activations. "
            "tensor-parallel keeps the original vLLM-style tensor-parallel "
            "synthetic runner."
        ),
    )
    parser.add_argument("--dtype",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        default="auto")
    parser.add_argument("--attention-backend",
                        choices=[
                            "auto",
                            "triton-prefill",
                            "flash-attn-varlen",
                            "sdpa",
                        ],
                        default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--check-reference", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Path to write benchmark results as JSON (rank 0 only).",
    )
    return parser.parse_args()


def verify_cuda_kernels(device: torch.device) -> None:
    try:
        probe = torch.randn(1, device=device, dtype=torch.float16)
        del probe
    except torch.AcceleratorError as exc:
        raise RuntimeError(
            "PyTorch cannot run CUDA kernels on this GPU. Tesla V100 "
            "(compute capability 7.0) is not supported by torch 2.11; install "
            "torch 2.6+cu124 in the vllm-py312 conda env:\n"
            "  conda activate vllm-py312\n"
            "  pip install torch==2.6.0+cu124 torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu124\n"
            "The runner defaults to --attention-backend sdpa on V100. See "
            "single_llama_block_runner/README.md for details."
        ) from exc


def setup_runtime(tp_size: int) -> Runtime:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if tp_size == 1:
        rank = 0
        local_rank = 0
        world_size = 1
        is_distributed = False
    else:
        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise RuntimeError(
                f"tp_size={tp_size} requires torchrun; missing env vars: {missing}"
            )
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != tp_size:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} must match --tp-size={tp_size}."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        is_distributed = True

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    verify_cuda_kernels(device)
    capability = torch.cuda.get_device_capability(device)
    gpu_name = torch.cuda.get_device_name(device)
    return Runtime(
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        tp_size=tp_size,
        is_distributed=is_distributed,
        gpu_name=gpu_name,
        capability=capability,
    )


def resolve_dtype(dtype_arg: str, runtime: Runtime) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.float16
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "bfloat16":
        if runtime.capability[0] < 8:
            raise RuntimeError(
                "bfloat16 requires compute capability >= 8.0. "
                f"{runtime.gpu_name} has {runtime.capability[0]}."
                f"{runtime.capability[1]}; use --dtype float16 on V100."
            )
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def resolve_attention_backend(backend_arg: str, runtime: Runtime) -> str:
    if backend_arg == "auto":
        if runtime.capability < (7, 5):
            return "sdpa"
        if "A100" in runtime.gpu_name and runtime.capability >= (8, 0):
            from vllm.v1.attention.backends.fa_utils import (
                is_flash_attn_varlen_func_available,
            )

            if is_flash_attn_varlen_func_available():
                return "flash-attn-varlen"
        return "triton-prefill"
    if backend_arg == "triton-prefill" and runtime.capability < (7, 5):
        raise RuntimeError(
            "triton-prefill requires vLLM kernels built for this GPU and "
            "PyTorch with Volta (sm_70) support. On V100, use the default "
            "auto backend (sdpa) or pass --attention-backend sdpa."
        )
    if backend_arg == "flash-attn-varlen":
        if runtime.capability[0] < 8:
            raise RuntimeError(
                "flash-attn-varlen requires compute capability >= 8.0 in "
                "this vLLM build; use --attention-backend triton-prefill "
                "on V100."
            )
        from vllm.v1.attention.backends.fa_utils import (
            is_flash_attn_varlen_func_available,
        )

        if not is_flash_attn_varlen_func_available():
            raise RuntimeError(
                "flash-attn-varlen was requested, but vLLM's "
                "flash_attn_varlen_func is not available in this environment."
            )
    return backend_arg


def rank0_print(runtime: Runtime, message: str) -> None:
    if runtime.rank == 0:
        print(message, flush=True)


def all_reduce_sum(x: torch.Tensor, runtime: Runtime) -> torch.Tensor:
    if runtime.is_distributed:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def all_gather_sequence_heads(local: torch.Tensor, runtime: Runtime,
                              num_seqs: int) -> torch.Tensor:
    if not runtime.is_distributed:
        return local
    local_seq_len = local.shape[0] // num_seqs
    local_4d = local.view(num_seqs, local_seq_len, local.shape[1],
                          local.shape[2]).contiguous()
    gathered = [torch.empty_like(local_4d) for _ in range(runtime.world_size)]
    dist.all_gather(gathered, local_4d)
    return torch.cat(gathered, dim=1).reshape(num_seqs * local_seq_len *
                                              runtime.world_size,
                                              local.shape[1], local.shape[2])


def make_randn(shape: tuple[int, ...],
               *,
               device: torch.device,
               dtype: torch.dtype,
               seed: int,
               scale: float) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(shape, device=device, dtype=dtype, generator=gen) * scale


def make_sequence_metadata(num_seqs: int, seq_len: int, position_offset: int,
                           device: torch.device
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      torch.Tensor]:
    positions = (
        torch.arange(position_offset,
                     position_offset + seq_len,
                     device=device,
                     dtype=torch.long).repeat(num_seqs)
    )
    b_start_loc = torch.arange(num_seqs, device=device,
                               dtype=torch.int32) * seq_len
    b_seq_len = torch.full((num_seqs, ), seq_len, device=device, dtype=torch.int32)
    cu_seqlens = torch.arange(num_seqs + 1, device=device,
                              dtype=torch.int32) * seq_len
    return positions, b_start_loc, b_seq_len, cu_seqlens


def make_case_inputs(case_name: str, runtime: Runtime, dtype: torch.dtype,
                     seed: int, parallel_strategy: str
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor, int, int, int, int]:
    case = CASES[case_name]
    num_seqs = case.num_seqs
    seq_len = case.seq_len
    total_tokens = num_seqs * seq_len
    x = make_randn((total_tokens, D),
                   device=runtime.device,
                   dtype=dtype,
                   seed=seed + 11,
                   scale=0.02)
    local_num_seqs = num_seqs
    local_seq_len = seq_len
    position_offset = 0

    if runtime.world_size > 1 and parallel_strategy == "benchmark":
        x_3d = x.view(num_seqs, seq_len, D)
        if case.strategy == "batch_sharded":
            if num_seqs % runtime.world_size != 0:
                raise RuntimeError(
                    f"{case_name} num_seqs={num_seqs} must be divisible by "
                    f"world_size={runtime.world_size} for batch_sharded."
                )
            local_num_seqs = num_seqs // runtime.world_size
            start = runtime.rank * local_num_seqs
            x = x_3d.narrow(0, start, local_num_seqs).contiguous().view(
                local_num_seqs * seq_len, D)
        else:
            if seq_len % runtime.world_size != 0:
                raise RuntimeError(
                    f"{case_name} seq_len={seq_len} must be divisible by "
                    f"world_size={runtime.world_size} for sequence_sharded."
                )
            local_seq_len = seq_len // runtime.world_size
            position_offset = runtime.rank * local_seq_len
            x = x_3d.narrow(1, position_offset, local_seq_len).contiguous().view(
                num_seqs * local_seq_len, D)

    positions, b_start_loc, b_seq_len, cu_seqlens = make_sequence_metadata(
        local_num_seqs, local_seq_len, position_offset, runtime.device)
    return (x, positions, b_start_loc, b_seq_len, cu_seqlens, local_num_seqs,
            local_seq_len, num_seqs, seq_len)


def make_weights(runtime: Runtime, dtype: torch.dtype, parallel_strategy: str,
                 seed: int) -> tuple[BlockWeights, int, int, int]:
    shard_count = runtime.tp_size if parallel_strategy == "tensor-parallel" else 1
    if N_Q_HEADS % shard_count != 0:
        raise RuntimeError("N_Q_HEADS must be divisible by tp_size.")
    if N_KV_HEADS % shard_count != 0:
        raise RuntimeError("N_KV_HEADS must be divisible by tp_size.")
    if FFN_INTER % shard_count != 0:
        raise RuntimeError("FFN_INTER must be divisible by tp_size.")

    q_heads_local = N_Q_HEADS // shard_count
    kv_heads_local = N_KV_HEADS // shard_count
    ffn_inter_local = FFN_INTER // shard_count

    rank_seed = runtime.rank if parallel_strategy == "tensor-parallel" else 0
    shard_seed = seed + 100000 + rank_seed * 1000
    device = runtime.device

    weights = BlockWeights(
        gam_a=torch.ones((D, ), device=device, dtype=dtype),
        gam_f=torch.ones((D, ), device=device, dtype=dtype),
        wq=make_randn((D, q_heads_local, HEAD_DIM),
                      device=device,
                      dtype=dtype,
                      seed=shard_seed + 1,
                      scale=1.0 / math.sqrt(D)),
        wk=make_randn((D, kv_heads_local, HEAD_DIM),
                      device=device,
                      dtype=dtype,
                      seed=shard_seed + 2,
                      scale=1.0 / math.sqrt(D)),
        wv=make_randn((D, kv_heads_local, HEAD_DIM),
                      device=device,
                      dtype=dtype,
                      seed=shard_seed + 3,
                      scale=1.0 / math.sqrt(D)),
        wo=make_randn((q_heads_local, HEAD_DIM, D),
                      device=device,
                      dtype=dtype,
                      seed=shard_seed + 4,
                      scale=1.0 / math.sqrt(N_Q_HEADS * HEAD_DIM)),
        w_gu=make_randn((D, 2 * ffn_inter_local),
                        device=device,
                        dtype=dtype,
                        seed=shard_seed + 5,
                        scale=1.0 / math.sqrt(D)),
        w_down=make_randn((ffn_inter_local, D),
                          device=device,
                          dtype=dtype,
                          seed=shard_seed + 6,
                          scale=1.0 / math.sqrt(FFN_INTER)),
    )
    return weights, q_heads_local, kv_heads_local, ffn_inter_local


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    out = x.float() * torch.rsqrt(variance + eps)
    return (out * weight.float()).to(x.dtype)


def rope_cache(seq_len: int, device: torch.device,
               dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        10000**(torch.arange(0, HEAD_DIM, 2, device=device,
                             dtype=torch.float32) / HEAD_DIM))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope_neox(x: torch.Tensor, positions: torch.Tensor,
                    cos_cache: torch.Tensor,
                    sin_cache: torch.Tensor) -> torch.Tensor:
    cos = cos_cache.index_select(0, positions).unsqueeze(1)
    sin = sin_cache.index_select(0, positions).unsqueeze(1)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat((out1, out2), dim=-1)


def attention_triton_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             b_start_loc: torch.Tensor,
                             b_seq_len: torch.Tensor,
                             seq_len: int) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

    out = torch.empty_like(q)
    context_attention_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        o=out,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=seq_len,
        is_causal=True,
        softmax_scale=HEAD_DIM**-0.5,
    )
    return out


def attention_flash_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           cu_seqlens: torch.Tensor,
                           seq_len: int) -> torch.Tensor:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    out = flash_attn_varlen_func(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        dropout_p=0.0,
        softmax_scale=HEAD_DIM**-0.5,
        causal=True,
        fa_version=2,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   num_seqs: int, seq_len: int) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if q_heads % kv_heads != 0:
        raise RuntimeError(
            f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}.")

    q4 = q.view(num_seqs, seq_len, q_heads, HEAD_DIM).permute(0, 2, 1, 3)
    k4 = k.view(num_seqs, seq_len, kv_heads, HEAD_DIM).permute(0, 2, 1, 3)
    v4 = v.view(num_seqs, seq_len, kv_heads, HEAD_DIM).permute(0, 2, 1, 3)
    repeat = q_heads // kv_heads
    if repeat != 1:
        k4 = k4.repeat_interleave(repeat, dim=1)
        v4 = v4.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(q4,
                                          k4,
                                          v4,
                                          dropout_p=0.0,
                                          is_causal=True)
    return out.permute(0, 2, 1, 3).reshape(num_seqs * seq_len, q_heads, HEAD_DIM)


def attention_sdpa_sequence_sharded(q: torch.Tensor, k: torch.Tensor,
                                    v: torch.Tensor, num_seqs: int,
                                    local_seq_len: int, full_seq_len: int,
                                    query_offset: int) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if q_heads % kv_heads != 0:
        raise RuntimeError(
            f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}.")

    q4 = q.view(num_seqs, local_seq_len, q_heads, HEAD_DIM).permute(0, 2, 1, 3)
    k4 = k.view(num_seqs, full_seq_len, kv_heads, HEAD_DIM).permute(0, 2, 1, 3)
    v4 = v.view(num_seqs, full_seq_len, kv_heads, HEAD_DIM).permute(0, 2, 1, 3)
    repeat = q_heads // kv_heads
    if repeat != 1:
        k4 = k4.repeat_interleave(repeat, dim=1)
        v4 = v4.repeat_interleave(repeat, dim=1)

    q_pos = torch.arange(query_offset,
                         query_offset + local_seq_len,
                         device=q.device)[:, None]
    k_pos = torch.arange(full_seq_len, device=q.device)[None, :]
    attn_mask = (k_pos <= q_pos)[None, None, :, :]
    out = F.scaled_dot_product_attention(q4,
                                          k4,
                                          v4,
                                          attn_mask=attn_mask,
                                          dropout_p=0.0,
                                          is_causal=False)
    return out.permute(0, 2, 1, 3).reshape(num_seqs * local_seq_len, q_heads,
                                           HEAD_DIM)


def packed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     b_start_loc: torch.Tensor, b_seq_len: torch.Tensor,
                     cu_seqlens: torch.Tensor, num_seqs: int, seq_len: int,
                     backend: str) -> torch.Tensor:
    if backend == "triton-prefill":
        return attention_triton_prefill(q, k, v, b_start_loc, b_seq_len, seq_len)
    if backend == "flash-attn-varlen":
        return attention_flash_varlen(q, k, v, cu_seqlens, seq_len)
    if backend == "sdpa":
        return attention_sdpa(q, k, v, num_seqs, seq_len)
    raise ValueError(f"Unsupported attention backend: {backend}")


def llama_block_forward(x: torch.Tensor, positions: torch.Tensor,
                        b_start_loc: torch.Tensor, b_seq_len: torch.Tensor,
                        cu_seqlens: torch.Tensor, cos_cache: torch.Tensor,
                        sin_cache: torch.Tensor, num_seqs: int, seq_len: int,
                        global_num_seqs: int, global_seq_len: int,
                        weights: BlockWeights, runtime: Runtime, backend: str,
                        case_strategy: str, parallel_strategy: str
                        ) -> torch.Tensor:
    norm_x = rmsnorm(x, weights.gam_a, EPS)

    q_3d = torch.einsum("sd,dhm->shm", norm_x, weights.wq)
    k_3d = torch.einsum("sd,dhm->shm", norm_x, weights.wk)
    v_3d = torch.einsum("sd,dhm->shm", norm_x, weights.wv)

    q_r = apply_rope_neox(q_3d, positions, cos_cache, sin_cache)
    k_r = apply_rope_neox(k_3d, positions, cos_cache, sin_cache)

    if (parallel_strategy == "benchmark" and runtime.world_size > 1
            and case_strategy == "sequence_sharded"):
        k_full = all_gather_sequence_heads(k_r, runtime, global_num_seqs)
        v_full = all_gather_sequence_heads(v_3d, runtime, global_num_seqs)
        query_offset = runtime.rank * seq_len
        attn_ctx = attention_sdpa_sequence_sharded(q_r, k_full, v_full,
                                                   global_num_seqs, seq_len,
                                                   global_seq_len,
                                                   query_offset)
    else:
        attn_ctx = packed_attention(q_r, k_r, v_3d, b_start_loc, b_seq_len,
                                    cu_seqlens, num_seqs, seq_len, backend)
    attn_out_partial = torch.einsum("shm,hmd->sd", attn_ctx, weights.wo)
    attn_out = (all_reduce_sum(attn_out_partial, runtime)
                if parallel_strategy == "tensor-parallel" else attn_out_partial)

    x_after = x + attn_out
    norm2 = rmsnorm(x_after, weights.gam_f, EPS)

    gate_up = norm2 @ weights.w_gu
    gate, up = gate_up.chunk(2, dim=-1)
    intermed = F.silu(gate) * up

    ffn_out_partial = intermed @ weights.w_down
    ffn_out = (all_reduce_sum(ffn_out_partial, runtime)
               if parallel_strategy == "tensor-parallel" else ffn_out_partial)
    return x_after + ffn_out


def synchronize(runtime: Runtime) -> None:
    torch.cuda.synchronize(runtime.device)
    if runtime.is_distributed:
        dist.barrier()


def benchmark_case(case_name: str, runtime: Runtime, dtype: torch.dtype,
                   backend: str,
                   args: argparse.Namespace) -> CaseBenchmarkResult:
    case = CASES[case_name]
    if backend == "sdpa" and case_name == "1kx128":
        rank0_print(
            runtime,
            "warning: sdpa on 1kx128 may require a very large attention "
            "workspace and can OOM; triton-prefill is recommended.",
        )

    effective_backend = backend
    if (args.parallel_strategy == "benchmark" and runtime.world_size > 1
            and case.strategy == "sequence_sharded"):
        effective_backend = "sdpa"
    if effective_backend != backend:
        rank0_print(
            runtime,
            f"case={case_name} strategy=sequence_sharded uses sdpa attention "
            "for offset-aware causal masking after K/V gather.",
        )

    (x, positions, b_start_loc, b_seq_len, cu_seqlens, num_seqs, seq_len,
     global_num_seqs, global_seq_len) = make_case_inputs(
         case_name, runtime, dtype, args.seed, args.parallel_strategy)
    cos_cache, sin_cache = rope_cache(global_seq_len, runtime.device, dtype)
    weights, _, _, _ = make_weights(runtime, dtype, args.parallel_strategy,
                                    args.seed)
    total_tokens = global_num_seqs * global_seq_len
    local_tokens = num_seqs * seq_len

    def run_once(selected_backend: str) -> torch.Tensor:
        return llama_block_forward(x, positions, b_start_loc, b_seq_len,
                                   cu_seqlens, cos_cache, sin_cache,
                                   num_seqs, seq_len, global_num_seqs,
                                   global_seq_len, weights, runtime,
                                   selected_backend, case.strategy,
                                   args.parallel_strategy)

    with torch.inference_mode():
        output = None
        for _ in range(args.warmup_iters):
            output = run_once(effective_backend)
        synchronize(runtime)

        torch.cuda.reset_peak_memory_stats(runtime.device)
        forward_ms: list[float] = []
        for _ in range(args.benchmark_iters):
            synchronize(runtime)
            start = time.perf_counter()
            output = run_once(effective_backend)
            synchronize(runtime)
            forward_ms.append((time.perf_counter() - start) * 1000.0)

        assert output is not None
        mean_ms = statistics.mean(forward_ms)
        std_ms = statistics.stdev(forward_ms) if len(forward_ms) > 1 else 0.0
        min_ms = min(forward_ms)
        max_ms = max(forward_ms)
        tok_per_s = total_tokens / (mean_ms / 1000.0)
        peak_gib = torch.cuda.max_memory_allocated(runtime.device) / (1024**3)
        reference_check: str | None = None

        if args.check_reference:
            if case_name == "1kx128":
                reference_check = "skipped:sdpa_may_oom"
                rank0_print(
                    runtime,
                    "reference_check=skipped case=1kx128 reason=sdpa_may_oom",
                )
            elif effective_backend == "sdpa":
                reference_check = "skipped:backend_is_sdpa"
                rank0_print(
                    runtime,
                    f"reference_check=skipped case={case_name} "
                    "reason=backend_is_sdpa",
                )
            else:
                ref_output = run_once("sdpa")
                torch.testing.assert_close(output,
                                           ref_output,
                                           rtol=5e-2,
                                           atol=5e-2)
                reference_check = "passed"
                rank0_print(runtime, f"reference_check=passed case={case_name}")

    result = CaseBenchmarkResult(
        case=case_name,
        strategy=("tensor_parallel" if args.parallel_strategy == "tensor-parallel"
                  else case.strategy),
        num_seqs=num_seqs,
        seq_len=seq_len,
        local_num_seqs=num_seqs,
        local_seq_len=seq_len,
        total_tokens=total_tokens,
        local_tokens=local_tokens,
        attention_backend=effective_backend,
        forward_ms=forward_ms,
        mean_forward_ms=mean_ms,
        std_forward_ms=std_ms,
        min_forward_ms=min_ms,
        max_forward_ms=max_ms,
        tokens_per_second=tok_per_s,
        peak_memory_gib=peak_gib,
        output_shape=tuple(output.shape),
        reference_check=reference_check,
    )

    rank0_print(
        runtime,
        " ".join([
            f"case={case_name}",
            f"strategy={result.strategy}",
            f"tp_size={runtime.tp_size}",
            f"dtype={str(dtype).replace('torch.', '')}",
            f"global_num_seqs={global_num_seqs}",
            f"global_seq_len={global_seq_len}",
            f"local_num_seqs={num_seqs}",
            f"local_seq_len={seq_len}",
            f"total_tokens={total_tokens}",
            f"local_tokens={local_tokens}",
            f"attention_backend={effective_backend}",
            f"mean_forward_ms={mean_ms:.3f}",
            f"std_forward_ms={std_ms:.3f}",
            f"min_forward_ms={min_ms:.3f}",
            f"max_forward_ms={max_ms:.3f}",
            f"tokens_per_second={tok_per_s:.2f}",
            f"peak_memory_gib={peak_gib:.2f}",
            f"output_shape={tuple(output.shape)}",
        ]),
    )

    del x, positions, b_start_loc, b_seq_len, cu_seqlens, cos_cache, sin_cache
    del weights, output
    torch.cuda.empty_cache()
    return result


def write_results_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build_results_payload(
    *,
    runtime: Runtime,
    dtype: torch.dtype,
    backend: str,
    args: argparse.Namespace,
    case_results: list[CaseBenchmarkResult],
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": runtime.gpu_name,
        "capability": f"{runtime.capability[0]}.{runtime.capability[1]}",
        "tp_size": runtime.tp_size,
        "world_size": runtime.world_size,
        "parallel_strategy": args.parallel_strategy,
        "dtype": str(dtype).replace("torch.", ""),
        "requested_attention_backend": backend,
        "seed": args.seed,
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
        "check_reference": args.check_reference,
        "cases": [asdict(result) for result in case_results],
    }


def main() -> None:
    args = parse_args()
    runtime = setup_runtime(args.tp_size)
    dtype = resolve_dtype(args.dtype, runtime)
    backend = resolve_attention_backend(args.attention_backend, runtime)

    rank0_print(
        runtime,
        " ".join([
            f"gpu={runtime.gpu_name!r}",
            f"capability={runtime.capability[0]}.{runtime.capability[1]}",
            f"tp_size={runtime.tp_size}",
            f"parallel_strategy={args.parallel_strategy}",
            f"dtype={str(dtype).replace('torch.', '')}",
            f"requested_attention_backend={backend}",
        ]),
    )

    case_names = list(CASES) if args.case == "all" else [args.case]
    case_results: list[CaseBenchmarkResult] = []
    try:
        for case_name in case_names:
            case_results.append(
                benchmark_case(case_name, runtime, dtype, backend, args))
    finally:
        if runtime.rank == 0 and args.output_json:
            payload = build_results_payload(
                runtime=runtime,
                dtype=dtype,
                backend=backend,
                args=args,
                case_results=case_results,
            )
            write_results_json(args.output_json, payload)
            rank0_print(runtime, f"results_json={args.output_json}")
        if runtime.is_distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
