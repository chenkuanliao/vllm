#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Standalone synthetic single Llama block benchmark.

This runner intentionally avoids the vLLM engine, scheduler, tokenizer, model
loader, and KV cache. It uses vLLM's memory-efficient Triton prefill attention
kernel by default, with optional FlashAttention varlen and PyTorch SDPA paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    # Maximum sequence length; equals every sequence's length unless
    # sequence_lengths is set.
    seq_len: int
    strategy: str
    # Ordered per-sequence lengths for variable-length packed cases.
    sequence_lengths: tuple[int, ...] | None = None
    sequence_pattern: dict[str, Any] | None = None

    @property
    def lengths(self) -> tuple[int, ...]:
        if self.sequence_lengths is not None:
            return self.sequence_lengths
        return (self.seq_len, ) * self.num_seqs

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)


# Variable-length packed workloads shared with the Einsummable experiments
# (experiements/catalog.json and experiements/sequence_patterns.py in the
# vldb-experiments branch). Each keeps 1kx128's 128-sequence / 131072-token
# budget. The ordering is reproduced with the same portable sha256-sort-v1
# recipe and checked against the checksum Einsummable records in every
# sequence_pattern.json, so both systems run the identical sequence layout.
VARIABLE_LENGTH_SPECS: dict[str, dict[str, Any]] = {
    "var-moderate-128": {
        "algorithm": "sha256-sort-v1",
        "seed": 42,
        "length_counts": [[512, 32], [768, 32], [1280, 32], [1536, 32]],
        "batch_size": 128,
        "total_tokens": 131072,
        "sequence_lengths_sha256":
        "98788f83854bfb6ff2b6e5fedf6661ef3781bbd086b5f18e186447357edd8cba",
    },
    "var-strong-128": {
        "algorithm": "sha256-sort-v1",
        "seed": 42,
        "length_counts": [[256, 96], [1536, 16], [5120, 16]],
        "batch_size": 128,
        "total_tokens": 131072,
        "sequence_lengths_sha256":
        "b1206b5a6e85f731b6554817dd9dc7750727586e118db04966901adcb8c8b222",
    },
}


def sha256_sort_v1(seed: int, length_counts: list[list[int]]) -> list[int]:
    """Expand length_counts in order, then permute by SHA256("seed:i")."""
    expanded = [length for length, count in length_counts for _ in range(count)]
    order = sorted(range(len(expanded)),
                   key=lambda i: (hashlib.sha256(
                       f"{seed}:{i}".encode("ascii")).digest(), i))
    return [expanded[i] for i in order]


def sequence_lengths_sha256(lengths: list[int]) -> str:
    payload = json.dumps(lengths, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def make_variable_length_case(name: str) -> LlamaCase:
    spec = VARIABLE_LENGTH_SPECS[name]
    lengths = sha256_sort_v1(spec["seed"], spec["length_counts"])
    checksum = sequence_lengths_sha256(lengths)
    if (len(lengths) != spec["batch_size"]
            or sum(lengths) != spec["total_tokens"]
            or checksum != spec["sequence_lengths_sha256"]):
        raise RuntimeError(
            f"{name}: generated sequence pattern does not match the "
            f"Einsummable reference (sha256={checksum}).")
    return LlamaCase(num_seqs=len(lengths),
                     seq_len=max(lengths),
                     strategy="batch_sharded",
                     sequence_lengths=tuple(lengths),
                     sequence_pattern={
                         "algorithm": spec["algorithm"],
                         "seed": spec["seed"],
                         "length_counts": spec["length_counts"],
                         "sequence_lengths_sha256": checksum,
                     })


ORIGINAL_CASES = ("1k", "8k", "128k", "1kx8", "1kx128")
VARIABLE_LENGTH_CASES = tuple(VARIABLE_LENGTH_SPECS)

CASES: dict[str, LlamaCase] = {
    "1k": LlamaCase(num_seqs=1, seq_len=1024, strategy="sequence_sharded"),
    "8k": LlamaCase(num_seqs=1, seq_len=8192, strategy="sequence_sharded"),
    "128k": LlamaCase(num_seqs=1, seq_len=131072, strategy="sequence_sharded"),
    "1kx8": LlamaCase(num_seqs=8, seq_len=1024, strategy="batch_sharded"),
    "1kx128": LlamaCase(num_seqs=128, seq_len=1024, strategy="batch_sharded"),
    **{name: make_variable_length_case(name) for name in VARIABLE_LENGTH_CASES},
}

# Case groups accepted by --case in addition to individual case names. "all"
# keeps its original meaning; the variable-length cases are opt-in, matching
# the Einsummable setup where they have their own launchers.
CASE_GROUPS: dict[str, tuple[str, ...]] = {
    "all": ORIGINAL_CASES,
    "variable-length": VARIABLE_LENGTH_CASES,
}

SDPA_OOM_RISK_CASES = frozenset({"128k", "1kx128"})


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
    # How the timed forwards ran: "cuda_graph" replay, or "eager" only when
    # capture is not supported (capture_error says why).
    execution: str = "cuda_graph"
    capture_warmups: int = 0
    warmup_capture_ms: float = 0.0
    warmup_forward_ms: list[float] | None = None
    capture_error: str | None = None
    # Variable-length cases only: the exact ordered layout plus how the
    # batch_sharded split distributed tokens and attention work over ranks.
    sequence_pattern: dict[str, Any] | None = None


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
                        choices=[*CASES.keys(), *CASE_GROUPS.keys()],
                        default="all",
                        help=("A single case, 'all' for the original five "
                              "cases, or 'variable-length' for "
                              f"{', '.join(VARIABLE_LENGTH_CASES)}."))
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
    parser.add_argument("--warmup-iters", type=int, default=2,
                        help="Timed CUDA graph replays excluded from stats.")
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument(
        "--capture-warmups", type=int, default=1,
        help="Untimed eager forwards before CUDA graph capture (lazy init, "
        "Triton JIT, NCCL communicators).")
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


# Same input ranges as Einsummable's experiements/workloads.py (uniform fill).
X_RANGE = (-0.1, 0.1)
GAMMA_RANGE = (0.9, 1.1)
WEIGHT_RANGE = (-0.02, 0.02)


def make_uniform(shape: tuple[int, ...],
                 *,
                 device: torch.device,
                 dtype: torch.dtype,
                 seed: int,
                 value_range: tuple[float, float]) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    low, high = value_range
    return torch.empty(shape, device=device, dtype=dtype).uniform_(
        low, high, generator=gen)


def make_sequence_metadata(lengths: tuple[int, ...], position_offset: int,
                           device: torch.device
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      torch.Tensor]:
    """Packed-sequence metadata; RoPE positions restart for each sequence."""
    b_seq_len = torch.tensor(lengths, device=device, dtype=torch.int32)
    cu_seqlens = torch.zeros(len(lengths) + 1, device=device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(b_seq_len, dim=0)
    b_start_loc = cu_seqlens[:-1].clone()
    total = sum(lengths)
    positions = (torch.arange(total, device=device, dtype=torch.long) -
                 torch.repeat_interleave(b_start_loc.long(), b_seq_len.long(),
                                         output_size=total) + position_offset)
    return positions, b_start_loc, b_seq_len, cu_seqlens


def batch_shard_bounds(num_seqs: int, world_size: int, rank: int,
                       case_name: str) -> tuple[int, int]:
    if num_seqs % world_size != 0:
        raise RuntimeError(
            f"{case_name} num_seqs={num_seqs} must be divisible by "
            f"world_size={world_size} for batch_sharded.")
    local_num_seqs = num_seqs // world_size
    return rank * local_num_seqs, local_num_seqs


def make_case_inputs(case_name: str, runtime: Runtime, dtype: torch.dtype,
                     seed: int, parallel_strategy: str
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor, int, int, int, int,
                                tuple[int, ...]]:
    case = CASES[case_name]
    num_seqs = case.num_seqs
    seq_len = case.seq_len
    lengths = case.lengths
    x = make_uniform((case.total_tokens, D),
                     device=runtime.device,
                     dtype=dtype,
                     seed=seed + 11,
                     value_range=X_RANGE)
    local_lengths = lengths
    position_offset = 0

    if runtime.world_size > 1 and parallel_strategy == "benchmark":
        if case.strategy == "batch_sharded":
            # Contiguous, equal-count blocks of whole sequences per rank, as
            # for 1kx128. With variable lengths the per-rank token counts
            # differ; no balancing is applied.
            start, local_num_seqs = batch_shard_bounds(
                num_seqs, runtime.world_size, runtime.rank, case_name)
            local_lengths = lengths[start:start + local_num_seqs]
            token_start = sum(lengths[:start])
            x = x.narrow(0, token_start, sum(local_lengths)).contiguous()
        else:
            if case.sequence_lengths is not None:
                raise RuntimeError(
                    f"{case_name}: sequence_sharded needs equal lengths.")
            if seq_len % runtime.world_size != 0:
                raise RuntimeError(
                    f"{case_name} seq_len={seq_len} must be divisible by "
                    f"world_size={runtime.world_size} for sequence_sharded."
                )
            local_seq_len = seq_len // runtime.world_size
            position_offset = runtime.rank * local_seq_len
            x_3d = x.view(num_seqs, seq_len, D)
            x = x_3d.narrow(1, position_offset, local_seq_len).contiguous().view(
                num_seqs * local_seq_len, D)
            local_lengths = (local_seq_len, ) * num_seqs

    positions, b_start_loc, b_seq_len, cu_seqlens = make_sequence_metadata(
        local_lengths, position_offset, runtime.device)
    return (x, positions, b_start_loc, b_seq_len, cu_seqlens,
            len(local_lengths), max(local_lengths), num_seqs, seq_len,
            local_lengths)


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
        gam_a=make_uniform((D, ),
                           device=device,
                           dtype=dtype,
                           seed=seed + 100000 - 2,
                           value_range=GAMMA_RANGE),
        gam_f=make_uniform((D, ),
                           device=device,
                           dtype=dtype,
                           seed=seed + 100000 - 1,
                           value_range=GAMMA_RANGE),
        wq=make_uniform((D, q_heads_local, HEAD_DIM),
                        device=device,
                        dtype=dtype,
                        seed=shard_seed + 1,
                        value_range=WEIGHT_RANGE),
        wk=make_uniform((D, kv_heads_local, HEAD_DIM),
                        device=device,
                        dtype=dtype,
                        seed=shard_seed + 2,
                        value_range=WEIGHT_RANGE),
        wv=make_uniform((D, kv_heads_local, HEAD_DIM),
                        device=device,
                        dtype=dtype,
                        seed=shard_seed + 3,
                        value_range=WEIGHT_RANGE),
        wo=make_uniform((q_heads_local, HEAD_DIM, D),
                        device=device,
                        dtype=dtype,
                        seed=shard_seed + 4,
                        value_range=WEIGHT_RANGE),
        w_gu=make_uniform((D, 2 * ffn_inter_local),
                          device=device,
                          dtype=dtype,
                          seed=shard_seed + 5,
                          value_range=WEIGHT_RANGE),
        w_down=make_uniform((ffn_inter_local, D),
                            device=device,
                            dtype=dtype,
                            seed=shard_seed + 6,
                            value_range=WEIGHT_RANGE),
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


def attention_sequence_sharded(q: torch.Tensor, k: torch.Tensor,
                               v: torch.Tensor, num_seqs: int,
                               local_seq_len: int, full_seq_len: int,
                               query_offset: int,
                               backend: str) -> torch.Tensor:
    """Causal attention for one rank's query chunk against gathered K/V.

    Keys after this rank's last query position are always masked, so they are
    dropped. What remains is ordinary causal attention aligned to the
    bottom-right corner of the (local_q x kept_k) score matrix, which flash
    kernels support directly (no explicit mask).
    """
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if q_heads % kv_heads != 0:
        raise RuntimeError(
            f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}.")
    kept = query_offset + local_seq_len
    k4 = k.view(num_seqs, full_seq_len, kv_heads, HEAD_DIM)[:, :kept]
    v4 = v.view(num_seqs, full_seq_len, kv_heads, HEAD_DIM)[:, :kept]

    if backend == "flash-attn-varlen":
        # FA2 aligns the causal mask bottom-right when seqlen_q < seqlen_k.
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        seq_idx = torch.arange(num_seqs + 1, device=q.device,
                               dtype=torch.int32)
        out = flash_attn_varlen_func(
            q=q.contiguous(),
            k=k4.reshape(num_seqs * kept, kv_heads, HEAD_DIM),
            v=v4.reshape(num_seqs * kept, kv_heads, HEAD_DIM),
            cu_seqlens_q=seq_idx * local_seq_len,
            cu_seqlens_k=seq_idx * kept,
            max_seqlen_q=local_seq_len,
            max_seqlen_k=kept,
            dropout_p=0.0,
            softmax_scale=HEAD_DIM**-0.5,
            causal=True,
            fa_version=2,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out

    # SDPA (V100): its fused kernels need equal head counts, so expand K/V
    # after the gather.
    from torch.nn.attention.bias import causal_lower_right

    q4 = q.contiguous().view(num_seqs, local_seq_len, q_heads,
                             HEAD_DIM).permute(0, 2, 1, 3)
    k4 = k4.permute(0, 2, 1, 3)
    v4 = v4.permute(0, 2, 1, 3)
    repeat = q_heads // kv_heads
    if repeat != 1:
        k4 = k4.repeat_interleave(repeat, dim=1)
        v4 = v4.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(
        q4, k4.contiguous(), v4.contiguous(),
        attn_mask=causal_lower_right(local_seq_len, kept), dropout_p=0.0)
    return out.permute(0, 2, 1, 3).reshape(num_seqs * local_seq_len, q_heads,
                                           HEAD_DIM)


def attention_sdpa_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          lengths: tuple[int, ...]) -> torch.Tensor:
    # SDPA has no packed variable-length input, so run it per sequence. A
    # block-diagonal mask over all tokens would not fit in memory.
    out = torch.empty_like(q)
    start = 0
    for length in lengths:
        end = start + length
        out[start:end] = attention_sdpa(q[start:end], k[start:end],
                                        v[start:end], 1, length)
        start = end
    return out


def packed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     b_start_loc: torch.Tensor, b_seq_len: torch.Tensor,
                     cu_seqlens: torch.Tensor, num_seqs: int, seq_len: int,
                     lengths: tuple[int, ...], backend: str) -> torch.Tensor:
    # seq_len is the maximum local sequence length.
    if backend == "triton-prefill":
        return attention_triton_prefill(q, k, v, b_start_loc, b_seq_len, seq_len)
    if backend == "flash-attn-varlen":
        return attention_flash_varlen(q, k, v, cu_seqlens, seq_len)
    if backend == "sdpa":
        if len(set(lengths)) > 1:
            return attention_sdpa_varlen(q, k, v, lengths)
        return attention_sdpa(q, k, v, num_seqs, seq_len)
    raise ValueError(f"Unsupported attention backend: {backend}")


def llama_block_forward(x: torch.Tensor, positions: torch.Tensor,
                        b_start_loc: torch.Tensor, b_seq_len: torch.Tensor,
                        cu_seqlens: torch.Tensor, cos_cache: torch.Tensor,
                        sin_cache: torch.Tensor, num_seqs: int, seq_len: int,
                        local_lengths: tuple[int, ...],
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
        attn_ctx = attention_sequence_sharded(q_r, k_full, v_full,
                                              global_num_seqs, seq_len,
                                              global_seq_len, query_offset,
                                              backend)
    else:
        attn_ctx = packed_attention(q_r, k_r, v_3d, b_start_loc, b_seq_len,
                                    cu_seqlens, num_seqs, seq_len,
                                    local_lengths, backend)
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


def max_over_ranks(value: float, runtime: Runtime) -> float:
    if not runtime.is_distributed:
        return value
    tensor = torch.tensor([value], device=runtime.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def all_ranks_true(flag: bool, runtime: Runtime) -> bool:
    if not runtime.is_distributed:
        return flag
    tensor = torch.tensor([1 if flag else 0], device=runtime.device,
                          dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def elapsed_ms(work, runtime: Runtime) -> float:
    """Host wall time of one step: all ranks start together (sync + barrier),
    each times until its own device is idle, and the slowest rank counts."""
    synchronize(runtime)
    start = time.perf_counter()
    work()
    torch.cuda.synchronize(runtime.device)
    return max_over_ranks((time.perf_counter() - start) * 1000.0, runtime)


def capture_forward(step, runtime: Runtime, capture_warmups: int
                    ) -> tuple[Any, torch.Tensor, dict[str, Any],
                               torch.cuda.CUDAGraph | None]:
    """Untimed eager warmups, then capture `step` in a CUDA graph.

    Returns the callable to time (graph replay; eager `step` only if capture is
    not supported), its output tensor, a record of what was done, and the graph.
    Mirrors Einsummable's untimed first execution (eager warmup + capture).
    """
    device = runtime.device
    synchronize(runtime)
    start = time.perf_counter()
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(capture_warmups):
            step()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    # Return eager-warmup blocks so the graph's private pool can reuse them.
    torch.cuda.empty_cache()
    synchronize(runtime)
    graph = torch.cuda.CUDAGraph()
    output = None
    error = None
    try:
        with torch.cuda.graph(graph):
            output = step()
    except Exception as exc:  # noqa: BLE001 - unsupported; fall back to eager
        error = f"{type(exc).__name__}: {exc}"
    torch.cuda.synchronize(device)
    captured = all_ranks_true(error is None, runtime)
    if captured:
        graph.replay()
    else:
        graph.reset()
        graph = None
        torch.cuda.empty_cache()
        output = step()
    torch.cuda.synchronize(device)
    record = {
        "execution": "cuda_graph" if captured else "eager",
        "capture_warmups": capture_warmups,
        "warmup_capture_ms": max_over_ranks(
            (time.perf_counter() - start) * 1000.0, runtime),
        "capture_error": None if captured else (
            error or "capture failed on another rank"),
    }
    if not captured:
        rank0_print(runtime,
                    f"warning: CUDA graph capture unsupported, timing eager: "
                    f"{record['capture_error']}")
    return (graph.replay if captured else step), output, record, graph


def sequence_pattern_record(case: LlamaCase, case_name: str, runtime: Runtime,
                            parallel_strategy: str) -> dict[str, Any] | None:
    if case.sequence_pattern is None:
        return None
    lengths = list(case.lengths)
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    if parallel_strategy == "benchmark":
        _, per_rank = batch_shard_bounds(case.num_seqs, runtime.world_size, 0,
                                         case_name)
        shards = [lengths[r * per_rank:(r + 1) * per_rank]
                  for r in range(runtime.world_size)]
    else:
        shards = [lengths]
    return {
        **case.sequence_pattern,
        "sequence_lengths": lengths,
        "cu_seqlens_with_zero": cu,
        "max_sequence_length": max(lengths),
        "attention_work_proxy_sum_squared_lengths": sum(n * n for n in lengths),
        "per_rank_num_seqs": [len(s) for s in shards],
        "per_rank_tokens": [sum(s) for s in shards],
        "per_rank_attention_work_proxy": [sum(n * n for n in s) for s in shards],
    }


def benchmark_case(case_name: str, runtime: Runtime, dtype: torch.dtype,
                   backend: str,
                   args: argparse.Namespace) -> CaseBenchmarkResult:
    case = CASES[case_name]
    if backend == "sdpa" and case_name in SDPA_OOM_RISK_CASES:
        rank0_print(
            runtime,
            f"warning: sdpa on {case_name} may require a very large attention "
            "workspace and can OOM; triton-prefill is recommended.",
        )

    effective_backend = backend
    if (args.parallel_strategy == "benchmark" and runtime.world_size > 1
            and case.strategy == "sequence_sharded"
            and backend == "triton-prefill"):
        # The Triton prefill kernel has no query offset; use SDPA with a
        # bottom-right causal mask instead.
        effective_backend = "sdpa"
        rank0_print(
            runtime,
            f"case={case_name} strategy=sequence_sharded uses sdpa attention "
            "(triton-prefill has no query offset).",
        )

    (x, positions, b_start_loc, b_seq_len, cu_seqlens, num_seqs, seq_len,
     global_num_seqs, global_seq_len, local_lengths) = make_case_inputs(
         case_name, runtime, dtype, args.seed, args.parallel_strategy)
    cos_cache, sin_cache = rope_cache(global_seq_len, runtime.device, dtype)
    weights, _, _, _ = make_weights(runtime, dtype, args.parallel_strategy,
                                    args.seed)
    total_tokens = case.total_tokens
    local_tokens = x.shape[0]

    def run_once(selected_backend: str) -> torch.Tensor:
        return llama_block_forward(x, positions, b_start_loc, b_seq_len,
                                   cu_seqlens, cos_cache, sin_cache,
                                   num_seqs, seq_len, local_lengths,
                                   global_num_seqs, global_seq_len,
                                   weights, runtime,
                                   selected_backend, case.strategy,
                                   args.parallel_strategy)

    with torch.inference_mode():
        # Reset before the eager warmup so the peak covers the activations; the
        # graph's private pool is not re-allocated during replay.
        torch.cuda.reset_peak_memory_stats(runtime.device)
        run, output, execution, graph = capture_forward(
            lambda: run_once(effective_backend), runtime, args.capture_warmups)
        warmup_forward_ms = [
            elapsed_ms(run, runtime) for _ in range(args.warmup_iters)
        ]
        forward_ms = [
            elapsed_ms(run, runtime) for _ in range(args.benchmark_iters)
        ]

        assert output is not None
        mean_ms = statistics.mean(forward_ms)
        std_ms = statistics.stdev(forward_ms) if len(forward_ms) > 1 else 0.0
        min_ms = min(forward_ms)
        max_ms = max(forward_ms)
        tok_per_s = total_tokens / (mean_ms / 1000.0)
        peak_gib = torch.cuda.max_memory_allocated(runtime.device) / (1024**3)
        reference_check: str | None = None

        if args.check_reference:
            if case_name in SDPA_OOM_RISK_CASES:
                reference_check = "skipped:sdpa_may_oom"
                rank0_print(
                    runtime,
                    f"reference_check=skipped case={case_name} "
                    "reason=sdpa_may_oom",
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
        warmup_forward_ms=warmup_forward_ms,
        **execution,
        sequence_pattern=sequence_pattern_record(case, case_name, runtime,
                                                 args.parallel_strategy),
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
            f"execution={execution['execution']}",
            f"mean_forward_ms={mean_ms:.3f}",
            f"std_forward_ms={std_ms:.3f}",
            f"min_forward_ms={min_ms:.3f}",
            f"max_forward_ms={max_ms:.3f}",
            f"tokens_per_second={tok_per_s:.2f}",
            f"peak_memory_gib={peak_gib:.2f}",
            f"output_shape={tuple(output.shape)}",
            *([f"sequence_lengths_sha256="
               f"{result.sequence_pattern['sequence_lengths_sha256']}"]
              if result.sequence_pattern else []),
        ]),
    )

    del x, positions, b_start_loc, b_seq_len, cu_seqlens, cos_cache, sin_cache
    del weights, output, run
    if graph is not None:
        graph.reset()
    del graph
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
        "capture_warmups": args.capture_warmups,
        "timing": ("host wall time per step; sync + barrier before, device "
                   "sync after; max over ranks; CUDA graph replay"),
        "input_init": ("uniform: x [-0.1, 0.1], gamma [0.9, 1.1], "
                       "weights [-0.02, 0.02]"),
        "fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        "check_reference": args.check_reference,
        "cases": [asdict(result) for result in case_results],
    }


def main() -> None:
    args = parse_args()
    runtime = setup_runtime(args.tp_size)
    # Match Einsummable's cuBLAS setup: FP16 GEMMs accumulate in FP32
    # (CUBLAS_COMPUTE_32F), and reduced-precision split-K reductions stay
    # allowed (its handle never sets
    # CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION). PyTorch's default.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
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

    case_names = list(CASE_GROUPS.get(args.case, (args.case, )))
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
