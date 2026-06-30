# Single Llama Block Runner

This directory contains a standalone synthetic-weight benchmark for one Llama
decoder block. It is designed to run from inside this directory on V100 and
A100 systems with 1, 4, and 8 GPU launches.

The runner does not use a KV cache, tokenizer, checkpoint loader, vLLM engine,
or vLLM scheduler. It uses PyTorch for the block math and vLLM's Triton prefill
attention op by default.

By default, distributed runs use the same high-level strategies as
`benchmarks/jax-llama` and `benchmarks/pytorch-llama`: replicated weights with
batch-sharded or sequence-sharded activations. The original tensor-parallel
synthetic mode is still available with `--parallel-strategy tensor-parallel`.

## Environment Setup

Install vLLM from the repo root with either a uv virtualenv or the
`vllm-py312` conda environment. The launch scripts prefer `../.venv` when
present, otherwise the `vllm-py312` conda env (override with `VLLM_CONDA_ENV`),
otherwise `python`/`torchrun` from `PATH`.

uv virtualenv:

```bash
cd /path/to/vllm

uv venv --python 3.12
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Conda on V100 (this machine):

PyTorch 2.11 does not ship CUDA kernels for Tesla V100 (compute capability
7.0). Use torch 2.6+cu124 and let the runner default to the SDPA attention
backend:

```bash
cd /path/to/vllm

conda create -n vllm-py312 python=3.12 -y
conda activate vllm-py312

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0+cu124 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
```

No vLLM install is required for the default V100 path (`--attention-backend
auto` selects `sdpa`). For A100 FlashAttention-2 varlen or `triton-prefill`,
build vLLM from source against the same torch 2.6+cu124 stack.

Conda on A100 and newer (compute capability >= 8.0):

```bash
cd /path/to/vllm

conda create -n vllm-py312 python=3.12 -y
conda activate vllm-py312

conda install -c conda-forge uv -y
uv pip install --python "$CONDA_PREFIX/bin/python" --upgrade setuptools wheel
uv pip install --python "$CONDA_PREFIX/bin/python" \
  torch==2.11.0 torchaudio torchvision \
  --index-url https://download.pytorch.org/whl/cu129
VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 VLLM_USE_PRECOMPILED=1 \
  uv pip install --python "$CONDA_PREFIX/bin/python" -e . \
  --torch-backend=auto
```

Fresh 8xA100 setup and run:

```bash
cd /path/to/vllm

conda create -n vllm-py312 python=3.12 -y
conda activate vllm-py312

conda install -c conda-forge uv -y
uv pip install --python "$CONDA_PREFIX/bin/python" --upgrade setuptools wheel
uv pip install --python "$CONDA_PREFIX/bin/python" \
  torch==2.11.0 torchaudio torchvision \
  --index-url https://download.pytorch.org/whl/cu129
VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 VLLM_USE_PRECOMPILED=1 \
  uv pip install --python "$CONDA_PREFIX/bin/python" -e . \
  --torch-backend=auto

cd single_llama_block_runner
./run_all.sh
```

`run_all.sh` runs `run_1gpu.sh`, `run_4gpu.sh`, then `run_8gpu.sh`. The full
script therefore requires at least 8 visible A100 GPUs. If the machine has
fewer visible GPUs, run only the matching launch script. The A100 default
`--attention-backend auto` resolves to `flash-attn-varlen` backed by FA2, so
the editable vLLM install above is required.

The launch scripts prefer `../.venv` when present, otherwise the `vllm-py312`
conda env (override with `VLLM_CONDA_ENV`), otherwise `python`/`torchrun`
from `PATH`. You do not need to activate conda if `vllm-py312` exists, but
activating it is fine:

```bash
conda activate vllm-py312
cd single_llama_block_runner
./run_1gpu.sh
```

Your NVIDIA driver must support the PyTorch CUDA build you install. If you
previously installed `cu130` wheels on a driver that reports `CUDA Version:
12.4`, use `cu124` on V100 or `cu129` on newer GPUs.

On V100, keep the default `float16` dtype and expect `auto` to select `sdpa`.
On A100, the default `float16` works, and `--dtype bfloat16` is optional.

## Cases

| Case | Shape | Total tokens |
|---|---:|---:|
| `1k` | 1 sequence x 1024 | 1024 |
| `8k` | 1 sequence x 8192 | 8192 |
| `1kx8` | 8 sequences x 1024 packed | 8192 |
| `1kx128` | 128 sequences x 1024 packed | 131072 |

`1kx128` means 128 independent packed 1024-token causal sequences, not one
131072-token sequence.

## Distributed Strategies

Default `--parallel-strategy benchmark` matches the benchmark directory:

| Case | Strategy | Distributed behavior |
|---|---|---|
| `1k` | `sequence_sharded` | Split the sequence dimension across ranks, all-gather K/V, then use offset-aware causal attention. |
| `8k` | `sequence_sharded` | Same as `1k`, with a longer global sequence. |
| `1kx8` | `batch_sharded` | Split independent packed sequences across ranks and run local packed causal attention. |
| `1kx128` | `batch_sharded` | Same as `1kx8`, with more local sequences per rank. |

Batch-sharded cases keep using the selected packed vLLM attention backend on
each rank. Sequence-sharded cases need an offset-aware causal mask after K/V
gather; the runner uses SDPA for that attention step and reports it as the
per-case effective `attention_backend`.

Legacy tensor-parallel mode shards Q/K/V heads and FFN intermediate weights,
keeps full inputs on every rank, and all-reduces the output projection and FFN
down projection:

```bash
./run_8gpu.sh --parallel-strategy tensor-parallel
```

## Quick Start

From the vLLM repo root:

```bash
cd single_llama_block_runner
./run_1gpu.sh
./run_4gpu.sh
./run_8gpu.sh
```

Run all launch sizes in order:

```bash
./run_all.sh
```

The launch scripts resolve Python and `torchrun` automatically via `env.sh`.

## Direct Commands

With the environment active:

```bash
python run_block.py --case all --tp-size 1
torchrun --nproc-per-node=4 run_block.py --case all --tp-size 4
torchrun --nproc-per-node=8 run_block.py --case all --tp-size 8
```

## Hardware Defaults

The script detects the GPU name and compute capability at runtime.

- V100: default `--dtype auto` resolves to `float16`.
- A100: default `--dtype auto` resolves to `float16`.
- A100 optional BF16: pass `--dtype bfloat16`.

The default attention backend is `--attention-backend auto`:
- V100: resolves to `sdpa` (PyTorch 2.11+ does not ship Volta kernels).
- A100: resolves to `flash-attn-varlen`, which uses vLLM's FlashAttention-2
  varlen kernel.

## Optional Attention Backends

Triton prefill fallback:

```bash
./run_8gpu.sh --attention-backend triton-prefill --dtype float16
```

A100 FlashAttention-2 varlen path:

```bash
./run_8gpu.sh --attention-backend flash-attn-varlen
```

`flash-attn-varlen` is selected automatically on A100 and newer when available.
If requested on V100, the script exits with a clear error because this vLLM
build requires compute capability >= 8.0 for that path.

PyTorch SDPA debug path:

```bash
./run_1gpu.sh --case 1k --attention-backend sdpa
```

SDPA can require a large attention workspace for `1kx128`; use
`flash-attn-varlen` or `triton-prefill` for large packed cases.

## Reference Check

For small cases, compare the selected backend against SDPA:

```bash
./run_1gpu.sh --case 1k --warmup-iters 1 --benchmark-iters 1 --check-reference
```

The reference check is skipped for `1kx128` because SDPA may run out of memory.

## CLI

```bash
python run_block.py \
  --case {1k,8k,1kx8,1kx128,all} \
  --tp-size {1,4,8} \
  --parallel-strategy {benchmark,tensor-parallel} \
  --dtype {auto,float16,bfloat16,float32} \
  --attention-backend {auto,triton-prefill,flash-attn-varlen,sdpa} \
  --seed 0 \
  --warmup-iters 5 \
  --benchmark-iters 20 \
  --output-json results/custom.json
```

## Output

Rank 0 prints one line per case:

```text
case=1kx128 strategy=batch_sharded tp_size=8 dtype=float16
global_num_seqs=128 global_seq_len=1024 local_num_seqs=16 local_seq_len=1024
total_tokens=131072 local_tokens=16384 attention_backend=flash-attn-varlen
mean_forward_ms=... std_forward_ms=... min_forward_ms=... max_forward_ms=...
tokens_per_second=... peak_memory_gib=... output_shape=(16384, 4096)
```

The launch scripts write a timestamped JSON file under `results/`, for example
`results/tp8_20260519T153045Z.json`. Override the path with `--output-json`:

```bash
./run_1gpu.sh --output-json results/my_run.json
```

Each JSON file includes run metadata (GPU, dtype, requested backend, parallel
strategy, iteration counts) and per-case stats: strategy, effective attention
backend, global/local shape, every benchmark iteration time in `forward_ms`, plus
`mean_forward_ms`, `std_forward_ms`, `min_forward_ms`, `max_forward_ms`,
`tokens_per_second`, `peak_memory_gib`, and `output_shape`.

## Model Shape

The synthetic block uses:

```text
D = 4096
n_q_head = 32
n_kv_heads = 8
head_dim = 128
ffn_inter = 14336
gate_up = 28672
```

The forward graph is:

```text
rmsnorm -> q/k/v projections -> RoPE -> packed causal GQA attention
-> output projection -> add+rmsnorm -> gate/up -> silu*up
-> down projection -> residual add
```
