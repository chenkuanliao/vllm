# Single Llama Block Runner

This directory contains a standalone synthetic-weight benchmark for one Llama
decoder block. It is designed to run from inside this directory on V100 and
A100 systems with tensor parallel sizes 1, 4, and 8.

The runner does not use a KV cache, tokenizer, checkpoint loader, vLLM engine,
or vLLM scheduler. It uses PyTorch for the block math and vLLM's Triton prefill
attention op by default.

## Environment Setup

Create and install the vLLM development environment from the repo root:

```bash
cd /path/to/vllm

uv venv --python 3.12
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

The one-GPU script calls `../.venv/bin/python` directly. The multi-GPU scripts
use `torchrun`, so activate the environment first:

```bash
source ../.venv/bin/activate
```

On V100, keep the default `float16` dtype. On A100, the default `float16` works,
and `--dtype bfloat16` is optional.

## Cases

| Case | Shape | Total tokens |
|---|---:|---:|
| `1k` | 1 sequence x 1024 | 1024 |
| `8k` | 1 sequence x 8192 | 8192 |
| `1kx8` | 8 sequences x 1024 packed | 8192 |
| `1kx128` | 128 sequences x 1024 packed | 131072 |

`1kx128` means 128 independent packed 1024-token causal sequences, not one
131072-token sequence.

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

The one-GPU script uses `../.venv/bin/python`. The multi-GPU scripts use
`torchrun`, which should come from the active vLLM environment.

## Direct Commands

```bash
../.venv/bin/python run_block.py --case all --tp-size 1
torchrun --nproc-per-node=4 run_block.py --case all --tp-size 4
torchrun --nproc-per-node=8 run_block.py --case all --tp-size 8
```

## Hardware Defaults

The script detects the GPU name and compute capability at runtime.

- V100: default `--dtype auto` resolves to `float16`.
- A100: default `--dtype auto` resolves to `float16`.
- A100 optional BF16: pass `--dtype bfloat16`.

The default attention backend is `--attention-backend auto`, which resolves to
`triton-prefill` on both V100 and A100. This backend avoids materializing the
full attention matrix and is the recommended path for `1kx128`.

## Optional Attention Backends

Portable default:

```bash
./run_8gpu.sh --attention-backend triton-prefill --dtype float16
```

A100 FlashAttention varlen experiment:

```bash
./run_8gpu.sh --attention-backend flash-attn-varlen
```

`flash-attn-varlen` is never selected automatically. If requested on V100, the
script exits with a clear error because this vLLM build requires compute
capability >= 8.0 for that path.

PyTorch SDPA debug path:

```bash
./run_1gpu.sh --case 1k --attention-backend sdpa
```

SDPA can require a large attention workspace for `1kx128`; use
`triton-prefill` for large packed cases.

## Reference Check

For small cases, compare the selected backend against SDPA:

```bash
./run_1gpu.sh --case 1k --warmup-iters 1 --benchmark-iters 1 --check-reference
```

The reference check is skipped for `1kx128` because SDPA may run out of memory.

## CLI

```bash
../.venv/bin/python run_block.py \
  --case {1k,8k,1kx8,1kx128,all} \
  --tp-size {1,4,8} \
  --dtype {auto,float16,bfloat16,float32} \
  --attention-backend {auto,triton-prefill,flash-attn-varlen,sdpa} \
  --seed 0 \
  --warmup-iters 5 \
  --benchmark-iters 20
```

## Output

Rank 0 prints one line per case:

```text
case=1kx128 tp_size=8 dtype=float16 num_seqs=128 seq_len=1024
total_tokens=131072 attention_backend=triton-prefill mean_forward_ms=...
tokens_per_second=... peak_memory_gib=... output_shape=(131072, 4096)
```

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
