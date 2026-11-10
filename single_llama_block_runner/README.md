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

This repository uses one conda env, `vllm-experiments`, created or updated by
`setup_env.sh` for the GPU type of the machine:

```bash
cd single_llama_block_runner
./setup_env.sh v100     # or: ./setup_env.sh a100
./run_all.sh
```

Re-running `setup_env.sh` updates the env in place. The `run_*.sh` scripts
find it through `env.sh` (override the name with `VLLM_CONDA_ENV`), so no
activation is needed.

| GPU | What `setup_env.sh` installs |
|---|---|
| V100 | Python 3.12, torch 2.11.0 (CUDA 12.6 build: ships V100 sm_70 kernels, runs on the CUDA 12.4 driver). vLLM itself is not installed: the default V100 attention backend is PyTorch SDPA. |
| A100 | Python 3.12, torch 2.11.0 (CUDA 12.9 build), and vLLM from this repository with its precompiled CUDA 12.9 kernels (`VLLM_USE_PRECOMPILED=1`), needed for the default `flash-attn-varlen` backend and for `triton-prefill`. |

torch 2.11.0 is the version this vLLM source pins (`requirements/cuda.txt`).
The script checks the torch version and, on A100, that FlashAttention-2 varlen
is available. `run_all.sh` runs `run_1gpu.sh`, `run_4gpu.sh`, then
`run_8gpu.sh`, so it needs 8 visible GPUs; otherwise run only the matching
launch script.

On V100, keep the default `float16` dtype and expect `auto` to select `sdpa`.
On A100, the default `float16` works, and `--dtype bfloat16` is optional.

## Cases

| Case | Shape | Total tokens |
|---|---:|---:|
| `1k` | 1 sequence x 1024 | 1024 |
| `8k` | 1 sequence x 8192 | 8192 |
| `128k` | 1 sequence x 131072 | 131072 |
| `1kx8` | 8 sequences x 1024 packed | 8192 |
| `1kx128` | 128 sequences x 1024 packed | 131072 |

`128k` is one packed 131072-token causal sequence. `1kx128` means 128
independent packed 1024-token causal sequences, not one 131072-token sequence.
Both `128k` and `1kx128` use 131072 total tokens but different sharding
strategies.

Opt-in variable-length cases (not part of `--case all`):

| Case | Lengths x count | Sequences | Total tokens |
|---|---|---:|---:|
| `var-moderate-128` | 512 x 32, 768 x 32, 1280 x 32, 1536 x 32 | 128 | 131072 |
| `var-strong-128` | 256 x 96, 1536 x 16, 5120 x 16 | 128 | 131072 |
| `var-moderate-128-sorted` | as `var-moderate-128`, sorted short to long | 128 | 131072 |
| `var-strong-128-sorted` | as `var-strong-128`, sorted short to long | 128 | 131072 |

See [Variable-Length Cases](#variable-length-cases) and
[Sorted Variable-Length Cases](#sorted-variable-length-cases).

## Distributed Strategies

Default `--parallel-strategy benchmark` matches the benchmark directory:

| Case | Strategy | Distributed behavior |
|---|---|---|
| `1k` | `sequence_sharded` | Split the sequence dimension across ranks, all-gather K/V, then use offset-aware causal attention. |
| `8k` | `sequence_sharded` | Same as `1k`, with a longer global sequence. |
| `128k` | `sequence_sharded` | Same as `8k`, with a 131072-token global sequence. |
| `1kx8` | `batch_sharded` | Split independent packed sequences across ranks and run local packed causal attention. |
| `1kx128` | `batch_sharded` | Same as `1kx8`, with more local sequences per rank. |
| `var-moderate-128` | `batch_sharded` | Whole sequences per rank, split token-balanced (see below). Per-rank sequence counts differ. |
| `var-strong-128` | `batch_sharded` | Same as `var-moderate-128`. |

Batch-sharded cases keep using the selected packed vLLM attention backend on
each rank. In sequence-sharded cases each rank all-gathers K/V, keeps only the
keys up to its last query position, and runs ordinary causal attention aligned
to the bottom-right of the (local queries x kept keys) score matrix. This is
exact and needs no explicit mask, so `flash-attn-varlen` (A100) handles it
directly; `sdpa` (V100) uses `causal_lower_right`. `triton-prefill` has no
query offset, so those cases fall back to SDPA and report it as the per-case
effective `attention_backend`.

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

## Variable-Length Cases

`var-moderate-128` and `var-strong-128` match the Einsummable experiments of
the same name (`experiements/variable_length/` on the vldb-experiments
branch). Each keeps `1kx128`'s 128-sequence / 131072-token budget and varies
the sequence lengths. Run them at 1, 4, and 8 GPUs on V100 or A100 with:

```bash
./run_variable_length.sh
./run_variable_length.sh --case var-strong-128   # one case only
./run_4gpu.sh --case variable-length             # one GPU count only
```

Results go to `results/variable_length_tp{1,4,8}_<timestamp>.json`.

**Same layout as Einsummable.** The ordered lengths are generated with
Einsummable's `sha256-sort-v1` recipe (seed 42): expand the length histogram
in order, then sort index `i` by `SHA256("42:i")`. No library RNG is used.
At import, `run_block.py` checks the result against the checksum Einsummable
writes to every `sequence_pattern.json` (`sequence_lengths_sha256`: SHA256 of
the JSON length array with no spaces) and refuses to run on a mismatch:

| Case | `sequence_lengths_sha256` |
|---|---|
| `var-moderate-128` | `98788f83854bfb6ff2b6e5fedf6661ef3781bbd086b5f18e186447357edd8cba` |
| `var-strong-128` | `b1206b5a6e85f731b6554817dd9dc7750727586e118db04966901adcb8c8b222` |

Inputs are one packed buffer with those boundaries. RoPE positions restart at
zero for each sequence, and attention is causal within each sequence. Nothing
is padded to the maximum length.

**Distribution and kernels (no tuning).** These cases use `batch_sharded`
like `1kx128`, but split token-balanced: each rank gets a contiguous run of
whole sequences, in pattern order, ending at the sequence boundary closest to
an equal share of tokens (ties go to the earlier boundary; every rank gets at
least one sequence). The rule is the same as in jax-pytorch-experiments'
`sequence_patterns.token_balanced_cuts`. Sequence counts per rank differ and
token counts are near-equal (max/mean 1.01-1.08); attention work still
differs, since long sequences cost more per token. Attention uses the
default backend: vLLM's FlashAttention-2 varlen kernel on A100 and PyTorch
SDPA on V100. SDPA has no packed variable-length input, so the runner calls it
once per sequence.

Each case's JSON entry has a `sequence_pattern` record with the ordered
`sequence_lengths`, `cu_seqlens_with_zero`, the checksum, and
`per_rank_num_seqs`, `per_rank_tokens`, and `per_rank_attention_work_proxy`
(sum of squared lengths). For these cases `seq_len` and `local_seq_len` are the
maximum (local) sequence length.

## Sorted Variable-Length Cases

`var-moderate-128-sorted` and `var-strong-128-sorted` match Einsummable's
`experiements/variable_length_sorted/` setups: the same histograms as above,
packed short to long (`length-ascending-v1`, no seed). Run them with:

```bash
./run_variable_length_sorted.sh
./run_variable_length_sorted.sh --case var-strong-128-sorted   # one case only
./run_4gpu.sh --case variable-length-sorted                    # one GPU count only
```

Results go to `results/variable_length_sorted_tp{1,4,8}_<timestamp>.json`.
The checksums are verified at import as for the shuffled cases:

| Case | `sequence_lengths_sha256` |
|---|---|
| `var-moderate-128-sorted` | `0ffa21a4e662f1466228c20111a8a369b72f3e03cd7be6f9b7b12b0415177f5c` |
| `var-strong-128-sorted` | `c950be1808c585c308b9c74727163471481ce4812800868517fbc08625400cfa` |

**Split.** The same token-balanced split as the shuffled cases. It matters
more here: equal sequence counts would give the last ranks all the long
sequences (up to 5x the average tokens on `var-strong-128-sorted` at 8 GPUs);
balanced, per-rank token counts are within 1.25x of the average. The
`sequence_pattern` record includes `split` and the per-rank counts. Everything
else (kernels, timing, CUDA graphs) is as for the shuffled cases.

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

SDPA can require a large attention workspace for `128k` and `1kx128`; use
`flash-attn-varlen` or `triton-prefill` for large packed cases.

## Reference Check

For small cases, compare the selected backend against SDPA:

```bash
./run_1gpu.sh --case 1k --warmup-iters 1 --benchmark-iters 1 --check-reference
```

The reference check is skipped for `128k` and `1kx128` because SDPA may run
out of memory.

## CLI

```bash
python run_block.py \
  --case {1k,8k,128k,1kx8,1kx128,var-moderate-128,var-strong-128,var-moderate-128-sorted,var-strong-128-sorted,all,variable-length,variable-length-sorted} \
  --tp-size {1,4,8} \
  --parallel-strategy {benchmark,tensor-parallel} \
  --dtype {auto,float16,bfloat16,float32} \
  --attention-backend {auto,triton-prefill,flash-attn-varlen,sdpa} \
  --seed 0 \
  --capture-warmups 1 \
  --warmup-iters 2 \
  --benchmark-iters 10 \
  --output-json results/custom.json
```

## Timing, Inputs, and Precision

These match the Einsummable Llama experiments (`experiements/worker.py`) and the
`benchmarks/jax-llama` / `benchmarks/pytorch-llama` runners:

- Each case runs `--capture-warmups` (default 1) untimed eager forwards on a
  side stream, then captures the forward in a CUDA graph and launches it once
  (recorded as `warmup_capture_ms`). NCCL collectives are captured too.
- `--warmup-iters` (default 2) graph replays are timed and recorded in
  `warmup_forward_ms` but excluded from the stats; `--benchmark-iters`
  (default 10) replays are measured.
- Each step is host wall time: device sync + barrier before the step, timed
  until the local device is idle, and the slowest rank's time is reported.
- If capture is not supported for a case, `execution` is `eager` and
  `capture_error` says why; otherwise `execution` is `cuda_graph`.
- Inputs are uniform with Einsummable's ranges: `x` in [-0.1, 0.1], RMSNorm
  gammas in [0.9, 1.1], weights in [-0.02, 0.02]. FP16 GEMMs accumulate in
  FP32; RMSNorm computes in FP32.

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
`execution`, `warmup_capture_ms`, `warmup_forward_ms`,
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
