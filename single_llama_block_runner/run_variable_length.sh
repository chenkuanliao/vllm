#!/usr/bin/env bash
# Variable-length packed Llama cases (var-moderate-128, var-strong-128) at
# 1, 4, and 8 GPUs. Same sequence layout as the Einsummable experiments.
# Extra args are forwarded, e.g. --case var-strong-128 to run only one case.
set -euo pipefail
cd "$(dirname "$0")"
export RESULTS_TAG="${RESULTS_TAG:-variable_length}"
./run_1gpu.sh --case variable-length "$@"
./run_4gpu.sh --case variable-length "$@"
./run_8gpu.sh --case variable-length "$@"
