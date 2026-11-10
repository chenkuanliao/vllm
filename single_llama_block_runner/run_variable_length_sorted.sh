#!/usr/bin/env bash
# Sorted variable-length packed Llama cases (var-moderate-128-sorted, var-strong-128-sorted) at
# 1, 4, and 8 GPUs. Same layout as Einsummable variable_length_sorted/; split token-balanced.
# Extra args are forwarded, e.g. --case var-strong-128-sorted to run only one case.
set -euo pipefail
cd "$(dirname "$0")"
export RESULTS_TAG="${RESULTS_TAG:-variable_length_sorted}"
./run_1gpu.sh --case variable-length-sorted "$@"
./run_4gpu.sh --case variable-length-sorted "$@"
./run_8gpu.sh --case variable-length-sorted "$@"
