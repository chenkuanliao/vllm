#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./run_1gpu.sh "$@"
./run_4gpu.sh "$@"
./run_8gpu.sh "$@"
