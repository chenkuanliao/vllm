#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
torchrun --nproc-per-node=8 run_block.py --case all --tp-size 8 "$@"
