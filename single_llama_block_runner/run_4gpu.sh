#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=env.sh
source "./env.sh"

"$TORCHRUN" --nproc-per-node=4 run_block.py --case all --tp-size 4 "$@"
