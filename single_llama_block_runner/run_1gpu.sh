#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=env.sh
source "./env.sh"

"$PYTHON" run_block.py --case all --tp-size 1 \
  --output-json "$(results_json_path 1)" "$@"
