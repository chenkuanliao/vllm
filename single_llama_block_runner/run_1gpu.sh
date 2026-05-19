#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
../.venv/bin/python run_block.py --case all --tp-size 1 "$@"
