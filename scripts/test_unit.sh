#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python3 -m unittest discover -s tests -v
python3 -m compileall -q \
  src scripts clients/python

bash -n scripts/configure_kaggle_tpu.sh
bash -n scripts/run_g0_g2.sh
bash -n scripts/start.sh
bash -n scripts/stop.sh
bash -n scripts/status.sh
