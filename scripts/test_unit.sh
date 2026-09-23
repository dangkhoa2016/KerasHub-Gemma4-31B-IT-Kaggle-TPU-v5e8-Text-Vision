#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts clients/python

for script in   scripts/_common.sh   scripts/configure_kaggle_tpu.sh   scripts/run_tunnel.sh   scripts/start.sh   scripts/status.sh   scripts/stop.sh   scripts/stop_tunnel.sh
do
  bash -n "$script"
done
