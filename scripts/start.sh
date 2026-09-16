#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"

[[ -f .env ]] || cp .env.example .env
load_env
source scripts/configure_kaggle_tpu.sh

export KERAS_BACKEND=jax
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"

mkdir -p logs state
if [[ -f state/server.pid ]] \
  && kill -0 "$(cat state/server.pid)" 2>/dev/null; then
  echo "Server already running PID=$(cat state/server.pid)"
  exit 0
fi

nohup python3 src/server.py \
  > logs/server.stdout.log 2>&1 &
echo $! > state/server.pid
echo "Started coordinator PID=$(cat state/server.pid)"
