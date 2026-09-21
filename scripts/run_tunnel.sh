#!/usr/bin/env bash
set -Eeuo pipefail
PORT="${PORT:-7860}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed" >&2
  exit 1
fi

mkdir -p state logs
nohup cloudflared tunnel \
  --url "http://127.0.0.1:$PORT" \
  > logs/cloudflared.log 2>&1 &
echo $! > state/tunnel.pid
echo "Quick Tunnel started; inspect logs/cloudflared.log."
