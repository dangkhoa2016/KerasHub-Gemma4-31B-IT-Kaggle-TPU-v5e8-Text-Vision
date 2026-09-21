#!/usr/bin/env bash
set -Eeuo pipefail
if [[ -f state/tunnel.pid ]]; then
  pid="$(cat state/tunnel.pid)"
  kill "$pid" 2>/dev/null || true
  rm -f state/tunnel.pid
fi
