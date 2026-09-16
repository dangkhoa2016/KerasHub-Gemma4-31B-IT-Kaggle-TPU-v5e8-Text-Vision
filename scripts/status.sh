#!/usr/bin/env bash
set -Eeuo pipefail
for name in server worker; do
  file="state/$name.pid"
  if [[ -f "$file" ]]; then
    pid="$(cat "$file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name: RUNNING pid=$pid"
    else
      echo "$name: STALE pid=$pid"
    fi
  else
    echo "$name: NOT RUNNING"
  fi
done
