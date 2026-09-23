#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

load_env() {
  local process_api_key="${API_KEY-}"
  local process_restart_secret="${RESTART_SECRET-}"
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
  if [[ -n "$process_api_key" ]]; then
    export API_KEY="$process_api_key"
  fi
  if [[ -n "$process_restart_secret" ]]; then
    export RESTART_SECRET="$process_restart_secret"
  fi
  export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
}
