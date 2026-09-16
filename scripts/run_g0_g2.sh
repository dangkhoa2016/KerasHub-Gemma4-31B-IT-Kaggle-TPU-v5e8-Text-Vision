#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"

[[ -f .env ]] || cp .env.example .env
load_env

export KERAS_BACKEND=jax
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"

# Must run in the same process tree that later imports JAX/libtpu.
source scripts/configure_kaggle_tpu.sh

case "${INSTALL_PYTHON_DEPS:-auto}" in
  auto|true|1|yes)
    python3 -m pip install \
      --disable-pip-version-check \
      --root-user-action=ignore \
      --no-warn-conflicts \
      --progress-bar off \
      -r requirements.txt
    ;;
  false|0|no) ;;
  *)
    echo "Invalid INSTALL_PYTHON_DEPS" >&2
    exit 2
    ;;
esac

python3 scripts/ensure_libtpu.py
python3 scripts/check_tpu_preflight.py

preferred="${MODEL_PATH:-${MODEL_BASE:-}}"
export MODEL_PATH="$(
  python3 scripts/resolve_kaggle_model_path.py \
    --preferred "$preferred" \
    --preset-name "${MODEL_PRESET:-gemma4_instruct_31b}"
)"
echo "[model] resolved: $MODEL_PATH"

python3 scripts/g0_g2_strict_load.py
