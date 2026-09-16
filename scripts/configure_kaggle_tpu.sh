#!/usr/bin/env bash
set -Eeuo pipefail

mode="${KAGGLE_TPU_FALLBACK:-auto}"
case "$mode" in
  auto|true|false) ;;
  *) echo "Invalid KAGGLE_TPU_FALLBACK" >&2; return 2 2>/dev/null || exit 2 ;;
esac

is_kaggle=false
if [[ -d /kaggle/working || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
  is_kaggle=true
fi

metadata=false
if command -v curl >/dev/null 2>&1; then
  if curl -fsS --connect-timeout 1 --max-time 2 \
    -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type' \
    >/dev/null 2>&1; then
    metadata=true
  fi
fi

apply=false
[[ "$mode" == true ]] && apply=true
if [[ "$mode" == auto && "$is_kaggle" == true && "$metadata" == false \
      && "${REQUIRE_V5E8:-true}" == true \
      && "${EXPECTED_TPU_DEVICES:-8}" == 8 ]]; then
  apply=true
fi

if [[ "$apply" == true ]]; then
  export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
  export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-${TPU_CHIPS_PER_HOST_BOUNDS_FALLBACK:-2,4,1}}"
  export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-${TPU_HOST_BOUNDS_FALLBACK:-1,1,1}}"
  export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-${TPU_ACCELERATOR_TYPE_FALLBACK:-v5e-8}}"
  export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-$(hostname)}"
  export G4_TPU_FALLBACK_APPLIED=true
  echo "[tpu-config] Kaggle fallback applied: $TPU_ACCELERATOR_TYPE"
else
  export G4_TPU_FALLBACK_APPLIED=false
  echo "[tpu-config] normal TPU discovery retained"
fi
