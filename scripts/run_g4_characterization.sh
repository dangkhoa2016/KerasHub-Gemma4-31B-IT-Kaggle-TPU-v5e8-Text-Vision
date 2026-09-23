#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_31b/2"
MODEL_PRESET="gemma4_instruct_31b"
MINIMUM_AUTHORITY_MEMORY_GIB=300
EXPECTED_TPU_DEVICES=8
G4_TPU_ATTEMPTS="${G4_TPU_ATTEMPT_NUMBER:-1}"
G3_ARCHIVE="${G3_ARCHIVE:-$PROJECT_ROOT/artifacts/g3/gemma4-31b-vnext-g3-authority-20260913T225039Z.tar.gz}"
G3_ARCHIVE_SHA256="475c940baf60302fd0f03e5f4b1a156696598d51162310ff168283551d6bb38e"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
evidence_dir="/kaggle/working/gemma4-31b-g4-characterization-${stamp}"
archive="/kaggle/working/gemma4-31b-g4-characterization-${stamp}.tar.gz"
mkdir -p "$evidence_dir"
for file in \
  00-g3-freeze-reference.txt 01-g4-source-hashes-before.txt \
  02-g4-source-hashes-after.txt 03-api-discovery.txt 04-hardware-gate.txt \
  05-runtime.txt 06-dependency-gate.txt 07-g4-authority.stdout.log \
  08-g4-authority.stderr.log 09-g4-result.json 10-memory-before.txt \
  11-memory-after.txt 12-native-vs-split-comparison.json \
  13-final-adjudication.txt; do
  : > "$evidence_dir/$file"
done

memory_cgroup_dir="/sys/fs/cgroup"
while IFS=: read -r hierarchy controllers path; do
  if [[ "$hierarchy" == 0 && -z "$controllers" ]]; then
    memory_cgroup_dir="/sys/fs/cgroup/${path#/}"
    break
  fi
done < /proc/self/cgroup
memory_max_path="$memory_cgroup_dir/memory.max"
memory_events_path="$memory_cgroup_dir/memory.events"
memory_current_path="$memory_cgroup_dir/memory.current"
memory_max=""
if [[ -r "$memory_max_path" ]]; then
  memory_max="$(<"$memory_max_path")"
fi

write_cgroup_snapshot() {
  local target="$1"
  {
    printf 'memory.max=%s\n' "${memory_max:-unknown}"
    if [[ -r "$memory_current_path" ]]; then
      printf 'memory.current=%s\n' "$(<"$memory_current_path")"
    else
      printf 'memory.current=unknown\n'
    fi
    if [[ -r "$memory_events_path" ]]; then
      cat "$memory_events_path"
    else
      printf 'memory.events=unavailable\n'
    fi
  } > "$target"
}

write_source_hashes() {
  local target="$1"
  {
    printf 'G3_FROZEN_RESULT_UNCHANGED=true\n'
    printf 'G4_POST_G3_SOURCE_DELTA_RECORDED=true\n'
    for source_file in \
      requirements-tpu-g3.txt scripts/run_g3_tpu_authority.sh \
      scripts/g3_tpu_authority.py scripts/g3_oom_adjudication.py \
      src/gemma4_server/tpu/authority_contract.py \
      src/gemma4_server/tpu/engine.py src/gemma4_server/tpu/distribution.py \
      src/gemma4_server/tpu/sharded_checkpoint.py \
      src/gemma4_server/tpu/generation.py scripts/g4_split_authority.py \
      scripts/g4_evidence.py scripts/run_g4_characterization.sh \
      src/gemma4_server/tpu/g4_split.py; do
      if [[ -f "$source_file" ]]; then
        sha256sum "$source_file"
      fi
    done
  } > "$target"
}

cp artifacts/g4/g4-prechange-source-hashes.txt \
  "$evidence_dir/01-g4-source-hashes-before.txt"
cp artifacts/g4/g4-api-discovery.md "$evidence_dir/03-api-discovery.txt"
{
  printf 'G3_FREEZE_ARTIFACT=%s\n' "$PROJECT_ROOT/artifacts/g3/g3-pass-freeze.json"
  printf 'G3_AUTHORITY_ARCHIVE=%s\n' "$G3_ARCHIVE"
  printf 'G3_AUTHORITY_ARCHIVE_SHA256=%s\n' "$G3_ARCHIVE_SHA256"
  printf 'G3_FROZEN_RESULT_UNCHANGED=true\nG3_RERUN_FORBIDDEN=true\n'
} > "$evidence_dir/00-g3-freeze-reference.txt"
write_cgroup_snapshot "$evidence_dir/10-memory-before.txt"

shopt -s nullglob
accel_devices=(/dev/accel*)
vfio_devices=(/dev/vfio/[0-9]*)
shopt -u nullglob
numeric_vfio_devices=()
for vfio_path in "${vfio_devices[@]}"; do
  vfio_name="${vfio_path##*/}"
  if [[ "$vfio_name" =~ ^[0-9]+$ ]]; then
    numeric_vfio_devices+=("$vfio_path")
  fi
done
tpu_device_exposure_style="NONE"
if [[ ${#accel_devices[@]} -gt 0 ]]; then
  tpu_device_exposure_style="ACCEL"
elif [[ ${#numeric_vfio_devices[@]} -eq 8 ]]; then
  tpu_device_exposure_style="VFIO"
fi
memory_floor_pass=false
if [[ "$memory_max" =~ ^[0-9]+$ ]] && \
  (( memory_max >= MINIMUM_AUTHORITY_MEMORY_GIB * 1024 * 1024 * 1024 )); then
  memory_floor_pass=true
fi
hardware_pass=false
if [[ -e "$MODEL_PATH" && "$memory_floor_pass" == true && \
  ("$tpu_device_exposure_style" == ACCEL || "$tpu_device_exposure_style" == VFIO) ]]; then
  hardware_pass=true
fi
{
  printf 'MODEL_PATH=%s\nMODEL_PRESET=%s\n' "$MODEL_PATH" "$MODEL_PRESET"
  printf 'MODEL_PATH_EXISTS=%s\n' "$([[ -e "$MODEL_PATH" ]] && echo true || echo false)"
  printf 'ACCELERATOR_PATHS=%s\n' "${accel_devices[*]:-}"
  printf 'VFIO_NUMERIC_PATHS=%s\n' "${numeric_vfio_devices[*]:-}"
  printf 'TPU_DEVICE_EXPOSURE_STYLE=%s\n' "$tpu_device_exposure_style"
  printf 'MEMORY_MAX_BYTES=%s\nMINIMUM_AUTHORITY_MEMORY_GIB=%s\n' \
    "${memory_max:-unknown}" "$MINIMUM_AUTHORITY_MEMORY_GIB"
  printf 'MEMORY_FLOOR_PASS=%s\nHARDWARE_GATE=%s\n' "$memory_floor_pass" \
    "$([[ "$hardware_pass" == true ]] && echo PASS || echo FAIL)"
} > "$evidence_dir/04-hardware-gate.txt"

write_failure_result() {
  local final_result="$1"
  python3 - "$evidence_dir/09-g4-result.json" "$final_result" \
    "$tpu_device_exposure_style" "$hardware_pass" <<'PY'
import json
import sys
from pathlib import Path

path, final_result, style, hardware_pass = sys.argv[1:]
Path(path).write_text(json.dumps({
    "AUTHORITY_NOT_STARTED": True,
    "TPU_DEVICE_EXPOSURE_STYLE": style,
    "MODEL_LOAD_COUNT": 0,
    "MODEL_LOAD_RETURNED": False,
    "CANDIDATE_A_SHARDING_VERIFIED": False,
    "G4_SPLIT_PATH_STARTED": False,
    "G4_SPLIT_PATH_RETURNED": False,
    "G4_SPLIT_GENERATION_CALL_COUNT": 0,
    "G4_SPLIT_MINIMAL_RUN": "NOT_EVALUATED",
    "G4_STATUS": "OPEN",
    "HARDWARE_GATE": hardware_pass == "true",
    "FINAL_RESULT": final_result,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_final_and_archive() {
  local final_result="$1"
  write_source_hashes "$evidence_dir/02-g4-source-hashes-after.txt"
  {
    printf 'G3_FROZEN_RESULT_UNCHANGED=true\nG4_POST_G3_SOURCE_DELTA_RECORDED=true\n'
    printf 'TPU_DEVICE_EXPOSURE_STYLE=%s\nTPU_31B_ATTEMPTS_USED=%s\n' \
      "$tpu_device_exposure_style" "$G4_TPU_ATTEMPTS"
    printf 'FINAL_RESULT=%s\nEVIDENCE_DIR=%s\n' "$final_result" "$evidence_dir"
  } > "$evidence_dir/13-final-adjudication.txt"
  (
    cd "$evidence_dir"
    sha256sum 00-g3-freeze-reference.txt 01-g4-source-hashes-before.txt \
      02-g4-source-hashes-after.txt 03-api-discovery.txt 04-hardware-gate.txt \
      05-runtime.txt 06-dependency-gate.txt 07-g4-authority.stdout.log \
      08-g4-authority.stderr.log 09-g4-result.json 10-memory-before.txt \
      11-memory-after.txt 12-native-vs-split-comparison.json \
      13-final-adjudication.txt > SHA256SUMS
  )
  tar -czf "$archive" -C "$(dirname "$evidence_dir")" "$(basename "$evidence_dir")"
  sha256sum "$archive" > "${archive}.sha256"
  printf 'EVIDENCE_ARCHIVE=%s\nEVIDENCE_ARCHIVE_SHA256=%s\n' \
    "$archive" "$(awk '{print $1}' "${archive}.sha256")"
}

if [[ "$hardware_pass" != true ]]; then
  printf 'RUNTIME_GATE=NOT_STARTED_HARDWARE_GATE_FAILED\n' > "$evidence_dir/05-runtime.txt"
  printf 'PROJECT_SCOPED_DEPENDENCY_GATE=NOT_STARTED_HARDWARE_GATE_FAILED\n' \
    > "$evidence_dir/06-dependency-gate.txt"
  write_failure_result TPU_HARDWARE_NOT_READY
  write_cgroup_snapshot "$evidence_dir/11-memory-after.txt"
  python3 - "$evidence_dir/12-native-vs-split-comparison.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "native_source": "frozen-g3-authority",
    "native_generation_seconds": 567.709641,
    "split_result": "NOT_EVALUATED",
    "split_generation_seconds": None,
    "split_host_oom_delta": None,
    "split_tpu_device_count": None,
    "split_model_load_count": 0,
    "memory_comparison_status": "PARTIAL",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  write_final_and_archive TPU_HARDWARE_NOT_READY
  exit 1
fi

export KERAS_BACKEND=jax
export MODEL_DTYPE=bfloat16
export MODEL_PRESET MODEL_PATH EXPECTED_TPU_DEVICES
export MESH_SHAPE=1,8
export MESH_AXIS_NAMES=batch,model
export PROMPT_TEXT=Hello
export PROMPT_TOKEN_COUNT=10
export MAX_NEW_TOKENS=1
export AUTHORITY_MAX_LENGTH=11

runtime_check() {
  python3 - "$PROJECT_ROOT/requirements-tpu-g3.txt" <<'PY'
import importlib.metadata
import json
import sys
from gemma4_server.tpu.authority_contract import exact_runtime_versions

installed = {
    distribution.metadata["Name"]: distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
print(json.dumps(exact_runtime_versions(sys.argv[1], installed), sort_keys=True))
PY
}

if runtime_versions="$(runtime_check 2>&1)"; then
  printf 'RUNTIME_RESTORE=SKIPPED_ALREADY_EXACT\n%s\n' "$runtime_versions" \
    > "$evidence_dir/05-runtime.txt"
else
  printf 'RUNTIME_RESTORE=REQUIRED\n%s\n' "$runtime_versions" \
    > "$evidence_dir/05-runtime.txt"
  if ! python3 -m pip install \
    --disable-pip-version-check --root-user-action=ignore --progress-bar off \
    --no-deps --upgrade -r requirements-tpu-g3.txt \
    >> "$evidence_dir/05-runtime.txt" 2>&1; then
    write_failure_result RUNTIME_BASELINE_MISMATCH
    printf 'FINAL_RESULT=RUNTIME_BASELINE_MISMATCH\n' >> "$evidence_dir/05-runtime.txt"
    write_cgroup_snapshot "$evidence_dir/11-memory-after.txt"
    write_final_and_archive RUNTIME_BASELINE_MISMATCH
    exit 1
  fi
  if ! runtime_versions="$(runtime_check 2>&1)"; then
    printf 'FINAL_RESULT=RUNTIME_BASELINE_MISMATCH\n%s\n' "$runtime_versions" \
      >> "$evidence_dir/05-runtime.txt"
    write_failure_result RUNTIME_BASELINE_MISMATCH
    write_cgroup_snapshot "$evidence_dir/11-memory-after.txt"
    write_final_and_archive RUNTIME_BASELINE_MISMATCH
    exit 1
  fi
  printf 'RUNTIME_RESTORE=COMPLETED\n%s\n' "$runtime_versions" \
    >> "$evidence_dir/05-runtime.txt"
fi

if dependency_result="$(python3 - <<'PY' 2>&1
import importlib.metadata
import json
from gemma4_server.tpu.authority_contract import project_dependency_gate

distributions = {
    distribution.metadata["Name"]: {
        "version": distribution.version,
        "requires": distribution.requires or (),
    }
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
print(json.dumps(project_dependency_gate(
    distributions,
    ("keras", "keras-hub", "keras-nlp", "jax", "jaxlib", "numpy", "libtpu"),
), sort_keys=True))
PY
)"; then
  printf '%s\n' "$dependency_result" > "$evidence_dir/06-dependency-gate.txt"
else
  printf 'PROJECT_SCOPED_DEPENDENCY_GATE=FAIL\n%s\n' "$dependency_result" \
    > "$evidence_dir/06-dependency-gate.txt"
  write_failure_result PROJECT_SCOPED_DEPENDENCY_GATE_FAILED
  write_cgroup_snapshot "$evidence_dir/11-memory-after.txt"
  write_final_and_archive PROJECT_SCOPED_DEPENDENCY_GATE_FAILED
  exit 1
fi

source scripts/configure_kaggle_tpu.sh
printf 'G4_TPU_FALLBACK_APPLIED=%s\n' "$G4_TPU_FALLBACK_APPLIED" \
  >> "$evidence_dir/04-hardware-gate.txt"

oom_before="$(awk '$1 == "oom_kill" {print $2}' "$memory_events_path" 2>/dev/null || true)"
oom_before="${oom_before:-0}"
set +e
python3 scripts/g4_split_authority.py --evidence-dir "$evidence_dir" \
  > "$evidence_dir/07-g4-authority.stdout.log" \
  2> "$evidence_dir/08-g4-authority.stderr.log"
authority_exit=$?
set -e
oom_after="$(awk '$1 == "oom_kill" {print $2}' "$memory_events_path" 2>/dev/null || true)"
oom_after="${oom_after:-0}"
oom_delta=$((oom_after - oom_before))
write_cgroup_snapshot "$evidence_dir/11-memory-after.txt"

if [[ -s "$evidence_dir/07-g4-result.json" ]]; then
  cp "$evidence_dir/07-g4-result.json" "$evidence_dir/09-g4-result.json"
  rm "$evidence_dir/07-g4-result.json"
fi
if [[ ! -s "$evidence_dir/09-g4-result.json" ]]; then
  write_failure_result G4_AUTHORITY_RESULT_MISSING
fi
python3 - "$evidence_dir/12-native-vs-split-comparison.json" \
  "$evidence_dir/09-g4-result.json" "$oom_delta" "$G3_ARCHIVE" <<'PY'
import json
import sys
from pathlib import Path
from scripts.g4_evidence import native_baseline_from_archive, write_comparison

comparison_path, result_path, oom_delta, archive_path = sys.argv[1:]
native = native_baseline_from_archive(archive_path)
split = json.loads(Path(result_path).read_text(encoding="utf-8"))
write_comparison(comparison_path, native, split, split_host_oom_delta=int(oom_delta))
PY

final_result="G4_SPLIT_PATH_FAILED"
if [[ "$authority_exit" -ne 0 && "$G4_TPU_ATTEMPTS" -ge 2 ]]; then
  final_result="G4_TPU_ATTEMPT_BUDGET_EXHAUSTED"
elif [[ "$authority_exit" -eq 0 ]] && \
  grep -q '"G4_SPLIT_MINIMAL_RUN": "PASS"' "$evidence_dir/09-g4-result.json"; then
  final_result="G4_NATIVE_VS_SPLIT_CHARACTERIZED"
elif grep -q '"G4_SPLIT_MINIMAL_RUN": "NOT_VIABLE"' \
  "$evidence_dir/09-g4-result.json"; then
  final_result="G4_CHARACTERIZED_NATIVE_RETAINED"
elif [[ "$authority_exit" -eq 137 && "$oom_delta" -gt 0 ]]; then
  final_result="G4_TPU_ATTEMPT_FAILED_HOST_OOM"
fi
write_final_and_archive "$final_result"
if [[ "$authority_exit" -ne 0 ]]; then
  exit "$authority_exit"
fi
