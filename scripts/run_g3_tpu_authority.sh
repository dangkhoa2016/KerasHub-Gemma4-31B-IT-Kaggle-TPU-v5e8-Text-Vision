#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_31b/2"
MODEL_PRESET="gemma4_instruct_31b"
MINIMUM_AUTHORITY_MEMORY_GIB=300
EXPECTED_TPU_DEVICES=8

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
evidence_dir="/kaggle/working/gemma4-31b-vnext-g3-authority-${stamp}"
archive="/kaggle/working/gemma4-31b-vnext-g3-authority-${stamp}.tar.gz"
mkdir -p "$evidence_dir"
for file in \
  00-source-hashes.txt 01-hardware-gate.txt 02-runtime-versions.txt \
  03-project-dependency-gate.txt 04-cgroup-before.txt 05-authority.stdout.log \
  06-authority.stderr.log 07-authority-result.json 08-cgroup-after.txt \
  09-final-adjudication.txt; do
  : > "$evidence_dir/$file"
done
printf '{}\n' > "$evidence_dir/07-authority-result.json"

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
shopt -s nullglob
accel_devices=(/dev/accel*)
vfio_devices=(/dev/vfio/[0-9]*)
shopt -u nullglob
tpu_devices=("${accel_devices[@]}" "${vfio_devices[@]}")

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

write_final_and_archive() {
  local result="$1"
  {
    for source_file in requirements-tpu-g3.txt scripts/run_g3_tpu_authority.sh \
      scripts/g3_tpu_authority.py scripts/g3_oom_adjudication.py \
      src/gemma4_server/tpu/authority_contract.py \
      src/gemma4_server/tpu/engine.py src/gemma4_server/tpu/distribution.py \
      src/gemma4_server/tpu/sharded_checkpoint.py src/gemma4_server/tpu/generation.py; do
      if [[ -f "$source_file" ]]; then
        sha256sum "$source_file"
      fi
    done
  } > "$evidence_dir/00-source-hashes.txt"
  {
    printf 'AUTHORITY_NOT_STARTED=%s\n' "${AUTHORITY_NOT_STARTED:-false}"
    printf 'FINAL_RESULT=%s\n' "$result"
    printf 'G3_TEXT_GENERATION=%s\n' "${g3_text_generation:-NOT_EVALUATED}"
    printf 'G3_STATUS=%s\n' "${g3_status:-OPEN}"
    printf 'G4_ENTRY_ELIGIBLE=%s\n' "${g4_entry_eligible:-false}"
    printf 'G4_STARTED=%s\n' "${g4_started:-false}"
    printf 'GENERATION_STARTED=%s\n' "${generation_started:-unknown}"
    printf 'EVIDENCE_DIR=%s\n' "$evidence_dir"
  } > "$evidence_dir/09-final-adjudication.txt"
  {
    cd "$evidence_dir"
    sha256sum 00-source-hashes.txt 01-hardware-gate.txt 02-runtime-versions.txt \
      03-project-dependency-gate.txt 04-cgroup-before.txt 05-authority.stdout.log \
      06-authority.stderr.log 07-authority-result.json 08-cgroup-after.txt \
      09-final-adjudication.txt > SHA256SUMS
  }
  tar -czf "$archive" -C "$(dirname "$evidence_dir")" "$(basename "$evidence_dir")"
  sha256sum "$archive" > "${archive}.sha256"
  printf 'EVIDENCE_ARCHIVE=%s\nEVIDENCE_ARCHIVE_SHA256=%s\n' \
    "$archive" "${archive}.sha256"
}

write_cgroup_snapshot "$evidence_dir/04-cgroup-before.txt"
oom_before="$(awk '$1 == "oom_kill" {print $2}' "$memory_events_path" 2>/dev/null || true)"
oom_before="${oom_before:-0}"
{
  printf 'MODEL_PATH=%s\n' "$MODEL_PATH"
  printf 'MODEL_PATH_EXISTS=%s\n' "$([[ -e "$MODEL_PATH" ]] && echo true || echo false)"
  printf 'ACCELERATOR_PATHS=%s\n' "${tpu_devices[*]:-}"
  printf 'ACCELERATOR_PRESENT=%s\n' "$([[ ${#tpu_devices[@]} -gt 0 ]] && echo true || echo false)"
  printf 'MEMORY_MAX_BYTES=%s\n' "${memory_max:-unknown}"
  printf 'MINIMUM_AUTHORITY_MEMORY_GIB=%s\n' "$MINIMUM_AUTHORITY_MEMORY_GIB"
} > "$evidence_dir/01-hardware-gate.txt"

memory_floor_pass=false
if [[ "$memory_max" =~ ^[0-9]+$ ]] && (( memory_max >= MINIMUM_AUTHORITY_MEMORY_GIB * 1024 * 1024 * 1024 )); then
  memory_floor_pass=true
fi
if [[ ! -e "$MODEL_PATH" || ${#tpu_devices[@]} -eq 0 || "$memory_floor_pass" != true ]]; then
  AUTHORITY_NOT_STARTED=true
  export AUTHORITY_NOT_STARTED
  printf 'AUTHORITY_NOT_STARTED=true\nFINAL_RESULT=TPU_HARDWARE_NOT_READY\n' \
    > "$evidence_dir/09-final-adjudication.txt"
  printf 'AUTHORITY_NOT_STARTED=true\nFINAL_RESULT=TPU_HARDWARE_NOT_READY\n'
  write_final_and_archive TPU_HARDWARE_NOT_READY
  exit 1
fi

export KERAS_BACKEND=jax
export MODEL_DTYPE=bfloat16
export MODEL_PRESET
export MODEL_PATH
export EXPECTED_TPU_DEVICES
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
result = exact_runtime_versions(sys.argv[1], installed)
print(json.dumps(result, sort_keys=True))
PY
}

if runtime_versions="$(runtime_check 2>&1)"; then
  printf 'RUNTIME_RESTORE=SKIPPED_ALREADY_EXACT\n%s\n' "$runtime_versions" \
    > "$evidence_dir/02-runtime-versions.txt"
else
  printf 'RUNTIME_RESTORE=REQUIRED\n%s\n' "$runtime_versions" \
    > "$evidence_dir/02-runtime-versions.txt"
  if ! python3 -m pip install \
    --disable-pip-version-check \
    --root-user-action=ignore \
    --progress-bar off \
    --no-deps \
    --upgrade \
    -r requirements-tpu-g3.txt >> "$evidence_dir/02-runtime-versions.txt" 2>&1; then
    AUTHORITY_NOT_STARTED=true
    export AUTHORITY_NOT_STARTED
    write_final_and_archive RUNTIME_BASELINE_MISMATCH
    exit 1
  fi
  if ! runtime_versions="$(runtime_check 2>&1)"; then
    printf 'FINAL_RESULT=RUNTIME_BASELINE_MISMATCH\n%s\n' "$runtime_versions" \
      >> "$evidence_dir/02-runtime-versions.txt"
    AUTHORITY_NOT_STARTED=true
    export AUTHORITY_NOT_STARTED
    write_final_and_archive RUNTIME_BASELINE_MISMATCH
    exit 1
  fi
  printf 'RUNTIME_RESTORE=COMPLETED\n%s\n' "$runtime_versions" \
    >> "$evidence_dir/02-runtime-versions.txt"
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
result = project_dependency_gate(
    distributions,
    ("keras", "keras-hub", "keras-nlp", "jax", "jaxlib", "numpy", "libtpu"),
)
print(json.dumps(result, sort_keys=True))
PY
)"; then
  printf '%s\n' "$dependency_result" > "$evidence_dir/03-project-dependency-gate.txt"
else
  printf 'PROJECT_SCOPED_DEPENDENCY_GATE=FAIL\n%s\n' "$dependency_result" \
    > "$evidence_dir/03-project-dependency-gate.txt"
  AUTHORITY_NOT_STARTED=true
  export AUTHORITY_NOT_STARTED
  write_final_and_archive PROJECT_SCOPED_DEPENDENCY_GATE_FAILED
  exit 1
fi

source scripts/configure_kaggle_tpu.sh
set +e
python3 scripts/g3_tpu_authority.py --evidence-dir "$evidence_dir" \
  > "$evidence_dir/05-authority.stdout.log" \
  2> "$evidence_dir/06-authority.stderr.log"
authority_exit=$?
set -e
write_cgroup_snapshot "$evidence_dir/08-cgroup-after.txt"

oom_after="$(awk '$1 == "oom_kill" {print $2}' "$memory_events_path" 2>/dev/null || true)"
oom_after="${oom_after:-0}"
oom_delta=$((oom_after - oom_before))
final_result="$(sed -n 's/^  *"FINAL_RESULT": *"\([^"]*\)".*/\1/p' "$evidence_dir/07-authority-result.json" | head -n 1)"
adjudication="$(python3 scripts/g3_oom_adjudication.py \
  --authority-exit "$authority_exit" \
  --oom-delta "$oom_delta" \
  --result-json "$evidence_dir/07-authority-result.json")"
while IFS='=' read -r key value; do
  case "$key" in
    FINAL_RESULT) final_result="$value" ;;
    GENERATION_STARTED) generation_started="$value" ;;
    G3_TEXT_GENERATION) g3_text_generation="$value" ;;
    G3_STATUS) g3_status="$value" ;;
    G4_ENTRY_ELIGIBLE) g4_entry_eligible="$value" ;;
    G4_STARTED) g4_started="$value" ;;
  esac
done <<< "$adjudication"
AUTHORITY_NOT_STARTED=false
export AUTHORITY_NOT_STARTED
write_final_and_archive "$final_result"
printf 'FINAL_RESULT=%s\nG3_TEXT_GENERATION=%s\nG3_STATUS=%s\nG4_ENTRY_ELIGIBLE=%s\nG4_STARTED=%s\nGENERATION_STARTED=%s\n' \
  "$final_result" "$g3_text_generation" "$g3_status" "$g4_entry_eligible" \
  "$g4_started" "$generation_started"
exit "$authority_exit"
