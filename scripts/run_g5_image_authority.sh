#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_31b/2"
MODEL_PRESET="gemma4_instruct_31b"
MINIMUM_AUTHORITY_MEMORY_GIB=300
EXPECTED_TPU_DEVICES=8
PRIMARY_G5_TPU_ATTEMPTS=1
CORRECTIVE_G5_TPU_RERUNS_ALLOWED=1
MAX_TOTAL_G5_31B_TPU_ATTEMPTS=2
G5_TPU_ATTEMPT_NUMBER="${G5_TPU_ATTEMPT_NUMBER:-1}"
G4_FREEZE_MANIFEST="$PROJECT_ROOT/artifacts/g4/g4-pass-freeze.json"
G4_ARCHIVE_SHA256="5ae0609df40393aaaf719c9615715610c176cbbd1f6584e987f3b6aa6a3bc662"

if ! [[ "$G5_TPU_ATTEMPT_NUMBER" =~ ^[12]$ ]]; then
  echo "G5_TPU_ATTEMPT_NUMBER must be 1 or 2" >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
evidence_dir="/kaggle/working/gemma4-31b-g5-image-authority-${stamp}"
archive="/kaggle/working/gemma4-31b-g5-image-authority-${stamp}.tar.gz"
fixture_path="$evidence_dir/g5-fixture.png"
mkdir -p "$evidence_dir"
for file in \
  00-g4-freeze-reference.txt 01-g5-source-hashes-before.txt \
  02-g5-source-hashes-after.txt 03-g5-api-discovery.txt \
  04-image-fixture.txt 05-hardware-gate.txt 06-runtime.txt \
  07-dependency-gate.txt 08-memory-before.txt \
  09-g5-authority.stdout.log 10-g5-authority.stderr.log \
  11-g5-result.json 12-memory-after.txt 13-final-adjudication.txt; do
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
    printf 'POST_G4_SOURCE_CHANGE=true\n'
    printf 'G4_FREEZE_MANIFEST=%s\n' "$G4_FREEZE_MANIFEST"
    for source_file in \
      src/gemma4_server/tpu/g5_vision.py \
      scripts/g5_image_authority.py scripts/run_g5_image_authority.sh \
      tests/test_g5_contract.py artifacts/g5/g5-api-discovery.md; do
      if [[ -f "$source_file" ]]; then
        sha256sum "$source_file"
      fi
    done
  } > "$target"
}

write_g4_reference() {
  {
    printf 'G4_FREEZE_MANIFEST=%s\n' "$G4_FREEZE_MANIFEST"
    printf 'G4_AUTHORITY_ARCHIVE=/kaggle/working/gemma4-31b-g4-characterization-20260914T003028Z.tar.gz\n'
    printf 'G4_AUTHORITY_ARCHIVE_SHA256=%s\n' "$G4_ARCHIVE_SHA256"
    printf 'G4_FREEZE_VERIFICATION=PASS\nG4_STATUS=CLOSED\n'
    printf 'G4_RECOMMENDED_PATH=SPLIT\nG4_IMMUTABLE=true\nG4_RERUN_FORBIDDEN=true\n'
  } > "$evidence_dir/00-g4-freeze-reference.txt"
}

write_failure_result() {
  local final_result="$1"
  python3 - "$evidence_dir/11-g5-result.json" "$final_result" \
    "$tpu_device_exposure_style" "$hardware_pass" "$G5_TPU_ATTEMPT_NUMBER" <<'PY'
import json
import sys
from pathlib import Path

path, final_result, style, hardware_pass, attempt = sys.argv[1:]
Path(path).write_text(json.dumps({
    "AUTHORITY_NOT_STARTED": True,
    "G5_PATH": "NATIVE_VISION",
    "TPU_DEVICE_EXPOSURE_STYLE": style,
    "TPU_DEVICE_COUNT": None,
    "MODEL_LOAD_COUNT": 0,
    "MODEL_LOAD_RETURNED": False,
    "CANDIDATE_A_SHARDING_VERIFIED": False,
    "IMAGE_INPUT_LOADED": False,
    "IMAGE_PREPROCESS_RETURNED": False,
    "VISION_CONDITIONING_PRESENT": False,
    "G5_GENERATION_STARTED": False,
    "G5_GENERATION_RETURNED": False,
    "G5_GENERATION_CALL_COUNT": 0,
    "G5_IMAGE_GENERATION": "NOT_EVALUATED",
    "G5_STATUS": "OPEN",
    "HARDWARE_GATE": hardware_pass == "true",
    "G5_TPU_31B_ATTEMPTS_USED": int(attempt),
    "G5_CORRECTIVE_TPU_RERUN_USED": int(attempt) == 2,
    "FINAL_RESULT": final_result,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_final_and_archive() {
  local final_result="$1"
  local authority_exit="$2"
  write_source_hashes "$evidence_dir/02-g5-source-hashes-after.txt"
  python3 - "$evidence_dir/11-g5-result.json" "$final_result" \
    "$G5_TPU_ATTEMPT_NUMBER" "$authority_exit" <<'PY'
import json
import sys
from pathlib import Path

path, final_result, attempt, authority_exit = sys.argv[1:]
target = Path(path)
result = json.loads(target.read_text(encoding="utf-8"))
result.update({
    "G5_CPU_PREPARATION": True,
    "G5_TPU_31B_ATTEMPTS_USED": int(attempt),
    "G5_CORRECTIVE_TPU_RERUN_USED": int(attempt) == 2,
    "AUTHORITY_EXIT_CODE": int(authority_exit),
    "FINAL_RESULT": final_result,
})
target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  {
    printf 'G5_CPU_PREPARATION=PASS\n'
    printf 'G5_PATH=NATIVE_VISION\n'
    printf 'G5_TPU_31B_ATTEMPTS_USED=%s\n' "$G5_TPU_ATTEMPT_NUMBER"
    printf 'G5_CORRECTIVE_TPU_RERUN_USED=%s\n' \
      "$([[ "$G5_TPU_ATTEMPT_NUMBER" == 2 ]] && echo true || echo false)"
    printf 'FINAL_RESULT=%s\nEVIDENCE_DIR=%s\n' "$final_result" "$evidence_dir"
    if [[ "$final_result" == "G5_IMAGE_CONDITIONED_GENERATION_PASS" ]]; then
      printf 'G5_IMAGE_GENERATION=PASS\nG5_STATUS=CLOSED\nG6_ENTRY_ELIGIBLE=true\n'
    else
      printf 'G5_IMAGE_GENERATION=OPEN\nG5_STATUS=OPEN\nG6_ENTRY_ELIGIBLE=false\n'
    fi
  } > "$evidence_dir/13-final-adjudication.txt"
  (
    cd "$evidence_dir"
    sha256sum 00-g4-freeze-reference.txt 01-g5-source-hashes-before.txt \
      02-g5-source-hashes-after.txt 03-g5-api-discovery.txt \
      04-image-fixture.txt 05-hardware-gate.txt 06-runtime.txt \
      07-dependency-gate.txt 08-memory-before.txt \
      09-g5-authority.stdout.log 10-g5-authority.stderr.log \
      11-g5-result.json 12-memory-after.txt 13-final-adjudication.txt \
      > SHA256SUMS
  )
  tar -czf "$archive" -C "$(dirname "$evidence_dir")" "$(basename "$evidence_dir")"
  sha256sum "$archive" > "${archive}.sha256"
  printf 'EVIDENCE_ARCHIVE=%s\nEVIDENCE_ARCHIVE_SHA256=%s\n' \
    "$archive" "$(awk '{print $1}' "${archive}.sha256")"
}

write_g4_reference
write_source_hashes "$evidence_dir/01-g5-source-hashes-before.txt"
cp artifacts/g5/g5-api-discovery.md "$evidence_dir/03-g5-api-discovery.txt"
write_cgroup_snapshot "$evidence_dir/08-memory-before.txt"

cpu_log="$(mktemp)"
cpu_status=0
python3 -m unittest discover -s tests -p 'test_*.py' > "$cpu_log" 2>&1 || cpu_status=$?
python3 -m compileall -q src scripts clients/python >> "$cpu_log" 2>&1 || cpu_status=$?
for shell_file in scripts/*.sh; do
  bash -n "$shell_file" >> "$cpu_log" 2>&1 || cpu_status=$?
done
if [[ "$cpu_status" -ne 0 ]]; then
  printf 'G5_CPU_PREPARATION=FAIL\n%s\n' "$(<"$cpu_log")" \
    > "$evidence_dir/13-final-adjudication.txt"
  rm -f "$cpu_log"
  tpu_device_exposure_style="NONE"
  hardware_pass=false
  write_failure_result G5_CPU_PREPARATION_FAILED
  write_cgroup_snapshot "$evidence_dir/12-memory-after.txt"
  printf 'RUNTIME_GATE=NOT_STARTED_CPU_PREPARATION_FAILED\n' > "$evidence_dir/06-runtime.txt"
  printf 'PROJECT_SCOPED_DEPENDENCY_GATE=NOT_STARTED_CPU_PREPARATION_FAILED\n' \
    > "$evidence_dir/07-dependency-gate.txt"
  write_final_and_archive G5_CPU_PREPARATION_FAILED "$cpu_status"
  exit "$cpu_status"
fi
rm -f "$cpu_log"

python3 - "$fixture_path" <<'PY' > "$evidence_dir/04-image-fixture.txt"
import sys
from gemma4_server.tpu.g5_vision import create_synthetic_fixture, load_fixture_rgb

path = sys.argv[1]
digest = create_synthetic_fixture(path)
image = load_fixture_rgb(path)
print(f"IMAGE_FIXTURE_PATH={path}")
print(f"IMAGE_FIXTURE_SHA256={digest}")
print(f"IMAGE_FIXTURE_SHAPE={list(image.shape)}")
print(f"IMAGE_FIXTURE_DTYPE={image.dtype}")
print("IMAGE_FIXTURE_NETWORK_DOWNLOAD=false")
PY
fixture_sha256="$(sed -n 's/^IMAGE_FIXTURE_SHA256=//p' "$evidence_dir/04-image-fixture.txt")"

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
  printf 'G5_CPU_PREPARATION=PASS\n'
} > "$evidence_dir/05-hardware-gate.txt"

if [[ "$hardware_pass" != true ]]; then
  printf 'RUNTIME_GATE=NOT_STARTED_HARDWARE_GATE_FAILED\n' > "$evidence_dir/06-runtime.txt"
  printf 'PROJECT_SCOPED_DEPENDENCY_GATE=NOT_STARTED_HARDWARE_GATE_FAILED\n' \
    > "$evidence_dir/07-dependency-gate.txt"
  write_failure_result TPU_HARDWARE_NOT_READY
  write_cgroup_snapshot "$evidence_dir/12-memory-after.txt"
  write_final_and_archive TPU_HARDWARE_NOT_READY 1
  exit 1
fi

export KERAS_BACKEND=jax
export MODEL_DTYPE=bfloat16
export MODEL_PRESET MODEL_PATH EXPECTED_TPU_DEVICES
export MESH_SHAPE=1,8
export MESH_AXIS_NAMES=batch,model
export G5_IMAGE_FIXTURE_PATH="$fixture_path"
export G5_IMAGE_FIXTURE_SHA256="$fixture_sha256"

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
    > "$evidence_dir/06-runtime.txt"
else
  printf 'RUNTIME_RESTORE=REQUIRED\n%s\n' "$runtime_versions" \
    > "$evidence_dir/06-runtime.txt"
  write_failure_result RUNTIME_BASELINE_MISMATCH
  write_cgroup_snapshot "$evidence_dir/12-memory-after.txt"
  write_final_and_archive RUNTIME_BASELINE_MISMATCH 1
  exit 1
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
  printf '%s\n' "$dependency_result" > "$evidence_dir/07-dependency-gate.txt"
else
  printf 'PROJECT_SCOPED_DEPENDENCY_GATE=FAIL\n%s\n' "$dependency_result" \
    > "$evidence_dir/07-dependency-gate.txt"
  write_failure_result PROJECT_SCOPED_DEPENDENCY_GATE_FAILED
  write_cgroup_snapshot "$evidence_dir/12-memory-after.txt"
  write_final_and_archive PROJECT_SCOPED_DEPENDENCY_GATE_FAILED 1
  exit 1
fi

source scripts/configure_kaggle_tpu.sh
printf 'G4_TPU_FALLBACK_APPLIED=%s\n' "$G4_TPU_FALLBACK_APPLIED" \
  >> "$evidence_dir/05-hardware-gate.txt"
oom_before="$(awk '$1 == "oom_kill" {print $2}' "$memory_events_path" 2>/dev/null || true)"
oom_before="${oom_before:-0}"
set +e
python3 scripts/g5_image_authority.py --evidence-dir "$evidence_dir" \
  > "$evidence_dir/09-g5-authority.stdout.log" \
  2> "$evidence_dir/10-g5-authority.stderr.log"
authority_exit=$?
set -e
oom_after="$(awk '$1 == "oom_kill" {print $2}' "$memory_events_path" 2>/dev/null || true)"
oom_after="${oom_after:-0}"
oom_delta=$((oom_after - oom_before))
write_cgroup_snapshot "$evidence_dir/12-memory-after.txt"

python3 - "$evidence_dir/11-g5-result.json" "$tpu_device_exposure_style" \
  "$oom_delta" "$G5_TPU_ATTEMPT_NUMBER" <<'PY'
import json
import sys
from pathlib import Path

path, style, oom_delta, attempt = sys.argv[1:]
target = Path(path)
result = json.loads(target.read_text(encoding="utf-8")) if target.stat().st_size else {}
result["TPU_DEVICE_EXPOSURE_STYLE"] = style
result["G5_HOST_OOM_KILL_DELTA"] = int(oom_delta)
result["G5_TPU_31B_ATTEMPTS_USED"] = int(attempt)
result["G5_CORRECTIVE_TPU_RERUN_USED"] = int(attempt) == 2
target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

final_result="G5_IMAGE_GENERATION_FAILED"
if [[ "$authority_exit" -eq 0 ]] && \
  grep -q '"G5_IMAGE_GENERATION": "PASS"' "$evidence_dir/11-g5-result.json" && \
  grep -q '"VISION_CONDITIONING_PRESENT": true' "$evidence_dir/11-g5-result.json" && \
  grep -q '"G5_GENERATION_RETURNED": true' "$evidence_dir/11-g5-result.json"; then
  final_result="G5_IMAGE_CONDITIONED_GENERATION_PASS"
elif [[ "$G5_TPU_ATTEMPT_NUMBER" -ge 2 ]]; then
  final_result="G5_TPU_ATTEMPT_BUDGET_EXHAUSTED"
elif [[ "$authority_exit" -eq 137 && "$oom_delta" -gt 0 ]]; then
  final_result="G5_TPU_ATTEMPT_FAILED_HOST_OOM"
fi
write_final_and_archive "$final_result" "$authority_exit"
if [[ "$authority_exit" -ne 0 ]]; then
  exit "$authority_exit"
fi
