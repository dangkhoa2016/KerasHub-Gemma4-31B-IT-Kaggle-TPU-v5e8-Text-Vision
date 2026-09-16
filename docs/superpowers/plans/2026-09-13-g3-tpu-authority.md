# G3 TPU Authority Integrated Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one idempotent Kaggle TPU v5e-8 command that gates hardware/runtime, executes the existing R3-backed Gemma4 strict load, verifies Candidate-A, performs exact one-token G3 generation, and emits compact evidence.

**Architecture:** Pure standard-library contract helpers are kept separate from the JAX authority process so shell and CPU tests can validate gates without importing JAX. The shell runner performs admission, exact package restoration, project-scoped dependency validation, environment setup, and evidence packaging; the sole Python authority process handles TPU initialization, model load, Candidate-A verification, cleanup, generation, and result persistence.

**Tech Stack:** Bash, Python 3 standard library, pytest/unittest, Keras 3.15.0, KerasHub 0.29.1, Keras-NLP 0.29.1, JAX/JAXlib 0.10.2, NumPy 2.5.0, libtpu 0.0.17.

**Spec:** `docs/superpowers/specs/2026-09-13-g3-tpu-authority-design.md`

## Global Constraints

- `MODEL_PRESET=gemma4_instruct_31b` and `MODEL_PATH=/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_31b/2`.
- `KERAS_BACKEND=jax`, `MODEL_DTYPE=bfloat16`, TPU v5e-8, exactly 8 devices.
- `MESH_SHAPE=[1,8]`, `MESH_AXES=[batch,model]`.
- Candidate-A remains the existing `gemma4_31b_dense_candidate_a_v1` layout.
- R3 remains `jax.make_array_from_callback()` with shard-local host slices followed by normal `Variable.assign`.
- Forbidden R3 operations remain absent: `_direct_assign`, full `jnp.asarray(full_source)`, full `jax.device_put(full_source)`, checkpoint mutation, dtype/mesh/Candidate-A changes, and `skip_mismatch`.
- `TPU_INIT_HARD_MAX_SECONDS=180`; `STRICT_LOAD_HARD_MAX_SECONDS=2304`.
- G3 is exactly `Hello`, prompt count 10, `max_new_tokens=1`, `max_length=11`, `strip_prompt=True`, and one generation call.
- Do not use global `pip check` as an admission gate; unrelated Kaggle conflicts are ignored.
- Do not redownload the model, build JAX/libtpu from source, tag, release, or push.

### Task 1: Add pure authority contract helpers and focused tests

**Files:**
- Create: `src/gemma4_server/tpu/authority_contract.py`
- Create: `tests/test_authority_contract.py`

**Interfaces:**
- `hardware_gate(model_path: str | Path, accel_glob: Iterable[str], memory_max_bytes: int | None, minimum_memory_gib: int = 300) -> dict[str, object]` returns admission facts and raises `AuthorityGateError` on failure.
- `exact_runtime_versions(requirements_path: str | Path, installed: Mapping[str, str]) -> dict[str, str]` returns normalized exact package versions.
- `project_dependency_gate(distributions: Mapping[str, Mapping[str, object]], roots: Iterable[str]) -> dict[str, object]` validates only the transitive metadata closure of the named roots.
- `verify_candidate_a(model) -> dict[str, object]` returns the required token embedding facts and raises `CandidateAVerificationError` when any frozen invariant fails; this function must not import JAX.

- [ ] **Step 1: Write failing tests** for missing accelerator, sub-300-GiB memory, exact requirements matching/mismatch, unrelated dependency conflicts being ignored, Candidate-A success, and Candidate-A failure.
- [ ] **Step 2: Run `python -m pytest tests/test_authority_contract.py -q` and confirm the new imports/functions fail because the module is absent.
- [ ] **Step 3: Implement only standard-library contract functions and explicit exceptions; parse cgroup values and requirement lines without importing JAX/Keras.
- [ ] **Step 4: Run the focused test file and confirm it passes.
- [ ] **Step 5: Run the existing source/R3 tests that touch the unchanged engine contracts.

### Task 2: Add exact runtime manifest and integrated shell runner

**Files:**
- Create: `requirements-tpu-g3.txt`
- Create: `scripts/run_g3_tpu_authority.sh`
- Modify: `scripts/configure_kaggle_tpu.sh` only if the existing exported defaults cannot be reused without changing behavior.
- Create: `tests/test_g3_runner_contract.py`

**Interfaces:**
- Shell entrypoint: `bash scripts/run_g3_tpu_authority.sh`.
- The runner creates `/kaggle/working/gemma4-31b-vnext-g3-authority-<UTC>/` with exactly the prescribed evidence names, and archives it beside the directory.
- Early hardware failure writes `AUTHORITY_NOT_STARTED=true` and `FINAL_RESULT=TPU_HARDWARE_NOT_READY`, exits non-zero, and never invokes a JAX-importing process.

- [ ] **Step 1: Write failing source-contract tests** for exact manifest content, gate ordering before any JAX reference, no reinstall when all metadata matches, `--no-deps --upgrade -r requirements-tpu-g3.txt` restoration, evidence names, and non-TPU result markers.
- [ ] **Step 2: Run the new runner contract tests and confirm they fail because the manifest and runner are absent.
- [ ] **Step 3: Add the seven exact manifest lines and implement the shell runner with `set -Eeuo pipefail`, safe timestamped evidence paths, hardware gate, exact metadata check, conditional install, project dependency gate, environment configuration, one Python child, cgroup snapshots, adjudication, hashes, and one archive plus sidecar.
- [ ] **Step 4: Run the runner contract tests and confirm they pass.
- [ ] **Step 5: Execute the runner in the current environment; expect the hardware gate to return `TPU_HARDWARE_NOT_READY` before any JAX import.

### Task 3: Add the single Python TPU authority process

**Files:**
- Create: `scripts/g3_tpu_authority.py`
- Modify: `src/gemma4_server/tpu/engine.py` to pass `max_new_tokens=1` explicitly on the authority native call while preserving the existing exact planner and strip behavior.
- Create: `tests/test_g3_authority_source.py`

**Interfaces:**
- Authority entrypoint: `python3 scripts/g3_tpu_authority.py --evidence-dir <dir>`.
- It writes `07-authority-result.json`, emits durable marker lines, and returns a non-zero status for every non-pass outcome.
- It imports JAX only after the shell gate has admitted hardware; it calls `jax.devices("tpu")` once for the device check and requires exactly eight devices.

- [ ] **Step 1: Write failing source and fake-engine tests** for watchdog constants, one JAX authority import path, device count, strict-load marker ordering, Candidate-A verification, cleanup ordering, one generation call, explicit `max_length=11`, explicit `max_new_tokens=1`, and G4 remaining false.
- [ ] **Step 2: Run the new tests and confirm they fail because the authority script and explicit generation keyword are absent.
- [ ] **Step 3: Implement the watchdog thread with a 180-second hard deadline, one strict-load 2304-second deadline, event/result persistence, cgroup memory snapshots, Candidate-A call, `gc.collect()`/`malloc_trim` via existing engine cleanup, and one native authority generation call.
- [ ] **Step 4: Update the engine authority call to pass both `max_length=plan.max_length` and `max_new_tokens=plan.max_new_tokens`.
- [ ] **Step 5: Run focused authority/source tests and existing generation/R3 tests; fix only failures caused by this integration.

### Task 4: Add optional minimal Kaggle notebook and run the required verification

**Files:**
- Create: `notebooks/kaggle-tpu-v5e8-g3.ipynb` only if the existing project workflow requires a notebook entrypoint.
- Modify: focused tests only if assertions need to reflect the integrated contract.

**Interfaces:**
- Notebook has exactly three conceptual cells: source/config display, shell runner invocation, and final JSON display; it contains no duplicate TPU/model logic.

- [ ] **Step 1: Inspect the existing notebook workflow and add the minimal three-cell notebook only when required.
- [ ] **Step 2: Run `python -m pytest tests/test_authority_contract.py tests/test_g3_runner_contract.py tests/test_g3_authority_source.py tests/test_generation.py tests/test_sharded_checkpoint.py tests/test_source_contract.py -q`.
- [ ] **Step 3: Run `bash -n scripts/run_g3_tpu_authority.sh` and `python -m py_compile scripts/g3_tpu_authority.py src/gemma4_server/tpu/authority_contract.py`.
- [ ] **Step 4: Run the integrated runner once in the current environment and record the expected hardware-deferred result.
- [ ] **Step 5: Verify the generated evidence contents and SHA-256 sidecar without running any second TPU probe or authority process.
