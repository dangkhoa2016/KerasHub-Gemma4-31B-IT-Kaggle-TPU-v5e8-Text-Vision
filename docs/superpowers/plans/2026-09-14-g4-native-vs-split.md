# G4 Native-vs-Split Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute and characterize one-token Gemma4 native generation versus direct cache-based split prefill/decode without changing frozen G3.

**Architecture:** Reuse `Gemma4TPUEngine.load()` for one model and Candidate-A sharding, then use the installed `Gemma4CausalLM._build_cache()` and `.call_with_cache()` primitives in a G4-only helper. A shell runner gates hardware/runtime/dependencies before one authority process and packages result, memory, baseline, comparison, and hashes.

**Tech Stack:** Python 3.12, Keras 3.15.0, KerasHub 0.29.1, JAX 0.10.2, BF16, Keras ModelParallel, Bash, unittest.

**Spec:** `docs/superpowers/specs/2026-09-14-g4-native-vs-split-design.md`

## Global Constraints

- G3 is immutable; do not rerun `scripts/run_g3_tpu_authority.sh`.
- Use `Hello`, prompt token count 10, requested new tokens 1, exact max length 11.
- Preserve model path, `gemma4_instruct_31b`, BF16, mesh `[1,8]`, axes `[batch,model]`, and Candidate-A.
- Use one Python authority process and one model load per real attempt.
- Accept `/dev/accel*` or eight numeric `/dev/vfio/*` nodes with `memory.max >= 300 GiB`.
- Do not pass `max_new_tokens` to `Gemma4CausalLM.generate()`.
- Do not add a third TPU attempt; one corrective rerun is allowed only after a narrow understood defect and CPU verification.

---

### Task 1: Add test-first G4 contracts

**Files:**
- Create: `tests/test_g4_contract.py`
- Create: `src/gemma4_server/tpu/g4_split.py`

**Interfaces:**
- Produces `build_split_inputs(model, rendered_prompt, sequence_length)`, `split_one_token(model, inputs)`, and `run_split_generation(model, prompt)`.

- [ ] **Step 1: Write failing tests** for exact prompt/length, direct cache primitive use, greedy one-token selection, no `generate()` call, and optional-key-safe preprocessing.
- [ ] **Step 2: Run `python3 -m unittest tests.test_g4_contract -v` and confirm failure because `g4_split` does not exist.
- [ ] **Step 3: Implement the smallest helper using `generate_preprocess([rendered_prompt], sequence_length=11)`, `_build_cache`, `call_with_cache`, `keras.ops.argmax`, and tokenizer postprocessing.
- [ ] **Step 4: Run the focused test again and confirm it passes without importing JAX devices.
- [ ] **Step 5: Run the full CPU suite and compileall.

### Task 2: Add one-process G4 authority

**Files:**
- Create: `scripts/g4_split_authority.py`
- Modify: `tests/test_g4_contract.py`

**Interfaces:**
- Consumes `Gemma4TPUEngine`, `verify_candidate_a`, `run_split_generation`, and an evidence directory.
- Produces `07-g4-result.json` with load, Candidate-A, split, prompt, generation, and memory markers.

- [ ] **Step 1: Add source tests for one `Gemma4TPUEngine` construction, one `.load()` call, Candidate-A-before-split ordering, and no G3 import/invocation.
- [ ] **Step 2: Run focused tests and observe the new source-contract failures.
- [ ] **Step 3: Implement the authority with a single JAX import, TPU count hard gate, 2304-second load watchdog, load cleanup, split execution, non-empty result check, and JSON writes.
- [ ] **Step 4: Run focused tests, compile the script, and inspect the source for forbidden duplicate model/weight operations.

### Task 3: Add frozen native baseline parser and evidence comparison

**Files:**
- Create: `scripts/g4_evidence.py`
- Create: `artifacts/g4/g4-native-baseline.json`
- Modify: `tests/test_g4_contract.py`

**Interfaces:**
- Produces `native_baseline_from_archive(archive_path)`, `write_comparison(...)`, and parseable JSON with the frozen native metrics only.

- [ ] **Step 1: Add tests using the frozen archive path and assert no native runner reference is executed.
- [ ] **Step 2: Run tests and observe missing parser failure.
- [ ] **Step 3: Implement archive JSON extraction and comparison fields, preserving unavailable metrics as unavailable instead of inventing values.
- [ ] **Step 4: Generate `g4-native-baseline.json` and run tests plus JSON parsing.

### Task 4: Add G4 runner and pre-TPU gates

**Files:**
- Create: `scripts/run_g4_characterization.sh`
- Modify: `tests/test_g4_contract.py`

**Interfaces:**
- Consumes project root, frozen G3 archive, current device/cgroup state, and runtime manifest.
- Produces the required G4 evidence directory, archive, `.sha256` sidecar, and final adjudication.

- [ ] **Step 1: Add source tests for gate ordering, ACCEL/VFIO style markers, low-memory rejection, no G3 invocation, exact runtime restore, dependency gate, and required evidence names.
- [ ] **Step 2: Run tests and observe runner missing failure.
- [ ] **Step 3: Implement pre-JAX gates, Kaggle fallback sourcing, one authority launch, phase cgroup captures, baseline comparison, adjudication, SHA256SUMS, archive, and sidecar.
- [ ] **Step 4: Run source tests, `bash -n`, and non-TPU dry-run checks without initializing JAX.

### Task 5: CPU gate and primary TPU characterization

**Files:**
- Modify: generated G4 evidence only; preserve all G3 artifacts.

- [ ] **Step 1: Run `python3 -m unittest discover -s tests -p 'test_*.py'`, `python3 -m compileall -q src scripts clients/python`, and shell syntax checks.
- [ ] **Step 2: Record `G4_CPU_PREPARATION=PASS` only after exit code 0 and zero test failures.
- [ ] **Step 3: Run `bash scripts/run_g4_characterization.sh` exactly once if the hardware gate passes; otherwise preserve a pre-authority evidence package.
- [ ] **Step 4: Inspect result, memory, comparison, adjudication, archive hash, and source delta.

### Task 6: Conditional corrective rerun and closeout

**Files:**
- Modify: only narrow G4 implementation files if a direct first-attempt defect is observed.

- [ ] **Step 1: If and only if the first real attempt exposes an understood direct defect, patch narrowly and rerun the full CPU/static gate.
- [ ] **Step 2: Consume at most one corrective TPU attempt on the same healthy allocation.
- [ ] **Step 3: If the second attempt fails, mark `G4_TPU_ATTEMPT_BUDGET_EXHAUSTED`; never start a third attempt.
- [ ] **Step 4: Verify G3 hashes/artifacts unchanged, G4 post-G3 hashes recorded, and report the exact operational summary requested by the directive.
