# G3 Minimum-Shape cgroup-OOM Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the supplied G3 directive with CPU-only gates followed by at most one TPU authority process, producing either G3 PASS or directly proven minimum-shape cgroup OOM evidence.

**Architecture:** Keep the authority launcher outside the model project so preflight can validate files and configuration without importing JAX/Keras. Reuse the existing `Gemma4TPUEngine`, native KerasHub loader, frozen `(1,8)` distribution, and project generation path. The outer launcher owns cgroup-before/after snapshots, process status, telemetry, adjudication, and immutable evidence packaging.

**Tech Stack:** Python 3 standard library for preflight/launcher/evidence, existing Keras 3.15.0, KerasHub 0.29.1, Keras-NLP 0.29.1, JAX 0.10.2, JAXlib 0.10.2, NumPy 2.5.0, libtpu 0.0.17, TPU v5e-8.

**Spec:** `/kaggle/working/gemma4-31b-g3-minimum-shape-generation-cgroup-oom-proof-directive-2026-09-12.md`

## Global Constraints

- Preserve `G0=CLOSED`, `G1=CLOSED`, `G2_STATUS=CLOSED`, `G3_STATUS=OPEN`, `G4_ENTRY_ELIGIBLE=false`, and `G4_STARTED=false` until fresh evidence adjudicates G3.
- Keep `MODEL_PATH=/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_31b/2`, `KERAS_BACKEND=jax`, `MODEL_DTYPE=bfloat16`, mesh shape `[1,8]`, and axes `[batch,model]`.
- Use `PROMPT_TEXT=Hello`, `MAX_NEW_TOKENS=1`, exactly one TPU authority process, one model load, and one generation call.
- Do not rerun G2, change checkpoint/dtype/mesh/sharding, use fallback/quantization, run warmup/second generation/G4/REST/vision/benchmark, or modify site-packages.
- Do not initialize JAX/TPU for CPU discovery, token proof, planner checks, cgroup snapshots, or preflight.
- Declare direct cgroup OOM only when the authority exits 137/SIGKILL after generation starts and `oom_kill_after - oom_kill_before > 0`.

### Task 1: Add failing CPU tests for minimum-shape and evidence contracts

**Files:**
- Create: `/kaggle/working/test_gemma4_g3_minimum_shape_authority.py`
- Test: `/kaggle/working/test_gemma4_g3_minimum_shape_authority.py`

**Interfaces:**
- Consumes: future `/kaggle/working/gemma4-g3-minimum-shape-authority.py` CLI and current project source/config.
- Produces: subprocess tests that fail before the harness exists and later prove no runtime import, exact one-token contract, smallest legal bucket, cgroup adjudication, and evidence schema.

- [ ] Write tests for valid CPU preflight, invalid paths/parameters, explicit path precedence, no JAX/Keras import, `max_new_tokens=1`, and no generation in preflight.
- [ ] Write tests for `plan_generation()` proving a legal bucket list selects the smallest bucket at least `prompt_count + 1`, and that a zero/negative OOM delta cannot classify direct proof.
- [ ] Run `python3 -m unittest -v /kaggle/working/test_gemma4_g3_minimum_shape_authority.py` and confirm the failures are due to the missing harness, not test syntax.

### Task 2: Implement CPU-only source contract and preflight

**Files:**
- Create: `/kaggle/working/gemma4-g3-minimum-shape-authority.py`
- Read: `src/gemma4_server/tpu/engine.py`, `src/gemma4_server/tpu/generation.py`, `src/gemma4_server/core/config.py`, model metadata files.

**Interfaces:**
- Produces `--preflight-only`, `g3-preflight.json`, source-contract text, model-path/weight-map validation, cgroup path proof, and minimum-shape proof without runtime imports.

- [ ] Implement standard-library-only argument parsing and canonical path/file validation; resolve every `model.weights.json` reference and reject missing/empty shards.
- [ ] Record the actual path: `chat_prompt()` → `Gemma4TPUEngine.generate_text()` → `_generate()` → `generate_preprocess()` → `plan_generation()` → `model.generate(..., strip_prompt=True)`.
- [ ] Obtain exact token IDs/count for the rendered `Hello` prompt through the native tokenizer on CPU only; fail closed if this cannot be obtained without loading the 31B model or touching TPU.
- [ ] Read current configured legal buckets from project source/config, compute `required_length = prompt_token_count + 1`, and select the smallest legal bucket; do not reuse attempt #3's bucket 32 unless current source proves it.
- [ ] Discover unified cgroup-v2 path from `/proc/self/cgroup`, verify `memory.events` and `memory.max`, and persist raw path proof.

### Task 3: Implement tests and one-process authority/evidence logic

**Files:**
- Modify: `/kaggle/working/gemma4-g3-minimum-shape-authority.py`

**Interfaces:**
- Consumes: validated preflight contract.
- Produces: one deferred-import TPU runtime with exact load/generation counters, RSS telemetry, token-embedding sharding metadata, and machine-readable authority result.

- [ ] Import JAX/Keras/KerasHub only after preflight succeeds; verify exactly 8 TPU devices and expected v5e-8 environment.
- [ ] Build frozen distribution once, construct one `Gemma4TPUEngine`, call `load()` once, verify model object and Candidate-A token embedding sharding, and record RSS boundaries.
- [ ] Call `engine.generate_text("Hello", "", 1)` exactly once with the proven minimum shape; record generation start before the call and never synthesize a Python exception for OS kill.
- [ ] Write result fields for both normal return and exception paths, including output validation and generation metadata; return success only for a non-empty string.

### Task 4: Run all CPU gates and freeze the source snapshot

**Files:**
- Create: `/kaggle/working/gemma4-31b-g3-minimum-shape-cgroup-oom-proof-evidence-<timestamp>/`
- Create: `/kaggle/working/gemma4-31b-g3-minimum-shape-cgroup-oom-proof-evidence-<timestamp>/01-authority-source-snapshot/`

**Interfaces:**
- Produces all required CPU gate logs, source-contract report, prompt-token proof, minimum-shape proof, runtime versions, model preflight, weight-map preflight, and static safety scan.

- [ ] Run `pip check`, project CPU tests, harness tests, planner tests, model path/weight-map checks, and static R3 scan without TPU initialization.
- [ ] Persist source snapshot and authority marker; after this point do not change runtime source before the single authority invocation.
- [ ] Verify the evidence directory is new and never overwrite prior attempt evidence.

### Task 5: Capture cgroup BEFORE, run exactly one TPU authority, capture AFTER

**Files:**
- Create in the evidence directory: `12-cgroup-memory-events-before.txt`, `13-cgroup-memory-events-after.txt`, `14-memory-telemetry.txt`, `15-authority-stdout.log`, `16-authority-stderr.log`, `17-authority-exit.txt`, `18-g3-minimum-shape-authority-result.json`.

**Interfaces:**
- Produces the only authorized TPU runtime observation and exact before/after OOM arithmetic.

- [ ] Persist `memory.max`, `memory.current`, `memory.peak`, `memory.events`, and `memory.events.local` before launching the authority process; abort before TPU if persistence fails.
- [ ] Launch exactly one Python/JAX authority process with canonical environment and explicit project `PYTHONPATH`; capture exit code and signal externally.
- [ ] Read the same cgroup files after exit and calculate `oom`, `oom_kill`, and `oom_group_kill` deltas as numbers.
- [ ] Adjudicate only G3 PASS, directly proven minimum-shape cgroup OOM, or an inconclusive/invalid single invocation; stop immediately after the first terminal outcome.

### Task 6: Package and verify immutable evidence

**Files:**
- Create: `00-directive.txt` through `19-final-adjudication.txt`, `SHA256SUMS`, archive, and external `.sha256` sidecar.

**Interfaces:**
- Produces the final report fields required by the directive and a checksum-verified archive without rerunning model code.

- [ ] Generate internal checksums only after all evidence contents are final and verify `sha256sum -c`.
- [ ] Create and verify the UTC archive and sidecar; verify archive listing.
- [ ] Write final adjudication with all CPU/TPU/cgroup fields, including BEFORE/AFTER `oom_kill` values, then stop without G4.
