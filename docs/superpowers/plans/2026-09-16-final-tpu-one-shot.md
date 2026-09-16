# Final TPU One-Shot G9/G10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add direct G9 compile/cache evidence and a complete tracked G10 orchestration path without changing frozen Gemma4 generation semantics.

**Architecture:** A request-scoped JAX log observer wraps the existing native `model.generate()` call. A tracked Python orchestrator owns source, hardware, model, REST, semantic, memory, and evidence checks; the Kaggle notebook becomes a thin caller.

**Tech Stack:** Python 3.12, unittest, Flask, urllib, JAX 0.10.x, Keras 3.15.x, KerasHub 0.29.x, Jupyter notebook JSON.

**Spec:** `docs/superpowers/specs/2026-09-16-final-tpu-one-shot-design.md`

## Global Constraints

- Preserve `gemma4_instruct_31b`, JAX, BF16, TPU v5e-8, mesh `[1,8]`, axes `[batch,model]`, Candidate-A, native preset loader, strict loading, `skip_mismatch=false`, production buckets `(16,512,768,1024,1536,2048)`, sampling, and REST payload semantics.
- Do not modify installed packages, clear caches, or add warmups/retries/extra model calls.
- No live generation, Kaggle restart, TPU release, tag, or release during pre-live implementation.
- Exclude weights, caches, credentials, `.env`, PID/state files, private logs, databases, and restart secrets from evidence.
- `NOT_AVAILABLE` is not zero and fails required direct-evidence gates.
- Run focused tests, the full suite, compileall, Bash syntax checks, notebook JSON parsing, and canonical CI before live G9.

---

### Task 1: Add direct request-scoped compile/cache observation

**Files:**
- Create: `src/gemma4_server/tpu/observability.py`
- Create: `tests/test_observability.py`

**Interfaces:**
- `CompilationEvidenceCapture(logger_names: tuple[str, ...] = ("jax", "jax._src"))` context manager.
- `CompilationEvidenceCapture.snapshot() -> dict[str, object]` with `source`, `available`, `events`, `compile_event_count`, `compile_seconds`, `persistent_cache_hits`, `persistent_cache_misses`, and `status`.
- `adjudicate_hot_cache(prime: dict, hot1: dict, hot2: dict) -> dict[str, object]` with `hot_cache_reuse`, `hot_prefill_compile_seconds`, `hot_decode_compile_seconds`, `compile_evidence`, and `reason`.

- [ ] **Step 1: Write failing tests**

Test parsing of `Compiling ...`, persistent-cache hit, and persistent-cache
miss messages; test handler cleanup on success and exception; test empty direct
capture; test HOT PASS only when both HOT captures are direct, have zero compile
events, and have no cache miss. Use this fixture shape:

```python
capture = CompilationEvidenceCapture(logger_names=("test-jax",))
with capture:
    logging.getLogger("test-jax").warning(
        "Compiling jit(generate) with global shapes and types ..."
    )
assert capture.snapshot()["compile_event_count"] == 1
```

- [ ] **Step 2: Run and observe failure**

```bash
python3 -m unittest tests.test_observability -v
```

Expected: import/attribute failures because the observer is absent.

- [ ] **Step 3: Implement the minimal observer**

Use a temporary `logging.Handler` and remove it in `finally`. Match only
semantic JAX compile/cache fragments and retain compact raw messages. Do not
call `jax.clear_caches()`, alter JAX configuration, enumerate devices, or run
model code. Return explicit unavailable status on handler/parsing failure.

- [ ] **Step 4: Run focused tests**

```bash
python3 -m unittest tests.test_observability -v
```

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_server/tpu/observability.py tests/test_observability.py
git commit -m "feat: add direct TPU compile evidence observer"
```

### Task 2: Integrate evidence without changing generation semantics

**Files:**
- Modify: `src/gemma4_server/tpu/engine.py` in `_generate`
- Modify: `tests/test_generation.py`
- Modify: `tests/test_source_contract.py` when needed for frozen-call assertions

**Interfaces:**
- `_generate()` still returns `(text: str, metrics: dict)`.
- Metrics retain all existing keys and add `compile_cache_evidence: dict`.
- The model call remains `self.model.generate(inputs, max_length=plan.max_length, strip_prompt=True)`.

- [ ] **Step 1: Write failing fake-model tests**

Patch the observer to emit one synthetic event and assert nested metrics plus
the exact existing model arguments. Add a test that observer failure yields
`status="unavailable"` while text/result behavior is unchanged.

- [ ] **Step 2: Run focused tests and observe failure**

```bash
python3 -m unittest tests.test_generation tests.test_source_contract -v
```

- [ ] **Step 3: Implement the narrow wrapper**

Import the observer locally in `_generate`, wrap only the one existing
`model.generate()` call, and add the snapshot to metrics. Add no warmup,
barrier, retry, alternate sampler, changed length, or second generation.

- [ ] **Step 4: Re-run tests and compile changed modules**

```bash
python3 -m unittest tests.test_generation tests.test_source_contract -v
python3 -m py_compile src/gemma4_server/tpu/engine.py src/gemma4_server/tpu/observability.py
```

- [ ] **Step 5: Commit**

```bash
git add src/gemma4_server/tpu/engine.py tests/test_generation.py tests/test_source_contract.py
git commit -m "feat: expose generation compile evidence metrics"
```

### Task 3: Build standard-library orchestration and evidence utilities

**Files:**
- Create: `scripts/final_tpu_one_shot.py`
- Create: `tests/test_final_tpu_one_shot.py`

**Interfaces:**
- `read_runtime_identity() -> dict[str, str]`
- `read_cgroup_snapshot(root: pathlib.Path = pathlib.Path("/sys/fs/cgroup")) -> dict[str, object]`
- `sha256_file(path: pathlib.Path) -> str`
- `assert_source_identity(repo: pathlib.Path, expected_sha: str) -> dict[str, object]`
- `request_json(base_url: str, path: str, method: str = "GET", payload: dict | None = None, headers: dict[str, str] | None = None, timeout: float = 30.0) -> tuple[int, dict]`
- `poll_job(base_url: str, job_id: str, headers: dict[str, str], timeout: float, interval: float = 2.0) -> dict`
- `package_evidence(directory: pathlib.Path) -> pathlib.Path`

- [ ] **Step 1: Write failing fake-filesystem and fake-HTTP tests**

Cover runtime identity, cgroup integer/OOM parsing, SHA calculation, exact
SHA/clean-tree rejection, request payload/headers, queued→processing→completed
polling, timeout failure, and checksums that exclude forbidden filenames.

- [ ] **Step 2: Run and observe failure**

```bash
python3 -m unittest tests.test_final_tpu_one_shot -v
```

- [ ] **Step 3: Implement utilities using only the standard library**

Use `urllib.request`, `json`, `hashlib`, `subprocess`, `pathlib`, and `time`.
Redact fields containing `key`, `secret`, `token`, or `credential`. Missing
cgroup fields remain explicit unavailable values. Do not import JAX here.

- [ ] **Step 4: Run focused tests and compile**

```bash
python3 -m unittest tests.test_final_tpu_one_shot -v
python3 -m py_compile scripts/final_tpu_one_shot.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/final_tpu_one_shot.py tests/test_final_tpu_one_shot.py
git commit -m "feat: add TPU one-shot evidence utilities"
```

### Task 4: Implement G9 PRIME/HOT and semantic REST orchestration

**Files:**
- Modify: `scripts/final_tpu_one_shot.py`
- Create: `tests/test_g9_orchestration.py`

**Interfaces:**
- `run_g9(args: argparse.Namespace) -> int`
- `run_text_acceptance(client, evidence_dir, label: str) -> dict`
- `run_vision_acceptance(client, evidence_dir, label: str) -> dict`
- `adjudicate_g9(rows: dict[str, object]) -> dict[str, object]`

- [ ] **Step 1: Write failing fake-REST tests**

Assert exactly one PRIME and two identical HOT submissions, no `/restart`,
direct evidence required, nonempty results, semantic text/vision checks,
memory/OOM checks, required endpoint checks, and fail-closed adjudication.

- [ ] **Step 2: Run and observe failure**

```bash
python3 -m unittest tests.test_g9_orchestration -v
```

- [ ] **Step 3: Implement explicit G9 mode**

Verify exact execution SHA and clean tree; capture source/runtime/cgroup
baseline; inspect `/`, `/health/live`, `/health/ready`, and `/info`; start the
existing server only when no healthy runtime can be reused; wait for ready;
submit one async text PRIME; submit the exact same payload twice for HOT-1 and
HOT-2; parse metrics through `adjudicate_hot_cache`; run tracked semantic text
and repository-local/reconstructible vision acceptance; verify lifecycle and
collector completion; capture after-state memory/OOM; write the required
compact G9 evidence files and `SHA256SUMS`.

A started replacement runtime records `G9_MODEL_RELOAD_COUNT=1`; reuse records
zero only with source/config equivalence. PRIME failure is terminal; never
retry, reload, clear caches, or invoke `/restart` in G9.

- [ ] **Step 4: Run focused tests**

```bash
python3 -m unittest tests.test_g9_orchestration tests.test_final_tpu_one_shot -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/final_tpu_one_shot.py tests/test_g9_orchestration.py
git commit -m "feat: orchestrate G9 prime hot acceptance"
```

### Task 5: Implement complete G10 fresh-session Run All mode

**Files:**
- Modify: `scripts/final_tpu_one_shot.py`
- Modify: `notebooks/kaggle-tpu-v5e8-final-run-all.ipynb`
- Create: `tests/test_g10_orchestration.py`

**Interfaces:**
- `run_g10(args: argparse.Namespace) -> int`
- `adjudicate_g10(rows: dict[str, object]) -> dict[str, object]`
- Notebook invokes `python3 scripts/final_tpu_one_shot.py --mode g10 --expected-sha "$FINAL_TPU_EXECUTION_SHA"`.

- [ ] **Step 1: Write failing G10 fake-runtime tests**

Test rejection of stale session identity, wrong SHA, fewer than eight devices,
wrong preset/dtype/classes, missing strict-load/Candidate-A metadata, empty
text/vision results, failed polling, and positive OOM deltas. Test that all
rows produce `G10_STATUS="CLOSED/PASS"` and `EVIDENCE_PACKAGED=true`.

- [ ] **Step 2: Run and observe failure**

```bash
python3 -m unittest tests.test_g10_orchestration -v
```

- [ ] **Step 3: Implement G10 mode and thin notebook**

Capture fresh-session identity (`boot_id`, hostname, PID 1, and available
Jupyter marker), clone the public repository into a fresh path, checkout the
exact execution SHA, reject a dirty tree, configure frozen environment values,
start the production server once, wait for readiness, and require metadata for
Gemma4CausalLM/Gemma4Backbone, BF16, eight devices, strict loading,
`skip_mismatch=false`, Candidate-A, and native preset loading. Run text and
vision over async REST, verify all health/info/job/collector rows, capture
before/after cgroup state, package the exact G10 filenames, and exit nonzero on
any missing row.

Replace notebook model code with configuration, canonical clone/checkout, and
one subprocess call to the tracked script. The notebook prints exact SHA and
final adjudication and contains no model implementation.

- [ ] **Step 4: Run focused tests and parse notebook JSON**

```bash
python3 -m unittest tests.test_g10_orchestration tests.test_g9_orchestration -v
python3 -m json.tool notebooks/kaggle-tpu-v5e8-final-run-all.ipynb >/dev/null
```

- [ ] **Step 5: Commit**

```bash
git add scripts/final_tpu_one_shot.py notebooks/kaggle-tpu-v5e8-final-run-all.ipynb tests/test_g10_orchestration.py
git commit -m "feat: complete fresh TPU Run All orchestration"
```

### Task 6: Complete pre-live verification and freeze the execution SHA

**Files:**
- Modify only tests/docs if a verification finding requires a narrow correction.

- [ ] **Step 1: Run focused acceptance tests**

```bash
python3 -m unittest tests.test_observability tests.test_generation tests.test_final_tpu_one_shot tests.test_g9_orchestration tests.test_g10_orchestration
```

- [ ] **Step 2: Run the complete CPU suite once**

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

- [ ] **Step 3: Run static checks**

```bash
python3 -m compileall -q src scripts clients/python
for f in scripts/*.sh; do bash -n "$f"; done
python3 -m json.tool notebooks/kaggle-tpu-v5e8-final-run-all.ipynb >/dev/null
```

- [ ] **Step 4: Review frozen invariants**

```bash
rg -n 'model\.generate\(|clear_caches|skip_mismatch|bfloat16|gemma4_instruct_31b|generation_length_buckets|mesh_shape|model_axis' src scripts tests
git diff e7a58636ba2fba0f4ed49b2d2bd4fb282968ab26 -- src scripts notebooks
```

Confirm only evidence wrapping, orchestration, notebook wiring, and tests
changed. Confirm no prompt, sampler, bucket, checkpoint, mesh, REST payload,
or strict-load behavior changed.

- [ ] **Step 5: Commit only if verification added tracked material**

```bash
git status --short
git log --oneline --decorate -8
```

### Task 7: Push, observe canonical CI, then execute the directive gates

**Files:**
- Create compact runtime evidence under `artifacts/g9/`, `artifacts/g10/`,
  and later `artifacts/final/` only after the corresponding live gate.
- Modify final docs/release files only after G9 and G10 PASS.

- [ ] **Step 1: Push tested execution source**

Fetch `origin`, integrate the tested commits into canonical `main`, push
`main`, and record the resulting 40-character `FINAL_TPU_EXECUTION_SHA`.
Require remote `main` to point to it before any generation.

- [ ] **Step 2: Wait for CI PASS**

Observe canonical CI for exactly that SHA and record its run ID. If CI fails,
correct only pre-live source, repeat the required checks, push a new SHA, and
wait again.

- [ ] **Step 3: Run G9 exactly once**

Inspect current runtime without generation, reuse only if all equivalence
conditions pass, otherwise load one server/worker in the current TPU
allocation. Run one PRIME and two HOT requests. Stop immediately on failure;
do not restart or release TPU.

- [ ] **Step 4: Perform exactly one G9→G10 Kaggle Restart Session**

After verified G9 PASS, record the checksum/SHA/runtime checkpoint, request the
single restart, prove actual freshness, and run the tracked notebook with
Restart Session → Run All. Stop without release on any G10 failure.

- [ ] **Step 5: Finish only after G10 PASS**

Reconcile README/README.vi, roadmap, status, changelog, and release notes;
package final evidence; prove runtime-code equivalence; observe CI on the
release candidate; verify `v1.0.0` does not exist; create/push the annotated
tag; create the verified GitHub release; and verify remote tag, release,
assets, checksums, main, and CI. Never publish private material or weights.
