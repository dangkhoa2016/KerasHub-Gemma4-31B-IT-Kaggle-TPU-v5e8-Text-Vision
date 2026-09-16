# G9 Corrective Authority and Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and statically qualify the G9 corrective for environment-secret precedence, pre-PRIME authority gating, and direct JAX compile/cache evidence without running live G9/G10 or advancing release state.

**Architecture:** Keep the shell fix local to `load_env`, add a structured authority-gate helper to the existing one-shot orchestrator, and make JAX logging configuration/coverage explicit in the existing observability module. The G9 runner consumes those helpers before any generation POST and writes the gate/cache fields into its primary acceptance evidence.

**Tech Stack:** Bash, Python 3, `unittest`, Flask test doubles, JAX logging/configuration, JSON evidence, GitHub Actions CI.

**Spec:** `docs/superpowers/specs/2026-09-16-g9-corrective-authority-observability-design.md`

## Global Constraints

- Preserve `G9_INITIAL_LIVE_ATTEMPT=FAIL` and the existing evidence directory and SHA256 manifest.
- Do not release TPU, switch accelerator to CPU, deliberately restart G9→G10, run G10, create a tag, or create a release.
- A non-empty process environment `API_KEY` or `RESTART_SECRET` must survive `.env` loading.
- Do not disable authentication, commit credentials, or write generated secrets into evidence.
- Do not change model.generate arguments, sampler, bucket algorithm, dtype, mesh, Candidate-A, loader, or G10 orchestration.
- Empty compile captures are authoritative only with direct observer coverage and enabled production JAX compile logging.
- Before corrective live G9, bucket 16 must be proven cold in the new process/cache authority or another deterministic production bucket must be explicitly adjudicated.

### Task 1: Protect orchestrator credentials during `.env` loading

**Files:**
- Create: `tests/test_start_env.py`
- Modify: `scripts/_common.sh:load_env`

**Interfaces:**
- Consumes: `scripts/_common.sh` and a temporary project-local `.env`.
- Produces: `load_env()` where non-empty inherited `API_KEY` and `RESTART_SECRET` override `.env`, while ordinary `.env` variables still load.

- [ ] **Step 1: Write the failing tests**

Copy the production `_common.sh` into a temporary `scripts/` directory, write `.env` with blank secret assignments and `HOST=from-dotenv`, then run:

```python
completed = subprocess.run(
    ["bash", "-c", "source scripts/_common.sh; load_env; printf '%s\\n' \"$API_KEY|$RESTART_SECRET|$HOST\""],
    cwd=project,
    env={**os.environ, "API_KEY": "process-api", "RESTART_SECRET": "process-restart"},
    check=True,
    capture_output=True,
    text=True,
)
assert completed.stdout.strip() == "process-api|process-restart|from-dotenv"
```

Add a second case with no inherited secret variables and `.env` values `API_KEY=dotenv-api` and `RESTART_SECRET=dotenv-restart`; assert those values load. Use only synthetic values.

- [ ] **Step 2: Run the focused tests and verify RED**

Run `python3 -m unittest tests.test_start_env -v`. Expected: the inherited-secret case fails because the current `source .env` overwrites process values with blanks.

- [ ] **Step 3: Implement the minimal shell fix**

In `load_env`, snapshot `${API_KEY-}` and `${RESTART_SECRET-}` before `source .env`; source the file exactly as before; restore/export each snapshot only when non-empty; then preserve the existing `PYTHONPATH` export. Do not alter unrelated variables or authentication settings.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run `python3 -m unittest tests.test_start_env -v`. Expected: all precedence and normal `.env` behavior tests pass.

- [ ] **Step 5: Commit the isolated task**

```bash
git add scripts/_common.sh tests/test_start_env.py
git commit -m "fix: preserve orchestrator secrets during env loading"
```

### Task 2: Add and enforce the pre-PRIME authority gate

**Files:**
- Modify: `scripts/final_tpu_one_shot.py:request helpers, run_g9`
- Modify: `tests/test_g9_orchestration.py:FakeClient and G9OrchestrationTests`

**Interfaces:**
- Consumes: a `RestClient`-compatible object with `get(path)` and the expected source SHA.
- Produces: `pre_prime_authority_gate(client, expected_sha) -> dict[str, Any]` with `passed`, endpoint checks, source-SHA check, runtime-contract check, and redacted failure details; `run_g9` invokes it before the first generation POST.

- [ ] **Step 1: Write the failing gate tests**

Extend the G9 fake `/info` runtime with the complete frozen `_model_contract` fields and `source_sha="a" * 40`. Add an `info_status` constructor option whose `401` path returns `401, {"error": "Unauthorized"}` while `/health/ready` still returns `200, {"ready": True}`. In the test, construct `SimpleNamespace(client=client, evidence_dir=Path(tmp), expected_sha="a" * 40, source_identity={"head": "a" * 40, "expected_sha": "a" * 40, "git_sha_exact": True, "worktree_clean": True}, model_reload_count=0)`, call `run_g9(args)`, and assert:

```python
self.assertEqual(result, 1)
self.assertFalse(evidence["passed"])
self.assertEqual(evidence["PRE_PRIME_AUTHORITY_GATE"], "FAIL")
self.assertEqual(evidence["GENERATION_POST_COUNT"], 0)
self.assertEqual([path for path, _payload in client.posts], [])
```

Add a passing gate test requiring all four endpoints, `ready=true`, exact `runtime.source_sha`, and the frozen model contract; add a source-SHA mismatch test.

- [ ] **Step 2: Run the focused test and verify RED**

Run `python3 -m unittest tests.test_g9_orchestration.G9OrchestrationTests.test_pre_prime_gate_stops_before_generation_when_info_is_unauthorized -v`. Expected: failure because `run_g9` currently calls `/generate/async` after endpoint checks.

- [ ] **Step 3: Implement the structured gate**

Add `pre_prime_authority_gate` near `_ensure_live_runtime`. Check `/`, `/health/live`, `/health/ready`, and `/info`; require HTTP 200 for all, `ready=true`, exact `runtime.source_sha`, and `_model_contract(runtime)["passed"] is True`. Store only safe statuses and contract booleans, never headers or credentials. Request failures return a failed structured result.

In `run_g9`, initialize `generation_post_count=0`, execute the gate before PRIME, and on failure write `00-pre-prime-authority-gate.json`, readiness/final adjudication evidence, `PRE_PRIME_AUTHORITY_GATE=FAIL`, and `GENERATION_POST_COUNT=0`; return `1` without any generation route. Add an optional counter to `_submit_and_poll` and increment immediately before each POST so successful evidence is accurate.

- [ ] **Step 4: Run all G9 orchestration tests and verify GREEN**

Run `python3 -m unittest tests.test_g9_orchestration -v`. Expected: all existing one-PRIME/two-identical-HOT assertions and new zero-POST gate tests pass.

- [ ] **Step 5: Commit the isolated task**

```bash
git add scripts/final_tpu_one_shot.py tests/test_g9_orchestration.py
git commit -m "feat: gate G9 prime on authenticated runtime authority"
```

### Task 3: Make JAX compile/cache evidence directly observable

**Files:**
- Modify: `src/gemma4_server/tpu/observability.py:CompilationEvidenceCapture and adjudicate_hot_cache`
- Modify: `src/gemma4_server/workers/worker.py:model_worker_main loader`
- Modify: `tests/test_observability.py`
- Modify: `src/gemma4_server/tpu/generation.py` only if an explicit capture status parameter is required; preserve the existing `model.generate` call exactly.

**Interfaces:**
- Consumes: production JAX configuration and the existing `jax` logger hierarchy.
- Produces: `enable_jax_compile_logging() -> dict[str, Any]`, capture snapshots with `compile_logging_enabled` and `coverage_verified`, and hot adjudication fields `HOT_CACHE_REUSE`, `HOT_PREFILL_COMPILE_SECONDS`, and `HOT_DECODE_COMPILE_SECONDS`.

- [ ] **Step 1: Write failing observability tests**

Patch `enable_jax_compile_logging` with an enabled result and assert an empty capture has `available=True`, `status="direct"`, `compile_logging_enabled=True`, `coverage_verified=True`, and `compile_event_count=0`. Add an unavailable-coverage case asserting `available=False`. Update hot adjudication tests to assert:

```python
assert result["HOT_CACHE_REUSE"] is True
assert result["HOT_PREFILL_COMPILE_SECONDS"] == 0.0
assert result["HOT_DECODE_COMPILE_SECONDS"] == 0.0
```

Add a failing case for a HOT compile event and a HOT persistent-cache miss.

- [ ] **Step 2: Run the focused tests and verify RED**

Run `python3 -m unittest tests.test_observability -v`. Expected: failure because the current observer neither enables JAX compile logging nor exposes/validates coverage and uppercase primary fields.

- [ ] **Step 3: Implement minimal observability-only instrumentation**

Implement `enable_jax_compile_logging` by importing JAX, calling `jax.config.update("jax_log_compiles", True)`, and verifying `bool(jax.config.jax_log_compiles)`; return a redacted status/error structure. Have capture entry attach to the existing covered logger hierarchy and set `coverage_verified` only when attachment and enabled logging both succeed. Call the helper once in `model_worker_main` after JAX import and before model load/generation.

Update `adjudicate_hot_cache` so passing requires direct, coverage-verified PRIME/HOT captures, zero HOT compile events, and zero HOT persistent-cache misses. Return existing lowercase fields plus exact uppercase fields; passing sets both hot numeric fields to `0.0`, failing sets them to `None`. Keep cleanup and event filtering behavior unchanged.

- [ ] **Step 4: Run focused and dependent generation tests**

Run `python3 -m unittest tests.test_observability tests.test_generation -v`. Expected: PASS without changing any generation plan or `model.generate` call signature.

- [ ] **Step 5: Commit the isolated task**

```bash
git add src/gemma4_server/tpu/observability.py src/gemma4_server/workers/worker.py tests/test_observability.py
git commit -m "feat: expose direct JAX compile observer coverage"
```

### Task 4: Publish primary G9 evidence and pre-live bucket adjudication

**Files:**
- Modify: `scripts/final_tpu_one_shot.py:adjudicate_g9 and run_g9 evidence rows`
- Create: `docs/G9-CORRECTIVE-BUCKET-ADJUDICATION.md`
- Modify: `tests/test_g9_orchestration.py` and `tests/test_source_contract.py` if static evidence-contract assertions need updating.

**Interfaces:**
- Consumes: the structured pre-PRIME gate and direct hot-cache result from Tasks 2–3; existing G8 evidence in `/kaggle/working/temp/g8-final-check.gDzq7j/artifacts/g8/`.
- Produces: primary G9 acceptance/evidence with direct `PRE_PRIME_AUTHORITY_GATE`, `GENERATION_POST_COUNT`, `HOT_CACHE_REUSE`, `HOT_PREFILL_COMPILE_SECONDS`, and `HOT_DECODE_COMPILE_SECONDS`; a committed bucket rationale that precedes any live attempt.

- [ ] **Step 1: Write failing evidence assertions**

Extend G9 adjudication tests to assert `adjudicate_g9` preserves the uppercase hot fields and returns `G9_STATUS=CLOSED/PASS` only with direct zero values. Add a static-contract test that the canonical one-shot script contains the gate before the first G9 generation call and that the G10 function and `/restart` flow remain present.

- [ ] **Step 2: Run focused tests and verify RED**

Run `python3 -m unittest tests.test_g9_orchestration tests.test_source_contract -v`. Expected: failure because current G9 output does not publish the required uppercase hot fields or a bucket adjudication artifact.

- [ ] **Step 3: Implement primary evidence and rationale**

Add direct hot fields to `rows`, `adjudicate_g9`, `07-cache-compile-evidence.json`, and `13-g9-acceptance.json`; include gate status and generation POST count in acceptance/final adjudication files. Create `docs/G9-CORRECTIVE-BUCKET-ADJUDICATION.md` stating: request `prompt=hello`, `system=''`, `max_new_tokens=1`; normal production bucket `16`; G8 cold semantics were the new worker lifecycle and first-inference/warm behavior was deferred to G9; historical G9 evidence is not proof of a new cold cache. Require a new process/cache authority proving bucket 16 cold before corrective live execution; otherwise stop and explicitly adjudicate another deterministic existing production bucket. Do not alter the request to manufacture a pass.

- [ ] **Step 4: Run focused tests and inspect exact diff**

Run `python3 -m unittest tests.test_g9_orchestration tests.test_source_contract -v` and inspect `git diff HEAD~1 -- scripts/final_tpu_one_shot.py docs/G9-CORRECTIVE-BUCKET-ADJUDICATION.md`. Expected: tests pass; diff contains only evidence/gate wiring and no model semantics or G10 changes.

- [ ] **Step 5: Commit the isolated task**

```bash
git add scripts/final_tpu_one_shot.py docs/G9-CORRECTIVE-BUCKET-ADJUDICATION.md tests/test_g9_orchestration.py tests/test_source_contract.py
git commit -m "feat: publish G9 authority and hot-cache evidence"
```

### Task 5: Run complete static qualification and promotion checks

**Files:**
- Modify only if a test or static contract reveals a real regression; otherwise no source changes.
- Read-only inputs: `notebooks/kaggle-tpu-v5e8-final-run-all.ipynb`, historical evidence under `/kaggle/working/temp` and `/kaggle/working/artifacts/g9`.

**Interfaces:**
- Consumes: the committed corrective tree and existing frozen evidence.
- Produces: verified green local source, clean diff, a new main SHA candidate, and CI status; it does not authorize live G9 until all gates pass.

- [ ] **Step 1: Run focused regression suite**

```bash
python3 -m unittest tests.test_start_env tests.test_g9_orchestration tests.test_observability tests.test_generation -v
```

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: PASS with zero failures/errors.

- [ ] **Step 3: Run static and syntax checks**

```bash
python3 -m compileall -q src scripts tests
bash -n scripts/_common.sh scripts/start.sh scripts/stop.sh
python3 -m json.tool notebooks/kaggle-tpu-v5e8-final-run-all.ipynb >/dev/null
```

Expected: all commands exit `0`.

- [ ] **Step 4: Run security, provenance, and semantic diff checks**

Confirm no credential files are tracked or newly added:

```bash
git status --short
git diff --check
git ls-files | grep -Ei '(^|/)(\.env|.*(api[_-]?key|restart[_-]?secret|credential|token).*)$' || true
git diff -- scripts/start.sh scripts/_common.sh scripts/final_tpu_one_shot.py src/gemma4_server/tpu/observability.py src/gemma4_server/workers/worker.py src/gemma4_server/tpu/generation.py
```

Expected: no credentials tracked; no generation semantic changes; G10 orchestration remains intact; initial G9 evidence remains untouched.

- [ ] **Step 5: Commit any final qualification-only correction and push `main`**

If Step 4 identifies a required source correction, repeat its failing-test-first cycle, then run Steps 1–4 again. When clean:

```bash
git status --short --branch
git push origin main
```

Expected: push succeeds; record the new `FINAL_TPU_EXECUTION_SHA` from `git rev-parse HEAD` only after canonical CI reports PASS. Do not start live G9, release TPU, restart the session, run G10, tag, or release in this plan execution.
