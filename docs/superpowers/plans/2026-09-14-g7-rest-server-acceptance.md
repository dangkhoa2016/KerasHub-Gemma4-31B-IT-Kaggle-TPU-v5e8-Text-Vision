# G7 REST Server Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve G7 from the canonical repository, prove the REST contract on CPU/static paths, and close G7 only if the canonical scope is fully satisfied without a TPU run unless live model-backed REST is explicitly required.

**Architecture:** Preserve the Flask + Waitress CPU coordinator and its single spawned TPU worker. Add only the narrow CPU-verifiable guard needed to keep coordinator startup idempotent, then exercise HTTP behavior with a fake manager/model boundary. Record scope, topology, verification, and adjudication as G7 evidence that references frozen G6 rather than rewriting it.

**Tech Stack:** Python 3, Flask test client, unittest/pytest, Waitress source inspection, JSON/Markdown evidence, shell syntax checks, SHA-256.

**Spec:** `/kaggle/working/gemma4-31b-g7-rest-server-acceptance-cpu-first-fast-track-codex-directive-2026-09-14.md`

## Global Constraints

- `TAG=false` and `RELEASE=false` throughout.
- G3, G4, G5, and G6 are immutable authority inputs; do not rerun or mutate their science/evidence.
- Keep TPU OFF through discovery, scope resolution, source review, tests, and evidence packaging.
- Preserve `MODEL_PRESET=gemma4_instruct_31b`, JAX, `bfloat16`, 8 devices, mesh `[1,8]`, and Candidate-A.
- Do not consume G8 async/cold-compile/restart/lifecycle acceptance or G9 PRIME/HOT acceptance.
- A live TPU attempt is permitted only if canonical documents require live model-backed REST and CPU verification is PASS; maximum one attempt, no automatic corrective rerun.

### Task 1: Freeze G6 input and resolve the canonical G7 contract

**Files:**
- Create: `artifacts/g7/g7-scope-resolution.md`
- Create: `artifacts/g7/g7-scope-resolution.json`
- Create: `artifacts/g7/g7-server-inventory.md`
- Create: `artifacts/g7/g7-server-inventory.json`

**Interfaces:**
- Consumes: frozen G6 archive and `artifacts/g6/*`, `docs/G6-CLOSEOUT.md`, current status/roadmap/README/architecture/API documents.
- Produces: exact G7 requirement rows, G7/G8/G9 boundary classification, and the one-process/one-worker source inventory used by later adjudication.

- [ ] **Step 1: Verify the frozen G6 outer hash, sidecar, internal `SHA256SUMS`, acceptance JSON, and final adjudication.**
- [ ] **Step 2: Record `G6_EXPLICIT_FREEZE_ARTIFACT_PRESENT` and whether repository conventions require a new freeze; reuse the existing G6 closeout when it is sufficient.
- [ ] **Step 3: Read the canonical English and Vietnamese status, roadmap, README, architecture, API, and Kaggle documents and enumerate each REST requirement with source context and verification class.
- [ ] **Step 4: Inventory `src/server.py`, `src/gemma4_server/api/app.py`, the manager/worker, validation, scripts, and existing API tests without importing Gemma/JAX.
- [ ] **Step 5: Write the scope and server inventory artifacts, explicitly deferring full async/cold-compile/restart/lifecycle work to G8 and PRIME/HOT work to G9.**

### Task 2: Make coordinator startup idempotent under a failing CPU test

**Files:**
- Create: `tests/test_g7_rest_contract.py`
- Modify: `src/gemma4_server/workers/manager.py`

**Interfaces:**
- Consumes: `GenerationManager.start_async()` and the CPU-only fake manager/job boundary.
- Produces: a `start_async()` call that cannot create a second worker when the configured single worker is already active or starting.

- [ ] **Step 1: Add a focused test that calls `start_async()` twice with `_start_worker` replaced by a CPU-only counter and asserts one collector and one worker start.**
- [ ] **Step 2: Run `python3 -m pytest -q tests/test_g7_rest_contract.py -k start_async` and confirm it fails because the current implementation starts twice.**
- [ ] **Step 3: Add the smallest lock-protected active/startup guard in `GenerationManager.start_async()`; do not alter worker loading, model, checkpoint, mesh, or restart science.**
- [ ] **Step 4: Re-run the focused test and then the complete G7 contract test file.**

### Task 3: Complete CPU/static REST contract coverage

**Files:**
- Modify: `tests/test_g7_rest_contract.py`
- Create: `artifacts/g7/g7-rest-contract-test-matrix.md`

**Interfaces:**
- Consumes: `create_app(Runtime(...))`, `Config.for_tests()`, fake manager/job responses, and source-level route/topology inspection.
- Produces: evidence-backed tests for route registration, auth, health/readiness, info, validation/error schema, text/image submission mechanics, result lookup, and single-worker configuration; lifecycle and cold-compile execution remain G8.

- [ ] **Step 1: Add CPU tests for all canonical G7 HTTP surface rows: `/`, `/health/live`, `/health/ready`, `/info`, `/generate`, `/generate/async`, `/generate/image`, `/generate/image/async`, and `/result/<job_id>`.**
- [ ] **Step 2: Add tests for bearer/API-key authentication, request IDs, malformed/missing JSON, invalid generation limits, image fixture parsing, status codes, JSON content types, and exception-to-error-schema mapping.**
- [ ] **Step 3: Add static assertions that `src/server.py` uses one Waitress process, no auto-reload or multi-worker option, and the manager has exactly one configured worker identity.**
- [ ] **Step 4: Run the complete G7 test file and the repository-native test suite, then run `compileall`, `bash -n`, and project JSON parsing.**
- [ ] **Step 5: Record exact test counts and each row's result in the contract test matrix, including deferred G8/G9 rows.**

### Task 4: Decide live necessity, package evidence, and adjudicate G7

**Files:**
- Create: `artifacts/g7/g7-cpu-verification.txt`
- Create: `artifacts/g7/g7-live-necessity-decision.md`
- Create: `artifacts/g7/g7-live-necessity-decision.json`
- Create: `artifacts/g7/g7-acceptance-matrix.md`
- Create: `artifacts/g7/g7-acceptance.json`
- Create: `docs/G7-CLOSEOUT.md` only if every mandatory G7 row passes
- Create: one compact G7 archive, internal `SHA256SUMS`, and outer `.sha256` only if G7 closes

**Interfaces:**
- Consumes: Tasks 1–3 evidence and the canonical source requirements.
- Produces: a truthful CPU-only G7 closeout when live model REST is not canonical, or a precise open/blocker result when a required live proof is unavailable or fails.

- [ ] **Step 1: Set `G7_CPU_VERIFICATION=PASS` only after all CPU/static checks pass and record any harness-only G4 archive-path normalization separately from G7 evidence.**
- [ ] **Step 2: Derive `G7_LIVE_MODEL_BACKED_REST_REQUIRED` and all live requirement flags from canonical documents; do not infer them from this plan or preference.**
- [ ] **Step 3: Keep TPU off when canonical scope is CPU-satisfiable; otherwise stop before enabling TPU unless the exact one-run authority plan is complete and explicitly authorized by the resolved contract.**
- [ ] **Step 4: Write the acceptance matrix and structured adjudication with `NEW_TPU_RUN_STARTED=false`, `TAG=false`, and `RELEASE=false` for the CPU-only path.**
- [ ] **Step 5: If and only if closed, package and verify internal and outer hashes, then create `docs/G7-CLOSEOUT.md`; otherwise create a progress/blocker report and leave G7 open.**
