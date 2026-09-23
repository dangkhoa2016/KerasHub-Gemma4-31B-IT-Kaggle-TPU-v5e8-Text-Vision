# G9 Corrective Authority and Observability Design

## Goal

Close the confirmed G9 authentication/orchestration defect with the smallest production-safe corrective, while adding a real pre-PRIME authority gate and direct, auditable JAX compile/cache evidence. Preserve the historical failed G9 attempt and keep the G10/release gates untouched.

## Scope and invariants

- The historical G9 attempt remains `FAIL` and its evidence directory and SHA256 manifest are never overwritten.
- No TPU release, CPU fallback, deliberate G9→G10 restart, G10 run, tag, or release occurs before a corrective G9 closeout.
- A non-empty process environment `API_KEY` or `RESTART_SECRET` has precedence over blank/default values sourced from `.env`.
- Authentication remains enabled; no credential is committed or written into evidence.
- Frozen model-generation semantics remain unchanged: native generation, existing request payloads, sampler, bucket policy, dtype, mesh, Candidate-A, and loader are out of scope.
- A corrective G9 PRIME is allowed only after static qualification and canonical CI pass on the new source SHA.

## Design

### Environment precedence

Update `scripts/_common.sh:load_env` to snapshot only the two orchestrator-supplied secret variables when they are non-empty, source `.env` for normal configuration behavior, then restore those snapshots. This preserves all unrelated `.env` loading and prevents blank/default `.env` assignments from replacing explicit credentials.

Regression coverage runs the shell function in a temporary project directory and verifies both secret variables survive blank `.env` values. A companion case verifies ordinary `.env` configuration still loads. Test values are synthetic and never emitted into evidence or committed.

### Pre-PRIME authority gate

Add a small gate in `scripts/final_tpu_one_shot.py` that runs before the first `/generate/async` POST. It records checks for `/`, `/health/live`, `/health/ready` with `ready=true`, and authenticated `/info`. It also validates the exact expected `runtime.source_sha` and the existing frozen runtime contract. The gate returns a structured result with `passed`, endpoint checks, source-SHA check, runtime-contract check, and a redacted failure reason.

`run_g9` must stop immediately on gate failure. It must not call any generation route, must record `PRE_PRIME_AUTHORITY_GATE=FAIL` and `GENERATION_POST_COUNT=0`, package only non-secret gate evidence, and return failure. The existing test doubles will be extended with `source_sha` and the frozen contract; a regression test will make `/health/ready` pass and `/info` return `401`, then assert zero `/generate/async` calls.

### Direct JAX compile/cache observability

Keep instrumentation observability-only. The production worker will explicitly enable JAX compile logging before generation capture, while the capture exposes whether the handler was attached to the covered production logger hierarchy and whether coverage was verified. Empty captures are authoritative only when the observer is available, direct, and coverage verification succeeds.

The capture snapshot will retain direct event counts, durations, cache hits/misses, observer status, and coverage fields. Hot-cache adjudication will publish `HOT_CACHE_REUSE`, `HOT_PREFILL_COMPILE_SECONDS`, and `HOT_DECODE_COMPILE_SECONDS` directly in the primary G9 acceptance/evidence payload, with zero values only after direct HOT-1/HOT-2 zero-compile and zero-cache-miss checks pass.

Tests cover successful coverage verification, unavailable coverage, direct zero-compile adjudication, and failure on HOT compile/cache-miss evidence. No model `generate` arguments or runtime generation policy are changed.

### PRIME bucket adjudication

Before any corrective live attempt, add a timestamped authority note identifying the production request shape (`hello`, `max_new_tokens=1`) and its normal bucket (`16`), summarize the existing G8 bucket-16 evidence from `temp`, and state whether bucket 16 is directly proven cold in the new process/cache authority. If it is not proven cold, the note must explicitly select and justify another deterministic production bucket rather than changing a request opportunistically. This note is a prerequisite, not a live retry.

## Verification and promotion gates

After implementation, run focused tests, the full test suite, `compileall`, `bash -n` on relevant scripts, notebook JSON/static contract checks, a no-secrets check, and a diff audit proving frozen runtime semantics and G10 orchestration are unchanged. Commit and push the corrective to `main`, wait for canonical CI PASS, and only then freeze `FINAL_TPU_EXECUTION_SHA` and authorize the separate corrective G9 live sequence.

