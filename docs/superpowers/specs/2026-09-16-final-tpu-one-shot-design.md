# Final TPU One-Shot G9/G10 Design

## Goal

Make the frozen repository capable of producing authoritative G9 PRIME/HOT
cache/compile evidence and of executing the complete fresh-session G10
acceptance contract, while preserving the frozen Gemma4 runtime semantics.

## Scope and non-goals

This design covers only the pre-live corrective and orchestration work needed
before G9. It does not authorize a generation request, a Kaggle restart, a TPU
release, a tag, or a GitHub release. Those are separate gated operations from
the supplied final directive.

The corrective must not change model arguments, sampler behavior, prompt
construction, bucket selection, dtype, mesh, checkpoint loading, sharding,
or REST payload semantics. It may add evidence capture and deterministic
acceptance orchestration around the existing interfaces.

## Current evidence

- Frozen CPU authority is `e7a58636ba2fba0f4ed49b2d2bd4fb282968ab26`.
- The canonical repository baseline is clean and its full CPU suite passes
  133 tests.
- The installed runtime versions are JAX 0.10.2, Keras 3.15.0, and
  KerasHub 0.29.1.
- The current engine emits prompt, bucket, generation-time, and mode fields,
  but no direct compile/cache events.
- The current final notebook clones the repository and proves the SHA, then
  prints `FINAL_TPU_ONE_SHOT=READY`; it does not run model, vision, REST, or
  memory acceptance.
- No healthy server/worker is currently available for safe G9 reuse.

## Architecture

### 1. Request-scoped compile evidence

Add a project-owned observability helper that installs a temporary logging
handler on the JAX compilation logger for one generation call. It records
structured events only when JAX emits its authoritative compile/cache messages
and always removes the handler in `finally`. The helper must be inert when
disabled and must not clear caches, alter JAX configuration, or invoke any
extra model operation.

The engine wraps the existing native `model.generate()` call with this helper
and adds an `observability` object to returned metrics. The existing generation
arguments and returned text remain unchanged. The metrics distinguish:

- observed compilation events during the request;
- observed persistent-cache hits/misses when JAX emits them;
- request-scoped compile event count and elapsed compile seconds;
- whether the evidence is direct, unavailable, or malformed.

For HOT acceptance, the same request shape is sent three times in one worker
process. `HOT_CACHE_REUSE=true` is emitted only when the second and third
requests have direct request-scoped evidence of zero compilation events and
the evidence source is available. Missing logs remain `NOT_AVAILABLE` and
fail the gate; wall-clock speed is never used as a proxy.

### 2. Tracked TPU orchestrator

Add one tracked Python script, `scripts/final_tpu_one_shot.py`, as the
single orchestration entry point. It owns only orchestration and evidence
packaging; it imports the production server and clients rather than duplicating
model implementation. It will:

1. verify the canonical SHA and clean tree;
2. capture session, host, dependency, TPU, cgroup, source, and runtime
   metadata without secrets;
3. resolve the model path and validate the eight-device TPU contract;
4. start the production server once and wait for readiness;
5. call `/`, health, and `/info`;
6. execute tracked semantic text and vision requests through async REST;
7. poll results and record direct engine metrics, including compile evidence;
8. verify lifecycle endpoints and collector completion;
9. capture memory and OOM counters before and after;
10. write compact G9 or G10 evidence files and verified `SHA256SUMS`;
11. print a machine-readable final adjudication and exit nonzero on any
    mandatory failure.

The script must support an explicit G9 mode and an explicit G10 mode so that
G9 cannot accidentally perform the G10 restart. It must never call `/restart`
as part of G9.

### 3. Thin G10 notebook

Replace the current scaffold cells with a thin notebook that sets the pinned
SHA and mode, clones the public repository after the fresh restart, invokes the
tracked orchestrator, and displays the orchestrator's adjudication. The
notebook itself will not contain model-loading or generation implementation.
The orchestrator will prove fresh-session identity from runtime values and
will fail if the exact pinned SHA, eight TPU devices, BF16 metadata, Candidate-A
sharding, strict loading, text, vision, REST, or OOM requirements are absent.

## Error and gate behavior

- Static/orchestration failures before live G9 may be corrected, tested, and
  pushed as a new execution SHA.
- A failed live PRIME is terminal for this campaign; no HOT retry, G10 restart,
  release, or tag follows.
- `NOT_AVAILABLE` compile/cache evidence is a failure, not a zero.
- The orchestrator writes evidence on failure when possible, but never labels
  an incomplete gate as PASS.
- Evidence excludes weights, caches, credentials, `.env`, PID files, private
  logs, runtime databases, and restart secrets.
- After G10 PASS, only documentation/evidence/release files may change; no
  production runtime or notebook orchestration changes are allowed.

## Testing strategy

Before any live request:

- add CPU unit tests for event parsing, handler cleanup, unavailable evidence,
  HOT zero-compile adjudication, source/contract checks, and orchestrator
  failure behavior using fakes;
- run the focused tests;
- run the complete existing unittest suite;
- run `python3 -m compileall -q src scripts clients/python`;
- run Bash syntax checks and notebook JSON parsing;
- inspect the diff for forbidden runtime-semantic changes;
- commit and push the corrective/orchestrator;
- wait for canonical CI PASS and freeze the new SHA.

## Operational boundaries

The existing TPU allocation is preserved. No JAX device enumeration, model
load, generation, or cache clearing is performed during this pre-live code
phase. The only intentional Kaggle session restart remains the G9→G10
boundary after G9 has closed PASS.
