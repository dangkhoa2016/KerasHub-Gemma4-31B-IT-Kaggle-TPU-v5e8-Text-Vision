# G8 live-model necessity decision

```text
G8_SCOPE_RESOLUTION=PASS
G8_CPU_VERIFICATION=PASS
G8_COLD_COMPILE_SEMANTICS_RESOLVED=true
G8_LIVE_MODEL_LIFECYCLE_REQUIRED=false
G8_LIVE_COLD_COMPILE_REQUIRED=false
G8_LIVE_RESTART_REQUIRED=false
G8_TPU_REQUIRED=false
NEW_TPU_RUN_STARTED=false
```

## Decision

CPU-only closure is authorized for Task 6. No canonical repository document
explicitly requires a real 31B model load, first-request compile measurement,
live readiness observation, live worker restart, or post-restart model-backed
request for G8. The canonical roadmap labels G8 as async/cold compile plus
restart/lifecycle acceptance and labels G9 as PRIME/HOT. The API and
architecture documents define the lifecycle surface and one-worker topology;
they do not add a live-model authority requirement.

## Evidence coverage

- Async submission, polling, terminal/error behavior, restart job behavior,
  stale-generation isolation, readiness transitions, one-worker ownership,
  and shutdown are covered by the deterministic CPU tests in
  `artifacts/g8/g8-test-matrix.md`; the complete repository suite and required
  static checks are recorded in `artifacts/g8/g8-cpu-verification.txt`.
- Cold compile is resolved to the new-worker load boundary and the explicit
  `model.compile(..., run_eagerly=True)` call, as recorded in
  `artifacts/g8/g8-cold-compile-resolution.md/json`.
- Warm steady-state, hot-cache, latency, throughput, PRIME, and HOT evidence
  are not required for G8 and remain deferred to G9.

## Canonical citations

- `docs/ROADMAP.md:38-49`: G8 is async/cold compile/restart/lifecycle and G9
  is PRIME/HOT.
- `docs/ARCHITECTURE.md:7-24`: CPU coordinator, bounded queue, one spawned TPU
  worker, one logical model, and long-lived load/compile lifecycle.
- `docs/API.md:11-41`: async, result polling, health, and authenticated restart
  endpoints; no live authority wording.
- `docs/superpowers/specs/2026-09-14-g8-async-cold-compile-restart-lifecycle-design.md:86-111,123-129`:
  deterministic fake CPU lifecycle proof, resolved cold-compile interpretation,
  and live TPU only when canonical material makes it mandatory.
- `directive:715-746`: live TPU is required only for an explicit canonical
  live behavior that CPU/static evidence cannot honestly prove.

No TPU preparation, JAX/Gemma import, model load, checkpoint access, server
process, or live lifecycle was performed.
