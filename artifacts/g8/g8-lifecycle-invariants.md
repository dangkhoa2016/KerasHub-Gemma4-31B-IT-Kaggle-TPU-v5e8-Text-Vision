# G8 lifecycle invariants

| ID | Required invariant | Exact citation | Task 1 status |
|---|---|---|---|
| G8-INV-001 | One worker identity/owner: only `tpu-0` is active for a lifecycle. | `src/gemma4_server/workers/manager.py:GenerationManager.WORKER_ID,_start_worker`; `docs/ARCHITECTURE.md` | Source topology observed; concurrent ownership acceptance pending. |
| G8-INV-002 | `start_async()` is idempotent while a worker is starting/loading/ready/busy. | `src/gemma4_server/workers/manager.py:GenerationManager.start_async` | Source guard observed; deterministic concurrency proof pending. |
| G8-INV-003 | Restart is serialized. | `src/gemma4_server/api/app.py:create_app.restart`; `docs/superpowers/specs/2026-09-14-g8-async-cold-compile-restart-lifecycle-design.md:Lifecycle serialization` | HTTP admission lock observed; manager lifecycle serialization is required G8 work. |
| G8-INV-004 | Readiness is false during loading and restart. | `src/gemma4_server/workers/manager.py:GenerationManager.health,restart_worker`; G8 design `Lifecycle serialization` | Required; current source needs G8 hardening against stale-ready conditions. |
| G8-INV-005 | Readiness becomes true only after current-generation `worker_ready` with expected devices. | `src/gemma4_server/workers/manager.py:GenerationManager._handle,health`; `src/gemma4_server/workers/worker.py:model_worker_main` | Event ordering/source check observed; CPU proof pending. |
| G8-INV-006 | A stale worker event cannot mutate lifecycle or job state. | `src/gemma4_server/workers/manager.py:GenerationManager._handle` | Source returns on generation mismatch before kind dispatch. |
| G8-INV-007 | Terminal jobs are immutable: completed/failed cannot be overwritten. | G8 design `Lifecycle serialization` and `Canonical restart and job behavior`; target methods `src/gemma4_server/jobs/store.py:mark_completed,mark_failed` | Required correction: current methods lack terminal-state guards. |
| G8-INV-008 | Shutdown is idempotent. | `src/gemma4_server/workers/manager.py:GenerationManager.shutdown` | Source returns true if `_shutting_down` is already set; CPU proof pending. |
| G8-INV-009 | Shutdown cannot cause a replacement worker. | `src/gemma4_server/workers/manager.py:GenerationManager._monitor,shutdown`; G8 design `Lifecycle serialization` | Source monitor exits when shutdown flag is set; race-proof pending. |
| G8-INV-010 | Each worker lifecycle invokes the model loader exactly once. | `src/gemma4_server/workers/worker.py:model_worker_main`; `src/gemma4_server/tpu/engine.py:Gemma4TPUEngine.load`; G8 design `Current lifecycle model` | Single call-site observed; deterministic CPU loader-count proof pending. |

These are acceptance invariants, not completion claims. “Pending” identifies work deliberately left to later G8 tasks; it prevents evidence drift from the frozen G7 baseline.
