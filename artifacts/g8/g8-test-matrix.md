# G8 CPU Lifecycle Test Matrix

This matrix maps the canonical G8 lifecycle invariants and scoped failure
paths to deterministic CPU-only evidence. The tests use standard-library
fakes; concurrent ordering is controlled with `threading.Event` or
`threading.Barrier`, not sleep-only timing. They do not load Gemma, JAX, TPU
devices, or checkpoints.

| ID / scope | Test evidence | Verification class | Expected state transition | Observed result | Live model evidence required |
|---|---|---|---|---|---|
| G8-INV-001 one worker owner | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_start_and_restart_race_has_one_worker_owner` | CPU_INTEGRATION | `none -> starting(g1)` exactly once while a restart contends | PASS: one start and generation 1 | No |
| G8-INV-002 idempotent start | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_start_async_is_idempotent_during_loading` | CPU_INTEGRATION | `none -> starting(g1)`; second start remains no-op | PASS: `[True, False]`, one start | No |
| G8-INV-003 serialized restart | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_two_simultaneous_restarts_have_one_transition` | CPU_INTEGRATION | `ready(g1) -> restarting -> starting(g2)` once | PASS: one replacement, one rejected contender | No |
| G8-INV-004 readiness gates | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_health_is_not_ready_during_loading`; `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_health_is_not_ready_during_restart` | CPU_UNIT | `ready metadata -> loading/restarting`, `ready=False` | PASS: readiness is false in both states | No |
| G8-INV-005 current ready event | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_health_becomes_ready_only_after_worker_ready`; `tests.test_g8_worker_protocol.G8WorkerProtocolTests.test_successful_fake_load_emits_ready_before_job_lifecycle` | CPU_INTEGRATION | `loading -> ready -> busy -> ready` | PASS: ready follows successful fake load only | No |
| G8-INV-006 stale event isolation | `tests.test_g8_async_jobs.AsyncManagerTests.test_old_completion_after_replacement_generation_is_current_is_ignored`; `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_restart_invalidates_stopping_generation_before_queued_events` | CPU_INTEGRATION | `processing(g2) -> failed/replacement starting(g3)`; stale `completed(g2)` is ignored | PASS: job remains failed and replacement remains starting | No |
| G8-INV-007 terminal job immutability | `tests.test_g8_async_jobs.AsyncJobStoreTests.test_failed_job_cannot_be_completed_by_late_worker_event`; `tests.test_g8_async_jobs.AsyncJobStoreTests.test_completed_result_survives_restart` | CPU_UNIT | `queued/processing -> terminal`; later terminal event is no-op | PASS: first terminal result/error survives | No |
| G8-INV-008 idempotent shutdown | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_shutdown_is_idempotent`; `tests.test_g8_worker_protocol.G8WorkerProtocolTests.test_shutdown_is_idempotent` | CPU_INTEGRATION | `ready -> stopping/stopped`; repeated shutdown has no extra work | PASS: one join and one fake-loader call | No |
| G8-INV-009 no replacement on shutdown | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_shutdown_does_not_spawn_replacement` | CPU_INTEGRATION | `ready -> stopping`; monitor exit cannot transition to replacement | PASS: zero replacement starts | No |
| G8-INV-010 one loader per lifecycle | `tests.test_g8_worker_protocol.G8WorkerProtocolTests.test_successful_fake_load_emits_ready_before_job_lifecycle`; `tests.test_g8_worker_protocol.G8WorkerProtocolTests.test_shutdown_is_idempotent` | CPU_UNIT | `loading -> ready/stopped` with one loader invocation | PASS: loader count is one per protocol invocation | No |
| Loader raises | `tests.test_g8_worker_protocol.G8WorkerProtocolTests.test_loader_failure_emits_error_without_ready` | CPU_UNIT | `loading -> worker_load_error` | PASS: no ready event; monitor stopped | No |
| Simulated compile raises in loader | `tests.test_g8_worker_protocol.G8WorkerProtocolTests.test_compile_failure_inside_loader_emits_load_error_without_ready` | CPU_UNIT | `loading -> worker_load_error` after event-released fake compile | PASS: no ready event; exact compile error emitted | No |
| Generation raises | `tests.test_g8_worker_protocol.G8WorkerProtocolTests.test_generation_failure_emits_job_failed` | CPU_UNIT | `ready -> busy -> job_failed -> stopped` | PASS: terminal job failure event includes generation exception | No |
| Cleanup raises | `tests.test_g8_worker_protocol.G8WorkerProtocolTests.test_cleanup_failure_follows_completed_job_event` | CPU_UNIT | `ready -> completed`; monitor cleanup then raises | PASS: `job_completed` is emitted before the injected cleanup exception propagates | No |
| Replacement load failure | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_replacement_load_failure_leaves_manager_unavailable` | CPU_INTEGRATION | `ready(g1) -> restarting -> starting(g2) -> failed/unavailable` | PASS: one replacement attempt; readiness and acceptance remain false | No |
| Restart during loading | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_start_and_restart_race_has_one_worker_owner` | CPU_INTEGRATION | `starting(g1)` with restart contender | PASS: restart is rejected while start owns lifecycle transition | No |
| Restart during busy | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_restart_during_busy_fails_job_before_replacement_starts` | CPU_INTEGRATION | `busy(g1) -> restarting`, pending job failed, then `starting(g2)` | PASS: job fails before replacement; contender rejected | No |
| Two start calls during loading | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_start_async_is_idempotent_during_loading` | CPU_INTEGRATION | `starting(g1)` plus concurrent start | PASS: exactly one start | No |
| Two restart calls | `tests.test_g8_manager_lifecycle.ManagerLifecycleTests.test_two_simultaneous_restarts_have_one_transition` | CPU_INTEGRATION | `ready(g1) -> restarting -> starting(g2)` | PASS: exactly one transition | No |
| Async polling/failure representation | `tests.test_g8_async_jobs.AsyncManagerTests.test_async_polling_reports_queued_processing_and_terminal_states`; `tests.test_g8_async_jobs.AsyncManagerTests.test_generation_failure_is_http_500_with_json_error` | CPU_INTEGRATION | `queued -> processing -> completed/failed` | PASS: 202 pending, 200 completed, 500 failed JSON | No |
| G9 PRIME/HOT behavior | Deferred by `docs/superpowers/specs/2026-09-14-g8-async-cold-compile-restart-lifecycle-design.md:Non-goals` | DEFERRED_TO_G9 | Not exercised in G8 | DEFERRED_TO_G9 | Yes |

Focused execution evidence: `python3 -m unittest -v tests.test_g8_worker_protocol tests.test_g8_manager_lifecycle tests.test_g8_async_jobs` (31 tests, zero failures/errors).

Evidence source files: [worker protocol](../../tests/test_g8_worker_protocol.py),
[manager lifecycle](../../tests/test_g8_manager_lifecycle.py),
[async jobs](../../tests/test_g8_async_jobs.py), and
[lifecycle invariants](g8-lifecycle-invariants.md). Every test reference in
the matrix is an exact runnable `module.Class.method` identifier.
