# G8 Async / Cold Compile / Restart / Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Close G8 with evidence-backed async job, cold worker load/compile, restart, and shutdown lifecycle acceptance while preserving frozen G3-G7 authority.

**Architecture:** Keep the existing Flask/Waitress coordinator, one spawned tpu-0 worker, multiprocessing queues, generation-token stale-event barrier, and JobStore. Add a CPU-testable worker protocol seam, serialize manager lifecycle operations, make readiness accurately unavailable during restart, and prevent terminal job results from being overwritten by stale worker events.

**Tech Stack:** Python 3, unittest, threading.Event/Barrier, multiprocessing, Flask test client, JSON artifacts, Bash/Python verification commands.

**Spec:** docs/superpowers/specs/2026-09-14-g8-async-cold-compile-restart-lifecycle-design.md

## Global Constraints

- Continue only from G7=CLOSED/PASS; do not rerun or rewrite G3-G7 authority.
- MODEL_PRESET=gemma4_instruct_31b, KERAS_BACKEND=jax, MODEL_DTYPE=bfloat16, TPU v5e-8, eight devices, mesh [1,8], axes [batch,model] remain frozen.
- Preserve Candidate-A sharding and the frozen strict-load method; do not change checkpoint identity, dtype, mesh, or sharding.
- Preserve run_eagerly=True; do not introduce whole-model JIT or speculative compile paths.
- CPU tests must never load Gemma, JAX TPU devices, or a real checkpoint.
- Use deterministic synchronization primitives; do not rely on sleep-only race tests.
- No live TPU run is permitted before CPU/static verification and the live-necessity decision pass.
- If live authority is required, allow exactly one primary attempt and zero automatic corrective reruns.
- Do not perform G9 PRIME/HOT, warm benchmark, throughput, or latency characterization.
- Keep TAG=false and RELEASE=false.
- The repository is not Git-backed; do not invent commit/HEAD/worktree claims.

---

### Task 1: Freeze G8 scope, inventory, and lifecycle contract evidence

**Files:**
- Create: artifacts/g8/g8-scope-resolution.md
- Create: artifacts/g8/g8-scope-resolution.json
- Create: artifacts/g8/g8-lifecycle-inventory.md
- Create: artifacts/g8/g8-lifecycle-inventory.json
- Create: artifacts/g8/g8-lifecycle-invariants.md
- Create: artifacts/g8/g8-restart-contract.md
- Create: artifacts/g8/g8-async-job-state-machine.md

**Interfaces:**
- Consumes: DEVELOPMENT-STATUS.md, roadmap/status/architecture/API docs, and current manager, worker, JobStore, and Flask source.
- Produces: explicit G8 requirement rows, source-derived lifecycle states/transitions, restart behavior, async job behavior, and invariant IDs consumed by later acceptance artifacts.

- [ ] Step 1: Record the verified continuity baseline.

Use the verified archive /kaggle/working/gemma4-31b-g7-rest-server-acceptance-20260914T043113Z.tar.gz and record outer SHA-256 4f1547d6d86d796aca7ce7d396021d42829abb7f932d1713093345d69035c0a2. Record GIT_PRESENT=false, G8_REPO_ROOT as the canonical repo path, and G8_HEAD=NOT_APPLICABLE, G8_WORKTREE_CLEAN=NOT_APPLICABLE.

- [ ] Step 2: Write scope-resolution rows.

Each JSON row must contain exactly: requirement_id, canonical_source, source_heading_or_context, requirement_summary, verification_class, live_model_required, restart_required, cold_compile_required, belongs_to_gate, and notes. Include rows for async submission/polling, readiness, worker startup/load, restart authentication and asynchronous response, restart job behavior, stale event isolation, shutdown, single-worker ownership, cold load/compile semantics, and G9 PRIME/HOT deferral. Use only STATIC, CPU_UNIT, CPU_INTEGRATION, LIVE_REST, or LIVE_MODEL_LIFECYCLE. Mark every PRIME/HOT row belongs_to_gate=G9 and Result=DEFERRED_TO_G9.

- [ ] Step 3: Write the source-derived lifecycle inventory.

Map GenerationManager, model_worker_main, JobStore, Flask request handlers, task queue, result queue, worker process, collector thread, monitor threads, restart thread, and model engine. For each component record constructor, start, stop, restart, identity, locks/events, state fields, load location, and cleanup location. Include only observed worker states starting, loading, ready, busy, failed, and stopped, plus manager-derived loading, ready, restarting, and unavailable.

- [ ] Step 4: Write the canonical restart contract and async job state machine.

Record POST /restart as authenticated plus X-Restart-Secret, returning 202 and running in one background thread; concurrent HTTP restart requests return 409. Record wait_for_jobs=true as waiting for pending jobs, false as failing pending jobs immediately, preservation of completed results, and stale worker events being ignored by generation. For queued, processing, completed, and failed, record entry/exit conditions, HTTP representation, terminality, restart behavior, and error behavior.

- [ ] Step 5: Write lifecycle invariants with source citations.

Include IDs for one worker identity, idempotent start, serialized restart, readiness false during loading/restart, readiness only after worker_ready, generation-token stale-event rejection, terminal job immutability, shutdown idempotence, no replacement after shutdown, and one model loader invocation per worker lifecycle. Cite the exact source method or canonical document for every invariant.

- [ ] Step 6: Validate the evidence files.

~~~bash
python3 -m json.tool artifacts/g8/g8-scope-resolution.json >/dev/null
python3 -m json.tool artifacts/g8/g8-lifecycle-inventory.json >/dev/null
rg -n 'G9|PRIME|HOT|G8|worker|restart|queued|processing|completed|failed' artifacts/g8/g8-*.md
~~~

Expected: both JSON commands exit 0; all G9 rows are explicitly deferred; no G3-G7 file is modified.

---

### Task 2: Add a deterministic CPU worker protocol seam and failure tests

**Files:**
- Modify: src/gemma4_server/workers/worker.py
- Create: tests/test_g8_worker_protocol.py

**Interfaces:**
- Consumes: existing worker event schema and model_worker_main arguments.
- Produces: worker_protocol_loop(worker_id, generation, task_queue, result_queue, shutdown_event, loader, monitor) as a CPU-testable top-level function; model_worker_main remains the multiprocessing entrypoint and supplies the real TPU loader closure.

- [ ] Step 1: Write the first failing test for successful fake load and readiness ordering.

Use queue.Queue, threading.Event, a fake monitor, and a fake engine. Call worker_protocol_loop in a thread with loader() returning (engine, {"device_count": 8, "dtype": "bfloat16"}). Assert the first events are worker_state=loading then worker_ready, readiness is emitted only after loader return, and one task emits job_started followed by job_completed.

- [ ] Step 2: Run the focused test and verify a meaningful RED failure.

~~~bash
python3 -m unittest -v tests.test_g8_worker_protocol
~~~

Expected: import or attribute failure because worker_protocol_loop does not exist yet; fix only test setup errors, not the missing production behavior.

- [ ] Step 3: Implement the minimal protocol loop.

Move the common event protocol from model_worker_main into:

~~~python
def worker_protocol_loop(
    worker_id,
    generation,
    task_queue,
    result_queue,
    shutdown_event,
    loader,
    monitor,
):
    def emit(event_type, **payload):
        result_queue.put({
            "type": event_type,
            "worker_id": worker_id,
            "generation": generation,
            **payload,
        })

    emit("worker_state", state="loading")
    try:
        engine, metadata = loader()
    except Exception as exc:
        emit("worker_load_error", error=repr(exc))
        monitor.stop()
        return

    emit("worker_ready", metadata=metadata)
    while not shutdown_event.is_set():
        task = task_queue.get()
        if task is None:
            break
        emit("job_started", job_id=task["job_id"])
        try:
            if task.get("image") is not None:
                output, metrics = engine.generate_image(
                    task["image"],
                    task["prompt"],
                    task["system"],
                    task["max_tokens"],
                )
            else:
                output, metrics = engine.generate_text(
                    task["prompt"],
                    task["system"],
                    task["max_tokens"],
                )
            emit(
                "job_completed",
                job_id=task["job_id"],
                result=output,
                metrics=metrics,
            )
        except Exception as exc:
            emit("job_failed", job_id=task["job_id"], error=repr(exc))

    monitor.stop()
    emit("worker_stopped", state="stopped")
~~~

The loop emits worker_state=loading, calls loader once, emits worker_load_error and returns on loader exception, emits worker_ready only after success, processes one task at a time, emits job_started, job_completed or job_failed, stops the monitor, and emits worker_stopped on graceful shutdown. model_worker_main retains TPU imports/configuration and passes a closure that constructs exactly one Gemma4TPUEngine and calls engine.load().

- [ ] Step 4: Run the focused test and verify GREEN.

Run the same unittest command. Expected: the successful fake lifecycle test passes with no TPU/JAX import.

- [ ] Step 5: Add controlled failure and shutdown tests.

Add test_loader_failure_emits_error_without_ready, test_generation_failure_emits_job_failed, test_shutdown_is_idempotent, and test_slow_loader_keeps_ready_unemitted. Use an Event to hold/release a slow loader and assert event ordering; call shutdown twice and assert no second loader call and no extra worker event.

- [ ] Step 6: Run RED/GREEN for the new failure tests.

Run each new test before and after the minimal implementation change. Expected final result: all worker protocol tests pass and the fake loader count is one per protocol lifecycle.

---

### Task 3: Serialize manager start/restart/shutdown and prove readiness transitions

**Files:**
- Modify: src/gemma4_server/workers/manager.py
- Create: tests/test_g8_manager_lifecycle.py

**Interfaces:**
- Consumes: GenerationManager.start_async(), restart_worker(), shutdown(), _handle(), _monitor(), and the existing worker event schema.
- Produces: one lifecycle lock protecting start/restart/shutdown, readiness false during _restart_pending, and a restart sequence that joins the old process before starting a replacement.

- [ ] Step 1: Write failing concurrent-start/restart tests.

Add test_start_async_is_idempotent_during_loading, test_start_and_restart_race_has_one_worker_owner, and test_two_simultaneous_restarts_have_one_transition. Replace _start_worker with a controlled fake that blocks on an Event, count starts, and use a Barrier to release two caller threads at once. Assert one active worker generation and one restart transition.

- [ ] Step 2: Run the focused tests and confirm RED.

~~~bash
python3 -m unittest -v tests.test_g8_manager_lifecycle
~~~

Expected: the concurrent restart test demonstrates duplicate transition/start behavior or an unprotected lifecycle path in the current implementation.

- [ ] Step 3: Implement lifecycle serialization minimally.

Add a manager-owned reentrant lifecycle lock. Acquire it in start_async, restart_worker, shutdown, and the automatic restart branch of _monitor. Set _accepting=False and _restart_pending=True before stopping; mark the current worker stopping; join/terminate the old process; set _worker=None; replace the task/result queues and shutdown event; then start exactly one replacement. A second direct restart_worker call returns False while the first transition owns the lock. Keep the existing HTTP Runtime.restart_lock and 409 behavior unchanged.

- [ ] Step 4: Add and run readiness RED/GREEN tests.

Add test_health_is_not_ready_during_restart, test_health_is_not_ready_during_loading, test_health_becomes_ready_only_after_worker_ready, and test_failed_load_is_unavailable. Assert health()["ready"] is false whenever _restart_pending or _accepting is false, regardless of stale ready metadata; assert it becomes true only after the current generation sends worker_ready with the expected device count.

- [ ] Step 5: Add stop-order and shutdown tests.

Add test_restart_joins_old_worker_before_new_start, test_shutdown_joins_worker, test_shutdown_is_idempotent, and test_shutdown_does_not_spawn_replacement. The fake process records join, terminate, and replacement-start events; assert all old-process stop events precede the new-start event and no monitor restart occurs after _shutting_down is set.

- [ ] Step 6: Run the focused manager suite and frozen G7 topology tests.

~~~bash
python3 -m unittest -v tests.test_g8_manager_lifecycle tests.test_g7_rest_contract
~~~

Expected: all new lifecycle tests and existing G7 tests pass; no G7 source/evidence files are edited.

---

### Task 4: Protect async job terminal states and stale worker results

**Files:**
- Modify: src/gemma4_server/jobs/store.py
- Modify: src/gemma4_server/workers/manager.py
- Create: tests/test_g8_async_jobs.py

**Interfaces:**
- Consumes: JobStore.mark_processing, mark_completed, mark_failed, fail_pending, GenerationManager._handle, and Flask result lookup behavior.
- Produces: terminal job immutability, canonical restart failure behavior, and stale-generation protection for queued/processing jobs.

- [ ] Step 1: Write failing terminal-state tests.

Add test_failed_job_cannot_be_completed_by_late_worker_event and test_completed_result_survives_restart. Create a Job, mark it failed with TPU worker restarted, then call mark_completed; assert status remains failed. Create a completed job, invoke restart failure handling, and assert its output remains queryable.

- [ ] Step 2: Run the tests and verify RED.

~~~bash
python3 -m unittest -v tests.test_g8_async_jobs
~~~

Expected: current mark_completed overwrites a failed job, proving the regression test catches the lifecycle defect.

- [ ] Step 3: Implement terminal-state guards.

Change JobStore.mark_completed and mark_failed to update only jobs whose current status is queued or processing; leave completed and failed unchanged. Preserve done.set(), image cleanup, metrics, runtime, and TTL behavior for the first terminal transition.

- [ ] Step 4: Add stale-generation and restart-state tests.

Add test_stale_old_generation_cannot_complete_failed_job, test_restart_without_wait_fails_pending_jobs, test_restart_waits_for_processing_job_when_requested, test_async_polling_reports_queued_processing_and_terminal_states, and test_generation_failure_is_http_500_with_json_error. Feed _handle events for old and current generations and use controlled events for a blocking fake generation; assert old events are ignored and /result/job-123 returns 202 for nonterminal, 200 for completed, and 500 for failed.

- [ ] Step 5: Run focused async tests and existing API tests.

~~~bash
python3 -m unittest -v tests.test_g8_async_jobs tests.test_jobs tests.test_api_contract tests.test_g7_rest_contract
~~~

Expected: all terminal-state, stale-event, async polling, and existing API contract tests pass.

---

### Task 5: Complete CPU failure injection and lifecycle test matrix

**Files:**
- Modify: tests/test_g8_worker_protocol.py
- Modify: tests/test_g8_manager_lifecycle.py
- Modify: tests/test_g8_async_jobs.py
- Create: artifacts/g8/g8-test-matrix.md

**Interfaces:**
- Consumes: corrected manager/store/worker protocol.
- Produces: deterministic coverage for every canonical G8 lifecycle transition and failure path.

- [ ] Step 1: Add the remaining controlled failure tests.

Cover loader raises, simulated compile raises inside the fake loader, generation raises, cleanup raises, replacement load failure after restart, restart during loading, restart during busy, shutdown twice, two start calls during loading, two restart calls, and old worker completion after replacement generation becomes current. Every test uses Event or Barrier to control ordering.

- [ ] Step 2: Run the complete G8 CPU suite and inspect failures.

~~~bash
python3 -m unittest -v tests.test_g8_worker_protocol tests.test_g8_manager_lifecycle tests.test_g8_async_jobs
~~~

Expected: zero failures and zero errors. If a race test fails, keep the failing test, diagnose ordering from recorded events, and fix only the minimum lifecycle code before rerunning.

- [ ] Step 3: Write the test matrix with evidence links.

For each invariant and scope row, record test name, verification class, expected state transition, observed result, and whether live model evidence is required. Mark G9 rows DEFERRED_TO_G9.

- [ ] Step 4: Run the repository-native suite before static verification.

~~~bash
python3 -m unittest discover -s tests -p 'test_*.py'
~~~

Expected: existing and G8 tests pass; record the exact test count and any pre-existing warning without altering frozen G5/G7 evidence.

---

### Task 6: Resolve cold-compile semantics and complete CPU/static verification

**Files:**
- Create: artifacts/g8/g8-cpu-verification.txt
- Create: artifacts/g8/g8-cold-compile-resolution.md
- Create: artifacts/g8/g8-cold-compile-resolution.json
- Create: artifacts/g8/g8-live-necessity-decision.md
- Create: artifacts/g8/g8-live-necessity-decision.json

**Interfaces:**
- Consumes: source/docs inventory, G8 tests, repository-native test output, engine load implementation, and frozen G7 authority reference.
- Produces: G8_CPU_VERIFICATION=PASS, G8_COLD_COMPILE_SEMANTICS_RESOLVED=true, and a cited G8_TPU_REQUIRED decision.

- [ ] Step 1: Run all required CPU/static checks.

~~~bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q src scripts clients/python
find scripts -type f -name '*.sh' -print0 | xargs -0 -r -n1 bash -n
python3 - <<'PY'
import json
from pathlib import Path
for path in Path('artifacts').rglob('*.json'):
    json.loads(path.read_text(encoding='utf-8'))
print('PROJECT_JSON_PARSE=PASS')
PY
PYTHONPATH=src python3 -c 'from gemma4_server.api.app import Runtime, create_app; from gemma4_server.core.config import Config; create_app(Runtime(Config.for_tests(), object()))'
PYTHONPATH=src python3 - <<'PY'
import importlib.metadata
from gemma4_server.tpu.authority_contract import project_dependency_gate
roots = ('keras', 'keras-hub', 'jax', 'flask', 'waitress')
packages = {}
for dist in importlib.metadata.distributions():
    packages[dist.metadata['Name']] = {'version': dist.version, 'requires': list(dist.requires or ())}
print(project_dependency_gate(packages, roots))
PY
~~~

Record each command, exit code, test count, and output in g8-cpu-verification.txt; use G8_CPU_VERIFICATION=PASS only when every command exits 0. Do not use global pip check as a blocker.

- [ ] Step 2: Resolve cold compile from current source.

Record that a new worker lifecycle is the cold boundary; Gemma4TPUEngine.load() performs native preset strict load, model.compile(sampler="greedy", run_eagerly=True), post-load cleanup, and only then emits ready; the worker optionally configures the persistent JAX compilation cache. Record that whole-model JIT is forbidden by frozen G3 safety evidence and that no canonical source requires first-inference timing or warm-cache characterization.

- [ ] Step 3: Decide live necessity without starting TPU.

Set G8_LIVE_MODEL_LIFECYCLE_REQUIRED=false, G8_LIVE_COLD_COMPILE_REQUIRED=false, G8_LIVE_RESTART_REQUIRED=false, and G8_TPU_REQUIRED=false only if every G8 row is honestly covered by source/static/CPU evidence and canonical docs do not explicitly require a real 31B model lifecycle. Otherwise record the exact mandatory live row, set CPU status open for live authority, and stop before TPU preparation.

- [ ] Step 4: Validate the decision JSON.

~~~bash
python3 -m json.tool artifacts/g8/g8-cold-compile-resolution.json >/dev/null
python3 -m json.tool artifacts/g8/g8-live-necessity-decision.json >/dev/null
rg -n 'G8_CPU_VERIFICATION=PASS|G8_COLD_COMPILE_SEMANTICS_RESOLVED|G8_TPU_REQUIRED|NEW_TPU_RUN_STARTED' artifacts/g8/g8-cpu-verification.txt artifacts/g8/g8-live-necessity-decision.md
~~~

Expected for CPU-only closure: required PASS markers are present and NEW_TPU_RUN_STARTED=false.

---

### Task 7: Build the final G8 acceptance matrix and CPU-only closeout

**Files:**
- Create: artifacts/g8/g8-acceptance-matrix.md
- Create: artifacts/g8/g8-acceptance.json
- Create: docs/G8-CLOSEOUT.md
- Create: /kaggle/working/gemma4-31b-g8-async-cold-compile-restart-lifecycle-20260914T000000Z.tar.gz
- Create: /kaggle/working/gemma4-31b-g8-async-cold-compile-restart-lifecycle-20260914T000000Z.tar.gz.sha256

**Interfaces:**
- Consumes: all artifacts/g8 evidence, frozen G7 archive reference, and fresh CPU/static output.
- Produces: machine-readable acceptance, closeout markers, compact archive, internal SHA256SUMS, and outer SHA-256 sidecar.

- [ ] Step 1: Write acceptance rows.

Use columns ID, Requirement, Canonical source, Gate, Verification method, CPU/static complete, Live model required, Evidence, Result, and Blocking. Every non-G9 row cites a concrete test or source artifact. Every G9 row has Result=DEFERRED_TO_G9 and Blocking=false.

- [ ] Step 2: Write the CPU-only final adjudication.

Use these exact markers only after fresh verification:

~~~text
G8_SCOPE_RESOLUTION=PASS
G8_CPU_VERIFICATION=PASS
G8_COLD_COMPILE_SEMANTICS_RESOLVED=true
G8_LIVE_MODEL_LIFECYCLE_REQUIRED=false
G8_LIVE_COLD_COMPILE_REQUIRED=false
G8_LIVE_RESTART_REQUIRED=false
G8_TPU_REQUIRED=false
NEW_TPU_RUN_STARTED=false
G8_ASYNC_ACCEPTANCE=PASS
G8_RESTART_LIFECYCLE_ACCEPTANCE=PASS
G8_COLD_COMPILE_ACCEPTANCE=PASS
G8_STATUS=CLOSED
G9_ENTRY_ELIGIBLE=true
G9_STARTED=false
TAG=false
RELEASE=false
FINAL_RESULT=G8_ASYNC_COLD_COMPILE_RESTART_LIFECYCLE_PASS
~~~

- [ ] Step 3: Package only G8 evidence and references.

Create a staging directory with artifacts/g8/, docs/G8-CLOSEOUT.md, relevant new G8 tests, modified lifecycle source files, and G7-AUTHORITY-REFERENCE.txt containing the G7 archive path, expected SHA-256, and G7_STATUS=CLOSED. Generate internal SHA256SUMS from the staged payload, create the compact tarball, and create its outer sidecar with sha256sum. Do not include or rewrite prior G3-G7 archives.

- [ ] Step 4: Verify package integrity and final status.

~~~bash
PACKAGE_ROOT=$(mktemp -d /kaggle/working/g8-package-verify.XXXXXX)
tar -xzf /kaggle/working/gemma4-31b-g8-async-cold-compile-restart-lifecycle-20260914T000000Z.tar.gz -C "$PACKAGE_ROOT"
CHECK_ROOT=$(find "$PACKAGE_ROOT" -type f -name SHA256SUMS -print -quit | xargs -r dirname)
test -n "$CHECK_ROOT"
(cd "$CHECK_ROOT" && sha256sum -c SHA256SUMS)
sha256sum /kaggle/working/gemma4-31b-g8-async-cold-compile-restart-lifecycle-20260914T000000Z.tar.gz
test "$(sha256sum /kaggle/working/gemma4-31b-g8-async-cold-compile-restart-lifecycle-20260914T000000Z.tar.gz | cut -d' ' -f1)" = "$(cut -d' ' -f1 /kaggle/working/gemma4-31b-g8-async-cold-compile-restart-lifecycle-20260914T000000Z.tar.gz.sha256)"
python3 -m json.tool artifacts/g8/g8-acceptance.json >/dev/null
~~~

Then run the complete CPU/static verification command set from Task 6 again. Expected: internal sums pass, outer sidecar matches, all mandatory acceptance rows pass, G9_STARTED=false, TAG=false, and RELEASE=false.

---

## Execution notes

- The plan intentionally stops before any live TPU authority. If Task 6 finds an explicit canonical live-model requirement, create the directive-required g8-authority-plan.md and run only the smallest one-process/one-worker/one-active-instance sequence after an explicit user decision; do not convert a missing live run into PASS.
- If CPU verification fails, leave G8_STATUS=OPEN, set G9_ENTRY_ELIGIBLE=false, preserve the failing evidence, and do not package a closeout archive.
- Before claiming completion, invoke superpowers:verification-before-completion and use fresh command output for every PASS claim.
