# G8 canonical restart contract

## HTTP admission

`POST /restart` requires normal API authentication (`Authorization: Bearer` or `X-API-Key`) and an `X-Restart-Secret` matching `config.restart_secret` (`src/gemma4_server/api/app.py:auth_required`, `restart`). Valid input is parsed by `parse_restart_options`: `wait_for_jobs` is boolean and `timeout` is numeric, positive, and capped at 900 seconds.

The handler acquires `Runtime.restart_lock` non-blocking. If held, it returns `409 {"error":"Restart already in progress"}`. If admitted, it creates exactly one non-daemon `http-restart` background thread, returns `202 {"status":"restarting","pid":...}`, and releases the HTTP lock in the thread's `finally` block.

## Lifecycle behavior

The canonical G8 behavior required by `docs/superpowers/specs/2026-09-14-g8-async-cold-compile-restart-lifecycle-design.md` is:

- Set acceptance false and restart-pending true before stopping the old worker; readiness is false throughout loading/restart.
- With `wait_for_jobs=true`, wait for queued/processing jobs up to timeout. If the wait expires, fail those pending jobs with public error `TPU worker restarted`.
- With `wait_for_jobs=false`, fail queued/processing jobs immediately with public error `TPU worker restarted`.
- Preserve `completed` results; they remain pollable until ordinary JobStore TTL/capacity cleanup.
- Stop/join the old worker, recreate task/result queues and shutdown event, and start one replacement generation. A successful current-generation `worker_ready` is the only readiness transition back to true.
- Ignore every stale event whose generation differs from the status generation (`GenerationManager._handle`).

Current source evidence: `GenerationManager.restart_worker` already disables acceptance, has the wait/fail decision, stops the worker, replaces queues/events, and starts a worker; `create_app.restart` already provides HTTP serialization. Manager-wide lifecycle serialization and the readiness hardening are G8 acceptance work, not asserted as complete by this Task 1 inventory.
