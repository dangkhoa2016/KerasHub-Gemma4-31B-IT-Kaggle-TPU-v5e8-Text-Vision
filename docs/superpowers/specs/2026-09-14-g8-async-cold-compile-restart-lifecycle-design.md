# G8 Async / Cold Compile / Restart / Lifecycle Design

**Date:** 2026-09-14
**Project:** KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision
**Gate:** G8 only

## Goal

Close the G8 async-job, cold-start/compile, worker restart, and lifecycle
acceptance scope with deterministic CPU evidence first. G3-G7 remain frozen;
G9 PRIME/HOT behavior remains deferred.

## Authority and scope

The repository is the canonical source tree at
`/kaggle/working/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision`.
The G7 compact archive has matching outer SHA-256, matching internal
`SHA256SUMS`, and matching current hashes for every archived G7 source and
evidence file. The repository has no Git metadata, so no historical commit or
worktree state is available.

Canonical repository material defines the G8 boundary as async/cold compile
plus restart/lifecycle acceptance. The documented HTTP surface is:
`POST /generate/async`, `POST /generate/image/async`, `GET /result/<job_id>`,
`GET /health/live`, `GET /health/ready`, and authenticated `POST /restart`.
The architecture specifies one CPU coordinator, one spawned TPU worker, one
logical model, a bounded queue, and a `JobStore`. PRIME/HOT, warm steady-state
performance, and throughput characterization are G9 and are explicitly out
of scope.

## Current lifecycle model

The current implementation uses `GenerationManager` as the lifecycle owner.
It starts a single process with worker identity `tpu-0`, passes work through a
multiprocessing task queue, receives worker events through a result queue, and
publishes job state through `JobStore`. Worker events carry a generation token;
events for a non-current generation are ignored.

Observed worker states are `starting`, `loading`, `ready`, `busy`, `failed`,
and `stopped`. The manager exposes a derived `loading`, `ready`,
`restarting`, or `unavailable` health state. Jobs use the states `queued`,
`processing`, `completed`, and `failed`.

The worker imports JAX/Keras/KerasHub only inside the spawned process, loads
one `Gemma4TPUEngine`, compiles the model with
`run_eagerly=True`, emits readiness only after load and post-load cleanup,
then processes jobs until its shutdown event is set. The configured JAX
persistent compilation cache is retained. G8 must not introduce whole-model
JIT, change Candidate-A sharding, or alter model/checkpoint identity.

## Design

### Lifecycle serialization

Make restart lifecycle ownership explicit in `GenerationManager`. A restart
must serialize against another restart and against worker start, set the
manager unavailable before stopping the old worker, stop and join the old
worker, replace its queues/events, and only then create the replacement
worker. `start_async()` remains idempotent for the single worker identity.
The manager must never expose readiness while `_restart_pending` is true or
while the manager is not accepting jobs.

The generation token remains the stale-event barrier. A completion or state
event from an old worker must not mutate the replacement worker status or
overwrite a job already failed by restart. Shutdown remains idempotent and
must not trigger an automatic replacement worker.

### Canonical restart and job behavior

The existing HTTP contract remains asynchronous: an authenticated restart
request with `X-Restart-Secret` returns `202` and starts one background
lifecycle transition; a concurrent restart request returns `409`. The
`wait_for_jobs` option preserves completed results, waits for pending jobs when
true, and marks queued/processing jobs failed with the canonical restart error
when false or when the wait expires. Completed results remain queryable until
normal TTL cleanup.

During load or restart, readiness is false and new submissions are rejected by
the existing not-ready response. After successful replacement load, readiness
returns only on the new worker's ready event. A replacement load failure is
observable as failed/unavailable and does not create another model worker as a
corrective retry.

### Test seam

Add a CPU-only deterministic lifecycle harness around the manager/worker event
protocol. The fake loader/model must support successful load, controlled slow
load, load failure, generation success/failure, blocking generation, cleanup
failure, and shutdown. Tests use `threading.Event`, `threading.Barrier`, and
explicit hooks rather than sleep-only race timing. No Gemma model, JAX TPU
device, or real checkpoint is loaded by G8 CPU tests.

The tests cover idempotent start, start/restart races, concurrent restart
serialization, readiness transitions, one worker and one model owner per
lifecycle, stale events, async job transitions, restart behavior for queued,
running, and completed jobs, failure propagation, and idempotent shutdown.

## Cold-compile interpretation

The canonical source material distinguishes the long-lived worker's load and
compile cost from later warm behavior, and the engine currently calls
`model.compile(..., run_eagerly=True)` during `load()`, while configuring a
persistent JAX compilation cache. The project does not authorize reintroducing
the historically unsafe whole-model JIT path.

The implementation will record this interpretation and explicitly test the
state boundary around worker load/compile. It will not claim a real 31B first
inference compile, latency, or warm-cache result unless canonical material
proves that such live evidence is a mandatory G8 requirement. If canonical
scope remains insufficient to decide whether a live model lifecycle is
required, G8 stops as `G8_COLD_COMPILE_SCOPE_AMBIGUOUS` rather than guessing.

## Evidence and acceptance

Create the directive-required files under `artifacts/g8/`: scope resolution,
lifecycle inventory, invariants, test matrix, CPU verification, cold-compile
resolution, restart contract, async state machine, live-necessity decision,
acceptance matrix, and machine-readable acceptance. Create
`docs/G8-CLOSEOUT.md` and a compact archive only if every mandatory G8 row has
fresh evidence and passes. Reference G7 by archive name/path/SHA-256 without
rewriting it.

CPU acceptance requires the repository-native suite, G8 tests, `compileall`,
shell syntax checks, JSON parsing, server import, and project-scoped dependency
verification to pass. TPU remains off until this gate passes and the live
necessity decision cites an explicit canonical live requirement. If live TPU is
required, authorize one process, one model worker, at most one active model
instance, eight devices, and one minimal lifecycle sequence; automatic
corrective reruns are forbidden.

## Non-goals

- Reopening or rewriting G3-G7 authority.
- Changing model dtype, mesh, Candidate-A sharding, checkpoint loading, or
  generation architecture.
- PRIME/HOT acceptance, warm benchmarks, throughput characterization, or
  latency closure.
- Tagging or releasing.
