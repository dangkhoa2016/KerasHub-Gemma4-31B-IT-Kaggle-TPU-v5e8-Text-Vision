# G8 async job state machine

| State | Entry | Exit | HTTP representation | Terminal | Restart behavior | Error behavior |
|---|---|---|---|---|---|---|
| `queued` | `GenerationManager.submit` creates a `Job` and `JobStore.put` succeeds before task queue placement. | `job_started` marks processing; restart failure marks failed. Queue-full rolls back by delete. | Async submit: 202 with job id/status/result URL. Poll: 202, no result. | No | `wait_for_jobs=true` waits; `false` fails immediately. | Submission may be 429 for full queue or 503 when not ready/store full. |
| `processing` | Current-generation `job_started` reaches `GenerationManager._handle`, then `JobStore.mark_processing`. | `job_completed` or `job_failed`; restart failure/expired wait marks failed. | Poll: 202, no result. | No | `wait_for_jobs=true` waits; `false` fails immediately. | Worker generation exception emits `job_failed`. |
| `completed` | Current-generation `job_completed` calls `JobStore.mark_completed`. | TTL/capacity cleanup only; required G8 invariant prevents later terminal overwrite. | Poll: 200 with output, runtime, metrics. | Yes | Preserved across restart and remains queryable until normal cleanup. | No HTTP error. |
| `failed` | Current-generation `job_failed`, restart pending failure, or required failure transition calls `JobStore.mark_failed`. | TTL/capacity cleanup only; required G8 invariant prevents late completion overwrite. | Poll: 500 with JSON `error`. | Yes | Pending job becomes failed with `TPU worker restarted`; previously failed remains terminal. | Public error is returned; internal error remains stored. |

Source anchors: `src/gemma4_server/jobs/models.py:Job`; `src/gemma4_server/jobs/store.py:mark_processing,mark_completed,mark_failed,fail_pending`; `src/gemma4_server/workers/manager.py:submit,_handle,restart_worker`; `src/gemma4_server/api/app.py:submit,result`.

The terminal immutability entries are canonical G8 acceptance requirements from the G8 lifecycle design. The Task 1 source inventory records that the pre-G8 `mark_completed`/`mark_failed` code currently updates any extant job; later G8 work must add the guards before acceptance can claim this transition rule.
