# G7 Server Inventory

## Entrypoint and topology

`src/server.py` constructs one `GenerationManager`, one `Runtime`, one Flask
application, calls `manager.start_async()`, and invokes Waitress `serve()` in
the coordinator process. It has no gunicorn, multiprocessing pool, auto-reload,
or multi-worker server option. `GenerationManager.WORKER_ID` is the single
`tpu-0` model worker identity. The model is constructed and loaded inside that
spawned worker, never per request.

The source-level initial topology is therefore:

```text
one OS coordinator process
  -> one Waitress server
  -> one spawned TPU worker (tpu-0)
  -> one Gemma4TPUEngine instance
  -> one engine.load() call
```

`MAX_WORKER_RESTARTS` permits a later sequential worker generation after an
unexpected exit. That is a lifecycle concern and is excluded from the CPU-only
G7 authority; any future live G7 run would set the restart budget to zero and
would still use one process, one worker, one model instance, and one load.

## HTTP surface

| Route | Auth | Handler/call path |
|---|---|---|
| `GET /` | Public | `index` returns route list and service metadata |
| `GET /health/live` | Public | `live` returns process liveness without manager/model access |
| `GET /health/ready` | Public | `ready` calls `manager.health()` and returns 200/503 |
| `GET /info` | API key | `manager.health()` -> `build_info_payload` |
| `POST /generate` | API key | JSON validation -> manager queue -> text engine |
| `POST /generate/async` | API key | JSON validation -> manager queue -> 202 job envelope |
| `POST /generate/image` | API key | JSON/multipart image validation -> manager queue -> vision engine |
| `POST /generate/image/async` | API key | JSON/multipart validation -> 202 job envelope |
| `GET /result/<job_id>` | API key | `JobStore.get()` -> public job JSON |
| `POST /restart` | API key + `X-Restart-Secret` | non-blocking restart thread -> manager restart |

The exact request path is:

```text
HTTP request
 -> Flask before_request/request-id
 -> auth and parser
 -> GenerationManager.submit
 -> bounded task queue
 -> spawned model_worker_main
 -> Gemma4TPUEngine.generate_text/generate_image
 -> result queue
 -> JobStore
 -> HTTP JSON response or result polling
```

Health and readiness do not construct a model. JAX/Keras/KerasHub imports occur
only in the spawned worker after TPU configuration. `src/server.py` installs
SIGTERM/SIGINT hooks and shuts down the manager in `finally`; complete graceful
restart/lifecycle proof is G8 scope.

## Contract and evidence boundary

G7 proves route registration and CPU HTTP mechanics with an injected fake
manager/job boundary. It does not claim live 31B text/image output, cold
compile, restart, or PRIME/HOT behavior. Those claims remain respectively
outside this CPU-only G7 closeout or in G8/G9 as resolved in
`g7-scope-resolution.md`.
