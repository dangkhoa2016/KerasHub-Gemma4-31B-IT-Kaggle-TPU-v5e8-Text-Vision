# G8 cold-compile resolution

```text
G8_COLD_COMPILE_SEMANTICS_RESOLVED=true
G8_COLD_COMPILE_LIVE_TPU_REQUIRED=false
```

## Resolved boundary

The canonical cold boundary is a new model-worker lifecycle: worker start,
model construction, native strict preset load, model compile, validation,
post-load host cleanup, and the current-generation `worker_ready` event. The
manager remains not ready while the worker is loading or restarting; readiness
is restored only by the current-generation ready event. This is a lifecycle
state boundary, not a permission to add a whole-model JIT path.

## Canonical evidence

- `src/gemma4_server/tpu/engine.py:207-232` imports the TPU/model runtime only
  inside `Gemma4TPUEngine.load()`, performs native
  `Gemma4CausalLM.from_preset(..., load_weights=True, dtype=...)`, then calls
  `model.compile(sampler="greedy", run_eagerly=True)`.
- `src/gemma4_server/tpu/engine.py:235-256,287-288` validates the loaded
  model/sharding contract and performs post-load host cleanup before
  `src/gemma4_server/tpu/engine.py:284` emits the engine `ready` phase.
- `src/gemma4_server/workers/worker.py:40-48,114-165` emits loading, invokes
  the loader once, emits worker-ready only after the loader returns, and
  optionally configures the JAX persistent compilation cache at lines 99-112.
- `src/gemma4_server/workers/manager.py:184-203` accepts only the current
  generation and turns readiness on at the ready event; the lifecycle tests
  prove the loading/restart not-ready boundary.
- `docs/ARCHITECTURE.md:22-24` describes a long-lived worker so load/compile
  costs can be amortized. `docs/API.md:11-41` defines async, polling, health,
  and restart semantics but specifies no first-inference timing or warm-cache
  acceptance.
- The frozen directive's safety rule (`directive:207-223`) and
  `docs/G3-CLOSEOUT.md:28-29` preserve `run_eagerly=True` because the prior
  native JIT generation path caused host-memory OOM. G8 therefore does not
  reintroduce whole-model `jax.jit`.

## What is and is not claimed

Compile is resolved as the explicit `model.compile()` operation executed in
`load()`, within the initial/new worker lifecycle. The canonical material does
not require or define a real first-inference compilation measurement, split
generation first-call compilation, native-vision first-call compilation,
latency, throughput, or warm-cache characterization. Those claims remain
outside this CPU-only G8 resolution; PRIME/HOT and warm behavior are deferred
to G9 by the approved G8 scope.

HTTP requests during loading/restart use the existing not-ready behavior. CPU
tests prove the state transition and rejection semantics with a deterministic
fake loader; no model, checkpoint, TPU device, or worker process was started.
