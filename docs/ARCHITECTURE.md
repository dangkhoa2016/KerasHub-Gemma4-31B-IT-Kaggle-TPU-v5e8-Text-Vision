# Architecture

```text
HTTP
 |
 v
Flask + Waitress CPU coordinator
 |
bounded multiprocessing queue + JobStore
 |
ONE spawned TPU worker
 |
Gemma4TPUEngine
 |
Keras ModelParallel mesh [1,8]
 |
TPU0 ... TPU7
```

The model is one logical model, not eight replicas.

JAX/Keras/KerasHub imports happen inside the spawned worker after TPU
configuration. The worker remains long-lived so model-load and compile costs
can be amortized across requests.

Candidate A shards dominant dense text weights and keeps the vision encoder
replicated under the qualified ModelParallel layout.

## Generation path

The engine keeps KerasHub Gemma4 cache/prefill semantics and compiles the model
with:

```text
sampler=StableGemma4GreedySampler
run_eagerly=True
```

The stable sampler preserves a fixed Python callable identity for the JAX
`lax.while_loop` cond/body functions. Prompt tokens, cache, decode index,
padding mask, stop-token IDs, and model variable values remain dynamic loop
state rather than request-specific closure state.

This is important because the pre-corrective path created fresh nested loop
callables on each request, which caused JAX tracing/compile cache misses and
repeated executable metadata recovery.

The outer `run_eagerly=False` alternative was investigated but was not adopted:
on the qualified 31B runtime it drove host memory toward the cgroup limit while
materializing PJRT executable sharding/layout metadata.

## Qualified hot reuse

Production-source TPU qualification used one model load and two identical text
requests:

```text
warm request              541.409 s
hot identical request       5.239 s
hot compile attempts         0
hot compile events           0
same output                 true
```

The numbers above are evidence for the tested qualification request, not a
universal latency guarantee for all request shapes.

## Vision semantics

Image conditioning remains in the KerasHub Gemma4 prefill/cache path. The
stable sampler performs token-by-token decode against the already-prefilled
cache and does not replace the native vision encoder or image-conditioning
logic.
