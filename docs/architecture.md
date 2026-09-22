# Architecture

```text
HTTP
 |
 v
Flask + Waitress coordinator
 |
bounded queue + in-memory JobStore
 |
one long-lived spawned TPU worker
 |
Gemma4TPUEngine
 |
Keras ModelParallel mesh [1,8]
 |
8 TPU devices
```

The deployment runs one logical Gemma 4 31B model sharded across all eight TPU
devices. It does not run eight independent replicas.

## Model loading

JAX, Keras and KerasHub are configured inside the spawned worker. The model is
loaded strictly through the KerasHub preset loader under the ModelParallel
scope. Host-side temporary allocations are garbage-collected and trimmed after
load.

## Generation

KerasHub handles native prefill/cache semantics. The production greedy sampler
keeps the JAX `lax.while_loop` cond/body callable identities stable across
requests while prompt, cache, padding mask, stop IDs and variable values remain
dynamic state. This allows hot requests with compatible shapes to reuse the
compiled executable.

## Vision

Image conditioning is produced during KerasHub preprocessing and prefill. The
decode loop consumes the already-conditioned cache; it does not replace the
native vision encoder.

## REST process model

The Flask coordinator never owns the TPU model. It validates requests, stores
job state, and communicates with the worker through bounded multiprocessing
queues. Worker restart is explicit and protected by a separate secret.
