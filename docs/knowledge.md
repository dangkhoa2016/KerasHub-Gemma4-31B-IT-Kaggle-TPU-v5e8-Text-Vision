# Technical Notes

## Why one model across eight TPU devices?

The 31B model is too large to treat each TPU device as an independent replica
under the target memory envelope. Keras ModelParallel distributes dominant
parameters across the model axis while preserving one logical model.

## Why keep the worker alive?

Model loading and JAX compilation are expensive. A long-lived worker preserves
model state and allows compatible hot requests to reuse compiled executables.

## Why stable loop callables matter

JAX compilation caching is sensitive to Python callable identity. Recreating
nested `lax.while_loop` callables per request can force retracing and
recompilation. The production sampler keeps those callables stable and moves
request-specific values into loop state.

## Why not outer `run_eagerly=False`?

That approach was investigated on the qualified 31B runtime but caused very
large host-memory pressure while PJRT materialized executable
sharding/layout metadata. The stable inner loop achieved reuse without that
host-memory behavior.

## Why generation buckets?

Buckets bound the number of compiled shapes and make executable reuse more
predictable. Small requests use a 16-token bucket; larger requests map to the
configured bucket sequence up to the maximum generation length.
