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
configuration. The worker remains long-lived so load/compile costs can be
amortized.

Candidate A shards dominant dense text weights and keeps the vision encoder
replicated until strict-load feasibility is proven.

Initial G3/G5 generation is explicitly marked
`keras_hub_native_unvalidated`. G4 may replace it with a Gemma4-native split
prefill/decode implementation after real TPU characterization.
