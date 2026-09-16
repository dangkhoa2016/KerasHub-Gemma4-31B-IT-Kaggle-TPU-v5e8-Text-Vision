# G7 Live-Model Necessity Decision

```text
G7_SCOPE_RESOLUTION=PASS
G7_CPU_VERIFICATION=PASS
G7_LIVE_MODEL_BACKED_REST_REQUIRED=false
G7_LIVE_TEXT_REST_REQUIRED=false
G7_LIVE_IMAGE_REST_REQUIRED=false
G7_STREAMING_REQUIRED=false
G7_AUTH_REQUIRED=true
G7_GRACEFUL_SHUTDOWN_REQUIRED=false
G7_TPU_REQUIRED=false
NEW_TPU_RUN_STARTED=false
```

The canonical roadmap defines G7 as REST server acceptance and separately
defines G8 as async/cold compile plus restart/lifecycle acceptance and G9 as
PRIME/HOT acceptance. The README and API documents specify the HTTP routes,
authentication, request formats, and health surface, but do not state that G7
requires a fresh live model-backed request. The architecture places Gemma/JAX
imports and model loading inside the spawned TPU worker, so the documented HTTP
surface is legitimately verified with static checks and a CPU fake/stub manager.

Accordingly, no TPU run is required for this G7 contract. This is not a claim
that the fake output is Gemma evidence; prior frozen G3/G5/G6 authority remains
the model/runtime authority.
