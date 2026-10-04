# Limitations

- The qualified target is Gemma 4 31B Instruct on Kaggle TPU v5e-8 with eight
  TPU devices. Other accelerators are not covered by the same qualification.
- Audio generation is not implemented.
- Hot latency depends on request shape, generation bucket, cache state and
  compilation-cache reuse. The measured 5.768-second identical hot request is not a
  universal SLA.
- The first model load and first compile can take many minutes.
- The in-memory job store is intentionally process-local and not a durable
  distributed queue.
- Quick Tunnel is suitable for demonstrations, not as a permanent ingress or
  authentication boundary.
- Model availability and licensing are governed by the upstream Gemma model.
- This project does not claim compatibility with every future Keras, KerasHub,
  JAX or libtpu release; use the qualified versions for reproducibility.
