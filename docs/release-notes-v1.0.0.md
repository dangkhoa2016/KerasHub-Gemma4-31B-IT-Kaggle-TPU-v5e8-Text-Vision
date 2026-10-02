# v1.0.0 Release Notes

The first public release focuses on a reproducible Gemma 4 31B Instruct
inference stack for Kaggle TPU v5e-8.

## Included

- KerasHub Gemma 4 31B strict preset loading
- Keras ModelParallel sharding across eight TPU devices
- BF16 runtime
- text and image-conditioned text generation
- stable JAX decode-loop reuse for compatible hot requests
- synchronous and asynchronous REST APIs
- API-key authentication, request IDs and protected worker restart
- Python and Node.js clients
- Kaggle operational scripts and production notebook
- bilingual English / Vietnamese documentation
- compact qualification evidence

## Qualified reference

The qualified production-source run used one model load. The first compatible
request paid the compile cost; the identical hot request reused the executable
without a new compile event. See [qualification.md](qualification.md).

## Scope

This release targets Kaggle TPU v5e-8 / v5litepod-8. Other accelerators and
future dependency versions are not covered by the same qualification.
