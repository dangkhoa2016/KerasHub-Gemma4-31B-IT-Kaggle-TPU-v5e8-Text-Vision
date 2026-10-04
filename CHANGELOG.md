# Changelog

All notable public-facing changes are documented here.

## Unreleased

No public-facing changes have been recorded after v1.0.0 yet.

## v1.0.0 — 2026-09-26

### Documentation and repository presentation

- Added bilingual English / Vietnamese landing pages and documentation.
- Added issue templates, pull-request guidance, Dependabot configuration and
  a compact public qualification record.
- Simplified the public source tree by removing development-stage gate
  artifacts and internal runbooks.
- Renamed notebooks, requirements and tests around user-visible behavior
  instead of internal qualification labels.
- Added Vietnamese counterparts for the client README, changelog,
  contribution guide and security policy.
- Rewrote the production notebook around the public repository workflow and
  added static validation for notebook references, local links, bilingual
  document pairs and internal development tokens.

### Runtime

- Added stable JAX decode-loop callable reuse for compatible hot requests while
  keeping native KerasHub prefill/cache semantics.
- Preserved strict checkpoint loading, ModelParallel sharding, text generation,
  image-conditioned generation, authenticated REST APIs and async jobs.
- Published qualified runtime metadata in the API and documentation.
- Aligned the TPU dependency manifest with the qualified runtime.
- Qualified KerasHub 0.32.0 and libtpu 0.0.49, and raised the cold model-load timeout to 3600 seconds so the 60 GB checkpoint can finish loading on a cold Kaggle mount.

### Qualification

- Qualified the production source on Kaggle TPU v5e-8 with one model load.
- Recorded a 38.652-second first compatible request and a
  5.768-second identical hot request (6.70x warm-to-hot speedup) with no new compile event.
- Published compact bilingual qualification documentation and machine-readable
  evidence.
