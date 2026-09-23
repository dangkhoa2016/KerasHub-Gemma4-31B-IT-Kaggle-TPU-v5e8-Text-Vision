# Changelog

All notable public-facing changes are documented here.

## Unreleased

### Documentation and repository presentation

- Added bilingual English / Vietnamese landing pages and documentation.
- Added issue templates, pull-request guidance, Dependabot configuration and
  a compact public qualification record.
- Simplified the public source tree by removing development-stage gate
  artifacts and internal runbooks.
- Renamed notebooks, requirements and tests around user-visible behavior
  instead of internal qualification labels.

### Runtime

- Added stable JAX decode-loop callable reuse for compatible hot requests while
  keeping native KerasHub prefill/cache semantics.
- Preserved strict checkpoint loading, ModelParallel sharding, text generation,
  image-conditioned generation, authenticated REST APIs and async jobs.
- Published qualified runtime metadata in the API and documentation.

## v1.0.0 — planned

The first public release will package the qualified Gemma 4 31B Kaggle TPU
runtime after final release review. No public v1.0.0 tag is created by this
changelog entry alone.
