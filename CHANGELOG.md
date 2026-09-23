# Changelog

## Unreleased

### Post-G10 generation-loop corrective and publication readiness

- Closed G9 PRIME/HOT and G10 fresh-session Run All before the post-G10
  corrective work.
- Localized the multi-minute warm-request latency to repeated JAX/PXLA
  compilation and executable metadata recovery around the generation
  `lax.while_loop`.
- Identified fresh nested while-loop callable identity as the cause of
  per-request tracing/compile cache misses.
- Added `StableGemma4GreedySampler` while retaining `run_eagerly=True`.
  Request data remains dynamic loop state; the stable loop callables do not
  capture the previous request's padding mask or token state.
- Kept the KerasHub Gemma4 cache/prefill path and native vision conditioning.
- Production-source TPU qualification used one model load and recorded:
  - warm request: `541.409262515 s`
  - hot identical request: `5.239080924 s`
  - warm-to-hot speedup: `103.340503873x`
  - hot compile attempts/events: `0 / 0`
  - same model object and same output: `true`
  - OOM counter increase: `0`
- Full local unit suite passed `192/192`.
- Public GitHub Actions CI passed for the corrective source.
- Lightweight CI skips the two JAX runtime subprocess regression tests when
  JAX is not installed; those tests run and pass on the qualified JAX runtime.
- No version bump, v1.0.0 tag, or GitHub release has been created yet.

### Final CPU preparation before TPU acceptance

- Recorded G0-G8 closed state and the final G8 live-generation authority.
- Preserved the post-fix collector fallback and its nested-metrics regression
  test in the canonical source tree.
- Added deterministic G9 PRIME/HOT and G10 fresh-session Run All runbooks.
- Added a static final Kaggle notebook and v1.0.0 release preflight.

### G3 PASS freeze and closeout

- Closed G3 with authoritative text-generation PASS on Kaggle TPU v5e-8.
- Verified 8 TPU devices, Candidate-A sharding, one exact generation call,
  and the immutable PASS-producing source hash set.
- G4 kickoff started repository-scoped native-vs-split characterization
  planning; no expensive TPU workload, tag, or release was created.

### Corrective R1 — native Gemma 4 preset loading

- Replaced the R0 whole-task generic `model.load_weights(model.weights.json)` path.
- G2 now uses `Gemma4CausalLM.from_preset(..., load_weights=True, dtype=bfloat16)` inside the ModelParallel scope.
- KerasHub therefore routes sharded `model.weights.json` through `task.backbone`, matching KerasHub preset semantics.
- Added explicit loader-strategy metadata and source-contract tests.
- Keras/KerasHub/JAX/libtpu, mesh `[1,8]`, BF16 and Candidate-A sharding remain unchanged.

- Independent Gemma 4 31B Instruct project identity.
- Kaggle TPU v5e-8 preflight and attached-model resolver.
- Candidate-A Keras ModelParallel `[1,8]` distribution.
- Strict-load, byte-weighted sharding and memory gates.
- Long-lived spawned TPU worker, queue, async jobs, auth, restart and shutdown.
- Python and Node clients.
- Source-ZIP-first Kaggle notebook.
- No public release yet.
