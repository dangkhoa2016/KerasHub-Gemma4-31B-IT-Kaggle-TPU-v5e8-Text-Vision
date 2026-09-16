# Changelog

## Unreleased

### Final CPU preparation before TPU acceptance

- Recorded G0-G8 closed state and the final G8 live-generation authority.
- Preserved the post-fix collector fallback and its nested-metrics regression
  test in the canonical source tree.
- Added deterministic G9 PRIME/HOT and G10 fresh-session Run All runbooks.
- Added a static final Kaggle notebook and v1.0.0 release preflight.
- No G9/G10 runtime execution, tag, release, or public publication was made.

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
- Provisional native text/image generation for G3/G5 characterization.
- Long-lived spawned TPU worker, queue, async jobs, auth, restart and shutdown.
- Python and Node clients.
- Source-ZIP-first Kaggle notebook.
- No public release yet.
