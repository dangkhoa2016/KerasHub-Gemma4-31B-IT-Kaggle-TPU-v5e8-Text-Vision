# G8 scope resolution

## Verified continuity baseline

- `G8_REPO_ROOT=/kaggle/working/KerasHub-Gemma4-31B-IT-Kaggle-TPU-v5e8-Text-Vision`
- `GIT_PRESENT=false`; `G8_HEAD=NOT_APPLICABLE`; `G8_WORKTREE_CLEAN=NOT_APPLICABLE`.
- Verified archive: `/kaggle/working/gemma4-31b-g7-rest-server-acceptance-20260914T043113Z.tar.gz`
- Outer SHA-256: `4f1547d6d86d796aca7ce7d396021d42829abb7f932d1713093345d69035c0a2`
- `DEVELOPMENT-STATUS.md` and `docs/ROADMAP.md` record `G7=CLOSED/PASS` and `G8_ENTRY_ELIGIBLE=true`.

## Resolution

Machine-readable requirement rows are in `g8-scope-resolution.json`. G8 covers CPU-verifiable async job, worker lifecycle, restart, shutdown, stale-event, and cold load/compile-boundary behavior. No TPU was run for this task. `G8-SCOPE-010` and `G8-SCOPE-011` are explicitly `Result=DEFERRED_TO_G9`: G9 owns PRIME/HOT, warm, throughput, and latency acceptance.

The cold-compile row is deliberately static: `Gemma4TPUEngine.load()` calls `model.compile(..., run_eagerly=True)` and the worker configures the persistent JAX cache, but this task makes no first-inference, warm-cache, latency, or throughput claim.
