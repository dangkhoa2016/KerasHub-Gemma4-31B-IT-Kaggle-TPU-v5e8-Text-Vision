# v1.0.0 Release Notes — Preparation Copy

This document is prepared for publication after G9 and G10 pass. It is not a
published release.

## Included

- Gemma 4 31B Instruct text and vision server for Kaggle TPU v5e-8.
- Candidate-A Keras ModelParallel mesh `[1,8]` with axes `[batch, model]`.
- Native KerasHub preset loading with strict checkpoint semantics.
- Authenticated synchronous/asynchronous REST API with lifecycle and restart
  handling.
- Python and Node clients.
- Frozen G0-G8 source and compact evidence packages under `artifacts/`.

## Validation boundary

G8 is closed with final live text generation authority and post-fix CPU/static
qualification. G9 PRIME/HOT and G10 fresh Kaggle Restart Session → Run All are
the remaining operational validation phase. Their results must be added before
this document becomes release material.

## Known limitations

- Audio is outside this 31B target.
- PRIME/HOT cache and performance claims are not made until G9 directly
  records them.
- Public v1.0.0 publication is intentionally disabled until G10 passes.

## Assets and integrity

The release must include only tracked source, documentation, clients, compact
evidence, and checksums. Never publish model weights, checkpoint payloads,
Kaggle caches, JAX caches, TPU lockfiles, PID/state files, logs containing
private data, credentials, API keys, or temporary Codex transcripts.
