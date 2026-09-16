# G4 Troubleshooting

## Primary attempt issue

Observed: watchdog cancel/join before thread start, with
`CandidateAVerificationError` local binding unsafe.

Fix: start the watchdog before `import jax`; join only when the thread is
alive; bind `CandidateAVerificationError` before the `try` block.

## Evidence plumbing issue

Observed: the real result was written to `07-g4-result.json` while the runner
expected canonical `09-g4-result.json`.

Fix: copy/use the real authority result as canonical result, regenerate the
comparison and `SHA256SUMS`, then repack the archive. This evidence-plumbing
correction did not rerun the model and did not consume another TPU attempt.

## Non-blocking warnings

- Transparent hugepages warning.
- Splash Attention was not applicable to `q_seq_len=11` / `block=128`.
- Runtime fell back to dot-product attention.

These warnings did not invalidate G4.
