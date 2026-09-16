# G8 Formal Final Closeout

## Adjudication

G8 is now **CLOSED/PASS**. Every mandatory G8 acceptance criterion is
satisfied by the frozen authority and existing post-fix evidence sidecar.

## Final authority

- Final successful job: `job-f555faed03dc450f88a9a336`
- Job status: `completed`
- Request: production async path, `prompt="Hello"`, `system=""`,
  `max_new_tokens=1`
- Prompt tokens: `10`
- Required length: `11`
- Generation max length: `16`
- Bucket 16 live result: `PASS`
- Inference seconds: `413.141671`
- Generated output: non-empty
- TPU topology: 8 devices (`TPU v5e-8`)
- OOM kill delta: `0`
- OOM group-kill delta: `0`

## Post-fix and provenance

- Post-fix full suite: `133/133 PASS`
- Compileall: `PASS`
- Collector corrective regression: `PASS`
- Source provenance: `PASS`
- Required current source hashes match the established SHA256 values.
- Git persistence: `NOT_PERFORMED_IN_THIS_CHECKOUT`; no commit SHA is claimed.

The corrective preserves the narrow worker collector contract: top-level
`inference_seconds` takes precedence, with fallback to
`metrics.generation_seconds`, while passing `metrics` through unchanged.

Historical failures remain immutable and represented. In particular, the
earlier timeout, preflight, and runtime-probe failures are not rewritten by
this closeout.

No additional generation was performed during formal closeout.

## G9 boundary and release boundary

`G9_ENTRY_ELIGIBLE=true` means G8 no longer blocks entry; it does not authorize
G9 execution. `G9_STARTED=false`, `TAG=false`, and `RELEASE=false`.

The next action is to await separate G9 authorization and canonical Git
persistence.
