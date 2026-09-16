# G9 Corrective PRIME Bucket Adjudication

## Resolved request shape

The corrective G9 PRIME/HOT sequence retains the existing production request:

```json
{
  "prompt": "hello",
  "system": "",
  "max_new_tokens": 1
}
```

The production planner normally maps this shape to bucket `16`. The request,
bucket policy, and generation arguments are frozen and are not changed to
manufacture a pass.

## Existing evidence boundary

The G8 cold-compile resolution at
`/kaggle/working/temp/g8-final-check.gDzq7j/artifacts/g8/g8-cold-compile-resolution.md`
defines cold semantics as the new worker lifecycle through native load,
explicit `model.compile(..., run_eagerly=True)`, validation, cleanup, and the
current-generation `worker_ready` event. It explicitly does not claim a
first-inference compile measurement or warm-cache characterization; those
claims were deferred to G9.

The historical G9 evidence under `/kaggle/working/artifacts/g9` and the
initial failed live attempt are preserved authority, but they do not prove
that bucket `16` is cold in the new corrective G9 process/cache authority.

## Pre-live decision

Before a corrective G9 PRIME is authorized, capture a new timestamped process
and cache authority that directly proves bucket `16` is cold. If that proof is
not available, stop before PRIME and explicitly adjudicate another
deterministic production bucket whose cold state can be directly proven. No
automatic retry, request mutation, bucket-policy change, sampler change,
dtype change, mesh change, loader change, or G10 transition is permitted.

