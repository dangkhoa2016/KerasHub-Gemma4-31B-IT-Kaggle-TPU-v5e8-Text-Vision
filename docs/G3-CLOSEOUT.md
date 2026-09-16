# G3 Closeout

```text
G3=CLOSED/PASS
G3_TEXT_GENERATION=PASS
G4_ENTRY_ELIGIBLE=true
G4_STARTED=false
```

The authoritative PASS path was Kaggle TPU v5e-8 with 8 TPU devices,
Candidate-A sharding verified, one exact authority generation call, prompt
token count 10, and authority `max_length=11`. Generation returned
successfully from the KerasHub native path.

The final authority archive is referenced at
`/kaggle/working/gemma4-31b-vnext-g3-authority-20260913T225039Z.tar.gz` with
SHA-256
`475c940baf60302fd0f03e5f4b1a156696598d51162310ff168283551d6bb38e`.

PASS-producing compatibility facts:

- TPU device admission supports Kaggle VFIO exposure (`/dev/vfio/0..7`).
- Project-scoped dependency handling supports wildcard constraints such as
  `==1.*`.
- KerasHub 0.29.1 authority generation uses native `max_length=11`; the
  `max_new_tokens=1` value remains authority metadata/contract because this
  KerasHub API does not accept that keyword.
- Generation compiles with `run_eagerly=True` to avoid the observed
  host-memory OOM of the JIT generation path.

G3 is immutable from this point. Any later change to a G3-authority source is
a post-G3 change and does not alter this frozen result. `TAG=false` and
`RELEASE=false`.
