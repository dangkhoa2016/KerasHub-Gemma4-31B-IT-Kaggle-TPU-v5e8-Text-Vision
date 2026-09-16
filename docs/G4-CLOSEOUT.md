# G4 Closeout

```text
G4=CLOSED/PASS
G4_RECOMMENDED_PATH=SPLIT
G5_ENTRY_ELIGIBLE=true
G5_STARTED=false
```

G4 is frozen from the authoritative native-vs-split characterization archive:

```text
archive=/kaggle/working/gemma4-31b-g4-characterization-20260914T003028Z.tar.gz
sha256=5ae0609df40393aaaf719c9615715610c176cbbd1f6584e987f3b6aa6a3bc662
```

The frozen native G3 baseline was `567.709641 s`; the split one-token
characterization was `26.965903 s`, an approximate `21.05x` ratio. This is a
one-token characterization only, not a production throughput benchmark.

G4 used one model load, eight TPU devices, and verified Candidate-A sharding.
The G3 freeze remains immutable. No G4 model execution is authorized during
freeze verification, and later G5 source edits must be treated as
`POST_G4_SOURCE_CHANGE` without changing `artifacts/g4/g4-pass-freeze.json`.

## Limitations

- Text-only one-token characterization.
- The split path uses installed implementation cache primitives.
- `_build_cache()` and `call_with_cache()` are integration primitives, not a
  public API stability guarantee.
- `<|channel>` is non-empty execution proof, not a quality benchmark.
