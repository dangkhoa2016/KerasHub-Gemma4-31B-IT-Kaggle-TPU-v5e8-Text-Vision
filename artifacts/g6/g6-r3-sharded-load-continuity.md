# G6 R3 Sharded Load Continuity

The current source matches the frozen G3/G4 R3 source hashes and the frozen G5 source hashes. Static source inspection confirms the R3 contract:

```text
make_array_from_callback()
shard-local host slices
Keras Variable.assign(already-sharded JAX value)
R3_SHARDED_LOAD_CONTINUITY=PASS
```

The forbidden full conversion/device placement patterns are absent from the R3 helper and its source contract tests. The existing `skip_mismatch=False` setting is strict-load configuration and is not used to bypass loading.

This continuity record does not alter any frozen manifest.
