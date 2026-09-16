# G2 Corrective R1 — Native KerasHub Preset Loader

## R0 evidence

The first authority run closed G0/G1 and reached Gemma 4 construction on eight TPU devices, but strict loading failed with:

```text
A total of 1187 objects could not be loaded.
```

The attached `model.weights.json` also contained 1187 weight-map entries. Its paths were backbone-oriented (for example `/layers/gemma4_vision_encoder/...`), while R0 loaded the index directly on the whole `Gemma4CausalLM` task object.

KerasHub 0.31.1's native preset loader reconstructs the task (using `task.json` when present) and loads `model.weights.json` via `task.backbone.load_weights(...)`.

## R1 change

R0:

```python
model = Gemma4CausalLM.from_preset(path, load_weights=False)
model.load_weights(model_weights_index, skip_mismatch=False)
```

R1:

```python
with distribution.scope():
    model = Gemma4CausalLM.from_preset(
        path,
        load_weights=True,
        dtype="bfloat16",
    )
```

No `skip_mismatch=True` is introduced. The native KerasHub loader remains strict.

## Frozen variables

R1 intentionally keeps unchanged:

- Keras 3.15.1
- KerasHub 0.31.1
- Kaggle JAX/JAXLIB
- libtpu policy
- BF16
- 8 TPU devices
- mesh `[1,8]`
- Candidate-A sharding rules
- 300 GiB memory guard
- 80% byte-weighted sharding gate

This isolation lets the next authority run determine whether loader targeting alone resolves the R0 failure.
