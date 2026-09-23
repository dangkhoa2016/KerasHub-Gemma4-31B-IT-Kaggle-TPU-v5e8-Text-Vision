# Qualification

The public production qualification used the following runtime:

```text
Keras            3.15.1
KerasHub         0.31.1
JAX / JAXLIB     0.10.2 / 0.10.2
libtpu           0.0.17
TPU              Kaggle v5e-8
TPU devices      8
Mesh             [1,8]
Dtype            bfloat16
```

## Production generation result

```text
model loads             1
same model object       true
same output             true
warm request            541.409 s
hot identical request     5.239 s
warm-to-hot speedup     103.34x
hot compile attempts      0
hot compile events        0
new OOM events            0
```

The result demonstrates executable reuse for the tested compatible hot request.
It does not promise identical latency for different prompt lengths, buckets,
vision requests or future dependency versions.

A compact machine-readable record is stored in `evidence/qualification.json`.
