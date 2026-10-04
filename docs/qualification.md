# Qualification

The dedicated TPU qualification for this release uses source ref `v1.0.0` and the following runtime:

```text
Keras            3.15.1
KerasHub         0.32.0
JAX / JAXLIB     0.10.2 / 0.10.2
libtpu           0.0.49
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
warm request            38.652 s
hot identical request   5.768 s
warm-to-hot speedup     6.70x
hot compile attempts    0
hot compile events      0
new OOM events           0
```

The result demonstrates executable reuse for the tested compatible hot request.
It does not promise identical latency for different prompt lengths, buckets,
vision requests or future dependency versions.

A compact machine-readable record is stored in `evidence/qualification.json`.
