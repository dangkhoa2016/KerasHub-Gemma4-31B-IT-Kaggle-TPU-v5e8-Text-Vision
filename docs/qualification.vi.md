# Qualification

Runtime public đã qualification:

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

## Kết quả production generation

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

Kết quả chứng minh executable reuse cho hot request tương thích đã test, không
phải cam kết latency cho mọi prompt, bucket, vision request hoặc dependency
version tương lai.

Machine-readable record nằm ở `evidence/qualification.json`.
