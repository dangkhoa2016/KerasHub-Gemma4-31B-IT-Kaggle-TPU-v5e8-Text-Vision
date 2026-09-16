# G5 Troubleshooting

The installed preprocessor produced:

```text
pixel_values shape [1, 2520, 768]
vision_indices rank-1
```

The first helper assumed higher ranks. The correction accepted the installed unbatched/trailing-dimension shapes, used trailing dimensions rather than hard-coded rank assumptions, and added regression tests before the corrective TPU attempt.

```text
corrective attempt 2/2 PASS
no third attempt used
no new host OOM kill
```

Pallas/Splash Attention was unavailable with the installed libtpu, so the run used the JAX native `dot_product_attention` fallback. G5 still PASSed.
