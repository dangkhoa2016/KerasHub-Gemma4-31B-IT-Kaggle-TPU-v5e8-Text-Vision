# G2 Corrective R1 — Native KerasHub Preset Loader

Authority run R0 đã PASS G0/G1 nhưng strict load fail với đúng 1.187 objects. `model.weights.json` cũng có 1.187 weight-map entries và mang hierarchy của backbone.

R0 đã gọi generic `model.load_weights(model.weights.json)` trên toàn bộ `Gemma4CausalLM` task. KerasHub 0.31.1 thực tế load preset bằng cách reconstruct task rồi nạp `model.weights.json` qua `task.backbone`.

R1 vì vậy đổi duy nhất loader strategy sang:

```python
with distribution.scope():
    model = Gemma4CausalLM.from_preset(
        path,
        load_weights=True,
        dtype="bfloat16",
    )
```

Không dùng `skip_mismatch=True`. Keras/KerasHub/JAX/libtpu, BF16, mesh `[1,8]`, Candidate-A sharding, memory guard và sharding gate đều giữ nguyên để lần authority kế tiếp chỉ kiểm chứng loader corrective.
