from __future__ import annotations
import math
from typing import Any

LAYOUT_PROFILE = "gemma4_31b_dense_candidate_a_v1"

def build_layout_map(
    keras_module: Any,
    mesh: Any,
    data_axis="batch",
    model_axis="model",
):
    layout = keras_module.distribution.LayoutMap(mesh)

    layout[r".*token_embedding/embeddings$"] = (model_axis, data_axis)

    layout[r".*decoder_block.*attention/query/kernel$"] = (
        model_axis, data_axis, None
    )
    layout[r".*decoder_block.*attention/(key|value)/kernel$"] = (
        None, data_axis, None
    )
    layout[r".*decoder_block.*attention/attention_output/kernel$"] = (
        model_axis, None, data_axis
    )

    layout[r".*decoder_block.*ffw_gating/kernel$"] = (data_axis, model_axis)
    layout[r".*decoder_block.*ffw_gating_2/kernel$"] = (data_axis, model_axis)
    layout[r".*decoder_block.*ffw_linear/kernel$"] = (model_axis, data_axis)

    # Candidate A deliberately keeps the vision encoder replicated.
    return layout

def build_distribution(
    keras_module: Any,
    jax_module: Any,
    *,
    shape=(1,8),
    axis_names=("batch","model"),
    data_axis="batch",
    model_axis="model",
):
    devices = list(jax_module.devices("tpu"))
    if len(devices) != math.prod(shape):
        raise RuntimeError(
            f"DeviceMesh requires {math.prod(shape)} TPU devices, "
            f"found {len(devices)}"
        )
    mesh = keras_module.distribution.DeviceMesh(
        shape=shape, axis_names=axis_names, devices=devices
    )
    layout_map = build_layout_map(
        keras_module, mesh, data_axis=data_axis, model_axis=model_axis
    )
    distribution = keras_module.distribution.ModelParallel(
        layout_map=layout_map, batch_dim_name=data_axis
    )
    return mesh, layout_map, distribution
