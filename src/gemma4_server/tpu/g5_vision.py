"""CPU-first helpers for the minimal Gemma4 image-generation proof."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .generation import scalar_text, vision_prompt


G5_PROMPT = "Describe the image."
G5_SEQUENCE_LENGTH = 512
G5_IMAGE_SIZE = (64, 64)


def create_synthetic_fixture(path: str | Path) -> str:
    """Write the small deterministic RGB fixture and return its SHA-256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = np.arange(G5_IMAGE_SIZE[0], dtype=np.uint8)[:, None]
    cols = np.arange(G5_IMAGE_SIZE[1], dtype=np.uint8)[None, :]
    pixels = np.empty((*G5_IMAGE_SIZE, 3), dtype=np.uint8)
    pixels[..., 0] = rows
    pixels[..., 1] = cols
    pixels[..., 2] = rows ^ cols
    Image.fromarray(pixels, mode="RGB").save(target, format="PNG")
    return sha256_file(target)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixture_rgb(path: str | Path) -> np.ndarray:
    """Load one fixture as a rank-3 uint8 RGB array."""
    with Image.open(path) as source:
        image = source.convert("RGB").copy()
    result = np.asarray(image)
    if result.shape != (*G5_IMAGE_SIZE, 3) or result.dtype != np.uint8:
        raise ValueError(f"fixture must be {G5_IMAGE_SIZE} RGB uint8")
    return result


def prepare_image_inputs(
    preprocessor,
    image: np.ndarray,
    prompt: str = G5_PROMPT,
    sequence_length: int = G5_SEQUENCE_LENGTH,
):
    """Run the observed Gemma4 image preprocessor input contract."""
    rendered = vision_prompt(prompt)
    inputs = {"prompts": rendered, "images": np.asarray(image)}
    processed = preprocessor.generate_preprocess(
        inputs,
        sequence_length=int(sequence_length),
    )
    if not isinstance(processed, dict):
        raise RuntimeError("Gemma4 image preprocessor did not return a mapping")
    return processed, rendered


def _shape(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(dimension) for dimension in shape)


def _count_true(value) -> int:
    if value is None:
        return 0
    return int(np.asarray(value).astype(bool).sum())


def inspect_vision_preprocess(processed: dict) -> dict[str, object]:
    """Summarize only fields emitted by the installed preprocessor."""
    expected = (
        "pixel_values",
        "pixel_position_ids",
        "token_ids",
        "vision_indices",
        "vision_mask",
        "padding_mask",
    )
    present = [key for key in expected if processed.get(key) is not None]
    pixel_shape = _shape(processed.get("pixel_values"))
    position_shape = _shape(processed.get("pixel_position_ids"))
    token_shape = _shape(processed.get("token_ids"))
    vision_mask_true_count = _count_true(processed.get("vision_mask"))
    indices_shape = _shape(processed.get("vision_indices"))
    has_image_tensor = len(pixel_shape) >= 2 and pixel_shape[-2] > 0
    has_vision_positions = vision_mask_true_count > 0 and (
        len(indices_shape) >= 1
        and indices_shape[-1] > 0
    )
    return {
        "image_preprocess_returned": all(key in present for key in expected),
        "vision_conditioning_present": (
            has_image_tensor and has_vision_positions
        ),
        "preprocessed_keys": present,
        "pixel_values_shape": list(pixel_shape),
        "pixel_position_ids_shape": list(position_shape),
        "token_ids_shape": list(token_shape),
        "vision_mask_true_count": vision_mask_true_count,
        "vision_indices_width": (
            indices_shape[-1] if len(indices_shape) >= 1 else 0
        ),
    }


def validate_non_empty_generation(value) -> str:
    result = scalar_text(value).strip()
    if not result:
        raise RuntimeError("G5 image generation returned empty output")
    return result
