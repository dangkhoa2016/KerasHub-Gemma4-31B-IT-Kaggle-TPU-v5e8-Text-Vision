from __future__ import annotations
import base64, binascii, io
from PIL import Image
from .errors import ValidationError

def _max_tokens(data, config):
    raw = data.get("max_new_tokens", config.default_output_tokens)
    if isinstance(raw, bool):
        raise ValidationError("max_new_tokens must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError("max_new_tokens must be an integer") from exc
    if not 1 <= value <= config.max_output_tokens:
        raise ValidationError(
            f"max_new_tokens must be between 1 and {config.max_output_tokens}"
        )
    return value

def _prompt(data, config):
    value = data.get("prompt")
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Field 'prompt' must be a non-empty string")
    if len(value) > config.max_input_chars:
        raise ValidationError("Prompt is too large")
    return value.strip()

def _system(data, config):
    value = data.get("system", "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValidationError("Field 'system' must be a string")
    if len(value) > config.max_input_chars:
        raise ValidationError("System prompt is too large")
    return value.strip()

def parse_generate_payload(data, config):
    return {
        "prompt": _prompt(data, config),
        "system": _system(data, config),
        "max_tokens": _max_tokens(data, config),
    }

def parse_image_binary(binary: bytes, data: dict, config):
    if not config.vision_enabled:
        raise ValidationError("Vision generation is disabled")
    if not binary or len(binary) > config.max_image_bytes:
        raise ValidationError("Image is empty or too large")
    try:
        with Image.open(io.BytesIO(binary)) as src:
            if src.width * src.height > config.max_image_pixels:
                raise ValidationError("Image has too many pixels")
            image = src.convert("RGB").copy()
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("Unsupported image") from exc
    payload = parse_generate_payload(data, config)
    payload["image"] = image
    return payload

def parse_image_payload(data, config):
    raw = data.get("image_base64", data.get("image"))
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError("Field 'image_base64' must be a base64 string")
    value = raw.strip()
    if value.startswith("data:"):
        if "," not in value:
            raise ValidationError("Invalid data URL")
        value = value.split(",", 1)[1]
    try:
        binary = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("Invalid base64 image") from exc
    return parse_image_binary(binary, data, config)
