from __future__ import annotations
from dataclasses import dataclass

TURN_END = "<turn|>"

def chat_prompt(prompt: str, system: str = "") -> str:
    parts = []
    if system:
        parts.append(f"<|turn>system\n{system}{TURN_END}\n")
    parts.append(f"<|turn>user\n{prompt}{TURN_END}\n")
    parts.append("<|turn>model\n")
    return "".join(parts)

def vision_prompt(prompt: str, system: str = "") -> str:
    parts = []
    if system:
        parts.append(f"<|turn>system\n{system}{TURN_END}\n")
    parts.append(f"<|turn>user\n<|image|>\n{prompt}{TURN_END}\n")
    parts.append("<|turn>model\n")
    return "".join(parts)

@dataclass(frozen=True)
class GenerationPlan:
    max_length: int
    max_new_tokens: int
    bucketed: bool
    mode: str = "CURRENT_PRODUCTION_BUCKET_POLICY"

def plan_generation(
    prompt_tokens: int,
    max_new_tokens: int,
    buckets,
    max_length: int,
):
    required = prompt_tokens + max_new_tokens
    if required > max_length:
        raise ValueError(
            f"Prompt + completion requires {required} tokens, "
            f"above MAX_GENERATION_LENGTH={max_length}"
        )
    for bucket in buckets:
        if required <= bucket:
            return GenerationPlan(bucket, max_new_tokens, bucket != required)
    return GenerationPlan(required, max_new_tokens, False)


def plan_authority_generation(
    prompt_tokens: int,
    max_new_tokens: int,
    max_generation_length: int,
):
    """Plan the narrow G3 authority call at its exact required length.

    The production bucket planner remains deliberately separate.  This path
    is only for a single authority invocation whose prompt and completion
    lengths are already fixed and source-verified.
    """
    prompt_tokens = int(prompt_tokens)
    max_new_tokens = int(max_new_tokens)
    max_generation_length = int(max_generation_length)
    required = prompt_tokens + max_new_tokens
    if prompt_tokens < 1 or max_new_tokens < 1:
        raise ValueError("prompt_tokens and max_new_tokens must be positive")
    if required > max_generation_length:
        raise ValueError(
            f"Prompt + completion requires {required} tokens, "
            f"above MAX_GENERATION_LENGTH={max_generation_length}"
        )
    return GenerationPlan(
        max_length=required,
        max_new_tokens=max_new_tokens,
        bucketed=False,
        mode="EXACT_LENGTH_AUTHORITY_PATH",
    )

def scalar_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return scalar_text(value[0])
    try:
        if hasattr(value, "item"):
            item = value.item()
            if isinstance(item, str):
                return item
    except Exception:
        pass
    return str(value)
