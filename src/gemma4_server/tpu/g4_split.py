"""Small Gemma4-native split prefill/decode characterization path.

This module deliberately uses the same cache primitives that the installed
Gemma4CausalLM implementation uses inside its own ``generate_step``. It is a
G4 characterization helper, not a replacement production generation engine.
"""

from __future__ import annotations

import time

import numpy as np

from .generation import chat_prompt, scalar_text


G4_PROMPT = "Hello"
G4_EXPECTED_PROMPT_TOKENS = 10
G4_REQUESTED_NEW_TOKENS = 1
G4_MAX_LENGTH = G4_EXPECTED_PROMPT_TOKENS + G4_REQUESTED_NEW_TOKENS


def rendered_authority_prompt(prompt: str = G4_PROMPT) -> str:
    """Return the exact chat-formatted prompt used by the native authority."""
    return chat_prompt(prompt, "")


def _batched_inputs(model, prompt: str, sequence_length: int) -> dict:
    inputs = model.preprocessor.generate_preprocess(
        [prompt], sequence_length=int(sequence_length)
    )
    if not isinstance(inputs, dict):
        raise RuntimeError("Gemma4 preprocessor did not return a mapping")
    token_ids = np.asarray(inputs["token_ids"])
    padding_mask = np.asarray(inputs["padding_mask"])
    if token_ids.ndim != 2 or padding_mask.ndim != 2:
        raise RuntimeError(
            "G4 split authority requires batched token_ids and padding_mask"
        )
    if token_ids.shape != padding_mask.shape:
        raise RuntimeError("token_ids and padding_mask shapes differ")
    if token_ids.shape[0] != 1 or token_ids.shape[1] != sequence_length:
        raise RuntimeError(
            "G4 split authority requires one batch at the exact sequence length"
        )
    return inputs


def build_split_inputs(
    model,
    rendered_prompt: str,
    sequence_length: int = G4_MAX_LENGTH,
) -> dict:
    """Preprocess one exact-length text prompt for split execution."""
    return _batched_inputs(model, rendered_prompt, int(sequence_length))


def split_one_token(model, inputs: dict) -> dict[str, object]:
    """Run one explicit prefill and one explicit cached decode."""
    token_ids = np.asarray(inputs["token_ids"])
    padding_mask = np.asarray(inputs["padding_mask"], dtype=bool)
    prompt_lengths = np.sum(padding_mask, axis=-1)
    if len(prompt_lengths) != 1:
        raise RuntimeError("G4 split authority supports one sample")
    prompt_tokens = int(prompt_lengths[0])
    if prompt_tokens != G4_EXPECTED_PROMPT_TOKENS:
        raise RuntimeError(
            f"expected {G4_EXPECTED_PROMPT_TOKENS} prompt tokens, "
            f"found {prompt_tokens}"
        )
    if prompt_tokens >= token_ids.shape[1]:
        raise RuntimeError("split prompt leaves no decode position")

    prefill_started = time.perf_counter()
    _, cache = model._build_cache(
        token_ids=token_ids,
        img_embeddings=None,
        vision_mask=None,
        padding_mask=padding_mask,
        vision_indices=None,
    )
    prefill_seconds = time.perf_counter() - prefill_started

    decode_position = prompt_tokens - 1
    decode_token_ids = token_ids[:, decode_position : decode_position + 1]
    decode_update_mask = ~padding_mask[:, decode_position : decode_position + 1]
    decode_started = time.perf_counter()
    logits, _, _ = model.call_with_cache(
        token_ids=decode_token_ids,
        cache=cache,
        cache_update_index=decode_position,
        cache_update_mask=decode_update_mask,
    )
    next_token_ids = np.argmax(np.asarray(logits)[:, -1, :], axis=-1).astype(
        np.int32
    )
    decode_seconds = time.perf_counter() - decode_started

    generated_ids = next_token_ids[:, None]
    generated_mask = np.ones(generated_ids.shape, dtype=bool)
    generated_text = scalar_text(
        model.preprocessor.generate_postprocess(
            {"token_ids": generated_ids, "padding_mask": generated_mask}
        )
    ).strip()
    if not generated_text:
        raise RuntimeError("G4 split authority returned empty output")
    return {
        "result": generated_text,
        "prompt_tokens": prompt_tokens,
        "requested_new_tokens": G4_REQUESTED_NEW_TOKENS,
        "max_length": int(token_ids.shape[1]),
        "generation_call_count": 1,
        "split_prefill_seconds": round(prefill_seconds, 6),
        "split_decode_seconds": round(decode_seconds, 6),
        "split_generation_seconds": round(
            prefill_seconds + decode_seconds, 6
        ),
    }


def run_split_generation(
    model,
    prompt: str = G4_PROMPT,
) -> dict[str, object]:
    """Execute the fixed G4 one-token split characterization."""
    rendered = rendered_authority_prompt(prompt)
    inputs = build_split_inputs(model, rendered, G4_MAX_LENGTH)
    result = split_one_token(model, inputs)
    result.update(
        {
            "prompt_text": prompt,
            "rendered_prompt": rendered,
            "split_path": "Gemma4CausalLM._build_cache+call_with_cache",
        }
    )
    return result
