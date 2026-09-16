# G4 CPU API discovery

Date: 2026-09-14
Runtime inspected from installed KerasHub 0.29.1, Keras 3.15.0, JAX 0.10.2.
This inspection did not enumerate or initialize JAX devices.

## OBSERVED_API

- `keras_hub.models.Gemma4CausalLM` subclasses `CausalLM` and constructs a
  `Gemma4Backbone`; `Gemma4CausalLM.generate(inputs, max_length=None,
  stop_token_ids="auto", strip_prompt=False, assistant_model=None)` delegates
  to the inherited high-level generation path.
- `Gemma4CausalLM.call_with_cache(token_ids, cache, cache_update_index,
  img_embeddings=None, vision_mask=None, padding_mask=None,
  vision_indices=None, audio_embeddings=None, audio_indices=None,
  audio_mask=None, cache_update_mask=None)` returns
  `(logits, hidden_states, cache)`.
- `Gemma4CausalLM._build_cache(token_ids, img_embeddings, vision_mask,
  padding_mask, vision_indices, audio_embeddings=None, audio_indices=None,
  audio_mask=None)` allocates the cache and seeds it using one
  `call_with_cache` pass.
- The cache allocation is
  `[batch, num_layers, 2, max_length, num_key_value_heads, max_head_dim]`,
  where `max_head_dim=max(head_dim, global_head_dim)` when a global head is
  present.
- `Gemma4CausalLM.generate_step()` uses exactly this cache API: it builds the
  cache, computes `row_lengths` from `padding_mask`, then its nested `next()`
  slices one token and calls `call_with_cache` with
  `cache_update_index=index-1` and a one-token `cache_update_mask`.
- `Gemma4CausalLMPreprocessor.generate_preprocess(x, sequence_length=...)`
  accepts raw text or a dict with `prompts`; text-only input yields
  `token_ids` and `padding_mask`. Multimodal models additionally expose empty
  or populated pixel/audio fields, but the text-only authority does not need
  to pass them into `call_with_cache`.
- `CausalLM.generate()` accepts `max_length` and `strip_prompt`; the installed
  signature does not accept `max_new_tokens`.

## INFERENCE

- The smallest repository-consistent split prototype is one model process
  using the already-loaded model: preprocess `Hello` at exact length 11,
  prefill with `_build_cache`, decode one token with `call_with_cache`, and
  greedy-select the largest logit. This exercises the model's explicit
  prefill/cache/decode primitives without invoking native `generate()`.
- `cache_update_mask=False` for the final prompt token preserves the cache
  slot seeded by prefill; the returned logits are the next-token logits.
- A one-token result can be postprocessed with the installed tokenizer through
  `generate_postprocess` using a one-token `token_ids`/`padding_mask` result.
- The native path's whole-model JIT risk is avoided because the candidate does
  not call `generate()` and does not create a second model or checkpoint
  buffer. The process remains `run_eagerly=True` after the frozen load.

## UNSUPPORTED_ASSUMPTIONS

- No undocumented cache object, cache-update class, or separate public
  `prefill()`/`decode()` method is assumed.
- No claim is made that the prototype is production-ready, faster than native
  generation, or semantically equivalent for multimodal prompts.
- The prototype relies on the installed implementation's underscore methods
  because those are the actual primitives used by its own `generate_step()`;
  this is an integration characterization, not a public API guarantee.
