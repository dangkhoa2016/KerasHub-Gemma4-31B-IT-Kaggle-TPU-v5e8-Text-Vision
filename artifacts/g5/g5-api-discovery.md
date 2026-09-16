# G5 CPU API Discovery

Date: 2026-09-14. Inspection was performed before any G5 model execution.

## OBSERVED_API

- Installed `Gemma4CausalLMPreprocessor.generate_preprocess(x,
  sequence_length=...)` accepts a mapping with `prompts` and `images`.
- A scalar prompt is normalized by the installed generation path to a
  one-element prompt batch; a rank-3 RGB NumPy image is accepted and is
  promoted to one image for that batch.
- For a multimodal Gemma4 preprocessor, the image path returns
  `pixel_values`, `pixel_position_ids`, `token_ids`, `vision_indices`,
  `vision_mask`, and `padding_mask`.
- `pixel_values` are patch tensors and `pixel_position_ids` are patch
  coordinates. `vision_mask` marks image-token positions and
  `vision_indices` maps image embeddings to those positions.
- `Gemma4CausalLM.generate_step()` invokes the backbone `vision_encoder` on
  `pixel_values`/`pixel_position_ids`, then passes the resulting image
  embeddings to `_build_cache()`.
- `Gemma4CausalLM.generate()` accepts `max_length` and `strip_prompt`; the
  installed signature does not accept `max_new_tokens`.
- The existing project image flow uses `vision_prompt()` and
  `{"prompts": rendered_prompt, "images": np.asarray(image)}`.

## INFERENCE

- A small deterministic RGB PNG can be loaded as one rank-3 `uint8` array and
  handed to the installed preprocessor without a dataset or network asset.
- `VISION_CONDITIONING_PRESENT` can be established from non-empty image patch
  tensors plus a non-empty `vision_mask`/`vision_indices` result. The native
  `generate_step()` is then the installed, observed route that materializes
  image embeddings and conditions the text decoder.
- The G5 proof should use the native multimodal generation path with one
  loaded model and one `generate()` call. Preprocessing inspection is done
  once before that call and released before generation where safe.

## UNSUPPORTED_ASSUMPTIONS

- No public `prefill()`/`decode()` vision API is assumed.
- No invented `img_embeddings`, `vision_mask`, or `vision_indices` values are
  manufactured when the installed preprocessor does not return them.
- The proof does not claim image quality, production throughput, or public
  stability of private cache methods.
