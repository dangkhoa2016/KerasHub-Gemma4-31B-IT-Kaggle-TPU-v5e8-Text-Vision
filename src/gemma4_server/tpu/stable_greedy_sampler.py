from __future__ import annotations


def make_stable_gemma4_greedy_sampler():
    """Return a Gemma4-specific greedy sampler with stable JAX loop callables."""
    import itertools

    import jax
    import keras
    from keras import ops
    from keras_hub.src.samplers.greedy_sampler import GreedySampler
    from keras_hub.src.utils.tensor_utils import any_equal

    class StableGemma4GreedySampler(GreedySampler):
        def __init__(self):
            super().__init__()
            # Define these once so JAX sees stable Python callable identities.
            self._stable_cond = self._make_stable_cond()
            self._stable_body = self._make_stable_body()

        def _make_stable_cond(self):
            def stable_cond(carry):
                (
                    state,
                    prompt,
                    cache,
                    index,
                    mask,
                    stop_ids,
                    stop_enabled,
                    current_iter,
                    max_iter,
                ) = carry
                del state, cache, index
                end_tokens = any_equal(prompt, stop_ids, ~mask)
                prompt_done = ops.any(end_tokens, axis=-1)
                stop_done = ops.all(prompt_done)
                keep_sampling = ops.logical_or(
                    ops.logical_not(stop_enabled),
                    ops.logical_not(stop_done),
                )
                return ops.logical_and(current_iter < max_iter, keep_sampling)

            return stable_cond

        def _make_stable_body(self):
            def stable_body(carry):
                (
                    state,
                    prompt,
                    cache,
                    index,
                    mask,
                    stop_ids,
                    stop_enabled,
                    current_iter,
                    max_iter,
                ) = carry
                (
                    sampler_variables,
                    trainable_variables,
                    non_trainable_variables,
                ) = state

                mapping = itertools.chain(
                    zip(self.variables, sampler_variables),
                    zip(self._stable_model.trainable_variables, trainable_variables),
                    zip(
                        self._stable_model.non_trainable_variables,
                        non_trainable_variables,
                    ),
                )
                with keras.StatelessScope(state_mapping=mapping) as scope:
                    cache_update_index = index - 1
                    batch_size = ops.shape(prompt)[0]
                    prompt_slice = ops.slice(
                        prompt, [0, index - 1], [batch_size, 1]
                    )
                    cache_update_mask = ops.slice(
                        ~mask, [0, index - 1], [batch_size, 1]
                    )
                    logits, _, cache = self._stable_model.call_with_cache(
                        token_ids=prompt_slice,
                        cache=cache,
                        cache_update_index=cache_update_index,
                        cache_update_mask=cache_update_mask,
                    )
                    logits = ops.squeeze(logits, axis=1)
                    probabilities = self.compute_probabilities(logits)
                    next_token = self.get_next_token(probabilities)
                    next_token = ops.cast(next_token, prompt.dtype)
                    next_token = ops.where(
                        mask[:, index], prompt[:, index], next_token
                    )
                    prompt = ops.slice_update(
                        prompt, [0, index], next_token[:, None]
                    )

                updated_sampler_variables = []
                for variable in self.variables:
                    current = scope.get_current_value(variable)
                    updated_sampler_variables.append(
                        current if current is not None else variable
                    )
                state = (
                    updated_sampler_variables,
                    trainable_variables,
                    non_trainable_variables,
                )
                return (
                    state,
                    prompt,
                    cache,
                    index + 1,
                    mask,
                    stop_ids,
                    stop_enabled,
                    current_iter + 1,
                    max_iter,
                )

            return stable_body

        def __call__(
            self,
            next,
            prompt,
            cache=None,
            index=0,
            mask=None,
            stop_token_ids=None,
            hidden_states=None,
            model=None,
            **kwargs,
        ):
            del next, hidden_states
            if keras.config.backend() != "jax":
                raise RuntimeError("StableGemma4GreedySampler requires JAX")
            if model is None or model.__class__.__name__ != "Gemma4CausalLM":
                raise RuntimeError("StableGemma4GreedySampler requires Gemma4CausalLM")
            if kwargs.get("draft_next") is not None or kwargs.get("verify_next") is not None:
                raise RuntimeError(
                    "StableGemma4GreedySampler does not support speculative decoding"
                )
            if getattr(model, "_assistant_model", None) is not None:
                raise RuntimeError(
                    "StableGemma4GreedySampler does not support assistant models"
                )

            self._stable_model = model
            max_length = ops.cast(ops.shape(prompt)[-1], "int32")
            index = ops.cast(index, "int32")
            if mask is None:
                mask = ops.zeros_like(prompt, dtype="bool")
            else:
                mask = ops.cast(mask, "bool")
            cache = () if cache is None else cache

            if stop_token_ids is None:
                stop_ids = ops.convert_to_tensor([-1], dtype=prompt.dtype)
                stop_enabled = ops.convert_to_tensor(False, dtype="bool")
            else:
                stop_ids = ops.convert_to_tensor(stop_token_ids, dtype=prompt.dtype)
                stop_enabled = ops.convert_to_tensor(True, dtype="bool")

            sampler_variables = [ops.convert_to_tensor(v) for v in self.variables]
            trainable_variables = [
                ops.convert_to_tensor(v) for v in model.trainable_variables
            ]
            non_trainable_variables = [
                ops.convert_to_tensor(v) for v in model.non_trainable_variables
            ]
            state = (
                sampler_variables,
                trainable_variables,
                non_trainable_variables,
            )
            carry = (
                state,
                prompt,
                cache,
                index,
                mask,
                stop_ids,
                stop_enabled,
                ops.cast(0, "int32"),
                max_length - index,
            )
            carry = jax.lax.while_loop(
                self._stable_cond,
                self._stable_body,
                carry,
            )
            state, prompt, _, _, _, _, _, _, _ = carry
            for ref_v, value in zip(self.variables, state[0]):
                ref_v.assign(value)
            return prompt

    return StableGemma4GreedySampler()
