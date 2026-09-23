from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StableGreedySamplerTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("jax") is not None,
        "JAX is not installed in the lightweight CI environment",
    )
    def test_same_shape_requests_do_not_capture_previous_padding_mask(self):
        script = textwrap.dedent(
            r"""
            import numpy as np
            import jax
            import keras
            from keras import ops
            from gemma4_server.tpu.stable_greedy_sampler import (
                make_stable_gemma4_greedy_sampler,
            )

            class Gemma4CausalLM:
                def __init__(self):
                    self.trainable_variables = []
                    self.non_trainable_variables = []
                    self._assistant_model = None

                def call_with_cache(
                    self,
                    token_ids,
                    cache,
                    cache_update_index,
                    cache_update_mask,
                ):
                    del token_ids, cache_update_index, cache_update_mask
                    logits = ops.one_hot(
                        ops.convert_to_tensor([[4]], dtype="int32"),
                        8,
                    )
                    logits = ops.cast(logits, "float32") * 10.0
                    hidden = ops.zeros((1, 1, 1), dtype="float32")
                    return logits, hidden, cache

            sampler = make_stable_gemma4_greedy_sampler()
            model = Gemma4CausalLM()

            def run(prompt, mask, index):
                result = sampler(
                    next=None,
                    prompt=ops.convert_to_tensor(
                        np.asarray(prompt, dtype=np.int32)
                    ),
                    cache=(),
                    index=index,
                    mask=ops.convert_to_tensor(
                        np.asarray(mask, dtype=bool)
                    ),
                    stop_token_ids=None,
                    model=model,
                )
                jax.block_until_ready(result)
                return np.asarray(result).tolist()

            first = run(
                [[1, 2, 3, 0, 0, 0]],
                [[1, 1, 1, 0, 0, 0]],
                3,
            )
            second = run(
                [[9, 8, 0, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0]],
                2,
            )
            assert first == [[1, 2, 3, 4, 4, 4]], first
            assert second == [[9, 8, 4, 4, 4, 4]], second
            print("STABLE_MASK_REUSE_PASS")
            """
        )
        env = os.environ.copy()
        env["KERAS_BACKEND"] = "jax"
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout}\nstderr={result.stderr}",
        )
        self.assertIn("STABLE_MASK_REUSE_PASS", result.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("jax") is not None,
        "JAX is not installed in the lightweight CI environment",
    )
    def test_stop_token_semantics_stop_after_new_eos(self):
        script = textwrap.dedent(
            r"""
            import numpy as np
            import jax
            from keras import ops
            from gemma4_server.tpu.stable_greedy_sampler import (
                make_stable_gemma4_greedy_sampler,
            )

            class Gemma4CausalLM:
                def __init__(self):
                    self.trainable_variables = []
                    self.non_trainable_variables = []
                    self._assistant_model = None

                def call_with_cache(
                    self,
                    token_ids,
                    cache,
                    cache_update_index,
                    cache_update_mask,
                ):
                    del token_ids, cache_update_index, cache_update_mask
                    logits = ops.one_hot(
                        ops.convert_to_tensor([[4]], dtype="int32"),
                        8,
                    )
                    logits = ops.cast(logits, "float32") * 10.0
                    hidden = ops.zeros((1, 1, 1), dtype="float32")
                    return logits, hidden, cache

            sampler = make_stable_gemma4_greedy_sampler()
            model = Gemma4CausalLM()
            result = sampler(
                next=None,
                prompt=ops.convert_to_tensor(
                    np.asarray([[1, 2, 3, 0, 0, 0]], dtype=np.int32)
                ),
                cache=(),
                index=3,
                mask=ops.convert_to_tensor(
                    np.asarray([[1, 1, 1, 0, 0, 0]], dtype=bool)
                ),
                stop_token_ids=(4,),
                model=model,
            )
            jax.block_until_ready(result)
            observed = np.asarray(result).tolist()
            assert observed == [[1, 2, 3, 4, 0, 0]], observed
            print("STABLE_STOP_TOKEN_PASS")
            """
        )
        env = os.environ.copy()
        env["KERAS_BACKEND"] = "jax"
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout}\nstderr={result.stderr}",
        )
        self.assertIn("STABLE_STOP_TOKEN_PASS", result.stdout)

    def test_source_fails_closed_for_assistant_models(self):
        source = (
            ROOT / "src/gemma4_server/tpu/stable_greedy_sampler.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "does not support speculative decoding",
            source,
        )
        self.assertIn(
            "does not support assistant models",
            source,
        )


if __name__ == "__main__":
    unittest.main()
