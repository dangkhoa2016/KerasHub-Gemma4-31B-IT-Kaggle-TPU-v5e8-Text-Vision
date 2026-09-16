from __future__ import annotations

import contextlib
import unittest

import numpy as np

from gemma4_server.tpu.sharded_checkpoint import (
    R3AssignmentError,
    make_r3_assign_wrapper,
    prepare_sharded_value,
    sharded_checkpoint_assignment,
)


class FakeNamedSharding:
    def __init__(self, mesh="mesh", spec=("model", "batch")):
        self.mesh = mesh
        self.spec = spec

    def __str__(self):
        return f"NamedSharding(mesh={self.mesh!r}, spec={self.spec!r})"


class FakeArray:
    def __init__(self, shape, dtype, sharding, payload=None):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.sharding = sharding
        self.addressable_shards = [object()] * 8
        self.payload = payload

    def __getitem__(self, index):
        if self.payload is None:
            raise AssertionError("fake target has no payload")
        return self.payload[index]


class FakeJax:
    class sharding:
        NamedSharding = FakeNamedSharding

    Array = FakeArray

    def __init__(self, indices):
        self.indices = list(indices)
        self.callback_indices = []
        self.result_payload = None

    def make_array_from_callback(self, shape, sharding, callback, dtype=None):
        payload = []
        for index in self.indices:
            self.callback_indices.append(index)
            payload.append(callback(index))
        return FakeArray(
            shape,
            dtype,
            sharding,
            payload=(
                self.result_payload if self.result_payload is not None else payload
            ),
        )

    @staticmethod
    def device_get(value):
        return value


class FakeTarget:
    def __init__(self, value):
        self.value = value
        self.shape = value.shape
        self.dtype = value.dtype
        self.assigned = None


class ShardedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.sharding = FakeNamedSharding()
        self.indices = [
            (slice(0, 2), slice(None)),
            (slice(2, 4), slice(None)),
        ]
        self.jax = FakeJax(self.indices)
        self.source = np.arange(24, dtype=np.float32).reshape(4, 6)
        self.target = FakeTarget(
            FakeArray(
                self.source.shape,
                self.source.dtype,
                self.sharding,
                payload=self.source.copy(),
            )
        )

    def test_non_numpy_source_uses_normal_path(self):
        calls = []
        wrapper = make_r3_assign_wrapper(
            self._normal_assign(calls),
            jax_module=self.jax,
            min_source_nbytes=0,
        )

        value = wrapper(self.target, "not-an-array")

        self.assertEqual(value, "normal-result")
        self.assertEqual(calls, ["not-an-array"])
        self.assertEqual(self.jax.callback_indices, [])

    def test_unsharded_target_uses_normal_path(self):
        self.target.value.sharding = None
        calls = []
        wrapper = make_r3_assign_wrapper(
            self._normal_assign(calls),
            jax_module=self.jax,
            min_source_nbytes=0,
        )

        wrapper(self.target, self.source)

        self.assertEqual(calls, [self.source])
        self.assertEqual(self.jax.callback_indices, [])

    def test_small_source_uses_normal_path(self):
        calls = []
        wrapper = make_r3_assign_wrapper(
            self._normal_assign(calls),
            jax_module=self.jax,
            min_source_nbytes=self.source.nbytes + 1,
        )

        wrapper(self.target, self.source)

        self.assertEqual(calls, [self.source])
        self.assertEqual(self.jax.callback_indices, [])

    def test_shape_mismatch_fails_closed(self):
        calls = []
        wrapper = make_r3_assign_wrapper(
            self._normal_assign(calls),
            jax_module=self.jax,
            min_source_nbytes=0,
        )

        with self.assertRaises(R3AssignmentError):
            wrapper(self.target, np.zeros((3, 6), dtype=np.float32))

        self.assertEqual(calls, [])

    def test_incompatible_dtype_fails_closed(self):
        calls = []
        wrapper = make_r3_assign_wrapper(
            self._normal_assign(calls),
            jax_module=self.jax,
            min_source_nbytes=0,
        )

        with self.assertRaises(R3AssignmentError):
            wrapper(self.target, self.source.astype(np.float16))

        self.assertEqual(calls, [])

    def test_callback_receives_only_shard_local_indices(self):
        result = prepare_sharded_value(
            self.source,
            self.target,
            jax_module=self.jax,
            min_source_nbytes=0,
            sample_rows=(),
            sample_columns=(),
        )

        self.assertEqual(self.jax.callback_indices, self.indices)
        self.assertEqual(result.payload[0].shape, (2, 6))
        np.testing.assert_array_equal(result.payload[0], self.source[:2])
        np.testing.assert_array_equal(result.payload[1], self.source[2:])

    def test_context_restores_patch_on_success(self):
        original = object()
        FakeVariable = type("FakeVariable", (), {"assign": original})

        with sharded_checkpoint_assignment(
            variable_class=FakeVariable,
            jax_module=self.jax,
            min_source_nbytes=0,
        ):
            self.assertIsNot(FakeVariable.assign, original)

        self.assertIs(FakeVariable.assign, original)

    def test_context_restores_patch_on_exception(self):
        original = object()
        FakeVariable = type("FakeVariable", (), {"assign": original})

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with sharded_checkpoint_assignment(
                variable_class=FakeVariable,
                jax_module=self.jax,
                min_source_nbytes=0,
            ):
                raise RuntimeError("boom")

        self.assertIs(FakeVariable.assign, original)

    def test_success_assigns_sharded_value_and_samples_it(self):
        self.jax.result_payload = self.source.copy()
        events = []

        def assign(target, value):
            target.value = value
            return value

        wrapper = make_r3_assign_wrapper(
            assign,
            jax_module=self.jax,
            min_source_nbytes=0,
            event_callback=lambda event, payload: events.append(event),
            sample_rows=(0, 2),
            sample_columns=(0, 5),
        )

        result = wrapper(self.target, self.source)

        self.assertIs(result, self.target.value)
        self.assertEqual(
            events,
            [
                "R3_TARGET_ENTER",
                "SOURCE_READY",
                "TARGET_SHARDING_OBSERVED",
                "SHARDED_TRANSFER_BEGIN",
                "SHARD_CALLBACK_ENTER",
                "SHARD_CALLBACK_RETURN",
                "SHARD_CALLBACK_ENTER",
                "SHARD_CALLBACK_RETURN",
                "SHARDED_VALUE_CREATED",
                "SHARDED_VALUE_SHARDING_VERIFIED",
                "ORIGINAL_ASSIGN_WITH_SHARDED_VALUE_ENTER",
                "ORIGINAL_ASSIGN_WITH_SHARDED_VALUE_SUCCESS",
                "TARGET_SHARDING_AFTER_VERIFIED",
                "R3_SAMPLE_VALUE_CHECK_PASS",
            ],
        )

    def test_changed_target_sharding_fails_closed_after_assign(self):
        self.jax.result_payload = self.source.copy()

        def assign(target, value):
            value.sharding = FakeNamedSharding(mesh="different")
            target.value = value
            return value

        wrapper = make_r3_assign_wrapper(
            assign,
            jax_module=self.jax,
            min_source_nbytes=0,
            sample_rows=(),
            sample_columns=(),
        )

        with self.assertRaises(R3AssignmentError):
            wrapper(self.target, self.source)

    def test_noncanonical_large_tensor_skips_embedding_samples(self):
        self.jax.result_payload = self.source.copy()

        def assign(target, value):
            target.value = value
            return value

        wrapper = make_r3_assign_wrapper(
            assign,
            jax_module=self.jax,
            min_source_nbytes=0,
        )

        result = wrapper(self.target, self.source)

        self.assertIs(result, self.target.value)

    def test_r3_helper_does_not_use_full_conversion_or_device_put(self):
        from pathlib import Path

        module_text = Path(
            __file__
        ).parents[1].joinpath(
            "src/gemma4_server/tpu/sharded_checkpoint.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("jnp.asarray", module_text)
        self.assertNotIn("jax.device_put", module_text)

    @staticmethod
    def _normal_assign(calls):
        def assign(target, value):
            calls.append(value)
            return "normal-result"

        return assign


if __name__ == "__main__":
    unittest.main()
