from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Iterable

import numpy as np


DEFAULT_MIN_SOURCE_NBYTES = 1 << 30
DEFAULT_SAMPLE_ROWS = (
    0,
    32767,
    32768,
    65535,
    65536,
    98303,
    98304,
    131071,
    131072,
    163839,
    163840,
    196607,
    196608,
    229375,
    229376,
    262143,
)
DEFAULT_SAMPLE_COLUMNS = (0, 1, 5375)


class R3AssignmentError(RuntimeError):
    """A large sharded checkpoint assignment could not be made safely."""


def _emit(callback, event: str, **payload):
    if callback is not None:
        callback(event, payload)


def _dtype_compatible(source_dtype, target_dtype) -> bool:
    try:
        return np.dtype(source_dtype) == np.dtype(target_dtype)
    except TypeError:
        return str(source_dtype) == str(target_dtype)


def _addressable_shards(value) -> list[Any]:
    shards = getattr(value, "addressable_shards", None)
    if shards is None:
        return []
    return list(shards)


def _candidate_target(source, target, jax_module, min_source_nbytes):
    if not isinstance(source, np.ndarray):
        return None

    target_value = getattr(target, "value", None)
    if target_value is None or not isinstance(target_value, jax_module.Array):
        return None

    target_sharding = getattr(target_value, "sharding", None)
    named_sharding = getattr(jax_module.sharding, "NamedSharding", None)
    if named_sharding is None or not isinstance(target_sharding, named_sharding):
        return None

    target_shards = _addressable_shards(target_value)
    if len(target_shards) <= 1:
        return None

    if source.nbytes <= min_source_nbytes:
        return None

    target_shape = tuple(int(dimension) for dimension in target_value.shape)
    if tuple(source.shape) != target_shape:
        raise R3AssignmentError(
            "R3 shape invariant failed: "
            f"source={source.shape}, target={target_value.shape}"
        )

    if not _dtype_compatible(source.dtype, target_value.dtype):
        raise R3AssignmentError(
            "R3 dtype invariant failed: "
            f"source={source.dtype}, target={target_value.dtype}"
        )

    return target_value, target_sharding, target_shards


def _verify_sharding(value, expected_sharding, expected_count, jax_module):
    if not isinstance(value, jax_module.Array):
        raise R3AssignmentError(
            f"R3 value is not a JAX array: {type(value).__name__}"
        )

    observed_sharding = getattr(value, "sharding", None)
    named_sharding = getattr(jax_module.sharding, "NamedSharding", None)
    if not isinstance(observed_sharding, named_sharding):
        raise R3AssignmentError(
            "R3 value does not have the required NamedSharding"
        )

    if observed_sharding.mesh != expected_sharding.mesh:
        raise R3AssignmentError("R3 value changed the target mesh")
    if observed_sharding.spec != expected_sharding.spec:
        raise R3AssignmentError("R3 value changed the target PartitionSpec")

    observed_shards = _addressable_shards(value)
    if len(observed_shards) != expected_count:
        raise R3AssignmentError(
            "R3 value changed the logical shard count: "
            f"expected={expected_count}, observed={len(observed_shards)}"
        )
    return observed_sharding, observed_shards


def _sharding_payload(value, sharding, shards):
    shapes = []
    for shard in shards:
        data = getattr(shard, "data", shard)
        shape = getattr(data, "shape", None)
        if shape is not None:
            shapes.append([int(dimension) for dimension in shape])
    return {
        "value_class": (
            f"{type(value).__module__}.{type(value).__name__}"
        ),
        "sharding_class": (
            f"{type(sharding).__module__}.{type(sharding).__name__}"
        ),
        "sharding_repr": repr(sharding),
        "sharding_spec": repr(getattr(sharding, "spec", None)),
        "sharding_mesh": repr(getattr(sharding, "mesh", None)),
        "addressable_shard_count": len(shards),
        "shard_shapes": shapes,
    }


def prepare_sharded_value(
    source: np.ndarray,
    target,
    *,
    jax_module=None,
    min_source_nbytes: int = DEFAULT_MIN_SOURCE_NBYTES,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    sample_rows: Iterable[int] | None = DEFAULT_SAMPLE_ROWS,
    sample_columns: Iterable[int] | None = DEFAULT_SAMPLE_COLUMNS,
):
    """Build a shard-local JAX value, or return ``None`` for ordinary values."""
    if jax_module is None:
        import jax as jax_module

    candidate = _candidate_target(
        source, target, jax_module, int(min_source_nbytes)
    )
    if candidate is None:
        return None

    target_value, target_sharding, target_shards = candidate
    target_path = str(
        getattr(target, "path", getattr(target, "name", "unknown"))
    )
    _emit(event_callback, "R3_TARGET_ENTER", target_path=target_path)
    _emit(
        event_callback,
        "SOURCE_READY",
        target_path=target_path,
        source_class=type(source).__name__,
        source_shape=[int(dimension) for dimension in source.shape],
        source_dtype=str(source.dtype),
        source_nbytes=int(source.nbytes),
    )
    _emit(
        event_callback,
        "TARGET_SHARDING_OBSERVED",
        target_path=target_path,
        **{
            f"target_{key}": value
            for key, value in _sharding_payload(
                target_value, target_sharding, target_shards
            ).items()
        },
    )

    def callback(index):
        _emit(
            event_callback,
            "SHARD_CALLBACK_ENTER",
            target_path=target_path,
            index=repr(index),
        )
        local_source = source[index]
        if not isinstance(local_source, np.ndarray):
            raise R3AssignmentError(
                "R3 callback did not return a NumPy shard-local view"
            )
        _emit(
            event_callback,
            "SHARD_CALLBACK_RETURN",
            target_path=target_path,
            index=repr(index),
            local_shape=[int(dimension) for dimension in local_source.shape],
        )
        return local_source

    _emit(
        event_callback,
        "SHARDED_TRANSFER_BEGIN",
        target_path=target_path,
    )
    try:
        sharded_value = jax_module.make_array_from_callback(
            source.shape,
            target_sharding,
            callback,
            dtype=source.dtype,
        )
    except Exception as exc:
        raise R3AssignmentError(
            "R3 shard-local construction failed"
        ) from exc

    _emit(
        event_callback,
        "SHARDED_VALUE_CREATED",
        target_path=target_path,
        **{
            f"sharded_{key}": value
            for key, value in _sharding_payload(
                sharded_value,
                getattr(sharded_value, "sharding", None),
                _addressable_shards(sharded_value),
            ).items()
        },
    )
    _verify_sharding(
        sharded_value,
        target_sharding,
        len(target_shards),
        jax_module,
    )
    _emit(
        event_callback,
        "SHARDED_VALUE_SHARDING_VERIFIED",
        target_path=target_path,
        **{
            f"sharded_{key}": value
            for key, value in _sharding_payload(
                sharded_value,
                sharded_value.sharding,
                _addressable_shards(sharded_value),
            ).items()
        },
    )
    return sharded_value


def sample_value_check(
    source: np.ndarray,
    target_value,
    *,
    jax_module=None,
    rows: Iterable[int] = DEFAULT_SAMPLE_ROWS,
    columns: Iterable[int] = DEFAULT_SAMPLE_COLUMNS,
):
    """Compare deterministic scalar samples without downloading the tensor."""
    if jax_module is None:
        import jax as jax_module

    for row in rows:
        for column in columns:
            expected = np.asarray(source[row, column]).item()
            observed = jax_module.device_get(target_value[row, column])
            observed = np.asarray(observed).item()
            if not np.array_equal(np.asarray(observed), np.asarray(expected)):
                raise R3AssignmentError(
                    "R3 sample mismatch at "
                    f"row={row}, column={column}: "
                    f"expected={expected!r}, observed={observed!r}"
                )
    return True


def _sample_indices_fit(source, rows, columns) -> bool:
    return (
        all(0 <= int(row) < source.shape[0] for row in rows)
        and all(0 <= int(column) < source.shape[1] for column in columns)
    )


def make_r3_assign_wrapper(
    original_assign,
    *,
    jax_module=None,
    min_source_nbytes: int = DEFAULT_MIN_SOURCE_NBYTES,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    sample_rows: Iterable[int] | None = DEFAULT_SAMPLE_ROWS,
    sample_columns: Iterable[int] | None = DEFAULT_SAMPLE_COLUMNS,
):
    if jax_module is None:
        import jax as jax_module

    @functools.wraps(original_assign)
    def wrapped(self, value):
        before_value = getattr(self, "value", None)
        before_sharding = getattr(before_value, "sharding", None)
        before_count = len(_addressable_shards(before_value))
        try:
            sharded_value = prepare_sharded_value(
                value,
                self,
                jax_module=jax_module,
                min_source_nbytes=min_source_nbytes,
                event_callback=event_callback,
                sample_rows=sample_rows,
                sample_columns=sample_columns,
            )
        except R3AssignmentError:
            raise

        if sharded_value is None:
            return original_assign(self, value)

        _emit(
            event_callback,
            "ORIGINAL_ASSIGN_WITH_SHARDED_VALUE_ENTER",
            target_path=str(
                getattr(self, "path", getattr(self, "name", "unknown"))
            ),
        )
        result = original_assign(self, sharded_value)
        _emit(
            event_callback,
            "ORIGINAL_ASSIGN_WITH_SHARDED_VALUE_SUCCESS",
            target_path=str(
                getattr(self, "path", getattr(self, "name", "unknown"))
            ),
        )

        after_value = getattr(self, "value", None)
        if before_sharding is None:
            raise R3AssignmentError(
                "R3 target has no sharding after assignment"
            )
        _verify_sharding(
            after_value,
            before_sharding,
            before_count,
            jax_module,
        )
        _emit(
            event_callback,
            "TARGET_SHARDING_AFTER_VERIFIED",
            target_path=str(
                getattr(self, "path", getattr(self, "name", "unknown"))
            ),
            **{
                f"target_{key}": value
                for key, value in _sharding_payload(
                    after_value,
                    after_value.sharding,
                    _addressable_shards(after_value),
                ).items()
            },
        )

        rows = tuple(sample_rows) if sample_rows is not None else None
        columns = (
            tuple(sample_columns) if sample_columns is not None else None
        )
        if (
            rows is not None
            and columns is not None
            and _sample_indices_fit(value, rows, columns)
        ):
            sample_value_check(
                value,
                after_value,
                jax_module=jax_module,
                rows=rows,
                columns=columns,
            )
            _emit(
                event_callback,
                "R3_SAMPLE_VALUE_CHECK_PASS",
                target_path=str(
                    getattr(self, "path", getattr(self, "name", "unknown"))
                ),
            )
        return result

    return wrapped


@contextlib.contextmanager
def sharded_checkpoint_assignment(
    *,
    variable_class=None,
    jax_module=None,
    min_source_nbytes: int = DEFAULT_MIN_SOURCE_NBYTES,
    event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    sample_rows: Iterable[int] | None = DEFAULT_SAMPLE_ROWS,
    sample_columns: Iterable[int] | None = DEFAULT_SAMPLE_COLUMNS,
):
    if variable_class is None:
        from keras.src.backend.common.variables import Variable

        variable_class = Variable
    original_assign = variable_class.assign
    variable_class.assign = make_r3_assign_wrapper(
        original_assign,
        jax_module=jax_module,
        min_source_nbytes=min_source_nbytes,
        event_callback=event_callback,
        sample_rows=sample_rows,
        sample_columns=sample_columns,
    )
    try:
        yield
    finally:
        variable_class.assign = original_assign
