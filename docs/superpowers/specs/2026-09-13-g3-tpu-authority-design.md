# G3 TPU Authority Integrated Path

## Goal

Add one canonical, idempotent vNext path that admits only a real Kaggle TPU
v5e-8 session, restores the exact frozen runtime when necessary, loads the
Gemma4 31B model with the existing R3 sharded-checkpoint engine, verifies
Candidate-A, performs host cleanup, executes the exact one-token G3 text
generation, and emits a compact evidence archive.

## Architecture

`run_g3_tpu_authority.sh` owns all pre-import admission and runtime checks. It
checks the model path, TPU device nodes, cgroup memory floor, exact package
metadata, and the project-scoped dependency closure before starting the single
Python authority process. It configures the established Kaggle TPU fallback
environment only after admission and never imports JAX itself.

`g3_tpu_authority.py` is the sole JAX-importing authority process. It starts an
in-process TPU initialization watchdog, verifies exactly eight TPU devices,
constructs the existing Candidate-A distribution, calls
`Gemma4TPUEngine.load()` under the existing R3 assignment context, verifies the
token embedding contract, runs the existing cleanup, and invokes native
generation exactly once with `max_length=11`, `max_new_tokens=1`, and
`strip_prompt=True`.

The shell runner captures only the required evidence files, writes final
adjudication markers for early gates and authority outcomes, and archives the
single evidence directory. A non-TPU environment exits before any JAX/Keras
import and records `TPU_HARDWARE_NOT_READY`.

## Frozen contract

- Model preset: `gemma4_instruct_31b`
- Model path: `/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_31b/2`
- Backend/dtype: `KERAS_BACKEND=jax`, `bfloat16`
- TPU: v5e-8, exactly 8 devices
- Mesh: shape `[1,8]`, axes `[batch,model]`
- Candidate-A: the existing `LAYOUT_PROFILE` and layout map
- R3: `jax.make_array_from_callback()` with shard-local source slices, then normal `Variable.assign`
- G3: prompt `Hello`, prompt token count `10`, one new token, authority max length `11`, stripped prompt

## Error handling

Hardware and runtime gates fail before JAX import. The TPU watchdog writes a
timeout marker, flushes evidence, and exits 124 after 180 seconds. The strict
load has one absolute 2304-second cap and no retries/extensions. Device-count,
Candidate-A, generation, and OOM outcomes are persisted with the prescribed
final result and the process stops without creating a corrective loop.

## Testing

Focused CPU-only tests cover hardware-gate predicates, exact runtime matching,
project-scoped dependency filtering, authority source contracts, generation
single-call behavior, Candidate-A verification, timeout constants, and the
absence of forbidden full-buffer R3 operations. The real authority command is
run only when the hardware gate admits a TPU session.
