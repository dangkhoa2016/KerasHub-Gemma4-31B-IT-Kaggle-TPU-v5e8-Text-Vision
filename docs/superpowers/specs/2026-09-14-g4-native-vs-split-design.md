# G4 Native-vs-Split Characterization Design

## Goal

Characterize the frozen native G3 path against the smallest repository-consistent Gemma4-native split prefill/decode path while preserving G3 source/result boundaries and using at most two real 31B TPU attempts.

## Architecture

Add a G4-only split helper and authority script. The authority reuses the frozen `Gemma4TPUEngine` loader, Candidate-A distribution, checkpoint assignment, and `run_eagerly=True` model, then calls the installed Gemma4 cache primitives directly: `_build_cache` for prefill and `call_with_cache` for one-token decode. The runner performs hardware/runtime/dependency gates before starting exactly one Python authority process, records cgroup memory at each phase, parses frozen G3 evidence for the native baseline, and packages comparison evidence.

## Contract

- Prompt is `Hello`; prompt token count is 10; requested new tokens is 1; exact max length is 11.
- One Python G4 authority process initializes TPU, loads exactly one model once, verifies Candidate-A, executes split prefill/decode once, and writes JSON.
- Candidate-A remains `[262144,5376]`, BF16, eight `[32768,5376]` shards, `P('model','batch')`.
- No G3 runner invocation, native rerun, second model, whole-buffer `device_put`, or G3 freeze mutation.
- Valid TPU exposure is `/dev/accel*` or eight numeric `/dev/vfio/*` nodes with `memory.max >= 300 GiB`.
- First direct implementation defect may receive one narrow corrective TPU rerun after CPU/static verification; no third attempt.

## Components and data flow

1. `g4_split.py` exposes pure input/result helpers plus `run_split_generation(model, prompt, ...)`.
2. `g4_split_authority.py` owns the one JAX/model process, load markers, Candidate-A check, memory snapshots, split call, and result JSON.
3. `run_g4_characterization.sh` owns pre-JAX hardware/runtime/dependency gates, fallback configuration, evidence logs, baseline parsing, adjudication, archive, and hashes.
4. Focused unit/source tests cover gate ordering, call contracts, one-load/no-duplicate invariants, and parseable evidence.

## Error handling

Hardware/runtime/dependency failures stop before model initialization. Candidate-A drift stops immediately with `G4_CANDIDATE_A_DRIFT`. A direct split integration error is recorded, fixed narrowly, retested on CPU, and may consume the one corrective attempt. If split is proven incompatible with the installed API without changing model/checkpoint semantics, adjudication is `NOT_VIABLE` with native retained. Any second real TPU failure closes the budget as exhausted.

## Testing

Use `python3 -m unittest discover -s tests -p 'test_*.py'`, `compileall`, and `bash -n`; do not install pytest or rg. CPU/static tests must pass before the primary TPU process starts. Final claims require fresh verification and archive/hash inspection.
