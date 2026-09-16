# G8 Final Acceptance Matrix

| Area | Acceptance condition | Result | Existing evidence |
|---|---|---|---|
| CPU/static qualification | G0-G7 closed; G8 async/lifecycle/cold-compile static and CPU evidence qualified | PASS | `artifacts/g8/g8-acceptance.json`, `g8-test-matrix.md`, `g8-cpu-verification.txt` |
| Post-fix regression | Complete post-fix suite is 133/133 PASS | PASS | `artifacts/g8/post-fix-closeout-20260916/final-adjudication.txt`, `README.md` |
| Compileall | Post-fix compileall passes | PASS | `artifacts/g8/post-fix-closeout-20260916/final-adjudication.txt`, `README.md` |
| TPU runtime qualification | Runtime requalified; JAX has 8 devices and default backend TPU | PASS | `post-fix-closeout-20260916/runtime-evidence.json` |
| Model load | Gemma4 load and readiness completed | PASS | `runtime-evidence.json` readiness/runtime metadata |
| Production readiness | HTTP readiness 200, ready=true, accepting_jobs=true | PASS | `runtime-evidence.json` |
| Planner/bucket | Prompt 10 tokens, required length 11, generation max length 16, bucketed; bucket 16 live PASS | PASS | `runtime-evidence.json`, `server-transition.log`, `final-adjudication.txt` |
| REST generation | Canonical async job completed with nonempty output | PASS | `runtime-evidence.json`, `server-transition.log` |
| Collector correctness | Nested `metrics.generation_seconds` fallback is tested and source hash matches | PASS | `source-provenance.txt`, current manager/test hashes |
| OOM status | OOM kill and group-kill deltas are zero | PASS | `runtime-evidence.json`, `final-adjudication.txt` |
| Source provenance | Required manager and lifecycle-test SHA256 values match | PASS | `03-source-provenance-verification.txt`, `source-provenance.txt` |
| Historical failure preservation | Earlier timeout/preflight/runtime failures remain represented and unmodified | PASS | `06-historical-attempt-classification.txt`, retained temp packages |

All rows are supported by existing evidence. No row is inferred from a new
live run. G9 PRIME/HOT, warm, throughput, and latency work remains deferred.
