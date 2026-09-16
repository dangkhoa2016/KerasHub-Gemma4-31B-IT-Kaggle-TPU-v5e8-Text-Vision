# G7 Scope Resolution

```text
G7_SCOPE_RESOLUTION=PASS
G7_ENTRY_ELIGIBLE=true
G7_STARTED=true
G6_EXPLICIT_FREEZE_ARTIFACT_PRESENT=true
G6_FREEZE_REQUIRED_FOR_G7=false
NEW_TPU_RUN_STARTED=false
```

## Frozen G6 input

The existing `docs/G6-CLOSEOUT.md` and `artifacts/g6/g6-acceptance.json` are
explicit immutable downstream markers. The authority archive used for
verification is:

```text
/kaggle/working/temp/gemma4-31b-g6-final-sharding-memory-evidence-20260914T034943Z.tar.gz
SHA256=52c6ac41100e736984895588d54a47d04bb4e41228b36e425d42c9a2f53c25f0
```

The archive digest matches the known G6 authority value and every entry in its
internal `SHA256SUMS` verifies. The supplied sidecar contains the same digest
but names the historical `/kaggle/working/` path; the actual local archive is
under `/kaggle/working/temp/`. No G6 artifact or science was modified.

## Canonical resolution

The canonical roadmap names G7 as REST server acceptance, G8 as async/cold
compile plus restart/lifecycle acceptance, and G9 as PRIME/HOT acceptance.
The README and API documents define the route and request surface but do not
require a fresh live model-backed REST request as a G7 acceptance condition.
Therefore the HTTP contract can be accepted with static and CPU fake/stub
proof; no TPU is required for G7.

| ID | Source | Resolved requirement | Class | Live model | Gate |
|---|---|---|---|---:|---|
| G7-REST-001 | `docs/ROADMAP.md`, gate list | G7 is REST server acceptance | STATIC | No | G7 |
| G7-REST-002 | `README.md`, REST endpoints | Documented REST route surface is registered | STATIC | No | G7 |
| G7-REST-003 | `docs/API.md`, Authentication | Bearer and X-API-Key protection works | CPU_UNIT | No | G7 |
| G7-REST-004 | `docs/API.md`, Text | Text request and async result HTTP mechanics work | CPU_INTEGRATION | No | G7 |
| G7-REST-005 | `docs/API.md`, Image | JSON base64 and multipart image HTTP mechanics work | CPU_INTEGRATION | No | G7 |
| G7-REST-006 | `docs/API.md`, Health | Live/ready/info JSON health surface works | CPU_UNIT | No | G7 |
| G7-REST-007 | `docs/API.md`, Restart | Restart route and secret gate are registered | STATIC | No | G7 |
| G7-BOUNDARY-001 | `docs/ROADMAP.md`, gate list | Full async/cold-compile/restart/lifecycle acceptance | STATIC | No | G8 |
| G7-BOUNDARY-002 | `docs/ROADMAP.md`, gate list | PRIME/HOT acceptance | STATIC | No | G9 |

`G7-REST-004` and `G7-REST-005` use an injected CPU fake/stub only for server
mechanics. They do not claim text or image model-science evidence. No streaming
requirement appears in the canonical G7 documents.

## Live-necessity decision

```text
G7_LIVE_MODEL_BACKED_REST_REQUIRED=false
G7_LIVE_TEXT_REST_REQUIRED=false
G7_LIVE_IMAGE_REST_REQUIRED=false
G7_STREAMING_REQUIRED=false
G7_AUTH_REQUIRED=true
G7_GRACEFUL_SHUTDOWN_REQUIRED=false
G7_TPU_REQUIRED=false
```
