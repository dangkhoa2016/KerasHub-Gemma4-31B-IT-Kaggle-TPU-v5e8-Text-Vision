# G7 REST Contract Test Matrix

| ID | Requirement | Canonical source | Verification method | CPU/static complete | Live REST required | Live model required | Evidence file | Result | Blocking |
|---|---|---|---|---:|---:|---:|---|---|---:|
| G7-REST-001 | G7 is REST server acceptance | `docs/ROADMAP.md`, gate list | Source/scope review | Yes | No | No | `g7-scope-resolution.md` | PASS | No |
| G7-REST-002 | Documented route surface is registered | `README.md`, REST endpoints | Flask URL-map test | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-REST-003 | Bearer and X-API-Key authentication | `docs/API.md`, Authentication | Flask CPU auth tests | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-REST-004 | Text sync/async request mechanics and result envelope | `docs/API.md`, Text | CPU fake-manager HTTP tests and validation tests | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-REST-005 | JSON base64 and multipart image request mechanics | `docs/API.md`, Image | CPU Pillow fixture + fake-manager HTTP tests | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-REST-006 | Live/ready/info health surface | `docs/API.md`, Health | CPU status/schema tests; readiness uses injected manager | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-REST-007 | Restart route and secret gate are registered | `docs/API.md`, Restart | Static route test and unauthorized secret tests | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-TOPO-001 | One coordinator process and one model worker identity | `docs/ARCHITECTURE.md`, source | Source inventory and idempotent startup test | Yes | No | No | `g7-server-inventory.md`, `tests/test_g7_rest_contract.py` | PASS | No |
| G7-TOPO-002 | Health-only paths do not create a model | `docs/ARCHITECTURE.md`, import boundary | Source inspection and fake-manager readiness test | Yes | No | No | `g7-server-inventory.md`, `tests/test_g7_rest_contract.py` | PASS | No |
| G7-BOUNDARY-001 | Full async/cold compile/restart/lifecycle acceptance | `docs/ROADMAP.md`, G8 gate | Explicitly deferred; no lifecycle authority run | N/A | N/A | N/A | `g7-scope-resolution.md` | DEFERRED_TO_G8 | No |
| G7-BOUNDARY-002 | PRIME/HOT acceptance | `docs/ROADMAP.md`, G9 gate | Explicitly deferred; no PRIME/HOT run | N/A | N/A | N/A | `g7-scope-resolution.md` | DEFERRED_TO_G9 | No |

The fake manager/model boundary is used only for HTTP mechanics. No fake output
is used as model-science evidence.
