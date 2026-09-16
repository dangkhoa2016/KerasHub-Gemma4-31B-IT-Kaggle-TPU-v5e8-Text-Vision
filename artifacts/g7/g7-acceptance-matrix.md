# G7 Acceptance Matrix

| ID | Requirement | Canonical source | Verification method | CPU/static complete | Live REST required | Live model required | Evidence file | Result | Blocking |
|---|---|---|---|---:|---:|---:|---|---|---:|
| G7-REST-001 | G7 is REST server acceptance | `docs/ROADMAP.md` gate list | Scope resolution | Yes | No | No | `g7-scope-resolution.md` | PASS | No |
| G7-REST-002 | Documented REST route surface is registered | `README.md` REST endpoints | Flask URL-map test | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-REST-003 | Bearer and X-API-Key authentication | `docs/API.md` Authentication | CPU auth tests | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-REST-004 | Text sync/async request mechanics and result envelope | `docs/API.md` Text | CPU fake-manager HTTP tests | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-REST-005 | JSON base64 and multipart image mechanics | `docs/API.md` Image | CPU Pillow fixture + HTTP tests | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-REST-006 | Live/ready/info health surface | `docs/API.md` Health | CPU status/schema tests | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-REST-007 | Restart route and secret gate are registered | `docs/API.md` Restart | Static route + unauthorized secret tests | Yes | No | No | `tests/test_g7_rest_contract.py` | PASS | No |
| G7-TOPO-001 | One coordinator process and one model worker identity | `docs/ARCHITECTURE.md` | Source inventory + idempotent startup test | Yes | No | No | `g7-server-inventory.md`, `tests/test_g7_rest_contract.py` | PASS | No |
| G7-TOPO-002 | Health-only paths do not create a model | `docs/ARCHITECTURE.md` | Source/import boundary + readiness test | Yes | No | No | `g7-server-inventory.md`, `tests/test_g7_rest_contract.py` | PASS | No |
| G7-BOUNDARY-001 | Full async/cold compile/restart/lifecycle acceptance | `docs/ROADMAP.md` G8 | Explicit deferral | N/A | N/A | N/A | `g7-scope-resolution.md` | DEFERRED_TO_G8 | No |
| G7-BOUNDARY-002 | PRIME/HOT acceptance | `docs/ROADMAP.md` G9 | Explicit deferral | N/A | N/A | N/A | `g7-scope-resolution.md` | DEFERRED_TO_G9 | No |

All mandatory G7 rows have evidence. Fake/stub output proves only REST
mechanics; it is not model-science evidence.
