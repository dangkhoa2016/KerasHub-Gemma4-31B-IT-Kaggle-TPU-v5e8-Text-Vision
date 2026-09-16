from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:
    sys.path.insert(0,str(ROOT/"src"))

import unittest
import importlib.util

FLASK_AVAILABLE = importlib.util.find_spec('flask') is not None
from gemma4_server.api.app import (
    Runtime, create_app
)
from gemma4_server.core.config import Config

class DummyStore:
    def get(self,_):
        return None

class DummyManager:
    store=DummyStore()

    def health(self):
        return {
            "state":"loading",
            "ready":False,
            "ready_workers":0,
            "expected_workers":1,
            "accepting_jobs":False,
            "worker_generation":0,
            "automatic_restarts_used":0,
            "jobs":{},
            "workers":[],
            "runtime":{
                "model":"gemma4_instruct_31b",
                "mesh":[1,8],
            },
        }

    def submit(self,*a,**k):
        raise RuntimeError("not used")

@unittest.skipUnless(FLASK_AVAILABLE, 'Flask is optional for CPU-only source validation')
class T(unittest.TestCase):
    def test_index(self):
        app=create_app(
            Runtime(
                Config.for_tests(),
                DummyManager(),
            )
        )
        response=app.test_client().get("/")
        self.assertEqual(
            response.status_code,200
        )
        self.assertIn(
            "/generate",
            response.get_json()["endpoints"],
        )

    def test_info_auth(self):
        app=create_app(
            Runtime(
                Config.for_tests(),
                DummyManager(),
            )
        )
        client=app.test_client()
        self.assertEqual(
            client.get("/info").status_code,
            401,
        )
        self.assertEqual(
            client.get(
                "/info",
                headers={
                    "Authorization":
                    "Bearer test-api-key"
                },
            ).status_code,
            200,
        )
