from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:
    sys.path.insert(0,str(ROOT/"src"))

import unittest
from gemma4_server.jobs.models import Job
from gemma4_server.jobs.store import JobStore

class T(unittest.TestCase):
    def test_lifecycle(self):
        store=JobStore(4,60)
        job=Job("j","p","",16)
        store.put(job)
        store.mark_processing("j","tpu-0")
        store.mark_completed(
            "j","ok",1.0
        )
        self.assertEqual(
            store.get("j").public_dict()["output"],
            "ok",
        )
