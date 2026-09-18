from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:
    sys.path.insert(0,str(ROOT/"src"))

import tempfile, unittest
from gemma4_server.core.model_path import (
    model_complete, discover_from_base
)

def make(path):
    (path/"assets/tokenizer").mkdir(parents=True)
    for rel in (
        "config.json",
        "preprocessor.json",
        "assets/tokenizer/vocabulary.spm",
        "model.weights.json",
        "model_00000.weights.h5",
    ):
        p=path/rel
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text("{}")

class T(unittest.TestCase):
    def test_complete(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"gemma4_instruct_31b"
            make(p)
            self.assertTrue(model_complete(p))

    def test_highest_revision(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)/"gemma4_instruct_31b"
            for n in ("1","3","2"):
                make(base/n)
            self.assertEqual(
                discover_from_base(base).name,
                "3",
            )
