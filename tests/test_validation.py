from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:
    sys.path.insert(0,str(ROOT/"src"))

import unittest
from gemma4_server.core.config import Config
from gemma4_server.core.validation import (
    parse_generate_payload
)
from gemma4_server.core.errors import (
    ValidationError
)

class T(unittest.TestCase):
    def setUp(self):
        self.config=Config.for_tests()

    def test_text(self):
        payload=parse_generate_payload(
            {
                "prompt":"hello",
                "max_new_tokens":32,
            },
            self.config,
        )
        self.assertEqual(
            payload["max_tokens"],32
        )

    def test_empty(self):
        with self.assertRaises(
            ValidationError
        ):
            parse_generate_payload(
                {"prompt":""},
                self.config,
            )
