from __future__ import annotations

import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ensure_libtpu.py"

class EnsureLibtpuTests(unittest.TestCase):
    def test_mismatched_existing_version_is_replaced_with_expected_pin(self):
        env = {"EXPECTED_LIBTPU_VERSION": "0.0.49", "INSTALL_LIBTPU_IF_MISSING": "auto"}
        with patch.dict(os.environ, env, clear=False), \
             patch("importlib.metadata.version", return_value="0.0.17"), \
             patch("subprocess.run") as run:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertIn("libtpu==0.0.49", command)
        self.assertIn("--no-deps", command)

if __name__ == "__main__":
    unittest.main()
