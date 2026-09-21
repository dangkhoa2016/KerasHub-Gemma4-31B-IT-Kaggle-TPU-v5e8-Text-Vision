from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "scripts" / "_common.sh"


class LoadEnvTests(unittest.TestCase):
    def run_load_env(self, dotenv: str, overrides: dict[str, str]) -> str:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            scripts = project / "scripts"
            scripts.mkdir()
            shutil.copy2(COMMON, scripts / "_common.sh")
            (project / ".env").write_text(dotenv, encoding="utf-8")

            environment = os.environ.copy()
            environment.pop("API_KEY", None)
            environment.pop("RESTART_SECRET", None)
            environment.update(overrides)
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        "source scripts/_common.sh; "
                        "load_env; "
                        "printf '%s\\n' \"$API_KEY|$RESTART_SECRET|$HOST\""
                    ),
                ],
                cwd=project,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()

    def test_nonempty_process_secrets_override_blank_dotenv_values(self):
        result = self.run_load_env(
            "API_KEY=\nRESTART_SECRET=\nHOST=from-dotenv\n",
            {
                "API_KEY": "process-api",
                "RESTART_SECRET": "process-restart",
            },
        )

        self.assertEqual(result, "process-api|process-restart|from-dotenv")

    def test_dotenv_secrets_are_loaded_when_process_values_are_absent(self):
        result = self.run_load_env(
            "API_KEY=dotenv-api\nRESTART_SECRET=dotenv-restart\nHOST=from-dotenv\n",
            {},
        )

        self.assertEqual(result, "dotenv-api|dotenv-restart|from-dotenv")


if __name__ == "__main__":
    unittest.main()
