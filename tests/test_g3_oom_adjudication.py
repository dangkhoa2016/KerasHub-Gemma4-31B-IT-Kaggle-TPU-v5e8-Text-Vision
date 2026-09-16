from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ADJUDICATOR = ROOT / "scripts/g3_oom_adjudication.py"


def run_adjudication(result: dict[str, object] | None) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as td:
        result_path = Path(td) / "07-authority-result.json"
        if result is not None:
            result_path.write_text(json.dumps(result), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(ADJUDICATOR),
                "--authority-exit",
                "137",
                "--oom-delta",
                "1",
                "--result-json",
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr or completed.stdout)
        return dict(
            line.split("=", 1)
            for line in completed.stdout.splitlines()
            if "=" in line
        )


class G3OomAdjudicationTests(unittest.TestCase):
    def test_generation_oom_requires_explicit_started_true(self):
        result = run_adjudication({"GENERATION_STARTED": True})

        self.assertEqual(
            result["FINAL_RESULT"],
            "G3_EXACT_LENGTH_GENERATION_HOST_OOM",
        )

    def test_pre_generation_oom_is_classified_separately(self):
        result = run_adjudication({"GENERATION_STARTED": False})

        self.assertEqual(
            result["FINAL_RESULT"],
            "AUTHORITY_PRE_GENERATION_HOST_OOM",
        )
        self.assertEqual(result["G3_TEXT_GENERATION"], "NOT_EVALUATED")
        self.assertEqual(result["G3_STATUS"], "OPEN")
        self.assertEqual(result["G4_ENTRY_ELIGIBLE"], "false")
        self.assertEqual(result["G4_STARTED"], "false")

    def test_unknown_oom_stage_is_not_generation_oom(self):
        result = run_adjudication(None)

        self.assertEqual(
            result["FINAL_RESULT"],
            "AUTHORITY_HOST_OOM_STAGE_UNKNOWN",
        )
        self.assertEqual(result["G3_TEXT_GENERATION"], "NOT_EVALUATED")

    def test_generation_oom_never_occurs_without_boolean_true_marker(self):
        for marker in (None, False, 0, "true"):
            with self.subTest(marker=marker):
                result = run_adjudication(
                    {} if marker is None else {"GENERATION_STARTED": marker}
                )

                self.assertNotEqual(
                    result["FINAL_RESULT"],
                    "G3_EXACT_LENGTH_GENERATION_HOST_OOM",
                )


if __name__ == "__main__":
    unittest.main()
