from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_g3_tpu_authority.sh"
MANIFEST = ROOT / "requirements-tpu-g3.txt"


class G3RunnerContractTests(unittest.TestCase):
    def test_manifest_is_exactly_frozen(self):
        self.assertEqual(
            MANIFEST.read_text(encoding="utf-8"),
            """keras==3.15.0
keras-hub==0.29.1
keras-nlp==0.29.1
jax==0.10.2
jaxlib==0.10.2
numpy==2.5.0
libtpu==0.0.17
""",
        )

    def test_hardware_gate_precedes_jax_and_runtime_restore(self):
        text = RUNNER.read_text(encoding="utf-8")
        hardware = text.index("/dev/accel")
        runtime = text.index("importlib.metadata")
        jax = text.index("\npython3 scripts/g3_tpu_authority.py")
        self.assertLess(hardware, runtime)
        self.assertLess(runtime, jax)

    def test_hardware_gate_accepts_vfio_tpu_mapping(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("vfio_devices", text)
        self.assertIn("/dev/vfio/[0-9]*", text)

    def test_runner_uses_exact_conditional_restore_command(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("--no-deps", text)
        self.assertIn("--upgrade", text)
        self.assertIn("-r requirements-tpu-g3.txt", text)
        self.assertIn("RUNTIME_RESTORE=SKIPPED_ALREADY_EXACT", text)

    def test_runner_contains_only_minimal_evidence_names(self):
        text = RUNNER.read_text(encoding="utf-8")
        for name in (
            "00-source-hashes.txt",
            "01-hardware-gate.txt",
            "02-runtime-versions.txt",
            "03-project-dependency-gate.txt",
            "04-cgroup-before.txt",
            "05-authority.stdout.log",
            "06-authority.stderr.log",
            "07-authority-result.json",
            "08-cgroup-after.txt",
            "09-final-adjudication.txt",
            "SHA256SUMS",
        ):
            self.assertIn(name, text)

    def test_runner_adjudicates_oom_from_structured_authority_result(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("scripts/g3_oom_adjudication.py", text)
        self.assertIn("--result-json", text)
        self.assertIn('GENERATION_STARTED) generation_started=', text)
        self.assertNotIn(
            'if [[ "$authority_exit" -eq 137 && "$oom_delta" -gt 0 ]]; then\n'
            '  final_result=G3_EXACT_LENGTH_GENERATION_HOST_OOM',
            text,
        )

    def test_runner_non_tpu_path_stops_before_authority(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("FINAL_RESULT=TPU_HARDWARE_NOT_READY", text)
        self.assertIn("write_final_and_archive TPU_HARDWARE_NOT_READY", text)
        self.assertIn("exit 1", text)


if __name__ == "__main__":
    unittest.main()
