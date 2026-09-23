from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class RuntimeSourceContractTests(unittest.TestCase):
    def test_project_targets_gemma4_31b_only(self):
        corpus = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in (ROOT / "src").rglob("*.py")
        )
        self.assertIn("gemma4_instruct_31b", corpus)
        self.assertNotIn("TranslateGemmaTPUEngine", corpus)
        self.assertNotIn("Gemma3CausalLM", corpus)

    def test_engine_uses_native_preset_loader(self):
        text = (ROOT / "src/gemma4_server/tpu/engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"keras_hub_native_preset_loader"', text)
        self.assertIn("load_weights=True", text)
        self.assertIn("dtype=self.dtype", text)
        self.assertNotIn("self.model.load_weights(", text)
        self.assertIn('"checkpoint_target": "task.backbone"', text)

    def test_checkpoint_assignment_wraps_native_load(self):
        text = (ROOT / "src/gemma4_server/tpu/engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("sharded_checkpoint_assignment(", text)
        self.assertIn("with self.distribution.scope(),", text)

    def test_post_load_cleanup_precedes_ready_phase(self):
        text = (ROOT / "src/gemma4_server/tpu/engine.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            text.index("self._post_load_host_cleanup()"),
            text.index('self._phase("ready")'),
        )
        self.assertIn("gc.collect()", text)
        self.assertIn("malloc_trim", text)

    def test_engine_records_tpu_backend_and_sharding_summary(self):
        text = (ROOT / "src/gemma4_server/tpu/engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"jax_default_backend": jax.default_backend()', text)
        self.assertIn('"sharded_parameter_percent_by_bytes"', text)


if __name__ == "__main__":
    unittest.main()
