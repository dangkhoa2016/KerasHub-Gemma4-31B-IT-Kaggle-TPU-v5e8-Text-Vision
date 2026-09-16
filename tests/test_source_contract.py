from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:
    sys.path.insert(0,str(ROOT/"src"))

import unittest

class T(unittest.TestCase):
    def test_identity(self):
        corpus="\n".join(
            p.read_text(
                encoding="utf-8",
                errors="ignore",
            )
            for p in (ROOT/"src").rglob("*.py")
        )
        self.assertIn(
            "gemma4_instruct_31b",
            corpus,
        )
        self.assertNotIn(
            "TranslateGemmaTPUEngine",
            corpus,
        )
        self.assertNotIn(
            "Gemma3CausalLM",
            corpus,
        )

    def test_g2_does_not_generate(self):
        text=(
            ROOT/"scripts/g0_g2_strict_load.py"
        ).read_text()
        self.assertNotIn(
            ".generate(",
            text,
        )

    def test_r1_uses_native_preset_loader(self):
        text=(
            ROOT/"src/gemma4_server/tpu/engine.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"keras_hub_native_preset_loader"',
            text,
        )
        self.assertIn(
            "load_weights=True",
            text,
        )
        self.assertIn(
            "dtype=self.dtype",
            text,
        )
        self.assertNotIn(
            "self.model.load_weights(",
            text,
        )
        self.assertIn(
            '"checkpoint_target": "task.backbone"',
            text,
        )

    def test_r3_wraps_native_checkpoint_load_only(self):
        text = (
            ROOT / "src/gemma4_server/tpu/engine.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "sharded_checkpoint_assignment(",
            text,
        )
        self.assertIn(
            "event_callback=self.r3_event_callback",
            text,
        )
        self.assertIn(
            "with self.distribution.scope(),",
            text,
        )

    def test_authority_generation_uses_exact_native_length(self):
        text = (ROOT / "src/gemma4_server/tpu/engine.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def generate_text_authority", text)
        self.assertIn("plan_authority_generation", text)
        self.assertIn('"EXACT_LENGTH_AUTHORITY_PATH"', text)
        self.assertIn("max_length=plan.max_length", text)
        self.assertIn("strip_prompt=True", text)

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
