from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "scripts/g3_tpu_authority.py"
ENGINE = ROOT / "src/gemma4_server/tpu/engine.py"


class G3AuthoritySourceTests(unittest.TestCase):
    def test_authority_has_required_hard_caps_and_one_process_path(self):
        text = AUTHORITY.read_text(encoding="utf-8")
        self.assertIn("TPU_INIT_TARGET_SECONDS = 120", text)
        self.assertIn("TPU_INIT_HARD_MAX_SECONDS = 180", text)
        self.assertIn("STRICT_LOAD_HARD_MAX_SECONDS = 2304", text)
        self.assertIn("os._exit(124)", text)
        self.assertEqual(text.count("import jax"), 1)

    def test_authority_preserves_exact_generation_contract(self):
        text = AUTHORITY.read_text(encoding="utf-8")
        self.assertIn('prompt = "Hello"', text)
        self.assertIn("prompt_token_count = 10", text)
        self.assertIn("max_new_tokens = 1", text)
        self.assertIn("authority_max_length = 11", text)
        self.assertIn("generation_call_count = 1", text)
        self.assertIn("generation_call_count >= 1", text)
        self.assertIn("strip_prompt=True", ENGINE.read_text(encoding="utf-8"))

    def test_authority_requires_candidate_a_before_generation(self):
        text = AUTHORITY.read_text(encoding="utf-8")
        self.assertLess(text.index("verify_candidate_a"), text.index("generate_text_authority"))
        self.assertIn("CANDIDATE_A_SHARDING_VERIFIED", text)
        self.assertIn("G4_ENTRY_ELIGIBLE", text)
        self.assertIn('"G4_STARTED": False', text)

    def test_authority_records_load_markers_and_cgroup_boundaries(self):
        text = AUTHORITY.read_text(encoding="utf-8")
        self.assertIn('result["MODEL_LOAD_STARTED"] = True', text)
        self.assertIn('result["MODEL_LOAD_RETURNED"] = True', text)
        self.assertIn("memory.events", text)
        self.assertIn("HOST_MEMORY_BEFORE_CLEANUP", text)
        self.assertIn("HOST_MEMORY_AFTER_GC", text)
        self.assertIn("HOST_MEMORY_AFTER_MALLOC_TRIM", text)

    def test_r3_forbidden_full_buffer_operations_remain_absent(self):
        text = (ROOT / "src/gemma4_server/tpu/sharded_checkpoint.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_direct_assign", text)
        self.assertNotIn("jnp.asarray", text)
        self.assertNotIn("jax.device_put", text)


if __name__ == "__main__":
    unittest.main()
