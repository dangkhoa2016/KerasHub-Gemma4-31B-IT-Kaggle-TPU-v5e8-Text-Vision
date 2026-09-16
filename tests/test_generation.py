from __future__ import annotations
import os
import sys
from unittest.mock import patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:
    sys.path.insert(0,str(ROOT/"src"))

import unittest
from unittest.mock import patch
import gemma4_server.core.config as config_module
from gemma4_server.tpu.generation import (
    chat_prompt,
    vision_prompt,
    plan_generation,
    plan_authority_generation,
)
from gemma4_server.tpu.engine import Gemma4TPUEngine

class T(unittest.TestCase):
    def production_config(self, **environment):
        with patch.dict(os.environ, environment, clear=True), \
             patch.object(
                 config_module,
                 "resolve_model_path",
                 return_value=Path("/tmp/model"),
             ), \
             patch.object(
                 config_module,
                 "load_or_create_secret",
                 return_value="test-secret",
             ):
            return config_module.Config.from_env()

    def test_chat_tokens(self):
        p=chat_prompt("Hello","Be concise")
        self.assertIn("<|turn>system",p)
        self.assertIn("<|turn>user",p)
        self.assertTrue(
            p.endswith("<|turn>model\n")
        )

    def test_vision_token(self):
        self.assertIn(
            "<|image|>",
            vision_prompt("Describe"),
        )

    def test_bucket(self):
        plan=plan_generation(
            300,100,(512,768,1024),1024
        )
        self.assertEqual(
            plan.max_length,512
        )

    def test_production_default_small_request_uses_small_bucket(self):
        config = self.production_config()

        plan = plan_generation(
            10,
            1,
            config.generation_length_buckets,
            config.max_generation_length,
        )

        self.assertEqual(plan.max_length, 16)
        self.assertTrue(plan.bucketed)
        self.assertEqual(plan.mode, "CURRENT_PRODUCTION_BUCKET_POLICY")

    def test_production_buckets_preserve_larger_boundary_mapping(self):
        config = self.production_config()
        expected = {
            11: 16,
            16: 16,
            17: 512,
            512: 512,
            513: 768,
            768: 768,
            769: 1024,
            1024: 1024,
            1025: 1536,
            1536: 1536,
            1537: 2048,
        }

        for required, selected in expected.items():
            plan = plan_generation(
                required - 1,
                1,
                config.generation_length_buckets,
                config.max_generation_length,
            )
            with self.subTest(required=required):
                self.assertEqual(plan.max_length, selected)

    def test_production_config_accepts_small_bucket_and_normalizes_order(self):
        config = self.production_config(
            GENERATION_LENGTH_BUCKETS="512,16,512,768",
            MAX_GENERATION_LENGTH="2048",
        )

        self.assertEqual(config.generation_length_buckets, (16, 512, 768))

    def test_production_planner_never_uses_authority_mode(self):
        config = self.production_config()

        plan = plan_generation(
            10,
            1,
            config.generation_length_buckets,
            config.max_generation_length,
        )

        self.assertNotEqual(plan.mode, "EXACT_LENGTH_AUTHORITY_PATH")
        self.assertFalse(hasattr(plan, "authority_generation_path"))

    def test_engine_normalizes_constructor_buckets_without_model_load(self):
        engine = Gemma4TPUEngine(
            "/tmp/model",
            "bfloat16",
            object(),
            generation_length_buckets=(512, 16, 512, 768),
            max_generation_length=2048,
        )

        self.assertEqual(engine.buckets, (16, 512, 768))

    def test_limit(self):
        with self.assertRaises(ValueError):
            plan_generation(
                1000,100,(1024,),1024
            )

    def test_production_maximum_rejection_is_preserved(self):
        config = self.production_config()

        with self.assertRaises(ValueError):
            plan_generation(
                config.max_generation_length,
                1,
                config.generation_length_buckets,
                config.max_generation_length,
            )

    def test_authority_plan_is_exact_and_does_not_use_production_buckets(self):
        plan = plan_authority_generation(10, 1, 2048)

        self.assertEqual(plan.max_length, 11)
        self.assertEqual(plan.max_new_tokens, 1)
        self.assertFalse(plan.bucketed)
        self.assertEqual(plan.mode, "EXACT_LENGTH_AUTHORITY_PATH")

    def test_authority_plan_rejects_length_over_maximum(self):
        with self.assertRaises(ValueError):
            plan_authority_generation(2048, 1, 2048)

    def test_generate_exposes_compile_evidence_without_changing_native_call(self):
        class FakePreprocessor:
            pass

        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return object()

        class FakeCapture:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def snapshot(self):
                return {
                    "source": "jax_logging",
                    "available": True,
                    "events": [],
                    "compile_event_count": 0,
                    "compile_seconds": 0.0,
                    "persistent_cache_hits": 0,
                    "persistent_cache_misses": 0,
                    "status": "direct",
                    "error": None,
                }

        engine = Gemma4TPUEngine(
            "/tmp/model",
            "bfloat16",
            object(),
            generation_length_buckets=(16, 512),
            max_generation_length=512,
        )
        engine.model = FakeModel()
        engine.preprocessor = FakePreprocessor()

        with patch.object(
            engine,
            "_preprocess_prompt_tokens",
            return_value=10,
        ), patch(
            "gemma4_server.tpu.engine.scalar_text",
            return_value="done",
        ), patch(
            "gemma4_server.tpu.observability.CompilationEvidenceCapture",
            return_value=FakeCapture(),
        ):
            result, metrics = engine._generate("hello", 1)

        self.assertEqual(result, "done")
        self.assertEqual(metrics["compile_cache_evidence"]["status"], "direct")
        self.assertEqual(
            engine.model.calls,
            [
                (("hello",), {
                    "max_length": 16,
                    "strip_prompt": True,
                })
            ],
        )
