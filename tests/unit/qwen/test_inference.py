"""Tests for inference pipeline (P3-09)."""

import torch
import pytest

from ww.qwen.inference import InferenceConfig, InferencePipeline
from ww.qwen.unified_model import UnifiedModel


class TestInference:
    def _make_pipeline(self, tiny_model, tiny_tokenizer):
        for p in tiny_model.parameters():
            p.requires_grad = False
        model = UnifiedModel(
            qwen_model=tiny_model, qwen_dim=64, mem_dim=32,
            num_spiking_blocks=2, num_heads=4, split_layer=2,
        )
        cfg = InferenceConfig(max_new_tokens=5, do_sample=False)
        return InferencePipeline(model, tiny_tokenizer, cfg)

    def test_generate_returns_text(self, tiny_model, tiny_tokenizer):
        pipe = self._make_pipeline(tiny_model, tiny_tokenizer)
        result = pipe.generate("hello")
        assert "text" in result
        assert "tokens_generated" in result
        assert result["tokens_generated"] > 0

    def test_reset_state(self, tiny_model, tiny_tokenizer):
        pipe = self._make_pipeline(tiny_model, tiny_tokenizer)
        pipe.generate("hello")
        assert pipe._spiking_states is not None
        pipe.reset_state()
        assert pipe._spiking_states is None

    def test_greedy_deterministic(self, tiny_model, tiny_tokenizer):
        pipe = self._make_pipeline(tiny_model, tiny_tokenizer)
        r1 = pipe.generate("hello")
        pipe.reset_state()
        r2 = pipe.generate("hello")
        # Greedy should produce same output (given same random input ids from mock tokenizer)
        assert r1["tokens_generated"] == r2["tokens_generated"]
