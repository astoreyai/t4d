"""Tests for unified model (P3-05)."""

import torch
import pytest

from t4dm.qwen.unified_model import UnifiedModel


class TestUnifiedModel:
    def _make_model(self, tiny_model):
        return UnifiedModel(
            qwen_model=tiny_model,
            qwen_dim=64,
            mem_dim=32,
            num_spiking_blocks=2,
            num_heads=4,
            split_layer=2,  # split at layer 2 of 4
        )

    def test_forward_produces_logits(self, tiny_model):
        model = self._make_model(tiny_model)
        input_ids = torch.randint(0, 100, (1, 5))
        output = model(input_ids)
        assert "logits" in output
        assert output["logits"].shape == (1, 5, 100)  # B, S, vocab

    def test_spiking_states_returned(self, tiny_model):
        model = self._make_model(tiny_model)
        input_ids = torch.randint(0, 100, (1, 5))
        output = model(input_ids)
        assert "spiking_states" in output
        assert "spiking_metrics" in output

    def test_encoded_memory_shape(self, tiny_model):
        model = self._make_model(tiny_model)
        input_ids = torch.randint(0, 100, (1, 5))
        output = model(input_ids)
        assert output["encoded_memory"].shape == (1, 5, 32)

    def test_trainable_param_count(self, tiny_model):
        for p in tiny_model.parameters():
            p.requires_grad = False
        model = self._make_model(tiny_model)
        counts = model.trainable_param_count()
        assert counts["spiking"] > 0
        assert counts["projection"] > 0
        assert counts["gate"] > 0
        assert counts["total"] > 0
