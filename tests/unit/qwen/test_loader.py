"""Tests for Qwen loader (P3-01) — uses tiny mock, no real model download."""

import pytest

from ww.qwen.loader import QwenConfig, get_hidden_dim, get_num_layers


class TestLoader:
    def test_qwen_config_defaults(self):
        cfg = QwenConfig()
        assert cfg.use_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"

    def test_get_hidden_dim(self, tiny_model):
        assert get_hidden_dim(tiny_model) == 64

    def test_get_num_layers(self, tiny_model):
        assert get_num_layers(tiny_model) == 4

    def test_model_frozen(self, tiny_model):
        """Verify we can freeze all params."""
        for p in tiny_model.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
        assert trainable == 0
