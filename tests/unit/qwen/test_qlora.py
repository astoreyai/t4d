"""Tests for QLoRA adapter setup (P3-02)."""

import pytest
import torch.nn as nn

from ww.qwen.qlora import QLoRAConfig, apply_qlora, get_trainable_params


class TestQLoRA:
    def test_config_defaults(self):
        cfg = QLoRAConfig()
        assert cfg.r == 16
        assert cfg.target_modules == ["q_proj", "v_proj"]

    def test_apply_qlora_on_tiny_model(self, tiny_model):
        """Apply LoRA to a model that has linear layers (not q_proj/v_proj)."""
        # Freeze base
        for p in tiny_model.parameters():
            p.requires_grad = False

        # PEFT requires prepare_inputs_for_generation
        if not hasattr(tiny_model, "prepare_inputs_for_generation"):
            tiny_model.prepare_inputs_for_generation = lambda *a, **kw: {}

        # Apply with target_modules matching our tiny model
        cfg = QLoRAConfig(r=4, target_modules=["linear"])
        peft_model = apply_qlora(tiny_model, cfg)

        trainable = get_trainable_params(peft_model)
        assert trainable > 0  # LoRA params should be trainable

    def test_get_trainable_params(self, tiny_model):
        total = get_trainable_params(tiny_model)
        assert total > 0  # unfrozen model has trainable params
