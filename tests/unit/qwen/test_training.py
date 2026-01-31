"""Tests for Phase 1 training loop (P3-06)."""

import torch
import pytest

from t4dm.qwen.training import Phase1Trainer, TrainingConfig
from t4dm.qwen.unified_model import UnifiedModel


class TestTraining:
    def _make_model(self, tiny_model):
        for p in tiny_model.parameters():
            p.requires_grad = False
        return UnifiedModel(
            qwen_model=tiny_model, qwen_dim=64, mem_dim=32,
            num_spiking_blocks=2, num_heads=4, split_layer=2,
        )

    def test_train_step_returns_loss(self, tiny_model):
        model = self._make_model(tiny_model)
        cfg = TrainingConfig(gradient_accumulation=1)
        trainer = Phase1Trainer(model, cfg)

        input_ids = torch.randint(2, 100, (1, 8))
        labels = input_ids.clone()

        result = trainer.train_step(input_ids, labels)
        assert "loss" in result
        assert "ce_loss" in result
        assert result["loss"] > 0

    def test_loss_decreases(self, tiny_model):
        model = self._make_model(tiny_model)
        cfg = TrainingConfig(lr=1e-3, gradient_accumulation=1)
        trainer = Phase1Trainer(model, cfg)

        input_ids = torch.randint(2, 100, (1, 8))
        labels = input_ids.clone()

        losses = []
        for _ in range(10):
            result = trainer.train_step(input_ids, labels)
            losses.append(result["loss"])

        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0]

    def test_step_counter(self, tiny_model):
        model = self._make_model(tiny_model)
        trainer = Phase1Trainer(model, TrainingConfig(gradient_accumulation=1))
        input_ids = torch.randint(2, 100, (1, 5))
        trainer.train_step(input_ids, input_ids)
        assert trainer.step == 1
