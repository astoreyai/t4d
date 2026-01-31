"""Tests for Phase 2 local learning (P3-07)."""

import torch
import pytest

from t4dm.qwen.local_learning import LocalLearner, LocalLearningConfig
from t4dm.qwen.unified_model import UnifiedModel


class TestLocalLearning:
    def _make_model(self, tiny_model):
        for p in tiny_model.parameters():
            p.requires_grad = False
        return UnifiedModel(
            qwen_model=tiny_model, qwen_dim=64, mem_dim=32,
            num_spiking_blocks=2, num_heads=4, split_layer=2,
        )

    def test_ach_gate(self, tiny_model):
        model = self._make_model(tiny_model)
        learner = LocalLearner(model, LocalLearningConfig(freeze_qlora=False))
        result = learner.update([], ach_level=0.1)
        assert result["gated"] is True

    def test_update_with_metrics(self, tiny_model):
        model = self._make_model(tiny_model)
        learner = LocalLearner(model, LocalLearningConfig(freeze_qlora=False))
        metrics = [{"attn": torch.randn(4, 4)} for _ in range(2)]
        result = learner.update(metrics, da_level=1.0, ach_level=0.8)
        assert result["gated"] is False

    def test_reset_eligibility(self, tiny_model):
        model = self._make_model(tiny_model)
        learner = LocalLearner(model, LocalLearningConfig(freeze_qlora=False))
        metrics = [{"attn": torch.randn(4, 4)}]
        learner.update(metrics)
        assert len(learner._eligibility_traces) > 0
        learner.reset_eligibility()
        assert len(learner._eligibility_traces) == 0
