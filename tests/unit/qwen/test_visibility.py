"""Tests for activation visibility hooks (P3-10)."""

import torch
import pytest

from t4dm.qwen.visibility import ActivationCollector


class TestVisibility:
    def test_attach_qwen_and_capture(self, tiny_model):
        collector = ActivationCollector()
        collector.attach_qwen(tiny_model, layers=[0, 1])

        input_ids = torch.randint(0, 100, (1, 5))
        tiny_model(input_ids)

        acts = collector.activations
        assert "qwen_layer_0" in acts
        assert "qwen_layer_1" in acts
        assert acts["qwen_layer_0"].shape == (1, 5, 64)

        collector.detach_all()

    def test_summary(self, tiny_model):
        collector = ActivationCollector()
        collector.attach_qwen(tiny_model, layers=[0])
        tiny_model(torch.randint(0, 100, (1, 3)))
        summary = collector.summary()
        assert "qwen_layer_0" in summary
        assert summary["qwen_layer_0"] == (1, 3, 64)
        collector.detach_all()

    def test_clear(self, tiny_model):
        collector = ActivationCollector()
        collector.attach_qwen(tiny_model, layers=[0])
        tiny_model(torch.randint(0, 100, (1, 3)))
        assert len(collector.activations) > 0
        collector.clear()
        assert len(collector.activations) == 0
        collector.detach_all()

    def test_detach_stops_collection(self, tiny_model):
        collector = ActivationCollector()
        collector.attach_qwen(tiny_model, layers=[0])
        collector.detach_all()
        collector.clear()
        tiny_model(torch.randint(0, 100, (1, 3)))
        assert len(collector.activations) == 0
