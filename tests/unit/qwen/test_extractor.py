"""Tests for hidden state extractor (P3-03)."""

import torch
import pytest

from t4dm.qwen.extractor import HiddenStateExtractor


class TestExtractor:
    def test_attach_and_capture(self, tiny_model):
        ext = HiddenStateExtractor(tiny_model, tap_layer=1)
        ext.attach()

        input_ids = torch.randint(0, 100, (1, 5))
        tiny_model(input_ids)

        assert ext.hidden_states is not None
        assert ext.hidden_states.shape == (1, 5, 64)  # B, S, D

        ext.detach()

    def test_detach(self, tiny_model):
        ext = HiddenStateExtractor(tiny_model, tap_layer=0)
        ext.attach()
        ext.detach()
        # Should not capture after detach
        ext._hidden = None
        input_ids = torch.randint(0, 100, (1, 5))
        tiny_model(input_ids)
        assert ext.hidden_states is None

    def test_tap_layer_property(self, tiny_model):
        ext = HiddenStateExtractor(tiny_model, tap_layer=2)
        assert ext.tap_layer == 2
