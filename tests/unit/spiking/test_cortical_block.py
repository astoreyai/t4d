"""Tests for cortical block."""

import pytest
import torch

from ww.spiking.cortical_block import CorticalBlock


class TestCorticalBlock:
    @pytest.fixture
    def block(self):
        return CorticalBlock(dim=32, num_heads=4)

    def test_forward_shape(self, block):
        x = torch.randn(2, 10, 32)
        out, state, metrics = block(x)
        assert out.shape == (2, 10, 32)

    def test_state_tracking(self, block):
        x = torch.randn(1, 5, 32)
        _, state, _ = block(x)
        assert "u2" in state
        assert "rwkv" in state
        assert "u6" in state

    def test_metrics_returned(self, block):
        x = torch.randn(1, 5, 32)
        _, _, metrics = block(x)
        assert "pe" in metrics
        assert "goodness" in metrics
        assert "attn" in metrics

    def test_residual_connection(self, block):
        # Output should be normalized (LayerNorm)
        x = torch.randn(2, 10, 32)
        out, _, _ = block(x)
        # LayerNorm ensures roughly zero mean, unit variance per feature
        assert out.mean().abs() < 1.0

    def test_with_context(self):
        block = CorticalBlock(dim=32, context_dim=16, num_heads=4)
        x = torch.randn(2, 10, 32)
        ctx = torch.randn(2, 10, 16)
        out, _, _ = block(x, context=ctx)
        assert out.shape == (2, 10, 32)

    def test_state_reuse(self, block):
        x1 = torch.randn(1, 5, 32)
        x2 = torch.randn(1, 5, 32)
        _, state1, _ = block(x1)
        out2a, _, _ = block(x2, state=state1)
        out2b, _, _ = block(x2)
        assert not torch.allclose(out2a, out2b, atol=1e-5)
