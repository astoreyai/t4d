"""Tests for cortical stack."""

import pytest
import torch

from ww.spiking.cortical_stack import CorticalStack


class TestCorticalStack:
    @pytest.fixture
    def stack(self):
        return CorticalStack(dim=32, num_blocks=3, num_heads=4)

    def test_forward_shape(self, stack):
        x = torch.randn(2, 10, 32)
        out, states, metrics = stack(x)
        assert out.shape == (2, 10, 32)
        assert len(states) == 3
        assert len(metrics) == 3

    def test_per_block_state(self, stack):
        x = torch.randn(1, 5, 32)
        _, states, _ = stack(x)
        for s in states:
            assert "u2" in s
            assert "rwkv" in s

    def test_state_reuse(self, stack):
        x1 = torch.randn(1, 5, 32)
        x2 = torch.randn(1, 5, 32)
        _, states1, _ = stack(x1)
        out_a, _, _ = stack(x2, states=states1)
        out_b, _, _ = stack(x2)
        assert not torch.allclose(out_a, out_b, atol=1e-5)

    def test_shared_context(self):
        stack = CorticalStack(dim=32, num_blocks=2, context_dim=16, num_heads=4)
        x = torch.randn(1, 5, 32)
        ctx = torch.randn(1, 5, 16)
        out, _, _ = stack(x, context=ctx)
        assert out.shape == (1, 5, 32)
