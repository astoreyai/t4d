"""Tests for thalamic gate."""

import pytest
import torch

from ww.spiking.thalamic_gate import ThalamicGate


class TestThalamicGate:
    @pytest.fixture
    def gate(self):
        return ThalamicGate(input_dim=16)

    def test_multiplicative_gating(self, gate):
        x = torch.ones(2, 16)
        out = gate(x)
        # Output should be elementwise gated (all values between 0 and max)
        assert out.shape == (2, 16)

    def test_ach_modulation(self, gate):
        x = torch.ones(2, 16)
        out_low = gate(x, ach_level=0.0)
        out_high = gate(x, ach_level=1.0)
        # Higher ACh = stronger gate signal
        assert out_high.abs().sum() > out_low.abs().sum()

    def test_context_input(self):
        gate = ThalamicGate(input_dim=16, context_dim=8)
        x = torch.ones(2, 16)
        ctx = torch.randn(2, 8)
        out = gate(x, context=ctx)
        assert out.shape == (2, 16)

    def test_zero_input(self, gate):
        x = torch.zeros(2, 16)
        out = gate(x)
        assert torch.allclose(out, torch.zeros(2, 16))
