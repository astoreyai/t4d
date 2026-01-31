"""Tests for spike attention."""

import pytest
import torch

from ww.spiking.spike_attention import SpikeAttention


class TestSpikeAttention:
    @pytest.fixture
    def attn(self):
        return SpikeAttention(dim=32, num_heads=4)

    def test_output_shape(self, attn):
        x = torch.randn(2, 10, 32)
        out, weights = attn(x)
        assert out.shape == (2, 10, 32)
        assert weights.shape == (4,)

    def test_linear_complexity_scales(self, attn):
        # Just verify it works with different sequence lengths
        for seq_len in [5, 50, 100]:
            x = torch.randn(1, seq_len, 32)
            out, _ = attn(x)
            assert out.shape == (1, seq_len, 32)

    def test_stdp_weights_learnable(self, attn):
        x = torch.randn(2, 10, 32)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()
        assert attn.stdp_weights.grad is not None

    def test_dim_divisibility(self):
        with pytest.raises(AssertionError):
            SpikeAttention(dim=33, num_heads=4)
