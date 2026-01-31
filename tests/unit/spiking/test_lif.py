"""Tests for LIF neuron with surrogate gradient."""

import pytest
import torch

from ww.spiking.lif import LIFNeuron


class TestLIFNeuron:
    @pytest.fixture
    def lif(self):
        return LIFNeuron(size=32, alpha=0.9, v_thresh=1.0)

    def test_spike_generation(self, lif):
        # Large input should produce spikes
        x = torch.ones(1, 32) * 2.0
        spikes, u = lif(x)
        assert spikes.shape == (1, 32)
        assert (spikes == 1.0).any()

    def test_subthreshold_no_spike(self, lif):
        x = torch.ones(1, 32) * 0.1
        spikes, u = lif(x)
        assert (spikes == 0.0).all()

    def test_soft_reset(self, lif):
        # After spiking, membrane should be reduced but not zeroed
        x = torch.ones(1, 32) * 1.5
        spikes, u = lif(x)
        # Where spiked: u = 1.5 - 1.0*1.0 = 0.5 (soft reset)
        spiked_u = u[spikes == 1.0]
        if spiked_u.numel() > 0:
            assert (spiked_u > 0).all()
            assert (spiked_u < 1.0).all()

    def test_leak(self, lif):
        # Without input, membrane should decay
        x = torch.zeros(1, 32)
        _, u1 = lif(x, torch.ones(1, 32) * 0.5)
        assert torch.allclose(u1, torch.ones(1, 32) * 0.45, atol=0.01)

    def test_state_persistence(self, lif):
        x1 = torch.ones(1, 32) * 0.3
        _, u1 = lif(x1)
        x2 = torch.ones(1, 32) * 0.3
        _, u2 = lif(x2, u1)
        # u2 should be higher than u1 due to accumulation
        assert (u2 > u1).all()

    def test_gradient_flows_through_ste(self, lif):
        x = torch.randn(1, 32, requires_grad=True)
        spikes, u = lif(x)
        loss = spikes.sum()
        loss.backward()
        assert x.grad is not None
        # Gradient should be nonzero (surrogate gradient)
        assert x.grad.abs().sum() > 0

    def test_batch_processing(self, lif):
        x = torch.randn(8, 32)
        spikes, u = lif(x)
        assert spikes.shape == (8, 32)
        assert u.shape == (8, 32)
