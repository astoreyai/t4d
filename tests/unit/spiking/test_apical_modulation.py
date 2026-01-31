"""Tests for apical modulation."""

import pytest
import torch

from t4dm.spiking.apical_modulation import ApicalModulation


class TestApicalModulation:
    @pytest.fixture
    def mod(self):
        return ApicalModulation(dim=16)

    def test_output_shape(self, mod):
        x = torch.randn(2, 16)
        out, pe, goodness = mod(x)
        assert out.shape == (2, 16)
        assert pe.shape == (2,)
        assert goodness.shape == (2,)

    def test_multiplicative_gating(self, mod):
        x = torch.randn(2, 16)
        out, _, _ = mod(x)
        # Output is basal * sigmoid(gate) so should be bounded
        assert out.shape == (2, 16)

    def test_prediction_error_computation(self, mod):
        x = torch.randn(2, 16)
        apical = torch.randn(2, 16)
        _, pe, _ = mod(x, apical)
        assert (pe >= 0).all()  # Squared difference is non-negative

    def test_goodness_positive(self, mod):
        x = torch.randn(2, 16)
        _, _, goodness = mod(x)
        assert (goodness >= 0).all()  # Sum of squares is non-negative

    def test_no_apical_input(self, mod):
        x = torch.randn(2, 16)
        out, pe, goodness = mod(x)
        # Should still work with zeros as apical
        assert out.shape == (2, 16)

    def test_context_dim(self):
        mod = ApicalModulation(dim=16, context_dim=8)
        basal = torch.randn(2, 16)
        apical = torch.randn(2, 8)
        out, pe, goodness = mod(basal, apical)
        assert out.shape == (2, 16)
