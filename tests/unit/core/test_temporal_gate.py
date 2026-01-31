"""Tests for τ(t) temporal gate."""

import pytest
import torch

from t4dm.core.temporal_gate import TemporalGate


class TestTemporalGate:
    @pytest.fixture
    def gate(self):
        return TemporalGate()

    def test_output_range(self, gate):
        pe = torch.tensor(0.5)
        nov = torch.tensor(0.3)
        rew = torch.tensor(0.8)
        tau = gate(pe, nov, rew)
        assert 0.0 < tau.item() < 1.0

    def test_batch_input(self, gate):
        pe = torch.rand(4)
        nov = torch.rand(4)
        rew = torch.rand(4)
        tau = gate(pe, nov, rew)
        assert tau.shape == (4,)
        assert (tau > 0).all() and (tau < 1).all()

    def test_gradient_flows(self, gate):
        pe = torch.tensor(0.5, requires_grad=True)
        nov = torch.tensor(0.3, requires_grad=True)
        rew = torch.tensor(0.8, requires_grad=True)
        tau = gate(pe, nov, rew)
        tau.backward()
        assert pe.grad is not None
        assert gate.lambdas.grad is not None

    def test_high_signals_high_gate(self):
        gate = TemporalGate(lambda_epsilon=5.0, lambda_delta=5.0, lambda_r=5.0)
        tau = gate(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
        assert tau.item() > 0.99

    def test_zero_signals(self, gate):
        tau = gate(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        assert abs(tau.item() - 0.5) < 0.01  # sigmoid(0) = 0.5
