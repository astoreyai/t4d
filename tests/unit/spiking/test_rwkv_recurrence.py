"""Tests for RWKV recurrence."""

import pytest
import torch

from ww.spiking.rwkv_recurrence import RWKVRecurrence


class TestRWKVRecurrence:
    @pytest.fixture
    def rwkv(self):
        return RWKVRecurrence(dim=32)

    def test_output_shape(self, rwkv):
        x = torch.randn(2, 10, 32)
        out, state = rwkv(x)
        assert out.shape == (2, 10, 32)

    def test_state_persistence(self, rwkv):
        x1 = torch.randn(1, 5, 32)
        x2 = torch.randn(1, 5, 32)
        _, state1 = rwkv(x1)
        out_with_state, _ = rwkv(x2, state1)
        out_no_state, _ = rwkv(x2)
        # Outputs should differ when state is provided
        assert not torch.allclose(out_with_state, out_no_state, atol=1e-5)

    def test_constant_state_size(self, rwkv):
        # State size should not grow with sequence length
        x_short = torch.randn(1, 5, 32)
        x_long = torch.randn(1, 50, 32)
        _, state_short = rwkv(x_short)
        _, state_long = rwkv(x_long)
        # Both states should have same structure
        assert state_short["tm"]["wkv_num"].shape == state_long["tm"]["wkv_num"].shape

    def test_gradient_flow(self, rwkv):
        x = torch.randn(1, 5, 32, requires_grad=True)
        out, _ = rwkv(x)
        out.sum().backward()
        assert x.grad is not None
