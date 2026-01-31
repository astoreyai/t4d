"""Tests for memory projection layers (P3-04)."""

import torch
import pytest

from t4dm.qwen.projections import MemoryProjection


class TestProjections:
    def test_encode_shape(self):
        proj = MemoryProjection(qwen_dim=64, mem_dim=32)
        x = torch.randn(2, 10, 64)
        out = proj.encode(x)
        assert out.shape == (2, 10, 32)

    def test_decode_shape(self):
        proj = MemoryProjection(qwen_dim=64, mem_dim=32)
        x = torch.randn(2, 10, 32)
        out = proj.decode(x)
        assert out.shape == (2, 10, 64)

    def test_forward_returns_both(self):
        proj = MemoryProjection(qwen_dim=64, mem_dim=32)
        x = torch.randn(2, 10, 64)
        encoded, decoded = proj(x)
        assert encoded.shape == (2, 10, 32)
        assert decoded.shape == (2, 10, 64)

    def test_reconstruction_loss_decreases(self):
        """After training, reconstruction should be reasonable."""
        proj = MemoryProjection(qwen_dim=64, mem_dim=32)
        opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
        x = torch.randn(4, 5, 64)

        loss_start = None
        for _ in range(50):
            encoded, decoded = proj(x)
            loss = torch.nn.functional.mse_loss(decoded, x)
            if loss_start is None:
                loss_start = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert loss.item() < loss_start  # loss decreased
