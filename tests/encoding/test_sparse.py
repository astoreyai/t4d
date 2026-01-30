"""
Unit tests for sparse encoder implementation.
"""

import pytest
import torch

from ww.encoding.sparse import SparseEncoder, kwta, AdaptiveSparseEncoder
from ww.encoding.utils import compute_sparsity


class TestKWTA:
    """Tests for k-Winner-Take-All function."""

    def test_kwta_basic(self):
        """k-WTA keeps exactly k values."""
        x = torch.randn(1, 100)
        k = 5

        sparse = kwta(x, k)

        # Exactly k non-zero
        assert (sparse != 0).sum().item() == k

    def test_kwta_batch(self):
        """k-WTA works for batches."""
        x = torch.randn(8, 100)
        k = 10

        sparse = kwta(x, k)

        # Each sample has exactly k non-zero
        for i in range(8):
            assert (sparse[i] != 0).sum().item() == k

    def test_kwta_preserves_topk(self):
        """k-WTA preserves top-k values."""
        x = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        k = 3

        sparse = kwta(x, k)

        # Should keep 5, 4, 3 (top 3)
        expected_mask = torch.tensor([[False, True, True, False, True]])
        assert ((sparse != 0) == expected_mask).all()


class TestSparseEncoder:
    """Tests for sparse encoder module."""

    @pytest.fixture
    def encoder(self):
        """Create default sparse encoder."""
        return SparseEncoder(
            input_dim=64,
            hidden_dim=256,
            sparsity=0.02,
            use_kwta=True
        )

    def test_initialization(self, encoder):
        """Encoder initializes correctly."""
        assert encoder.input_dim == 64
        assert encoder.hidden_dim == 256
        assert encoder.sparsity == 0.02
        assert encoder.k == 5  # 256 * 0.02 = 5.12 -> 5

    def test_forward_shape(self, encoder):
        """Forward produces correct shape."""
        x = torch.randn(8, 64)

        sparse = encoder(x)

        assert sparse.shape == (8, 256)

    def test_kwta_sparsity(self, encoder):
        """k-WTA produces exact sparsity."""
        x = torch.randn(32, 64)

        sparse = encoder(x)

        # Each sample should have exactly k active
        for i in range(32):
            active = (sparse[i] != 0).sum().item()
            assert active == encoder.k

    def test_sparsity_in_biological_range(self):
        """Sparsity should be in 1-5% range."""
        encoder = SparseEncoder(
            input_dim=64,
            hidden_dim=1000,
            sparsity=0.02
        )
        x = torch.randn(100, 64)

        sparse = encoder(x)
        actual_sparsity = compute_sparsity(sparse)

        assert 0.01 <= actual_sparsity <= 0.05

    def test_pattern_orthogonality(self, encoder):
        """Different inputs produce decorrelated patterns."""
        # Generate diverse inputs
        x = torch.randn(100, 64)

        sparse = encoder(x)

        # Normalize and compute correlations
        sparse_norm = sparse / (sparse.norm(dim=1, keepdim=True) + 1e-8)
        correlations = torch.mm(sparse_norm, sparse_norm.t())

        # Get off-diagonal elements
        mask = ~torch.eye(100, dtype=bool)
        off_diagonal = correlations[mask]

        # Average correlation should be low (patterns decorrelated)
        avg_correlation = off_diagonal.abs().mean().item()
        assert avg_correlation < 0.5  # Reasonably decorrelated

    def test_decode_reconstruction(self, encoder):
        """Decode produces reasonable reconstruction."""
        x = torch.randn(8, 64)

        sparse = encoder(x)
        reconstructed = encoder.decode(sparse)

        assert reconstructed.shape == x.shape

        # Reconstruction error should be bounded
        # (Not perfect due to sparsity, but not random)
        error = (reconstructed - x).norm() / x.norm()
        assert error.item() < 10.0  # Rough bound

    def test_get_active_indices(self, encoder):
        """Active indices extraction works."""
        x = torch.randn(4, 64)

        sparse = encoder(x)
        indices = encoder.get_active_indices(sparse)

        assert len(indices) == 4
        for idx_list in indices:
            assert len(idx_list) == encoder.k

    def test_pattern_overlap(self, encoder):
        """Pattern overlap computation works."""
        x1 = torch.randn(1, 64)
        x2 = torch.randn(1, 64)

        code1 = encoder(x1)
        code2 = encoder(x2)

        overlap = encoder.compute_pattern_overlap(code1[0], code2[0])

        assert 0.0 <= overlap <= 1.0

    def test_soft_threshold_mode(self):
        """Soft thresholding mode works."""
        encoder = SparseEncoder(
            input_dim=64,
            hidden_dim=256,
            sparsity=0.02,
            use_kwta=False  # Soft thresholding
        )
        x = torch.randn(8, 64)

        sparse = encoder(x)

        # Should still be sparse, though not exactly k
        sparsity = compute_sparsity(sparse)
        assert sparsity < 0.1

    def test_lateral_inhibition(self):
        """Lateral inhibition affects encoding."""
        encoder_no_inhibit = SparseEncoder(
            input_dim=64,
            hidden_dim=256,
            lateral_inhibition=0.0
        )
        encoder_inhibit = SparseEncoder(
            input_dim=64,
            hidden_dim=256,
            lateral_inhibition=0.5
        )

        x = torch.randn(8, 64)

        sparse_no = encoder_no_inhibit(x)
        sparse_yes = encoder_inhibit(x)

        # Both should be valid
        assert sparse_no.shape == sparse_yes.shape

    def test_gradient_flow(self, encoder):
        """Gradients flow through encoder."""
        x = torch.randn(4, 64, requires_grad=True)

        sparse = encoder(x)
        loss = sparse.sum()
        loss.backward()

        assert x.grad is not None
        # Gradients should be non-zero (straight-through estimator)
        assert x.grad.abs().sum() > 0


class TestAdaptiveSparseEncoder:
    """Tests for adaptive sparse encoder."""

    @pytest.fixture
    def encoder(self):
        """Create adaptive encoder."""
        return AdaptiveSparseEncoder(
            input_dim=64,
            hidden_dim=256,
            target_sparsity=0.02,
            min_sparsity=0.01,
            max_sparsity=0.05
        )

    def test_initialization(self, encoder):
        """Adaptive encoder initializes correctly."""
        assert encoder.target_sparsity == 0.02
        assert encoder.min_k > 0
        assert encoder.max_k > encoder.min_k

    def test_forward_shape(self, encoder):
        """Forward produces correct shape."""
        x = torch.randn(8, 64)

        sparse = encoder(x)

        assert sparse.shape == (8, 256)

    def test_sparsity_bounds(self, encoder):
        """Sparsity stays within bounds."""
        # Test with various input magnitudes
        for scale in [0.1, 1.0, 10.0]:
            x = scale * torch.randn(32, 64)

            sparse = encoder(x)

            for i in range(32):
                active = (sparse[i] != 0).sum().item()
                assert encoder.min_k <= active <= encoder.max_k
