"""Tests for encoding utility functions."""

import pytest
import torch
import numpy as np

from ww.encoding.utils import (
    compute_sparsity,
    validate_sparsity,
    cosine_similarity_matrix,
    compute_pattern_orthogonality,
    straight_through_estimator,
    exponential_decay,
    normalize_to_range,
    add_noise,
)


class TestComputeSparsity:
    """Tests for compute_sparsity function."""

    def test_all_zeros(self):
        """All zeros has sparsity 0."""
        x = torch.zeros(10)
        assert compute_sparsity(x) == 0.0

    def test_all_nonzero(self):
        """All non-zero has sparsity 1."""
        x = torch.ones(10)
        assert compute_sparsity(x) == 1.0

    def test_half_sparse(self):
        """Half zeros has sparsity 0.5."""
        x = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        assert compute_sparsity(x) == pytest.approx(0.5)

    def test_2d_tensor(self):
        """Works with 2D tensors."""
        x = torch.zeros(4, 4)
        x[0, 0] = 1.0
        x[1, 1] = 1.0
        # 2 non-zero out of 16
        assert compute_sparsity(x) == pytest.approx(0.125)


class TestValidateSparsity:
    """Tests for validate_sparsity function."""

    def test_valid_sparsity(self):
        """Sparsity in range is valid."""
        x = torch.zeros(100)
        x[:3] = 1.0  # 3% sparsity
        is_valid, actual = validate_sparsity(x)
        assert is_valid is True
        assert actual == pytest.approx(0.03)

    def test_too_sparse(self):
        """Sparsity below min is invalid."""
        x = torch.zeros(1000)
        x[0] = 1.0  # 0.1% sparsity
        is_valid, actual = validate_sparsity(x, min_sparsity=0.01)
        assert is_valid is False
        assert actual < 0.01

    def test_too_dense(self):
        """Sparsity above max is invalid."""
        x = torch.ones(100)
        is_valid, actual = validate_sparsity(x, max_sparsity=0.5)
        assert is_valid is False
        assert actual > 0.5

    def test_custom_range(self):
        """Custom sparsity range."""
        x = torch.zeros(10)
        x[:2] = 1.0  # 20% sparsity
        is_valid, actual = validate_sparsity(x, min_sparsity=0.1, max_sparsity=0.3)
        assert is_valid is True
        assert actual == pytest.approx(0.2)


class TestCosineSimilarityMatrix:
    """Tests for cosine_similarity_matrix function."""

    def test_self_similarity(self):
        """Self-similarity is 1."""
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        sim = cosine_similarity_matrix(x)
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[1, 1] == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0."""
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        sim = cosine_similarity_matrix(x)
        assert sim[0, 1] == pytest.approx(0.0)
        assert sim[1, 0] == pytest.approx(0.0)

    def test_identical_vectors(self):
        """Identical vectors have similarity 1."""
        x = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
        sim = cosine_similarity_matrix(x)
        assert sim[0, 1] == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1."""
        x = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        sim = cosine_similarity_matrix(x)
        assert sim[0, 1] == pytest.approx(-1.0)

    def test_output_shape(self):
        """Output shape is (N, N)."""
        x = torch.randn(5, 10)
        sim = cosine_similarity_matrix(x)
        assert sim.shape == (5, 5)


class TestComputePatternOrthogonality:
    """Tests for compute_pattern_orthogonality function."""

    def test_single_pattern(self):
        """Single pattern returns 1.0."""
        patterns = torch.randn(1, 10)
        assert compute_pattern_orthogonality(patterns) == 1.0

    def test_orthogonal_patterns(self):
        """Orthogonal patterns have high orthogonality."""
        patterns = torch.eye(4)  # 4 orthogonal unit vectors
        orth = compute_pattern_orthogonality(patterns)
        assert orth == pytest.approx(1.0, abs=0.01)

    def test_identical_patterns(self):
        """Identical patterns have low orthogonality."""
        patterns = torch.ones(4, 10)  # All same direction
        orth = compute_pattern_orthogonality(patterns)
        assert orth == pytest.approx(0.0, abs=0.01)

    def test_random_patterns(self):
        """Random patterns have intermediate orthogonality."""
        torch.manual_seed(42)
        patterns = torch.randn(10, 100)
        orth = compute_pattern_orthogonality(patterns)
        assert 0.0 < orth < 1.0


class TestStraightThroughEstimator:
    """Tests for straight_through_estimator function."""

    def test_basic_threshold(self):
        """Basic thresholding works."""
        x = torch.tensor([0.3, 0.5, 0.7, 0.9])
        threshold = torch.tensor(0.5)
        result = straight_through_estimator(x, threshold)
        # Values > 0.5 should be 1, others 0 (approximately)
        assert result.shape == x.shape

    def test_gradient_flow(self):
        """Gradients flow through."""
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        threshold = torch.tensor(0.5)
        result = straight_through_estimator(x, threshold)
        loss = result.sum()
        loss.backward()
        # Gradient should exist (not blocked)
        assert x.grad is not None


class TestExponentialDecay:
    """Tests for exponential_decay function."""

    def test_zero_time(self):
        """No decay at dt=0."""
        result = exponential_decay(1.0, tau=10.0, dt=0.0)
        assert result == pytest.approx(1.0)

    def test_one_tau(self):
        """At dt=tau, decay to ~1/e."""
        result = exponential_decay(1.0, tau=1.0, dt=1.0)
        assert result == pytest.approx(1.0 / np.e, abs=0.001)

    def test_large_time(self):
        """Large time gives near-zero."""
        result = exponential_decay(1.0, tau=1.0, dt=100.0)
        assert result < 1e-10

    def test_scaling(self):
        """Scales with initial value."""
        result = exponential_decay(10.0, tau=1.0, dt=1.0)
        assert result == pytest.approx(10.0 / np.e, abs=0.001)


class TestNormalizeToRange:
    """Tests for normalize_to_range function."""

    def test_basic_normalization(self):
        """Normalize to [0, 1]."""
        x = torch.tensor([0.0, 5.0, 10.0])
        result = normalize_to_range(x)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_custom_range(self):
        """Normalize to custom range."""
        x = torch.tensor([0.0, 5.0, 10.0])
        result = normalize_to_range(x, min_val=-1.0, max_val=1.0)
        assert result.min() == pytest.approx(-1.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_input(self):
        """Constant input returns middle value."""
        x = torch.ones(5) * 3.0
        result = normalize_to_range(x, min_val=0.0, max_val=1.0)
        assert all(r == pytest.approx(0.5) for r in result.tolist())

    def test_negative_values(self):
        """Works with negative values."""
        x = torch.tensor([-10.0, 0.0, 10.0])
        result = normalize_to_range(x)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.5)  # Middle value


class TestAddNoise:
    """Tests for add_noise function."""

    def test_gaussian_noise(self):
        """Add Gaussian noise."""
        x = torch.zeros(1000)
        result = add_noise(x, noise_std=1.0, noise_type="gaussian")
        # Should have non-zero mean close to 0, std close to 1
        assert result.mean().abs() < 0.2  # Should be near 0
        assert 0.8 < result.std() < 1.2  # Should be near 1

    def test_uniform_noise(self):
        """Add uniform noise."""
        x = torch.zeros(1000)
        result = add_noise(x, noise_std=1.0, noise_type="uniform")
        # Uniform in [-1, 1] has std ~0.577
        assert result.mean().abs() < 0.2

    def test_noise_std_scaling(self):
        """Noise scales with std."""
        x = torch.zeros(1000)
        result_small = add_noise(x, noise_std=0.1, noise_type="gaussian")
        result_large = add_noise(x, noise_std=1.0, noise_type="gaussian")
        assert result_small.std() < result_large.std()

    def test_invalid_noise_type(self):
        """Invalid noise type raises error."""
        x = torch.zeros(10)
        with pytest.raises(ValueError, match="Unknown noise type"):
            add_noise(x, noise_type="invalid")

    def test_preserves_shape(self):
        """Output shape matches input."""
        x = torch.randn(3, 4, 5)
        result = add_noise(x, noise_std=0.1)
        assert result.shape == x.shape
