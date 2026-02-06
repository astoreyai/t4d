"""
Unit Tests for Uncertainty-Aware Memory Storage (W2-01).

Verifies MC Dropout-based uncertainty estimation for memory embeddings
following Friston's Free Energy Principle.

Evidence Base: Friston (2010) "The free-energy principle: a unified brain theory?"
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4


class TestUncertaintyAwareItem:
    """Test UncertaintyAwareItem data structure."""

    def test_item_creation(self):
        """Should create item with mean and covariance."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyAwareItem

        item = UncertaintyAwareItem(
            id=uuid4(),
            vector_mean=np.random.randn(1024),
            vector_cov=np.random.rand(1024),  # Diagonal covariance
            content="Test content",
            kappa=0.5,
            importance=1.0,
        )

        assert item.vector_mean.shape == (1024,)
        assert item.vector_cov.shape == (1024,)

    def test_uncertainty_from_diagonal_covariance(self):
        """Uncertainty should be sum of variances for diagonal cov."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyAwareItem

        variances = np.ones(1024) * 0.1  # Uniform variance
        item = UncertaintyAwareItem(
            id=uuid4(),
            vector_mean=np.zeros(1024),
            vector_cov=variances,
            content="Test",
            kappa=0.5,
            importance=1.0,
        )

        expected_uncertainty = 1024 * 0.1
        assert abs(item.uncertainty - expected_uncertainty) < 0.001

    def test_uncertainty_from_full_covariance(self):
        """Uncertainty should be trace of covariance for full matrix."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyAwareItem

        # Create 64-dim for efficiency
        full_cov = np.eye(64) * 0.5  # Diagonal matrix
        item = UncertaintyAwareItem(
            id=uuid4(),
            vector_mean=np.zeros(64),
            vector_cov=full_cov,
            content="Test",
            kappa=0.5,
            importance=1.0,
        )

        expected_uncertainty = 64 * 0.5
        assert abs(item.uncertainty - expected_uncertainty) < 0.001

    def test_confidence_inverse_of_uncertainty(self):
        """Confidence should be 1 / (1 + uncertainty)."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyAwareItem

        # Low uncertainty → high confidence
        low_var = np.ones(64) * 0.01
        item_low = UncertaintyAwareItem(
            id=uuid4(),
            vector_mean=np.zeros(64),
            vector_cov=low_var,
            content="Test",
            kappa=0.5,
            importance=1.0,
        )

        # High uncertainty → low confidence
        high_var = np.ones(64) * 1.0
        item_high = UncertaintyAwareItem(
            id=uuid4(),
            vector_mean=np.zeros(64),
            vector_cov=high_var,
            content="Test",
            kappa=0.5,
            importance=1.0,
        )

        assert item_low.confidence > item_high.confidence
        assert 0 < item_low.confidence <= 1
        assert 0 < item_high.confidence <= 1

    def test_zero_variance_gives_max_confidence(self):
        """Zero variance should give confidence = 1."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyAwareItem

        item = UncertaintyAwareItem(
            id=uuid4(),
            vector_mean=np.zeros(64),
            vector_cov=np.zeros(64),
            content="Test",
            kappa=0.5,
            importance=1.0,
        )

        assert item.confidence == 1.0


class TestUncertaintyEstimator:
    """Test MC Dropout-based uncertainty estimation."""

    @pytest.fixture
    def mock_model(self):
        """Create mock embedding model with dropout."""
        model = Mock()
        # Simulate dropout by adding noise
        def encode_with_noise(text):
            base = np.random.randn(1024)
            noise = np.random.randn(1024) * 0.1
            return base + noise

        model.encode = encode_with_noise
        model.train = Mock()
        model.eval = Mock()
        return model

    def test_estimator_creation(self):
        """Should create estimator with model and method."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyEstimator

        model = Mock()
        estimator = UncertaintyEstimator(model, method="mc_dropout")

        assert estimator.model is model
        assert estimator.method == "mc_dropout"
        assert estimator.n_samples > 0

    def test_mc_dropout_produces_mean_and_variance(self, mock_model):
        """MC Dropout should produce mean and variance estimates."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyEstimator

        estimator = UncertaintyEstimator(mock_model, method="mc_dropout")
        estimator.n_samples = 10

        mean, var = estimator.embed_with_uncertainty("Test text")

        assert mean.shape == (1024,)
        assert var.shape == (1024,)
        assert np.all(var >= 0), "Variance should be non-negative"

    def test_mc_dropout_enables_training_mode(self, mock_model):
        """MC Dropout should enable training mode for dropout."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyEstimator

        estimator = UncertaintyEstimator(mock_model, method="mc_dropout")
        estimator.n_samples = 5

        estimator.embed_with_uncertainty("Test text")

        mock_model.train.assert_called()
        mock_model.eval.assert_called()

    def test_variance_reflects_model_uncertainty(self):
        """Higher dropout noise should produce higher variance."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyEstimator

        # Model with low noise
        low_noise_model = Mock()
        low_noise_model.encode = lambda t: np.zeros(128) + np.random.randn(128) * 0.01
        low_noise_model.train = Mock()
        low_noise_model.eval = Mock()

        # Model with high noise
        high_noise_model = Mock()
        high_noise_model.encode = lambda t: np.zeros(128) + np.random.randn(128) * 0.5
        high_noise_model.train = Mock()
        high_noise_model.eval = Mock()

        est_low = UncertaintyEstimator(low_noise_model, method="mc_dropout")
        est_low.n_samples = 20
        est_high = UncertaintyEstimator(high_noise_model, method="mc_dropout")
        est_high.n_samples = 20

        _, var_low = est_low.embed_with_uncertainty("Test")
        _, var_high = est_high.embed_with_uncertainty("Test")

        assert np.sum(var_high) > np.sum(var_low), \
            "Higher noise should produce higher variance"

    def test_invalid_method_raises(self, mock_model):
        """Invalid method should raise ValueError."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyEstimator

        estimator = UncertaintyEstimator(mock_model, method="invalid")

        with pytest.raises(ValueError, match="Unknown method"):
            estimator.embed_with_uncertainty("Test")


class TestUncertaintyConfig:
    """Test uncertainty configuration."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyConfig

        config = UncertaintyConfig()

        assert config.n_samples >= 5
        assert config.use_diagonal_covariance is True
        assert config.method == "mc_dropout"

    def test_config_override(self):
        """Should be able to override config values."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyConfig

        config = UncertaintyConfig(
            n_samples=20,
            use_diagonal_covariance=False,
            method="ensemble",
        )

        assert config.n_samples == 20
        assert config.use_diagonal_covariance is False
        assert config.method == "ensemble"


class TestUncertaintyAwareSearch:
    """Test search with uncertainty consideration."""

    def test_search_returns_confidence(self):
        """Search results should include confidence scores."""
        from t4dm.storage.t4dx.uncertainty import (
            UncertaintyAwareItem,
            UncertaintyAwareSearch,
        )

        # Create mock engine with items
        engine = Mock()
        items = [
            UncertaintyAwareItem(
                id=uuid4(),
                vector_mean=np.random.randn(64),
                vector_cov=np.random.rand(64) * 0.1,
                content=f"Item {i}",
                kappa=0.5,
                importance=1.0,
            )
            for i in range(5)
        ]
        engine.search = Mock(return_value=items)

        search = UncertaintyAwareSearch(engine)
        results = search.search_with_confidence(np.random.randn(64), k=5)

        assert len(results) == 5
        for result in results:
            assert hasattr(result, "confidence")
            assert 0 < result.confidence <= 1

    def test_high_confidence_results_ranked_higher(self):
        """High confidence results should be ranked higher when using confidence weighting."""
        from t4dm.storage.t4dx.uncertainty import (
            UncertaintyAwareItem,
            UncertaintyAwareSearch,
        )

        engine = Mock()

        # Create items with varying uncertainty
        high_conf_item = UncertaintyAwareItem(
            id=uuid4(),
            vector_mean=np.random.randn(64),
            vector_cov=np.ones(64) * 0.01,  # Low variance
            content="High confidence",
            kappa=0.5,
            importance=1.0,
        )
        low_conf_item = UncertaintyAwareItem(
            id=uuid4(),
            vector_mean=np.random.randn(64),
            vector_cov=np.ones(64) * 1.0,  # High variance
            content="Low confidence",
            kappa=0.5,
            importance=1.0,
        )

        # Mock equal similarity scores
        engine.search = Mock(return_value=[low_conf_item, high_conf_item])

        search = UncertaintyAwareSearch(engine)
        results = search.search_with_confidence(
            np.random.randn(64),
            k=2,
            confidence_weight=0.5,
        )

        # High confidence should come first
        assert results[0].confidence > results[1].confidence


class TestStorageOverhead:
    """Test that uncertainty storage is efficient."""

    def test_diagonal_covariance_size(self):
        """Diagonal covariance should use O(d) space, not O(d^2)."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyAwareItem
        import sys

        # Diagonal covariance: 1024 floats
        diagonal_item = UncertaintyAwareItem(
            id=uuid4(),
            vector_mean=np.random.randn(1024),
            vector_cov=np.random.rand(1024),
            content="Test",
            kappa=0.5,
            importance=1.0,
        )

        # Size should be roughly 2x the vector size (mean + diagonal cov)
        mean_size = diagonal_item.vector_mean.nbytes
        cov_size = diagonal_item.vector_cov.nbytes

        # Diagonal cov should be same size as mean (1024 floats)
        assert cov_size == mean_size, "Diagonal cov should be O(d), not O(d^2)"

    def test_overhead_under_2x(self):
        """Total storage overhead should be under 2x."""
        from t4dm.storage.t4dx.uncertainty import UncertaintyAwareItem

        # Without uncertainty (just mean)
        base_size = np.random.randn(1024).nbytes

        # With uncertainty (mean + diagonal cov)
        item = UncertaintyAwareItem(
            id=uuid4(),
            vector_mean=np.random.randn(1024),
            vector_cov=np.random.rand(1024),
            content="Test",
            kappa=0.5,
            importance=1.0,
        )

        with_uncertainty = item.vector_mean.nbytes + item.vector_cov.nbytes

        overhead = with_uncertainty / base_size
        assert overhead <= 2.0, f"Overhead {overhead}x should be <= 2x"
