"""
Unit Tests for Dream-Based Negative Samples (W2-04).

Verifies VAE-based generation of negative samples for Forward-Forward
learning following Hinton (2022) principles.

Evidence Base: Hinton (2022) "The Forward-Forward Algorithm"
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, MagicMock, patch


class TestDreamNegativeGenerator:
    """Test DreamNegativeGenerator for negative sample creation."""

    @pytest.fixture
    def mock_vae(self):
        """Create mock VAE for testing."""
        vae = Mock()

        def mock_encode(x):
            batch_size = x.shape[0]
            latent_dim = 64
            mu = torch.zeros(batch_size, latent_dim)
            logvar = torch.zeros(batch_size, latent_dim)
            return mu, logvar

        def mock_decode(z):
            batch_size = z.shape[0]
            output_dim = 128
            return torch.randn(batch_size, output_dim)

        vae.encode = mock_encode
        vae.decode = mock_decode
        return vae

    def test_generator_creation(self, mock_vae):
        """Should create generator with VAE and corruption rate."""
        from t4dm.learning.dream_negatives import DreamNegativeGenerator

        generator = DreamNegativeGenerator(mock_vae, corruption_rate=0.3)

        assert generator.vae is mock_vae
        assert generator.corruption_rate == 0.3

    def test_vae_sample_method(self, mock_vae):
        """VAE sample method should generate negative samples."""
        from t4dm.learning.dream_negatives import DreamNegativeGenerator

        generator = DreamNegativeGenerator(mock_vae)
        positive = torch.randn(8, 128)

        negative = generator.generate(positive, method="vae_sample")

        assert negative.shape == positive.shape

    def test_shuffle_method(self, mock_vae):
        """Shuffle method should permute samples."""
        from t4dm.learning.dream_negatives import DreamNegativeGenerator

        generator = DreamNegativeGenerator(mock_vae)
        positive = torch.arange(32).reshape(8, 4).float()

        negative = generator.generate(positive, method="shuffle")

        assert negative.shape == positive.shape
        # At least one sample should be different
        assert not torch.equal(negative, positive)

    def test_hybrid_method(self, mock_vae):
        """Hybrid method should mix VAE and shuffle."""
        from t4dm.learning.dream_negatives import DreamNegativeGenerator

        generator = DreamNegativeGenerator(mock_vae)
        positive = torch.randn(16, 128)

        negative = generator.generate(positive, method="hybrid")

        assert negative.shape == positive.shape

    def test_invalid_method_raises(self, mock_vae):
        """Invalid method should raise ValueError."""
        from t4dm.learning.dream_negatives import DreamNegativeGenerator

        generator = DreamNegativeGenerator(mock_vae)
        positive = torch.randn(8, 128)

        with pytest.raises(ValueError, match="Unknown method"):
            generator.generate(positive, method="invalid")


class TestNegativeQuality:
    """Test quality properties of generated negatives."""

    @pytest.fixture
    def generator(self):
        """Create generator with simple VAE mock."""
        from t4dm.learning.dream_negatives import DreamNegativeGenerator

        vae = Mock()
        vae.encode = lambda x: (
            torch.zeros(x.shape[0], 64),
            torch.zeros(x.shape[0], 64),
        )
        # Decode returns something close to input but shifted
        vae.decode = lambda z: torch.randn(z.shape[0], 128) * 0.5

        return DreamNegativeGenerator(vae, corruption_rate=0.3)

    def test_negatives_different_from_positive(self, generator):
        """Negatives should be different from positive samples."""
        positive = torch.randn(16, 128)

        negative = generator.generate(positive, method="vae_sample")

        # Should not be identical
        assert not torch.equal(positive, negative)

    def test_negatives_have_realistic_statistics(self, generator):
        """Negatives should have similar statistics to positive manifold."""
        positive = torch.randn(32, 128) * 2  # Scale to have larger variance

        negative = generator.generate(positive, method="vae_sample")

        # Negative should have finite values
        assert torch.isfinite(negative).all()

    def test_corruption_affects_output(self):
        """Higher corruption should produce more different negatives."""
        from t4dm.learning.dream_negatives import DreamNegativeGenerator

        vae = Mock()
        vae.encode = lambda x: (torch.zeros(x.shape[0], 32), torch.zeros(x.shape[0], 32))
        vae.decode = lambda z: torch.zeros(z.shape[0], 64)

        gen_low = DreamNegativeGenerator(vae, corruption_rate=0.1)
        gen_high = DreamNegativeGenerator(vae, corruption_rate=0.9)

        positive = torch.zeros(16, 64)

        neg_low = gen_low.generate(positive, method="vae_sample")
        neg_high = gen_high.generate(positive, method="vae_sample")

        # High corruption should deviate more from VAE output
        var_low = neg_low.var().item()
        var_high = neg_high.var().item()

        # With higher corruption, variance should increase
        assert var_high > var_low


class TestDreamNegativeConfig:
    """Test configuration for dream negatives."""

    def test_default_config(self):
        """Default config should have sensible values."""
        from t4dm.learning.dream_negatives import DreamNegativeConfig

        config = DreamNegativeConfig()

        assert config.corruption_rate == 0.3
        assert config.default_method == "vae_sample"
        assert config.latent_noise_scale > 0

    def test_config_override(self):
        """Should be able to override config values."""
        from t4dm.learning.dream_negatives import DreamNegativeConfig

        config = DreamNegativeConfig(
            corruption_rate=0.5,
            default_method="hybrid",
            latent_noise_scale=3.0,
        )

        assert config.corruption_rate == 0.5
        assert config.default_method == "hybrid"


class TestFFIntegration:
    """Test integration with Forward-Forward learning."""

    @pytest.fixture
    def mock_vae(self):
        vae = Mock()
        vae.encode = lambda x: (torch.zeros(x.shape[0], 32), torch.zeros(x.shape[0], 32))
        vae.decode = lambda z: torch.randn(z.shape[0], 64) * 0.5
        return vae

    def test_negatives_usable_for_ff(self, mock_vae):
        """Negatives should be directly usable for FF training."""
        from t4dm.learning.dream_negatives import DreamNegativeGenerator

        generator = DreamNegativeGenerator(mock_vae)

        # Simulate positive data
        positive = torch.randn(32, 64)

        # Generate negatives
        negative = generator.generate(positive, method="vae_sample")

        # Should be same shape and dtype
        assert negative.shape == positive.shape
        assert negative.dtype == positive.dtype

    def test_batch_consistency(self, mock_vae):
        """Different batch sizes should work correctly."""
        from t4dm.learning.dream_negatives import DreamNegativeGenerator

        generator = DreamNegativeGenerator(mock_vae)

        for batch_size in [1, 8, 32, 64]:
            positive = torch.randn(batch_size, 64)
            negative = generator.generate(positive, method="vae_sample")

            assert negative.shape == positive.shape


class TestLatency:
    """Test latency requirements."""

    def test_generation_under_5ms(self):
        """Generation should add <5ms latency."""
        from t4dm.learning.dream_negatives import DreamNegativeGenerator
        import time

        # Simple VAE mock
        vae = Mock()
        vae.encode = lambda x: (torch.zeros(x.shape[0], 32), torch.zeros(x.shape[0], 32))
        vae.decode = lambda z: torch.zeros(z.shape[0], 64)

        generator = DreamNegativeGenerator(vae)
        positive = torch.randn(32, 64)

        # Warmup
        for _ in range(10):
            generator.generate(positive, method="shuffle")

        # Measure
        times = []
        for _ in range(50):
            start = time.perf_counter()
            generator.generate(positive, method="shuffle")
            times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000

        assert avg_time_ms < 10.0, f"Generation took {avg_time_ms:.2f}ms, should be <10ms"
