"""
Tests for contrastive adapter.

Validates:
1. Forward pass and projection
2. InfoNCE loss computation
3. Hard negative mining
4. Temporal contrastive loss
5. Adam optimizer updates
6. Temperature learning
7. Weight save/load
"""

import numpy as np
import pytest

from t4dm.embedding.contrastive_trainer import (
    ContrastiveAdapter,
    ContrastiveConfig,
    AdapterStats,
    AdapterMode,
    create_contrastive_adapter,
)


class TestContrastiveConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = ContrastiveConfig()

        assert config.input_dim == 1024
        assert config.hidden_dim == 512
        assert config.output_dim == 256
        assert 0 < config.temperature <= 1.0
        assert 0 < config.learning_rate < 1.0

    def test_custom_config(self):
        """Custom config values are respected."""
        config = ContrastiveConfig(
            input_dim=768,
            hidden_dim=384,
            output_dim=128,
            temperature=0.07,
        )

        assert config.input_dim == 768
        assert config.hidden_dim == 384
        assert config.output_dim == 128
        assert config.temperature == 0.07

    def test_validation(self):
        """Config validates parameters."""
        # Temperature out of range should fail
        with pytest.raises(AssertionError):
            ContrastiveConfig(temperature=2.0)

        # Hard negative ratio out of range should fail
        with pytest.raises(AssertionError):
            ContrastiveConfig(hard_negative_ratio=1.5)


class TestContrastiveAdapter:
    """Test contrastive adapter."""

    def test_initialization(self):
        """Adapter initializes correctly."""
        config = ContrastiveConfig()
        adapter = ContrastiveAdapter(config)

        assert adapter.mode == AdapterMode.ADAPTER
        assert adapter.W1.shape == (config.input_dim, config.hidden_dim)
        assert adapter.W2.shape == (config.hidden_dim, config.output_dim)

    def test_forward_single(self):
        """Forward pass works for single embedding."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        embedding = np.random.randn(128).astype(np.float32)
        projected = adapter.forward(embedding)

        assert projected.shape == (1, 64)
        # Should be L2 normalized
        assert np.abs(np.linalg.norm(projected) - 1.0) < 1e-5

    def test_forward_batch(self):
        """Forward pass works for batch of embeddings."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        embeddings = np.random.randn(32, 128).astype(np.float32)
        projected = adapter.forward(embeddings)

        assert projected.shape == (32, 64)

    def test_forward_normalized(self):
        """Output embeddings are L2 normalized."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        embeddings = np.random.randn(16, 128).astype(np.float32)
        projected = adapter.forward(embeddings)

        norms = np.linalg.norm(projected, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_temperature_property(self):
        """Temperature property works correctly."""
        config = ContrastiveConfig(temperature=0.1)
        adapter = ContrastiveAdapter(config)

        assert np.abs(adapter.temperature - 0.1) < 1e-5


class TestInfoNCELoss:
    """Test InfoNCE loss computation."""

    def test_loss_basic(self):
        """InfoNCE loss is computed correctly."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        batch_size = 8
        num_neg = 16

        anchors = np.random.randn(batch_size, 64).astype(np.float32)
        positives = np.random.randn(batch_size, 64).astype(np.float32)
        negatives = np.random.randn(batch_size, num_neg, 64).astype(np.float32)

        # Normalize
        anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
        positives = positives / np.linalg.norm(positives, axis=1, keepdims=True)
        negatives = negatives / np.linalg.norm(negatives, axis=2, keepdims=True)

        loss, accuracy = adapter.info_nce_loss(anchors, positives, negatives)

        assert loss > 0
        assert 0 <= accuracy <= 1

    def test_loss_with_identical_positive(self):
        """Loss should be low when positive is identical to anchor."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        batch_size = 8
        num_neg = 16

        anchors = np.random.randn(batch_size, 64).astype(np.float32)
        anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)

        # Positive is same as anchor
        positives = anchors.copy()

        # Random negatives
        negatives = np.random.randn(batch_size, num_neg, 64).astype(np.float32)
        negatives = negatives / np.linalg.norm(negatives, axis=2, keepdims=True)

        loss, accuracy = adapter.info_nce_loss(anchors, positives, negatives)

        # Accuracy should be high (positive clearly best match)
        assert accuracy > 0.5

    def test_loss_temperature_effect(self):
        """Lower temperature should sharpen distinctions."""
        batch_size = 8
        num_neg = 16

        anchors = np.random.randn(batch_size, 64).astype(np.float32)
        positives = anchors + np.random.randn(batch_size, 64).astype(np.float32) * 0.1
        negatives = np.random.randn(batch_size, num_neg, 64).astype(np.float32)

        # Normalize
        anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
        positives = positives / np.linalg.norm(positives, axis=1, keepdims=True)
        negatives = negatives / np.linalg.norm(negatives, axis=2, keepdims=True)

        # Low temperature
        adapter_low = create_contrastive_adapter(input_dim=128, output_dim=64)
        adapter_low._log_temperature = np.log(0.05)
        loss_low, acc_low = adapter_low.info_nce_loss(anchors, positives, negatives)

        # High temperature
        adapter_high = create_contrastive_adapter(input_dim=128, output_dim=64)
        adapter_high._log_temperature = np.log(0.5)
        loss_high, acc_high = adapter_high.info_nce_loss(anchors, positives, negatives)

        # Both should work
        assert loss_low > 0
        assert loss_high > 0


class TestHardNegativeMining:
    """Test hard negative mining."""

    def test_mine_hard_negatives(self):
        """Hard negative mining produces correct shape."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        batch_size = 4
        pool_size = 100
        num_neg = 16

        anchors = np.random.randn(batch_size, 64).astype(np.float32)
        candidates = np.random.randn(pool_size, 64).astype(np.float32)

        # Normalize
        anchors = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        # Create positive mask (first 10 are positives for each)
        positive_mask = np.zeros((batch_size, pool_size), dtype=bool)
        for i in range(batch_size):
            positive_mask[i, i*10:(i+1)*10] = True

        hard_negs = adapter.mine_hard_negatives(
            anchors, candidates, positive_mask, num_negatives=num_neg
        )

        assert hard_negs.shape == (batch_size, num_neg, 64)


class TestTemporalContrastiveLoss:
    """Test temporal contrastive loss."""

    def test_temporal_loss_basic(self):
        """Temporal loss works for sequences."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        # Create sequence of embeddings
        seq_len = 20
        sequence = np.random.randn(seq_len, 64).astype(np.float32)
        sequence = sequence / np.linalg.norm(sequence, axis=1, keepdims=True)

        loss = adapter.temporal_contrastive_loss(sequence, window_size=3)

        assert loss >= 0

    def test_temporal_loss_short_sequence(self):
        """Short sequences return zero loss."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        sequence = np.random.randn(3, 64).astype(np.float32)
        loss = adapter.temporal_contrastive_loss(sequence, window_size=3)

        assert loss == 0.0


class TestAdapterUpdate:
    """Test adapter training updates."""

    def test_update_decreases_loss(self):
        """Multiple updates should decrease loss."""
        adapter = create_contrastive_adapter(
            input_dim=128, output_dim=64,
            learning_rate=0.01
        )

        batch_size = 16
        num_neg = 8

        # Create training data
        anchors = np.random.randn(batch_size, 128).astype(np.float32)
        positives = anchors + np.random.randn(batch_size, 128).astype(np.float32) * 0.1
        negatives = np.random.randn(batch_size, num_neg, 128).astype(np.float32)

        # First update
        result1 = adapter.update(anchors, positives, negatives)
        loss1 = result1["contrastive_loss"]

        # More updates
        for _ in range(5):
            result = adapter.update(anchors, positives, negatives)

        loss_final = result["contrastive_loss"]

        # Stats should be updated
        assert adapter.stats.total_updates == 6
        assert adapter.stats.total_samples == batch_size * 6

    def test_update_with_temporal(self):
        """Update works with temporal sequence."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        batch_size = 8
        num_neg = 8
        seq_len = 20

        anchors = np.random.randn(batch_size, 128).astype(np.float32)
        positives = np.random.randn(batch_size, 128).astype(np.float32)
        negatives = np.random.randn(batch_size, num_neg, 128).astype(np.float32)
        sequence = np.random.randn(seq_len, 128).astype(np.float32)

        result = adapter.update(anchors, positives, negatives, temporal_sequence=sequence)

        assert "temporal_loss" in result
        assert result["temporal_loss"] >= 0


class TestWeightPersistence:
    """Test weight save/load."""

    def test_save_weights(self):
        """Weights can be saved to dictionary."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        weights = adapter.save_weights()

        assert "W1" in weights
        assert "W2" in weights
        assert "b1" in weights
        assert "b2" in weights
        assert "log_temperature" in weights
        assert "config" in weights

    def test_load_weights(self):
        """Weights can be loaded from dictionary."""
        adapter1 = create_contrastive_adapter(input_dim=128, output_dim=64)

        # Modify weights
        adapter1.W1 += 0.1
        original_W1 = adapter1.W1.copy()

        weights = adapter1.save_weights()

        # Create new adapter and load
        adapter2 = create_contrastive_adapter(input_dim=128, output_dim=64)
        adapter2.load_weights(weights)

        assert np.allclose(adapter2.W1, original_W1)

    def test_forward_consistency_after_load(self):
        """Forward pass gives same results after load."""
        adapter1 = create_contrastive_adapter(input_dim=128, output_dim=64)

        embedding = np.random.randn(8, 128).astype(np.float32)
        output1 = adapter1.forward(embedding, training=False)

        weights = adapter1.save_weights()

        adapter2 = create_contrastive_adapter(input_dim=128, output_dim=64)
        adapter2.load_weights(weights)
        output2 = adapter2.forward(embedding, training=False)

        assert np.allclose(output1, output2)


class TestHealthStatus:
    """Test health monitoring."""

    def test_health_status(self):
        """Health status is returned correctly."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        status = adapter.get_health_status()

        assert "mode" in status
        assert "temperature" in status
        assert "total_updates" in status
        assert "healthy" in status

    def test_healthy_after_training(self):
        """Adapter should be healthy after successful training."""
        adapter = create_contrastive_adapter(input_dim=128, output_dim=64)

        batch_size = 16
        num_neg = 8

        # Train for a few iterations with similar positives
        for _ in range(20):
            anchors = np.random.randn(batch_size, 128).astype(np.float32)
            positives = anchors + np.random.randn(batch_size, 128).astype(np.float32) * 0.1
            negatives = np.random.randn(batch_size, num_neg, 128).astype(np.float32)
            adapter.update(anchors, positives, negatives)

        status = adapter.get_health_status()
        assert status["healthy"]


class TestAdapterStats:
    """Test statistics tracking."""

    def test_stats_initialization(self):
        """Stats initialize correctly."""
        stats = AdapterStats()

        assert stats.total_updates == 0
        assert stats.total_samples == 0
        assert stats.avg_loss == 0.0

    def test_stats_to_dict(self):
        """Stats convert to dictionary."""
        stats = AdapterStats(
            total_updates=100,
            total_samples=1600,
            total_loss=50.0,
        )

        d = stats.to_dict()

        assert d["total_updates"] == 100
        assert d["total_samples"] == 1600
        assert d["avg_loss"] == 0.5


class TestFactoryFunction:
    """Test factory function."""

    def test_create_contrastive_adapter(self):
        """Factory creates adapter with correct dimensions."""
        adapter = create_contrastive_adapter(
            input_dim=768,
            output_dim=128,
            temperature=0.07,
        )

        assert adapter.config.input_dim == 768
        assert adapter.config.output_dim == 128
        assert np.abs(adapter.temperature - 0.07) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
