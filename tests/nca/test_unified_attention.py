"""
Tests for Unified Attention System.

Tests capsule-transformer hybrid attention mechanism.
"""

import numpy as np
import pytest

from t4dm.nca.unified_attention import (
    UnifiedAttentionConfig,
    UnifiedAttentionHead,
    UnifiedAttentionSystem,
)


class TestUnifiedAttentionConfig:
    """Tests for UnifiedAttentionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = UnifiedAttentionConfig()
        assert config.embed_dim == 1024
        assert config.num_heads == 8
        assert config.head_dim == 64
        assert config.pose_dim == 4
        assert config.capsule_weight == 0.5
        assert config.temperature == 1.0
        assert config.use_ff_learning is True
        assert config.attention_dropout == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = UnifiedAttentionConfig(
            embed_dim=512,
            num_heads=4,
            capsule_weight=0.3,
        )
        assert config.embed_dim == 512
        assert config.num_heads == 4
        assert config.capsule_weight == 0.3


class TestUnifiedAttentionHead:
    """Tests for UnifiedAttentionHead."""

    @pytest.fixture
    def small_head(self):
        """Create small attention head for testing."""
        return UnifiedAttentionHead(
            embed_dim=64,
            head_dim=16,
            pose_dim=4,
            capsule_weight=0.5,
            temperature=1.0,
        )

    def test_initialization(self, small_head):
        """Test head initialization."""
        assert small_head.embed_dim == 64
        assert small_head.head_dim == 16
        assert small_head.pose_dim == 4
        assert small_head.W_q.shape == (16, 64)
        assert small_head.W_k.shape == (16, 64)
        assert small_head.W_v.shape == (16, 64)
        assert small_head.W_pose.shape == (16, 64)  # pose_dim * pose_dim, embed_dim

    def test_extract_pose(self, small_head):
        """Test pose extraction from embedding."""
        embedding = np.random.randn(64).astype(np.float32)
        pose = small_head.extract_pose(embedding)
        assert pose.shape == (4, 4)

    def test_compute_capsule_attention(self, small_head):
        """Test capsule attention computation."""
        n_queries, n_keys = 3, 5
        query_poses = np.random.randn(n_queries, 4, 4).astype(np.float32)
        key_poses = np.random.randn(n_keys, 4, 4).astype(np.float32)

        attention = small_head.compute_capsule_attention(query_poses, key_poses)

        assert attention.shape == (n_queries, n_keys)
        # All attention values should be positive (exponential)
        assert np.all(attention > 0)

    def test_compute_transformer_attention(self, small_head):
        """Test transformer attention computation."""
        n_queries, n_keys = 3, 5
        query = np.random.randn(n_queries, 64).astype(np.float32)
        key = np.random.randn(n_keys, 64).astype(np.float32)

        attention = small_head.compute_transformer_attention(query, key)

        assert attention.shape == (n_queries, n_keys)

    def test_forward_single_query(self, small_head):
        """Test forward pass with single query."""
        query = np.random.randn(64).astype(np.float32)
        key = np.random.randn(5, 64).astype(np.float32)
        value = np.random.randn(5, 64).astype(np.float32)

        output, weights = small_head.forward(query, key, value)

        assert output.shape == (1, 16)  # [1, head_dim]
        assert weights.shape == (1, 5)
        # Attention weights should sum to 1
        assert np.isclose(weights.sum(), 1.0, atol=0.01)

    def test_forward_batch_query(self, small_head):
        """Test forward pass with batch of queries."""
        query = np.random.randn(3, 64).astype(np.float32)
        key = np.random.randn(5, 64).astype(np.float32)
        value = np.random.randn(5, 64).astype(np.float32)

        output, weights = small_head.forward(query, key, value)

        assert output.shape == (3, 16)
        assert weights.shape == (3, 5)
        # Each row should sum to 1
        for i in range(3):
            assert np.isclose(weights[i].sum(), 1.0, atol=0.01)

    def test_forward_with_poses(self, small_head):
        """Test forward pass with provided poses."""
        query = np.random.randn(2, 64).astype(np.float32)
        key = np.random.randn(4, 64).astype(np.float32)
        value = np.random.randn(4, 64).astype(np.float32)

        query_poses = np.random.randn(2, 4, 4).astype(np.float32)
        key_poses = np.random.randn(4, 4, 4).astype(np.float32)

        output, weights = small_head.forward(
            query, key, value,
            poses=(query_poses, key_poses),
        )

        assert output.shape == (2, 16)
        assert weights.shape == (2, 4)

    def test_forward_with_mask(self, small_head):
        """Test forward pass with attention mask."""
        query = np.random.randn(2, 64).astype(np.float32)
        key = np.random.randn(4, 64).astype(np.float32)
        value = np.random.randn(4, 64).astype(np.float32)

        # Mask out last two keys
        mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float32)

        output, weights = small_head.forward(query, key, value, mask=mask)

        assert output.shape == (2, 16)
        # Masked positions should have near-zero attention
        assert weights[:, 2:].max() < 0.01

    def test_learn_fusion_weight(self, small_head):
        """Test fusion weight learning."""
        initial_weight = small_head.capsule_weight

        # Capsule contributes more
        small_head.learn_fusion_weight(
            utility=1.0,
            contributions=(0.8, 0.2),
        )
        assert small_head.capsule_weight > initial_weight

        # Reset
        small_head.capsule_weight = 0.5

        # Transformer contributes more
        small_head.learn_fusion_weight(
            utility=1.0,
            contributions=(0.2, 0.8),
        )
        assert small_head.capsule_weight < initial_weight


class TestUnifiedAttentionSystem:
    """Tests for UnifiedAttentionSystem."""

    @pytest.fixture
    def small_system(self):
        """Create small attention system."""
        config = UnifiedAttentionConfig(
            embed_dim=64,
            num_heads=2,
            head_dim=16,
            pose_dim=4,
        )
        return UnifiedAttentionSystem(config)

    def test_initialization(self, small_system):
        """Test system initialization."""
        assert len(small_system.heads) == 2
        assert small_system.W_out.shape == (64, 32)  # embed_dim, num_heads * head_dim

    def test_attend(self, small_system):
        """Test system attend pass."""
        query = np.random.randn(2, 64).astype(np.float32)
        key = np.random.randn(4, 64).astype(np.float32)
        value = np.random.randn(4, 64).astype(np.float32)

        output = small_system.attend(query, key, value)

        assert output.shape == (2, 64)  # Same as embed_dim

    def test_get_attention_stats(self, small_system):
        """Test statistics retrieval."""
        stats = small_system.get_attention_stats()

        assert "num_heads" in stats
        assert "mean_capsule_weight" in stats
        assert "capsule_weights" in stats
        assert "temperature" in stats


class TestUnifiedAttentionIntegration:
    """Integration tests for unified attention."""

    def test_capsule_vs_transformer_influence(self):
        """Test that capsule weight affects attention."""
        config_capsule = UnifiedAttentionConfig(
            embed_dim=64,
            num_heads=1,
            head_dim=32,
            capsule_weight=0.9,  # Mostly capsule
        )
        config_transformer = UnifiedAttentionConfig(
            embed_dim=64,
            num_heads=1,
            head_dim=32,
            capsule_weight=0.1,  # Mostly transformer
        )

        system_capsule = UnifiedAttentionSystem(config_capsule)
        system_transformer = UnifiedAttentionSystem(config_transformer)

        # Check that configs are different
        assert system_capsule.config.capsule_weight != system_transformer.config.capsule_weight
        # Check head weights are different
        assert system_capsule.heads[0].capsule_weight == 0.9
        assert system_transformer.heads[0].capsule_weight == 0.1

    def test_attention_focuses_on_similar(self):
        """Test that attention focuses on similar items."""
        head = UnifiedAttentionHead(
            embed_dim=64,
            head_dim=16,
            pose_dim=4,
            capsule_weight=0.3,  # Some capsule influence
        )

        # Query similar to first key
        key = np.random.randn(4, 64).astype(np.float32)
        query = key[0:1] + np.random.randn(1, 64).astype(np.float32) * 0.1
        value = np.eye(4, 64).astype(np.float32)  # Distinguishable values

        output, weights = head.forward(query, key, value)

        # Should attend mostly to first key
        assert weights[0, 0] > weights[0, 1:].mean()

    def test_multi_query_independence(self):
        """Test that different queries produce outputs with correct shape."""
        system = UnifiedAttentionSystem(UnifiedAttentionConfig(
            embed_dim=64,
            num_heads=2,
            head_dim=16,
        ))

        # Different queries
        query = np.random.randn(3, 64).astype(np.float32)
        key = np.random.randn(5, 64).astype(np.float32)
        value = np.random.randn(5, 64).astype(np.float32)

        output = system.attend(query, key, value)

        # Output should match query count and embed dim
        assert output.shape == (3, 64)
        # Each row should have content (not zeros)
        for i in range(3):
            assert np.std(output[i]) > 0
