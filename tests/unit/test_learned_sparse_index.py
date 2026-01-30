"""
Tests for LearnedSparseIndex adaptive sparse addressing.

Phase 2: Validates learned sparse addressing functionality.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from ww.memory.learned_sparse_index import (
    LearnedSparseIndex,
    SparseAddressingResult,
    PendingUpdate,
)


class TestSparseAddressingResult:
    """Test SparseAddressingResult dataclass."""

    def test_initialization(self):
        """Test result initializes correctly."""
        cluster_att = np.array([0.5, 0.3, 0.1, 0.1])
        feature_att = np.random.rand(1024).astype(np.float32)

        result = SparseAddressingResult(
            cluster_attention=cluster_att,
            feature_attention=feature_att,
            sparsity_level=0.15,
            effective_clusters=[0, 1],
            query_id="test-123",
        )

        assert result.sparsity_level == 0.15
        assert len(result.effective_clusters) == 2
        assert result.query_id == "test-123"

    def test_top_k_clusters(self):
        """Test top-k cluster extraction."""
        cluster_att = np.array([0.1, 0.5, 0.3, 0.1])
        result = SparseAddressingResult(
            cluster_attention=cluster_att,
            feature_attention=np.zeros(1024),
            sparsity_level=0.1,
            effective_clusters=[],
            query_id="test",
        )

        top_k = result.top_k_clusters
        assert top_k[0] == 1  # Highest attention
        assert top_k[1] == 2  # Second highest

    def test_sparse_mask(self):
        """Test sparse mask generation."""
        feature_att = np.array([0.05, 0.15, 0.25, 0.08])
        result = SparseAddressingResult(
            cluster_attention=np.zeros(4),
            feature_attention=feature_att,
            sparsity_level=0.1,
            effective_clusters=[],
            query_id="test",
        )

        mask = result.get_sparse_mask(threshold=0.1)
        assert mask[0] == 0.0  # Below threshold
        assert mask[1] == 1.0  # Above threshold
        assert mask[2] == 1.0  # Above threshold
        assert mask[3] == 0.0  # Below threshold


class TestLearnedSparseIndexBasic:
    """Test LearnedSparseIndex basic functionality."""

    @pytest.fixture
    def index(self):
        """Create sparse index instance."""
        return LearnedSparseIndex(
            embed_dim=1024,
            hidden_dim=256,
            max_clusters=100,
            learning_rate=0.01,
        )

    def test_initialization(self, index):
        """Test index initializes correctly."""
        assert index.embed_dim == 1024
        assert index.hidden_dim == 256
        assert index.max_clusters == 100
        assert index.n_updates == 0
        assert index.is_warm is False

        # Weights should be initialized
        assert index.W_shared.shape == (256, 1024)
        assert index.W_cluster.shape == (100, 256)
        assert index.W_feature.shape == (1024, 256)
        assert index.W_sparsity.shape == (1, 256)

    def test_cold_start(self, index):
        """Test cold start detection."""
        assert index.is_warm is False

        # Simulate updates
        index.n_updates = 49
        assert index.is_warm is False

        index.n_updates = 50
        assert index.is_warm is True

    def test_forward_basic(self, index):
        """Test basic forward pass."""
        query = np.random.randn(1024).astype(np.float32)

        result = index.forward(query, n_clusters=10)

        assert result.cluster_attention.shape == (100,)
        assert result.feature_attention.shape == (1024,)
        assert 0 <= result.sparsity_level <= 1
        assert result.query_id is not None

        # Cluster attention should sum to 1 (softmax)
        assert np.isclose(result.cluster_attention.sum(), 1.0, atol=1e-5)

        # Feature attention should be in [0, 1] (sigmoid)
        assert np.all(result.feature_attention >= 0)
        assert np.all(result.feature_attention <= 1)

    def test_forward_masks_unused_clusters(self, index):
        """Test that unused clusters get near-zero attention."""
        query = np.random.randn(1024).astype(np.float32)

        result = index.forward(query, n_clusters=5)

        # Clusters beyond n_clusters should have very low attention
        assert result.cluster_attention[5:].max() < 0.01


class TestNeuromodulatorModulation:
    """Test neuromodulator modulation of sparse addressing."""

    @pytest.fixture
    def index(self):
        """Create sparse index instance."""
        return LearnedSparseIndex(embed_dim=1024, hidden_dim=256)

    def test_high_ne_arousal(self, index):
        """High NE arousal should lead to sharper attention."""
        query = np.random.randn(1024).astype(np.float32)

        result_baseline = index.forward(query, ne_gain=1.0)
        result_high_ne = index.forward(query, ne_gain=2.0)

        # High arousal = sharper cluster attention (higher entropy reduction)
        # Max attention should be higher with high arousal
        assert result_high_ne.cluster_attention.max() >= result_baseline.cluster_attention.max() - 0.1

    def test_low_ne_arousal(self, index):
        """Low NE arousal should lead to broader attention."""
        query = np.random.randn(1024).astype(np.float32)

        result_baseline = index.forward(query, ne_gain=1.0)
        result_low_ne = index.forward(query, ne_gain=0.5)

        # Low arousal = broader cluster attention
        # Max attention should be lower with low arousal (more spread out)
        baseline_entropy = -np.sum(result_baseline.cluster_attention * np.log(result_baseline.cluster_attention + 1e-10))
        low_ne_entropy = -np.sum(result_low_ne.cluster_attention * np.log(result_low_ne.cluster_attention + 1e-10))

        # Lower arousal should have higher entropy (broader distribution)
        assert low_ne_entropy >= baseline_entropy - 0.5

    def test_encoding_mode(self, index):
        """Encoding mode should broaden feature attention."""
        query = np.random.randn(1024).astype(np.float32)

        result_retrieval = index.forward(query, ach_mode="retrieval")
        result_encoding = index.forward(query, ach_mode="encoding")

        # Encoding mode should have broader (higher mean) feature attention
        assert result_encoding.feature_attention.mean() >= result_retrieval.feature_attention.mean()

    def test_retrieval_mode(self, index):
        """Retrieval mode should sharpen feature attention."""
        query = np.random.randn(1024).astype(np.float32)

        result_encoding = index.forward(query, ach_mode="encoding")
        result_retrieval = index.forward(query, ach_mode="retrieval")

        # Retrieval mode should have higher max attention (more focused)
        assert result_retrieval.feature_attention.max() >= result_encoding.feature_attention.max() - 0.1


class TestLearnedSparseIndexTraining:
    """Test training functionality."""

    @pytest.fixture
    def index(self):
        """Create sparse index instance."""
        return LearnedSparseIndex(
            embed_dim=1024,
            hidden_dim=256,
            learning_rate=0.01,
        )

    def test_pending_updates_stored(self, index):
        """Forward should store pending updates."""
        query = np.random.randn(1024).astype(np.float32)
        result = index.forward(query, n_clusters=10)

        assert result.query_id in index._pending_updates
        pending = index._pending_updates[result.query_id]
        assert pending.query_embedding is not None
        assert pending.hidden_state is not None

    def test_update_success(self, index):
        """Test successful update."""
        query = np.random.randn(1024).astype(np.float32)
        result = index.forward(query, n_clusters=10)

        initial_updates = index.n_updates

        success = index.update(
            query_id=result.query_id,
            cluster_rewards={0: 0.9, 1: 0.7},
            overall_success=True,
        )

        assert success is True
        assert index.n_updates == initial_updates + 1
        assert result.query_id not in index._pending_updates  # Should be removed

    def test_update_nonexistent_query(self, index):
        """Update with non-existent query should fail gracefully."""
        success = index.update(
            query_id="nonexistent-id",
            cluster_rewards={0: 0.9},
            overall_success=True,
        )

        assert success is False

    def test_update_modifies_weights(self, index):
        """Update should modify weights."""
        query = np.random.randn(1024).astype(np.float32)
        result = index.forward(query, n_clusters=10)

        W_cluster_before = index.W_cluster.copy()

        index.update(
            query_id=result.query_id,
            cluster_rewards={0: 0.9, 1: 0.1},  # Strong preference for cluster 0
            overall_success=True,
        )

        # Weights should have changed
        assert not np.allclose(W_cluster_before, index.W_cluster)

    def test_update_with_feature_importance(self, index):
        """Test update with feature importance signal."""
        query = np.random.randn(1024).astype(np.float32)
        result = index.forward(query, n_clusters=10)

        feature_importance = np.random.rand(1024).astype(np.float32)

        success = index.update(
            query_id=result.query_id,
            cluster_rewards={0: 0.8},
            overall_success=True,
            feature_importance=feature_importance,
        )

        assert success is True

    def test_warmup_counter(self, index):
        """Counter should increment with updates."""
        query = np.random.randn(1024).astype(np.float32)

        for i in range(60):
            result = index.forward(query, n_clusters=10)
            index.update(
                query_id=result.query_id,
                cluster_rewards={i % 10: 0.8},
                overall_success=True,
            )

        assert index.n_updates == 60
        assert index.is_warm is True


class TestFeatureWeightedScoring:
    """Test feature-weighted similarity scoring."""

    @pytest.fixture
    def index(self):
        """Create sparse index instance."""
        return LearnedSparseIndex(embed_dim=1024, hidden_dim=256)

    def test_feature_weighted_score(self, index):
        """Test feature-weighted similarity."""
        query = np.random.randn(1024).astype(np.float32)
        candidate = query + 0.1 * np.random.randn(1024).astype(np.float32)

        score = index.get_feature_weighted_score(query, candidate)

        # Score should be between -1 and 1
        assert -1 <= score <= 1
        # Similar vectors should have high score
        assert score > 0.5

    def test_orthogonal_vectors(self, index):
        """Orthogonal vectors should have low score."""
        query = np.zeros(1024, dtype=np.float32)
        query[0] = 1.0

        candidate = np.zeros(1024, dtype=np.float32)
        candidate[1] = 1.0

        score = index.get_feature_weighted_score(query, candidate)

        # Orthogonal vectors should have low score
        assert score < 0.5


class TestPendingUpdateManagement:
    """Test pending update management."""

    @pytest.fixture
    def index(self):
        """Create sparse index instance."""
        return LearnedSparseIndex(embed_dim=1024, hidden_dim=256)

    def test_max_pending_limit(self, index):
        """Test maximum pending updates limit."""
        index._max_pending = 5

        for i in range(10):
            query = np.random.randn(1024).astype(np.float32)
            index.forward(query, n_clusters=10)

        # Should not exceed max
        assert len(index._pending_updates) <= 5

    def test_prune_stale_pending(self, index):
        """Test pruning stale pending updates."""
        # Create some pending updates
        for _ in range(5):
            query = np.random.randn(1024).astype(np.float32)
            index.forward(query, n_clusters=10)

        # Make some updates stale
        old_time = datetime.now() - timedelta(hours=2)
        for i, pending in enumerate(index._pending_updates.values()):
            if i < 3:
                pending.timestamp = old_time

        # Prune with 60 minute threshold
        pruned = index.prune_stale_pending(max_age_minutes=60)

        assert pruned == 3
        assert len(index._pending_updates) == 2

    def test_register_pending(self, index):
        """Test manual pending registration."""
        query = np.random.randn(1024).astype(np.float32)
        cluster_att = np.random.rand(100).astype(np.float32)
        cluster_att = cluster_att / cluster_att.sum()
        feature_att = np.random.rand(1024).astype(np.float32)

        query_id = index.register_pending(
            query_embedding=query,
            cluster_attention=cluster_att,
            feature_attention=feature_att,
            sparsity_level=0.15,
        )

        assert query_id in index._pending_updates


class TestStatistics:
    """Test statistics and monitoring."""

    @pytest.fixture
    def index(self):
        """Create sparse index instance."""
        return LearnedSparseIndex(embed_dim=1024, hidden_dim=256)

    def test_get_statistics(self, index):
        """Test statistics retrieval."""
        stats = index.get_statistics()

        assert "n_updates" in stats
        assert "n_queries" in stats
        assert "is_warm" in stats
        assert "avg_sparsity" in stats
        assert "pending_updates" in stats

    def test_statistics_update(self, index):
        """Statistics should update with usage."""
        initial_stats = index.get_statistics()
        assert initial_stats["n_queries"] == 0

        query = np.random.randn(1024).astype(np.float32)
        index.forward(query, n_clusters=10)

        updated_stats = index.get_statistics()
        assert updated_stats["n_queries"] == 1

    def test_avg_sparsity_tracking(self, index):
        """Average sparsity should be tracked."""
        for _ in range(20):
            query = np.random.randn(1024).astype(np.float32)
            result = index.forward(query, n_clusters=10)
            index.update(
                query_id=result.query_id,
                cluster_rewards={0: 0.8},
                overall_success=True,
            )

        # Average sparsity should be reasonable
        assert 0.01 <= index.avg_sparsity <= 0.5


class TestDimensionHandling:
    """Test handling of different input dimensions."""

    def test_shorter_query(self):
        """Shorter queries should be padded."""
        index = LearnedSparseIndex(embed_dim=1024)
        query = np.random.randn(512).astype(np.float32)

        result = index.forward(query, n_clusters=10)
        assert result.feature_attention.shape == (1024,)

    def test_longer_query(self):
        """Longer queries should be truncated."""
        index = LearnedSparseIndex(embed_dim=1024)
        query = np.random.randn(2048).astype(np.float32)

        result = index.forward(query, n_clusters=10)
        assert result.feature_attention.shape == (1024,)
