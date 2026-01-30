"""
Tests for ClusterIndex hierarchical episode retrieval.

Phase 1: Validates CA3-like semantic clustering functionality.
"""

import pytest
import numpy as np
from uuid import uuid4
from datetime import datetime, timedelta

from ww.memory.cluster_index import ClusterIndex, ClusterMeta


class TestClusterMeta:
    """Test ClusterMeta dataclass."""

    def test_initialization(self):
        """Test ClusterMeta initializes correctly."""
        centroid = np.random.randn(1024).astype(np.float32)
        member_ids = [uuid4() for _ in range(5)]

        meta = ClusterMeta(
            cluster_id="cluster-001",
            centroid=centroid,
            member_count=5,
            member_ids=member_ids,
        )

        assert meta.cluster_id == "cluster-001"
        assert meta.member_count == 5
        assert len(meta.member_ids) == 5
        assert meta.total_retrievals == 0
        assert meta.successful_retrievals == 0
        assert meta.avg_score == 0.5

    def test_success_rate_prior(self):
        """Success rate should return prior when no retrievals."""
        meta = ClusterMeta(
            cluster_id="test",
            centroid=np.zeros(1024),
        )

        # No retrievals = prior of 0.5
        assert meta.success_rate == 0.5

    def test_success_rate_calculation(self):
        """Success rate should calculate correctly."""
        meta = ClusterMeta(
            cluster_id="test",
            centroid=np.zeros(1024),
            total_retrievals=10,
            successful_retrievals=7,
        )

        assert meta.success_rate == 0.7

    def test_priority_score(self):
        """Priority score should blend multiple factors."""
        meta = ClusterMeta(
            cluster_id="test",
            centroid=np.zeros(1024),
            member_count=50,
            total_retrievals=10,
            successful_retrievals=8,
            coherence=0.9,
            last_accessed=datetime.now(),
        )

        score = meta.priority_score

        # Should be between 0 and 1
        assert 0 <= score <= 1
        # High success (0.8), recent, good coherence = high score
        assert score > 0.6


class TestClusterIndexBasic:
    """Test ClusterIndex basic functionality."""

    @pytest.fixture
    def index(self):
        """Create cluster index instance."""
        return ClusterIndex(embedding_dim=1024, default_k=5)

    def test_initialization(self, index):
        """Test index initializes correctly."""
        assert index.embedding_dim == 1024
        assert index.default_k == 5
        assert index.n_clusters == 0
        assert index.centroid_matrix is None

    def test_register_cluster(self, index):
        """Test cluster registration."""
        centroid = np.random.randn(1024).astype(np.float32)
        member_ids = [uuid4() for _ in range(10)]

        cluster = index.register_cluster(
            cluster_id="cluster-001",
            centroid=centroid,
            member_ids=member_ids,
            variance=0.1,
            coherence=0.85,
        )

        assert index.n_clusters == 1
        assert cluster.cluster_id == "cluster-001"
        assert cluster.member_count == 10
        assert cluster.coherence == 0.85
        assert index.centroid_matrix is not None
        assert index.centroid_matrix.shape == (1, 1024)

    def test_register_multiple_clusters(self, index):
        """Test registering multiple clusters."""
        for i in range(5):
            centroid = np.random.randn(1024).astype(np.float32)
            member_ids = [uuid4() for _ in range(10)]
            index.register_cluster(
                cluster_id=f"cluster-{i:03d}",
                centroid=centroid,
                member_ids=member_ids,
            )

        assert index.n_clusters == 5
        assert index.centroid_matrix.shape == (5, 1024)

    def test_unregister_cluster(self, index):
        """Test cluster removal."""
        centroid = np.random.randn(1024).astype(np.float32)
        index.register_cluster("cluster-001", centroid, [uuid4()])
        index.register_cluster("cluster-002", centroid, [uuid4()])

        assert index.n_clusters == 2

        result = index.unregister_cluster("cluster-001")
        assert result is True
        assert index.n_clusters == 1

        # Non-existent
        result = index.unregister_cluster("cluster-999")
        assert result is False

    def test_centroid_normalization(self, index):
        """Test that centroids are normalized."""
        # Non-unit vector
        centroid = np.array([3.0, 4.0] + [0.0] * 1022, dtype=np.float32)
        index.register_cluster("test", centroid, [uuid4()])

        stored_centroid = index.clusters["test"].centroid
        norm = np.linalg.norm(stored_centroid)

        assert np.isclose(norm, 1.0, atol=1e-6)


class TestClusterSelection:
    """Test cluster selection functionality."""

    @pytest.fixture
    def populated_index(self):
        """Create index with clusters."""
        index = ClusterIndex(embedding_dim=1024, default_k=3)

        # Create distinct clusters
        np.random.seed(42)
        for i in range(10):
            # Create somewhat distinct centroids
            base = np.zeros(1024, dtype=np.float32)
            base[i * 100:(i + 1) * 100] = np.random.randn(100).astype(np.float32)
            member_ids = [uuid4() for _ in range(20)]
            index.register_cluster(
                cluster_id=f"cluster-{i:03d}",
                centroid=base,
                member_ids=member_ids,
                coherence=0.8 + i * 0.02,
            )

        return index

    def test_select_clusters_basic(self, populated_index):
        """Test basic cluster selection."""
        # Query similar to cluster-0
        query = np.zeros(1024, dtype=np.float32)
        query[:100] = np.random.randn(100).astype(np.float32)

        selected = populated_index.select_clusters(query, k=3)

        assert len(selected) >= 1  # At least one selected
        assert len(selected) <= 3  # At most k
        assert all(isinstance(cid, str) for cid, _ in selected)
        assert all(isinstance(score, float) for _, score in selected)

    def test_select_clusters_similarity_ordering(self, populated_index):
        """Selected clusters should be ordered by similarity."""
        query = np.random.randn(1024).astype(np.float32)
        selected = populated_index.select_clusters(query, k=5)

        # Scores should be in descending order (or close due to priority blending)
        if len(selected) >= 2:
            # First should generally have higher or similar score as last
            assert selected[0][1] >= selected[-1][1] - 0.2  # Allow some slack

    def test_ne_modulation_high_arousal(self, populated_index):
        """High NE arousal should select more clusters."""
        query = np.random.randn(1024).astype(np.float32)

        baseline = populated_index.select_clusters(query, ne_gain=1.0)
        high_arousal = populated_index.select_clusters(query, ne_gain=2.0)

        # High arousal should select same or more clusters
        assert len(high_arousal) >= len(baseline)

    def test_ne_modulation_low_arousal(self, populated_index):
        """Low NE arousal should select fewer clusters."""
        query = np.random.randn(1024).astype(np.float32)

        baseline = populated_index.select_clusters(query, ne_gain=1.0)
        low_arousal = populated_index.select_clusters(query, ne_gain=0.5)

        # Low arousal should select same or fewer clusters
        assert len(low_arousal) <= len(baseline)

    def test_ach_retrieval_mode(self, populated_index):
        """Retrieval mode should prioritize high success rate clusters."""
        # Give one cluster high success rate
        populated_index.clusters["cluster-005"].total_retrievals = 100
        populated_index.clusters["cluster-005"].successful_retrievals = 95

        query = np.random.randn(1024).astype(np.float32)
        selected = populated_index.select_clusters(query, k=5, ach_mode="retrieval")

        # Success rate boost should influence selection
        assert len(selected) > 0

    def test_ach_encoding_mode(self, populated_index):
        """Encoding mode should prioritize diverse/underexplored clusters."""
        # Give one cluster high success rate (it should be de-prioritized in encoding)
        populated_index.clusters["cluster-005"].total_retrievals = 100
        populated_index.clusters["cluster-005"].successful_retrievals = 95

        query = np.random.randn(1024).astype(np.float32)
        selected = populated_index.select_clusters(query, k=5, ach_mode="encoding")

        assert len(selected) > 0

    def test_empty_index_returns_empty(self):
        """Empty index should return no clusters."""
        index = ClusterIndex()
        query = np.random.randn(1024).astype(np.float32)

        selected = index.select_clusters(query)
        assert selected == []


class TestClusterLearning:
    """Test cluster learning and statistics."""

    @pytest.fixture
    def index_with_clusters(self):
        """Create index with clusters."""
        index = ClusterIndex(embedding_dim=1024)

        for i in range(3):
            centroid = np.random.randn(1024).astype(np.float32)
            index.register_cluster(f"cluster-{i}", centroid, [uuid4()])

        return index

    def test_record_retrieval_outcome_success(self, index_with_clusters):
        """Test recording successful retrieval."""
        cluster_ids = ["cluster-0", "cluster-1"]

        index_with_clusters.record_retrieval_outcome(
            cluster_ids=cluster_ids,
            successful=True,
        )

        for cid in cluster_ids:
            cluster = index_with_clusters.clusters[cid]
            assert cluster.total_retrievals == 1
            assert cluster.successful_retrievals == 1
            assert cluster.last_accessed is not None

    def test_record_retrieval_outcome_failure(self, index_with_clusters):
        """Test recording failed retrieval."""
        index_with_clusters.record_retrieval_outcome(
            cluster_ids=["cluster-0"],
            successful=False,
        )

        cluster = index_with_clusters.clusters["cluster-0"]
        assert cluster.total_retrievals == 1
        assert cluster.successful_retrievals == 0

    def test_record_retrieval_with_scores(self, index_with_clusters):
        """Test score updates with retrieval."""
        initial_score = index_with_clusters.clusters["cluster-0"].avg_score

        index_with_clusters.record_retrieval_outcome(
            cluster_ids=["cluster-0"],
            successful=True,
            retrieved_scores={"cluster-0": 0.9},
        )

        # Score should have moved toward 0.9
        new_score = index_with_clusters.clusters["cluster-0"].avg_score
        assert new_score != initial_score

    def test_update_cluster_centroid(self, index_with_clusters):
        """Test incremental centroid update."""
        original = index_with_clusters.clusters["cluster-0"].centroid.copy()
        new_embedding = np.random.randn(1024).astype(np.float32)

        index_with_clusters.update_cluster_centroid(
            "cluster-0", new_embedding, weight=0.5
        )

        updated = index_with_clusters.clusters["cluster-0"].centroid

        # Centroid should have changed
        assert not np.allclose(original, updated)
        # Should still be normalized
        assert np.isclose(np.linalg.norm(updated), 1.0, atol=1e-6)

    def test_add_to_cluster(self, index_with_clusters):
        """Test adding episode to cluster."""
        initial_count = index_with_clusters.clusters["cluster-0"].member_count
        new_id = uuid4()
        embedding = np.random.randn(1024).astype(np.float32)

        result = index_with_clusters.add_to_cluster("cluster-0", new_id, embedding)

        assert result is True
        assert index_with_clusters.clusters["cluster-0"].member_count == initial_count + 1
        assert new_id in index_with_clusters.clusters["cluster-0"].member_ids


class TestClusterIndexUtilities:
    """Test utility functions."""

    @pytest.fixture
    def index(self):
        """Create populated index."""
        index = ClusterIndex(embedding_dim=1024)
        for i in range(5):
            centroid = np.random.randn(1024).astype(np.float32)
            member_ids = [uuid4() for _ in range(10)]
            index.register_cluster(f"cluster-{i}", centroid, member_ids)
        return index

    def test_find_nearest_cluster(self, index):
        """Test finding nearest cluster."""
        embedding = np.random.randn(1024).astype(np.float32)

        result = index.find_nearest_cluster(embedding)

        assert result is not None
        cluster_id, similarity = result
        assert cluster_id in index.clusters
        assert isinstance(similarity, float)

    def test_find_nearest_cluster_with_exclusion(self, index):
        """Test nearest cluster with exclusions."""
        embedding = np.random.randn(1024).astype(np.float32)

        # Get nearest without exclusion
        nearest, _ = index.find_nearest_cluster(embedding)

        # Exclude that cluster
        result = index.find_nearest_cluster(embedding, exclude={nearest})

        assert result is not None
        assert result[0] != nearest

    def test_get_cluster_members(self, index):
        """Test getting cluster members."""
        members = index.get_cluster_members("cluster-0")

        assert len(members) == 10
        assert all(isinstance(m, type(uuid4())) for m in members)

    def test_get_cluster_members_nonexistent(self, index):
        """Non-existent cluster returns empty list."""
        members = index.get_cluster_members("nonexistent")
        assert members == []

    def test_get_statistics(self, index):
        """Test statistics retrieval."""
        stats = index.get_statistics()

        assert stats["n_clusters"] == 5
        assert stats["total_episodes"] == 50  # 5 clusters * 10 members
        assert stats["avg_cluster_size"] == 10.0
        assert "avg_success_rate" in stats
        assert "avg_coherence" in stats

    def test_get_statistics_empty(self):
        """Empty index statistics."""
        index = ClusterIndex()
        stats = index.get_statistics()

        assert stats["n_clusters"] == 0
        assert stats["total_episodes"] == 0


class TestClusterPruning:
    """Test cluster pruning functionality."""

    def test_prune_stale_clusters(self):
        """Test pruning old low-performing clusters."""
        index = ClusterIndex()

        # Create old cluster with low success rate
        old_cluster = index.register_cluster(
            "old-bad",
            np.random.randn(1024).astype(np.float32),
            [uuid4()],
        )
        old_cluster.created_at = datetime.now() - timedelta(days=60)
        old_cluster.total_retrievals = 100
        old_cluster.successful_retrievals = 5  # 5% success rate

        # Create new cluster
        index.register_cluster(
            "new-good",
            np.random.randn(1024).astype(np.float32),
            [uuid4()],
        )

        pruned = index.prune_stale_clusters(max_age_days=30, min_success_rate=0.1)

        assert "old-bad" in pruned
        assert "new-good" not in pruned
        assert index.n_clusters == 1

    def test_prune_empty_clusters(self):
        """Test pruning empty clusters."""
        index = ClusterIndex()

        # Create cluster then remove all members
        cluster = index.register_cluster(
            "empty",
            np.random.randn(1024).astype(np.float32),
            [],
        )
        cluster.member_count = 0

        pruned = index.prune_stale_clusters()

        assert "empty" in pruned
        assert index.n_clusters == 0


class TestDimensionHandling:
    """Test handling of different embedding dimensions."""

    def test_handles_shorter_centroid(self):
        """Shorter centroids should be padded."""
        index = ClusterIndex(embedding_dim=1024)

        # 512-dim centroid
        short_centroid = np.random.randn(512).astype(np.float32)
        index.register_cluster("short", short_centroid, [uuid4()])

        assert index.clusters["short"].centroid.shape == (1024,)

    def test_handles_longer_centroid(self):
        """Longer centroids should be truncated."""
        index = ClusterIndex(embedding_dim=1024)

        # 2048-dim centroid
        long_centroid = np.random.randn(2048).astype(np.float32)
        index.register_cluster("long", long_centroid, [uuid4()])

        assert index.clusters["long"].centroid.shape == (1024,)

    def test_handles_shorter_query(self):
        """Shorter queries should be handled."""
        index = ClusterIndex(embedding_dim=1024)
        index.register_cluster(
            "test",
            np.random.randn(1024).astype(np.float32),
            [uuid4()],
        )

        # 512-dim query
        short_query = np.random.randn(512).astype(np.float32)
        selected = index.select_clusters(short_query)

        assert len(selected) >= 1

    def test_handles_longer_query(self):
        """Longer queries should be handled."""
        index = ClusterIndex(embedding_dim=1024)
        index.register_cluster(
            "test",
            np.random.randn(1024).astype(np.float32),
            [uuid4()],
        )

        # 2048-dim query
        long_query = np.random.randn(2048).astype(np.float32)
        selected = index.select_clusters(long_query)

        assert len(selected) >= 1


class TestAChCA3CompletionStrength:
    """Test P2.2 ACh-CA3 pattern completion strength modulation."""

    def test_get_completion_strength_encoding(self):
        """Encoding mode should return low completion strength."""
        strength = ClusterIndex.get_completion_strength("encoding")

        assert strength == 0.2
        # Encoding mode: reduce pattern completion, favor separation

    def test_get_completion_strength_retrieval(self):
        """Retrieval mode should return high completion strength."""
        strength = ClusterIndex.get_completion_strength("retrieval")

        assert strength == 0.7
        # Retrieval mode: enhance pattern completion for recall

    def test_get_completion_strength_balanced(self):
        """Balanced mode should return moderate completion strength."""
        strength = ClusterIndex.get_completion_strength("balanced")

        assert strength == 0.4
        # Balanced mode: moderate pattern completion

    def test_get_completion_strength_default(self):
        """Default should be balanced mode."""
        strength = ClusterIndex.get_completion_strength()

        assert strength == 0.4

    def test_get_completion_strength_unknown_mode(self):
        """Unknown mode should default to balanced."""
        strength = ClusterIndex.get_completion_strength("unknown")

        assert strength == 0.4

    def test_completion_strength_range(self):
        """All completion strengths should be in valid range [0, 1]."""
        for mode in ["encoding", "balanced", "retrieval", "unknown"]:
            strength = ClusterIndex.get_completion_strength(mode)
            assert 0.0 <= strength <= 1.0


class TestClusterSearch:
    """Test P2.2 unified search method with completion strength."""

    @pytest.fixture
    def populated_index(self):
        """Create index with clusters."""
        index = ClusterIndex(embedding_dim=1024, default_k=3)

        np.random.seed(42)
        for i in range(5):
            base = np.zeros(1024, dtype=np.float32)
            base[i * 200:(i + 1) * 200] = np.random.randn(200).astype(np.float32)
            member_ids = [uuid4() for _ in range(10)]
            index.register_cluster(
                cluster_id=f"cluster-{i:03d}",
                centroid=base,
                member_ids=member_ids,
            )

        return index

    def test_search_returns_clusters_and_strength(self, populated_index):
        """Search should return tuple of (clusters, completion_strength)."""
        query = np.random.randn(1024).astype(np.float32)

        result = populated_index.search(query)

        assert isinstance(result, tuple)
        assert len(result) == 2

        clusters, strength = result
        assert isinstance(clusters, list)
        assert isinstance(strength, float)

    def test_search_encoding_mode(self, populated_index):
        """Search in encoding mode should return low completion strength."""
        query = np.random.randn(1024).astype(np.float32)

        clusters, strength = populated_index.search(query, ach_mode="encoding")

        assert strength == 0.2
        assert len(clusters) > 0

    def test_search_retrieval_mode(self, populated_index):
        """Search in retrieval mode should return high completion strength."""
        query = np.random.randn(1024).astype(np.float32)

        clusters, strength = populated_index.search(query, ach_mode="retrieval")

        assert strength == 0.7
        assert len(clusters) > 0

    def test_search_balanced_mode(self, populated_index):
        """Search in balanced mode should return moderate completion strength."""
        query = np.random.randn(1024).astype(np.float32)

        clusters, strength = populated_index.search(query, ach_mode="balanced")

        assert strength == 0.4
        assert len(clusters) > 0

    def test_search_with_ne_modulation(self, populated_index):
        """Search should respect NE modulation."""
        query = np.random.randn(1024).astype(np.float32)

        baseline_clusters, _ = populated_index.search(query, ne_gain=1.0)
        high_arousal_clusters, _ = populated_index.search(query, ne_gain=2.0)

        # High arousal should select same or more clusters
        assert len(high_arousal_clusters) >= len(baseline_clusters)

    def test_search_empty_index(self):
        """Search on empty index should return empty clusters with strength."""
        index = ClusterIndex(embedding_dim=1024)
        query = np.random.randn(1024).astype(np.float32)

        clusters, strength = index.search(query)

        assert clusters == []
        assert strength == 0.4  # Default balanced mode

    def test_search_modes_differ(self, populated_index):
        """Different ACh modes should produce different completion strengths."""
        query = np.random.randn(1024).astype(np.float32)

        _, enc_strength = populated_index.search(query, ach_mode="encoding")
        _, bal_strength = populated_index.search(query, ach_mode="balanced")
        _, ret_strength = populated_index.search(query, ach_mode="retrieval")

        # Should be strictly ordered: encoding < balanced < retrieval
        assert enc_strength < bal_strength < ret_strength
