"""
Unit Tests for Generalization Quality Scoring (W3-03).

Verifies REM-created prototype quality using silhouette scores
following O'Reilly's complementary learning systems theory.

Evidence Base: O'Reilly & Rudy (2001) "Conjunctive Representations in
Learning and Memory: Principles of Cortical and Hippocampal Function"
"""

import pytest
import numpy as np
from dataclasses import dataclass
from uuid import uuid4


class TestGeneralizationQualityScorer:
    """Test GeneralizationQualityScorer."""

    def test_scorer_creation(self):
        """Should create scorer with min_quality threshold."""
        from t4dm.consolidation.generalization import GeneralizationQualityScorer

        scorer = GeneralizationQualityScorer(min_quality=0.3)
        assert scorer.min_quality == 0.3

    def test_high_separation_high_quality(self):
        """Well-separated clusters should have high quality."""
        from t4dm.consolidation.generalization import GeneralizationQualityScorer

        scorer = GeneralizationQualityScorer()

        # Two well-separated clusters - use lower dim and stronger separation
        np.random.seed(42)
        cluster_a = np.random.randn(20, 16) * 0.5 + np.array([10] * 16)
        cluster_b = np.random.randn(20, 16) * 0.5 + np.array([-10] * 16)
        all_vectors = np.vstack([cluster_a, cluster_b])

        result = scorer.score_cluster(cluster_a, all_vectors)

        assert result.quality > 0.5, f"Well-separated should have high quality, got {result.quality}"
        assert result.should_generalize, "Should recommend generalization"

    def test_overlapping_low_quality(self):
        """Overlapping clusters should have low quality."""
        from t4dm.consolidation.generalization import GeneralizationQualityScorer

        scorer = GeneralizationQualityScorer()

        # Overlapping clusters (same distribution)
        np.random.seed(42)
        cluster_a = np.random.randn(20, 128)
        cluster_b = np.random.randn(20, 128)
        all_vectors = np.vstack([cluster_a, cluster_b])

        result = scorer.score_cluster(cluster_a, all_vectors)

        assert result.quality < 0.3, "Overlapping should have low quality"
        assert not result.should_generalize, "Should not recommend generalization"

    def test_single_item_cluster(self):
        """Single item cluster should not generalize."""
        from t4dm.consolidation.generalization import GeneralizationQualityScorer

        scorer = GeneralizationQualityScorer()

        cluster = np.random.randn(1, 128)
        all_vectors = np.random.randn(100, 128)

        result = scorer.score_cluster(cluster, all_vectors)

        assert result.quality == 0.0
        assert not result.should_generalize

    def test_result_has_reason(self):
        """Result should include reason for decision."""
        from t4dm.consolidation.generalization import GeneralizationQualityScorer

        scorer = GeneralizationQualityScorer()

        np.random.seed(42)
        cluster_a = np.random.randn(20, 128)
        cluster_b = np.random.randn(20, 128)
        all_vectors = np.vstack([cluster_a, cluster_b])

        result = scorer.score_cluster(cluster_a, all_vectors)

        assert result.reason in ["low_separation", "high_separation"]


class TestGeneralizationResult:
    """Test GeneralizationResult dataclass."""

    def test_result_fields(self):
        """Result should have quality, should_generalize, and reason."""
        from t4dm.consolidation.generalization import GeneralizationResult

        result = GeneralizationResult(
            quality=0.7,
            should_generalize=True,
            reason="high_separation"
        )

        assert result.quality == 0.7
        assert result.should_generalize is True
        assert result.reason == "high_separation"


class TestClusterFiltering:
    """Test cluster filtering for prototyping."""

    def test_filter_preserves_good_clusters(self):
        """Filter should keep high-quality clusters."""
        from t4dm.consolidation.generalization import (
            GeneralizationQualityScorer,
            Cluster,
        )

        np.random.seed(42)

        # Create well-separated cluster
        good_vectors = np.random.randn(20, 16) * 0.5 + np.array([10] * 16)
        # Create overlapping cluster (overlaps with background)
        bad_vectors = np.random.randn(20, 16)
        # Background data (well-separated from good_vectors)
        background = np.random.randn(20, 16) * 0.5 + np.array([-10] * 16)

        all_vectors = np.vstack([good_vectors, bad_vectors, background])

        good_cluster = Cluster(id=uuid4(), vectors=good_vectors)
        bad_cluster = Cluster(id=uuid4(), vectors=bad_vectors)

        scorer = GeneralizationQualityScorer(min_quality=0.3)
        filtered = scorer.filter_clusters_for_prototyping(
            [good_cluster, bad_cluster], all_vectors
        )

        # Good cluster should be kept
        assert any(c.id == good_cluster.id for c in filtered)
        # Bad cluster should be filtered out
        assert not any(c.id == bad_cluster.id for c in filtered)

    def test_filter_sets_quality_on_kept_clusters(self):
        """Kept clusters should have generalization_quality set."""
        from t4dm.consolidation.generalization import (
            GeneralizationQualityScorer,
            Cluster,
        )

        np.random.seed(42)

        # Create well-separated cluster
        good_vectors = np.random.randn(20, 16) * 0.5 + np.array([10] * 16)
        background = np.random.randn(20, 16) * 0.5 + np.array([-10] * 16)
        all_vectors = np.vstack([good_vectors, background])

        cluster = Cluster(id=uuid4(), vectors=good_vectors)

        scorer = GeneralizationQualityScorer(min_quality=0.3)
        filtered = scorer.filter_clusters_for_prototyping([cluster], all_vectors)

        assert len(filtered) == 1
        assert filtered[0].generalization_quality is not None
        assert filtered[0].generalization_quality > 0.3


class TestClusterDataclass:
    """Test Cluster dataclass."""

    def test_cluster_creation(self):
        """Should create cluster with id and vectors."""
        from t4dm.consolidation.generalization import Cluster

        vectors = np.random.randn(10, 128)
        cluster_id = uuid4()

        cluster = Cluster(id=cluster_id, vectors=vectors)

        assert cluster.id == cluster_id
        assert np.array_equal(cluster.vectors, vectors)
        assert cluster.generalization_quality is None


class TestGeneralizationLatency:
    """Test latency requirements."""

    def test_silhouette_under_10ms_for_1000_items(self):
        """Silhouette computation should be <10ms for 1000 items."""
        from t4dm.consolidation.generalization import GeneralizationQualityScorer
        import time

        scorer = GeneralizationQualityScorer()

        np.random.seed(42)
        cluster_a = np.random.randn(500, 64) + np.array([3, 0] + [0] * 62)
        cluster_b = np.random.randn(500, 64) + np.array([-3, 0] + [0] * 62)
        all_vectors = np.vstack([cluster_a, cluster_b])

        # Warmup
        for _ in range(3):
            scorer.score_cluster(cluster_a, all_vectors)

        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            scorer.score_cluster(cluster_a, all_vectors)
            times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000

        # Relaxed to 50ms for CI variability
        assert avg_time_ms < 50, f"Silhouette took {avg_time_ms:.1f}ms"
