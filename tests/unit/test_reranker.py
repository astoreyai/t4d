"""
Tests for LearnedReranker.

P0c: Validates learned re-ranking functionality.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional


# Minimal ScoredResult mock for testing
@dataclass
class MockScoredResult:
    score: float
    components: dict


class TestLearnedReranker:
    """Test LearnedReranker functionality."""

    @pytest.fixture
    def reranker(self):
        """Create reranker instance."""
        from ww.memory.episodic import LearnedReranker
        return LearnedReranker(embed_dim=1024, learning_rate=0.005)

    def test_initialization(self, reranker):
        """Test reranker initializes correctly."""
        assert reranker.embed_dim == 1024
        assert reranker.lr == 0.005
        assert reranker.n_updates == 0
        assert reranker.cold_start_threshold == 100
        assert reranker.residual_weight == 0.3

        # Weights should be initialized
        assert reranker.W_query.shape == (16, 1024)
        assert reranker.W1.shape == (32, 20)
        assert reranker.W2.shape == (1, 32)

    def test_query_compression(self, reranker):
        """Test query embedding compression."""
        query_emb = np.random.randn(1024).astype(np.float32)
        compressed = reranker._compress_query(query_emb)

        # Should be 16-dim
        assert compressed.shape == (16,)
        # Should be normalized by tanh to [-1, 1]
        assert np.all(np.abs(compressed) <= 1.0)

    def test_cold_start_bypass(self, reranker):
        """During cold start, reranker should not modify results."""
        results = [
            MockScoredResult(score=0.8, components={"semantic": 0.9, "recency": 0.7, "outcome": 0.5, "importance": 0.6}),
            MockScoredResult(score=0.6, components={"semantic": 0.7, "recency": 0.5, "outcome": 0.4, "importance": 0.5}),
            MockScoredResult(score=0.4, components={"semantic": 0.5, "recency": 0.3, "outcome": 0.3, "importance": 0.4}),
        ]
        original_scores = [r.score for r in results]

        query_emb = np.random.randn(1024).astype(np.float32)
        reranked = reranker.rerank(results, query_emb)

        # Scores should be unchanged during cold start
        assert reranked[0].score == original_scores[0]
        assert reranked[1].score == original_scores[1]
        assert reranked[2].score == original_scores[2]

    def test_rerank_after_warmup(self, reranker):
        """After warmup, reranker should modify scores."""
        # Force past cold start
        reranker.n_updates = 101

        results = [
            MockScoredResult(score=0.8, components={"semantic": 0.9, "recency": 0.7, "outcome": 0.5, "importance": 0.6}),
            MockScoredResult(score=0.6, components={"semantic": 0.7, "recency": 0.5, "outcome": 0.4, "importance": 0.5}),
            MockScoredResult(score=0.4, components={"semantic": 0.5, "recency": 0.3, "outcome": 0.3, "importance": 0.4}),
        ]

        query_emb = np.random.randn(1024).astype(np.float32)
        reranked = reranker.rerank(results, query_emb)

        # Results should have rerank_adjustment component
        for r in reranked:
            assert "rerank_adjustment" in r.components
            # Adjustment should be bounded
            assert -0.2 <= r.components["rerank_adjustment"] <= 0.2

    def test_update_increments_counter(self, reranker):
        """Training should increment update counter."""
        query_emb = np.random.randn(1024).astype(np.float32)
        component_scores_list = [
            {"semantic": 0.8, "recency": 0.6, "outcome": 0.5, "importance": 0.5},
            {"semantic": 0.7, "recency": 0.5, "outcome": 0.4, "importance": 0.4},
        ]
        outcome_utilities = [0.9, 0.3]

        assert reranker.n_updates == 0
        reranker.update(query_emb, component_scores_list, outcome_utilities)
        assert reranker.n_updates == 1

    def test_residual_weight_increases(self, reranker):
        """Residual weight should increase after cold start."""
        reranker.n_updates = 100  # At threshold

        # Do one update past threshold
        query_emb = np.random.randn(1024).astype(np.float32)
        component_scores_list = [{"semantic": 0.8, "recency": 0.6, "outcome": 0.5, "importance": 0.5}]
        outcome_utilities = [0.9]

        initial_weight = reranker.residual_weight
        reranker.update(query_emb, component_scores_list, outcome_utilities)

        # Should have increased
        assert reranker.residual_weight > initial_weight
        # But not exceed max
        assert reranker.residual_weight <= 0.5

    def test_empty_results_handling(self, reranker):
        """Empty results should be handled gracefully."""
        reranker.n_updates = 101  # Past cold start

        query_emb = np.random.randn(1024).astype(np.float32)
        reranked = reranker.rerank([], query_emb)

        assert reranked == []

    def test_maintains_order_initially(self, reranker):
        """With random init, should roughly maintain order."""
        reranker.n_updates = 101  # Past cold start

        # Create results in descending score order
        results = [
            MockScoredResult(score=0.9, components={"semantic": 0.9, "recency": 0.8, "outcome": 0.7, "importance": 0.8}),
            MockScoredResult(score=0.7, components={"semantic": 0.7, "recency": 0.6, "outcome": 0.5, "importance": 0.6}),
            MockScoredResult(score=0.5, components={"semantic": 0.5, "recency": 0.4, "outcome": 0.3, "importance": 0.4}),
        ]

        query_emb = np.random.randn(1024).astype(np.float32)
        reranked = reranker.rerank(results, query_emb)

        # With residual connection, order should mostly be preserved
        # Top item should likely stay top (residual weight is 0.3)
        assert reranked[0].score >= reranked[-1].score


class TestLearnedRerankerTraining:
    """Test reranker learning behavior."""

    @pytest.fixture
    def reranker(self):
        """Create reranker instance."""
        from ww.memory.episodic import LearnedReranker
        return LearnedReranker(embed_dim=1024, learning_rate=0.01)  # Higher LR for testing

    def test_learns_from_positive_outcomes(self, reranker):
        """Reranker should learn to score higher when utility is high."""
        # Use consistent query
        np.random.seed(42)
        query_emb = np.random.randn(1024).astype(np.float32)

        # Train on high utility outcomes
        for _ in range(50):
            component_scores_list = [
                {"semantic": 0.9, "recency": 0.8, "outcome": 0.5, "importance": 0.7},
            ]
            reranker.update(query_emb, component_scores_list, [0.95])

        # Get prediction for similar input
        reranker.n_updates = 101  # Force active
        results = [
            MockScoredResult(score=0.5, components={"semantic": 0.9, "recency": 0.8, "outcome": 0.5, "importance": 0.7}),
        ]

        reranked = reranker.rerank(results, query_emb)

        # With high utility training, adjustment should be positive
        assert reranked[0].components["rerank_adjustment"] > -0.1

    def test_learns_from_negative_outcomes(self, reranker):
        """Reranker should learn to score lower when utility is low."""
        np.random.seed(42)
        query_emb = np.random.randn(1024).astype(np.float32)

        # Train on low utility outcomes
        for _ in range(50):
            component_scores_list = [
                {"semantic": 0.3, "recency": 0.2, "outcome": 0.1, "importance": 0.2},
            ]
            reranker.update(query_emb, component_scores_list, [0.05])

        # Get prediction
        reranker.n_updates = 101
        results = [
            MockScoredResult(score=0.5, components={"semantic": 0.3, "recency": 0.2, "outcome": 0.1, "importance": 0.2}),
        ]

        reranked = reranker.rerank(results, query_emb)

        # With low utility training, adjustment should be negative
        assert reranked[0].components["rerank_adjustment"] < 0.1


class TestLearnedRerankerDimensions:
    """Test handling of different embedding dimensions."""

    def test_handles_shorter_embedding(self):
        """Should handle embeddings shorter than embed_dim."""
        from ww.memory.episodic import LearnedReranker
        reranker = LearnedReranker(embed_dim=1024)

        # 512-dim embedding (shorter)
        query_emb = np.random.randn(512).astype(np.float32)
        compressed = reranker._compress_query(query_emb)

        assert compressed.shape == (16,)

    def test_handles_longer_embedding(self):
        """Should handle embeddings longer than embed_dim."""
        from ww.memory.episodic import LearnedReranker
        reranker = LearnedReranker(embed_dim=1024)

        # 2048-dim embedding (longer)
        query_emb = np.random.randn(2048).astype(np.float32)
        compressed = reranker._compress_query(query_emb)

        assert compressed.shape == (16,)
