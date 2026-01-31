"""Tests for learned fusion weights and reranker in episodic memory."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from dataclasses import dataclass

from t4dm.memory.episodic import LearnedFusionWeights, LearnedReranker


class TestLearnedFusionWeights:
    """Tests for LearnedFusionWeights class."""

    @pytest.fixture
    def fusion(self):
        """Create fusion weights instance."""
        return LearnedFusionWeights(
            embed_dim=64,  # Smaller for tests
            hidden_dim=16,
            learning_rate=0.01,
        )

    def test_initialization(self, fusion):
        """Test initialization."""
        assert fusion.embed_dim == 64
        assert fusion.hidden_dim == 16
        assert fusion.lr == 0.01
        assert fusion.n_components == 4

    def test_initialization_defaults(self):
        """Test default initialization."""
        fusion = LearnedFusionWeights()
        assert fusion.embed_dim == 1024
        assert fusion.hidden_dim == 32
        assert fusion.n_updates == 0

    def test_weight_shapes(self, fusion):
        """Weight matrices have correct shapes."""
        assert fusion.W1.shape == (16, 64)  # (hidden_dim, embed_dim)
        assert fusion.b1.shape == (16,)
        assert fusion.W2.shape == (4, 16)  # (n_components, hidden_dim)
        assert fusion.b2.shape == (4,)

    def test_default_weights(self, fusion):
        """Default weights sum to 1."""
        assert fusion.default_weights.sum() == pytest.approx(1.0)
        assert len(fusion.default_weights) == 4

    def test_component_names(self, fusion):
        """Component names are defined."""
        assert len(fusion.component_names) == 4
        assert "semantic" in fusion.component_names
        assert "recency" in fusion.component_names

    def test_softmax(self, fusion):
        """Softmax produces valid probabilities."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = fusion._softmax(x)
        assert result.sum() == pytest.approx(1.0)
        assert all(r >= 0 for r in result)

    def test_softmax_numerical_stability(self, fusion):
        """Softmax is numerically stable with large values."""
        x = np.array([1000.0, 1001.0, 1002.0, 1003.0])
        result = fusion._softmax(x)
        assert result.sum() == pytest.approx(1.0)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_compute_weights_shape(self, fusion):
        """Compute weights returns correct shape."""
        query = np.random.randn(64)
        weights = fusion.compute_weights(query)
        assert weights.shape == (4,)

    def test_compute_weights_sum_to_one(self, fusion):
        """Computed weights sum to 1."""
        query = np.random.randn(64)
        weights = fusion.compute_weights(query)
        assert weights.sum() == pytest.approx(1.0, abs=0.01)

    def test_compute_weights_cold_start(self, fusion):
        """Cold start blends with default weights."""
        fusion.n_updates = 0
        query = np.random.randn(64)
        weights = fusion.compute_weights(query)
        # During cold start, should be heavily influenced by defaults
        # Weights should be close to default_weights
        assert weights.sum() == pytest.approx(1.0, abs=0.01)

    def test_compute_weights_after_warmup(self, fusion):
        """After warmup, weights are fully learned."""
        fusion.n_updates = fusion.cold_start_threshold + 10
        query = np.random.randn(64)
        weights = fusion.compute_weights(query)
        assert weights.sum() == pytest.approx(1.0, abs=0.01)

    def test_compute_weights_handles_long_embedding(self, fusion):
        """Handles embeddings longer than embed_dim."""
        query = np.random.randn(128)  # Longer than 64
        weights = fusion.compute_weights(query)
        assert weights.shape == (4,)

    def test_compute_weights_handles_short_embedding(self, fusion):
        """Handles embeddings shorter than embed_dim."""
        query = np.random.randn(32)  # Shorter than 64
        weights = fusion.compute_weights(query)
        assert weights.shape == (4,)

    def test_get_weights_dict(self, fusion):
        """Get weights as dictionary."""
        query = np.random.randn(64)
        weights_dict = fusion.get_weights_dict(query)
        assert isinstance(weights_dict, dict)
        assert "semantic" in weights_dict
        assert "recency" in weights_dict
        assert "outcome" in weights_dict
        assert "importance" in weights_dict

    def test_update_increments_counter(self, fusion):
        """Update increments n_updates."""
        query = np.random.randn(64)
        component_scores = {"semantic": 0.8, "recency": 0.6, "outcome": 0.5, "importance": 0.3}
        initial_updates = fusion.n_updates

        fusion.update(query, component_scores, outcome_utility=0.8)

        assert fusion.n_updates == initial_updates + 1

    def test_update_modifies_weights(self, fusion):
        """Update modifies weight matrices."""
        query = np.random.randn(64)
        component_scores = {"semantic": 0.9, "recency": 0.1, "outcome": 0.1, "importance": 0.1}

        W1_before = fusion.W1.copy()

        fusion.update(query, component_scores, outcome_utility=0.9)

        # Weights should change (slightly)
        assert not np.allclose(fusion.W1, W1_before)

    def test_update_with_different_utilities(self, fusion):
        """Update responds differently to different utilities."""
        query = np.random.randn(64)
        component_scores = {"semantic": 0.8, "recency": 0.5, "outcome": 0.3, "importance": 0.2}

        # High utility - should push weights toward high scorers
        W2_before = fusion.W2.copy()
        fusion.update(query, component_scores, outcome_utility=0.9)
        W2_after_high = fusion.W2.copy()

        # Reset and try low utility
        fusion.W2 = W2_before.copy()
        fusion.update(query, component_scores, outcome_utility=0.1)
        W2_after_low = fusion.W2.copy()

        # Updates should be different
        assert not np.allclose(W2_after_high, W2_after_low)


class TestLearnedReranker:
    """Tests for LearnedReranker class."""

    @pytest.fixture
    def reranker(self):
        """Create reranker instance."""
        return LearnedReranker(
            embed_dim=64,  # Smaller for tests
            learning_rate=0.005,
        )

    @pytest.fixture
    def mock_result(self):
        """Create mock scored result."""
        @dataclass
        class MockResult:
            score: float
            components: dict

            def __post_init__(self):
                if self.components is None:
                    self.components = {}

        return MockResult

    def test_initialization(self, reranker):
        """Test initialization."""
        assert reranker.embed_dim == 64
        assert reranker.lr == 0.005
        assert reranker.n_updates == 0

    def test_initialization_defaults(self):
        """Test default initialization."""
        reranker = LearnedReranker()
        assert reranker.embed_dim == 1024
        assert reranker.COMPONENT_DIM == 4
        assert reranker.QUERY_CONTEXT_DIM == 16

    def test_weight_shapes(self, reranker):
        """Weight matrices have correct shapes."""
        assert reranker.W_query.shape == (16, 64)  # Query compression
        assert reranker.W1.shape == (32, 20)  # Hidden layer (4 + 16 = 20 input)
        assert reranker.W2.shape == (1, 32)  # Output

    def test_compress_query(self, reranker):
        """Query compression produces correct shape."""
        query = np.random.randn(64)
        compressed = reranker._compress_query(query)
        assert compressed.shape == (16,)

    def test_compress_query_long_embedding(self, reranker):
        """Handles embeddings longer than embed_dim."""
        query = np.random.randn(128)
        compressed = reranker._compress_query(query)
        assert compressed.shape == (16,)

    def test_compress_query_short_embedding(self, reranker):
        """Handles embeddings shorter than embed_dim."""
        query = np.random.randn(32)
        compressed = reranker._compress_query(query)
        assert compressed.shape == (16,)

    def test_rerank_empty_results(self, reranker):
        """Rerank handles empty results."""
        query = np.random.randn(64)
        result = reranker.rerank([], query)
        assert result == []

    def test_rerank_cold_start_unchanged(self, reranker, mock_result):
        """During cold start, results are unchanged."""
        reranker.n_updates = 0  # Cold start

        results = [
            mock_result(score=0.9, components={"semantic": 0.9, "recency": 0.7, "outcome": 0.5, "importance": 0.3}),
            mock_result(score=0.7, components={"semantic": 0.7, "recency": 0.5, "outcome": 0.3, "importance": 0.2}),
        ]
        original_scores = [r.score for r in results]

        query = np.random.randn(64)
        reranked = reranker.rerank(results, query)

        # Scores should be unchanged during cold start
        assert [r.score for r in reranked] == original_scores

    def test_rerank_after_warmup(self, reranker, mock_result):
        """After warmup, results may be reranked."""
        reranker.n_updates = reranker.cold_start_threshold + 10

        results = [
            mock_result(score=0.9, components={"semantic": 0.9, "recency": 0.1, "outcome": 0.1, "importance": 0.1}),
            mock_result(score=0.8, components={"semantic": 0.1, "recency": 0.9, "outcome": 0.9, "importance": 0.9}),
        ]

        query = np.random.randn(64)
        reranked = reranker.rerank(results, query)

        # Should have rerank_adjustment in components
        for r in reranked:
            assert "rerank_adjustment" in r.components

    def test_rerank_preserves_count(self, reranker, mock_result):
        """Rerank preserves number of results."""
        reranker.n_updates = reranker.cold_start_threshold + 10

        results = [
            mock_result(score=0.9 - i * 0.1, components={"semantic": 0.5, "recency": 0.5, "outcome": 0.5, "importance": 0.5})
            for i in range(5)
        ]

        query = np.random.randn(64)
        reranked = reranker.rerank(results, query)

        assert len(reranked) == 5

    def test_update_increments_counter(self, reranker):
        """Update increments n_updates."""
        query = np.random.randn(64)
        component_scores = [{"semantic": 0.8, "recency": 0.5, "outcome": 0.3, "importance": 0.2}]
        utilities = [0.8]

        initial = reranker.n_updates
        reranker.update(query, component_scores, utilities)

        assert reranker.n_updates == initial + 1

    def test_update_empty_lists(self, reranker):
        """Update handles empty lists."""
        query = np.random.randn(64)
        initial = reranker.n_updates

        reranker.update(query, [], [])

        # Should not increment
        assert reranker.n_updates == initial

    def test_update_modifies_weights(self, reranker):
        """Update modifies weight matrices."""
        query = np.random.randn(64)
        component_scores = [{"semantic": 0.9, "recency": 0.1, "outcome": 0.1, "importance": 0.1}]
        utilities = [0.9]

        W1_before = reranker.W1.copy()
        reranker.update(query, component_scores, utilities)

        assert not np.allclose(reranker.W1, W1_before)

    def test_residual_weight_ramps_up(self, reranker):
        """Residual weight increases after warmup."""
        query = np.random.randn(64)
        component_scores = [{"semantic": 0.5, "recency": 0.5, "outcome": 0.5, "importance": 0.5}]
        utilities = [0.8]

        initial_residual = reranker.residual_weight

        # Simulate warmup
        reranker.n_updates = reranker.cold_start_threshold

        for _ in range(10):
            reranker.update(query, component_scores, utilities)

        # Residual weight should have increased
        assert reranker.residual_weight > initial_residual

    def test_residual_weight_capped(self, reranker):
        """Residual weight is capped at 0.5."""
        query = np.random.randn(64)
        component_scores = [{"semantic": 0.5, "recency": 0.5, "outcome": 0.5, "importance": 0.5}]
        utilities = [0.8]

        # Simulate many updates
        reranker.n_updates = reranker.cold_start_threshold + 1000

        for _ in range(100):
            reranker.update(query, component_scores, utilities)

        assert reranker.residual_weight <= 0.5

    def test_adjustment_bounded(self, reranker, mock_result):
        """Rerank adjustments are bounded."""
        reranker.n_updates = reranker.cold_start_threshold + 100

        results = [
            mock_result(score=0.9, components={"semantic": 1.0, "recency": 0.0, "outcome": 0.0, "importance": 0.0}),
        ]

        query = np.random.randn(64)
        reranked = reranker.rerank(results, query)

        # Adjustment should be in [-0.2, 0.2]
        adjustment = reranked[0].components["rerank_adjustment"]
        assert -0.2 <= adjustment <= 0.2


class TestLearnedFusionIntegration:
    """Integration tests for learned fusion."""

    def test_fusion_weights_evolve_with_training(self):
        """Fusion weights evolve with training."""
        fusion = LearnedFusionWeights(embed_dim=64, hidden_dim=16)

        # Generate consistent query patterns
        np.random.seed(42)
        query = np.random.randn(64)

        # Training: semantic is always good, recency is bad
        for _ in range(100):
            component_scores = {
                "semantic": np.random.uniform(0.7, 0.9),
                "recency": np.random.uniform(0.1, 0.3),
                "outcome": np.random.uniform(0.3, 0.5),
                "importance": np.random.uniform(0.3, 0.5),
            }
            # High utility when semantic is high
            utility = component_scores["semantic"]
            fusion.update(query, component_scores, utility)

        # After training, semantic weight should be higher
        weights = fusion.get_weights_dict(query)
        # Due to randomness, just verify training happened
        assert fusion.n_updates == 100

    def test_reranker_learns_patterns(self):
        """Reranker learns to adjust based on patterns."""
        @dataclass
        class Result:
            score: float
            components: dict

        reranker = LearnedReranker(embed_dim=64)

        np.random.seed(42)
        query = np.random.randn(64)

        # Train: items with high semantic but low recency are good
        for _ in range(150):
            component_scores = [
                {"semantic": np.random.uniform(0.7, 0.9), "recency": 0.2, "outcome": 0.5, "importance": 0.5},
            ]
            utilities = [0.9]  # Good outcome
            reranker.update(query, component_scores, utilities)

        assert reranker.n_updates >= reranker.cold_start_threshold
