"""
Tests for FFRetrievalScorer bridge (P6.4).

FF retrieval scoring uses goodness to boost confident pattern matches.
"""

import numpy as np
import pytest

from t4dm.bridges import (
    FFRetrievalConfig,
    FFRetrievalScorer,
    FFRetrievalState,
    RetrievalScore,
)


class TestFFRetrievalConfig:
    """Test FFRetrievalConfig dataclass."""

    def test_default_config(self):
        """Default config values are sensible."""
        config = FFRetrievalConfig()
        assert config.normalize_goodness is True
        assert config.max_boost == 0.3
        assert config.learn_from_outcomes is True
        assert config.positive_outcome_threshold == 0.6

    def test_custom_config(self):
        """Custom config values are respected."""
        config = FFRetrievalConfig(
            max_boost=0.5,
            learn_from_outcomes=False,
        )
        assert config.max_boost == 0.5
        assert config.learn_from_outcomes is False


class TestRetrievalScore:
    """Test RetrievalScore dataclass."""

    def test_creation(self):
        """RetrievalScore creation works."""
        score = RetrievalScore(
            memory_id="test_mem",
            goodness=2.5,
            normalized_goodness=1.25,
            confidence=0.7,
            boost=0.15,
            is_confident=True,
        )
        assert score.memory_id == "test_mem"
        assert score.goodness == 2.5
        assert score.is_confident is True


class TestFFRetrievalScorer:
    """Test FFRetrievalScorer class."""

    @pytest.fixture
    def config(self):
        """Standard test config."""
        return FFRetrievalConfig(
            max_boost=0.3,
            learn_from_outcomes=True,
        )

    @pytest.fixture
    def scorer(self, config):
        """Scorer without FF layer (fallback mode)."""
        return FFRetrievalScorer(ff_layer=None, config=config)

    def test_initialization(self, config):
        """Scorer initializes correctly."""
        scorer = FFRetrievalScorer(config=config)
        assert scorer.config == config
        assert isinstance(scorer.state, FFRetrievalState)
        assert scorer.ff_layer is None

    def test_score_candidate(self, scorer):
        """Single candidate scoring works."""
        embedding = np.random.randn(1024).astype(np.float32)
        score = scorer.score_candidate(embedding, "mem_1")

        assert isinstance(score, RetrievalScore)
        assert score.memory_id == "mem_1"
        assert 0 <= score.confidence <= 1
        assert 0 <= score.boost <= scorer.config.max_boost

    def test_score_candidates_batch(self, scorer):
        """Batch candidate scoring works."""
        embeddings = [np.random.randn(1024).astype(np.float32) for _ in range(5)]
        ids = [f"mem_{i}" for i in range(5)]

        scores = scorer.score_candidates(embeddings, ids)

        assert len(scores) == 5
        for i, score in enumerate(scores):
            assert score.memory_id == f"mem_{i}"

    def test_state_updates(self, scorer):
        """Scorer state updates correctly."""
        embedding = np.random.randn(1024).astype(np.float32)

        assert scorer.state.n_candidates_scored == 0

        scorer.score_candidate(embedding, "mem_1")
        assert scorer.state.n_candidates_scored == 1

        scorer.score_candidate(embedding, "mem_2")
        assert scorer.state.n_candidates_scored == 2

    def test_cache_functionality(self, scorer):
        """Goodness caching works."""
        embedding = np.random.randn(1024).astype(np.float32)

        # First call should cache
        score1 = scorer.score_candidate(embedding, "mem_1", use_cache=True)

        # Second call with same ID should use cache
        score2 = scorer.score_candidate(embedding, "mem_1", use_cache=True)

        # Goodness should be the same
        assert score1.goodness == score2.goodness

    def test_clear_cache(self, scorer):
        """Cache clearing works."""
        embedding = np.random.randn(1024).astype(np.float32)
        scorer.score_candidate(embedding, "mem_1")

        assert len(scorer._goodness_cache) > 0

        scorer.clear_cache()

        assert len(scorer._goodness_cache) == 0

    def test_learn_from_outcome_disabled(self, scorer):
        """Learning disabled returns skip."""
        scorer.config.learn_from_outcomes = False
        embeddings = [np.random.randn(1024).astype(np.float32)]
        ids = ["mem_1"]

        stats = scorer.learn_from_outcome(embeddings, ids, outcome_score=0.8)

        assert stats.get("skipped") is True

    def test_learn_from_outcome_no_layer(self, scorer):
        """Learning without FF layer returns skip."""
        embeddings = [np.random.randn(1024).astype(np.float32)]
        ids = ["mem_1"]

        stats = scorer.learn_from_outcome(embeddings, ids, outcome_score=0.8)

        assert stats.get("skipped") is True
        assert "no FF layer" in stats.get("reason", "")

    def test_get_stats(self, scorer):
        """Statistics retrieval works."""
        embedding = np.random.randn(1024).astype(np.float32)
        scorer.score_candidate(embedding, "mem_1")

        stats = scorer.get_stats()

        assert "n_candidates_scored" in stats
        assert stats["n_candidates_scored"] == 1
        assert "cache_size" in stats
        assert "history_size" in stats

    def test_adaptive_threshold(self, scorer):
        """Adaptive threshold calculation works."""
        # With no history, returns default
        threshold = scorer.get_adaptive_threshold()
        assert threshold == 1.0

        # Add history
        for _ in range(20):
            embedding = np.random.randn(1024).astype(np.float32)
            scorer.score_candidate(embedding, f"mem_{_}")

        threshold = scorer.get_adaptive_threshold()
        assert threshold > 0  # Should be mean + std

    def test_confidence_to_boost_threshold(self, scorer):
        """Low confidence gives no boost."""
        # Low confidence
        boost_low = scorer._confidence_to_boost(0.3)
        assert boost_low == 0.0

        # High confidence
        boost_high = scorer._confidence_to_boost(0.9)
        assert boost_high > 0
        assert boost_high <= scorer.config.max_boost


class TestFFRetrievalScorerWithLayer:
    """Test FFRetrievalScorer with actual FF layer."""

    @pytest.fixture
    def ff_layer(self):
        """Create a real FF layer for testing."""
        try:
            from t4dm.nca.forward_forward import ForwardForwardConfig, ForwardForwardLayer
            config = ForwardForwardConfig(
                input_dim=1024,
                hidden_dim=512,
                learning_rate=0.03,
                threshold_theta=2.0,
            )
            return ForwardForwardLayer(config)
        except Exception:
            pytest.skip("FF layer not available")

    @pytest.fixture
    def scorer_with_layer(self, ff_layer):
        """Scorer with real FF layer."""
        config = FFRetrievalConfig(max_boost=0.3, learn_from_outcomes=True)
        return FFRetrievalScorer(ff_layer=ff_layer, config=config)

    def test_score_with_real_layer(self, scorer_with_layer):
        """Scoring with real FF layer works."""
        embedding = np.random.randn(1024).astype(np.float32)
        score = scorer_with_layer.score_candidate(embedding, "mem_1")

        assert isinstance(score, RetrievalScore)
        assert score.goodness > 0  # FF layer computes real goodness

    def test_learn_from_positive_outcome(self, scorer_with_layer):
        """Learning from positive outcome trains layer."""
        embeddings = [np.random.randn(1024).astype(np.float32)]
        ids = ["mem_1"]

        # Record initial state
        initial_n_positive = scorer_with_layer.state.n_positive_outcomes

        stats = scorer_with_layer.learn_from_outcome(embeddings, ids, outcome_score=0.9)

        assert stats.get("skipped") is not True
        assert stats.get("positive_learning", 0) > 0
        assert scorer_with_layer.state.n_positive_outcomes > initial_n_positive

    def test_learn_from_negative_outcome(self, scorer_with_layer):
        """Learning from negative outcome trains layer."""
        embeddings = [np.random.randn(1024).astype(np.float32)]
        ids = ["mem_1"]

        # Record initial state
        initial_n_negative = scorer_with_layer.state.n_negative_outcomes

        stats = scorer_with_layer.learn_from_outcome(embeddings, ids, outcome_score=0.2)

        assert stats.get("skipped") is not True
        assert stats.get("negative_learning", 0) > 0
        assert scorer_with_layer.state.n_negative_outcomes > initial_n_negative

    def test_cache_invalidated_after_learning(self, scorer_with_layer):
        """Cache entries invalidated after outcome learning."""
        embedding = np.random.randn(1024).astype(np.float32)

        # Score to populate cache
        scorer_with_layer.score_candidate(embedding, "mem_1")
        assert "mem_1" in scorer_with_layer._goodness_cache

        # Learn - should invalidate cache
        scorer_with_layer.learn_from_outcome([embedding], ["mem_1"], outcome_score=0.8)

        assert "mem_1" not in scorer_with_layer._goodness_cache

    def test_batch_scoring_with_layer(self, scorer_with_layer):
        """Batch scoring with real layer works."""
        embeddings = [np.random.randn(1024).astype(np.float32) for _ in range(10)]
        ids = [f"mem_{i}" for i in range(10)]

        scores = scorer_with_layer.score_candidates(embeddings, ids)

        assert len(scores) == 10
        assert scorer_with_layer.state.last_batch_size == 10
        # All should have positive goodness from FF
        for score in scores:
            assert score.goodness > 0
