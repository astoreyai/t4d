"""Tests for norepinephrine arousal/attention system."""
import pytest
import numpy as np
from datetime import datetime

from t4dm.learning.norepinephrine import (
    ArousalState,
    NorepinephrineSystem,
)


class TestArousalState:
    """Tests for ArousalState dataclass."""

    def test_creation(self):
        """Test state can be created."""
        state = ArousalState(
            tonic_level=0.5,
            phasic_burst=0.3,
            combined_gain=1.2,
            novelty_score=0.7,
            uncertainty_score=0.4,
        )

        assert state.tonic_level == 0.5
        assert state.phasic_burst == 0.3
        assert state.combined_gain == 1.2

    def test_timestamp_default(self):
        """Test timestamp defaults to now."""
        state = ArousalState(
            tonic_level=0.5,
            phasic_burst=0.0,
            combined_gain=1.0,
            novelty_score=0.5,
            uncertainty_score=0.5,
        )

        assert isinstance(state.timestamp, datetime)

    def test_exploration_bias(self):
        """Test exploration_bias property."""
        # Low gain
        state_low = ArousalState(
            tonic_level=0.3,
            phasic_burst=0.0,
            combined_gain=0.6,
            novelty_score=0.3,
            uncertainty_score=0.3,
        )
        assert state_low.exploration_bias < 0.5

        # High gain
        state_high = ArousalState(
            tonic_level=0.8,
            phasic_burst=0.5,
            combined_gain=1.8,
            novelty_score=0.9,
            uncertainty_score=0.7,
        )
        assert state_high.exploration_bias > state_low.exploration_bias


class TestNorepinephrineSystem:
    """Tests for NorepinephrineSystem class."""

    def test_init_default(self):
        """Test default initialization."""
        ne = NorepinephrineSystem()

        assert ne.baseline_arousal == 0.5
        assert ne.min_gain == 0.5
        assert ne.max_gain == 2.0

    def test_init_custom_params(self):
        """Test custom parameter initialization."""
        ne = NorepinephrineSystem(
            baseline_arousal=0.6,
            min_gain=0.3,
            max_gain=2.5,
        )

        assert ne.baseline_arousal == 0.6
        assert ne.min_gain == 0.3
        assert ne.max_gain == 2.5

    def test_compute_novelty_first_query(self):
        """Test first query is maximally novel."""
        ne = NorepinephrineSystem()
        query = np.random.randn(128)

        novelty = ne.compute_novelty(query)

        assert novelty == 1.0

    def test_compute_novelty_similar_query(self):
        """Test similar queries have low novelty."""
        ne = NorepinephrineSystem()
        query = np.random.randn(128)

        # Add to history
        ne._query_history.append(query / np.linalg.norm(query))

        # Same query should have low novelty
        novelty = ne.compute_novelty(query)

        assert novelty < 0.2

    def test_compute_novelty_different_query(self):
        """Test different queries have high novelty."""
        ne = NorepinephrineSystem()
        query1 = np.array([1.0] * 64 + [0.0] * 64)
        query2 = np.array([0.0] * 64 + [1.0] * 64)  # Orthogonal

        ne._query_history.append(query1 / np.linalg.norm(query1))
        novelty = ne.compute_novelty(query2)

        assert novelty > 0.5

    def test_compute_uncertainty_uniform(self):
        """Test uniform scores have high uncertainty."""
        ne = NorepinephrineSystem()

        # Uniform distribution
        scores = [0.5, 0.5, 0.5, 0.5]
        uncertainty = ne.compute_uncertainty(scores)

        # Should be close to 1 (max entropy)
        assert uncertainty > 0.9

    def test_compute_uncertainty_peaked(self):
        """Test peaked scores have low uncertainty."""
        ne = NorepinephrineSystem()

        # Highly peaked distribution
        scores = [0.9, 0.05, 0.03, 0.02]
        uncertainty = ne.compute_uncertainty(scores)

        # Should be low (low entropy)
        assert uncertainty < 0.5

    def test_compute_uncertainty_empty(self):
        """Test empty scores return default."""
        ne = NorepinephrineSystem()

        uncertainty = ne.compute_uncertainty([])

        assert uncertainty == 0.5

    def test_compute_uncertainty_single(self):
        """Test single score returns default."""
        ne = NorepinephrineSystem()

        uncertainty = ne.compute_uncertainty([0.8])

        assert uncertainty == 0.5

    def test_update_returns_state(self):
        """Test update returns ArousalState."""
        ne = NorepinephrineSystem()
        query = np.random.randn(128)

        state = ne.update(query)

        assert isinstance(state, ArousalState)

    def test_update_with_retrieval_scores(self):
        """Test update with retrieval scores."""
        ne = NorepinephrineSystem()
        query = np.random.randn(128)
        scores = [0.8, 0.6, 0.4]

        state = ne.update(query, retrieval_scores=scores)

        assert state.uncertainty_score != 0.5  # Should be computed

    def test_update_with_urgency(self):
        """Test external urgency increases gain."""
        ne = NorepinephrineSystem(max_gain=3.0)  # Higher max to see effect
        query = np.random.randn(128)

        state_normal = ne.update(query, external_urgency=0.0)
        ne.reset_history()
        state_urgent = ne.update(query, external_urgency=1.0)

        assert state_urgent.combined_gain >= state_normal.combined_gain

    def test_update_adds_to_history(self):
        """Test update adds query to history."""
        ne = NorepinephrineSystem()
        query = np.random.randn(128)

        ne.update(query)

        assert len(ne._query_history) == 1

    def test_phasic_decay(self):
        """Test phasic level decays over updates."""
        ne = NorepinephrineSystem(phasic_decay=0.5)

        # High novelty first query
        query1 = np.random.randn(128)
        state1 = ne.update(query1)

        # Same query (low novelty)
        state2 = ne.update(query1)

        # Phasic should decay
        assert state2.phasic_burst < state1.phasic_burst

    def test_get_current_gain_no_state(self):
        """Test get_current_gain returns 1.0 with no state."""
        ne = NorepinephrineSystem()

        gain = ne.get_current_gain()

        assert gain == 1.0

    def test_get_current_gain_with_state(self):
        """Test get_current_gain returns state gain."""
        ne = NorepinephrineSystem()
        ne.update(np.random.randn(128))

        gain = ne.get_current_gain()

        assert ne.min_gain <= gain <= ne.max_gain

    def test_get_current_novelty_no_state(self):
        """Test get_current_novelty returns 0.5 with no state."""
        ne = NorepinephrineSystem()

        novelty = ne.get_current_novelty()

        assert novelty == 0.5

    def test_modulate_learning_rate(self):
        """Test learning rate modulation."""
        ne = NorepinephrineSystem()
        ne.update(np.random.randn(128))

        base_lr = 0.1
        modulated_lr = ne.modulate_learning_rate(base_lr)

        # Should be scaled by gain
        expected = base_lr * ne.get_current_gain()
        assert modulated_lr == pytest.approx(expected, rel=0.01)

    def test_modulate_retrieval_threshold(self):
        """Test retrieval threshold modulation."""
        ne = NorepinephrineSystem()
        ne.update(np.random.randn(128))

        base_threshold = 0.8
        modulated_threshold = ne.modulate_retrieval_threshold(base_threshold)

        # High gain = lower threshold
        expected = base_threshold / ne.get_current_gain()
        assert modulated_threshold == pytest.approx(expected, rel=0.01)

    def test_modulate_separation_strength(self):
        """Test pattern separation modulation."""
        ne = NorepinephrineSystem()
        ne.update(np.random.randn(128))

        base_separation = 1.0
        modulated = ne.modulate_separation_strength(base_separation)

        # Should be scaled by gain
        expected = base_separation * ne.get_current_gain()
        assert modulated == pytest.approx(expected, rel=0.01)

    def test_get_exploration_bias_no_state(self):
        """Test exploration_bias returns 0.5 with no state."""
        ne = NorepinephrineSystem()

        bias = ne.get_exploration_bias()

        assert bias == 0.5

    def test_get_stats_empty(self):
        """Test get_stats with no updates."""
        ne = NorepinephrineSystem()

        stats = ne.get_stats()

        assert stats["total_updates"] == 0
        assert stats["avg_gain"] == 1.0

    def test_get_stats_with_updates(self):
        """Test get_stats after updates."""
        ne = NorepinephrineSystem()

        ne.update(np.random.randn(128))
        ne.update(np.random.randn(128))

        stats = ne.get_stats()

        assert stats["total_updates"] == 2
        assert "avg_novelty" in stats
        assert "avg_uncertainty" in stats


class TestConfigurationSetters:
    """Tests for runtime configuration setters."""

    def test_set_baseline_arousal(self):
        """Test setting baseline arousal."""
        ne = NorepinephrineSystem()

        ne.set_baseline_arousal(0.7)

        assert ne.baseline_arousal == 0.7

    def test_set_baseline_arousal_clipped(self):
        """Test baseline arousal is clipped."""
        ne = NorepinephrineSystem()

        ne.set_baseline_arousal(1.5)

        assert ne.baseline_arousal == 1.0

    def test_set_arousal_bounds(self):
        """Test setting arousal bounds."""
        ne = NorepinephrineSystem()

        ne.set_arousal_bounds(min_gain=0.3, max_gain=3.0)

        assert ne.min_gain == 0.3
        assert ne.max_gain == 3.0

    def test_set_arousal_bounds_enforces_order(self):
        """Test max_gain > min_gain."""
        ne = NorepinephrineSystem()

        ne.set_arousal_bounds(min_gain=1.0, max_gain=1.0)

        assert ne.max_gain > ne.min_gain

    def test_set_arousal_gain(self):
        """Test directly setting arousal gain."""
        ne = NorepinephrineSystem(min_gain=0.5, max_gain=2.0)

        ne.set_arousal_gain(1.5)

        assert ne.get_current_gain() == 1.5

    def test_set_arousal_gain_clipped(self):
        """Test arousal gain is clipped to bounds."""
        ne = NorepinephrineSystem(min_gain=0.5, max_gain=2.0)

        ne.set_arousal_gain(5.0)

        assert ne.get_current_gain() == 2.0

    def test_boost_arousal(self):
        """Test boosting arousal."""
        ne = NorepinephrineSystem(max_gain=5.0)  # Higher max to see boost effect
        ne.set_arousal_gain(1.0)  # Start at 1.0
        initial_gain = ne.get_current_gain()

        new_gain = ne.boost_arousal(1.5)

        assert new_gain > initial_gain

    def test_boost_arousal_clipped(self):
        """Test boost multiplier is clipped."""
        ne = NorepinephrineSystem(max_gain=2.0)
        ne.update(np.random.randn(128))

        new_gain = ne.boost_arousal(10.0)  # Will be clipped to 3.0

        assert new_gain <= ne.max_gain

    def test_set_novelty_decay(self):
        """Test setting novelty decay."""
        ne = NorepinephrineSystem()

        ne.set_novelty_decay(0.9)

        assert ne.novelty_decay == 0.9

    def test_set_phasic_decay(self):
        """Test setting phasic decay."""
        ne = NorepinephrineSystem()

        ne.set_phasic_decay(0.7)

        assert ne.phasic_decay == 0.7

    def test_get_config(self):
        """Test get_config returns configuration."""
        ne = NorepinephrineSystem(baseline_arousal=0.6, min_gain=0.4)

        config = ne.get_config()

        assert config["baseline_arousal"] == 0.6
        assert config["min_gain"] == 0.4

    def test_reset_history(self):
        """Test reset_history clears state."""
        ne = NorepinephrineSystem(baseline_arousal=0.5)

        ne.update(np.random.randn(128))
        ne.update(np.random.randn(128))
        ne.reset_history()

        assert len(ne._query_history) == 0
        assert len(ne._arousal_history) == 0
        assert ne._tonic_level == 0.5
        assert ne._current_state is None
