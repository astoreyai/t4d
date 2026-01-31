"""Tests for acetylcholine encoding/retrieval mode switch."""
import pytest
import numpy as np
from datetime import datetime

from t4dm.learning.acetylcholine import (
    AcetylcholineState,
    AcetylcholineSystem,
    CognitiveMode,
)


class TestCognitiveMode:
    """Tests for CognitiveMode enum."""

    def test_modes_exist(self):
        """Test all cognitive modes exist."""
        assert CognitiveMode.ENCODING.value == "encoding"
        assert CognitiveMode.BALANCED.value == "balanced"
        assert CognitiveMode.RETRIEVAL.value == "retrieval"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert CognitiveMode("encoding") == CognitiveMode.ENCODING
        assert CognitiveMode("retrieval") == CognitiveMode.RETRIEVAL


class TestAcetylcholineState:
    """Tests for AcetylcholineState dataclass."""

    def test_creation(self):
        """Test state can be created."""
        state = AcetylcholineState(
            ach_level=0.7,
            mode=CognitiveMode.ENCODING,
            encoding_weight=0.7,
            retrieval_weight=0.58,
            attention_gate=0.85,
        )

        assert state.ach_level == 0.7
        assert state.mode == CognitiveMode.ENCODING
        assert state.encoding_weight == 0.7

    def test_timestamp_default(self):
        """Test timestamp defaults to now."""
        state = AcetylcholineState(
            ach_level=0.5,
            mode=CognitiveMode.BALANCED,
            encoding_weight=0.5,
            retrieval_weight=0.7,
            attention_gate=0.75,
        )

        assert isinstance(state.timestamp, datetime)

    def test_learning_rate_modifier(self):
        """Test learning_rate_modifier property."""
        state = AcetylcholineState(
            ach_level=0.8,
            mode=CognitiveMode.ENCODING,
            encoding_weight=0.8,
            retrieval_weight=0.52,
            attention_gate=0.9,
        )

        # Higher ACh = higher LR modifier
        assert state.learning_rate_modifier == pytest.approx(1.3, rel=0.01)

    def test_pattern_completion_strength(self):
        """Test pattern_completion_strength property."""
        state = AcetylcholineState(
            ach_level=0.2,
            mode=CognitiveMode.RETRIEVAL,
            encoding_weight=0.2,
            retrieval_weight=0.88,
            attention_gate=0.6,
        )

        # Lower ACh = stronger pattern completion
        assert state.pattern_completion_strength == pytest.approx(0.88, rel=0.01)


class TestAcetylcholineSystem:
    """Tests for AcetylcholineSystem class."""

    def test_init_default(self):
        """Test default initialization."""
        ach = AcetylcholineSystem()

        assert ach.baseline_ach == 0.5
        assert ach.encoding_threshold == 0.7
        assert ach.retrieval_threshold == 0.3

    def test_init_custom_params(self):
        """Test custom parameter initialization."""
        ach = AcetylcholineSystem(
            baseline_ach=0.6,
            encoding_threshold=0.8,
            retrieval_threshold=0.2,
        )

        assert ach.baseline_ach == 0.6
        assert ach.encoding_threshold == 0.8
        assert ach.retrieval_threshold == 0.2

    def test_compute_encoding_demand_novelty(self):
        """Test encoding demand increases with novelty."""
        ach = AcetylcholineSystem()

        demand_low = ach.compute_encoding_demand(query_novelty=0.2)
        demand_high = ach.compute_encoding_demand(query_novelty=0.9)

        assert demand_high > demand_low

    def test_compute_encoding_demand_statement(self):
        """Test statements increase encoding demand."""
        ach = AcetylcholineSystem()

        demand_question = ach.compute_encoding_demand(query_novelty=0.5, is_statement=False)
        demand_statement = ach.compute_encoding_demand(query_novelty=0.5, is_statement=True)

        assert demand_statement > demand_question

    def test_compute_encoding_demand_importance(self):
        """Test explicit importance increases encoding demand."""
        ach = AcetylcholineSystem()

        demand_normal = ach.compute_encoding_demand(query_novelty=0.5)
        demand_important = ach.compute_encoding_demand(query_novelty=0.5, explicit_importance=0.9)

        assert demand_important > demand_normal

    def test_compute_retrieval_demand_question(self):
        """Test questions increase retrieval demand."""
        ach = AcetylcholineSystem()

        demand_statement = ach.compute_retrieval_demand(is_question=False)
        demand_question = ach.compute_retrieval_demand(is_question=True)

        assert demand_question > demand_statement

    def test_compute_retrieval_demand_match_quality(self):
        """Test memory match quality increases retrieval demand."""
        ach = AcetylcholineSystem()

        demand_low = ach.compute_retrieval_demand(memory_match_quality=0.2)
        demand_high = ach.compute_retrieval_demand(memory_match_quality=0.9)

        assert demand_high > demand_low

    def test_compute_retrieval_demand_specificity(self):
        """Test query specificity increases retrieval demand."""
        ach = AcetylcholineSystem()

        demand_vague = ach.compute_retrieval_demand(query_specificity=0.2)
        demand_specific = ach.compute_retrieval_demand(query_specificity=0.9)

        assert demand_specific > demand_vague

    def test_update_returns_state(self):
        """Test update returns AcetylcholineState."""
        ach = AcetylcholineSystem()

        state = ach.update(encoding_demand=0.5, retrieval_demand=0.3)

        assert isinstance(state, AcetylcholineState)

    def test_update_high_encoding_demand(self):
        """Test high encoding demand increases ACh."""
        ach = AcetylcholineSystem(baseline_ach=0.5)

        state = ach.update(encoding_demand=0.9, retrieval_demand=0.1)

        assert state.ach_level > 0.5

    def test_update_high_retrieval_demand(self):
        """Test high retrieval demand decreases ACh."""
        ach = AcetylcholineSystem(baseline_ach=0.5)

        state = ach.update(encoding_demand=0.1, retrieval_demand=0.9)

        assert state.ach_level < 0.5

    def test_update_arousal_gain(self):
        """Test arousal gain modulates ACh level."""
        ach = AcetylcholineSystem()

        state_low = ach.update(encoding_demand=0.5, retrieval_demand=0.5, arousal_gain=0.8)
        ach.reset()
        state_high = ach.update(encoding_demand=0.5, retrieval_demand=0.5, arousal_gain=1.2)

        assert state_high.ach_level > state_low.ach_level

    def test_update_mode_encoding(self):
        """Test mode becomes ENCODING with high ACh."""
        ach = AcetylcholineSystem(encoding_threshold=0.7)

        # Force high encoding demand
        for _ in range(5):
            state = ach.update(encoding_demand=1.0, retrieval_demand=0.0)

        assert state.mode == CognitiveMode.ENCODING

    def test_update_mode_retrieval(self):
        """Test mode becomes RETRIEVAL with low ACh."""
        ach = AcetylcholineSystem(retrieval_threshold=0.3)

        # Force high retrieval demand
        for _ in range(5):
            state = ach.update(encoding_demand=0.0, retrieval_demand=1.0)

        assert state.mode == CognitiveMode.RETRIEVAL

    def test_get_current_mode_default(self):
        """Test get_current_mode returns BALANCED before any update."""
        ach = AcetylcholineSystem()

        mode = ach.get_current_mode()

        assert mode == CognitiveMode.BALANCED

    def test_get_current_level(self):
        """Test get_current_level returns ACh level."""
        ach = AcetylcholineSystem(baseline_ach=0.6)

        level = ach.get_current_level()

        assert level == 0.6

    def test_should_prioritize_encoding(self):
        """Test should_prioritize_encoding in encoding mode."""
        ach = AcetylcholineSystem()

        # Force encoding mode
        for _ in range(5):
            ach.update(encoding_demand=1.0, retrieval_demand=0.0)

        assert ach.should_prioritize_encoding() is True
        assert ach.should_prioritize_retrieval() is False

    def test_should_prioritize_retrieval(self):
        """Test should_prioritize_retrieval in retrieval mode."""
        ach = AcetylcholineSystem()

        # Force retrieval mode
        for _ in range(5):
            ach.update(encoding_demand=0.0, retrieval_demand=1.0)

        assert ach.should_prioritize_retrieval() is True
        assert ach.should_prioritize_encoding() is False

    def test_modulate_learning_rate(self):
        """Test learning rate modulation by ACh level."""
        ach = AcetylcholineSystem()

        # Low ACh
        for _ in range(5):
            ach.update(encoding_demand=0.0, retrieval_demand=1.0)
        lr_low = ach.modulate_learning_rate(0.1)

        ach.reset()

        # High ACh
        for _ in range(5):
            ach.update(encoding_demand=1.0, retrieval_demand=0.0)
        lr_high = ach.modulate_learning_rate(0.1)

        assert lr_high > lr_low

    def test_modulate_learning_rate_no_state(self):
        """Test learning rate with no state returns base_lr."""
        ach = AcetylcholineSystem()

        lr = ach.modulate_learning_rate(0.1)

        assert lr == 0.1

    def test_modulate_pattern_completion(self):
        """Test pattern completion modulation by ACh level."""
        ach = AcetylcholineSystem()

        # High ACh (weak pattern completion)
        for _ in range(5):
            ach.update(encoding_demand=1.0, retrieval_demand=0.0)
        strength_high = ach.modulate_pattern_completion(1.0)

        ach.reset()

        # Low ACh (strong pattern completion)
        for _ in range(5):
            ach.update(encoding_demand=0.0, retrieval_demand=1.0)
        strength_low = ach.modulate_pattern_completion(1.0)

        assert strength_low > strength_high

    def test_modulate_pattern_completion_no_state(self):
        """Test pattern completion with no state returns base_strength."""
        ach = AcetylcholineSystem()

        strength = ach.modulate_pattern_completion(1.0)

        assert strength == 1.0

    def test_get_attention_weights_encoding_mode(self):
        """Test attention weights in encoding mode."""
        ach = AcetylcholineSystem()

        # Force encoding mode
        for _ in range(5):
            ach.update(encoding_demand=1.0, retrieval_demand=0.0)

        weights = ach.get_attention_weights(["episodic", "semantic", "other"])

        assert weights["episodic"] > weights["semantic"]

    def test_get_attention_weights_retrieval_mode(self):
        """Test attention weights in retrieval mode."""
        ach = AcetylcholineSystem()

        # Force retrieval mode
        for _ in range(5):
            ach.update(encoding_demand=0.0, retrieval_demand=1.0)

        weights = ach.get_attention_weights(["episodic", "semantic", "other"])

        assert weights["semantic"] > weights["episodic"]

    def test_get_attention_weights_no_state(self):
        """Test attention weights with no state returns 1.0 for all."""
        ach = AcetylcholineSystem()

        weights = ach.get_attention_weights(["episodic", "semantic"])

        assert all(w == 1.0 for w in weights.values())

    def test_get_reconsolidation_eligibility(self):
        """Test reconsolidation eligibility based on ACh level."""
        ach = AcetylcholineSystem()

        # High ACh = high eligibility
        for _ in range(5):
            ach.update(encoding_demand=1.0, retrieval_demand=0.0)
        elig_high = ach.get_reconsolidation_eligibility()

        ach.reset()

        # Low ACh = low eligibility
        for _ in range(5):
            ach.update(encoding_demand=0.0, retrieval_demand=1.0)
        elig_low = ach.get_reconsolidation_eligibility()

        assert elig_high > elig_low

    def test_get_reconsolidation_eligibility_no_state(self):
        """Test eligibility with no state returns 0.5."""
        ach = AcetylcholineSystem()

        eligibility = ach.get_reconsolidation_eligibility()

        assert eligibility == 0.5

    def test_get_stats_empty(self):
        """Test get_stats with no updates."""
        ach = AcetylcholineSystem()

        stats = ach.get_stats()

        assert stats["total_updates"] == 0
        assert stats["current_mode"] == "balanced"

    def test_get_stats_with_updates(self):
        """Test get_stats after updates."""
        ach = AcetylcholineSystem()

        ach.update(encoding_demand=0.8, retrieval_demand=0.2)
        ach.update(encoding_demand=0.7, retrieval_demand=0.3)

        stats = ach.get_stats()

        assert stats["total_updates"] == 2
        assert "avg_ach" in stats
        assert "mode_counts" in stats
        assert "config" in stats


class TestConfigurationSetters:
    """Tests for runtime configuration setters."""

    def test_force_mode_encoding(self):
        """Test force_mode sets encoding mode."""
        ach = AcetylcholineSystem()

        state = ach.force_mode(CognitiveMode.ENCODING)

        assert state.mode == CognitiveMode.ENCODING
        assert ach._ach_level >= ach.encoding_threshold

    def test_force_mode_retrieval(self):
        """Test force_mode sets retrieval mode."""
        ach = AcetylcholineSystem()

        state = ach.force_mode(CognitiveMode.RETRIEVAL)

        assert state.mode == CognitiveMode.RETRIEVAL
        assert ach._ach_level <= ach.retrieval_threshold

    def test_force_mode_from_string(self):
        """Test force_mode accepts string."""
        ach = AcetylcholineSystem()

        state = ach.force_mode("encoding")

        assert state.mode == CognitiveMode.ENCODING

    def test_set_thresholds(self):
        """Test setting thresholds."""
        ach = AcetylcholineSystem()

        ach.set_thresholds(encoding=0.8, retrieval=0.2)

        assert ach.encoding_threshold == 0.8
        assert ach.retrieval_threshold == 0.2

    def test_set_thresholds_clipped(self):
        """Test thresholds are clipped to valid range."""
        ach = AcetylcholineSystem()

        ach.set_thresholds(encoding=0.99, retrieval=0.01)

        assert ach.encoding_threshold == 0.9  # Max
        assert ach.retrieval_threshold == 0.1  # Min

    def test_set_thresholds_enforces_order(self):
        """Test encoding threshold > retrieval threshold."""
        ach = AcetylcholineSystem()

        ach.set_thresholds(encoding=0.5, retrieval=0.5)  # Same value

        assert ach.encoding_threshold > ach.retrieval_threshold

    def test_set_adaptation_rate(self):
        """Test setting adaptation rate."""
        ach = AcetylcholineSystem()

        ach.set_adaptation_rate(0.5)

        assert ach.adaptation_rate == 0.5

    def test_set_adaptation_rate_clipped(self):
        """Test adaptation rate is clipped."""
        ach = AcetylcholineSystem()

        ach.set_adaptation_rate(5.0)  # Too high

        assert ach.adaptation_rate == 1.0

    def test_set_baseline_ach(self):
        """Test setting baseline ACh."""
        ach = AcetylcholineSystem()

        ach.set_baseline_ach(0.6)

        assert ach.baseline_ach == 0.6

    def test_set_ach_bounds(self):
        """Test setting ACh bounds."""
        ach = AcetylcholineSystem()

        ach.set_ach_bounds(min_ach=0.2, max_ach=0.8)

        assert ach.min_ach == 0.2
        assert ach.max_ach == 0.8

    def test_set_ach_bounds_enforces_order(self):
        """Test max_ach > min_ach."""
        ach = AcetylcholineSystem()

        ach.set_ach_bounds(min_ach=0.5, max_ach=0.5)  # Same value

        assert ach.max_ach > ach.min_ach

    def test_get_config(self):
        """Test get_config returns configuration."""
        ach = AcetylcholineSystem(baseline_ach=0.6, adaptation_rate=0.3)

        config = ach.get_config()

        assert config["baseline_ach"] == 0.6
        assert config["adaptation_rate"] == 0.3

    def test_reset(self):
        """Test reset clears state."""
        ach = AcetylcholineSystem(baseline_ach=0.5)

        ach.update(encoding_demand=0.9, retrieval_demand=0.1)
        ach.reset()

        assert ach._ach_level == 0.5
        assert ach._current_state is None
        assert len(ach._state_history) == 0
