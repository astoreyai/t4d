"""Tests for integrated neuromodulator system."""

import pytest
import numpy as np
from datetime import datetime
from uuid import uuid4

from t4dm.learning.neuromodulators import (
    NeuromodulatorState,
    NeuromodulatorOrchestra,
    create_neuromodulator_orchestra,
)
from t4dm.learning.dopamine import DopamineSystem
from t4dm.learning.norepinephrine import NorepinephrineSystem
from t4dm.learning.acetylcholine import AcetylcholineSystem
from t4dm.learning.serotonin import SerotoninSystem
from t4dm.learning.inhibition import InhibitoryNetwork


class TestNeuromodulatorState:
    """Tests for NeuromodulatorState dataclass."""

    def test_state_creation(self):
        """Create state with all fields."""
        state = NeuromodulatorState(
            dopamine_rpe=0.5,
            norepinephrine_gain=1.2,
            acetylcholine_mode="encoding",
            serotonin_mood=0.6,
            inhibition_sparsity=0.8,
        )
        assert state.dopamine_rpe == 0.5
        assert state.norepinephrine_gain == 1.2
        assert state.acetylcholine_mode == "encoding"
        assert state.serotonin_mood == 0.6
        assert state.inhibition_sparsity == 0.8

    def test_state_timestamp(self):
        """State has timestamp."""
        state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        assert state.timestamp is not None
        assert isinstance(state.timestamp, datetime)

    def test_effective_learning_rate_encoding(self):
        """Learning rate boosted in encoding mode."""
        state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="encoding",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        lr = state.effective_learning_rate
        assert lr > 1.0  # Boosted (1.0 * 1.3 * (0.7 + 0.6*1) = 1.69)

    def test_effective_learning_rate_retrieval(self):
        """Learning rate modestly boosted in retrieval mode.

        BIO-002/LOGIC-001 FIX: Both encoding (2.0x) and retrieval (1.2x) modes
        should BOOST learning, not reduce it. This matches biological evidence
        that ACh enhances plasticity in both modes, just more strongly during
        active encoding.
        """
        state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="retrieval",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        lr = state.effective_learning_rate
        # Retrieval mode: 1.2x boost (less than encoding's 2.0x)
        assert lr >= 1.0  # Modest boost, not reduction

    def test_exploration_exploitation_balance_high_arousal(self):
        """High arousal shifts toward exploration."""
        state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.5,  # High arousal
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        balance = state.exploration_exploitation_balance
        assert balance > 0  # Exploration bias

    def test_exploration_exploitation_balance_encoding(self):
        """Encoding mode shifts toward exploration."""
        state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="encoding",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        balance = state.exploration_exploitation_balance
        assert balance > 0  # Exploration bias from encoding mode

    def test_exploration_exploitation_balance_retrieval(self):
        """Retrieval mode shifts toward exploitation."""
        state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="retrieval",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        balance = state.exploration_exploitation_balance
        assert balance < 0  # Exploitation bias

    def test_to_dict(self):
        """State converts to dictionary."""
        state = NeuromodulatorState(
            dopamine_rpe=0.2,
            norepinephrine_gain=1.1,
            acetylcholine_mode="balanced",
            serotonin_mood=0.6,
            inhibition_sparsity=0.7,
        )
        d = state.to_dict()
        assert d["dopamine_rpe"] == 0.2
        assert d["norepinephrine_gain"] == 1.1
        assert d["acetylcholine_mode"] == "balanced"
        assert "effective_learning_rate" in d
        assert "exploration_balance" in d
        assert "timestamp" in d


class TestNeuromodulatorOrchestra:
    """Tests for NeuromodulatorOrchestra class."""

    @pytest.fixture
    def orchestra(self):
        """Create orchestra instance."""
        return NeuromodulatorOrchestra()

    def test_orchestra_creation(self, orchestra):
        """Orchestra initializes with all systems."""
        assert orchestra.dopamine is not None
        assert orchestra.norepinephrine is not None
        assert orchestra.acetylcholine is not None
        assert orchestra.serotonin is not None
        assert orchestra.inhibitory is not None

    def test_orchestra_systems_types(self, orchestra):
        """Systems are correct types."""
        assert isinstance(orchestra.dopamine, DopamineSystem)
        assert isinstance(orchestra.norepinephrine, NorepinephrineSystem)
        assert isinstance(orchestra.acetylcholine, AcetylcholineSystem)
        assert isinstance(orchestra.serotonin, SerotoninSystem)
        assert isinstance(orchestra.inhibitory, InhibitoryNetwork)

    def test_get_current_state_none_initially(self, orchestra):
        """Current state is None before processing."""
        assert orchestra.get_current_state() is None

    def test_process_query(self, orchestra):
        """Process query updates state."""
        query_embedding = np.random.randn(128)
        state = orchestra.process_query(query_embedding, is_question=True)
        assert isinstance(state, NeuromodulatorState)
        assert orchestra.get_current_state() is not None

    def test_process_query_statement(self, orchestra):
        """Process statement query."""
        query_embedding = np.random.randn(128)
        state = orchestra.process_query(query_embedding, is_question=False)
        assert isinstance(state, NeuromodulatorState)

    def test_process_query_with_importance(self, orchestra):
        """Process query with explicit importance."""
        query_embedding = np.random.randn(128)
        state = orchestra.process_query(
            query_embedding, is_question=False, explicit_importance=0.9
        )
        assert isinstance(state, NeuromodulatorState)

    def test_get_learning_rate(self, orchestra):
        """Get modulated learning rate."""
        query_embedding = np.random.randn(128)
        orchestra.process_query(query_embedding)
        lr = orchestra.get_learning_rate(0.01)
        assert lr > 0

    def test_get_retrieval_threshold(self, orchestra):
        """Get modulated retrieval threshold."""
        threshold = orchestra.get_retrieval_threshold(0.5)
        assert 0 < threshold < 1

    def test_should_encode(self, orchestra):
        """Check encoding priority."""
        result = orchestra.should_encode()
        assert isinstance(result, bool)

    def test_should_retrieve(self, orchestra):
        """Check retrieval priority."""
        result = orchestra.should_retrieve()
        assert isinstance(result, bool)

    def test_get_attention_weights(self, orchestra):
        """Get attention weights for sources."""
        sources = ["episodic", "semantic", "procedural"]
        weights = orchestra.get_attention_weights(sources)
        assert isinstance(weights, dict)
        for source in sources:
            assert source in weights

    def test_start_session(self, orchestra):
        """Start tracking session."""
        orchestra.start_session("test-session", goal="test goal")
        # Should not raise

    def test_end_session(self, orchestra):
        """End session returns credits."""
        orchestra.start_session("test-session")
        credits = orchestra.end_session("test-session", outcome=0.8)
        assert isinstance(credits, dict)

    def test_reset(self, orchestra):
        """Reset clears state."""
        query_embedding = np.random.randn(128)
        orchestra.process_query(query_embedding)
        assert orchestra.get_current_state() is not None
        orchestra.reset()
        assert orchestra.get_current_state() is None

    def test_get_stats(self, orchestra):
        """Get combined statistics."""
        stats = orchestra.get_stats()
        assert "dopamine" in stats
        assert "norepinephrine" in stats
        assert "acetylcholine" in stats
        assert "serotonin" in stats
        assert "inhibitory" in stats
        assert "total_states" in stats


class TestCreateNeuromodulatorOrchestra:
    """Tests for factory function."""

    def test_create_default(self):
        """Create with default config."""
        orchestra = create_neuromodulator_orchestra()
        assert orchestra is not None
        assert isinstance(orchestra, NeuromodulatorOrchestra)

    def test_create_with_configs(self):
        """Create with custom configs."""
        orchestra = create_neuromodulator_orchestra(
            dopamine_config={},
            norepinephrine_config={},
            acetylcholine_config={},
            serotonin_config={},
            inhibitory_config={},
        )
        assert orchestra is not None


class TestStateProperties:
    """Tests for NeuromodulatorState computed properties."""

    def test_high_arousal_high_learning(self):
        """High arousal increases learning rate."""
        state_low = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=0.5,
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        state_high = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.5,
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        assert state_high.effective_learning_rate > state_low.effective_learning_rate

    def test_mood_affects_learning(self):
        """Mood affects learning rate."""
        state_low_mood = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="balanced",
            serotonin_mood=0.1,  # Low mood
            inhibition_sparsity=0.5,
        )
        state_moderate_mood = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,  # Optimal mood
            inhibition_sparsity=0.5,
        )
        # Moderate mood should be optimal
        assert state_moderate_mood.effective_learning_rate >= state_low_mood.effective_learning_rate

    def test_exploration_balance_clipped(self):
        """Exploration balance is clipped to [-1, 1]."""
        # Extreme values
        state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=10.0,  # Very high
            acetylcholine_mode="encoding",
            serotonin_mood=0.0,
            inhibition_sparsity=0.5,
        )
        balance = state.exploration_exploitation_balance
        assert -1.0 <= balance <= 1.0
