"""Tests for unified temporal dynamics coordinator."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4

from t4dm.temporal.dynamics import (
    TemporalPhase,
    TemporalState,
    TemporalConfig,
    TemporalDynamics,
)
from t4dm.embedding.modulated import CognitiveMode, NeuromodulatorState


class TestTemporalPhase:
    """Tests for TemporalPhase enum."""

    def test_phases_exist(self):
        """All phases are defined."""
        assert TemporalPhase.ACTIVE.value == "active"
        assert TemporalPhase.IDLE.value == "idle"
        assert TemporalPhase.CONSOLIDATING.value == "consolidating"
        assert TemporalPhase.SLEEPING.value == "sleeping"

    def test_phase_count(self):
        """Four phases are defined."""
        assert len(TemporalPhase) == 4


class TestTemporalState:
    """Tests for TemporalState dataclass."""

    def test_default_values(self):
        """State has correct defaults."""
        state = TemporalState()
        assert state.phase == TemporalPhase.ACTIVE
        assert state.cognitive_mode == CognitiveMode.ENCODING
        assert state.retrieval_count == 0
        assert state.encoding_count == 0

    def test_timestamps(self):
        """State has timestamp fields."""
        state = TemporalState()
        assert state.session_start is not None
        assert state.last_activity is not None
        assert isinstance(state.session_start, datetime)

    def test_neuromodulator_state(self):
        """State contains neuromodulator state."""
        state = TemporalState()
        assert state.neuromodulator is not None
        assert isinstance(state.neuromodulator, NeuromodulatorState)

    def test_to_dict(self):
        """State converts to dictionary."""
        state = TemporalState()
        d = state.to_dict()
        assert "phase" in d
        assert "cognitive_mode" in d
        assert "neuromodulator" in d
        assert "session_start" in d
        assert "retrieval_count" in d

    def test_to_dict_neuromodulator(self):
        """Neuromodulator is serialized properly."""
        state = TemporalState()
        d = state.to_dict()
        neuro = d["neuromodulator"]
        assert "acetylcholine" in neuro
        assert "dopamine" in neuro
        assert "norepinephrine" in neuro
        assert "serotonin" in neuro


class TestTemporalConfig:
    """Tests for TemporalConfig dataclass."""

    def test_default_values(self):
        """Config has correct defaults."""
        config = TemporalConfig()
        assert config.idle_threshold_seconds == 300.0
        assert config.consolidation_threshold_seconds == 1800.0
        assert config.reconsolidation_enabled is True
        assert config.homeostatic_enabled is True
        assert config.serotonin_enabled is True

    def test_custom_values(self):
        """Config accepts custom values."""
        config = TemporalConfig(
            idle_threshold_seconds=600.0,
            reconsolidation_enabled=False,
        )
        assert config.idle_threshold_seconds == 600.0
        assert config.reconsolidation_enabled is False

    def test_ach_settings(self):
        """ACh settings present."""
        config = TemporalConfig()
        assert config.ach_encoding_bias == 0.8
        assert config.ach_retrieval_bias == 0.2


class TestTemporalDynamics:
    """Tests for TemporalDynamics coordinator."""

    @pytest.fixture
    def dynamics(self):
        """Create dynamics instance."""
        return TemporalDynamics()

    @pytest.fixture
    def dynamics_disabled(self):
        """Create dynamics with subsystems disabled."""
        config = TemporalConfig(
            reconsolidation_enabled=False,
            homeostatic_enabled=False,
            serotonin_enabled=False,
        )
        return TemporalDynamics(config=config)

    def test_initialization(self, dynamics):
        """Dynamics initializes properly."""
        assert dynamics.state is not None
        assert dynamics.phase == TemporalPhase.ACTIVE

    def test_initialization_with_config(self, dynamics_disabled):
        """Dynamics respects config."""
        assert dynamics_disabled._reconsolidation is None
        assert dynamics_disabled._homeostatic is None
        assert dynamics_disabled._serotonin is None

    def test_get_neuromodulator_state(self, dynamics):
        """Get neuromodulator state."""
        state = dynamics.get_neuromodulator_state()
        assert isinstance(state, NeuromodulatorState)

    def test_begin_session(self, dynamics):
        """Begin session updates state."""
        dynamics.begin_session("test-session", mode=CognitiveMode.ENCODING)
        assert dynamics.phase == TemporalPhase.ACTIVE
        assert dynamics.state.cognitive_mode == CognitiveMode.ENCODING
        assert "test-session" in dynamics._sessions

    def test_begin_session_retrieval_mode(self, dynamics):
        """Begin session in retrieval mode."""
        dynamics.begin_session("test-session", mode=CognitiveMode.RETRIEVAL)
        assert dynamics.state.cognitive_mode == CognitiveMode.RETRIEVAL

    def test_begin_session_exploration_mode(self, dynamics):
        """Begin session in exploration mode."""
        dynamics.begin_session("test-session", mode=CognitiveMode.EXPLORATION)
        assert dynamics.state.cognitive_mode == CognitiveMode.EXPLORATION

    def test_end_session(self, dynamics):
        """End session cleans up session."""
        dynamics.begin_session("test-session")
        dynamics.end_session("test-session", success=True)
        # Session should be removed
        assert "test-session" not in dynamics._sessions

    def test_end_session_not_found(self, dynamics):
        """End session handles missing session."""
        credits = dynamics.end_session("nonexistent", success=True)
        # Should not raise

    def test_record_retrieval(self, dynamics):
        """Record retrieval updates counts."""
        dynamics.begin_session("test-session")
        memory_ids = [uuid4() for _ in range(3)]
        query_embedding = np.random.randn(128)

        dynamics.record_retrieval(memory_ids, query_embedding)
        assert dynamics.state.retrieval_count == 1

    def test_record_encoding(self, dynamics):
        """Record encoding updates counts."""
        dynamics.begin_session("test-session")
        memory_id = uuid4()
        embedding = np.random.randn(128)

        dynamics.record_encoding(memory_id, embedding)
        assert dynamics.state.encoding_count == 1

    def test_record_outcome(self, dynamics):
        """Record outcome processes signal."""
        dynamics.begin_session("test-session")
        mem_id = uuid4()
        dynamics.record_retrieval([mem_id], np.random.randn(128))
        dynamics.record_outcome(1.0)
        # Should not raise

    def test_set_cognitive_mode(self, dynamics):
        """Set cognitive mode."""
        dynamics.begin_session("test-session", mode=CognitiveMode.ENCODING)
        dynamics.set_cognitive_mode(CognitiveMode.RETRIEVAL)
        assert dynamics.state.cognitive_mode == CognitiveMode.RETRIEVAL

    def test_set_cognitive_mode_exploration(self, dynamics):
        """Set exploration mode updates NE."""
        dynamics.begin_session("test-session")
        dynamics.set_cognitive_mode(CognitiveMode.EXPLORATION)
        assert dynamics.state.cognitive_mode == CognitiveMode.EXPLORATION
        assert dynamics.state.neuromodulator.norepinephrine == 0.9

    def test_get_stats(self, dynamics):
        """Get system statistics."""
        stats = dynamics.get_stats()
        assert isinstance(stats, dict)
        assert "state" in stats
        assert "active_sessions" in stats
        assert "subsystems" in stats

    def test_register_phase_callback(self, dynamics):
        """Register phase transition callback."""
        called = []
        def callback(phase):
            called.append(phase)
        dynamics.register_phase_callback(callback)
        assert len(dynamics._phase_callbacks) == 1

    def test_register_state_callback(self, dynamics):
        """Register state change callback."""
        called = []
        def callback(state):
            called.append(state)
        dynamics.register_state_callback(callback)
        assert len(dynamics._state_callbacks) == 1


class TestTemporalDynamicsPhases:
    """Tests for phase transitions."""

    @pytest.fixture
    def dynamics(self):
        """Create dynamics with short thresholds for testing."""
        config = TemporalConfig(
            idle_threshold_seconds=0.1,
            consolidation_threshold_seconds=0.5,
        )
        return TemporalDynamics(config=config)

    def test_update_phase_transitions(self, dynamics):
        """Update triggers phase transition checks."""
        dynamics.begin_session("test-session")
        # Initially active
        assert dynamics.phase == TemporalPhase.ACTIVE
        dynamics.update()
        # Should still be active if recent activity

    def test_transition_to_idle(self, dynamics):
        """Transition to idle phase after inactivity."""
        dynamics.begin_session("test-session")
        # Artificially set last_activity to past
        dynamics._state.last_activity = datetime.now() - timedelta(seconds=10)
        dynamics.update()
        # Should transition to idle
        assert dynamics.phase == TemporalPhase.IDLE


class TestTemporalDynamicsReconsolidation:
    """Tests for reconsolidation integration."""

    @pytest.fixture
    def dynamics(self):
        """Create dynamics with reconsolidation enabled."""
        return TemporalDynamics()

    def test_pending_updates_empty_initially(self, dynamics):
        """Pending updates empty initially."""
        assert len(dynamics._pending_updates) == 0

    def test_record_retrieval_adds_pending(self, dynamics):
        """Record retrieval adds to pending updates."""
        dynamics.begin_session("test-session")
        dynamics.record_retrieval([uuid4()], np.random.randn(128))
        assert len(dynamics._pending_updates) > 0

    def test_record_outcome_clears_pending(self, dynamics):
        """Record outcome clears pending updates."""
        dynamics.begin_session("test-session")
        dynamics.record_retrieval([uuid4()], np.random.randn(128))
        dynamics.record_outcome(1.0)
        assert len(dynamics._pending_updates) == 0


class TestTemporalDynamicsActivity:
    """Tests for activity tracking."""

    @pytest.fixture
    def dynamics(self):
        """Create dynamics instance."""
        return TemporalDynamics()

    def test_multiple_retrievals(self, dynamics):
        """Track multiple retrievals."""
        dynamics.begin_session("test-session")
        for _ in range(5):
            dynamics.record_retrieval([uuid4()], np.random.randn(128))
        assert dynamics.state.retrieval_count == 5

    def test_multiple_encodings(self, dynamics):
        """Track multiple encodings."""
        dynamics.begin_session("test-session")
        for _ in range(3):
            dynamics.record_encoding(uuid4(), np.random.randn(128))
        assert dynamics.state.encoding_count == 3

    def test_mixed_activity(self, dynamics):
        """Track mixed activity."""
        dynamics.begin_session("test-session")
        dynamics.record_retrieval([uuid4()], np.random.randn(128))
        dynamics.record_encoding(uuid4(), np.random.randn(128))
        dynamics.record_retrieval([uuid4()], np.random.randn(128))
        assert dynamics.state.retrieval_count == 2
        assert dynamics.state.encoding_count == 1
