"""
Unit tests for temporal dynamics layer.

Tests TemporalDynamics coordinator and SessionManager.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4

from t4dm.embedding.modulated import CognitiveMode, NeuromodulatorState
from t4dm.temporal.dynamics import (
    TemporalPhase,
    TemporalState,
    TemporalConfig,
    TemporalDynamics,
    create_temporal_dynamics,
)
from t4dm.temporal.session import (
    SessionContext,
    SessionManager,
    get_session_manager,
)


class TestTemporalPhase:
    """Tests for TemporalPhase enum."""

    def test_phase_values(self):
        assert TemporalPhase.ACTIVE.value == "active"
        assert TemporalPhase.IDLE.value == "idle"
        assert TemporalPhase.CONSOLIDATING.value == "consolidating"
        assert TemporalPhase.SLEEPING.value == "sleeping"


class TestTemporalState:
    """Tests for TemporalState dataclass."""

    def test_default_values(self):
        state = TemporalState()
        assert state.phase == TemporalPhase.ACTIVE
        assert state.cognitive_mode == CognitiveMode.ENCODING
        assert state.retrieval_count == 0
        assert state.encoding_count == 0

    def test_to_dict(self):
        state = TemporalState()
        data = state.to_dict()

        assert "phase" in data
        assert "cognitive_mode" in data
        assert "neuromodulator" in data
        assert data["phase"] == "active"


class TestTemporalConfig:
    """Tests for TemporalConfig."""

    def test_default_values(self):
        config = TemporalConfig()
        assert config.idle_threshold_seconds == 300.0
        assert config.consolidation_threshold_seconds == 1800.0
        assert config.reconsolidation_enabled is True

    def test_custom_values(self):
        config = TemporalConfig(
            idle_threshold_seconds=60.0,
            reconsolidation_enabled=False,
        )
        assert config.idle_threshold_seconds == 60.0
        assert config.reconsolidation_enabled is False


class TestTemporalDynamics:
    """Tests for TemporalDynamics coordinator."""

    @pytest.fixture
    def dynamics(self):
        return TemporalDynamics()

    def test_creation(self, dynamics):
        assert dynamics.phase == TemporalPhase.ACTIVE
        assert dynamics.state is not None

    def test_begin_session(self, dynamics):
        dynamics.begin_session("test-session", mode=CognitiveMode.ENCODING)

        assert dynamics.phase == TemporalPhase.ACTIVE
        assert dynamics.state.cognitive_mode == CognitiveMode.ENCODING

    def test_begin_session_retrieval_mode(self, dynamics):
        dynamics.begin_session("test-session", mode=CognitiveMode.RETRIEVAL)

        neuro = dynamics.get_neuromodulator_state()
        assert neuro.acetylcholine < 0.3  # Low ACh for retrieval

    def test_begin_session_encoding_mode(self, dynamics):
        dynamics.begin_session("test-session", mode=CognitiveMode.ENCODING)

        neuro = dynamics.get_neuromodulator_state()
        assert neuro.acetylcholine > 0.7  # High ACh for encoding

    def test_end_session(self, dynamics):
        dynamics.begin_session("test-session")
        dynamics.end_session("test-session", outcome_score=0.8)

        # Should transition to idle when no sessions
        assert dynamics.phase == TemporalPhase.IDLE

    def test_set_cognitive_mode(self, dynamics):
        dynamics.set_cognitive_mode(CognitiveMode.EXPLORATION)

        assert dynamics.state.cognitive_mode == CognitiveMode.EXPLORATION

    def test_record_retrieval(self, dynamics):
        memory_ids = [uuid4(), uuid4()]
        query_emb = np.random.randn(128).astype(np.float32)

        dynamics.record_retrieval(memory_ids, query_emb)

        assert dynamics.state.retrieval_count == 1

    def test_record_encoding(self, dynamics):
        memory_id = uuid4()
        embedding = np.random.randn(128).astype(np.float32)

        dynamics.record_encoding(memory_id, embedding)

        assert dynamics.state.encoding_count == 1

    def test_record_outcome(self, dynamics):
        # First record a retrieval
        memory_ids = [uuid4()]
        query_emb = np.random.randn(128).astype(np.float32)
        dynamics.record_retrieval(memory_ids, query_emb)

        # Then record outcome
        updates = dynamics.record_outcome(0.9)

        assert dynamics.state.pending_reconsolidations == 0
        assert dynamics.state.update_count >= 1

    def test_update_phase_transition(self, dynamics):
        # Manually set last_activity to past
        dynamics._state.last_activity = datetime.now() - timedelta(seconds=400)

        dynamics.update()

        assert dynamics.phase == TemporalPhase.IDLE

    def test_get_stats(self, dynamics):
        stats = dynamics.get_stats()

        assert "state" in stats
        assert "active_sessions" in stats
        assert "subsystems" in stats

    def test_phase_callback(self, dynamics):
        phases_seen = []

        def callback(phase):
            phases_seen.append(phase)

        dynamics.register_phase_callback(callback)
        dynamics.begin_session("test")
        dynamics.end_session("test")

        assert TemporalPhase.IDLE in phases_seen


class TestCreateTemporalDynamics:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        dynamics = create_temporal_dynamics()
        assert dynamics is not None
        assert dynamics.phase == TemporalPhase.ACTIVE

    def test_create_with_config(self):
        config = TemporalConfig(reconsolidation_enabled=False)
        dynamics = create_temporal_dynamics(config=config)

        assert dynamics._reconsolidation is None


class TestSessionContext:
    """Tests for SessionContext."""

    def test_creation(self):
        ctx = SessionContext()
        assert ctx.id is not None
        assert ctx.is_active is True

    def test_with_goal(self):
        ctx = SessionContext(goal="debug issue", project="ww")
        assert ctx.goal == "debug issue"
        assert ctx.project == "ww"

    def test_record_retrieval(self):
        ctx = SessionContext()
        memory_id = uuid4()

        ctx.record_retrieval(memory_id)

        assert ctx.retrieval_count == 1
        assert memory_id in ctx.memory_ids_accessed

    def test_record_encoding(self):
        ctx = SessionContext()
        memory_id = uuid4()

        ctx.record_encoding(memory_id)

        assert ctx.encoding_count == 1

    def test_set_outcome(self):
        ctx = SessionContext()

        ctx.set_outcome(0.9)

        assert ctx.outcome_score == 0.9
        assert ctx.success is True

    def test_set_outcome_failure(self):
        ctx = SessionContext()

        ctx.set_outcome(0.3)

        assert ctx.outcome_score == 0.3
        assert ctx.success is False

    def test_close(self):
        ctx = SessionContext()

        ctx.close()

        assert ctx.is_active is False
        assert ctx.end_time is not None

    def test_duration(self):
        ctx = SessionContext()
        # Just check it returns a float
        assert isinstance(ctx.duration_seconds, float)

    def test_to_dict(self):
        ctx = SessionContext(goal="test", project="ww")
        data = ctx.to_dict()

        assert data["goal"] == "test"
        assert data["project"] == "ww"
        assert "duration_seconds" in data


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def manager(self):
        return SessionManager()

    def test_start_session(self, manager):
        ctx = manager.start_session(goal="test goal")

        assert ctx is not None
        assert ctx.goal == "test goal"
        assert manager.active_count == 1

    def test_end_session(self, manager):
        ctx = manager.start_session()
        result = manager.end_session(ctx.id, outcome_score=0.8)

        assert result is not None
        assert result.outcome_score == 0.8
        assert manager.active_count == 0

    def test_end_nonexistent_session(self, manager):
        result = manager.end_session("nonexistent")
        assert result is None

    def test_current_session(self, manager):
        ctx = manager.start_session()

        assert manager.current_session == ctx

    def test_get_session(self, manager):
        ctx = manager.start_session()

        found = manager.get_session(ctx.id)
        assert found == ctx

    def test_context_manager(self, manager):
        with manager.session(goal="context goal") as ctx:
            ctx.record_retrieval(uuid4())
            ctx.set_outcome(1.0)

        assert manager.active_count == 0
        assert len(manager._completed_sessions) == 1

    def test_switch_mode(self, manager):
        ctx = manager.start_session()

        manager.switch_mode(ctx.id, CognitiveMode.RETRIEVAL)

        assert ctx.current_mode == CognitiveMode.RETRIEVAL

    def test_get_stats(self, manager):
        manager.start_session()

        stats = manager.get_stats()

        assert stats["active_sessions"] == 1

    def test_get_active_sessions(self, manager):
        manager.start_session()
        manager.start_session()

        active = manager.get_active_sessions()

        assert len(active) == 2

    def test_get_recent_sessions(self, manager):
        for i in range(5):
            ctx = manager.start_session()
            manager.end_session(ctx.id, outcome_score=0.5)

        recent = manager.get_recent_sessions(limit=3)

        assert len(recent) == 3


class TestGetSessionManager:
    """Tests for singleton session manager."""

    def test_returns_same_instance(self):
        m1 = get_session_manager()
        m2 = get_session_manager()

        assert m1 is m2

    def test_returns_session_manager(self):
        m = get_session_manager()
        assert isinstance(m, SessionManager)


class TestDynamicsIntegration:
    """Integration tests for temporal dynamics."""

    @pytest.fixture
    def dynamics(self):
        return TemporalDynamics()

    @pytest.fixture
    def manager(self):
        return SessionManager()

    def test_full_session_flow(self, dynamics, manager):
        """Test complete session workflow."""
        # Start session
        ctx = manager.start_session(goal="test task")
        dynamics.begin_session(ctx.id, mode=CognitiveMode.ENCODING)

        # Record activity
        memory_id = uuid4()
        embedding = np.random.randn(128).astype(np.float32)

        dynamics.record_encoding(memory_id, embedding)
        ctx.record_encoding(memory_id)

        # Switch to retrieval
        dynamics.set_cognitive_mode(CognitiveMode.RETRIEVAL)
        manager.switch_mode(ctx.id, CognitiveMode.RETRIEVAL)

        dynamics.record_retrieval([memory_id], embedding)
        ctx.record_retrieval(memory_id)

        # Record outcome
        ctx.set_outcome(0.9)
        dynamics.record_outcome(0.9, session_id=ctx.id)

        # End session
        dynamics.end_session(ctx.id, outcome_score=0.9)
        manager.end_session(ctx.id, outcome_score=0.9)

        # Verify
        assert dynamics.state.encoding_count == 1
        assert dynamics.state.retrieval_count == 1
        assert ctx.success is True
