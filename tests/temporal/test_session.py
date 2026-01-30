"""Tests for temporal session management."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
import time

from ww.temporal.session import SessionContext, SessionManager
from ww.embedding.modulated import CognitiveMode


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_default_values(self):
        """SessionContext has correct defaults."""
        ctx = SessionContext()
        assert ctx.id is not None
        assert ctx.start_time is not None
        assert ctx.end_time is None
        assert ctx.initial_mode == CognitiveMode.ENCODING
        assert ctx.current_mode == CognitiveMode.ENCODING
        assert ctx.retrieval_count == 0
        assert ctx.encoding_count == 0

    def test_custom_values(self):
        """SessionContext accepts custom values."""
        ctx = SessionContext(
            goal="Fix bug",
            project="ww",
            tags=["testing", "bugs"],
        )
        assert ctx.goal == "Fix bug"
        assert ctx.project == "ww"
        assert "testing" in ctx.tags

    def test_duration_seconds_active(self):
        """Duration is calculated for active session."""
        ctx = SessionContext()
        time.sleep(0.1)  # Small delay
        assert ctx.duration_seconds >= 0.1

    def test_duration_seconds_closed(self):
        """Duration is fixed for closed session."""
        ctx = SessionContext()
        ctx.end_time = ctx.start_time + timedelta(seconds=10)
        assert ctx.duration_seconds == pytest.approx(10.0, abs=0.1)

    def test_is_active(self):
        """is_active reflects end_time."""
        ctx = SessionContext()
        assert ctx.is_active is True
        ctx.close()
        assert ctx.is_active is False

    def test_record_retrieval(self):
        """Record retrieval updates counts."""
        ctx = SessionContext()
        mem_id = uuid4()

        ctx.record_retrieval(mem_id)
        assert ctx.retrieval_count == 1
        assert mem_id in ctx.memory_ids_accessed

        # Second retrieval of same memory doesn't add to accessed list
        ctx.record_retrieval(mem_id)
        assert ctx.retrieval_count == 2
        assert len(ctx.memory_ids_accessed) == 1

    def test_record_encoding(self):
        """Record encoding updates counts."""
        ctx = SessionContext()
        mem_id = uuid4()

        ctx.record_encoding(mem_id)
        assert ctx.encoding_count == 1
        assert mem_id in ctx.memory_ids_accessed

    def test_set_outcome(self):
        """Set outcome updates scores."""
        ctx = SessionContext()

        ctx.set_outcome(0.8)
        assert ctx.outcome_score == 0.8
        assert ctx.success is True  # 0.8 > 0.5

        ctx.set_outcome(0.3)
        assert ctx.success is False  # 0.3 < 0.5

    def test_set_outcome_explicit_success(self):
        """Explicit success overrides threshold."""
        ctx = SessionContext()
        ctx.set_outcome(0.3, success=True)
        assert ctx.outcome_score == 0.3
        assert ctx.success is True

    def test_close(self):
        """Close sets end_time."""
        ctx = SessionContext()
        assert ctx.end_time is None
        ctx.close()
        assert ctx.end_time is not None


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.fixture
    def manager(self):
        """Create session manager."""
        return SessionManager()

    def test_start_session(self, manager):
        """Start session returns context."""
        ctx = manager.start_session(goal="Test goal", project="test-proj")
        assert ctx is not None
        assert ctx.goal == "Test goal"
        assert ctx.project == "test-proj"

    def test_current_session(self, manager):
        """Current session property returns active session."""
        ctx = manager.start_session()
        current = manager.current_session
        assert current is ctx

    def test_current_session_none(self, manager):
        """Current session is None when no session."""
        assert manager.current_session is None

    def test_end_session(self, manager):
        """End session closes session."""
        ctx = manager.start_session()
        manager.end_session(ctx.id, outcome_score=0.9)

        assert ctx.end_time is not None
        assert ctx.outcome_score == 0.9
        assert manager.current_session is None

    def test_session_context_manager(self, manager):
        """Context manager handles session lifecycle."""
        with manager.session(goal="Context test") as ctx:
            assert ctx.is_active
            assert ctx.goal == "Context test"

        assert not ctx.is_active

    def test_multiple_sessions(self, manager):
        """Multiple sessions track history."""
        ctx1 = manager.start_session(goal="First")
        manager.end_session(ctx1.id, outcome_score=0.5)

        ctx2 = manager.start_session(goal="Second")
        manager.end_session(ctx2.id, outcome_score=0.8)

        history = manager.get_recent_sessions()
        assert len(history) == 2
        assert history[0].goal == "First"
        assert history[1].goal == "Second"

    def test_record_activity_on_context(self, manager):
        """Record activity on session context."""
        ctx = manager.start_session()
        mem_id = uuid4()

        ctx.record_retrieval(mem_id)
        ctx.record_encoding(mem_id)

        assert ctx.retrieval_count == 1
        assert ctx.encoding_count == 1

    def test_get_session(self, manager):
        """Get session by ID."""
        ctx = manager.start_session(goal="Test")
        retrieved = manager.get_session(ctx.id)
        assert retrieved is ctx

    def test_get_session_not_found(self, manager):
        """Get session returns None for unknown ID."""
        assert manager.get_session("unknown-id") is None

    def test_active_count(self, manager):
        """Active count tracks open sessions."""
        assert manager.active_count == 0
        ctx1 = manager.start_session()
        assert manager.active_count == 1
        ctx2 = manager.start_session()
        assert manager.active_count == 2
        manager.end_session(ctx1.id)
        assert manager.active_count == 1

    def test_switch_mode(self, manager):
        """Switch mode changes session cognitive mode."""
        ctx = manager.start_session()
        assert ctx.current_mode == CognitiveMode.ENCODING
        manager.switch_mode(ctx.id, CognitiveMode.RETRIEVAL)
        assert ctx.current_mode == CognitiveMode.RETRIEVAL

    def test_get_stats(self, manager):
        """Get stats returns session statistics."""
        ctx = manager.start_session()
        manager.end_session(ctx.id, outcome_score=0.8, success=True)

        stats = manager.get_stats()
        assert stats["completed_sessions"] == 1
        assert stats["average_outcome"] == 0.8
        assert stats["success_rate"] == 1.0

    def test_get_active_sessions(self, manager):
        """Get active sessions returns list."""
        ctx1 = manager.start_session(goal="First")
        ctx2 = manager.start_session(goal="Second")

        active = manager.get_active_sessions()
        assert len(active) == 2
        assert ctx1 in active
        assert ctx2 in active
