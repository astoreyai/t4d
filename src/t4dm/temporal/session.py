"""
Session Context Management for World Weaver.

Manages session lifecycle and context for temporal dynamics.
Sessions provide the temporal boundary for credit assignment
and neuromodulator state management.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from t4dm.embedding.modulated import CognitiveMode

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """
    Context for a single session.

    Tracks temporal and cognitive context throughout session lifecycle.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    # Cognitive state
    initial_mode: CognitiveMode = CognitiveMode.ENCODING
    current_mode: CognitiveMode = CognitiveMode.ENCODING

    # Session metadata
    goal: str | None = None
    project: str | None = None
    tags: list[str] = field(default_factory=list)

    # Activity tracking
    retrieval_count: int = 0
    encoding_count: int = 0
    memory_ids_accessed: list[UUID] = field(default_factory=list)

    # Outcomes
    outcome_score: float | None = None
    success: bool | None = None

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.end_time is None

    def record_retrieval(self, memory_id: UUID) -> None:
        """Record a memory retrieval."""
        self.retrieval_count += 1
        if memory_id not in self.memory_ids_accessed:
            self.memory_ids_accessed.append(memory_id)

    def record_encoding(self, memory_id: UUID) -> None:
        """Record a memory encoding."""
        self.encoding_count += 1
        if memory_id not in self.memory_ids_accessed:
            self.memory_ids_accessed.append(memory_id)

    def set_outcome(self, score: float, success: bool | None = None) -> None:
        """Set session outcome."""
        self.outcome_score = score
        self.success = success if success is not None else (score > 0.5)

    def close(self) -> None:
        """Close the session."""
        self.end_time = datetime.now()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "initial_mode": self.initial_mode.value,
            "current_mode": self.current_mode.value,
            "goal": self.goal,
            "project": self.project,
            "tags": self.tags,
            "retrieval_count": self.retrieval_count,
            "encoding_count": self.encoding_count,
            "memory_count": len(self.memory_ids_accessed),
            "outcome_score": self.outcome_score,
            "success": self.success,
            "is_active": self.is_active,
        }


class SessionManager:
    """
    Manages session lifecycle and context.

    Provides a clean API for session management that integrates
    with temporal dynamics and neuromodulator systems.

    Example:
        manager = SessionManager()

        # Context manager style
        with manager.session("my-goal") as ctx:
            # Do work
            ctx.record_retrieval(memory_id)
            ctx.set_outcome(1.0)

        # Manual style
        ctx = manager.start_session(goal="debug issue")
        manager.end_session(ctx.id, outcome_score=0.8)
    """

    def __init__(self, max_history: int = 100):
        """Initialize session manager."""
        self._active_sessions: dict[str, SessionContext] = {}
        self._completed_sessions: list[SessionContext] = []
        self._max_history = max_history
        self._current_session_id: str | None = None

    @property
    def current_session(self) -> SessionContext | None:
        """Get current active session."""
        if self._current_session_id:
            return self._active_sessions.get(self._current_session_id)
        return None

    @property
    def active_count(self) -> int:
        """Get number of active sessions."""
        return len(self._active_sessions)

    def start_session(
        self,
        goal: str | None = None,
        mode: CognitiveMode = CognitiveMode.ENCODING,
        project: str | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> SessionContext:
        """
        Start a new session.

        Args:
            goal: Session goal for credit assignment
            mode: Initial cognitive mode
            project: Project context
            tags: Session tags
            session_id: Optional custom session ID

        Returns:
            Session context
        """
        ctx = SessionContext(
            id=session_id or str(uuid4()),
            initial_mode=mode,
            current_mode=mode,
            goal=goal,
            project=project,
            tags=tags or [],
        )

        self._active_sessions[ctx.id] = ctx
        self._current_session_id = ctx.id

        logger.debug(f"Started session {ctx.id} with goal={goal}")
        return ctx

    def end_session(
        self,
        session_id: str,
        outcome_score: float | None = None,
        success: bool | None = None,
    ) -> SessionContext | None:
        """
        End a session.

        Args:
            session_id: Session to end
            outcome_score: Final outcome score
            success: Whether session was successful

        Returns:
            Completed session context or None if not found
        """
        ctx = self._active_sessions.pop(session_id, None)
        if ctx is None:
            logger.warning(f"Session {session_id} not found")
            return None

        # Set outcome and close
        if outcome_score is not None:
            ctx.set_outcome(outcome_score, success)
        ctx.close()

        # Add to history
        self._completed_sessions.append(ctx)
        if len(self._completed_sessions) > self._max_history:
            self._completed_sessions.pop(0)

        # Update current session
        if self._current_session_id == session_id:
            self._current_session_id = None

        logger.debug(f"Ended session {session_id} with outcome={outcome_score}")
        return ctx

    def get_session(self, session_id: str) -> SessionContext | None:
        """Get session by ID (active or completed)."""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        for ctx in self._completed_sessions:
            if ctx.id == session_id:
                return ctx

        return None

    @contextmanager
    def session(
        self,
        goal: str | None = None,
        mode: CognitiveMode = CognitiveMode.ENCODING,
        project: str | None = None,
    ):
        """
        Context manager for session lifecycle.

        Example:
            with manager.session("my goal") as ctx:
                # Work within session
                ctx.record_retrieval(memory_id)
        """
        ctx = self.start_session(goal=goal, mode=mode, project=project)
        try:
            yield ctx
        finally:
            self.end_session(ctx.id, outcome_score=ctx.outcome_score)

    def switch_mode(
        self,
        session_id: str,
        mode: CognitiveMode,
    ) -> None:
        """Switch cognitive mode for a session."""
        ctx = self._active_sessions.get(session_id)
        if ctx:
            ctx.current_mode = mode

    def get_stats(self) -> dict:
        """Get session statistics."""
        completed_outcomes = [
            s.outcome_score for s in self._completed_sessions
            if s.outcome_score is not None
        ]

        return {
            "active_sessions": self.active_count,
            "completed_sessions": len(self._completed_sessions),
            "current_session_id": self._current_session_id,
            "average_outcome": (
                sum(completed_outcomes) / len(completed_outcomes)
                if completed_outcomes else None
            ),
            "success_rate": (
                sum(1 for s in self._completed_sessions if s.success) /
                len(self._completed_sessions)
                if self._completed_sessions else None
            ),
        }

    def get_active_sessions(self) -> list[SessionContext]:
        """Get all active sessions."""
        return list(self._active_sessions.values())

    def get_recent_sessions(self, limit: int = 10) -> list[SessionContext]:
        """Get recent completed sessions."""
        return self._completed_sessions[-limit:]


# Singleton instance
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


__all__ = [
    "SessionContext",
    "SessionManager",
    "get_session_manager",
]
