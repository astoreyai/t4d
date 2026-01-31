"""
Unified Temporal Dynamics Coordinator for World Weaver.

Integrates all temporal aspects of the memory system:
1. Neuromodulator state management (ACh, DA, NE, 5-HT)
2. State-dependent embedding modulation
3. Reconsolidation and plasticity
4. Long-term credit assignment
5. Sleep-based consolidation

Hinton-inspired: Memory representations are dynamic patterns that
depend on current temporal and cognitive context, not static vectors.

CompBio-inspired: The brain's memory systems operate on multiple
timescales - milliseconds for retrieval, hours for consolidation,
days for long-term plasticity.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID

import numpy as np

from t4dm.embedding.modulated import (
    CognitiveMode,
    ModulationConfig,
    NeuromodulatorState,
)
from t4dm.learning.homeostatic import HomeostaticPlasticity, HomeostaticState
from t4dm.learning.reconsolidation import ReconsolidationEngine, ReconsolidationUpdate
from t4dm.learning.serotonin import SerotoninSystem

logger = logging.getLogger(__name__)


class TemporalPhase(Enum):
    """Current temporal phase of the system."""
    ACTIVE = "active"           # Active encoding/retrieval
    IDLE = "idle"               # Low activity, maintenance
    CONSOLIDATING = "consolidating"  # Offline consolidation
    SLEEPING = "sleeping"       # Full sleep cycle


@dataclass
class TemporalState:
    """
    Current temporal state of the memory system.

    Captures the full temporal context for memory operations.
    """
    phase: TemporalPhase = TemporalPhase.ACTIVE
    cognitive_mode: CognitiveMode = CognitiveMode.ENCODING
    neuromodulator: NeuromodulatorState = field(default_factory=NeuromodulatorState)

    # Timing
    session_start: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    phase_duration: timedelta = field(default_factory=lambda: timedelta(0))

    # Activity metrics
    retrieval_count: int = 0
    encoding_count: int = 0
    update_count: int = 0

    # Health metrics
    homeostatic_state: HomeostaticState | None = None
    active_traces: int = 0
    pending_reconsolidations: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "cognitive_mode": self.cognitive_mode.value,
            "neuromodulator": {
                "acetylcholine": self.neuromodulator.acetylcholine,
                "dopamine": self.neuromodulator.dopamine,
                "norepinephrine": self.neuromodulator.norepinephrine,
                "serotonin": self.neuromodulator.serotonin,
            },
            "session_start": self.session_start.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "phase_duration_seconds": self.phase_duration.total_seconds(),
            "retrieval_count": self.retrieval_count,
            "encoding_count": self.encoding_count,
            "update_count": self.update_count,
            "active_traces": self.active_traces,
            "pending_reconsolidations": self.pending_reconsolidations,
        }


@dataclass
class TemporalConfig:
    """Configuration for temporal dynamics."""

    # Phase transition thresholds
    idle_threshold_seconds: float = 300.0  # 5 min to idle
    consolidation_threshold_seconds: float = 1800.0  # 30 min to consolidate

    # Neuromodulator dynamics
    ach_encoding_bias: float = 0.8  # ACh during encoding
    ach_retrieval_bias: float = 0.2  # ACh during retrieval
    ne_decay_rate: float = 0.1  # NE decay per minute
    da_baseline: float = 0.5  # Baseline dopamine

    # Plasticity settings
    reconsolidation_enabled: bool = True
    homeostatic_enabled: bool = True
    serotonin_enabled: bool = True

    # Integration settings
    update_interval_seconds: float = 1.0  # State update frequency
    trace_cleanup_interval: float = 60.0  # Trace cleanup frequency


class TemporalDynamics:
    """
    Unified coordinator for temporal aspects of memory.

    Manages the flow of time through the memory system, coordinating:
    1. Phase transitions (active -> idle -> consolidating)
    2. Neuromodulator state updates
    3. Plasticity and reconsolidation
    4. Long-term credit assignment

    Example:
        dynamics = TemporalDynamics()

        # Start session in encoding mode
        dynamics.begin_session("session-123", mode=CognitiveMode.ENCODING)

        # Record retrieval and outcome
        dynamics.record_retrieval(memory_ids, query_emb)
        dynamics.record_outcome(1.0)  # Positive outcome

        # Get current state for embedding modulation
        state = dynamics.get_neuromodulator_state()

        # End session
        dynamics.end_session("session-123", success=True)
    """

    def __init__(
        self,
        config: TemporalConfig | None = None,
        modulation_config: ModulationConfig | None = None,
    ):
        """Initialize temporal dynamics."""
        self._config = config or TemporalConfig()
        self._modulation_config = modulation_config or ModulationConfig()

        # Current state
        self._state = TemporalState()

        # Sub-systems
        self._reconsolidation = ReconsolidationEngine() if self._config.reconsolidation_enabled else None
        self._homeostatic = HomeostaticPlasticity() if self._config.homeostatic_enabled else None
        self._serotonin = SerotoninSystem() if self._config.serotonin_enabled else None

        # Active sessions
        self._sessions: dict[str, datetime] = {}

        # Pending reconsolidation updates
        self._pending_updates: list[dict] = []

        # Event callbacks
        self._phase_callbacks: list[Callable[[TemporalPhase], None]] = []
        self._state_callbacks: list[Callable[[TemporalState], None]] = []

        # Background task
        self._update_task: asyncio.Task | None = None
        self._running = False

        logger.info("TemporalDynamics initialized")

    @property
    def state(self) -> TemporalState:
        """Get current temporal state."""
        return self._state

    @property
    def phase(self) -> TemporalPhase:
        """Get current phase."""
        return self._state.phase

    def get_neuromodulator_state(self) -> NeuromodulatorState:
        """Get current neuromodulator state for embedding modulation."""
        return self._state.neuromodulator

    def begin_session(
        self,
        session_id: str,
        mode: CognitiveMode = CognitiveMode.ENCODING,
        goal: str | None = None,
    ) -> None:
        """
        Begin a new session.

        Args:
            session_id: Unique session identifier
            mode: Initial cognitive mode
            goal: Optional session goal for credit assignment
        """
        now = datetime.now()
        self._sessions[session_id] = now

        # Update state
        self._state.phase = TemporalPhase.ACTIVE
        self._state.cognitive_mode = mode
        self._state.session_start = now
        self._state.last_activity = now

        # Set neuromodulator state for mode
        if mode == CognitiveMode.ENCODING:
            self._state.neuromodulator = NeuromodulatorState.for_encoding()
        elif mode == CognitiveMode.RETRIEVAL:
            self._state.neuromodulator = NeuromodulatorState.for_retrieval()
        elif mode == CognitiveMode.EXPLORATION:
            self._state.neuromodulator = NeuromodulatorState.for_exploration()

        # Start serotonin context if enabled
        if self._serotonin:
            self._serotonin.start_context(session_id, goal)

        logger.debug(f"Session {session_id} started in {mode.value} mode")

    def end_session(
        self,
        session_id: str,
        success: bool | None = None,
        outcome_score: float | None = None,
    ) -> None:
        """
        End a session.

        Args:
            session_id: Session identifier
            success: Whether session was successful (for credit assignment)
            outcome_score: Numerical outcome score (0-1)
        """
        if session_id in self._sessions:
            del self._sessions[session_id]

        # End serotonin context with outcome
        if self._serotonin:
            self._serotonin.end_context(session_id)
            # Distribute credit via eligibility traces
            if outcome_score is not None:
                self._serotonin.receive_outcome(
                    outcome_score=outcome_score,
                    context_id=session_id,
                )

        # Transition to idle if no active sessions
        if not self._sessions:
            self._transition_to_phase(TemporalPhase.IDLE)

        logger.debug(f"Session {session_id} ended with outcome={outcome_score}")

    def set_cognitive_mode(self, mode: CognitiveMode) -> None:
        """
        Set cognitive mode.

        Adjusts neuromodulator state accordingly.
        """
        self._state.cognitive_mode = mode

        # Update neuromodulator for mode
        if mode == CognitiveMode.ENCODING:
            self._state.neuromodulator.acetylcholine = self._config.ach_encoding_bias
        elif mode == CognitiveMode.RETRIEVAL:
            self._state.neuromodulator.acetylcholine = self._config.ach_retrieval_bias
        elif mode == CognitiveMode.EXPLORATION:
            self._state.neuromodulator.norepinephrine = 0.9
        elif mode == CognitiveMode.EXPLOITATION:
            self._state.neuromodulator.dopamine = 0.8

        self._state.last_activity = datetime.now()

    def record_retrieval(
        self,
        memory_ids: list[UUID],
        query_embedding: np.ndarray,
        scores: list[float] | None = None,
        original_embeddings: list[np.ndarray] | None = None,
    ) -> None:
        """
        Record a retrieval event.

        Creates eligibility traces for credit assignment.

        Args:
            memory_ids: IDs of retrieved memories
            query_embedding: Query embedding used for retrieval
            scores: Optional relevance scores for each memory
            original_embeddings: Optional original embeddings for reconsolidation
                                (P3-QUALITY-001 fix: enables accurate embedding updates)
        """
        self._state.retrieval_count += 1
        self._state.last_activity = datetime.now()

        # Create eligibility traces if serotonin enabled
        if self._serotonin:
            for i, memory_id in enumerate(memory_ids):
                score = scores[i] if scores else 1.0
                self._serotonin.add_eligibility(
                    memory_id=memory_id,
                    strength=score,
                )
            self._state.active_traces = len(self._serotonin.get_memories_with_traces())

        # Store for potential reconsolidation
        if self._reconsolidation:
            for i, memory_id in enumerate(memory_ids):
                # P3-QUALITY-001 FIX: Store original embedding if provided
                original_emb = None
                if original_embeddings is not None and i < len(original_embeddings):
                    original_emb = original_embeddings[i].copy()

                self._pending_updates.append({
                    "memory_id": memory_id,
                    "query_embedding": query_embedding.copy(),
                    "original_embedding": original_emb,
                    "timestamp": datetime.now(),
                })
            self._state.pending_reconsolidations = len(self._pending_updates)

    def record_encoding(self, memory_id: UUID, embedding: np.ndarray) -> None:
        """Record an encoding event."""
        self._state.encoding_count += 1
        self._state.last_activity = datetime.now()

        # Update homeostatic statistics
        if self._homeostatic:
            self._homeostatic.update_statistics(embedding)

    def record_outcome(
        self,
        outcome_score: float,
        session_id: str | None = None,
    ) -> list[ReconsolidationUpdate]:
        """
        Record an outcome and trigger updates.

        Args:
            outcome_score: Outcome score (0-1)
            session_id: Optional session for credit assignment

        Returns:
            List of reconsolidation updates applied
        """
        updates = []

        # Apply reconsolidation to pending retrievals
        if self._reconsolidation and self._pending_updates:
            for pending in self._pending_updates:
                # P3-QUALITY-001 FIX: Use stored original embedding if available
                original_emb = pending.get("original_embedding")
                if original_emb is None:
                    # Fallback: use query embedding if original not provided
                    original_emb = pending["query_embedding"]

                update = ReconsolidationUpdate(
                    memory_id=pending["memory_id"],
                    query_embedding=pending["query_embedding"],
                    original_embedding=original_emb,
                    updated_embedding=original_emb,  # Actual update computed by reconsolidation
                    outcome_score=outcome_score,
                    advantage=outcome_score - 0.5,
                    learning_rate=0.01,
                )
                updates.append(update)
                self._state.update_count += 1

            self._pending_updates.clear()
            self._state.pending_reconsolidations = 0

        # Update serotonin long-term values
        if self._serotonin:
            self._serotonin.receive_outcome(
                outcome_score=outcome_score,
                context_id=session_id,
            )

        # Update dopamine based on surprise
        self._update_dopamine(outcome_score)

        return updates

    def _update_dopamine(self, outcome: float) -> None:
        """Update dopamine based on outcome surprise."""
        expected = self._config.da_baseline
        prediction_error = outcome - expected

        # Dopamine increases on positive surprise, decreases on negative
        new_da = self._state.neuromodulator.dopamine + 0.1 * prediction_error
        self._state.neuromodulator.dopamine = np.clip(new_da, 0.1, 0.9)

    def _transition_to_phase(self, new_phase: TemporalPhase) -> None:
        """Transition to a new phase."""
        old_phase = self._state.phase
        if old_phase == new_phase:
            return

        self._state.phase = new_phase
        self._state.phase_duration = timedelta(0)

        logger.info(f"Phase transition: {old_phase.value} -> {new_phase.value}")

        # Notify callbacks
        for callback in self._phase_callbacks:
            try:
                callback(new_phase)
            except Exception as e:
                logger.error(f"Phase callback error: {e}")

    def update(self) -> None:
        """
        Update temporal state.

        Should be called periodically to manage phase transitions.
        """
        now = datetime.now()
        idle_duration = (now - self._state.last_activity).total_seconds()

        # Phase transitions based on idle time
        if self._state.phase == TemporalPhase.ACTIVE:
            if idle_duration > self._config.idle_threshold_seconds:
                self._transition_to_phase(TemporalPhase.IDLE)

        elif self._state.phase == TemporalPhase.IDLE:
            if idle_duration > self._config.consolidation_threshold_seconds:
                self._transition_to_phase(TemporalPhase.CONSOLIDATING)

        # Decay norepinephrine over time
        decay = self._config.ne_decay_rate * (idle_duration / 60.0)
        self._state.neuromodulator.norepinephrine = max(
            0.3, self._state.neuromodulator.norepinephrine - decay
        )

        # Update homeostatic state
        if self._homeostatic:
            self._state.homeostatic_state = self._homeostatic.get_state()

        # Clean up expired traces
        if self._serotonin:
            self._serotonin.cleanup_expired_traces()
            self._state.active_traces = len(self._serotonin.get_memories_with_traces())

        # Update phase duration
        self._state.phase_duration = now - self._state.session_start

    async def start_background_updates(self) -> None:
        """Start background update loop."""
        if self._running:
            return

        self._running = True
        self._update_task = asyncio.create_task(self._background_loop())
        logger.info("Started temporal dynamics background updates")

    async def stop_background_updates(self) -> None:
        """Stop background update loop."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        logger.info("Stopped temporal dynamics background updates")

    async def _background_loop(self) -> None:
        """Background update loop."""
        while self._running:
            try:
                self.update()

                # Notify state callbacks
                for callback in self._state_callbacks:
                    try:
                        callback(self._state)
                    except Exception as e:
                        logger.error(f"State callback error: {e}")

                await asyncio.sleep(self._config.update_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background update error: {e}")

    def register_phase_callback(
        self,
        callback: Callable[[TemporalPhase], None],
    ) -> None:
        """Register callback for phase transitions."""
        self._phase_callbacks.append(callback)

    def register_state_callback(
        self,
        callback: Callable[[TemporalState], None],
    ) -> None:
        """Register callback for state updates."""
        self._state_callbacks.append(callback)

    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        return {
            "state": self._state.to_dict(),
            "active_sessions": len(self._sessions),
            "subsystems": {
                "reconsolidation_enabled": self._reconsolidation is not None,
                "homeostatic_enabled": self._homeostatic is not None,
                "serotonin_enabled": self._serotonin is not None,
            },
            "pending_updates": len(self._pending_updates),
        }


def create_temporal_dynamics(
    config: TemporalConfig | None = None,
    modulation_config: ModulationConfig | None = None,
) -> TemporalDynamics:
    """Factory function to create temporal dynamics coordinator."""
    return TemporalDynamics(
        config=config,
        modulation_config=modulation_config,
    )


__all__ = [
    "TemporalConfig",
    "TemporalDynamics",
    "TemporalPhase",
    "TemporalState",
    "create_temporal_dynamics",
]
