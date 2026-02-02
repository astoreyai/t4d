"""
P7.2: Memory Lifecycle Manager for T4DM.

Connects temporal dynamics to memory operations, ensuring that:
1. Encoding uses appropriate neuromodulator states
2. Retrieval uses appropriate neuromodulator states
3. Consolidation uses sleep/replay states
4. Outcomes trigger reconsolidation with state consistency

This creates a coherent lifecycle where memory operations are
state-aware and learning is modulated by the system's current mode.

Biological Basis:
- Acetylcholine gates encoding vs retrieval mode (Hasselmo 1999)
- Dopamine signals importance for consolidation (Lisman & Grace 2005)
- Sleep states enable memory transfer (Diekelmann & Born 2010)

Integration Points:
- ww.memory.episodic: EpisodicMemory.create/recall
- ww.memory.semantic: SemanticMemory.recall
- ww.consolidation.service: ConsolidationService
- ww.temporal.integration: PlasticityCoordinator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from t4dm.core.bridge_container import get_bridge_container
from t4dm.temporal.integration import (
    PlasticityConfig,
    PlasticityCoordinator,
    adapt_orchestra_state,
    get_consolidation_state,
    get_pattern_separation_state,
    get_sleep_replay_state,
)

if TYPE_CHECKING:
    from t4dm.embedding.modulated import NeuromodulatorState
    from t4dm.learning.neuromodulators import NeuromodulatorOrchestra

logger = logging.getLogger(__name__)


class LifecyclePhase(Enum):
    """Memory lifecycle phases."""
    ENCODING = "encoding"
    RETRIEVAL = "retrieval"
    CONSOLIDATION = "consolidation"
    SLEEP_REPLAY = "sleep_replay"
    PATTERN_SEPARATION = "pattern_separation"
    IDLE = "idle"


@dataclass
class LifecycleState:
    """Current state of the memory lifecycle.

    Attributes:
        phase: Current lifecycle phase
        phase_start: When current phase started
        encoding_count: Number of encodings in current phase
        retrieval_count: Number of retrievals in current phase
        consolidation_count: Number of consolidation cycles
        last_outcome_time: When last outcome was processed
        session_id: Current session ID
    """
    phase: LifecyclePhase = LifecyclePhase.IDLE
    phase_start: datetime = field(default_factory=datetime.now)
    encoding_count: int = 0
    retrieval_count: int = 0
    consolidation_count: int = 0
    last_outcome_time: datetime | None = None
    session_id: str = "default"


@dataclass
class LifecycleConfig:
    """Configuration for lifecycle manager.

    Attributes:
        auto_phase_transition: Automatically transition phases based on activity
        encoding_to_retrieval_ratio: Ratio triggering mode switch (encoding/retrieval)
        consolidation_idle_seconds: Idle time before suggesting consolidation
        enable_plasticity: Enable plasticity coordination
        enable_temporal_dynamics: Enable NCA temporal dynamics
    """
    auto_phase_transition: bool = True
    encoding_to_retrieval_ratio: float = 2.0
    consolidation_idle_seconds: float = 300.0
    enable_plasticity: bool = True
    enable_temporal_dynamics: bool = True


class MemoryLifecycleManager:
    """
    Coordinates temporal dynamics with memory operations.

    Implements P7.2: Connect temporal module to memory lifecycle.

    The manager ensures that:
    - Encoding operations use high-ACh states (DG/CA3 mode)
    - Retrieval operations use low-ACh states (CA1 output mode)
    - Consolidation uses specialized sleep states
    - Outcomes trigger appropriate reconsolidation

    Usage:
        ```python
        from t4dm.temporal.lifecycle import get_lifecycle_manager

        manager = get_lifecycle_manager()

        # Signal encoding start
        state = manager.start_encoding()
        embedding = modulated_adapter.encode(content, state)

        # Signal retrieval
        state = manager.start_retrieval()
        results = await memory.recall(query)

        # Process outcome
        await manager.process_outcome(results, outcome_score=0.8)
        ```
    """

    def __init__(
        self,
        config: LifecycleConfig | None = None,
        session_id: str = "default",
    ):
        """
        Initialize lifecycle manager.

        Args:
            config: Lifecycle configuration
            session_id: Session identifier
        """
        self.config = config or LifecycleConfig()
        self.state = LifecycleState(session_id=session_id)

        # Bridge container for NCA integration
        self._bridge_container = get_bridge_container(session_id)

        # Plasticity coordinator
        self._plasticity = PlasticityCoordinator(
            config=PlasticityConfig()
        ) if self.config.enable_plasticity else None

        # Orchestra reference (set externally)
        self._orchestra: NeuromodulatorOrchestra | None = None

        # State history for analysis
        self._phase_history: list[tuple[datetime, LifecyclePhase]] = []
        self._max_history = 1000

        logger.info(
            f"P7.2: MemoryLifecycleManager initialized "
            f"(session={session_id}, plasticity={self.config.enable_plasticity})"
        )

    def set_orchestra(self, orchestra: NeuromodulatorOrchestra) -> None:
        """Set neuromodulator orchestra for state adaptation."""
        self._orchestra = orchestra
        logger.debug("P7.2: Orchestra connected to lifecycle manager")

    def _transition_phase(self, new_phase: LifecyclePhase) -> None:
        """Transition to a new lifecycle phase."""
        old_phase = self.state.phase

        if old_phase != new_phase:
            self.state.phase = new_phase
            self.state.phase_start = datetime.now()

            # Reset counters for new phase
            self.state.encoding_count = 0
            self.state.retrieval_count = 0

            # Track history
            self._phase_history.append((datetime.now(), new_phase))
            if len(self._phase_history) > self._max_history:
                self._phase_history = self._phase_history[-self._max_history:]

            logger.info(f"P7.2: Phase transition {old_phase.value} -> {new_phase.value}")

    def _get_current_neuromod_state(self) -> NeuromodulatorState:
        """Get current neuromodulator state from orchestra or default."""
        if self._orchestra is not None:
            try:
                orchestra_state = self._orchestra.get_current_state()
                return adapt_orchestra_state(orchestra_state)
            except Exception as e:
                logger.debug(f"Orchestra state unavailable: {e}")

        # Fallback to phase-appropriate defaults
        from t4dm.embedding.modulated import NeuromodulatorState

        if self.state.phase == LifecyclePhase.ENCODING:
            return NeuromodulatorState.for_encoding()
        elif self.state.phase == LifecyclePhase.RETRIEVAL:
            return NeuromodulatorState.for_retrieval()
        elif self.state.phase == LifecyclePhase.CONSOLIDATION:
            return get_consolidation_state()
        elif self.state.phase == LifecyclePhase.SLEEP_REPLAY:
            return get_sleep_replay_state()
        elif self.state.phase == LifecyclePhase.PATTERN_SEPARATION:
            return get_pattern_separation_state()
        else:
            return NeuromodulatorState.balanced()

    def start_encoding(self) -> NeuromodulatorState:
        """
        Signal start of encoding operation.

        Returns appropriate neuromodulator state for encoding:
        - High ACh for input processing
        - Moderate NE for attention

        Returns:
            NeuromodulatorState for encoding
        """
        self._transition_phase(LifecyclePhase.ENCODING)
        self.state.encoding_count += 1

        # Update NCA bridge if available
        nca_bridge = self._bridge_container.get_nca_bridge()
        if nca_bridge is not None:
            try:
                nca_bridge.step(dt=0.01)  # Small step to update state
            except Exception as e:
                logger.debug(f"NCA bridge step failed: {e}")

        state = self._get_current_neuromod_state()
        logger.debug(f"P7.2: Encoding started (ACh={state.acetylcholine:.2f})")

        return state

    def start_retrieval(self) -> NeuromodulatorState:
        """
        Signal start of retrieval operation.

        Returns appropriate neuromodulator state for retrieval:
        - Low ACh for output mode
        - High DA for pattern matching

        Returns:
            NeuromodulatorState for retrieval
        """
        self._transition_phase(LifecyclePhase.RETRIEVAL)
        self.state.retrieval_count += 1

        # Update NCA bridge
        nca_bridge = self._bridge_container.get_nca_bridge()
        if nca_bridge is not None:
            try:
                nca_bridge.step(dt=0.01)
            except Exception as e:
                logger.debug(f"NCA bridge step failed: {e}")

        state = self._get_current_neuromod_state()
        logger.debug(f"P7.2: Retrieval started (ACh={state.acetylcholine:.2f})")

        return state

    def start_consolidation(self) -> NeuromodulatorState:
        """
        Signal start of consolidation.

        Returns appropriate neuromodulator state for consolidation:
        - Very low ACh (sleep-like)
        - High DA for importance weighting
        - High 5-HT for long-term integration

        Returns:
            NeuromodulatorState for consolidation
        """
        self._transition_phase(LifecyclePhase.CONSOLIDATION)
        self.state.consolidation_count += 1

        state = get_consolidation_state()
        logger.debug(f"P7.2: Consolidation started (DA={state.dopamine:.2f})")

        return state

    def start_sleep_replay(self) -> NeuromodulatorState:
        """
        Signal start of sleep replay (SWR-like).

        Returns appropriate neuromodulator state for replay:
        - Minimal ACh (hippocampal output mode)
        - Moderate DA for reactivation

        Returns:
            NeuromodulatorState for sleep replay
        """
        self._transition_phase(LifecyclePhase.SLEEP_REPLAY)

        state = get_sleep_replay_state()
        logger.debug(f"P7.2: Sleep replay started (ACh={state.acetylcholine:.2f})")

        return state

    async def process_outcome(
        self,
        memory_ids: list[str | UUID],
        outcome_score: float,
        query_embedding: np.ndarray | None = None,
        retrieved_embeddings: list[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """
        Process outcome and trigger reconsolidation.

        Args:
            memory_ids: IDs of retrieved memories
            outcome_score: Outcome score (0-1)
            query_embedding: Query embedding that triggered retrieval
            retrieved_embeddings: Embeddings of retrieved memories

        Returns:
            Dict with reconsolidation statistics
        """
        self.state.last_outcome_time = datetime.now()

        result = {
            "memory_ids": [str(m) for m in memory_ids],
            "outcome_score": outcome_score,
            "updates_applied": 0,
            "phase": self.state.phase.value,
        }

        if not self.config.enable_plasticity or self._plasticity is None:
            return result

        # Require embeddings for reconsolidation
        if query_embedding is None or not retrieved_embeddings:
            return result

        try:
            # Get current state for consistent updates
            current_state = self._get_current_neuromod_state()

            # Process through plasticity coordinator
            updates = await self._plasticity.process_outcome(
                outcome_score=outcome_score,
                retrieved_embeddings=retrieved_embeddings,
                query_embedding=query_embedding,
                memory_ids=[str(m) for m in memory_ids],
                current_state=current_state,
            )

            result["updates_applied"] = len(updates)
            logger.debug(
                f"P7.2: Outcome processed - score={outcome_score:.2f}, "
                f"updates={len(updates)}"
            )

        except Exception as e:
            logger.warning(f"Outcome processing failed: {e}")
            result["error"] = str(e)

        return result

    def suggest_phase(self) -> LifecyclePhase:
        """
        Suggest appropriate lifecycle phase based on activity.

        Uses heuristics to suggest transitions:
        - Many encodings without retrieval → suggest retrieval
        - Long idle → suggest consolidation
        - High encoding/retrieval ratio → suggest pattern separation

        Returns:
            Suggested lifecycle phase
        """
        now = datetime.now()
        idle_seconds = (now - self.state.phase_start).total_seconds()

        # Long idle suggests consolidation
        if idle_seconds > self.config.consolidation_idle_seconds:
            return LifecyclePhase.CONSOLIDATION

        # High encoding suggests pattern separation might help
        if (self.state.encoding_count > 10 and
            self.state.retrieval_count < 3):
            return LifecyclePhase.PATTERN_SEPARATION

        # Balance encoding and retrieval
        if self.state.encoding_count > 5 and self.state.retrieval_count == 0:
            return LifecyclePhase.RETRIEVAL

        return self.state.phase

    def get_statistics(self) -> dict[str, Any]:
        """Get lifecycle manager statistics."""
        return {
            "current_phase": self.state.phase.value,
            "phase_duration_seconds": (
                datetime.now() - self.state.phase_start
            ).total_seconds(),
            "encoding_count": self.state.encoding_count,
            "retrieval_count": self.state.retrieval_count,
            "consolidation_count": self.state.consolidation_count,
            "last_outcome_time": (
                self.state.last_outcome_time.isoformat()
                if self.state.last_outcome_time else None
            ),
            "phase_history_length": len(self._phase_history),
            "config": {
                "auto_phase_transition": self.config.auto_phase_transition,
                "enable_plasticity": self.config.enable_plasticity,
                "enable_temporal_dynamics": self.config.enable_temporal_dynamics,
            },
        }


# -----------------------------------------------------------------------------
# Singleton Pattern for Session-Scoped Managers
# -----------------------------------------------------------------------------

_managers: dict[str, MemoryLifecycleManager] = {}


def get_lifecycle_manager(
    session_id: str = "default",
    config: LifecycleConfig | None = None,
) -> MemoryLifecycleManager:
    """
    Get or create lifecycle manager for session.

    Args:
        session_id: Session identifier
        config: Optional configuration (only used on first creation)

    Returns:
        MemoryLifecycleManager for the session
    """
    if session_id not in _managers:
        _managers[session_id] = MemoryLifecycleManager(
            config=config,
            session_id=session_id,
        )
    return _managers[session_id]


def clear_lifecycle_managers() -> None:
    """Clear all lifecycle managers (for testing)."""
    _managers.clear()
    logger.info("P7.2: All lifecycle managers cleared")


__all__ = [
    "LifecyclePhase",
    "LifecycleState",
    "LifecycleConfig",
    "MemoryLifecycleManager",
    "get_lifecycle_manager",
    "clear_lifecycle_managers",
]
