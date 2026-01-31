"""
Lability Window (Protein Synthesis Gate) for Memory Reconsolidation.

Phase 7: Implements biological reconsolidation constraints.

Biological Basis:
After retrieval, memories enter a labile (modifiable) state for ~4-6 hours,
during which they can be updated, strengthened, or destabilized. This window
is gated by protein synthesis - blocking protein synthesis during this window
prevents reconsolidation.

Key Research:
- Nader et al. (2000): Reconsolidation and protein synthesis
- Tronson & Taylor (2007): Molecular mechanisms of reconsolidation
- Dudai (2012): The restless engram

Implementation:
- Memories become labile upon retrieval
- Lability window = 6 hours by default
- During lability: memory can be reconsolidated (updated)
- After window closes: memory is stable until next retrieval
- Reconsolidation updates embeddings based on new context

Usage:
    from t4dm.consolidation.lability import LabilityManager, is_reconsolidation_eligible

    # Check if memory is eligible for reconsolidation
    if is_reconsolidation_eligible(last_retrieval_time):
        # Safe to update memory embedding/associations
        await memory.reconsolidate(new_context)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


# Biological constants (Nader et al. 2000, Dudai 2012)
DEFAULT_LABILITY_WINDOW_HOURS = 6.0
MIN_LABILITY_WINDOW_HOURS = 4.0
MAX_LABILITY_WINDOW_HOURS = 8.0

# Reconsolidation strength factors
RETRIEVAL_STRENGTH_DECAY = 0.1  # Per hour
EMOTIONAL_BOOST_FACTOR = 1.5  # High emotion extends window


class LabilityPhase(Enum):
    """Memory lability phases."""
    STABLE = "stable"       # Memory is consolidated and stable
    LABILE = "labile"       # Memory is in reconsolidation window
    DESTABILIZED = "destabilized"  # Memory was retrieved but not reconsolidated
    RECONSOLIDATING = "reconsolidating"  # Active reconsolidation in progress


@dataclass
class LabilityConfig:
    """Configuration for lability window behavior.

    Attributes:
        window_hours: Duration of lability window after retrieval
        min_retrieval_strength: Minimum retrieval strength to trigger lability
        emotional_modulation: Whether emotional valence affects window duration
        require_prediction_error: Only enter lability if prediction error > threshold
        prediction_error_threshold: Threshold for PE-gated reconsolidation
    """
    window_hours: float = DEFAULT_LABILITY_WINDOW_HOURS
    min_retrieval_strength: float = 0.3
    emotional_modulation: bool = True
    require_prediction_error: bool = True
    prediction_error_threshold: float = 0.1

    def __post_init__(self):
        """Validate biological constraints."""
        if not (MIN_LABILITY_WINDOW_HOURS <= self.window_hours <= MAX_LABILITY_WINDOW_HOURS):
            logger.warning(
                f"Lability window {self.window_hours}h outside biological range "
                f"[{MIN_LABILITY_WINDOW_HOURS}, {MAX_LABILITY_WINDOW_HOURS}]. "
                f"Clamping to valid range."
            )
            self.window_hours = np.clip(
                self.window_hours,
                MIN_LABILITY_WINDOW_HOURS,
                MAX_LABILITY_WINDOW_HOURS
            )


@dataclass
class LabilityState:
    """State of a memory's lability.

    Attributes:
        memory_id: Identifier of the memory
        last_retrieval: When memory was last retrieved
        phase: Current lability phase
        retrieval_strength: How strongly memory was activated
        emotional_valence: Emotional content of memory
        prediction_error: Prediction error during retrieval
        reconsolidation_count: How many times reconsolidated
    """
    memory_id: UUID
    last_retrieval: datetime | None = None
    phase: LabilityPhase = LabilityPhase.STABLE
    retrieval_strength: float = 0.0
    emotional_valence: float = 0.5
    prediction_error: float = 0.0
    reconsolidation_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "memory_id": str(self.memory_id),
            "last_retrieval": self.last_retrieval.isoformat() if self.last_retrieval else None,
            "phase": self.phase.value,
            "retrieval_strength": self.retrieval_strength,
            "emotional_valence": self.emotional_valence,
            "prediction_error": self.prediction_error,
            "reconsolidation_count": self.reconsolidation_count,
        }


class LabilityManager:
    """
    Manages memory lability states for reconsolidation.

    Tracks which memories are in their lability window and eligible
    for reconsolidation updates. Implements biological constraints
    from memory reconsolidation research.

    Example:
        ```python
        manager = LabilityManager()

        # On memory retrieval
        manager.on_retrieval(memory_id, strength=0.8, emotion=0.7)

        # Check if eligible for update
        if manager.is_labile(memory_id):
            # Update memory content/embedding
            manager.on_reconsolidation(memory_id, success=True)
        ```
    """

    def __init__(self, config: LabilityConfig | None = None):
        """
        Initialize lability manager.

        Args:
            config: Lability configuration
        """
        self.config = config or LabilityConfig()
        self._states: dict[UUID, LabilityState] = {}
        self._total_retrievals = 0
        self._total_reconsolidations = 0

        logger.info(
            f"LabilityManager initialized: window={self.config.window_hours}h"
        )

    def on_retrieval(
        self,
        memory_id: UUID,
        strength: float = 1.0,
        emotional_valence: float = 0.5,
        prediction_error: float = 0.0,
    ) -> LabilityState:
        """
        Record memory retrieval and enter lability window.

        Called when a memory is retrieved/recalled. Memory enters labile
        state if retrieval strength exceeds threshold.

        Args:
            memory_id: Identifier of retrieved memory
            strength: How strongly memory was activated (0-1)
            emotional_valence: Emotional content (0-1)
            prediction_error: Prediction error during retrieval

        Returns:
            Updated lability state
        """
        now = datetime.now()
        self._total_retrievals += 1

        # Check if retrieval strong enough to trigger lability
        if strength < self.config.min_retrieval_strength:
            logger.debug(
                f"Retrieval too weak for lability: {strength:.2f} < "
                f"{self.config.min_retrieval_strength}"
            )
            # Return stable state
            state = self._states.get(memory_id, LabilityState(memory_id=memory_id))
            state.phase = LabilityPhase.STABLE
            return state

        # Check prediction error threshold (if required)
        if self.config.require_prediction_error:
            if abs(prediction_error) < self.config.prediction_error_threshold:
                logger.debug(
                    f"Prediction error too low for lability: {prediction_error:.3f} < "
                    f"{self.config.prediction_error_threshold}"
                )
                # Retrieval without surprise doesn't destabilize memory
                state = self._states.get(memory_id, LabilityState(memory_id=memory_id))
                state.last_retrieval = now
                state.retrieval_strength = strength
                state.phase = LabilityPhase.STABLE
                self._states[memory_id] = state
                return state

        # Get or create state
        state = self._states.get(memory_id, LabilityState(memory_id=memory_id))

        # Update state
        state.last_retrieval = now
        state.retrieval_strength = strength
        state.emotional_valence = emotional_valence
        state.prediction_error = prediction_error
        state.phase = LabilityPhase.LABILE

        self._states[memory_id] = state

        logger.debug(
            f"Memory {memory_id} entered lability: strength={strength:.2f}, "
            f"PE={prediction_error:.3f}"
        )

        return state

    def is_labile(
        self,
        memory_id: UUID,
        now: datetime | None = None,
    ) -> bool:
        """
        Check if memory is currently in labile state.

        Memory is labile if:
        1. It was recently retrieved (within window)
        2. Retrieval strength exceeded threshold
        3. Prediction error exceeded threshold (if required)

        Args:
            memory_id: Memory to check
            now: Current time (defaults to datetime.now())

        Returns:
            True if memory is labile and can be updated
        """
        if memory_id not in self._states:
            return False

        state = self._states[memory_id]
        if state.last_retrieval is None:
            return False

        if now is None:
            now = datetime.now()

        # Calculate effective window (emotion extends it)
        window_hours = self.config.window_hours
        if self.config.emotional_modulation:
            # High emotion extends window, low emotion shortens it
            emotion_factor = 0.5 + state.emotional_valence  # 0.5 to 1.5
            window_hours *= emotion_factor

        # Check if within window
        elapsed = (now - state.last_retrieval).total_seconds() / 3600
        in_window = elapsed < window_hours

        # Update phase based on window status
        if in_window and state.phase != LabilityPhase.RECONSOLIDATING:
            state.phase = LabilityPhase.LABILE
        elif not in_window and state.phase == LabilityPhase.LABILE:
            # Window closed without reconsolidation
            state.phase = LabilityPhase.DESTABILIZED

        return in_window

    def get_window_remaining(
        self,
        memory_id: UUID,
        now: datetime | None = None,
    ) -> float:
        """
        Get remaining time in lability window.

        Args:
            memory_id: Memory to check
            now: Current time

        Returns:
            Hours remaining (0 if not labile)
        """
        if memory_id not in self._states:
            return 0.0

        state = self._states[memory_id]
        if state.last_retrieval is None:
            return 0.0

        if now is None:
            now = datetime.now()

        # Calculate effective window
        window_hours = self.config.window_hours
        if self.config.emotional_modulation:
            emotion_factor = 0.5 + state.emotional_valence
            window_hours *= emotion_factor

        elapsed = (now - state.last_retrieval).total_seconds() / 3600
        remaining = max(0.0, window_hours - elapsed)

        return remaining

    def on_reconsolidation(
        self,
        memory_id: UUID,
        success: bool = True,
    ) -> LabilityState | None:
        """
        Record that memory was reconsolidated.

        Called after successfully updating memory content/embedding
        during the lability window.

        Args:
            memory_id: Memory that was reconsolidated
            success: Whether reconsolidation succeeded

        Returns:
            Updated state, or None if not found
        """
        if memory_id not in self._states:
            logger.warning(f"Reconsolidation called for unknown memory: {memory_id}")
            return None

        state = self._states[memory_id]

        if success:
            state.phase = LabilityPhase.STABLE
            state.reconsolidation_count += 1
            self._total_reconsolidations += 1

            logger.debug(
                f"Memory {memory_id} reconsolidated (count={state.reconsolidation_count})"
            )
        else:
            # Failed reconsolidation leaves memory destabilized
            state.phase = LabilityPhase.DESTABILIZED

        return state

    def get_labile_memories(
        self,
        now: datetime | None = None,
    ) -> list[UUID]:
        """
        Get all currently labile memories.

        Returns:
            List of memory IDs in labile state
        """
        if now is None:
            now = datetime.now()

        return [
            memory_id for memory_id in self._states
            if self.is_labile(memory_id, now)
        ]

    def get_state(self, memory_id: UUID) -> LabilityState | None:
        """Get lability state for a memory."""
        return self._states.get(memory_id)

    def get_stats(self) -> dict:
        """Get lability manager statistics."""
        now = datetime.now()
        labile_count = len(self.get_labile_memories(now))

        phase_counts = {phase.value: 0 for phase in LabilityPhase}
        for state in self._states.values():
            phase_counts[state.phase.value] += 1

        return {
            "tracked_memories": len(self._states),
            "currently_labile": labile_count,
            "total_retrievals": self._total_retrievals,
            "total_reconsolidations": self._total_reconsolidations,
            "phase_distribution": phase_counts,
            "config": {
                "window_hours": self.config.window_hours,
                "emotional_modulation": self.config.emotional_modulation,
                "require_prediction_error": self.config.require_prediction_error,
            },
        }

    def clear(self) -> None:
        """Clear all tracked states."""
        self._states.clear()
        logger.info("LabilityManager state cleared")


# ============================================================================
# Convenience Functions
# ============================================================================


def is_reconsolidation_eligible(
    last_retrieval_time: datetime,
    window_hours: float = DEFAULT_LABILITY_WINDOW_HOURS,
    now: datetime | None = None,
) -> bool:
    """
    Check if a memory is eligible for reconsolidation based on time since retrieval.

    Simple stateless check for reconsolidation eligibility.

    Args:
        last_retrieval_time: When memory was last retrieved
        window_hours: Lability window duration
        now: Current time (defaults to datetime.now())

    Returns:
        True if within reconsolidation window

    Example:
        ```python
        if is_reconsolidation_eligible(episode.last_accessed):
            # Safe to update episode
            episode.embedding = new_embedding
        ```
    """
    if now is None:
        now = datetime.now()

    elapsed = (now - last_retrieval_time).total_seconds() / 3600
    return elapsed < window_hours


def compute_reconsolidation_strength(
    retrieval_strength: float,
    emotional_valence: float,
    prediction_error: float,
    hours_elapsed: float,
) -> float:
    """
    Compute how strongly reconsolidation should update the memory.

    Higher values mean more substantial updates are appropriate.

    Args:
        retrieval_strength: How strongly memory was activated
        emotional_valence: Emotional content (0-1)
        prediction_error: Prediction error during retrieval
        hours_elapsed: Hours since retrieval

    Returns:
        Reconsolidation strength (0-1)
    """
    # Base strength from retrieval
    base = retrieval_strength

    # Emotion boost (high emotion = stronger update)
    emotion_factor = 0.7 + 0.6 * emotional_valence  # 0.7 to 1.3

    # Prediction error boost (surprise = need to update)
    pe_factor = 1.0 + abs(prediction_error)  # 1.0 to 2.0

    # Time decay (stronger updates early in window)
    decay = max(0.1, 1.0 - hours_elapsed * RETRIEVAL_STRENGTH_DECAY)

    strength = base * emotion_factor * pe_factor * decay

    return float(np.clip(strength, 0.0, 1.0))


def get_reconsolidation_learning_rate(
    base_lr: float,
    reconsolidation_strength: float,
    reconsolidation_count: int,
) -> float:
    """
    Get learning rate for reconsolidation updates.

    Learning rate decreases with repeated reconsolidations to prevent
    runaway updates.

    Args:
        base_lr: Base learning rate
        reconsolidation_strength: Strength of reconsolidation
        reconsolidation_count: How many times memory has been reconsolidated

    Returns:
        Adjusted learning rate
    """
    # Strength multiplier
    strength_mult = reconsolidation_strength

    # Decay with repeated reconsolidations (diminishing returns)
    count_decay = 1.0 / (1.0 + 0.2 * reconsolidation_count)

    return base_lr * strength_mult * count_decay


# ============================================================================
# Singleton
# ============================================================================

_lability_manager: LabilityManager | None = None


def get_lability_manager() -> LabilityManager:
    """Get or create singleton lability manager."""
    global _lability_manager
    if _lability_manager is None:
        _lability_manager = LabilityManager()
    return _lability_manager


def reset_lability_manager() -> None:
    """Reset singleton (for testing)."""
    global _lability_manager
    _lability_manager = None


__all__ = [
    "LabilityManager",
    "LabilityConfig",
    "LabilityState",
    "LabilityPhase",
    "is_reconsolidation_eligible",
    "compute_reconsolidation_strength",
    "get_reconsolidation_learning_rate",
    "get_lability_manager",
    "reset_lability_manager",
    "DEFAULT_LABILITY_WINDOW_HOURS",
    "MIN_LABILITY_WINDOW_HOURS",
    "MAX_LABILITY_WINDOW_HOURS",
]
