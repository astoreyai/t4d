"""
Cognitive State Attractors for KATIE-style Dynamics.

Biological Basis:
- Brain states exist as attractor basins in NT space
- Transitions between states (e.g., alert -> focus -> rest)
- Metastability allows flexible state switching

Attractor States:
- ALERT: High DA, NE; Low GABA (ready for action)
- FOCUS: High ACh, Glu; Moderate DA (sustained attention)
- REST: High 5-HT, GABA; Low NE (default mode)
- EXPLORE: High DA, NE, ACh (novelty seeking)
- CONSOLIDATE: High GABA, 5-HT; Low Glu (memory consolidation)

Integration:
- Attractor basins modulate memory operations
- State transitions trigger learning signals
- Implemented by Hinton agent using energy-based framework
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class CognitiveState(Enum):
    """Discrete cognitive states as attractor basins."""
    ALERT = auto()        # High DA, NE; Low GABA
    FOCUS = auto()        # High ACh, Glu; Moderate DA
    REST = auto()         # High 5-HT, GABA; Low NE
    EXPLORE = auto()      # High DA, NE, ACh
    CONSOLIDATE = auto()  # High GABA, 5-HT; Low Glu


@dataclass
class AttractorBasin:
    """
    Definition of a cognitive state attractor.

    Each attractor is defined by:
    - Target NT concentrations (basin center)
    - Basin width (how far states can deviate)
    - Transition thresholds to other states
    """
    state: CognitiveState
    center: np.ndarray  # [DA, 5HT, ACh, NE, GABA, Glu]
    width: float = 0.2  # Basin radius in NT space
    stability: float = 0.5  # How strongly state is maintained

    @classmethod
    def get_default_attractors(cls) -> dict[CognitiveState, AttractorBasin]:
        """Get biologically-plausible default attractors."""
        return {
            CognitiveState.ALERT: cls(
                state=CognitiveState.ALERT,
                center=np.array([0.7, 0.4, 0.5, 0.8, 0.3, 0.5]),
                width=0.2,
                stability=0.4,  # Easily transitions
            ),
            CognitiveState.FOCUS: cls(
                state=CognitiveState.FOCUS,
                center=np.array([0.5, 0.5, 0.8, 0.5, 0.4, 0.7]),
                width=0.15,
                stability=0.6,  # More stable
            ),
            CognitiveState.REST: cls(
                state=CognitiveState.REST,
                center=np.array([0.3, 0.7, 0.4, 0.2, 0.7, 0.3]),
                width=0.25,
                stability=0.5,
            ),
            CognitiveState.EXPLORE: cls(
                state=CognitiveState.EXPLORE,
                center=np.array([0.8, 0.4, 0.7, 0.7, 0.3, 0.6]),
                width=0.2,
                stability=0.3,  # Unstable, leads to transitions
            ),
            CognitiveState.CONSOLIDATE: cls(
                state=CognitiveState.CONSOLIDATE,
                center=np.array([0.3, 0.6, 0.3, 0.3, 0.8, 0.2]),
                width=0.2,
                stability=0.7,  # Very stable during sleep
            ),
        }


@dataclass
class StateTransition:
    """Record of a cognitive state transition."""
    from_state: CognitiveState
    to_state: CognitiveState
    timestamp: datetime
    nt_state_before: np.ndarray
    nt_state_after: np.ndarray
    trigger: str = "spontaneous"  # What caused transition


class StateTransitionManager:
    """
    Manage cognitive state transitions and attractor dynamics.

    Uses energy landscape from energy.py to determine:
    - Current attractor basin
    - Transition probabilities
    - State stability

    Key Methods:
    - classify_state: Map NT state to nearest attractor basin
    - update: Process state transitions with hysteresis
    - get_stats: Return transition statistics
    """

    def __init__(
        self,
        attractors: dict[CognitiveState, AttractorBasin] | None = None,
        transition_threshold: float = 0.3,
        hysteresis: float = 0.1
    ):
        """
        Initialize state transition manager.

        Args:
            attractors: Attractor basin definitions
            transition_threshold: Distance threshold for state change
            hysteresis: Prevents rapid oscillation between states
        """
        self.attractors = attractors or AttractorBasin.get_default_attractors()
        self.transition_threshold = transition_threshold
        self.hysteresis = hysteresis

        # Current state
        self._current_state = CognitiveState.REST
        self._state_duration = 0.0
        self._last_transition: datetime | None = None

        # Transition history
        self._transitions: list[StateTransition] = []
        self._max_history = 1000

        logger.info(
            f"StateTransitionManager initialized with "
            f"{len(self.attractors)} attractors"
        )

    def get_current_state(self) -> CognitiveState:
        """Get current cognitive state."""
        return self._current_state

    def classify_state(
        self,
        nt_state: NeurotransmitterState
    ) -> tuple[CognitiveState, float]:
        """
        Classify NT state into nearest attractor basin.

        Args:
            nt_state: Current NT concentrations

        Returns:
            (nearest_state, distance_to_center)
        """
        from ww.nca.neural_field import NeurotransmitterState

        if isinstance(nt_state, NeurotransmitterState):
            U = nt_state.to_array()
        else:
            U = np.asarray(nt_state)

        best_state = self._current_state
        best_dist = float('inf')

        for state, attractor in self.attractors.items():
            dist = np.linalg.norm(U - attractor.center)

            # Apply hysteresis: current state gets bonus
            if state == self._current_state:
                dist -= self.hysteresis

            if dist < best_dist:
                best_dist = dist
                best_state = state

        return best_state, best_dist

    def update(
        self,
        nt_state: NeurotransmitterState,
        dt: float = 0.01
    ) -> StateTransition | None:
        """
        Update state based on current NT concentrations.

        Args:
            nt_state: Current NT state
            dt: Time step

        Returns:
            StateTransition if state changed, None otherwise
        """
        from ww.nca.neural_field import NeurotransmitterState

        if isinstance(nt_state, NeurotransmitterState):
            U = nt_state.to_array()
        else:
            U = np.asarray(nt_state)

        new_state, distance = self.classify_state(nt_state)

        self._state_duration += dt

        # Check for state transition
        if new_state != self._current_state:
            current_attractor = self.attractors[self._current_state]

            # Compute distance from current attractor center
            dist_from_current = np.linalg.norm(U - current_attractor.center)

            # Only transition if outside current basin
            if dist_from_current > current_attractor.width + self.hysteresis:
                transition = StateTransition(
                    from_state=self._current_state,
                    to_state=new_state,
                    timestamp=datetime.now(),
                    nt_state_before=U.copy(),
                    nt_state_after=U.copy(),
                    trigger="nt_dynamics"
                )

                self._transitions.append(transition)
                if len(self._transitions) > self._max_history:
                    self._transitions = self._transitions[-self._max_history:]

                old_state = self._current_state
                self._current_state = new_state
                self._state_duration = 0.0
                self._last_transition = datetime.now()

                logger.debug(
                    f"State transition: {old_state.name} -> {new_state.name}"
                )

                return transition

        return None

    def force_state(
        self,
        state: CognitiveState,
        reason: str = "forced"
    ) -> StateTransition:
        """
        Force transition to specific state.

        Args:
            state: Target state
            reason: Why transition was forced

        Returns:
            StateTransition record
        """
        if state == self._current_state:
            return None

        attractor = self.attractors[state]
        transition = StateTransition(
            from_state=self._current_state,
            to_state=state,
            timestamp=datetime.now(),
            nt_state_before=self.attractors[self._current_state].center,
            nt_state_after=attractor.center,
            trigger=reason
        )

        self._transitions.append(transition)
        self._current_state = state
        self._state_duration = 0.0
        self._last_transition = datetime.now()

        logger.info(f"Forced state transition to {state.name}: {reason}")

        return transition

    def get_attractor_force(
        self,
        nt_state: NeurotransmitterState
    ) -> np.ndarray:
        """
        Compute force pulling NT state toward current attractor.

        Used by neural_field.py to add attractor dynamics.

        Args:
            nt_state: Current NT state

        Returns:
            Force vector toward attractor center [6,]
        """
        from ww.nca.neural_field import NeurotransmitterState

        if isinstance(nt_state, NeurotransmitterState):
            U = nt_state.to_array()
        else:
            U = np.asarray(nt_state)

        attractor = self.attractors[self._current_state]

        # Force toward attractor center, scaled by stability
        direction = attractor.center - U
        force = attractor.stability * direction

        return force

    def get_transition_probability(
        self,
        from_state: CognitiveState,
        to_state: CognitiveState
    ) -> float:
        """
        Get probability of state transition based on history.

        Args:
            from_state: Origin state
            to_state: Target state

        Returns:
            Estimated transition probability
        """
        if not self._transitions:
            return 0.1  # Default uniform

        matching = sum(
            1 for t in self._transitions
            if t.from_state == from_state and t.to_state == to_state
        )
        from_total = sum(
            1 for t in self._transitions
            if t.from_state == from_state
        )

        if from_total == 0:
            return 0.1

        return matching / from_total

    def get_stats(self) -> dict:
        """Get state management statistics."""
        return {
            "current_state": self._current_state.name,
            "state_duration": self._state_duration,
            "total_transitions": len(self._transitions),
            "last_transition": (
                self._last_transition.isoformat()
                if self._last_transition else None
            ),
            "transition_counts": {
                state.name: sum(
                    1 for t in self._transitions
                    if t.to_state == state
                )
                for state in CognitiveState
            }
        }


__all__ = [
    "CognitiveState",
    "AttractorBasin",
    "StateTransitionManager",
    "StateTransition",
]
