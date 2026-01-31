"""Tests for NCA cognitive state attractors."""

import numpy as np
import pytest

from t4dm.nca.attractors import (
    AttractorBasin,
    CognitiveState,
    StateTransitionManager,
)
from t4dm.nca.neural_field import NeurotransmitterState


class TestCognitiveState:
    """Tests for CognitiveState enum."""

    def test_all_states_defined(self):
        """All expected cognitive states should exist."""
        assert CognitiveState.ALERT
        assert CognitiveState.FOCUS
        assert CognitiveState.REST
        assert CognitiveState.EXPLORE
        assert CognitiveState.CONSOLIDATE

    def test_state_count(self):
        """Should have exactly 5 cognitive states."""
        assert len(CognitiveState) == 5


class TestAttractorBasin:
    """Tests for AttractorBasin dataclass."""

    def test_default_attractors(self):
        """Default attractors should cover all states."""
        attractors = AttractorBasin.get_default_attractors()

        assert len(attractors) == 5
        for state in CognitiveState:
            assert state in attractors

    def test_attractor_centers_valid(self):
        """Attractor centers should be in [0, 1] for all NTs."""
        attractors = AttractorBasin.get_default_attractors()

        for basin in attractors.values():
            assert basin.center.shape == (6,)
            assert np.all(basin.center >= 0)
            assert np.all(basin.center <= 1)

    def test_alert_state_characteristics(self):
        """ALERT state should have high DA, NE and low GABA."""
        attractors = AttractorBasin.get_default_attractors()
        alert = attractors[CognitiveState.ALERT]

        # DA (index 0) should be high
        assert alert.center[0] > 0.5
        # NE (index 3) should be high
        assert alert.center[3] > 0.5
        # GABA (index 4) should be low
        assert alert.center[4] < 0.5

    def test_focus_state_characteristics(self):
        """FOCUS state should have high ACh, Glu."""
        attractors = AttractorBasin.get_default_attractors()
        focus = attractors[CognitiveState.FOCUS]

        # ACh (index 2) should be high
        assert focus.center[2] > 0.5
        # Glu (index 5) should be high
        assert focus.center[5] > 0.5

    def test_rest_state_characteristics(self):
        """REST state should have high 5-HT, GABA and low NE."""
        attractors = AttractorBasin.get_default_attractors()
        rest = attractors[CognitiveState.REST]

        # 5-HT (index 1) should be high
        assert rest.center[1] > 0.5
        # GABA (index 4) should be high
        assert rest.center[4] > 0.5
        # NE (index 3) should be low
        assert rest.center[3] < 0.5

    def test_consolidate_stability(self):
        """CONSOLIDATE state should be most stable."""
        attractors = AttractorBasin.get_default_attractors()

        consolidate = attractors[CognitiveState.CONSOLIDATE]
        explore = attractors[CognitiveState.EXPLORE]

        # Consolidate should be more stable than explore
        assert consolidate.stability > explore.stability


class TestStateTransitionManager:
    """Tests for StateTransitionManager."""

    def test_initialization(self):
        """Manager should initialize with default state."""
        manager = StateTransitionManager()
        assert manager.get_current_state() == CognitiveState.REST

    def test_classify_state_rest(self):
        """State at REST center should classify as REST."""
        manager = StateTransitionManager()
        rest_center = manager.attractors[CognitiveState.REST].center
        state = NeurotransmitterState.from_array(rest_center)

        classified, distance = manager.classify_state(state)
        assert classified == CognitiveState.REST
        assert distance < 0.1

    def test_classify_state_alert(self):
        """State at ALERT center should classify as ALERT."""
        manager = StateTransitionManager()
        alert_center = manager.attractors[CognitiveState.ALERT].center
        state = NeurotransmitterState.from_array(alert_center)

        # First force to REST to avoid hysteresis
        manager.force_state(CognitiveState.REST)

        classified, distance = manager.classify_state(state)
        assert classified == CognitiveState.ALERT

    def test_update_no_transition(self):
        """Small perturbations should not cause transition."""
        manager = StateTransitionManager()

        # Small perturbation from REST
        rest_center = manager.attractors[CognitiveState.REST].center
        perturbed = rest_center + np.random.randn(6) * 0.05
        perturbed = np.clip(perturbed, 0, 1)
        state = NeurotransmitterState.from_array(perturbed)

        transition = manager.update(state, dt=0.01)
        assert transition is None
        assert manager.get_current_state() == CognitiveState.REST

    def test_update_with_transition(self):
        """Large move to different attractor should trigger transition."""
        manager = StateTransitionManager()

        # Move to ALERT center
        alert_center = manager.attractors[CognitiveState.ALERT].center
        state = NeurotransmitterState.from_array(alert_center)

        # May need multiple updates due to hysteresis
        transition = None
        for _ in range(10):
            transition = manager.update(state, dt=0.1)
            if transition is not None:
                break

        assert transition is not None
        assert transition.to_state == CognitiveState.ALERT

    def test_force_state(self):
        """force_state should immediately change state."""
        manager = StateTransitionManager()

        transition = manager.force_state(CognitiveState.EXPLORE, "test")

        assert manager.get_current_state() == CognitiveState.EXPLORE
        assert transition.to_state == CognitiveState.EXPLORE
        assert transition.trigger == "test"

    def test_force_state_same(self):
        """Forcing to current state should return None."""
        manager = StateTransitionManager()

        result = manager.force_state(CognitiveState.REST)
        assert result is None

    def test_attractor_force(self):
        """get_attractor_force should pull toward current attractor."""
        manager = StateTransitionManager()

        # State away from REST center
        state = NeurotransmitterState(
            dopamine=0.8,  # High DA (not REST-like)
            norepinephrine=0.8  # High NE (not REST-like)
        )

        force = manager.get_attractor_force(state)
        assert force.shape == (6,)

        # Force should be non-zero
        assert np.linalg.norm(force) > 0

    def test_transition_probability(self):
        """Transition probability should be based on history."""
        manager = StateTransitionManager()

        # No history yet
        prob = manager.get_transition_probability(
            CognitiveState.REST,
            CognitiveState.ALERT
        )
        assert prob == 0.1  # Default

        # Add some transitions
        manager.force_state(CognitiveState.ALERT)
        manager.force_state(CognitiveState.REST)
        manager.force_state(CognitiveState.ALERT)

        prob = manager.get_transition_probability(
            CognitiveState.REST,
            CognitiveState.ALERT
        )
        assert prob > 0  # Should have some history now

    def test_stats(self):
        """Stats should include expected keys."""
        manager = StateTransitionManager()
        stats = manager.get_stats()

        assert "current_state" in stats
        assert "state_duration" in stats
        assert "total_transitions" in stats
        assert "transition_counts" in stats


class TestStateTransitionDynamics:
    """Integration tests for state transition dynamics."""

    def test_hysteresis(self):
        """Hysteresis should prevent rapid oscillation at basin boundaries."""
        manager = StateTransitionManager(hysteresis=0.2)

        # Create a state at the boundary between REST and ALERT
        rest_center = manager.attractors[CognitiveState.REST].center
        alert_center = manager.attractors[CognitiveState.ALERT].center

        # Midpoint is at the boundary
        midpoint = (rest_center + alert_center) / 2

        # Small oscillations around the midpoint should be dampened by hysteresis
        transitions = 0
        for i in range(100):
            # Small perturbation around midpoint
            noise = np.random.randn(6) * 0.05
            perturbed = np.clip(midpoint + noise, 0, 1)
            state = NeurotransmitterState.from_array(perturbed)

            if manager.update(state, dt=0.01):
                transitions += 1

        # With hysteresis, small oscillations around boundary shouldn't cause
        # frequent transitions - should stay in current state
        assert transitions < 20

    def test_state_duration_tracking(self):
        """State duration should accumulate during updates."""
        manager = StateTransitionManager()

        # Stay in REST
        rest_state = NeurotransmitterState.from_array(
            manager.attractors[CognitiveState.REST].center
        )

        for _ in range(100):
            manager.update(rest_state, dt=0.01)

        assert manager._state_duration >= 0.9  # ~100 * 0.01
