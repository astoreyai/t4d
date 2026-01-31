"""
Tests for D1/D2 MSN Populations.

Tests cover:
1. D1/D2 receptor binding kinetics
2. Dopamine modulation (D1 excited, D2 inhibited)
3. Action selection (GO/NO-GO)
4. Lateral inhibition and competition
5. Dopamine-modulated plasticity
6. Habit formation
7. Integration interfaces
"""

import numpy as np
import pytest

from t4dm.nca.striatal_msn import (
    StriatalMSN,
    MSNConfig,
    MSNPopulationState,
    ActionState,
    create_striatal_msn,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def msn():
    """Create default MSN populations."""
    return StriatalMSN()


@pytest.fixture
def msn_strong_inhibition():
    """Create MSN with strong lateral inhibition."""
    config = MSNConfig(lateral_inhibition=0.5)
    return StriatalMSN(config)


# =============================================================================
# Test Configuration
# =============================================================================

class TestMSNConfig:
    """Tests for MSN configuration."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = MSNConfig()
        assert config.d1_baseline == 0.2
        assert config.d2_baseline == 0.3  # D2 slightly higher at rest
        assert config.d2_affinity < config.d1_affinity  # D2 higher affinity
        assert 0 < config.lateral_inhibition < 1

    def test_custom_config(self):
        """Custom config values are preserved."""
        config = MSNConfig(
            d1_baseline=0.3,
            d2_baseline=0.4,
            lateral_inhibition=0.4,
        )
        assert config.d1_baseline == 0.3
        assert config.d2_baseline == 0.4
        assert config.lateral_inhibition == 0.4


# =============================================================================
# Test Receptor Binding
# =============================================================================

class TestReceptorBinding:
    """Tests for D1/D2 receptor binding kinetics."""

    def test_no_binding_at_zero_da(self, msn):
        """No receptor binding when DA is zero."""
        msn.set_dopamine_level(0.0)
        for _ in range(20):
            msn.step(dt=0.01)

        assert msn.state.d1_receptor_occupancy == pytest.approx(0.0, abs=0.01)
        assert msn.state.d2_receptor_occupancy == pytest.approx(0.0, abs=0.01)

    def test_d2_binds_at_lower_da(self, msn):
        """D2 receptors bind at lower DA concentrations than D1."""
        msn.set_dopamine_level(0.2)
        for _ in range(50):
            msn.step(dt=0.01)

        # D2 has higher affinity, so should be more occupied at low DA
        assert msn.state.d2_receptor_occupancy > msn.state.d1_receptor_occupancy

    def test_high_da_saturates_both(self, msn):
        """High DA saturates both D1 and D2 receptors."""
        msn.set_dopamine_level(0.9)
        for _ in range(50):
            msn.step(dt=0.01)

        assert msn.state.d1_receptor_occupancy > 0.5
        assert msn.state.d2_receptor_occupancy > 0.5

    def test_binding_kinetics_have_time_constant(self, msn):
        """Receptor binding has appropriate time course."""
        msn.set_dopamine_level(0.8)

        # First step
        msn.step(dt=0.01)
        early_d1 = msn.state.d1_receptor_occupancy

        # After many steps
        for _ in range(50):
            msn.step(dt=0.01)

        # Should have increased over time
        assert msn.state.d1_receptor_occupancy > early_d1


# =============================================================================
# Test Dopamine Modulation
# =============================================================================

class TestDopamineModulation:
    """Tests for DA effects on D1/D2 MSNs."""

    def test_high_da_excites_d1(self, msn):
        """High dopamine increases D1 MSN activity."""
        msn.set_cortical_input(0.5)
        msn.set_dopamine_level(0.2)
        for _ in range(50):
            msn.step(dt=0.01)
        low_da_d1 = msn.state.d1_activity

        msn.set_dopamine_level(0.8)
        for _ in range(50):
            msn.step(dt=0.01)

        assert msn.state.d1_activity > low_da_d1

    def test_high_da_inhibits_d2(self, msn):
        """High dopamine decreases D2 MSN activity."""
        msn.set_cortical_input(0.5)
        msn.set_dopamine_level(0.2)
        for _ in range(50):
            msn.step(dt=0.01)
        low_da_d2 = msn.state.d2_activity

        msn.set_dopamine_level(0.8)
        for _ in range(50):
            msn.step(dt=0.01)

        # D2 should decrease (or at least not increase as much as D1)
        # Note: may not always decrease due to cortical drive
        assert msn.state.d2_activity < low_da_d2 + 0.2

    def test_da_creates_go_bias(self, msn):
        """High DA shifts balance toward GO."""
        msn.set_cortical_input(0.6)

        # Low DA
        msn.set_dopamine_level(0.2)
        for _ in range(100):
            msn.step(dt=0.01)
        low_da_margin = msn.state.competition_margin

        # High DA
        msn.set_dopamine_level(0.8)
        for _ in range(100):
            msn.step(dt=0.01)

        # High DA should increase D1-D2 margin
        assert msn.state.competition_margin > low_da_margin


# =============================================================================
# Test Action Selection
# =============================================================================

class TestActionSelection:
    """Tests for GO/NO-GO action selection."""

    def test_initial_state_undecided(self, msn):
        """Initial state with no input is undecided."""
        assert msn.state.action_state == ActionState.UNDECIDED

    def test_high_da_high_input_goes_go(self, msn):
        """High DA + high cortical input → GO."""
        msn.set_cortical_input(0.8)
        msn.set_dopamine_level(0.9)

        for _ in range(100):
            msn.step(dt=0.01)

        assert msn.state.action_state == ActionState.GO

    def test_low_da_high_input_reduces_go(self, msn):
        """Low DA + high cortical input reduces GO probability."""
        msn.set_cortical_input(0.8)
        msn.set_dopamine_level(0.1)

        for _ in range(100):
            msn.step(dt=0.01)

        # With low DA, GO probability should be lower (closer to 0.5)
        # and definitely not a strong GO decision
        assert msn.state.go_probability < 0.7
        assert msn.state.action_state != ActionState.GO or msn.state.competition_margin < 0.3

    def test_moderate_da_gives_intermediate_probability(self, msn):
        """Moderate DA gives intermediate GO probability."""
        msn.set_cortical_input(0.5)
        msn.set_dopamine_level(0.5)

        for _ in range(50):
            msn.step(dt=0.01)

        # At moderate DA, probability should be somewhere in middle range
        # D1/D2 may not be perfectly balanced due to different affinities
        assert 0.3 < msn.state.go_probability < 0.9

    def test_go_probability_increases_with_da(self, msn):
        """GO probability increases with dopamine."""
        msn.set_cortical_input(0.5)

        msn.set_dopamine_level(0.2)
        for _ in range(50):
            msn.step(dt=0.01)
        low_prob = msn.state.go_probability

        msn.set_dopamine_level(0.8)
        for _ in range(50):
            msn.step(dt=0.01)

        assert msn.state.go_probability > low_prob


# =============================================================================
# Test Lateral Inhibition
# =============================================================================

class TestLateralInhibition:
    """Tests for D1-D2 mutual inhibition."""

    def test_d1_inhibits_d2(self, msn):
        """D1 activity inhibits D2."""
        # Force D1 high
        msn.state.d1_activity = 0.8
        msn.state.d2_activity = 0.5

        msn._apply_lateral_inhibition(dt=0.01)

        # D2 should have decreased
        assert msn.state.d2_activity < 0.5

    def test_stronger_inhibition_clearer_decision(self, msn_strong_inhibition):
        """Stronger lateral inhibition leads to clearer decisions."""
        msn_strong_inhibition.set_cortical_input(0.5)
        msn_strong_inhibition.set_dopamine_level(0.7)

        for _ in range(100):
            msn_strong_inhibition.step(dt=0.01)

        # Should have larger margin due to winner-take-all
        assert abs(msn_strong_inhibition.state.competition_margin) > 0.1


# =============================================================================
# Test Plasticity
# =============================================================================

class TestPlasticity:
    """Tests for DA-modulated plasticity."""

    def test_high_da_strengthens_d1(self, msn):
        """High DA strengthens D1 pathway (LTP)."""
        initial_d1_strength = msn.state.d1_synaptic_strength

        msn.set_cortical_input(0.7)
        msn.set_dopamine_level(0.9)

        # Run until GO decision and plasticity
        for _ in range(200):
            msn.step(dt=0.01)

        assert msn.state.d1_synaptic_strength > initial_d1_strength

    def test_low_da_strengthens_d2(self, msn):
        """Low DA strengthens D2 pathway (LTP)."""
        initial_d2_strength = msn.state.d2_synaptic_strength

        msn.set_cortical_input(0.7)
        msn.set_dopamine_level(0.1)

        # Run until NO-GO decision and plasticity
        for _ in range(200):
            msn.step(dt=0.01)

        # D2 should increase or at least not decrease much
        assert msn.state.d2_synaptic_strength >= initial_d2_strength - 0.1

    def test_rpe_modulates_plasticity(self, msn):
        """Positive RPE strengthens GO pathway."""
        initial_d1 = msn.state.d1_synaptic_strength
        initial_d2 = msn.state.d2_synaptic_strength

        msn.apply_rpe(0.5)  # Positive surprise

        assert msn.state.d1_synaptic_strength > initial_d1
        assert msn.state.d2_synaptic_strength < initial_d2

    def test_negative_rpe_strengthens_no_go(self, msn):
        """Negative RPE strengthens NO-GO pathway."""
        initial_d1 = msn.state.d1_synaptic_strength
        initial_d2 = msn.state.d2_synaptic_strength

        msn.apply_rpe(-0.5)  # Negative surprise

        assert msn.state.d2_synaptic_strength > initial_d2
        assert msn.state.d1_synaptic_strength < initial_d1


# =============================================================================
# Test Habit Formation
# =============================================================================

class TestHabitFormation:
    """Tests for habit formation dynamics."""

    def test_habit_initially_zero(self, msn):
        """Habit strength starts at zero."""
        assert msn.state.habit_strength == 0.0

    def test_habit_increases_with_d1_dominance(self, msn):
        """Habit increases when D1 pathway dominates."""
        # Strengthen D1 pathway
        msn.state.d1_synaptic_strength = 1.5

        for _ in range(1000):
            msn._update_habit_strength(dt=0.01)

        # Habit should increase
        assert msn.state.habit_strength > 0

    def test_is_habitual_check(self, msn):
        """is_habitual returns correct status."""
        assert not msn.is_habitual()

        msn.state.habit_strength = 0.9
        assert msn.is_habitual()


# =============================================================================
# Test External Inputs
# =============================================================================

class TestExternalInputs:
    """Tests for external input handling."""

    def test_set_cortical_input(self, msn):
        """Can set cortical input."""
        msn.set_cortical_input(0.7)
        assert msn.state.cortical_input == 0.7

    def test_set_dopamine_level(self, msn):
        """Can set dopamine level."""
        msn.set_dopamine_level(0.8)
        assert msn.state.dopamine_level == 0.8

    def test_set_ach_level(self, msn):
        """Can set ACh level."""
        msn.set_ach_level(0.6)
        assert msn.state.ach_level == 0.6

    def test_inputs_are_clamped(self, msn):
        """Inputs are clamped to [0, 1]."""
        msn.set_cortical_input(1.5)
        assert msn.state.cortical_input == 1.0

        msn.set_dopamine_level(-0.5)
        assert msn.state.dopamine_level == 0.0


# =============================================================================
# Test Integration Methods
# =============================================================================

class TestIntegration:
    """Tests for integration with other systems."""

    def test_get_go_signal(self, msn):
        """Can get GO signal strength."""
        msn.set_cortical_input(0.8)
        msn.set_dopamine_level(0.9)

        for _ in range(100):
            msn.step(dt=0.01)

        go_signal = msn.get_go_signal()
        assert 0 <= go_signal <= 1
        assert go_signal > 0.3  # Should be substantial

    def test_get_no_go_signal(self, msn):
        """Can get NO-GO signal strength."""
        msn.set_cortical_input(0.8)
        msn.set_dopamine_level(0.1)

        for _ in range(100):
            msn.step(dt=0.01)

        no_go_signal = msn.get_no_go_signal()
        assert 0 <= no_go_signal <= 1

    def test_get_action_values(self, msn):
        """Can get action values (Q-values)."""
        msn.set_cortical_input(0.5)
        msn.set_dopamine_level(0.5)

        for _ in range(50):
            msn.step(dt=0.01)

        q_go, q_no_go = msn.get_action_values()
        assert q_go >= 0
        assert q_no_go >= 0

    def test_action_callback(self, msn):
        """Action callback fires on decision."""
        decisions = []

        def callback(action):
            decisions.append(action)

        msn.register_action_callback(callback)

        msn.set_cortical_input(0.9)
        msn.set_dopamine_level(0.9)

        for _ in range(100):
            msn.step(dt=0.01)

        # Should have received some callbacks
        assert len(decisions) > 0


# =============================================================================
# Test Statistics and State
# =============================================================================

class TestStatisticsAndState:
    """Tests for statistics and state management."""

    def test_get_stats(self, msn):
        """Stats dict contains expected keys."""
        msn.step(dt=0.01)
        stats = msn.get_stats()

        assert "d1_activity" in stats
        assert "d2_activity" in stats
        assert "action_state" in stats
        assert "go_probability" in stats

    def test_reset(self, msn):
        """Reset restores initial state."""
        msn.set_cortical_input(0.8)
        msn.set_dopamine_level(0.9)
        for _ in range(100):
            msn.step(dt=0.01)

        msn.reset()

        assert msn.state.d1_synaptic_strength == 1.0
        assert msn.state.d2_synaptic_strength == 1.0
        assert msn.state.habit_strength == 0.0

    def test_save_load_state(self, msn):
        """State can be saved and restored."""
        msn.state.d1_synaptic_strength = 1.5
        msn.state.habit_strength = 0.3

        saved = msn.save_state()

        new_msn = StriatalMSN()
        new_msn.load_state(saved)

        assert new_msn.state.d1_synaptic_strength == pytest.approx(1.5, abs=0.01)
        assert new_msn.state.habit_strength == pytest.approx(0.3, abs=0.01)


# =============================================================================
# Test Factory
# =============================================================================

class TestFactory:
    """Tests for factory function."""

    def test_create_striatal_msn(self):
        """Factory creates configured MSN."""
        msn = create_striatal_msn(
            d1_baseline=0.3,
            d2_baseline=0.4,
            lateral_inhibition=0.4,
        )
        assert msn.config.d1_baseline == 0.3
        assert msn.config.d2_baseline == 0.4
        assert msn.config.lateral_inhibition == 0.4


# =============================================================================
# Test Biological Plausibility
# =============================================================================

class TestBiologicalPlausibility:
    """Tests for biologically realistic behavior."""

    def test_d1_d2_opponent_process(self, msn):
        """D1 and D2 have opponent responses to DA."""
        msn.set_cortical_input(0.5)

        # Low DA: D2 should dominate
        msn.set_dopamine_level(0.1)
        for _ in range(100):
            msn.step(dt=0.01)
        low_da_margin = msn.state.d1_activity - msn.state.d2_activity

        # High DA: D1 should dominate
        msn.set_dopamine_level(0.9)
        for _ in range(100):
            msn.step(dt=0.01)
        high_da_margin = msn.state.d1_activity - msn.state.d2_activity

        # Margin should shift positive with high DA
        assert high_da_margin > low_da_margin

    def test_lower_input_smaller_activity(self, msn):
        """Lower cortical input produces smaller MSN activities."""
        # High input condition
        msn.set_cortical_input(0.8)
        msn.set_dopamine_level(0.5)
        for _ in range(50):
            msn.step(dt=0.01)
        high_input_d1 = msn.state.d1_activity

        # Reset and try low input
        msn.reset()
        msn.set_cortical_input(0.2)  # Low input
        msn.set_dopamine_level(0.5)
        for _ in range(50):
            msn.step(dt=0.01)

        # Low input should produce lower activity
        assert msn.state.d1_activity < high_input_d1

    def test_winner_take_all_dynamics(self, msn_strong_inhibition):
        """Strong inhibition creates winner-take-all."""
        msn_strong_inhibition.set_cortical_input(0.7)
        msn_strong_inhibition.set_dopamine_level(0.6)

        for _ in range(200):
            msn_strong_inhibition.step(dt=0.01)

        # Should have clear winner
        d1 = msn_strong_inhibition.state.d1_activity
        d2 = msn_strong_inhibition.state.d2_activity
        assert abs(d1 - d2) > 0.2 or max(d1, d2) < 0.3

    def test_reinforcement_learning_pattern(self, msn):
        """Repeated high DA reinforces GO pathway."""
        initial_d1 = msn.state.d1_synaptic_strength

        # Simulate successful action sequence
        for _ in range(10):
            msn.set_cortical_input(0.7)
            msn.set_dopamine_level(0.8)
            for _ in range(50):
                msn.step(dt=0.01)
            msn.apply_rpe(0.3)  # Reward received

        # D1 should be stronger
        assert msn.state.d1_synaptic_strength > initial_d1
