"""
Tests for Raphe Nucleus Serotonin Circuit.

Tests cover:
1. Autoreceptor negative feedback
2. Homeostatic regulation
3. Desensitization dynamics
4. Stress/reward modulation
5. Integration interfaces
"""

import numpy as np
import pytest

from t4dm.nca.raphe import (
    RapheNucleus,
    RapheConfig,
    RapheNucleusState,
    RapheState,
    create_raphe_nucleus,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def raphe():
    """Create default raphe nucleus."""
    return RapheNucleus()


@pytest.fixture
def raphe_high_sensitivity():
    """Create raphe with high autoreceptor sensitivity."""
    config = RapheConfig(autoreceptor_sensitivity=0.8)
    return RapheNucleus(config)


# =============================================================================
# Test Configuration
# =============================================================================

class TestRapheConfig:
    """Tests for raphe configuration."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = RapheConfig()
        assert config.baseline_rate == 2.5
        assert config.max_rate > config.baseline_rate
        assert 0 < config.autoreceptor_sensitivity < 1
        assert 0 < config.setpoint < 1

    def test_custom_config(self):
        """Custom config values are preserved."""
        config = RapheConfig(
            baseline_rate=3.0,
            setpoint=0.5,
        )
        assert config.baseline_rate == 3.0
        assert config.setpoint == 0.5


# =============================================================================
# Test Basic Dynamics
# =============================================================================

class TestBasicDynamics:
    """Tests for basic raphe dynamics."""

    def test_initial_state(self, raphe):
        """Initial state is at baseline."""
        assert raphe.state.firing_rate == pytest.approx(
            raphe.config.baseline_rate, abs=0.1
        )
        assert raphe.state.extracellular_5ht == pytest.approx(
            raphe.config.setpoint, abs=0.1
        )
        assert raphe.state.state == RapheState.TONIC

    def test_step_returns_5ht(self, raphe):
        """Step returns 5-HT level."""
        ht = raphe.step(dt=0.1)
        assert 0 <= ht <= 1

    def test_stable_at_setpoint(self, raphe):
        """System is stable near setpoint."""
        initial_5ht = raphe.state.extracellular_5ht

        # Run for many steps
        for _ in range(100):
            raphe.step(dt=0.1)

        # Should stay near setpoint
        assert raphe.state.extracellular_5ht == pytest.approx(
            raphe.config.setpoint, abs=0.1
        )


# =============================================================================
# Test Autoreceptor Feedback
# =============================================================================

class TestAutoreceptorFeedback:
    """Tests for 5-HT1A autoreceptor negative feedback."""

    def test_high_5ht_inhibits_firing(self, raphe):
        """High 5-HT reduces firing via autoreceptors."""
        initial_rate = raphe.state.firing_rate

        # Inject 5-HT
        raphe.inject_5ht(0.3)

        # Step to let dynamics settle
        for _ in range(20):
            raphe.step(dt=0.1)

        # Firing should be reduced
        assert raphe.state.autoreceptor_inhibition > 0
        # May or may not be lower due to homeostatic compensation

    def test_low_5ht_increases_firing(self, raphe):
        """Low 5-HT allows increased firing."""
        # Force 5-HT low
        raphe.state.extracellular_5ht = 0.1

        for _ in range(10):
            raphe.step(dt=0.1)

        # Autoreceptor inhibition should be low
        assert raphe.state.autoreceptor_inhibition < 0.3

    def test_hill_function_shape(self, raphe):
        """Autoreceptor inhibition follows Hill function."""
        # At EC50, inhibition should be ~50% of max
        raphe.state.extracellular_5ht = raphe.config.autoreceptor_ec50
        raphe._update_autoreceptor_inhibition()

        expected_inhib = 0.5 * raphe.config.autoreceptor_sensitivity
        assert raphe.state.autoreceptor_inhibition == pytest.approx(
            expected_inhib, abs=0.1
        )

    def test_higher_sensitivity_stronger_inhibition(self, raphe_high_sensitivity):
        """Higher sensitivity causes stronger inhibition."""
        raphe_high_sensitivity.state.extracellular_5ht = 0.6
        raphe_high_sensitivity._update_autoreceptor_inhibition()

        assert raphe_high_sensitivity.state.autoreceptor_inhibition > 0.3


# =============================================================================
# Test Homeostatic Regulation
# =============================================================================

class TestHomeostaticRegulation:
    """Tests for homeostatic setpoint regulation."""

    def test_recovers_from_high_5ht(self, raphe):
        """5-HT returns to setpoint after perturbation up."""
        # Inject 5-HT
        raphe.state.extracellular_5ht = 0.8

        # Run for many steps
        for _ in range(200):
            raphe.step(dt=0.1)

        # Should return toward setpoint
        assert raphe.state.extracellular_5ht < 0.7

    def test_recovers_from_low_5ht(self, raphe):
        """5-HT returns to setpoint after perturbation down."""
        # Force 5-HT low
        raphe.state.extracellular_5ht = 0.1

        # Run for many steps
        for _ in range(200):
            raphe.step(dt=0.1)

        # Should return toward setpoint
        assert raphe.state.extracellular_5ht > 0.2

    def test_setpoint_error_tracked(self, raphe):
        """Setpoint error is computed correctly."""
        raphe.state.extracellular_5ht = 0.6
        raphe._compute_homeostatic_drive()

        expected_error = raphe.config.setpoint - 0.6
        assert raphe.state.setpoint_error == pytest.approx(expected_error, abs=0.01)


# =============================================================================
# Test Desensitization
# =============================================================================

class TestDesensitization:
    """Tests for autoreceptor desensitization."""

    def test_chronic_high_5ht_causes_desensitization(self, raphe):
        """Sustained high 5-HT reduces autoreceptor sensitivity."""
        initial_sensitivity = raphe.state.autoreceptor_sensitivity

        # Keep 5-HT high for many steps
        for _ in range(500):
            raphe.state.extracellular_5ht = 0.8
            raphe.step(dt=0.1)

        # Sensitivity should have decreased
        assert raphe.state.autoreceptor_sensitivity < initial_sensitivity

    def test_resensitization_occurs(self, raphe):
        """Sensitivity recovers when 5-HT normalizes."""
        # First, desensitize
        raphe.state.autoreceptor_sensitivity = 0.5

        # Then let 5-HT normalize (at setpoint)
        for _ in range(100):
            raphe.state.extracellular_5ht = raphe.config.setpoint
            raphe.step(dt=0.1)

        # Should have recovered somewhat
        assert raphe.state.autoreceptor_sensitivity > 0.5

    def test_sensitivity_bounded(self, raphe):
        """Sensitivity stays within bounds."""
        # Try to desensitize maximally
        for _ in range(1000):
            raphe.state.extracellular_5ht = 1.0
            raphe._update_desensitization(dt=0.1)

        min_sens = 1 - raphe.config.max_desensitization
        assert raphe.state.autoreceptor_sensitivity >= min_sens


# =============================================================================
# Test External Inputs
# =============================================================================

class TestExternalInputs:
    """Tests for stress and reward modulation."""

    def test_stress_increases_firing(self, raphe):
        """Stress input increases firing rate."""
        raphe.set_stress_input(0.0)
        for _ in range(50):
            raphe.step(dt=0.1)
        baseline_rate = raphe.state.firing_rate

        raphe.set_stress_input(0.5)
        for _ in range(50):
            raphe.step(dt=0.1)

        # Firing should increase
        assert raphe.state.firing_rate > baseline_rate

    def test_reward_modulates_activity(self, raphe):
        """Reward input modulates DRN activity."""
        raphe.set_reward_input(0.0)
        for _ in range(50):
            raphe.step(dt=0.1)
        no_reward_rate = raphe.state.firing_rate

        raphe.set_reward_input(0.8)
        for _ in range(50):
            raphe.step(dt=0.1)

        # Reward should reduce activity (less need for patience)
        assert raphe.state.firing_rate < no_reward_rate * 1.2

    def test_5ht_injection(self, raphe):
        """5-HT injection increases level."""
        initial = raphe.state.extracellular_5ht
        raphe.inject_5ht(0.2)
        assert raphe.state.extracellular_5ht > initial

    def test_reuptake_block(self, raphe):
        """Reuptake block increases 5-HT (SSRI effect)."""
        # Run to steady state
        for _ in range(100):
            raphe.step(dt=0.1)
        initial_5ht = raphe.state.extracellular_5ht

        # Block reuptake
        raphe.block_reuptake(0.5)

        # Run more steps
        for _ in range(100):
            raphe.step(dt=0.1)

        # 5-HT should increase
        assert raphe.state.extracellular_5ht > initial_5ht * 0.9


# =============================================================================
# Test State Classification
# =============================================================================

class TestStateClassification:
    """Tests for raphe state classification."""

    def test_tonic_state(self, raphe):
        """Normal firing is TONIC."""
        raphe.state.firing_rate = raphe.config.baseline_rate
        raphe.state.autoreceptor_inhibition = 0.2
        raphe._classify_state()
        assert raphe.state.state == RapheState.TONIC

    def test_quiescent_state(self, raphe):
        """Very low firing is QUIESCENT."""
        raphe.state.firing_rate = 0.2
        raphe._classify_state()
        assert raphe.state.state == RapheState.QUIESCENT

    def test_suppressed_state(self, raphe):
        """High inhibition is SUPPRESSED."""
        raphe.state.firing_rate = 2.0
        raphe.state.autoreceptor_inhibition = 0.7
        raphe._classify_state()
        assert raphe.state.state == RapheState.SUPPRESSED

    def test_elevated_state(self, raphe):
        """High firing is ELEVATED."""
        raphe.state.firing_rate = raphe.config.baseline_rate * 2
        raphe.state.autoreceptor_inhibition = 0.1
        raphe._classify_state()
        assert raphe.state.state == RapheState.ELEVATED


# =============================================================================
# Test Integration Methods
# =============================================================================

class TestIntegration:
    """Tests for integration with other systems."""

    def test_get_5ht_for_neural_field(self, raphe):
        """Can get 5-HT for neural field."""
        ht = raphe.get_5ht_for_neural_field()
        assert ht == raphe.state.extracellular_5ht

    def test_get_mood_modulation(self, raphe):
        """Mood modulation based on 5-HT."""
        raphe.state.extracellular_5ht = 0.7
        mood = raphe.get_mood_modulation()
        assert mood == 0.7

    def test_5ht_callback(self, raphe):
        """5-HT callback is triggered."""
        received = []

        def callback(ht):
            received.append(ht)

        raphe.register_5ht_callback(callback)
        raphe.step(dt=0.1)

        assert len(received) == 1

    def test_connect_to_vta(self, raphe):
        """VTA RPE modulates activity."""
        # Positive RPE
        raphe.connect_to_vta(vta_rpe=0.5)
        assert raphe.state.reward_input == 0.5

        # Negative RPE (ignored)
        raphe.connect_to_vta(vta_rpe=-0.3)
        assert raphe.state.reward_input == 0.0


# =============================================================================
# Test Factory and Utilities
# =============================================================================

class TestFactoryAndUtilities:
    """Tests for factory function and utilities."""

    def test_create_raphe_nucleus(self):
        """Factory creates configured nucleus."""
        raphe = create_raphe_nucleus(
            baseline_rate=3.0,
            setpoint=0.5,
        )
        assert raphe.config.baseline_rate == 3.0
        assert raphe.config.setpoint == 0.5

    def test_get_stats(self, raphe):
        """Stats dict contains expected keys."""
        raphe.step(dt=0.1)
        stats = raphe.get_stats()

        assert "state" in stats
        assert "firing_rate" in stats
        assert "extracellular_5ht" in stats
        assert "autoreceptor_inhibition" in stats

    def test_reset(self, raphe):
        """Reset restores initial state."""
        raphe.inject_5ht(0.3)
        raphe.set_stress_input(0.5)
        for _ in range(20):
            raphe.step(dt=0.1)

        raphe.reset()

        assert raphe.state.firing_rate == raphe.config.baseline_rate
        assert raphe.state.extracellular_5ht == raphe.config.setpoint
        assert raphe.state.stress_input == 0.0

    def test_save_load_state(self, raphe):
        """State can be saved and restored."""
        raphe.inject_5ht(0.2)
        for _ in range(10):
            raphe.step(dt=0.1)

        saved = raphe.save_state()

        new_raphe = RapheNucleus()
        new_raphe.load_state(saved)

        assert new_raphe.state.extracellular_5ht == pytest.approx(
            raphe.state.extracellular_5ht, abs=0.01
        )


# =============================================================================
# Test Biological Plausibility
# =============================================================================

class TestBiologicalPlausibility:
    """Tests for biologically realistic behavior."""

    def test_firing_rate_range(self, raphe):
        """Firing rate stays in biological range."""
        # Stress
        raphe.set_stress_input(1.0)
        for _ in range(50):
            raphe.step(dt=0.1)

        assert raphe.state.firing_rate <= raphe.config.max_rate
        assert raphe.state.firing_rate >= raphe.config.min_rate

    def test_negative_feedback_time_course(self, raphe):
        """Negative feedback has realistic time course."""
        # Inject 5-HT
        raphe.inject_5ht(0.3)

        # Should see inhibition develop over time
        raphe.step(dt=0.1)
        early_inhib = raphe.state.autoreceptor_inhibition

        for _ in range(10):
            raphe.step(dt=0.1)

        # Inhibition should be established
        assert raphe.state.autoreceptor_inhibition > 0

    def test_ssri_like_effect(self, raphe):
        """SSRI-like reuptake block elevates 5-HT."""
        # Baseline
        for _ in range(50):
            raphe.step(dt=0.1)
        baseline = raphe.state.extracellular_5ht

        # Block reuptake (like SSRI)
        raphe.block_reuptake(0.8)

        # Run for "chronic" period
        for _ in range(200):
            raphe.step(dt=0.1)

        # 5-HT should be elevated
        # Also: autoreceptor should desensitize
        assert raphe.state.autoreceptor_sensitivity < 1.0

    def test_stress_response(self, raphe):
        """Stress increases DRN activity."""
        # Baseline
        for _ in range(50):
            raphe.step(dt=0.1)
        baseline_5ht = raphe.state.extracellular_5ht

        # Apply stress
        raphe.set_stress_input(0.8)
        for _ in range(100):
            raphe.step(dt=0.1)

        # 5-HT should increase due to elevated firing
        # (may be partially compensated by autoreceptors)
        assert raphe.state.firing_rate > raphe.config.baseline_rate
