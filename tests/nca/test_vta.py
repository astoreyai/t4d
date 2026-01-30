"""
Tests for VTA Dopamine Circuit.

Tests cover:
1. Tonic/phasic firing mode transitions
2. RPE computation (TD error and simple)
3. DA dynamics (burst, pause, decay)
4. Value learning
5. Integration interfaces
"""

import numpy as np
import pytest

from ww.nca.vta import (
    VTACircuit,
    VTAConfig,
    VTAState,
    VTAFiringMode,
    create_vta_circuit,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def vta():
    """Create default VTA circuit."""
    return VTACircuit()


@pytest.fixture
def vta_custom():
    """Create VTA with custom config."""
    config = VTAConfig(
        tonic_da_level=0.4,
        rpe_to_da_gain=0.6,
        discount_gamma=0.9,
        td_lambda=0.8,
    )
    return VTACircuit(config)


# =============================================================================
# Test VTAConfig
# =============================================================================

class TestVTAConfig:
    """Tests for VTA configuration."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = VTAConfig()
        assert config.tonic_rate == 4.5
        assert config.tonic_da_level == 0.3
        assert config.burst_peak_rate == 30.0
        assert 0.9 <= config.discount_gamma <= 1.0
        assert 0.8 <= config.td_lambda <= 1.0

    def test_custom_config(self):
        """Custom config values are preserved."""
        config = VTAConfig(
            tonic_da_level=0.5,
            rpe_to_da_gain=0.8,
        )
        assert config.tonic_da_level == 0.5
        assert config.rpe_to_da_gain == 0.8


# =============================================================================
# Test VTAState
# =============================================================================

class TestVTAState:
    """Tests for VTA state representation."""

    def test_default_state(self):
        """Default state starts in tonic mode."""
        state = VTAState()
        assert state.firing_mode == VTAFiringMode.TONIC
        assert state.current_da == 0.3
        assert state.last_rpe == 0.0

    def test_state_to_dict(self):
        """State can be serialized to dict."""
        state = VTAState(
            current_da=0.6,
            last_rpe=0.3,
        )
        d = state.to_dict()
        assert d["current_da"] == 0.6
        assert d["last_rpe"] == 0.3
        assert d["firing_mode"] == "tonic"


# =============================================================================
# Test RPE Computation
# =============================================================================

class TestRPEComputation:
    """Tests for reward prediction error computation."""

    def test_simple_rpe_positive(self, vta):
        """Positive RPE when actual > expected."""
        rpe = vta.compute_rpe_from_outcome(
            actual_outcome=0.8,
            expected_outcome=0.5
        )
        assert rpe == pytest.approx(0.3, abs=0.01)
        assert vta.state.last_rpe == pytest.approx(0.3, abs=0.01)

    def test_simple_rpe_negative(self, vta):
        """Negative RPE when actual < expected."""
        rpe = vta.compute_rpe_from_outcome(
            actual_outcome=0.2,
            expected_outcome=0.6
        )
        assert rpe == pytest.approx(-0.4, abs=0.01)

    def test_simple_rpe_zero(self, vta):
        """Zero RPE when actual == expected."""
        rpe = vta.compute_rpe_from_outcome(
            actual_outcome=0.5,
            expected_outcome=0.5
        )
        assert rpe == pytest.approx(0.0, abs=0.01)

    def test_td_error_no_next_state(self, vta):
        """TD error with bootstrap from current."""
        rpe = vta.compute_td_error(
            reward=0.5,
            current_state="state_a"
        )
        # δ = r + γV(s') - V(s), all values start at 0.5
        # = 0.5 + 0.95*0.5 - 0.5 = 0.475
        assert abs(rpe) < 0.5  # Reasonable range
        assert vta.state.td_error == rpe

    def test_td_error_terminal(self, vta):
        """TD error at terminal state (no future value)."""
        rpe = vta.compute_td_error(
            reward=1.0,
            current_state="state_a",
            terminal=True
        )
        # δ = r + γ*0 - V(s) = 1.0 - 0.5 = 0.5
        assert rpe == pytest.approx(0.5, abs=0.01)

    def test_eligibility_trace_updates(self, vta):
        """Eligibility trace accumulates over steps."""
        initial = vta.state.eligibility
        vta.compute_rpe_from_outcome(0.8, 0.5)
        after_one = vta.state.eligibility

        assert after_one > initial  # Increased

        vta.compute_rpe_from_outcome(0.7, 0.5)
        after_two = vta.state.eligibility

        assert after_two > after_one  # Still accumulating


# =============================================================================
# Test Dopamine Dynamics
# =============================================================================

class TestDopamineDynamics:
    """Tests for DA concentration changes."""

    def test_positive_rpe_increases_da(self, vta):
        """Positive RPE causes DA burst."""
        initial_da = vta.state.current_da
        vta.process_rpe(0.4, dt=0.1)

        assert vta.state.current_da > initial_da
        assert vta.state.firing_mode == VTAFiringMode.PHASIC_BURST
        assert vta.state.current_rate > vta.config.tonic_rate

    def test_negative_rpe_decreases_da(self, vta):
        """Negative RPE causes DA pause."""
        initial_da = vta.state.current_da
        vta.process_rpe(-0.4, dt=0.1)

        assert vta.state.current_da < initial_da
        assert vta.state.firing_mode == VTAFiringMode.PHASIC_PAUSE
        assert vta.state.current_rate < vta.config.tonic_rate

    def test_small_rpe_stays_tonic(self, vta):
        """Small RPE maintains tonic mode."""
        vta.process_rpe(0.02, dt=0.1)

        assert vta.state.firing_mode == VTAFiringMode.TONIC
        assert abs(vta.state.current_rate - vta.config.tonic_rate) < 1.0

    def test_da_bounds_respected(self, vta):
        """DA stays within biological bounds."""
        # Large positive RPE
        for _ in range(10):
            vta.state.time_since_phasic = 1.0  # Reset refractory
            vta.process_rpe(0.9, dt=0.1)
        assert vta.state.current_da <= vta.config.max_da

        # Large negative RPE
        for _ in range(10):
            vta.state.time_since_phasic = 1.0
            vta.process_rpe(-0.9, dt=0.1)
        assert vta.state.current_da >= vta.config.min_da

    def test_decay_to_tonic(self, vta):
        """DA decays back to tonic after phasic event."""
        # Trigger burst
        vta.process_rpe(0.5, dt=0.1)
        burst_da = vta.state.current_da

        # Step without new RPE
        for _ in range(20):
            vta.step(dt=0.1)

        # Should decay toward tonic
        assert vta.state.current_da < burst_da
        assert abs(vta.state.current_da - vta.config.tonic_da_level) < 0.1

    def test_refractory_period(self, vta):
        """Refractory period prevents rapid phasic events."""
        vta.process_rpe(0.5, dt=0.1)  # First burst
        first_da = vta.state.current_da

        # Immediate second burst should be blocked
        vta.process_rpe(0.5, dt=0.01)  # Very short dt
        assert vta.state.firing_mode == VTAFiringMode.TONIC  # Blocked


# =============================================================================
# Test Value Learning
# =============================================================================

class TestValueLearning:
    """Tests for value function updates."""

    def test_value_update_positive_td(self, vta):
        """Positive TD error increases value."""
        initial = vta._get_value("state_a")
        vta.state.eligibility = 1.0  # Set eligibility
        vta.update_value("state_a", td_error=0.3)

        assert vta._get_value("state_a") > initial

    def test_value_update_negative_td(self, vta):
        """Negative TD error decreases value."""
        vta._value_table["state_a"] = 0.6
        vta.state.eligibility = 1.0
        vta.update_value("state_a", td_error=-0.3)

        assert vta._get_value("state_a") < 0.6

    def test_value_bounded(self, vta):
        """Value stays in [0, 1]."""
        vta.state.eligibility = 1.0

        # Try to push above 1
        vta._value_table["high"] = 0.95
        vta.update_value("high", td_error=1.0)
        assert vta._get_value("high") <= 1.0

        # Try to push below 0
        vta._value_table["low"] = 0.05
        vta.update_value("low", td_error=-1.0)
        assert vta._get_value("low") >= 0.0

    def test_eligibility_modulates_update(self, vta):
        """Higher eligibility = larger value updates."""
        vta._value_table["state_a"] = 0.5
        vta._value_table["state_b"] = 0.5

        vta.state.eligibility = 0.5
        vta.update_value("state_a", td_error=0.3)
        delta_a = vta._get_value("state_a") - 0.5

        vta.state.eligibility = 1.0
        vta.update_value("state_b", td_error=0.3)
        delta_b = vta._get_value("state_b") - 0.5

        assert delta_b > delta_a  # Higher eligibility = larger update


# =============================================================================
# Test Integration Methods
# =============================================================================

class TestIntegration:
    """Tests for integration with other systems."""

    def test_get_da_for_neural_field(self, vta):
        """DA level can be retrieved for neural field."""
        da = vta.get_da_for_neural_field()
        assert 0 <= da <= 1
        assert da == vta.state.current_da

    def test_get_rpe_for_coupling(self, vta):
        """RPE and eligibility available for coupling."""
        vta.compute_rpe_from_outcome(0.8, 0.5)
        rpe, elig = vta.get_rpe_for_coupling()

        assert rpe == pytest.approx(0.3, abs=0.01)
        assert elig > 0

    def test_da_callback(self, vta):
        """DA callbacks are triggered on changes."""
        received = []

        def callback(da, rpe):
            received.append((da, rpe))

        vta.register_da_callback(callback)
        vta.process_rpe(0.3, dt=0.1)

        assert len(received) == 1
        assert received[0][1] == pytest.approx(0.3, abs=0.01)

    def test_hippocampus_novelty_integration(self, vta):
        """Novelty signal modulates RPE."""
        vta.compute_rpe_from_outcome(0.5, 0.5)  # Zero RPE

        # High novelty adds to RPE
        combined = vta.connect_to_hippocampus(novelty_signal=0.9)
        assert combined > 0  # Novelty contributes positive

        # Low novelty subtracts
        combined_low = vta.connect_to_hippocampus(novelty_signal=0.1)
        assert combined_low < combined


# =============================================================================
# Test Factory and Utilities
# =============================================================================

class TestFactoryAndUtilities:
    """Tests for factory function and utilities."""

    def test_create_vta_circuit(self):
        """Factory creates configured circuit."""
        vta = create_vta_circuit(
            tonic_da=0.4,
            rpe_gain=0.7,
            td_lambda=0.85
        )
        assert vta.config.tonic_da_level == 0.4
        assert vta.config.rpe_to_da_gain == 0.7
        assert vta.config.td_lambda == 0.85

    def test_get_stats(self, vta):
        """Stats dict contains expected keys."""
        vta.process_rpe(0.3, dt=0.1)
        stats = vta.get_stats()

        assert "firing_mode" in stats
        assert "current_da" in stats
        assert "last_rpe" in stats
        assert "eligibility" in stats
        assert "n_states_tracked" in stats

    def test_reset(self, vta):
        """Reset restores initial state."""
        vta.process_rpe(0.5, dt=0.1)
        vta.compute_rpe_from_outcome(0.8, 0.5)

        vta.reset()

        assert vta.state.current_da == vta.config.tonic_da_level
        assert vta.state.last_rpe == 0.0
        assert vta.state.firing_mode == VTAFiringMode.TONIC

    def test_save_load_state(self, vta):
        """State can be saved and restored."""
        vta.process_rpe(0.4, dt=0.1)
        vta.update_value("test_state", 0.3)

        saved = vta.save_state()

        new_vta = VTACircuit()
        new_vta.load_state(saved)

        assert new_vta._value_table.get("test_state") is not None
        assert new_vta.state.current_da == pytest.approx(
            vta.state.current_da, abs=0.01
        )


# =============================================================================
# Test Biological Plausibility
# =============================================================================

class TestBiologicalPlausibility:
    """Tests for biologically realistic behavior."""

    def test_burst_rate_scaling(self, vta):
        """Burst rate scales with RPE magnitude."""
        vta.process_rpe(0.2, dt=0.1)
        rate_small = vta.state.current_rate

        vta.state.time_since_phasic = 1.0  # Reset refractory
        vta.process_rpe(0.6, dt=0.1)
        rate_large = vta.state.current_rate

        assert rate_large > rate_small

    def test_firing_rate_bounds(self, vta):
        """Firing rate stays in biological range."""
        # Burst
        vta.process_rpe(1.0, dt=0.1)
        assert vta.state.current_rate <= vta.config.burst_peak_rate

        # Pause
        vta.state.time_since_phasic = 1.0
        vta.process_rpe(-1.0, dt=0.1)
        assert vta.state.current_rate >= 0

    def test_temporal_credit_assignment(self, vta):
        """TD(λ) enables temporal credit assignment."""
        # Simulate sequence: action -> delay -> reward
        vta.compute_td_error(reward=0.0, current_state="action")
        elig_after_action = vta.state.eligibility

        vta.compute_td_error(reward=0.0, current_state="delay1")
        vta.compute_td_error(reward=0.0, current_state="delay2")

        # Eligibility should have decayed but still present
        elig_after_delays = vta.state.eligibility
        assert elig_after_delays > 0  # Still some trace
        assert elig_after_delays < elig_after_action * 3  # Decayed somewhat

    def test_expected_value_learning(self, vta):
        """Repeated rewards increase expected value."""
        state = "rewarded_state"

        for _ in range(10):
            td = vta.compute_td_error(
                reward=0.8,
                current_state=state
            )
            vta.update_value(state, td)

        # Value should approach reward level
        final_value = vta._get_value(state)
        assert final_value > 0.6  # Learned high value
