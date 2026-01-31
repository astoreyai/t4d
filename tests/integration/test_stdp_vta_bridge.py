"""
Phase 1B: Tests for STDP-VTA dopamine modulation bridge.

Tests:
- High DA increases LTP, decreases LTD
- Low DA decreases LTP, increases LTD
- Baseline DA produces no modulation
- DA gating blocks learning when DA too low
- Bridge integration with VTA circuit
"""

import pytest
from datetime import datetime, timedelta

from t4dm.integration.stdp_vta_bridge import (
    STDPVTABridge,
    STDPVTAConfig,
    get_stdp_vta_bridge,
    reset_stdp_vta_bridge,
)
from t4dm.learning.stdp import STDPLearner, STDPConfig, reset_stdp_learner
from t4dm.nca.vta import VTACircuit, VTAConfig


@pytest.fixture
def stdp_learner():
    """Create fresh STDP learner for testing."""
    reset_stdp_learner()
    config = STDPConfig(
        a_plus=0.01,
        a_minus=0.0105,
        multiplicative=True,
        mu=0.5
    )
    return STDPLearner(config)


@pytest.fixture
def vta_circuit():
    """Create VTA circuit for testing."""
    config = VTAConfig(
        tonic_da_level=0.5,  # Baseline
        rpe_to_da_gain=0.5
    )
    return VTACircuit(config)


@pytest.fixture
def bridge(stdp_learner, vta_circuit):
    """Create STDP-VTA bridge for testing."""
    reset_stdp_vta_bridge()
    return STDPVTABridge(stdp_learner, vta_circuit)


class TestDopamineModulation:
    """Test DA modulation of STDP learning rates."""

    def test_high_da_increases_ltp(self, bridge, stdp_learner):
        """Test high DA (reward) increases LTP amplitude."""
        # Get baseline rates
        a_plus_base = stdp_learner.config.a_plus
        a_minus_base = stdp_learner.config.a_minus

        # High DA (reward signal)
        high_da = 0.9
        a_plus_mod, a_minus_mod = bridge.get_da_modulated_rates(da_level=high_da)

        # LTP should increase
        assert a_plus_mod > a_plus_base, "High DA should increase LTP"

        # LTD should decrease
        assert a_minus_mod < a_minus_base, "High DA should decrease LTD"

        # Verify ratios are reasonable
        ltp_increase = a_plus_mod / a_plus_base
        ltd_decrease = a_minus_mod / a_minus_base
        assert 1.0 < ltp_increase < 2.0, "LTP increase should be moderate"
        assert 0.5 < ltd_decrease < 1.0, "LTD decrease should be moderate"

    def test_low_da_increases_ltd(self, bridge, stdp_learner):
        """Test low DA (punishment) increases LTD amplitude."""
        # Get baseline rates
        a_plus_base = stdp_learner.config.a_plus
        a_minus_base = stdp_learner.config.a_minus

        # Low DA (punishment signal)
        low_da = 0.1
        a_plus_mod, a_minus_mod = bridge.get_da_modulated_rates(da_level=low_da)

        # LTP should decrease
        assert a_plus_mod < a_plus_base, "Low DA should decrease LTP"

        # LTD should increase
        assert a_minus_mod > a_minus_base, "Low DA should increase LTD"

        # Verify ratios are reasonable
        ltp_decrease = a_plus_mod / a_plus_base
        ltd_increase = a_minus_mod / a_minus_base
        assert 0.1 < ltp_decrease < 1.0, "LTP decrease should be moderate"
        assert 1.0 < ltd_increase < 2.0, "LTD increase should be moderate"

    def test_baseline_da_no_modulation(self, bridge, stdp_learner):
        """Test baseline DA produces minimal modulation."""
        # Get baseline rates
        a_plus_base = stdp_learner.config.a_plus
        a_minus_base = stdp_learner.config.a_minus

        # Baseline DA (neutral)
        baseline_da = 0.5
        a_plus_mod, a_minus_mod = bridge.get_da_modulated_rates(da_level=baseline_da)

        # Should be very close to baseline (within 1% due to threshold)
        assert abs(a_plus_mod - a_plus_base) < a_plus_base * 0.01
        assert abs(a_minus_mod - a_minus_base) < a_minus_base * 0.01

    def test_da_modulation_gradual(self, bridge):
        """Test DA modulation is smooth and gradual."""
        # Test a range of DA levels
        da_levels = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        a_plus_values = []
        a_minus_values = []

        for da in da_levels:
            a_plus, a_minus = bridge.get_da_modulated_rates(da_level=da)
            a_plus_values.append(a_plus)
            a_minus_values.append(a_minus)

        # LTP should increase monotonically with DA
        for i in range(len(da_levels) - 1):
            assert a_plus_values[i] <= a_plus_values[i + 1], \
                f"LTP should increase with DA: {da_levels[i]} -> {da_levels[i+1]}"

        # LTD should decrease monotonically with DA
        for i in range(len(da_levels) - 1):
            assert a_minus_values[i] >= a_minus_values[i + 1], \
                f"LTD should decrease with DA: {da_levels[i]} -> {da_levels[i+1]}"


class TestSTDPVTAIntegration:
    """Test integration of STDP with VTA circuit."""

    def test_vta_connection(self, stdp_learner):
        """Test connecting VTA circuit to bridge."""
        bridge = STDPVTABridge(stdp_learner)

        # Initially no VTA
        assert bridge._vta is None

        # Connect VTA
        vta = VTACircuit()
        bridge.set_vta_circuit(vta)

        assert bridge._vta is vta
        assert bridge.get_current_da() > 0  # Should get DA from VTA

    def test_da_from_vta_reward(self, bridge, vta_circuit):
        """Test DA level changes based on VTA reward signals."""
        # Process positive RPE (reward)
        vta_circuit.process_rpe(rpe=0.5, dt=0.1)
        da_level = bridge.get_current_da()

        # DA should be elevated above baseline
        assert da_level > vta_circuit.config.tonic_da_level

        # Modulated rates should reflect reward
        a_plus, a_minus = bridge.get_da_modulated_rates()
        assert a_plus > bridge.stdp.config.a_plus  # Enhanced LTP
        assert a_minus < bridge.stdp.config.a_minus  # Reduced LTD

    def test_da_from_vta_punishment(self, bridge, vta_circuit):
        """Test DA level changes based on VTA punishment signals."""
        # Process negative RPE (punishment)
        vta_circuit.process_rpe(rpe=-0.5, dt=0.1)
        da_level = bridge.get_current_da()

        # DA should be reduced below baseline
        assert da_level < vta_circuit.config.tonic_da_level

        # Modulated rates should reflect punishment
        a_plus, a_minus = bridge.get_da_modulated_rates()
        assert a_plus < bridge.stdp.config.a_plus  # Reduced LTP
        assert a_minus > bridge.stdp.config.a_minus  # Enhanced LTD

    def test_stdp_with_vta_modulation(self, bridge, stdp_learner, vta_circuit):
        """Test STDP weight changes modulated by VTA DA."""
        # Setup: Record spikes with 10ms positive timing (LTP condition)
        now = datetime.now()
        stdp_learner.record_spike("pre", timestamp=now, strength=1.0)
        stdp_learner.record_spike("post", timestamp=now + timedelta(milliseconds=10), strength=1.0)

        # Baseline: Compute STDP without DA modulation
        baseline_update = stdp_learner.compute_update("pre", "post", current_weight=0.5)
        assert baseline_update is not None
        baseline_delta = baseline_update.delta_weight

        # Reset for next test
        stdp_learner.clear_spikes()

        # Test 1: With reward (high DA)
        stdp_learner.record_spike("pre", timestamp=now, strength=1.0)
        stdp_learner.record_spike("post", timestamp=now + timedelta(milliseconds=10), strength=1.0)
        vta_circuit.process_rpe(rpe=0.5, dt=0.1)  # Positive RPE
        da_level = bridge.get_current_da()

        reward_update = stdp_learner.compute_update(
            "pre", "post", current_weight=0.5, da_level=da_level
        )
        assert reward_update is not None
        reward_delta = reward_update.delta_weight

        # Reward should enhance LTP
        assert reward_delta > baseline_delta, "Reward should increase LTP magnitude"

        # Reset for next test
        stdp_learner.clear_spikes()
        vta_circuit.reset()

        # Test 2: With punishment (low DA)
        stdp_learner.record_spike("pre", timestamp=now, strength=1.0)
        stdp_learner.record_spike("post", timestamp=now + timedelta(milliseconds=10), strength=1.0)
        vta_circuit.process_rpe(rpe=-0.5, dt=0.1)  # Negative RPE
        da_level = bridge.get_current_da()

        punishment_update = stdp_learner.compute_update(
            "pre", "post", current_weight=0.5, da_level=da_level
        )
        assert punishment_update is not None
        punishment_delta = punishment_update.delta_weight

        # Punishment should reduce LTP
        assert punishment_delta < baseline_delta, "Punishment should decrease LTP magnitude"


class TestLearningGating:
    """Test DA-based learning gating."""

    def test_learning_gated_at_low_da(self, bridge):
        """Test learning is blocked when DA too low."""
        config = STDPVTAConfig(
            enable_gating=True,
            min_da_for_learning=0.2
        )
        bridge.config = config

        # Very low DA should gate learning
        assert not bridge.should_gate_learning(da_level=0.1)

        # Sufficient DA should allow learning
        assert bridge.should_gate_learning(da_level=0.3)

    def test_gating_disabled(self, bridge):
        """Test gating can be disabled."""
        config = STDPVTAConfig(
            enable_gating=False,
            min_da_for_learning=0.5
        )
        bridge.config = config

        # Even low DA should allow learning if gating disabled
        assert bridge.should_gate_learning(da_level=0.1)

    def test_gated_stdp_returns_zero(self, bridge):
        """Test gated STDP returns zero weight change."""
        config = STDPVTAConfig(
            enable_gating=True,
            min_da_for_learning=0.5
        )
        bridge.config = config

        # Very low DA
        delta = bridge.compute_modulated_stdp_delta(
            delta_t_ms=10.0,
            current_weight=0.5,
            da_level=0.1  # Below threshold
        )

        assert delta == 0.0, "Gated learning should return zero"

        # Sufficient DA
        delta = bridge.compute_modulated_stdp_delta(
            delta_t_ms=10.0,
            current_weight=0.5,
            da_level=0.6  # Above threshold
        )

        assert delta != 0.0, "Ungated learning should return nonzero"


class TestBridgeStatistics:
    """Test bridge statistics and monitoring."""

    def test_bridge_stats(self, bridge, vta_circuit):
        """Test bridge statistics reporting."""
        # Process some RPE
        vta_circuit.process_rpe(rpe=0.3, dt=0.1)

        stats = bridge.get_stats()

        # Check required fields
        assert "da_level" in stats
        assert "ltp_modulation" in stats
        assert "ltd_modulation" in stats
        assert "a_plus_modulated" in stats
        assert "a_minus_modulated" in stats
        assert "a_plus_base" in stats
        assert "a_minus_base" in stats
        assert "learning_gated" in stats
        assert "vta_connected" in stats

        # Verify VTA connection
        assert stats["vta_connected"] is True

        # DA should be elevated
        assert stats["da_level"] > 0.5

    def test_stats_without_vta(self, stdp_learner):
        """Test stats when VTA not connected."""
        bridge = STDPVTABridge(stdp_learner)

        stats = bridge.get_stats()

        assert stats["vta_connected"] is False
        assert stats["da_level"] == bridge.config.baseline_da  # Should use baseline


class TestModulationMechanics:
    """Test detailed mechanics of DA modulation."""

    def test_ltp_enhancement_mechanism(self, bridge):
        """Test LTP enhancement mechanism in detail."""
        # High DA should use formula: A+ * (1 + gain * da_mod)
        high_da = 0.8
        ltp_mod, ltd_mod = bridge.compute_da_modulation(da_level=high_da)

        # DA mod = (0.8 - 0.5) / 0.5 = 0.6
        # LTP mod = 1 + 0.5 * 0.6 = 1.3
        expected_ltp_mod = 1.0 + bridge.config.ltp_da_gain * (
            (high_da - bridge.config.baseline_da) / bridge.config.baseline_da
        )

        assert abs(ltp_mod - expected_ltp_mod) < 0.01

    def test_ltd_suppression_mechanism(self, bridge):
        """Test LTD suppression mechanism in detail."""
        # High DA should use formula: A- * (1 - gain * da_mod)
        high_da = 0.8
        ltp_mod, ltd_mod = bridge.compute_da_modulation(da_level=high_da)

        # DA mod = (0.8 - 0.5) / 0.5 = 0.6
        # LTD mod = 1 - 0.3 * 0.6 = 0.82
        expected_ltd_mod = 1.0 - bridge.config.ltd_da_gain * (
            (high_da - bridge.config.baseline_da) / bridge.config.baseline_da
        )

        assert abs(ltd_mod - expected_ltd_mod) < 0.01

    def test_minimum_modulation_bounds(self, bridge):
        """Test modulation doesn't completely suppress learning."""
        # Even at extreme DA levels, modulation shouldn't go to zero
        for da in [0.0, 1.0]:
            ltp_mod, ltd_mod = bridge.compute_da_modulation(da_level=da)
            assert ltp_mod >= 0.1, "LTP shouldn't be completely suppressed"
            assert ltd_mod >= 0.1, "LTD shouldn't be completely suppressed"

    def test_da_threshold_filtering(self, bridge):
        """Test DA threshold filters small fluctuations."""
        config = STDPVTAConfig(
            baseline_da=0.5,
            da_threshold=0.1  # 10% threshold
        )
        bridge.config = config

        # Small DA change (within threshold)
        small_da = 0.51
        ltp_mod, ltd_mod = bridge.compute_da_modulation(da_level=small_da)

        # Should be close to 1.0 (no modulation)
        assert abs(ltp_mod - 1.0) < 0.05
        assert abs(ltd_mod - 1.0) < 0.05


class TestGlobalBridge:
    """Test global bridge singleton."""

    def test_get_global_bridge(self, stdp_learner, vta_circuit):
        """Test getting global bridge instance."""
        reset_stdp_vta_bridge()

        bridge = get_stdp_vta_bridge(stdp_learner, vta_circuit)
        assert bridge is not None
        assert bridge.stdp is stdp_learner
        assert bridge._vta is vta_circuit

        # Should return same instance
        bridge2 = get_stdp_vta_bridge()
        assert bridge2 is bridge

    def test_reset_global_bridge(self):
        """Test resetting global bridge."""
        bridge1 = get_stdp_vta_bridge()
        reset_stdp_vta_bridge()
        bridge2 = get_stdp_vta_bridge()

        # Should be different instances
        assert bridge1 is not bridge2


class TestBiologicalConsistency:
    """Test biological consistency of DA modulation."""

    def test_reward_prediction_error_pattern(self, bridge, vta_circuit, stdp_learner):
        """Test DA follows RPE pattern (Schultz)."""
        now = datetime.now()

        # Scenario 1: Unexpected reward (positive RPE)
        vta_circuit.process_rpe(rpe=0.8, dt=0.1)
        stdp_learner.record_spike("pre", timestamp=now, strength=1.0)
        stdp_learner.record_spike("post", timestamp=now + timedelta(milliseconds=10), strength=1.0)

        da_level = bridge.get_current_da()
        update = stdp_learner.compute_update("pre", "post", current_weight=0.5, da_level=da_level)

        # Should strongly potentiate (reward learning)
        assert update.update_type == "ltp"
        assert update.delta_weight > 0

        # Reset
        stdp_learner.clear_spikes()
        vta_circuit.reset()

        # Scenario 2: Expected reward (zero RPE)
        vta_circuit.process_rpe(rpe=0.0, dt=0.1)
        stdp_learner.record_spike("pre", timestamp=now, strength=1.0)
        stdp_learner.record_spike("post", timestamp=now + timedelta(milliseconds=10), strength=1.0)

        da_level = bridge.get_current_da()
        update_baseline = stdp_learner.compute_update("pre", "post", current_weight=0.5, da_level=da_level)

        # Should be less than unexpected reward
        assert update_baseline.delta_weight < update.delta_weight

        # Reset
        stdp_learner.clear_spikes()
        vta_circuit.reset()

        # Scenario 3: Omitted reward (negative RPE)
        vta_circuit.process_rpe(rpe=-0.8, dt=0.1)
        stdp_learner.record_spike("pre", timestamp=now, strength=1.0)
        stdp_learner.record_spike("post", timestamp=now + timedelta(milliseconds=10), strength=1.0)

        da_level = bridge.get_current_da()
        update_omit = stdp_learner.compute_update("pre", "post", current_weight=0.5, da_level=da_level)

        # Should be weakly potentiated or even depressed
        assert update_omit.delta_weight < update_baseline.delta_weight

    def test_d1_d2_receptor_analog(self, bridge):
        """Test modulation analogous to D1/D2 receptor dynamics."""
        # D1 receptors: enhance LTP (cAMP/PKA pathway)
        # D2 receptors: suppress LTD (reduced Ca2+)

        high_da = 0.9
        a_plus, a_minus = bridge.get_da_modulated_rates(da_level=high_da)

        base_a_plus = bridge.stdp.config.a_plus
        base_a_minus = bridge.stdp.config.a_minus

        # D1-like: LTP enhancement should be stronger than LTD suppression
        ltp_change_pct = (a_plus - base_a_plus) / base_a_plus
        ltd_change_pct = (base_a_minus - a_minus) / base_a_minus

        # Both should be positive changes
        assert ltp_change_pct > 0
        assert ltd_change_pct > 0

        # LTP enhancement (D1) typically stronger than LTD suppression (D2)
        # This matches the gain parameters: ltp_gain=0.5, ltd_gain=0.3
        assert ltp_change_pct > ltd_change_pct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
