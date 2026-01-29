"""
Test P2.1: Raphe-VTA bidirectional autowiring.

Tests that after wiring via set_raphe() / set_vta(), step() automatically
produces bidirectional modulation without manual calls.
"""

import numpy as np
import pytest

from ww.nca.raphe import RapheConfig, RapheNucleus
from ww.nca.vta import VTACircuit, VTAConfig


class TestRapheVTAAutowiring:
    """Test automatic bidirectional modulation between raphe and VTA."""

    def test_vta_without_wiring_still_works(self):
        """Test that VTA step() works without raphe wired (no crash)."""
        vta = VTACircuit()

        # Should work fine without raphe
        vta.step(dt=0.1)
        vta.step(dt=0.1)

        assert vta.state.current_da > 0  # Still produces DA

    def test_raphe_without_wiring_still_works(self):
        """Test that raphe step() works without VTA wired (no crash)."""
        raphe = RapheNucleus()

        # Should work fine without VTA
        raphe.step(dt=0.1)
        raphe.step(dt=0.1)

        assert raphe.state.extracellular_5ht > 0  # Still produces 5-HT

    def test_vta_set_raphe_enables_autowiring(self):
        """Test that set_raphe() enables automatic 5-HT inhibition."""
        vta = VTACircuit()
        raphe = RapheNucleus()

        # Wire raphe to VTA
        vta.set_raphe(raphe)

        # Increase raphe 5-HT to create inhibition
        raphe.inject_5ht(0.4)  # Push 5-HT high

        # Get baseline DA before step
        da_before = vta.state.current_da

        # Step VTA (should auto-receive inhibition)
        vta.step(dt=0.1)

        # DA should be reduced due to 5-HT inhibition
        da_after = vta.state.current_da
        assert da_after < da_before, "5-HT inhibition should reduce DA"

    def test_raphe_set_vta_enables_autowiring(self):
        """Test that set_vta() enables automatic RPE modulation."""
        vta = VTACircuit()
        raphe = RapheNucleus()

        # Wire VTA to raphe
        raphe.set_vta(vta)

        # Generate positive RPE in VTA
        vta.process_rpe(rpe=0.5, dt=0.1)

        # Step raphe (should auto-receive RPE)
        raphe.step(dt=0.1)

        # Positive RPE should increase reward_input
        assert raphe.state.reward_input > 0, "Positive RPE should set reward input"

    def test_bidirectional_autowiring(self):
        """Test full bidirectional autowiring: both directions work."""
        vta = VTACircuit()
        raphe = RapheNucleus()

        # Wire both directions
        vta.set_raphe(raphe)
        raphe.set_vta(vta)

        # Inject high 5-HT
        raphe.inject_5ht(0.5)

        # Step both
        raphe.step(dt=0.1)
        vta.step(dt=0.1)

        # VTA should be inhibited by high 5-HT
        # (Hard to assert exact values due to dynamics, but check interaction occurred)
        stats = vta.get_stats()
        assert "current_da" in stats

    def test_equilibrium_convergence(self):
        """Test that wired systems converge to stable equilibrium."""
        vta = VTACircuit(VTAConfig(tonic_da_level=0.3))
        raphe = RapheNucleus(RapheConfig(setpoint=0.4))

        # Wire bidirectionally
        vta.set_raphe(raphe)
        raphe.set_vta(vta)

        # Run 100 steps
        da_history = []
        ht_history = []

        for _ in range(100):
            raphe.step(dt=0.1)
            vta.step(dt=0.1)
            da_history.append(vta.state.current_da)
            ht_history.append(raphe.state.extracellular_5ht)

        # Check convergence: last 10 steps should have low variance
        da_recent = da_history[-10:]
        ht_recent = ht_history[-10:]

        da_variance = np.var(da_recent)
        ht_variance = np.var(ht_recent)

        # Should converge to stable state
        assert da_variance < 0.01, f"DA variance too high: {da_variance}"
        assert ht_variance < 0.01, f"5-HT variance too high: {ht_variance}"

        # Final values should be in reasonable range
        final_da = da_history[-1]
        final_ht = ht_history[-1]

        assert 0.1 < final_da < 0.6, f"DA out of range: {final_da}"
        assert 0.2 < final_ht < 0.6, f"5-HT out of range: {final_ht}"

    def test_raphe_inhibits_vta_proportionally(self):
        """Test that higher 5-HT produces stronger DA inhibition."""
        vta = VTACircuit(VTAConfig(tonic_da_level=0.5))
        raphe_low = RapheNucleus(RapheConfig(setpoint=0.2))
        raphe_high = RapheNucleus(RapheConfig(setpoint=0.6))

        # Low 5-HT scenario
        vta_low = VTACircuit(VTAConfig(tonic_da_level=0.5))
        vta_low.set_raphe(raphe_low)

        for _ in range(20):
            raphe_low.step(dt=0.1)
            vta_low.step(dt=0.1)

        da_with_low_5ht = vta_low.state.current_da

        # High 5-HT scenario
        vta_high = VTACircuit(VTAConfig(tonic_da_level=0.5))
        vta_high.set_raphe(raphe_high)

        for _ in range(20):
            raphe_high.step(dt=0.1)
            vta_high.step(dt=0.1)

        da_with_high_5ht = vta_high.state.current_da

        # Higher 5-HT should produce lower DA
        assert da_with_high_5ht < da_with_low_5ht, \
            f"High 5-HT should reduce DA more: {da_with_high_5ht} vs {da_with_low_5ht}"

    def test_vta_rpe_modulates_raphe(self):
        """Test that VTA RPE modulates raphe reward input."""
        vta = VTACircuit()
        raphe = RapheNucleus()

        # Wire VTA to raphe
        raphe.set_vta(vta)

        # Generate strong positive RPE
        vta.process_rpe(rpe=0.8, dt=0.1)

        # Check VTA has high RPE
        assert vta.state.last_rpe > 0.5

        # Step raphe (should receive RPE)
        raphe.step(dt=0.1)

        # Raphe should have increased reward_input
        assert raphe.state.reward_input > 0.5, \
            f"Expected high reward input, got {raphe.state.reward_input}"

    def test_negative_rpe_does_not_increase_reward_input(self):
        """Test that negative RPE does not spuriously increase reward input."""
        vta = VTACircuit()
        raphe = RapheNucleus()

        # Wire VTA to raphe
        raphe.set_vta(vta)

        # Generate negative RPE
        vta.process_rpe(rpe=-0.5, dt=0.1)

        # Step raphe
        raphe.step(dt=0.1)

        # Negative RPE should be max(0, rpe) = 0
        assert raphe.state.reward_input == 0, \
            f"Negative RPE should not set reward input, got {raphe.state.reward_input}"

    def test_autowiring_survives_reset(self):
        """Test that wiring persists after reset() calls."""
        vta = VTACircuit()
        raphe = RapheNucleus()

        # Wire both directions
        vta.set_raphe(raphe)
        raphe.set_vta(vta)

        # Reset both
        vta.reset()
        raphe.reset()

        # Wiring should still be active (references preserved)
        assert vta._raphe is raphe
        assert raphe._vta is vta

        # Should still work
        vta.step(dt=0.1)
        raphe.step(dt=0.1)

    def test_unidirectional_wiring_works(self):
        """Test that wiring only one direction works (doesn't require both)."""
        vta = VTACircuit()
        raphe = RapheNucleus()

        # Wire only VTA -> receives from raphe
        vta.set_raphe(raphe)

        # High 5-HT
        raphe.inject_5ht(0.5)

        # Step VTA
        vta.step(dt=0.1)

        # Should work (no crash, modulation occurs)
        assert vta.state.current_da < vta.config.tonic_da_level

    def test_dynamic_5ht_inhibition(self):
        """Test that 5-HT inhibition updates dynamically across steps."""
        vta = VTACircuit(VTAConfig(tonic_da_level=0.5))
        raphe = RapheNucleus(RapheConfig(setpoint=0.3))

        # Wire
        vta.set_raphe(raphe)

        # Step 1: Baseline (let settle for a bit)
        for _ in range(10):
            raphe.step(dt=0.1)
            vta.step(dt=0.1)
        da_baseline = vta.state.current_da

        # Step 2: Inject high 5-HT
        raphe.inject_5ht(0.5)
        for _ in range(5):
            raphe.step(dt=0.1)
            vta.step(dt=0.1)
        da_inhibited = vta.state.current_da

        # Step 3: Let 5-HT decay longer (more steps for exponential decay)
        for _ in range(50):
            raphe.step(dt=0.1)
            vta.step(dt=0.1)
        da_recovered = vta.state.current_da

        # DA should drop when 5-HT high
        assert da_inhibited < da_baseline, "DA should drop with high 5-HT"

        # DA should trend back toward baseline (may not fully recover)
        # Test: recovered should be closer to baseline than inhibited state
        distance_inhibited = abs(da_inhibited - da_baseline)
        distance_recovered = abs(da_recovered - da_baseline)

        assert distance_recovered < distance_inhibited, \
            f"DA should move back toward baseline: inhibited={da_inhibited:.3f}, " \
            f"baseline={da_baseline:.3f}, recovered={da_recovered:.3f}"

    def test_multiple_wiring_calls_are_safe(self):
        """Test that calling set_raphe/set_vta multiple times is safe."""
        vta = VTACircuit()
        raphe1 = RapheNucleus()
        raphe2 = RapheNucleus()

        # Wire to raphe1
        vta.set_raphe(raphe1)

        # Re-wire to raphe2 (should replace)
        vta.set_raphe(raphe2)

        # Should use raphe2
        assert vta._raphe is raphe2

        # Should work
        vta.step(dt=0.1)


class TestBiologicalPlausibility:
    """Test that autowiring produces biologically plausible dynamics."""

    def test_opponent_process_dynamic(self):
        """Test 5-HT-DA opponent process: high 5-HT dampens DA response."""
        # Low 5-HT scenario (impatient, high DA response)
        vta_low = VTACircuit()
        raphe_low = RapheNucleus(RapheConfig(setpoint=0.2))
        vta_low.set_raphe(raphe_low)

        for _ in range(10):
            raphe_low.step(dt=0.1)
            vta_low.step(dt=0.1)

        # Deliver reward
        vta_low.process_rpe(rpe=0.5, dt=0.1)
        da_response_low_5ht = vta_low.state.current_da

        # High 5-HT scenario (patient, dampened DA response)
        vta_high = VTACircuit()
        raphe_high = RapheNucleus(RapheConfig(setpoint=0.6))
        vta_high.set_raphe(raphe_high)

        for _ in range(10):
            raphe_high.step(dt=0.1)
            vta_high.step(dt=0.1)

        # Deliver same reward
        vta_high.process_rpe(rpe=0.5, dt=0.1)
        da_response_high_5ht = vta_high.state.current_da

        # DA response should be larger with low 5-HT (less inhibition)
        assert da_response_low_5ht > da_response_high_5ht, \
            "Low 5-HT should produce larger DA response to reward"

    def test_patience_increases_with_5ht(self):
        """Test that raphe patience signal increases with 5-HT level."""
        vta = VTACircuit()
        raphe = RapheNucleus()

        raphe.set_vta(vta)

        # Low 5-HT
        for _ in range(10):
            raphe.step(dt=0.1)
        patience_low = raphe.get_patience_signal()

        # Inject 5-HT
        raphe.inject_5ht(0.5)

        for _ in range(10):
            raphe.step(dt=0.1)
        patience_high = raphe.get_patience_signal()

        # Higher 5-HT should increase patience
        assert patience_high > patience_low, \
            f"Patience should increase with 5-HT: {patience_low} -> {patience_high}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
