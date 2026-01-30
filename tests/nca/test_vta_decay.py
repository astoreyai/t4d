"""
Test VTA exponential decay implementation (Fix 1).

Verifies that VTA DA decay follows exponential kinetics from Grace & Bunney 1984.
"""

import numpy as np
import pytest

from ww.nca.vta import VTACircuit, VTAConfig


class TestVTAExponentialDecay:
    """Test exponential decay of VTA dopamine levels."""

    def test_exponential_decay_curve(self):
        """Verify DA decays exponentially with tau_decay time constant."""
        # Create VTA with known parameters
        config = VTAConfig(
            tonic_da_level=0.3,
            tau_decay=0.2,  # 200ms time constant
            rpe_to_da_gain=0.5,
            burst_duration=0.0,  # No burst duration for clean decay test
        )
        vta = VTACircuit(config)

        # Boost DA level with positive RPE
        vta.process_rpe(rpe=0.5, dt=0.01)
        # Manually set to burst peak to start decay from known point
        vta.state.current_da = 0.55  # Known starting point
        vta.state.phasic_remaining = 0.0  # Force immediate decay
        initial_da = vta.state.current_da

        # Record decay trajectory
        times = []
        da_levels = []
        dt = 0.01  # 10ms timesteps

        for i in range(200):  # 2 seconds
            vta.step(dt=dt)
            times.append(i * dt)
            da_levels.append(vta.state.current_da)

        times = np.array(times)
        da_levels = np.array(da_levels)

        # Expected exponential decay: da(t) = target + (da0 - target) * exp(-t/tau)
        target = config.tonic_da_level
        tau = config.tau_decay
        expected = target + (initial_da - target) * np.exp(-times / tau)

        # Check close match to exponential (allow 10% error due to discrete timesteps)
        relative_error = np.abs((da_levels - expected) / expected)
        assert np.mean(relative_error) < 0.10, (
            f"Mean relative error {np.mean(relative_error):.3f} exceeds 10% "
            "- decay is not exponential"
        )

        # Verify decay reaches ~95% of target after 3*tau
        time_3tau = 3 * tau
        idx_3tau = int(time_3tau / dt)
        if idx_3tau < len(da_levels):
            da_at_3tau = da_levels[idx_3tau]
            expected_at_3tau = target + (initial_da - target) * np.exp(-3)
            assert abs(da_at_3tau - expected_at_3tau) < 0.05, (
                f"DA at 3*tau ({da_at_3tau:.3f}) should match "
                f"exponential prediction ({expected_at_3tau:.3f})"
            )

    def test_exponential_faster_than_linear(self):
        """Verify exponential decay has correct half-life."""
        config = VTAConfig(
            tonic_da_level=0.3,
            tau_decay=0.2,
            burst_duration=0.0,
        )
        vta = VTACircuit(config)

        # Set to known starting point
        vta.state.current_da = 0.55
        vta.state.phasic_remaining = 0.0
        initial_da = vta.state.current_da

        # Compute half-life for exponential: t_half = tau * ln(2)
        t_half = config.tau_decay * np.log(2)

        # Step to half-life
        dt = 0.01
        n_steps = int(t_half / dt)
        for _ in range(n_steps):
            vta.step(dt=dt)

        # DA should be approximately halfway between initial and target
        target = config.tonic_da_level
        expected_half = target + (initial_da - target) / 2
        actual = vta.state.current_da

        # Allow for discrete timestep error
        assert abs(actual - expected_half) < 0.02, (
            f"DA at half-life ({actual:.3f}) should be ~{expected_half:.3f}"
        )

    def test_decay_preserves_target(self):
        """Verify decay converges to tonic_da_level."""
        config = VTAConfig(
            tonic_da_level=0.3,
            tau_decay=0.2,
        )
        vta = VTACircuit(config)

        # Boost DA
        vta.process_rpe(rpe=0.5, dt=0.01)

        # Decay for 5*tau (should be >99% converged)
        dt = 0.01
        n_steps = int(5 * config.tau_decay / dt)
        for _ in range(n_steps):
            vta.step(dt=dt)

        # Should be very close to target
        assert abs(vta.state.current_da - config.tonic_da_level) < 0.01, (
            f"DA after 5*tau ({vta.state.current_da:.3f}) should converge "
            f"to target ({config.tonic_da_level:.3f})"
        )

    def test_decay_from_below(self):
        """Verify exponential decay also works when starting below target."""
        config = VTAConfig(
            tonic_da_level=0.3,
            tau_decay=0.2,
        )
        vta = VTACircuit(config)

        # Suppress DA with negative RPE
        vta.process_rpe(rpe=-0.5, dt=0.01)
        initial_da = vta.state.current_da
        assert initial_da < 0.3, "DA should decrease after negative RPE"

        # Decay should increase toward target
        dt = 0.01
        for _ in range(100):
            vta.step(dt=dt)

        # Should be moving toward target
        assert vta.state.current_da > initial_da, (
            "DA should increase toward target when starting below"
        )

    def test_tau_decay_parameter_effect(self):
        """Verify different tau values produce different decay rates."""
        # Fast decay (small tau)
        config_fast = VTAConfig(tonic_da_level=0.3, tau_decay=0.1, burst_duration=0.0)
        vta_fast = VTACircuit(config_fast)
        vta_fast.state.current_da = 0.55
        vta_fast.state.phasic_remaining = 0.0

        # Slow decay (large tau)
        config_slow = VTAConfig(tonic_da_level=0.3, tau_decay=0.4, burst_duration=0.0)
        vta_slow = VTACircuit(config_slow)
        vta_slow.state.current_da = 0.55
        vta_slow.state.phasic_remaining = 0.0

        # Step both
        dt = 0.01
        n_steps = 20  # 200ms
        for _ in range(n_steps):
            vta_fast.step(dt=dt)
            vta_slow.step(dt=dt)

        # Fast should be closer to target
        fast_distance = abs(vta_fast.state.current_da - 0.3)
        slow_distance = abs(vta_slow.state.current_da - 0.3)

        assert fast_distance < slow_distance, (
            f"Fast decay (tau=0.1) should be closer to target than slow (tau=0.4): "
            f"fast={fast_distance:.3f}, slow={slow_distance:.3f}"
        )

    def test_biological_timescale(self):
        """Verify tau_decay=0.2s produces biologically plausible dynamics."""
        config = VTAConfig(
            tonic_da_level=0.3,
            tau_decay=0.2,  # Grace & Bunney 1984: ~200ms
        )
        vta = VTACircuit(config)

        # Burst
        vta.process_rpe(rpe=0.5, dt=0.01)
        peak_da = vta.state.current_da

        # Measure time to decay to baseline + 10% of peak
        threshold = config.tonic_da_level + 0.1 * (peak_da - config.tonic_da_level)

        dt = 0.01
        time = 0.0
        max_time = 2.0  # 2 seconds max

        while vta.state.current_da > threshold and time < max_time:
            vta.step(dt=dt)
            time += dt

        # Should decay within 1 second (biologically plausible)
        assert time < 1.0, (
            f"Decay to 10% above baseline took {time:.3f}s, "
            "should be <1s for biological plausibility"
        )

        # Should not be instantaneous (>50ms)
        assert time > 0.05, (
            f"Decay to 10% above baseline took {time:.3f}s, "
            "should be >50ms for biological plausibility"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
