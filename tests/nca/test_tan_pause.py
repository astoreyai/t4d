"""
Test TAN pause mechanism implementation (Fix 2).

Verifies cholinergic interneuron pause responses following Aosaki et al. 1994.
"""

import numpy as np
import pytest

from ww.nca.striatal_msn import (
    CholinergicInterneuron,
    MSNConfig,
    StriatalMSN,
    TANState,
)


class TestTANPauseMechanism:
    """Test TAN (cholinergic interneuron) pause responses."""

    def test_tan_pause_triggers_on_surprise(self):
        """Verify TAN pauses when RPE exceeds threshold."""
        config = MSNConfig(
            tan_pause_threshold=0.3,
            tan_pause_ms=200.0,  # ATOM-P4-2: Use tan_pause_ms instead of tan_pause_duration
            tan_baseline_ach=0.5,
            tan_pause_ach=0.1,
        )
        tan = CholinergicInterneuron(config)

        # Baseline: should be active
        assert tan.state.state == TANState.ACTIVE
        assert tan.state.ach_level == config.tan_baseline_ach

        # Small RPE: should not pause
        tan.process_reward_surprise(rpe=0.2, dt=0.01)
        assert tan.state.state == TANState.ACTIVE
        assert tan.state.ach_level == config.tan_baseline_ach

        # Large RPE: should pause
        tan.process_reward_surprise(rpe=0.5, dt=0.01)
        assert tan.state.state == TANState.PAUSED
        assert tan.state.ach_level == config.tan_pause_ach
        assert tan.state.pause_remaining > 0

    def test_pause_duration_200ms(self):
        """Verify pause lasts ~200ms (Aosaki et al. 1994)."""
        config = MSNConfig(
            tan_pause_threshold=0.3,
            tan_pause_ms=200.0,  # ATOM-P4-2: Use tan_pause_ms instead of tan_pause_duration
        )
        tan = CholinergicInterneuron(config)

        # Trigger pause
        tan.process_reward_surprise(rpe=0.5, dt=0.01)
        assert tan.is_paused()

        # Step through time
        dt = 0.01  # 10ms timesteps
        pause_start_time = 0.0
        time = pause_start_time

        while tan.is_paused() and time < 0.5:  # Max 500ms
            tan.step(dt=dt)
            time += dt

        pause_duration = time - pause_start_time

        # Should be close to 200ms
        assert abs(pause_duration - 0.2) < 0.02, (
            f"Pause duration {pause_duration:.3f}s should be ~0.2s"
        )

        # Should have resumed
        assert tan.state.state == TANState.ACTIVE
        assert tan.state.ach_level == config.tan_baseline_ach

    def test_ach_drops_during_pause(self):
        """Verify ACh drops during pause and recovers after."""
        config = MSNConfig(
            tan_baseline_ach=0.5,
            tan_pause_ach=0.1,
        )
        tan = CholinergicInterneuron(config)

        baseline_ach = tan.get_ach_level()
        assert baseline_ach == 0.5

        # Trigger pause
        tan.process_reward_surprise(rpe=0.5, dt=0.01)

        # ACh should drop immediately
        assert tan.get_ach_level() == config.tan_pause_ach
        assert tan.get_ach_level() < baseline_ach

        # Wait for pause to end
        dt = 0.01
        for _ in range(30):  # 300ms
            tan.step(dt=dt)

        # ACh should recover
        assert tan.get_ach_level() == baseline_ach

    def test_negative_rpe_also_triggers_pause(self):
        """Verify both positive and negative surprise trigger pause."""
        config = MSNConfig(tan_pause_threshold=0.3)
        tan = CholinergicInterneuron(config)

        # Positive surprise
        tan.process_reward_surprise(rpe=0.5, dt=0.01)
        assert tan.is_paused()

        # Reset
        tan.reset()
        assert not tan.is_paused()

        # Negative surprise
        tan.process_reward_surprise(rpe=-0.5, dt=0.01)
        assert tan.is_paused()

    def test_no_double_trigger(self):
        """Verify pause cannot be re-triggered while already paused."""
        config = MSNConfig(tan_pause_ms=200.0)  # ATOM-P4-2: Use tan_pause_ms
        tan = CholinergicInterneuron(config)

        # Trigger pause
        tan.process_reward_surprise(rpe=0.5, dt=0.01)
        initial_remaining = tan.state.pause_remaining

        # Try to trigger again immediately
        tan.process_reward_surprise(rpe=0.7, dt=0.01)

        # Pause should not reset
        assert tan.state.pause_remaining < initial_remaining, (
            "Pause duration should continue decreasing, not reset"
        )

    def test_msn_integration(self):
        """Verify TAN is integrated into StriatalMSN."""
        msn = StriatalMSN()

        # Should have TAN
        assert hasattr(msn, "tan")
        assert isinstance(msn.tan, CholinergicInterneuron)

        # Baseline ACh
        assert msn.state.ach_level == msn.config.tan_baseline_ach

        # Apply RPE (should trigger TAN pause)
        msn.apply_rpe(rpe=0.5, dt=0.01)

        # Step to sync ACh level
        msn.step(dt=0.01)

        # TAN should be paused
        assert msn.tan.is_paused()
        assert msn.state.ach_level == msn.config.tan_pause_ach

    def test_tan_pause_enhances_d1_plasticity(self):
        """Verify TAN pause (low ACh) enhances D1 pathway."""
        msn = StriatalMSN()

        # Set cortical input and DA
        msn.set_cortical_input(0.5)
        msn.set_dopamine_level(0.6)

        # Baseline D1 activity (with tonic ACh)
        msn.step(dt=0.01)
        baseline_d1 = msn.state.d1_activity

        # Trigger TAN pause
        msn.apply_rpe(rpe=0.5, dt=0.01)
        assert msn.tan.is_paused()

        # Step again (with paused ACh)
        msn.step(dt=0.01)
        paused_d1 = msn.state.d1_activity

        # D1 should be enhanced during pause (Fix 2)
        # Note: Enhancement is modest (~30% max), so check for increase
        assert paused_d1 > baseline_d1, (
            f"D1 activity during TAN pause ({paused_d1:.3f}) should exceed "
            f"baseline ({baseline_d1:.3f})"
        )

    def test_tan_statistics(self):
        """Verify TAN statistics tracking."""
        tan = CholinergicInterneuron()

        # Trigger multiple pauses
        for rpe in [0.5, 0.6, -0.4, 0.7]:
            tan.process_reward_surprise(rpe=rpe, dt=0.01)
            # Wait for pause to complete
            for _ in range(30):
                tan.step(dt=0.01)

        stats = tan.get_stats()

        assert "state" in stats
        assert "ach_level" in stats
        assert "n_pauses" in stats
        assert "avg_pause_trigger" in stats

        assert stats["n_pauses"] == 4
        assert stats["avg_pause_trigger"] > 0

    def test_temporal_credit_assignment(self):
        """Verify TAN pause marks 'when' reinforcement occurred."""
        msn = StriatalMSN()
        msn.set_cortical_input(0.5)
        msn.set_dopamine_level(0.6)

        # Record ACh trace
        times = []
        ach_levels = []

        # Baseline period
        dt = 0.01
        for i in range(20):  # 200ms
            msn.step(dt=dt)
            times.append(i * dt)
            ach_levels.append(msn.state.ach_level)

        # Reward event at t=0.2s
        reward_time = len(times) * dt
        msn.apply_rpe(rpe=0.5, dt=dt)

        # Continue recording (step syncs ACh)
        for i in range(20, 50):  # Until 500ms
            msn.step(dt=dt)
            times.append(i * dt)
            ach_levels.append(msn.state.ach_level)

        times = np.array(times)
        ach_levels = np.array(ach_levels)

        # Find pause onset (ACh drop)
        # Look for where ACh drops below threshold
        pause_ach = msn.config.tan_pause_ach
        baseline_ach = msn.config.tan_baseline_ach

        # Find first point where ACh is at pause level
        pause_indices = np.where(ach_levels < baseline_ach * 0.5)[0]

        if len(pause_indices) == 0:
            pytest.fail("No TAN pause detected in ACh trace")

        pause_onset_idx = pause_indices[0]
        pause_onset_time = times[pause_onset_idx]

        # Pause should occur near reward time (within one timestep)
        assert abs(pause_onset_time - reward_time) < 0.03, (
            f"TAN pause onset ({pause_onset_time:.3f}s) should occur near "
            f"reward time ({reward_time:.3f}s)"
        )

        # Find pause end (ACh recovery)
        recovery_indices = np.where(
            (times > pause_onset_time) &
            (ach_levels > baseline_ach * 0.9)
        )[0]

        if len(recovery_indices) > 0:
            pause_end_idx = recovery_indices[0]
            pause_duration = times[pause_end_idx] - times[pause_onset_idx]

            assert abs(pause_duration - 0.2) < 0.05, (
                f"Pause duration ({pause_duration:.3f}s) should be ~0.2s"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
