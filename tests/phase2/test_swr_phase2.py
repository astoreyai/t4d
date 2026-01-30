"""
Phase 2 Tests: SWR Timing and Wake-Sleep Separation.

Tests the Phase 2 enhancements to SWR coupling:
- Ripple frequency validation (150-250 Hz)
- Wake-sleep state separation
- State-dependent SWR gating
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, "src")

from ww.nca.swr_coupling import (
    SWRNeuralFieldCoupling,
    SWRConfig,
    WakeSleepMode,
    RIPPLE_FREQ_MIN,
    RIPPLE_FREQ_MAX,
    RIPPLE_FREQ_OPTIMAL,
    create_swr_coupling,
)


class TestRippleFrequencyConstants:
    """Test biological frequency constants."""

    def test_ripple_freq_min(self):
        """Minimum ripple frequency should be 150 Hz."""
        assert RIPPLE_FREQ_MIN == 150.0

    def test_ripple_freq_max(self):
        """Maximum ripple frequency should be 250 Hz."""
        assert RIPPLE_FREQ_MAX == 250.0

    def test_ripple_freq_optimal(self):
        """Optimal ripple frequency should be 180 Hz."""
        assert RIPPLE_FREQ_OPTIMAL == 180.0

    def test_frequency_range_valid(self):
        """Frequency range should be biologically plausible."""
        assert RIPPLE_FREQ_MIN < RIPPLE_FREQ_OPTIMAL < RIPPLE_FREQ_MAX


class TestWakeSleepMode:
    """Test wake/sleep state enumeration."""

    def test_all_modes_exist(self):
        """All expected wake/sleep modes should exist."""
        expected = ["active_wake", "quiet_wake", "nrem_light", "nrem_deep", "rem"]
        actual = [m.value for m in WakeSleepMode]
        assert set(expected) == set(actual)

    def test_mode_count(self):
        """Should have exactly 5 wake/sleep modes."""
        assert len(list(WakeSleepMode)) == 5


class TestSWRConfigValidation:
    """Test SWR configuration validation."""

    def test_default_frequency_valid(self):
        """Default ripple frequency should be valid."""
        config = SWRConfig()
        assert RIPPLE_FREQ_MIN <= config.ripple_frequency <= RIPPLE_FREQ_MAX

    def test_valid_frequency_accepted(self):
        """Valid ripple frequencies should be accepted."""
        config = SWRConfig(ripple_frequency=180.0)
        assert config.ripple_frequency == 180.0

    def test_invalid_frequency_low_rejected(self):
        """Frequency below minimum should raise error."""
        with pytest.raises(ValueError, match="Ripple frequency"):
            SWRConfig(ripple_frequency=100.0)

    def test_invalid_frequency_high_rejected(self):
        """Frequency above maximum should raise error."""
        with pytest.raises(ValueError, match="Ripple frequency"):
            SWRConfig(ripple_frequency=300.0)

    def test_state_swr_probability(self):
        """State-specific SWR probabilities should be valid."""
        config = SWRConfig()

        # Active wake should have lowest probability
        p_active = config.get_state_swr_probability(WakeSleepMode.ACTIVE_WAKE)
        assert 0 <= p_active <= 0.1

        # Deep NREM should have highest probability
        p_deep = config.get_state_swr_probability(WakeSleepMode.NREM_DEEP)
        assert p_deep >= 0.8


class TestSWRWakeSleepIntegration:
    """Test SWR-wake/sleep coupling."""

    def test_initial_state_quiet_wake(self):
        """Initial state should be quiet wake."""
        swr = create_swr_coupling()
        swr.step(dt=0.01)
        # Default should be quiet wake with moderate ACh/NE
        assert swr.state.wake_sleep_mode in [
            WakeSleepMode.QUIET_WAKE,
            WakeSleepMode.NREM_LIGHT,
        ]

    def test_set_wake_sleep_mode(self):
        """Manual wake/sleep mode setting should work."""
        swr = create_swr_coupling(enable_state_gating=True)

        swr.set_wake_sleep_mode(WakeSleepMode.NREM_DEEP)
        assert swr.state.wake_sleep_mode == WakeSleepMode.NREM_DEEP

        swr.set_wake_sleep_mode(WakeSleepMode.ACTIVE_WAKE)
        assert swr.state.wake_sleep_mode == WakeSleepMode.ACTIVE_WAKE

    def test_validate_ripple_frequency(self):
        """Ripple frequency validation method should work."""
        swr = create_swr_coupling()

        assert swr.validate_ripple_frequency(180.0) is True
        assert swr.validate_ripple_frequency(100.0) is False
        assert swr.validate_ripple_frequency(300.0) is False

    def test_get_swr_probability(self):
        """SWR probability should vary with wake/sleep state."""
        swr = create_swr_coupling(enable_state_gating=True)

        # Set to active wake (low probability, may be 0)
        swr.set_wake_sleep_mode(WakeSleepMode.ACTIVE_WAKE)
        # Active wake has high ACh/NE
        swr.state.ach_level = 0.8
        swr.state.ne_level = 0.7
        p_active = swr.get_swr_probability()

        # Set to deep NREM (high probability)
        swr.set_wake_sleep_mode(WakeSleepMode.NREM_DEEP)
        # Deep NREM has low ACh/NE
        swr.state.ach_level = 0.1
        swr.state.ne_level = 0.1
        p_deep = swr.get_swr_probability()

        # Active wake should have lower probability than deep NREM
        # Note: p_active may be 0 which is valid for active wake
        assert p_active <= p_deep
        assert p_deep > 0  # Deep NREM should definitely have SWRs

    def test_state_dependent_ripple_frequency(self):
        """Ripple frequency should vary with state."""
        swr = create_swr_coupling(enable_state_gating=True)

        # Check that frequency stays in valid range across states
        for mode in WakeSleepMode:
            swr.set_wake_sleep_mode(mode)
            swr.step(dt=0.01)
            freq = swr.state.current_ripple_freq
            assert RIPPLE_FREQ_MIN <= freq <= RIPPLE_FREQ_MAX


class TestSWRStatistics:
    """Test Phase 2 statistics."""

    def test_stats_include_phase2_fields(self):
        """Statistics should include Phase 2 fields."""
        swr = create_swr_coupling()
        swr.step(dt=0.01)
        stats = swr.get_stats()

        assert "wake_sleep_mode" in stats
        assert "current_ripple_freq" in stats
        assert "swr_count_wake" in stats
        assert "swr_count_sleep" in stats

    def test_swr_count_tracking(self):
        """SWR counts should be tracked separately for wake and sleep."""
        swr = create_swr_coupling(enable_state_gating=True)

        # Initial counts should be zero
        assert swr.state.swr_count_wake == 0
        assert swr.state.swr_count_sleep == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
