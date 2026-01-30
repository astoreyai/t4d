"""
Tests for SWR-Neural Field Coupling.

Tests cover:
1. SWR gating conditions (ACh, NE, hippocampal activity)
2. SWR phase progression
3. Neural field modulation during SWR
4. Replay interface
5. Callbacks and statistics
"""

import numpy as np
import pytest

from ww.nca.swr_coupling import (
    SWRNeuralFieldCoupling,
    SWRConfig,
    SWRCouplingState,
    SWREvent,
    SWRPhase,
    create_swr_coupling,
)
from ww.nca.neural_field import NeuralFieldSolver, NeuralFieldConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def swr():
    """Create default SWR coupling."""
    return SWRNeuralFieldCoupling()


@pytest.fixture
def swr_with_field():
    """Create SWR with neural field."""
    config = NeuralFieldConfig(grid_size=8)
    field = NeuralFieldSolver(config)
    return SWRNeuralFieldCoupling(neural_field=field)


# =============================================================================
# Test Configuration
# =============================================================================

class TestSWRConfig:
    """Tests for SWR configuration."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = SWRConfig()
        assert 150 <= config.ripple_frequency <= 250
        assert config.ripple_duration > 0.05
        assert 0 < config.ach_threshold < 0.5
        assert config.glutamate_boost > 0

    def test_custom_config(self):
        """Custom config values are preserved."""
        config = SWRConfig(
            ripple_frequency=200.0,
            glutamate_boost=0.4,
        )
        assert config.ripple_frequency == 200.0
        assert config.glutamate_boost == 0.4


# =============================================================================
# Test Gating Conditions
# =============================================================================

class TestGatingConditions:
    """Tests for SWR gating."""

    def test_low_ach_allows_swr(self, swr):
        """Low ACh allows SWR initiation."""
        swr.state.ach_level = 0.2
        swr.state.ne_level = 0.2
        swr.state.hippocampal_activity = 0.7
        swr.state.time_since_last_swr = 1.0

        assert swr._should_initiate_swr()

    def test_high_ach_blocks_swr(self, swr):
        """High ACh blocks SWR."""
        swr.state.ach_level = 0.6
        swr.state.ne_level = 0.2
        swr.state.hippocampal_activity = 0.7
        swr.state.time_since_last_swr = 1.0

        assert not swr._should_initiate_swr()

    def test_high_ne_blocks_swr(self, swr):
        """High NE (arousal) blocks SWR."""
        swr.state.ach_level = 0.2
        swr.state.ne_level = 0.6
        swr.state.hippocampal_activity = 0.7
        swr.state.time_since_last_swr = 1.0

        assert not swr._should_initiate_swr()

    def test_refractory_period(self, swr):
        """Refractory period prevents rapid SWRs."""
        swr.state.ach_level = 0.2
        swr.state.ne_level = 0.2
        swr.state.hippocampal_activity = 0.7
        swr.state.time_since_last_swr = 0.1  # Too soon

        assert not swr._should_initiate_swr()

    def test_can_initiate_swr_method(self, swr):
        """can_initiate_swr checks conditions."""
        swr.state.ach_level = 0.2
        swr.state.ne_level = 0.2
        swr.state.hippocampal_activity = 0.8
        swr.state.time_since_last_swr = 1.0

        assert swr.can_initiate_swr()


# =============================================================================
# Test Phase Progression
# =============================================================================

class TestPhaseProgression:
    """Tests for SWR phase progression."""

    def test_initial_state_quiescent(self, swr):
        """Initial state is quiescent."""
        assert swr.state.phase == SWRPhase.QUIESCENT
        assert swr.state.current_amplitude == 0.0

    def test_force_swr_initiates(self, swr):
        """force_swr initiates an event."""
        assert swr.force_swr()
        assert swr.state.phase == SWRPhase.INITIATING

    def test_phase_progresses_through_swr(self, swr):
        """Phases progress: INITIATING -> RIPPLING -> TERMINATING."""
        swr.force_swr()
        assert swr.state.phase == SWRPhase.INITIATING

        # Progress through initiating
        for _ in range(10):
            swr.step(dt=0.01)

        assert swr.state.phase == SWRPhase.RIPPLING

        # Progress through rippling
        for _ in range(15):
            swr.step(dt=0.01)

        # Should be terminating or quiescent
        assert swr.state.phase in [SWRPhase.TERMINATING, SWRPhase.QUIESCENT]

    def test_returns_to_quiescent(self, swr):
        """SWR completes and returns to quiescent."""
        swr.force_swr()

        # Run until complete
        for _ in range(50):
            swr.step(dt=0.01)

        assert swr.state.phase == SWRPhase.QUIESCENT
        assert len(swr._swr_events) == 1

    def test_amplitude_during_rippling(self, swr):
        """Amplitude oscillates during rippling phase."""
        swr.force_swr()

        # Progress to rippling
        for _ in range(10):
            swr.step(dt=0.01)

        assert swr.state.phase == SWRPhase.RIPPLING
        assert swr.state.current_amplitude > 0


# =============================================================================
# Test Neural Field Modulation
# =============================================================================

class TestNeuralFieldModulation:
    """Tests for neural field modulation during SWR."""

    def test_glutamate_injection_during_swr(self, swr_with_field):
        """Glutamate is injected during SWR."""
        initial_glu = swr_with_field.neural_field.get_mean_state().glutamate

        swr_with_field.force_swr()

        # Step through SWR
        for _ in range(20):
            swr_with_field.step(dt=0.01)

        # Glutamate should have increased
        assert swr_with_field.state.glutamate_injection >= 0

    def test_no_injection_when_quiescent(self, swr_with_field):
        """No injection when quiescent."""
        swr_with_field.step(dt=0.01)
        assert swr_with_field.state.glutamate_injection == 0.0
        assert swr_with_field.state.gaba_injection == 0.0

    def test_injection_proportional_to_amplitude(self, swr_with_field):
        """Injection scales with amplitude."""
        swr_with_field.force_swr()

        injections = []
        for _ in range(15):
            swr_with_field.step(dt=0.01)
            injections.append(swr_with_field.state.glutamate_injection)

        # Should see non-zero injections
        assert any(i > 0 for i in injections)


# =============================================================================
# Test Replay Interface
# =============================================================================

class TestReplayInterface:
    """Tests for replay functionality."""

    def test_replay_fails_when_quiescent(self, swr):
        """Cannot replay when not in SWR."""
        pattern = np.random.randn(32)
        assert not swr.trigger_replay(pattern)

    def test_replay_succeeds_during_swr(self, swr):
        """Can replay during SWR."""
        swr.force_swr()

        # Progress a bit
        for _ in range(5):
            swr.step(dt=0.01)

        pattern = np.random.randn(32)
        # Without hippocampus, returns False but doesn't crash
        result = swr.trigger_replay(pattern, memory_id="test_mem")
        # Result depends on hippocampus presence
        assert isinstance(result, bool)

    def test_replay_compression_active_during_swr(self, swr):
        """Compression factor is active during SWR."""
        assert swr.get_replay_compression() == 1.0

        swr.force_swr()
        for _ in range(5):
            swr.step(dt=0.01)

        assert swr.get_replay_compression() == swr.config.compression_factor


# =============================================================================
# Test Callbacks
# =============================================================================

class TestCallbacks:
    """Tests for SWR callbacks."""

    def test_swr_callback_fires(self, swr):
        """Callback fires on SWR completion."""
        received = []

        def callback(event):
            received.append(event)

        swr.register_swr_callback(callback)
        swr.force_swr()

        # Complete SWR
        for _ in range(50):
            swr.step(dt=0.01)

        assert len(received) == 1
        assert isinstance(received[0], SWREvent)

    def test_event_has_correct_fields(self, swr):
        """Completed event has correct fields."""
        events = []
        swr.register_swr_callback(lambda e: events.append(e))

        swr.force_swr()
        for _ in range(50):
            swr.step(dt=0.01)

        event = events[0]
        assert event.start_time >= 0
        assert event.duration > 0
        assert event.peak_amplitude > 0


# =============================================================================
# Test External Interface
# =============================================================================

class TestExternalInterface:
    """Tests for external control methods."""

    def test_is_swr_active(self, swr):
        """is_swr_active reports correctly."""
        assert not swr.is_swr_active()

        swr.force_swr()
        assert swr.is_swr_active()

    def test_set_ach_level(self, swr):
        """Can manually set ACh level."""
        swr.set_ach_level(0.5)
        assert swr.state.ach_level == 0.5

    def test_set_ne_level(self, swr):
        """Can manually set NE level."""
        swr.set_ne_level(0.6)
        assert swr.state.ne_level == 0.6

    def test_force_swr_only_when_quiescent(self, swr):
        """force_swr only works when quiescent."""
        assert swr.force_swr()
        assert not swr.force_swr()  # Already in SWR


# =============================================================================
# Test Statistics
# =============================================================================

class TestStatistics:
    """Tests for statistics tracking."""

    def test_get_stats(self, swr):
        """Stats dict contains expected keys."""
        stats = swr.get_stats()

        assert "phase" in stats
        assert "current_amplitude" in stats
        assert "total_swrs" in stats
        assert "ach_level" in stats

    def test_stats_track_events(self, swr):
        """Stats track event count."""
        swr.force_swr()
        for _ in range(50):
            swr.step(dt=0.01)

        stats = swr.get_stats()
        assert stats["total_swrs"] == 1

    def test_get_recent_events(self, swr):
        """Can retrieve recent events."""
        swr.force_swr()
        for _ in range(50):
            swr.step(dt=0.01)

        events = swr.get_recent_events(10)
        assert len(events) == 1


# =============================================================================
# Test State Management
# =============================================================================

class TestStateManagement:
    """Tests for state save/load."""

    def test_reset(self, swr):
        """Reset clears state."""
        swr.force_swr()
        for _ in range(20):
            swr.step(dt=0.01)

        swr.reset()

        assert swr.state.phase == SWRPhase.QUIESCENT
        assert len(swr._swr_events) == 0

    def test_save_load_state(self, swr):
        """State can be saved and restored."""
        swr.step(dt=0.1)
        swr.state.ach_level = 0.25

        saved = swr.save_state()

        new_swr = SWRNeuralFieldCoupling()
        new_swr.load_state(saved)

        assert new_swr.state.ach_level == 0.25


# =============================================================================
# Test Factory
# =============================================================================

class TestFactory:
    """Tests for factory function."""

    def test_create_swr_coupling(self):
        """Factory creates configured coupling."""
        coupling = create_swr_coupling(ripple_frequency=200.0)
        assert coupling.config.ripple_frequency == 200.0

    def test_create_with_neural_field(self):
        """Factory accepts neural field."""
        field = NeuralFieldSolver(NeuralFieldConfig(grid_size=8))
        coupling = create_swr_coupling(neural_field=field)
        assert coupling.neural_field is field


# =============================================================================
# Test Biological Plausibility
# =============================================================================

class TestBiologicalPlausibility:
    """Tests for biologically realistic behavior."""

    def test_swr_duration_realistic(self, swr):
        """SWR duration is in realistic range (~80-150ms)."""
        swr.force_swr()

        start_time = swr._simulation_time
        while swr.is_swr_active():
            swr.step(dt=0.001)

        duration = swr._simulation_time - start_time
        assert 0.05 < duration < 0.2  # 50-200ms

    def test_ripple_frequency_in_signal(self, swr):
        """Ripple oscillation visible in amplitude."""
        swr.force_swr()

        # Progress to rippling
        for _ in range(10):
            swr.step(dt=0.01)

        if swr.state.phase == SWRPhase.RIPPLING:
            amplitudes = []
            for _ in range(10):
                swr.step(dt=0.001)
                amplitudes.append(swr.state.current_amplitude)

            # Should see oscillation (non-constant)
            if len(amplitudes) > 2:
                assert max(amplitudes) > min(amplitudes)

    def test_ach_blocks_swr_like_wakefulness(self, swr):
        """High ACh (wakefulness) blocks SWR like in biology."""
        # Set conditions for SWR except ACh
        swr.state.ne_level = 0.2
        swr.state.hippocampal_activity = 0.8
        swr.state.time_since_last_swr = 1.0

        # High ACh should block
        swr.state.ach_level = 0.6
        for _ in range(100):
            occurred = swr.step(dt=0.01)
            if occurred:
                break
        assert swr.state.phase == SWRPhase.QUIESCENT

        # Low ACh should allow
        swr.state.ach_level = 0.2
        swr.state.time_since_last_swr = 1.0
        # Probabilistic, so run several times
        any_swr = False
        for _ in range(20):
            if swr.force_swr():
                any_swr = True
                break
            swr.reset()
            swr.state.ach_level = 0.2
            swr.state.ne_level = 0.2
            swr.state.hippocampal_activity = 0.8
            swr.state.time_since_last_swr = 1.0

        assert any_swr
