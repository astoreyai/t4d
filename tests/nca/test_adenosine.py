"""
Tests for adenosine sleep-wake dynamics.

Validates:
1. Adenosine accumulation during wake
2. Adenosine clearance during sleep
3. Sleep pressure thresholds
4. NT modulation effects
5. Caffeine dynamics
6. Neural field integration
"""

import numpy as np
import pytest

from ww.nca.adenosine import (
    AdenosineConfig,
    AdenosineDynamics,
    AdenosineState,
    SleepPressureIntegrator,
    SleepWakeState,
    compute_sleep_need,
    create_adenosine_system,
)


class TestAdenosineConfig:
    """Test configuration validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AdenosineConfig()
        assert config.baseline_level == 0.1
        assert config.max_level == 1.0
        assert 0 < config.accumulation_rate < 0.1
        assert config.sleep_onset_threshold > config.wake_threshold

    def test_biological_thresholds(self):
        """Test that thresholds follow biological order."""
        config = AdenosineConfig()
        # Order: baseline < wake < drowsy < sleep_onset < exhausted < max
        assert config.baseline_level < config.wake_threshold
        assert config.wake_threshold < config.drowsy_threshold
        assert config.drowsy_threshold < config.sleep_onset_threshold
        assert config.sleep_onset_threshold < config.exhausted_threshold
        assert config.exhausted_threshold <= config.max_level


class TestAdenosineDynamics:
    """Test adenosine accumulation and clearance."""

    def test_initialization(self):
        """Test initial state is rested."""
        dynamics = AdenosineDynamics()
        assert dynamics.state.level == pytest.approx(0.1, abs=0.01)
        assert dynamics.state.state == SleepWakeState.WAKE_ALERT
        assert dynamics.state.cognitive_efficiency > 0.9

    def test_accumulation_during_wake(self):
        """Test adenosine accumulates during wake."""
        dynamics = AdenosineDynamics()
        initial = dynamics.state.level

        # Simulate 8 hours wake
        for _ in range(8):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.7)

        assert dynamics.state.level > initial
        assert dynamics.state.wake_duration_hours == 8.0

    def test_full_day_accumulation(self):
        """Test 16 hours of wake approaches high adenosine."""
        dynamics = AdenosineDynamics()

        # Simulate full day wake
        for _ in range(16):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.6)

        # Should be drowsy or exhausted
        assert dynamics.state.level > dynamics.config.drowsy_threshold
        assert dynamics.state.state in [
            SleepWakeState.WAKE_DROWSY,
            SleepWakeState.WAKE_EXHAUSTED
        ]

    def test_activity_affects_accumulation(self):
        """Test higher activity increases accumulation rate."""
        low_activity = AdenosineDynamics()
        high_activity = AdenosineDynamics()

        # Same duration, different activity
        for _ in range(8):
            low_activity.step_wake(dt_hours=1.0, activity_level=0.2)
            high_activity.step_wake(dt_hours=1.0, activity_level=0.9)

        assert high_activity.state.level > low_activity.state.level

    def test_clearance_during_sleep(self):
        """Test adenosine clears during sleep."""
        dynamics = AdenosineDynamics()

        # Build up adenosine
        for _ in range(12):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.7)

        high_level = dynamics.state.level

        # Enter sleep
        dynamics.enter_sleep()

        # Sleep for 4 hours
        for _ in range(4):
            dynamics.step_sleep(dt_hours=1.0, sleep_phase="deep")

        assert dynamics.state.level < high_level
        assert dynamics.state.sleep_duration_hours == 4.0

    def test_deep_sleep_faster_clearance(self):
        """Test deep sleep clears adenosine faster than light sleep."""
        # Set up two systems at same high adenosine
        deep = AdenosineDynamics()
        light = AdenosineDynamics()

        for d in [deep, light]:
            for _ in range(12):
                d.step_wake(dt_hours=1.0, activity_level=0.7)
            d.enter_sleep()

        # Same sleep duration, different phases
        for _ in range(4):
            deep.step_sleep(dt_hours=1.0, sleep_phase="deep")
            light.step_sleep(dt_hours=1.0, sleep_phase="light")

        assert deep.state.level < light.state.level

    def test_rem_slower_clearance(self):
        """Test REM sleep has slower adenosine clearance."""
        deep = AdenosineDynamics()
        rem = AdenosineDynamics()

        for d in [deep, rem]:
            for _ in range(12):
                d.step_wake(dt_hours=1.0, activity_level=0.7)
            d.enter_sleep()

        for _ in range(4):
            deep.step_sleep(dt_hours=1.0, sleep_phase="deep")
            rem.step_sleep(dt_hours=1.0, sleep_phase="rem")

        # REM clears slower
        assert rem.state.level > deep.state.level


class TestSleepPressure:
    """Test sleep pressure and thresholds."""

    def test_should_sleep_trigger(self):
        """Test should_sleep triggers at threshold."""
        dynamics = AdenosineDynamics()
        assert not dynamics.should_sleep()

        # Build to threshold
        while not dynamics.should_sleep():
            dynamics.step_wake(dt_hours=1.0, activity_level=0.8)
            if dynamics.state.wake_duration_hours > 24:
                break  # Safety limit

        assert dynamics.should_sleep()
        assert dynamics.state.sleep_pressure >= dynamics.config.sleep_onset_threshold

    def test_can_wake_after_sleep(self):
        """Test can_wake returns true after adequate sleep."""
        dynamics = AdenosineDynamics()

        # Build pressure
        for _ in range(16):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.7)

        assert not dynamics.can_wake()  # High adenosine

        # Sleep
        dynamics.enter_sleep()
        for _ in range(8):
            dynamics.step_sleep(dt_hours=1.0, sleep_phase="deep")

        assert dynamics.can_wake()

    def test_cognitive_efficiency_declines(self):
        """Test cognitive efficiency decreases with sleep pressure."""
        dynamics = AdenosineDynamics()
        initial_efficiency = dynamics.state.cognitive_efficiency

        for _ in range(16):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.8)

        assert dynamics.state.cognitive_efficiency < initial_efficiency
        assert dynamics.state.cognitive_efficiency > 0  # Never zero


class TestCaffeine:
    """Test caffeine antagonist effects."""

    def test_caffeine_blocks_sleep_pressure(self):
        """Test caffeine reduces perceived sleep pressure."""
        no_coffee = AdenosineDynamics()
        with_coffee = AdenosineDynamics()

        # Same wake period
        for _ in range(8):
            no_coffee.step_wake(dt_hours=1.0, activity_level=0.6)
            with_coffee.step_wake(dt_hours=1.0, activity_level=0.6, caffeine_dose=0.3)

        # Both have same adenosine, but coffee blocks receptors
        assert with_coffee.state.caffeine_level > 0
        assert with_coffee.state.sleep_pressure < no_coffee.state.sleep_pressure

    def test_caffeine_half_life(self):
        """Test caffeine decays with half-life."""
        dynamics = AdenosineDynamics()
        dynamics.add_caffeine(1.0)  # Full dose

        initial = dynamics.state.caffeine_level
        half_life = dynamics.config.caffeine_half_life_hours

        # Wait one half-life
        dynamics.step_wake(dt_hours=half_life, activity_level=0.5)

        # Should be ~50% remaining
        assert dynamics.state.caffeine_level == pytest.approx(initial * 0.5, rel=0.1)

    def test_caffeine_delays_sleep_need(self):
        """Test caffeine delays should_sleep trigger."""
        no_coffee = AdenosineDynamics()
        with_coffee = AdenosineDynamics()

        hours_no_coffee = 0
        hours_with_coffee = 0

        while not no_coffee.should_sleep() and hours_no_coffee < 24:
            no_coffee.step_wake(dt_hours=1.0, activity_level=0.6)
            hours_no_coffee += 1

        while not with_coffee.should_sleep() and hours_with_coffee < 30:
            caffeine = 0.3 if hours_with_coffee in [4, 8, 12] else 0
            with_coffee.step_wake(dt_hours=1.0, activity_level=0.6, caffeine_dose=caffeine)
            hours_with_coffee += 1

        assert hours_with_coffee > hours_no_coffee


class TestNTModulation:
    """Test neurotransmitter modulation by adenosine."""

    def test_modulation_at_baseline(self):
        """Test minimal modulation at baseline adenosine."""
        dynamics = AdenosineDynamics()
        mod = dynamics.get_nt_modulation()

        # Near baseline, modulation should be ~1.0
        for nt, value in mod.items():
            assert 0.8 < value < 1.2, f"{nt} modulation out of range"

    def test_high_adenosine_suppresses_da(self):
        """Test high adenosine suppresses dopamine."""
        dynamics = AdenosineDynamics()

        # Build high adenosine
        for _ in range(20):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.9)

        mod = dynamics.get_nt_modulation()
        assert mod["da"] < 0.9  # DA suppressed

    def test_high_adenosine_potentiates_gaba(self):
        """Test high adenosine potentiates GABA."""
        dynamics = AdenosineDynamics()

        for _ in range(20):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.9)

        mod = dynamics.get_nt_modulation()
        assert mod["gaba"] > 1.0  # GABA potentiated

    def test_ne_suppression(self):
        """Test norepinephrine is suppressed by adenosine."""
        dynamics = AdenosineDynamics()

        for _ in range(16):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.8)

        mod = dynamics.get_nt_modulation()
        assert mod["ne"] < 1.0  # NE suppressed


class TestReceptorAdaptation:
    """Test adenosine receptor dynamics."""

    def test_chronic_high_adenosine_reduces_sensitivity(self):
        """Test receptor downregulation with chronic high adenosine."""
        dynamics = AdenosineDynamics()
        initial_sensitivity = dynamics.state.receptor_sensitivity

        # Many hours at high adenosine (above drowsy threshold)
        for _ in range(20):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.9)

        # Receptors should downregulate
        assert dynamics.state.receptor_sensitivity < initial_sensitivity

    def test_sensitivity_recovers_during_sleep(self):
        """Test receptor sensitivity recovers during sleep."""
        dynamics = AdenosineDynamics()

        # Build chronic fatigue
        for _ in range(20):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.9)

        reduced_sensitivity = dynamics.state.receptor_sensitivity

        # Sleep
        dynamics.enter_sleep()
        for _ in range(8):
            dynamics.step_sleep(dt_hours=1.0, sleep_phase="deep")

        assert dynamics.state.receptor_sensitivity > reduced_sensitivity


class TestConsolidationIntegration:
    """Test integration with consolidation system."""

    def test_consolidation_signal(self):
        """Test consolidation signal increases with adenosine."""
        dynamics = AdenosineDynamics()
        initial = dynamics.get_consolidation_signal()

        for _ in range(12):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.7)

        assert dynamics.get_consolidation_signal() > initial

    def test_consolidation_need_cleared_during_sleep(self):
        """Test consolidation need decreases during sleep."""
        dynamics = AdenosineDynamics()

        for _ in range(12):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.7)

        high_need = dynamics.state.consolidation_need

        dynamics.enter_sleep()
        for _ in range(6):
            dynamics.step_sleep(dt_hours=1.0, sleep_phase="deep")

        assert dynamics.state.consolidation_need < high_need


class TestSleepPressureIntegrator:
    """Test the sleep pressure integrator."""

    def test_integrator_creation(self):
        """Test integrator initialization."""
        dynamics = AdenosineDynamics()
        integrator = SleepPressureIntegrator(dynamics)
        assert integrator.adenosine is dynamics

    def test_check_consolidation_needed(self):
        """Test consolidation check logic."""
        dynamics = AdenosineDynamics()
        integrator = SleepPressureIntegrator(dynamics)

        assert not integrator.check_consolidation_needed()

        # Build pressure
        for _ in range(20):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.8)

        assert integrator.check_consolidation_needed()

    def test_oscillator_modulation(self):
        """Test oscillator modulation based on sleep pressure."""
        dynamics = AdenosineDynamics()
        integrator = SleepPressureIntegrator(dynamics)

        # Alert state
        mod_alert = integrator.get_oscillator_modulation()
        assert mod_alert["theta_power"] > 0.8
        assert mod_alert["gamma_power"] > 0.7

        # Build fatigue
        for _ in range(16):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.8)

        mod_tired = integrator.get_oscillator_modulation()
        assert mod_tired["theta_power"] < mod_alert["theta_power"]
        assert mod_tired["gamma_power"] < mod_alert["gamma_power"]

    def test_day_simulation(self):
        """Test full day simulation."""
        dynamics = AdenosineDynamics()
        integrator = SleepPressureIntegrator(dynamics)

        results = integrator.simulate_day(wake_hours=16, sleep_hours=8)

        assert len(results) == 24
        # First should be wake
        assert results[0]["phase"] == "wake"
        # Last should be sleep
        assert "sleep" in results[-1]["phase"]
        # Should end with low adenosine
        assert results[-1]["state"]["level"] < 0.3


class TestNeuralFieldIntegration:
    """Test integration with neural field solver."""

    def test_solver_with_adenosine(self):
        """Test neural field solver accepts adenosine."""
        from ww.nca import NeuralFieldSolver, AdenosineDynamics

        adenosine = AdenosineDynamics()
        solver = NeuralFieldSolver(adenosine=adenosine)

        assert solver.adenosine is adenosine

    def test_adenosine_affects_dynamics(self):
        """Test adenosine modulates field dynamics."""
        from ww.nca import NeuralFieldSolver, AdenosineDynamics

        # Rested system
        adenosine_rested = AdenosineDynamics()
        solver_rested = NeuralFieldSolver(adenosine=adenosine_rested)

        # Fatigued system
        adenosine_tired = AdenosineDynamics()
        for _ in range(16):
            adenosine_tired.step_wake(dt_hours=1.0, activity_level=0.8)
        solver_tired = NeuralFieldSolver(adenosine=adenosine_tired)

        # Run both
        for _ in range(10):
            state_rested = solver_rested.step()
            state_tired = solver_tired.step()

        # Tired should have different NT balance
        # Higher GABA (adenosine potentiates), lower DA (adenosine suppresses)
        # Note: these are relative comparisons
        assert solver_tired.get_cognitive_efficiency() < solver_rested.get_cognitive_efficiency()

    def test_sleep_pressure_accessible(self):
        """Test sleep pressure getter works."""
        from ww.nca import NeuralFieldSolver, AdenosineDynamics

        adenosine = AdenosineDynamics()
        solver = NeuralFieldSolver(adenosine=adenosine)

        assert solver.get_sleep_pressure() == pytest.approx(0.0, abs=0.1)

        for _ in range(12):
            adenosine.step_wake(dt_hours=1.0, activity_level=0.8)

        assert solver.get_sleep_pressure() > 0.3

    def test_should_consolidate(self):
        """Test should_consolidate accessor."""
        from ww.nca import NeuralFieldSolver, AdenosineDynamics

        adenosine = AdenosineDynamics()
        solver = NeuralFieldSolver(adenosine=adenosine)

        assert not solver.should_consolidate()

        # Build to threshold
        for _ in range(20):
            adenosine.step_wake(dt_hours=1.0, activity_level=0.8)

        assert solver.should_consolidate()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_adenosine_system(self):
        """Test factory function."""
        dynamics = create_adenosine_system(baseline=0.15, accumulation_rate=0.05)
        assert dynamics.config.baseline_level == 0.15
        assert dynamics.config.accumulation_rate == 0.05

    def test_compute_sleep_need(self):
        """Test sleep need calculation."""
        # 16 hours wake should need ~8 hours sleep
        need = compute_sleep_need(wake_hours=16.0, activity_level=0.6)
        assert 6 < need < 12

        # Higher activity needs more sleep
        high_need = compute_sleep_need(wake_hours=16.0, activity_level=0.9)
        low_need = compute_sleep_need(wake_hours=16.0, activity_level=0.3)
        assert high_need > low_need


class TestStateManagement:
    """Test state management functions."""

    def test_reset(self):
        """Test reset returns to rested state."""
        dynamics = AdenosineDynamics()

        # Accumulate fatigue
        for _ in range(20):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.9)

        dynamics.reset()

        assert dynamics.state.level == pytest.approx(dynamics.config.baseline_level)
        assert dynamics.state.state == SleepWakeState.WAKE_ALERT
        assert dynamics.state.cognitive_efficiency > 0.9

    def test_get_stats(self):
        """Test stats retrieval."""
        dynamics = AdenosineDynamics()

        for _ in range(8):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.6)

        stats = dynamics.get_stats()

        assert "state" in stats
        assert "should_sleep" in stats
        assert "sleep_debt_hours" in stats
        assert "nt_modulation" in stats
        assert stats["history_length"] > 0

    def test_state_to_dict(self):
        """Test state serialization."""
        state = AdenosineState()
        d = state.to_dict()

        assert "level" in d
        assert "sleep_pressure" in d
        assert "cognitive_efficiency" in d
        assert "state" in d

    def test_optimal_sleep_duration(self):
        """Test optimal sleep calculation."""
        dynamics = AdenosineDynamics()

        # Rested needs no sleep
        assert dynamics.get_optimal_sleep_duration() == 0.0

        # Build fatigue
        for _ in range(16):
            dynamics.step_wake(dt_hours=1.0, activity_level=0.7)

        # Now needs sleep
        optimal = dynamics.get_optimal_sleep_duration()
        assert optimal > 0
        assert optimal < 12  # Reasonable bound


class TestAstrocyteIntegration:
    """Test astrocyte-adenosine interaction."""

    def test_astrocyte_boosts_clearance(self):
        """Test astrocyte activity boosts adenosine clearance."""
        normal = AdenosineDynamics()
        boosted = AdenosineDynamics()

        # Build same fatigue
        for d in [normal, boosted]:
            for _ in range(12):
                d.step_wake(dt_hours=1.0, activity_level=0.7)
            d.enter_sleep()

        # Boost one with astrocyte activity
        boosted.set_astrocyte_activity(0.8)

        # Same sleep period
        for _ in range(4):
            normal.step_sleep(dt_hours=1.0, sleep_phase="deep")
            boosted.step_sleep(dt_hours=1.0, sleep_phase="deep")

        # Boosted should clear faster
        assert boosted.state.level < normal.state.level
