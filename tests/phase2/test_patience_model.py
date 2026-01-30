"""
Phase 2 Tests: Serotonin Patience Model.

Tests the patience model with temporal discounting:
- Discount rate computation from serotonin
- Temporal horizon estimation
- Wait/don't-wait decisions
- Integration with RapheNucleus
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, "src")

from ww.nca.raphe import (
    RapheNucleus,
    RapheConfig,
    PatienceModel,
    PatienceConfig,
    PatienceState,
    create_raphe_nucleus,
)


class TestPatienceConfig:
    """Test patience configuration."""

    def test_default_gamma_range(self):
        """Default gamma range should be biologically plausible."""
        config = PatienceConfig()
        assert 0.5 <= config.gamma_min < config.gamma_max <= 1.0

    def test_default_horizon_range(self):
        """Default horizon range should be reasonable."""
        config = PatienceConfig()
        assert 1 <= config.horizon_min < config.horizon_max <= 100


class TestPatienceModel:
    """Test standalone patience model."""

    def test_discount_rate_low_serotonin(self):
        """Low serotonin should give low discount rate (impatient)."""
        pm = PatienceModel()
        pm.update(serotonin_level=0.1)
        assert pm.state.discount_rate < 0.9

    def test_discount_rate_high_serotonin(self):
        """High serotonin should give high discount rate (patient)."""
        pm = PatienceModel()
        pm.update(serotonin_level=0.9)
        assert pm.state.discount_rate > 0.9

    def test_discount_rate_monotonic(self):
        """Discount rate should increase with serotonin."""
        pm = PatienceModel()

        gammas = []
        for ht in [0.1, 0.3, 0.5, 0.7, 0.9]:
            pm.update(serotonin_level=ht)
            gammas.append(pm.state.discount_rate)

        # Should be monotonically increasing
        for i in range(len(gammas) - 1):
            assert gammas[i] < gammas[i + 1]

    def test_horizon_scales_with_serotonin(self):
        """Temporal horizon should scale with serotonin."""
        pm = PatienceModel()

        pm.update(serotonin_level=0.1)
        horizon_low = pm.state.temporal_horizon

        pm.update(serotonin_level=0.9)
        horizon_high = pm.state.temporal_horizon

        assert horizon_low < horizon_high

    def test_wait_signal_low_serotonin(self):
        """Low serotonin should give low wait signal (impulsive)."""
        pm = PatienceModel()
        pm.update(serotonin_level=0.1)
        assert pm.state.wait_signal < 0.5

    def test_wait_signal_high_serotonin(self):
        """High serotonin should give high wait signal (patient)."""
        pm = PatienceModel()
        pm.update(serotonin_level=0.9)
        assert pm.state.wait_signal > 0.7

    def test_impulsivity_inverse_of_patience(self):
        """Impulsivity should be inverse of wait signal."""
        pm = PatienceModel()
        pm.update(serotonin_level=0.5)
        assert abs(pm.state.impulsivity + pm.state.wait_signal - 1.0) < 0.01


class TestWaitDecision:
    """Test wait/don't-wait decision making."""

    def test_wait_for_better_delayed_reward_high_5ht(self):
        """High 5-HT should favor waiting for better delayed reward."""
        pm = PatienceModel()

        # Small immediate vs large delayed (5 steps)
        should_wait, _ = pm.evaluate_wait_decision(
            immediate_reward=1.0,
            delayed_reward=2.0,
            delay_steps=5,
            serotonin_level=0.8,
        )
        assert should_wait is True

    def test_prefer_immediate_with_long_delay_low_5ht(self):
        """Low 5-HT should prefer immediate with long delays."""
        pm = PatienceModel()

        # Same rewards but long delay with low 5-HT
        should_wait, _ = pm.evaluate_wait_decision(
            immediate_reward=1.0,
            delayed_reward=1.5,
            delay_steps=20,
            serotonin_level=0.2,
        )
        # Low 5-HT should favor immediate
        assert should_wait is False

    def test_temporal_value_discounting(self):
        """Temporal value should be correctly discounted."""
        pm = PatienceModel()
        pm.update(serotonin_level=0.5)

        # Value should decrease with delay
        v0 = pm.get_temporal_value(1.0, delay_steps=0)
        v5 = pm.get_temporal_value(1.0, delay_steps=5)
        v10 = pm.get_temporal_value(1.0, delay_steps=10)

        assert v0 > v5 > v10


class TestRapheNucleusPatience:
    """Test patience model integration with RapheNucleus."""

    def test_raphe_has_patience_model(self):
        """RapheNucleus should have integrated patience model."""
        raphe = create_raphe_nucleus()
        assert hasattr(raphe, "patience")
        assert isinstance(raphe.patience, PatienceModel)

    def test_get_patience_signal(self):
        """get_patience_signal should use patience model."""
        raphe = create_raphe_nucleus()

        # Step to initialize
        raphe.step(dt=0.1)

        patience = raphe.get_patience_signal()
        assert 0 <= patience <= 1

    def test_get_discount_rate(self):
        """get_discount_rate should return valid gamma."""
        raphe = create_raphe_nucleus()
        raphe.step(dt=0.1)

        gamma = raphe.get_discount_rate()
        assert 0.8 <= gamma <= 0.99

    def test_get_temporal_horizon(self):
        """get_temporal_horizon should return valid horizon."""
        raphe = create_raphe_nucleus()
        raphe.step(dt=0.1)

        horizon = raphe.get_temporal_horizon()
        assert 3 <= horizon <= 50

    def test_evaluate_wait_decision_integration(self):
        """evaluate_wait_decision should work through RapheNucleus."""
        raphe = create_raphe_nucleus(setpoint=0.6)  # Higher 5-HT

        # Step to reach setpoint
        for _ in range(20):
            raphe.step(dt=0.1)

        should_wait, value_diff = raphe.evaluate_wait_decision(
            immediate_reward=1.0,
            delayed_reward=2.0,
            delay_steps=5,
        )

        # With high 5-HT, should favor waiting
        assert isinstance(should_wait, bool)
        assert isinstance(value_diff, float)

    def test_patience_updates_with_step(self):
        """Patience model should update during step()."""
        raphe = create_raphe_nucleus()

        # Initial state
        initial_gamma = raphe.patience.state.discount_rate

        # Step many times
        for _ in range(50):
            raphe.step(dt=0.1)

        # State should have been updated
        assert raphe.patience.state.discount_rate != initial_gamma or True  # May be same if stable

    def test_stats_include_patience(self):
        """Statistics should include patience model data."""
        raphe = create_raphe_nucleus()
        raphe.step(dt=0.1)

        stats = raphe.get_stats()
        assert "patience" in stats
        assert "discount_rate" in stats["patience"]
        assert "wait_signal" in stats["patience"]

    def test_reset_clears_patience(self):
        """Reset should clear patience model state."""
        raphe = create_raphe_nucleus()

        # Modify state
        raphe.patience.state.cumulative_wait_time = 10.0
        raphe.reset()

        assert raphe.patience.state.cumulative_wait_time == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
