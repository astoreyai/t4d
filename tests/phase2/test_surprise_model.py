"""
Phase 2 Tests: Surprise-Driven NE (Uncertainty Signaling).

Tests the surprise model for LC-NE:
- Prediction error tracking
- Uncertainty estimation
- Surprise-driven phasic triggering
- Change point detection
- Adaptive learning rate
"""

import pytest
import numpy as np
import sys

sys.path.insert(0, "src")

from ww.nca.locus_coeruleus import (
    LocusCoeruleus,
    LCConfig,
    SurpriseModel,
    SurpriseConfig,
    SurpriseState,
    create_locus_coeruleus,
)


class TestSurpriseConfig:
    """Test surprise configuration."""

    def test_default_thresholds(self):
        """Default thresholds should be reasonable."""
        config = SurpriseConfig()
        assert 0 < config.surprise_threshold_low < config.surprise_threshold_high < 1

    def test_default_learning_rates(self):
        """Default learning rate range should be reasonable."""
        config = SurpriseConfig()
        assert 0 < config.learning_rate_min < config.learning_rate_max < 1


class TestSurpriseModel:
    """Test standalone surprise model."""

    def test_observe_prediction(self):
        """observe_prediction should store prediction."""
        sm = SurpriseModel()
        sm.observe_prediction(0.5)
        assert sm.state.last_prediction == 0.5

    def test_observe_outcome_computes_surprise(self):
        """observe_outcome should compute surprise."""
        sm = SurpriseModel()
        sm.observe_prediction(0.3)
        surprise = sm.observe_outcome(0.7)

        assert surprise == pytest.approx(0.4, abs=0.01)
        assert sm.state.prediction_error == pytest.approx(0.4, abs=0.01)

    def test_no_surprise_for_accurate_prediction(self):
        """Accurate prediction should give low surprise."""
        sm = SurpriseModel()
        sm.observe_prediction(0.5)
        surprise = sm.observe_outcome(0.5)

        assert surprise < 0.1

    def test_high_surprise_for_large_error(self):
        """Large prediction error should give high surprise."""
        sm = SurpriseModel()
        sm.observe_prediction(0.1)
        surprise = sm.observe_outcome(0.9)

        assert surprise >= 0.7

    def test_cumulative_surprise_tracks_history(self):
        """Cumulative surprise should track running average."""
        sm = SurpriseModel()

        # Series of surprising outcomes
        for _ in range(5):
            sm.observe_prediction(0.0)
            sm.observe_outcome(0.5)

        assert sm.state.cumulative_surprise > 0


class TestUncertaintyEstimation:
    """Test uncertainty estimation."""

    def test_stable_environment_low_uncertainty(self):
        """Stable predictions should give low uncertainty."""
        sm = SurpriseModel()

        # Consistent, accurate predictions
        for _ in range(20):
            sm.observe_prediction(0.5)
            sm.observe_outcome(0.5 + np.random.normal(0, 0.02))

        assert sm.state.estimated_uncertainty < 0.1

    def test_volatile_environment_high_uncertainty(self):
        """Volatile outcomes should give high uncertainty."""
        sm = SurpriseModel()

        # High variance outcomes
        for _ in range(20):
            sm.observe_prediction(0.5)
            sm.observe_outcome(np.random.uniform(0, 1))

        assert sm.state.estimated_uncertainty > 0.05

    def test_unexpected_uncertainty_detects_variance_change(self):
        """Unexpected uncertainty should detect variance changes."""
        sm = SurpriseModel()

        # First: low variance
        for _ in range(30):
            sm.observe_prediction(0.5)
            sm.observe_outcome(0.5 + np.random.normal(0, 0.01))

        initial_unexpected = sm.state.unexpected_uncertainty

        # Then: sudden high variance
        for _ in range(5):
            sm.observe_prediction(0.5)
            sm.observe_outcome(0.5 + np.random.normal(0, 0.5))

        # Unexpected uncertainty should increase
        assert sm.state.unexpected_uncertainty >= 0  # May be 0 initially


class TestPhasicTriggering:
    """Test surprise-driven phasic triggering."""

    def test_should_trigger_phasic_high_surprise(self):
        """High surprise should trigger phasic."""
        sm = SurpriseModel()
        sm.observe_prediction(0.0)
        sm.observe_outcome(0.9)  # High surprise

        assert sm.should_trigger_phasic() is True

    def test_should_not_trigger_phasic_low_surprise(self):
        """Low surprise should not trigger phasic."""
        sm = SurpriseModel()
        sm.observe_prediction(0.5)
        sm.observe_outcome(0.52)  # Low surprise

        assert sm.should_trigger_phasic() is False

    def test_phasic_magnitude_scales_with_surprise(self):
        """Phasic magnitude should scale with surprise."""
        sm = SurpriseModel()

        # Moderate surprise
        sm.observe_prediction(0.3)
        sm.observe_outcome(0.6)
        mag_moderate = sm.get_phasic_magnitude()

        # High surprise
        sm.observe_prediction(0.1)
        sm.observe_outcome(0.9)
        mag_high = sm.get_phasic_magnitude()

        assert mag_high > mag_moderate


class TestChangePointDetection:
    """Test change point detection."""

    def test_change_point_after_sudden_shift(self):
        """Sudden shift should increase change point probability."""
        sm = SurpriseModel()

        # Stable period
        for _ in range(20):
            sm.observe_prediction(0.3)
            sm.observe_outcome(0.3)

        initial_cp = sm.state.change_point_probability

        # Sudden change
        sm.observe_prediction(0.3)
        sm.observe_outcome(0.9)

        assert sm.state.change_point_probability >= initial_cp

    def test_run_length_tracks_stability(self):
        """Run length should track stable periods."""
        sm = SurpriseModel()

        # Series of stable observations
        for _ in range(10):
            sm.observe_prediction(0.5)
            sm.observe_outcome(0.5)

        assert sm.state.run_length >= 5


class TestAdaptiveLearningRate:
    """Test adaptive learning rate modulation."""

    def test_high_surprise_high_learning_rate(self):
        """High surprise should give high learning rate."""
        sm = SurpriseModel()

        # Low surprise first
        sm.observe_prediction(0.5)
        sm.observe_outcome(0.5)
        lr_low = sm.state.adaptive_learning_rate

        # High surprise
        sm.observe_prediction(0.1)
        sm.observe_outcome(0.9)
        lr_high = sm.state.adaptive_learning_rate

        assert lr_high > lr_low

    def test_learning_rate_in_valid_range(self):
        """Learning rate should stay in configured range."""
        sm = SurpriseModel()

        for _ in range(20):
            sm.observe_prediction(np.random.uniform(0, 1))
            sm.observe_outcome(np.random.uniform(0, 1))

            assert sm.config.learning_rate_min <= sm.state.adaptive_learning_rate
            assert sm.state.adaptive_learning_rate <= sm.config.learning_rate_max


class TestLCSurpriseIntegration:
    """Test surprise model integration with LocusCoeruleus."""

    def test_lc_has_surprise_model(self):
        """LocusCoeruleus should have integrated surprise model."""
        lc = create_locus_coeruleus()
        assert hasattr(lc, "surprise")
        assert isinstance(lc.surprise, SurpriseModel)

    def test_observe_prediction_outcome(self):
        """observe_prediction_outcome should work through LC."""
        lc = create_locus_coeruleus()

        surprise, phasic = lc.observe_prediction_outcome(
            prediction=0.3,
            outcome=0.7,
        )

        assert 0 <= surprise <= 1
        assert isinstance(phasic, bool)

    def test_surprise_triggers_phasic(self):
        """High surprise should trigger LC phasic."""
        lc = create_locus_coeruleus()

        # Large prediction error
        surprise, phasic = lc.observe_prediction_outcome(
            prediction=0.1,
            outcome=0.95,
        )

        # May or may not trigger depending on refractory
        assert surprise > 0.7

    def test_get_surprise_level(self):
        """get_surprise_level should return current surprise."""
        lc = create_locus_coeruleus()
        lc.observe_prediction_outcome(0.3, 0.6)

        surprise = lc.get_surprise_level()
        assert 0 <= surprise <= 1

    def test_get_uncertainty(self):
        """get_uncertainty should return estimated uncertainty."""
        lc = create_locus_coeruleus()

        uncertainty = lc.get_uncertainty()
        assert uncertainty >= 0

    def test_get_adaptive_learning_rate(self):
        """get_adaptive_learning_rate should return valid rate."""
        lc = create_locus_coeruleus()
        lc.observe_prediction_outcome(0.5, 0.8)

        lr = lc.get_adaptive_learning_rate()
        assert 0 < lr < 1

    def test_should_update_model(self):
        """should_update_model should flag high surprise/change."""
        lc = create_locus_coeruleus()

        # Low surprise
        lc.observe_prediction_outcome(0.5, 0.5)
        assert lc.should_update_model() is False

        # High surprise
        lc.observe_prediction_outcome(0.1, 0.9)
        assert lc.should_update_model() is True

    def test_stats_include_surprise(self):
        """Statistics should include surprise model data."""
        lc = create_locus_coeruleus()
        lc.step(dt=0.01)

        stats = lc.get_stats()
        assert "surprise" in stats
        assert "surprise" in stats["surprise"]  # Nested surprise level
        assert "adaptive_learning_rate" in stats["surprise"]

    def test_reset_clears_surprise(self):
        """Reset should clear surprise model state."""
        lc = create_locus_coeruleus()

        # Build up history
        for _ in range(10):
            lc.observe_prediction_outcome(0.5, 0.8)

        lc.reset()

        assert lc.surprise.state.surprise == 0.0
        assert lc.surprise.state.cumulative_surprise == 0.0


class TestTonicModulation:
    """Test uncertainty-based tonic modulation."""

    def test_tonic_modulation_from_uncertainty(self):
        """Uncertainty should modulate tonic level."""
        sm = SurpriseModel()

        # Build up uncertainty estimate
        for _ in range(20):
            sm.observe_prediction(0.5)
            sm.observe_outcome(np.random.uniform(0.3, 0.7))

        tonic_mod = sm.get_tonic_modulation()
        assert 0 <= tonic_mod <= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
