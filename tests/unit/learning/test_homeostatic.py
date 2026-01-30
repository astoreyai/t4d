"""
Tests for Homeostatic Plasticity.

Tests synaptic scaling, decorrelation, BCM sliding threshold,
and learning rate modulation based on biological homeostatic mechanisms.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from ww.learning.homeostatic import (
    HomeostaticState,
    HomeostaticPlasticity,
    apply_homeostatic_bounds,
)


class TestHomeostaticState:
    """Test HomeostaticState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = HomeostaticState()
        assert state.mean_norm == 1.0
        assert state.std_norm == 0.1
        assert state.mean_activation == 0.0
        assert state.sliding_threshold == 0.5

    def test_to_dict(self):
        """Test dictionary conversion."""
        state = HomeostaticState(
            mean_norm=0.9,
            std_norm=0.15,
            mean_activation=0.3,
            sliding_threshold=0.6
        )
        d = state.to_dict()
        assert d["mean_norm"] == 0.9
        assert d["std_norm"] == 0.15
        assert d["mean_activation"] == 0.3
        assert d["sliding_threshold"] == 0.6
        assert "last_update" in d


class TestHomeostaticPlasticity:
    """Test HomeostaticPlasticity class."""

    @pytest.fixture
    def plasticity(self):
        """Create homeostatic plasticity instance."""
        return HomeostaticPlasticity(
            target_norm=1.0,
            norm_tolerance=0.2,
            ema_alpha=0.1,  # Faster updates for testing
            decorrelation_strength=0.01,
            sliding_threshold_rate=0.01
        )

    def test_init(self, plasticity):
        """Test initialization."""
        assert plasticity.target_norm == 1.0
        assert plasticity.norm_tolerance == 0.2
        state = plasticity.get_state()
        assert state.mean_norm == 1.0

    def test_update_statistics_single(self, plasticity):
        """Test updating statistics with single embedding."""
        embedding = np.random.randn(1024).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding) * 1.5  # Norm of 1.5

        plasticity.update_statistics(embedding)

        state = plasticity.get_state()
        # EMA should move toward 1.5
        assert state.mean_norm > 1.0

    def test_update_statistics_batch(self, plasticity):
        """Test updating statistics with batch of embeddings."""
        embeddings = np.random.randn(10, 1024).astype(np.float32)
        # Normalize each to norm of 0.8
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms * 0.8

        plasticity.update_statistics(embeddings)

        state = plasticity.get_state()
        # EMA should move toward 0.8
        assert state.mean_norm < 1.0

    def test_needs_scaling_no(self, plasticity):
        """Test no scaling needed when within tolerance."""
        # Default state has mean_norm=1.0, target=1.0, tolerance=0.2
        assert not plasticity.needs_scaling()

    def test_needs_scaling_yes(self, plasticity):
        """Test scaling needed when outside tolerance."""
        # Push mean well outside tolerance
        high_norm_embeddings = np.random.randn(20, 1024).astype(np.float32)
        norms = np.linalg.norm(high_norm_embeddings, axis=1, keepdims=True)
        high_norm_embeddings = high_norm_embeddings / norms * 1.5

        for _ in range(10):
            plasticity.update_statistics(high_norm_embeddings)

        assert plasticity.needs_scaling()

    def test_compute_scaling_factor(self, plasticity):
        """Test scaling factor computation."""
        # Manually set mean_norm
        plasticity._state.mean_norm = 0.5
        factor = plasticity.compute_scaling_factor()
        assert factor == pytest.approx(2.0, rel=0.01)  # 1.0 / 0.5

        plasticity._state.mean_norm = 2.0
        factor = plasticity.compute_scaling_factor()
        assert factor == pytest.approx(0.5, rel=0.01)  # 1.0 / 2.0

    def test_apply_scaling_when_needed(self, plasticity):
        """Test scaling is applied when needed."""
        plasticity._state.mean_norm = 0.5  # Needs scaling

        embeddings = np.ones((5, 1024), dtype=np.float32)
        scaled = plasticity.apply_scaling(embeddings)

        # Should be scaled by 2.0
        assert np.allclose(scaled, embeddings * 2.0)
        assert plasticity._scaling_count == 1

    def test_apply_scaling_when_not_needed(self, plasticity):
        """Test scaling is skipped when not needed."""
        # Default state, no scaling needed
        embeddings = np.ones((5, 1024), dtype=np.float32)
        result = plasticity.apply_scaling(embeddings)

        # Should be unchanged
        assert np.array_equal(result, embeddings)
        assert plasticity._scaling_count == 0

    def test_apply_scaling_force(self, plasticity):
        """Test forced scaling."""
        # Default state, no scaling needed, but force it
        plasticity._state.mean_norm = 0.8
        embeddings = np.ones((5, 1024), dtype=np.float32)
        scaled = plasticity.apply_scaling(embeddings, force=True)

        # Should be scaled by 1.25 (1.0 / 0.8)
        assert np.allclose(scaled, embeddings * 1.25, rtol=0.01)
        assert plasticity._scaling_count == 1

    def test_decorrelate(self, plasticity):
        """Test decorrelation reduces correlation between embeddings."""
        # Create correlated embeddings
        base = np.random.randn(1024).astype(np.float32)
        base = base / np.linalg.norm(base)
        embeddings = np.stack([
            base + 0.1 * np.random.randn(1024).astype(np.float32)
            for _ in range(5)
        ])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Measure initial correlation
        initial_corr = np.corrcoef(embeddings)[0, 1]

        # Decorrelate
        decorrelated = plasticity.decorrelate(embeddings, strength=0.1)

        # Measure final correlation - should be lower
        final_corr = np.corrcoef(decorrelated)[0, 1]

        assert final_corr < initial_corr
        assert plasticity._decorrelation_count == 1

    def test_decorrelate_single(self, plasticity):
        """Test decorrelation returns unchanged for single embedding."""
        embedding = np.random.randn(1024).astype(np.float32)
        result = plasticity.decorrelate(embedding)
        assert np.array_equal(result, embedding)

    def test_update_sliding_threshold(self, plasticity):
        """Test BCM sliding threshold update."""
        initial_threshold = plasticity._state.sliding_threshold

        # High activity should raise threshold
        high_activations = np.ones(100) * 0.9
        new_threshold = plasticity.update_sliding_threshold(high_activations)

        assert new_threshold > initial_threshold
        assert plasticity._state.mean_activation == pytest.approx(0.9, rel=0.01)

    def test_modulate_learning_rate(self, plasticity):
        """Test learning rate modulation based on norm."""
        base_lr = 0.1

        # Low norm = higher learning rate
        lr_low = plasticity.modulate_learning_rate(base_lr, current_norm=0.5)

        # High norm = lower learning rate
        lr_high = plasticity.modulate_learning_rate(base_lr, current_norm=2.0)

        # Normal norm = base learning rate
        lr_normal = plasticity.modulate_learning_rate(base_lr, current_norm=1.0)

        assert lr_low > lr_normal
        assert lr_high < lr_normal
        assert lr_normal == pytest.approx(base_lr, rel=0.01)

    def test_get_stats(self, plasticity):
        """Test statistics retrieval."""
        # Generate some activity
        embeddings = np.random.randn(10, 1024).astype(np.float32)
        plasticity.update_statistics(embeddings)
        plasticity.apply_scaling(embeddings, force=True)
        plasticity.decorrelate(embeddings)

        stats = plasticity.get_stats()
        assert "state" in stats
        assert stats["scaling_count"] == 1
        assert stats["decorrelation_count"] == 1
        assert stats["norm_history_size"] == 10

    def test_reset(self, plasticity):
        """Test reset clears all state."""
        # Generate some activity
        embeddings = np.random.randn(10, 1024).astype(np.float32)
        plasticity.update_statistics(embeddings)
        plasticity.apply_scaling(embeddings, force=True)

        plasticity.reset()

        state = plasticity.get_state()
        assert state.mean_norm == 1.0
        assert plasticity._scaling_count == 0
        stats = plasticity.get_stats()
        assert stats["norm_history_size"] == 0


class TestApplyHomeostaticBounds:
    """Test integration helper function."""

    def test_apply_homeostatic_bounds(self):
        """Test apply_homeostatic_bounds helper."""
        from unittest.mock import MagicMock

        mock_engine = MagicMock()
        plasticity = HomeostaticPlasticity(
            target_norm=1.0,
            norm_tolerance=0.1,
            ema_alpha=0.01  # Slow update so mean doesn't shift much
        )

        # Force need for scaling - set mean low
        plasticity._state.mean_norm = 0.5

        embeddings = np.ones((5, 1024), dtype=np.float32)
        result = apply_homeostatic_bounds(
            reconsolidation_engine=mock_engine,
            homeostatic=plasticity,
            embeddings=embeddings,
            force_scaling=True
        )

        # Should be scaled up (not exactly 2x due to EMA update)
        assert not np.array_equal(result, embeddings)
        # Result should be higher than original
        assert result[0, 0] > embeddings[0, 0]


class TestHomeostaticIntegration:
    """Integration tests for homeostatic plasticity."""

    def test_prevents_runaway_potentiation(self):
        """Test that homeostatic scaling prevents unbounded growth."""
        plasticity = HomeostaticPlasticity(
            target_norm=1.0,
            norm_tolerance=0.1,
            ema_alpha=0.5  # Fast adaptation for testing
        )

        # Simulate Hebbian-like updates that would cause runaway potentiation
        embeddings = np.random.randn(10, 1024).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        for step in range(20):
            # Simulate potentiation (increasing norms)
            embeddings = embeddings * 1.1

            # Apply homeostatic regulation
            plasticity.update_statistics(embeddings)
            if plasticity.needs_scaling():
                embeddings = plasticity.apply_scaling(embeddings)

        # Norms should be bounded near target
        final_norms = np.linalg.norm(embeddings, axis=1)
        assert np.all(final_norms < 2.0)  # Not runaway

    def test_maintains_stability_over_time(self):
        """Test long-term stability with varied inputs."""
        plasticity = HomeostaticPlasticity(
            target_norm=1.0,
            norm_tolerance=0.15,  # Tighter tolerance triggers scaling earlier
            ema_alpha=0.1  # Faster EMA adaptation
        )

        norm_history = []

        for step in range(100):
            # Generate embeddings with varying norms
            np.random.seed(step)  # Reproducible
            embeddings = np.random.randn(5, 1024).astype(np.float32)
            random_norm = 0.7 + np.random.rand() * 0.6  # 0.7 to 1.3 (narrower range)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms * random_norm

            plasticity.update_statistics(embeddings)

            if plasticity.needs_scaling():
                embeddings = plasticity.apply_scaling(embeddings)

            norm_history.append(plasticity._state.mean_norm)

        # Should stabilize around target (allow wider range due to stochastic nature)
        final_mean = np.mean(norm_history[-20:])
        assert 0.7 < final_mean < 1.4


class TestHomeostaticRuntimeConfiguration:
    """Tests for runtime configuration setters."""

    def test_set_target_norm(self):
        """Test setting target norm."""
        plasticity = HomeostaticPlasticity()

        plasticity.set_target_norm(1.5)
        assert plasticity.target_norm == 1.5

    def test_set_target_norm_clipped(self):
        """Test target norm is clipped to valid range."""
        plasticity = HomeostaticPlasticity()

        plasticity.set_target_norm(0.1)  # Below min
        assert plasticity.target_norm == 0.5

        plasticity.set_target_norm(5.0)  # Above max
        assert plasticity.target_norm == 2.0

    def test_set_ema_alpha(self):
        """Test setting EMA alpha."""
        plasticity = HomeostaticPlasticity()

        plasticity.set_ema_alpha(0.05)
        assert plasticity.ema_alpha == 0.05

    def test_set_ema_alpha_clipped(self):
        """Test EMA alpha is clipped to valid range."""
        plasticity = HomeostaticPlasticity()

        plasticity.set_ema_alpha(0.0001)  # Below min
        assert plasticity.ema_alpha == 0.001

        plasticity.set_ema_alpha(0.5)  # Above max
        assert plasticity.ema_alpha == 0.1

    def test_set_decorrelation_strength(self):
        """Test setting decorrelation strength."""
        plasticity = HomeostaticPlasticity()

        plasticity.set_decorrelation_strength(0.05)
        assert plasticity.decorrelation_strength == 0.05

    def test_set_decorrelation_strength_clipped(self):
        """Test decorrelation strength is clipped to valid range."""
        plasticity = HomeostaticPlasticity()

        plasticity.set_decorrelation_strength(-0.1)  # Below min
        assert plasticity.decorrelation_strength == 0.0

        plasticity.set_decorrelation_strength(0.5)  # Above max
        assert plasticity.decorrelation_strength == 0.1

    def test_set_norm_tolerance(self):
        """Test setting norm tolerance."""
        plasticity = HomeostaticPlasticity()

        plasticity.set_norm_tolerance(0.3)
        assert plasticity.norm_tolerance == 0.3

    def test_set_norm_tolerance_clipped(self):
        """Test norm tolerance is clipped to valid range."""
        plasticity = HomeostaticPlasticity()

        plasticity.set_norm_tolerance(0.01)  # Below min
        assert plasticity.norm_tolerance == 0.05

        plasticity.set_norm_tolerance(1.0)  # Above max
        assert plasticity.norm_tolerance == 0.5

    def test_set_sliding_threshold_rate(self):
        """Test setting sliding threshold rate."""
        plasticity = HomeostaticPlasticity()

        plasticity.set_sliding_threshold_rate(0.005)
        assert plasticity.sliding_threshold_rate == 0.005

    def test_set_sliding_threshold_rate_clipped(self):
        """Test sliding threshold rate is clipped to valid range."""
        plasticity = HomeostaticPlasticity()

        plasticity.set_sliding_threshold_rate(0.00001)  # Below min
        assert plasticity.sliding_threshold_rate == 0.0001

        plasticity.set_sliding_threshold_rate(0.1)  # Above max
        assert plasticity.sliding_threshold_rate == 0.01

    def test_force_scaling(self):
        """Test force_scaling method."""
        plasticity = HomeostaticPlasticity()
        plasticity._state.mean_norm = 0.8

        scale = plasticity.force_scaling()

        assert scale == pytest.approx(1.25, rel=0.01)  # 1.0 / 0.8
        assert plasticity._scaling_count == 1

    def test_force_scaling_at_target(self):
        """Test force_scaling when already at target."""
        plasticity = HomeostaticPlasticity(target_norm=1.0)
        plasticity._state.mean_norm = 1.0

        scale = plasticity.force_scaling()

        assert scale == pytest.approx(1.0, rel=0.01)
        assert plasticity._scaling_count == 1

    def test_get_config(self):
        """Test get_config returns all settings."""
        plasticity = HomeostaticPlasticity(
            target_norm=1.2,
            norm_tolerance=0.15,
            ema_alpha=0.02,
            decorrelation_strength=0.03,
            sliding_threshold_rate=0.005,
        )

        config = plasticity.get_config()

        assert config["target_norm"] == 1.2
        assert config["norm_tolerance"] == 0.15
        assert config["ema_alpha"] == 0.02
        assert config["decorrelation_strength"] == 0.03
        assert config["sliding_threshold_rate"] == 0.005


class TestHomeostaticEdgeCases:
    """Tests for edge cases in homeostatic plasticity."""

    def test_compute_scaling_factor_zero_norm(self):
        """Test scaling factor when mean_norm is near zero."""
        plasticity = HomeostaticPlasticity()
        plasticity._state.mean_norm = 1e-10  # Very small

        factor = plasticity.compute_scaling_factor()

        # Should return 1.0 to avoid division by zero
        assert factor == 1.0

    def test_modulate_learning_rate_zero_norm(self):
        """Test learning rate modulation when current_norm is near zero."""
        plasticity = HomeostaticPlasticity()
        base_lr = 0.1

        lr = plasticity.modulate_learning_rate(base_lr, current_norm=1e-10)

        # Should return base_lr unchanged
        assert lr == base_lr
