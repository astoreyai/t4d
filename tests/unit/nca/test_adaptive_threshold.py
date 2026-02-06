"""
Tests for Adaptive Layer Thresholds (W1-02).

Implements Hinton's Forward-Forward Algorithm extension with homeostatic
threshold adaptation per layer. Thresholds adapt to maintain target firing rate.

Evidence Base: Hinton (2022) "The Forward-Forward Algorithm"

Test Strategy (TDD):
1. Config tests for parameter validation
2. Single threshold adaptation tests
3. Per-layer independence tests
4. Convergence to target firing rate
5. Boundary condition tests
"""

import pytest
import numpy as np


class TestAdaptiveThresholdConfig:
    """Test AdaptiveThresholdConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from t4dm.nca.adaptive_threshold import AdaptiveThresholdConfig

        config = AdaptiveThresholdConfig()
        assert config.target_firing_rate == 0.15
        assert config.adaptation_rate == 0.01
        assert config.min_threshold == 0.1
        assert config.max_threshold == 10.0
        assert config.window_size == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        from t4dm.nca.adaptive_threshold import AdaptiveThresholdConfig

        config = AdaptiveThresholdConfig(
            target_firing_rate=0.2,
            adaptation_rate=0.05,
            min_threshold=0.5,
            max_threshold=5.0,
            window_size=50,
        )
        assert config.target_firing_rate == 0.2
        assert config.adaptation_rate == 0.05
        assert config.min_threshold == 0.5
        assert config.max_threshold == 5.0
        assert config.window_size == 50


class TestAdaptiveThreshold:
    """Test AdaptiveThreshold class."""

    @pytest.fixture
    def threshold(self):
        """Create default adaptive threshold."""
        from t4dm.nca.adaptive_threshold import AdaptiveThreshold

        return AdaptiveThreshold()

    @pytest.fixture
    def fast_threshold(self):
        """Create fast-adapting threshold for testing."""
        from t4dm.nca.adaptive_threshold import (
            AdaptiveThreshold,
            AdaptiveThresholdConfig,
        )

        config = AdaptiveThresholdConfig(
            adaptation_rate=0.1,  # Fast adaptation
            window_size=10,
        )
        return AdaptiveThreshold(config)

    def test_initial_theta(self, threshold):
        """Initial threshold should be 1.0."""
        assert threshold.theta == 1.0

    def test_update_returns_theta(self, threshold):
        """Update should return current theta."""
        import torch

        goodness = torch.randn(100) * 2
        theta = threshold.update(goodness)
        assert theta == threshold.theta

    def test_threshold_increases_when_firing_rate_high(self, fast_threshold):
        """When firing rate > target, threshold should increase."""
        import torch

        initial_theta = fast_threshold.theta

        # High goodness values → high firing rate → increase theta
        for _ in range(50):
            goodness = torch.ones(100) * 10  # All above threshold
            fast_threshold.update(goodness)

        assert fast_threshold.theta > initial_theta, (
            f"Theta should increase from {initial_theta} when firing rate is high"
        )

    def test_threshold_decreases_when_firing_rate_low(self, fast_threshold):
        """When firing rate < target, threshold should decrease."""
        import torch

        # First set theta high
        fast_threshold.theta = 5.0
        initial_theta = fast_threshold.theta

        # Low goodness values → low firing rate → decrease theta
        for _ in range(50):
            goodness = torch.zeros(100)  # All below threshold
            fast_threshold.update(goodness)

        assert fast_threshold.theta < initial_theta, (
            f"Theta should decrease from {initial_theta} when firing rate is low"
        )

    def test_threshold_stays_in_bounds_upper(self):
        """Threshold should not exceed max_threshold."""
        from t4dm.nca.adaptive_threshold import (
            AdaptiveThreshold,
            AdaptiveThresholdConfig,
        )
        import torch

        config = AdaptiveThresholdConfig(
            min_threshold=0.1,
            max_threshold=10.0,
            adaptation_rate=0.5,  # Very fast
            window_size=5,
        )
        threshold = AdaptiveThreshold(config)

        # Extreme high goodness → should try to increase theta massively
        for _ in range(1000):
            goodness = torch.ones(100) * 1000
            threshold.update(goodness)

        assert threshold.theta <= 10.0, f"Threshold {threshold.theta} exceeded max 10.0"

    def test_threshold_stays_in_bounds_lower(self):
        """Threshold should not go below min_threshold."""
        from t4dm.nca.adaptive_threshold import (
            AdaptiveThreshold,
            AdaptiveThresholdConfig,
        )
        import torch

        config = AdaptiveThresholdConfig(
            min_threshold=0.1,
            max_threshold=10.0,
            adaptation_rate=0.5,  # Very fast
            window_size=5,
        )
        threshold = AdaptiveThreshold(config)

        # Zero goodness → should try to decrease theta massively
        for _ in range(1000):
            goodness = torch.zeros(100)
            threshold.update(goodness)

        assert threshold.theta >= 0.1, f"Threshold {threshold.theta} below min 0.1"

    def test_threshold_converges_to_target_rate(self):
        """Threshold should adapt until firing rate approximates target."""
        from t4dm.nca.adaptive_threshold import (
            AdaptiveThreshold,
            AdaptiveThresholdConfig,
        )
        import torch

        config = AdaptiveThresholdConfig(
            target_firing_rate=0.15,
            adaptation_rate=0.02,
            window_size=50,
        )
        threshold = AdaptiveThreshold(config)

        # Simulate 1000 steps with random goodness values
        np.random.seed(42)
        for _ in range(1000):
            # Goodness from standard normal * 2 → varied firing rates
            goodness = torch.randn(100) * 2
            threshold.update(goodness)

        # Measure final firing rate
        final_rates = []
        for _ in range(100):
            goodness = torch.randn(100) * 2
            fired = (goodness > threshold.theta).float().mean().item()
            final_rates.append(fired)

        avg_final_rate = np.mean(final_rates)
        assert abs(avg_final_rate - 0.15) < 0.05, (
            f"Firing rate {avg_final_rate:.3f} not within ±5% of target 0.15"
        )

    def test_reset_clears_history(self, threshold):
        """Reset should clear firing history."""
        import torch

        # Add some history
        for _ in range(20):
            threshold.update(torch.randn(100))

        assert len(threshold.firing_history) > 0

        threshold.reset()

        assert len(threshold.firing_history) == 0
        assert threshold.theta == 1.0

    def test_get_stats(self, threshold):
        """get_stats should return useful metrics."""
        import torch

        for _ in range(50):
            threshold.update(torch.randn(100))

        stats = threshold.get_stats()

        assert "theta" in stats
        assert "avg_firing_rate" in stats
        assert "target_firing_rate" in stats
        assert "updates" in stats


class TestAdaptiveThresholdManager:
    """Test AdaptiveThresholdManager for multi-layer networks."""

    @pytest.fixture
    def manager(self):
        """Create manager with 3 layers."""
        from t4dm.nca.adaptive_threshold import AdaptiveThresholdManager

        return AdaptiveThresholdManager(num_layers=3)

    def test_creates_per_layer_thresholds(self, manager):
        """Should create independent threshold per layer."""
        assert len(manager.thresholds) == 3
        assert all(t.theta == 1.0 for t in manager.thresholds)

    def test_update_layer_modifies_correct_threshold(self, manager):
        """update_layer should only modify specified layer."""
        import torch

        # Update layer 0 with high goodness
        for _ in range(100):
            goodness = torch.ones(100) * 10
            manager.update_layer(0, goodness)

        # Update layer 2 with low goodness
        for _ in range(100):
            goodness = torch.zeros(100)
            manager.update_layer(2, goodness)

        # Layer 0 should have increased theta
        # Layer 1 should be unchanged (initial)
        # Layer 2 should have decreased theta (but clamped to min)
        assert manager.thresholds[0].theta > 1.0
        assert manager.thresholds[1].theta == 1.0
        assert manager.thresholds[2].theta < 1.0 or manager.thresholds[2].theta == 0.1

    def test_per_layer_independence(self, manager):
        """Each layer should adapt independently."""
        import torch

        # Different inputs to different layers
        for i in range(100):
            # Layer 0: high goodness
            manager.update_layer(0, torch.ones(100) * 5)
            # Layer 1: medium goodness
            manager.update_layer(1, torch.randn(100) * 2)
            # Layer 2: low goodness
            manager.update_layer(2, torch.ones(100) * 0.1)

        # All thresholds should differ
        thetas = [t.theta for t in manager.thresholds]
        assert thetas[0] != thetas[1] or thetas[1] != thetas[2], (
            f"Thresholds should differ: {thetas}"
        )

    def test_get_theta(self, manager):
        """get_theta should return layer-specific theta."""
        import torch

        # Modify layer 1
        for _ in range(100):
            manager.update_layer(1, torch.ones(100) * 10)

        theta_1 = manager.get_theta(1)
        theta_0 = manager.get_theta(0)

        assert theta_1 > theta_0, "Modified layer should have different theta"

    def test_get_all_stats(self, manager):
        """get_all_stats should return stats for all layers."""
        import torch

        for layer in range(3):
            for _ in range(50):
                manager.update_layer(layer, torch.randn(100))

        stats = manager.get_all_stats()

        assert len(stats) == 3
        for layer_stats in stats:
            assert "theta" in layer_stats
            assert "avg_firing_rate" in layer_stats


class TestIntegrationWithForwardForward:
    """Integration tests with ForwardForward layer."""

    def test_ff_layer_uses_adaptive_threshold(self):
        """ForwardForward layer should use adaptive threshold when enabled."""
        from t4dm.nca.adaptive_threshold import AdaptiveThresholdConfig
        from t4dm.nca.forward_forward import ForwardForwardConfig, ForwardForwardLayer

        config = ForwardForwardConfig(
            input_dim=64,
            hidden_dim=32,
            use_adaptive_threshold=True,
            adaptive_threshold_config=AdaptiveThresholdConfig(
                target_firing_rate=0.2,
                adaptation_rate=0.05,
            ),
        )

        layer = ForwardForwardLayer(config)

        # Layer should have adaptive threshold
        assert layer.adaptive_threshold is not None
        assert layer.adaptive_threshold.config.target_firing_rate == 0.2


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_handles_empty_batch(self):
        """Should handle empty goodness tensor."""
        from t4dm.nca.adaptive_threshold import AdaptiveThreshold
        import torch

        threshold = AdaptiveThreshold()

        # Empty tensor should not crash
        goodness = torch.tensor([])
        theta = threshold.update(goodness)

        assert not np.isnan(theta)
        assert not np.isinf(theta)

    def test_handles_nan_values(self):
        """Should handle NaN in goodness gracefully."""
        from t4dm.nca.adaptive_threshold import AdaptiveThreshold
        import torch

        threshold = AdaptiveThreshold()

        # NaN values should be filtered
        goodness = torch.tensor([1.0, float("nan"), 2.0])
        theta = threshold.update(goodness)

        assert not np.isnan(theta)

    def test_handles_extreme_values(self):
        """Should handle extreme goodness values."""
        from t4dm.nca.adaptive_threshold import AdaptiveThreshold
        import torch

        threshold = AdaptiveThreshold()

        # Very large values
        goodness = torch.ones(100) * 1e10
        theta = threshold.update(goodness)

        assert not np.isnan(theta)
        assert not np.isinf(theta)
        assert threshold.config.min_threshold <= theta <= threshold.config.max_threshold
