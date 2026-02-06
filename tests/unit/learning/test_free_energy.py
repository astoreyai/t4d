"""
Tests for Free Energy Objective (W1-01).

Implements Friston's variational free energy minimization:
F = E_q[log p(o|s)] + β·KL[q(s|o) || p(s)]

Evidence Base: Friston (2010) "The free-energy principle: a unified brain theory?"

Test Strategy (TDD):
1. Unit tests for each component (reconstruction, KL, free energy)
2. Integration test for F decreasing during wake phase
3. Edge cases and numerical stability
"""

import pytest
import numpy as np
import torch
from torch.distributions import Normal, kl_divergence


class TestFreeEnergyConfig:
    """Test FreeEnergyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from t4dm.learning.free_energy import FreeEnergyConfig

        config = FreeEnergyConfig()
        assert config.beta == 1.0
        assert config.reconstruction_weight == 1.0
        assert config.kl_weight == 0.1
        assert config.use_elbo is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from t4dm.learning.free_energy import FreeEnergyConfig

        config = FreeEnergyConfig(
            beta=2.0,
            reconstruction_weight=0.5,
            kl_weight=0.2,
            use_elbo=False,
        )
        assert config.beta == 2.0
        assert config.reconstruction_weight == 0.5
        assert config.kl_weight == 0.2
        assert config.use_elbo is False


class TestFreeEnergyResult:
    """Test FreeEnergyResult dataclass."""

    def test_result_fields(self):
        """Test result dataclass has required fields."""
        from t4dm.learning.free_energy import FreeEnergyResult

        result = FreeEnergyResult(
            free_energy=torch.tensor(1.5),
            reconstruction=torch.tensor(1.0),
            complexity=torch.tensor(0.5),
            learning_rate_scale=0.8,
        )
        assert result.free_energy.item() == 1.5
        assert result.reconstruction.item() == 1.0
        assert result.complexity.item() == 0.5
        assert result.learning_rate_scale == 0.8


class TestFreeEnergyMinimizer:
    """Test FreeEnergyMinimizer class."""

    @pytest.fixture
    def minimizer(self):
        """Create minimizer instance with default config."""
        from t4dm.learning.free_energy import FreeEnergyMinimizer

        return FreeEnergyMinimizer()

    @pytest.fixture
    def custom_minimizer(self):
        """Create minimizer with custom config."""
        from t4dm.learning.free_energy import FreeEnergyConfig, FreeEnergyMinimizer

        config = FreeEnergyConfig(
            beta=0.5,
            reconstruction_weight=2.0,
            kl_weight=0.05,
        )
        return FreeEnergyMinimizer(config)

    def test_kl_divergence_identical_distributions(self, minimizer):
        """KL term should be 0 when q == p."""
        q = Normal(torch.tensor(0.0), torch.tensor(1.0))
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        kl = kl_divergence(q, p)
        assert kl.item() < 1e-6, "KL should be ~0 when distributions are identical"

    def test_kl_divergence_different_distributions(self, minimizer):
        """KL term should be > 0 when q != p."""
        q = Normal(torch.tensor(1.0), torch.tensor(0.5))
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        kl = kl_divergence(q, p)
        assert kl.item() > 0, "KL should be > 0 when distributions differ"

    def test_reconstruction_error_zero_for_perfect_prediction(self, minimizer):
        """Reconstruction error should be 0 for perfect predictions."""
        prediction = torch.randn(1024)
        observation = prediction.clone()

        q = Normal(torch.tensor(0.0), torch.tensor(1.0))
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        result = minimizer.compute_free_energy(prediction, observation, q, p)
        assert (
            result.reconstruction.item() < 1e-6
        ), "Reconstruction error should be ~0 for identical tensors"

    def test_reconstruction_error_increases_with_noise(self, minimizer):
        """Reconstruction error should increase with prediction-observation gap."""
        prediction = torch.zeros(1024)

        q = Normal(torch.tensor(0.0), torch.tensor(1.0))
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        # Small noise
        obs_small_noise = prediction + 0.1 * torch.randn(1024)
        result_small = minimizer.compute_free_energy(prediction, obs_small_noise, q, p)

        # Large noise
        obs_large_noise = prediction + 1.0 * torch.randn(1024)
        result_large = minimizer.compute_free_energy(prediction, obs_large_noise, q, p)

        assert (
            result_large.reconstruction > result_small.reconstruction
        ), "Larger noise should give larger reconstruction error"

    def test_free_energy_composition(self, minimizer):
        """F should equal reconstruction + β·KL·kl_weight."""
        prediction = torch.randn(1024)
        observation = prediction + 0.1 * torch.randn(1024)

        q = Normal(torch.tensor(0.5), torch.tensor(0.8))
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        result = minimizer.compute_free_energy(prediction, observation, q, p)

        # Manually compute expected F
        expected_kl = kl_divergence(q, p)
        expected_F = (
            minimizer.config.reconstruction_weight * result.reconstruction
            + minimizer.config.beta * minimizer.config.kl_weight * expected_kl
        )

        torch.testing.assert_close(
            result.free_energy, expected_F, rtol=1e-4, atol=1e-4
        )

    def test_learning_rate_scale_computation(self, minimizer):
        """Learning rate scale should be in valid range."""
        prediction = torch.randn(1024)
        observation = prediction + 0.1 * torch.randn(1024)

        q = Normal(torch.tensor(0.0), torch.tensor(1.0))
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        result = minimizer.compute_free_energy(prediction, observation, q, p)

        assert 0.0 <= result.learning_rate_scale <= 2.0, (
            f"Learning rate scale {result.learning_rate_scale} should be in [0, 2]"
        )

    def test_custom_config_affects_output(self, custom_minimizer):
        """Custom config should affect free energy computation."""
        prediction = torch.randn(1024)
        observation = prediction + 0.5 * torch.randn(1024)

        q = Normal(torch.tensor(1.0), torch.tensor(0.5))
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        result = custom_minimizer.compute_free_energy(prediction, observation, q, p)

        # With custom config (reconstruction_weight=2.0), F should be different
        # than default config
        assert result.free_energy.item() > 0


class TestFreeEnergyDynamics:
    """Test free energy dynamics during wake phase."""

    def test_free_energy_decreases_during_learning(self):
        """F should decrease as predictions improve during wake phase."""
        from t4dm.learning.free_energy import FreeEnergyMinimizer

        torch.manual_seed(42)
        minimizer = FreeEnergyMinimizer()

        # Simple linear model for testing
        model = torch.nn.Linear(64, 64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Target: identity-like transformation
        observation = torch.randn(32, 64)

        F_values = []
        for step in range(100):
            optimizer.zero_grad()

            prediction = model(observation)

            # Use learned posterior approximation
            pred_mean = prediction.mean()
            pred_std = prediction.std().clamp(min=0.1)
            q = Normal(pred_mean, pred_std)
            p = Normal(torch.tensor(0.0), torch.tensor(1.0))

            result = minimizer.compute_free_energy(prediction, observation, q, p)
            F_values.append(result.free_energy.item())

            # Backprop through free energy
            result.free_energy.backward()
            optimizer.step()

        # F should trend downward (negative slope)
        F_array = np.array(F_values)
        slope = np.polyfit(range(len(F_array)), F_array, 1)[0]

        assert slope < 0, f"Free energy should decrease during wake (slope={slope:.4f})"

    def test_free_energy_stabilizes(self):
        """F should eventually stabilize (variance decreases)."""
        from t4dm.learning.free_energy import FreeEnergyMinimizer

        torch.manual_seed(42)
        minimizer = FreeEnergyMinimizer()

        model = torch.nn.Linear(32, 32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        observation = torch.randn(16, 32)

        F_values = []
        for step in range(200):
            optimizer.zero_grad()

            prediction = model(observation)
            q = Normal(prediction.mean(), prediction.std().clamp(min=0.1))
            p = Normal(torch.tensor(0.0), torch.tensor(1.0))

            result = minimizer.compute_free_energy(prediction, observation, q, p)
            F_values.append(result.free_energy.item())

            result.free_energy.backward()
            optimizer.step()

        # Variance of last 50 steps should be less than first 50
        early_var = np.var(F_values[:50])
        late_var = np.var(F_values[-50:])

        assert late_var < early_var, (
            f"F variance should decrease (early={early_var:.4f}, late={late_var:.4f})"
        )


class TestFreeEnergyNumericalStability:
    """Test numerical stability edge cases."""

    @pytest.fixture
    def minimizer(self):
        """Create minimizer instance."""
        from t4dm.learning.free_energy import FreeEnergyMinimizer

        return FreeEnergyMinimizer()

    def test_handles_zero_std_gracefully(self, minimizer):
        """Should handle near-zero std without NaN."""
        prediction = torch.zeros(100)
        observation = torch.zeros(100)

        # Very small std - should be clamped internally
        q = Normal(torch.tensor(0.0), torch.tensor(1e-8))
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        result = minimizer.compute_free_energy(prediction, observation, q, p)

        assert not torch.isnan(result.free_energy), "Should not produce NaN"
        assert not torch.isinf(result.free_energy), "Should not produce Inf"

    def test_handles_large_values(self, minimizer):
        """Should handle large tensor values without overflow."""
        prediction = torch.randn(100) * 1000
        observation = torch.randn(100) * 1000

        q = Normal(torch.tensor(0.0), torch.tensor(100.0))
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        result = minimizer.compute_free_energy(prediction, observation, q, p)

        assert not torch.isnan(result.free_energy), "Should not produce NaN"
        assert not torch.isinf(result.free_energy), "Should not produce Inf"

    def test_gradient_flow(self, minimizer):
        """Gradients should flow through free energy computation."""
        prediction = torch.randn(100, requires_grad=True)
        observation = torch.randn(100)

        q_loc = torch.tensor(0.0, requires_grad=True)
        q_scale = torch.tensor(1.0, requires_grad=True)
        q = Normal(q_loc, q_scale)
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        result = minimizer.compute_free_energy(prediction, observation, q, p)
        result.free_energy.backward()

        assert prediction.grad is not None, "Should have gradient for prediction"
        assert q_loc.grad is not None, "Should have gradient for q_loc"
        assert q_scale.grad is not None, "Should have gradient for q_scale"


class TestFreeEnergyIntegration:
    """Integration tests with other T4DM components."""

    def test_reconstruction_correlation_with_accuracy(self):
        """Reconstruction error should correlate with prediction accuracy (r > 0.9)."""
        from t4dm.learning.free_energy import FreeEnergyMinimizer

        minimizer = FreeEnergyMinimizer()

        # Generate varying noise levels
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        reconstruction_errors = []
        mse_values = []

        prediction = torch.randn(1024)
        q = Normal(torch.tensor(0.0), torch.tensor(1.0))
        p = Normal(torch.tensor(0.0), torch.tensor(1.0))

        for noise in noise_levels:
            observation = prediction + noise * torch.randn(1024)
            result = minimizer.compute_free_energy(prediction, observation, q, p)

            reconstruction_errors.append(result.reconstruction.item())
            mse_values.append(((prediction - observation) ** 2).mean().item())

        # Compute correlation
        r = np.corrcoef(reconstruction_errors, mse_values)[0, 1]

        assert r > 0.9, f"Correlation between reconstruction and MSE should be > 0.9, got {r:.4f}"
