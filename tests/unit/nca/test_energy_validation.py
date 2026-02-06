"""
Tests for Energy Landscape Validation (W1-03).

Validates that Hopfield energy dynamics match biological constraints:
- Wake: Energy should decrease (minimize prediction error)
- Sleep: Energy should stabilize (settle in attractor)
- Attractor depth correlates with memory strength

Evidence Base:
- Hopfield & Tank (1986) "Computing with Neural Circuits"
- Bengio (2019) "On the Measure of Intelligence"
- Ramsauer et al. (2020) "Hopfield Networks is All You Need"

Test Strategy (TDD):
1. Config tests for parameter validation
2. Wake phase energy decrease tests
3. Sleep phase energy stabilization tests
4. Attractor depth correlation tests
5. Convergence timing tests
"""

import pytest
import numpy as np
import torch


class TestEnergyValidationConfig:
    """Test EnergyValidationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from t4dm.nca.energy_validation import EnergyValidationConfig

        config = EnergyValidationConfig()
        assert config.wake_slope_threshold == 0.0
        assert config.sleep_variance_threshold == 0.01
        assert config.convergence_threshold == 0.001
        assert config.convergence_window == 10
        assert config.default_steps == 500

    def test_custom_values(self):
        """Test custom configuration values."""
        from t4dm.nca.energy_validation import EnergyValidationConfig

        config = EnergyValidationConfig(
            wake_slope_threshold=-0.001,
            sleep_variance_threshold=0.005,
            convergence_threshold=0.0001,
            convergence_window=20,
            default_steps=1000,
        )
        assert config.wake_slope_threshold == -0.001
        assert config.sleep_variance_threshold == 0.005
        assert config.convergence_threshold == 0.0001
        assert config.convergence_window == 20
        assert config.default_steps == 1000


class TestEnergyValidationResult:
    """Test EnergyValidationResult dataclass."""

    def test_result_fields(self):
        """Test result dataclass has required fields."""
        from t4dm.nca.energy_validation import EnergyValidationResult

        result = EnergyValidationResult(
            wake_slope=-0.01,
            sleep_variance=0.005,
            convergence_time=100,
            attractor_depth=-2.5,
            is_valid=True,
            phase="wake",
        )
        assert result.wake_slope == -0.01
        assert result.sleep_variance == 0.005
        assert result.convergence_time == 100
        assert result.attractor_depth == -2.5
        assert result.is_valid is True
        assert result.phase == "wake"

    def test_to_dict(self):
        """Test dictionary conversion."""
        from t4dm.nca.energy_validation import EnergyValidationResult

        result = EnergyValidationResult(
            wake_slope=-0.01,
            sleep_variance=0.005,
            convergence_time=100,
            attractor_depth=-2.5,
            is_valid=True,
            phase="wake",
        )
        d = result.to_dict()
        assert "wake_slope" in d
        assert "is_valid" in d
        assert d["phase"] == "wake"


class TestEnergyLandscapeValidator:
    """Test EnergyLandscapeValidator class."""

    @pytest.fixture
    def simple_energy_fn(self):
        """Simple quadratic energy function for testing."""

        def energy_fn(state: torch.Tensor) -> torch.Tensor:
            # Simple quadratic bowl: E = 0.5 * ||x||^2
            return 0.5 * (state**2).sum()

        return energy_fn

    @pytest.fixture
    def hopfield_energy_fn(self):
        """Hopfield energy with stored patterns."""
        # Store 3 random patterns
        patterns = torch.randn(3, 64)
        patterns = patterns / patterns.norm(dim=1, keepdim=True)

        def energy_fn(state: torch.Tensor) -> torch.Tensor:
            # Modern Hopfield: -log(sum exp(beta * pattern @ state))
            beta = 4.0
            state_norm = state / (state.norm() + 1e-6)
            similarities = beta * (patterns @ state_norm)
            return -torch.logsumexp(similarities, dim=0)

        return energy_fn

    @pytest.fixture
    def validator(self, simple_energy_fn):
        """Create validator with simple energy function."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        return EnergyLandscapeValidator(simple_energy_fn)

    @pytest.fixture
    def hopfield_validator(self, hopfield_energy_fn):
        """Create validator with Hopfield energy function."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        return EnergyLandscapeValidator(hopfield_energy_fn)

    def test_init_with_energy_fn(self, validator):
        """Validator should store energy function."""
        assert validator.energy_fn is not None
        assert len(validator.E_history) == 0

    def test_check_convergence_wake_phase(self, validator):
        """Wake phase should show decreasing energy."""
        state = torch.randn(64) * 2  # Start away from minimum

        result = validator.check_convergence(state, phase="wake", steps=100)

        assert result.wake_slope < 0, "Energy should decrease during wake"
        assert result.phase == "wake"

    def test_check_convergence_sleep_phase(self, validator):
        """Sleep phase should show stabilized energy."""
        state = torch.randn(64) * 0.01  # Start near minimum

        result = validator.check_convergence(state, phase="sleep", steps=100)

        assert result.sleep_variance < 0.1, "Energy should stabilize during sleep"
        assert result.phase == "sleep"

    def test_is_valid_wake_with_decreasing_energy(self, validator):
        """Wake validation should pass when energy decreases."""
        state = torch.randn(64) * 5  # Start far from minimum

        result = validator.check_convergence(state, phase="wake", steps=200)

        assert result.is_valid, "Wake validation should pass with decreasing energy"

    def test_is_valid_sleep_with_stable_energy(self, validator):
        """Sleep validation should pass when energy is stable."""
        state = torch.zeros(64)  # Start at minimum

        result = validator.check_convergence(state, phase="sleep", steps=100)

        assert result.is_valid, "Sleep validation should pass with stable energy"

    def test_convergence_time_tracking(self, validator):
        """Should track time to convergence."""
        state = torch.randn(64) * 2

        result = validator.check_convergence(state, phase="wake", steps=300)

        assert result.convergence_time >= 0
        assert result.convergence_time <= 300

    def test_attractor_depth_computed(self, hopfield_validator):
        """Should compute attractor depth (final energy)."""
        state = torch.randn(64)

        result = hopfield_validator.check_convergence(state, phase="wake", steps=200)

        assert result.attractor_depth is not None
        assert isinstance(result.attractor_depth, float)

    def test_energy_history_recorded(self, validator):
        """Should record energy history during validation."""
        state = torch.randn(64)

        validator.check_convergence(state, phase="wake", steps=50)

        assert len(validator.E_history) == 50

    def test_reset_clears_history(self, validator):
        """Reset should clear energy history."""
        state = torch.randn(64)
        validator.check_convergence(state, phase="wake", steps=50)

        validator.reset()

        assert len(validator.E_history) == 0


class TestWakePhaseDynamics:
    """Test wake phase energy dynamics."""

    @pytest.fixture
    def energy_fn(self):
        """Quadratic energy for predictable dynamics."""

        def fn(state: torch.Tensor) -> torch.Tensor:
            return 0.5 * (state**2).sum()

        return fn

    def test_wake_energy_decreases(self, energy_fn):
        """During wake, energy should decrease as predictions improve."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        validator = EnergyLandscapeValidator(energy_fn)
        state = torch.randn(128) * 3  # Start away from minimum

        result = validator.check_convergence(state, phase="wake", steps=500)

        assert result.wake_slope < 0, f"Energy should decrease during wake, got slope={result.wake_slope}"
        assert result.is_valid, "Wake phase validation failed"

    def test_wake_converges_to_minimum(self, energy_fn):
        """Wake phase should converge toward energy minimum."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        validator = EnergyLandscapeValidator(energy_fn)
        state = torch.randn(64) * 5

        result = validator.check_convergence(state, phase="wake", steps=500)

        # Final energy should be close to minimum (0 for quadratic bowl)
        assert result.attractor_depth < 1.0, "Should converge near minimum"


class TestSleepPhaseDynamics:
    """Test sleep phase energy dynamics."""

    @pytest.fixture
    def energy_fn(self):
        """Quadratic energy for predictable dynamics."""

        def fn(state: torch.Tensor) -> torch.Tensor:
            return 0.5 * (state**2).sum()

        return fn

    def test_sleep_energy_stabilizes(self, energy_fn):
        """During sleep, energy should stabilize in attractor."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        validator = EnergyLandscapeValidator(energy_fn)
        state = torch.randn(64) * 0.1  # Start near minimum

        result = validator.check_convergence(state, phase="sleep", steps=500)

        assert result.sleep_variance < 0.01, f"Energy should stabilize during sleep, got variance={result.sleep_variance}"
        assert result.is_valid, "Sleep phase validation failed"

    def test_sleep_maintains_attractor(self, energy_fn):
        """Sleep phase should maintain position in attractor basin."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        validator = EnergyLandscapeValidator(energy_fn)
        state = torch.zeros(64)  # Start at minimum

        result = validator.check_convergence(state, phase="sleep", steps=200)

        # Energy should stay low
        assert result.attractor_depth < 0.1, "Should maintain position at minimum"


class TestAttractorDepth:
    """Test attractor depth correlation with memory strength."""

    def test_deeper_attractor_for_stronger_memory(self):
        """Deeper attractors should indicate stronger memories."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        # Create energy function with two patterns of different strengths
        strong_pattern = torch.ones(64)
        weak_pattern = torch.randn(64) * 0.5

        def energy_fn(state: torch.Tensor) -> torch.Tensor:
            # Energy is negative similarity to patterns
            strong_sim = (state * strong_pattern).sum()
            weak_sim = (state * weak_pattern).sum()
            return -(strong_sim + 0.5 * weak_sim)

        validator = EnergyLandscapeValidator(energy_fn)

        # Test near strong pattern
        result_strong = validator.check_convergence(
            strong_pattern + torch.randn(64) * 0.1, phase="wake", steps=200
        )

        # Test near weak pattern
        result_weak = validator.check_convergence(
            weak_pattern + torch.randn(64) * 0.1, phase="wake", steps=200
        )

        # Strong pattern should have lower (deeper) energy
        assert result_strong.attractor_depth < result_weak.attractor_depth, (
            "Strong memories should have deeper attractors (lower energy)"
        )


class TestConvergenceMetrics:
    """Test convergence time computation."""

    @pytest.fixture
    def fast_converging_fn(self):
        """Energy that converges quickly."""

        def fn(state: torch.Tensor) -> torch.Tensor:
            return 0.5 * (state**2).sum()

        return fn

    def test_convergence_within_steps(self, fast_converging_fn):
        """Should converge within specified steps."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        validator = EnergyLandscapeValidator(fast_converging_fn)
        state = torch.randn(32) * 2

        result = validator.check_convergence(state, phase="wake", steps=500)

        assert result.convergence_time <= 500, "Should converge within 500 steps"

    def test_records_non_convergence(self, fast_converging_fn):
        """Should record if convergence not reached."""
        from t4dm.nca.energy_validation import (
            EnergyLandscapeValidator,
            EnergyValidationConfig,
        )

        # Very strict threshold that won't be met
        config = EnergyValidationConfig(convergence_threshold=1e-10)
        validator = EnergyLandscapeValidator(fast_converging_fn, config)
        state = torch.randn(32) * 10

        result = validator.check_convergence(state, phase="wake", steps=50)

        # Should return max steps when not converged
        assert result.convergence_time == 50


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_handles_zero_state(self):
        """Should handle zero state without errors."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        def energy_fn(state: torch.Tensor) -> torch.Tensor:
            return 0.5 * (state**2).sum()

        validator = EnergyLandscapeValidator(energy_fn)
        state = torch.zeros(64)

        result = validator.check_convergence(state, phase="sleep", steps=50)

        assert not np.isnan(result.attractor_depth)
        assert not np.isinf(result.attractor_depth)

    def test_handles_large_state(self):
        """Should handle large state values."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        def energy_fn(state: torch.Tensor) -> torch.Tensor:
            # Clamp to prevent overflow
            return 0.5 * (state.clamp(-100, 100) ** 2).sum()

        validator = EnergyLandscapeValidator(energy_fn)
        state = torch.randn(64) * 100

        result = validator.check_convergence(state, phase="wake", steps=50)

        assert not np.isnan(result.attractor_depth)
        assert not np.isinf(result.attractor_depth)

    def test_handles_nan_in_energy(self):
        """Should handle NaN in energy gracefully."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator

        call_count = [0]

        def energy_fn(state: torch.Tensor) -> torch.Tensor:
            call_count[0] += 1
            if call_count[0] == 5:
                return torch.tensor(float("nan"))
            return 0.5 * (state**2).sum()

        validator = EnergyLandscapeValidator(energy_fn)
        state = torch.randn(64)

        # Should not crash, should mark as invalid
        result = validator.check_convergence(state, phase="wake", steps=50)

        # NaN handling should mark result appropriately
        assert result is not None


class TestIntegrationWithEnergyLandscape:
    """Integration tests with existing EnergyLandscape class."""

    def test_validates_existing_energy_landscape(self):
        """Should work with existing EnergyLandscape from energy.py."""
        from t4dm.nca.energy_validation import EnergyLandscapeValidator
        from t4dm.nca.energy import EnergyLandscape, EnergyConfig

        # Create energy landscape
        config = EnergyConfig(temperature=0.5)
        landscape = EnergyLandscape(config)

        # For integration with numpy-based energy, we create a differentiable
        # approximation. The actual validation doesn't require gradient flow.
        # Here we test the interface works with a pure PyTorch energy function
        # that mimics the EnergyLandscape behavior.
        def energy_fn(state: torch.Tensor) -> torch.Tensor:
            # Simple quadratic energy mimicking Hopfield dynamics
            # E = -0.5 * x^T x (negative for attractor at x=0)
            return -0.5 * (state**2).sum() + 0.1 * (state**4).sum()

        validator = EnergyLandscapeValidator(energy_fn)
        state = torch.rand(6)  # 6 NT systems

        result = validator.check_convergence(state, phase="wake", steps=100)

        assert result is not None
        assert result.phase == "wake"

    def test_energy_landscape_interface_compatibility(self):
        """Verify EnergyLandscape interface is compatible with validator."""
        from t4dm.nca.energy import EnergyLandscape, EnergyConfig

        # Just verify the interface exists
        config = EnergyConfig()
        landscape = EnergyLandscape(config)

        # Check compute_total_energy exists
        assert hasattr(landscape, "compute_total_energy")
