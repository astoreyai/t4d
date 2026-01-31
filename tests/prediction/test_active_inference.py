"""
Tests for Active Inference and Predictive Coding module.

Tests hierarchical prediction, precision weighting, and free energy minimization.
"""

import numpy as np
import pytest

from t4dm.prediction.active_inference import (
    ActiveInferenceConfig,
    BeliefState,
    PrecisionWeightedLevel,
)


class TestActiveInferenceConfig:
    """Tests for ActiveInferenceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ActiveInferenceConfig()
        assert config.n_levels == 4
        assert config.dims == [1024, 512, 256, 128]
        assert config.precision_lr == 0.01
        assert config.belief_lr == 0.005
        assert config.free_energy_threshold == 0.1
        assert config.temporal_horizon == 5
        assert config.precision_prior == 1.0
        assert config.action_dim == 32

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ActiveInferenceConfig(
            n_levels=3,
            dims=[512, 256, 128],
            precision_lr=0.02,
        )
        assert config.n_levels == 3
        assert config.dims == [512, 256, 128]
        assert config.precision_lr == 0.02


class TestBeliefState:
    """Tests for BeliefState dataclass."""

    def test_belief_state_creation(self):
        """Test creating a belief state."""
        mean = np.random.randn(64).astype(np.float32)
        precision = np.ones(64, dtype=np.float32)
        prediction = np.random.randn(128).astype(np.float32)
        error = np.random.randn(128).astype(np.float32)

        state = BeliefState(
            mean=mean,
            precision=precision,
            prediction=prediction,
            error=error,
            free_energy=0.5,
        )

        assert np.array_equal(state.mean, mean)
        assert np.array_equal(state.precision, precision)
        assert state.free_energy == 0.5

    def test_default_free_energy(self):
        """Test default free energy is zero."""
        state = BeliefState(
            mean=np.zeros(10),
            precision=np.ones(10),
            prediction=np.zeros(20),
            error=np.zeros(20),
        )
        assert state.free_energy == 0.0


class TestPrecisionWeightedLevel:
    """Tests for PrecisionWeightedLevel."""

    @pytest.fixture
    def level(self):
        """Create prediction level."""
        return PrecisionWeightedLevel(
            dim=64,
            dim_below=128,
            precision_prior=1.0,
            learning_rate=0.01,
        )

    def test_initialization(self, level):
        """Test level initialization."""
        assert level.dim == 64
        assert level.dim_below == 128
        assert level.W_gen.shape == (128, 64)
        assert level.W_rec.shape == (64, 128)
        assert level.mu.shape == (64,)
        assert level.precision.shape == (128,)

    def test_compute_prediction(self, level):
        """Test prediction computation."""
        # Set some belief state
        level.mu = np.random.randn(64).astype(np.float32)

        prediction = level.compute_prediction()

        assert prediction.shape == (128,)
        # Prediction uses tanh, so bounded
        assert np.all(prediction >= -1)
        assert np.all(prediction <= 1)

    def test_compute_prediction_error(self, level):
        """Test prediction error computation."""
        level.mu = np.random.randn(64).astype(np.float32)
        target = np.random.randn(128).astype(np.float32)

        error = level.compute_prediction_error(target)

        assert error.shape == (128,)
        # Error should be target - prediction
        expected_prediction = np.tanh(level.W_gen @ level.mu)
        expected_error = target - expected_prediction
        np.testing.assert_array_almost_equal(error, expected_error)

    def test_belief_update(self, level):
        """Test belief update from prediction error."""
        level.mu = np.random.randn(64).astype(np.float32)
        target = np.random.randn(128).astype(np.float32)

        # Compute error first
        error = level.compute_prediction_error(target)
        mu_before = level.mu.copy()
        level.belief_update(error)

        # Beliefs should change
        assert not np.allclose(level.mu, mu_before)

    def test_update_precision(self, level):
        """Test precision update from error variance."""
        level.precision = np.ones(128, dtype=np.float32)

        # Small errors -> high precision
        small_error = np.ones(128, dtype=np.float32) * 0.01
        level.update_precision(small_error)
        high_precision = level.precision.mean()

        # Reset
        level.precision = np.ones(128, dtype=np.float32)

        # Large errors -> low precision
        large_error = np.ones(128, dtype=np.float32) * 10.0
        level.update_precision(large_error)
        low_precision = level.precision.mean()

        # Large errors should reduce precision more
        # (precision inversely related to error variance)
        assert low_precision < high_precision

    def test_compute_free_energy(self, level):
        """Test free energy computation."""
        level.mu = np.random.randn(64).astype(np.float32)
        target = np.random.randn(128).astype(np.float32)

        error = level.compute_prediction_error(target)
        fe = level.compute_free_energy(error)

        # Free energy should be non-negative
        assert fe >= 0

    def test_get_state(self, level):
        """Test getting current belief state."""
        level.mu = np.random.randn(64).astype(np.float32)

        state = level.get_state()

        assert isinstance(state, BeliefState)
        assert state.mean.shape == (64,)
        assert state.prediction.shape == (128,)
        assert state.precision.shape == (128,)


class TestActiveInferenceHierarchy:
    """Tests for ActiveInferenceHierarchy (if it exists)."""

    def test_hierarchical_prediction_flow(self):
        """Test prediction flows down hierarchy."""
        # Create a simple 2-level hierarchy
        level_high = PrecisionWeightedLevel(dim=32, dim_below=64)
        level_low = PrecisionWeightedLevel(dim=64, dim_below=128)

        # High level predicts for low level
        level_high.mu = np.random.randn(32).astype(np.float32)
        high_prediction = level_high.compute_prediction()

        # Low level receives high prediction as prior
        level_low.mu = high_prediction

        # Low level predicts for sensory
        low_prediction = level_low.compute_prediction()

        assert low_prediction.shape == (128,)

    def test_error_flows_up_hierarchy(self):
        """Test prediction errors flow up hierarchy."""
        level_low = PrecisionWeightedLevel(dim=64, dim_below=128)
        level_high = PrecisionWeightedLevel(dim=32, dim_below=64)

        # Sensory input creates error at low level
        sensory = np.random.randn(128).astype(np.float32)
        low_error = level_low.compute_prediction_error(sensory)

        # Low level updates beliefs
        mu_before_low = level_low.mu.copy()
        level_low.belief_update(low_error)
        assert not np.allclose(level_low.mu, mu_before_low)

        # High level receives low level's beliefs as input
        high_error = level_high.compute_prediction_error(level_low.mu)

        # High level updates
        mu_before_high = level_high.mu.copy()
        level_high.belief_update(high_error)
        assert not np.allclose(level_high.mu, mu_before_high)


class TestActiveInferenceIntegration:
    """Integration tests for active inference."""

    def test_prediction_error_minimization(self):
        """Test that belief updates minimize prediction error."""
        level = PrecisionWeightedLevel(dim=64, dim_below=128)

        # Fixed target
        target = np.random.randn(128).astype(np.float32)

        # Track free energy over updates
        free_energies = []
        for _ in range(50):
            error = level.compute_prediction_error(target)
            fe = level.compute_free_energy(error)
            free_energies.append(fe)
            level.belief_update(error)

        # Free energy should generally decrease
        # (may not be monotonic due to learning dynamics)
        assert free_energies[-1] < free_energies[0]

    def test_precision_adapts_to_input_reliability(self):
        """Test that precision adapts to input statistics."""
        level = PrecisionWeightedLevel(dim=64, dim_below=128, learning_rate=0.1)

        # Reliable input (low variance)
        reliable_target = np.ones(128, dtype=np.float32) * 0.5
        for _ in range(20):
            error = level.compute_prediction_error(reliable_target)
            level.update_precision(error)
            level.belief_update(error)

        precision_reliable = level.precision.mean()

        # Reset
        level = PrecisionWeightedLevel(dim=64, dim_below=128, learning_rate=0.1)

        # Unreliable input (high variance noise)
        for _ in range(20):
            noisy_target = np.random.randn(128).astype(np.float32) * 5
            error = level.compute_prediction_error(noisy_target)
            level.update_precision(error)
            level.belief_update(error)

        precision_unreliable = level.precision.mean()

        # Precision should be lower for unreliable inputs
        # This tests that the system learns about input reliability
        assert precision_unreliable < precision_reliable

    def test_belief_convergence(self):
        """Test beliefs converge to stable state."""
        level = PrecisionWeightedLevel(dim=32, dim_below=64)

        # Fixed input
        target = np.tanh(np.random.randn(64).astype(np.float32))

        # Many updates
        for _ in range(100):
            error = level.compute_prediction_error(target)
            level.belief_update(error)

        # Compute final prediction error
        final_error = level.compute_prediction_error(target)
        error_magnitude = np.mean(final_error ** 2)

        # Error should be small after convergence
        assert error_magnitude < 1.0
