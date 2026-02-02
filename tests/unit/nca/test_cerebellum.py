"""Tests for the cerebellar circuit module."""

import numpy as np
import pytest

from t4dm.nca.cerebellum import (
    CerebellarConfig,
    CerebellarModule,
    CerebellarState,
    GranuleCellLayer,
    PurkinjeCellLayer,
    create_cerebellar_module,
)


@pytest.fixture
def config():
    """Small config for fast tests."""
    return CerebellarConfig(
        input_dim=32,
        granule_expansion=10,
        num_purkinje=20,
        dcn_dim=16,
        timing_bins=10,
    )


@pytest.fixture
def cerebellum(config):
    return CerebellarModule(config, random_seed=42)


class TestForwardPass:
    """Test basic forward pass through the circuit."""

    def test_forward_shape(self, cerebellum, config):
        mossy = np.random.randn(config.input_dim).astype(np.float32)
        state = cerebellum.forward(mossy)

        assert isinstance(state, CerebellarState)
        assert state.purkinje_output.shape == (config.num_purkinje,)
        assert state.dcn_output.shape == (config.dcn_dim,)

    def test_forward_with_error(self, cerebellum, config):
        mossy = np.random.randn(config.input_dim).astype(np.float32)
        error = np.random.randn(config.num_purkinje).astype(np.float32) * 0.1

        state = cerebellum.forward(mossy, climbing_fiber_error=error)
        assert state.prediction_error > 0.0

    def test_forward_no_error_zero_prediction_error(self, cerebellum, config):
        mossy = np.random.randn(config.input_dim).astype(np.float32)
        state = cerebellum.forward(mossy)
        assert state.prediction_error == 0.0


class TestGranuleExpansion:
    """Test granule cell expansion ratio."""

    def test_expansion_ratio(self, config):
        assert config.granule_dim == config.input_dim * config.granule_expansion

    def test_granule_output_shape(self, config):
        layer = GranuleCellLayer(config, seed=42)
        mossy = np.random.randn(config.input_dim).astype(np.float32)
        output = layer.process(mossy)
        assert output.shape == (config.granule_dim,)

    def test_granule_sparsity(self, config):
        layer = GranuleCellLayer(config, seed=42)
        mossy = np.random.randn(config.input_dim).astype(np.float32)
        output = layer.process(mossy)

        active_fraction = np.count_nonzero(output) / len(output)
        # Should be close to configured sparsity (within 2x tolerance)
        assert active_fraction <= config.granule_sparsity * 2.0


class TestErrorDrivenLearning:
    """Test that CF error-driven learning reduces prediction error."""

    def test_learning_reduces_error(self):
        cfg = CerebellarConfig(
            input_dim=32,
            granule_expansion=10,
            num_purkinje=20,
            dcn_dim=16,
            timing_bins=10,
            learning_rate=0.05,  # Higher LR for test convergence
        )
        cerebellum = CerebellarModule(cfg, random_seed=42)
        rng = np.random.default_rng(123)

        # Fixed input-output pair to learn
        mossy = rng.standard_normal(cfg.input_dim).astype(np.float32)
        target_pc = np.full(cfg.num_purkinje, 0.3, dtype=np.float32)

        errors = []
        for _ in range(200):
            state = cerebellum.forward(mossy)
            error = state.purkinje_output - target_pc  # CF signals "too high"
            cerebellum.forward(mossy, climbing_fiber_error=error)
            errors.append(float(np.mean(np.abs(error))))

        # Error should decrease over training
        assert np.mean(errors[-20:]) < np.mean(errors[:20])

    def test_ltd_weight_decrease(self):
        """Test that positive error leads to weight decrease (LTD)."""
        # Use config with no clamping floor so we can see decrease
        cfg = CerebellarConfig(
            input_dim=32, granule_expansion=10, num_purkinje=20,
            dcn_dim=16, timing_bins=10,
            min_weight=-10.0, max_weight=10.0,  # Wide bounds
        )
        layer = PurkinjeCellLayer(cfg, seed=42)
        granule = np.abs(np.random.default_rng(42).standard_normal(
            cfg.granule_dim
        ).astype(np.float32))  # All positive activations

        # Process to set eligibility
        layer.process(granule)
        weights_before = layer.weights.copy()

        # Positive error -> LTD (weight decrease): delta_W = -lr * outer(g, e)
        error = np.ones(cfg.num_purkinje, dtype=np.float32) * 0.5
        layer.update_weights(error)
        weights_after = layer.weights

        # With positive granule and positive error, LTD should decrease weights
        assert np.sum(weights_after) < np.sum(weights_before)


class TestTimingPrediction:
    """Test interval timing estimation."""

    def test_timing_returns_positive(self, cerebellum, config):
        context = np.random.randn(config.input_dim).astype(np.float32)
        timing = cerebellum.predict_timing(context)
        assert isinstance(timing, float)
        assert timing > 0.0

    def test_timing_from_forward(self, cerebellum, config):
        mossy = np.random.randn(config.input_dim).astype(np.float32)
        state = cerebellum.forward(mossy)
        assert state.timing_estimate > 0.0


class TestForwardModel:
    """Test predictive forward model."""

    def test_predict_outcome_shape(self, cerebellum):
        state = np.random.randn(16).astype(np.float32)
        action = np.random.randn(8).astype(np.float32)
        predicted = cerebellum.predict_outcome(state, action)
        assert predicted.shape == (16,)  # Same as state dim

    def test_update_returns_error(self, cerebellum):
        predicted = np.zeros(16, dtype=np.float32)
        actual = np.ones(16, dtype=np.float32)
        error = cerebellum.update(predicted, actual)
        assert error > 0.0

    def test_forward_model_learning(self, cerebellum):
        """Test that repeated updates reduce forward model error."""
        rng = np.random.default_rng(99)
        state = rng.standard_normal(16).astype(np.float32)
        action = rng.standard_normal(8).astype(np.float32)
        actual = np.tanh(state * 0.5)  # Target

        errors = []
        for _ in range(100):
            predicted = cerebellum.predict_outcome(state, action)
            err = cerebellum.update(predicted, actual)
            errors.append(err)

        # Error should decrease (bias learning at minimum)
        assert np.mean(errors[-20:]) < np.mean(errors[:20])


class TestFactory:
    """Test factory function."""

    def test_create_cerebellar_module(self):
        cb = create_cerebellar_module(input_dim=64, num_purkinje=50)
        assert cb.config.input_dim == 64
        assert cb.config.num_purkinje == 50

    def test_state_to_dict(self, cerebellum, config):
        mossy = np.random.randn(config.input_dim).astype(np.float32)
        state = cerebellum.forward(mossy)
        d = state.to_dict()
        assert "timing_estimate" in d
        assert "prediction_error" in d


class TestStatsAndPersistence:
    """Test statistics and save/load."""

    def test_get_stats(self, cerebellum, config):
        mossy = np.random.randn(config.input_dim).astype(np.float32)
        cerebellum.forward(mossy)
        stats = cerebellum.get_stats()
        assert "step_count" in stats
        assert stats["step_count"] == 1

    def test_save_load_roundtrip(self, config):
        cb1 = CerebellarModule(config, random_seed=42)
        mossy = np.random.randn(config.input_dim).astype(np.float32)
        cb1.forward(mossy)

        saved = cb1.save_state()
        cb2 = CerebellarModule(config, random_seed=99)
        cb2.load_state(saved)

        assert cb2._step_count == cb1._step_count
