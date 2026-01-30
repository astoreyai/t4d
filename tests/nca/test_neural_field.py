"""Tests for NCA neural field PDE solver."""

import numpy as np
import pytest

from ww.nca.neural_field import (
    NeuralFieldConfig,
    NeuralFieldSolver,
    NeurotransmitterState,
    NeurotransmitterType,
)


class TestNeurotransmitterState:
    """Tests for NeurotransmitterState dataclass."""

    def test_default_values(self):
        """Default state should be 0.5 for all NTs."""
        state = NeurotransmitterState()
        assert state.dopamine == 0.5
        assert state.serotonin == 0.5
        assert state.acetylcholine == 0.5
        assert state.norepinephrine == 0.5
        assert state.gaba == 0.5
        assert state.glutamate == 0.5

    def test_to_array(self):
        """to_array should return [DA, 5HT, ACh, NE, GABA, Glu]."""
        state = NeurotransmitterState(
            dopamine=0.1,
            serotonin=0.2,
            acetylcholine=0.3,
            norepinephrine=0.4,
            gaba=0.5,
            glutamate=0.6
        )
        arr = state.to_array()
        assert arr.shape == (6,)
        np.testing.assert_array_almost_equal(
            arr, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        )

    def test_from_array(self):
        """from_array should reconstruct state."""
        arr = np.array([0.7, 0.8, 0.9, 0.6, 0.5, 0.4])
        state = NeurotransmitterState.from_array(arr)
        assert state.dopamine == pytest.approx(0.7)
        assert state.serotonin == pytest.approx(0.8)
        assert state.acetylcholine == pytest.approx(0.9)
        assert state.norepinephrine == pytest.approx(0.6)
        assert state.gaba == pytest.approx(0.5)
        assert state.glutamate == pytest.approx(0.4)

    def test_clamping(self):
        """Values should be clamped to [0, 1]."""
        state = NeurotransmitterState(
            dopamine=1.5,
            serotonin=-0.5
        )
        assert state.dopamine == 1.0
        assert state.serotonin == 0.0

    def test_roundtrip(self):
        """array -> state -> array should be identity."""
        original = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        state = NeurotransmitterState.from_array(original)
        recovered = state.to_array()
        np.testing.assert_array_almost_equal(original, recovered)


class TestNeuralFieldConfig:
    """Tests for NeuralFieldConfig."""

    def test_default_values(self):
        """Config should have biologically plausible defaults."""
        config = NeuralFieldConfig()

        # Glutamate should have fastest clearance (excitotoxicity)
        assert config.alpha_glu >= config.alpha_da

        # ACh should be faster than 5-HT
        assert config.alpha_ach > config.alpha_5ht

        # Time step should be small for stability
        assert config.dt <= 0.01

    def test_custom_values(self):
        """Config should accept custom values."""
        config = NeuralFieldConfig(
            alpha_da=1.0,
            dt=0.001,
            grid_size=64
        )
        assert config.alpha_da == 1.0
        assert config.dt == 0.001
        assert config.grid_size == 64


class TestNeuralFieldSolver:
    """Tests for NeuralFieldSolver."""

    def test_initialization(self):
        """Solver should initialize with baseline fields."""
        solver = NeuralFieldSolver()
        state = solver.get_mean_state()

        # All NTs should start at baseline (0.5)
        assert state.dopamine == pytest.approx(0.5)
        assert state.serotonin == pytest.approx(0.5)

    def test_step_decay(self):
        """Without stimulus, fields should decay toward 0."""
        config = NeuralFieldConfig(dt=0.1)
        solver = NeuralFieldSolver(config=config)

        # Inject high DA
        solver.inject_stimulus(NeurotransmitterType.DOPAMINE, 0.3)

        initial_da = solver.get_mean_state().dopamine

        # Step forward
        for _ in range(10):
            solver.step()

        final_da = solver.get_mean_state().dopamine

        # DA should have decayed
        assert final_da < initial_da

    def test_stimulus_injection(self):
        """Stimulus injection should increase NT level."""
        solver = NeuralFieldSolver()
        initial = solver.get_mean_state().dopamine

        solver.inject_stimulus(NeurotransmitterType.DOPAMINE, 0.2)

        after = solver.get_mean_state().dopamine
        assert after > initial

    def test_clamping_high(self):
        """Fields should not exceed 1.0."""
        solver = NeuralFieldSolver()

        # Inject very high stimulus
        for _ in range(10):
            solver.inject_stimulus(NeurotransmitterType.DOPAMINE, 0.5)

        state = solver.get_mean_state()
        assert state.dopamine <= 1.0

    def test_clamping_low(self):
        """Fields should not go below 0.0."""
        solver = NeuralFieldSolver()

        # Inject negative stimulus
        for _ in range(10):
            solver.inject_stimulus(NeurotransmitterType.DOPAMINE, -0.5)

        state = solver.get_mean_state()
        assert state.dopamine >= 0.0

    def test_reset(self):
        """Reset should return to baseline."""
        solver = NeuralFieldSolver()

        # Perturb state
        solver.inject_stimulus(NeurotransmitterType.DOPAMINE, 0.3)
        for _ in range(10):
            solver.step()

        # Reset
        solver.reset()

        state = solver.get_mean_state()
        assert state.dopamine == pytest.approx(0.5)
        assert solver._time == 0.0
        assert solver._step_count == 0

    def test_stats(self):
        """Stats should return expected keys."""
        solver = NeuralFieldSolver()
        solver.step()

        stats = solver.get_stats()
        assert "time" in stats
        assert "step_count" in stats
        assert "mean_state" in stats
        assert "config" in stats


class TestNeuralFieldIntegration:
    """Integration tests for neural field dynamics."""

    def test_ei_balance(self):
        """GABA and Glu should show opposing dynamics."""
        solver = NeuralFieldSolver()

        # High Glu should... (need coupling for full test)
        # This is a placeholder for when coupling is integrated
        solver.inject_stimulus(NeurotransmitterType.GLUTAMATE, 0.3)
        solver.step()

        state = solver.get_mean_state()
        assert state.glutamate > 0.5

    def test_multiple_steps_stability(self):
        """Solver should remain stable over many steps."""
        solver = NeuralFieldSolver()

        # Run for many steps
        for i in range(1000):
            # Add random small perturbations
            if i % 100 == 0:
                nt = list(NeurotransmitterType)[i % 6]
                solver.inject_stimulus(nt, 0.1)
            solver.step()

        state = solver.get_mean_state()

        # All NTs should still be in valid range
        arr = state.to_array()
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)
