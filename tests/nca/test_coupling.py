"""Tests for NCA coupling matrix."""

import numpy as np
import pytest

from ww.nca.coupling import (
    BiologicalBounds,
    CouplingConfig,
    LearnableCoupling,
)
from ww.nca.neural_field import NeurotransmitterState


class TestBiologicalBounds:
    """Tests for BiologicalBounds."""

    def test_default_bounds(self):
        """Default bounds should be 6x6 matrices."""
        bounds = BiologicalBounds()
        assert bounds.K_MIN.shape == (6, 6)
        assert bounds.K_MAX.shape == (6, 6)

    def test_min_less_than_max(self):
        """K_MIN should always be <= K_MAX."""
        bounds = BiologicalBounds()
        assert np.all(bounds.K_MIN <= bounds.K_MAX)

    def test_clamp(self):
        """Clamp should enforce bounds."""
        bounds = BiologicalBounds()

        # Create out-of-bounds matrix
        K_too_high = np.ones((6, 6)) * 10
        K_clamped = bounds.clamp(K_too_high)

        assert np.all(K_clamped <= bounds.K_MAX)
        assert np.all(K_clamped >= bounds.K_MIN)

    def test_is_valid(self):
        """is_valid should correctly identify valid matrices."""
        bounds = BiologicalBounds()

        # Midpoint should always be valid
        K_mid = (bounds.K_MIN + bounds.K_MAX) / 2
        assert bounds.is_valid(K_mid)

        # Out of bounds should be invalid
        K_bad = np.ones((6, 6)) * 100
        assert not bounds.is_valid(K_bad)

    def test_ei_balance_constraint(self):
        """GABA-Glu coupling should be negative (mutual inhibition)."""
        bounds = BiologicalBounds()

        # GABA -> Glu (index [4, 5]) should have negative max
        # Glu -> GABA (index [5, 4]) should have negative max
        assert bounds.K_MAX[4, 5] <= 0  # GABA inhibits Glu
        assert bounds.K_MAX[5, 4] <= 0  # Glu activates GABA (which inhibits)


class TestCouplingConfig:
    """Tests for CouplingConfig."""

    def test_default_init_coupling(self):
        """Default init should be at biological midpoint."""
        config = CouplingConfig()
        bounds = BiologicalBounds()

        expected = (bounds.K_MIN + bounds.K_MAX) / 2
        np.testing.assert_array_almost_equal(config.init_coupling, expected)

    def test_custom_learning_rate(self):
        """Config should accept custom learning rate."""
        config = CouplingConfig(learning_rate=0.05)
        assert config.learning_rate == 0.05


class TestLearnableCoupling:
    """Tests for LearnableCoupling."""

    def test_initialization(self):
        """Coupling should initialize within bounds."""
        coupling = LearnableCoupling()
        assert coupling.bounds.is_valid(coupling.K)

    def test_compute_coupling(self):
        """compute_coupling should return 6-element vector."""
        coupling = LearnableCoupling()
        state = NeurotransmitterState()

        result = coupling.compute_coupling(state)
        assert result.shape == (6,)

    def test_compute_coupling_with_array(self):
        """compute_coupling should accept numpy array."""
        coupling = LearnableCoupling()
        U = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        result = coupling.compute_coupling(U)
        assert result.shape == (6,)

    def test_update_from_rpe_positive(self):
        """Positive RPE should update coupling."""
        coupling = LearnableCoupling()
        initial_K = coupling.K.copy()

        state = NeurotransmitterState(dopamine=0.8)
        coupling.update_from_rpe(state, rpe=0.5)

        # K should have changed
        assert not np.allclose(coupling.K, initial_K)

    def test_update_from_rpe_bounds_enforced(self):
        """Updates should stay within biological bounds."""
        coupling = LearnableCoupling()

        state = NeurotransmitterState()

        # Many large updates
        for _ in range(100):
            coupling.update_from_rpe(state, rpe=1.0)

        assert coupling.bounds.is_valid(coupling.K)

    def test_ei_balance_enforcement(self):
        """E/I balance should be maintained after updates."""
        config = CouplingConfig(enforce_ei_balance=True)
        coupling = LearnableCoupling(config=config)

        state = NeurotransmitterState()

        # Many updates
        for _ in range(50):
            coupling.update_from_rpe(state, rpe=0.5)

        # GABA-Glu coupling should remain inhibitory
        assert coupling.K[4, 5] < 0  # GABA -> Glu
        assert coupling.K[5, 4] < 0  # Glu -> GABA

    def test_coupling_strength(self):
        """get_coupling_strength should return positive scalar."""
        coupling = LearnableCoupling()
        strength = coupling.get_coupling_strength()

        assert isinstance(strength, float)
        assert strength >= 0

    def test_stats(self):
        """Stats should include expected keys."""
        coupling = LearnableCoupling()
        stats = coupling.get_stats()

        assert "update_count" in stats
        assert "coupling_norm" in stats
        assert "bounds_valid" in stats
        assert "ei_balance" in stats

    def test_save_load_state(self):
        """State should be recoverable via save/load."""
        coupling = LearnableCoupling()

        # Make some updates
        state = NeurotransmitterState(dopamine=0.8)
        coupling.update_from_rpe(state, rpe=0.3)

        # Save
        saved = coupling.save_state()

        # Create new coupling and load
        coupling2 = LearnableCoupling()
        coupling2.load_state(saved)

        np.testing.assert_array_almost_equal(coupling.K, coupling2.K)
        assert coupling._update_count == coupling2._update_count


class TestCouplingDynamics:
    """Integration tests for coupling dynamics."""

    def test_da_ne_antagonism(self):
        """DA and NE should show antagonistic coupling."""
        bounds = BiologicalBounds()

        # DA -> NE (index [0, 3]) should be negative (antagonistic)
        # This is based on biological constraint that DA inhibits NE
        assert bounds.K_MIN[0, 3] < 0

    def test_ach_da_interaction(self):
        """ACh and DA should show striatal interaction."""
        bounds = BiologicalBounds()

        # ACh can increase DA (index [2, 0])
        assert bounds.K_MAX[2, 0] > 0
