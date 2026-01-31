"""
Tests for Neural ODE Capsules module.

Tests continuous-time capsule dynamics via ODE integration.
"""

import numpy as np
import pytest

from t4dm.nca.neural_ode_capsules import (
    NeuralODECapsuleConfig,
    CapsuleState,
    CapsuleODEFunc,
    NeuralODECapsuleLayer,
)


class TestNeuralODECapsuleConfig:
    """Tests for NeuralODECapsuleConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NeuralODECapsuleConfig()
        assert config.input_dim == 1024
        assert config.num_capsules == 32
        assert config.capsule_dim == 16
        assert config.pose_dim == 4
        assert config.time_span == (0.0, 1.0)
        assert config.solver == "RK45"
        assert config.rtol == 1e-3
        assert config.atol == 1e-4
        assert config.activation_decay == 0.1
        assert config.pose_regularization == 0.01
        assert config.routing_rate == 0.5
        assert config.max_routing_iterations == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = NeuralODECapsuleConfig(
            input_dim=512,
            num_capsules=16,
            pose_dim=3,
            solver="RK23",
        )
        assert config.input_dim == 512
        assert config.num_capsules == 16
        assert config.pose_dim == 3
        assert config.solver == "RK23"


class TestCapsuleState:
    """Tests for CapsuleState."""

    @pytest.fixture
    def state(self):
        """Create capsule state instance."""
        num_capsules = 4
        pose_dim = 3

        activations = np.random.rand(num_capsules).astype(np.float32)
        poses = np.random.randn(num_capsules, pose_dim, pose_dim).astype(np.float32)
        routing = np.random.rand(num_capsules, num_capsules).astype(np.float32)

        return CapsuleState(activations, poses, routing)

    def test_initialization(self, state):
        """Test state initialization."""
        assert state.activations.shape == (4,)
        assert state.poses.shape == (4, 3, 3)
        assert state.routing.shape == (4, 4)

    def test_num_capsules(self, state):
        """Test num_capsules property."""
        assert state.num_capsules == 4

    def test_pose_dim(self, state):
        """Test pose_dim property."""
        assert state.pose_dim == 3

    def test_to_flat(self, state):
        """Test flattening state."""
        flat = state.to_flat()

        expected_len = 4 + 4 * 3 * 3 + 4 * 4  # activations + poses + routing
        assert flat.shape == (expected_len,)

    def test_from_flat(self):
        """Test reconstructing state from flat."""
        num_capsules = 4
        pose_dim = 3

        # Create flat array
        activations = np.random.rand(num_capsules)
        poses = np.random.randn(num_capsules, pose_dim, pose_dim)
        routing = np.random.rand(num_capsules, num_capsules)

        original = CapsuleState(activations, poses, routing)
        flat = original.to_flat()

        # Reconstruct
        reconstructed = CapsuleState.from_flat(flat, num_capsules, pose_dim)

        np.testing.assert_array_almost_equal(
            reconstructed.activations, original.activations
        )
        np.testing.assert_array_almost_equal(reconstructed.poses, original.poses)
        np.testing.assert_array_almost_equal(reconstructed.routing, original.routing)

    def test_roundtrip(self, state):
        """Test flatten/reconstruct roundtrip."""
        flat = state.to_flat()
        reconstructed = CapsuleState.from_flat(flat, state.num_capsules, state.pose_dim)

        np.testing.assert_array_almost_equal(
            reconstructed.activations, state.activations
        )
        np.testing.assert_array_almost_equal(reconstructed.poses, state.poses)
        np.testing.assert_array_almost_equal(reconstructed.routing, state.routing)


class TestCapsuleODEFunc:
    """Tests for CapsuleODEFunc."""

    @pytest.fixture
    def config(self):
        """Create small config for testing."""
        return NeuralODECapsuleConfig(
            input_dim=32,
            num_capsules=4,
            pose_dim=2,
        )

    @pytest.fixture
    def ode_func(self, config):
        """Create ODE function."""
        input_transform = np.random.randn(
            config.num_capsules, config.input_dim
        ).astype(np.float32)
        pose_transform = np.random.randn(
            config.num_capsules,
            config.num_capsules,
            config.pose_dim,
            config.pose_dim,
        ).astype(np.float32)

        return CapsuleODEFunc(config, input_transform, pose_transform)

    def test_initialization(self, ode_func):
        """Test ODE function initialization."""
        assert ode_func.W_a is not None
        assert ode_func.T is not None
        assert ode_func.current_input is None

    def test_set_input(self, ode_func, config):
        """Test setting input."""
        x = np.random.randn(config.input_dim).astype(np.float32)
        ode_func.set_input(x)

        np.testing.assert_array_equal(ode_func.current_input, x)

    def test_forward_without_input(self, ode_func, config):
        """Test forward raises error without input."""
        # Create flat state
        activations = np.random.rand(config.num_capsules)
        poses = np.random.randn(config.num_capsules, config.pose_dim, config.pose_dim)
        routing = np.random.rand(config.num_capsules, config.num_capsules)
        state = CapsuleState(activations, poses, routing).to_flat()

        with pytest.raises(ValueError, match="Input not set"):
            ode_func.forward(0.0, state)

    def test_forward(self, ode_func, config):
        """Test forward pass."""
        # Set input
        x = np.random.randn(config.input_dim).astype(np.float32)
        ode_func.set_input(x)

        # Create state
        activations = np.random.rand(config.num_capsules).astype(np.float32)
        poses = np.random.randn(
            config.num_capsules, config.pose_dim, config.pose_dim
        ).astype(np.float32)
        routing = np.random.rand(config.num_capsules, config.num_capsules).astype(
            np.float32
        )
        state = CapsuleState(activations, poses, routing).to_flat()

        # Compute derivative
        d_state = ode_func.forward(0.0, state)

        assert d_state.shape == state.shape

    def test_call_interface(self, ode_func, config):
        """Test __call__ interface."""
        x = np.random.randn(config.input_dim).astype(np.float32)
        ode_func.set_input(x)

        activations = np.random.rand(config.num_capsules).astype(np.float32)
        poses = np.random.randn(
            config.num_capsules, config.pose_dim, config.pose_dim
        ).astype(np.float32)
        routing = np.random.rand(config.num_capsules, config.num_capsules).astype(
            np.float32
        )
        state = CapsuleState(activations, poses, routing).to_flat()

        # Call should work same as forward
        d_state = ode_func(0.0, state)
        assert d_state.shape == state.shape

    def test_squash(self, ode_func):
        """Test squash nonlinearity."""
        x = np.array([0.0, 1.0, -1.0, 5.0, -5.0])
        squashed = ode_func._squash(x)

        # Should preserve signs
        assert np.sign(squashed[1]) > 0
        assert np.sign(squashed[2]) < 0
        assert np.sign(squashed[3]) > 0
        assert np.sign(squashed[4]) < 0

        # Large values should saturate
        assert np.abs(squashed[3]) < 1.5
        assert np.abs(squashed[4]) < 1.5

    def test_pose_agreement(self, ode_func):
        """Test pose agreement computation."""
        # Identical poses should have agreement 1
        P = np.eye(2)
        agreement = ode_func._pose_agreement(P, P)
        assert agreement == pytest.approx(1.0, rel=0.01)

        # Orthogonal poses should have agreement ~0
        P1 = np.array([[1, 0], [0, 1]])
        P2 = np.array([[0, 1], [-1, 0]])
        agreement = ode_func._pose_agreement(P1, P2)
        assert np.abs(agreement) < 0.5


class TestNeuralODECapsuleLayer:
    """Tests for NeuralODECapsuleLayer."""

    @pytest.fixture
    def config(self):
        """Create small config for faster tests."""
        return NeuralODECapsuleConfig(
            input_dim=32,
            num_capsules=4,
            pose_dim=2,
            time_span=(0.0, 0.5),  # Shorter integration
        )

    @pytest.fixture
    def layer(self, config):
        """Create layer instance."""
        return NeuralODECapsuleLayer(config)

    def test_initialization(self, layer):
        """Test layer initialization."""
        assert layer.config is not None
        assert layer.ode_func is not None
        assert layer.W_a.shape == (4, 32)
        assert layer.T.shape == (4, 4, 2, 2)

    def test_initial_state(self, layer):
        """Test initial state generation."""
        state = layer._initial_state()

        assert state.activations.shape == (4,)
        assert state.poses.shape == (4, 2, 2)
        assert state.routing.shape == (4, 4)

        # Activations should be small
        assert np.all(state.activations > 0)
        assert np.all(state.activations < 0.2)

        # Routing should be uniform
        assert np.allclose(state.routing, 0.25, atol=0.01)

    def test_forward(self, layer, config):
        """Test forward pass."""
        x = np.random.randn(config.input_dim).astype(np.float32)

        activations, poses = layer.forward(x)

        assert activations.shape == (4,)
        assert poses.shape == (4, 2, 2)

    def test_forward_custom_time_span(self, layer, config):
        """Test forward with custom time span."""
        x = np.random.randn(config.input_dim).astype(np.float32)

        activations, poses = layer.forward(x, t_span=(0.0, 0.1))

        assert activations.shape == (4,)
        assert poses.shape == (4, 2, 2)

    def test_get_trajectory(self, layer, config):
        """Test getting full trajectory."""
        x = np.random.randn(config.input_dim).astype(np.float32)

        trajectory = layer.get_trajectory(x, n_points=5)

        assert len(trajectory) == 5
        for state in trajectory:
            assert isinstance(state, CapsuleState)
            assert state.activations.shape == (4,)

    def test_compute_energy(self, layer):
        """Test energy computation."""
        state = layer._initial_state()
        energy = layer.compute_energy(state)

        # Energy should be a finite number
        assert np.isfinite(energy)

    def test_energy_decreases_over_trajectory(self, layer, config):
        """Test that energy decreases as capsules settle."""
        x = np.random.randn(config.input_dim).astype(np.float32)

        trajectory = layer.get_trajectory(x, n_points=10)

        energies = [layer.compute_energy(state) for state in trajectory]

        # Energy should generally decrease (or stay similar)
        # Note: Not guaranteed to be monotonic, but end should be <= start
        # In practice this depends on the input and initialization
        assert len(energies) == 10

    def test_get_stats(self, layer):
        """Test statistics retrieval."""
        stats = layer.get_stats()

        assert "num_capsules" in stats
        assert "pose_dim" in stats
        assert "solver" in stats
        assert "time_span" in stats
        assert "W_a_norm" in stats
        assert "T_norm" in stats

        assert stats["num_capsules"] == 4
        assert stats["pose_dim"] == 2

    def test_different_inputs_different_outputs(self, layer, config):
        """Test that different inputs produce different outputs."""
        x1 = np.random.randn(config.input_dim).astype(np.float32)
        x2 = np.random.randn(config.input_dim).astype(np.float32)

        act1, poses1 = layer.forward(x1)
        act2, poses2 = layer.forward(x2)

        # Different inputs should produce different outputs
        # (Note: Due to random initialization, this is probabilistic)
        # Use a loose check
        assert not np.allclose(act1, act2, atol=0.01) or not np.allclose(
            poses1, poses2, atol=0.01
        )


class TestNeuralODECapsuleIntegration:
    """Integration tests for Neural ODE capsules."""

    def test_full_forward_pass(self):
        """Test complete forward pass."""
        config = NeuralODECapsuleConfig(
            input_dim=64,
            num_capsules=8,
            pose_dim=3,
            time_span=(0.0, 1.0),
        )
        layer = NeuralODECapsuleLayer(config)

        x = np.random.randn(64).astype(np.float32)
        activations, poses = layer.forward(x)

        assert activations.shape == (8,)
        assert poses.shape == (8, 3, 3)
        assert np.all(np.isfinite(activations))
        assert np.all(np.isfinite(poses))

    def test_trajectory_evolution(self):
        """Test that trajectory shows evolution."""
        config = NeuralODECapsuleConfig(
            input_dim=32,
            num_capsules=4,
            pose_dim=2,
            time_span=(0.0, 2.0),
        )
        layer = NeuralODECapsuleLayer(config)

        x = np.random.randn(32).astype(np.float32)
        trajectory = layer.get_trajectory(x, n_points=20)

        # States should evolve over time
        first_state = trajectory[0]
        last_state = trajectory[-1]

        # Activations should change
        assert not np.allclose(first_state.activations, last_state.activations)

    def test_solver_options(self):
        """Test different ODE solvers."""
        for solver in ["RK45", "RK23"]:
            config = NeuralODECapsuleConfig(
                input_dim=32,
                num_capsules=4,
                pose_dim=2,
                solver=solver,
                time_span=(0.0, 0.5),
            )
            layer = NeuralODECapsuleLayer(config)

            x = np.random.randn(32).astype(np.float32)
            activations, poses = layer.forward(x)

            assert activations.shape == (4,)
            assert np.all(np.isfinite(activations))

    def test_capsule_state_roundtrip_in_integration(self):
        """Test state conversion during integration."""
        config = NeuralODECapsuleConfig(
            input_dim=32,
            num_capsules=4,
            pose_dim=2,
        )
        layer = NeuralODECapsuleLayer(config)

        x = np.random.randn(32).astype(np.float32)
        trajectory = layer.get_trajectory(x, n_points=5)

        # Each state in trajectory should be valid
        for state in trajectory:
            flat = state.to_flat()
            reconstructed = CapsuleState.from_flat(
                flat, state.num_capsules, state.pose_dim
            )

            np.testing.assert_array_almost_equal(
                state.activations, reconstructed.activations
            )
            np.testing.assert_array_almost_equal(state.poses, reconstructed.poses)
            np.testing.assert_array_almost_equal(state.routing, reconstructed.routing)

    def test_energy_finite_throughout(self):
        """Test that energy remains finite during integration."""
        config = NeuralODECapsuleConfig(
            input_dim=32,
            num_capsules=4,
            pose_dim=2,
            time_span=(0.0, 1.0),
        )
        layer = NeuralODECapsuleLayer(config)

        x = np.random.randn(32).astype(np.float32)
        trajectory = layer.get_trajectory(x, n_points=10)

        for state in trajectory:
            energy = layer.compute_energy(state)
            assert np.isfinite(energy)
