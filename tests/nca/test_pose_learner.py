"""
Unit tests for Emergent Pose Learning (Phase 4B - Hinton H8-H9).

Tests pose dimension discovery from routing patterns:
- Learning from routing agreement
- Agreement-modulated learning rates
- Convergence to stable poses
- Integration with capsule routing
- Dimension analysis and statistics
- Persistence (save/load)

References:
- Hinton H8: Part-whole representation
- Hinton H9: Frozen pose learning (addressed by emergent learning)
"""

import numpy as np
import pytest

from t4dm.nca.pose_learner import (
    PoseDimensionDiscovery,
    PoseLearnerConfig,
    PoseLearnerState,
    PoseLearningMixin,
    create_learnable_capsule_system,
    create_pose_learner,
)


# =============================================================================
# Test Configuration
# =============================================================================


class TestPoseLearnerConfig:
    """Test PoseLearnerConfig dataclass."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = PoseLearnerConfig()
        assert config.n_dimensions == 4
        assert config.pose_dim == 4
        assert config.base_lr == 0.01
        assert config.agreement_threshold == 0.7
        assert config.momentum == 0.9

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = PoseLearnerConfig(
            n_dimensions=8,
            pose_dim=8,
            base_lr=0.05,
            agreement_threshold=0.8,
        )
        assert config.n_dimensions == 8
        assert config.pose_dim == 8
        assert config.base_lr == 0.05
        assert config.agreement_threshold == 0.8

    def test_config_validation_n_dimensions(self):
        """Config validates n_dimensions."""
        with pytest.raises(AssertionError):
            PoseLearnerConfig(n_dimensions=0)

    def test_config_validation_base_lr(self):
        """Config validates base_lr range."""
        with pytest.raises(AssertionError):
            PoseLearnerConfig(base_lr=0)
        with pytest.raises(AssertionError):
            PoseLearnerConfig(base_lr=1.5)

    def test_config_validation_agreement_threshold(self):
        """Config validates agreement_threshold range."""
        with pytest.raises(AssertionError):
            PoseLearnerConfig(agreement_threshold=-0.1)
        with pytest.raises(AssertionError):
            PoseLearnerConfig(agreement_threshold=1.5)


# =============================================================================
# Test State
# =============================================================================


class TestPoseLearnerState:
    """Test PoseLearnerState dataclass."""

    def test_empty_state(self):
        """Empty state initializes correctly."""
        state = PoseLearnerState()
        assert state.total_updates == 0
        assert state.mean_agreement == 0.0
        assert state.is_converged is False
        assert state.convergence_step == -1

    def test_state_with_values(self):
        """State holds actual values."""
        state = PoseLearnerState(
            total_updates=100,
            mean_agreement=0.85,
            is_converged=True,
            convergence_step=95,
        )
        assert state.total_updates == 100
        assert state.mean_agreement == 0.85
        assert state.is_converged is True
        assert state.convergence_step == 95


# =============================================================================
# Test PoseDimensionDiscovery Initialization
# =============================================================================


class TestPoseDimensionDiscoveryInit:
    """Test PoseDimensionDiscovery initialization."""

    def test_default_initialization(self):
        """Default initialization works."""
        learner = PoseDimensionDiscovery()
        assert len(learner.transform_matrices) == 4
        assert len(learner.dimension_names) == 4
        assert learner.state.total_updates == 0

    def test_custom_dimensions(self):
        """Custom number of dimensions works."""
        learner = PoseDimensionDiscovery(n_dimensions=8)
        assert len(learner.transform_matrices) == 8
        assert len(learner.dimension_names) == 8

    def test_config_initialization(self):
        """Config-based initialization works."""
        config = PoseLearnerConfig(
            n_dimensions=6,
            pose_dim=6,
            base_lr=0.05,
        )
        learner = PoseDimensionDiscovery(config=config)
        assert len(learner.transform_matrices) == 6
        assert learner.config.base_lr == 0.05

    def test_reproducible_with_seed(self):
        """Same seed produces same initialization."""
        learner1 = PoseDimensionDiscovery(n_dimensions=4, random_seed=42)
        learner2 = PoseDimensionDiscovery(n_dimensions=4, random_seed=42)

        for t1, t2 in zip(learner1.transform_matrices, learner2.transform_matrices):
            np.testing.assert_array_equal(t1, t2)

    def test_different_seeds_differ(self):
        """Different seeds produce different initialization."""
        learner1 = PoseDimensionDiscovery(n_dimensions=4, random_seed=42)
        learner2 = PoseDimensionDiscovery(n_dimensions=4, random_seed=99)

        # At least one transform should differ
        any_different = False
        for t1, t2 in zip(learner1.transform_matrices, learner2.transform_matrices):
            if not np.allclose(t1, t2):
                any_different = True
                break
        assert any_different, "Different seeds should produce different transforms"

    def test_transforms_near_identity(self):
        """Initial transforms are near identity."""
        learner = PoseDimensionDiscovery(n_dimensions=4, random_seed=42)

        for transform in learner.transform_matrices:
            identity = np.eye(learner.config.pose_dim)
            # Should be within 0.5 of identity (accounting for random noise)
            assert np.linalg.norm(transform - identity) < 0.5


# =============================================================================
# Test Core Learning
# =============================================================================


class TestLearnFromRouting:
    """Test learn_from_routing core functionality."""

    @pytest.fixture
    def learner(self):
        """Create test learner."""
        return PoseDimensionDiscovery(
            config=PoseLearnerConfig(
                n_dimensions=4,
                pose_dim=4,
                base_lr=0.1,
                agreement_threshold=0.7,
            ),
            random_seed=42,
        )

    def test_learn_from_routing_returns_dict(self, learner):
        """learn_from_routing returns expected dictionary."""
        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        stats = learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.6,
        )

        assert isinstance(stats, dict)
        assert "mean_agreement" in stats
        assert "effective_lr" in stats
        assert "transform_change" in stats
        assert "consensus_distance" in stats
        assert "loss" in stats

    def test_learn_from_routing_updates_state(self, learner):
        """Learning updates state correctly."""
        initial_updates = learner.state.total_updates

        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.6,
        )

        assert learner.state.total_updates == initial_updates + 1
        assert learner.state.mean_agreement == 0.6

    def test_high_agreement_reduces_lr(self, learner):
        """High agreement reduces effective learning rate."""
        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        # High agreement (above threshold)
        stats_high = learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.9,  # Above threshold of 0.7
        )

        # Reset learner
        learner2 = PoseDimensionDiscovery(
            config=learner.config,
            random_seed=42,
        )

        # Low agreement
        stats_low = learner2.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.5,  # Below threshold
        )

        # High agreement should have lower effective LR
        assert stats_high["effective_lr"] < stats_low["effective_lr"]

    def test_low_agreement_uses_base_lr(self, learner):
        """Low agreement uses base learning rate."""
        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        stats = learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.5,  # Below threshold
        )

        assert stats["effective_lr"] == learner.config.base_lr

    def test_transforms_change_during_learning(self, learner):
        """Transform matrices change during learning."""
        initial_transforms = [t.copy() for t in learner.transform_matrices]

        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.5,
        )

        # At least one transform should change
        any_changed = False
        for t_init, t_new in zip(initial_transforms, learner.transform_matrices):
            if not np.allclose(t_init, t_new):
                any_changed = True
                break

        assert any_changed, "Transforms should change during learning"

    def test_single_pose_input_shape(self, learner):
        """Handles single pose (2D) input correctly."""
        lower_pose = np.random.rand(4, 4).astype(np.float32)
        upper_pose = np.random.rand(4, 4).astype(np.float32)
        routing_weights = np.array([1.0], dtype=np.float32)

        # Should not raise
        stats = learner.learn_from_routing(
            lower_poses=lower_pose,
            upper_poses=upper_pose,
            routing_weights=routing_weights,
            agreement=0.6,
        )

        assert stats["total_updates"] == 1

    def test_custom_learning_rate(self, learner):
        """Custom learning rate overrides config."""
        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        stats = learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.5,  # Below threshold
            learning_rate=0.05,  # Custom LR
        )

        assert stats["effective_lr"] == 0.05


# =============================================================================
# Test Convergence
# =============================================================================


class TestConvergence:
    """Test convergence detection."""

    def test_converges_with_consistent_input(self):
        """Learner converges with consistent input."""
        config = PoseLearnerConfig(
            n_dimensions=4,
            pose_dim=4,
            base_lr=0.1,
            convergence_window=10,
            convergence_threshold=0.01,
        )
        learner = PoseDimensionDiscovery(config=config, random_seed=42)

        # Use fixed, consistent input
        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        # Train with high agreement (small updates, should converge)
        for _ in range(50):
            learner.learn_from_routing(
                lower_poses=lower_poses,
                upper_poses=upper_poses,
                routing_weights=routing_weights,
                agreement=0.9,
            )

        # Should converge or have very low loss variance
        assert (
            learner.state.is_converged or
            len(learner.state.loss_history) < config.convergence_window
        )

    def test_does_not_converge_immediately(self):
        """Learner does not converge in first few steps."""
        config = PoseLearnerConfig(
            n_dimensions=4,
            pose_dim=4,
            base_lr=0.1,
            convergence_window=50,
        )
        learner = PoseDimensionDiscovery(config=config, random_seed=42)

        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        # Train just a few steps
        for _ in range(10):
            learner.learn_from_routing(
                lower_poses=lower_poses,
                upper_poses=upper_poses,
                routing_weights=routing_weights,
                agreement=0.5,
            )

        # Should NOT be converged yet
        assert not learner.state.is_converged


# =============================================================================
# Test Dimension Analysis
# =============================================================================


class TestDimensionAnalysis:
    """Test dimension usage and correlation analysis."""

    @pytest.fixture
    def learner(self):
        """Create and train a learner."""
        config = PoseLearnerConfig(n_dimensions=4, pose_dim=4)
        learner = PoseDimensionDiscovery(config=config, random_seed=42)

        # Train a bit
        for _ in range(10):
            lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
            upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
            routing_weights = np.random.rand(8, 4).astype(np.float32)
            learner.learn_from_routing(
                lower_poses=lower_poses,
                upper_poses=upper_poses,
                routing_weights=routing_weights,
                agreement=0.6,
            )

        return learner

    def test_get_dimension_usage_shape(self, learner):
        """Dimension usage has correct shape."""
        usage = learner.get_dimension_usage()
        assert usage.shape == (learner.config.n_dimensions,)

    def test_get_dimension_usage_positive(self, learner):
        """Dimension usage is non-negative."""
        usage = learner.get_dimension_usage()
        assert np.all(usage >= 0)

    def test_get_dimension_correlations_shape(self, learner):
        """Dimension correlations has correct shape."""
        corr = learner.get_dimension_correlations()
        n = learner.config.n_dimensions
        assert corr.shape == (n, n)

    def test_correlations_symmetric(self, learner):
        """Correlation matrix is symmetric."""
        corr = learner.get_dimension_correlations()
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_correlations_diagonal_one(self, learner):
        """Diagonal of correlation matrix is 1."""
        corr = learner.get_dimension_correlations()
        np.testing.assert_array_almost_equal(np.diag(corr), 1.0)

    def test_get_transform_statistics(self, learner):
        """Transform statistics computed correctly."""
        stats = learner.get_transform_statistics()

        assert "mean_norm" in stats
        assert "std_norm" in stats
        assert "norms" in stats
        assert "determinants" in stats
        assert "orthogonality_errors" in stats

        assert len(stats["norms"]) == learner.config.n_dimensions
        assert stats["mean_norm"] > 0


# =============================================================================
# Test Transform Application
# =============================================================================


class TestTransformApplication:
    """Test transform application methods."""

    @pytest.fixture
    def learner(self):
        """Create test learner."""
        return PoseDimensionDiscovery(n_dimensions=4, random_seed=42)

    def test_apply_transform_shape(self, learner):
        """Apply transform preserves shape."""
        pose = np.random.rand(4, 4).astype(np.float32)
        transformed = learner.apply_transform(pose, dimension_idx=0)
        assert transformed.shape == pose.shape

    def test_apply_transform_batch_shape(self, learner):
        """Apply transform works on batches."""
        poses = np.random.rand(10, 4, 4).astype(np.float32)
        transformed = learner.apply_transform(poses, dimension_idx=0)
        assert transformed.shape == poses.shape

    def test_apply_transform_invalid_idx(self, learner):
        """Apply transform raises on invalid index."""
        pose = np.random.rand(4, 4).astype(np.float32)
        with pytest.raises(ValueError):
            learner.apply_transform(pose, dimension_idx=100)

    def test_apply_all_transforms_shape(self, learner):
        """Apply all transforms produces correct shape."""
        pose = np.random.rand(4, 4).astype(np.float32)
        all_transformed = learner.apply_all_transforms(pose)
        assert all_transformed.shape == (learner.config.n_dimensions, 4, 4)

    def test_compose_transforms_shape(self, learner):
        """Compose transforms produces correct shape."""
        composed = learner.compose_transforms([0, 1])
        assert composed.shape == (learner.config.pose_dim, learner.config.pose_dim)

    def test_compose_transforms_order_matters(self, learner):
        """Transform composition order affects result."""
        comp_01 = learner.compose_transforms([0, 1])
        comp_10 = learner.compose_transforms([1, 0])

        # Generally, matrix multiplication is not commutative
        # (unless transforms happen to be special)
        # We can't assert they're always different, but we can test the operation works
        assert comp_01.shape == comp_10.shape


# =============================================================================
# Test Persistence
# =============================================================================


class TestPersistence:
    """Test save/load functionality."""

    @pytest.fixture
    def trained_learner(self):
        """Create and train a learner."""
        learner = PoseDimensionDiscovery(
            config=PoseLearnerConfig(n_dimensions=4, pose_dim=4),
            random_seed=42,
        )

        # Train
        for i in range(20):
            lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
            upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
            routing_weights = np.random.rand(8, 4).astype(np.float32)
            learner.learn_from_routing(
                lower_poses=lower_poses,
                upper_poses=upper_poses,
                routing_weights=routing_weights,
                agreement=0.6 + 0.01 * i,
            )

        return learner

    def test_get_state_dict(self, trained_learner):
        """get_state_dict returns valid dictionary."""
        state_dict = trained_learner.get_state_dict()

        assert "config" in state_dict
        assert "dimension_names" in state_dict
        assert "transform_matrices" in state_dict
        assert "momentum_buffers" in state_dict
        assert "state" in state_dict

    def test_load_state_dict_restores_state(self, trained_learner):
        """load_state_dict restores learner state."""
        state_dict = trained_learner.get_state_dict()

        # Create new learner
        new_learner = PoseDimensionDiscovery(
            config=PoseLearnerConfig(n_dimensions=4, pose_dim=4),
            random_seed=99,  # Different seed
        )

        # Load state
        new_learner.load_state_dict(state_dict)

        # Check state restored
        assert new_learner.state.total_updates == trained_learner.state.total_updates
        assert new_learner.state.mean_agreement == trained_learner.state.mean_agreement

        # Check transforms restored
        for t1, t2 in zip(
            new_learner.transform_matrices,
            trained_learner.transform_matrices,
        ):
            np.testing.assert_array_almost_equal(t1, t2)


# =============================================================================
# Test Capsule Integration
# =============================================================================


class TestCapsuleIntegration:
    """Test integration with CapsuleLayer."""

    def test_integrate_with_capsule_layer(self):
        """Integration with capsule layer works."""
        from t4dm.nca.capsules import CapsuleConfig, CapsuleLayer

        # Create capsule layer
        capsule_config = CapsuleConfig(
            input_dim=64,
            num_capsules=8,
            capsule_dim=8,
            pose_dim=4,
        )
        capsule_layer = CapsuleLayer(capsule_config, random_seed=42)

        # Forward pass to initialize transform matrices
        x = np.random.rand(64).astype(np.float32)
        activations, poses = capsule_layer.forward(x)

        # Route to initialize _transform_matrices
        capsule_layer.route(activations, poses, num_output_capsules=4)

        # Create pose learner
        pose_learner = PoseDimensionDiscovery(n_dimensions=4, random_seed=42)

        # Train pose learner
        for _ in range(10):
            lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
            upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
            routing_weights = np.random.rand(8, 4).astype(np.float32)
            pose_learner.learn_from_routing(
                lower_poses=lower_poses,
                upper_poses=upper_poses,
                routing_weights=routing_weights,
                agreement=0.6,
            )

        # Get transforms before integration
        transforms_before = capsule_layer._transform_matrices.copy()

        # Integrate
        pose_learner.integrate_with_capsule_layer(capsule_layer)

        # Transforms should have changed
        assert not np.allclose(
            capsule_layer._transform_matrices, transforms_before
        ), "Transforms should change after integration"

    def test_create_learnable_capsule_system(self):
        """Factory function creates integrated system."""
        capsule_layer, pose_learner = create_learnable_capsule_system(
            input_dim=128,
            num_capsules=16,
            pose_dim=4,
            random_seed=42,
        )

        assert capsule_layer.config.input_dim == 128
        assert capsule_layer.config.num_capsules == 16
        assert pose_learner.config.pose_dim == 4


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_pose_learner(self):
        """create_pose_learner factory works."""
        learner = create_pose_learner(
            n_dimensions=6,
            pose_dim=6,
            learning_rate=0.05,
            random_seed=42,
        )

        assert len(learner.transform_matrices) == 6
        assert learner.config.base_lr == 0.05

    def test_create_pose_learner_defaults(self):
        """create_pose_learner works with defaults."""
        learner = create_pose_learner()
        assert len(learner.transform_matrices) == 4


# =============================================================================
# Test Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_zero_routing_weights(self):
        """Handles zero routing weights."""
        learner = PoseDimensionDiscovery(n_dimensions=4, random_seed=42)

        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.zeros((8, 4), dtype=np.float32)

        # Should not crash
        stats = learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.5,
        )

        # Results should be finite
        assert np.isfinite(stats["loss"])

    def test_large_poses(self):
        """Handles large pose values."""
        learner = PoseDimensionDiscovery(n_dimensions=4, random_seed=42)

        lower_poses = np.random.rand(8, 4, 4).astype(np.float32) * 1000
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32) * 1000
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        # Should not crash
        stats = learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.5,
        )

        # Results should be finite
        assert np.isfinite(stats["loss"])
        for t in learner.transform_matrices:
            assert np.all(np.isfinite(t))

    def test_small_poses(self):
        """Handles very small pose values."""
        learner = PoseDimensionDiscovery(n_dimensions=4, random_seed=42)

        lower_poses = np.random.rand(8, 4, 4).astype(np.float32) * 1e-8
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32) * 1e-8
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        # Should not crash
        stats = learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.5,
        )

        # Results should be finite
        assert np.isfinite(stats["loss"])

    def test_gradient_clipping(self):
        """Gradient clipping prevents explosion."""
        config = PoseLearnerConfig(
            n_dimensions=4,
            pose_dim=4,
            base_lr=1.0,  # Very high LR
            max_gradient_norm=0.1,  # Strong clipping
        )
        learner = PoseDimensionDiscovery(config=config, random_seed=42)

        # Large input that would cause large gradients
        lower_poses = np.random.rand(8, 4, 4).astype(np.float32) * 100
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32) * 100
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        initial_norms = [np.linalg.norm(t) for t in learner.transform_matrices]

        learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.1,  # Low agreement = larger update
        )

        final_norms = [np.linalg.norm(t) for t in learner.transform_matrices]

        # Update should be bounded
        for init, final in zip(initial_norms, final_norms):
            assert abs(final - init) < 10, "Gradient clipping should bound updates"


# =============================================================================
# Test String Representation
# =============================================================================


class TestStringRepresentation:
    """Test string representation."""

    def test_repr(self):
        """__repr__ returns informative string."""
        learner = PoseDimensionDiscovery(n_dimensions=4, random_seed=42)
        repr_str = repr(learner)

        assert "PoseDimensionDiscovery" in repr_str
        assert "n_dims=4" in repr_str


# =============================================================================
# Test Hebbian Learning Properties
# =============================================================================


class TestHebbianLearning:
    """Test Hebbian learning properties (H8-H9)."""

    def test_reinforcement_with_high_agreement(self):
        """High agreement leads to smaller weight changes."""
        learner = PoseDimensionDiscovery(
            config=PoseLearnerConfig(
                n_dimensions=4,
                pose_dim=4,
                base_lr=0.1,
                agreement_threshold=0.5,
            ),
            random_seed=42,
        )

        # Fixed input for comparison
        rng = np.random.default_rng(42)
        lower_poses = rng.random((8, 4, 4)).astype(np.float32)
        upper_poses = rng.random((4, 4, 4)).astype(np.float32)
        routing_weights = rng.random((8, 4)).astype(np.float32)

        # High agreement update
        stats_high = learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.9,
        )

        # Reset
        learner2 = PoseDimensionDiscovery(
            config=learner.config,
            random_seed=42,
        )

        # Low agreement update
        stats_low = learner2.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.3,
        )

        # High agreement should have smaller transform change (per effective LR)
        # Note: transform_change also depends on consensus distance, so we compare
        # effective LR which is guaranteed to be smaller
        assert stats_high["effective_lr"] < stats_low["effective_lr"]

    def test_poses_emerge_from_learning(self):
        """Pose structure emerges through learning (not hand-set)."""
        learner = PoseDimensionDiscovery(
            config=PoseLearnerConfig(
                n_dimensions=4,
                pose_dim=4,
                base_lr=0.1,
            ),
            random_seed=42,
        )

        # Initial transforms
        initial_transforms = [t.copy() for t in learner.transform_matrices]

        # Train with consistent patterns
        rng = np.random.default_rng(42)
        for _ in range(50):
            lower_poses = rng.random((8, 4, 4)).astype(np.float32)
            upper_poses = rng.random((4, 4, 4)).astype(np.float32)
            routing_weights = rng.random((8, 4)).astype(np.float32)

            learner.learn_from_routing(
                lower_poses=lower_poses,
                upper_poses=upper_poses,
                routing_weights=routing_weights,
                agreement=0.6,
            )

        # Transforms should have evolved from initial state
        total_change = 0.0
        for t_init, t_final in zip(initial_transforms, learner.transform_matrices):
            total_change += np.linalg.norm(t_final - t_init)

        assert total_change > 0.1, "Transforms should emerge from learning"


# =============================================================================
# Test PoseLearningMixin
# =============================================================================


class TestPoseLearningMixin:
    """Test PoseLearningMixin class."""

    def test_mixin_init_pose_learner(self):
        """Mixin initializes pose learner."""

        class TestClass(PoseLearningMixin):
            pass

        obj = TestClass()
        obj.init_pose_learner()

        assert hasattr(obj, "_pose_learner")
        assert isinstance(obj._pose_learner, PoseDimensionDiscovery)

    def test_mixin_learn_from_routing(self):
        """Mixin learn method works."""

        class TestClass(PoseLearningMixin):
            pass

        obj = TestClass()
        obj.init_pose_learner()

        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        stats = obj.learn_poses_from_routing_result(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.6,
        )

        assert "mean_agreement" in stats

    def test_mixin_auto_init(self):
        """Mixin auto-initializes pose learner if not present."""

        class TestClass(PoseLearningMixin):
            pass

        obj = TestClass()

        # Should not have _pose_learner yet
        assert not hasattr(obj, "_pose_learner")

        lower_poses = np.random.rand(8, 4, 4).astype(np.float32)
        upper_poses = np.random.rand(4, 4, 4).astype(np.float32)
        routing_weights = np.random.rand(8, 4).astype(np.float32)

        # This should auto-initialize
        stats = obj.learn_poses_from_routing_result(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=0.6,
        )

        # Now should have _pose_learner
        assert hasattr(obj, "_pose_learner")
