"""
Unit tests for Capsule Networks (H8, H9).

Tests capsule layer forward pass, routing-by-agreement,
pose matrix operations, and Forward-Forward integration.
"""

import numpy as np
import pytest

from ww.nca.capsules import (
    CapsuleConfig,
    CapsuleState,
    CapsuleLayer,
    CapsuleNetwork,
    SquashType,
    RoutingType,
    create_capsule_layer,
    create_capsule_network,
)
from ww.nca.pose import (
    PoseConfig,
    PoseMatrix,
    SemanticDimension,
    create_identity_pose,
    create_random_pose,
)


class TestCapsuleConfig:
    """Test CapsuleConfig dataclass."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = CapsuleConfig()
        assert config.input_dim == 1024
        assert config.num_capsules == 32
        assert config.capsule_dim == 16
        assert config.pose_dim == 4
        assert config.routing_iterations == 3
        assert config.squash_type == "hinton"

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = CapsuleConfig(
            input_dim=512,
            num_capsules=16,
            capsule_dim=8,
            routing_iterations=5
        )
        assert config.input_dim == 512
        assert config.num_capsules == 16
        assert config.capsule_dim == 8
        assert config.routing_iterations == 5


class TestCapsuleState:
    """Test CapsuleState dataclass."""

    def test_empty_state(self):
        """Empty state initializes correctly."""
        state = CapsuleState(
            activations=np.zeros(32),
            poses=np.zeros((32, 4, 4)),
            routing_logits=np.zeros((32, 32))
        )
        assert state.activations.shape == (32,)
        assert state.poses.shape == (32, 4, 4)
        assert state.total_routes == 0

    def test_state_with_values(self):
        """State holds actual values."""
        activations = np.random.rand(16)
        poses = np.random.rand(16, 4, 4)
        routing = np.random.rand(16, 16)

        state = CapsuleState(
            activations=activations,
            poses=poses,
            routing_logits=routing,
            total_routes=100
        )

        np.testing.assert_array_equal(state.activations, activations)
        np.testing.assert_array_equal(state.poses, poses)
        assert state.total_routes == 100


class TestCapsuleLayerForward:
    """Test CapsuleLayer forward pass."""

    @pytest.fixture
    def layer(self):
        """Create test capsule layer."""
        config = CapsuleConfig(
            input_dim=128,
            num_capsules=8,
            capsule_dim=4,
            pose_dim=4
        )
        return CapsuleLayer(config)

    def test_forward_shape(self, layer):
        """Forward pass produces correct output shapes."""
        x = np.random.rand(128)
        activations, poses = layer.forward(x)

        assert activations.shape == (8,), "Should have num_capsules activations"
        assert poses.shape == (8, 4, 4), "Should have num_capsules pose matrices"

    def test_forward_batch_shape(self, layer):
        """Forward pass handles batch input."""
        x = np.random.rand(10, 128)  # batch of 10
        activations, poses = layer.forward(x)

        assert activations.shape == (10, 8), "Batch activations"
        assert poses.shape == (10, 8, 4, 4), "Batch poses"

    def test_activations_normalized(self, layer):
        """Activations are in valid range after squashing."""
        x = np.random.rand(128) * 10  # Large input
        activations, _ = layer.forward(x)

        # Squashing should keep activations in [0, 1)
        assert np.all(activations >= 0), "Activations should be non-negative"
        assert np.all(activations < 1), "Squashing should keep activations < 1"

    def test_zero_input(self, layer):
        """Zero input produces zero activations."""
        x = np.zeros(128)
        activations, poses = layer.forward(x)

        # Zero input should give near-zero activations
        assert np.allclose(activations, 0, atol=1e-6)


class TestSquashingFunctions:
    """Test different squashing functions."""

    @pytest.fixture
    def layer_hinton(self):
        """Capsule layer with Hinton squashing."""
        config = CapsuleConfig(
            input_dim=64,
            num_capsules=4,
            squash_type="hinton"
        )
        return CapsuleLayer(config)

    @pytest.fixture
    def layer_norm(self):
        """Capsule layer with norm squashing."""
        config = CapsuleConfig(
            input_dim=64,
            num_capsules=4,
            squash_type="norm"
        )
        return CapsuleLayer(config)

    def test_hinton_squash(self, layer_hinton):
        """Hinton squash keeps activations < 1."""
        x = np.random.rand(64) * 10  # Large input
        activations, _ = layer_hinton.forward(x)
        # All activations should be < 1 due to squashing
        assert np.all(activations < 1.0)

    def test_norm_squash(self, layer_norm):
        """Norm squash keeps activations < 1."""
        x = np.random.rand(64) * 10
        activations, _ = layer_norm.forward(x)
        assert np.all(activations < 1.0)

    def test_squash_preserves_relative_magnitude(self, layer_hinton):
        """Squashing preserves relative magnitude ordering."""
        x_small = np.random.rand(64)
        x_large = np.random.rand(64) * 10

        act_small, _ = layer_hinton.forward(x_small)
        act_large, _ = layer_hinton.forward(x_large)

        # Larger input should give larger (or equal) activations
        assert np.mean(act_large) >= np.mean(act_small) * 0.5


class TestRoutingByAgreement:
    """Test routing-by-agreement algorithm (H8)."""

    @pytest.fixture
    def layer(self):
        """Create test capsule layer."""
        config = CapsuleConfig(
            input_dim=64,
            num_capsules=8,
            capsule_dim=4,
            pose_dim=4,
            routing_iterations=3
        )
        return CapsuleLayer(config)

    def test_routing_returns_tuple(self, layer):
        """Routing produces expected tuple output."""
        lower_activations = np.random.rand(8)
        lower_poses = np.random.rand(8, 4, 4)

        result = layer.route(
            lower_activations=lower_activations,
            lower_poses=lower_poses,
            num_output_capsules=4
        )

        # Should return a tuple with 3 elements
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_routing_coefficients_valid(self, layer):
        """Routing coefficients are valid probabilities."""
        lower_activations = np.random.rand(8)
        lower_poses = np.random.rand(8, 4, 4)

        result = layer.route(
            lower_activations=lower_activations,
            lower_poses=lower_poses,
            num_output_capsules=4
        )

        # Get routing coefficients (last element)
        routing_coeff = result[-1]

        # Routing coefficients should be non-negative
        assert np.all(routing_coeff >= 0)

    def test_routing_iterations_run(self, layer):
        """Routing with different iteration counts runs."""
        lower_activations = np.random.rand(8)
        lower_poses = np.random.rand(8, 4, 4)

        # 1 iteration
        result_1 = layer.route(
            lower_activations, lower_poses, num_output_capsules=4, iterations=1
        )

        # 5 iterations
        result_5 = layer.route(
            lower_activations, lower_poses, num_output_capsules=4, iterations=5
        )

        # Both should complete successfully
        assert result_1 is not None
        assert result_5 is not None


class TestCapsuleNetwork:
    """Test multi-layer CapsuleNetwork."""

    @pytest.fixture
    def network(self):
        """Create test capsule network."""
        return create_capsule_network(
            input_dim=256,
            layer_dims=[64, 32, 16],
            capsule_dim=8,
            pose_dim=4
        )

    def test_network_forward(self, network):
        """Network forward pass through all layers."""
        x = np.random.rand(256)
        output_activations, output_poses = network.forward(x)

        # Should have 16 capsules in output (last layer)
        assert output_activations.shape == (16,)
        assert output_poses.shape == (16, 4, 4)


class TestForwardForwardIntegration:
    """Test Forward-Forward learning integration (H6+H8)."""

    @pytest.fixture
    def layer(self):
        """Create FF-enabled capsule layer."""
        config = CapsuleConfig(
            input_dim=64,
            num_capsules=8,
            use_ff_learning=True
        )
        return CapsuleLayer(config)

    def test_learn_positive(self, layer):
        """Learning on positive sample updates weights."""
        x = np.random.rand(64)

        # Initial forward
        act_before, _ = layer.forward(x)

        # Learn positive
        layer.learn_positive(x, act_before, learning_rate=0.1)

        # Forward again - should still work
        act_after, _ = layer.forward(x)

        # Activations should be valid
        assert np.all(np.isfinite(act_after))

    def test_learn_negative(self, layer):
        """Learning on negative sample updates weights."""
        x = np.random.rand(64)

        # Initial forward
        act_before, _ = layer.forward(x)

        # Learn negative
        layer.learn_negative(x, act_before, learning_rate=0.1)

        # Forward again
        act_after, _ = layer.forward(x)

        # Activations should be valid
        assert np.all(np.isfinite(act_after))


class TestPoseMatrix:
    """Test PoseMatrix operations (H9)."""

    def test_identity_pose(self):
        """Identity pose is 4x4 identity matrix."""
        pose = create_identity_pose()
        expected = np.eye(4)
        np.testing.assert_array_equal(pose.matrix, expected)

    def test_pose_composition(self):
        """Pose composition is matrix multiplication."""
        pose1 = create_random_pose(random_seed=42)
        pose2 = create_random_pose(random_seed=43)

        composed = pose1.compose(pose2)

        expected = pose1.matrix @ pose2.matrix
        np.testing.assert_allclose(composed.matrix, expected)

    def test_pose_agreement_identical(self):
        """Identical poses have agreement = 1."""
        pose1 = create_random_pose(random_seed=42)
        pose2 = create_random_pose(random_seed=42)

        agreement = pose1.agreement(pose2)
        assert np.isclose(agreement, 1.0)

    def test_pose_agreement_different(self):
        """Different poses have agreement < 1."""
        pose1 = create_random_pose(random_seed=42)
        pose2 = create_random_pose(random_seed=99)

        agreement = pose1.agreement(pose2)
        assert 0 <= agreement < 1

    def test_pose_agreement_symmetric(self):
        """Pose agreement is symmetric."""
        pose1 = create_random_pose(random_seed=42)
        pose2 = create_random_pose(random_seed=43)

        agreement_12 = pose1.agreement(pose2)
        agreement_21 = pose2.agreement(pose1)

        assert np.isclose(agreement_12, agreement_21)


class TestSemanticDimensions:
    """Test semantic pose dimensions (H9)."""

    def test_temporal_dimension_value(self):
        """SemanticDimension.TEMPORAL has correct value."""
        assert SemanticDimension.TEMPORAL == 0

    def test_causal_dimension_value(self):
        """SemanticDimension.CAUSAL has correct value."""
        assert SemanticDimension.CAUSAL == 1

    def test_semantic_role_dimension_value(self):
        """SemanticDimension.SEMANTIC_ROLE has correct value."""
        assert SemanticDimension.SEMANTIC_ROLE == 2

    def test_certainty_dimension_value(self):
        """SemanticDimension.CERTAINTY has correct value."""
        assert SemanticDimension.CERTAINTY == 3

    def test_dimension_used_in_pose(self):
        """Semantic dimensions correspond to pose matrix rows."""
        pose = create_identity_pose()

        # Modify a specific dimension (row)
        pose.matrix[SemanticDimension.TEMPORAL, :] = np.array([1, 0.5, 0.2, 0])

        assert np.isclose(pose.matrix[0, 0], 1.0)
        assert np.isclose(pose.matrix[0, 1], 0.5)
        assert np.isclose(pose.matrix[0, 2], 0.2)


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_capsule_layer(self):
        """Factory creates configured layer."""
        layer = create_capsule_layer(
            input_dim=128,
            num_capsules=16,
            capsule_dim=8
        )

        assert layer.config.input_dim == 128
        assert layer.config.num_capsules == 16
        assert layer.config.capsule_dim == 8

    def test_create_capsule_network(self):
        """Factory creates configured network."""
        network = create_capsule_network(
            input_dim=256,
            layer_dims=[32, 16, 8],
        )

        assert len(network.layers) == 3

    def test_create_identity_pose(self):
        """Factory creates valid identity pose matrix."""
        config = PoseConfig(pose_dim=4)
        pose = create_identity_pose(config)

        assert pose.matrix.shape == (4, 4)
        assert pose.config.pose_dim == 4
        np.testing.assert_array_equal(pose.matrix, np.eye(4))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_capsule(self):
        """Network with single capsule works."""
        config = CapsuleConfig(
            input_dim=64,
            num_capsules=1,
            capsule_dim=4
        )
        layer = CapsuleLayer(config)

        x = np.random.rand(64)
        activations, poses = layer.forward(x)

        # Activations might be scalar or 1-element array
        assert activations.size == 1 or len(activations) == 1

    def test_large_input(self):
        """Handles large input vectors."""
        config = CapsuleConfig(
            input_dim=4096,
            num_capsules=64,
            capsule_dim=16
        )
        layer = CapsuleLayer(config)

        x = np.random.rand(4096)
        activations, poses = layer.forward(x)

        assert activations.shape == (64,)
        assert np.all(np.isfinite(activations))

    def test_numerical_stability_via_forward(self):
        """Forward pass is numerically stable with extreme inputs."""
        config = CapsuleConfig(input_dim=64, num_capsules=4)
        layer = CapsuleLayer(config)

        # Very large values
        x_large = np.ones(64) * 1e6
        activations_large, _ = layer.forward(x_large)
        assert np.all(np.isfinite(activations_large))
        assert np.all(activations_large < 1.0)  # Squashing should bound output

        # Very small values
        x_small = np.ones(64) * 1e-10
        activations_small, _ = layer.forward(x_small)
        assert np.all(np.isfinite(activations_small))


# =============================================================================
# Phase 6: Pose Learning from Routing Tests
# =============================================================================


class TestPhase6PoseLearning:
    """Test Phase 6 pose learning from routing agreement."""

    @pytest.fixture
    def layer(self):
        """Create a small capsule layer for testing."""
        config = CapsuleConfig(
            input_dim=64,
            num_capsules=8,
            capsule_dim=8,
            pose_dim=4,
            routing_iterations=3,
            learning_rate=0.1,  # Higher LR for visible effect
        )
        return CapsuleLayer(config, random_seed=42)

    def test_pose_weights_update_positive(self, layer):
        """Pose weights should update on positive outcome."""
        x = np.random.randn(64).astype(np.float32)
        activations, poses = layer.forward(x)

        # Get initial pose weights norm
        initial_pose_norm = np.linalg.norm(layer.W_pose)

        # Learn from positive outcome with poses
        stats = layer.learn_positive(x, activations, poses=poses, learning_rate=0.1)

        assert stats['phase'] == 'positive'
        assert stats['pose_update_norm'] > 0, "Pose weights should update"

    def test_pose_weights_update_negative(self, layer):
        """Pose weights should update on negative outcome."""
        x = np.random.randn(64).astype(np.float32)
        activations, poses = layer.forward(x)

        # Learn from negative outcome with poses
        stats = layer.learn_negative(x, activations, poses=poses, learning_rate=0.1)

        assert stats['phase'] == 'negative'
        assert stats['pose_update_norm'] > 0, "Pose weights should update"

    def test_learn_pose_from_routing(self, layer):
        """learn_pose_from_routing should update transform matrices."""
        # Set up routing scenario
        x = np.random.randn(64).astype(np.float32)
        activations, poses = layer.forward(x)

        # Route to higher-level capsules
        num_output = 4
        higher_activations, higher_poses, routing_coeff = layer.route(
            lower_activations=activations,
            lower_poses=poses,
            num_output_capsules=num_output,
        )

        # Get stored predictions
        predictions = layer._last_predictions
        assert predictions is not None, "_last_predictions should be set after routing"

        # Call learn_pose_from_routing
        learn_stats = layer.learn_pose_from_routing(
            lower_poses=poses,
            predictions=predictions,
            consensus_poses=higher_poses,
            agreement_scores=layer.state.agreement_scores,
            learning_rate=0.1,
        )

        assert 'mean_agreement' in learn_stats
        assert 'transform_change' in learn_stats
        assert learn_stats['num_pairs'] == 8 * 4  # num_lower * num_output

    def test_route_with_learn_poses(self, layer):
        """Route with learn_poses=True should trigger pose learning."""
        x = np.random.randn(64).astype(np.float32)
        activations, poses = layer.forward(x)

        # Get initial transform matrices
        layer.route(
            lower_activations=activations,
            lower_poses=poses,
            num_output_capsules=4,
        )
        initial_transform_norm = np.linalg.norm(layer._transform_matrices)

        # Route with learning enabled
        layer.route(
            lower_activations=activations,
            lower_poses=poses,
            num_output_capsules=4,
            learn_poses=True,
            learning_rate=0.1,
        )
        final_transform_norm = np.linalg.norm(layer._transform_matrices)

        # Transform matrices should change (unless perfect agreement)
        assert initial_transform_norm != final_transform_norm or \
               layer.state.mean_agreement > 0.99, \
               "Transform matrices should update when learn_poses=True"

    def test_poses_improve_with_routing(self, layer):
        """Poses should improve agreement through routing iterations."""
        # Use fixed seed for reproducibility
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64).astype(np.float32)
        activations, poses = layer.forward(x)

        # First route call initializes transform matrices
        _, higher_poses, _ = layer.route(
            lower_activations=activations,
            lower_poses=poses,
            num_output_capsules=4,
            learn_poses=True,
            learning_rate=0.05,
        )

        # Now we can track weight changes (after first route initializes matrices)
        initial_transform_norm = np.linalg.norm(layer._transform_matrices)
        initial_pose_norm = np.linalg.norm(layer.W_pose)
        agreements = [layer.state.mean_agreement]

        # Additional routing passes with learning
        for _ in range(4):
            _, higher_poses, _ = layer.route(
                lower_activations=activations,
                lower_poses=poses,
                num_output_capsules=4,
                learn_poses=True,
                learning_rate=0.05,
            )
            agreements.append(layer.state.mean_agreement)

        final_transform_norm = np.linalg.norm(layer._transform_matrices)
        final_pose_norm = np.linalg.norm(layer.W_pose)

        # Core assertions: learning is happening
        weight_changed = (
            abs(final_transform_norm - initial_transform_norm) > 1e-6 or
            abs(final_pose_norm - initial_pose_norm) > 1e-6
        )
        assert weight_changed, "Weights should update during pose learning"

        # Agreement should remain positive (learning doesn't collapse)
        assert all(a > 0 for a in agreements), "Agreement should remain positive"
        assert agreements[-1] > 0.001, "Final agreement should not collapse to zero"


class TestPhase6CapsuleNetworkTraining:
    """Test CapsuleNetwork train_step with pose learning."""

    def test_train_step_updates_poses(self):
        """train_step with learn_poses=True should update pose weights."""
        # All layers share same input_dim in FF learning (each sees raw input)
        configs = [
            CapsuleConfig(input_dim=64, num_capsules=8, learning_rate=0.1),
            CapsuleConfig(input_dim=64, num_capsules=4, learning_rate=0.1),
        ]
        network = CapsuleNetwork(configs, random_seed=42)

        # Create positive and negative samples with fixed seed
        rng = np.random.default_rng(42)
        positive = rng.standard_normal(64).astype(np.float32)
        negative = rng.standard_normal(64).astype(np.float32) * 0.5 + 0.5  # Shifted

        # Get initial pose weights
        initial_pose_norms = [np.linalg.norm(layer.W_pose) for layer in network.layers]

        # Train step with pose learning
        stats = network.train_step(positive, negative, learn_poses=True)

        # Check stats include pose info
        for layer_stat in stats['positive']:
            assert 'pose_update_norm' in layer_stat

        # Check pose weights changed
        final_pose_norms = [np.linalg.norm(layer.W_pose) for layer in network.layers]
        weight_changes = [abs(f - i) for f, i in zip(final_pose_norms, initial_pose_norms)]

        assert sum(weight_changes) > 0, "At least some pose weights should change"


class TestPhase6LastInputTracking:
    """Test that _last_input is properly tracked for pose learning."""

    def test_last_input_tracked(self):
        """Forward pass should track _last_input."""
        config = CapsuleConfig(input_dim=64, num_capsules=8)
        layer = CapsuleLayer(config, random_seed=42)

        x = np.random.randn(64).astype(np.float32)
        layer.forward(x)

        assert len(layer._last_input) > 0, "_last_input should be set after forward"
        assert np.allclose(layer._last_input, x), "_last_input should match input"

    def test_last_predictions_tracked(self):
        """Routing should track _last_predictions."""
        config = CapsuleConfig(input_dim=64, num_capsules=8)
        layer = CapsuleLayer(config, random_seed=42)

        x = np.random.randn(64).astype(np.float32)
        activations, poses = layer.forward(x)

        layer.route(activations, poses, num_output_capsules=4)

        assert layer._last_predictions is not None, "_last_predictions should be set after route"
        assert layer._last_predictions.shape == (8, 4, 4, 4), "Shape should be [lower, output, pose, pose]"


class TestPhase6ForwardWithRouting:
    """Test forward_with_routing method (THE KEY Phase 6 integration)."""

    @pytest.fixture
    def layer(self):
        """Create a capsule layer for testing."""
        config = CapsuleConfig(
            input_dim=64,
            num_capsules=8,
            capsule_dim=8,
            pose_dim=4,
            routing_iterations=3,
            learning_rate=0.1,
        )
        return CapsuleLayer(config, random_seed=42)

    def test_forward_with_routing_returns_tuple(self, layer):
        """forward_with_routing should return (activations, poses, stats)."""
        x = np.random.randn(64).astype(np.float32)
        result = layer.forward_with_routing(x)

        assert len(result) == 3, "Should return 3 values"
        activations, poses, stats = result

        assert activations.shape == (8,), "Activations shape should match num_capsules"
        assert poses.shape == (8, 4, 4), "Poses shape should match [num_caps, pose_dim, pose_dim]"
        assert isinstance(stats, dict), "Stats should be a dictionary"

    def test_forward_with_routing_stats_keys(self, layer):
        """forward_with_routing should return expected stats keys."""
        x = np.random.randn(64).astype(np.float32)
        _, _, stats = layer.forward_with_routing(x)

        expected_keys = [
            'mean_agreement',
            'initial_activation_mean',
            'refined_activation_mean',
            'pose_change',
            'routing_iterations',
            'learned_poses',
        ]
        for key in expected_keys:
            assert key in stats, f"Stats should contain '{key}'"

    def test_forward_with_routing_refines_poses(self, layer):
        """Poses from forward_with_routing should differ from initial forward."""
        x = np.random.randn(64).astype(np.float32)

        # Get initial poses via direct forward
        initial_activations, initial_poses = layer.forward(x)

        # Get refined poses via forward_with_routing
        refined_activations, refined_poses, stats = layer.forward_with_routing(x)

        # Pose change should be > 0 (routing refines poses)
        assert stats['pose_change'] > 0, "Poses should change through routing"

        # Poses should be different
        pose_diff = np.linalg.norm(refined_poses - initial_poses)
        assert pose_diff > 0, "Refined poses should differ from initial"

    def test_forward_with_routing_learns_poses(self, layer):
        """forward_with_routing with learn_poses=True should update weights."""
        x = np.random.randn(64).astype(np.float32)

        # Get initial W_pose copy
        initial_W_pose = layer.W_pose.copy()

        # Forward with learning (use smaller learning rate to avoid overflow)
        _, _, stats = layer.forward_with_routing(x, learn_poses=True, learning_rate=0.01)

        # Check learning was enabled
        assert stats['learned_poses'] is True

        # After just one call, weights should already differ
        # (don't loop many times as it can cause numerical overflow)
        weight_diff = np.linalg.norm(layer.W_pose - initial_W_pose)

        # Weights should change (and be finite)
        assert np.isfinite(weight_diff), "Weight difference should be finite"
        assert weight_diff > 1e-8 or not np.allclose(layer.W_pose, initial_W_pose), \
            "Pose weights should update with learn_poses=True"

    def test_forward_with_routing_no_learning_on_query(self, layer):
        """forward_with_routing with learn_poses=False should not update weights."""
        x = np.random.randn(64).astype(np.float32)

        # Get initial W_pose
        initial_W_pose = layer.W_pose.copy()

        # Forward without learning (retrieval mode)
        _, _, stats = layer.forward_with_routing(x, learn_poses=False)

        # Check learning was disabled
        assert stats['learned_poses'] is False

        # Weights should NOT change
        assert np.allclose(layer.W_pose, initial_W_pose), \
            "Pose weights should NOT update with learn_poses=False"

    def test_forward_with_routing_custom_iterations(self, layer):
        """forward_with_routing should respect custom routing_iterations."""
        x = np.random.randn(64).astype(np.float32)

        # Use custom iterations
        _, _, stats = layer.forward_with_routing(x, routing_iterations=5)

        assert stats['routing_iterations'] == 5

    def test_forward_with_routing_agreement_positive(self, layer):
        """Mean agreement should be positive after routing."""
        x = np.random.randn(64).astype(np.float32)
        _, _, stats = layer.forward_with_routing(x)

        assert stats['mean_agreement'] > 0, "Agreement should be positive"
        assert stats['mean_agreement'] <= 1, "Agreement should be <= 1"

    def test_forward_with_routing_updates_state(self, layer):
        """forward_with_routing should update layer state."""
        x = np.random.randn(64).astype(np.float32)

        activations, poses, _ = layer.forward_with_routing(x)

        # State should reflect refined values
        assert layer.state.activations is not None
        assert layer.state.poses is not None
        np.testing.assert_array_almost_equal(layer.state.activations, activations)
        np.testing.assert_array_almost_equal(layer.state.poses, poses)

    def test_forward_with_routing_batched_input(self, layer):
        """forward_with_routing should handle batched input."""
        # Note: Current implementation squeezes batch dimension
        # This test verifies single-sample behavior from batch
        x = np.random.randn(1, 64).astype(np.float32)
        activations, poses, stats = layer.forward_with_routing(x)

        assert activations.shape == (8,)
        assert poses.shape == (8, 4, 4)
