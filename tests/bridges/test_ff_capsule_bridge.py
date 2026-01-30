"""
Tests for FF-Capsule Bridge (Phase 6A).

Tests the FFCapsuleBridge which unifies Forward-Forward goodness with
capsule routing agreement for combined confidence scoring and joint learning.
"""

import numpy as np
import pytest

from ww.bridges.ff_capsule_bridge import (
    CapsuleState,
    FFCapsuleBridge,
    FFCapsuleBridgeConfig,
    FFCapsuleBridgeState,
    create_ff_capsule_bridge,
)


# =============================================================================
# Mock Classes for Testing
# =============================================================================


class MockFFEncoder:
    """Mock FF encoder for testing."""

    def __init__(self, output_dim: int = 1024):
        self.output_dim = output_dim
        self._fixed_goodness: float | None = None
        self._encode_calls = 0
        self._learn_calls = 0
        self.state = type('State', (), {
            'mean_goodness': 2.0,
            'total_encodes': 0,
        })()

    def encode(self, embedding: np.ndarray, training: bool = False) -> np.ndarray:
        """Mock encode - return transformed embedding."""
        self._encode_calls += 1
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        # Simple projection to output_dim
        if embedding.shape[1] != self.output_dim:
            output = np.zeros(self.output_dim, dtype=np.float32)
            output[:min(embedding.shape[1], self.output_dim)] = embedding[0, :self.output_dim]
        else:
            output = embedding[0].copy()
        return output

    def get_goodness(self, embedding: np.ndarray) -> float:
        """Return fixed or computed goodness."""
        if self._fixed_goodness is not None:
            return self._fixed_goodness
        # Compute proxy goodness
        return float(np.sum(embedding ** 2) / max(len(embedding), 1))

    def learn_from_outcome(
        self,
        embedding: np.ndarray,
        outcome_score: float,
        three_factor_signal=None,
        effective_lr: float | None = None,
    ) -> dict:
        """Mock learning."""
        self._learn_calls += 1
        return {
            "phase": "positive" if outcome_score > 0.5 else "negative",
            "outcome_score": outcome_score,
            "effective_lr": effective_lr or 0.03,
        }

    def set_goodness(self, goodness: float) -> None:
        """Set fixed goodness for testing."""
        self._fixed_goodness = goodness


class MockCapsuleLayer:
    """Mock capsule layer for testing."""

    def __init__(self, num_capsules: int = 32, pose_dim: int = 4):
        self.num_capsules = num_capsules
        self.pose_dim = pose_dim
        self._fixed_agreement: float | None = None
        self._forward_calls = 0
        self._learn_positive_calls = 0
        self._learn_negative_calls = 0
        self.state = type('State', (), {
            'activations': np.zeros(num_capsules),
            'poses': np.zeros((num_capsules, pose_dim, pose_dim)),
            'mean_agreement': 0.5,
        })()

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mock forward pass."""
        self._forward_calls += 1
        activations = np.abs(x[:self.num_capsules]) if len(x) >= self.num_capsules else np.zeros(self.num_capsules)
        activations = activations / (np.linalg.norm(activations) + 1e-8)
        poses = np.eye(self.pose_dim, dtype=np.float32)[np.newaxis, :, :].repeat(self.num_capsules, axis=0)
        return activations, poses

    def forward_with_routing(
        self,
        x: np.ndarray,
        routing_iterations: int | None = None,
        learn_poses: bool = True,
        learning_rate: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Mock forward with routing."""
        activations, poses = self.forward(x)

        if self._fixed_agreement is not None:
            agreement = self._fixed_agreement
        else:
            agreement = float(np.mean(activations))

        routing_stats = {
            'mean_agreement': agreement,
            'routing_iterations': routing_iterations or 3,
            'learned_poses': learn_poses,
            'pose_change': 0.01 if learn_poses else 0.0,
        }

        return activations, poses, routing_stats

    def learn_pose_from_routing(
        self,
        lower_poses: np.ndarray,
        predictions: np.ndarray,
        consensus_poses: np.ndarray,
        agreement_scores: np.ndarray,
        learning_rate: float | None = None,
    ) -> dict:
        """Mock pose learning."""
        return {
            'mean_agreement': float(np.mean(agreement_scores)),
            'learning_rate': learning_rate or 0.01,
        }

    def learn_positive(
        self,
        x: np.ndarray,
        activations: np.ndarray,
        poses: np.ndarray | None = None,
        learning_rate: float | None = None,
    ) -> dict:
        """Mock positive learning."""
        self._learn_positive_calls += 1
        return {
            'phase': 'positive',
            'goodness': float(np.sum(activations ** 2)),
            'learning_rate': learning_rate or 0.01,
        }

    def learn_negative(
        self,
        x: np.ndarray,
        activations: np.ndarray,
        poses: np.ndarray | None = None,
        learning_rate: float | None = None,
    ) -> dict:
        """Mock negative learning."""
        self._learn_negative_calls += 1
        return {
            'phase': 'negative',
            'goodness': float(np.sum(activations ** 2)),
            'learning_rate': learning_rate or 0.01,
        }

    def set_agreement(self, agreement: float) -> None:
        """Set fixed agreement for testing."""
        self._fixed_agreement = agreement


# =============================================================================
# Configuration Tests
# =============================================================================


class TestFFCapsuleBridgeConfig:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FFCapsuleBridgeConfig()
        assert config.ff_weight == 0.6
        assert config.capsule_weight == pytest.approx(0.4, abs=1e-6)  # Auto-computed
        assert config.goodness_threshold == 2.0
        assert config.agreement_threshold == 0.3
        assert config.joint_learning is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = FFCapsuleBridgeConfig(
            ff_weight=0.7,
            goodness_threshold=3.0,
            joint_learning=False
        )
        assert config.ff_weight == 0.7
        assert config.capsule_weight == pytest.approx(0.3, abs=1e-6)
        assert config.goodness_threshold == 3.0
        assert config.joint_learning is False

    def test_capsule_weight_auto_computed(self):
        """Test that capsule_weight is auto-computed from ff_weight."""
        config = FFCapsuleBridgeConfig(ff_weight=0.8)
        assert config.capsule_weight == pytest.approx(0.2, abs=1e-6)

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(AssertionError):
            FFCapsuleBridgeConfig(ff_weight=1.5)  # Out of range

        with pytest.raises(AssertionError):
            FFCapsuleBridgeConfig(ff_weight=-0.1)  # Negative

        with pytest.raises(AssertionError):
            FFCapsuleBridgeConfig(goodness_threshold=0)  # Must be positive

        with pytest.raises(AssertionError):
            FFCapsuleBridgeConfig(agreement_threshold=1.5)  # Out of range


class TestCapsuleState:
    """Tests for CapsuleState dataclass."""

    def test_creation(self):
        """Test CapsuleState creation."""
        activations = np.array([0.1, 0.5, 0.9])
        poses = np.eye(4)[np.newaxis, :, :].repeat(3, axis=0)

        state = CapsuleState(
            activations=activations,
            poses=poses,
            agreement=0.75,
        )

        assert len(state.activations) == 3
        assert state.poses.shape == (3, 4, 4)
        assert state.agreement == 0.75

    def test_routing_stats(self):
        """Test CapsuleState with routing stats."""
        state = CapsuleState(
            activations=np.zeros(32),
            poses=np.zeros((32, 4, 4)),
            agreement=0.5,
            routing_stats={'iterations': 3, 'pose_change': 0.01}
        )
        assert state.routing_stats['iterations'] == 3


# =============================================================================
# Bridge Core Functionality Tests
# =============================================================================


class TestFFCapsuleBridgeInitialization:
    """Tests for bridge initialization."""

    @pytest.fixture
    def mock_encoder(self):
        return MockFFEncoder()

    @pytest.fixture
    def mock_capsule(self):
        return MockCapsuleLayer()

    def test_initialization_with_both(self, mock_encoder, mock_capsule):
        """Test initialization with both encoder and capsule layer."""
        bridge = FFCapsuleBridge(
            ff_encoder=mock_encoder,
            capsule_layer=mock_capsule
        )
        assert bridge.ff_encoder is not None
        assert bridge.capsule_layer is not None
        assert bridge.state.total_forwards == 0

    def test_initialization_encoder_only(self, mock_encoder):
        """Test initialization with encoder only."""
        bridge = FFCapsuleBridge(ff_encoder=mock_encoder)
        assert bridge.ff_encoder is not None
        assert bridge.capsule_layer is None

    def test_initialization_capsule_only(self, mock_capsule):
        """Test initialization with capsule layer only."""
        bridge = FFCapsuleBridge(capsule_layer=mock_capsule)
        assert bridge.ff_encoder is None
        assert bridge.capsule_layer is not None

    def test_initialization_empty(self):
        """Test initialization without any components (fallback mode)."""
        bridge = FFCapsuleBridge()
        assert bridge.ff_encoder is None
        assert bridge.capsule_layer is None

    def test_set_ff_encoder(self, mock_encoder):
        """Test setting FF encoder after initialization."""
        bridge = FFCapsuleBridge()
        bridge.set_ff_encoder(mock_encoder)
        assert bridge.ff_encoder is not None

    def test_set_capsule_layer(self, mock_capsule):
        """Test setting capsule layer after initialization."""
        bridge = FFCapsuleBridge()
        bridge.set_capsule_layer(mock_capsule)
        assert bridge.capsule_layer is not None


class TestFFCapsuleBridgeForward:
    """Tests for forward pass."""

    @pytest.fixture
    def bridge(self):
        return FFCapsuleBridge(
            ff_encoder=MockFFEncoder(),
            capsule_layer=MockCapsuleLayer()
        )

    @pytest.fixture
    def bridge_fallback(self):
        """Bridge without components (uses fallbacks)."""
        return FFCapsuleBridge()

    def test_forward_returns_correct_types(self, bridge):
        """Test forward returns correct types."""
        embedding = np.random.randn(1024).astype(np.float32)
        ff_output, capsule_state, confidence = bridge.forward(embedding)

        assert isinstance(ff_output, np.ndarray)
        assert isinstance(capsule_state, CapsuleState)
        assert isinstance(confidence, float)

    def test_forward_output_shapes(self, bridge):
        """Test forward output shapes."""
        embedding = np.random.randn(1024).astype(np.float32)
        ff_output, capsule_state, confidence = bridge.forward(embedding)

        assert ff_output.shape == (1024,)
        assert capsule_state.activations.shape == (32,)
        assert capsule_state.poses.shape == (32, 4, 4)

    def test_forward_confidence_range(self, bridge):
        """Test confidence is in [0, 1]."""
        for _ in range(20):
            embedding = np.random.randn(1024).astype(np.float32)
            _, _, confidence = bridge.forward(embedding)
            assert 0.0 <= confidence <= 1.0

    def test_forward_updates_state(self, bridge):
        """Test forward updates state."""
        embedding = np.random.randn(1024).astype(np.float32)
        initial_forwards = bridge.state.total_forwards

        bridge.forward(embedding)

        assert bridge.state.total_forwards == initial_forwards + 1
        assert bridge.state.last_ff_goodness > 0
        assert 0.0 <= bridge.state.last_routing_agreement <= 1.0
        assert 0.0 <= bridge.state.last_confidence <= 1.0

    def test_forward_fallback_mode(self, bridge_fallback):
        """Test forward works without encoder/capsule (fallback)."""
        embedding = np.random.randn(1024).astype(np.float32)
        ff_output, capsule_state, confidence = bridge_fallback.forward(embedding)

        # Should still return valid outputs
        assert isinstance(ff_output, np.ndarray)
        assert isinstance(capsule_state, CapsuleState)
        assert 0.0 <= confidence <= 1.0

    def test_forward_stores_last_values(self, bridge):
        """Test forward stores last values for learning."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)

        assert bridge._last_embedding is not None
        assert bridge._last_ff_output is not None
        assert bridge._last_capsule_state is not None


class TestCombinedConfidence:
    """Tests for combined confidence calculation."""

    @pytest.fixture
    def bridge(self):
        return FFCapsuleBridge(
            ff_encoder=MockFFEncoder(),
            capsule_layer=MockCapsuleLayer(),
            config=FFCapsuleBridgeConfig(ff_weight=0.6)
        )

    def test_high_goodness_high_agreement(self, bridge):
        """Test confidence when both signals are high."""
        bridge.ff_encoder.set_goodness(5.0)  # Above threshold
        bridge.capsule_layer.set_agreement(0.9)

        embedding = np.random.randn(1024).astype(np.float32)
        _, _, confidence = bridge.forward(embedding)

        # Both high -> high confidence
        assert confidence > 0.7

    def test_low_goodness_low_agreement(self, bridge):
        """Test confidence when both signals are low."""
        bridge.ff_encoder.set_goodness(0.5)  # Below threshold
        bridge.capsule_layer.set_agreement(0.1)

        embedding = np.random.randn(1024).astype(np.float32)
        _, _, confidence = bridge.forward(embedding)

        # Both low -> low confidence
        assert confidence < 0.5

    def test_high_goodness_low_agreement(self, bridge):
        """Test confidence with mixed signals."""
        bridge.ff_encoder.set_goodness(5.0)  # High
        bridge.capsule_layer.set_agreement(0.1)  # Low

        embedding = np.random.randn(1024).astype(np.float32)
        _, _, confidence = bridge.forward(embedding)

        # Mixed -> medium confidence
        assert 0.3 < confidence < 0.8

    def test_ff_weight_affects_confidence(self):
        """Test that ff_weight properly affects confidence."""
        # High FF weight
        config_high_ff = FFCapsuleBridgeConfig(ff_weight=0.9)
        bridge_high_ff = FFCapsuleBridge(
            ff_encoder=MockFFEncoder(),
            capsule_layer=MockCapsuleLayer(),
            config=config_high_ff
        )
        bridge_high_ff.ff_encoder.set_goodness(5.0)
        bridge_high_ff.capsule_layer.set_agreement(0.1)

        # Low FF weight
        config_low_ff = FFCapsuleBridgeConfig(ff_weight=0.1)
        bridge_low_ff = FFCapsuleBridge(
            ff_encoder=MockFFEncoder(),
            capsule_layer=MockCapsuleLayer(),
            config=config_low_ff
        )
        bridge_low_ff.ff_encoder.set_goodness(5.0)
        bridge_low_ff.capsule_layer.set_agreement(0.1)

        embedding = np.random.randn(1024).astype(np.float32)
        _, _, conf_high_ff = bridge_high_ff.forward(embedding)
        _, _, conf_low_ff = bridge_low_ff.forward(embedding)

        # High FF weight should give higher confidence when FF is good
        assert conf_high_ff > conf_low_ff

    def test_compute_confidence_convenience(self, bridge):
        """Test compute_confidence convenience method."""
        embedding = np.random.randn(1024).astype(np.float32)

        # Both methods should give same result
        _, _, conf1 = bridge.forward(embedding)
        conf2 = bridge.compute_confidence(embedding)

        # Should be similar (might differ slightly due to randomness in mock)
        assert abs(conf1 - conf2) < 0.1


# =============================================================================
# Learning Tests
# =============================================================================


class TestFFCapsuleBridgeLearning:
    """Tests for joint learning."""

    @pytest.fixture
    def bridge(self):
        return FFCapsuleBridge(
            ff_encoder=MockFFEncoder(),
            capsule_layer=MockCapsuleLayer(),
            config=FFCapsuleBridgeConfig(joint_learning=True)
        )

    @pytest.fixture
    def bridge_no_learning(self):
        return FFCapsuleBridge(
            ff_encoder=MockFFEncoder(),
            capsule_layer=MockCapsuleLayer(),
            config=FFCapsuleBridgeConfig(joint_learning=False)
        )

    def test_learn_positive_outcome(self, bridge):
        """Test learning from positive outcome."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)

        stats = bridge.learn(outcome=0.9)

        assert stats['outcome'] == 0.9
        assert 'ff_stats' in stats
        assert 'capsule_stats' in stats
        assert bridge.state.total_positive_outcomes == 1

    def test_learn_negative_outcome(self, bridge):
        """Test learning from negative outcome."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)

        stats = bridge.learn(outcome=0.2)

        assert stats['outcome'] == 0.2
        assert bridge.state.total_negative_outcomes == 1

    def test_learn_updates_state(self, bridge):
        """Test learn updates state."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)

        initial_learns = bridge.state.total_learn_calls

        bridge.learn(outcome=0.8)

        assert bridge.state.total_learn_calls == initial_learns + 1

    def test_learn_without_forward_uses_provided_embedding(self, bridge):
        """Test learn can use provided embedding."""
        embedding = np.random.randn(1024).astype(np.float32)

        stats = bridge.learn(outcome=0.7, embedding=embedding)

        assert 'ff_stats' in stats
        assert stats['ff_stats'].get('phase') == 'positive'

    def test_learn_without_embedding_fails(self, bridge):
        """Test learn without any embedding returns error."""
        # Don't call forward, don't provide embedding
        stats = bridge.learn(outcome=0.5)

        assert 'error' in stats

    def test_learn_disabled(self, bridge_no_learning):
        """Test learning is disabled when configured."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge_no_learning.forward(embedding)

        stats = bridge_no_learning.learn(outcome=0.8)

        assert stats['status'] == 'joint_learning_disabled'
        assert bridge_no_learning.state.total_learn_calls == 0

    def test_learn_calls_ff_encoder(self, bridge):
        """Test learn calls FF encoder's learn method."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)

        initial_calls = bridge.ff_encoder._learn_calls
        bridge.learn(outcome=0.8)

        assert bridge.ff_encoder._learn_calls == initial_calls + 1

    def test_learn_calls_capsule_layer(self, bridge):
        """Test learn calls capsule layer's learn method."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)

        initial_positive = bridge.capsule_layer._learn_positive_calls
        initial_negative = bridge.capsule_layer._learn_negative_calls

        bridge.learn(outcome=0.8)

        assert bridge.capsule_layer._learn_positive_calls == initial_positive + 1

        bridge.learn(outcome=0.2)

        assert bridge.capsule_layer._learn_negative_calls == initial_negative + 1

    def test_learn_with_custom_lr(self, bridge):
        """Test learning with custom learning rate."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)

        stats = bridge.learn(outcome=0.8, effective_lr=0.1)

        assert stats['ff_stats'].get('effective_lr') == 0.1


# =============================================================================
# State Tracking Tests
# =============================================================================


class TestFFCapsuleBridgeState:
    """Tests for state tracking."""

    @pytest.fixture
    def bridge(self):
        return FFCapsuleBridge(
            ff_encoder=MockFFEncoder(),
            capsule_layer=MockCapsuleLayer(),
            config=FFCapsuleBridgeConfig(track_history=True)
        )

    def test_history_tracking(self, bridge):
        """Test history is tracked during forward passes."""
        for _ in range(10):
            embedding = np.random.randn(1024).astype(np.float32)
            bridge.forward(embedding)

        assert len(bridge.state.goodness_history) == 10
        assert len(bridge.state.agreement_history) == 10
        assert len(bridge.state.confidence_history) == 10

    def test_history_bounded(self, bridge):
        """Test history is bounded to max_history."""
        bridge.config.max_history = 5

        for _ in range(20):
            embedding = np.random.randn(1024).astype(np.float32)
            bridge.forward(embedding)

        assert len(bridge.state.goodness_history) <= 5

    def test_mean_values_updated(self, bridge):
        """Test mean values are updated via EMA."""
        for _ in range(100):
            embedding = np.random.randn(1024).astype(np.float32)
            bridge.forward(embedding)

        # Mean values should be reasonable
        assert bridge.state.mean_ff_goodness > 0
        assert 0.0 <= bridge.state.mean_routing_agreement <= 1.0
        assert 0.0 <= bridge.state.mean_confidence <= 1.0

    def test_outcome_history(self, bridge):
        """Test outcome history is tracked."""
        embedding = np.random.randn(1024).astype(np.float32)

        for outcome in [0.9, 0.8, 0.3, 0.2]:
            bridge.forward(embedding)
            bridge.learn(outcome=outcome)

        assert len(bridge.state.outcome_history) == 4

    def test_reset_statistics(self, bridge):
        """Test reset clears all statistics."""
        # Do some operations
        embedding = np.random.randn(1024).astype(np.float32)
        for _ in range(10):
            bridge.forward(embedding)

        bridge.reset_statistics()

        assert bridge.state.total_forwards == 0
        assert len(bridge.state.goodness_history) == 0
        assert bridge._last_embedding is None


class TestGetStatistics:
    """Tests for statistics retrieval."""

    @pytest.fixture
    def bridge(self):
        return FFCapsuleBridge(
            ff_encoder=MockFFEncoder(),
            capsule_layer=MockCapsuleLayer()
        )

    def test_get_statistics_structure(self, bridge):
        """Test statistics have correct structure."""
        # Do some operations
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)
        bridge.learn(outcome=0.8)

        stats = bridge.get_statistics()

        assert 'total_forwards' in stats
        assert 'ff' in stats
        assert 'capsule' in stats
        assert 'combined' in stats
        assert 'learning' in stats
        assert 'config' in stats

    def test_get_statistics_ff_section(self, bridge):
        """Test FF statistics section."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)

        stats = bridge.get_statistics()

        assert 'last_goodness' in stats['ff']
        assert 'mean_goodness' in stats['ff']
        assert 'above_threshold_ratio' in stats['ff']

    def test_get_statistics_capsule_section(self, bridge):
        """Test capsule statistics section."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)

        stats = bridge.get_statistics()

        assert 'last_agreement' in stats['capsule']
        assert 'mean_agreement' in stats['capsule']

    def test_get_statistics_learning_section(self, bridge):
        """Test learning statistics section."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(embedding)
        bridge.learn(outcome=0.8)

        stats = bridge.get_statistics()

        assert 'total_learn_calls' in stats['learning']
        assert 'positive_outcomes' in stats['learning']
        assert 'negative_outcomes' in stats['learning']


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateFFCapsuleBridge:
    """Tests for factory function."""

    def test_factory_default(self):
        """Test factory with defaults."""
        bridge = create_ff_capsule_bridge()

        assert bridge.config.ff_weight == 0.6
        assert bridge.config.goodness_threshold == 2.0
        assert bridge.config.joint_learning is True

    def test_factory_custom(self):
        """Test factory with custom parameters."""
        bridge = create_ff_capsule_bridge(
            ff_weight=0.8,
            goodness_threshold=3.0,
            joint_learning=False
        )

        assert bridge.config.ff_weight == 0.8
        assert bridge.config.goodness_threshold == 3.0
        assert bridge.config.joint_learning is False

    def test_factory_with_encoder(self):
        """Test factory with encoder."""
        encoder = MockFFEncoder()
        bridge = create_ff_capsule_bridge(ff_encoder=encoder)

        assert bridge.ff_encoder is not None

    def test_factory_with_capsule_layer(self):
        """Test factory with capsule layer."""
        capsule = MockCapsuleLayer()
        bridge = create_ff_capsule_bridge(capsule_layer=capsule)

        assert bridge.capsule_layer is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestFFCapsuleIntegration:
    """Integration tests with actual FF and Capsule components."""

    def test_with_real_ff_encoder(self):
        """Test integration with real FFEncoder."""
        try:
            from ww.encoding.ff_encoder import FFEncoder, FFEncoderConfig

            ff_config = FFEncoderConfig(
                input_dim=1024,
                hidden_dims=(256,),
                output_dim=1024
            )
            ff_encoder = FFEncoder(ff_config)

            bridge = FFCapsuleBridge(ff_encoder=ff_encoder)

            embedding = np.random.randn(1024).astype(np.float32)
            ff_output, capsule_state, confidence = bridge.forward(embedding)

            assert ff_output.shape == (1024,)
            assert 0.0 <= confidence <= 1.0

        except ImportError:
            pytest.skip("FFEncoder not available")

    def test_with_real_capsule_layer(self):
        """Test integration with real CapsuleLayer."""
        try:
            from ww.nca.capsules import CapsuleLayer, CapsuleConfig

            caps_config = CapsuleConfig(
                input_dim=1024,
                num_capsules=16,
                capsule_dim=8,
                pose_dim=4
            )
            capsule_layer = CapsuleLayer(caps_config)

            bridge = FFCapsuleBridge(capsule_layer=capsule_layer)

            embedding = np.random.randn(1024).astype(np.float32)
            ff_output, capsule_state, confidence = bridge.forward(embedding)

            assert 0.0 <= confidence <= 1.0
            assert capsule_state.activations is not None

        except ImportError:
            pytest.skip("CapsuleLayer not available")

    def test_with_both_real_components(self):
        """Test integration with both real components."""
        try:
            from ww.encoding.ff_encoder import FFEncoder, FFEncoderConfig
            from ww.nca.capsules import CapsuleLayer, CapsuleConfig

            # Create real components
            ff_config = FFEncoderConfig(input_dim=1024, hidden_dims=(256,), output_dim=1024)
            ff_encoder = FFEncoder(ff_config)

            caps_config = CapsuleConfig(input_dim=1024, num_capsules=16, capsule_dim=8, pose_dim=4)
            capsule_layer = CapsuleLayer(caps_config)

            # Create bridge
            bridge = FFCapsuleBridge(
                ff_encoder=ff_encoder,
                capsule_layer=capsule_layer
            )

            # Test forward
            embedding = np.random.randn(1024).astype(np.float32)
            ff_output, capsule_state, confidence = bridge.forward(embedding)

            assert 0.0 <= confidence <= 1.0

            # Test learning
            stats = bridge.learn(outcome=0.8)
            assert 'ff_stats' in stats

        except ImportError:
            pytest.skip("FFEncoder or CapsuleLayer not available")


class TestFFCapsuleBridgeRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ returns useful string."""
        bridge = FFCapsuleBridge(
            ff_encoder=MockFFEncoder(),
            capsule_layer=MockCapsuleLayer()
        )

        repr_str = repr(bridge)

        assert "FFCapsuleBridge" in repr_str
        assert "ff_weight" in repr_str
        assert "has_encoder=True" in repr_str
        assert "has_capsule=True" in repr_str
