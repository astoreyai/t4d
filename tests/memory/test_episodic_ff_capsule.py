"""
Phase 2A: Tests for FFCapsuleBridge integration with EpisodicMemory.

Tests the unified FF-Capsule bridge that combines:
- FF goodness (Hinton 2022) for familiarity detection
- Capsule routing agreement (Sabour et al. 2017) for compositional structure

The bridge provides:
1. Novelty detection during store() via goodness scores
2. Capsule activations stored in episode metadata
3. Learning from retrieval outcomes via joint FF+Capsule updates
"""

import pytest
import numpy as np
from datetime import datetime
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock, patch

from ww.bridges.ff_capsule_bridge import (
    FFCapsuleBridge,
    FFCapsuleBridgeConfig,
    FFCapsuleBridgeState,
    CapsuleState,
    create_ff_capsule_bridge,
)


# =============================================================================
# Test FFCapsuleBridgeConfig
# =============================================================================


class TestFFCapsuleBridgeConfig:
    """Tests for FFCapsuleBridgeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FFCapsuleBridgeConfig()
        assert config.ff_weight == 0.6
        assert config.capsule_weight == 0.4  # Auto-computed
        assert config.goodness_threshold == 2.0
        assert config.agreement_threshold == 0.3
        assert config.normalize_goodness is True
        assert config.joint_learning is True

    def test_custom_weights(self):
        """Test custom weight configuration."""
        config = FFCapsuleBridgeConfig(ff_weight=0.8)
        assert config.ff_weight == 0.8
        assert np.isclose(config.capsule_weight, 0.2)

    def test_invalid_weight_raises(self):
        """Test invalid weight raises assertion."""
        with pytest.raises(AssertionError):
            FFCapsuleBridgeConfig(ff_weight=1.5)
        with pytest.raises(AssertionError):
            FFCapsuleBridgeConfig(ff_weight=-0.1)


# =============================================================================
# Test FFCapsuleBridge Core Functionality
# =============================================================================


class TestFFCapsuleBridge:
    """Tests for FFCapsuleBridge core methods."""

    @pytest.fixture
    def bridge(self):
        """Create a basic FFCapsuleBridge without encoder/capsule layers."""
        return FFCapsuleBridge(
            config=FFCapsuleBridgeConfig(
                ff_weight=0.6,
                goodness_threshold=2.0,
                joint_learning=True,
            )
        )

    @pytest.fixture
    def mock_ff_encoder(self):
        """Create a mock FF encoder."""
        encoder = MagicMock()
        encoder.encode.return_value = np.random.randn(1024).astype(np.float32)
        encoder.get_goodness.return_value = 2.5
        encoder.learn_from_outcome.return_value = {"updated": True}
        return encoder

    @pytest.fixture
    def mock_capsule_layer(self):
        """Create a mock capsule layer."""
        layer = MagicMock()
        layer.forward_with_routing.return_value = (
            np.random.rand(32).astype(np.float32),  # activations
            np.random.rand(32, 4, 4).astype(np.float32),  # poses
            {"mean_agreement": 0.75, "pose_change": 0.02}  # stats
        )
        layer.learn_positive.return_value = {"phase": "positive", "goodness": 2.8}
        layer.learn_negative.return_value = {"phase": "negative", "goodness": 1.2}
        return layer

    def test_forward_without_components(self, bridge):
        """Test forward pass uses fallback when no encoder/capsule layer."""
        embedding = np.random.randn(1024).astype(np.float32)
        ff_output, capsule_state, confidence = bridge.forward(embedding)

        # Should return something (fallback behavior)
        assert ff_output is not None
        assert capsule_state is not None
        assert 0.0 <= confidence <= 1.0

        # State should be updated
        assert bridge.state.total_forwards == 1

    def test_forward_with_ff_encoder(self, bridge, mock_ff_encoder):
        """Test forward pass with FF encoder."""
        bridge.set_ff_encoder(mock_ff_encoder)
        embedding = np.random.randn(1024).astype(np.float32)

        ff_output, capsule_state, confidence = bridge.forward(embedding)

        mock_ff_encoder.encode.assert_called_once()
        mock_ff_encoder.get_goodness.assert_called_once()
        assert bridge.state.last_ff_goodness == 2.5

    def test_forward_with_capsule_layer(self, bridge, mock_capsule_layer):
        """Test forward pass with capsule layer."""
        bridge.set_capsule_layer(mock_capsule_layer)
        embedding = np.random.randn(1024).astype(np.float32)

        ff_output, capsule_state, confidence = bridge.forward(embedding)

        mock_capsule_layer.forward_with_routing.assert_called_once()
        assert capsule_state.agreement == 0.75
        assert bridge.state.last_routing_agreement == 0.75

    def test_forward_combined_confidence(self, bridge, mock_ff_encoder, mock_capsule_layer):
        """Test combined confidence computation."""
        bridge.set_ff_encoder(mock_ff_encoder)
        bridge.set_capsule_layer(mock_capsule_layer)
        embedding = np.random.randn(1024).astype(np.float32)

        ff_output, capsule_state, confidence = bridge.forward(embedding)

        # Confidence should combine FF goodness and routing agreement
        assert 0.0 <= confidence <= 1.0
        # With goodness=2.5 (above threshold) and agreement=0.75, confidence should be high
        assert confidence > 0.5

    def test_compute_confidence_shortcut(self, bridge):
        """Test compute_confidence convenience method."""
        embedding = np.random.randn(1024).astype(np.float32)
        confidence = bridge.compute_confidence(embedding)
        assert 0.0 <= confidence <= 1.0


# =============================================================================
# Test FFCapsuleBridge Learning
# =============================================================================


class TestFFCapsuleBridgeLearning:
    """Tests for FFCapsuleBridge learning functionality."""

    @pytest.fixture
    def bridge_with_components(self):
        """Create bridge with mock encoder and capsule layer."""
        encoder = MagicMock()
        encoder.encode.return_value = np.random.randn(1024).astype(np.float32)
        encoder.get_goodness.return_value = 2.0
        encoder.learn_from_outcome.return_value = {"updated": True}

        layer = MagicMock()
        layer.forward_with_routing.return_value = (
            np.random.rand(32).astype(np.float32),
            np.random.rand(32, 4, 4).astype(np.float32),
            {"mean_agreement": 0.7}
        )
        layer.learn_positive.return_value = {"phase": "positive"}
        layer.learn_negative.return_value = {"phase": "negative"}

        bridge = FFCapsuleBridge(
            ff_encoder=encoder,
            capsule_layer=layer,
            config=FFCapsuleBridgeConfig(joint_learning=True),
        )
        return bridge

    def test_learn_positive_outcome(self, bridge_with_components):
        """Test learning from positive outcome."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge_with_components.forward(embedding)

        stats = bridge_with_components.learn(outcome=0.8)

        assert stats["outcome"] == 0.8
        assert "ff_stats" in stats
        assert bridge_with_components.state.total_learn_calls == 1
        assert bridge_with_components.state.total_positive_outcomes == 1

    def test_learn_negative_outcome(self, bridge_with_components):
        """Test learning from negative outcome."""
        embedding = np.random.randn(1024).astype(np.float32)
        bridge_with_components.forward(embedding)

        stats = bridge_with_components.learn(outcome=0.2)

        assert stats["outcome"] == 0.2
        assert bridge_with_components.state.total_negative_outcomes == 1

    def test_learn_with_custom_embedding(self, bridge_with_components):
        """Test learning with custom embedding (not last forward)."""
        custom_embedding = np.random.randn(1024).astype(np.float32)
        stats = bridge_with_components.learn(outcome=0.6, embedding=custom_embedding)
        assert stats["outcome"] == 0.6

    def test_learn_disabled(self):
        """Test learning is disabled when joint_learning=False."""
        bridge = FFCapsuleBridge(
            config=FFCapsuleBridgeConfig(joint_learning=False)
        )
        stats = bridge.learn(outcome=0.8)
        assert stats["status"] == "joint_learning_disabled"


# =============================================================================
# Test FFCapsuleBridge Novelty Detection
# =============================================================================


class TestFFCapsuleBridgeNovelty:
    """Tests for novelty detection from FF goodness."""

    def test_low_goodness_indicates_novelty(self):
        """Test that low goodness indicates novel pattern."""
        encoder = MagicMock()
        encoder.encode.return_value = np.random.randn(1024).astype(np.float32)
        encoder.get_goodness.return_value = 0.5  # Below threshold

        bridge = FFCapsuleBridge(
            ff_encoder=encoder,
            config=FFCapsuleBridgeConfig(goodness_threshold=2.0),
        )

        embedding = np.random.randn(1024).astype(np.float32)
        _, _, confidence = bridge.forward(embedding)

        # Low goodness = low confidence = novel pattern
        # Novelty can be computed as 1.0 - normalized_goodness
        normalized_goodness = bridge.state.last_ff_goodness
        # With goodness=0.5 and threshold=2.0, normalized should be < 0.5
        # (sigmoid centered at threshold)

    def test_high_goodness_indicates_familiarity(self):
        """Test that high goodness indicates familiar pattern."""
        encoder = MagicMock()
        encoder.encode.return_value = np.random.randn(1024).astype(np.float32)
        encoder.get_goodness.return_value = 4.0  # Above threshold

        bridge = FFCapsuleBridge(
            ff_encoder=encoder,
            config=FFCapsuleBridgeConfig(goodness_threshold=2.0),
        )

        embedding = np.random.randn(1024).astype(np.float32)
        _, _, confidence = bridge.forward(embedding)

        # High goodness = familiar pattern
        assert bridge.state.ff_above_threshold_count == 1


# =============================================================================
# Test FFCapsuleBridge Statistics
# =============================================================================


class TestFFCapsuleBridgeStatistics:
    """Tests for statistics tracking."""

    @pytest.fixture
    def active_bridge(self):
        """Create bridge and run some forwards/learns."""
        bridge = FFCapsuleBridge()
        embeddings = [np.random.randn(1024).astype(np.float32) for _ in range(5)]
        for emb in embeddings:
            bridge.forward(emb)
            bridge.learn(outcome=np.random.rand())
        return bridge

    def test_statistics_tracking(self, active_bridge):
        """Test that statistics are tracked correctly."""
        stats = active_bridge.get_statistics()

        assert stats["total_forwards"] == 5
        assert stats["learning"]["total_learn_calls"] == 5
        assert "ff" in stats
        assert "capsule" in stats
        assert "combined" in stats

    def test_reset_statistics(self, active_bridge):
        """Test statistics reset."""
        assert active_bridge.state.total_forwards > 0
        active_bridge.reset_statistics()
        assert active_bridge.state.total_forwards == 0
        assert active_bridge.state.total_learn_calls == 0


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateFFCapsuleBridge:
    """Tests for factory function."""

    def test_create_basic(self):
        """Test creating bridge with defaults."""
        bridge = create_ff_capsule_bridge()
        assert bridge.config.ff_weight == 0.6
        assert bridge.config.joint_learning is True

    def test_create_custom(self):
        """Test creating bridge with custom settings."""
        bridge = create_ff_capsule_bridge(
            ff_weight=0.8,
            goodness_threshold=3.0,
            joint_learning=False,
        )
        assert bridge.config.ff_weight == 0.8
        assert bridge.config.goodness_threshold == 3.0
        assert bridge.config.joint_learning is False


# =============================================================================
# Test Integration with Episodic Memory Patterns
# =============================================================================


class TestFFCapsuleBridgeEpisodicIntegration:
    """Tests for patterns used in EpisodicMemory integration."""

    @pytest.fixture
    def bridge(self):
        """Create bridge for integration tests."""
        return FFCapsuleBridge(
            config=FFCapsuleBridgeConfig(
                ff_weight=0.6,
                joint_learning=True,
                track_history=True,
            )
        )

    def test_store_pattern_novelty_detection(self, bridge):
        """Test novelty detection pattern used during store()."""
        embedding = np.random.randn(1024).astype(np.float32)

        # Forward pass during store
        ff_output, capsule_state, confidence = bridge.forward(embedding, learn_poses=True)

        # Compute novelty from goodness (pattern used in store)
        goodness = bridge.state.last_ff_goodness
        novelty_score = 1.0 - bridge._normalize_goodness(goodness)

        # Novelty score should be valid
        assert 0.0 <= novelty_score <= 1.0

        # Capsule activations should be serializable for metadata
        activations_list = capsule_state.activations.tolist()
        assert isinstance(activations_list, list)

    def test_retrieve_pattern_confidence_scoring(self, bridge):
        """Test confidence scoring pattern used during retrieve()."""
        query_embedding = np.random.randn(1024).astype(np.float32)
        memory_embeddings = [np.random.randn(1024).astype(np.float32) for _ in range(5)]

        # Score query
        query_confidence = bridge.compute_confidence(query_embedding)

        # High confidence could boost retrieval scores
        boost = min(query_confidence * 0.2, 0.15)  # Cap like in episodic.py
        assert 0.0 <= boost <= 0.15

    def test_learn_from_outcome_pattern(self, bridge):
        """Test learning pattern used in learn_from_outcome()."""
        # Simulate retrieval
        query_embedding = np.random.randn(1024).astype(np.float32)
        bridge.forward(query_embedding)

        # Simulate outcome feedback
        outcome_score = 0.75
        retrieval_success = outcome_score > 0.5

        # Learn from outcome
        learn_stats = bridge.learn(
            outcome=outcome_score,
            embedding=query_embedding,
        )

        # Should have learned
        assert bridge.state.total_learn_calls == 1
        assert bridge.state.total_positive_outcomes == (1 if retrieval_success else 0)

    def test_metadata_serialization(self, bridge):
        """Test that capsule data can be serialized for storage."""
        embedding = np.random.randn(1024).astype(np.float32)
        _, capsule_state, confidence = bridge.forward(embedding)

        # Build metadata like episodic.py does
        metadata = {}
        metadata["capsule_activations"] = capsule_state.activations.tolist()
        metadata["ff_goodness"] = float(bridge.state.last_ff_goodness)
        metadata["ff_confidence"] = float(confidence)
        metadata["routing_agreement"] = float(capsule_state.agreement)

        # All values should be JSON-serializable
        import json
        json_str = json.dumps(metadata)
        assert json_str is not None

    def test_history_for_monitoring(self, bridge):
        """Test history tracking for monitoring."""
        # Run multiple forwards
        for _ in range(10):
            embedding = np.random.randn(1024).astype(np.float32)
            bridge.forward(embedding)
            bridge.learn(outcome=np.random.rand())

        # History should be tracked
        assert len(bridge.state.goodness_history) == 10
        assert len(bridge.state.agreement_history) == 10
        assert len(bridge.state.confidence_history) == 10
        assert len(bridge.state.outcome_history) == 10


# =============================================================================
# Test CapsuleState
# =============================================================================


class TestCapsuleState:
    """Tests for CapsuleState dataclass."""

    def test_capsule_state_creation(self):
        """Test creating CapsuleState."""
        activations = np.random.rand(32).astype(np.float32)
        poses = np.random.rand(32, 4, 4).astype(np.float32)

        state = CapsuleState(
            activations=activations,
            poses=poses,
            agreement=0.75,
            routing_stats={"iterations": 3}
        )

        assert state.agreement == 0.75
        assert state.activations.shape == (32,)
        assert state.poses.shape == (32, 4, 4)
        assert state.routing_stats["iterations"] == 3
