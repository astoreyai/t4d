"""
Tests for Capsule-Retrieval Bridge (P6.2).

Tests the CapsuleRetrievalBridge which uses capsule network representations
to enhance memory retrieval scoring.
"""

import numpy as np
import pytest

from ww.bridges.capsule_bridge import (
    CapsuleBridgeConfig,
    CapsuleBridgeState,
    CapsuleRepresentation,
    CapsuleRetrievalBridge,
    create_capsule_bridge,
)


class MockCapsuleLayer:
    """Mock capsule layer for testing."""

    def __init__(self, num_capsules: int = 32, pose_dim: int = 4):
        self.num_capsules = num_capsules
        self.pose_dim = pose_dim

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate mock activations and poses from embedding."""
        # Derive activations from embedding
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Use parts of embedding for activations
        chunk_size = x.shape[1] // self.num_capsules
        activations = np.zeros(self.num_capsules)
        for i in range(self.num_capsules):
            start = i * chunk_size
            end = start + chunk_size
            activations[i] = np.abs(x[0, start:end].sum() / chunk_size)

        # Normalize activations to [0, 1]
        activations = activations / (activations.max() + 1e-8)

        # Generate poses based on embedding patterns
        poses = np.zeros((self.num_capsules, self.pose_dim, self.pose_dim))
        for i in range(self.num_capsules):
            poses[i] = np.eye(self.pose_dim) * (0.5 + 0.5 * activations[i])
            # Add some embedding-derived rotation
            if i * chunk_size < x.shape[1]:
                angle = x[0, i * chunk_size] * 0.1
                poses[i, 0, 1] = np.sin(angle)
                poses[i, 1, 0] = -np.sin(angle)

        return activations, poses


class TestCapsuleBridgeConfig:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CapsuleBridgeConfig()
        assert config.activation_weight == 0.4
        assert config.pose_weight == 0.6
        assert config.agreement_threshold == 0.3
        assert config.max_boost == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = CapsuleBridgeConfig(
            activation_weight=0.5,
            pose_weight=0.5,
            max_boost=0.5
        )
        assert config.activation_weight == 0.5
        assert config.max_boost == 0.5


class TestCapsuleRepresentation:
    """Tests for capsule representation dataclass."""

    def test_creation(self):
        """Test representation creation."""
        rep = CapsuleRepresentation(
            activations=np.array([0.1, 0.5, 0.9]),
            poses=np.eye(4)[np.newaxis, :, :].repeat(3, axis=0)
        )
        assert len(rep.activations) == 3
        assert rep.poses.shape == (3, 4, 4)


class TestCapsuleRetrievalBridge:
    """Tests for CapsuleRetrievalBridge class."""

    @pytest.fixture
    def mock_layer(self):
        """Create mock capsule layer."""
        return MockCapsuleLayer(num_capsules=32, pose_dim=4)

    @pytest.fixture
    def bridge(self, mock_layer):
        """Create bridge with mock layer."""
        return CapsuleRetrievalBridge(capsule_layer=mock_layer)

    @pytest.fixture
    def bridge_no_layer(self):
        """Create bridge without capsule layer."""
        return CapsuleRetrievalBridge()

    def test_initialization(self, bridge):
        """Test bridge initializes correctly."""
        assert bridge.capsule_layer is not None
        assert bridge.config.activation_weight == 0.4
        assert bridge.state.n_queries == 0

    def test_initialization_no_layer(self, bridge_no_layer):
        """Test bridge works without capsule layer (fallback)."""
        assert bridge_no_layer.capsule_layer is None

    def test_set_capsule_layer(self, bridge_no_layer, mock_layer):
        """Test setting capsule layer."""
        bridge_no_layer.set_capsule_layer(mock_layer)
        assert bridge_no_layer.capsule_layer is not None

    def test_compute_activation_similarity(self, bridge):
        """Test activation similarity computation."""
        query_acts = np.array([0.8, 0.1, 0.6])
        memory_acts = np.array([0.7, 0.2, 0.5])

        sim = bridge.compute_activation_similarity(query_acts, memory_acts)

        assert 0.0 <= sim <= 1.0
        assert sim > 0.5  # Similar patterns

    def test_activation_similarity_identical(self, bridge):
        """Test similarity for identical activations."""
        acts = np.array([0.5, 0.5, 0.5])
        sim = bridge.compute_activation_similarity(acts, acts)
        assert sim > 0.9  # Should be very high

    def test_activation_similarity_different(self, bridge):
        """Test similarity for very different activations."""
        query_acts = np.array([0.9, 0.0, 0.0])
        memory_acts = np.array([0.0, 0.0, 0.9])

        sim = bridge.compute_activation_similarity(query_acts, memory_acts)
        assert sim < 0.5  # Should be low

    def test_activation_similarity_zero(self, bridge):
        """Test similarity with zero activations."""
        query_acts = np.zeros(3)
        memory_acts = np.array([0.5, 0.5, 0.5])

        sim = bridge.compute_activation_similarity(query_acts, memory_acts)
        assert sim == 0.0

    def test_compute_pose_agreement(self, bridge):
        """Test pose agreement computation."""
        # Create similar poses
        query_poses = np.stack([np.eye(4) for _ in range(3)])
        memory_poses = np.stack([np.eye(4) * 1.1 for _ in range(3)])
        acts = np.array([0.8, 0.8, 0.8])

        agreement = bridge.compute_pose_agreement(
            query_poses, memory_poses, acts, acts
        )

        assert 0.0 <= agreement <= 1.0
        assert agreement > 0.5  # Similar poses

    def test_pose_agreement_identical(self, bridge):
        """Test pose agreement for identical poses."""
        poses = np.stack([np.eye(4) for _ in range(3)])
        acts = np.array([0.8, 0.8, 0.8])

        agreement = bridge.compute_pose_agreement(poses, poses, acts, acts)
        assert agreement > 0.95  # Should be very high

    def test_pose_agreement_inactive(self, bridge):
        """Test pose agreement with inactive capsules."""
        poses = np.stack([np.eye(4) for _ in range(3)])
        acts = np.array([0.0, 0.0, 0.0])  # All inactive

        agreement = bridge.compute_pose_agreement(poses, poses, acts, acts)
        assert agreement == 0.0  # No agreement from inactive

    def test_compute_boost(self, bridge):
        """Test boost computation for single memory."""
        query_emb = np.random.randn(1024)
        memory_emb = query_emb + np.random.randn(1024) * 0.1  # Similar

        boost = bridge.compute_boost(query_emb, memory_emb)

        assert 0.0 <= boost <= bridge.config.max_boost

    def test_compute_boost_identical(self, bridge):
        """Test boost for identical embeddings."""
        emb = np.random.randn(1024)
        boost = bridge.compute_boost(emb, emb)

        # Identical should get high boost
        assert boost > 0

    def test_compute_boost_different(self, bridge):
        """Test boost for very different embeddings."""
        query_emb = np.random.randn(1024)
        memory_emb = -query_emb  # Opposite

        boost = bridge.compute_boost(query_emb, memory_emb)

        # Different should get low/no boost
        assert boost <= bridge.config.max_boost

    def test_compute_boosts_batch(self, bridge):
        """Test batch boost computation."""
        query_emb = np.random.randn(1024)
        memory_embs = [np.random.randn(1024) for _ in range(10)]

        boosts = bridge.compute_boosts(query_emb, memory_embs)

        assert len(boosts) == 10
        assert all(0.0 <= b <= bridge.config.max_boost for b in boosts)

    def test_compute_boosts_empty(self, bridge):
        """Test batch with empty memories."""
        query_emb = np.random.randn(1024)
        boosts = bridge.compute_boosts(query_emb, [])
        assert boosts == []

    def test_cache_functionality(self, bridge):
        """Test capsule representation caching."""
        emb = np.random.randn(1024)

        # First access - cache miss
        bridge._get_capsule_representation(emb)
        assert bridge.state.cache_misses == 1
        assert bridge.state.cache_hits == 0

        # Second access - cache hit
        bridge._get_capsule_representation(emb)
        assert bridge.state.cache_hits == 1

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        config = CapsuleBridgeConfig(cache_size=5)
        bridge = CapsuleRetrievalBridge(config=config)

        # Fill cache beyond limit
        for i in range(10):
            emb = np.random.randn(1024) * i  # Different each time
            bridge._get_capsule_representation(emb)

        # Cache should be at limit
        assert len(bridge._cache) <= config.cache_size

    def test_clear_cache(self, bridge):
        """Test cache clearing."""
        emb = np.random.randn(1024)
        bridge._get_capsule_representation(emb)

        bridge.clear_cache()

        assert len(bridge._cache) == 0
        assert bridge.state.cache_hits == 0
        assert bridge.state.cache_misses == 0

    def test_get_capsule_features(self, bridge):
        """Test feature extraction."""
        emb = np.random.randn(1024)
        features = bridge.get_capsule_features(emb)

        assert "activations" in features
        assert "poses" in features
        assert "active_capsules" in features
        assert "mean_activation" in features
        assert "pose_complexity" in features

    def test_get_statistics(self, bridge):
        """Test statistics retrieval."""
        # Do some operations
        query_emb = np.random.randn(1024)
        memory_embs = [np.random.randn(1024) for _ in range(5)]
        bridge.compute_boosts(query_emb, memory_embs)

        stats = bridge.get_statistics()

        assert stats["n_queries"] == 1
        assert stats["n_comparisons"] == 5
        assert "mean_boost" in stats
        assert "cache_hit_rate" in stats
        assert "config" in stats

    def test_state_updates(self, bridge):
        """Test state updates during operation."""
        initial_queries = bridge.state.n_queries
        initial_comparisons = bridge.state.n_comparisons

        query_emb = np.random.randn(1024)
        memory_embs = [np.random.randn(1024) for _ in range(3)]
        bridge.compute_boosts(query_emb, memory_embs)

        assert bridge.state.n_queries == initial_queries + 1
        assert bridge.state.n_comparisons == initial_comparisons + 3

    def test_fallback_without_layer(self, bridge_no_layer):
        """Test bridge works with fallback when no layer provided."""
        query_emb = np.random.randn(1024)
        memory_emb = np.random.randn(1024)

        # Should not raise, just use fallback
        boost = bridge_no_layer.compute_boost(query_emb, memory_emb)
        assert 0.0 <= boost <= bridge_no_layer.config.max_boost


class TestCapsuleIntegration:
    """Integration tests with actual CapsuleLayer."""

    def test_with_real_capsule_layer(self):
        """Test integration with actual CapsuleLayer (if available)."""
        try:
            from ww.nca.capsules import CapsuleLayer, CapsuleConfig

            # Create real capsule layer
            caps_config = CapsuleConfig(
                input_dim=1024,
                num_capsules=16,
                capsule_dim=8,
                pose_dim=4
            )
            capsule_layer = CapsuleLayer(caps_config)

            # Create bridge
            bridge = CapsuleRetrievalBridge(capsule_layer=capsule_layer)

            # Test boost computation
            query_emb = np.random.randn(1024).astype(np.float32)
            memory_embs = [
                np.random.randn(1024).astype(np.float32)
                for _ in range(5)
            ]

            boosts = bridge.compute_boosts(query_emb, memory_embs)

            assert len(boosts) == 5
            assert all(0.0 <= b <= bridge.config.max_boost for b in boosts)

        except ImportError:
            pytest.skip("CapsuleLayer not available")


class TestCreateCapsuleBridge:
    """Tests for factory function."""

    def test_factory_default(self):
        """Test factory with defaults."""
        bridge = create_capsule_bridge()
        assert bridge.config.activation_weight == 0.4
        assert bridge.config.pose_weight == 0.6

    def test_factory_custom(self):
        """Test factory with custom params."""
        bridge = create_capsule_bridge(
            activation_weight=0.5,
            pose_weight=0.5,
            max_boost=0.5
        )
        assert bridge.config.activation_weight == 0.5
        assert bridge.config.max_boost == 0.5

    def test_factory_with_layer(self):
        """Test factory with capsule layer."""
        layer = MockCapsuleLayer()
        bridge = create_capsule_bridge(capsule_layer=layer)
        assert bridge.capsule_layer is not None
