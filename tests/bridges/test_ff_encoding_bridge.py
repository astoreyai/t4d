"""
Tests for FF-Encoding Bridge (P6.3).

Tests the FFEncodingBridge which uses Forward-Forward algorithm
to provide novelty detection and encoding guidance.
"""

import numpy as np
import pytest

from t4dm.bridges.ff_encoding_bridge import (
    EncodingGuidance,
    FFEncodingBridge,
    FFEncodingConfig,
    FFEncodingState,
    create_ff_encoding_bridge,
)


class MockForwardForwardLayer:
    """Mock FF layer for testing."""

    def __init__(self, hidden_dim: int = 512):
        self.hidden_dim = hidden_dim
        self._fixed_goodness = None  # If set, return this directly
        self._state = type('State', (), {'goodness': 0.0})()

    @property
    def state(self):
        return self._state

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Mock forward pass - just return transformed input."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Simple projection (normalize to prevent extreme values)
        hidden = np.abs(x[:, :self.hidden_dim])
        hidden = hidden / (np.linalg.norm(hidden) + 1e-8)  # Normalize
        return hidden.flatten()

    def compute_goodness(self, h: np.ndarray) -> float:
        """Compute goodness as sum of squared activations."""
        if self._fixed_goodness is not None:
            goodness = self._fixed_goodness
        else:
            # With normalized h, this will be close to 1/hidden_dim
            goodness = float(np.sum(h ** 2))
        self._state.goodness = goodness
        return goodness

    def positive_phase(self, x: np.ndarray, learning_rate: float = None) -> float:
        """Mock positive phase."""
        h = self.forward(x)
        return self.compute_goodness(h)

    def negative_phase(self, x: np.ndarray, learning_rate: float = None) -> float:
        """Mock negative phase."""
        h = self.forward(x)
        return self.compute_goodness(h)

    def set_goodness_scale(self, scale: float) -> None:
        """Control goodness output for testing."""
        self._fixed_goodness = scale


class TestFFEncodingConfig:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FFEncodingConfig()
        assert config.goodness_threshold == 2.0
        assert config.novelty_boost == 0.5
        assert config.familiar_dampening == 0.2
        assert config.online_learning is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = FFEncodingConfig(
            goodness_threshold=3.0,
            novelty_boost=0.8,
            online_learning=False
        )
        assert config.goodness_threshold == 3.0
        assert config.novelty_boost == 0.8
        assert config.online_learning is False


class TestEncodingGuidance:
    """Tests for guidance dataclass."""

    def test_creation(self):
        """Test guidance creation."""
        guidance = EncodingGuidance(
            is_novel=True,
            goodness=1.5,
            encoding_multiplier=1.2,
            priority=0.8,
            confidence=0.5
        )
        assert guidance.is_novel is True
        assert guidance.goodness == 1.5
        assert guidance.encoding_multiplier == 1.2

    def test_optional_embedding(self):
        """Test with enhanced embedding."""
        emb = np.random.randn(512)
        guidance = EncodingGuidance(
            is_novel=False,
            goodness=3.0,
            encoding_multiplier=0.9,
            priority=0.4,
            confidence=1.0,
            enhanced_embedding=emb
        )
        assert guidance.enhanced_embedding is not None
        assert len(guidance.enhanced_embedding) == 512


class TestFFEncodingBridge:
    """Tests for FFEncodingBridge class."""

    @pytest.fixture
    def mock_layer(self):
        """Create mock FF layer."""
        return MockForwardForwardLayer(hidden_dim=512)

    @pytest.fixture
    def bridge(self, mock_layer):
        """Create bridge with mock layer."""
        return FFEncodingBridge(ff_layer=mock_layer)

    @pytest.fixture
    def bridge_no_layer(self):
        """Create bridge without FF layer."""
        return FFEncodingBridge()

    def test_initialization(self, bridge):
        """Test bridge initializes correctly."""
        assert bridge.ff_layer is not None
        assert bridge.config.goodness_threshold == 2.0
        assert bridge.state.n_embeddings_processed == 0

    def test_initialization_no_layer(self, bridge_no_layer):
        """Test bridge works without FF layer (fallback)."""
        assert bridge_no_layer.ff_layer is None

    def test_set_ff_layer(self, bridge_no_layer, mock_layer):
        """Test setting FF layer."""
        bridge_no_layer.set_ff_layer(mock_layer)
        assert bridge_no_layer.ff_layer is not None

    def test_process_novel(self, bridge, mock_layer):
        """Test processing novel pattern (low goodness)."""
        # Set low goodness scale for novel detection
        mock_layer.set_goodness_scale(0.1)  # Will give low goodness

        embedding = np.random.randn(1024)
        guidance = bridge.process(embedding)

        # Low goodness should be novel
        assert guidance.is_novel is True
        assert guidance.encoding_multiplier > 1.0  # Boost for novel
        assert guidance.priority > 0.5  # Higher priority

    def test_process_familiar(self, bridge, mock_layer):
        """Test processing familiar pattern (high goodness)."""
        # Set high goodness scale for familiar detection
        mock_layer.set_goodness_scale(10.0)  # Will give high goodness

        embedding = np.random.randn(1024)
        guidance = bridge.process(embedding)

        # High goodness should be familiar
        assert guidance.is_novel is False
        assert guidance.encoding_multiplier <= 1.0  # No boost or dampening

    def test_encoding_multiplier_range(self, bridge):
        """Test encoding multiplier stays in expected range."""
        for _ in range(20):
            embedding = np.random.randn(1024)
            guidance = bridge.process(embedding)

            # Should be in [0.8, 1.5] range
            assert 0.8 <= guidance.encoding_multiplier <= 1.5

    def test_priority_range(self, bridge):
        """Test priority stays in [0, 1]."""
        for _ in range(20):
            embedding = np.random.randn(1024)
            guidance = bridge.process(embedding)

            assert 0.0 <= guidance.priority <= 1.0

    def test_process_batch(self, bridge):
        """Test batch processing."""
        embeddings = [np.random.randn(1024) for _ in range(10)]
        guidances = bridge.process_batch(embeddings)

        assert len(guidances) == 10
        assert all(isinstance(g, EncodingGuidance) for g in guidances)

    def test_state_updates(self, bridge):
        """Test state updates during processing."""
        initial_count = bridge.state.n_embeddings_processed

        embedding = np.random.randn(1024)
        bridge.process(embedding)

        assert bridge.state.n_embeddings_processed == initial_count + 1
        assert bridge.state.n_novel_detected + bridge.state.n_familiar_detected == 1

    def test_novelty_ratio_tracking(self, bridge, mock_layer):
        """Test novelty ratio is tracked correctly."""
        # Process some novel patterns
        mock_layer.set_goodness_scale(0.05)
        for _ in range(5):
            bridge.process(np.random.randn(1024))

        # Process some familiar patterns
        mock_layer.set_goodness_scale(10.0)
        for _ in range(5):
            bridge.process(np.random.randn(1024))

        # Should have both novel and familiar
        assert bridge.state.n_novel_detected > 0
        assert bridge.state.n_familiar_detected > 0
        assert 0.0 <= bridge.state.novelty_ratio <= 1.0

    def test_online_learning(self, bridge, mock_layer):
        """Test online learning is called when enabled."""
        bridge.config.online_learning = True
        embedding = np.random.randn(1024)

        # Should not raise
        bridge.process(embedding)

    def test_online_learning_disabled(self, bridge):
        """Test online learning can be disabled."""
        bridge.config.online_learning = False
        embedding = np.random.randn(1024)

        # Should not raise
        guidance = bridge.process(embedding)
        assert guidance is not None

    def test_generate_negative(self, bridge):
        """Test negative sample generation."""
        embedding = np.random.randn(1024)
        negative = bridge._generate_negative(embedding)

        assert negative.shape == embedding.shape
        # Should be different from original
        assert not np.allclose(embedding, negative)
        # Should be normalized
        assert np.abs(np.linalg.norm(negative) - 1.0) < 0.01

    def test_adaptive_threshold(self, bridge, mock_layer):
        """Test adaptive threshold computation."""
        # Process many samples to build history
        for scale in np.linspace(0.1, 10.0, 100):
            mock_layer.set_goodness_scale(scale)
            bridge.process(np.random.randn(1024))

        # Should have adaptive threshold
        adaptive = bridge.get_adaptive_threshold()
        assert adaptive > 0

    def test_update_threshold(self, bridge):
        """Test threshold update."""
        old_threshold = bridge.config.goodness_threshold
        new_threshold = 3.5

        bridge.update_threshold(new_threshold)

        assert bridge.config.goodness_threshold == new_threshold
        assert bridge.config.goodness_threshold != old_threshold

    def test_get_statistics(self, bridge):
        """Test statistics retrieval."""
        # Do some processing
        for _ in range(5):
            bridge.process(np.random.randn(1024))

        stats = bridge.get_statistics()

        assert stats["n_embeddings_processed"] == 5
        assert "novelty_ratio" in stats
        assert "mean_goodness" in stats
        assert "current_threshold" in stats
        assert "config" in stats

    def test_reset_statistics(self, bridge):
        """Test statistics reset."""
        # Do some processing
        for _ in range(5):
            bridge.process(np.random.randn(1024))

        bridge.reset_statistics()

        assert bridge.state.n_embeddings_processed == 0
        assert bridge.state.n_novel_detected == 0
        assert len(bridge._goodness_history) == 0

    def test_fallback_without_layer(self, bridge_no_layer):
        """Test bridge works with fallback when no layer provided."""
        embedding = np.random.randn(1024)

        # Should not raise, just use fallback
        guidance = bridge_no_layer.process(embedding)

        assert guidance.is_novel is not None
        assert 0.0 <= guidance.priority <= 1.0


class TestFFEncodingIntegration:
    """Integration tests with actual FF layer."""

    def test_with_real_ff_layer(self):
        """Test integration with actual ForwardForwardLayer."""
        try:
            from t4dm.nca.forward_forward import ForwardForwardLayer, ForwardForwardConfig

            # Create real FF layer
            ff_config = ForwardForwardConfig(input_dim=1024, hidden_dim=256)
            ff_layer = ForwardForwardLayer(ff_config)

            # Create bridge
            bridge = FFEncodingBridge(ff_layer=ff_layer)

            # Process embeddings
            for _ in range(10):
                embedding = np.random.randn(1024).astype(np.float32)
                guidance = bridge.process(embedding)

                assert guidance is not None
                assert 0.0 <= guidance.priority <= 1.0
                assert 0.8 <= guidance.encoding_multiplier <= 1.5

        except ImportError:
            pytest.skip("ForwardForwardLayer not available")


class TestCreateFFEncodingBridge:
    """Tests for factory function."""

    def test_factory_default(self):
        """Test factory with defaults."""
        bridge = create_ff_encoding_bridge()
        assert bridge.config.goodness_threshold == 2.0
        assert bridge.config.novelty_boost == 0.5

    def test_factory_custom(self):
        """Test factory with custom params."""
        bridge = create_ff_encoding_bridge(
            goodness_threshold=3.0,
            novelty_boost=0.8,
            online_learning=False
        )
        assert bridge.config.goodness_threshold == 3.0
        assert bridge.config.novelty_boost == 0.8
        assert bridge.config.online_learning is False

    def test_factory_with_layer(self):
        """Test factory with FF layer."""
        layer = MockForwardForwardLayer()
        bridge = create_ff_encoding_bridge(ff_layer=layer)
        assert bridge.ff_layer is not None
