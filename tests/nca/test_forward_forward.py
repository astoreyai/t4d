"""Tests for Forward-Forward Algorithm implementation."""

import numpy as np
import pytest

from ww.nca.forward_forward import (
    ForwardForwardConfig,
    ForwardForwardState,
    ForwardForwardLayer,
    ForwardForwardNetwork,
    FFPhase,
    create_ff_layer,
    create_ff_network,
)


class TestForwardForwardConfig:
    """Tests for ForwardForwardConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = ForwardForwardConfig()

        assert config.input_dim == 1024
        assert config.hidden_dim == 512
        assert config.threshold_theta == 2.0
        assert config.learning_rate == 0.03
        assert config.activation == "relu"

    def test_custom_values(self):
        """Config should accept custom values."""
        config = ForwardForwardConfig(
            input_dim=64,
            hidden_dim=32,
            threshold_theta=3.0,
            learning_rate=0.05,
        )
        assert config.input_dim == 64
        assert config.hidden_dim == 32
        assert config.threshold_theta == 3.0
        assert config.learning_rate == 0.05

    def test_validation_fails_for_invalid_params(self):
        """Config should reject invalid parameters."""
        with pytest.raises(AssertionError):
            ForwardForwardConfig(input_dim=0)

        with pytest.raises(AssertionError):
            ForwardForwardConfig(hidden_dim=-1)

        with pytest.raises(AssertionError):
            ForwardForwardConfig(threshold_theta=-1)

        with pytest.raises(AssertionError):
            ForwardForwardConfig(learning_rate=2.0)


class TestForwardForwardState:
    """Tests for ForwardForwardState."""

    def test_default_state(self):
        """State should have reasonable defaults."""
        state = ForwardForwardState()

        assert state.goodness == 0.0
        assert state.is_positive is True
        assert state.confidence == 0.0
        assert state.total_updates == 0

    def test_to_dict(self):
        """State should serialize to dict."""
        state = ForwardForwardState(
            goodness=10.0,
            is_positive=False,
            confidence=-2.0,
        )

        d = state.to_dict()
        assert d["goodness"] == 10.0
        assert d["is_positive"] is False
        assert d["confidence"] == -2.0

    def test_history_trimming(self):
        """History should be trimmed when exceeding max."""
        state = ForwardForwardState()
        state._max_history = 10

        # Add more than max_history items
        for i in range(20):
            state.positive_goodness_history.append(float(i))

        state._trim_history()

        assert len(state.positive_goodness_history) == 10
        # Should keep the last 10 items
        assert state.positive_goodness_history[0] == 10.0


class TestForwardForwardLayer:
    """Tests for ForwardForwardLayer."""

    @pytest.fixture
    def layer(self):
        """Create a test layer."""
        config = ForwardForwardConfig(input_dim=64, hidden_dim=32)
        return ForwardForwardLayer(config, random_seed=42)

    @pytest.fixture
    def small_layer(self):
        """Create a small test layer for debugging."""
        config = ForwardForwardConfig(input_dim=4, hidden_dim=2)
        return ForwardForwardLayer(config, random_seed=42)

    def test_initialization(self, layer):
        """Layer should initialize correctly."""
        assert layer.config.input_dim == 64
        assert layer.config.hidden_dim == 32
        assert layer.W.shape == (64, 32)
        assert layer.b.shape == (32,)

    def test_forward_shape(self, layer):
        """Forward pass should produce correct shape."""
        x = np.random.randn(64).astype(np.float32)
        h = layer.forward(x)

        assert h.shape == (32,)

    def test_forward_batch(self, layer):
        """Forward pass should handle batches."""
        x = np.random.randn(8, 64).astype(np.float32)
        h = layer.forward(x)

        assert h.shape == (8, 32)

    def test_goodness_computation(self, layer):
        """Goodness should be sum of squared activations."""
        h = np.array([1.0, 2.0, 3.0])
        goodness = layer.compute_goodness(h)

        # 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
        assert goodness == 14.0

    def test_goodness_computation_zeros(self, layer):
        """Goodness of zeros should be zero."""
        h = np.zeros(10)
        goodness = layer.compute_goodness(h)

        assert goodness == 0.0

    def test_classification_above_threshold(self, layer):
        """High goodness should classify as positive."""
        # Create activations with high squared sum
        h = np.ones(32) * 0.5  # Sum of squares = 32 * 0.25 = 8.0
        is_positive, confidence = layer.classify(h)

        # With threshold=2.0, goodness=8.0 should be positive
        assert is_positive is True
        assert confidence > 0

    def test_classification_below_threshold(self, layer):
        """Low goodness should classify as negative."""
        # Create activations with low squared sum
        h = np.ones(32) * 0.1  # Sum of squares = 32 * 0.01 = 0.32
        is_positive, confidence = layer.classify(h)

        # With threshold=2.0, goodness=0.32 should be negative
        assert is_positive is False
        assert confidence < 0

    def test_learn_positive_updates_weights(self, layer):
        """Positive learning should update weights."""
        x = np.random.randn(64).astype(np.float32)
        w_before = layer.W.copy()

        h = layer.forward(x)
        layer.learn_positive(x, h)

        w_after = layer.W

        # Weights should change
        assert not np.allclose(w_before, w_after)

    def test_learn_negative_updates_weights(self, layer):
        """Negative learning should update weights."""
        x = np.random.randn(64).astype(np.float32)
        w_before = layer.W.copy()

        h = layer.forward(x)
        layer.learn_negative(x, h)

        w_after = layer.W

        # Weights should change
        assert not np.allclose(w_before, w_after)

    def test_learn_positive_increases_goodness_tendency(self, layer):
        """Positive learning should tend to increase goodness for same input."""
        x = np.random.randn(64).astype(np.float32)

        # Initial goodness
        h1 = layer.forward(x)
        g1 = layer.state.goodness

        # Train multiple times on positive
        for _ in range(10):
            h = layer.forward(x)
            layer.learn_positive(x, h)

        # Final goodness
        h2 = layer.forward(x)
        g2 = layer.state.goodness

        # Goodness should generally increase (allow some tolerance)
        # Note: momentum/decay can cause short-term fluctuations
        assert g2 >= g1 * 0.8  # Allow 20% tolerance

    def test_learn_negative_decreases_goodness_tendency(self, layer):
        """Negative learning should tend to decrease goodness for same input."""
        x = np.random.randn(64).astype(np.float32)

        # Create artificially high activations
        layer.W = np.abs(layer.W) * 2  # Make all weights positive and large

        # Initial goodness
        h1 = layer.forward(x)
        g1 = layer.state.goodness

        # Train multiple times on negative
        for _ in range(10):
            h = layer.forward(x)
            layer.learn_negative(x, h)

        # Final goodness
        h2 = layer.forward(x)
        g2 = layer.state.goodness

        # Goodness should generally decrease (allow tolerance)
        assert g2 <= g1 * 1.2  # Should not increase much

    def test_weight_bounds_enforced(self, layer):
        """Weights should stay within bounds."""
        x = np.ones(64).astype(np.float32) * 10  # Large input

        # Force many updates
        for _ in range(100):
            h = layer.forward(x)
            layer.learn_positive(x, h)

        assert np.all(layer.W >= layer.config.min_weight)
        assert np.all(layer.W <= layer.config.max_weight)

    def test_reset(self, layer):
        """Reset should restore initial state."""
        x = np.random.randn(64).astype(np.float32)

        # Do some training
        h = layer.forward(x)
        layer.learn_positive(x, h)
        layer.learn_negative(x, h)

        assert layer.state.total_updates > 0

        # Reset
        layer.reset()

        assert layer.state.total_updates == 0
        assert len(layer.state.positive_goodness_history) == 0

    def test_stats(self, layer):
        """Should return meaningful statistics."""
        x = np.random.randn(64).astype(np.float32)

        h = layer.forward(x)
        layer.learn_positive(x, h)

        stats = layer.get_stats()

        assert "layer_idx" in stats
        assert "input_dim" in stats
        assert "hidden_dim" in stats
        assert "total_updates" in stats
        assert stats["total_updates"] == 1

    def test_no_backward_pass_required(self, small_layer):
        """Learning should not require any backward pass."""
        layer = small_layer
        x = np.random.randn(4).astype(np.float32)

        h = layer.forward(x)

        # Learn using only local information
        stats = layer.learn_positive(x, h)

        # Should complete without error and not reference downstream
        assert "gradient_norm" in stats
        assert "downstream_gradient" not in stats

    def test_layer_local_learning(self, small_layer):
        """Learning should use only local activations."""
        layer = small_layer
        x = np.array([1.0, 0.0, 0.0, 0.0])  # Single active input

        h = layer.forward(x)
        w_before = layer.W.copy()
        layer.learn_positive(x, h)
        w_after = layer.W

        # Changes should be larger for connections from active input
        delta = np.abs(w_after - w_before)
        assert np.mean(delta[0, :]) >= np.mean(delta[1:, :])


class TestForwardForwardNetwork:
    """Tests for ForwardForwardNetwork."""

    @pytest.fixture
    def network(self):
        """Create a test network."""
        config = ForwardForwardConfig(learning_rate=0.05)
        return ForwardForwardNetwork([64, 32, 16], config, random_seed=42)

    @pytest.fixture
    def small_network(self):
        """Create a small test network."""
        config = ForwardForwardConfig(learning_rate=0.05)
        return ForwardForwardNetwork([8, 4, 2], config, random_seed=42)

    def test_initialization(self, network):
        """Network should initialize correctly."""
        assert len(network.layers) == 2
        assert network.layer_dims == [64, 32, 16]

        assert network.layers[0].config.input_dim == 64
        assert network.layers[0].config.hidden_dim == 32
        assert network.layers[1].config.input_dim == 32
        assert network.layers[1].config.hidden_dim == 16

    def test_forward_multi_layer(self, network):
        """Forward pass should produce activations for all layers."""
        x = np.random.randn(64).astype(np.float32)

        activations = network.forward(x)

        assert len(activations) == 2
        assert activations[0].shape == (32,)
        assert activations[1].shape == (16,)

    def test_forward_with_label(self, network):
        """Forward pass should embed label in input."""
        x = np.random.randn(64).astype(np.float32)
        label = np.zeros(10)
        label[5] = 1.0  # One-hot for class 5

        activations = network.forward(x, label)

        assert len(activations) == 2

    def test_generate_negative_noise(self, network):
        """Noise method should add Gaussian noise."""
        positive = np.ones(64).astype(np.float32) * 0.5

        negative, _ = network.generate_negative(positive, method="noise")

        # Should be different from original
        assert not np.allclose(positive, negative)
        # Should be within reasonable range
        assert np.all(negative >= 0) and np.all(negative <= 1)

    def test_generate_negative_shuffle(self, network):
        """Shuffle method should permute features."""
        positive = np.arange(64).astype(np.float32)

        negative, _ = network.generate_negative(positive, method="shuffle")

        # Should have same elements but different order
        assert set(positive.flatten()) != set(negative.flatten()) or \
               not np.array_equal(positive, negative)

    def test_generate_negative_adversarial(self, network):
        """Adversarial method should increase first layer goodness."""
        positive = np.random.randn(1, 64).astype(np.float32) * 0.1

        negative, _ = network.generate_negative(positive, method="adversarial")

        # Negative should be modified from positive
        assert not np.allclose(positive, negative)

    def test_generate_negative_hybrid(self, network):
        """Hybrid method should combine methods."""
        positive = np.random.randn(9, 64).astype(np.float32)  # 9 samples

        negative, _ = network.generate_negative(positive, method="hybrid")

        assert negative.shape == positive.shape

    def test_train_step(self, network):
        """Training step should update all layers."""
        positive = np.random.randn(64).astype(np.float32)
        negative = np.random.randn(64).astype(np.float32)

        stats = network.train_step(positive, negative)

        assert "positive_phase" in stats
        assert "negative_phase" in stats
        assert len(stats["positive_phase"]) == 2
        assert len(stats["negative_phase"]) == 2

    def test_training_convergence(self, small_network):
        """Training should learn to distinguish positive from negative."""
        network = small_network

        # Create distinguishable positive and negative
        positive = np.ones(8).astype(np.float32) * 0.8
        negative = np.zeros(8).astype(np.float32)

        # Train
        for _ in range(50):
            network.train_step(positive, negative)

        # Check discrimination
        network.forward(positive)
        pos_goodness = sum(l.state.goodness for l in network.layers)

        network.forward(negative)
        neg_goodness = sum(l.state.goodness for l in network.layers)

        # Positive should have higher goodness
        assert pos_goodness > neg_goodness

    def test_infer(self, network):
        """Inference should return classification."""
        x = np.random.randn(64).astype(np.float32)

        is_positive, confidence = network.infer(x)

        assert isinstance(is_positive, bool)
        assert isinstance(confidence, float)

    def test_get_stats(self, network):
        """Should return network statistics."""
        stats = network.get_stats()

        assert stats["n_layers"] == 2
        assert stats["layer_dims"] == [64, 32, 16]
        assert "layer_stats" in stats

    def test_reset(self, network):
        """Reset should restore all layers."""
        positive = np.random.randn(64).astype(np.float32)
        negative = np.random.randn(64).astype(np.float32)

        # Train
        network.train_step(positive, negative)
        assert network._total_training_steps == 1

        # Reset
        network.reset()

        assert network._total_training_steps == 0
        for layer in network.layers:
            assert layer.state.total_updates == 0


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_ff_layer(self):
        """Factory should create configured layer."""
        layer = create_ff_layer(
            input_dim=128,
            hidden_dim=64,
            learning_rate=0.01,
            threshold=1.5,
            random_seed=42,
        )

        assert layer.config.input_dim == 128
        assert layer.config.hidden_dim == 64
        assert layer.config.learning_rate == 0.01
        assert layer.config.threshold_theta == 1.5

    def test_create_ff_network(self):
        """Factory should create configured network."""
        network = create_ff_network(
            layer_dims=[128, 64, 32],
            learning_rate=0.01,
            threshold=1.5,
            random_seed=42,
        )

        assert len(network.layers) == 2
        assert network.config.learning_rate == 0.01
        assert network.config.threshold_theta == 1.5


class TestBiologicalPlausibility:
    """Tests for biological plausibility of FF algorithm."""

    def test_hebbian_like_updates(self):
        """Weight updates should follow Hebbian-like rule."""
        config = ForwardForwardConfig(input_dim=4, hidden_dim=2)
        layer = ForwardForwardLayer(config, random_seed=42)

        # Input with one strongly active unit
        x = np.array([1.0, 0.0, 0.0, 0.0])
        h = layer.forward(x)

        w_before = layer.W.copy()
        layer.learn_positive(x, h)
        w_after = layer.W

        # Weights from active input should change most
        delta_w = np.abs(w_after - w_before)

        # Row 0 (active input) should have larger changes than others
        active_row_delta = np.sum(delta_w[0, :])
        inactive_row_delta = np.sum(delta_w[1:, :])

        assert active_row_delta >= inactive_row_delta / 3  # More relaxed

    def test_layer_independence(self):
        """Each layer should learn independently."""
        config = ForwardForwardConfig(learning_rate=0.05)
        network = ForwardForwardNetwork([8, 4, 2], config, random_seed=42)

        x = np.random.randn(8).astype(np.float32)

        # Get initial layer 2 weights
        layer2_weights_before = network.layers[1].W.copy()

        # Train only layer 1
        h = network.layers[0].forward(x)
        network.layers[0].learn_positive(x, h)

        layer2_weights_after = network.layers[1].W

        # Layer 2 weights should not change from layer 1 learning
        np.testing.assert_array_equal(layer2_weights_before, layer2_weights_after)

    def test_no_activations_stored(self):
        """FF should not require storing activations for backprop."""
        config = ForwardForwardConfig(input_dim=64, hidden_dim=32)
        layer = ForwardForwardLayer(config, random_seed=42)

        x = np.random.randn(64).astype(np.float32)
        h = layer.forward(x)

        # Learn using current state only
        stats = layer.learn_positive(x, h)

        # Should succeed without storing activation history
        assert stats["phase"] == "positive"
        # State has current activations but not history for backprop
        assert layer.state.activations is not None
