"""
Tests for the learnable Forward-Forward Encoder (Phase 5).

The FFEncoder is THE LEARNING GAP FIX - it sits between the frozen embedder
and storage, allowing the system to learn representations through use.
"""

import numpy as np
import pytest

from t4dm.encoding.ff_encoder import (
    FFEncoder,
    FFEncoderConfig,
    FFEncoderState,
    create_ff_encoder,
    get_ff_encoder,
    reset_ff_encoder,
)


class TestFFEncoderConfig:
    """Test FFEncoderConfig dataclass."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = FFEncoderConfig()
        assert config.input_dim == 1024
        assert config.hidden_dims == (512, 256)
        assert config.output_dim == 1024
        assert config.learning_rate == 0.03
        assert config.use_residual is True
        assert config.normalize_output is True

    def test_custom_config(self):
        """Custom config values are respected."""
        config = FFEncoderConfig(
            input_dim=768,
            hidden_dims=(384, 192),
            output_dim=768,
            learning_rate=0.05,
        )
        assert config.input_dim == 768
        assert config.hidden_dims == (384, 192)

    def test_validation(self):
        """Config validates parameters."""
        with pytest.raises(AssertionError):
            FFEncoderConfig(input_dim=-1)

        with pytest.raises(AssertionError):
            FFEncoderConfig(hidden_dims=())

        with pytest.raises(AssertionError):
            FFEncoderConfig(learning_rate=2.0)


class TestFFEncoderState:
    """Test FFEncoderState dataclass."""

    def test_initial_state(self):
        """Initial state is zeroed."""
        state = FFEncoderState()
        assert state.total_encodes == 0
        assert state.total_positive_updates == 0
        assert state.total_negative_updates == 0
        assert state.mean_goodness == 0.0

    def test_to_dict(self):
        """State serializes to dict."""
        state = FFEncoderState()
        state.total_encodes = 100
        d = state.to_dict()
        assert d["total_encodes"] == 100


class TestFFEncoder:
    """Test FFEncoder class."""

    @pytest.fixture
    def config(self):
        """Standard test config."""
        return FFEncoderConfig(
            input_dim=64,  # Small for fast testing
            hidden_dims=(32, 16),
            output_dim=64,
            learning_rate=0.05,
        )

    @pytest.fixture
    def encoder(self, config):
        """Create encoder."""
        return FFEncoder(config, random_seed=42)

    def test_initialization(self, encoder, config):
        """Encoder initializes correctly."""
        assert encoder.config == config
        assert len(encoder._layers) == 2
        assert encoder.state.total_encodes == 0

    def test_encode_single(self, encoder):
        """Encoding a single vector works."""
        x = np.random.randn(64).astype(np.float32)
        encoded = encoder.encode(x)

        assert encoded.shape == (64,)
        assert encoder.state.total_encodes == 1

    def test_encode_batch(self, encoder):
        """Encoding a batch works."""
        x = np.random.randn(4, 64).astype(np.float32)
        encoded = encoder.encode(x)

        assert encoded.shape == (4, 64)
        assert encoder.state.total_encodes == 4

    def test_encode_changes_input(self, encoder):
        """Encoding produces different output than input."""
        x = np.random.randn(64).astype(np.float32)
        encoded = encoder.encode(x)

        # With residual connection, output is close but not identical
        diff = np.linalg.norm(encoded - x)
        assert diff > 0.01, "Encoding should transform the input"

    def test_normalize_output(self, encoder):
        """Output is L2 normalized."""
        x = np.random.randn(64).astype(np.float32)
        encoded = encoder.encode(x)

        norm = np.linalg.norm(encoded)
        assert np.isclose(norm, 1.0, atol=0.01), "Output should be normalized"

    def test_get_goodness(self, encoder):
        """Goodness score works."""
        x = np.random.randn(64).astype(np.float32)
        encoder.encode(x)

        goodness = encoder.get_goodness(x)
        assert goodness > 0, "Goodness should be positive"

    def test_learn_from_positive_outcome(self, encoder):
        """Learning from positive outcome updates weights."""
        x = np.random.randn(64).astype(np.float32)
        encoder.encode(x)

        initial_positive = encoder.state.total_positive_updates

        stats = encoder.learn_from_outcome(x, outcome_score=0.9)

        assert stats["phase"] == "positive"
        assert encoder.state.total_positive_updates > initial_positive

    def test_learn_from_negative_outcome(self, encoder):
        """Learning from negative outcome updates weights."""
        x = np.random.randn(64).astype(np.float32)
        encoder.encode(x)

        initial_negative = encoder.state.total_negative_updates

        stats = encoder.learn_from_outcome(x, outcome_score=0.2)

        assert stats["phase"] == "negative"
        assert encoder.state.total_negative_updates > initial_negative

    def test_learn_with_effective_lr(self, encoder):
        """Custom effective learning rate is used."""
        x = np.random.randn(64).astype(np.float32)
        encoder.encode(x)

        stats = encoder.learn_from_outcome(x, outcome_score=0.8, effective_lr=0.1)

        assert stats["effective_lr"] == 0.1

    def test_replay_consolidation(self, encoder):
        """Consolidation replay works."""
        # Encode enough samples for replay
        for _ in range(20):
            x = np.random.randn(64).astype(np.float32)
            encoder.encode(x)

        stats = encoder.replay_consolidation(n_samples=10)

        assert stats["n_positive"] > 0
        assert stats["n_negative"] > 0
        assert "mean_positive_goodness" in stats

    def test_replay_requires_history(self, encoder):
        """Replay requires sufficient history."""
        stats = encoder.replay_consolidation(n_samples=10)

        assert stats["status"] == "insufficient_history"

    def test_save_load_weights(self, encoder):
        """Weights can be saved and loaded."""
        x = np.random.randn(64).astype(np.float32)
        encoder.encode(x)
        encoder.learn_from_outcome(x, outcome_score=0.9)

        # Save weights
        weights = encoder.save_weights()

        # Create new encoder
        new_encoder = FFEncoder(encoder.config, random_seed=123)

        # Verify different state before load
        assert new_encoder.state.total_positive_updates == 0

        # Load weights
        new_encoder.load_weights(weights)

        # Verify state restored
        assert new_encoder.state.total_positive_updates > 0

    def test_reset(self, encoder):
        """Reset returns to initial state."""
        x = np.random.randn(64).astype(np.float32)
        encoder.encode(x)
        encoder.learn_from_outcome(x, outcome_score=0.9)

        encoder.reset()

        assert encoder.state.total_encodes == 0
        assert encoder.state.total_positive_updates == 0

    def test_get_stats(self, encoder):
        """Statistics are computed correctly."""
        x = np.random.randn(64).astype(np.float32)
        encoder.encode(x)

        stats = encoder.get_stats()

        assert "state" in stats
        assert "config" in stats
        assert "n_layers" in stats
        assert stats["n_layers"] == 2


class TestFFEncoderResidual:
    """Test residual connection behavior."""

    def test_with_residual(self):
        """Residual connection preserves input structure."""
        config = FFEncoderConfig(
            input_dim=64,
            hidden_dims=(32,),
            output_dim=64,
            use_residual=True,
        )
        encoder = FFEncoder(config, random_seed=42)

        x = np.random.randn(64).astype(np.float32)
        encoded = encoder.encode(x)

        # With residual, output should have high correlation with input
        correlation = np.corrcoef(x, encoded)[0, 1]
        assert correlation > 0.5, "Residual should preserve input structure"

    def test_without_residual(self):
        """Without residual, output can differ more."""
        config = FFEncoderConfig(
            input_dim=64,
            hidden_dims=(32,),
            output_dim=64,
            use_residual=False,
        )
        encoder = FFEncoder(config, random_seed=42)

        x = np.random.randn(64).astype(np.float32)
        encoded = encoder.encode(x)

        # Without residual, output may differ significantly
        # Just verify it produces valid output
        assert encoded.shape == (64,)


class TestFFEncoderLearning:
    """Test that FFEncoder actually learns from outcomes."""

    def test_positive_outcomes_increase_goodness(self):
        """Positive outcomes should increase goodness for similar patterns."""
        config = FFEncoderConfig(
            input_dim=64,
            hidden_dims=(32,),
            output_dim=64,
            learning_rate=0.1,  # Higher LR for visible effect
        )
        encoder = FFEncoder(config, random_seed=42)

        # Create a pattern
        x = np.random.randn(64).astype(np.float32)
        x = x / np.linalg.norm(x)  # Normalize

        # Get initial goodness
        encoder.encode(x)
        initial_goodness = encoder.get_goodness(x)

        # Train with positive outcomes
        for _ in range(10):
            encoder.learn_from_outcome(x, outcome_score=0.95)

        # Get final goodness
        encoder.encode(x)
        final_goodness = encoder.get_goodness(x)

        assert final_goodness >= initial_goodness, \
            "Positive outcomes should increase or maintain goodness"

    def test_weights_change_after_learning(self):
        """Verify weights actually change after learning."""
        config = FFEncoderConfig(
            input_dim=64,
            hidden_dims=(32,),
            output_dim=64,
            learning_rate=0.1,
        )
        encoder = FFEncoder(config, random_seed=42)

        # Get initial output weights
        initial_weights = encoder._output_weights.copy()

        # Learn from outcomes
        x = np.random.randn(64).astype(np.float32)
        encoder.encode(x)
        for _ in range(5):
            encoder.learn_from_outcome(x, outcome_score=0.9)

        # Verify weights changed
        weight_diff = np.linalg.norm(encoder._output_weights - initial_weights)
        assert weight_diff > 0.001, "Weights should change after learning"


class TestFFEncoderSingleton:
    """Test global singleton management."""

    def test_get_ff_encoder_returns_singleton(self):
        """get_ff_encoder returns same instance."""
        reset_ff_encoder()

        encoder1 = get_ff_encoder()
        encoder2 = get_ff_encoder()

        assert encoder1 is encoder2

    def test_reset_creates_new_instance(self):
        """reset_ff_encoder creates new instance."""
        encoder1 = get_ff_encoder()
        reset_ff_encoder()
        encoder2 = get_ff_encoder()

        assert encoder1 is not encoder2

    def test_factory_function(self):
        """create_ff_encoder factory works."""
        encoder = create_ff_encoder(
            input_dim=128,
            hidden_dims=(64, 32),
            output_dim=128,
        )

        assert encoder.config.input_dim == 128
        assert encoder.config.hidden_dims == (64, 32)


class TestFFEncoderThreeFactorIntegration:
    """Test integration with three-factor learning."""

    def test_learn_with_three_factor_signal(self):
        """Learning with three-factor signal works."""
        from t4dm.learning.three_factor import ThreeFactorSignal
        from uuid import uuid4

        config = FFEncoderConfig(
            input_dim=64,
            hidden_dims=(32,),
            output_dim=64,
        )
        encoder = FFEncoder(config, random_seed=42)

        # Create mock three-factor signal
        signal = ThreeFactorSignal(
            memory_id=uuid4(),
            eligibility=0.9,
            neuromod_gate=1.2,
            ach_mode_factor=1.5,
            ne_arousal_factor=1.0,
            serotonin_mood_factor=0.9,
            dopamine_surprise=0.8,
            rpe_raw=0.3,
            effective_lr_multiplier=2.0,
        )

        x = np.random.randn(64).astype(np.float32)
        encoder.encode(x)

        stats = encoder.learn_from_outcome(
            embedding=x,
            outcome_score=0.85,
            three_factor_signal=signal,
        )

        # Effective LR should be base_lr * three_factor_multiplier
        expected_lr = encoder.config.learning_rate * signal.effective_lr_multiplier
        assert np.isclose(stats["effective_lr"], expected_lr, atol=0.001)


class TestFFEncoderBiologicalConstraints:
    """Test biological plausibility of FFEncoder."""

    def test_local_learning(self):
        """Learning uses local gradients (no backprop through layers)."""
        config = FFEncoderConfig(
            input_dim=64,
            hidden_dims=(32, 16),
            output_dim=64,
        )
        encoder = FFEncoder(config, random_seed=42)

        x = np.random.randn(64).astype(np.float32)
        encoder.encode(x)

        # Each layer learns independently via FF algorithm
        stats = encoder.learn_from_outcome(x, outcome_score=0.9)

        # Verify each layer was updated
        assert len(stats["layer_stats"]) == 2
        for layer_stat in stats["layer_stats"]:
            assert "layer_idx" in layer_stat

    def test_replay_prevents_catastrophic_forgetting(self):
        """Interleaved replay helps maintain old patterns."""
        config = FFEncoderConfig(
            input_dim=64,
            hidden_dims=(32,),
            output_dim=64,
            learning_rate=0.05,
            max_history=100,
        )
        encoder = FFEncoder(config, random_seed=42)

        # Learn initial patterns
        old_patterns = [np.random.randn(64).astype(np.float32) for _ in range(10)]
        for p in old_patterns:
            encoder.encode(p)
            encoder.learn_from_outcome(p, outcome_score=0.9)

        # Get goodness for old patterns
        old_goodness = [encoder.get_goodness(p) for p in old_patterns]

        # Learn new patterns
        new_patterns = [np.random.randn(64).astype(np.float32) for _ in range(10)]
        for p in new_patterns:
            encoder.encode(p)
            encoder.learn_from_outcome(p, outcome_score=0.9)

        # Do consolidation replay
        encoder.replay_consolidation(n_samples=20)

        # Verify old patterns still have reasonable goodness
        new_old_goodness = [encoder.get_goodness(p) for p in old_patterns]

        # Old patterns should maintain similar goodness (not zeroed out)
        for old_g, new_g in zip(old_goodness, new_old_goodness):
            assert new_g > old_g * 0.3, "Replay should prevent catastrophic forgetting"
