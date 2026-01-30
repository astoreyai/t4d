"""
Tests for Temporal Attention module.

Tests theta-gamma positional encoding and temporal attention mechanisms.
"""

import numpy as np
import pytest

from ww.nca.temporal_attention import (
    TemporalAttentionConfig,
    PositionalEncoding,
    RelativePositionEmbedding,
)


class TestTemporalAttentionConfig:
    """Tests for TemporalAttentionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TemporalAttentionConfig()
        assert config.embed_dim == 1024
        assert config.max_sequence_length == 512
        assert config.positional_type == "learnable"
        assert config.use_theta_modulation is True
        assert config.gamma_slots_per_theta == 7
        assert config.theta_position_weight == 0.3
        assert config.use_relative_positions is True
        assert config.max_relative_distance == 128
        assert config.num_heads == 8
        assert config.head_dim == 64

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TemporalAttentionConfig(
            embed_dim=512,
            max_sequence_length=256,
            gamma_slots_per_theta=5,
        )
        assert config.embed_dim == 512
        assert config.max_sequence_length == 256
        assert config.gamma_slots_per_theta == 5


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    @pytest.fixture
    def config(self):
        """Create small config for testing."""
        return TemporalAttentionConfig(
            embed_dim=64,
            max_sequence_length=32,
            use_theta_modulation=True,
        )

    @pytest.fixture
    def sinusoidal_config(self):
        """Create config for sinusoidal encoding."""
        return TemporalAttentionConfig(
            embed_dim=64,
            max_sequence_length=32,
            positional_type="sinusoidal",
            use_theta_modulation=True,
        )

    def test_initialization_learnable(self, config):
        """Test learnable positional encoding initialization."""
        pe = PositionalEncoding(config)

        assert pe.encodings.shape == (32, 64)
        assert hasattr(pe, 'W_theta')

    def test_initialization_sinusoidal(self, sinusoidal_config):
        """Test sinusoidal positional encoding initialization."""
        pe = PositionalEncoding(sinusoidal_config)

        assert pe.encodings.shape == (32, 64)
        # Sinusoidal should have specific patterns
        assert pe.config.positional_type == "sinusoidal"

    def test_encode_positions(self, config):
        """Test encoding positions."""
        pe = PositionalEncoding(config)

        encodings = pe.encode_positions(length=10)

        assert encodings.shape == (10, 64)
        assert encodings.dtype == np.float32

    def test_encode_positions_with_theta(self, config):
        """Test encoding positions with theta modulation."""
        pe = PositionalEncoding(config)

        encodings = pe.encode_positions(
            length=10,
            theta_phase=np.pi / 2,
        )

        assert encodings.shape == (10, 64)

    def test_encode_positions_exceeds_max(self, config):
        """Test that exceeding max length raises error."""
        pe = PositionalEncoding(config)

        with pytest.raises(ValueError, match="exceeds max"):
            pe.encode_positions(length=100)  # More than max 32

    def test_theta_modulated_encoding(self, config):
        """Test theta-modulated encoding."""
        pe = PositionalEncoding(config)

        modulation = pe.get_theta_modulated_encoding(
            position=5,
            theta_cycle=0,
            gamma_slot=3,
        )

        assert modulation.shape == (64,)
        assert modulation.dtype == np.float32

    def test_different_gamma_slots_produce_different_modulation(self, config):
        """Test that different gamma slots produce different modulations."""
        pe = PositionalEncoding(config)

        mod0 = pe.get_theta_modulated_encoding(position=0, theta_cycle=0, gamma_slot=0)
        mod1 = pe.get_theta_modulated_encoding(position=1, theta_cycle=0, gamma_slot=1)
        mod3 = pe.get_theta_modulated_encoding(position=3, theta_cycle=0, gamma_slot=3)

        # Different gamma slots should produce different modulations
        assert not np.allclose(mod0, mod1)
        assert not np.allclose(mod0, mod3)

    def test_different_theta_cycles_produce_different_modulation(self, config):
        """Test that different theta cycles produce different modulations."""
        pe = PositionalEncoding(config)

        mod_cycle0 = pe.get_theta_modulated_encoding(position=0, theta_cycle=0, gamma_slot=0)
        mod_cycle1 = pe.get_theta_modulated_encoding(position=7, theta_cycle=1, gamma_slot=0)
        mod_cycle5 = pe.get_theta_modulated_encoding(position=35, theta_cycle=5, gamma_slot=0)

        # Different cycles should produce different encodings
        assert not np.allclose(mod_cycle0, mod_cycle1)
        assert not np.allclose(mod_cycle0, mod_cycle5)

    def test_no_theta_modulation(self):
        """Test encoding without theta modulation."""
        config = TemporalAttentionConfig(
            embed_dim=64,
            max_sequence_length=32,
            use_theta_modulation=False,
        )
        pe = PositionalEncoding(config)

        # Should not have W_theta
        assert not hasattr(pe, 'W_theta')

        # Should still encode positions
        encodings = pe.encode_positions(length=10)
        assert encodings.shape == (10, 64)


class TestRelativePositionEmbedding:
    """Tests for RelativePositionEmbedding."""

    @pytest.fixture
    def embedding(self):
        """Create relative position embedding."""
        return RelativePositionEmbedding(
            max_relative_distance=16,
            embed_dim=64,
        )

    def test_initialization(self, embedding):
        """Test initialization."""
        assert embedding.max_relative_distance == 16
        assert embedding.embed_dim == 64
        # 2 * 16 + 1 = 33 possible relative positions
        assert embedding.embeddings.shape == (33, 64)

    def test_get_relative_embeddings(self, embedding):
        """Test getting relative embeddings for sequence."""
        rel_emb = embedding.get_relative_embeddings(length=8)

        assert rel_emb.shape == (8, 8, 64)

    def test_relative_embeddings_symmetry(self, embedding):
        """Test relative embeddings for symmetric positions."""
        rel_emb = embedding.get_relative_embeddings(length=8)

        # Position (0, 1) has relative distance 1
        # Position (1, 0) has relative distance -1
        # These should use different embeddings
        assert rel_emb.shape == (8, 8, 64)


class TestTemporalAttentionIntegration:
    """Integration tests for temporal attention components."""

    def test_full_positional_encoding_pipeline(self):
        """Test complete positional encoding with theta-gamma."""
        config = TemporalAttentionConfig(
            embed_dim=128,
            max_sequence_length=64,
            use_theta_modulation=True,
            gamma_slots_per_theta=7,
            theta_position_weight=0.3,
        )

        pe = PositionalEncoding(config)

        # Encode a sequence
        encodings = pe.encode_positions(
            length=21,  # 3 theta cycles
            theta_phase=0.0,
        )

        assert encodings.shape == (21, 128)

        # Check that different positions have different encodings
        assert not np.allclose(encodings[0], encodings[7])  # Different theta cycles
        assert not np.allclose(encodings[0], encodings[1])  # Different gamma slots

    def test_sinusoidal_encoding_consistency(self):
        """Test that sinusoidal encoding is consistent."""
        config = TemporalAttentionConfig(
            embed_dim=64,
            max_sequence_length=32,
            positional_type="sinusoidal",
            use_theta_modulation=False,
        )

        pe1 = PositionalEncoding(config)
        pe2 = PositionalEncoding(config)

        # Sinusoidal should be deterministic
        np.testing.assert_array_equal(pe1.encodings, pe2.encodings)

    def test_relative_and_absolute_positions_combined(self):
        """Test using both absolute and relative positions."""
        config = TemporalAttentionConfig(
            embed_dim=64,
            max_sequence_length=32,
            use_relative_positions=True,
            max_relative_distance=16,
        )

        pe = PositionalEncoding(config)
        rel_emb = RelativePositionEmbedding(
            max_relative_distance=config.max_relative_distance,
            embed_dim=config.embed_dim,
        )

        length = 10

        # Get absolute encodings
        abs_enc = pe.encode_positions(length)

        # Get relative embeddings
        rel_enc = rel_emb.get_relative_embeddings(length)

        assert abs_enc.shape == (length, 64)
        assert rel_enc.shape == (length, length, 64)
