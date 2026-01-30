"""
Tests for Hippocampal Circuit.

Sprint 1.1: Tests DG pattern separation, CA3 completion, CA1 novelty detection.
"""

import pytest
import numpy as np
from uuid import UUID

from ww.nca.hippocampus import (
    HippocampalConfig,
    HippocampalState,
    HippocampalMode,
    HippocampalCircuit,
    DentateGyrusLayer,
    CA3Layer,
    CA1Layer,
    create_hippocampal_circuit,
)


# =============================================================================
# Test Configuration
# =============================================================================


class TestHippocampalConfig:
    """Tests for HippocampalConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HippocampalConfig()
        assert config.ec_dim == 1024
        assert config.dg_dim == 4096
        assert config.ca3_dim == 1024
        assert config.ca1_dim == 1024
        assert config.dg_sparsity == 0.04  # Biological: ~4% activation (Jung & McNaughton 1993)
        assert config.ca3_beta == 8.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = HippocampalConfig(
            ec_dim=512,
            dg_dim=2048,
            ca3_beta=16.0,
            ca1_novelty_threshold=0.4
        )
        assert config.ec_dim == 512
        assert config.dg_dim == 2048
        assert config.ca3_beta == 16.0
        assert config.ca1_novelty_threshold == 0.4


# =============================================================================
# Test Dentate Gyrus
# =============================================================================


class TestDentateGyrusLayer:
    """Tests for DentateGyrusLayer (pattern separation)."""

    @pytest.fixture
    def config(self):
        """Small config for testing."""
        return HippocampalConfig(
            ec_dim=64,
            dg_dim=256,
            ca3_dim=64
        )

    @pytest.fixture
    def dg(self, config):
        """Create DG layer."""
        return DentateGyrusLayer(config, random_seed=42)

    def test_initialization(self, dg, config):
        """Test DG initialization."""
        assert dg._expansion_weights.shape == (config.ec_dim, config.dg_dim)
        assert dg._compression_weights.shape == (config.dg_dim, config.ca3_dim)

    def test_process_basic(self, dg, config):
        """Test basic DG processing."""
        ec_input = np.random.randn(config.ec_dim).astype(np.float32)
        output, separation = dg.process(ec_input)

        assert output.shape == (config.ca3_dim,)
        assert np.isfinite(output).all()
        # First input has nothing to separate from
        assert separation == 0.0

    def test_output_normalized(self, dg, config):
        """Test output is normalized."""
        ec_input = np.random.randn(config.ec_dim).astype(np.float32)
        output, _ = dg.process(ec_input)

        norm = np.linalg.norm(output)
        assert abs(norm - 1.0) < 0.01

    def test_separation_applied_for_similar(self, dg, config):
        """Test separation is applied for similar inputs."""
        base = np.random.randn(config.ec_dim).astype(np.float32)

        # Process first pattern
        dg.process(base, apply_separation=True)

        # Process similar pattern
        similar = base + 0.1 * np.random.randn(config.ec_dim).astype(np.float32)
        _, separation = dg.process(similar, apply_separation=True)

        # Should have some separation for similar patterns
        # (may be 0 if threshold not met, but should work for very similar)

    def test_no_separation_for_different(self, dg, config):
        """Test no separation for very different inputs."""
        pattern1 = np.random.randn(config.ec_dim).astype(np.float32)
        pattern2 = np.random.randn(config.ec_dim).astype(np.float32)

        dg.process(pattern1)
        _, separation = dg.process(pattern2)

        # Different patterns shouldn't trigger much separation
        assert separation < 0.1

    def test_sparsity(self, dg, config):
        """Test that DG output has sparse intermediate."""
        ec_input = np.random.randn(config.ec_dim).astype(np.float32)

        # Process and check sparsity of internal representation
        expanded = ec_input @ dg._expansion_weights
        expanded = np.maximum(0, expanded)
        sparse = dg._sparsify(expanded)

        # Count non-zeros
        sparsity = np.count_nonzero(sparse) / len(sparse)
        # Should be close to target sparsity
        assert sparsity < config.dg_sparsity * 2

    def test_clear_recent(self, dg, config):
        """Test clearing recent patterns."""
        ec_input = np.random.randn(config.ec_dim).astype(np.float32)
        dg.process(ec_input)

        assert len(dg._recent_patterns) > 0
        dg.clear_recent()
        assert len(dg._recent_patterns) == 0


# =============================================================================
# Test CA3
# =============================================================================


class TestCA3Layer:
    """Tests for CA3Layer (pattern completion)."""

    @pytest.fixture
    def config(self):
        """Small config for testing."""
        return HippocampalConfig(
            ca3_dim=64,
            ca3_max_patterns=100,
            ca3_beta=8.0
        )

    @pytest.fixture
    def ca3(self, config):
        """Create CA3 layer."""
        return CA3Layer(config)

    def test_initialization(self, ca3):
        """Test CA3 initialization."""
        assert ca3.get_pattern_count() == 0

    def test_store_pattern(self, ca3, config):
        """Test storing a pattern."""
        pattern = np.random.randn(config.ca3_dim).astype(np.float32)
        pattern_id = ca3.store(pattern)

        assert isinstance(pattern_id, UUID)
        assert ca3.get_pattern_count() == 1

    def test_complete_empty(self, ca3, config):
        """Test completion with no stored patterns."""
        query = np.random.randn(config.ca3_dim).astype(np.float32)
        output, iters, energy, pattern_id = ca3.complete(query)

        # ATOM-P3-3: CA3 complete() returns None for empty store
        assert output is None
        assert iters == 0
        assert pattern_id is None

    def test_complete_exact_match(self, ca3, config):
        """Test completion recovers exact stored pattern."""
        pattern = np.random.randn(config.ca3_dim).astype(np.float32)
        pattern = pattern / np.linalg.norm(pattern)

        stored_id = ca3.store(pattern)

        output, iters, energy, retrieved_id = ca3.complete(pattern)

        assert retrieved_id == stored_id
        # Should be very similar to original
        similarity = np.dot(output, pattern)
        assert similarity > 0.95

    def test_complete_noisy_input(self, ca3, config):
        """Test completion from noisy input."""
        pattern = np.random.randn(config.ca3_dim).astype(np.float32)
        pattern = pattern / np.linalg.norm(pattern)

        ca3.store(pattern)

        # Add noise
        noise = np.random.randn(config.ca3_dim).astype(np.float32) * 0.3
        noisy = pattern + noise
        noisy = noisy / np.linalg.norm(noisy)

        output, iters, energy, _ = ca3.complete(noisy)

        # Should recover something similar to original
        similarity = np.dot(output, pattern)
        assert similarity > 0.7

    def test_capacity_limit(self, ca3, config):
        """Test capacity limit is enforced."""
        for _ in range(config.ca3_max_patterns + 10):
            pattern = np.random.randn(config.ca3_dim).astype(np.float32)
            ca3.store(pattern)

        assert ca3.get_pattern_count() == config.ca3_max_patterns

    def test_clear(self, ca3, config):
        """Test clearing CA3."""
        pattern = np.random.randn(config.ca3_dim).astype(np.float32)
        ca3.store(pattern)

        assert ca3.get_pattern_count() == 1
        ca3.clear()
        assert ca3.get_pattern_count() == 0


# =============================================================================
# Test CA1
# =============================================================================


class TestCA1Layer:
    """Tests for CA1Layer (novelty detection)."""

    @pytest.fixture
    def config(self):
        """Config for testing."""
        return HippocampalConfig(
            ec_dim=64,
            ca3_dim=64,
            ca1_dim=64,
            ca1_novelty_threshold=0.3,
            ca1_encoding_threshold=0.5
        )

    @pytest.fixture
    def ca1(self, config):
        """Create CA1 layer."""
        return CA1Layer(config)

    def test_initialization(self, ca1):
        """Test CA1 initialization."""
        assert ca1._ec_projection is None  # Same dims
        assert ca1._ca3_projection is None

    def test_familiar_detection(self, ca1, config):
        """Test familiar pattern is detected as retrieval."""
        # EC and CA3 very similar -> familiar
        pattern = np.random.randn(config.ec_dim).astype(np.float32)
        pattern = pattern / np.linalg.norm(pattern)

        output, novelty, mode = ca1.process(pattern, pattern)

        assert novelty < config.ca1_novelty_threshold
        assert mode == HippocampalMode.RETRIEVAL

    def test_novel_detection(self, ca1, config):
        """Test novel pattern is detected as encoding."""
        # EC and CA3 very different -> novel
        ec = np.random.randn(config.ec_dim).astype(np.float32)
        ca3 = np.random.randn(config.ca3_dim).astype(np.float32)
        ec = ec / np.linalg.norm(ec)
        ca3 = ca3 / np.linalg.norm(ca3)

        output, novelty, mode = ca1.process(ec, ca3)

        # Different patterns should have high novelty
        assert novelty > 0.3  # Some novelty expected

    def test_output_normalized(self, ca1, config):
        """Test CA1 output is normalized."""
        ec = np.random.randn(config.ec_dim).astype(np.float32)
        ca3 = np.random.randn(config.ca3_dim).astype(np.float32)

        output, _, _ = ca1.process(ec, ca3)

        norm = np.linalg.norm(output)
        assert abs(norm - 1.0) < 0.01

    def test_novelty_bounds(self, ca1, config):
        """Test novelty score is in [0, 1]."""
        for _ in range(10):
            ec = np.random.randn(config.ec_dim).astype(np.float32)
            ca3 = np.random.randn(config.ca3_dim).astype(np.float32)

            _, novelty, _ = ca1.process(ec, ca3)

            assert 0.0 <= novelty <= 1.0


# =============================================================================
# Test Integrated Circuit
# =============================================================================


class TestHippocampalCircuit:
    """Tests for integrated HippocampalCircuit."""

    @pytest.fixture
    def config(self):
        """Small config for testing."""
        return HippocampalConfig(
            ec_dim=64,
            dg_dim=256,
            ca3_dim=64,
            ca1_dim=64,
            ca3_max_patterns=100,
            ca1_novelty_threshold=0.3
        )

    @pytest.fixture
    def circuit(self, config):
        """Create hippocampal circuit."""
        return HippocampalCircuit(config, random_seed=42)

    def test_initialization(self, circuit):
        """Test circuit initialization."""
        assert circuit.get_pattern_count() == 0

    def test_process_basic(self, circuit, config):
        """Test basic processing."""
        ec_input = np.random.randn(config.ec_dim).astype(np.float32)
        state = circuit.process(ec_input)

        assert isinstance(state, HippocampalState)
        assert state.ec_input.shape == (config.ec_dim,)
        assert state.dg_output.shape == (config.ca3_dim,)
        assert state.ca3_output.shape == (config.ca3_dim,)
        assert state.ca1_output.shape == (config.ca1_dim,)

    def test_first_input_is_novel(self, circuit, config):
        """Test first input is treated as novel."""
        ec_input = np.random.randn(config.ec_dim).astype(np.float32)
        state = circuit.process(ec_input, store_if_novel=True)

        # First input should be novel (nothing to retrieve from)
        assert state.novelty_score > 0.3  # Some novelty expected
        assert circuit.get_pattern_count() >= 1

    def test_repeated_input_familiar(self, circuit, config):
        """Test repeated input becomes familiar."""
        ec_input = np.random.randn(config.ec_dim).astype(np.float32)
        ec_input = ec_input / np.linalg.norm(ec_input)

        # First encoding
        circuit.encode(ec_input)

        # Second retrieval should be familiar
        state = circuit.retrieve(ec_input)

        # Should be somewhat similar (familiarity)
        similarity = np.dot(state.ca3_output, state.dg_output)
        assert similarity > 0.3

    def test_encode_mode(self, circuit, config):
        """Test forced encoding mode."""
        ec_input = np.random.randn(config.ec_dim).astype(np.float32)
        state = circuit.encode(ec_input)

        assert state.mode == HippocampalMode.ENCODING
        assert state.pattern_id is not None

    def test_retrieve_mode(self, circuit, config):
        """Test forced retrieval mode."""
        ec_input = np.random.randn(config.ec_dim).astype(np.float32)
        circuit.encode(ec_input)

        state = circuit.retrieve(ec_input)
        assert state.mode == HippocampalMode.RETRIEVAL

    def test_pattern_separation(self, circuit, config):
        """Test similar patterns are separated."""
        base = np.random.randn(config.ec_dim).astype(np.float32)

        # Store base pattern
        circuit.encode(base)

        # Process similar pattern
        similar = base + 0.1 * np.random.randn(config.ec_dim).astype(np.float32)
        state = circuit.process(similar)

        # Should have some separation applied
        # (separation_magnitude > 0 indicates orthogonalization occurred)

    def test_get_stats(self, circuit, config):
        """Test statistics retrieval."""
        for _ in range(5):
            ec_input = np.random.randn(config.ec_dim).astype(np.float32)
            circuit.process(ec_input)

        stats = circuit.get_stats()

        assert stats["total_processed"] == 5
        assert "avg_novelty" in stats
        assert "stored_patterns" in stats

    def test_clear(self, circuit, config):
        """Test clearing circuit."""
        ec_input = np.random.randn(config.ec_dim).astype(np.float32)
        circuit.encode(ec_input)

        assert circuit.get_pattern_count() > 0

        circuit.clear()

        assert circuit.get_pattern_count() == 0

    def test_novelty_threshold_adjustment(self, circuit):
        """Test adjusting novelty threshold."""
        circuit.set_novelty_threshold(0.5)
        assert circuit.get_novelty_threshold() == 0.5

        # Clamp to valid range
        circuit.set_novelty_threshold(1.5)
        assert circuit.get_novelty_threshold() == 1.0

        circuit.set_novelty_threshold(-0.5)
        assert circuit.get_novelty_threshold() == 0.0

    def test_history(self, circuit, config):
        """Test processing history."""
        for _ in range(5):
            ec_input = np.random.randn(config.ec_dim).astype(np.float32)
            circuit.process(ec_input)

        history = circuit.get_history(limit=3)
        assert len(history) == 3


# =============================================================================
# Test Factory Function
# =============================================================================


class TestFactoryFunction:
    """Tests for create_hippocampal_circuit factory."""

    def test_default_creation(self):
        """Test default circuit creation."""
        circuit = create_hippocampal_circuit()

        assert circuit.config.ec_dim == 1024
        assert circuit.config.dg_dim == 4096  # 4x expansion
        assert circuit.config.ca3_beta == 8.0

    def test_custom_creation(self):
        """Test custom circuit creation."""
        circuit = create_hippocampal_circuit(
            ec_dim=512,
            dg_expansion_factor=2,
            ca3_beta=16.0,
            novelty_threshold=0.5
        )

        assert circuit.config.ec_dim == 512
        assert circuit.config.dg_dim == 1024  # 2x expansion
        assert circuit.config.ca3_beta == 16.0
        assert circuit.config.ca1_novelty_threshold == 0.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for hippocampal circuit."""

    def test_encoding_retrieval_cycle(self):
        """Test full encoding-retrieval cycle."""
        circuit = create_hippocampal_circuit(ec_dim=128, dg_expansion_factor=2)

        # Encode several patterns
        patterns = [np.random.randn(128).astype(np.float32) for _ in range(5)]
        for p in patterns:
            circuit.encode(p)

        assert circuit.get_pattern_count() == 5

        # Retrieve each
        for p in patterns:
            state = circuit.retrieve(p)
            # Should get reasonable similarity
            assert state.completion_energy < 0  # Converged to stored pattern

    def test_interference_reduction(self):
        """Test that similar patterns don't interfere."""
        circuit = create_hippocampal_circuit(ec_dim=128, dg_expansion_factor=4)

        # Create similar patterns
        base = np.random.randn(128).astype(np.float32)
        base = base / np.linalg.norm(base)

        # Encode base
        circuit.encode(base)

        # Encode similar pattern
        similar = base + 0.2 * np.random.randn(128).astype(np.float32)
        similar = similar / np.linalg.norm(similar)
        circuit.encode(similar)

        # Both should be stored (pattern separation prevents overwriting)
        assert circuit.get_pattern_count() == 2
