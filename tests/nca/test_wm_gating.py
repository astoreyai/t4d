"""
Tests for Working Memory Gating module.

Tests theta-gamma modulation, encoding/retrieval gates, and maintenance.
"""

import numpy as np
import pytest

from ww.nca.wm_gating import (
    WMGatingConfig,
    WMItem,
    EncodingGate,
    RetrievalGate,
    MaintenanceController,
)


class TestWMGatingConfig:
    """Tests for WMGatingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WMGatingConfig()
        assert config.wm_capacity == 7
        assert config.embed_dim == 1024
        assert config.encoding_threshold == 0.5
        assert config.retrieval_threshold == 0.5
        assert config.eviction_threshold == 0.2
        assert config.decay_rate == 0.1
        assert config.theta_frequency == 6.0
        assert config.gamma_frequency == 40.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WMGatingConfig(
            wm_capacity=5,
            embed_dim=512,
            encoding_threshold=0.3,
        )
        assert config.wm_capacity == 5
        assert config.embed_dim == 512
        assert config.encoding_threshold == 0.3


class TestWMItem:
    """Tests for WMItem dataclass."""

    def test_default_values(self):
        """Test default WMItem values."""
        embedding = np.random.randn(64).astype(np.float32)
        item = WMItem(embedding=embedding)
        assert item.activation == 1.0
        assert item.attention == 1.0
        assert item.age == 0
        assert item.gamma_phase == 0.0
        assert item.priority == 1.0

    def test_custom_values(self):
        """Test custom WMItem values."""
        embedding = np.random.randn(64).astype(np.float32)
        item = WMItem(
            embedding=embedding,
            activation=0.5,
            attention=0.8,
            age=10,
        )
        assert item.activation == 0.5
        assert item.attention == 0.8
        assert item.age == 10


class TestEncodingGate:
    """Tests for EncodingGate."""

    @pytest.fixture
    def gate(self):
        """Create encoding gate."""
        return EncodingGate(threshold=0.5, alpha_weight=0.5)

    def test_initialization(self, gate):
        """Test gate initialization."""
        assert gate.threshold == 0.5
        assert gate.alpha_weight == 0.5

    def test_compute_gate_high_signal(self, gate):
        """Test gate with high encoding signal."""
        # High signal, low alpha, plenty of capacity
        result = gate.compute_gate(
            encoding_signal=1.0,
            alpha=0.0,
            capacity=2,
            max_capacity=7,
        )
        # Should be above threshold (sigmoid of 0.5)
        assert result > 0.5
        assert 0 <= result <= 1

    def test_compute_gate_low_signal(self, gate):
        """Test gate with low encoding signal."""
        result = gate.compute_gate(
            encoding_signal=0.0,
            alpha=0.0,
            capacity=2,
            max_capacity=7,
        )
        # Should be low gate value
        assert result < 0.5

    def test_compute_gate_alpha_inhibition(self, gate):
        """Test that alpha inhibits encoding."""
        # Same signal, but with high alpha
        result_low_alpha = gate.compute_gate(
            encoding_signal=0.8,
            alpha=0.0,
            capacity=2,
            max_capacity=7,
        )
        result_high_alpha = gate.compute_gate(
            encoding_signal=0.8,
            alpha=1.0,
            capacity=2,
            max_capacity=7,
        )
        # Alpha should reduce gate value
        assert result_high_alpha < result_low_alpha

    def test_compute_gate_capacity_suppression(self, gate):
        """Test that full capacity suppresses encoding."""
        result_available = gate.compute_gate(
            encoding_signal=0.8,
            alpha=0.0,
            capacity=2,
            max_capacity=7,
        )
        result_full = gate.compute_gate(
            encoding_signal=0.8,
            alpha=0.0,
            capacity=7,
            max_capacity=7,
        )
        # Full capacity should strongly suppress
        assert result_full < result_available * 0.2

    def test_should_encode_true(self, gate):
        """Test should_encode returns True for high priority."""
        result = gate.should_encode(item_priority=0.9, gate_value=0.8)
        assert result is True

    def test_should_encode_false(self, gate):
        """Test should_encode returns False for low values."""
        result = gate.should_encode(item_priority=0.3, gate_value=0.3)
        assert result is False


class TestRetrievalGate:
    """Tests for RetrievalGate."""

    @pytest.fixture
    def gate(self):
        """Create retrieval gate."""
        return RetrievalGate(threshold=0.5)

    def test_initialization(self, gate):
        """Test gate initialization."""
        assert gate.threshold == 0.5

    def test_compute_gate_high_retrieval(self, gate):
        """Test gate with high retrieval signal."""
        result = gate.compute_gate(
            retrieval_signal=1.0,
            attention=0.9,
            activation=0.9,
        )
        # Should be above threshold
        assert result > 0.4
        assert 0 <= result <= 1

    def test_compute_gate_low_attention(self, gate):
        """Test that low attention reduces retrieval."""
        result_high_att = gate.compute_gate(
            retrieval_signal=0.8,
            attention=1.0,
            activation=0.9,
        )
        result_low_att = gate.compute_gate(
            retrieval_signal=0.8,
            attention=0.2,
            activation=0.9,
        )
        assert result_low_att < result_high_att

    def test_compute_gate_low_activation(self, gate):
        """Test that low activation reduces retrieval."""
        result_high_act = gate.compute_gate(
            retrieval_signal=0.8,
            attention=0.9,
            activation=1.0,
        )
        result_low_act = gate.compute_gate(
            retrieval_signal=0.8,
            attention=0.9,
            activation=0.2,
        )
        assert result_low_act < result_high_act

    def test_retrieve_strength(self, gate):
        """Test retrieve_strength computation."""
        strength = gate.retrieve_strength(gate_value=0.8, activation=0.9)
        assert strength == pytest.approx(0.72, rel=0.01)


class TestMaintenanceController:
    """Tests for MaintenanceController."""

    @pytest.fixture
    def controller(self):
        """Create maintenance controller."""
        return MaintenanceController(
            decay_rate=0.1,
            rehearsal_boost=0.3,
            gamma_weight=0.3,
        )

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller.decay_rate == 0.1
        assert controller.rehearsal_boost == 0.3
        assert controller.gamma_weight == 0.3

    def test_update_activations_decay(self, controller):
        """Test that activations decay over time."""
        activations = np.array([1.0, 0.8, 0.6])
        attention = np.array([0.0, 0.0, 0.0])  # No attention = pure decay

        new_activations = controller.update_activations(
            activations=activations,
            attention=attention,
            gamma=0.0,
            alpha=0.0,
            dt=1.0,
        )

        # Activations should decrease
        assert all(new_activations <= activations)

    def test_update_activations_maintenance(self, controller):
        """Test that attention maintains activations."""
        activations = np.array([0.5, 0.5])
        attention = np.array([1.0, 0.0])  # First item attended

        new_activations = controller.update_activations(
            activations=activations,
            attention=attention,
            gamma=0.8,  # High gamma
            alpha=0.0,
            dt=1.0,
        )

        # Attended item should be maintained better
        assert new_activations[0] >= new_activations[1]


class TestWMGatingIntegration:
    """Integration tests for WM gating system."""

    def test_encoding_retrieval_cycle(self):
        """Test complete encode-maintain-retrieve cycle."""
        encoding_gate = EncodingGate(threshold=0.3)  # Lower threshold for encoding
        retrieval_gate = RetrievalGate(threshold=0.5)
        maintenance = MaintenanceController()

        # 1. Encoding: Strong signal should pass
        encode_gate = encoding_gate.compute_gate(
            encoding_signal=0.9,
            alpha=0.2,
            capacity=2,
            max_capacity=7,
        )
        should_encode = encoding_gate.should_encode(
            item_priority=0.8,
            gate_value=encode_gate,
        )
        assert should_encode is True

        # 2. Maintenance: Item stays active
        activations = np.array([1.0])
        attention = np.array([0.8])
        new_act = maintenance.update_activations(
            activations=activations,
            attention=attention,
            gamma=0.5,
            alpha=0.2,
        )
        assert new_act[0] > 0.3  # Still active

        # 3. Retrieval: Can retrieve active item
        retrieve_gate = retrieval_gate.compute_gate(
            retrieval_signal=0.8,
            attention=0.8,
            activation=new_act[0],
        )
        assert retrieve_gate > 0.3  # Retrievable

    def test_alpha_inhibition_blocks_distractors(self):
        """Test that high alpha blocks distractor encoding."""
        gate = EncodingGate(threshold=0.5, alpha_weight=0.8)

        # Moderate signal with high alpha inhibition
        result = gate.compute_gate(
            encoding_signal=0.6,
            alpha=0.9,
            capacity=3,
            max_capacity=7,
        )

        # Should be suppressed
        assert result < 0.3

    def test_capacity_limits_respected(self):
        """Test that capacity limits are enforced."""
        gate = EncodingGate(threshold=0.3)

        # Even with high signal, full capacity blocks encoding
        result_full = gate.compute_gate(
            encoding_signal=1.0,
            alpha=0.0,
            capacity=7,
            max_capacity=7,
        )

        result_partial = gate.compute_gate(
            encoding_signal=1.0,
            alpha=0.0,
            capacity=3,
            max_capacity=7,
        )

        assert result_full < result_partial * 0.5
