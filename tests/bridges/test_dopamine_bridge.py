"""
Tests for Dopamine Bridge module.

Tests predictive coding to dopamine system integration.
"""

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

from ww.bridges.dopamine_bridge import (
    BridgeConfig,
    BridgeState,
    PredictiveCodingDopamineBridge,
    create_pc_dopamine_bridge,
)


class TestBridgeConfig:
    """Tests for BridgeConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BridgeConfig()
        assert config.pe_to_rpe_gain == 0.5
        assert config.baseline_error == 0.3
        assert config.blend_ratio == 0.5
        assert config.precision_floor == 0.01
        assert config.update_da_expectations is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BridgeConfig(
            pe_to_rpe_gain=0.8,
            baseline_error=0.5,
            blend_ratio=0.7,
        )
        assert config.pe_to_rpe_gain == 0.8
        assert config.baseline_error == 0.5
        assert config.blend_ratio == 0.7


class TestBridgeState:
    """Tests for BridgeState dataclass."""

    def test_default_values(self):
        """Test default BridgeState values."""
        state = BridgeState()
        assert state.last_pe_signal == 0.0
        assert state.last_blended_rpe == 0.0
        assert state.n_signals_processed == 0
        assert state.mean_pe == 0.0
        assert isinstance(state.timestamp, datetime)

    def test_custom_values(self):
        """Test custom BridgeState values."""
        state = BridgeState(
            last_pe_signal=0.5,
            last_blended_rpe=0.3,
            n_signals_processed=10,
        )
        assert state.last_pe_signal == 0.5
        assert state.last_blended_rpe == 0.3
        assert state.n_signals_processed == 10


class TestPredictiveCodingDopamineBridge:
    """Tests for PredictiveCodingDopamineBridge."""

    @pytest.fixture
    def mock_hierarchy(self):
        """Mock predictive coding hierarchy."""
        hierarchy = MagicMock()
        hierarchy.compute_dopamine_signal.return_value = 0.5
        hierarchy.get_precision_weighted_error.return_value = np.array([0.2, 0.3, 0.4])
        hierarchy.process.return_value = MagicMock()
        return hierarchy

    @pytest.fixture
    def mock_dopamine(self):
        """Mock dopamine system."""
        da = MagicMock()
        rpe_result = MagicMock()
        rpe_result.delta = 0.3
        da.compute_rpe.return_value = rpe_result
        da.update_expectations.return_value = None
        return da

    @pytest.fixture
    def bridge(self, mock_hierarchy, mock_dopamine):
        """Create bridge with mocked dependencies."""
        return PredictiveCodingDopamineBridge(
            hierarchy=mock_hierarchy,
            dopamine=mock_dopamine,
            config=BridgeConfig(),
        )

    def test_initialization(self, bridge):
        """Test bridge initialization."""
        assert bridge.config is not None
        assert bridge.hierarchy is not None
        assert bridge.dopamine is not None
        assert bridge.state is not None

    def test_compute_pe_signal(self, bridge, mock_hierarchy):
        """Test computing PE signal from hierarchy."""
        pe_signal = bridge.compute_pe_signal()

        assert pe_signal == 0.5
        mock_hierarchy.compute_dopamine_signal.assert_called_once()
        assert bridge.state.last_pe_signal == 0.5
        assert bridge.state.n_signals_processed == 1

    def test_blend_with_internal_rpe(self, bridge, mock_hierarchy, mock_dopamine):
        """Test blending PE with internal RPE."""
        memory_id = uuid4()
        outcome = 0.8

        blended = bridge.blend_with_internal_rpe(memory_id, outcome)

        # PE = 0.5, internal RPE = 0.3, blend_ratio = 0.5
        # blended = 0.5 * 0.5 + 0.5 * 0.3 = 0.4
        assert blended == pytest.approx(0.4, rel=0.01)
        mock_dopamine.compute_rpe.assert_called_once_with(memory_id, outcome)
        mock_dopamine.update_expectations.assert_called_once()

    def test_blend_without_dopamine_system(self, mock_hierarchy):
        """Test blending when no dopamine system provided."""
        bridge = PredictiveCodingDopamineBridge(
            hierarchy=mock_hierarchy,
            dopamine=None,
        )

        memory_id = uuid4()
        outcome = 0.7

        blended = bridge.blend_with_internal_rpe(memory_id, outcome)

        # PE = 0.5, internal RPE = outcome (0.7), blend_ratio = 0.5
        # blended = 0.5 * 0.5 + 0.5 * 0.7 = 0.6
        assert blended == pytest.approx(0.6, rel=0.01)

    def test_process_with_dopamine(self, bridge, mock_hierarchy):
        """Test full processing pipeline."""
        sensory_input = np.random.randn(128).astype(np.float32)
        memory_id = uuid4()
        outcome = 0.9

        state = bridge.process_with_dopamine(sensory_input, memory_id, outcome)

        assert isinstance(state, BridgeState)
        mock_hierarchy.process.assert_called_once()
        assert state.n_signals_processed > 0

    def test_process_without_outcome(self, bridge, mock_hierarchy):
        """Test processing without outcome (no blending)."""
        sensory_input = np.random.randn(128).astype(np.float32)
        memory_id = uuid4()

        state = bridge.process_with_dopamine(sensory_input, memory_id, outcome=None)

        assert isinstance(state, BridgeState)
        # Should only compute PE, not blend
        assert state.last_pe_signal != 0.0

    def test_get_learning_modulation(self, bridge):
        """Test learning modulation computation."""
        # Set PE signal
        bridge.state.last_pe_signal = 0.0
        mod = bridge.get_learning_modulation()
        assert mod == pytest.approx(1.0, rel=0.01)

        # Positive PE = enhanced learning
        bridge.state.last_pe_signal = 1.0
        mod = bridge.get_learning_modulation()
        assert mod > 1.0
        assert mod <= 2.0

        # Negative PE = reduced learning
        bridge.state.last_pe_signal = -1.0
        mod = bridge.get_learning_modulation()
        assert mod < 1.0
        assert mod >= 0.5

    def test_get_statistics(self, bridge):
        """Test statistics retrieval."""
        stats = bridge.get_statistics()

        assert "last_pe_signal" in stats
        assert "last_blended_rpe" in stats
        assert "n_signals_processed" in stats
        assert "mean_pe" in stats
        assert "learning_modulation" in stats
        assert "config" in stats

    def test_mean_pe_updates(self, bridge, mock_hierarchy):
        """Test that mean PE is updated with EMA."""
        # Compute multiple signals
        for i in range(5):
            mock_hierarchy.compute_dopamine_signal.return_value = 0.5
            bridge.compute_pe_signal()

        # Mean should approach 0.5
        assert 0.1 < bridge.state.mean_pe < 0.6

    def test_config_affects_blend(self, mock_hierarchy, mock_dopamine):
        """Test that config blend_ratio affects blending."""
        # Mostly PE
        config_pe = BridgeConfig(blend_ratio=0.9)
        bridge_pe = PredictiveCodingDopamineBridge(
            hierarchy=mock_hierarchy,
            dopamine=mock_dopamine,
            config=config_pe,
        )

        # Mostly internal RPE
        config_rpe = BridgeConfig(blend_ratio=0.1)
        bridge_rpe = PredictiveCodingDopamineBridge(
            hierarchy=mock_hierarchy,
            dopamine=mock_dopamine,
            config=config_rpe,
        )

        memory_id = uuid4()
        blended_pe = bridge_pe.blend_with_internal_rpe(memory_id, 0.8)
        blended_rpe = bridge_rpe.blend_with_internal_rpe(memory_id, 0.8)

        # PE-weighted should be closer to PE signal (0.5)
        # RPE-weighted should be closer to internal RPE (0.3)
        assert blended_pe > blended_rpe


class TestCreatePCDopamineBridge:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Test factory with default blend ratio."""
        hierarchy = MagicMock()
        hierarchy.compute_dopamine_signal.return_value = 0.5

        bridge = create_pc_dopamine_bridge(hierarchy)

        assert bridge.config.blend_ratio == 0.5
        assert bridge.dopamine is None

    def test_create_with_custom_blend(self):
        """Test factory with custom blend ratio."""
        hierarchy = MagicMock()
        hierarchy.compute_dopamine_signal.return_value = 0.5
        dopamine = MagicMock()

        bridge = create_pc_dopamine_bridge(
            hierarchy,
            dopamine=dopamine,
            blend_ratio=0.7,
        )

        assert bridge.config.blend_ratio == 0.7
        assert bridge.dopamine is dopamine


class TestDopamineBridgeIntegration:
    """Integration tests for dopamine bridge."""

    def test_full_signal_flow(self):
        """Test complete signal flow from PE to learning modulation."""
        # Mock hierarchy with specific behavior
        hierarchy = MagicMock()
        hierarchy.compute_dopamine_signal.return_value = 0.7
        hierarchy.process.return_value = MagicMock()

        # Mock DA system
        da = MagicMock()
        rpe_result = MagicMock()
        rpe_result.delta = 0.4
        da.compute_rpe.return_value = rpe_result

        bridge = PredictiveCodingDopamineBridge(
            hierarchy=hierarchy,
            dopamine=da,
        )

        # Full processing
        sensory = np.random.randn(128).astype(np.float32)
        memory_id = uuid4()

        state = bridge.process_with_dopamine(sensory, memory_id, outcome=0.8)

        # Should produce valid state
        assert state.last_pe_signal == 0.7
        assert state.n_signals_processed >= 1

        # Learning modulation should be elevated (positive PE)
        mod = bridge.get_learning_modulation()
        assert mod > 1.0

    def test_repeated_signals_update_state(self):
        """Test that repeated signals update internal state."""
        hierarchy = MagicMock()
        hierarchy.compute_dopamine_signal.return_value = 0.5
        hierarchy.process.return_value = MagicMock()

        bridge = PredictiveCodingDopamineBridge(hierarchy=hierarchy)

        # Compute multiple signals
        for _ in range(10):
            bridge.compute_pe_signal()

        assert bridge.state.n_signals_processed == 10
        assert bridge.state.mean_pe > 0  # Should have accumulated

    def test_disable_da_expectations_update(self):
        """Test disabling DA expectations update."""
        hierarchy = MagicMock()
        hierarchy.compute_dopamine_signal.return_value = 0.5

        da = MagicMock()
        rpe_result = MagicMock()
        rpe_result.delta = 0.3
        da.compute_rpe.return_value = rpe_result

        config = BridgeConfig(update_da_expectations=False)
        bridge = PredictiveCodingDopamineBridge(
            hierarchy=hierarchy,
            dopamine=da,
            config=config,
        )

        memory_id = uuid4()
        bridge.blend_with_internal_rpe(memory_id, 0.8)

        # Should NOT call update_expectations
        da.update_expectations.assert_not_called()
