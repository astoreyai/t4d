"""
Unit Tests for Reconsolidation Lability (W2-02).

Verifies memory reconsolidation with lability windows following
O'Reilly's CLS principles and Nader et al. (2000) protein synthesis findings.

Evidence Base: Nader et al. (2000) "Fear memories require protein synthesis"
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4


class TestReconsolidationConfig:
    """Test reconsolidation configuration."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        from t4dm.learning.reconsolidation import ReconsolidationConfig

        config = ReconsolidationConfig()

        assert config.lability_window_seconds == 300  # 5 minutes
        assert config.kappa_drop_threshold == 0.7
        assert config.kappa_drop_amount == 0.4
        assert config.mismatch_threshold == 0.3

    def test_config_override(self):
        """Should be able to override config values."""
        from t4dm.learning.reconsolidation import ReconsolidationConfig

        config = ReconsolidationConfig(
            lability_window_seconds=600,
            kappa_drop_threshold=0.8,
            kappa_drop_amount=0.5,
            mismatch_threshold=0.2,
        )

        assert config.lability_window_seconds == 600
        assert config.kappa_drop_threshold == 0.8


class TestReconsolidationResult:
    """Test reconsolidation result data structure."""

    def test_result_creation_triggered(self):
        """Should create result for triggered reconsolidation."""
        from t4dm.learning.reconsolidation import ReconsolidationResult

        result = ReconsolidationResult(
            triggered=True,
            reason="mismatch_detected",
            mismatch_score=0.45,
            new_kappa=0.5,
            lability_expires=time.time() + 300,
        )

        assert result.triggered is True
        assert result.reason == "mismatch_detected"
        assert result.mismatch_score == 0.45

    def test_result_creation_not_triggered(self):
        """Should create result for non-triggered case."""
        from t4dm.learning.reconsolidation import ReconsolidationResult

        result = ReconsolidationResult(
            triggered=False,
            reason="not_consolidated",
        )

        assert result.triggered is False
        assert result.reason == "not_consolidated"
        assert result.mismatch_score is None


class TestReconsolidationManager:
    """Test ReconsolidationManager behavior."""

    @pytest.fixture
    def config(self):
        """Create test config with short window."""
        from t4dm.learning.reconsolidation import ReconsolidationConfig

        return ReconsolidationConfig(
            lability_window_seconds=0.5,  # 500ms for faster tests
            kappa_drop_threshold=0.7,
            kappa_drop_amount=0.4,
            mismatch_threshold=0.3,
        )

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        engine = Mock()
        return engine

    def test_manager_creation(self, config, mock_engine):
        """Should create manager with config and engine."""
        from t4dm.learning.reconsolidation import ReconsolidationManager

        manager = ReconsolidationManager(config, mock_engine)

        assert manager.config is config
        assert manager.engine is mock_engine

    def test_no_reconsolidation_for_low_kappa(self, config, mock_engine):
        """Low-κ memories should not trigger reconsolidation."""
        from t4dm.learning.reconsolidation import ReconsolidationManager

        manager = ReconsolidationManager(config, mock_engine)

        # Create unconsolidated memory
        memory_id = uuid4()
        mock_memory = Mock()
        mock_memory.kappa = 0.2  # Below threshold
        mock_memory.vector = np.random.randn(64)
        mock_engine.get.return_value = mock_memory

        # Reactivate with mismatch
        mismatched_context = np.random.randn(64)
        result = manager.on_reactivation(memory_id, mismatched_context)

        assert not result.triggered
        assert result.reason == "not_consolidated"

    def test_no_reconsolidation_when_no_mismatch(self, config, mock_engine):
        """Similar context should not trigger reconsolidation."""
        from t4dm.learning.reconsolidation import ReconsolidationManager

        manager = ReconsolidationManager(config, mock_engine)

        # Create consolidated memory
        memory_id = uuid4()
        original_vector = np.random.randn(64)
        original_vector = original_vector / np.linalg.norm(original_vector)

        mock_memory = Mock()
        mock_memory.kappa = 0.9  # Above threshold
        mock_memory.vector = original_vector
        mock_engine.get.return_value = mock_memory

        # Reactivate with similar context (low mismatch)
        similar_context = original_vector + np.random.randn(64) * 0.01
        similar_context = similar_context / np.linalg.norm(similar_context)

        result = manager.on_reactivation(memory_id, similar_context)

        assert not result.triggered
        assert result.reason == "no_mismatch"

    def test_reconsolidation_on_mismatch(self, config, mock_engine):
        """Mismatch should trigger lability window."""
        from t4dm.learning.reconsolidation import ReconsolidationManager

        manager = ReconsolidationManager(config, mock_engine)

        # Create consolidated memory
        memory_id = uuid4()
        original_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 61)
        original_vector = original_vector / np.linalg.norm(original_vector)

        mock_memory = Mock()
        mock_memory.kappa = 0.9  # Above threshold
        mock_memory.vector = original_vector
        mock_engine.get.return_value = mock_memory

        # Reactivate with mismatched context (very different)
        mismatched_context = np.array([0.0, 1.0, 0.0] + [0.0] * 61)
        mismatched_context = mismatched_context / np.linalg.norm(mismatched_context)

        result = manager.on_reactivation(memory_id, mismatched_context)

        assert result.triggered
        assert result.reason == "mismatch_detected"
        assert result.mismatch_score > config.mismatch_threshold

    def test_kappa_drops_when_labile(self, config, mock_engine):
        """κ should drop when memory becomes labile."""
        from t4dm.learning.reconsolidation import ReconsolidationManager

        manager = ReconsolidationManager(config, mock_engine)

        memory_id = uuid4()
        original_kappa = 0.9

        mock_memory = Mock()
        mock_memory.kappa = original_kappa
        mock_memory.vector = np.array([1.0] + [0.0] * 63)
        mock_engine.get.return_value = mock_memory

        # Trigger reconsolidation
        mismatched = np.array([0.0, 1.0] + [0.0] * 62)
        result = manager.on_reactivation(memory_id, mismatched)

        # Verify update_fields was called with dropped kappa
        mock_engine.update_fields.assert_called()
        call_args = mock_engine.update_fields.call_args
        updates = call_args[0][1]

        expected_new_kappa = original_kappa - config.kappa_drop_amount
        assert updates["kappa"] == expected_new_kappa
        assert updates["labile"] is True

    def test_lability_window_tracked(self, config, mock_engine):
        """Lability window should be tracked."""
        from t4dm.learning.reconsolidation import ReconsolidationManager

        manager = ReconsolidationManager(config, mock_engine)

        memory_id = uuid4()
        mock_memory = Mock()
        mock_memory.kappa = 0.9
        mock_memory.vector = np.array([1.0] + [0.0] * 63)
        mock_engine.get.return_value = mock_memory

        # Trigger reconsolidation
        mismatched = np.array([0.0, 1.0] + [0.0] * 62)
        manager.on_reactivation(memory_id, mismatched)

        # Memory should be in labile_memories
        assert memory_id in manager.labile_memories

    def test_lability_window_expires(self, config, mock_engine):
        """Lability window should close after timeout."""
        from t4dm.learning.reconsolidation import ReconsolidationManager

        manager = ReconsolidationManager(config, mock_engine)

        memory_id = uuid4()

        # First call: trigger lability (high kappa to pass threshold)
        mock_memory_initial = Mock()
        mock_memory_initial.kappa = 0.9  # Above threshold
        mock_memory_initial.labile = False
        mock_memory_initial.vector = np.array([1.0] + [0.0] * 63)
        mock_engine.get.return_value = mock_memory_initial

        # Trigger reconsolidation
        mismatched = np.array([0.0, 1.0] + [0.0] * 62)
        manager.on_reactivation(memory_id, mismatched)

        # Update mock for window close check (now labile with dropped kappa)
        mock_memory_labile = Mock()
        mock_memory_labile.kappa = 0.5  # After drop
        mock_memory_labile.labile = True
        mock_engine.get.return_value = mock_memory_labile

        # Wait for window to expire
        time.sleep(config.lability_window_seconds + 0.1)

        # Close windows
        manager.close_lability_windows()

        # Memory should no longer be in labile_memories
        assert memory_id not in manager.labile_memories

        # κ should be restored (at least 2 calls: one for drop, one for restore)
        assert mock_engine.update_fields.call_count >= 2

    def test_kappa_restored_if_not_updated(self, config, mock_engine):
        """κ should be restored if memory not updated during lability."""
        from t4dm.learning.reconsolidation import ReconsolidationManager

        manager = ReconsolidationManager(config, mock_engine)

        memory_id = uuid4()
        original_kappa = 0.9
        dropped_kappa = original_kappa - config.kappa_drop_amount

        # First call: trigger lability
        mock_memory_initial = Mock()
        mock_memory_initial.kappa = original_kappa
        mock_memory_initial.labile = False
        mock_memory_initial.vector = np.array([1.0] + [0.0] * 63)

        mock_engine.get.return_value = mock_memory_initial

        mismatched = np.array([0.0, 1.0] + [0.0] * 62)
        manager.on_reactivation(memory_id, mismatched)

        # Update mock for window close
        mock_memory_labile = Mock()
        mock_memory_labile.kappa = dropped_kappa
        mock_memory_labile.labile = True
        mock_engine.get.return_value = mock_memory_labile

        # Wait and close
        time.sleep(config.lability_window_seconds + 0.1)
        manager.close_lability_windows()

        # Check that kappa was restored
        calls = mock_engine.update_fields.call_args_list
        last_call = calls[-1]
        updates = last_call[0][1]

        assert updates["kappa"] == dropped_kappa + config.kappa_drop_amount
        assert updates["labile"] is False


class TestMismatchCalculation:
    """Test mismatch (cosine distance) calculation."""

    def test_identical_vectors_no_mismatch(self):
        """Identical vectors should have zero mismatch."""
        from t4dm.learning.reconsolidation import compute_mismatch

        v = np.random.randn(64)
        v = v / np.linalg.norm(v)

        mismatch = compute_mismatch(v, v)

        assert abs(mismatch) < 0.001

    def test_orthogonal_vectors_max_mismatch(self):
        """Orthogonal vectors should have mismatch = 1."""
        from t4dm.learning.reconsolidation import compute_mismatch

        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        mismatch = compute_mismatch(v1, v2)

        assert abs(mismatch - 1.0) < 0.001

    def test_opposite_vectors_max_mismatch(self):
        """Opposite vectors should have mismatch = 2."""
        from t4dm.learning.reconsolidation import compute_mismatch

        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])

        mismatch = compute_mismatch(v1, v2)

        assert abs(mismatch - 2.0) < 0.001


class TestLabilityWindowManagement:
    """Test lability window edge cases."""

    @pytest.fixture
    def config(self):
        from t4dm.learning.reconsolidation import ReconsolidationConfig

        return ReconsolidationConfig(lability_window_seconds=0.2)

    @pytest.fixture
    def mock_engine(self):
        return Mock()

    def test_multiple_reactivations_extend_window(self, config, mock_engine):
        """Multiple reactivations should extend lability window."""
        from t4dm.learning.reconsolidation import ReconsolidationManager

        manager = ReconsolidationManager(config, mock_engine)

        memory_id = uuid4()
        mock_memory = Mock()
        mock_memory.kappa = 0.9
        mock_memory.labile = False
        mock_memory.vector = np.array([1.0] + [0.0] * 63)
        mock_engine.get.return_value = mock_memory

        mismatched = np.array([0.0, 1.0] + [0.0] * 62)

        # First reactivation
        manager.on_reactivation(memory_id, mismatched)
        first_expires = manager.labile_memories[memory_id]

        time.sleep(0.1)

        # Second reactivation should extend
        mock_memory.kappa = 0.9  # Reset for second trigger
        mock_memory.labile = True
        manager.on_reactivation(memory_id, mismatched)
        second_expires = manager.labile_memories[memory_id]

        assert second_expires > first_expires

    def test_close_windows_handles_empty(self, config, mock_engine):
        """close_lability_windows should handle empty state."""
        from t4dm.learning.reconsolidation import ReconsolidationManager

        manager = ReconsolidationManager(config, mock_engine)

        # Should not raise
        manager.close_lability_windows()

        assert len(manager.labile_memories) == 0
