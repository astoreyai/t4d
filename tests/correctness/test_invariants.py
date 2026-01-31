"""Tests for NCA invariants."""
import numpy as np
import pytest
from t4dm.nca.invariants import InvariantViolation, check_no_nan, check_sparsity, check_bounded


class TestInvariants:
    def test_check_no_nan_detects_nan(self):
        arr = np.array([1.0, float('nan'), 3.0])
        with pytest.raises(InvariantViolation):
            check_no_nan(arr, "test")

    def test_check_no_nan_detects_inf(self):
        arr = np.array([1.0, float('inf'), 3.0])
        with pytest.raises(InvariantViolation):
            check_no_nan(arr, "test")

    def test_check_no_nan_passes_clean(self):
        arr = np.array([1.0, 2.0, 3.0])
        check_no_nan(arr, "test")  # Should not raise

    def test_check_sparsity_detects_violation(self):
        arr = np.ones(100)  # 100% nonzero
        with pytest.raises(InvariantViolation):
            check_sparsity(arr, max_sparsity=0.1, name="test")

    def test_check_sparsity_passes(self):
        arr = np.zeros(100)
        arr[:5] = 1.0  # 5% nonzero
        check_sparsity(arr, max_sparsity=0.1, name="test")  # Should not raise

    def test_check_bounded_detects_violation_high(self):
        with pytest.raises(InvariantViolation):
            check_bounded(1.5, 0.0, 1.0, "test")

    def test_check_bounded_detects_violation_low(self):
        with pytest.raises(InvariantViolation):
            check_bounded(-0.5, 0.0, 1.0, "test")

    def test_check_bounded_passes(self):
        check_bounded(0.5, 0.0, 1.0, "test")  # Should not raise

    def test_check_bounded_array_violation(self):
        arr = np.array([0.5, 1.5, 0.3])
        with pytest.raises(InvariantViolation):
            check_bounded(arr, 0.0, 1.0, "test")

    def test_check_bounded_array_passes(self):
        arr = np.array([0.5, 0.8, 0.3])
        check_bounded(arr, 0.0, 1.0, "test")  # Should not raise
