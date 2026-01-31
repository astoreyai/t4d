"""Runtime invariant checking for NCA modules.

Provides a decorator that validates pre/post conditions for NCA methods.
"""

from __future__ import annotations
import functools
import logging
import numpy as np

logger = logging.getLogger(__name__)


class InvariantViolation(RuntimeError):
    """Raised when an NCA invariant is violated."""
    pass


def check_no_nan(result, name: str):
    """Check that result contains no NaN/Inf values."""
    if isinstance(result, np.ndarray):
        if not np.all(np.isfinite(result)):
            raise InvariantViolation(
                f"{name}: result contains NaN/Inf "
                f"(count: {np.count_nonzero(~np.isfinite(result))})"
            )


def check_sparsity(result, max_sparsity: float, name: str):
    """Check that sparsity doesn't exceed target."""
    if isinstance(result, np.ndarray):
        nonzero_frac = np.count_nonzero(result) / max(len(result), 1)
        if nonzero_frac > max_sparsity + 0.01:
            raise InvariantViolation(
                f"{name}: sparsity violation. "
                f"Nonzero fraction {nonzero_frac:.3f} > target {max_sparsity:.3f}"
            )


def check_unit_norm(result, name: str, tolerance: float = 0.1):
    """Check that result is approximately unit-normalized."""
    if isinstance(result, np.ndarray) and len(result) > 0:
        norm = np.linalg.norm(result)
        if norm > 0 and abs(norm - 1.0) > tolerance:
            logger.warning(f"{name}: norm={norm:.4f}, expected ~1.0")


def check_bounded(result, low: float, high: float, name: str):
    """Check that result is within bounds."""
    if isinstance(result, (int, float)):
        if not (low <= result <= high):
            raise InvariantViolation(
                f"{name}: value {result} outside bounds [{low}, {high}]"
            )
    elif isinstance(result, np.ndarray):
        if np.any(result < low) or np.any(result > high):
            raise InvariantViolation(
                f"{name}: array values outside bounds [{low}, {high}]"
            )
