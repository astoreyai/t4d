"""
Numba JIT-optimized STDP computation kernels.

Provides 10-100x speedup for STDP weight updates by JIT-compiling
the inner loops. These functions are drop-in replacements for the
corresponding NumPy implementations in stdp.py.

Usage:
    from t4dm.learning.stdp_jit import compute_stdp_delta_jit, batch_stdp_update_jit

    # Single update
    delta = compute_stdp_delta_jit(delta_t_ms, current_weight, a_plus, a_minus,
                                    tau_plus, tau_minus, w_max, mu)

    # Batch update
    deltas = batch_stdp_update_jit(delta_t_array, weights, ...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

logger = logging.getLogger(__name__)

# Check if Numba is available
try:
    from numba import njit, prange, float64, int64, boolean

    NUMBA_AVAILABLE = True
    logger.info("Numba JIT compilation enabled for STDP")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, STDP will use NumPy fallback")

    # Create dummy decorator for fallback
    def njit(*args, **kwargs):
        """Dummy njit decorator when Numba is not available."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args, **kwargs):
        """Dummy prange when Numba is not available."""
        return range(*args)


# ============================================================================
# Core STDP Kernel (JIT-compiled)
# ============================================================================

@njit(cache=True, fastmath=True)
def _stdp_delta_kernel(
    delta_t_s: float,
    current_weight: float,
    a_plus: float,
    a_minus: float,
    tau_plus: float,
    tau_minus: float,
    w_max: float,
    mu: float,
    multiplicative: bool,
) -> float:
    """
    JIT-compiled STDP delta computation kernel.

    This is the innermost function that gets called millions of times
    during STDP updates. JIT compilation provides ~10-100x speedup.

    Args:
        delta_t_s: Time difference (post - pre) in seconds
        current_weight: Current synaptic weight
        a_plus: LTP amplitude
        a_minus: LTD amplitude
        tau_plus: LTP time constant (seconds)
        tau_minus: LTD time constant (seconds)
        w_max: Maximum weight
        mu: Weight dependence exponent
        multiplicative: Use multiplicative STDP

    Returns:
        Weight change delta
    """
    # Clamp weight to valid range
    w = min(max(current_weight, 0.0), w_max)

    if multiplicative:
        if delta_t_s > 0:
            # LTP: pre before post
            # Δw = A+ * (w_max - w)^μ * exp(-Δt/τ+)
            weight_factor = (w_max - w) ** mu
            return a_plus * weight_factor * np.exp(-delta_t_s / tau_plus)
        else:
            # LTD: post before pre
            # Δw = -A- * w^μ * exp(Δt/τ-)
            weight_factor = w ** mu
            return -a_minus * weight_factor * np.exp(delta_t_s / tau_minus)
    else:
        # Additive STDP
        if delta_t_s > 0:
            return a_plus * np.exp(-delta_t_s / tau_plus)
        else:
            return -a_minus * np.exp(delta_t_s / tau_minus)


@njit(cache=True, fastmath=True)
def _da_modulated_rates_kernel(
    da_level: float,
    a_plus_base: float,
    a_minus_base: float,
    ltp_gain: float,
    ltd_gain: float,
    baseline_da: float,
) -> tuple:
    """
    JIT-compiled dopamine modulation of STDP rates.

    Args:
        da_level: Dopamine level [0, 1]
        a_plus_base: Base LTP amplitude
        a_minus_base: Base LTD amplitude
        ltp_gain: LTP modulation strength
        ltd_gain: LTD modulation strength
        baseline_da: Baseline DA level

    Returns:
        (a_plus_modulated, a_minus_modulated)
    """
    # Normalize around baseline: range [-1, 1]
    da_mod = (da_level - baseline_da) / baseline_da
    da_mod = min(max(da_mod, -1.0), 1.0)

    # LTP modulation: High DA increases LTP
    ltp_mod = max(0.1, 1.0 + ltp_gain * da_mod)

    # LTD modulation: High DA decreases LTD (inverse)
    ltd_mod = max(0.1, 1.0 - ltd_gain * da_mod)

    return a_plus_base * ltp_mod, a_minus_base * ltd_mod


# ============================================================================
# Public API
# ============================================================================

def compute_stdp_delta_jit(
    delta_t_ms: float,
    current_weight: float = 0.5,
    a_plus: float = 0.01,
    a_minus: float = 0.0105,
    tau_plus: float = 0.017,
    tau_minus: float = 0.034,
    w_max: float = 1.0,
    mu: float = 0.5,
    multiplicative: bool = True,
    da_level: float | None = None,
) -> float:
    """
    Compute STDP weight change with JIT acceleration.

    Drop-in replacement for STDPLearner.compute_stdp_delta().

    Args:
        delta_t_ms: Time difference (post - pre) in milliseconds
        current_weight: Current synaptic weight
        a_plus: LTP amplitude
        a_minus: LTD amplitude
        tau_plus: LTP time constant (seconds)
        tau_minus: LTD time constant (seconds)
        w_max: Maximum weight
        mu: Weight dependence exponent
        multiplicative: Use multiplicative STDP
        da_level: Optional dopamine level for modulation

    Returns:
        Weight change delta
    """
    # Convert ms to seconds
    delta_t_s = delta_t_ms / 1000.0

    if abs(delta_t_ms) < 0.1:
        return 0.0

    # Apply dopamine modulation if provided
    if da_level is not None:
        a_plus, a_minus = _da_modulated_rates_kernel(
            da_level, a_plus, a_minus, 0.5, 0.3, 0.5
        )

    return _stdp_delta_kernel(
        delta_t_s, current_weight, a_plus, a_minus,
        tau_plus, tau_minus, w_max, mu, multiplicative
    )


# ============================================================================
# Batch Operations (Parallel JIT)
# ============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _batch_stdp_kernel(
    delta_t_array: ndarray,
    weights: ndarray,
    a_plus: float,
    a_minus: float,
    tau_plus: float,
    tau_minus: float,
    w_max: float,
    mu: float,
    multiplicative: bool,
) -> ndarray:
    """
    Batch STDP computation with parallel execution.

    Processes multiple synapses in parallel using Numba's prange.

    Args:
        delta_t_array: Array of time differences (seconds)
        weights: Array of current weights
        a_plus, a_minus, tau_plus, tau_minus, w_max, mu: STDP params
        multiplicative: Use multiplicative STDP

    Returns:
        Array of weight deltas
    """
    n = len(delta_t_array)
    deltas = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        delta_t_s = delta_t_array[i]
        w = weights[i]

        if abs(delta_t_s) < 1e-4:
            deltas[i] = 0.0
            continue

        deltas[i] = _stdp_delta_kernel(
            delta_t_s, w, a_plus, a_minus,
            tau_plus, tau_minus, w_max, mu, multiplicative
        )

    return deltas


def batch_stdp_update_jit(
    delta_t_array: ndarray,
    weights: ndarray,
    a_plus: float = 0.01,
    a_minus: float = 0.0105,
    tau_plus: float = 0.017,
    tau_minus: float = 0.034,
    w_max: float = 1.0,
    mu: float = 0.5,
    multiplicative: bool = True,
) -> ndarray:
    """
    Compute STDP weight changes for a batch of synapses.

    This is the main entry point for batch STDP updates.
    Provides significant speedup for large-scale updates.

    Args:
        delta_t_array: Array of time differences in SECONDS
        weights: Array of current weights
        a_plus, a_minus, tau_plus, tau_minus, w_max, mu: STDP params
        multiplicative: Use multiplicative STDP

    Returns:
        Array of weight deltas
    """
    # Ensure arrays are contiguous float64 for Numba
    delta_t_array = np.ascontiguousarray(delta_t_array, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)

    return _batch_stdp_kernel(
        delta_t_array, weights, a_plus, a_minus,
        tau_plus, tau_minus, w_max, mu, multiplicative
    )


# ============================================================================
# Spike Train Processing
# ============================================================================

@njit(cache=True, fastmath=True)
def _compute_pairwise_stdp(
    pre_times: ndarray,
    post_times: ndarray,
    weights: ndarray,
    a_plus: float,
    a_minus: float,
    tau_plus: float,
    tau_minus: float,
    w_max: float,
    mu: float,
    multiplicative: bool,
    window_s: float,
) -> ndarray:
    """
    Compute STDP updates for all spike pairs within a time window.

    Args:
        pre_times: Presynaptic spike times (seconds)
        post_times: Postsynaptic spike times (seconds)
        weights: Weight matrix [n_pre, n_post]
        ...: STDP parameters
        window_s: Time window for STDP (seconds)

    Returns:
        Weight delta matrix [n_pre, n_post]
    """
    n_pre = len(pre_times)
    n_post = len(post_times)
    n_weights = len(weights)

    # Flatten 2D weight matrix assumption: weights[i*n_post + j]
    deltas = np.zeros_like(weights)

    for i in range(n_pre):
        t_pre = pre_times[i]

        for j in range(n_post):
            t_post = post_times[j]
            delta_t = t_post - t_pre

            # Skip if outside window
            if abs(delta_t) > window_s:
                continue

            # Get weight index (assuming flattened matrix)
            idx = i * n_post + j
            if idx >= n_weights:
                continue

            w = weights[idx]
            delta_w = _stdp_delta_kernel(
                delta_t, w, a_plus, a_minus,
                tau_plus, tau_minus, w_max, mu, multiplicative
            )
            deltas[idx] += delta_w

    return deltas


def pairwise_stdp_update(
    pre_times: ndarray,
    post_times: ndarray,
    weights: ndarray,
    a_plus: float = 0.01,
    a_minus: float = 0.0105,
    tau_plus: float = 0.017,
    tau_minus: float = 0.034,
    w_max: float = 1.0,
    mu: float = 0.5,
    multiplicative: bool = True,
    window_ms: float = 100.0,
) -> ndarray:
    """
    Compute STDP updates for all pre-post spike pairs.

    Efficiently computes weight changes for all synapses based on
    pre and post synaptic spike trains.

    Args:
        pre_times: Array of presynaptic spike times (seconds)
        post_times: Array of postsynaptic spike times (seconds)
        weights: Weight array (flattened matrix or 1D)
        ...: STDP parameters
        window_ms: Time window in milliseconds

    Returns:
        Array of accumulated weight deltas
    """
    pre_times = np.ascontiguousarray(pre_times, dtype=np.float64)
    post_times = np.ascontiguousarray(post_times, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)

    window_s = window_ms / 1000.0

    return _compute_pairwise_stdp(
        pre_times, post_times, weights,
        a_plus, a_minus, tau_plus, tau_minus,
        w_max, mu, multiplicative, window_s
    )


# ============================================================================
# Benchmark Utility
# ============================================================================

def benchmark_stdp_jit(n_synapses: int = 100000, n_warmup: int = 3) -> dict:
    """
    Benchmark JIT vs NumPy STDP computation.

    Args:
        n_synapses: Number of synapses to test
        n_warmup: Number of warmup iterations

    Returns:
        Dictionary with timing results
    """
    import time

    # Generate test data
    np.random.seed(42)
    delta_t = np.random.uniform(-0.05, 0.05, n_synapses)  # -50ms to +50ms
    weights = np.random.uniform(0.1, 0.9, n_synapses)

    results = {}

    # Warmup JIT compilation
    for _ in range(n_warmup):
        _ = batch_stdp_update_jit(delta_t[:1000], weights[:1000])

    # Benchmark JIT
    start = time.perf_counter()
    deltas_jit = batch_stdp_update_jit(delta_t, weights)
    jit_time = time.perf_counter() - start
    results["jit_time"] = jit_time
    results["jit_throughput"] = n_synapses / jit_time

    # Benchmark NumPy (loop version)
    def numpy_stdp(delta_t, weights):
        deltas = np.zeros_like(weights)
        for i in range(len(delta_t)):
            dt = delta_t[i]
            w = weights[i]
            if abs(dt) < 1e-4:
                continue
            if dt > 0:
                deltas[i] = 0.01 * (1.0 - w) ** 0.5 * np.exp(-dt / 0.017)
            else:
                deltas[i] = -0.0105 * w ** 0.5 * np.exp(dt / 0.034)
        return deltas

    start = time.perf_counter()
    deltas_np = numpy_stdp(delta_t, weights)
    numpy_time = time.perf_counter() - start
    results["numpy_time"] = numpy_time
    results["numpy_throughput"] = n_synapses / numpy_time

    # Speedup
    results["speedup"] = numpy_time / jit_time
    results["n_synapses"] = n_synapses

    # Verify correctness (should match within float tolerance)
    results["max_error"] = np.max(np.abs(deltas_jit - deltas_np))

    return results


__all__ = [
    "compute_stdp_delta_jit",
    "batch_stdp_update_jit",
    "pairwise_stdp_update",
    "benchmark_stdp_jit",
    "NUMBA_AVAILABLE",
]
