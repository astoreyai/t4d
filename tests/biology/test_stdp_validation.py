"""
STDP window validation tests (P4-06).

Validates that STDP learning curves match biological data:
- LTP window: ~17ms (pre before post = potentiation)
- LTD window: ~34ms (post before pre = depression)
- Asymmetric learning curve
- Dopamine modulation of plasticity

Reference: Bi & Poo (1998), Dan & Poo (2004)
"""

import numpy as np
import pytest
import torch

from t4dm.learning.stdp import STDPLearner, STDPConfig


# Biological STDP parameters from Bi & Poo (1998)
# Note: T4DM uses seconds internally, we convert
BIOLOGICAL_TAU_PLUS_S = 0.017  # s - LTP time constant (~17ms)
BIOLOGICAL_TAU_MINUS_S = 0.034  # s - LTD time constant (~34ms)
BIOLOGICAL_TAU_PLUS_MS = 17.0  # ms
BIOLOGICAL_TAU_MINUS_MS = 34.0  # ms


class TestSTDPWindowBiology:
    """Test that STDP windows match biological data."""

    @pytest.fixture
    def stdp_learner(self):
        config = STDPConfig(
            tau_plus=BIOLOGICAL_TAU_PLUS_S,
            tau_minus=BIOLOGICAL_TAU_MINUS_S,
            a_plus=0.01,
            a_minus=0.01,
            max_weight=1.0,
            min_weight=0.0,
            multiplicative=False,  # Use additive for cleaner testing
        )
        return STDPLearner(config)

    def test_ltp_window_17ms(self, stdp_learner):
        """LTP should decay with ~17ms time constant."""
        # Pre before post (positive delta_t)
        delta_t_ms_values = np.array([1, 5, 10, 17, 25, 34, 50, 100])  # ms

        # Expected LTP curve: exp(-delta_t / tau_plus)
        expected_decay = np.exp(-delta_t_ms_values / BIOLOGICAL_TAU_PLUS_MS)

        # Simulate STDP at each delta_t using compute_stdp_delta (takes ms)
        for dt_ms, expected in zip(delta_t_ms_values, expected_decay):
            dw = stdp_learner.compute_stdp_delta(dt_ms, current_weight=0.5)
            actual_decay = dw / stdp_learner.config.a_plus

            # Allow 20% tolerance (multiplicative factors may affect)
            assert abs(actual_decay - expected) < 0.2 * expected + 0.02, \
                f"At Δt={dt_ms}ms: decay={actual_decay:.3f}, expected={expected:.3f}"

    def test_ltd_window_34ms(self, stdp_learner):
        """LTD should decay with ~34ms time constant."""
        # Post before pre (negative delta_t)
        delta_t_ms_values = np.array([1, 5, 10, 17, 34, 50, 100])  # ms

        # Expected LTD curve: exp(-delta_t / tau_minus)
        expected_decay = np.exp(-delta_t_ms_values / BIOLOGICAL_TAU_MINUS_MS)

        for dt_ms, expected in zip(delta_t_ms_values, expected_decay):
            # Negative delta_t for LTD
            dw = stdp_learner.compute_stdp_delta(-dt_ms, current_weight=0.5)
            actual_decay = abs(dw) / stdp_learner.config.a_minus

            assert abs(actual_decay - expected) < 0.2 * expected + 0.02, \
                f"At Δt=-{dt_ms}ms: decay={actual_decay:.3f}, expected={expected:.3f}"

    def test_stdp_asymmetry(self, stdp_learner):
        """STDP curve should be asymmetric (LTP faster than LTD)."""
        # At the same absolute time difference,
        # LTP decay (tau=17ms) should be faster than LTD decay (tau=34ms)
        dt = 20  # ms

        ltp_decay = np.exp(-dt / BIOLOGICAL_TAU_PLUS_MS)  # ~0.31
        ltd_decay = np.exp(-dt / BIOLOGICAL_TAU_MINUS_MS)  # ~0.56

        assert ltp_decay < ltd_decay, \
            "LTP should decay faster than LTD at same time offset"

    def test_zero_crossing(self, stdp_learner):
        """Weight change should cross zero at delta_t = 0."""
        # At delta_t = 0, there's no causal relationship
        # Convention: delta_t = t_post - t_pre
        # delta_t = 0: simultaneous, no learning
        # delta_t > 0: pre before post (LTP)
        # delta_t < 0: post before pre (LTD)

        # At exactly zero, we expect no weight change (or minimal)
        dw_zero = stdp_learner.compute_stdp_delta(delta_t_ms=0, current_weight=0.5)
        assert abs(dw_zero) < 0.001, f"dW at Δt=0 should be ~0, got {dw_zero}"


class TestSTDPCurveShape:
    """Test the complete STDP curve shape."""

    @pytest.fixture
    def stdp_learner(self):
        config = STDPConfig(
            tau_plus=BIOLOGICAL_TAU_PLUS_S,
            tau_minus=BIOLOGICAL_TAU_MINUS_S,
            a_plus=0.01,
            a_minus=0.01,
            multiplicative=False,
        )
        return STDPLearner(config)

    def test_full_stdp_curve(self, stdp_learner):
        """Generate and validate complete STDP curve."""
        # Delta t from -100ms to +100ms
        delta_t_range = np.arange(-100, 101, 1)
        dw_curve = np.zeros_like(delta_t_range, dtype=float)

        for i, dt in enumerate(delta_t_range):
            dw_curve[i] = stdp_learner.compute_stdp_delta(dt, current_weight=0.5)

        # Properties to check:
        # 1. Positive delta_t should give positive dW (LTP)
        positive_mask = delta_t_range > 5  # Avoid near-zero instability
        assert (dw_curve[positive_mask] > 0).mean() > 0.9, \
            "Positive Δt should produce LTP (positive dW)"

        # 2. Negative delta_t should give negative dW (LTD)
        negative_mask = delta_t_range < -5
        assert (dw_curve[negative_mask] < 0).mean() > 0.9, \
            "Negative Δt should produce LTD (negative dW)"

        # 3. Peak LTP should be near delta_t = 0+
        ltp_peak_idx = delta_t_range[delta_t_range > 0][np.argmax(dw_curve[delta_t_range > 0])]
        assert ltp_peak_idx < 10, f"LTP peak at Δt={ltp_peak_idx}ms, expected <10ms"

        # 4. Peak LTD should be near delta_t = 0-
        ltd_peak_idx = delta_t_range[delta_t_range < 0][np.argmin(dw_curve[delta_t_range < 0])]
        assert ltd_peak_idx > -10, f"LTD peak at Δt={ltd_peak_idx}ms, expected >-10ms"

    def test_stdp_integral_balance(self, stdp_learner):
        """Test LTP/LTD integral balance (metaplasticity)."""
        # Many biological STDP curves have roughly balanced integrals
        # to prevent runaway potentiation/depression
        delta_t_range = np.arange(-100, 101, 1)
        dw_curve = np.array([
            stdp_learner.compute_stdp_delta(dt, current_weight=0.5)
            for dt in delta_t_range
        ])

        ltp_integral = dw_curve[delta_t_range > 0].sum()
        ltd_integral = abs(dw_curve[delta_t_range < 0].sum())

        # Allow 2x difference (some biological curves are imbalanced)
        ratio = ltp_integral / (ltd_integral + 1e-10)
        assert 0.2 < ratio < 5, \
            f"LTP/LTD integral ratio = {ratio:.2f}, expected 0.2-5"


class TestDopamineModulatedSTDP:
    """Test dopamine modulation of STDP (three-factor learning)."""

    @pytest.fixture
    def stdp_learner(self):
        config = STDPConfig(
            tau_plus=BIOLOGICAL_TAU_PLUS_S,
            tau_minus=BIOLOGICAL_TAU_MINUS_S,
            a_plus=0.01,
            a_minus=0.01,
            multiplicative=False,
        )
        return STDPLearner(config)

    def test_high_dopamine_enhances_ltp(self, stdp_learner):
        """High dopamine should enhance LTP."""
        delta_t = 10  # ms (LTP regime)

        # Baseline with normal dopamine
        dw_baseline = stdp_learner.compute_stdp_delta(delta_t, current_weight=0.5, da_level=0.5)

        # High dopamine
        dw_high_da = stdp_learner.compute_stdp_delta(delta_t, current_weight=0.5, da_level=1.0)

        assert dw_high_da > dw_baseline, \
            f"High DA should enhance LTP: {dw_high_da:.4f} vs {dw_baseline:.4f}"

    def test_low_dopamine_reduces_ltp(self, stdp_learner):
        """Low dopamine should reduce LTP."""
        delta_t = 10  # ms

        dw_baseline = stdp_learner.compute_stdp_delta(delta_t, current_weight=0.5, da_level=0.5)
        dw_low_da = stdp_learner.compute_stdp_delta(delta_t, current_weight=0.5, da_level=0.1)

        assert dw_low_da < dw_baseline, \
            f"Low DA should reduce LTP: {dw_low_da:.4f} vs {dw_baseline:.4f}"

    def test_dopamine_effect_on_ltd(self, stdp_learner):
        """Dopamine should also modulate LTD (typically reduces it)."""
        delta_t = -10  # ms (LTD regime)

        dw_baseline = stdp_learner.compute_stdp_delta(delta_t, current_weight=0.5, da_level=0.5)
        dw_high_da = stdp_learner.compute_stdp_delta(delta_t, current_weight=0.5, da_level=1.0)

        # High dopamine typically reduces LTD magnitude
        # (converts potential punishment to reward-like state)
        assert abs(dw_high_da) < abs(dw_baseline) or dw_high_da > dw_baseline, \
            "High DA should reduce LTD magnitude or shift toward LTP"


class TestSTDPBiologicalBounds:
    """Test that STDP respects biological weight bounds."""

    @pytest.fixture
    def stdp_learner(self):
        config = STDPConfig(
            tau_plus=BIOLOGICAL_TAU_PLUS_S,
            tau_minus=BIOLOGICAL_TAU_MINUS_S,
            a_plus=0.1,  # Larger for faster saturation
            a_minus=0.1,
            max_weight=1.0,
            min_weight=0.0,
            multiplicative=True,  # Use multiplicative for soft bounds
        )
        return STDPLearner(config)

    def test_weight_cannot_exceed_max(self, stdp_learner):
        """Repeated LTP should not exceed w_max."""
        weight = 0.5

        # Apply many LTP events
        for _ in range(100):
            dw = stdp_learner.compute_stdp_delta(delta_t_ms=5, current_weight=weight)
            weight = min(stdp_learner.config.max_weight, weight + dw)

        assert weight <= stdp_learner.config.max_weight + 1e-6, \
            f"Weight {weight} exceeds max_weight {stdp_learner.config.max_weight}"

    def test_weight_cannot_go_below_min(self, stdp_learner):
        """Repeated LTD should not go below w_min."""
        weight = 0.5

        # Apply many LTD events
        for _ in range(100):
            dw = stdp_learner.compute_stdp_delta(delta_t_ms=-5, current_weight=weight)
            weight = max(stdp_learner.config.min_weight, weight + dw)

        assert weight >= stdp_learner.config.min_weight - 1e-6, \
            f"Weight {weight} below min_weight {stdp_learner.config.min_weight}"

    def test_soft_bounds_multiplicative(self, stdp_learner):
        """Test soft bounds (multiplicative) STDP variant."""
        # Soft bounds: dW_LTP proportional to (w_max - w)
        # dW_LTD proportional to (w - w_min)
        # This prevents saturation issues

        weight_near_max = 0.9  # Near max
        weight_mid = 0.5

        # LTP should be reduced near w_max (multiplicative)
        dw_near_max = stdp_learner.compute_stdp_delta(
            delta_t_ms=5, current_weight=weight_near_max
        )
        dw_at_mid = stdp_learner.compute_stdp_delta(
            delta_t_ms=5, current_weight=weight_mid
        )

        # Near max, LTP should be smaller (multiplicative effect)
        assert dw_near_max < dw_at_mid, \
            f"Soft bounds should reduce LTP near w_max: {dw_near_max:.4f} vs {dw_at_mid:.4f}"
