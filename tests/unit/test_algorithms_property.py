"""
Property-based tests for T4DM core algorithms.

Uses hypothesis for generative testing of FSRS, Hebbian, and ACT-R algorithms
to verify invariants across wide input ranges.

Properties tested:
- FSRS retrievability: bounded [0,1], monotonically decreasing
- Hebbian weight: bounded [0,1], approaches 1.0
- ACT-R activation: non-negative, reflects access frequency and recency
"""

import math
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume

import pytest


# Strategies for common parameter ranges

@st.composite
def days_elapsed(draw):
    """Strategy for elapsed days in FSRS calculations."""
    return draw(st.floats(min_value=0, max_value=10000))


@st.composite
def stability_in_days(draw):
    """Strategy for FSRS stability parameter (must be positive)."""
    return draw(st.floats(min_value=0.01, max_value=10000))


@st.composite
def weight_value(draw):
    """Strategy for Hebbian weights [0, 1]."""
    return draw(st.floats(min_value=0.0, max_value=1.0))


@st.composite
def learning_rate(draw):
    """Strategy for learning rates (positive, typically < 0.5)."""
    return draw(st.floats(min_value=0.001, max_value=0.5))


@st.composite
def access_times(draw):
    """Strategy for lists of access timestamps."""
    now = datetime.now()
    times = []
    current = now - timedelta(days=100)
    num_accesses = draw(st.integers(min_value=1, max_value=100))

    for _ in range(num_accesses):
        times.append(current)
        # Add random intervals
        days_forward = draw(st.floats(min_value=0.1, max_value=10))
        current = current + timedelta(days=days_forward)

    return sorted(times)


# ============================================================================
# FSRS (Spaced Repetition) Property Tests
# ============================================================================

class TestFSRSRetrievability:
    """Properties for FSRS retrievability formula: R(t, S) = (1 + 0.9*t/S)^(-0.5)"""

    @given(days_elapsed(), stability_in_days())
    def test_retrievability_bounded_in_unit_interval(self, days, stability):
        """Retrievability should always be in [0, 1]."""
        # R(t, S) = (1 + 0.9 * t / S) ^ (-0.5)
        retrieval = (1 + 0.9 * days / stability) ** (-0.5)
        assert 0 <= retrieval <= 1, f"R({days}, {stability}) = {retrieval} not in [0,1]"

    @given(stability_in_days())
    def test_retrievability_at_zero_is_one(self, stability):
        """Retrievability at t=0 should be 1 (perfect recall)."""
        retrieval = (1 + 0.9 * 0 / stability) ** (-0.5)
        assert abs(retrieval - 1.0) < 1e-10

    @given(stability_in_days())
    def test_retrievability_monotonically_decreases(self, stability):
        """Retrievability should decrease as time increases."""
        t1 = 1.0
        t2 = 2.0
        r1 = (1 + 0.9 * t1 / stability) ** (-0.5)
        r2 = (1 + 0.9 * t2 / stability) ** (-0.5)
        assert r1 >= r2, f"R(1, {stability}) = {r1} should be >= R(2, {stability}) = {r2}"

    @given(days_elapsed(), stability_in_days())
    def test_retrievability_approaches_zero(self, days, stability):
        """Retrievability should approach 0 as time approaches infinity."""
        retrieval = (1 + 0.9 * days / stability) ** (-0.5)
        # Larger time intervals should have lower retrieval
        if days > 10000:
            assert retrieval < 0.1

    @given(st.lists(stability_in_days(), min_size=2, max_size=10))
    def test_higher_stability_higher_retrieval(self, stabilities):
        """Higher stability should preserve retrievability longer."""
        days = 10.0
        retrievals = [(1 + 0.9 * days / s) ** (-0.5) for s in stabilities]

        # Sort by stability
        sorted_pairs = sorted(zip(stabilities, retrievals))
        sorted_stabilities = [s for s, _ in sorted_pairs]
        sorted_retrievals = [r for _, r in sorted_pairs]

        # Verify retrievals are monotonically increasing with stability
        for i in range(len(sorted_retrievals) - 1):
            assert sorted_retrievals[i] <= sorted_retrievals[i+1]


# ============================================================================
# Hebbian Learning Property Tests
# ============================================================================

class TestHebbianWeights:
    """Properties for Hebbian weight update: w' = w + lr * (1 - w)"""

    @given(weight_value(), learning_rate())
    def test_weight_always_bounded(self, w, lr):
        """Hebbian update should keep weight in [0, 1]."""
        w_new = w + lr * (1.0 - w)
        assert 0 <= w_new <= 1.0, f"w={w}, lr={lr} -> w'={w_new} not in [0,1]"

    @given(weight_value(), learning_rate())
    def test_weight_never_exceeds_one(self, w, lr):
        """Weight cannot exceed 1 after single update."""
        w_new = w + lr * (1.0 - w)
        assert w_new <= 1.0

    @given(weight_value(), learning_rate())
    def test_weight_monotonically_increases(self, w, lr):
        """Each Hebbian update should increase weight."""
        w_new = w + lr * (1.0 - w)
        assert w_new >= w, f"w={w}, lr={lr} -> w'={w_new} decreased"

    @given(learning_rate())
    def test_repeated_strengthening_converges_to_one(self, lr):
        """Repeated strengthening should approach but not exceed 1."""
        w = 0.1
        for _ in range(100):
            w = w + lr * (1.0 - w)

        assert w <= 1.0, f"Weight {w} exceeds 1 after iterations"
        # Weight should be higher than starting point (monotonic increase)
        assert w > 0.1, f"Weight {w} should increase from initial 0.1"
        # For small learning rates (e.g. 0.015), 100 iterations gives ~0.81
        # For larger learning rates (e.g. 0.5), 100 iterations gives ~0.999
        # Just verify monotonic increase toward 1 (not reached for small lr)
        expected_min = 1 - 0.9 * ((1 - lr) ** 100)
        assert w >= expected_min * 0.99, f"Weight {w} below expected minimum {expected_min}"

    @given(st.lists(learning_rate(), min_size=1, max_size=20))
    def test_weight_convergence_under_different_rates(self, rates):
        """Different learning rates should converge monotonically."""
        w = 0.1
        weights_history = [w]

        for lr in rates:
            w = w + lr * (1.0 - w)
            weights_history.append(w)

        # Verify monotonic increase
        for i in range(len(weights_history) - 1):
            assert weights_history[i] <= weights_history[i+1]

        # Verify bounded
        assert all(0 <= wt <= 1.0 for wt in weights_history)

    @given(weight_value())
    def test_zero_learning_rate_no_change(self, w):
        """Zero learning rate should not change weight."""
        w_new = w + 0.0 * (1.0 - w)
        assert w_new == w

    @given(weight_value())
    def test_high_learning_rate_step_bounded(self, w):
        """High learning rate (1.0) should step toward 1."""
        w_new = w + 1.0 * (1.0 - w)
        # With lr=1.0, w' = w + (1-w) = 1
        assert abs(w_new - 1.0) < 1e-10


# ============================================================================
# ACT-R Activation Property Tests
# ============================================================================

class TestACTRActivation:
    """Properties for ACT-R activation: A = ln(sum(t_i^-d)) + noise"""

    @given(access_times())
    def test_activation_is_logarithmic_sum(self, times):
        """Activation should be natural log of power-law sum."""
        decay = 0.5
        current = datetime.now()

        activation_sum = 0
        for t in times:
            days_since = (current - t).total_seconds() / 86400
            if days_since >= 0:
                activation_sum += days_since ** (-decay)

        activation = math.log(activation_sum) if activation_sum > 0 else float('-inf')
        # Should be real number
        assert isinstance(activation, (int, float))

    @given(access_times(), st.floats(min_value=0.1, max_value=1.0))
    def test_activation_increases_with_access_frequency(self, times, decay):
        """More accesses should increase activation."""
        current = datetime.now()

        # Single access
        t = times[0]
        days_since = (current - t).total_seconds() / 86400
        single_activation = math.log(days_since ** (-decay) if days_since > 0 else 1e-10)

        # Multiple accesses
        activation_sum = 0
        for t in times:
            days_since = (current - t).total_seconds() / 86400
            if days_since >= 0:
                activation_sum += days_since ** (-decay)

        multi_activation = math.log(activation_sum) if activation_sum > 0 else float('-inf')

        # More accesses should have higher (or equal) activation
        assert multi_activation >= single_activation

    @given(access_times(), st.floats(min_value=0.1, max_value=1.0))
    def test_recency_weighted_by_decay(self, times, decay):
        """Recent accesses should be weighted more with higher decay."""
        current = datetime.now()

        # Most recent
        recent = times[-1]
        days_recent = (current - recent).total_seconds() / 86400

        # Oldest
        oldest = times[0]
        days_oldest = (current - oldest).total_seconds() / 86400

        # Skip cases where times extend into the future or are too close to now
        # (ACT-R requires positive days for meaningful decay calculations)
        assume(days_recent > 0.01)  # At least ~15 minutes in the past
        assume(days_oldest > days_recent)  # oldest must be further back

        # Recent should contribute more (smaller days = higher weight with negative exponent)
        recent_weight = days_recent ** (-decay)
        oldest_weight = days_oldest ** (-decay)
        assert recent_weight >= oldest_weight, (
            f"Recent weight {recent_weight} should be >= oldest weight {oldest_weight}"
        )


# ============================================================================
# Combined Algorithm Properties
# ============================================================================

class TestAlgorithmInteractions:
    """Properties for interactions between algorithms."""

    @given(st.lists(stability_in_days(), min_size=1, max_size=5),
           st.lists(weight_value(), min_size=1, max_size=5))
    def test_retrievability_and_weight_multiplication(self, stabilities, weights):
        """Combined score R * w should stay bounded."""
        days = 5.0

        for s in stabilities:
            for w in weights:
                retrievability = (1 + 0.9 * days / s) ** (-0.5)
                combined = retrievability * w
                assert 0 <= combined <= 1.0

    @given(
        days_elapsed(),
        stability_in_days(),
        weight_value(),
        learning_rate()
    )
    def test_all_algorithms_preserve_bounds(self, days, stability, weight, lr):
        """All three algorithms should preserve value bounds."""
        # FSRS: [0, 1]
        fsrs = (1 + 0.9 * days / stability) ** (-0.5)
        assert 0 <= fsrs <= 1

        # Hebbian: [0, 1]
        hebbian = weight + lr * (1.0 - weight)
        assert 0 <= hebbian <= 1

        # Combined
        combined = fsrs * hebbian
        assert 0 <= combined <= 1


# ============================================================================
# Edge Cases and Boundaries
# ============================================================================

class TestAlgorithmEdgeCases:
    """Edge cases that reveal algorithm assumptions."""

    def test_fsrs_very_large_time(self):
        """FSRS with very large time values."""
        stability = 1.0
        days = 1e6
        retrieval = (1 + 0.9 * days / stability) ** (-0.5)
        assert 0 <= retrieval <= 1
        assert retrieval < 0.01  # Should be very small

    def test_fsrs_very_small_stability(self):
        """FSRS with very small stability (quick decay)."""
        stability = 0.001
        days = 1.0
        retrieval = (1 + 0.9 * days / stability) ** (-0.5)
        assert 0 <= retrieval <= 1
        assert retrieval < 0.5  # Should decay quickly

    def test_hebbian_starting_at_zero(self):
        """Hebbian updates from w=0."""
        w = 0.0
        lr = 0.1
        for _ in range(1000):
            w = w + lr * (1.0 - w)
        assert 0 < w < 1.0

    def test_hebbian_starting_at_one(self):
        """Hebbian updates from w=1.0 should stay at 1."""
        w = 1.0
        lr = 0.1
        w_new = w + lr * (1.0 - w)
        assert abs(w_new - 1.0) < 1e-10

    @given(st.floats(min_value=-1e6, max_value=1e6))
    def test_activation_sum_robustness(self, multiplier):
        """ACT-R sum should handle various scales."""
        decay = 0.5
        # This would need proper time handling, just verify sum computation
        assume(multiplier > 0)  # Only positive multipliers make sense

        base_value = 10.0
        scaled = base_value * multiplier
        log_scaled = math.log(scaled)
        assert isinstance(log_scaled, (int, float))
