"""
Unit tests for eligibility trace system.
"""

import pytest
import time
import numpy as np

from t4dm.learning.eligibility import (
    EligibilityTrace,
    LayeredEligibilityTrace,
    EligibilityConfig,
    MAX_TRACES,
    MAX_TRACE_VALUE
)


class TestEligibilityTrace:
    """Tests for basic eligibility trace."""

    @pytest.fixture
    def trace(self):
        """Create default eligibility trace."""
        return EligibilityTrace(
            decay=0.95,
            tau_trace=1.0,  # Short tau for faster testing
            a_plus=0.1,
            min_trace=0.001
        )

    def test_initialization(self, trace):
        """Trace initializes correctly."""
        assert trace.decay == 0.95
        assert trace.tau_trace == 1.0
        assert trace.count == 0

    def test_security_validation(self):
        """Security limits are enforced."""
        # Invalid decay
        with pytest.raises(ValueError, match="decay"):
            EligibilityTrace(decay=1.5)

        # Invalid tau
        with pytest.raises(ValueError, match="tau_trace"):
            EligibilityTrace(tau_trace=-1)

        # Max traces limit
        with pytest.raises(ValueError, match="MAX_TRACES"):
            EligibilityTrace(max_traces=MAX_TRACES + 1)

    def test_update_creates_trace(self, trace):
        """Update creates new trace."""
        assert trace.count == 0

        trace.update("memory_1", activity=1.0)

        assert trace.count == 1
        assert trace.get_trace("memory_1") > 0

    def test_update_accumulates(self, trace):
        """Multiple updates accumulate trace."""
        trace.update("memory_1")
        initial = trace.get_trace("memory_1")

        trace.update("memory_1")
        accumulated = trace.get_trace("memory_1")

        assert accumulated > initial

    def test_exponential_decay(self, trace):
        """Traces decay exponentially."""
        trace.update("memory_1", activity=1.0)
        initial = trace.get_trace("memory_1")

        # Wait and decay
        time.sleep(0.1)
        trace.step(dt=0.5)  # Half second

        decayed = trace.get_trace("memory_1")
        expected_decay = np.exp(-0.5 / trace.tau_trace)

        # Should be roughly expected decay
        assert decayed < initial
        assert abs(decayed / initial - expected_decay) < 0.1

    def test_step_removes_weak_traces(self, trace):
        """Step removes traces below threshold."""
        trace.update("memory_1", activity=0.01)  # Small activity

        # Decay multiple times
        for _ in range(10):
            trace.step(dt=1.0)

        # Should be removed
        assert "memory_1" not in trace.traces

    def test_credit_assignment(self, trace):
        """Credit is assigned proportional to trace."""
        trace.update("memory_1", activity=1.0)
        trace.update("memory_2", activity=0.5)

        credits = trace.assign_credit(reward=10.0)

        assert "memory_1" in credits
        assert "memory_2" in credits
        assert credits["memory_1"] > credits["memory_2"]

    def test_credit_proportional_to_reward(self, trace):
        """Credit scales with reward magnitude."""
        trace.update("memory_1", activity=1.0)

        credits_low = trace.assign_credit(reward=1.0)
        credits_high = trace.assign_credit(reward=10.0)

        assert credits_high["memory_1"] == 10 * credits_low["memory_1"]

    def test_late_rewards_credit_earlier(self, trace):
        """Late rewards still credit earlier activations."""
        # Activate memory
        trace.update("memory_1", activity=1.0)

        # Wait (simulating delay)
        time.sleep(0.1)
        trace.step(dt=0.3)

        # Reward arrives later
        credits = trace.assign_credit(reward=10.0)

        # Should still get some credit (decayed but non-zero)
        assert "memory_1" in credits
        assert credits["memory_1"] > 0

    def test_trace_accumulation_pattern(self, trace):
        """Repeated activation builds strong trace."""
        # Single activation
        trace.update("single", activity=1.0)
        single_val = trace.get_trace("single")

        # Repeated activation
        for _ in range(5):
            trace.update("repeated", activity=1.0)
        repeated_val = trace.get_trace("repeated")

        # Repeated should be stronger
        assert repeated_val > single_val

    def test_get_all_active(self, trace):
        """Get all active traces above threshold."""
        trace.update("strong", activity=1.0)
        trace.update("weak", activity=0.001)

        active = trace.get_all_active(threshold=0.01)

        assert "strong" in active
        # Weak may or may not be there depending on activity

    def test_clear(self, trace):
        """Clear removes all traces."""
        for i in range(5):
            trace.update(f"memory_{i}")

        assert trace.count == 5

        trace.clear()

        assert trace.count == 0

    def test_max_trace_value(self, trace):
        """Trace value is capped."""
        # Many activations
        for _ in range(1000):
            trace.update("memory_1", activity=10.0)

        assert trace.get_trace("memory_1") <= MAX_TRACE_VALUE

    def test_capacity_eviction(self):
        """Weak traces evicted when at capacity."""
        trace = EligibilityTrace(max_traces=5)

        # Add weak traces
        for i in range(5):
            trace.update(f"memory_{i}", activity=0.1)

        assert trace.count == 5

        # Add strong trace
        trace.update("strong", activity=1.0)

        # Should still be at capacity, weakest evicted
        assert trace.count == 5
        assert "strong" in trace.traces

    def test_credit_with_decay(self, trace):
        """Negative rewards can decay traces."""
        trace.update("memory_1", activity=1.0)
        initial = trace.get_trace("memory_1")

        # Negative reward with decay
        trace.assign_credit_with_decay(reward=-1.0, apply_decay=True)

        final = trace.get_trace("memory_1")
        assert final < initial

    def test_stats(self, trace):
        """Statistics are tracked correctly."""
        trace.update("memory_1")
        trace.assign_credit(reward=5.0)

        stats = trace.get_stats()

        assert stats["count"] == 1
        assert stats["total_updates"] == 1
        assert stats["total_credits_assigned"] > 0


class TestLayeredEligibilityTrace:
    """Tests for multi-layer eligibility trace."""

    @pytest.fixture
    def layered_trace(self):
        """Create layered trace."""
        return LayeredEligibilityTrace(
            fast_tau=0.5,
            slow_tau=5.0,
            fast_weight=0.7,
            a_plus=0.1
        )

    def test_initialization(self, layered_trace):
        """Layered trace initializes correctly."""
        assert layered_trace.fast_tau == 0.5
        assert layered_trace.slow_tau == 5.0
        assert layered_trace.fast_weight == 0.7

    def test_update_both_layers(self, layered_trace):
        """Update affects both fast and slow traces."""
        layered_trace.update("memory_1")

        assert "memory_1" in layered_trace.fast_traces
        assert "memory_1" in layered_trace.slow_traces

    def test_fast_decays_faster(self, layered_trace):
        """Fast trace decays faster than slow."""
        layered_trace.update("memory_1", activity=1.0)

        fast_initial = layered_trace.fast_traces["memory_1"]
        slow_initial = layered_trace.slow_traces["memory_1"]

        # Decay
        layered_trace.step(dt=1.0)

        fast_after = layered_trace.fast_traces.get("memory_1", 0)
        slow_after = layered_trace.slow_traces.get("memory_1", 0)

        # Fast should decay more
        fast_ratio = fast_after / fast_initial if fast_initial > 0 else 0
        slow_ratio = slow_after / slow_initial if slow_initial > 0 else 0

        assert fast_ratio < slow_ratio

    def test_combined_credit(self, layered_trace):
        """Credit combines both trace layers."""
        layered_trace.update("memory_1")

        credits = layered_trace.assign_credit(reward=10.0)

        assert "memory_1" in credits
        assert credits["memory_1"] > 0

    def test_weighted_combination(self, layered_trace):
        """Credit is weighted combination of traces."""
        layered_trace.update("memory_1", activity=1.0)

        # Get individual traces
        fast_val = layered_trace.fast_traces["memory_1"]
        slow_val = layered_trace.slow_traces["memory_1"]

        # Expected combined
        expected = layered_trace.fast_weight * fast_val + layered_trace.slow_weight * slow_val

        # Actual from credit
        credits = layered_trace.assign_credit(reward=1.0)

        assert abs(credits["memory_1"] - expected) < 0.01

    def test_get_all_active_combined(self, layered_trace):
        """Get all active returns combined values."""
        layered_trace.update("memory_1")

        active = layered_trace.get_all_active(threshold=0.001)

        assert "memory_1" in active

    def test_clear_both_layers(self, layered_trace):
        """Clear removes from both layers."""
        layered_trace.update("memory_1")

        layered_trace.clear()

        assert len(layered_trace.fast_traces) == 0
        assert len(layered_trace.slow_traces) == 0

    def test_count_unique(self, layered_trace):
        """Count returns unique memories."""
        layered_trace.update("memory_1")
        layered_trace.update("memory_2")

        assert layered_trace.count == 2


class TestNoDoubleDecay:
    """LOGIC-009: Test that decay is applied once, not twice."""

    @pytest.fixture
    def trace(self):
        """Create trace with short tau for testing."""
        return EligibilityTrace(
            tau_trace=1.0,  # 1 second time constant
            a_plus=0.1,
            min_trace=0.0001
        )

    def test_step_updates_last_update(self, trace):
        """step() should update entry.last_update to prevent double decay."""
        trace.update("memory_1", activity=1.0)
        initial_last_update = trace.traces["memory_1"].last_update

        time.sleep(0.01)  # Small delay
        trace.step(dt=0.5)

        new_last_update = trace.traces["memory_1"].last_update
        # last_update should be updated by step()
        assert new_last_update > initial_last_update

    def test_no_double_decay_after_step(self, trace):
        """After step() + update(), total decay should be correct, not doubled."""
        trace.update("memory_1", activity=1.0)
        initial = trace.get_trace("memory_1")

        # Step applies decay of 0.5 seconds
        trace.step(dt=0.5)
        after_step = trace.get_trace("memory_1")

        # Now update() should NOT re-apply the 0.5s decay again
        # It should only apply decay from the time of step() to now
        time.sleep(0.01)  # Negligible time
        trace.update("memory_1", activity=0.0)  # No new activity, just check decay

        after_update = trace.get_trace("memory_1")

        # Without fix: update() would decay by exp(-0.5/1.0) again = 0.606
        # With fix: update() decays by exp(-0.01/1.0) ≈ 0.99 (negligible)

        # after_update should be very close to after_step (not halved again)
        assert after_update / after_step > 0.9, \
            f"Double decay detected: after_step={after_step}, after_update={after_update}"

    def test_correct_total_decay(self, trace):
        """Total decay over time should match expected exponential."""
        trace.update("memory_1", activity=1.0)
        initial = trace.get_trace("memory_1")

        # Apply decay in two parts: 0.3s then 0.2s
        trace.step(dt=0.3)
        trace.step(dt=0.2)
        final = trace.get_trace("memory_1")

        # Expected: exp(-0.5 / 1.0) = 0.606
        expected_ratio = np.exp(-0.5 / trace.tau_trace)
        actual_ratio = final / initial

        # Should match within 5%
        assert abs(actual_ratio - expected_ratio) < 0.05, \
            f"Expected ratio {expected_ratio:.4f}, got {actual_ratio:.4f}"


class TestEligibilityConfig:
    """Tests for eligibility configuration."""

    def test_default_config(self):
        """Default config has expected values."""
        config = EligibilityConfig()

        assert config.decay == 0.95
        assert config.tau_trace == 20.0
        assert config.a_plus == 0.005
        assert config.min_trace == 1e-4

    def test_config_creates_trace(self):
        """Config can be used to create trace."""
        config = EligibilityConfig(decay=0.9, tau_trace=10.0)

        trace = EligibilityTrace(
            decay=config.decay,
            tau_trace=config.tau_trace,
            a_plus=config.a_plus,
            a_minus=config.a_minus,
            min_trace=config.min_trace,
            max_traces=config.max_traces
        )

        assert trace.decay == 0.9
        assert trace.tau_trace == 10.0


class TestEligibilityValidation:
    """Tests for input validation in eligibility traces."""

    def test_update_memory_id_too_long(self):
        """Update rejects memory_id that exceeds max length."""
        trace = EligibilityTrace()
        long_id = "x" * 300  # Exceeds MAX_MEMORY_ID_LENGTH (256)

        with pytest.raises(ValueError, match="memory_id length"):
            trace.update(long_id)

    def test_update_memory_id_not_printable(self):
        """Update rejects non-printable memory_id."""
        trace = EligibilityTrace()

        with pytest.raises(ValueError, match="printable"):
            trace.update("mem\x00id")  # Contains null byte

    def test_update_activity_not_finite(self):
        """Update rejects non-finite activity."""
        trace = EligibilityTrace()

        with pytest.raises(ValueError, match="finite"):
            trace.update("memory_1", activity=float("inf"))

        with pytest.raises(ValueError, match="finite"):
            trace.update("memory_1", activity=float("nan"))

    def test_update_activity_negative(self):
        """Update rejects negative activity."""
        trace = EligibilityTrace()

        with pytest.raises(ValueError, match="non-negative"):
            trace.update("memory_1", activity=-0.5)

    def test_update_activity_exceeds_max(self):
        """Update rejects activity exceeding MAX_ACTIVITY."""
        trace = EligibilityTrace()

        with pytest.raises(ValueError, match="MAX_ACTIVITY"):
            trace.update("memory_1", activity=15.0)  # MAX_ACTIVITY is 10.0

    def test_step_dt_not_finite(self):
        """Step rejects non-finite dt."""
        trace = EligibilityTrace()

        with pytest.raises(ValueError, match="finite"):
            trace.step(dt=float("inf"))

        with pytest.raises(ValueError, match="finite"):
            trace.step(dt=float("nan"))

    def test_step_dt_negative(self):
        """Step rejects negative dt."""
        trace = EligibilityTrace()

        with pytest.raises(ValueError, match="non-negative"):
            trace.step(dt=-1.0)

    def test_step_dt_capped(self):
        """Step caps dt to MAX_DT."""
        trace = EligibilityTrace(tau_trace=1.0)
        trace.update("memory_1", activity=1.0)

        # Very large dt should be capped, not cause overflow
        trace.step(dt=100000.0)  # Way larger than MAX_DT (86400)

        # Should still work without error
        assert trace.count == 0  # Decayed to nothing

    def test_assign_credit_reward_not_finite(self):
        """assign_credit rejects non-finite reward."""
        trace = EligibilityTrace()

        with pytest.raises(ValueError, match="finite"):
            trace.assign_credit(float("inf"))

        with pytest.raises(ValueError, match="finite"):
            trace.assign_credit(float("nan"))

    def test_assign_credit_reward_clipped(self):
        """assign_credit clips extreme rewards to MAX_REWARD."""
        trace = EligibilityTrace()
        trace.update("memory_1", activity=1.0)

        # Large positive reward should be clipped
        credits = trace.assign_credit(10000.0)
        assert credits["memory_1"] <= trace.a_plus * 1000.0  # MAX_REWARD

        # Large negative reward should be clipped
        credits = trace.assign_credit(-10000.0)
        assert credits["memory_1"] >= -trace.a_plus * 1000.0

    def test_evict_weakest_empty(self):
        """_evict_weakest handles empty traces."""
        trace = EligibilityTrace()
        # Should not raise
        trace._evict_weakest()
        assert trace.count == 0

    def test_get_stats_with_traces(self):
        """get_stats returns statistics when traces exist."""
        trace = EligibilityTrace()
        trace.update("memory_1", activity=1.0)
        trace.update("memory_2", activity=0.5)

        stats = trace.get_stats()

        assert stats["count"] == 2
        assert "mean_trace" in stats
        assert "max_trace" in stats
        assert "min_trace" in stats
        assert stats["total_updates"] == 2


class TestLayeredEligibilityValidation:
    """Tests for input validation in layered eligibility traces."""

    def test_update_memory_id_too_long(self):
        """Layered update rejects memory_id that exceeds max length."""
        trace = LayeredEligibilityTrace()
        long_id = "x" * 300

        with pytest.raises(ValueError, match="memory_id length"):
            trace.update(long_id)

    def test_update_memory_id_not_printable(self):
        """Layered update rejects non-printable memory_id."""
        trace = LayeredEligibilityTrace()

        with pytest.raises(ValueError, match="printable"):
            trace.update("mem\x01id")

    def test_update_activity_not_finite(self):
        """Layered update rejects non-finite activity."""
        trace = LayeredEligibilityTrace()

        with pytest.raises(ValueError, match="finite"):
            trace.update("memory_1", activity=float("nan"))

    def test_update_activity_negative(self):
        """Layered update rejects negative activity."""
        trace = LayeredEligibilityTrace()

        with pytest.raises(ValueError, match="non-negative"):
            trace.update("memory_1", activity=-1.0)

    def test_update_activity_exceeds_max(self):
        """Layered update rejects activity exceeding MAX_ACTIVITY."""
        trace = LayeredEligibilityTrace()

        with pytest.raises(ValueError, match="MAX_ACTIVITY"):
            trace.update("memory_1", activity=20.0)

    def test_step_dt_not_finite(self):
        """Layered step rejects non-finite dt."""
        trace = LayeredEligibilityTrace()

        with pytest.raises(ValueError, match="finite"):
            trace.step(dt=float("inf"))

    def test_step_dt_negative(self):
        """Layered step rejects negative dt."""
        trace = LayeredEligibilityTrace()

        with pytest.raises(ValueError, match="non-negative"):
            trace.step(dt=-0.5)

    def test_assign_credit_reward_not_finite(self):
        """Layered assign_credit rejects non-finite reward."""
        trace = LayeredEligibilityTrace()

        with pytest.raises(ValueError, match="finite"):
            trace.assign_credit(float("nan"))

    def test_evict_weakest_layered_empty(self):
        """_evict_weakest_layered handles empty traces."""
        trace = LayeredEligibilityTrace()
        # Should not raise
        trace._evict_weakest_layered()
        assert trace.count == 0

    def test_evict_weakest_layered(self):
        """_evict_weakest_layered removes weakest combined trace."""
        trace = LayeredEligibilityTrace(max_traces=3)

        # Add traces with different strengths
        trace.update("strong", activity=1.0)
        trace.update("medium", activity=0.5)
        trace.update("weak", activity=0.1)

        # Manually evict
        trace._evict_weakest_layered()

        # Weak should be removed
        assert "weak" not in trace.fast_traces
        assert "weak" not in trace.slow_traces
        assert trace.count == 2

    def test_capacity_eviction_layered(self):
        """Layered trace evicts when at capacity."""
        trace = LayeredEligibilityTrace(max_traces=3)

        # Fill to capacity
        trace.update("m1", activity=0.5)
        trace.update("m2", activity=0.5)
        trace.update("m3", activity=0.5)

        assert trace.count == 3

        # Add new stronger trace - should evict one
        trace.update("strong", activity=1.0)

        # Should still be at capacity
        assert trace.count == 3
        assert "strong" in trace.fast_traces
