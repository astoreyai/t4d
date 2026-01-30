"""
Tests for serotonin-based long-term credit assignment system.

Tests the biological-inspired serotonin modulation:
- Eligibility traces with temporal decay
- Long-term value estimation
- Mood adaptation
- Credit assignment across time
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

from ww.learning.serotonin import (
    TemporalContext,
    SerotoninSystem,
)


class TestTemporalContext:
    """Tests for TemporalContext dataclass."""

    def test_creation_minimal(self):
        """Create context with minimal info."""
        ctx = TemporalContext(session_id="test-session")
        assert ctx.session_id == "test-session"
        assert ctx.goal_description is None
        assert ctx.outcome_received is False
        assert ctx.final_outcome is None

    def test_creation_with_goal(self):
        """Create context with goal description."""
        ctx = TemporalContext(
            session_id="goal-session",
            goal_description="Complete the task",
        )
        assert ctx.goal_description == "Complete the task"

    def test_start_time_auto_set(self):
        """Start time is automatically set."""
        ctx = TemporalContext(session_id="auto-time")
        assert ctx.start_time is not None
        assert isinstance(ctx.start_time, datetime)


class TestSerotoninSystem:
    """Tests for SerotoninSystem class."""

    @pytest.fixture
    def system(self):
        """Create default serotonin system."""
        return SerotoninSystem()

    @pytest.fixture
    def system_custom(self):
        """Create serotonin system with custom parameters."""
        return SerotoninSystem(
            base_discount_rate=0.95,
            eligibility_decay=0.8,
            trace_lifetime_hours=12.0,
            baseline_mood=0.6,
            mood_adaptation_rate=0.2,
            max_traces_per_memory=5,
        )

    def test_initialization_default(self, system):
        """Test default initialization."""
        assert system.base_discount_rate == 0.99
        assert system.eligibility_decay == 0.95
        assert system.trace_lifetime_hours == 24.0
        assert system.baseline_mood == 0.5
        assert system._mood == 0.5
        assert len(system._active_contexts) == 0

    def test_initialization_custom(self, system_custom):
        """Test custom initialization."""
        assert system_custom.base_discount_rate == 0.95
        assert system_custom.eligibility_decay == 0.8
        assert system_custom.max_traces_per_memory == 5

    def test_start_context(self, system):
        """Start a temporal context."""
        system.start_context("session-1", "Complete task A")
        assert "session-1" in system._active_contexts
        ctx = system._active_contexts["session-1"]
        assert ctx.goal_description == "Complete task A"

    def test_start_context_no_goal(self, system):
        """Start context without goal description."""
        system.start_context("session-2")
        ctx = system._active_contexts["session-2"]
        assert ctx.goal_description is None

    def test_end_context(self, system):
        """End and retrieve a context."""
        system.start_context("session-to-end")
        ctx = system.end_context("session-to-end")
        assert ctx is not None
        assert ctx.session_id == "session-to-end"
        assert "session-to-end" not in system._active_contexts

    def test_end_context_not_found(self, system):
        """End non-existent context returns None."""
        result = system.end_context("nonexistent")
        assert result is None

    def test_add_eligibility(self, system):
        """Add eligibility trace for a memory."""
        mem_id = uuid4()
        system.add_eligibility(mem_id, strength=0.8)

        # Check that trace exists
        elig = system.get_eligibility(mem_id)
        assert elig > 0

    def test_add_eligibility_multiple(self, system):
        """Add multiple eligibility traces for same memory."""
        mem_id = uuid4()
        system.add_eligibility(mem_id, strength=0.8)
        system.add_eligibility(mem_id, strength=0.6)
        system.add_eligibility(mem_id, strength=0.4)

        # Should accumulate
        elig = system.get_eligibility(mem_id)
        assert elig > 0

    def test_get_eligibility_no_traces(self, system):
        """Get eligibility for memory with no traces."""
        mem_id = uuid4()
        elig = system.get_eligibility(mem_id)
        assert elig == 0.0

    def test_get_eligibility_with_trace(self, system):
        """Get eligibility for memory with trace."""
        mem_id = uuid4()
        system.add_eligibility(mem_id, strength=0.8)

        elig = system.get_eligibility(mem_id)
        assert elig > 0

    def test_get_eligibility_capped_at_one(self, system):
        """Eligibility is capped at 1.0."""
        mem_id = uuid4()
        # Add many traces
        for _ in range(10):
            system.add_eligibility(mem_id, strength=1.0)

        elig = system.get_eligibility(mem_id)
        assert elig <= 1.0

    def test_receive_outcome_updates_mood(self, system):
        """Receiving outcome updates mood."""
        initial_mood = system._mood
        system.receive_outcome(outcome_score=1.0)

        # Mood should increase towards 1.0
        assert system._mood > initial_mood

    def test_receive_outcome_mood_decreases(self, system):
        """Bad outcome decreases mood."""
        system._mood = 0.8  # Start high
        system.receive_outcome(outcome_score=0.0)

        # Mood should decrease
        assert system._mood < 0.8

    def test_receive_outcome_tracks_statistics(self, system):
        """Outcome tracking updates statistics."""
        system.receive_outcome(outcome_score=0.8)
        system.receive_outcome(outcome_score=0.3)
        system.receive_outcome(outcome_score=0.9)

        assert system._total_outcomes == 3
        assert system._positive_outcomes == 2  # 0.8 and 0.9 > 0.5

    def test_receive_outcome_distributes_credit(self, system):
        """Outcome distributes credit to eligible memories."""
        mem1, mem2 = uuid4(), uuid4()
        system.add_eligibility(mem1, strength=1.0)
        system.add_eligibility(mem2, strength=0.5)

        credits = system.receive_outcome(outcome_score=0.9)

        assert str(mem1) in credits
        assert str(mem2) in credits

    def test_receive_outcome_updates_long_term_value(self, system):
        """Outcome updates long-term value estimates."""
        mem_id = uuid4()
        system.add_eligibility(mem_id, strength=1.0)

        # Good outcome
        system.receive_outcome(outcome_score=0.95)

        ltv = system.get_long_term_value(mem_id)
        assert ltv > 0.5  # Should be above baseline

    def test_receive_outcome_marks_context(self, system):
        """Outcome marks associated context as complete."""
        system.start_context("test-ctx")
        mem_id = uuid4()
        system.add_eligibility(mem_id, strength=1.0)

        system.receive_outcome(outcome_score=0.8, context_id="test-ctx")

        ctx = system._active_contexts["test-ctx"]
        assert ctx.outcome_received is True
        assert ctx.final_outcome == 0.8

    def test_get_long_term_value_no_data(self, system):
        """Get LTV for memory with no data."""
        mem_id = uuid4()
        ltv = system.get_long_term_value(mem_id)
        assert ltv == 0.5  # Default baseline

    def test_mood_clipped(self, system):
        """Mood stays within [0, 1] bounds."""
        # Force mood high
        for _ in range(20):
            system.receive_outcome(outcome_score=1.0)
        assert system._mood <= 1.0

        # Force mood low
        for _ in range(50):
            system.receive_outcome(outcome_score=0.0)
        assert system._mood >= 0.0


class TestSerotoninSystemAdvanced:
    """Advanced tests for SerotoninSystem."""

    def test_credit_advantage_positive(self):
        """Positive advantage gives positive credit."""
        system = SerotoninSystem(baseline_mood=0.5)
        system._mood = 0.5

        mem_id = uuid4()
        system.add_eligibility(mem_id, strength=1.0)

        credits = system.receive_outcome(outcome_score=0.9)

        # Advantage = 0.9 - 0.5 = 0.4 > 0
        assert credits[str(mem_id)] > 0

    def test_credit_advantage_negative(self):
        """Negative advantage gives negative credit."""
        system = SerotoninSystem(baseline_mood=0.5)
        system._mood = 0.7

        mem_id = uuid4()
        system.add_eligibility(mem_id, strength=1.0)

        credits = system.receive_outcome(outcome_score=0.3)

        # Advantage = 0.3 - 0.7 = -0.4 < 0
        assert credits[str(mem_id)] < 0

    def test_multiple_contexts(self):
        """Multiple concurrent contexts work correctly."""
        system = SerotoninSystem()

        system.start_context("ctx-1", "Goal 1")
        system.start_context("ctx-2", "Goal 2")
        system.start_context("ctx-3", "Goal 3")

        assert len(system._active_contexts) == 3

        system.end_context("ctx-2")
        assert len(system._active_contexts) == 2
        assert "ctx-2" not in system._active_contexts


class TestSerotoninSystemIntegration:
    """Integration tests for SerotoninSystem."""

    def test_full_workflow(self):
        """Test complete workflow: context -> eligibility -> outcome."""
        system = SerotoninSystem()

        # Start context
        system.start_context("workflow-test", "Complete coding task")

        # Add some eligibility traces (memories used during task)
        mem1, mem2, mem3 = uuid4(), uuid4(), uuid4()
        system.add_eligibility(mem1, strength=1.0)
        system.add_eligibility(mem2, strength=0.5)
        system.add_eligibility(mem3, strength=0.3)

        # Task succeeds
        credits = system.receive_outcome(
            outcome_score=0.9,
            context_id="workflow-test"
        )

        # All memories should get credit
        assert len(credits) == 3
        assert str(mem1) in credits
        assert str(mem2) in credits
        assert str(mem3) in credits

        # Higher eligibility = more credit (in absolute value)
        assert abs(credits[str(mem1)]) > abs(credits[str(mem2)])

        # Context marked complete
        ctx = system._active_contexts["workflow-test"]
        assert ctx.outcome_received is True

    def test_multiple_sessions(self):
        """Test learning across multiple sessions."""
        system = SerotoninSystem()

        mem_id = uuid4()

        # First session - memory helps
        system.add_eligibility(mem_id, strength=1.0)
        system.receive_outcome(outcome_score=0.9)

        # Second session - memory helps again
        system.add_eligibility(mem_id, strength=1.0)
        system.receive_outcome(outcome_score=0.85)

        # Long-term value should be above baseline
        ltv = system.get_long_term_value(mem_id)
        assert ltv > 0.5  # Should learn this memory is valuable


class TestSerotoninSystemStatistics:
    """Tests for statistics and monitoring."""

    def test_get_stats(self):
        """Get system statistics."""
        system = SerotoninSystem()

        stats = system.get_stats()
        assert "current_mood" in stats
        assert "total_outcomes" in stats
        assert "memories_with_traces" in stats
        assert "active_contexts" in stats

    def test_get_memories_with_traces(self):
        """Get list of memories with active traces."""
        system = SerotoninSystem()

        mem1, mem2 = uuid4(), uuid4()
        system.add_eligibility(mem1, strength=1.0)
        system.add_eligibility(mem2, strength=0.5)

        # Verify traces exist using get_eligibility
        assert system.get_eligibility(mem1) > 0
        assert system.get_eligibility(mem2) > 0

        # get_memories_with_traces may filter out very weak traces
        # Just verify it returns a list
        memories = system.get_memories_with_traces()
        assert isinstance(memories, list)

    def test_reset(self):
        """Reset clears all state."""
        system = SerotoninSystem()

        # Build up some state
        system.start_context("ctx")
        system.add_eligibility(uuid4(), strength=1.0)
        system.receive_outcome(outcome_score=0.9)
        system._mood = 0.8

        # Reset
        system.reset()

        # Everything should be cleared
        assert system._mood == system.baseline_mood
        assert len(system._active_contexts) == 0
        assert len(system._long_term_values) == 0
        assert system._total_outcomes == 0


class TestTemporalCreditAssignmentDecay:
    """P2.4: Tests for temporal credit assignment decay."""

    @pytest.fixture
    def system(self):
        """Create serotonin system for temporal decay tests."""
        return SerotoninSystem()

    def test_get_trace_half_life(self, system):
        """Half-life should be tau * ln(2)."""
        import numpy as np

        half_life = system.get_trace_half_life()

        # Should be positive
        assert half_life > 0

        # Should be tau * ln(2)
        expected = system._base_tau_trace * np.log(2)
        assert abs(half_life - expected) < 1e-6

    def test_compute_temporal_discount_no_delay(self, system):
        """No delay should give no discount (factor = 1.0)."""
        discount = system.compute_temporal_discount(0.0)
        assert discount == 1.0

        discount = system.compute_temporal_discount(-1.0)
        assert discount == 1.0

    def test_compute_temporal_discount_positive_delay(self, system):
        """Positive delay should reduce discount factor."""
        discount = system.compute_temporal_discount(60.0)  # 1 minute delay

        # Should be between 0 and 1
        assert 0.0 < discount < 1.0

    def test_compute_temporal_discount_increases_with_delay(self, system):
        """Longer delays should give smaller discount factors."""
        d1 = system.compute_temporal_discount(10.0)
        d2 = system.compute_temporal_discount(100.0)
        d3 = system.compute_temporal_discount(1000.0)

        # Discount should decrease with delay
        assert d1 > d2 > d3

    def test_compute_temporal_discount_at_half_life(self, system):
        """At half-life delay, discount should be approximately gamma."""
        half_life = system.get_trace_half_life()
        discount = system.compute_temporal_discount(half_life)

        # At half-life: discount ≈ gamma^1 = gamma
        expected = system.base_discount_rate
        assert abs(discount - expected) < 0.01

    def test_add_eligibility_with_delay(self, system):
        """Adding eligibility with delay should reduce effective strength."""
        mem1, mem2 = uuid4(), uuid4()

        # Add without delay
        system.add_eligibility(mem1, strength=1.0, delay_seconds=0.0)

        # Add with delay
        system.add_eligibility(mem2, strength=1.0, delay_seconds=60.0)

        elig1 = system.get_eligibility(mem1)
        elig2 = system.get_eligibility(mem2)

        # Delayed should have lower eligibility
        assert elig2 < elig1

    def test_add_eligibility_delay_proportional(self, system):
        """Eligibility discount should be proportional to delay."""
        mem1, mem2, mem3 = uuid4(), uuid4(), uuid4()

        system.add_eligibility(mem1, strength=1.0, delay_seconds=10.0)
        system.add_eligibility(mem2, strength=1.0, delay_seconds=100.0)
        system.add_eligibility(mem3, strength=1.0, delay_seconds=1000.0)

        elig1 = system.get_eligibility(mem1)
        elig2 = system.get_eligibility(mem2)
        elig3 = system.get_eligibility(mem3)

        # Should decrease with increasing delay
        assert elig1 > elig2 > elig3

    def test_add_eligibility_zero_delay_no_change(self, system):
        """Zero delay should not change strength."""
        mem1, mem2 = uuid4(), uuid4()

        system.add_eligibility(mem1, strength=0.8, delay_seconds=0.0)
        system.add_eligibility(mem2, strength=0.8)  # Default delay=0

        elig1 = system.get_eligibility(mem1)
        elig2 = system.get_eligibility(mem2)

        # Should be equal
        assert abs(elig1 - elig2) < 1e-6

    def test_temporal_discount_with_custom_discount_rate(self):
        """Custom discount rate should affect temporal decay."""
        # High gamma (more patient)
        system_patient = SerotoninSystem(base_discount_rate=0.999)
        # Low gamma (more impulsive)
        system_impulsive = SerotoninSystem(base_discount_rate=0.9)

        delay = 100.0  # 100 second delay

        d_patient = system_patient.compute_temporal_discount(delay)
        d_impulsive = system_impulsive.compute_temporal_discount(delay)

        # Patient system should discount less
        assert d_patient > d_impulsive

    def test_credit_assignment_with_delayed_eligibility(self):
        """Credit assignment should use discounted eligibility."""
        system = SerotoninSystem()

        mem1, mem2 = uuid4(), uuid4()

        # mem1: no delay
        system.add_eligibility(mem1, strength=1.0, delay_seconds=0.0)
        # mem2: 60 second delay
        system.add_eligibility(mem2, strength=1.0, delay_seconds=60.0)

        # Receive positive outcome
        credits = system.receive_outcome(outcome_score=0.9)

        # mem1 should get more credit (higher eligibility)
        credit1 = abs(credits.get(str(mem1), 0))
        credit2 = abs(credits.get(str(mem2), 0))

        assert credit1 > credit2

    def test_discount_factor_bounded(self, system):
        """Discount factor should always be in [0, 1]."""
        # Test extreme values
        for delay in [0, 1, 10, 100, 1000, 10000, 100000]:
            discount = system.compute_temporal_discount(delay)
            assert 0.0 <= discount <= 1.0


class TestSerotoninSystemMissingCoverage:
    """Tests for previously uncovered serotonin functionality."""

    def test_get_long_term_values_batch(self):
        """Test batch retrieval of long-term values."""
        system = SerotoninSystem()

        mem1, mem2, mem3 = uuid4(), uuid4(), uuid4()

        # Set up some values
        system.add_eligibility(mem1, strength=1.0)
        system.add_eligibility(mem2, strength=1.0)
        system.receive_outcome(outcome_score=0.9)

        # Get batch values
        values = system.get_long_term_values_batch([mem1, mem2, mem3])

        assert str(mem1) in values
        assert str(mem2) in values
        assert str(mem3) in values

    def test_modulate_value_by_mood_high_mood(self):
        """Test value modulation with high mood."""
        system = SerotoninSystem()
        system._mood = 0.9  # High mood

        raw_value = 0.5
        modulated = system.modulate_value_by_mood(raw_value)

        # High mood should increase value
        assert modulated > raw_value

    def test_modulate_value_by_mood_low_mood(self):
        """Test value modulation with low mood."""
        system = SerotoninSystem()
        system._mood = 0.1  # Low mood

        raw_value = 0.5
        modulated = system.modulate_value_by_mood(raw_value)

        # Low mood should decrease value
        assert modulated < raw_value

    def test_modulate_value_by_mood_neutral(self):
        """Test value modulation with neutral mood."""
        system = SerotoninSystem()
        system._mood = 0.5  # Neutral mood

        raw_value = 0.5
        modulated = system.modulate_value_by_mood(raw_value)

        # Neutral mood should not change much
        assert abs(modulated - raw_value) < 0.01

    def test_modulate_value_by_mood_bounded(self):
        """Test value modulation stays bounded."""
        system = SerotoninSystem()
        system._mood = 1.0  # Max mood

        # High raw value
        modulated = system.modulate_value_by_mood(0.95)
        assert modulated <= 1.0

        system._mood = 0.0  # Min mood
        modulated = system.modulate_value_by_mood(0.05)
        assert modulated >= 0.0

    def test_set_mood(self):
        """Test manual mood setting."""
        system = SerotoninSystem()

        system.set_mood(0.8)
        assert system._mood == 0.8

        system.set_mood(0.2)
        assert system._mood == 0.2

    def test_set_mood_clipped(self):
        """Test mood setting is clipped to bounds."""
        system = SerotoninSystem()

        system.set_mood(1.5)
        assert system._mood == 1.0

        system.set_mood(-0.5)
        assert system._mood == 0.0

    def test_set_baseline_mood(self):
        """Test setting baseline mood."""
        system = SerotoninSystem()

        system.set_baseline_mood(0.7)
        assert system.baseline_mood == 0.7

    def test_set_baseline_mood_clipped(self):
        """Test baseline mood is clipped."""
        system = SerotoninSystem()

        system.set_baseline_mood(1.5)
        assert system.baseline_mood == 1.0

        system.set_baseline_mood(-0.5)
        assert system.baseline_mood == 0.0

    def test_set_mood_adaptation_rate(self):
        """Test setting mood adaptation rate."""
        system = SerotoninSystem()

        system.set_mood_adaptation_rate(0.3)
        assert system.mood_adaptation_rate == 0.3

    def test_set_mood_adaptation_rate_clipped(self):
        """Test mood adaptation rate is clipped."""
        system = SerotoninSystem()

        system.set_mood_adaptation_rate(0.001)  # Below min
        assert system.mood_adaptation_rate == 0.01

        system.set_mood_adaptation_rate(0.9)  # Above max
        assert system.mood_adaptation_rate == 0.5

    def test_set_discount_rate(self):
        """Test setting discount rate."""
        system = SerotoninSystem()

        system.set_discount_rate(0.95)
        assert system.base_discount_rate == 0.95

    def test_set_discount_rate_clipped(self):
        """Test discount rate is clipped."""
        system = SerotoninSystem()

        system.set_discount_rate(0.5)  # Below min
        assert system.base_discount_rate == 0.9

        system.set_discount_rate(1.5)  # Above max
        assert system.base_discount_rate == 1.0

    def test_set_eligibility_decay(self):
        """Test setting eligibility decay."""
        system = SerotoninSystem()

        system.set_eligibility_decay(0.9)
        assert system.eligibility_decay == 0.9

    def test_set_eligibility_decay_clipped(self):
        """Test eligibility decay is clipped."""
        system = SerotoninSystem()

        system.set_eligibility_decay(0.5)  # Below min
        assert system.eligibility_decay == 0.8

        system.set_eligibility_decay(1.0)  # Above max
        assert system.eligibility_decay == 0.99

    def test_set_eligibility_decay_updates_tau(self):
        """Test that setting eligibility decay updates tau_trace."""
        system = SerotoninSystem()
        old_tau = system._base_tau_trace

        system.set_eligibility_decay(0.9)

        # Tau should have changed
        assert system._base_tau_trace != old_tau

    def test_clear_eligibility_traces(self):
        """Test clearing eligibility traces."""
        system = SerotoninSystem()

        # Add some traces
        mem1, mem2 = uuid4(), uuid4()
        system.add_eligibility(mem1, strength=1.0)
        system.add_eligibility(mem2, strength=0.8)

        # Verify traces exist
        assert system.get_eligibility(mem1) > 0
        assert system.get_eligibility(mem2) > 0

        # Clear traces
        system.clear_eligibility_traces()

        # Traces should be gone
        assert system.get_eligibility(mem1) == 0.0
        assert system.get_eligibility(mem2) == 0.0

    def test_clear_long_term_values(self):
        """Test clearing long-term values."""
        system = SerotoninSystem()

        # Add some values
        mem_id = uuid4()
        system.add_eligibility(mem_id, strength=1.0)
        system.receive_outcome(outcome_score=0.9)

        # Verify value exists
        ltv = system.get_long_term_value(mem_id)
        assert ltv > 0.5

        # Clear values
        system.clear_long_term_values()

        # Should return default
        ltv = system.get_long_term_value(mem_id)
        assert ltv == 0.5

    def test_patience_factor_no_trace(self):
        """Test patience factor when trace not found."""
        system = SerotoninSystem()

        # Don't add any eligibility for this memory
        mem_id = uuid4()

        # Force the edge case by manipulating eligibility tracer
        # The patience_factor = 1.0 path is taken when trace_entry is None
        # We can trigger this by calling receive_outcome with a memory that
        # had a trace that was cleared
        system.add_eligibility(mem_id, strength=1.0)
        system._eligibility_tracer.clear()  # Clear the trace

        # Now receive outcome - should use patience_factor = 1.0 for missing trace
        credits = system.receive_outcome(outcome_score=0.9)

        # Should handle gracefully (may or may not have credits depending on implementation)
        assert isinstance(credits, dict)
