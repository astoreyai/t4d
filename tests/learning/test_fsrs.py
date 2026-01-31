"""
Unit tests for FSRS (Free Spaced Repetition Scheduler).
"""

import pytest
import math
from datetime import datetime, timedelta

from t4dm.learning.fsrs import (
    Rating,
    FSRSParameters,
    MemoryState,
    SchedulingInfo,
    FSRS,
    FSRSMemoryTracker,
    create_fsrs,
)


class TestRating:
    """Tests for Rating enum."""

    def test_rating_values(self):
        """Rating values match FSRS convention."""
        assert Rating.AGAIN == 1
        assert Rating.HARD == 2
        assert Rating.GOOD == 3
        assert Rating.EASY == 4

    def test_rating_ordering(self):
        """Ratings are ordered by quality."""
        assert Rating.AGAIN < Rating.HARD < Rating.GOOD < Rating.EASY


class TestFSRSParameters:
    """Tests for FSRS parameters."""

    def test_default_parameters(self):
        """Default parameters are set."""
        params = FSRSParameters()
        assert params.w0 > 0  # Initial stability for Again
        assert params.w3 > params.w0  # Easy > Again stability
        assert params.request_retention == 0.9
        assert params.maximum_interval > 0

    def test_initial_stability(self):
        """Initial stability varies by rating."""
        params = FSRSParameters()

        s_again = params.initial_stability(Rating.AGAIN)
        s_hard = params.initial_stability(Rating.HARD)
        s_good = params.initial_stability(Rating.GOOD)
        s_easy = params.initial_stability(Rating.EASY)

        # Higher ratings = higher initial stability
        assert s_again < s_hard < s_good < s_easy


class TestMemoryState:
    """Tests for MemoryState dataclass."""

    def test_default_state(self):
        """New state has correct defaults."""
        state = MemoryState()
        assert state.stability == 0.0
        assert state.difficulty == 5.0
        assert state.last_review is None
        assert state.reps == 0
        assert state.lapses == 0

    def test_is_new(self):
        """is_new() correctly identifies new items."""
        new_state = MemoryState()
        assert new_state.is_new()

        reviewed_state = MemoryState(reps=1)
        assert not reviewed_state.is_new()


class TestFSRS:
    """Tests for FSRS scheduler."""

    @pytest.fixture
    def fsrs(self):
        """Create default FSRS scheduler."""
        return FSRS()

    def test_initialization(self, fsrs):
        """FSRS initializes with parameters."""
        assert fsrs.params is not None
        assert fsrs.params.request_retention == 0.9

    def test_retrievability_new_item(self, fsrs):
        """New items have zero retrievability."""
        state = MemoryState()
        R = fsrs.retrievability(state)
        assert R == 0.0

    def test_retrievability_just_reviewed(self, fsrs):
        """Just-reviewed items have 100% retrievability."""
        state = MemoryState(stability=10.0, elapsed_days=0)
        R = fsrs.retrievability(state)
        assert R == 1.0

    def test_retrievability_decay(self, fsrs):
        """Retrievability decays over time."""
        state = MemoryState(stability=10.0)

        R_0 = fsrs.retrievability(state, elapsed_days=0)
        R_5 = fsrs.retrievability(state, elapsed_days=5)
        R_10 = fsrs.retrievability(state, elapsed_days=10)

        assert R_0 > R_5 > R_10
        # At t=S (10 days), R should be ~0.9
        assert abs(R_10 - 0.9) < 0.01

    def test_retrievability_formula(self, fsrs):
        """Retrievability follows R = 0.9^(t/S)."""
        S = 10.0
        t = 5.0
        state = MemoryState(stability=S)

        R = fsrs.retrievability(state, elapsed_days=t)
        expected = math.pow(0.9, t / S)

        assert abs(R - expected) < 0.001

    def test_next_interval_formula(self, fsrs):
        """Interval follows I = S * ln(R) / ln(0.9)."""
        S = 10.0
        R = 0.9  # Default retention

        interval = fsrs.next_interval(S)
        # For R=0.9: I = S * ln(0.9) / ln(0.9) = S
        assert abs(interval - S) < 0.1

    def test_next_interval_higher_retention(self):
        """Higher retention = shorter intervals."""
        fsrs_90 = FSRS(FSRSParameters(request_retention=0.9))
        fsrs_95 = FSRS(FSRSParameters(request_retention=0.95))

        i_90 = fsrs_90.next_interval(10.0)
        i_95 = fsrs_95.next_interval(10.0)

        # 95% retention needs more frequent reviews
        assert i_95 < i_90

    def test_next_interval_bounds(self, fsrs):
        """Interval is bounded."""
        # Very small stability
        i_small = fsrs.next_interval(0.001)
        assert i_small >= 1.0

        # Very large stability
        i_large = fsrs.next_interval(1000000)
        assert i_large <= fsrs.params.maximum_interval

    def test_first_review_good(self, fsrs):
        """First review with Good rating."""
        state = MemoryState()
        new_state, info = fsrs.review(state, Rating.GOOD)

        assert new_state.stability > 0
        assert 1 <= new_state.difficulty <= 10
        assert new_state.reps == 1
        assert new_state.lapses == 0
        assert info.interval > 0

    def test_first_review_again(self, fsrs):
        """First review with Again rating."""
        state = MemoryState()
        new_state, info = fsrs.review(state, Rating.AGAIN)

        assert new_state.stability > 0
        assert new_state.reps == 1
        assert new_state.lapses == 1  # First failure

    def test_first_review_ratings_differ(self, fsrs):
        """Different first ratings produce different results."""
        state = MemoryState()

        _, info_again = fsrs.review(state, Rating.AGAIN)
        _, info_good = fsrs.review(state, Rating.GOOD)
        _, info_easy = fsrs.review(state, Rating.EASY)

        # Better ratings = longer intervals
        assert info_again.interval < info_good.interval < info_easy.interval

    def test_stability_increases_on_success(self, fsrs):
        """Stability increases after successful reviews."""
        state = MemoryState()

        # First review
        state, _ = fsrs.review(state, Rating.GOOD)
        s1 = state.stability

        # Simulate time passing
        state.elapsed_days = 3.0

        # Second review
        state, _ = fsrs.review(state, Rating.GOOD)
        s2 = state.stability

        assert s2 > s1

    def test_stability_decreases_on_lapse(self, fsrs):
        """Stability decreases after forgetting."""
        state = MemoryState()

        # Build up stability
        state, _ = fsrs.review(state, Rating.GOOD)
        state.elapsed_days = 3.0
        state, _ = fsrs.review(state, Rating.GOOD)
        s_before = state.stability

        # Lapse
        state.elapsed_days = 10.0
        state, _ = fsrs.review(state, Rating.AGAIN)
        s_after = state.stability

        assert s_after < s_before
        assert state.lapses == 1

    def test_difficulty_adjusts(self, fsrs):
        """Difficulty adjusts based on ratings over multiple reviews."""
        # Test that Hard increases difficulty relative to Easy
        state_hard = MemoryState()
        state_easy = MemoryState()

        # Both start with Good
        state_hard, _ = fsrs.review(state_hard, Rating.GOOD)
        state_easy, _ = fsrs.review(state_easy, Rating.GOOD)

        # Simulate multiple reviews with different ratings
        for _ in range(5):
            state_hard.elapsed_days = 3.0
            state_hard, _ = fsrs.review(state_hard, Rating.HARD)

            state_easy.elapsed_days = 3.0
            state_easy, _ = fsrs.review(state_easy, Rating.EASY)

        # Hard path should have higher difficulty than Easy path
        assert state_hard.difficulty > state_easy.difficulty

    def test_preview_ratings(self, fsrs):
        """Preview shows all rating options."""
        state = MemoryState()
        state, _ = fsrs.review(state, Rating.GOOD)

        state.elapsed_days = 3.0
        preview = fsrs.preview_ratings(state)

        assert Rating.AGAIN in preview
        assert Rating.HARD in preview
        assert Rating.GOOD in preview
        assert Rating.EASY in preview

        # Intervals should increase with rating
        assert preview[Rating.AGAIN].interval < preview[Rating.GOOD].interval

    def test_get_stats(self, fsrs):
        """Stats include parameters."""
        stats = fsrs.get_stats()

        assert "request_retention" in stats
        assert "maximum_interval" in stats
        assert "parameters" in stats
        assert "w0" in stats["parameters"]


class TestFSRSMemoryTracker:
    """Tests for memory tracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker."""
        return FSRSMemoryTracker()

    def test_get_state_creates_new(self, tracker):
        """get_state creates new state if missing."""
        state = tracker.get_state("memory_1")
        assert state.is_new()

    def test_review_updates_state(self, tracker):
        """Review updates memory state."""
        info = tracker.review("memory_1", Rating.GOOD)

        state = tracker.get_state("memory_1")
        assert state.reps == 1
        assert state.stability > 0
        assert info.interval > 0

    def test_review_history(self, tracker):
        """Reviews are tracked in history."""
        tracker.review("memory_1", Rating.GOOD)
        tracker.review("memory_1", Rating.GOOD)

        assert "memory_1" in tracker._history
        assert len(tracker._history["memory_1"]) == 2

    def test_get_due_items(self, tracker):
        """Get items due for review."""
        now = datetime.now()

        # Review an item
        tracker.review("memory_1", Rating.GOOD, now - timedelta(days=10))

        # Should be due now
        due = tracker.get_due_items(now)
        memory_ids = [item[0] for item in due]
        assert "memory_1" in memory_ids

    def test_get_retrievability(self, tracker):
        """Get current retrievability."""
        now = datetime.now()
        tracker.review("memory_1", Rating.GOOD, now)

        # Just reviewed - should be ~1.0
        R = tracker.get_retrievability("memory_1", now)
        assert R > 0.99

    def test_get_stats(self, tracker):
        """Stats reflect tracked items."""
        tracker.review("memory_1", Rating.GOOD)
        tracker.review("memory_2", Rating.AGAIN)

        stats = tracker.get_stats()

        assert stats["total_items"] == 2
        assert stats["total_reviews"] == 2
        assert stats["items_with_lapses"] == 1


class TestCreateFSRS:
    """Tests for convenience function."""

    def test_create_with_defaults(self):
        """Create with default settings."""
        fsrs = create_fsrs()
        assert fsrs.params.request_retention == 0.9
        assert fsrs.params.maximum_interval == 365

    def test_create_with_custom(self):
        """Create with custom settings."""
        fsrs = create_fsrs(request_retention=0.85, maximum_interval=180)
        assert fsrs.params.request_retention == 0.85
        assert fsrs.params.maximum_interval == 180


class TestFSRSIntegration:
    """Integration tests for realistic scenarios."""

    def test_learning_progression(self):
        """Test realistic learning progression."""
        fsrs = FSRS()
        state = MemoryState()
        now = datetime.now()

        intervals = []
        for day_offset in [0, 1, 3, 7, 14, 30]:
            review_time = now + timedelta(days=day_offset)
            state, info = fsrs.review(state, Rating.GOOD, review_time)
            intervals.append(info.interval)

        # Intervals should generally increase
        assert intervals[-1] > intervals[0]

    def test_mixed_ratings(self):
        """Test mixed rating scenario."""
        fsrs = FSRS()
        state = MemoryState()
        now = datetime.now()

        # Good, Good, Again, Hard, Good, Easy
        ratings = [Rating.GOOD, Rating.GOOD, Rating.AGAIN, Rating.HARD, Rating.GOOD, Rating.EASY]
        review_time = now

        for rating in ratings:
            state, info = fsrs.review(state, rating, review_time)
            review_time = info.due_date

        # Should still have reasonable state
        assert state.stability > 0
        assert 1 <= state.difficulty <= 10
        assert state.lapses == 1

    def test_long_absence(self):
        """Test handling long absence."""
        fsrs = FSRS()
        state = MemoryState()
        now = datetime.now()

        # Initial learning
        state, _ = fsrs.review(state, Rating.GOOD, now)

        # Long absence (100 days)
        review_after_absence = now + timedelta(days=100)
        R_before = fsrs.retrievability(state, 100)

        # Retrievability should be very low
        assert R_before < 0.5

        # But still possible to review
        state, info = fsrs.review(state, Rating.HARD, review_after_absence)
        assert state.stability > 0
