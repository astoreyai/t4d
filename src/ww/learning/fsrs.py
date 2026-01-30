"""
Free Spaced Repetition Scheduler (FSRS) for World Weaver.

FSRS is a modern spaced repetition algorithm that models memory
using two key variables:
- Stability (S): Time in days for retention to fall to 90%
- Difficulty (D): Inherent difficulty of the item (0-10 scale)

This implementation follows FSRS-4.5 with optimized parameters.

References:
- https://github.com/open-spaced-repetition/fsrs4anki
- https://github.com/open-spaced-repetition/py-fsrs
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum

logger = logging.getLogger(__name__)


class Rating(IntEnum):
    """Review rating (matches Anki/FSRS convention)."""
    AGAIN = 1  # Complete failure, reset
    HARD = 2   # Recalled with difficulty
    GOOD = 3   # Recalled correctly
    EASY = 4   # Recalled effortlessly


@dataclass
class FSRSParameters:
    """
    FSRS-4.5 parameters.

    Default values are from the FSRS-4.5 optimizer trained on
    large Anki datasets. These can be personalized per-user.
    """
    # w0-w3: Initial stability for each rating
    w0: float = 0.4072  # Again
    w1: float = 1.1829  # Hard
    w2: float = 3.1262  # Good
    w3: float = 15.4722  # Easy

    # w4-w5: Initial difficulty
    w4: float = 7.2102  # Base difficulty
    w5: float = 0.5316  # Rating adjustment

    # w6: Difficulty mean reversion
    w6: float = 1.0198

    # w7-w8: Difficulty update
    w7: float = 0.0028  # Base
    w8: float = 1.5489  # Rating factor

    # w9-w11: Stability increase on success
    w9: float = 0.1443   # exp factor
    w10: float = 0.0951  # Stability power
    w11: float = 2.7537  # Retrievability factor

    # w12-w15: Stability after failure (lapse)
    w12: float = 0.0967  # Base
    w13: float = 0.3550  # Difficulty power
    w14: float = 0.2261  # Stability power
    w15: float = 2.8898  # Retrievability factor

    # w16: Hard penalty
    w16: float = 0.5100

    # Target retention (default 90%)
    request_retention: float = 0.9

    # Maximum interval in days
    maximum_interval: float = 36500  # 100 years

    def initial_stability(self, rating: Rating) -> float:
        """Get initial stability for a rating."""
        return [self.w0, self.w1, self.w2, self.w3][rating - 1]


@dataclass
class MemoryState:
    """
    Memory state for a single item.

    Attributes:
        stability: Time in days for R to fall to 90%
        difficulty: Item difficulty (1-10 scale)
        last_review: When the item was last reviewed
        elapsed_days: Days since last review
        reps: Total review count
        lapses: Number of times forgotten (rated Again)
    """
    stability: float = 0.0
    difficulty: float = 5.0
    last_review: datetime | None = None
    elapsed_days: float = 0.0
    reps: int = 0
    lapses: int = 0

    def is_new(self) -> bool:
        """Check if this is a new (unreviewed) item."""
        return self.reps == 0


@dataclass
class SchedulingInfo:
    """Scheduling information for next review."""
    memory_state: MemoryState
    interval: float  # Days until next review
    due_date: datetime
    retrievability: float  # Current probability of recall


class FSRS:
    """
    Free Spaced Repetition Scheduler.

    Implements the FSRS-4.5 algorithm for optimal spaced repetition.

    Usage:
        fsrs = FSRS()

        # First review
        state = MemoryState()
        state, info = fsrs.review(state, Rating.GOOD)

        # Subsequent reviews
        state, info = fsrs.review(state, Rating.GOOD)
        logger.info(f"Next review in {info.interval:.1f} days")
    """

    def __init__(self, params: FSRSParameters | None = None):
        """
        Initialize FSRS scheduler.

        Args:
            params: FSRS parameters (uses defaults if None)
        """
        self.params = params or FSRSParameters()

    def retrievability(self, state: MemoryState, elapsed_days: float | None = None) -> float:
        """
        Calculate current retrievability (probability of recall).

        R(t) = exp(ln(0.9) * t / S)

        Args:
            state: Current memory state
            elapsed_days: Days since last review (uses state.elapsed_days if None)

        Returns:
            Probability of recall [0, 1]
        """
        if state.stability <= 0:
            return 0.0

        t = elapsed_days if elapsed_days is not None else state.elapsed_days
        if t <= 0:
            return 1.0

        # R = 0.9^(t/S) = exp(ln(0.9) * t / S)
        return math.exp(math.log(0.9) * t / state.stability)

    def next_interval(self, stability: float, request_retention: float | None = None) -> float:
        """
        Calculate optimal interval for target retention.

        I = S * ln(R) / ln(0.9)

        Args:
            stability: Current stability in days
            request_retention: Target retention (uses params default if None)

        Returns:
            Optimal interval in days
        """
        r = request_retention or self.params.request_retention

        if stability <= 0 or r <= 0 or r >= 1:
            return 1.0

        # I = S * ln(R) / ln(0.9)
        interval = stability * math.log(r) / math.log(0.9)

        # Clamp to [1, maximum_interval]
        return max(1.0, min(interval, self.params.maximum_interval))

    def review(
        self,
        state: MemoryState,
        rating: Rating,
        review_time: datetime | None = None
    ) -> tuple[MemoryState, SchedulingInfo]:
        """
        Process a review and update memory state.

        Args:
            state: Current memory state
            rating: User's rating of recall quality
            review_time: When the review occurred (default: now)

        Returns:
            Tuple of (new_state, scheduling_info)
        """
        now = review_time or datetime.now()

        # Calculate elapsed time
        if state.last_review:
            elapsed = (now - state.last_review).total_seconds() / 86400  # days
        else:
            elapsed = 0.0

        # Create new state (don't mutate input)
        new_state = MemoryState(
            stability=state.stability,
            difficulty=state.difficulty,
            last_review=now,
            elapsed_days=elapsed,
            reps=state.reps + 1,
            lapses=state.lapses
        )

        if state.is_new():
            # First review - initialize
            new_state = self._init_state(new_state, rating)
        else:
            # Update existing state
            new_state = self._update_state(new_state, state, rating)

        # Calculate next interval
        interval = self.next_interval(new_state.stability)
        due_date = now + timedelta(days=interval)
        retrievability = self.retrievability(new_state, 0)

        info = SchedulingInfo(
            memory_state=new_state,
            interval=interval,
            due_date=due_date,
            retrievability=retrievability
        )

        return new_state, info

    def _init_state(self, state: MemoryState, rating: Rating) -> MemoryState:
        """Initialize state for first review."""
        p = self.params

        # Initial stability based on rating
        state.stability = p.initial_stability(rating)

        # Initial difficulty: D0 = w4 - w5 * (rating - 3)
        # Clamp to [1, 10]
        state.difficulty = max(1.0, min(10.0,
            p.w4 - p.w5 * (rating - 3)
        ))

        if rating == Rating.AGAIN:
            state.lapses = 1

        return state

    def _update_state(
        self,
        new_state: MemoryState,
        old_state: MemoryState,
        rating: Rating
    ) -> MemoryState:
        """Update state after subsequent review."""
        p = self.params

        # Current retrievability before review
        R = self.retrievability(old_state, new_state.elapsed_days)

        # Update difficulty (FSRS-4.5 formula from py-fsrs):
        # Step 1: new_d = D - w6 * (rating - 3)
        # Step 2: final_d = w7 * D0 + (1 - w7) * new_d  (mean reversion)
        D0_3 = p.w4  # D0 for rating 3 (Good)
        d_new = old_state.difficulty - p.w6 * (rating - 3)
        new_state.difficulty = max(1.0, min(10.0,
            p.w7 * D0_3 + (1 - p.w7) * d_new
        ))

        if rating == Rating.AGAIN:
            # Lapse - stability decreases significantly
            new_state.lapses += 1
            new_state.stability = self._stability_after_lapse(
                old_state.stability,
                new_state.difficulty,
                R
            )
        else:
            # Success - stability increases
            new_state.stability = self._stability_after_success(
                old_state.stability,
                new_state.difficulty,
                R,
                rating
            )

        return new_state

    def _stability_after_success(
        self,
        S: float,
        D: float,
        R: float,
        rating: Rating
    ) -> float:
        """
        Calculate new stability after successful recall.

        S' = S * (1 + exp(w9) * (11 - D) * S^(-w10) * (exp(w11 * (1 - R)) - 1) * hard_penalty)
        """
        p = self.params

        # Hard penalty reduces stability gain
        hard_penalty = 1.0 if rating != Rating.HARD else p.w16

        # Stability increase factor
        # Higher for: low difficulty, low stability, low retrievability
        factor = (
            math.exp(p.w9) *
            (11 - D) *
            math.pow(S + 0.001, -p.w10) *  # +0.001 to avoid div by zero
            (math.exp(p.w11 * (1 - R)) - 1) *
            hard_penalty
        )

        new_S = S * (1 + factor)

        # Easy bonus
        if rating == Rating.EASY:
            new_S *= 1.3

        return max(0.1, new_S)

    def _stability_after_lapse(self, S: float, D: float, R: float) -> float:
        """
        Calculate new stability after forgetting (lapse).

        S' = w12 * D^(-w13) * ((S + 1)^w14 - 1) * exp(w15 * (1 - R))
        """
        p = self.params

        new_S = (
            p.w12 *
            math.pow(D, -p.w13) *
            (math.pow(S + 1, p.w14) - 1) *
            math.exp(p.w15 * (1 - R))
        )

        # Ensure minimum stability
        return max(0.1, min(new_S, S))  # Can't exceed previous stability

    def preview_ratings(
        self,
        state: MemoryState,
        review_time: datetime | None = None
    ) -> dict[Rating, SchedulingInfo]:
        """
        Preview scheduling for all possible ratings.

        Useful for showing users what will happen with each rating choice.

        Args:
            state: Current memory state
            review_time: When the review would occur

        Returns:
            Dict mapping each rating to its scheduling info
        """
        result = {}
        for rating in Rating:
            _, info = self.review(state, rating, review_time)
            result[rating] = info
        return result

    def get_stats(self) -> dict:
        """Get scheduler statistics and parameters."""
        return {
            "request_retention": self.params.request_retention,
            "maximum_interval": self.params.maximum_interval,
            "parameters": {
                f"w{i}": getattr(self.params, f"w{i}")
                for i in range(17)
            }
        }


class FSRSMemoryTracker:
    """
    Track FSRS states for multiple memory items.

    Integrates with World Weaver's memory system to provide
    spaced repetition scheduling for episodic memories.
    """

    def __init__(self, fsrs: FSRS | None = None):
        """
        Initialize tracker.

        Args:
            fsrs: FSRS scheduler instance (creates default if None)
        """
        self.fsrs = fsrs or FSRS()
        self._states: dict[str, MemoryState] = {}
        self._history: dict[str, list[tuple[datetime, Rating, float]]] = {}

    def get_state(self, memory_id: str) -> MemoryState:
        """Get or create memory state for an item."""
        if memory_id not in self._states:
            self._states[memory_id] = MemoryState()
        return self._states[memory_id]

    def review(
        self,
        memory_id: str,
        rating: Rating,
        review_time: datetime | None = None
    ) -> SchedulingInfo:
        """
        Record a review for a memory item.

        Args:
            memory_id: ID of the memory being reviewed
            rating: User's rating
            review_time: When the review occurred

        Returns:
            Scheduling info for next review
        """
        state = self.get_state(memory_id)
        now = review_time or datetime.now()

        new_state, info = self.fsrs.review(state, rating, now)
        self._states[memory_id] = new_state

        # Record history
        if memory_id not in self._history:
            self._history[memory_id] = []
        self._history[memory_id].append((now, rating, info.interval))

        logger.debug(
            f"FSRS review: {memory_id[:8]}... rating={rating.name} "
            f"S={new_state.stability:.2f} D={new_state.difficulty:.2f} "
            f"next={info.interval:.1f}d"
        )

        return info

    def get_due_items(
        self,
        now: datetime | None = None,
        limit: int = 100
    ) -> list[tuple[str, float, MemoryState]]:
        """
        Get items that are due for review.

        Args:
            now: Current time (default: now)
            limit: Maximum items to return

        Returns:
            List of (memory_id, days_overdue, state) sorted by urgency
        """
        now = now or datetime.now()
        due = []

        for memory_id, state in self._states.items():
            if state.last_review is None:
                # New items are always due
                due.append((memory_id, float("inf"), state))
            else:
                interval = self.fsrs.next_interval(state.stability)
                due_date = state.last_review + timedelta(days=interval)
                if now >= due_date:
                    days_overdue = (now - due_date).total_seconds() / 86400
                    due.append((memory_id, days_overdue, state))

        # Sort by urgency (most overdue first)
        due.sort(key=lambda x: -x[1])
        return due[:limit]

    def get_retrievability(self, memory_id: str, now: datetime | None = None) -> float:
        """Get current retrievability for an item."""
        state = self.get_state(memory_id)
        if state.last_review is None:
            return 0.0

        now = now or datetime.now()
        elapsed = (now - state.last_review).total_seconds() / 86400
        return self.fsrs.retrievability(state, elapsed)

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        if not self._states:
            return {
                "total_items": 0,
                "average_stability": 0,
                "average_difficulty": 0,
                "total_reviews": 0
            }

        stabilities = [s.stability for s in self._states.values() if s.stability > 0]
        difficulties = [s.difficulty for s in self._states.values()]
        total_reviews = sum(s.reps for s in self._states.values())

        return {
            "total_items": len(self._states),
            "average_stability": sum(stabilities) / len(stabilities) if stabilities else 0,
            "average_difficulty": sum(difficulties) / len(difficulties) if difficulties else 0,
            "total_reviews": total_reviews,
            "items_with_lapses": sum(1 for s in self._states.values() if s.lapses > 0)
        }


# Convenience function
def create_fsrs(
    request_retention: float = 0.9,
    maximum_interval: float = 365
) -> FSRS:
    """
    Create an FSRS scheduler with custom settings.

    Args:
        request_retention: Target retention rate (default 90%)
        maximum_interval: Maximum interval in days (default 1 year)

    Returns:
        Configured FSRS instance
    """
    params = FSRSParameters(
        request_retention=request_retention,
        maximum_interval=maximum_interval
    )
    return FSRS(params)


__all__ = [
    "FSRS",
    "FSRSMemoryTracker",
    "FSRSParameters",
    "MemoryState",
    "Rating",
    "SchedulingInfo",
    "create_fsrs",
]
