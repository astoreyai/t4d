"""
Serotonin-like Long-Term Credit Assignment for T4DM.

Biological Basis:
- 5-HT modulates patience and temporal discounting
- Low 5-HT leads to impulsive, short-term choices
- High 5-HT promotes waiting for larger delayed rewards
- 5-HT interacts with dopamine for temporal credit assignment

Implementation:
- Maintains eligibility traces that decay slowly
- Computes long-horizon value estimates
- Modulates temporal discounting in value learning
- Tracks session-level and multi-session outcomes

Integration Points:
1. EpisodicMemory.recall(): Add eligibility when memories retrieved
2. Session end hooks: Trigger outcome distribution
3. Reconsolidation: Use long-term value for importance weighting
4. NeuroSymbolicReasoner: Include long-term value in fusion

References:
- Daw et al. (2002): Serotonin and temporal discounting
- Cools et al. (2008): 5-HT and behavioral inhibition
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import numpy as np

# Import sophisticated eligibility trace from eligibility module
from t4dm.learning.eligibility import EligibilityTrace as EligibilityTracer

# Backward-compatible export for code that imports EligibilityTrace from serotonin
EligibilityTrace = EligibilityTracer

logger = logging.getLogger(__name__)


@dataclass
class TemporalContext:
    """Context for long-term outcome tracking."""

    session_id: str
    goal_description: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    outcome_received: bool = False
    final_outcome: float | None = None


class SerotoninSystem:
    """
    Long-term credit assignment inspired by serotonergic modulation.

    While dopamine signals immediate prediction errors, serotonin
    supports patience and long-term value estimation:

    1. Eligibility traces connect past memories to future outcomes
    2. Temporal discounting controls how far to look ahead
    3. Session-level outcomes update memories used throughout
    4. Mood/baseline affects all value estimates

    Key insight: Some memories are valuable not for immediate use,
    but because they set up future successes. Serotonin captures this.

    The system maintains:
    - Eligibility traces per memory (decay over time)
    - Long-term value estimates (learned across sessions)
    - Mood state (affects value modulation)
    - Active contexts for outcome tracking

    Note: Uses the sophisticated EligibilityTrace from eligibility.py
    which includes STDP-like learning, thread safety, and security bounds.
    """

    def __init__(
        self,
        base_discount_rate: float = 0.99,  # Per-step discount
        eligibility_decay: float = 0.95,   # Per-hour decay (mapped to tau_trace)
        trace_lifetime_hours: float = 24.0,
        baseline_mood: float = 0.5,
        mood_adaptation_rate: float = 0.1,
        max_traces_per_memory: int = 10
    ):
        """
        Initialize serotonin system.

        Args:
            base_discount_rate: Gamma for temporal discounting
            eligibility_decay: How fast traces decay per hour
            trace_lifetime_hours: Maximum trace lifetime
            baseline_mood: Default mood level [0, 1]
            mood_adaptation_rate: How fast mood adapts
            max_traces_per_memory: Max traces to maintain per memory
        """
        self.base_discount_rate = base_discount_rate
        self.eligibility_decay = eligibility_decay
        self.trace_lifetime_hours = trace_lifetime_hours
        self.baseline_mood = baseline_mood
        self.mood_adaptation_rate = mood_adaptation_rate
        self.max_traces_per_memory = max_traces_per_memory

        # Current mood level
        self._mood = baseline_mood

        # Convert decay rate to tau_trace time constant
        # eligibility_decay is per-hour, tau_trace is exponential time constant
        # decay = exp(-dt/tau) => tau = -dt/ln(decay)
        # For 1 hour: tau = -3600 / ln(eligibility_decay)
        self._base_tau_trace = -3600.0 / np.log(eligibility_decay) if eligibility_decay < 1.0 else 3600.0

        # Use sophisticated eligibility trace system
        self._eligibility_tracer = EligibilityTracer(
            decay=eligibility_decay,
            tau_trace=self._base_tau_trace,
            a_plus=0.005,  # STDP learning rate
            a_minus=0.00525,
            min_trace=0.01,
            max_traces=max_traces_per_memory * 1000  # Conservative upper bound
        )

        # Active temporal contexts (ongoing sessions/goals)
        self._active_contexts: dict[str, TemporalContext] = {}

        # Long-term value estimates (learned over sessions)
        self._long_term_values: dict[str, float] = {}

        # Statistics
        self._total_outcomes = 0
        self._positive_outcomes = 0

    def start_context(
        self,
        session_id: str,
        goal_description: str | None = None
    ) -> None:
        """
        Start a new temporal context for tracking.

        Args:
            session_id: Unique session identifier
            goal_description: Optional description of session goal
        """
        self._active_contexts[session_id] = TemporalContext(
            session_id=session_id,
            goal_description=goal_description
        )
        logger.debug(f"Started temporal context: {session_id}")

    def end_context(self, session_id: str) -> TemporalContext | None:
        """
        End and remove a temporal context.

        Args:
            session_id: Context to end

        Returns:
            The ended context, or None if not found
        """
        return self._active_contexts.pop(session_id, None)

    def add_eligibility(
        self,
        memory_id: UUID,
        strength: float = 1.0,
        context_id: str | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        """
        Add eligibility trace for a memory with temporal discount.

        Called when a memory is retrieved - makes it eligible
        for credit when outcomes arrive later.

        P2.4: Implements temporal credit assignment decay per Daw et al. (2002).
        If there's a delay between when the memory was accessed and when
        eligibility is registered, the initial strength is discounted.

        Args:
            memory_id: Memory that was used
            strength: Initial trace strength
            context_id: Optional context to associate with
            delay_seconds: Time delay since memory access (applies temporal discount)
        """
        mem_id_str = str(memory_id)

        # P2.4: Apply temporal discount based on delay
        # Half-life = tau * ln(2), discount = gamma ^ (delay / half_life)
        if delay_seconds > 0:
            trace_half_life = self._base_tau_trace * np.log(2)
            discount = self.base_discount_rate ** (delay_seconds / trace_half_life)
            effective_strength = strength * discount
            logger.debug(
                f"Temporal discount: delay={delay_seconds:.1f}s, "
                f"discount={discount:.3f}, strength {strength:.3f} → {effective_strength:.3f}"
            )
        else:
            effective_strength = strength

        # Update sophisticated eligibility trace with discounted strength
        self._eligibility_tracer.update(mem_id_str, activity=effective_strength)

    def get_eligibility(self, memory_id: UUID) -> float:
        """
        Get current total eligibility for a memory.

        Args:
            memory_id: Memory to check

        Returns:
            Total eligibility (sum of active traces)
        """
        mem_id_str = str(memory_id)
        return min(1.0, self._eligibility_tracer.get_trace(mem_id_str))

    def receive_outcome(
        self,
        outcome_score: float,
        context_id: str | None = None,
        decay_with_time: bool = True
    ) -> dict[str, float]:
        """
        Receive an outcome and distribute credit via eligibility.

        This is the serotonin equivalent of dopamine's RPE, but
        it assigns credit across time, not just to immediate actions.

        Args:
            outcome_score: Outcome value [0, 1]
            context_id: Optional context this outcome belongs to
            decay_with_time: Whether to apply temporal decay

        Returns:
            Memory ID -> credit assigned
        """
        # Update mood based on outcome
        self._mood += self.mood_adaptation_rate * (outcome_score - self._mood)
        self._mood = float(np.clip(self._mood, 0.0, 1.0))

        # Track statistics
        self._total_outcomes += 1
        if outcome_score > 0.5:
            self._positive_outcomes += 1

        # Compute advantage (centered on mood)
        advantage = outcome_score - self._mood

        # Distribute credit to eligible memories using sophisticated trace
        credits = self._eligibility_tracer.assign_credit(reward=advantage)

        # Update long-term value estimates
        # LEARNING-CRITICAL-001 FIX: Apply temporal discounting via patience factor
        current_time = time.time()
        for mem_id_str, credit in credits.items():
            current_value = self._long_term_values.get(mem_id_str, 0.5)

            # Get trace strength for learning rate scaling
            trace_strength = self._eligibility_tracer.get_trace(mem_id_str)
            learning_rate = 0.1 * trace_strength

            # LEARNING-CRITICAL-001 FIX: Compute temporal discount based on trace age
            # Get time since trace was last updated (proxy for steps to outcome)
            trace_entry = self._eligibility_tracer.traces.get(mem_id_str)
            if trace_entry is not None:
                # Convert elapsed time to "steps" (1 step = 1 second for this purpose)
                elapsed_seconds = current_time - trace_entry.last_update
                steps_to_outcome = max(1, int(elapsed_seconds))
                patience_factor = self.compute_patience_factor(steps_to_outcome)
            else:
                # No trace found (shouldn't happen but handle gracefully)
                patience_factor = 1.0

            # Apply patience factor to discount delayed outcomes
            discounted_advantage = advantage * patience_factor
            new_value = current_value + learning_rate * discounted_advantage
            self._long_term_values[mem_id_str] = float(np.clip(new_value, 0.0, 1.0))

        # Mark context as complete if provided
        if context_id and context_id in self._active_contexts:
            self._active_contexts[context_id].outcome_received = True
            self._active_contexts[context_id].final_outcome = outcome_score

        logger.debug(
            f"Distributed credit to {len(credits)} memories, "
            f"outcome={outcome_score:.3f}, mood={self._mood:.3f}"
        )

        return credits

    def get_long_term_value(self, memory_id: UUID) -> float:
        """
        Get long-term value estimate for a memory.

        This represents how often this memory has led to
        positive long-term outcomes.

        Args:
            memory_id: Memory to check

        Returns:
            Long-term value estimate [0, 1]
        """
        return self._long_term_values.get(str(memory_id), 0.5)

    def get_long_term_values_batch(self, memory_ids: list[UUID]) -> dict[str, float]:
        """
        Get long-term values for multiple memories.

        Args:
            memory_ids: List of memory IDs

        Returns:
            Memory ID -> long-term value
        """
        return {
            str(mem_id): self.get_long_term_value(mem_id)
            for mem_id in memory_ids
        }

    def compute_patience_factor(
        self,
        steps_to_outcome: int
    ) -> float:
        """
        Compute discount factor for delayed outcomes.

        Higher mood (5-HT) = more patience = less discounting.

        Args:
            steps_to_outcome: Expected steps until outcome

        Returns:
            Patience factor [0, 1]
        """
        # Base discount
        base_patience = self.base_discount_rate ** steps_to_outcome

        # Mood modulation: high mood reduces temporal discounting
        mood_bonus = 0.2 * self._mood
        effective_patience = base_patience + mood_bonus * (1 - base_patience)

        return float(np.clip(effective_patience, 0.0, 1.0))

    def modulate_value_by_mood(self, raw_value: float) -> float:
        """
        Modulate value estimate by current mood.

        Low mood pessimistically reduces values;
        high mood optimistically increases them.

        Args:
            raw_value: Unmodulated value

        Returns:
            Mood-modulated value
        """
        # Mood deviation from neutral
        mood_offset = self._mood - 0.5

        # Apply offset (bounded)
        modulated = raw_value + 0.2 * mood_offset
        return float(np.clip(modulated, 0.0, 1.0))

    def get_current_mood(self) -> float:
        """Get current mood level."""
        return self._mood

    def set_mood(self, mood: float) -> None:
        """
        Manually set mood level.

        Args:
            mood: New mood level [0, 1]
        """
        self._mood = float(np.clip(mood, 0.0, 1.0))

    def get_trace_half_life(self) -> float:
        """
        Get trace half-life in seconds.

        P2.4: Half-life = tau_trace * ln(2)
        This is the time for a trace to decay to 50% of its value.

        Returns:
            Trace half-life in seconds
        """
        return self._base_tau_trace * np.log(2)

    def compute_temporal_discount(self, delay_seconds: float) -> float:
        """
        Compute temporal discount factor for a given delay.

        P2.4: Implements exponential temporal discounting.
        discount = gamma ^ (delay / half_life)

        Args:
            delay_seconds: Time delay in seconds

        Returns:
            Discount factor [0, 1] where 1 = no discount, 0 = full discount
        """
        if delay_seconds <= 0:
            return 1.0

        half_life = self.get_trace_half_life()
        discount = self.base_discount_rate ** (delay_seconds / half_life)
        return float(np.clip(discount, 0.0, 1.0))

    def _get_effective_tau_trace(self) -> float:
        """
        Compute mood-modulated tau_trace.

        LEARNING-CRITICAL-002 FIX: 5-HT should modulate trace decay rate.
        High mood (5-HT) → slower decay → longer tau
        Low mood → faster decay → shorter tau

        Returns:
            Effective tau_trace based on current mood
        """
        # Mood modulation: scale tau by 0.5x (low mood) to 1.5x (high mood)
        # At mood=0.5, effective_tau = base_tau (no change)
        mood_factor = 0.5 + self._mood  # Range: [0.5, 1.5]
        return self._base_tau_trace * mood_factor

    def cleanup_expired_traces(self) -> int:
        """
        Remove expired eligibility traces with mood-modulated decay.

        LEARNING-CRITICAL-002 FIX: Apply mood-modulated tau_trace before stepping.

        Returns:
            Number of traces removed
        """
        # LEARNING-CRITICAL-002 FIX: Update tau_trace based on current mood
        effective_tau = self._get_effective_tau_trace()
        self._eligibility_tracer.tau_trace = effective_tau

        # Step the trace system to decay and clean up
        self._eligibility_tracer.step()

        # Return count (approximate since we don't track removals in this API)
        return 0

    def get_memories_with_traces(self) -> list[str]:
        """Get list of memory IDs that have active traces."""
        active_traces = self._eligibility_tracer.get_all_active(threshold=0.01)
        return list(active_traces.keys())

    def get_stats(self) -> dict:
        """Get serotonin system statistics."""
        trace_stats = self._eligibility_tracer.get_stats()

        return {
            "current_mood": self._mood,
            "total_outcomes": self._total_outcomes,
            "positive_outcome_rate": (
                self._positive_outcomes / self._total_outcomes
                if self._total_outcomes > 0 else 0.5
            ),
            "memories_with_traces": trace_stats.get("count", 0),
            "active_traces": trace_stats.get("count", 0),
            "memories_with_long_term_values": len(self._long_term_values),
            "active_contexts": len(self._active_contexts),
            "mean_trace_strength": trace_stats.get("mean_trace", 0.0),
            "total_credits_assigned": trace_stats.get("total_credits_assigned", 0.0),
            "config": self.get_config(),
        }

    # ==================== Runtime Configuration Setters ====================

    def set_baseline_mood(self, mood: float) -> None:
        """
        Set baseline mood level.

        Args:
            mood: Baseline mood [0.0, 1.0]
        """
        self.baseline_mood = float(np.clip(mood, 0.0, 1.0))
        logger.info(f"5-HT baseline_mood set to {self.baseline_mood}")

    def set_mood_adaptation_rate(self, rate: float) -> None:
        """
        Set mood adaptation rate.

        Args:
            rate: Adaptation rate [0.01, 0.5]
        """
        self.mood_adaptation_rate = float(np.clip(rate, 0.01, 0.5))
        logger.info(f"5-HT mood_adaptation_rate set to {self.mood_adaptation_rate}")

    def set_discount_rate(self, rate: float) -> None:
        """
        Set temporal discount rate (gamma).

        Args:
            rate: Discount rate [0.9, 1.0]
        """
        self.base_discount_rate = float(np.clip(rate, 0.9, 1.0))
        logger.info(f"5-HT base_discount_rate set to {self.base_discount_rate}")

    def set_eligibility_decay(self, decay: float) -> None:
        """
        Set eligibility trace decay rate.

        Args:
            decay: Per-hour decay rate [0.8, 0.99]
        """
        self.eligibility_decay = float(np.clip(decay, 0.8, 0.99))
        # Update the base tau_trace
        self._base_tau_trace = -3600.0 / np.log(self.eligibility_decay) if self.eligibility_decay < 1.0 else 3600.0
        self._eligibility_tracer.tau_trace = self._base_tau_trace
        logger.info(f"5-HT eligibility_decay set to {self.eligibility_decay}")

    def clear_eligibility_traces(self) -> None:
        """Clear all eligibility traces."""
        self._eligibility_tracer.clear()
        logger.info("5-HT eligibility traces cleared")

    def clear_long_term_values(self) -> None:
        """Clear all long-term value estimates."""
        self._long_term_values.clear()
        logger.info("5-HT long-term values cleared")

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "baseline_mood": self.baseline_mood,
            "mood_adaptation_rate": self.mood_adaptation_rate,
            "base_discount_rate": self.base_discount_rate,
            "eligibility_decay": self.eligibility_decay,
            "trace_lifetime_hours": self.trace_lifetime_hours,
            "max_traces_per_memory": self.max_traces_per_memory,
        }

    def reset(self) -> None:
        """Reset to baseline state."""
        self._mood = self.baseline_mood
        self._eligibility_tracer.clear()
        self._active_contexts.clear()
        self._long_term_values.clear()
        self._total_outcomes = 0
        self._positive_outcomes = 0


__all__ = [
    "SerotoninSystem",
    "TemporalContext",
]
