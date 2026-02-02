"""
Memory Reconsolidation for T4DM.

Addresses Hinton critique: Embeddings are frozen after creation, but should
update based on retrieval outcomes. When a memory is retrieved and the
outcome is known, the embedding should shift toward or away from the
query embedding based on whether the retrieval was helpful.

Biological Basis:
- Reconsolidation in neuroscience: Retrieved memories become labile
- Updating occurs during "reconsolidation window" after retrieval
- Memories are re-encoded with new contextual information

Implementation:
1. On positive outcome: Move memory embedding toward query embedding
2. On negative outcome: Move memory embedding away from query embedding
3. Learning rate scales with |advantage| (confidence in update)
4. Updates are bounded to prevent catastrophic drift

Three-Factor Integration (v2):
When ThreeFactorLearningRule is provided, learning rate is computed as:
    effective_lr = base_lr * eligibility * neuromod_gate * dopamine_surprise
This implements biologically-plausible credit assignment where:
- Only recently active memories get updated (eligibility gating)
- Updates require appropriate brain state (neuromodulator gating)
- Learning scales with prediction error (dopamine surprise)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


def _validate_embedding(arr: np.ndarray, name: str) -> None:
    """
    DATA-005 FIX: Validate embedding array for NaN/Inf values.

    Args:
        arr: Array to validate
        name: Name for error message

    Raises:
        ValueError: If array contains NaN or Inf values
    """
    if np.any(np.isnan(arr)):
        raise ValueError(f"NaN detected in {name}")
    if np.any(np.isinf(arr)):
        raise ValueError(f"Inf detected in {name}")

# Avoid circular import - only import for type checking
if TYPE_CHECKING:
    from t4dm.learning.three_factor import ThreeFactorLearningRule


@dataclass
class ReconsolidationUpdate:
    """Record of a reconsolidation update."""

    memory_id: UUID
    query_embedding: np.ndarray
    original_embedding: np.ndarray
    updated_embedding: np.ndarray
    outcome_score: float
    advantage: float
    learning_rate: float
    # Three-factor components (optional, for enhanced tracking)
    eligibility: float | None = None
    neuromod_gate: float | None = None
    dopamine_surprise: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def update_magnitude(self) -> float:
        """Magnitude of the embedding update."""
        return float(np.linalg.norm(
            self.updated_embedding - self.original_embedding
        ))

    @property
    def used_three_factor(self) -> bool:
        """Whether this update used three-factor learning."""
        return self.eligibility is not None


class ReconsolidationEngine:
    """
    Update memory embeddings based on retrieval outcomes.

    When a memory is retrieved and outcomes are observed:
    - Positive outcome: Pull embedding toward query direction
    - Negative outcome: Push embedding away from query direction
    - Neutral outcome: Small or no update

    The update follows a gradient-like rule:
        new_emb = old_emb + lr * advantage * direction
        direction = normalized(query_emb - old_emb)

    Where advantage = outcome_score - baseline (centered at 0.5).

    With three-factor integration:
        effective_lr = base_lr * eligibility * neuromod_gate * dopamine_surprise
    This gates learning by synaptic activity, brain state, and prediction error.

    Safeguards:
    - Maximum update magnitude prevents catastrophic drift
    - Normalized embeddings maintain unit sphere constraint
    - Cooldown period prevents over-updating recently modified memories
    """

    def __init__(
        self,
        base_learning_rate: float = 0.01,
        max_update_magnitude: float = 0.1,
        baseline: float = 0.5,
        cooldown_hours: float = 1.0,
        lability_window_hours: float = 6.0,
        three_factor: ThreeFactorLearningRule | None = None
    ):
        """
        Initialize reconsolidation engine.

        Args:
            base_learning_rate: Base learning rate for updates
            max_update_magnitude: Maximum allowed L2 norm of update
            baseline: Baseline for advantage computation (0.5 = neutral)
            cooldown_hours: (DEPRECATED, kept for backward compat) Alias for lability_window
            lability_window_hours: Hours after retrieval during which memory is labile
                (can be modified). Per Nader et al. (2000), ~6 hours is typical.
            three_factor: Optional ThreeFactorLearningRule for biologically-
                plausible learning rate computation. When provided, combines
                eligibility traces, neuromodulator state, and dopamine RPE.
        """
        self.base_learning_rate = base_learning_rate
        self.max_update_magnitude = max_update_magnitude
        self.baseline = baseline
        # BIO-MAJOR-001 FIX: Use lability window semantics (biology enables recent updates)
        # cooldown_hours is kept for backward compat but now means lability window
        self.lability_window_hours = lability_window_hours or cooldown_hours
        self.cooldown_hours = cooldown_hours  # Deprecated alias
        self.three_factor = three_factor

        # BIO-MAJOR-001 FIX: Track retrieval times (not update times) for lability
        # Memory becomes LABILE when retrieved, RESTABILIZES after window closes
        self._last_retrieval: dict[str, datetime] = {}
        self._max_lability_entries = 10000  # Maximum tracked memories

        # Keep backward compat alias
        self._last_update = self._last_retrieval  # Alias for backward compat
        self._max_cooldown_entries = self._max_lability_entries

        # History for analysis
        self._update_history: list[ReconsolidationUpdate] = []
        self._max_history_entries = 10000  # Maximum history entries

    def trigger_lability(self, memory_id: UUID) -> None:
        """
        Mark memory as labile due to retrieval.

        BIO-MAJOR-001: Implements biological reconsolidation semantics.
        Memory becomes modifiable for lability_window_hours after retrieval.

        Args:
            memory_id: Memory that was retrieved
        """
        mem_id_str = str(memory_id)
        self._last_retrieval[mem_id_str] = datetime.now()

        # Cleanup if over limit
        if len(self._last_retrieval) > self._max_lability_entries:
            self._cleanup_lability_tracking()

    def is_labile(self, memory_id: UUID) -> bool:
        """
        Check if memory is in labile state (can be modified).

        BIO-MAJOR-001 FIX: Implements correct biological semantics:
        - Memory becomes LABILE when retrieved (trigger_lability called)
        - Memory can be modified WITHIN lability window
        - Memory RESTABILIZES after window closes

        This is OPPOSITE to the old cooldown semantics which prevented
        recent updates.

        Args:
            memory_id: Memory to check

        Returns:
            True if memory is labile (within window after retrieval)
        """
        mem_id_str = str(memory_id)

        if mem_id_str not in self._last_retrieval:
            # Never retrieved = not labile (must be triggered first)
            # But we allow first update for new memories
            return True  # Allow initial encoding

        elapsed = datetime.now() - self._last_retrieval[mem_id_str]
        elapsed_hours = elapsed.total_seconds() / 3600

        # Memory is labile if WITHIN the window (biology: enables recent updates)
        return elapsed_hours < self.lability_window_hours

    def should_update(self, memory_id: UUID) -> bool:
        """
        Check if memory is eligible for reconsolidation.

        BIO-MAJOR-001 FIX: Now uses is_labile() for correct biological semantics.
        Previously used inverted cooldown logic (prevented recent updates).
        Now allows updates within lability window (enables recent updates).

        Args:
            memory_id: Memory to check

        Returns:
            True if memory is labile and can be updated
        """
        return self.is_labile(memory_id)

    def _cleanup_lability_tracking(self) -> None:
        """
        Remove oldest lability tracking entries to enforce memory limit.

        BIO-MAJOR-001 + MEM-008 FIX: Prevent unbounded growth of tracking dict.
        Removes entries past lability window first, then oldest if still over limit.
        """
        now = datetime.now()
        window_seconds = self.lability_window_hours * 3600

        # First, remove entries past lability window (memory has restabilized)
        expired_keys = [
            key for key, timestamp in self._last_retrieval.items()
            if (now - timestamp).total_seconds() > window_seconds
        ]
        for key in expired_keys:
            del self._last_retrieval[key]

        # If still over limit, remove oldest entries
        if len(self._last_retrieval) > self._max_lability_entries:
            # Sort by timestamp and keep newest
            sorted_entries = sorted(
                self._last_retrieval.items(),
                key=lambda x: x[1],
                reverse=True  # Newest first
            )
            # Keep only max entries
            self._last_retrieval = dict(sorted_entries[:self._max_lability_entries])
            # Update alias
            self._last_update = self._last_retrieval

    def _cleanup_cooldowns(self) -> None:
        """DEPRECATED: Alias for _cleanup_lability_tracking. Kept for backward compat."""
        self._cleanup_lability_tracking()

    def _trim_history(self) -> None:
        """Trim update history to max size. MEM-008 FIX."""
        if len(self._update_history) > self._max_history_entries:
            # Keep most recent entries
            self._update_history = self._update_history[-self._max_history_entries:]

    def compute_advantage(self, outcome_score: float) -> float:
        """
        Compute advantage (centered outcome score).

        Args:
            outcome_score: Raw outcome score [0, 1]

        Returns:
            Advantage value (positive = good, negative = bad)
        """
        return outcome_score - self.baseline

    def compute_importance_adjusted_lr(
        self,
        base_lr: float,
        importance: float
    ) -> float:
        """
        Compute learning rate adjusted for memory importance.

        More important memories get smaller updates to prevent catastrophic
        forgetting. Uses formula: effective_lr = base_lr / (1 + importance)

        Args:
            base_lr: Base learning rate
            importance: Memory importance score [0, infinity]
                - 0 = unimportant, full learning rate
                - 1 = moderately important, half learning rate
                - higher values = increasingly protected

        Returns:
            Adjusted learning rate
        """
        return base_lr / (1.0 + importance)

    def reconsolidate(
        self,
        memory_id: UUID,
        memory_embedding: np.ndarray,
        query_embedding: np.ndarray,
        outcome_score: float,
        learning_rate: float | None = None,
        importance: float = 0.0,
        lr_modulation: float = 1.0
    ) -> np.ndarray | None:
        """
        Update memory embedding based on retrieval outcome.

        Args:
            memory_id: Memory being updated
            memory_embedding: Current memory embedding
            query_embedding: Query that retrieved this memory
            outcome_score: Outcome of using this memory [0, 1]
            learning_rate: Override base learning rate
            importance: Memory importance for catastrophic forgetting protection
                Higher importance = smaller updates (default 0 = no protection)
            lr_modulation: Dopamine surprise modulation factor (default 1.0)
                |delta| = |actual - expected| scales learning rate
                Surprising outcomes (|delta| > 0) learn more
                Expected outcomes (|delta| ~= 0) learn less
                NOTE: Ignored when three_factor is configured (it handles this)

        Returns:
            Updated embedding, or None if update skipped

        Raises:
            ValueError: If inputs contain NaN or Inf values
        """
        # DATA-005 FIX: Validate inputs
        _validate_embedding(memory_embedding, "memory_embedding")
        _validate_embedding(query_embedding, "query_embedding")
        if np.isnan(outcome_score):
            raise ValueError("NaN detected in outcome_score")
        if np.isinf(outcome_score):
            raise ValueError("Inf detected in outcome_score")

        # Check cooldown
        if not self.should_update(memory_id):
            logger.debug(f"Skipping reconsolidation for {memory_id}: cooldown")
            return None

        # Three-factor learning rate computation
        eligibility = None
        neuromod_gate = None
        dopamine_surprise = None

        if self.three_factor is not None:
            # Use three-factor rule for biologically-plausible LR
            signal = self.three_factor.compute(
                memory_id=memory_id,
                base_lr=learning_rate if learning_rate is not None else self.base_learning_rate,
                outcome=outcome_score
            )

            # Store components for tracking
            eligibility = signal.eligibility
            neuromod_gate = signal.neuromod_gate
            dopamine_surprise = signal.dopamine_surprise

            # Skip if below eligibility threshold
            if eligibility < self.three_factor.min_eligibility_threshold:
                logger.debug(
                    f"Skipping reconsolidation for {memory_id}: "
                    f"low eligibility {eligibility:.4f}"
                )
                return None

            # Use three-factor effective LR, then apply importance protection
            base_lr = learning_rate if learning_rate is not None else self.base_learning_rate
            lr = base_lr * signal.effective_lr_multiplier
            lr = self.compute_importance_adjusted_lr(lr, importance)

            # Update dopamine expectations for future predictions
            self.three_factor.update_dopamine_expectations(memory_id, outcome_score)

        else:
            # Legacy path: simple modulation without three-factor
            base_lr = learning_rate if learning_rate is not None else self.base_learning_rate
            # Apply importance-weighted protection
            lr = self.compute_importance_adjusted_lr(base_lr, importance)
            # Apply dopamine surprise modulation
            lr = lr * lr_modulation

        advantage = self.compute_advantage(outcome_score)

        # LOGIC-001 FIX: Still track neutral outcomes for dopamine learning
        # Even if we don't update the embedding, we should record the outcome
        # so dopamine expectations can be updated for future predictions.
        if abs(advantage) < 0.01:
            # Record that we processed this memory (refreshes lability window)
            self._last_retrieval[str(memory_id)] = datetime.now()
            logger.debug(
                f"Neutral outcome for {memory_id}: advantage={advantage:.4f}, "
                "recorded for dopamine expectation learning"
            )
            return None

        # Compute update direction
        direction = query_embedding - memory_embedding
        norm = np.linalg.norm(direction)

        # CRASH-015 FIX: Use <= to avoid potential division by very small values
        if norm <= 1e-8:
            # Query and memory are essentially the same (or identical)
            return None

        direction = direction / norm

        # Compute update magnitude (scale by advantage)
        update = lr * advantage * direction

        # Clip update magnitude
        update_norm = np.linalg.norm(update)
        if update_norm > self.max_update_magnitude:
            update = update * (self.max_update_magnitude / update_norm)

        # Apply update
        new_embedding = memory_embedding + update

        # Normalize to unit sphere
        new_norm = np.linalg.norm(new_embedding)
        if new_norm > 0:
            new_embedding = new_embedding / new_norm

        # Record update
        self._last_update[str(memory_id)] = datetime.now()

        # MEM-008 FIX: Cleanup old cooldown entries if over limit
        if len(self._last_update) > self._max_cooldown_entries:
            self._cleanup_cooldowns()

        self._update_history.append(ReconsolidationUpdate(
            memory_id=memory_id,
            query_embedding=query_embedding.copy(),
            original_embedding=memory_embedding.copy(),
            updated_embedding=new_embedding.copy(),
            outcome_score=outcome_score,
            advantage=advantage,
            learning_rate=lr,
            eligibility=eligibility,
            neuromod_gate=neuromod_gate,
            dopamine_surprise=dopamine_surprise
        ))

        # MEM-008 FIX: Trim history if over limit
        self._trim_history()

        if self.three_factor is not None:
            logger.debug(
                f"Reconsolidated {memory_id}: advantage={advantage:.3f}, "
                f"elig={eligibility:.3f}, gate={neuromod_gate:.3f}, "
                f"DA={dopamine_surprise:.3f}, lr={lr:.4f}, "
                f"update_magnitude={np.linalg.norm(update):.4f}"
            )
        else:
            logger.debug(
                f"Reconsolidated {memory_id}: advantage={advantage:.3f}, "
                f"update_magnitude={np.linalg.norm(update):.4f}"
            )

        return new_embedding

    def batch_reconsolidate(
        self,
        memories: list[tuple[UUID, np.ndarray]],
        query_embedding: np.ndarray,
        outcome_score: float,
        per_memory_rewards: dict[str, float] | None = None,
        per_memory_importance: dict[str, float] | None = None,
        per_memory_lr_modulation: dict[str, float] | None = None
    ) -> dict[UUID, np.ndarray]:
        """
        Reconsolidate multiple memories from a single retrieval.

        Args:
            memories: List of (memory_id, embedding) tuples
            query_embedding: Query embedding
            outcome_score: Overall outcome score
            per_memory_rewards: Optional per-memory rewards to use instead
            per_memory_importance: Optional per-memory importance scores
                Higher importance = smaller updates (catastrophic forgetting protection)
            per_memory_lr_modulation: Optional dopamine surprise modulation per memory
                |delta| = |actual - expected| scales learning rate
                Surprising outcomes learn more, expected outcomes learn less
                NOTE: Ignored when three_factor is configured

        Returns:
            Dict of memory_id -> updated_embedding for memories that were updated
        """
        updates = {}

        for memory_id, embedding in memories:
            # Use per-memory reward if available
            if per_memory_rewards and str(memory_id) in per_memory_rewards:
                score = per_memory_rewards[str(memory_id)]
            else:
                score = outcome_score

            # Use per-memory importance if available
            importance = 0.0
            if per_memory_importance and str(memory_id) in per_memory_importance:
                importance = per_memory_importance[str(memory_id)]

            # Use per-memory learning rate modulation if available
            lr_modulation = 1.0
            if per_memory_lr_modulation and str(memory_id) in per_memory_lr_modulation:
                lr_modulation = per_memory_lr_modulation[str(memory_id)]

            new_embedding = self.reconsolidate(
                memory_id=memory_id,
                memory_embedding=embedding,
                query_embedding=query_embedding,
                outcome_score=score,
                importance=importance,
                lr_modulation=lr_modulation
            )

            if new_embedding is not None:
                updates[memory_id] = new_embedding

        return updates

    def get_update_history(
        self,
        memory_id: UUID | None = None,
        limit: int = 100
    ) -> list[ReconsolidationUpdate]:
        """
        Get recent update history.

        Args:
            memory_id: Filter to specific memory (optional)
            limit: Maximum updates to return

        Returns:
            List of ReconsolidationUpdate objects
        """
        history = self._update_history

        if memory_id is not None:
            history = [u for u in history if u.memory_id == memory_id]

        return history[-limit:]

    def get_stats(self) -> dict:
        """
        Get reconsolidation statistics.

        Returns:
            Dict with update counts and average magnitudes
        """
        if not self._update_history:
            return {
                "total_updates": 0,
                "positive_updates": 0,
                "negative_updates": 0,
                "avg_magnitude": 0.0,
                "avg_advantage": 0.0,
                "three_factor_updates": 0,
                "three_factor_enabled": self.three_factor is not None
            }

        positive = [u for u in self._update_history if u.advantage > 0]
        negative = [u for u in self._update_history if u.advantage < 0]
        three_factor_updates = [u for u in self._update_history if u.used_three_factor]

        stats = {
            "total_updates": len(self._update_history),
            "positive_updates": len(positive),
            "negative_updates": len(negative),
            "avg_magnitude": float(np.mean([
                u.update_magnitude for u in self._update_history
            ])),
            "avg_advantage": float(np.mean([
                u.advantage for u in self._update_history
            ])),
            "three_factor_updates": len(three_factor_updates),
            "three_factor_enabled": self.three_factor is not None
        }

        # Add three-factor specific stats if available
        if three_factor_updates:
            stats["avg_eligibility"] = float(np.mean([
                u.eligibility for u in three_factor_updates
            ]))
            stats["avg_neuromod_gate"] = float(np.mean([
                u.neuromod_gate for u in three_factor_updates
            ]))
            stats["avg_dopamine_surprise"] = float(np.mean([
                u.dopamine_surprise for u in three_factor_updates
            ]))

        return stats

    def clear_history(self) -> None:
        """Clear update history and cooldown tracking."""
        self._update_history.clear()
        self._last_update.clear()


class NeuromodulatorIntegratedReconsolidation:
    """
    Reconsolidation with full neuromodulator orchestra integration.

    This integrates all neuromodulatory signals for embedding updates:
    - Dopamine: Surprise-driven learning (RPE magnitude)
    - Serotonin: Long-term patience and eligibility traces
    - Norepinephrine: Arousal-modulated learning rate
    - Acetylcholine: Encoding/retrieval mode
    - Eligibility: Temporal credit assignment

    Uses multiplicative gating: all signals must align for strong updates.
    """

    def __init__(
        self,
        orchestra: NeuromodulatorOrchestra | None = None,
        base_learning_rate: float = 0.01,
        max_update_magnitude: float = 0.1,
        cooldown_hours: float = 1.0
    ):
        """
        Initialize neuromodulator-integrated reconsolidation.

        Args:
            orchestra: NeuromodulatorOrchestra instance (created if None)
            base_learning_rate: Base learning rate for embedding updates
            max_update_magnitude: Maximum L2 norm of embedding update
            cooldown_hours: Minimum hours between updates to same memory
        """
        # Avoid circular import
        if orchestra is None:
            from t4dm.learning.neuromodulators import NeuromodulatorOrchestra
            orchestra = NeuromodulatorOrchestra()

        self.orchestra = orchestra
        self.reconsolidation = ReconsolidationEngine(
            base_learning_rate=base_learning_rate,
            max_update_magnitude=max_update_magnitude,
            cooldown_hours=cooldown_hours
        )

    def update(
        self,
        memory_id: UUID,
        memory_embedding: np.ndarray,
        query_embedding: np.ndarray,
        outcome_score: float,
        importance: float = 0.0
    ) -> np.ndarray | None:
        """
        Update memory with full neuromodulator integration.

        This is the main entry point that:
        1. Gets integrated learning params from orchestra
        2. Applies multiplicative gating
        3. Updates the embedding
        4. Updates neuromodulator systems

        Args:
            memory_id: Memory being updated
            memory_embedding: Current memory embedding
            query_embedding: Query that retrieved this memory
            outcome_score: Actual outcome [0, 1]
            importance: Memory importance for catastrophic forgetting protection

        Returns:
            Updated embedding, or None if update skipped
        """
        # Get integrated learning parameters from orchestra
        params = self.orchestra.get_learning_params_with_outcome(
            memory_id, outcome_score
        )

        # Use multiplicative gating for learning rate
        # All signals gate each other: effective_lr * eligibility * surprise * patience
        lr_modulation = params.combined_learning_signal

        # Apply reconsolidation with integrated learning rate
        updated_embedding = self.reconsolidation.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=outcome_score,
            importance=importance,
            lr_modulation=lr_modulation
        )

        if updated_embedding is not None:
            logger.debug(
                f"Neuromodulator-integrated update: memory={memory_id}, "
                f"combined_signal={lr_modulation:.3f}, "
                f"(lr={params.effective_lr:.2f} * elig={params.eligibility:.2f} * "
                f"surprise={params.surprise:.2f} * patience={params.patience:.2f})"
            )

        return updated_embedding

    def batch_update(
        self,
        memories: list[tuple[UUID, np.ndarray]],
        query_embedding: np.ndarray,
        memory_outcomes: dict[str, float],
        per_memory_importance: dict[str, float] | None = None
    ) -> dict[UUID, np.ndarray]:
        """
        Batch update multiple memories with neuromodulator integration.

        Args:
            memories: List of (memory_id, embedding) tuples
            query_embedding: Query embedding
            memory_outcomes: Per-memory outcome scores
            per_memory_importance: Optional per-memory importance scores

        Returns:
            Dict of memory_id -> updated_embedding for updated memories
        """
        updates = {}

        for memory_id, embedding in memories:
            mem_id_str = str(memory_id)

            # Get outcome for this memory
            outcome = memory_outcomes.get(mem_id_str, 0.5)

            # Get importance for this memory
            importance = 0.0
            if per_memory_importance and mem_id_str in per_memory_importance:
                importance = per_memory_importance[mem_id_str]

            updated = self.update(
                memory_id=memory_id,
                memory_embedding=embedding,
                query_embedding=query_embedding,
                outcome_score=outcome,
                importance=importance
            )

            if updated is not None:
                updates[memory_id] = updated

        return updates

    def get_stats(self) -> dict:
        """Get combined statistics from both systems."""
        return {
            "reconsolidation": self.reconsolidation.get_stats(),
            "orchestra": self.orchestra.get_stats()
        }


class DopamineModulatedReconsolidation:
    """
    Reconsolidation with integrated dopamine surprise modulation.

    Combines the ReconsolidationEngine with DopamineSystem to create a
    biologically-plausible learning system where:
    - Surprising outcomes (high |RPE|) drive larger embedding updates
    - Expected outcomes (low |RPE|) result in minimal updates
    - Value expectations are updated based on observed outcomes

    This implements the key insight from dopamine research: learning should
    be proportional to prediction error, not raw reward.

    Note: For full three-factor learning (eligibility + neuromod + dopamine),
    use ReconsolidationEngine with three_factor parameter instead.
    """

    def __init__(
        self,
        base_learning_rate: float = 0.01,
        max_update_magnitude: float = 0.1,
        value_learning_rate: float = 0.1,
        cooldown_hours: float = 1.0,
        use_uncertainty_boost: bool = True
    ):
        """
        Initialize dopamine-modulated reconsolidation.

        Args:
            base_learning_rate: Base learning rate for embedding updates
            max_update_magnitude: Maximum L2 norm of embedding update
            value_learning_rate: Learning rate for value expectation updates
            cooldown_hours: Minimum hours between updates to same memory
            use_uncertainty_boost: Boost LR for memories with few observations
        """
        # Import here to avoid circular dependency
        from t4dm.learning.dopamine import DopamineSystem

        self.reconsolidation = ReconsolidationEngine(
            base_learning_rate=base_learning_rate,
            max_update_magnitude=max_update_magnitude,
            cooldown_hours=cooldown_hours
        )

        self.dopamine = DopamineSystem(
            value_learning_rate=value_learning_rate
        )

        self.use_uncertainty_boost = use_uncertainty_boost

    def update(
        self,
        memory_id: UUID,
        memory_embedding: np.ndarray,
        query_embedding: np.ndarray,
        outcome_score: float,
        importance: float = 0.0
    ) -> np.ndarray | None:
        """
        Update memory with dopamine-modulated reconsolidation.

        This is the main entry point that:
        1. Computes RPE (dopamine signal)
        2. Modulates learning rate by surprise magnitude
        3. Updates the embedding
        4. Updates value expectations for future predictions

        Args:
            memory_id: Memory being updated
            memory_embedding: Current memory embedding
            query_embedding: Query that retrieved this memory
            outcome_score: Actual outcome [0, 1]
            importance: Memory importance for catastrophic forgetting protection

        Returns:
            Updated embedding, or None if update skipped
        """
        # 1. Compute dopamine RPE signal
        rpe = self.dopamine.compute_rpe(memory_id, outcome_score)

        # 2. Compute surprise-modulated learning rate
        lr_modulation = self.dopamine.modulate_learning_rate(
            base_lr=1.0,  # Base LR is already in reconsolidation
            rpe=rpe,
            use_uncertainty=self.use_uncertainty_boost
        )

        # 3. Apply reconsolidation with modulated learning rate
        updated_embedding = self.reconsolidation.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=outcome_score,
            importance=importance,
            lr_modulation=lr_modulation
        )

        # 4. Update value expectations for future predictions
        self.dopamine.update_expectations(memory_id, outcome_score)

        if updated_embedding is not None:
            logger.debug(
                f"Dopamine-modulated update: memory={memory_id}, "
                f"RPE={rpe.rpe:.3f}, lr_mod={lr_modulation:.3f}"
            )

        return updated_embedding

    def batch_update(
        self,
        memories: list[tuple[UUID, np.ndarray]],
        query_embedding: np.ndarray,
        memory_outcomes: dict[str, float],
        per_memory_importance: dict[str, float] | None = None
    ) -> dict[UUID, np.ndarray]:
        """
        Batch update multiple memories with dopamine modulation.

        Args:
            memories: List of (memory_id, embedding) tuples
            query_embedding: Query embedding
            memory_outcomes: Per-memory outcome scores
            per_memory_importance: Optional per-memory importance scores

        Returns:
            Dict of memory_id -> updated_embedding for updated memories
        """
        updates = {}

        for memory_id, embedding in memories:
            mem_id_str = str(memory_id)

            # Get outcome for this memory
            outcome = memory_outcomes.get(mem_id_str, 0.5)

            # Get importance for this memory
            importance = 0.0
            if per_memory_importance and mem_id_str in per_memory_importance:
                importance = per_memory_importance[mem_id_str]

            updated = self.update(
                memory_id=memory_id,
                memory_embedding=embedding,
                query_embedding=query_embedding,
                outcome_score=outcome,
                importance=importance
            )

            if updated is not None:
                updates[memory_id] = updated

        return updates

    def get_stats(self) -> dict:
        """Get combined statistics from both systems."""
        return {
            "reconsolidation": self.reconsolidation.get_stats(),
            "dopamine": self.dopamine.get_stats()
        }


# Convenience function
def reconsolidate(
    memory_embedding: np.ndarray,
    query_embedding: np.ndarray,
    outcome_score: float,
    learning_rate: float = 0.01
) -> np.ndarray:
    """
    Simple reconsolidation update (stateless).

    Args:
        memory_embedding: Current memory embedding
        query_embedding: Query that retrieved this memory
        outcome_score: Outcome of using this memory [0, 1]
        learning_rate: Learning rate for update

    Returns:
        Updated embedding (normalized to unit sphere)

    Raises:
        ValueError: If inputs contain NaN or Inf values
    """
    # DATA-005 FIX: Validate inputs
    _validate_embedding(memory_embedding, "memory_embedding")
    _validate_embedding(query_embedding, "query_embedding")
    if np.isnan(outcome_score):
        raise ValueError("NaN detected in outcome_score")
    if np.isinf(outcome_score):
        raise ValueError("Inf detected in outcome_score")

    advantage = outcome_score - 0.5  # Centered

    direction = query_embedding - memory_embedding
    norm = np.linalg.norm(direction)

    # CRASH-015 FIX: Use <= to avoid potential division by very small values
    if norm <= 1e-8:
        return memory_embedding.copy()

    direction = direction / norm
    update = learning_rate * advantage * direction

    new_embedding = memory_embedding + update

    # Normalize
    new_norm = np.linalg.norm(new_embedding)
    if new_norm > 0:
        new_embedding = new_embedding / new_norm

    return new_embedding
