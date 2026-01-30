"""
Three-Factor Learning Rule for World Weaver.

Biological Basis:
In biological neural networks, synaptic plasticity requires three factors:
1. Eligibility trace (synaptic tag) - marks recently active synapses
2. Neuromodulatory gate (ACh, NE, etc.) - global "learning allowed" signal
3. Dopamine surprise (RPE) - scales learning by prediction error

Formula:
    effective_lr = base_lr * eligibility * neuromod_gate * dopamine_surprise

This implements the "three-factor rule" from computational neuroscience:
- Eligibility: Which synapses were active? (temporal credit assignment)
- Neuromodulator: Should we learn now? (encoding mode, arousal)
- Dopamine: How surprising was this? (prediction error magnitude)

Key insight: Learning should be strongest when:
1. A memory was recently active (high eligibility)
2. The system is in an encoding/learning state (neuromod gate open)
3. The outcome was surprising (high |RPE|)

References:
- Frémaux & Gerstner (2016): Neuromodulated STDP
- Gerstner et al. (2018): Eligibility traces and three-factor learning
- Schultz (1998): Dopamine reward prediction

Integration Points:
- ReconsolidationEngine: Replace simple lr_modulation with three-factor LR
- NeuroSymbolicReasoner: Use for fusion weight updates
- MemoryConsolidation: Prioritize high-eligibility memories
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

# P7.4: Import coupling for energy-based learning integration
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ww.learning.dopamine import DopamineSystem, RewardPredictionError
from ww.learning.eligibility import EligibilityTrace as EligibilityTraceSystem
from ww.learning.eligibility import LayeredEligibilityTrace
from ww.learning.neuromodulators import NeuromodulatorOrchestra, NeuromodulatorState

if TYPE_CHECKING:
    from ww.nca.coupling import LearnableCoupling


def _validate_scalar(value: float, name: str) -> None:
    """DATA-005 FIX: Validate scalar for NaN/Inf values."""
    if np.isnan(value):
        raise ValueError(f"NaN detected in {name}")
    if np.isinf(value):
        raise ValueError(f"Inf detected in {name}")

logger = logging.getLogger(__name__)


@dataclass
class ThreeFactorSignal:
    """
    Complete three-factor learning signal for a memory.

    Contains all components and the computed effective learning rate.
    """
    memory_id: UUID

    # Factor 1: Eligibility (temporal credit)
    eligibility: float  # [0, 1] from trace decay

    # Factor 2: Neuromodulator gate (global learning state)
    neuromod_gate: float  # Combined ACh + NE + 5-HT signal
    ach_mode_factor: float  # ACh encoding/retrieval contribution
    ne_arousal_factor: float  # NE arousal contribution
    serotonin_mood_factor: float  # 5-HT mood contribution

    # Factor 3: Dopamine surprise (prediction error)
    dopamine_surprise: float  # |RPE| magnitude
    rpe_raw: float  # Signed RPE for direction

    # Computed result
    effective_lr_multiplier: float  # Final learning rate multiplier

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "memory_id": str(self.memory_id),
            "eligibility": self.eligibility,
            "neuromod_gate": self.neuromod_gate,
            "ach_mode_factor": self.ach_mode_factor,
            "ne_arousal_factor": self.ne_arousal_factor,
            "serotonin_mood_factor": self.serotonin_mood_factor,
            "dopamine_surprise": self.dopamine_surprise,
            "rpe_raw": self.rpe_raw,
            "effective_lr_multiplier": self.effective_lr_multiplier,
            "timestamp": self.timestamp.isoformat()
        }


class ThreeFactorLearningRule:
    """
    Unified three-factor learning rule combining:
    1. Eligibility traces (which synapses were active)
    2. Neuromodulator gate (should we learn now)
    3. Dopamine surprise (how surprising was this)

    This creates biologically-plausible credit assignment:
    - Only recently active memories get updated (eligibility)
    - Updates only happen in appropriate brain states (neuromod)
    - Learning magnitude scales with surprise (dopamine)

    Usage:
        three_factor = ThreeFactorLearningRule(
            eligibility_trace=...,
            neuromodulator_orchestra=...,
            dopamine_system=...
        )

        # Get effective learning rate for a memory
        signal = three_factor.compute(memory_id, base_lr=0.01, outcome=0.8)
        effective_lr = signal.effective_lr_multiplier * base_lr
    """

    def __init__(
        self,
        eligibility_trace: EligibilityTraceSystem | None = None,
        neuromodulator_orchestra: NeuromodulatorOrchestra | None = None,
        dopamine_system: DopamineSystem | None = None,
        coupling: LearnableCoupling | None = None,  # P7.4: Energy-based coupling
        # Weighting for neuromodulator components
        ach_weight: float = 0.4,
        ne_weight: float = 0.35,
        serotonin_weight: float = 0.25,
        # Bounds
        # Note: Default a_plus in EligibilityTrace is 0.005, so single activation
        # with activity=1.0 produces trace=0.005. Threshold must be <= a_plus.
        min_eligibility_threshold: float = 0.001,
        min_effective_lr: float = 0.1,  # Floor to prevent zero learning
        max_effective_lr: float = 3.0,  # Cap to prevent instability
        # P7.4: Energy-based learning control
        enable_coupling_updates: bool = True,
    ):
        """
        Initialize three-factor learning rule.

        Args:
            eligibility_trace: Eligibility trace system (created if None)
            neuromodulator_orchestra: Neuromodulator orchestra (created if None)
            dopamine_system: Dopamine RPE system (created if None)
            coupling: P7.4: LearnableCoupling for energy-based updates
            ach_weight: Weight for ACh mode in neuromod gate
            ne_weight: Weight for NE arousal in neuromod gate
            serotonin_weight: Weight for 5-HT mood in neuromod gate
            min_eligibility_threshold: Skip memories below this eligibility
            min_effective_lr: Minimum learning rate multiplier
            max_effective_lr: Maximum learning rate multiplier
            enable_coupling_updates: P7.4: Enable coupling matrix updates
        """
        # Initialize neuromodulators first (needed for eligibility trace integration)
        self.neuromodulators = neuromodulator_orchestra or NeuromodulatorOrchestra()

        # LEARNING-HIGH-003 FIX: Use serotonin's eligibility trace if no explicit one provided
        # This ensures traces updated during retrieval (via process_retrieval) are visible
        # to the three-factor learning rule, enabling proper temporal credit assignment.
        if eligibility_trace is not None:
            self.eligibility = eligibility_trace
        elif hasattr(self.neuromodulators, "serotonin") and hasattr(self.neuromodulators.serotonin, "_eligibility_tracer"):
            # Share the eligibility trace with serotonin system
            self.eligibility = self.neuromodulators.serotonin._eligibility_tracer
            logger.debug("ThreeFactorLearningRule using serotonin's eligibility trace for integration")
        else:
            self.eligibility = EligibilityTraceSystem()

        self.dopamine = dopamine_system or DopamineSystem()

        # P7.4: Coupling matrix for energy-based learning
        self._coupling = coupling
        self._enable_coupling_updates = enable_coupling_updates

        # Neuromodulator component weights (sum to 1.0)
        # CRASH-014 FIX: Validate weights are positive before division
        total_weight = ach_weight + ne_weight + serotonin_weight
        if total_weight <= 0:
            raise ValueError(
                f"Neuromodulator weights must sum to positive value, "
                f"got ach={ach_weight}, ne={ne_weight}, serotonin={serotonin_weight}"
            )
        self.ach_weight = ach_weight / total_weight
        self.ne_weight = ne_weight / total_weight
        self.serotonin_weight = serotonin_weight / total_weight

        # Bounds
        self.min_eligibility_threshold = min_eligibility_threshold
        self.min_effective_lr = min_effective_lr
        self.max_effective_lr = max_effective_lr

        # MEM-007 FIX: Bounded history for analysis
        self._signal_history: list[ThreeFactorSignal] = []
        self._max_history_size = 10000  # Maximum signals to retain

    def set_coupling(self, coupling: LearnableCoupling) -> None:
        """
        P7.4: Set coupling matrix for energy-based updates.

        Args:
            coupling: LearnableCoupling instance to update during learning
        """
        self._coupling = coupling
        logger.debug("P7.4: ThreeFactorLearningRule coupling connected")

    def mark_active(self, memory_id: str, activity: float = 1.0) -> None:
        """
        Mark a memory as active (add/update eligibility trace).

        Call this when a memory is retrieved or used.

        Args:
            memory_id: Memory that was activated
            activity: Activity level (default 1.0)
        """
        self.eligibility.update(memory_id, activity)

    def step(self, dt: float | None = None) -> None:
        """
        Advance time, decaying eligibility traces.

        Args:
            dt: Time delta in seconds (auto-computed if None)
        """
        self.eligibility.step(dt)

    def get_eligibility(self, memory_id: str) -> float:
        """
        Get current eligibility for a memory.

        Args:
            memory_id: Memory to check

        Returns:
            Eligibility value [0, 1]
        """
        return self.eligibility.get_trace(memory_id)

    def _compute_neuromod_gate(
        self,
        state: NeuromodulatorState | None = None
    ) -> tuple[float, float, float, float]:
        """
        Compute neuromodulator gate from current state.

        Returns:
            Tuple of (combined_gate, ach_factor, ne_factor, 5ht_factor)
        """
        if state is None:
            state = self.neuromodulators.get_current_state()

        if state is None:
            # No state available, use neutral defaults
            return (1.0, 1.0, 1.0, 1.0)

        # ACh: encoding mode boosts learning, retrieval mode reduces it
        if state.acetylcholine_mode == "encoding":
            ach_factor = 1.5
        elif state.acetylcholine_mode == "retrieval":
            ach_factor = 0.6
        else:  # balanced
            ach_factor = 1.0

        # NE: arousal directly modulates (already in [0.5, 2.0] range typically)
        ne_factor = state.norepinephrine_gain

        # 5-HT: inverted U - moderate mood is optimal for learning
        # Too low (depressed) or too high (manic) reduces learning
        mood_deviation = abs(state.serotonin_mood - 0.5)
        serotonin_factor = 1.0 - 0.5 * mood_deviation  # [0.75, 1.0]

        # Combine with weights
        combined = (
            self.ach_weight * ach_factor +
            self.ne_weight * ne_factor +
            self.serotonin_weight * serotonin_factor
        )

        return (combined, ach_factor, ne_factor, serotonin_factor)

    def compute(
        self,
        memory_id: UUID,
        base_lr: float,
        outcome: float | None = None,
        neuromod_state: NeuromodulatorState | None = None,
        precomputed_rpe: RewardPredictionError | None = None
    ) -> ThreeFactorSignal:
        """
        Compute the three-factor learning signal for a memory.

        This is the main entry point. Returns a signal containing
        all three factors and the computed effective learning rate.

        Args:
            memory_id: Memory to compute learning rate for
            base_lr: Base learning rate to modulate
            outcome: Observed outcome [0, 1] for dopamine RPE
            neuromod_state: Override neuromodulator state
            precomputed_rpe: Use precomputed RPE instead of computing

        Returns:
            ThreeFactorSignal with all components and effective LR
        """
        memory_id_str = str(memory_id)

        # Factor 1: Eligibility trace
        eligibility = self.get_eligibility(memory_id_str)

        # Factor 2: Neuromodulator gate
        combined_gate, ach_factor, ne_factor, serotonin_factor = (
            self._compute_neuromod_gate(neuromod_state)
        )

        # Factor 3: Dopamine surprise
        if precomputed_rpe is not None:
            rpe = precomputed_rpe
        elif outcome is not None:
            rpe = self.dopamine.compute_rpe(memory_id, outcome)
        else:
            # No outcome, assume neutral
            rpe = RewardPredictionError(
                memory_id=memory_id,
                expected=0.5,
                actual=0.5,
                rpe=0.0
            )

        # Dopamine surprise is |RPE|, with minimum floor
        dopamine_surprise = max(rpe.surprise_magnitude, 0.1)

        # Compute effective learning rate multiplier
        # Three-factor rule: eligibility * neuromod * dopamine
        effective_multiplier = eligibility * combined_gate * dopamine_surprise

        # DATA-005 FIX: Validate computed values before using
        _validate_scalar(effective_multiplier, "effective_multiplier")

        # Apply bounds
        if eligibility < self.min_eligibility_threshold:
            # Below eligibility threshold, minimal learning
            effective_multiplier = self.min_effective_lr * 0.1
        else:
            effective_multiplier = np.clip(
                effective_multiplier,
                self.min_effective_lr,
                self.max_effective_lr
            )

        signal = ThreeFactorSignal(
            memory_id=memory_id,
            eligibility=eligibility,
            neuromod_gate=combined_gate,
            ach_mode_factor=ach_factor,
            ne_arousal_factor=ne_factor,
            serotonin_mood_factor=serotonin_factor,
            dopamine_surprise=dopamine_surprise,
            rpe_raw=rpe.rpe,
            effective_lr_multiplier=float(effective_multiplier)
        )

        self._signal_history.append(signal)

        # MEM-007 FIX: Trim history if over limit
        if len(self._signal_history) > self._max_history_size:
            self._signal_history = self._signal_history[-self._max_history_size:]

        # P7.4: Update coupling matrix via energy-based learning
        if self._enable_coupling_updates and self._coupling is not None:
            try:
                # Get NT state from neuromodulator orchestra
                neuromod_state = neuromod_state or self.neuromodulators.get_current_state()
                if neuromod_state is not None:
                    # Convert NeuromodulatorState to NT array for coupling
                    nt_array = np.array([
                        neuromod_state.dopamine_level,
                        neuromod_state.serotonin_mood,
                        neuromod_state.acetylcholine_level,
                        neuromod_state.norepinephrine_gain,
                        0.5,  # GABA (default baseline)
                        0.5,  # Glutamate (default baseline)
                    ], dtype=np.float32)

                    # Update coupling using energy-based contrastive divergence
                    # Pass RPE as modulator and eligibility as trace
                    self._coupling.update_from_energy(
                        data_state=nt_array,
                        n_gibbs_steps=3,  # Quick CD-3
                        eligibility=None,  # Use coupling's internal trace
                    )
                    logger.debug("P7.4: Coupling updated via energy-based learning")
            except Exception as e:
                logger.warning(f"P7.4: Coupling update failed: {e}")

        logger.debug(
            f"Three-factor signal: memory={memory_id_str[:8]}, "
            f"elig={eligibility:.3f}, gate={combined_gate:.3f}, "
            f"DA={dopamine_surprise:.3f}, effective_lr_mult={effective_multiplier:.3f}"
        )

        return signal

    def compute_effective_lr(
        self,
        memory_id: UUID,
        base_lr: float,
        outcome: float | None = None
    ) -> float:
        """
        Convenience method: compute and return just the effective LR.

        Args:
            memory_id: Memory to compute learning rate for
            base_lr: Base learning rate
            outcome: Observed outcome

        Returns:
            Effective learning rate (base_lr * multiplier)
        """
        signal = self.compute(memory_id, base_lr, outcome)
        return base_lr * signal.effective_lr_multiplier

    def batch_compute(
        self,
        memory_ids: list[UUID],
        base_lr: float,
        outcomes: dict[str, float] | None = None
    ) -> dict[str, ThreeFactorSignal]:
        """
        Compute three-factor signals for multiple memories.

        Args:
            memory_ids: List of memory IDs
            base_lr: Base learning rate
            outcomes: Optional memory_id -> outcome mapping

        Returns:
            memory_id -> ThreeFactorSignal mapping
        """
        results = {}

        for memory_id in memory_ids:
            outcome = None
            if outcomes:
                outcome = outcomes.get(str(memory_id))

            signal = self.compute(memory_id, base_lr, outcome)
            results[str(memory_id)] = signal

        return results

    def update_dopamine_expectations(
        self,
        memory_id: UUID,
        outcome: float
    ) -> float:
        """
        Update dopamine value expectations after outcome.

        Args:
            memory_id: Memory that received outcome
            outcome: Observed outcome

        Returns:
            New expected value
        """
        return self.dopamine.update_expectations(memory_id, outcome)

    def get_stats(self) -> dict:
        """Get combined statistics from all subsystems."""
        eligibility_stats = self.eligibility.get_stats()

        # Compute signal history stats
        if self._signal_history:
            recent_signals = self._signal_history[-100:]
            avg_eligibility = np.mean([s.eligibility for s in recent_signals])
            avg_gate = np.mean([s.neuromod_gate for s in recent_signals])
            avg_surprise = np.mean([s.dopamine_surprise for s in recent_signals])
            avg_effective_lr = np.mean([s.effective_lr_multiplier for s in recent_signals])
        else:
            avg_eligibility = avg_gate = avg_surprise = avg_effective_lr = 0.0

        return {
            "eligibility": eligibility_stats,
            "neuromodulators": self.neuromodulators.get_stats(),
            "dopamine": self.dopamine.get_stats(),
            "three_factor": {
                "total_signals": len(self._signal_history),
                "avg_eligibility": float(avg_eligibility),
                "avg_neuromod_gate": float(avg_gate),
                "avg_dopamine_surprise": float(avg_surprise),
                "avg_effective_lr_mult": float(avg_effective_lr)
            }
        }

    def clear_history(self) -> None:
        """Clear signal history to free memory."""
        self._signal_history.clear()


class ThreeFactorReconsolidation:
    """
    Reconsolidation with full three-factor learning integration.

    Extends DopamineModulatedReconsolidation with:
    - Eligibility trace gating
    - Full neuromodulator state integration
    - Three-factor learning rate computation

    This is the biologically-plausible version of memory updating.
    """

    def __init__(
        self,
        three_factor: ThreeFactorLearningRule | None = None,
        base_learning_rate: float = 0.01,
        max_update_magnitude: float = 0.1,
        cooldown_hours: float = 1.0,
    ):
        """
        Initialize three-factor reconsolidation.

        Args:
            three_factor: Three-factor rule (created if None)
            base_learning_rate: Base learning rate for updates
            max_update_magnitude: Maximum embedding update magnitude
            cooldown_hours: Minimum hours between updates
        """
        self.three_factor = three_factor or ThreeFactorLearningRule()
        self.base_learning_rate = base_learning_rate
        self.max_update_magnitude = max_update_magnitude
        self.cooldown_hours = cooldown_hours

        # Cooldown tracking with size limit (MEM-007 FIX)
        self._last_update: dict[str, datetime] = {}
        self._max_cooldown_entries = 10000

    def should_update(self, memory_id: UUID) -> bool:
        """Check if memory is eligible for update (cooldown elapsed)."""
        mem_id_str = str(memory_id)

        if mem_id_str not in self._last_update:
            return True

        elapsed = datetime.now() - self._last_update[mem_id_str]
        return elapsed.total_seconds() / 3600 >= self.cooldown_hours

    def _cleanup_cooldowns(self) -> None:
        """
        Remove oldest cooldown entries to enforce memory limit.

        MEM-007 FIX: Prevent unbounded growth of _last_update dict.
        """
        now = datetime.now()
        cooldown_seconds = self.cooldown_hours * 3600

        # Remove expired entries first
        expired_keys = [
            key for key, timestamp in self._last_update.items()
            if (now - timestamp).total_seconds() > cooldown_seconds
        ]
        for key in expired_keys:
            del self._last_update[key]

        # If still over limit, remove oldest entries
        if len(self._last_update) > self._max_cooldown_entries:
            sorted_entries = sorted(
                self._last_update.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self._last_update = dict(sorted_entries[:self._max_cooldown_entries])

    def reconsolidate(
        self,
        memory_id: UUID,
        memory_embedding: np.ndarray,
        query_embedding: np.ndarray,
        outcome_score: float,
        importance: float = 0.0
    ) -> np.ndarray | None:
        """
        Update memory embedding using three-factor learning.

        Args:
            memory_id: Memory being updated
            memory_embedding: Current embedding
            query_embedding: Query that retrieved this memory
            outcome_score: Observed outcome [0, 1]
            importance: Memory importance for protection

        Returns:
            Updated embedding, or None if skipped
        """
        # Check cooldown
        if not self.should_update(memory_id):
            return None

        # Get three-factor learning rate
        signal = self.three_factor.compute(
            memory_id=memory_id,
            base_lr=self.base_learning_rate,
            outcome=outcome_score
        )

        # Skip if eligibility is too low
        if signal.eligibility < self.three_factor.min_eligibility_threshold:
            logger.debug(f"Skipping {memory_id}: low eligibility {signal.eligibility:.4f}")
            return None

        # Compute advantage (outcome relative to baseline)
        advantage = outcome_score - 0.5

        # Skip neutral outcomes
        if abs(advantage) < 0.01:
            return None

        # Compute update direction
        direction = query_embedding - memory_embedding
        norm = np.linalg.norm(direction)

        if norm < 1e-8:
            return None

        direction = direction / norm

        # Apply three-factor learning rate with importance protection
        effective_lr = self.base_learning_rate * signal.effective_lr_multiplier
        effective_lr = effective_lr / (1.0 + importance)  # Importance protection

        # Compute update
        update = effective_lr * advantage * direction

        # Clip magnitude
        update_norm = np.linalg.norm(update)
        if update_norm > self.max_update_magnitude:
            update = update * (self.max_update_magnitude / update_norm)

        # Apply update
        new_embedding = memory_embedding + update

        # Normalize to unit sphere
        new_norm = np.linalg.norm(new_embedding)
        if new_norm > 0:
            new_embedding = new_embedding / new_norm

        # Update tracking
        self._last_update[str(memory_id)] = datetime.now()

        # MEM-007 FIX: Cleanup old cooldown entries
        if len(self._last_update) > self._max_cooldown_entries:
            self._cleanup_cooldowns()

        # Update dopamine expectations
        self.three_factor.update_dopamine_expectations(memory_id, outcome_score)

        logger.debug(
            f"Three-factor reconsolidation: memory={str(memory_id)[:8]}, "
            f"advantage={advantage:.3f}, effective_lr={effective_lr:.4f}, "
            f"update_magnitude={np.linalg.norm(update):.4f}"
        )

        return new_embedding


# Factory function
def create_three_factor_rule(
    use_layered_traces: bool = False,
    eligibility_config: dict | None = None,
    neuromodulator_config: dict | None = None,
    dopamine_config: dict | None = None,
    **kwargs
) -> ThreeFactorLearningRule:
    """
    Create a configured three-factor learning rule.

    Args:
        use_layered_traces: Use LayeredEligibilityTrace for fast/slow dynamics
        eligibility_config: Config for eligibility trace system
        neuromodulator_config: Config for neuromodulator orchestra
        dopamine_config: Config for dopamine system
        **kwargs: Additional args passed to ThreeFactorLearningRule

    Returns:
        Configured ThreeFactorLearningRule
    """
    from ww.learning.neuromodulators import create_neuromodulator_orchestra

    # Create eligibility trace
    if use_layered_traces:
        eligibility = LayeredEligibilityTrace(**(eligibility_config or {}))
    else:
        eligibility = EligibilityTraceSystem(**(eligibility_config or {}))

    # Create neuromodulator orchestra
    neuromodulators = create_neuromodulator_orchestra(**(neuromodulator_config or {}))

    # Create dopamine system
    dopamine = DopamineSystem(**(dopamine_config or {}))

    return ThreeFactorLearningRule(
        eligibility_trace=eligibility,
        neuromodulator_orchestra=neuromodulators,
        dopamine_system=dopamine,
        **kwargs
    )


__all__ = [
    "ThreeFactorLearningRule",
    "ThreeFactorReconsolidation",
    "ThreeFactorSignal",
    "create_three_factor_rule",
]
