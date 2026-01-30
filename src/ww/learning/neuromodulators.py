"""
Integrated Neuromodulator System for World Weaver.

Coordinates the four neuromodulatory systems + E/I balance:
1. Dopamine - Reward prediction error, surprise-driven learning
2. Norepinephrine - Arousal, attention, novelty detection
3. Acetylcholine - Encoding/retrieval mode switching
4. Serotonin - Long-term credit assignment, patience

Plus excitatory/inhibitory (E/I) balance system:
5. GABA/Glutamate - Competitive inhibition, sparse representations
   (Note: GABA/Glutamate are NOT neuromodulators - they are fast
   neurotransmitters for E/I balance. Grouped here for convenience
   in orchestration, but mechanistically distinct.)

These systems interact as an orchestra:
- NE novelty -> ACh encoding mode
- Dopamine surprise -> NE arousal boost
- Serotonin mood -> Dopamine baseline adjustment
- GABA/Glutamate E/I dynamics apply to all retrieval outputs

Key insight: In the brain, these systems don't operate independently.
They form a coordinated ensemble that shapes how the entire memory
system operates. This module provides that coordination.

Usage:
    orchestra = NeuromodulatorOrchestra()

    # Process a query
    state = orchestra.process_query(query_embedding, is_question=True)

    # Process retrieval results
    sharpened_scores = orchestra.process_retrieval(
        retrieved_ids, scores, embeddings
    )

    # Process outcomes
    learning_signals = orchestra.process_outcome(
        memory_outcomes, session_outcome=0.8
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import numpy as np

from ww.learning.acetylcholine import AcetylcholineSystem
from ww.learning.dopamine import DopamineSystem
from ww.learning.inhibition import InhibitoryNetwork
from ww.learning.norepinephrine import NorepinephrineSystem
from ww.learning.serotonin import SerotoninSystem

logger = logging.getLogger(__name__)


@dataclass
class LearningParams:
    """
    Integrated learning parameters from all neuromodulatory systems.

    These parameters combine signals from dopamine (surprise), serotonin
    (patience), eligibility traces (temporal credit), and baseline learning
    rate to produce a unified learning signal for reconsolidation.

    The combination is multiplicative: signals gate each other rather than
    simply adding. This creates more selective learning where all signals
    must agree for strong updates.
    """

    effective_lr: float  # Combined learning rate from NE, ACh, 5-HT
    eligibility: float  # Eligibility trace strength [0, 1]
    surprise: float  # Dopamine surprise magnitude (|RPE|)
    patience: float  # Serotonin patience/long-term value
    rpe: float  # Raw dopamine RPE (signed)

    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def combined_learning_signal(self) -> float:
        """
        Multiplicative combination of all learning signals with bootstrap.

        All signals gate each other:
        - effective_lr: Baseline from arousal, mode, mood
        - eligibility: Temporal credit assignment
        - surprise: Dopamine prediction error magnitude
        - patience: Serotonin long-term value

        FIXES BUG-006: Adds bootstrap signal to prevent zero-learning deadlock.
        New memories with eligibility=0 or patience=0 would get combined_signal=0,
        making them unable to ever learn. Bootstrap allows small learning even
        without full eligibility/patience, breaking the catch-22.

        Returns:
            Combined multiplicative signal (with bootstrap floor)
        """
        # Primary multiplicative gating
        multiplicative = self.effective_lr * self.eligibility * self.surprise * self.patience

        # Bootstrap: allow small learning even without eligibility/patience
        # This prevents deadlock where new memories can never get learning signals
        bootstrap = 0.01 * self.effective_lr * max(0.1, self.surprise)

        return multiplicative + bootstrap

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "effective_lr": self.effective_lr,
            "eligibility": self.eligibility,
            "surprise": self.surprise,
            "patience": self.patience,
            "rpe": self.rpe,
            "combined_signal": self.combined_learning_signal,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class NeuromodulatorState:
    """Combined state of all neuromodulatory systems."""

    dopamine_rpe: float  # Recent reward prediction error
    norepinephrine_gain: float  # Current arousal gain
    acetylcholine_mode: str  # encoding/balanced/retrieval
    serotonin_mood: float  # Current mood
    inhibition_sparsity: float  # Recent sparsity

    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def effective_learning_rate(self) -> float:
        """
        Compute combined learning rate modifier.

        Combines contributions from:
        - NE arousal (high arousal = faster learning)
        - ACh mode (encoding mode = faster learning)
        - 5-HT mood (moderate mood = optimal learning)

        Note: This is the base learning rate. For multiplicative gating
        that includes eligibility and surprise, use get_learning_params().
        """
        # Base learning rate (NE provides gain/arousal boost)
        lr = 1.0 + (self.norepinephrine_gain - 1.0) * 0.5

        # ACh mode modulation - FIXES BIO-002: Both modes BOOST learning
        # Per bio spec: encoding gets stronger boost (4x), retrieval gets modest boost (1.5x)
        # NOT a reduction in retrieval mode - that was backwards
        if self.acetylcholine_mode == "encoding":
            lr *= 2.0  # Strong boost for encoding (down from 4x for stability)
        elif self.acetylcholine_mode == "retrieval":
            lr *= 1.2  # Modest boost for retrieval (NOT a reduction)
        # balanced mode: no additional boost

        # Mood modulation - FIXES BUG-005: Inverted-U with correct range
        # Moderate mood (0.5) is optimal, extremes reduce learning
        mood_deviation = abs(self.serotonin_mood - 0.5)
        mood_factor = 1.0 - mood_deviation  # Range [0.5, 1.0]
        lr *= (0.5 + 0.5 * mood_factor)  # Range [0.75, 1.0]

        return lr

    @property
    def exploration_exploitation_balance(self) -> float:
        """
        Compute balance between exploration and exploitation.

        Returns value in [-1, 1] where:
        - Negative = exploitation (use known patterns)
        - Positive = exploration (try new things)
        """
        balance = 0.0

        # High arousal -> exploration
        balance += (self.norepinephrine_gain - 1.0) * 0.5

        # Encoding mode -> exploration
        if self.acetylcholine_mode == "encoding":
            balance += 0.3
        elif self.acetylcholine_mode == "retrieval":
            balance -= 0.3

        # High mood -> slight exploitation (trust current approach)
        balance -= (self.serotonin_mood - 0.5) * 0.2

        return float(np.clip(balance, -1.0, 1.0))

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "dopamine_rpe": self.dopamine_rpe,
            "norepinephrine_gain": self.norepinephrine_gain,
            "acetylcholine_mode": self.acetylcholine_mode,
            "serotonin_mood": self.serotonin_mood,
            "inhibition_sparsity": self.inhibition_sparsity,
            "effective_learning_rate": self.effective_learning_rate,
            "exploration_balance": self.exploration_exploitation_balance,
            "timestamp": self.timestamp.isoformat()
        }


class NeuromodulatorOrchestra:
    """
    Coordinates all neuromodulatory systems for unified brain-like dynamics.

    The orchestra ensures systems interact appropriately:
    - High novelty (NE) triggers encoding mode (ACh)
    - Surprise (DA) boosts arousal (NE)
    - Outcomes update both DA expectations and 5-HT traces
    - All retrieval passes through inhibitory sharpening (GABA)

    This creates emergent behaviors:
    - Novel situations trigger heightened learning
    - Familiar situations trigger efficient retrieval
    - Surprising outcomes drive adaptation
    - Long-term successes build valued memories
    """

    def __init__(
        self,
        dopamine: DopamineSystem | None = None,
        norepinephrine: NorepinephrineSystem | None = None,
        acetylcholine: AcetylcholineSystem | None = None,
        serotonin: SerotoninSystem | None = None,
        inhibitory: InhibitoryNetwork | None = None
    ):
        """
        Initialize neuromodulator orchestra.

        Args:
            dopamine: Dopamine system (created if None)
            norepinephrine: NE system (created if None)
            acetylcholine: ACh system (created if None)
            serotonin: 5-HT system (created if None)
            inhibitory: GABA network (created if None)
        """
        self.dopamine = dopamine or DopamineSystem()
        self.norepinephrine = norepinephrine or NorepinephrineSystem()
        self.acetylcholine = acetylcholine or AcetylcholineSystem()
        self.serotonin = serotonin or SerotoninSystem()
        self.inhibitory = inhibitory or InhibitoryNetwork()

        self._current_state: NeuromodulatorState | None = None
        self._state_history: list[NeuromodulatorState] = []

        # RPE cache: Store computed RPEs for retrieval in get_learning_params
        # Fixes BUG-004: get_learning_params returned hardcoded 0.0
        from ww.learning.dopamine import RewardPredictionError
        self._rpe_cache: dict[str, RewardPredictionError] = {}

    def process_query(
        self,
        query_embedding: np.ndarray,
        is_question: bool = False,
        explicit_importance: float | None = None
    ) -> NeuromodulatorState:
        """
        Process a query through all neuromodulatory systems.

        This is called at the start of a retrieval operation.
        It updates arousal, determines encoding/retrieval mode,
        and prepares the system for processing.

        Args:
            query_embedding: Query vector
            is_question: Whether query is a question
            explicit_importance: User-indicated importance

        Returns:
            Combined neuromodulator state
        """
        # 1. Update norepinephrine (novelty/arousal)
        ne_state = self.norepinephrine.update(query_embedding)

        # 2. Update acetylcholine (encoding/retrieval mode)
        encoding_demand = self.acetylcholine.compute_encoding_demand(
            query_novelty=ne_state.novelty_score,
            is_statement=not is_question,
            explicit_importance=explicit_importance
        )
        retrieval_demand = self.acetylcholine.compute_retrieval_demand(
            is_question=is_question
        )
        ach_state = self.acetylcholine.update(
            encoding_demand=encoding_demand,
            retrieval_demand=retrieval_demand,
            arousal_gain=ne_state.combined_gain
        )

        # 3. Create combined state (DA and 5-HT update on outcomes)
        self._current_state = NeuromodulatorState(
            dopamine_rpe=0.0,  # Updated when outcome received
            norepinephrine_gain=ne_state.combined_gain,
            acetylcholine_mode=ach_state.mode.value,
            serotonin_mood=self.serotonin.get_current_mood(),
            inhibition_sparsity=0.0  # Updated after retrieval
        )

        self._state_history.append(self._current_state)

        logger.debug(
            f"Query processed: mode={ach_state.mode.value}, "
            f"gain={ne_state.combined_gain:.2f}, "
            f"mood={self.serotonin.get_current_mood():.2f}"
        )

        return self._current_state

    def process_retrieval(
        self,
        retrieved_ids: list[UUID],
        scores: dict[str, float],
        embeddings: dict[str, np.ndarray] | None = None
    ) -> dict[str, float]:
        """
        Process retrieval results through inhibitory dynamics.

        Also adds eligibility traces for serotonin credit assignment.

        Args:
            retrieved_ids: IDs of retrieved memories
            scores: Memory ID -> score
            embeddings: Optional embeddings for similarity-based inhibition

        Returns:
            Inhibited scores (sharpened distribution)
        """
        # Add eligibility traces for 5-HT long-term credit
        for mem_id in retrieved_ids:
            self.serotonin.add_eligibility(
                mem_id,
                strength=scores.get(str(mem_id), 0.5)
            )

        # Apply inhibitory dynamics
        result = self.inhibitory.apply_inhibition(scores, embeddings)

        # Update state
        if self._current_state:
            self._current_state.inhibition_sparsity = result.sparsity

        return result.inhibited_scores

    def process_outcome(
        self,
        memory_outcomes: dict[str, float],
        session_outcome: float | None = None
    ) -> dict[str, float]:
        """
        Process outcomes through DA (immediate) and 5-HT (long-term).

        This is called when outcomes are known for retrieved memories.
        Dopamine handles immediate prediction errors; serotonin
        distributes credit across time.

        MULTIPLICATIVE GATING: Signals gate each other rather than add.
        Combined signal = dopamine_surprise * serotonin_patience * eligibility

        Args:
            memory_outcomes: Memory ID -> immediate outcome
            session_outcome: Optional overall session outcome

        Returns:
            Memory ID -> combined learning signal (multiplicative)
        """
        learning_signals: dict[str, float] = {}

        # Process through dopamine (immediate RPE)
        rpes = self.dopamine.batch_compute_rpe(memory_outcomes)

        # Cache RPEs for retrieval in get_learning_params (fixes BUG-004)
        self._rpe_cache.update(rpes)

        # Update dopamine expectations
        self.dopamine.batch_update_expectations(memory_outcomes)

        # Process through serotonin (long-term credit)
        # LOGIC-007 FIX: Call receive_outcome to update eligibility traces and long-term values
        # but use separate components to avoid double counting
        if session_outcome is not None:
            self.serotonin.receive_outcome(session_outcome)

        # Multiplicative combination of all signals
        all_memory_ids = set(rpes.keys())

        for mem_id in all_memory_ids:
            # Get dopamine surprise (unsigned magnitude)
            rpe_obj = rpes.get(mem_id)
            if rpe_obj:
                dopamine_surprise = rpe_obj.surprise_magnitude
                if rpe_obj.surprise_magnitude > 0.3:
                    logger.debug(f"High surprise for {mem_id[:8]}: {rpe_obj.surprise_magnitude:.2f}")
            else:
                dopamine_surprise = 0.0

            # LOGIC-007 FIX: Get serotonin patience (long-term value) separately from eligibility
            # Previously used serotonin_credits which included trace, causing double counting
            try:
                mem_uuid = UUID(mem_id) if isinstance(mem_id, str) else mem_id
                serotonin_patience = self.serotonin.get_long_term_value(mem_uuid)
                eligibility_strength = self.serotonin.get_eligibility(mem_uuid)
            except (ValueError, AttributeError):
                serotonin_patience = 0.5  # Default neutral patience
                eligibility_strength = 0.0

            # Multiplicative gating: all signals must align for strong learning
            # If any signal is weak/absent, combined signal is weak
            # Each factor represents a different dimension:
            # - dopamine_surprise: how unexpected (prediction error magnitude)
            # - serotonin_patience: how valuable historically (long-term value)
            # - eligibility_strength: how recently active (temporal relevance)
            combined_signal = dopamine_surprise * serotonin_patience * eligibility_strength

            learning_signals[mem_id] = combined_signal

        # Update state with average RPE
        if self._current_state and rpes:
            avg_rpe = float(np.mean([r.rpe for r in rpes.values()]))
            self._current_state.dopamine_rpe = avg_rpe

        return learning_signals

    def get_signed_rpe(self, memory_id: UUID) -> float:
        """
        Get signed reward prediction error for a memory.

        LOGIC-010 FIX: Returns the signed RPE (not just magnitude) so that
        negative prediction errors can lead to depression/weakening.

        Args:
            memory_id: Memory to get RPE for

        Returns:
            Signed RPE: positive for better-than-expected, negative for worse
        """
        mem_id_str = str(memory_id)
        rpe_obj = self._rpe_cache.get(mem_id_str)
        if rpe_obj:
            return rpe_obj.rpe
        return 0.0

    def get_learning_params(self, memory_id: UUID) -> LearningParams:
        """
        Get integrated learning parameters for a specific memory.

        This is the main integration point for reconsolidation. It combines:
        - Effective learning rate (from NE, ACh, 5-HT state)
        - Eligibility trace strength (temporal credit)
        - Dopamine surprise/RPE (prediction error)
        - Serotonin patience (long-term value)

        All signals are combined multiplicatively for selective learning.

        Args:
            memory_id: Memory to get parameters for

        Returns:
            LearningParams with all integrated signals
        """
        if self._current_state is None:
            # Return neutral/default parameters
            return LearningParams(
                effective_lr=1.0,
                eligibility=0.0,
                surprise=0.0,
                patience=0.0,
                rpe=0.0
            )

        # Get eligibility trace strength
        eligibility = self.serotonin.get_eligibility(memory_id)

        # Get dopamine surprise and RPE from cache (populated by process_outcome)
        # Fixes BUG-004: Was returning hardcoded 0.0 which broke all learning
        mem_id_str = str(memory_id)
        rpe_obj = self._rpe_cache.get(mem_id_str)
        if rpe_obj:
            surprise = rpe_obj.surprise_magnitude
            rpe = rpe_obj.rpe
        else:
            # Fallback: compute RPE if we have expected value but no cached RPE
            self.dopamine.get_expected_value(memory_id)
            surprise = 0.1  # Small default surprise for uncached memories
            rpe = 0.0

        # Get serotonin long-term value
        patience = self.serotonin.get_long_term_value(memory_id)

        return LearningParams(
            effective_lr=self._current_state.effective_learning_rate,
            eligibility=eligibility,
            surprise=surprise,
            patience=patience,
            rpe=rpe
        )

    def get_learning_params_with_outcome(
        self,
        memory_id: UUID,
        outcome: float
    ) -> LearningParams:
        """
        Get learning parameters including outcome-based signals.

        This version computes dopamine RPE and surprise based on the
        provided outcome, giving a complete set of learning parameters.

        Args:
            memory_id: Memory to get parameters for
            outcome: Observed outcome [0, 1]

        Returns:
            Complete LearningParams with all signals
        """
        if self._current_state is None:
            return LearningParams(
                effective_lr=1.0,
                eligibility=0.0,
                surprise=0.0,
                patience=0.0,
                rpe=0.0
            )

        # Compute dopamine RPE
        rpe_obj = self.dopamine.compute_rpe(memory_id, outcome)

        # Get eligibility trace strength
        eligibility = self.serotonin.get_eligibility(memory_id)

        # Get serotonin long-term value
        patience = self.serotonin.get_long_term_value(memory_id)

        return LearningParams(
            effective_lr=self._current_state.effective_learning_rate,
            eligibility=eligibility,
            surprise=rpe_obj.surprise_magnitude,
            patience=patience,
            rpe=rpe_obj.rpe
        )

    def get_learning_rate(self, base_lr: float) -> float:
        """
        Get combined learning rate from all systems.

        Args:
            base_lr: Base learning rate

        Returns:
            Modulated learning rate
        """
        if self._current_state is None:
            return base_lr

        return base_lr * self._current_state.effective_learning_rate

    def get_retrieval_threshold(self, base_threshold: float) -> float:
        """
        Get modulated retrieval threshold.

        Lower threshold = broader search (more exploration).

        Args:
            base_threshold: Base similarity threshold

        Returns:
            Modulated threshold
        """
        return self.norepinephrine.modulate_retrieval_threshold(base_threshold)

    def should_encode(self) -> bool:
        """Check if system should prioritize encoding."""
        return self.acetylcholine.should_prioritize_encoding()

    def should_retrieve(self) -> bool:
        """Check if system should prioritize retrieval."""
        return self.acetylcholine.should_prioritize_retrieval()

    def get_current_state(self) -> NeuromodulatorState | None:
        """Get current neuromodulator state."""
        return self._current_state

    def get_attention_weights(self, sources: list[str]) -> dict[str, float]:
        """
        Get attention weights for different memory sources.

        Args:
            sources: List of source names

        Returns:
            Source -> attention weight
        """
        return self.acetylcholine.get_attention_weights(sources)

    def get_long_term_value(self, memory_id: UUID) -> float:
        """
        Get long-term value estimate for a memory.

        Args:
            memory_id: Memory to check

        Returns:
            Long-term value [0, 1]
        """
        return self.serotonin.get_long_term_value(memory_id)

    def get_expected_value(self, memory_id: UUID) -> float:
        """
        Get expected value for a memory (dopamine).

        Args:
            memory_id: Memory to check

        Returns:
            Expected value [0, 1]
        """
        return self.dopamine.get_expected_value(memory_id)

    def get_stats(self) -> dict:
        """Get combined statistics from all systems."""
        return {
            "dopamine": self.dopamine.get_stats(),
            "norepinephrine": self.norepinephrine.get_stats(),
            "acetylcholine": self.acetylcholine.get_stats(),
            "serotonin": self.serotonin.get_stats(),
            "inhibitory": self.inhibitory.get_stats(),
            "total_states": len(self._state_history),
            "current_state": (
                self._current_state.to_dict() if self._current_state else None
            )
        }

    def start_session(self, session_id: str, goal: str | None = None) -> None:
        """
        Start a new session for tracking.

        Args:
            session_id: Unique session identifier
            goal: Optional session goal description
        """
        self.serotonin.start_context(session_id, goal)

    def end_session(self, session_id: str, outcome: float) -> dict[str, float]:
        """
        End a session and distribute final credit.

        Args:
            session_id: Session to end
            outcome: Final session outcome [0, 1]

        Returns:
            Memory ID -> credit assigned
        """
        credits = self.serotonin.receive_outcome(outcome, session_id)
        self.serotonin.end_context(session_id)
        return credits

    def reset(self) -> None:
        """Reset all systems to baseline."""
        self.dopamine.reset_history()
        self.norepinephrine.reset_history()
        self.acetylcholine.reset()
        self.serotonin.reset()
        self.inhibitory.reset_history()
        self._current_state = None
        self._state_history.clear()


# Factory function
def create_neuromodulator_orchestra(
    dopamine_config: dict | None = None,
    norepinephrine_config: dict | None = None,
    acetylcholine_config: dict | None = None,
    serotonin_config: dict | None = None,
    inhibitory_config: dict | None = None
) -> NeuromodulatorOrchestra:
    """
    Create a configured neuromodulator orchestra.

    Args:
        dopamine_config: DopamineSystem configuration
        norepinephrine_config: NorepinephrineSystem configuration
        acetylcholine_config: AcetylcholineSystem configuration
        serotonin_config: SerotoninSystem configuration
        inhibitory_config: InhibitoryNetwork configuration

    Returns:
        Configured NeuromodulatorOrchestra
    """
    return NeuromodulatorOrchestra(
        dopamine=DopamineSystem(**(dopamine_config or {})),
        norepinephrine=NorepinephrineSystem(**(norepinephrine_config or {})),
        acetylcholine=AcetylcholineSystem(**(acetylcholine_config or {})),
        serotonin=SerotoninSystem(**(serotonin_config or {})),
        inhibitory=InhibitoryNetwork(**(inhibitory_config or {}))
    )


__all__ = [
    "LearningParams",
    "NeuromodulatorOrchestra",
    "NeuromodulatorState",
    "create_neuromodulator_orchestra",
]
