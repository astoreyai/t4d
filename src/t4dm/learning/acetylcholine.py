"""
Acetylcholine-like Encoding/Retrieval Mode Switch for T4DM.

Biological Basis:
- High ACh in hippocampus promotes encoding, suppresses retrieval
- Low ACh promotes retrieval/pattern completion from cortex
- ACh levels modulated by novelty, attention, and uncertainty
- Sleep-wake cycle shifts ACh levels (low during SWS consolidation)

Implementation:
- Tracks encoding vs retrieval demands from query characteristics
- Modulates balance between new learning and pattern completion
- Influences reconsolidation (high ACh = more labile memories)
- Gates attention between memory systems

Integration Points:
1. EpisodicMemory: Weight encoding vs pattern completion
2. PatternCompletion (CA3): Modulate completion strength
3. ReconsolidationEngine: Only reconsolidate in encoding mode
4. UnifiedMemory: Attention-weight different memory sources

Reference: Hasselmo (2006) - The role of ACh in memory
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class CognitiveMode(str, Enum):
    """Current cognitive mode based on ACh level."""
    ENCODING = "encoding"      # High ACh: prioritize learning
    BALANCED = "balanced"      # Moderate ACh: normal operation
    RETRIEVAL = "retrieval"    # Low ACh: prioritize recall


@dataclass
class AcetylcholineState:
    """Current state of the acetylcholine system."""

    ach_level: float  # ACh concentration [0, 1]
    mode: CognitiveMode  # Current cognitive mode
    encoding_weight: float  # Weight for new encoding [0, 1]
    retrieval_weight: float  # Weight for pattern completion [0, 1]
    attention_gate: float  # Attention gating strength [0, 1]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def learning_rate_modifier(self) -> float:
        """Higher ACh = higher learning rate."""
        return 0.5 + self.ach_level

    @property
    def pattern_completion_strength(self) -> float:
        """Lower ACh = stronger pattern completion."""
        return 1.0 - self.ach_level * 0.6


class AcetylcholineSystem:
    """
    Encoding/retrieval mode switch inspired by basal forebrain ACh.

    The cholinergic system dynamically balances:
    1. Encoding new information (high ACh)
    2. Retrieving stored patterns (low ACh)
    3. Attention allocation (moderate-high ACh)

    Mode detection is based on:
    - Query type (question vs statement)
    - Novelty (novel content -> encoding)
    - Memory match quality (good matches -> retrieval)
    - Explicit signals (user marking something important)

    High ACh (encoding mode):
    - Prioritize new information over stored patterns
    - Strengthen hippocampal-like fast learning
    - Reduce cortical pattern completion

    Low ACh (retrieval mode):
    - Prioritize stored patterns over new encoding
    - Enhance pattern completion from partial cues
    - Reduce interference from new information
    """

    def __init__(
        self,
        baseline_ach: float = 0.5,
        encoding_threshold: float = 0.7,
        retrieval_threshold: float = 0.3,
        adaptation_rate: float = 0.2,
        min_ach: float = 0.1,
        max_ach: float = 0.9
    ):
        """
        Initialize acetylcholine system.

        Args:
            baseline_ach: Default ACh level
            encoding_threshold: ACh level above which encoding dominates
            retrieval_threshold: ACh level below which retrieval dominates
            adaptation_rate: How fast ACh adapts to demands
            min_ach: Minimum ACh level
            max_ach: Maximum ACh level
        """
        self.baseline_ach = baseline_ach
        self.encoding_threshold = encoding_threshold
        self.retrieval_threshold = retrieval_threshold
        self.adaptation_rate = adaptation_rate
        self.min_ach = min_ach
        self.max_ach = max_ach

        # Current ACh level
        self._ach_level = baseline_ach
        self._current_state: AcetylcholineState | None = None

        # History for analysis
        self._state_history: list[AcetylcholineState] = []

    def compute_encoding_demand(
        self,
        query_novelty: float,
        is_statement: bool = False,
        explicit_importance: float | None = None
    ) -> float:
        """
        Compute demand for encoding mode.

        Args:
            query_novelty: Novelty score from NE system [0, 1]
            is_statement: True if query is a statement (not question)
            explicit_importance: Optional user-indicated importance

        Returns:
            Encoding demand [0, 1]
        """
        demand = 0.0

        # Novel content demands encoding
        demand += 0.4 * query_novelty

        # Statements (vs questions) lean toward encoding
        if is_statement:
            demand += 0.2

        # Explicit importance strongly favors encoding
        if explicit_importance is not None:
            demand += 0.4 * explicit_importance

        return min(1.0, demand)

    def compute_retrieval_demand(
        self,
        is_question: bool = False,
        memory_match_quality: float | None = None,
        query_specificity: float = 0.5
    ) -> float:
        """
        Compute demand for retrieval mode.

        Args:
            is_question: True if query is a question
            memory_match_quality: Quality of best memory match [0, 1]
            query_specificity: How specific/answerable the query is [0, 1]

        Returns:
            Retrieval demand [0, 1]
        """
        demand = 0.0

        # Questions demand retrieval
        if is_question:
            demand += 0.3

        # Good memory matches suggest retrieval
        if memory_match_quality is not None:
            demand += 0.4 * memory_match_quality

        # Specific queries lean toward retrieval
        demand += 0.2 * query_specificity

        return min(1.0, demand)

    def update(
        self,
        encoding_demand: float,
        retrieval_demand: float,
        arousal_gain: float = 1.0
    ) -> AcetylcholineState:
        """
        Update ACh level based on current demands.

        Args:
            encoding_demand: Demand for encoding mode [0, 1]
            retrieval_demand: Demand for retrieval mode [0, 1]
            arousal_gain: Gain from NE system (high arousal boosts ACh)

        Returns:
            Current ACh state
        """
        # Compute target ACh level
        # High encoding demand -> high ACh
        # High retrieval demand -> low ACh
        demand_diff = encoding_demand - retrieval_demand
        target_ach = self.baseline_ach + 0.4 * demand_diff

        # Arousal modulates ACh (novel/urgent -> higher ACh)
        target_ach *= arousal_gain

        # Clamp to bounds
        target_ach = np.clip(target_ach, self.min_ach, self.max_ach)

        # Adapt toward target
        self._ach_level += self.adaptation_rate * (target_ach - self._ach_level)

        # Determine mode
        if self._ach_level >= self.encoding_threshold:
            mode = CognitiveMode.ENCODING
        elif self._ach_level <= self.retrieval_threshold:
            mode = CognitiveMode.RETRIEVAL
        else:
            mode = CognitiveMode.BALANCED

        # Compute weights
        # Encoding weight increases with ACh
        encoding_weight = self._ach_level
        retrieval_weight = 1.0 - self._ach_level * 0.6

        # Attention gate (ACh enhances attention)
        attention_gate = 0.5 + 0.5 * self._ach_level

        # Create state
        self._current_state = AcetylcholineState(
            ach_level=float(self._ach_level),
            mode=mode,
            encoding_weight=float(encoding_weight),
            retrieval_weight=float(retrieval_weight),
            attention_gate=float(attention_gate)
        )

        self._state_history.append(self._current_state)

        logger.debug(
            f"ACh update: level={self._ach_level:.3f}, mode={mode.value}, "
            f"enc={encoding_weight:.2f}, ret={retrieval_weight:.2f}"
        )

        return self._current_state

    def get_current_mode(self) -> CognitiveMode:
        """Get current cognitive mode."""
        if self._current_state is None:
            return CognitiveMode.BALANCED
        return self._current_state.mode

    def get_current_level(self) -> float:
        """Get current ACh level."""
        return self._ach_level

    def should_prioritize_encoding(self) -> bool:
        """Check if system should prioritize encoding over retrieval."""
        return self.get_current_mode() == CognitiveMode.ENCODING

    def should_prioritize_retrieval(self) -> bool:
        """Check if system should prioritize retrieval over encoding."""
        return self.get_current_mode() == CognitiveMode.RETRIEVAL

    def modulate_learning_rate(self, base_lr: float) -> float:
        """
        Modulate learning rate by ACh level.

        Higher ACh = higher learning rate (encoding mode).

        Args:
            base_lr: Base learning rate

        Returns:
            Modulated learning rate
        """
        if self._current_state is None:
            return base_lr
        return base_lr * self._current_state.learning_rate_modifier

    def modulate_pattern_completion(self, base_strength: float) -> float:
        """
        Modulate pattern completion strength by ACh level.

        Lower ACh = stronger pattern completion (retrieval mode).

        Args:
            base_strength: Base pattern completion strength

        Returns:
            Modulated strength
        """
        if self._current_state is None:
            return base_strength
        return base_strength * self._current_state.pattern_completion_strength

    def get_attention_weights(
        self,
        memory_sources: list[str]
    ) -> dict[str, float]:
        """
        Compute attention weights for different memory sources.

        In encoding mode: weight hippocampal (episodic) sources higher
        In retrieval mode: weight cortical (semantic) sources higher

        Args:
            memory_sources: List of memory source names

        Returns:
            Source name -> attention weight
        """
        if self._current_state is None:
            return dict.fromkeys(memory_sources, 1.0)

        weights = {}
        mode = self._current_state.mode

        for src in memory_sources:
            src_lower = src.lower()
            if mode == CognitiveMode.ENCODING:
                # Encoding: boost episodic, reduce semantic
                if "episodic" in src_lower:
                    weights[src] = 1.2
                elif "semantic" in src_lower:
                    weights[src] = 0.8
                else:
                    weights[src] = 1.0
            elif mode == CognitiveMode.RETRIEVAL:
                # Retrieval: boost semantic, reduce episodic
                if "semantic" in src_lower:
                    weights[src] = 1.2
                elif "episodic" in src_lower:
                    weights[src] = 0.8
                else:
                    weights[src] = 1.0
            else:
                weights[src] = 1.0

        return weights

    def get_reconsolidation_eligibility(self) -> float:
        """
        Get eligibility for reconsolidation based on mode.

        Reconsolidation requires labile state (high ACh).

        Returns:
            Eligibility [0, 1] where 1 = fully eligible
        """
        if self._current_state is None:
            return 0.5

        # Higher ACh = more eligible for reconsolidation
        return self._current_state.ach_level

    def get_stats(self) -> dict:
        """Get acetylcholine system statistics."""
        if not self._state_history:
            return {
                "total_updates": 0,
                "avg_ach": self.baseline_ach,
                "mode_counts": {},
                "current_mode": "balanced",
                "config": self.get_config(),
            }

        mode_counts: dict[str, int] = {}
        for state in self._state_history:
            mode_counts[state.mode.value] = mode_counts.get(state.mode.value, 0) + 1

        return {
            "total_updates": len(self._state_history),
            "avg_ach": float(np.mean([s.ach_level for s in self._state_history])),
            "mode_counts": mode_counts,
            "current_mode": self._current_state.mode.value if self._current_state else "balanced",
            "config": self.get_config(),
        }

    # ==================== Runtime Configuration Setters ====================

    def force_mode(self, mode: CognitiveMode | str) -> AcetylcholineState:
        """
        Force cognitive mode directly.

        Args:
            mode: Target mode (CognitiveMode or string: "encoding", "balanced", "retrieval")

        Returns:
            New ACh state
        """
        if isinstance(mode, str):
            mode = CognitiveMode(mode.lower())

        # Set ACh level to match mode
        if mode == CognitiveMode.ENCODING:
            self._ach_level = self.encoding_threshold + 0.1
        elif mode == CognitiveMode.RETRIEVAL:
            self._ach_level = self.retrieval_threshold - 0.1
        else:
            self._ach_level = self.baseline_ach

        self._ach_level = float(np.clip(self._ach_level, self.min_ach, self.max_ach))

        # Compute weights
        encoding_weight = self._ach_level
        retrieval_weight = 1.0 - self._ach_level * 0.6
        attention_gate = 0.5 + 0.5 * self._ach_level

        self._current_state = AcetylcholineState(
            ach_level=float(self._ach_level),
            mode=mode,
            encoding_weight=float(encoding_weight),
            retrieval_weight=float(retrieval_weight),
            attention_gate=float(attention_gate),
        )
        self._state_history.append(self._current_state)
        logger.info(f"ACh mode forced to {mode.value}, level={self._ach_level:.3f}")
        return self._current_state

    def set_thresholds(self, encoding: float, retrieval: float) -> None:
        """
        Set mode transition thresholds.

        Args:
            encoding: Encoding threshold [0.5, 0.9]
            retrieval: Retrieval threshold [0.1, 0.5]
        """
        self.encoding_threshold = float(np.clip(encoding, 0.5, 0.9))
        self.retrieval_threshold = float(np.clip(retrieval, 0.1, 0.5))
        # Ensure encoding > retrieval
        if self.encoding_threshold <= self.retrieval_threshold:
            self.encoding_threshold = self.retrieval_threshold + 0.2
        logger.info(f"ACh thresholds set: encoding={self.encoding_threshold}, retrieval={self.retrieval_threshold}")

    def set_adaptation_rate(self, rate: float) -> None:
        """
        Set mode adaptation rate.

        Args:
            rate: Adaptation rate [0.01, 1.0]
        """
        self.adaptation_rate = float(np.clip(rate, 0.01, 1.0))
        logger.info(f"ACh adaptation_rate set to {self.adaptation_rate}")

    def set_baseline_ach(self, level: float) -> None:
        """
        Set baseline ACh level.

        Args:
            level: Baseline level [0.1, 0.9]
        """
        self.baseline_ach = float(np.clip(level, 0.1, 0.9))
        logger.info(f"ACh baseline_ach set to {self.baseline_ach}")

    def set_ach_bounds(self, min_ach: float, max_ach: float) -> None:
        """
        Set ACh level bounds.

        Args:
            min_ach: Minimum ACh [0.0, 0.5]
            max_ach: Maximum ACh [0.5, 1.0]
        """
        self.min_ach = float(np.clip(min_ach, 0.0, 0.5))
        self.max_ach = float(np.clip(max_ach, 0.5, 1.0))
        if self.max_ach <= self.min_ach:
            self.max_ach = self.min_ach + 0.3
        logger.info(f"ACh bounds set to [{self.min_ach}, {self.max_ach}]")

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "baseline_ach": self.baseline_ach,
            "encoding_threshold": self.encoding_threshold,
            "retrieval_threshold": self.retrieval_threshold,
            "adaptation_rate": self.adaptation_rate,
            "min_ach": self.min_ach,
            "max_ach": self.max_ach,
        }

    def reset(self) -> None:
        """Reset to baseline state."""
        self._ach_level = self.baseline_ach
        self._current_state = None
        self._state_history.clear()


__all__ = [
    "AcetylcholineState",
    "AcetylcholineSystem",
    "CognitiveMode",
]
