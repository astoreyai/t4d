"""
Norepinephrine-like Arousal/Attention System for T4DM.

Biological Basis:
- Locus coeruleus modulates cortical and hippocampal gain
- Tonic NE sets baseline arousal; phasic NE signals novelty
- High NE promotes exploration; low NE promotes exploitation
- NE enhances pattern separation in dentate gyrus

Implementation:
- Tracks query novelty via embedding distance from recent history
- Computes uncertainty from retrieval result entropy
- Modulates global gain affecting all downstream systems
- Influences exploration-exploitation balance in retrieval

Integration Points:
1. EpisodicMemory.recall(): Modulate retrieval threshold
2. DentateGyrus.encode(): Increase separation under high arousal
3. ReconsolidationEngine: Scale learning rate by gain
4. NeuroSymbolicReasoner: Bias toward exploration when novel
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArousalState:
    """Current state of the norepinephrine system."""

    tonic_level: float  # Baseline arousal [0, 1]
    phasic_burst: float  # Transient novelty signal [0, 1]
    combined_gain: float  # Effective gain multiplier
    novelty_score: float  # How novel was the current query
    uncertainty_score: float  # Entropy of retrieval results
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def exploration_bias(self) -> float:
        """Higher values favor exploration over exploitation."""
        return min(1.0, self.combined_gain * 0.5)


class NorepinephrineSystem:
    """
    Global arousal and attention modulator inspired by locus coeruleus.

    The LC-NE system implements adaptive gain control:
    - Novelty detection via query embedding distance
    - Uncertainty estimation from retrieval entropy
    - Global gain modulation affecting all learning systems
    - Exploration-exploitation balance

    Key insight: NE doesn't encode specific information, but rather
    modulates HOW information is processed across the entire system.

    Reference: Aston-Jones & Cohen (2005) - Adaptive Gain Theory
    """

    def __init__(
        self,
        baseline_arousal: float = 0.5,
        novelty_decay: float = 0.95,
        history_size: int = 50,
        phasic_decay: float = 0.7,
        min_gain: float = 0.5,
        max_gain: float = 2.0,
        uncertainty_weight: float = 0.3,
        novelty_weight: float = 0.7
    ):
        """
        Initialize norepinephrine system.

        Args:
            baseline_arousal: Tonic NE level when no novelty
            novelty_decay: How fast novelty habituates
            history_size: Number of recent queries to track
            phasic_decay: Decay rate for phasic bursts
            min_gain: Minimum gain multiplier
            max_gain: Maximum gain multiplier
            uncertainty_weight: Weight for uncertainty in gain
            novelty_weight: Weight for novelty in gain
        """
        self.baseline_arousal = baseline_arousal
        self.novelty_decay = novelty_decay
        self.history_size = history_size
        self.phasic_decay = phasic_decay
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.uncertainty_weight = uncertainty_weight
        self.novelty_weight = novelty_weight

        # Recent query history for novelty detection
        self._query_history: deque[np.ndarray] = deque(maxlen=history_size)

        # Current state
        self._tonic_level = baseline_arousal
        self._phasic_level = 0.0
        self._current_state: ArousalState | None = None

        # LEARNING-HIGH-001 FIX: Arousal history for analysis (bounded to prevent memory leak)
        self._arousal_history: deque[ArousalState] = deque(maxlen=1000)

    def compute_novelty(self, query_embedding: np.ndarray) -> float:
        """
        Compute novelty of query relative to recent history.

        Uses average distance from recent queries. High distance = high novelty.

        Args:
            query_embedding: Current query vector

        Returns:
            Novelty score [0, 1]
        """
        if not self._query_history:
            return 1.0  # First query is maximally novel

        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Compute distance from each historical query
        distances = []
        for hist_query in self._query_history:
            # Cosine distance: 1 - cosine_similarity
            similarity = np.dot(query, hist_query)
            distance = 1.0 - similarity
            distances.append(distance)

        # Recency-weighted average (recent queries matter more)
        weights = np.array([
            self.novelty_decay ** i
            for i in range(len(distances) - 1, -1, -1)
        ])
        weights = weights / weights.sum()

        avg_distance = np.average(distances, weights=weights)

        # Normalize to [0, 1] (distance ranges from 0 to 2 for unit vectors)
        novelty = min(1.0, avg_distance / 1.5)

        return float(novelty)

    def compute_uncertainty(
        self,
        retrieval_scores: list[float]
    ) -> float:
        """
        Compute uncertainty from retrieval result distribution.

        Uses entropy of score distribution. Uniform = high uncertainty,
        peaked = low uncertainty.

        Args:
            retrieval_scores: Scores of retrieved items

        Returns:
            Uncertainty score [0, 1]
        """
        if not retrieval_scores or len(retrieval_scores) < 2:
            return 0.5  # Default uncertainty

        scores = np.array(retrieval_scores, dtype=np.float32)

        # Normalize to probability distribution
        scores = scores - scores.min() + 1e-8
        probs = scores / scores.sum()

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize by maximum entropy (uniform distribution)
        max_entropy = np.log(len(scores))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5

        return float(normalized_entropy)

    def update(
        self,
        query_embedding: np.ndarray,
        retrieval_scores: list[float] | None = None,
        external_urgency: float = 0.0
    ) -> ArousalState:
        """
        Update arousal state based on current query.

        Args:
            query_embedding: Current query vector
            retrieval_scores: Optional scores from retrieval
            external_urgency: Optional external urgency signal [0, 1]

        Returns:
            Current arousal state
        """
        # Normalize query
        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Compute novelty
        novelty = self.compute_novelty(query)

        # Compute uncertainty if scores provided
        uncertainty = 0.5
        if retrieval_scores:
            uncertainty = self.compute_uncertainty(retrieval_scores)

        # Update phasic level (novelty burst)
        # Phasic = max of current novelty and decayed previous phasic
        self._phasic_level = max(
            novelty * 0.8,  # New novelty burst
            self._phasic_level * self.phasic_decay  # Decay previous
        )

        # Update tonic level (slow adaptation)
        # High sustained novelty raises tonic; low novelty returns to baseline
        target_tonic = self.baseline_arousal + 0.3 * novelty
        self._tonic_level += 0.1 * (target_tonic - self._tonic_level)

        # Combine into gain
        arousal = (
            self._tonic_level +
            self._phasic_level +
            0.2 * external_urgency
        )

        # Weight by uncertainty and novelty
        weighted_arousal = (
            self.novelty_weight * novelty +
            self.uncertainty_weight * uncertainty
        ) * arousal

        # Compute final gain
        combined_gain = self.min_gain + (self.max_gain - self.min_gain) * weighted_arousal
        combined_gain = np.clip(combined_gain, self.min_gain, self.max_gain)

        # Store query in history
        self._query_history.append(query.copy())

        # Create state
        self._current_state = ArousalState(
            tonic_level=float(self._tonic_level),
            phasic_burst=float(self._phasic_level),
            combined_gain=float(combined_gain),
            novelty_score=novelty,
            uncertainty_score=uncertainty
        )

        self._arousal_history.append(self._current_state)

        logger.debug(
            f"NE update: novelty={novelty:.3f}, uncertainty={uncertainty:.3f}, "
            f"gain={combined_gain:.3f}"
        )

        return self._current_state

    def get_current_gain(self) -> float:
        """Get current gain multiplier for downstream systems."""
        if self._current_state is None:
            return 1.0
        return self._current_state.combined_gain

    def get_current_novelty(self) -> float:
        """Get most recent novelty score."""
        if self._current_state is None:
            return 0.5
        return self._current_state.novelty_score

    def modulate_learning_rate(self, base_lr: float) -> float:
        """
        Modulate learning rate by current arousal.

        Higher arousal = higher learning rate (more plastic).

        Args:
            base_lr: Base learning rate

        Returns:
            Modulated learning rate
        """
        return base_lr * self.get_current_gain()

    def modulate_retrieval_threshold(self, base_threshold: float) -> float:
        """
        Modulate retrieval threshold by arousal.

        Higher arousal = lower threshold (broader search, more exploration).

        Args:
            base_threshold: Base similarity threshold

        Returns:
            Modulated threshold
        """
        gain = self.get_current_gain()
        # Inverse relationship: high gain = low threshold
        return base_threshold / gain

    def modulate_separation_strength(self, base_separation: float) -> float:
        """
        Modulate pattern separation strength by arousal.

        Higher arousal = stronger separation (reduce interference).

        Args:
            base_separation: Base separation magnitude

        Returns:
            Modulated separation
        """
        return base_separation * self.get_current_gain()

    def get_exploration_bias(self) -> float:
        """
        Get current exploration vs exploitation bias.

        Returns:
            Exploration bias [0, 1] where higher = more exploration
        """
        if self._current_state is None:
            return 0.5
        return self._current_state.exploration_bias

    def get_stats(self) -> dict:
        """Get norepinephrine system statistics."""
        if not self._arousal_history:
            return {
                "total_updates": 0,
                "avg_novelty": 0.0,
                "avg_uncertainty": 0.0,
                "avg_gain": 1.0,
                "current_tonic": self._tonic_level,
                "current_phasic": self._phasic_level,
                "config": self.get_config(),
            }

        return {
            "total_updates": len(self._arousal_history),
            "avg_novelty": float(np.mean([s.novelty_score for s in self._arousal_history])),
            "avg_uncertainty": float(np.mean([s.uncertainty_score for s in self._arousal_history])),
            "avg_gain": float(np.mean([s.combined_gain for s in self._arousal_history])),
            "current_tonic": self._tonic_level,
            "current_phasic": self._phasic_level,
            "config": self.get_config(),
        }

    # ==================== Runtime Configuration Setters ====================

    def set_baseline_arousal(self, level: float) -> None:
        """
        Set baseline arousal level.

        Args:
            level: Baseline arousal [0.0, 1.0]
        """
        self.baseline_arousal = float(np.clip(level, 0.0, 1.0))
        logger.info(f"NE baseline_arousal set to {self.baseline_arousal}")

    def set_arousal_bounds(self, min_gain: float, max_gain: float) -> None:
        """
        Set arousal gain bounds.

        Args:
            min_gain: Minimum gain [0.1, 1.0]
            max_gain: Maximum gain [1.0, 5.0]
        """
        self.min_gain = float(np.clip(min_gain, 0.1, 1.0))
        self.max_gain = float(np.clip(max_gain, 1.0, 5.0))
        if self.max_gain <= self.min_gain:
            self.max_gain = self.min_gain + 0.5
        logger.info(f"NE arousal bounds set to [{self.min_gain}, {self.max_gain}]")

    def set_arousal_gain(self, gain: float) -> None:
        """
        Directly set arousal gain, overriding automatic computation.

        Args:
            gain: Direct gain value [min_gain, max_gain]
        """
        clamped_gain = float(np.clip(gain, self.min_gain, self.max_gain))
        # Create synthetic state with the target gain
        self._current_state = ArousalState(
            tonic_level=self._tonic_level,
            phasic_burst=self._phasic_level,
            combined_gain=clamped_gain,
            novelty_score=self._current_state.novelty_score if self._current_state else 0.5,
            uncertainty_score=self._current_state.uncertainty_score if self._current_state else 0.5,
        )
        logger.info(f"NE arousal gain directly set to {clamped_gain}")

    def boost_arousal(self, multiplier: float) -> float:
        """
        Temporarily boost arousal by a multiplier.

        Args:
            multiplier: Boost factor [0.5, 3.0]

        Returns:
            New gain value
        """
        multiplier = float(np.clip(multiplier, 0.5, 3.0))
        current_gain = self.get_current_gain()
        boosted_gain = float(np.clip(current_gain * multiplier, self.min_gain, self.max_gain))
        self.set_arousal_gain(boosted_gain)
        logger.info(f"NE arousal boosted by {multiplier}x to {boosted_gain}")
        return boosted_gain

    def set_novelty_decay(self, decay: float) -> None:
        """
        Set novelty decay rate.

        Args:
            decay: Decay rate [0.8, 0.99]
        """
        self.novelty_decay = float(np.clip(decay, 0.8, 0.99))
        logger.info(f"NE novelty_decay set to {self.novelty_decay}")

    def set_phasic_decay(self, decay: float) -> None:
        """
        Set phasic burst decay rate.

        Args:
            decay: Decay rate [0.5, 0.95]
        """
        self.phasic_decay = float(np.clip(decay, 0.5, 0.95))
        logger.info(f"NE phasic_decay set to {self.phasic_decay}")

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "baseline_arousal": self.baseline_arousal,
            "min_gain": self.min_gain,
            "max_gain": self.max_gain,
            "novelty_decay": self.novelty_decay,
            "phasic_decay": self.phasic_decay,
            "uncertainty_weight": self.uncertainty_weight,
            "novelty_weight": self.novelty_weight,
        }

    def reset_history(self) -> None:
        """Clear query and arousal history."""
        self._query_history.clear()
        self._arousal_history.clear()
        self._tonic_level = self.baseline_arousal
        self._phasic_level = 0.0
        self._current_state = None


__all__ = [
    "ArousalState",
    "NorepinephrineSystem",
]
