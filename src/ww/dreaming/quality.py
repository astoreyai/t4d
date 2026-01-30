"""
Dream Quality Evaluation.

P3-2: Evaluate dream quality for consolidation priority.

Biological Basis:
- Not all dreams are equally valuable for learning
- Coherent dreams that explore novel patterns are most useful
- Dreams that diverge into nonsense waste consolidation resources

DreamerV3 Insight:
- Evaluate imagined trajectories for learning value
- Prioritize dreams that challenge the world model
- Discard dreams that are too easy or too incoherent

Metrics:
- Coherence: Stays on learned manifold
- Smoothness: Temporal consistency between steps
- Novelty: Distance from known patterns
- Informativeness: Prediction uncertainty (learning signal)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ww.dreaming.trajectory import DreamTrajectory

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """Configuration for dream quality evaluation."""

    # Metric weights for overall score
    coherence_weight: float = 0.3
    smoothness_weight: float = 0.2
    novelty_weight: float = 0.25
    informativeness_weight: float = 0.25

    # Thresholds
    min_quality_threshold: float = 0.4  # Below this = discard dream
    high_quality_threshold: float = 0.7  # Above this = priority dream

    # Novelty computation
    novelty_neighbors: int = 10  # Compare to N nearest references
    novelty_threshold: float = 0.3  # Minimum distance for "novel"


@dataclass
class DreamQuality:
    """Quality assessment for a dream trajectory."""

    dream_id: str
    coherence_score: float  # [0, 1] - stays on manifold
    smoothness_score: float  # [0, 1] - temporal consistency
    novelty_score: float  # [0, 1] - explores new patterns
    informativeness_score: float  # [0, 1] - learning potential
    overall_score: float  # Weighted combination
    is_high_quality: bool
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dream_id": self.dream_id,
            "coherence_score": self.coherence_score,
            "smoothness_score": self.smoothness_score,
            "novelty_score": self.novelty_score,
            "informativeness_score": self.informativeness_score,
            "overall_score": self.overall_score,
            "is_high_quality": self.is_high_quality,
            "timestamp": self.timestamp.isoformat(),
        }


class DreamQualityEvaluator:
    """
    Evaluate dream quality for consolidation priority.

    P3-2: Determines which dreams are worth using for learning.

    Quality Dimensions:
        1. Coherence: Dream stays on learned manifold
        2. Smoothness: Transitions are gradual, not jumpy
        3. Novelty: Explores regions not well-covered by memory
        4. Informativeness: Prediction uncertainty is high (learning signal)

    Usage:
        evaluator = DreamQualityEvaluator(reference_embeddings)
        quality = evaluator.evaluate(dream_trajectory)
        if quality.is_high_quality:
            # Use for consolidation
    """

    def __init__(
        self,
        reference_embeddings: list[np.ndarray] | None = None,
        config: QualityConfig | None = None,
    ):
        """
        Initialize quality evaluator.

        Args:
            reference_embeddings: Known memory embeddings for novelty
            config: Evaluation configuration
        """
        self.config = config or QualityConfig()
        self._references = reference_embeddings or []

        # Statistics
        self._total_evaluated = 0
        self._high_quality_count = 0
        self._discarded_count = 0

        logger.info("DreamQualityEvaluator initialized")

    def set_references(self, embeddings: list[np.ndarray]) -> None:
        """Set reference embeddings for novelty computation."""
        self._references = embeddings

    def add_references(self, embeddings: list[np.ndarray]) -> None:
        """Add reference embeddings."""
        self._references.extend(embeddings)

    def evaluate(self, dream: DreamTrajectory) -> DreamQuality:
        """
        Evaluate a dream trajectory.

        Args:
            dream: Dream trajectory to evaluate

        Returns:
            DreamQuality assessment
        """
        if not dream.steps:
            return DreamQuality(
                dream_id=str(dream.id),
                coherence_score=0.0,
                smoothness_score=0.0,
                novelty_score=0.0,
                informativeness_score=0.0,
                overall_score=0.0,
                is_high_quality=False,
            )

        # Compute individual metrics
        coherence = self._compute_coherence(dream)
        smoothness = self._compute_smoothness(dream)
        novelty = self._compute_novelty(dream)
        informativeness = self._compute_informativeness(dream)

        # Weighted overall score
        overall = (
            self.config.coherence_weight * coherence +
            self.config.smoothness_weight * smoothness +
            self.config.novelty_weight * novelty +
            self.config.informativeness_weight * informativeness
        )

        is_high_quality = overall >= self.config.high_quality_threshold

        # Track statistics
        self._total_evaluated += 1
        if is_high_quality:
            self._high_quality_count += 1
        elif overall < self.config.min_quality_threshold:
            self._discarded_count += 1

        quality = DreamQuality(
            dream_id=str(dream.id),
            coherence_score=coherence,
            smoothness_score=smoothness,
            novelty_score=novelty,
            informativeness_score=informativeness,
            overall_score=overall,
            is_high_quality=is_high_quality,
        )

        logger.debug(
            f"Dream quality: overall={overall:.3f}, "
            f"coherence={coherence:.3f}, smoothness={smoothness:.3f}, "
            f"novelty={novelty:.3f}, info={informativeness:.3f}"
        )

        return quality

    def evaluate_batch(
        self,
        dreams: list[DreamTrajectory],
    ) -> list[DreamQuality]:
        """Evaluate multiple dreams."""
        return [self.evaluate(d) for d in dreams]

    def filter_high_quality(
        self,
        dreams: list[DreamTrajectory],
    ) -> list[tuple[DreamTrajectory, DreamQuality]]:
        """
        Filter to high-quality dreams only.

        Args:
            dreams: Dreams to filter

        Returns:
            List of (dream, quality) tuples for high-quality dreams
        """
        results = []
        for dream in dreams:
            quality = self.evaluate(dream)
            if quality.is_high_quality:
                results.append((dream, quality))
        return results

    def _compute_coherence(self, dream: DreamTrajectory) -> float:
        """
        Compute coherence score.

        Based on mean confidence and coherence from dream steps.
        High coherence = dream stays on learned manifold.
        """
        # Use trajectory's built-in metrics
        confidence = dream.mean_confidence
        coherence = dream.mean_coherence

        # Combine (both should be high for coherent dream)
        return (confidence + coherence) / 2

    def _compute_smoothness(self, dream: DreamTrajectory) -> float:
        """
        Compute smoothness score.

        Measures temporal consistency - are transitions gradual?
        Jumpy dreams indicate prediction instability.
        """
        embeddings = dream.embeddings
        if len(embeddings) < 2:
            return 1.0  # Single step = trivially smooth

        # Compute step-to-step similarities
        similarities = []
        for i in range(1, len(embeddings)):
            sim = float(np.dot(embeddings[i], embeddings[i - 1]))
            similarities.append(sim)

        # High mean similarity = smooth trajectory
        mean_sim = np.mean(similarities)

        # Also check for sudden jumps (low variance is good)
        variance = np.var(similarities)
        stability = 1.0 / (1.0 + variance * 10)

        return float((mean_sim + stability) / 2)

    def _compute_novelty(self, dream: DreamTrajectory) -> float:
        """
        Compute novelty score.

        How different is this dream from known memories?
        Novel dreams explore underrepresented regions.
        """
        if not self._references or not dream.steps:
            return 0.5  # No references = neutral novelty

        embeddings = dream.embeddings

        # Compute minimum distance to references for each step
        min_distances = []
        for emb in embeddings:
            # Find closest reference
            distances = [
                1.0 - float(np.dot(emb, ref))  # Cosine distance
                for ref in self._references[-self.config.novelty_neighbors * 10:]
            ]
            if distances:
                min_dist = min(distances)
                min_distances.append(min_dist)

        if not min_distances:
            return 0.5

        # Average minimum distance
        mean_min_dist = np.mean(min_distances)

        # Normalize: 0 = identical to known, 1 = very different
        # Use threshold to calibrate
        novelty = min(1.0, mean_min_dist / self.config.novelty_threshold)

        return float(novelty)

    def _compute_informativeness(self, dream: DreamTrajectory) -> float:
        """
        Compute informativeness score.

        Dreams with moderate prediction confidence are most informative.
        Too easy = nothing to learn. Too hard = noise.
        """
        if not dream.steps:
            return 0.0

        confidences = [s.confidence for s in dream.steps]
        mean_conf = np.mean(confidences)

        # Optimal is around 0.5-0.7 (challenging but achievable)
        # Bell curve centered at 0.6
        optimal = 0.6
        deviation = abs(mean_conf - optimal)
        score = 1.0 - min(1.0, deviation * 2)

        # Also reward trajectory length (longer = more learning)
        length_bonus = min(1.0, dream.length / 10)

        return float((score + length_bonus) / 2)

    def get_statistics(self) -> dict[str, Any]:
        """Get evaluator statistics."""
        return {
            "total_evaluated": self._total_evaluated,
            "high_quality_count": self._high_quality_count,
            "discarded_count": self._discarded_count,
            "high_quality_rate": (
                self._high_quality_count / max(1, self._total_evaluated)
            ),
            "reference_count": len(self._references),
        }

    def save_state(self) -> dict[str, Any]:
        """Save evaluator state."""
        return {
            "config": {
                "coherence_weight": self.config.coherence_weight,
                "smoothness_weight": self.config.smoothness_weight,
                "novelty_weight": self.config.novelty_weight,
                "informativeness_weight": self.config.informativeness_weight,
            },
            "statistics": self.get_statistics(),
        }
