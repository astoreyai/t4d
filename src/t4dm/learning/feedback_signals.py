"""
Feedback Signal Processing for T4DM.

Phase 1C: Converts retrieval feedback into learning signals compatible
with the three-factor learning rule and neuromodulator systems.

Biological Basis:
- Feedback signals must be translated into reward prediction errors (dopamine)
- Eligibility traces mark which synapses were recently active
- The brain uses feedback to update expectations for future predictions
- Surprise (prediction error) modulates learning rate

This module bridges:
- RetrievalFeedback (implicit user signals) -> LearningSignal (neuromodulator-compatible)
- LearningSignal -> Three-factor updates (eligibility * gate * dopamine)

References:
- Schultz (1998): Dopamine reward prediction error
- Friston (2010): Free energy principle, prediction errors
- Gerstner et al. (2018): Eligibility traces and three-factor learning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from t4dm.learning.retrieval_feedback import RetrievalFeedback

if TYPE_CHECKING:
    from t4dm.learning.three_factor import ThreeFactorLearningRule, ThreeFactorSignal

logger = logging.getLogger(__name__)


@dataclass
class LearningSignal:
    """
    Learning signal compatible with three-factor learning.

    Represents the teaching signal derived from feedback, ready to be
    combined with eligibility traces and neuromodulator state.
    """
    memory_id: str
    query_id: str

    # Core learning signal
    reward: float  # Transformed relevance [-1, 1]
    prediction_error: float  # Expected vs actual relevance

    # Components
    expected_relevance: float  # What we predicted
    actual_relevance: float  # What we observed

    # Confidence and gating
    confidence: float  # From feedback confidence
    should_update: bool  # Whether this signal should trigger learning

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    feedback_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "memory_id": self.memory_id,
            "query_id": self.query_id,
            "reward": self.reward,
            "prediction_error": self.prediction_error,
            "expected_relevance": self.expected_relevance,
            "actual_relevance": self.actual_relevance,
            "confidence": self.confidence,
            "should_update": self.should_update,
            "timestamp": self.timestamp.isoformat(),
            "feedback_id": self.feedback_id,
        }


class FeedbackSignalProcessor:
    """
    Process retrieval feedback into learning signals.

    Transforms implicit feedback (clicks, dwells) into signals compatible
    with the three-factor learning rule.

    The processor maintains expectations about result relevance and computes
    prediction errors when feedback arrives. This enables:
    1. Dopamine-like RPE signals for surprise-modulated learning
    2. Eligibility marking for temporal credit assignment
    3. Confidence-weighted updates for robust learning

    Usage:
        processor = FeedbackSignalProcessor()

        # Process single feedback
        signal = processor.feedback_to_learning_signal(feedback)

        # Use with three-factor rule
        if signal.should_update:
            three_factor.mark_active(signal.memory_id)
            update = three_factor.compute(
                memory_id=UUID(signal.memory_id),
                base_lr=0.01,
                outcome=signal.actual_relevance
            )

        # Batch processing
        signals = processor.process_batch(feedbacks)
    """

    def __init__(
        self,
        # Expectation parameters
        default_expectation: float = 0.5,  # Prior expected relevance
        expectation_lr: float = 0.1,  # Learning rate for expectations
        # Signal transformation
        reward_scale: float = 2.0,  # Scale for relevance -> reward
        min_prediction_error: float = 0.05,  # Minimum PE for update
        # Confidence thresholds
        min_confidence_for_update: float = 0.4,  # Below this, skip update
        confidence_scale: float = 1.0,  # Scale confidence effect
        # MEM-007: Bounded history
        max_history: int = 10000,
    ):
        """
        Initialize feedback signal processor.

        Args:
            default_expectation: Prior for expected relevance
            expectation_lr: How fast to update expectations
            reward_scale: Scaling for relevance to reward transformation
            min_prediction_error: PE threshold for triggering updates
            min_confidence_for_update: Confidence threshold for updates
            confidence_scale: How much confidence affects signal
            max_history: Maximum expectations to track (MEM-007)
        """
        self.default_expectation = default_expectation
        self.expectation_lr = expectation_lr
        self.reward_scale = reward_scale
        self.min_prediction_error = min_prediction_error
        self.min_confidence_for_update = min_confidence_for_update
        self.confidence_scale = confidence_scale
        self.max_history = max_history

        # Track expected relevance per memory (learned expectations)
        self._expectations: dict[str, float] = {}
        self._expectation_counts: dict[str, int] = {}

        # Statistics
        self._total_signals = 0
        self._total_updates = 0
        self._total_prediction_error = 0.0

    def feedback_to_learning_signal(
        self,
        feedback: RetrievalFeedback,
    ) -> LearningSignal:
        """
        Convert feedback to three-factor compatible signal.

        The conversion process:
        1. Get expected relevance for this memory (from history or prior)
        2. Compute prediction error (actual - expected)
        3. Transform relevance to reward signal
        4. Determine if this should trigger learning

        Args:
            feedback: RetrievalFeedback from collector

        Returns:
            LearningSignal ready for three-factor processing
        """
        memory_id = feedback.result_id

        # Get expected relevance
        expected = self._get_expectation(memory_id)

        # Actual relevance from feedback
        actual = feedback.relevance

        # Compute prediction error (RPE-like)
        prediction_error = actual - expected

        # Transform relevance to reward [-1, 1]
        # Center around 0.5, scale by reward_scale
        reward = (actual - 0.5) * self.reward_scale
        reward = float(np.clip(reward, -1.0, 1.0))

        # Determine if we should update
        should_update = (
            abs(prediction_error) >= self.min_prediction_error and
            feedback.confidence >= self.min_confidence_for_update
        )

        # Apply confidence scaling to reward
        scaled_reward = reward * (feedback.confidence ** self.confidence_scale)

        signal = LearningSignal(
            memory_id=memory_id,
            query_id=feedback.query_id,
            reward=scaled_reward,
            prediction_error=prediction_error,
            expected_relevance=expected,
            actual_relevance=actual,
            confidence=feedback.confidence,
            should_update=should_update,
            feedback_id=str(feedback.feedback_id),
        )

        # Update expectations with observed relevance
        self._update_expectation(memory_id, actual)

        # Track statistics
        self._total_signals += 1
        if should_update:
            self._total_updates += 1
        self._total_prediction_error += abs(prediction_error)

        logger.debug(
            f"Feedback -> Signal: memory={memory_id[:8]}, "
            f"PE={prediction_error:.3f}, reward={scaled_reward:.3f}, "
            f"update={should_update}"
        )

        return signal

    def process_batch(
        self,
        feedbacks: list[RetrievalFeedback],
    ) -> list[LearningSignal]:
        """
        Process batch of feedback events.

        Args:
            feedbacks: List of RetrievalFeedback

        Returns:
            List of LearningSignal
        """
        return [self.feedback_to_learning_signal(fb) for fb in feedbacks]

    def apply_to_three_factor(
        self,
        signal: LearningSignal,
        three_factor: ThreeFactorLearningRule,
        base_lr: float = 0.01,
    ) -> ThreeFactorSignal | None:
        """
        Apply learning signal to three-factor rule.

        Convenience method that:
        1. Marks the memory as active (eligibility)
        2. Computes the three-factor signal
        3. Returns the result for downstream processing

        Args:
            signal: LearningSignal to apply
            three_factor: ThreeFactorLearningRule instance
            base_lr: Base learning rate

        Returns:
            ThreeFactorSignal, or None if skipped
        """
        if not signal.should_update:
            return None

        # Mark active to update eligibility trace
        three_factor.mark_active(signal.memory_id, activity=signal.confidence)

        # Compute three-factor signal
        try:
            memory_uuid = UUID(signal.memory_id)
        except ValueError:
            # Generate deterministic UUID from string
            import hashlib
            hash_bytes = hashlib.md5(signal.memory_id.encode()).digest()
            memory_uuid = UUID(bytes=hash_bytes)

        result = three_factor.compute(
            memory_id=memory_uuid,
            base_lr=base_lr,
            outcome=signal.actual_relevance,
        )

        return result

    def _get_expectation(self, memory_id: str) -> float:
        """Get expected relevance for a memory."""
        if memory_id in self._expectations:
            return self._expectations[memory_id]
        return self.default_expectation

    def _update_expectation(self, memory_id: str, observed: float) -> None:
        """Update expected relevance based on observation."""
        # MEM-007: Enforce max history
        if memory_id not in self._expectations and len(self._expectations) >= self.max_history:
            self._evict_oldest_expectation()

        current = self._expectations.get(memory_id, self.default_expectation)
        count = self._expectation_counts.get(memory_id, 0) + 1

        # Exponential moving average
        new_expectation = current + self.expectation_lr * (observed - current)

        self._expectations[memory_id] = new_expectation
        self._expectation_counts[memory_id] = count

    def _evict_oldest_expectation(self) -> None:
        """Evict least-used expectation to make room."""
        if not self._expectation_counts:
            return

        # Find least-used memory
        least_used = min(
            self._expectation_counts.keys(),
            key=lambda k: self._expectation_counts[k]
        )

        del self._expectations[least_used]
        del self._expectation_counts[least_used]

    def get_expectation(self, memory_id: str) -> float:
        """
        Get current expected relevance for a memory.

        Public method for external access.

        Args:
            memory_id: Memory ID to query

        Returns:
            Expected relevance [0, 1]
        """
        return self._get_expectation(memory_id)

    def set_expectation(self, memory_id: str, expectation: float) -> None:
        """
        Manually set expectation for a memory.

        Useful for initialization from prior knowledge.

        Args:
            memory_id: Memory ID
            expectation: Expected relevance [0, 1]
        """
        expectation = float(np.clip(expectation, 0.0, 1.0))

        # MEM-007: Enforce max history
        if memory_id not in self._expectations and len(self._expectations) >= self.max_history:
            self._evict_oldest_expectation()

        self._expectations[memory_id] = expectation
        self._expectation_counts[memory_id] = self._expectation_counts.get(memory_id, 0) + 1

    def get_statistics(self) -> dict:
        """Get processor statistics."""
        avg_pe = (
            self._total_prediction_error / self._total_signals
            if self._total_signals > 0 else 0.0
        )
        update_rate = (
            self._total_updates / self._total_signals
            if self._total_signals > 0 else 0.0
        )

        return {
            "total_signals": self._total_signals,
            "total_updates": self._total_updates,
            "update_rate": update_rate,
            "avg_prediction_error": avg_pe,
            "num_tracked_memories": len(self._expectations),
            "default_expectation": self.default_expectation,
        }

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "expectations": self._expectations.copy(),
            "expectation_counts": self._expectation_counts.copy(),
            "total_signals": self._total_signals,
            "total_updates": self._total_updates,
            "total_prediction_error": self._total_prediction_error,
        }

    def load_state(self, state: dict) -> None:
        """Load state from persistence."""
        self._expectations = state.get("expectations", {})
        self._expectation_counts = state.get("expectation_counts", {})
        self._total_signals = state.get("total_signals", 0)
        self._total_updates = state.get("total_updates", 0)
        self._total_prediction_error = state.get("total_prediction_error", 0.0)

    def clear(self) -> None:
        """Clear all expectations and statistics."""
        self._expectations.clear()
        self._expectation_counts.clear()
        self._total_signals = 0
        self._total_updates = 0
        self._total_prediction_error = 0.0


class AdapterTrainingSignal:
    """
    Training signal for retrieval adapter/scorer.

    Aggregates feedback into training examples suitable for
    learning-to-rank or embedding adaptation.
    """

    def __init__(
        self,
        query_embedding: np.ndarray,
        positive_ids: list[str],
        negative_ids: list[str],
        positive_weights: dict[str, float] | None = None,
        negative_weights: dict[str, float] | None = None,
    ):
        """
        Initialize adapter training signal.

        Args:
            query_embedding: Query vector
            positive_ids: IDs of relevant results (clicked, high dwell)
            negative_ids: IDs of non-relevant results (skipped)
            positive_weights: Optional relevance weights for positives
            negative_weights: Optional anti-relevance weights for negatives
        """
        self.query_embedding = query_embedding
        self.positive_ids = positive_ids
        self.negative_ids = negative_ids
        self.positive_weights = positive_weights or {}
        self.negative_weights = negative_weights or {}


class FeedbackToAdapterBridge:
    """
    Bridge retrieval feedback to adapter training.

    Converts feedback signals into contrastive training examples
    for the learned retrieval scorer.
    """

    def __init__(
        self,
        positive_threshold: float = 0.6,  # Min relevance for positive
        negative_threshold: float = 0.3,  # Max relevance for negative
        min_confidence: float = 0.5,  # Min confidence to include
    ):
        """
        Initialize bridge.

        Args:
            positive_threshold: Relevance threshold for positive examples
            negative_threshold: Relevance threshold for negative examples
            min_confidence: Minimum confidence to include in training
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.min_confidence = min_confidence

    def feedback_to_training_signal(
        self,
        feedbacks: list[RetrievalFeedback],
        query_embedding: np.ndarray | None = None,
    ) -> AdapterTrainingSignal | None:
        """
        Convert feedback from single query to training signal.

        Separates results into positive/negative based on relevance
        thresholds, weighting by confidence.

        Args:
            feedbacks: Feedbacks from single query
            query_embedding: Optional query vector

        Returns:
            AdapterTrainingSignal, or None if insufficient data
        """
        if not feedbacks:
            return None

        positive_ids = []
        negative_ids = []
        positive_weights: dict[str, float] = {}
        negative_weights: dict[str, float] = {}

        for fb in feedbacks:
            if fb.confidence < self.min_confidence:
                continue

            if fb.relevance >= self.positive_threshold:
                positive_ids.append(fb.result_id)
                positive_weights[fb.result_id] = fb.relevance * fb.confidence
            elif fb.relevance <= self.negative_threshold:
                negative_ids.append(fb.result_id)
                # Invert relevance for negative weight
                negative_weights[fb.result_id] = (1 - fb.relevance) * fb.confidence

        # Need at least one positive and one negative
        if not positive_ids or not negative_ids:
            return None

        # Use provided embedding or create placeholder
        if query_embedding is None:
            query_embedding = np.zeros(1024, dtype=np.float32)

        return AdapterTrainingSignal(
            query_embedding=query_embedding,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            positive_weights=positive_weights,
            negative_weights=negative_weights,
        )


__all__ = [
    "LearningSignal",
    "FeedbackSignalProcessor",
    "AdapterTrainingSignal",
    "FeedbackToAdapterBridge",
]
