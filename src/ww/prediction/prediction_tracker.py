"""
Prediction Tracker for World Models.

P2-3: Track prediction errors for episodes to enable prioritized replay.

Biological Basis:
- Hippocampus tracks prediction errors for consolidation priority
- High-error experiences are replayed more frequently during sleep
- This implements the "surprisal" component of memory consolidation

Integration:
- Works with Episode.prediction_error field
- Provides top-k episodes by prediction error for replay
- Tracks prediction history for trend analysis

JEPA/DreamerV3 Insight:
- Prediction error is the primary learning signal
- Episodes with high error = model gaps = learning opportunities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np

from ww.prediction.context_encoder import ContextEncoder
from ww.prediction.latent_predictor import (
    LatentPredictor,
    Prediction,
    PredictionError,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """Configuration for prediction tracker."""

    # History limits
    max_pending_predictions: int = 1000  # Predictions awaiting outcomes
    max_error_history: int = 10000  # Historical errors to track
    prediction_timeout: timedelta = field(
        default_factory=lambda: timedelta(hours=24)
    )  # When to expire pending predictions

    # Error normalization
    error_ema_alpha: float = 0.1  # For running mean/std
    normalize_errors: bool = True  # Normalize to z-scores

    # Replay priority
    min_error_for_priority: float = 0.1  # Below this, don't prioritize
    priority_decay: float = 0.95  # Decay priority over time


@dataclass
class TrackedPrediction:
    """A prediction awaiting outcome resolution."""

    episode_id: UUID  # Episode we predicted AFTER
    prediction: Prediction  # The prediction we made
    context_ids: list[UUID]  # Episodes used as context
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    error: PredictionError | None = None


class PredictionTracker:
    """
    Track predictions and their errors for prioritized replay.

    P2-3: The bridge between prediction and consolidation.

    Workflow:
        1. Before new episode: make_prediction(context_episodes)
        2. After episode arrives: resolve_prediction(prediction_id, actual_embedding)
        3. During consolidation: get_high_error_episodes(k=10)

    Usage:
        tracker = PredictionTracker(context_encoder, predictor)

        # Make prediction from recent context
        tracked = tracker.make_prediction(recent_episodes)

        # When outcome arrives
        tracker.resolve_prediction(tracked.episode_id, new_episode.embedding)

        # Get episodes for prioritized replay
        priority_episodes = tracker.get_high_error_episodes(k=10)
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        predictor: LatentPredictor,
        config: TrackerConfig | None = None,
    ):
        """
        Initialize prediction tracker.

        Args:
            context_encoder: Encoder for context representations
            predictor: Latent state predictor
            config: Tracker configuration
        """
        self.context_encoder = context_encoder
        self.predictor = predictor
        self.config = config or TrackerConfig()

        # Pending predictions (episode_id -> prediction)
        self._pending: dict[UUID, TrackedPrediction] = {}

        # Error history (episode_id -> error)
        self._errors: dict[UUID, float] = {}
        self._error_timestamps: dict[UUID, datetime] = {}

        # Running statistics for normalization
        self._error_mean = 0.0
        self._error_var = 1.0
        self._error_count = 0

        # Priority queue (episode_id -> priority score)
        self._priority_scores: dict[UUID, float] = {}

        logger.info("PredictionTracker initialized")

    def make_prediction(
        self,
        context_episodes: list[tuple[UUID, np.ndarray]],
    ) -> TrackedPrediction:
        """
        Make a prediction from context episodes.

        Args:
            context_episodes: List of (episode_id, embedding) tuples

        Returns:
            TrackedPrediction awaiting resolution
        """
        if not context_episodes:
            raise ValueError("Need at least one context episode")

        # Extract embeddings and IDs
        context_ids = [ep_id for ep_id, _ in context_episodes]
        embeddings = [emb for _, emb in context_episodes]

        # Encode context
        encoded = self.context_encoder.encode(embeddings)

        # Make prediction
        prediction = self.predictor.predict(encoded.context_vector)

        # Use last context episode as the "before" episode
        before_episode_id = context_ids[-1]

        tracked = TrackedPrediction(
            episode_id=before_episode_id,
            prediction=prediction,
            context_ids=context_ids,
        )

        # Store pending prediction
        self._pending[before_episode_id] = tracked

        # Cleanup old pending predictions
        self._cleanup_expired()

        logger.debug(
            f"Made prediction after episode {before_episode_id}, "
            f"confidence={prediction.confidence:.3f}"
        )

        return tracked

    def resolve_prediction(
        self,
        before_episode_id: UUID,
        actual_embedding: np.ndarray,
        actual_episode_id: UUID,
    ) -> PredictionError | None:
        """
        Resolve a pending prediction with actual outcome.

        Args:
            before_episode_id: Episode ID the prediction was made after
            actual_embedding: Actual embedding of the new episode
            actual_episode_id: ID of the actual new episode

        Returns:
            PredictionError if prediction existed, None otherwise
        """
        if before_episode_id not in self._pending:
            logger.debug(f"No pending prediction for {before_episode_id}")
            return None

        tracked = self._pending.pop(before_episode_id)

        # Compute prediction error
        error = self.predictor.compute_error(
            tracked.prediction,
            actual_embedding,
            actual_episode_id,
        )

        # Mark as resolved
        tracked.resolved = True
        tracked.error = error

        # Store error
        self._record_error(actual_episode_id, error.combined_error)

        logger.debug(
            f"Resolved prediction: error={error.combined_error:.4f}, "
            f"L2={error.error_magnitude:.4f}, cosine={error.cosine_error:.4f}"
        )

        return error

    def _record_error(self, episode_id: UUID, error: float) -> None:
        """Record error and update statistics."""
        self._errors[episode_id] = error
        self._error_timestamps[episode_id] = datetime.now()

        # Update running statistics (Welford's algorithm)
        self._error_count += 1
        delta = error - self._error_mean
        self._error_mean += delta / self._error_count
        delta2 = error - self._error_mean
        self._error_var += delta * delta2

        # Compute priority score
        if self.config.normalize_errors and self._error_count > 10:
            std = np.sqrt(self._error_var / self._error_count)
            z_score = (error - self._error_mean) / (std + 1e-8)
            priority = max(0, z_score)  # Only positive z-scores matter
        else:
            priority = error

        if priority > self.config.min_error_for_priority:
            self._priority_scores[episode_id] = priority

        # Cleanup old errors
        if len(self._errors) > self.config.max_error_history:
            self._cleanup_old_errors()

    def get_prediction_error(self, episode_id: UUID) -> float | None:
        """
        Get prediction error for an episode.

        Args:
            episode_id: Episode to query

        Returns:
            Error value or None if not tracked
        """
        return self._errors.get(episode_id)

    def get_high_error_episodes(
        self,
        k: int = 10,
        decay_by_time: bool = True,
    ) -> list[tuple[UUID, float]]:
        """
        Get top-k episodes by prediction error for prioritized replay.

        Args:
            k: Number of episodes to return
            decay_by_time: Apply time decay to priorities

        Returns:
            List of (episode_id, priority_score) sorted by priority
        """
        if not self._priority_scores:
            return []

        scores = dict(self._priority_scores)

        # Apply time decay
        if decay_by_time:
            now = datetime.now()
            for ep_id, score in list(scores.items()):
                if ep_id in self._error_timestamps:
                    age = (now - self._error_timestamps[ep_id]).total_seconds()
                    # Decay factor: 0.95^(hours)
                    decay = self.config.priority_decay ** (age / 3600)
                    scores[ep_id] = score * decay

        # Sort by priority (descending)
        sorted_episodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_episodes[:k]

    def tag_episode_with_error(
        self,
        episode: Any,
        error: float | None = None,
    ) -> Any:
        """
        Tag an episode with its prediction error.

        Args:
            episode: Episode object with prediction_error field
            error: Error value (or use stored value)

        Returns:
            Episode with updated prediction_error field
        """
        if error is None:
            error = self._errors.get(episode.id)

        if error is not None:
            episode.prediction_error = float(error)
            episode.prediction_error_timestamp = datetime.now()

        return episode

    def get_statistics(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "pending_predictions": len(self._pending),
            "tracked_errors": len(self._errors),
            "priority_queue_size": len(self._priority_scores),
            "error_mean": self._error_mean,
            "error_std": np.sqrt(self._error_var / max(1, self._error_count)),
            "total_predictions": self._error_count,
        }

    def _cleanup_expired(self) -> None:
        """Remove expired pending predictions."""
        now = datetime.now()
        expired = [
            ep_id
            for ep_id, tracked in self._pending.items()
            if now - tracked.timestamp > self.config.prediction_timeout
        ]
        for ep_id in expired:
            del self._pending[ep_id]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired predictions")

    def _cleanup_old_errors(self) -> None:
        """Remove oldest errors to maintain history limit."""
        if len(self._errors) <= self.config.max_error_history:
            return

        # Sort by timestamp and keep most recent
        sorted_by_time = sorted(
            self._error_timestamps.items(),
            key=lambda x: x[1],
        )

        n_remove = len(self._errors) - self.config.max_error_history
        for ep_id, _ in sorted_by_time[:n_remove]:
            del self._errors[ep_id]
            del self._error_timestamps[ep_id]
            self._priority_scores.pop(ep_id, None)

    def save_state(self) -> dict[str, Any]:
        """Save tracker state."""
        return {
            "errors": {str(k): v for k, v in self._errors.items()},
            "error_timestamps": {
                str(k): v.isoformat() for k, v in self._error_timestamps.items()
            },
            "priority_scores": {str(k): v for k, v in self._priority_scores.items()},
            "error_mean": self._error_mean,
            "error_var": self._error_var,
            "error_count": self._error_count,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load tracker state."""
        from uuid import UUID

        self._errors = {UUID(k): v for k, v in state.get("errors", {}).items()}
        self._error_timestamps = {
            UUID(k): datetime.fromisoformat(v)
            for k, v in state.get("error_timestamps", {}).items()
        }
        self._priority_scores = {
            UUID(k): v for k, v in state.get("priority_scores", {}).items()
        }
        self._error_mean = state.get("error_mean", 0.0)
        self._error_var = state.get("error_var", 1.0)
        self._error_count = state.get("error_count", 0)
