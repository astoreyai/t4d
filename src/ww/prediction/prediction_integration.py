"""
Prediction Integration with Consolidation.

P2-4: Connect latent prediction to consolidation and memory lifecycle.

Biological Basis:
- Hippocampus continuously predicts upcoming states
- Prediction errors tag memories for priority consolidation
- During sleep, high-error memories are replayed more frequently

Integration Points:
1. Episode creation: Make prediction from context
2. Episode storage: Resolve prediction, record error
3. Consolidation: Use prediction errors for replay priority
4. Training: Update predictor during consolidation replay

JEPA/DreamerV3 Insight:
- The world model learns from its own prediction errors
- Consolidation replays surprising events to improve the model
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

import numpy as np

from ww.prediction.context_encoder import ContextEncoder, ContextEncoderConfig
from ww.prediction.latent_predictor import LatentPredictor, LatentPredictorConfig
from ww.prediction.prediction_tracker import PredictionTracker, TrackerConfig

logger = logging.getLogger(__name__)


class EpisodeProtocol(Protocol):
    """Protocol for Episode-like objects."""

    id: UUID
    embedding: np.ndarray | None
    prediction_error: float | None
    prediction_error_timestamp: datetime | None


class MemoryProtocol(Protocol):
    """Protocol for memory store."""

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        """Search for similar memories."""
        ...


@dataclass
class PredictionIntegrationConfig:
    """Configuration for prediction integration."""

    # Context window
    context_size: int = 5  # Episodes to use as context
    context_hours: int = 4  # Hours to look back for context

    # Training
    train_on_resolve: bool = True  # Train predictor when resolving
    train_during_replay: bool = True  # Train during consolidation replay
    training_lr: float = 0.001

    # Error thresholds
    high_error_threshold: float = 0.5  # Above this = high surprise
    low_error_threshold: float = 0.1  # Below this = well predicted

    # Encoder/predictor configs
    encoder_config: ContextEncoderConfig = field(
        default_factory=ContextEncoderConfig
    )
    predictor_config: LatentPredictorConfig = field(
        default_factory=LatentPredictorConfig
    )
    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)


class PredictionIntegration:
    """
    Integrate prediction with memory lifecycle and consolidation.

    P2-4: The bridge between world model and memory systems.

    Lifecycle:
        1. on_episode_created() - Make prediction for next state
        2. on_episode_stored() - Resolve prediction, tag with error
        3. during_consolidation() - Replay and train on errors
        4. get_priority_episodes() - High-error episodes for replay

    Usage:
        integration = PredictionIntegration()

        # When episode is about to be stored
        integration.on_episode_created(new_episode, recent_episodes)

        # After episode is stored
        integration.on_episode_stored(new_episode)

        # During consolidation
        for episode in episodes_to_replay:
            integration.on_replay(episode, target_embedding)

        # Get priority queue
        priority = integration.get_priority_episodes(k=10)
    """

    def __init__(
        self,
        config: PredictionIntegrationConfig | None = None,
    ):
        """
        Initialize prediction integration.

        Args:
            config: Integration configuration
        """
        self.config = config or PredictionIntegrationConfig()

        # Initialize components
        self.encoder = ContextEncoder(self.config.encoder_config)
        self.predictor = LatentPredictor(self.config.predictor_config)
        self.tracker = PredictionTracker(
            self.encoder,
            self.predictor,
            self.config.tracker_config,
        )

        # Recent episode buffer for context
        self._recent_episodes: deque[tuple[UUID, np.ndarray]] = deque(
            maxlen=self.config.context_size * 2
        )

        # Statistics
        self._predictions_made = 0
        self._predictions_resolved = 0
        self._high_error_count = 0
        self._low_error_count = 0
        self._total_training_steps = 0

        logger.info("PredictionIntegration initialized")

    def on_episode_created(
        self,
        episode: EpisodeProtocol,
        context_episodes: list[tuple[UUID, np.ndarray]] | None = None,
    ) -> None:
        """
        Handle new episode creation - make prediction.

        Call this BEFORE the episode is stored, using the previous
        episodes as context to predict this new episode.

        Args:
            episode: The newly created episode
            context_episodes: Optional explicit context, otherwise uses buffer
        """
        if episode.embedding is None:
            return

        # Build context from recent episodes if not provided
        if context_episodes is None:
            context_episodes = list(self._recent_episodes)[-self.config.context_size:]

        if not context_episodes:
            # No context yet, just add to buffer
            self._recent_episodes.append((episode.id, episode.embedding))
            return

        # Make prediction for this episode
        try:
            self.tracker.make_prediction(context_episodes)
            self._predictions_made += 1
        except Exception as e:
            logger.warning(f"Failed to make prediction: {e}")

        # Add to recent buffer
        self._recent_episodes.append((episode.id, episode.embedding))

    def on_episode_stored(
        self,
        episode: EpisodeProtocol,
        context_id: UUID | None = None,
    ) -> float | None:
        """
        Handle episode storage - resolve prediction and tag.

        Call this AFTER the episode is stored to resolve any
        pending predictions and tag the episode with error.

        Args:
            episode: The stored episode
            context_id: Optional context episode ID to resolve

        Returns:
            Prediction error if resolved, None otherwise
        """
        if episode.embedding is None:
            return None

        # Find the context episode to resolve
        if context_id is None and len(self._recent_episodes) >= 2:
            # Use second-to-last episode as context
            context_id = self._recent_episodes[-2][0]

        if context_id is None:
            return None

        # Resolve prediction
        error_obj = self.tracker.resolve_prediction(
            before_episode_id=context_id,
            actual_embedding=episode.embedding,
            actual_episode_id=episode.id,
        )

        if error_obj is None:
            return None

        self._predictions_resolved += 1
        error_value = error_obj.combined_error

        # Categorize error
        if error_value > self.config.high_error_threshold:
            self._high_error_count += 1
        elif error_value < self.config.low_error_threshold:
            self._low_error_count += 1

        # Tag episode
        self.tracker.tag_episode_with_error(episode, error_value)

        # Train predictor if enabled
        if self.config.train_on_resolve and len(self._recent_episodes) >= 2:
            context_embs = [
                emb for _, emb in list(self._recent_episodes)[-self.config.context_size-1:-1]
            ]
            if context_embs:
                encoded = self.encoder.encode(context_embs)
                self.predictor.train_step(
                    encoded.context_vector,
                    episode.embedding,
                    learning_rate=self.config.training_lr,
                )
                self._total_training_steps += 1

        return error_value

    def on_replay(
        self,
        episode: EpisodeProtocol,
        context_embeddings: list[np.ndarray] | None = None,
    ) -> float:
        """
        Handle episode replay during consolidation.

        Use this during consolidation to train the predictor
        on replayed episodes.

        Args:
            episode: Episode being replayed
            context_embeddings: Optional context for training

        Returns:
            Training loss
        """
        if episode.embedding is None or not self.config.train_during_replay:
            return 0.0

        if context_embeddings is None or len(context_embeddings) == 0:
            return 0.0

        # Encode context
        encoded = self.encoder.encode(context_embeddings)

        # Train on replay
        loss = self.predictor.train_step(
            encoded.context_vector,
            episode.embedding,
            learning_rate=self.config.training_lr * 0.5,  # Lower LR for replay
        )
        self._total_training_steps += 1

        return loss

    def get_priority_episodes(
        self,
        k: int = 10,
    ) -> list[tuple[UUID, float]]:
        """
        Get high-error episodes for prioritized replay.

        Args:
            k: Number of episodes to return

        Returns:
            List of (episode_id, priority_score)
        """
        return self.tracker.get_high_error_episodes(k=k)

    def get_prediction_error(self, episode_id: UUID) -> float | None:
        """Get prediction error for an episode."""
        return self.tracker.get_prediction_error(episode_id)

    def predict_next(
        self,
        context_embeddings: list[np.ndarray],
    ) -> np.ndarray:
        """
        Predict next embedding from context.

        Useful for anticipatory retrieval or planning.

        Args:
            context_embeddings: Recent episode embeddings

        Returns:
            Predicted next embedding
        """
        encoded = self.encoder.encode(context_embeddings)
        prediction = self.predictor.predict(encoded.context_vector)
        return prediction.predicted_embedding

    def get_statistics(self) -> dict[str, Any]:
        """Get integration statistics."""
        return {
            "predictions_made": self._predictions_made,
            "predictions_resolved": self._predictions_resolved,
            "high_error_count": self._high_error_count,
            "low_error_count": self._low_error_count,
            "total_training_steps": self._total_training_steps,
            "recent_buffer_size": len(self._recent_episodes),
            "tracker_stats": self.tracker.get_statistics(),
            "predictor_stats": self.predictor.get_statistics(),
        }

    def save_state(self) -> dict[str, Any]:
        """Save integration state."""
        return {
            "encoder": self.encoder.save_state(),
            "predictor": self.predictor.save_state(),
            "tracker": self.tracker.save_state(),
            "recent_episodes": [
                (str(ep_id), emb.tolist())
                for ep_id, emb in self._recent_episodes
            ],
            "statistics": {
                "predictions_made": self._predictions_made,
                "predictions_resolved": self._predictions_resolved,
                "high_error_count": self._high_error_count,
                "low_error_count": self._low_error_count,
                "total_training_steps": self._total_training_steps,
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load integration state."""
        from uuid import UUID

        self.encoder.load_state(state["encoder"])
        self.predictor.load_state(state["predictor"])
        self.tracker.load_state(state["tracker"])

        self._recent_episodes.clear()
        for ep_id_str, emb_list in state.get("recent_episodes", []):
            self._recent_episodes.append((
                UUID(ep_id_str),
                np.array(emb_list, dtype=np.float32),
            ))

        stats = state.get("statistics", {})
        self._predictions_made = stats.get("predictions_made", 0)
        self._predictions_resolved = stats.get("predictions_resolved", 0)
        self._high_error_count = stats.get("high_error_count", 0)
        self._low_error_count = stats.get("low_error_count", 0)
        self._total_training_steps = stats.get("total_training_steps", 0)


def create_prediction_integration(
    config: PredictionIntegrationConfig | None = None,
) -> PredictionIntegration:
    """
    Factory function for prediction integration.

    Args:
        config: Optional configuration

    Returns:
        Configured PredictionIntegration instance
    """
    return PredictionIntegration(config)
