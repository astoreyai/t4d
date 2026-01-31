"""
Dream-Based Consolidation.

P3-3: Integration with sleep consolidation.
P3-4: CONSOLIDATE attractor integration.

Biological Basis:
- REM sleep generates dreams from high-error memories
- Dreams are evaluated for learning value
- High-quality dreams inform next NREM replay priority
- Dreams may create new abstract concepts

DreamerV3 Insight:
- Train world model on imagined trajectories
- Prioritize experiences that challenge predictions
- Use dreams for credit assignment over long horizons

Integration:
- Works with SleepConsolidation from consolidation/sleep.py
- Uses PredictionTracker for high-error episode selection
- Generates dreams, evaluates quality, updates priorities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

import numpy as np

from t4dm.dreaming.quality import DreamQuality, DreamQualityEvaluator, QualityConfig
from t4dm.dreaming.trajectory import DreamingConfig, DreamingSystem, DreamTrajectory
from t4dm.prediction import ContextEncoder, LatentPredictor, PredictionTracker

logger = logging.getLogger(__name__)


class EpisodeProtocol(Protocol):
    """Protocol for Episode-like objects."""

    id: UUID
    embedding: np.ndarray | None
    prediction_error: float | None


@dataclass
class DreamConsolidationConfig:
    """Configuration for dream-based consolidation."""

    # Dream generation
    dreams_per_cycle: int = 5  # Dreams to generate per REM cycle
    seed_from_high_error: bool = True  # Use high-error episodes as seeds
    min_error_for_seed: float = 0.3  # Minimum prediction error for seeding

    # Quality filtering
    min_quality_for_replay: float = 0.5  # Minimum quality to use dream
    priority_boost_factor: float = 0.2  # How much to boost priority for dream sources

    # Training
    train_on_dreams: bool = True  # Train predictor on dream trajectories
    dream_learning_rate: float = 0.0005  # Lower LR for dream training

    # Dreaming system config
    dreaming_config: DreamingConfig = field(default_factory=DreamingConfig)
    quality_config: QualityConfig = field(default_factory=QualityConfig)


@dataclass
class DreamReplayEvent:
    """Record of a dream-based replay event."""

    dream_id: UUID
    seed_episode_id: UUID | None
    dream_length: int
    quality_score: float
    training_loss: float | None = None
    priority_updates: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dream_id": str(self.dream_id),
            "seed_episode_id": str(self.seed_episode_id) if self.seed_episode_id else None,
            "dream_length": self.dream_length,
            "quality_score": self.quality_score,
            "training_loss": self.training_loss,
            "priority_updates": self.priority_updates,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DreamCycleResult:
    """Result of a complete dream cycle."""

    cycle_id: UUID = field(default_factory=uuid4)
    dreams_generated: int = 0
    high_quality_dreams: int = 0
    discarded_dreams: int = 0
    training_steps: int = 0
    mean_quality: float = 0.0
    priority_updates: int = 0
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycle_id": str(self.cycle_id),
            "dreams_generated": self.dreams_generated,
            "high_quality_dreams": self.high_quality_dreams,
            "discarded_dreams": self.discarded_dreams,
            "training_steps": self.training_steps,
            "mean_quality": self.mean_quality,
            "priority_updates": self.priority_updates,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class DreamConsolidation:
    """
    Manage dream-based consolidation during REM sleep.

    P3-3 & P3-4: Integration with sleep consolidation.

    Workflow:
        1. Select high-error episodes from NREM as seeds
        2. Generate dream trajectories using DreamingSystem
        3. Evaluate dream quality using DreamQualityEvaluator
        4. Train predictor on high-quality dreams
        5. Update replay priorities based on dream insights

    Usage:
        consolidation = DreamConsolidation(encoder, predictor, tracker)

        # During REM phase
        result = consolidation.run_dream_cycle(
            recent_episodes=recent,
            reference_embeddings=all_embeddings,
        )

        # Get priority updates for next NREM
        priority_boost = consolidation.get_priority_updates()
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        latent_predictor: LatentPredictor,
        prediction_tracker: PredictionTracker | None = None,
        config: DreamConsolidationConfig | None = None,
    ):
        """
        Initialize dream consolidation.

        Args:
            context_encoder: Context encoder for dreaming
            latent_predictor: Predictor for trajectory generation
            prediction_tracker: Optional tracker for high-error episodes
            config: Consolidation configuration
        """
        self.config = config or DreamConsolidationConfig()

        # Initialize subsystems
        self.dreamer = DreamingSystem(
            context_encoder=context_encoder,
            latent_predictor=latent_predictor,
            config=self.config.dreaming_config,
        )
        self.evaluator = DreamQualityEvaluator(
            config=self.config.quality_config,
        )
        self.tracker = prediction_tracker
        self.encoder = context_encoder
        self.predictor = latent_predictor

        # Priority updates (episode_id -> boost amount)
        self._priority_boosts: dict[UUID, float] = {}

        # History
        self._cycle_history: list[DreamCycleResult] = []
        self._replay_events: list[DreamReplayEvent] = []
        self._max_history = 100

        # Statistics
        self._total_cycles = 0
        self._total_dreams = 0
        self._total_training_steps = 0

        logger.info("DreamConsolidation initialized")

    def run_dream_cycle(
        self,
        recent_episodes: list[tuple[UUID, np.ndarray]],
        reference_embeddings: list[np.ndarray] | None = None,
        high_error_episodes: list[tuple[UUID, np.ndarray]] | None = None,
    ) -> DreamCycleResult:
        """
        Run a complete dream cycle.

        Args:
            recent_episodes: Recent (episode_id, embedding) pairs for context
            reference_embeddings: Known embeddings for coherence checking
            high_error_episodes: Optional explicit high-error seeds

        Returns:
            DreamCycleResult with cycle statistics
        """
        start_time = datetime.now()
        result = DreamCycleResult()

        # Set up references for coherence and novelty
        if reference_embeddings:
            self.dreamer.add_reference_embeddings(reference_embeddings)
            self.evaluator.set_references(reference_embeddings)

        # Select seeds for dreaming
        seeds = self._select_dream_seeds(recent_episodes, high_error_episodes)

        if not seeds:
            logger.debug("No suitable seeds for dreaming")
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result

        # Build shared context from recent episodes
        context_embeddings = [emb for _, emb in recent_episodes[-5:]]

        # Generate dreams
        dreams: list[DreamTrajectory] = []
        qualities: list[DreamQuality] = []

        for seed_id, seed_emb in seeds[:self.config.dreams_per_cycle]:
            # Generate dream
            dream = self.dreamer.dream(
                seed_embedding=seed_emb,
                context_embeddings=context_embeddings,
                seed_episode_id=seed_id,
            )
            dreams.append(dream)

            # Evaluate quality
            quality = self.evaluator.evaluate(dream)
            qualities.append(quality)

            result.dreams_generated += 1

            if quality.is_high_quality:
                result.high_quality_dreams += 1

                # Train on high-quality dream
                if self.config.train_on_dreams:
                    loss = self._train_on_dream(dream)
                    result.training_steps += dream.length

                    # Record replay event
                    event = DreamReplayEvent(
                        dream_id=dream.id,
                        seed_episode_id=seed_id,
                        dream_length=dream.length,
                        quality_score=quality.overall_score,
                        training_loss=loss,
                    )
                    self._replay_events.append(event)

                # Update priority for seed episode
                if seed_id:
                    boost = quality.overall_score * self.config.priority_boost_factor
                    self._priority_boosts[seed_id] = (
                        self._priority_boosts.get(seed_id, 0) + boost
                    )
                    result.priority_updates += 1

            elif quality.overall_score < self.config.min_quality_for_replay:
                result.discarded_dreams += 1

        # Compute mean quality
        if qualities:
            result.mean_quality = float(np.mean([q.overall_score for q in qualities]))

        result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Update statistics
        self._total_cycles += 1
        self._total_dreams += result.dreams_generated
        self._total_training_steps += result.training_steps

        # Store in history
        self._cycle_history.append(result)
        if len(self._cycle_history) > self._max_history:
            self._cycle_history = self._cycle_history[-self._max_history:]

        logger.info(
            f"Dream cycle complete: {result.dreams_generated} dreams, "
            f"{result.high_quality_dreams} high-quality, "
            f"mean_quality={result.mean_quality:.3f}"
        )

        return result

    def _select_dream_seeds(
        self,
        recent_episodes: list[tuple[UUID, np.ndarray]],
        explicit_seeds: list[tuple[UUID, np.ndarray]] | None = None,
    ) -> list[tuple[UUID, np.ndarray]]:
        """Select episodes to seed dreams from."""
        seeds = []

        # Use explicit seeds if provided
        if explicit_seeds:
            seeds.extend(explicit_seeds)

        # Add high-error episodes from tracker
        if self.config.seed_from_high_error and self.tracker:
            high_error = self.tracker.get_high_error_episodes(
                k=self.config.dreams_per_cycle
            )
            # Match with recent episodes to get embeddings
            recent_map = {ep_id: emb for ep_id, emb in recent_episodes}
            for ep_id, error in high_error:
                if error >= self.config.min_error_for_seed and ep_id in recent_map:
                    seeds.append((ep_id, recent_map[ep_id]))

        # Fall back to most recent if no high-error seeds
        if not seeds and recent_episodes:
            seeds = recent_episodes[-self.config.dreams_per_cycle:]

        return seeds

    def _train_on_dream(self, dream: DreamTrajectory) -> float:
        """
        Train predictor on dream trajectory.

        Uses consecutive steps as (context → target) pairs.

        Args:
            dream: Dream trajectory to train on

        Returns:
            Average training loss
        """
        embeddings = dream.embeddings
        if len(embeddings) < 2:
            return 0.0

        losses = []
        for i in range(1, len(embeddings)):
            # Context: previous embeddings
            context_embs = embeddings[max(0, i - 5):i]
            target = embeddings[i]

            # Encode context
            encoded = self.encoder.encode(context_embs)

            # Train step
            loss = self.predictor.train_step(
                encoded.context_vector,
                target,
                learning_rate=self.config.dream_learning_rate,
            )
            losses.append(loss)

        return float(np.mean(losses)) if losses else 0.0

    def get_priority_updates(self) -> dict[UUID, float]:
        """
        Get priority boost updates for episodes.

        These should be added to consolidation priority scores.

        Returns:
            Dictionary of episode_id → priority boost
        """
        return dict(self._priority_boosts)

    def clear_priority_updates(self) -> None:
        """Clear priority updates after applying them."""
        self._priority_boosts.clear()

    def get_recent_dreams(self, n: int = 10) -> list[DreamTrajectory]:
        """Get most recent dreams."""
        return self.dreamer.get_recent_dreams(n)

    def get_high_quality_dreams(self, n: int = 10) -> list[DreamTrajectory]:
        """Get recent high-quality dreams."""
        return self.dreamer.get_high_quality_dreams(n)

    def get_statistics(self) -> dict[str, Any]:
        """Get consolidation statistics."""
        return {
            "total_cycles": self._total_cycles,
            "total_dreams": self._total_dreams,
            "total_training_steps": self._total_training_steps,
            "pending_priority_updates": len(self._priority_boosts),
            "dreamer_stats": self.dreamer.get_statistics(),
            "evaluator_stats": self.evaluator.get_statistics(),
        }

    def save_state(self) -> dict[str, Any]:
        """Save consolidation state."""
        return {
            "priority_boosts": {
                str(k): v for k, v in self._priority_boosts.items()
            },
            "dreamer_state": self.dreamer.save_state(),
            "evaluator_state": self.evaluator.save_state(),
            "statistics": self.get_statistics(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load consolidation state."""
        from uuid import UUID

        self._priority_boosts = {
            UUID(k): v for k, v in state.get("priority_boosts", {}).items()
        }

        if "dreamer_state" in state:
            self.dreamer.load_state(state["dreamer_state"])
