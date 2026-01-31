"""
Dream Trajectory Generation.

P3-1: Generate imagined trajectories from seed memories.

Biological Basis:
- REM sleep generates "dreams" by replaying and extending memories
- Dreams are constrained by learned patterns (latent manifold)
- Prediction errors during dreams indicate world model gaps

DreamerV3 Insight:
- Imagine 15 steps ahead in latent space
- Stochastic latent absorbs prediction uncertainty
- Train on imagined trajectories, not just real experience

Architecture:
- Seed: High-error episode embedding from NREM
- Context: Recent trajectory + working memory
- Prediction: LatentPredictor generates next step
- Termination: Confidence drops or coherence fails
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from t4dm.prediction import ContextEncoder, LatentPredictor

logger = logging.getLogger(__name__)


@dataclass
class DreamingConfig:
    """Configuration for dreaming system."""

    # Trajectory parameters
    max_dream_length: int = 15  # DreamerV3 uses 15 steps
    min_dream_length: int = 3  # Minimum for meaningful dream
    context_window: int = 5  # Episodes for context

    # Termination conditions
    confidence_threshold: float = 0.3  # Stop if confidence drops below
    coherence_threshold: float = 0.5  # Stop if trajectory diverges

    # Stochasticity (biological variability)
    noise_scale: float = 0.05  # Small noise for dream variability
    temperature: float = 1.0  # Sampling temperature

    # Seeding
    prefer_high_error: bool = True  # Seed from surprising episodes
    error_threshold: float = 0.3  # Minimum error for dream seeding


@dataclass
class DreamStep:
    """A single step in a dream trajectory."""

    embedding: np.ndarray  # Predicted/imagined embedding
    confidence: float  # Prediction confidence
    coherence: float  # Trajectory coherence at this step
    step_index: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embedding": self.embedding.tolist(),
            "confidence": self.confidence,
            "coherence": self.coherence,
            "step_index": self.step_index,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DreamTrajectory:
    """A complete dream trajectory."""

    id: UUID = field(default_factory=uuid4)
    seed_episode_id: UUID | None = None
    seed_embedding: np.ndarray | None = None
    steps: list[DreamStep] = field(default_factory=list)
    termination_reason: str = "incomplete"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    @property
    def length(self) -> int:
        """Number of steps in trajectory."""
        return len(self.steps)

    @property
    def embeddings(self) -> list[np.ndarray]:
        """All embeddings in trajectory."""
        embeddings = []
        if self.seed_embedding is not None:
            embeddings.append(self.seed_embedding)
        embeddings.extend([s.embedding for s in self.steps])
        return embeddings

    @property
    def mean_confidence(self) -> float:
        """Average confidence across trajectory."""
        if not self.steps:
            return 0.0
        return float(np.mean([s.confidence for s in self.steps]))

    @property
    def mean_coherence(self) -> float:
        """Average coherence across trajectory."""
        if not self.steps:
            return 0.0
        return float(np.mean([s.coherence for s in self.steps]))

    @property
    def duration_ms(self) -> float:
        """Dream duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "seed_episode_id": str(self.seed_episode_id) if self.seed_episode_id else None,
            "length": self.length,
            "mean_confidence": self.mean_confidence,
            "mean_coherence": self.mean_coherence,
            "termination_reason": self.termination_reason,
            "duration_ms": self.duration_ms,
            "steps": [s.to_dict() for s in self.steps],
        }


class DreamingSystem:
    """
    Generate imagined trajectories during REM sleep.

    P3-1: The core dreaming engine.

    Workflow:
        1. Receive seed (high-error episode from NREM)
        2. Build context from seed + recent memories
        3. Predict next state using LatentPredictor
        4. Add noise for dream variability
        5. Check coherence with learned manifold
        6. Repeat until termination condition

    Usage:
        dreamer = DreamingSystem(encoder, predictor)
        trajectory = dreamer.dream(seed_embedding, context_embeddings)
    """

    def __init__(
        self,
        context_encoder: ContextEncoder,
        latent_predictor: LatentPredictor,
        config: DreamingConfig | None = None,
    ):
        """
        Initialize dreaming system.

        Args:
            context_encoder: Encoder for context representations
            latent_predictor: Predictor for next latent state
            config: Dreaming configuration
        """
        self.encoder = context_encoder
        self.predictor = latent_predictor
        self.config = config or DreamingConfig()

        # Dream history
        self._dream_history: list[DreamTrajectory] = []
        self._max_history = 1000

        # Statistics
        self._total_dreams = 0
        self._total_steps = 0
        self._terminated_by_confidence = 0
        self._terminated_by_coherence = 0
        self._terminated_by_length = 0

        # Reference embeddings for coherence checking
        self._reference_embeddings: list[np.ndarray] = []
        self._max_references = 500

        logger.info(
            f"DreamingSystem initialized: "
            f"max_length={self.config.max_dream_length}, "
            f"confidence_threshold={self.config.confidence_threshold}"
        )

    def dream(
        self,
        seed_embedding: np.ndarray,
        context_embeddings: list[np.ndarray] | None = None,
        seed_episode_id: UUID | None = None,
    ) -> DreamTrajectory:
        """
        Generate a dream trajectory from seed.

        Args:
            seed_embedding: Starting point for dream
            context_embeddings: Optional additional context
            seed_episode_id: ID of seed episode (for tracking)

        Returns:
            DreamTrajectory with imagined steps
        """
        trajectory = DreamTrajectory(
            seed_episode_id=seed_episode_id,
            seed_embedding=seed_embedding.copy(),
        )

        # Build initial context
        context = [seed_embedding]
        if context_embeddings:
            context = list(context_embeddings[-self.config.context_window:]) + [seed_embedding]

        # Dream loop
        for step_idx in range(self.config.max_dream_length):
            # Encode context
            encoded = self.encoder.encode(context[-self.config.context_window:])

            # Predict next state
            prediction = self.predictor.predict(encoded.context_vector)

            # Add stochastic noise (dream variability)
            noisy_embedding = self._add_dream_noise(prediction.predicted_embedding)

            # Compute coherence with learned manifold
            coherence = self._compute_coherence(noisy_embedding, context)

            # Create dream step
            step = DreamStep(
                embedding=noisy_embedding,
                confidence=prediction.confidence,
                coherence=coherence,
                step_index=step_idx,
            )
            trajectory.steps.append(step)

            # Check termination conditions
            if prediction.confidence < self.config.confidence_threshold:
                trajectory.termination_reason = "low_confidence"
                self._terminated_by_confidence += 1
                break

            if coherence < self.config.coherence_threshold:
                trajectory.termination_reason = "low_coherence"
                self._terminated_by_coherence += 1
                break

            # Update context for next step
            context.append(noisy_embedding)

        else:
            trajectory.termination_reason = "max_length"
            self._terminated_by_length += 1

        trajectory.end_time = datetime.now()

        # Track statistics
        self._total_dreams += 1
        self._total_steps += trajectory.length

        # Store in history
        self._dream_history.append(trajectory)
        if len(self._dream_history) > self._max_history:
            self._dream_history = self._dream_history[-self._max_history:]

        logger.debug(
            f"Generated dream: {trajectory.length} steps, "
            f"confidence={trajectory.mean_confidence:.3f}, "
            f"coherence={trajectory.mean_coherence:.3f}, "
            f"termination={trajectory.termination_reason}"
        )

        return trajectory

    def dream_batch(
        self,
        seed_embeddings: list[tuple[UUID | None, np.ndarray]],
        shared_context: list[np.ndarray] | None = None,
    ) -> list[DreamTrajectory]:
        """
        Generate multiple dream trajectories.

        Args:
            seed_embeddings: List of (episode_id, embedding) seeds
            shared_context: Optional shared context for all dreams

        Returns:
            List of DreamTrajectory objects
        """
        trajectories = []
        for episode_id, embedding in seed_embeddings:
            trajectory = self.dream(
                seed_embedding=embedding,
                context_embeddings=shared_context,
                seed_episode_id=episode_id,
            )
            trajectories.append(trajectory)
        return trajectories

    def add_reference_embeddings(self, embeddings: list[np.ndarray]) -> None:
        """
        Add reference embeddings for coherence checking.

        These represent the "learned manifold" - patterns the
        dreaming system should stay close to.

        Args:
            embeddings: Embeddings to add as references
        """
        self._reference_embeddings.extend(embeddings)
        if len(self._reference_embeddings) > self._max_references:
            # Keep most recent
            self._reference_embeddings = self._reference_embeddings[-self._max_references:]

    def _add_dream_noise(self, embedding: np.ndarray) -> np.ndarray:
        """Add stochastic noise for dream variability."""
        noise = np.random.randn(*embedding.shape).astype(np.float32)
        noise *= self.config.noise_scale * self.config.temperature

        noisy = embedding + noise
        # Re-normalize
        norm = np.linalg.norm(noisy)
        if norm > 0:
            noisy = noisy / norm

        return noisy

    def _compute_coherence(
        self,
        embedding: np.ndarray,
        context: list[np.ndarray],
    ) -> float:
        """
        Compute coherence of embedding with trajectory and references.

        High coherence = stays on learned manifold.
        Low coherence = diverging into nonsense.
        """
        scores = []

        # Coherence with recent context (smooth trajectory)
        if context:
            last_emb = context[-1]
            context_sim = float(np.dot(embedding, last_emb))
            # Expect high similarity with previous step
            scores.append(max(0, context_sim))

        # Coherence with reference embeddings (on manifold)
        if self._reference_embeddings:
            ref_sims = [
                float(np.dot(embedding, ref))
                for ref in self._reference_embeddings[-50:]  # Check recent references
            ]
            max_ref_sim = max(ref_sims) if ref_sims else 0.5
            scores.append(max_ref_sim)

        if not scores:
            return 1.0  # No references = assume coherent

        return float(np.mean(scores))

    def get_recent_dreams(self, n: int = 10) -> list[DreamTrajectory]:
        """Get most recent dream trajectories."""
        return self._dream_history[-n:]

    def get_high_quality_dreams(
        self,
        n: int = 10,
        min_confidence: float = 0.5,
        min_coherence: float = 0.5,
        min_length: int = 5,
    ) -> list[DreamTrajectory]:
        """
        Get high-quality dreams for consolidation.

        Args:
            n: Maximum number to return
            min_confidence: Minimum mean confidence
            min_coherence: Minimum mean coherence
            min_length: Minimum trajectory length

        Returns:
            High-quality dream trajectories
        """
        qualified = [
            d for d in self._dream_history
            if d.mean_confidence >= min_confidence
            and d.mean_coherence >= min_coherence
            and d.length >= min_length
        ]

        # Sort by quality (confidence * coherence * length)
        qualified.sort(
            key=lambda d: d.mean_confidence * d.mean_coherence * d.length,
            reverse=True,
        )

        return qualified[:n]

    def get_statistics(self) -> dict[str, Any]:
        """Get dreaming statistics."""
        return {
            "total_dreams": self._total_dreams,
            "total_steps": self._total_steps,
            "average_length": self._total_steps / max(1, self._total_dreams),
            "terminated_by_confidence": self._terminated_by_confidence,
            "terminated_by_coherence": self._terminated_by_coherence,
            "terminated_by_length": self._terminated_by_length,
            "reference_embeddings": len(self._reference_embeddings),
            "dream_history_size": len(self._dream_history),
        }

    def save_state(self) -> dict[str, Any]:
        """Save dreaming system state."""
        return {
            "config": {
                "max_dream_length": self.config.max_dream_length,
                "min_dream_length": self.config.min_dream_length,
                "confidence_threshold": self.config.confidence_threshold,
                "coherence_threshold": self.config.coherence_threshold,
                "noise_scale": self.config.noise_scale,
            },
            "reference_embeddings": [
                emb.tolist() for emb in self._reference_embeddings[-100:]
            ],
            "statistics": self.get_statistics(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load dreaming system state."""
        self._reference_embeddings = [
            np.array(emb, dtype=np.float32)
            for emb in state.get("reference_embeddings", [])
        ]
        # Statistics are informational only, not restored
