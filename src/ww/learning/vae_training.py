"""
VAE Training System for World Weaver.

Phase 5A: Training pipeline for VAE-based generative replay.

This module connects episodic memory collection with VAE training,
enabling the VAE to generate realistic synthetic memories for replay
during sleep consolidation phases.

Workflow:
1. During WAKE: Collect recent embeddings into training buffer
2. Before SLEEP: Train VAE on collected samples
3. During SLEEP: VAE generates synthetic samples for replay

Biological Mapping:
- Wake collection = hippocampal encoding of new experiences
- VAE training = synaptic plasticity from rehearsal
- Sleep replay = memory consolidation via generative dreams

References:
- Shin et al. (2017): Continual Learning with Deep Generative Replay
- Kumaran et al. (2016): What Learning Systems do Intelligent Agents Need?
- McClelland et al. (1995): Why there are complementary learning systems
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol
from collections import deque

import numpy as np

if TYPE_CHECKING:
    from ww.learning.vae_generator import VAEGenerator

logger = logging.getLogger(__name__)

# Configuration limits
MAX_TRAINING_BUFFER = 10000
MAX_EPOCHS = 100
MIN_BATCH_SIZE = 4


class EpisodicMemoryProtocol(Protocol):
    """Protocol for episodic memory to enable type checking."""

    async def get_recent(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> list[Any]:
        """Get recent episodes."""
        ...


@dataclass
class VAETrainingConfig:
    """Configuration for VAE training.

    Attributes:
        buffer_size: Maximum samples in training buffer
        default_collection_hours: Default lookback for sample collection
        default_collection_limit: Default max samples to collect
        min_samples_for_training: Minimum samples before training allowed
        epochs_per_training: Epochs per training session
        batch_size: Training batch size
        warmup_epochs: Epochs before logging starts
        early_stopping_patience: Epochs without improvement before stopping
        early_stopping_min_delta: Minimum improvement to reset patience
        log_interval: Batches between log messages
    """
    buffer_size: int = MAX_TRAINING_BUFFER
    default_collection_hours: int = 24
    default_collection_limit: int = 100
    min_samples_for_training: int = 32
    epochs_per_training: int = 10
    batch_size: int = 32
    warmup_epochs: int = 2
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    log_interval: int = 10


@dataclass
class TrainingStats:
    """Statistics from a VAE training session.

    Attributes:
        epochs_completed: Number of epochs run
        total_batches: Total batches processed
        samples_trained: Number of samples trained on
        final_loss: Loss at end of training
        best_loss: Best loss achieved
        mean_loss: Mean loss across all batches
        stopped_early: Whether early stopping triggered
        training_time_seconds: Duration of training
        timestamp: When training completed
    """
    epochs_completed: int = 0
    total_batches: int = 0
    samples_trained: int = 0
    final_loss: float = float('inf')
    best_loss: float = float('inf')
    mean_loss: float = float('inf')
    stopped_early: bool = False
    training_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return {
            "epochs_completed": self.epochs_completed,
            "total_batches": self.total_batches,
            "samples_trained": self.samples_trained,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "mean_loss": self.mean_loss,
            "stopped_early": self.stopped_early,
            "training_time_seconds": self.training_time_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


class VAEReplayTrainer:
    """Trains VAE on collected wake-time experiences.

    Manages:
    1. Collection of recent embeddings from episodic memory
    2. Buffering of samples for efficient training
    3. VAE training on collected samples
    4. Generation of synthetic memories for replay
    """

    def __init__(
        self,
        vae: VAEGenerator,
        memory: EpisodicMemoryProtocol,
        config: VAETrainingConfig | None = None,
    ):
        """
        Initialize VAE trainer.

        Args:
            vae: Trained/untrained VAE generator
            memory: Episodic memory for sample collection
            config: Training configuration
        """
        self.vae = vae
        self.memory = memory
        self.config = config or VAETrainingConfig()

        # Training buffer (stores embeddings as vectors)
        self._training_buffer: deque = deque(maxlen=self.config.buffer_size)

        # Statistics tracking
        self._training_count = 0
        self._training_history: list[TrainingStats] = []
        self._total_samples_collected = 0

        logger.info(
            f"VAEReplayTrainer initialized: "
            f"buffer_size={self.config.buffer_size}, "
            f"min_samples={self.config.min_samples_for_training}"
        )

    async def collect_wake_samples(
        self,
        n_samples: int | None = None,
        hours: int | None = None,
    ) -> int:
        """
        Collect recent embeddings from episodic memory.

        Args:
            n_samples: Number of samples to collect
            hours: Lookback window in hours

        Returns:
            Number of samples collected
        """
        n_samples = n_samples or self.config.default_collection_limit
        hours = hours or self.config.default_collection_hours

        try:
            episodes = await self.memory.get_recent(hours=hours, limit=n_samples)

            collected = 0
            for episode in episodes:
                # Extract embedding from episode
                if hasattr(episode, 'embedding'):
                    embedding = episode.embedding
                    # Convert from list to array if needed
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)
                    self.add_sample(embedding)  # add_sample increments _total_samples_collected
                    collected += 1
            logger.debug(
                f"Collected {collected} samples from episodic memory "
                f"(last {hours} hours)"
            )
            return collected

        except Exception as e:
            logger.error(f"Error collecting wake samples: {e}")
            return 0

    def add_sample(self, embedding: np.ndarray) -> None:
        """
        Add single embedding to training buffer.

        Args:
            embedding: Embedding vector to add
        """
        if isinstance(embedding, (list, tuple)):
            embedding = np.array(embedding, dtype=np.float32)
        else:
            embedding = np.asarray(embedding, dtype=np.float32)

        self._training_buffer.append(embedding)
        self._total_samples_collected += 1

    def add_samples(self, embeddings: list[np.ndarray]) -> int:
        """
        Add multiple embeddings to buffer.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Number of samples added
        """
        added = 0
        for embedding in embeddings:
            self.add_sample(embedding)
            added += 1
        return added

    def get_buffer_size(self) -> int:
        """Get current number of samples in buffer."""
        return len(self._training_buffer)

    def clear_buffer(self) -> None:
        """Clear training buffer."""
        self._training_buffer.clear()
        logger.debug("Training buffer cleared")

    def train_vae(
        self,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> TrainingStats:
        """
        Train VAE on collected samples.

        Runs training for specified epochs using samples in the
        training buffer.

        Args:
            epochs: Number of epochs (default from config)
            batch_size: Batch size (default from config)

        Returns:
            Training statistics
        """
        epochs = min(epochs or self.config.epochs_per_training, MAX_EPOCHS)
        batch_size = max(batch_size or self.config.batch_size, MIN_BATCH_SIZE)

        start_time = datetime.now()
        self._training_count += 1

        stats = TrainingStats()

        # Check minimum samples
        if len(self._training_buffer) < self.config.min_samples_for_training:
            logger.warning(
                f"Insufficient samples for training: "
                f"{len(self._training_buffer)} < {self.config.min_samples_for_training}"
            )
            return stats

        # Convert buffer to array
        data = np.array(list(self._training_buffer))
        n_samples = len(data)

        logger.info(
            f"Starting VAE training: {epochs} epochs, "
            f"{n_samples} samples, batch_size={batch_size}"
        )

        all_losses = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            epoch_losses = []

            # Process batches
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_indices = indices[batch_start:batch_end]
                batch = data[batch_indices]

                # Train step - may return dict or float
                loss_result = self.vae.train_step(batch)
                # Handle both dict and float returns for compatibility
                if isinstance(loss_result, dict):
                    loss = float(loss_result.get('total_loss', 0.0))
                else:
                    loss = float(loss_result)
                epoch_losses.append(loss)
                stats.total_batches += 1

                # Periodic logging
                if stats.total_batches % self.config.log_interval == 0:
                    logger.debug(
                        f"VAE batch {stats.total_batches}: loss={loss:.4f}"
                    )

            epoch_loss = np.mean(epoch_losses)
            all_losses.extend(epoch_losses)

            # Log epoch
            if epoch >= self.config.warmup_epochs:
                logger.info(f"VAE epoch {epoch + 1}/{epochs}: loss={epoch_loss:.4f}")

            # Early stopping check
            if epoch_loss < best_loss - self.config.early_stopping_min_delta:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                stats.stopped_early = True
                break

        # Compute final stats
        stats.epochs_completed = epoch + 1
        stats.samples_trained = n_samples
        stats.training_time_seconds = (datetime.now() - start_time).total_seconds()

        if all_losses:
            stats.final_loss = float(all_losses[-1])
            stats.best_loss = float(min(all_losses))
            stats.mean_loss = float(np.mean(all_losses))

        self._training_history.append(stats)

        logger.info(
            f"VAE training complete: {stats.epochs_completed} epochs, "
            f"final_loss={stats.final_loss:.4f}, best={stats.best_loss:.4f}, "
            f"time={stats.training_time_seconds:.1f}s"
        )

        return stats

    def generate_for_replay(
        self,
        n_samples: int = 50,
        temperature: float = 0.8,
    ) -> list[np.ndarray]:
        """
        Generate synthetic memories for replay.

        Uses the trained VAE to generate synthetic embeddings
        for interleaved replay during sleep consolidation.

        Args:
            n_samples: Number of samples to generate
            temperature: Sampling temperature (lower = more realistic)

        Returns:
            List of generated embedding vectors
        """
        generated = self.vae.generate(n_samples, temperature=temperature)

        logger.debug(
            f"Generated {len(generated)} synthetic memories "
            f"(temp={temperature})"
        )

        return generated

    def interleave_with_real(
        self,
        synthetic: list[np.ndarray],
        real: list[np.ndarray],
        interleave_ratio: float = 0.5,
    ) -> list[np.ndarray]:
        """
        Interleave synthetic and real memories for replay.

        Mixes generated synthetic memories with real ones to
        prevent catastrophic forgetting (CLS theory).

        Args:
            synthetic: Synthetic embeddings from VAE
            real: Real embeddings from memory
            interleave_ratio: Proportion of synthetic to use

        Returns:
            Mixed list of synthetic and real embeddings
        """
        n_synthetic = int(len(real) * interleave_ratio)
        n_synthetic = min(n_synthetic, len(synthetic))

        # Select random synthetic samples
        indices = np.random.choice(len(synthetic), n_synthetic, replace=False)
        selected_synthetic = [synthetic[i] for i in indices]

        # Mix with real samples
        mixed = selected_synthetic + real

        logger.debug(
            f"Interleaved {n_synthetic} synthetic with {len(real)} real "
            f"(ratio={interleave_ratio})"
        )

        return mixed

    def get_training_history(self) -> list[dict]:
        """Get training history as dicts."""
        return [stats.to_dict() for stats in self._training_history]

    def get_statistics(self) -> dict:
        """Get current trainer statistics."""
        return {
            'training_count': self._training_count,
            'total_samples_collected': self._total_samples_collected,
            'current_buffer_size': len(self._training_buffer),
            'history_length': len(self._training_history),
        }
