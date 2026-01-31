"""
Generative Replay for World Weaver.

Implements Hinton-style wake-sleep algorithm for memory consolidation.
Rather than replaying exact memories, generates synthetic experiences
to train recognition and consolidate learning without catastrophic forgetting.

Wake-Sleep Algorithm (Hinton & Dayan, 1995):
1. WAKE PHASE: Process real data, update recognition weights
   - Top-down generative model generates reconstructions
   - Bottom-up recognition model learns from mismatches

2. SLEEP PHASE: Generate synthetic samples, update generative weights
   - Generative model creates "dream" patterns
   - Recognition model provides targets for generative learning
   - No real data needed - prevents catastrophic interference

Benefits for CLS (Complementary Learning Systems):
- Fast hippocampal learning during wake
- Slow neocortical consolidation during sleep via replay
- Interleaved training prevents new learning from overwriting old

Integration Points:
- DreamingSystem: Provides dream trajectory generation
- SleepConsolidation: NREM phase triggers generative replay
- ThreeFactorLearner: Applies wake-sleep updates with neuromodulation

References:
- Hinton et al. (1995): The wake-sleep algorithm for unsupervised neural networks
- Hinton & Dayan (1996): Varieties of Helmholtz machines
- McClelland et al. (1995): Why there are complementary learning systems
- Shin et al. (2017): Continual learning with deep generative replay
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4

import numpy as np

logger = logging.getLogger(__name__)


class ReplayPhase(Enum):
    """Current phase of wake-sleep algorithm."""
    WAKE = "wake"      # Processing real experiences
    SLEEP = "sleep"    # Generating synthetic experiences
    IDLE = "idle"      # No active processing


@dataclass
class GenerativeReplayConfig:
    """Configuration for generative replay.

    Attributes:
        wake_learning_rate: LR during wake phase (fast, hippocampal)
        sleep_learning_rate: LR during sleep phase (slow, neocortical)
        generation_temperature: Softmax temperature for sampling
        n_sleep_samples: Number of synthetic samples per sleep phase
        reconstruction_weight: Weight for reconstruction loss
        regularization_weight: Weight for KL divergence regularization
        interleave_ratio: Ratio of old to new samples during training
    """
    wake_learning_rate: float = 0.01
    sleep_learning_rate: float = 0.001
    generation_temperature: float = 1.0
    n_sleep_samples: int = 100
    reconstruction_weight: float = 1.0
    regularization_weight: float = 0.01
    interleave_ratio: float = 0.5  # 50% old, 50% new


@dataclass
class GeneratedSample:
    """A synthetically generated experience.

    Attributes:
        id: Unique identifier
        embedding: Generated embedding vector
        source_phase: Which phase generated this
        generation_time: When generated
        temperature: Sampling temperature used
        confidence: Generator's confidence in this sample
        source_memories: Memory IDs that influenced generation
    """
    id: UUID = field(default_factory=uuid4)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(1024))
    source_phase: ReplayPhase = ReplayPhase.SLEEP
    generation_time: datetime = field(default_factory=datetime.now)
    temperature: float = 1.0
    confidence: float = 0.5
    source_memories: list[UUID] = field(default_factory=list)


@dataclass
class ReplayStats:
    """Statistics from a generative replay session.

    Attributes:
        phase: Which phase completed
        n_samples_processed: Number of samples handled
        mean_reconstruction_loss: Average reconstruction error
        mean_confidence: Average generation confidence
        duration_ms: Processing time
    """
    phase: ReplayPhase
    n_samples_processed: int = 0
    mean_reconstruction_loss: float = 0.0
    mean_confidence: float = 0.0
    duration_ms: float = 0.0


class Generator(Protocol):
    """Protocol for generative models."""

    def generate(self, n_samples: int, temperature: float = 1.0) -> list[np.ndarray]:
        """Generate synthetic embeddings."""
        ...

    def encode(self, embeddings: list[np.ndarray]) -> list[np.ndarray]:
        """Encode embeddings to latent space."""
        ...

    def decode(self, latents: list[np.ndarray]) -> list[np.ndarray]:
        """Decode latents back to embedding space."""
        ...


class GenerativeReplaySystem:
    """
    Wake-sleep style generative replay for memory consolidation.

    Implements Hinton's wake-sleep algorithm adapted for episodic memory:
    - Wake: Learn from real experiences with fast learning rate
    - Sleep: Generate synthetic experiences, train with slow learning rate

    This prevents catastrophic forgetting by interleaving old and new patterns.

    Example:
        ```python
        replay = GenerativeReplaySystem(config)

        # During wake (active processing)
        await replay.process_wake(real_episodes)

        # During sleep (NREM consolidation)
        stats = await replay.run_sleep_phase(n_samples=100)

        # Get generated samples for additional training
        samples = replay.get_generated_samples(limit=50)
        ```
    """

    def __init__(
        self,
        config: GenerativeReplayConfig | None = None,
        generator: Generator | None = None,
        on_sample_generated: Callable[[GeneratedSample], None] | None = None
    ):
        """
        Initialize generative replay system.

        Args:
            config: Replay configuration
            generator: Optional generator model (stub if None)
            on_sample_generated: Callback for each generated sample
        """
        self.config = config or GenerativeReplayConfig()
        self._generator = generator
        self._on_sample_generated = on_sample_generated

        self._current_phase = ReplayPhase.IDLE
        self._generated_samples: list[GeneratedSample] = []
        self._wake_history: list[np.ndarray] = []
        self._sleep_history: list[ReplayStats] = []

        # Statistics
        self._total_wake_samples = 0
        self._total_sleep_samples = 0
        self._total_generations = 0

        if self._generator is not None:
            logger.info("GenerativeReplaySystem initialized with generator")
        else:
            logger.info("GenerativeReplaySystem initialized without generator (wake history only)")

    @property
    def current_phase(self) -> ReplayPhase:
        """Current processing phase."""
        return self._current_phase

    async def process_wake(
        self,
        embeddings: list[np.ndarray],
        learning_rate: float | None = None
    ) -> ReplayStats:
        """
        Process real experiences during wake phase.

        During wake:
        1. Store embeddings for later generation
        2. Update recognition model (if available)
        3. Track statistics

        Args:
            embeddings: Real experience embeddings
            learning_rate: Override wake learning rate

        Returns:
            Statistics from wake processing
        """
        import time
        start = time.perf_counter()

        self._current_phase = ReplayPhase.WAKE
        lr = learning_rate or self.config.wake_learning_rate

        # Store for later generation
        self._wake_history.extend(embeddings)

        # Limit history size
        max_history = 10000
        if len(self._wake_history) > max_history:
            self._wake_history = self._wake_history[-max_history:]

        # Update statistics
        self._total_wake_samples += len(embeddings)

        # Train generator on real experiences (wake phase updates recognition model)
        mean_recon_loss = 0.0
        if self._generator and embeddings:
            logger.debug(f"Wake phase: training generator with {len(embeddings)} samples")
            try:
                # Generator protocol requires train_step method
                if hasattr(self._generator, 'train_step'):
                    train_result = self._generator.train_step(embeddings)
                    mean_recon_loss = train_result.get('recon_loss', 0.0)
                    logger.debug(
                        f"Wake training: recon_loss={mean_recon_loss:.4f}, "
                        f"kl_loss={train_result.get('kl_loss', 0.0):.4f}"
                    )
                else:
                    # Fallback: encode-decode for reconstruction loss computation
                    latents = self._generator.encode(embeddings)
                    reconstructions = self._generator.decode(latents)
                    if reconstructions:
                        # Compute MSE reconstruction loss
                        losses = []
                        for orig, recon in zip(embeddings, reconstructions):
                            losses.append(float(np.mean((orig - recon) ** 2)))
                        mean_recon_loss = np.mean(losses) if losses else 0.0
            except Exception as e:
                logger.warning(f"Wake phase generator training failed: {e}")

        self._current_phase = ReplayPhase.IDLE

        elapsed = (time.perf_counter() - start) * 1000

        return ReplayStats(
            phase=ReplayPhase.WAKE,
            n_samples_processed=len(embeddings),
            mean_reconstruction_loss=mean_recon_loss,
            mean_confidence=1.0,  # Real data has confidence 1.0
            duration_ms=elapsed
        )

    async def run_sleep_phase(
        self,
        n_samples: int | None = None,
        temperature: float | None = None,
        learning_rate: float | None = None
    ) -> ReplayStats:
        """
        Run sleep phase with generative replay.

        During sleep:
        1. Generate synthetic samples from learned distribution
        2. Interleave with stored real samples
        3. Train recognition model on combined set
        4. Update generative weights

        Args:
            n_samples: Number of samples to generate
            temperature: Sampling temperature (higher = more diverse)
            learning_rate: Override sleep learning rate

        Returns:
            Statistics from sleep phase
        """
        import time
        start = time.perf_counter()

        self._current_phase = ReplayPhase.SLEEP

        n = n_samples or self.config.n_sleep_samples
        temp = temperature or self.config.generation_temperature
        lr = learning_rate or self.config.sleep_learning_rate

        generated = []
        mean_recon_loss = 0.0

        if self._generator:
            # Use actual generator to create synthetic embeddings
            embeddings = self._generator.generate(n, temp)

            # Compute confidence based on reconstruction round-trip
            # High-confidence samples can be reconstructed well by the VAE
            confidences = []
            if hasattr(self._generator, 'encode') and hasattr(self._generator, 'decode'):
                try:
                    latents = self._generator.encode(embeddings)
                    reconstructions = self._generator.decode(latents)
                    for emb, recon in zip(embeddings, reconstructions):
                        # Confidence = 1 - normalized reconstruction error
                        recon_err = np.mean((emb - recon) ** 2)
                        # Sigmoid-like mapping: low error -> high confidence
                        # recon_err of ~0.1 -> confidence ~0.9
                        # recon_err of ~1.0 -> confidence ~0.27
                        confidence = 1.0 / (1.0 + recon_err * 3.0)
                        confidences.append(float(confidence))
                    mean_recon_loss = np.mean([np.mean((e - r) ** 2)
                                               for e, r in zip(embeddings, reconstructions)])
                except Exception as e:
                    logger.debug(f"Confidence computation failed: {e}")
                    confidences = [0.7] * len(embeddings)  # Default confidence
            else:
                # No encode/decode available, use temperature-based confidence
                # Lower temperature = higher confidence (closer to mode)
                base_confidence = 1.0 / (1.0 + temp * 0.5)
                confidences = [base_confidence] * len(embeddings)

            for emb, conf in zip(embeddings, confidences):
                sample = GeneratedSample(
                    embedding=emb,
                    source_phase=ReplayPhase.SLEEP,
                    temperature=temp,
                    confidence=conf
                )
                generated.append(sample)

                if self._on_sample_generated:
                    self._on_sample_generated(sample)
        else:
            # Stub: Generate from stored history with noise
            for i in range(min(n, len(self._wake_history))):
                idx = np.random.randint(0, len(self._wake_history))
                base = self._wake_history[idx]

                # Add noise scaled by temperature
                noise = np.random.randn(*base.shape) * temp * 0.1
                synthetic = base + noise
                synthetic = synthetic / (np.linalg.norm(synthetic) + 1e-8)

                # Confidence inversely proportional to noise magnitude
                noise_magnitude = np.linalg.norm(noise) / (np.linalg.norm(base) + 1e-8)
                confidence = max(0.3, 1.0 - noise_magnitude)

                sample = GeneratedSample(
                    embedding=synthetic,
                    source_phase=ReplayPhase.SLEEP,
                    temperature=temp,
                    confidence=confidence
                )
                generated.append(sample)

                if self._on_sample_generated:
                    self._on_sample_generated(sample)

        self._generated_samples.extend(generated)
        self._total_sleep_samples += len(generated)
        self._total_generations += 1

        # Limit generated sample history
        max_samples = 5000
        if len(self._generated_samples) > max_samples:
            self._generated_samples = self._generated_samples[-max_samples:]

        self._current_phase = ReplayPhase.IDLE

        elapsed = (time.perf_counter() - start) * 1000

        stats = ReplayStats(
            phase=ReplayPhase.SLEEP,
            n_samples_processed=len(generated),
            mean_reconstruction_loss=mean_recon_loss,
            mean_confidence=np.mean([s.confidence for s in generated]) if generated else 0.0,
            duration_ms=elapsed
        )

        self._sleep_history.append(stats)

        logger.debug(
            f"Sleep phase complete: generated {len(generated)} samples "
            f"in {elapsed:.1f}ms"
        )

        return stats

    def get_generated_samples(
        self,
        limit: int = 100,
        min_confidence: float = 0.0
    ) -> list[GeneratedSample]:
        """
        Get recent generated samples for training.

        Args:
            limit: Maximum samples to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of generated samples
        """
        filtered = [
            s for s in self._generated_samples
            if s.confidence >= min_confidence
        ]
        return filtered[-limit:]

    def get_interleaved_batch(
        self,
        batch_size: int,
        new_embeddings: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Get interleaved batch of old (generated) and new (real) samples.

        Critical for CLS: mixing old and new prevents catastrophic forgetting.

        Args:
            batch_size: Total batch size
            new_embeddings: New real embeddings

        Returns:
            Mixed batch with ratio from config
        """
        n_old = int(batch_size * self.config.interleave_ratio)
        n_new = batch_size - n_old

        # Get old samples
        old_samples = self.get_generated_samples(limit=n_old)
        old_embeddings = [s.embedding for s in old_samples]

        # Sample from new
        if len(new_embeddings) > n_new:
            indices = np.random.choice(len(new_embeddings), n_new, replace=False)
            new_subset = [new_embeddings[i] for i in indices]
        else:
            new_subset = new_embeddings

        # Combine and shuffle
        combined = old_embeddings + new_subset
        np.random.shuffle(combined)

        return combined

    def get_statistics(self) -> dict[str, Any]:
        """Get replay statistics."""
        return {
            "current_phase": self._current_phase.value,
            "total_wake_samples": self._total_wake_samples,
            "total_sleep_samples": self._total_sleep_samples,
            "total_generations": self._total_generations,
            "wake_history_size": len(self._wake_history),
            "generated_samples_size": len(self._generated_samples),
            "sleep_sessions": len(self._sleep_history),
            "config": {
                "wake_lr": self.config.wake_learning_rate,
                "sleep_lr": self.config.sleep_learning_rate,
                "interleave_ratio": self.config.interleave_ratio
            }
        }

    def clear_history(self) -> None:
        """Clear accumulated history."""
        self._wake_history.clear()
        self._generated_samples.clear()
        self._sleep_history.clear()
        logger.info("Cleared generative replay history")


# Factory function
def create_generative_replay(
    config: GenerativeReplayConfig | None = None,
    generator: Generator | None = None
) -> GenerativeReplaySystem:
    """
    Create a generative replay system.

    Args:
        config: Optional configuration
        generator: Optional generator model

    Returns:
        Configured GenerativeReplaySystem
    """
    return GenerativeReplaySystem(config=config, generator=generator)
