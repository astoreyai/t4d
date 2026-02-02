"""
Homeostatic Plasticity for T4DM.

Biological Basis:
- Synaptic scaling maintains average firing rates despite Hebbian learning
- Prevents runaway potentiation/depression
- Operates on slower timescale than Hebbian plasticity
- BCM (Bienenstock-Cooper-Munro) theory: sliding threshold for LTP/LTD

Implementation:
1. Track running statistics of embedding norms and activations
2. Apply global scaling when statistics drift from target
3. Decorrelation penalty to reduce interference between memories
4. Sliding threshold for reconsolidation updates

Integration Points:
1. ReconsolidationEngine: Apply after batch updates
2. EpisodicMemory: Periodic normalization of stored embeddings
3. DopamineModulatedReconsolidation: Homeostatic bounds on learning rate
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HomeostaticState:
    """Current state of homeostatic regulation."""

    mean_norm: float = 1.0
    std_norm: float = 0.1
    mean_activation: float = 0.0
    sliding_threshold: float = 0.5
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mean_norm": self.mean_norm,
            "std_norm": self.std_norm,
            "mean_activation": self.mean_activation,
            "sliding_threshold": self.sliding_threshold,
            "last_update": self.last_update.isoformat(),
        }


class HomeostaticPlasticity:
    """
    Homeostatic regulation of memory embeddings.

    Prevents runaway potentiation by:
    1. Tracking running statistics of embedding norms
    2. Applying global scaling when norms drift from target
    3. Implementing BCM-like sliding threshold for updates
    4. Decorrelating embeddings to reduce interference

    The key insight from synaptic scaling: when Hebbian learning
    strengthens some synapses, homeostatic mechanisms globally
    scale all synapses to maintain average activity.
    """

    def __init__(
        self,
        target_norm: float = 1.0,
        norm_tolerance: float = 0.2,
        ema_alpha: float = 0.01,
        decorrelation_strength: float = 0.01,
        sliding_threshold_rate: float = 0.001,
        history_size: int = 1000
    ):
        """
        Initialize homeostatic plasticity.

        Args:
            target_norm: Target L2 norm for embeddings
            norm_tolerance: Allowed deviation before scaling triggers
            ema_alpha: Exponential moving average rate for statistics
            decorrelation_strength: Strength of decorrelation penalty
            sliding_threshold_rate: Rate of threshold adaptation (BCM)
            history_size: Number of recent norms to track
        """
        self.target_norm = target_norm
        self.norm_tolerance = norm_tolerance
        self.ema_alpha = ema_alpha
        self.decorrelation_strength = decorrelation_strength
        self.sliding_threshold_rate = sliding_threshold_rate

        # Running statistics
        self._norm_history: deque[float] = deque(maxlen=history_size)
        self._state = HomeostaticState()

        # Scaling events for analysis
        self._scaling_count = 0
        self._decorrelation_count = 0

    def update_statistics(self, embeddings: np.ndarray) -> None:
        """
        Update running statistics with new embeddings.

        Args:
            embeddings: Array of embeddings (N x D)
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        norms = np.linalg.norm(embeddings, axis=1)

        for norm in norms:
            self._norm_history.append(float(norm))

        # Update EMA statistics
        batch_mean = float(np.mean(norms))
        batch_std = float(np.std(norms)) if len(norms) > 1 else 0.1

        self._state.mean_norm = (
            (1 - self.ema_alpha) * self._state.mean_norm +
            self.ema_alpha * batch_mean
        )
        self._state.std_norm = (
            (1 - self.ema_alpha) * self._state.std_norm +
            self.ema_alpha * batch_std
        )
        self._state.last_update = datetime.now()

    def needs_scaling(self) -> bool:
        """
        Check if global scaling is needed.

        Returns:
            True if mean norm deviates beyond tolerance
        """
        deviation = abs(self._state.mean_norm - self.target_norm)
        return deviation > self.norm_tolerance

    def compute_scaling_factor(self) -> float:
        """
        Compute scaling factor to restore target norm.

        Returns:
            Multiplicative scaling factor
        """
        if self._state.mean_norm < 1e-8:
            return 1.0

        return self.target_norm / self._state.mean_norm

    def apply_scaling(
        self,
        embeddings: np.ndarray,
        force: bool = False
    ) -> np.ndarray:
        """
        Apply homeostatic scaling to embeddings.

        Args:
            embeddings: Array of embeddings (N x D) or single embedding (D,)
            force: Apply scaling even if not needed

        Returns:
            Scaled embeddings
        """
        if not force and not self.needs_scaling():
            return embeddings

        scale = self.compute_scaling_factor()
        self._scaling_count += 1

        logger.debug(
            f"Homeostatic scaling: mean_norm={self._state.mean_norm:.3f}, "
            f"target={self.target_norm:.3f}, scale={scale:.3f}"
        )

        return embeddings * scale

    def decorrelate(
        self,
        embeddings: np.ndarray,
        strength: float | None = None
    ) -> np.ndarray:
        """
        Apply decorrelation to reduce embedding interference.

        Uses whitening-like transform to reduce off-diagonal correlations.
        This is inspired by lateral inhibition and anti-Hebbian plasticity.

        Args:
            embeddings: Array of embeddings (N x D)
            strength: Override decorrelation strength

        Returns:
            Decorrelated embeddings
        """
        if embeddings.ndim == 1 or len(embeddings) < 2:
            return embeddings

        strength = strength if strength is not None else self.decorrelation_strength

        # Center embeddings
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean

        # Compute covariance
        np.cov(centered.T)

        # Compute off-diagonal penalty gradient
        # We want to reduce correlations between different embeddings
        n = len(embeddings)
        decorr_grad = np.zeros_like(embeddings)

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Gradient to reduce dot product between i and j
                    decorr_grad[i] -= strength * embeddings[j]

        # Apply decorrelation step
        decorrelated = embeddings + decorr_grad

        # Renormalize
        norms = np.linalg.norm(decorrelated, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        decorrelated = decorrelated / norms

        self._decorrelation_count += 1

        return decorrelated

    def update_sliding_threshold(
        self,
        recent_activations: np.ndarray
    ) -> float:
        """
        Update BCM-like sliding threshold based on recent activity.

        The sliding threshold determines whether updates potentiate or depress.
        High recent activity raises threshold (harder to potentiate).
        Low recent activity lowers threshold (easier to potentiate).

        Args:
            recent_activations: Recent activation values

        Returns:
            Updated threshold
        """
        mean_activation = float(np.mean(recent_activations))

        # Update sliding threshold toward recent mean
        self._state.sliding_threshold = (
            (1 - self.sliding_threshold_rate) * self._state.sliding_threshold +
            self.sliding_threshold_rate * mean_activation
        )

        self._state.mean_activation = mean_activation

        return self._state.sliding_threshold

    def modulate_learning_rate(
        self,
        base_lr: float,
        current_norm: float
    ) -> float:
        """
        Modulate learning rate based on homeostatic state.

        Embeddings with high norms get lower learning rates
        to prevent further potentiation (negative feedback).

        Args:
            base_lr: Base learning rate
            current_norm: Current embedding norm

        Returns:
            Modulated learning rate
        """
        if current_norm < 1e-8:
            return base_lr

        # Higher norm = lower learning rate
        norm_ratio = current_norm / self.target_norm

        # Sigmoid-like modulation: lr decreases as norm increases
        modulation = 2.0 / (1.0 + np.exp(norm_ratio - 1.0))

        return base_lr * modulation

    def get_state(self) -> HomeostaticState:
        """Get current homeostatic state."""
        return self._state

    def get_stats(self) -> dict:
        """Get homeostatic plasticity statistics."""
        return {
            "state": self._state.to_dict(),
            "scaling_count": self._scaling_count,
            "decorrelation_count": self._decorrelation_count,
            "norm_history_size": len(self._norm_history),
            "recent_norm_mean": float(np.mean(list(self._norm_history))) if self._norm_history else 0.0,
            "recent_norm_std": float(np.std(list(self._norm_history))) if len(self._norm_history) > 1 else 0.0,
            "config": {
                "target_norm": self.target_norm,
                "norm_tolerance": self.norm_tolerance,
                "ema_alpha": self.ema_alpha,
                "decorrelation_strength": self.decorrelation_strength,
                "sliding_threshold_rate": self.sliding_threshold_rate,
            },
        }

    # ==================== Runtime Configuration Setters ====================

    def set_target_norm(self, norm: float) -> None:
        """
        Set target embedding norm.

        Args:
            norm: Target L2 norm (0.5 to 2.0)
        """
        self.target_norm = float(np.clip(norm, 0.5, 2.0))
        logger.info(f"Homeostatic target_norm set to {self.target_norm}")

    def set_ema_alpha(self, alpha: float) -> None:
        """
        Set EMA rate for statistics tracking.

        Args:
            alpha: EMA rate (0.001 to 0.1)
        """
        self.ema_alpha = float(np.clip(alpha, 0.001, 0.1))
        logger.info(f"Homeostatic ema_alpha set to {self.ema_alpha}")

    def set_decorrelation_strength(self, strength: float) -> None:
        """
        Set decorrelation strength.

        Args:
            strength: Decorrelation strength (0.0 to 0.1)
        """
        self.decorrelation_strength = float(np.clip(strength, 0.0, 0.1))
        logger.info(f"Homeostatic decorrelation_strength set to {self.decorrelation_strength}")

    def set_norm_tolerance(self, tolerance: float) -> None:
        """
        Set norm tolerance before scaling triggers.

        Args:
            tolerance: Tolerance (0.05 to 0.5)
        """
        self.norm_tolerance = float(np.clip(tolerance, 0.05, 0.5))
        logger.info(f"Homeostatic norm_tolerance set to {self.norm_tolerance}")

    def set_sliding_threshold_rate(self, rate: float) -> None:
        """
        Set BCM sliding threshold adaptation rate.

        Args:
            rate: Adaptation rate (0.0001 to 0.01)
        """
        self.sliding_threshold_rate = float(np.clip(rate, 0.0001, 0.01))
        logger.info(f"Homeostatic sliding_threshold_rate set to {self.sliding_threshold_rate}")

    def force_scaling(self) -> float:
        """
        Force immediate scaling of tracked embeddings.

        Returns:
            The scaling factor applied
        """
        scale = self.compute_scaling_factor()
        self._scaling_count += 1
        logger.info(f"Forced homeostatic scaling with factor {scale:.4f}")
        return scale

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "target_norm": self.target_norm,
            "norm_tolerance": self.norm_tolerance,
            "ema_alpha": self.ema_alpha,
            "decorrelation_strength": self.decorrelation_strength,
            "sliding_threshold_rate": self.sliding_threshold_rate,
        }

    def reset(self) -> None:
        """Reset homeostatic state."""
        self._norm_history.clear()
        self._state = HomeostaticState()
        self._scaling_count = 0
        self._decorrelation_count = 0


# Integration helper for reconsolidation
def apply_homeostatic_bounds(
    reconsolidation_engine,
    homeostatic: HomeostaticPlasticity,
    embeddings: np.ndarray,
    force_scaling: bool = False
) -> np.ndarray:
    """
    Apply homeostatic regulation after reconsolidation updates.

    Args:
        reconsolidation_engine: ReconsolidationEngine instance (for stats)
        homeostatic: HomeostaticPlasticity instance
        embeddings: Updated embeddings
        force_scaling: Force scaling even if not needed

    Returns:
        Homeostatic-regulated embeddings
    """
    # Update statistics
    homeostatic.update_statistics(embeddings)

    # Apply scaling if needed
    if force_scaling or homeostatic.needs_scaling():
        embeddings = homeostatic.apply_scaling(embeddings, force=force_scaling)

    return embeddings


__all__ = [
    "HomeostaticPlasticity",
    "HomeostaticState",
    "apply_homeostatic_bounds",
]
