"""
Adaptive Layer Thresholds (W1-02).

Implements Hinton's Forward-Forward Algorithm extension with homeostatic
threshold adaptation per layer. Each layer maintains its own threshold
that adapts to achieve a target firing rate.

Evidence Base:
- Hinton (2022) "The Forward-Forward Algorithm: Some Preliminary Investigations"
- Turrigiano (2011) "Too Many Cooks? Intrinsic and Synaptic Homeostatic Mechanisms"

Key Insight:
    The goodness threshold θ determines positive/negative classification.
    A fixed θ doesn't account for layer-specific activation distributions.
    Adaptive θ maintains homeostatic balance: ~15% neurons "fire" (exceed θ).

Biological Interpretation:
    - Threshold ~ homeostatic set point for neural activity
    - Target firing rate ~ metabolic efficiency constraint
    - Adaptation ~ intrinsic plasticity / synaptic scaling

Integration with T4DM:
    - Each ForwardForward layer gets its own AdaptiveThreshold
    - AdaptiveThresholdManager coordinates multi-layer networks
    - Thresholds adapt during both positive and negative phases
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for homeostatic threshold adaptation.

    Attributes:
        target_firing_rate: Desired fraction of neurons exceeding threshold (default 15%).
        adaptation_rate: How quickly threshold adapts (0.01 = 1% per update).
        min_threshold: Floor for threshold (prevents runaway adaptation).
        max_threshold: Ceiling for threshold.
        window_size: Number of steps to average firing rate over.
        use_multiplicative_update: If True, use multiplicative update (more stable).
    """

    target_firing_rate: float = 0.15
    adaptation_rate: float = 0.01
    min_threshold: float = 0.1
    max_threshold: float = 10.0
    window_size: int = 100
    use_multiplicative_update: bool = True


class AdaptiveThreshold:
    """Layer-specific threshold that adapts to maintain target firing rate.

    The threshold θ determines the boundary between "positive" (goodness > θ)
    and "negative" (goodness < θ) classifications in the Forward-Forward algorithm.

    Adaptation Rule:
        If firing_rate > target: increase θ (make it harder to fire)
        If firing_rate < target: decrease θ (make it easier to fire)

    This maintains homeostatic balance where approximately `target_firing_rate`
    fraction of neurons exceed the threshold.

    Example:
        >>> threshold = AdaptiveThreshold()
        >>> for batch in dataloader:
        ...     goodness = layer.compute_goodness(batch)
        ...     theta = threshold.update(goodness)
        ...     is_positive = goodness > theta
    """

    def __init__(self, config: Optional[AdaptiveThresholdConfig] = None):
        """Initialize adaptive threshold.

        Args:
            config: Configuration. Uses defaults if not provided.
        """
        self.config = config or AdaptiveThresholdConfig()
        self.theta = 1.0  # Initial threshold
        self.firing_history: deque = deque(maxlen=self.config.window_size)
        self._update_count = 0

    def update(self, goodness: torch.Tensor) -> float:
        """Update threshold based on observed firing rate.

        If firing_rate > target: increase θ (make it harder to fire)
        If firing_rate < target: decrease θ (make it easier to fire)

        Args:
            goodness: Goodness values tensor [batch_size] or [batch_size, ...].

        Returns:
            Updated threshold value.
        """
        # Handle edge cases
        if goodness.numel() == 0:
            return self.theta

        # Filter NaN values
        goodness = goodness[~torch.isnan(goodness)]
        if goodness.numel() == 0:
            return self.theta

        # Compute current firing rate
        fired = (goodness > self.theta).float().mean().item()
        self.firing_history.append(fired)
        self._update_count += 1

        # Only adapt after we have enough history
        if len(self.firing_history) >= min(self.config.window_size, 10):
            avg_rate = np.mean(self.firing_history)

            if self.config.use_multiplicative_update:
                # Multiplicative update for stability
                # When firing_rate > target: need to INCREASE theta (harder to fire)
                # When firing_rate < target: need to DECREASE theta (easier to fire)
                # ratio > 1 when firing_rate > target → increase theta
                # ratio < 1 when firing_rate < target → decrease theta
                ratio = (avg_rate + 1e-6) / (self.config.target_firing_rate + 1e-6)

                # Smooth update: blend current theta with adjusted theta
                self.theta *= (
                    (1 - self.config.adaptation_rate)
                    + self.config.adaptation_rate * ratio
                )
            else:
                # Additive update (simpler but less stable)
                # Positive error (firing > target) → increase theta
                error = avg_rate - self.config.target_firing_rate
                self.theta += self.config.adaptation_rate * error * self.theta

            # Clamp to valid range
            self.theta = np.clip(
                self.theta, self.config.min_threshold, self.config.max_threshold
            )

        return self.theta

    def reset(self) -> None:
        """Reset threshold to initial state."""
        self.theta = 1.0
        self.firing_history.clear()
        self._update_count = 0

    def get_stats(self) -> dict:
        """Get current statistics.

        Returns:
            Dictionary with theta, firing rate stats, etc.
        """
        return {
            "theta": self.theta,
            "avg_firing_rate": (
                float(np.mean(self.firing_history)) if self.firing_history else 0.0
            ),
            "target_firing_rate": self.config.target_firing_rate,
            "updates": self._update_count,
            "history_size": len(self.firing_history),
        }


class AdaptiveThresholdManager:
    """Manager for per-layer adaptive thresholds in multi-layer networks.

    Each layer in a Forward-Forward network has its own threshold that
    adapts independently. This manager coordinates them.

    Example:
        >>> manager = AdaptiveThresholdManager(num_layers=6)
        >>> for layer_idx, layer in enumerate(network.layers):
        ...     goodness = layer.compute_goodness(x)
        ...     theta = manager.update_layer(layer_idx, goodness)
        ...     x = layer.forward(x, theta=theta)
    """

    def __init__(
        self,
        num_layers: int,
        config: Optional[AdaptiveThresholdConfig] = None,
    ):
        """Initialize manager with per-layer thresholds.

        Args:
            num_layers: Number of layers in network.
            config: Configuration shared by all layers (or None for defaults).
        """
        self.num_layers = num_layers
        self.config = config or AdaptiveThresholdConfig()
        self.thresholds = [
            AdaptiveThreshold(self.config) for _ in range(num_layers)
        ]

    def update_layer(self, layer_idx: int, goodness: torch.Tensor) -> float:
        """Update threshold for specific layer.

        Args:
            layer_idx: Index of layer to update.
            goodness: Goodness values from that layer.

        Returns:
            Updated threshold for that layer.
        """
        if 0 <= layer_idx < self.num_layers:
            return self.thresholds[layer_idx].update(goodness)
        else:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")

    def get_theta(self, layer_idx: int) -> float:
        """Get current threshold for layer.

        Args:
            layer_idx: Index of layer.

        Returns:
            Current threshold value.
        """
        if 0 <= layer_idx < self.num_layers:
            return self.thresholds[layer_idx].theta
        else:
            raise IndexError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")

    def get_all_thetas(self) -> list[float]:
        """Get all layer thresholds.

        Returns:
            List of threshold values.
        """
        return [t.theta for t in self.thresholds]

    def get_all_stats(self) -> list[dict]:
        """Get statistics for all layers.

        Returns:
            List of stat dictionaries.
        """
        return [t.get_stats() for t in self.thresholds]

    def reset_all(self) -> None:
        """Reset all thresholds to initial state."""
        for threshold in self.thresholds:
            threshold.reset()
