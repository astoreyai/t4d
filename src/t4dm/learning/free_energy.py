"""
Free Energy Objective (W1-01).

Implements Friston's variational free energy minimization for memory systems.

Evidence Base:
- Friston (2010) "The free-energy principle: a unified brain theory?"
- Friston (2009) "The free-energy principle: a rough guide to the brain"

Key Equation:
    F = E_q[log p(o|s)] + β·KL[q(s|o) || p(s)]

Where:
- First term: reconstruction accuracy (negative log-likelihood)
- Second term: complexity cost (deviation from prior)

This provides a principled objective for memory encoding that balances:
1. Accuracy: Memories should faithfully reconstruct observations
2. Efficiency: Memory representations should stay close to priors

The free energy principle unifies perception, action, and learning under
a single objective: minimize surprise (= maximize model evidence).
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.distributions import Distribution, kl_divergence


@dataclass
class FreeEnergyConfig:
    """Configuration for variational free energy minimization.

    Attributes:
        beta: Inverse temperature (complexity weight). Higher = stronger prior.
        reconstruction_weight: Weight for reconstruction term.
        kl_weight: Weight for KL divergence term.
        use_elbo: If True, compute Evidence Lower Bound (ELBO) form.
        lr_scale_min: Minimum learning rate scale.
        lr_scale_max: Maximum learning rate scale.
        lr_sensitivity: How sensitive LR is to dF/dt.
    """

    beta: float = 1.0
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    use_elbo: bool = True
    lr_scale_min: float = 0.1
    lr_scale_max: float = 2.0
    lr_sensitivity: float = 1.0


@dataclass
class FreeEnergyResult:
    """Result of free energy computation.

    Attributes:
        free_energy: Total free energy F.
        reconstruction: Reconstruction error term.
        complexity: KL divergence complexity term.
        learning_rate_scale: Suggested learning rate multiplier based on dF/dt.
    """

    free_energy: torch.Tensor
    reconstruction: torch.Tensor
    complexity: torch.Tensor
    learning_rate_scale: float = 1.0


class FreeEnergyMinimizer:
    """Friston's variational free energy minimizer.

    F = reconstruction + β·KL[q||p]

    This class computes the variational free energy and provides
    learning rate scaling based on the rate of free energy change.

    Example:
        >>> minimizer = FreeEnergyMinimizer()
        >>> result = minimizer.compute_free_energy(prediction, observation, q, p)
        >>> loss = result.free_energy
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(self, config: Optional[FreeEnergyConfig] = None):
        """Initialize minimizer.

        Args:
            config: Configuration. Uses defaults if not provided.
        """
        self.config = config or FreeEnergyConfig()
        self._prev_F: Optional[float] = None
        self._F_history: list[float] = []

    def compute_free_energy(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
        q_posterior: Distribution,
        p_prior: Distribution,
    ) -> FreeEnergyResult:
        """Compute variational free energy.

        F = E_q[log p(o|s)] + β·KL[q(s|o) || p(s)]

        Where:
        - First term: reconstruction accuracy (negative log-likelihood)
        - Second term: complexity cost (deviation from prior)

        Args:
            prediction: Model prediction tensor.
            observation: Ground truth observation tensor.
            q_posterior: Approximate posterior q(s|o).
            p_prior: Prior distribution p(s).

        Returns:
            FreeEnergyResult with total free energy and components.
        """
        # Reconstruction error (MSE as Gaussian negative log-likelihood proxy)
        reconstruction = self._reconstruction_error(prediction, observation)

        # Complexity cost (KL divergence)
        complexity = kl_divergence(q_posterior, p_prior)

        # Total free energy
        F = (
            self.config.reconstruction_weight * reconstruction
            + self.config.beta * self.config.kl_weight * complexity
        )

        # Compute learning rate scale based on dF/dt
        lr_scale = self._compute_lr_scale(F.item())

        return FreeEnergyResult(
            free_energy=F,
            reconstruction=reconstruction,
            complexity=complexity,
            learning_rate_scale=lr_scale,
        )

    def _reconstruction_error(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction error.

        Uses MSE as a proxy for Gaussian negative log-likelihood.
        For Gaussian p(o|s) with unit variance: -log p = 0.5 * ||o - s||^2

        Args:
            prediction: Predicted values.
            observation: Target values.

        Returns:
            Reconstruction error (scalar tensor).
        """
        return torch.nn.functional.mse_loss(prediction, observation, reduction="mean")

    def _compute_lr_scale(self, F: float) -> float:
        """Compute learning rate scale based on free energy dynamics.

        When F is decreasing rapidly, we can afford larger learning rates.
        When F is stable or increasing, we should be more conservative.

        Args:
            F: Current free energy value.

        Returns:
            Learning rate multiplier in [lr_scale_min, lr_scale_max].
        """
        self._F_history.append(F)

        if len(self._F_history) < 2:
            return 1.0

        # Compute rate of change (dF/dt)
        dF = self._F_history[-1] - self._F_history[-2]

        # Negative dF (improving) -> higher LR, positive dF (worsening) -> lower LR
        # Use sigmoid to map dF to scale factor
        scale = 1.0 - self.config.lr_sensitivity * dF

        # Clamp to valid range
        scale = max(self.config.lr_scale_min, min(self.config.lr_scale_max, scale))

        return scale

    def reset_history(self) -> None:
        """Reset free energy history (e.g., at start of new episode)."""
        self._F_history = []
        self._prev_F = None


class FreeEnergyTracker:
    """Track free energy over time for monitoring and visualization.

    This class maintains a sliding window of free energy values and
    computes statistics useful for understanding learning dynamics.
    """

    def __init__(self, window_size: int = 100):
        """Initialize tracker.

        Args:
            window_size: Number of values to keep in sliding window.
        """
        self.window_size = window_size
        self._values: list[float] = []
        self._reconstructions: list[float] = []
        self._complexities: list[float] = []

    def update(self, result: FreeEnergyResult) -> None:
        """Record a free energy result.

        Args:
            result: FreeEnergyResult from minimizer.
        """
        self._values.append(result.free_energy.item())
        self._reconstructions.append(result.reconstruction.item())
        self._complexities.append(result.complexity.item())

        # Trim to window size
        if len(self._values) > self.window_size:
            self._values = self._values[-self.window_size :]
            self._reconstructions = self._reconstructions[-self.window_size :]
            self._complexities = self._complexities[-self.window_size :]

    @property
    def mean(self) -> float:
        """Mean free energy over window."""
        return sum(self._values) / len(self._values) if self._values else 0.0

    @property
    def variance(self) -> float:
        """Variance of free energy over window."""
        if len(self._values) < 2:
            return 0.0
        mean = self.mean
        return sum((v - mean) ** 2 for v in self._values) / (len(self._values) - 1)

    @property
    def slope(self) -> float:
        """Linear trend (slope) of free energy.

        Negative slope indicates F is decreasing (good).
        Positive slope indicates F is increasing (bad).
        """
        if len(self._values) < 2:
            return 0.0

        n = len(self._values)
        x_mean = (n - 1) / 2
        y_mean = self.mean

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(self._values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator > 0 else 0.0

    @property
    def is_converging(self) -> bool:
        """Check if free energy is converging (stabilizing)."""
        if len(self._values) < self.window_size:
            return False

        # Check if variance is decreasing and slope is near zero
        return abs(self.slope) < 0.01 and self.variance < 0.1

    def get_stats(self) -> dict:
        """Get summary statistics.

        Returns:
            Dictionary with mean, variance, slope, and convergence status.
        """
        return {
            "mean": self.mean,
            "variance": self.variance,
            "slope": self.slope,
            "is_converging": self.is_converging,
            "reconstruction_mean": (
                sum(self._reconstructions) / len(self._reconstructions)
                if self._reconstructions
                else 0.0
            ),
            "complexity_mean": (
                sum(self._complexities) / len(self._complexities)
                if self._complexities
                else 0.0
            ),
        }
