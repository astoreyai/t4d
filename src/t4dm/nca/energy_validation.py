"""
Energy Landscape Validation (W1-03).

Validates that Hopfield energy dynamics match biological constraints:
- Wake: Energy should decrease (minimize prediction error)
- Sleep: Energy should stabilize (settle in attractor)
- Attractor depth correlates with memory strength

Evidence Base:
- Hopfield & Tank (1986) "Computing with Neural Circuits"
- Bengio (2019) "On the Measure of Intelligence"
- Ramsauer et al. (2020) "Hopfield Networks is All You Need"

Key Insight:
    Energy landscapes provide a unifying framework for memory:
    - Stored patterns = local minima (attractors)
    - Retrieval = gradient descent to nearest attractor
    - Memory strength = depth of attractor basin

Biological Interpretation:
    - Wake: Active processing, energy decreases as predictions improve
    - Sleep: Consolidation, energy stabilizes in attractor basins
    - Deeper attractors = stronger, more consolidated memories
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EnergyValidationConfig:
    """Configuration for energy landscape validation.

    Attributes:
        wake_slope_threshold: Maximum slope for valid wake phase (should be < 0).
        sleep_variance_threshold: Maximum variance for valid sleep phase.
        convergence_threshold: Std threshold to consider converged.
        convergence_window: Window size for convergence check.
        default_steps: Default number of dynamics steps.
        learning_rate: Step size for gradient descent dynamics.
        noise_scale: Scale of noise for stochastic dynamics (0 = deterministic).
    """

    wake_slope_threshold: float = 0.0  # Slope must be negative
    sleep_variance_threshold: float = 0.01  # Low variance = stable
    convergence_threshold: float = 0.001  # Std below this = converged
    convergence_window: int = 10  # Window for convergence check
    default_steps: int = 500  # Default dynamics steps
    learning_rate: float = 0.1  # Gradient descent step size
    noise_scale: float = 0.0  # Optional stochastic noise


@dataclass
class EnergyValidationResult:
    """Result of energy landscape validation.

    Attributes:
        wake_slope: dE/dt during wake phase (should be < 0).
        sleep_variance: Var(E) during sleep phase (should be low).
        convergence_time: Steps to reach attractor.
        attractor_depth: Energy at attractor basin (lower = deeper).
        is_valid: Whether validation passed.
        phase: Which phase was validated ("wake" or "sleep").
    """

    wake_slope: float
    sleep_variance: float
    convergence_time: int
    attractor_depth: float
    is_valid: bool
    phase: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "wake_slope": self.wake_slope,
            "sleep_variance": self.sleep_variance,
            "convergence_time": self.convergence_time,
            "attractor_depth": self.attractor_depth,
            "is_valid": self.is_valid,
            "phase": self.phase,
        }


class EnergyLandscapeValidator:
    """Validate Hopfield energy dynamics match biological constraints.

    This class verifies that energy-based memory systems exhibit:
    1. Decreasing energy during wake (active processing)
    2. Stable energy during sleep (attractor consolidation)
    3. Attractor depth correlating with memory strength

    The validator runs dynamics simulations and checks convergence properties.

    Example:
        >>> validator = EnergyLandscapeValidator(hopfield_energy)
        >>> result = validator.check_convergence(state, phase="wake")
        >>> assert result.is_valid, "Energy should decrease during wake"
    """

    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        config: Optional[EnergyValidationConfig] = None,
    ):
        """Initialize validator.

        Args:
            energy_fn: Function that computes energy from state tensor.
            config: Validation configuration. Uses defaults if not provided.
        """
        self.energy_fn = energy_fn
        self.config = config or EnergyValidationConfig()
        self.E_history: list[float] = []

    def check_convergence(
        self,
        state: torch.Tensor,
        phase: str = "wake",
        steps: Optional[int] = None,
    ) -> EnergyValidationResult:
        """Verify energy dynamics match expected behavior.

        Wake: E should decrease (minimize prediction error)
        Sleep: E should stabilize (settled in attractor)

        Args:
            state: Initial state tensor.
            phase: "wake" or "sleep".
            steps: Number of dynamics steps (uses default if None).

        Returns:
            EnergyValidationResult with validation outcome.
        """
        steps = steps or self.config.default_steps
        self.E_history = []

        # Clone state to avoid modifying input
        current_state = state.clone().detach().requires_grad_(True)

        for step in range(steps):
            # Compute energy
            E = self.energy_fn(current_state)

            # Handle NaN
            if torch.isnan(E):
                logger.warning(f"NaN energy at step {step}")
                # Fill remaining history with last valid value or 0
                last_valid = self.E_history[-1] if self.E_history else 0.0
                self.E_history.extend([last_valid] * (steps - step))
                break

            self.E_history.append(E.item())

            # Gradient descent dynamics
            current_state = self._dynamics_step(current_state, E)

        E_array = np.array(self.E_history)

        if phase == "wake":
            return self._validate_wake(E_array)
        else:
            return self._validate_sleep(E_array)

    def _dynamics_step(
        self, state: torch.Tensor, energy: torch.Tensor
    ) -> torch.Tensor:
        """Perform one step of gradient descent dynamics.

        dstate/dt = -grad_E(state) + noise

        Args:
            state: Current state (requires grad).
            energy: Current energy value.

        Returns:
            Updated state after one dynamics step.
        """
        # Compute gradient
        if state.grad is not None:
            state.grad.zero_()

        energy.backward(retain_graph=True)

        if state.grad is None:
            # No gradient - return unchanged
            return state.detach().requires_grad_(True)

        grad = state.grad.clone()

        # Gradient descent: state -= lr * grad_E
        with torch.no_grad():
            new_state = state - self.config.learning_rate * grad

            # Optional noise for stochastic dynamics
            if self.config.noise_scale > 0:
                noise = torch.randn_like(new_state) * self.config.noise_scale
                new_state = new_state + noise

        return new_state.detach().requires_grad_(True)

    def _validate_wake(self, E_array: np.ndarray) -> EnergyValidationResult:
        """Validate wake phase: energy should decrease.

        Args:
            E_array: Energy history array.

        Returns:
            Validation result for wake phase.
        """
        # Compute slope via linear regression
        x = np.arange(len(E_array))
        slope = np.polyfit(x, E_array, 1)[0]

        # Variance of full history
        variance = float(np.var(E_array))

        # Find convergence time
        convergence_time = self._find_convergence(E_array)

        # Attractor depth = final energy
        attractor_depth = float(E_array[-1])

        # Valid if slope is negative
        is_valid = slope < self.config.wake_slope_threshold

        return EnergyValidationResult(
            wake_slope=float(slope),
            sleep_variance=variance,
            convergence_time=convergence_time,
            attractor_depth=attractor_depth,
            is_valid=is_valid,
            phase="wake",
        )

    def _validate_sleep(self, E_array: np.ndarray) -> EnergyValidationResult:
        """Validate sleep phase: energy should stabilize.

        Args:
            E_array: Energy history array.

        Returns:
            Validation result for sleep phase.
        """
        # Use second half of history for stability check
        midpoint = len(E_array) // 2
        late_E = E_array[midpoint:]

        # Variance of late phase
        variance = float(np.var(late_E))

        # Slope (should be near zero for stable)
        x = np.arange(len(E_array))
        slope = np.polyfit(x, E_array, 1)[0]

        # Find convergence time
        convergence_time = self._find_convergence(E_array)

        # Attractor depth = mean of late energy
        attractor_depth = float(np.mean(late_E))

        # Valid if variance is small
        is_valid = variance < self.config.sleep_variance_threshold

        return EnergyValidationResult(
            wake_slope=float(slope),
            sleep_variance=variance,
            convergence_time=convergence_time,
            attractor_depth=attractor_depth,
            is_valid=is_valid,
            phase="sleep",
        )

    def _find_convergence(self, E_array: np.ndarray) -> int:
        """Find step where energy stabilizes.

        Args:
            E_array: Energy history array.

        Returns:
            Step index where convergence was detected, or len(E_array) if not.
        """
        window = self.config.convergence_window
        threshold = self.config.convergence_threshold

        for i in range(len(E_array) - window):
            window_E = E_array[i : i + window]
            if np.std(window_E) < threshold:
                return i

        # Did not converge
        return len(E_array)

    def reset(self) -> None:
        """Reset energy history."""
        self.E_history = []

    def get_stats(self) -> dict:
        """Get current statistics.

        Returns:
            Dictionary with energy history stats.
        """
        if not self.E_history:
            return {"mean_energy": 0.0, "final_energy": 0.0, "steps": 0}

        return {
            "mean_energy": float(np.mean(self.E_history)),
            "final_energy": float(self.E_history[-1]),
            "min_energy": float(np.min(self.E_history)),
            "max_energy": float(np.max(self.E_history)),
            "steps": len(self.E_history),
        }


class BatchEnergyValidator:
    """Validate energy dynamics across multiple states/patterns.

    Useful for validating that a memory system correctly stores
    and retrieves multiple patterns with appropriate attractor depths.
    """

    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        config: Optional[EnergyValidationConfig] = None,
    ):
        """Initialize batch validator.

        Args:
            energy_fn: Function that computes energy from state tensor.
            config: Validation configuration.
        """
        self.validator = EnergyLandscapeValidator(energy_fn, config)

    def validate_patterns(
        self,
        patterns: list[torch.Tensor],
        phase: str = "wake",
        steps: int = 200,
    ) -> dict:
        """Validate multiple patterns and compute correlation metrics.

        Args:
            patterns: List of initial state patterns.
            phase: "wake" or "sleep".
            steps: Steps per pattern.

        Returns:
            Dictionary with per-pattern results and aggregate metrics.
        """
        results = []
        attractor_depths = []

        for i, pattern in enumerate(patterns):
            result = self.validator.check_convergence(pattern, phase, steps)
            results.append(result)
            attractor_depths.append(result.attractor_depth)

        # Compute aggregate metrics
        valid_count = sum(1 for r in results if r.is_valid)
        avg_slope = np.mean([r.wake_slope for r in results])
        avg_variance = np.mean([r.sleep_variance for r in results])
        avg_convergence = np.mean([r.convergence_time for r in results])

        return {
            "results": results,
            "valid_count": valid_count,
            "total_count": len(patterns),
            "valid_fraction": valid_count / len(patterns) if patterns else 0.0,
            "avg_wake_slope": float(avg_slope),
            "avg_sleep_variance": float(avg_variance),
            "avg_convergence_time": float(avg_convergence),
            "attractor_depths": attractor_depths,
        }
