"""
Integration Metrics - IIT-inspired integration scoring (W3-04).

Compute Integrated Information Theory (IIT) inspired metrics
for measuring computational integration between subsystems.

IMPORTANT DISCLAIMER:
    These metrics measure *computational integration* between neural network
    subsystems, NOT consciousness or awareness. The Φ (phi) metric indicates
    how much mutual information exists between spiking outputs and memory
    state - a useful engineering diagnostic, not a consciousness claim.

Evidence Base: Tononi (2004) "An information integration theory of consciousness"
    Note: We use IIT as inspiration for integration metrics, not as a
    consciousness detection system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class IntegrationMetrics:
    """Computational integration metrics (IIT-inspired).

    Measures how integrated the processing is between subsystems.
    Higher integration suggests tighter coupling between spiking
    and memory components.

    Attributes:
        phi: Integration score (Φ). Higher = more integrated processing.
        surprise: |E_current - E_previous|. Energy change between steps.
        integration: Mutual info approximation between subsystems.
        differentiation: Entropy of activation patterns.
        integration_threshold: Threshold for "highly integrated" processing.
    """

    phi: float
    surprise: float
    integration: float
    differentiation: float
    integration_threshold: float = 0.5

    @property
    def conscious_threshold(self) -> float:
        """Backward compatibility alias for integration_threshold."""
        return self.integration_threshold

    @property
    def is_conscious(self) -> bool:
        """Backward compatibility alias for is_highly_integrated."""
        return self.is_highly_integrated

    @property
    def is_highly_integrated(self) -> bool:
        """Whether phi exceeds integration threshold.

        Indicates tight coupling between subsystems, useful for
        diagnosing whether spiking outputs are influencing memory.
        """
        return self.phi > self.integration_threshold


# Backward compatibility aliases
ConsciousnessMetrics = IntegrationMetrics


class IITMetricsComputer:
    """Compute IIT-inspired integration metrics.

    Φ (phi) measures how much information is shared between subsystems.
    We use approximations since exact IIT computation is intractable.

    IMPORTANT: This is an engineering diagnostic tool, NOT a consciousness
    detector. High phi indicates computational integration, which is useful
    for understanding spiking-memory coupling.

    Components:
    - Integration: Mutual information between subsystems
    - Differentiation: Entropy of activation patterns
    - Surprise: Energy change from previous state
    - Phi: Integration * Differentiation (simplified)

    Example:
        >>> def energy_fn(x): return torch.sum(x ** 2).item()
        >>> computer = IITMetricsComputer(energy_fn)
        >>> metrics = computer.compute(spiking_output, memory_state)
        >>> if metrics.is_highly_integrated:
        ...     print("Subsystems are tightly coupled")
    """

    def __init__(self, energy_fn: Callable[[torch.Tensor], float]):
        """Initialize IIT metrics computer.

        Args:
            energy_fn: Function to compute energy from activation tensor.
        """
        self.energy_fn = energy_fn
        self.previous_energy: Optional[float] = None

    def compute(
        self,
        spiking_output: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> IntegrationMetrics:
        """Compute integration metrics from current state.

        Φ = Σ I(A; B) for all bipartitions - not exactly computable,
        so we use approximations based on correlation and entropy.

        Args:
            spiking_output: Output from spiking network [batch, features].
            memory_state: Current memory state [batch, features].

        Returns:
            IntegrationMetrics with phi, surprise, integration, differentiation.
        """
        # Surprise = energy change
        current_energy = self.energy_fn(spiking_output)
        if self.previous_energy is not None:
            surprise = abs(current_energy - self.previous_energy)
        else:
            surprise = 0.0
        self.previous_energy = current_energy

        # Integration = mutual info between spiking and memory
        integration = self._mutual_information(spiking_output, memory_state)

        # Differentiation = entropy of activation pattern
        differentiation = self._entropy(spiking_output)

        # Φ approximation (simplified): product of integration and differentiation
        # True IIT uses minimum information partition, which is intractable
        phi = integration * differentiation

        return IntegrationMetrics(
            phi=phi,
            surprise=surprise,
            integration=integration,
            differentiation=differentiation,
        )

    def _mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Estimate mutual information via correlation.

        Uses Pearson correlation as a proxy for mutual information.
        True MI requires density estimation which is expensive.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Absolute correlation in [0, 1].
        """
        x_flat = x.flatten()
        y_flat = y.flatten()

        # Match lengths
        min_len = min(len(x_flat), len(y_flat))
        x_flat = x_flat[:min_len]
        y_flat = y_flat[:min_len]

        # Handle edge case of constant values
        if x_flat.std() < 1e-8 or y_flat.std() < 1e-8:
            return 0.0

        # Compute correlation
        corr = torch.corrcoef(torch.stack([x_flat, y_flat]))[0, 1]

        # Handle NaN
        if torch.isnan(corr):
            return 0.0

        return abs(corr.item())

    def _entropy(self, x: torch.Tensor) -> float:
        """Estimate entropy of activation pattern.

        Discretizes values into bins and computes Shannon entropy.

        Args:
            x: Activation tensor.

        Returns:
            Entropy in nats (natural log base).
        """
        x_flat = x.flatten()

        # Handle constant tensor
        if x_flat.std() < 1e-8:
            return 0.0

        # Discretize to bins
        hist = torch.histc(x_flat, bins=100)
        probs = hist / hist.sum()
        probs = probs[probs > 0]

        # Shannon entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))

        return entropy.item()

    def reset(self) -> None:
        """Reset previous energy state."""
        self.previous_energy = None
