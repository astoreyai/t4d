"""
P5.2: Bridge between PredictiveCoding and DopamineSystem.

Converts hierarchical prediction errors from Rao & Ballard / Friston predictive
coding into dopamine-like RPE signals for learning modulation.

Biological Basis:
- Cortical prediction errors modulate VTA dopamine neurons (Schultz 1997)
- Precision-weighted errors indicate reliability of surprise signal (Friston 2005)
- High PE + high precision = strong dopamine signal = enhanced learning

Integration Flow:
    Sensory Input
         ↓
    [PredictiveCodingHierarchy]
         ↓ compute_dopamine_signal()
    [PredictiveCodingDopamineBridge]
         ↓ inject_external_rpe() / blend_with_internal_rpe()
    [DopamineSystem]
         ↓ modulates learning rate
    [ThreeFactorLearning]

References:
- Schultz et al. (1997): A neural substrate of prediction and reward
- Friston (2005): A theory of cortical responses
- Rao & Ballard (1999): Predictive coding in the visual cortex
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


class PredictiveHierarchy(Protocol):
    """Protocol for predictive coding hierarchy."""

    def process(self, sensory_input: np.ndarray) -> Any:
        """Process sensory input through hierarchy."""
        ...

    def compute_dopamine_signal(
        self,
        baseline: float = 0.3,
        gain: float = 0.5
    ) -> float:
        """Compute RPE-like signal from prediction errors."""
        ...

    def get_precision_weighted_error(self) -> np.ndarray:
        """Get precision-weighted error per level."""
        ...


class DopamineSystemProtocol(Protocol):
    """Protocol for dopamine system."""

    def compute_rpe(
        self,
        memory_id: UUID,
        outcome: float
    ) -> Any:
        """Compute reward prediction error."""
        ...

    def update_expectations(
        self,
        memory_id: UUID,
        outcome: float
    ) -> None:
        """Update internal expectations."""
        ...


@dataclass
class BridgeConfig:
    """Configuration for PC-DA bridge.

    Attributes:
        pe_to_rpe_gain: Scaling from prediction error to RPE
        baseline_error: Expected average error (surprise threshold)
        blend_ratio: How much to blend external PE with internal RPE
        precision_floor: Minimum precision to avoid division issues
        update_da_expectations: Whether to update DA system expectations
    """
    pe_to_rpe_gain: float = 0.5
    baseline_error: float = 0.3
    blend_ratio: float = 0.5  # 0.5 = equal weight to PE and internal RPE
    precision_floor: float = 0.01
    update_da_expectations: bool = True


@dataclass
class BridgeState:
    """Current state of the bridge.

    Attributes:
        last_pe_signal: Last prediction error signal computed
        last_blended_rpe: Last blended RPE (PE + internal)
        n_signals_processed: Total signals processed
        mean_pe: Running mean of PE signals
        timestamp: When last updated
    """
    last_pe_signal: float = 0.0
    last_blended_rpe: float = 0.0
    n_signals_processed: int = 0
    mean_pe: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class PredictiveCodingDopamineBridge:
    """
    Bridge connecting PredictiveCodingHierarchy to DopamineSystem.

    Implements P5.2: hierarchical prediction errors → dopamine RPE.

    Example:
        ```python
        from t4dm.prediction import create_predictive_hierarchy
        from t4dm.learning import DopamineSystem
        from t4dm.bridges import PredictiveCodingDopamineBridge

        hierarchy = create_predictive_hierarchy()
        dopamine = DopamineSystem()
        bridge = PredictiveCodingDopamineBridge(hierarchy, dopamine)

        # Process sensory input
        state = bridge.process_with_dopamine(sensory_embedding, memory_id)
        logger.info(f"PE signal: {state.last_pe_signal:.3f}")
        logger.info(f"Blended RPE: {state.last_blended_rpe:.3f}")
        ```
    """

    def __init__(
        self,
        hierarchy: PredictiveHierarchy,
        dopamine: DopamineSystemProtocol | None = None,
        config: BridgeConfig | None = None
    ):
        """
        Initialize bridge.

        Args:
            hierarchy: PredictiveCodingHierarchy instance
            dopamine: Optional DopamineSystem instance
            config: Bridge configuration
        """
        self.hierarchy = hierarchy
        self.dopamine = dopamine
        self.config = config or BridgeConfig()
        self.state = BridgeState()

        logger.info(
            f"PredictiveCodingDopamineBridge initialized "
            f"(blend_ratio={self.config.blend_ratio})"
        )

    def compute_pe_signal(self) -> float:
        """
        Compute prediction error signal from hierarchy.

        Returns:
            RPE-like signal in [-1, 1]
        """
        pe_signal = self.hierarchy.compute_dopamine_signal(
            baseline=self.config.baseline_error,
            gain=self.config.pe_to_rpe_gain
        )

        # Update state
        self.state.last_pe_signal = pe_signal
        self.state.n_signals_processed += 1

        # Update running mean
        alpha = 0.1  # EMA smoothing
        self.state.mean_pe = (
            alpha * pe_signal + (1 - alpha) * self.state.mean_pe
        )
        self.state.timestamp = datetime.now()

        return pe_signal

    def blend_with_internal_rpe(
        self,
        memory_id: UUID,
        outcome: float
    ) -> float:
        """
        Blend external PE signal with internal dopamine RPE.

        Args:
            memory_id: Memory being processed
            outcome: Observed outcome (0-1)

        Returns:
            Blended RPE signal
        """
        # Get PE from hierarchy
        pe_signal = self.compute_pe_signal()

        # Get internal RPE from dopamine system
        if self.dopamine is not None:
            internal_rpe_obj = self.dopamine.compute_rpe(memory_id, outcome)
            # Extract delta from RPE object if it has that attribute
            internal_rpe = getattr(internal_rpe_obj, 'delta', outcome)
        else:
            internal_rpe = outcome

        # Blend: weighted average of PE and internal RPE
        r = self.config.blend_ratio
        blended = r * pe_signal + (1 - r) * internal_rpe

        # Update state
        self.state.last_blended_rpe = blended

        # Optionally update DA system expectations
        if self.dopamine is not None and self.config.update_da_expectations:
            # Inject PE-modified outcome
            modified_outcome = outcome + pe_signal * 0.2  # Small PE contribution
            self.dopamine.update_expectations(
                memory_id,
                np.clip(modified_outcome, 0.0, 1.0)
            )

        logger.debug(
            f"P5.2: Blended RPE - PE={pe_signal:.3f}, "
            f"internal={internal_rpe:.3f}, blended={blended:.3f}"
        )

        return float(blended)

    def process_with_dopamine(
        self,
        sensory_input: np.ndarray,
        memory_id: UUID,
        outcome: float | None = None
    ) -> BridgeState:
        """
        Full processing: hierarchy → PE → blended RPE.

        Args:
            sensory_input: Input embedding
            memory_id: Memory being processed
            outcome: Optional observed outcome

        Returns:
            Updated bridge state
        """
        # Process through hierarchy
        hierarchy_state = self.hierarchy.process(sensory_input)

        # Compute PE signal
        pe_signal = self.compute_pe_signal()

        # If outcome provided, blend with internal RPE
        if outcome is not None:
            self.blend_with_internal_rpe(memory_id, outcome)

        return self.state

    def get_learning_modulation(self) -> float:
        """
        Get learning rate modulation factor based on current PE.

        Returns:
            Multiplier for learning rate (typically 0.5 - 2.0)
        """
        # Higher PE = higher surprise = stronger learning
        pe = self.state.last_pe_signal

        # Map PE to learning modulation
        # PE = 0: modulation = 1.0 (baseline)
        # PE = +1: modulation = 2.0 (double learning)
        # PE = -1: modulation = 0.5 (half learning)
        modulation = 1.0 + pe * 0.5

        return float(np.clip(modulation, 0.5, 2.0))

    def get_statistics(self) -> dict[str, Any]:
        """Get bridge statistics."""
        return {
            "last_pe_signal": self.state.last_pe_signal,
            "last_blended_rpe": self.state.last_blended_rpe,
            "n_signals_processed": self.state.n_signals_processed,
            "mean_pe": self.state.mean_pe,
            "learning_modulation": self.get_learning_modulation(),
            "config": {
                "pe_to_rpe_gain": self.config.pe_to_rpe_gain,
                "blend_ratio": self.config.blend_ratio,
            }
        }


def create_pc_dopamine_bridge(
    hierarchy: PredictiveHierarchy,
    dopamine: DopamineSystemProtocol | None = None,
    blend_ratio: float = 0.5
) -> PredictiveCodingDopamineBridge:
    """
    Factory function for PC-DA bridge.

    Args:
        hierarchy: PredictiveCodingHierarchy instance
        dopamine: Optional DopamineSystem instance
        blend_ratio: How much to weight PE vs internal RPE

    Returns:
        Configured PredictiveCodingDopamineBridge
    """
    config = BridgeConfig(blend_ratio=blend_ratio)
    return PredictiveCodingDopamineBridge(
        hierarchy=hierarchy,
        dopamine=dopamine,
        config=config
    )
