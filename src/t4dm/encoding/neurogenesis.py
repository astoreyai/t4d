"""
Activity-Dependent Neurogenesis for T4DM.

Phase 4A: Neural Growth

Implements biologically-inspired neuron birth/death based on activity patterns
and novelty detection. This addresses the "frozen embeddings" problem by allowing
the neural architecture to adapt to new information patterns over time.

Biological Foundation:
======================

Adult neurogenesis in the hippocampal dentate gyrus:
- ~700 new neurons per day in adult human DG (Kempermann 2015)
- Survival depends on integration into active circuits (Tashiro 2007)
- Novelty and learning enhance neurogenesis (Gould 1999)
- Immature neurons have enhanced plasticity (Schmidt-Hieber 2004)

Key Mechanisms:
1. Birth: New neurons added in response to high novelty
2. Maturation: Gradual integration over ~4-8 weeks (compressed to epochs)
3. Pruning: Inactive neurons removed to maintain capacity
4. Competition: Neurons compete for synaptic integration

Integration with T4DM:
===============================
- ForwardForwardLayer: Each layer can grow/shrink neurons
- Novelty detection: Based on goodness scores and prediction error
- Learning modulation: Immature neurons have higher learning rates
- Capacity bounds: Prevent unlimited growth

References:
- Kempermann, G. (2015). Activity Dependency and Aging in Adult Hippocampal Neurogenesis
- Tashiro, A. et al. (2007). Experience-specific functional modification of DG granule cells
- Schmidt-Hieber, C. et al. (2004). Enhanced synaptic plasticity in newly generated neurons
- Gould, E. et al. (1999). Learning enhances adult neurogenesis in hippocampal formation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from t4dm.nca.forward_forward import ForwardForwardLayer

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class NeuronState(Enum):
    """Maturation state of a neuron."""

    IMMATURE = auto()  # Recently born, high plasticity
    MATURING = auto()  # Integrating into network
    MATURE = auto()  # Fully integrated, stable
    MARKED_FOR_DEATH = auto()  # Inactive, will be pruned


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NeurogenesisConfig:
    """
    Configuration for activity-dependent neurogenesis.

    Biological parameters scaled for computational efficiency:
    - Real DG: ~700 neurons/day, ~1M total neurons
    - Scaled: ~0.1% birth rate per epoch, capacity-limited growth

    Attributes:
        birth_rate: Probability of neuron birth per high-novelty event
        novelty_threshold: Goodness difference threshold for "high novelty"
        survival_threshold: Minimum activity to avoid pruning
        maturation_epochs: Epochs required for full maturation
        max_neurons_per_layer: Maximum neuron capacity (prevents unlimited growth)
        immature_lr_boost: Learning rate multiplier for immature neurons
        pruning_interval: Epochs between pruning cycles
        min_neurons_per_layer: Minimum neurons to maintain
        enable_birth: Allow neuron birth (can disable for ablation)
        enable_pruning: Allow neuron pruning (can disable for ablation)
    """

    # Birth parameters (Kempermann 2015)
    birth_rate: float = 0.001  # ~0.1% per novelty event
    novelty_threshold: float = 1.5  # Goodness std above mean
    min_novelty_score: float = 0.1  # Minimum novelty to trigger birth

    # Survival parameters (Tashiro 2007)
    survival_threshold: float = 0.1  # Mean activity threshold
    activity_window: int = 100  # Samples to track for activity

    # Maturation parameters (Schmidt-Hieber 2004)
    maturation_epochs: int = 10  # Compressed from ~4-8 weeks
    immature_lr_boost: float = 2.0  # Enhanced plasticity for new neurons
    integration_rate: float = 0.1  # Rate of maturity increase per epoch

    # Capacity constraints
    max_neurons_per_layer: int = 2048  # Hard limit on layer size
    min_neurons_per_layer: int = 32  # Minimum to maintain functionality
    growth_rate_limit: float = 0.2  # Max fractional growth per epoch

    # Pruning parameters
    pruning_interval: int = 5  # Epochs between pruning cycles
    pruning_batch_size: int = 10  # Max neurons to prune per cycle

    # Feature flags
    enable_birth: bool = True
    enable_pruning: bool = True
    enable_maturation: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert 0 < self.birth_rate < 1, "birth_rate must be in (0, 1)"
        assert self.survival_threshold >= 0, "survival_threshold must be non-negative"
        assert self.maturation_epochs > 0, "maturation_epochs must be positive"
        assert (
            self.min_neurons_per_layer < self.max_neurons_per_layer
        ), "min < max neurons required"


@dataclass
class NeuronMetadata:
    """
    Metadata for a single neuron in the network.

    Tracks maturation state, activity history, and birth epoch.
    """

    neuron_idx: int  # Index in layer weight matrix
    birth_epoch: int  # When neuron was created
    maturity: float = 0.0  # [0, 1]: 0=immature, 1=fully mature
    state: NeuronState = NeuronState.IMMATURE
    activity_history: list[float] = field(default_factory=list)
    total_activations: int = 0
    mean_activity: float = 0.0
    last_active_epoch: int = 0

    _max_history: int = 100  # Bounded history (MEM-007 pattern)

    def record_activity(self, activation: float, epoch: int) -> None:
        """Record neuron activation."""
        self.activity_history.append(activation)
        if len(self.activity_history) > self._max_history:
            self.activity_history = self.activity_history[-self._max_history :]

        self.total_activations += 1
        self.mean_activity = float(np.mean(self.activity_history))
        self.last_active_epoch = epoch

    def update_maturity(self, delta: float) -> None:
        """Increment maturity level."""
        self.maturity = min(1.0, self.maturity + delta)

        # Update state based on maturity
        if self.maturity < 0.3:
            self.state = NeuronState.IMMATURE
        elif self.maturity < 0.8:
            self.state = NeuronState.MATURING
        else:
            self.state = NeuronState.MATURE

    def get_lr_multiplier(self, config: NeurogenesisConfig) -> float:
        """
        Get learning rate multiplier based on maturity.

        Immature neurons have enhanced plasticity (Schmidt-Hieber 2004).
        """
        if not config.enable_maturation:
            return 1.0

        # Linear interpolation: immature -> mature
        # immature: boost, mature: 1.0
        return config.immature_lr_boost * (1.0 - self.maturity) + 1.0 * self.maturity

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "neuron_idx": self.neuron_idx,
            "birth_epoch": self.birth_epoch,
            "maturity": self.maturity,
            "state": self.state.name,
            "total_activations": self.total_activations,
            "mean_activity": self.mean_activity,
            "last_active_epoch": self.last_active_epoch,
        }


@dataclass
class NeurogenesisState:
    """State tracking for neurogenesis manager."""

    total_births: int = 0
    total_deaths: int = 0
    current_epoch: int = 0
    last_pruning_epoch: int = 0
    novelty_history: list[float] = field(default_factory=list)
    birth_history: list[int] = field(default_factory=list)  # Births per epoch
    death_history: list[int] = field(default_factory=list)  # Deaths per epoch
    neuron_count_history: list[int] = field(default_factory=list)

    _max_history: int = 1000

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "current_epoch": self.current_epoch,
            "last_pruning_epoch": self.last_pruning_epoch,
            "mean_novelty": (
                float(np.mean(self.novelty_history[-100:]))
                if self.novelty_history
                else 0.0
            ),
            "recent_births": sum(self.birth_history[-10:]),
            "recent_deaths": sum(self.death_history[-10:]),
            "current_neuron_count": (
                self.neuron_count_history[-1] if self.neuron_count_history else 0
            ),
        }

    def _trim_history(self) -> None:
        """Trim history to prevent unbounded growth."""
        if len(self.novelty_history) > self._max_history:
            self.novelty_history = self.novelty_history[-self._max_history :]
        if len(self.birth_history) > self._max_history:
            self.birth_history = self.birth_history[-self._max_history :]
        if len(self.death_history) > self._max_history:
            self.death_history = self.death_history[-self._max_history :]
        if len(self.neuron_count_history) > self._max_history:
            self.neuron_count_history = self.neuron_count_history[-self._max_history :]


# =============================================================================
# Neurogenesis Manager
# =============================================================================


class NeurogenesisManager:
    """
    Manage activity-dependent neuron birth/death in FF layers.

    This implements hippocampal adult neurogenesis (Kempermann 2015):
    1. New neurons born in response to novelty
    2. Immature neurons have enhanced plasticity
    3. Inactive neurons pruned to maintain capacity
    4. Gradual maturation process over epochs

    Example:
        ```python
        from t4dm.encoding.neurogenesis import NeurogenesisManager, NeurogenesisConfig
        from t4dm.nca.forward_forward import ForwardForwardLayer

        config = NeurogenesisConfig(birth_rate=0.002, maturation_epochs=10)
        manager = NeurogenesisManager(config)

        # During training loop
        for epoch in range(num_epochs):
            for sample in dataset:
                # Forward pass
                output = layer.forward(sample)

                # Check for novelty, potentially add neuron
                manager.maybe_add_neuron(
                    novelty_score=compute_novelty(output),
                    layer=layer,
                    epoch=epoch
                )

                # Record activity
                manager.update_activity(layer, output, epoch)

            # Periodic pruning
            manager.prune_inactive(layer, epoch)

            # Mature existing neurons
            manager.update_maturation(epoch)
        ```
    """

    def __init__(
        self, config: NeurogenesisConfig | None = None, random_seed: int | None = None
    ):
        """
        Initialize neurogenesis manager.

        Args:
            config: Neurogenesis configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config or NeurogenesisConfig()
        self._rng = np.random.default_rng(random_seed)
        self.state = NeurogenesisState()

        # Track neurons across layers
        self._neuron_metadata: dict[int, dict[int, NeuronMetadata]] = {}
        # layer_idx -> {neuron_idx -> metadata}

        logger.info(
            f"Phase 4A: NeurogenesisManager initialized "
            f"(birth_rate={self.config.birth_rate:.4f}, "
            f"maturation_epochs={self.config.maturation_epochs})"
        )

    def maybe_add_neuron(
        self, novelty_score: float, layer: ForwardForwardLayer, epoch: int
    ) -> bool:
        """
        Potentially add a neuron if high novelty detected.

        Biological rationale: Novel experiences enhance neurogenesis in DG (Gould 1999).

        Args:
            novelty_score: Measure of pattern novelty (e.g., prediction error)
            layer: Layer to potentially add neuron to
            epoch: Current training epoch

        Returns:
            True if neuron was added
        """
        if not self.config.enable_birth:
            return False

        # Record novelty
        self.state.novelty_history.append(novelty_score)
        self.state._trim_history()

        # Check capacity constraints
        current_neurons = layer.config.hidden_dim
        if current_neurons >= self.config.max_neurons_per_layer:
            return False

        # Check novelty threshold
        if novelty_score < self.config.min_novelty_score:
            return False

        # Adaptive threshold based on recent novelty
        if len(self.state.novelty_history) >= 10:
            mean_novelty = float(np.mean(self.state.novelty_history[-100:]))
            std_novelty = float(np.std(self.state.novelty_history[-100:]))
            threshold = mean_novelty + self.config.novelty_threshold * std_novelty
        else:
            threshold = self.config.min_novelty_score

        if novelty_score < threshold:
            return False

        # Stochastic birth with configured rate
        if self._rng.random() > self.config.birth_rate:
            return False

        # Check growth rate limit
        epoch_births = self.state.birth_history[-1] if self.state.birth_history else 0
        max_births_per_epoch = int(
            current_neurons * self.config.growth_rate_limit
        )
        if epoch_births >= max_births_per_epoch:
            return False

        # Add neuron to layer
        new_idx = layer.add_neuron(self._initialize_immature_weights(layer))

        # Track metadata
        if layer.layer_idx not in self._neuron_metadata:
            self._neuron_metadata[layer.layer_idx] = {}

        self._neuron_metadata[layer.layer_idx][new_idx] = NeuronMetadata(
            neuron_idx=new_idx,
            birth_epoch=epoch,
            maturity=0.0,
            state=NeuronState.IMMATURE,
        )

        # Update statistics
        self.state.total_births += 1
        if self.state.birth_history and self.state.current_epoch == epoch:
            self.state.birth_history[-1] += 1
        else:
            self.state.birth_history.append(1)
        self.state._trim_history()

        logger.debug(
            f"Phase 4A: Neuron born in layer {layer.layer_idx} "
            f"(novelty={novelty_score:.3f}, epoch={epoch}, "
            f"new_count={layer.config.hidden_dim})"
        )

        return True

    def _initialize_immature_weights(
        self, layer: ForwardForwardLayer
    ) -> np.ndarray:
        """
        Initialize weights for a new immature neuron.

        Immature neurons start with small random weights.
        """
        input_dim = layer.config.input_dim
        scale = np.sqrt(2.0 / input_dim) * 0.5  # Smaller init for immature
        weights = self._rng.normal(0, scale, size=input_dim).astype(np.float32)
        return weights

    def prune_inactive(
        self, layer: ForwardForwardLayer, epoch: int, force: bool = False
    ) -> int:
        """
        Remove neurons with low cumulative activity.

        Biological rationale: Newly born neurons that fail to integrate into
        active circuits undergo apoptosis (Tashiro 2007).

        Args:
            layer: Layer to prune
            epoch: Current training epoch
            force: Force pruning even if interval not reached

        Returns:
            Number of neurons pruned
        """
        if not self.config.enable_pruning:
            return 0

        # Check pruning interval
        if (
            not force
            and epoch - self.state.last_pruning_epoch < self.config.pruning_interval
        ):
            return 0

        # Don't prune below minimum
        current_neurons = layer.config.hidden_dim
        if current_neurons <= self.config.min_neurons_per_layer:
            return 0

        # Get activity statistics for all neurons
        if layer.layer_idx not in self._neuron_metadata:
            return 0

        metadata = self._neuron_metadata[layer.layer_idx]
        if not metadata:
            return 0

        # Identify inactive neurons
        candidates = []
        for neuron_idx, meta in metadata.items():
            if meta.mean_activity < self.config.survival_threshold:
                # Check if enough time since birth (give neurons time to integrate)
                if epoch - meta.birth_epoch >= self.config.maturation_epochs // 2:
                    candidates.append((neuron_idx, meta.mean_activity))

        if not candidates:
            return 0

        # Sort by activity (lowest first)
        candidates.sort(key=lambda x: x[1])

        # Limit pruning batch size
        max_prune = min(
            len(candidates),
            self.config.pruning_batch_size,
            current_neurons - self.config.min_neurons_per_layer,
        )

        pruned_count = 0
        for neuron_idx, _ in candidates[:max_prune]:
            # Remove from layer (done in reverse to maintain indices)
            removed_indices = [idx for idx, _ in candidates[:max_prune]]
            removed_indices.sort(reverse=True)

        # Actually remove neurons in reverse order
        for idx in removed_indices:
            layer.remove_neuron(idx)
            if idx in metadata:
                del metadata[idx]
            pruned_count += 1

        # Update statistics
        self.state.total_deaths += pruned_count
        self.state.last_pruning_epoch = epoch
        if self.state.death_history and self.state.current_epoch == epoch:
            self.state.death_history[-1] += pruned_count
        else:
            self.state.death_history.append(pruned_count)
        self.state._trim_history()

        if pruned_count > 0:
            logger.debug(
                f"Phase 4A: Pruned {pruned_count} neurons from layer {layer.layer_idx} "
                f"(epoch={epoch}, remaining={layer.config.hidden_dim})"
            )

        return pruned_count

    def update_activity(
        self, layer: ForwardForwardLayer, activations: np.ndarray, epoch: int
    ) -> None:
        """
        Record neuron activations for tracking activity levels.

        Args:
            layer: Layer that produced activations
            activations: Neuron activations [hidden_dim] or [batch, hidden_dim]
            epoch: Current training epoch
        """
        if layer.layer_idx not in self._neuron_metadata:
            return

        activations = np.atleast_2d(activations)
        mean_activations = np.mean(np.abs(activations), axis=0)

        metadata = self._neuron_metadata[layer.layer_idx]

        for neuron_idx in range(len(mean_activations)):
            if neuron_idx in metadata:
                metadata[neuron_idx].record_activity(
                    float(mean_activations[neuron_idx]), epoch
                )

    def update_maturation(self, epoch: int) -> dict:
        """
        Update maturity levels for all tracked neurons.

        Neurons gradually mature over configured epochs (Schmidt-Hieber 2004).

        Args:
            epoch: Current training epoch

        Returns:
            Maturation statistics
        """
        if not self.config.enable_maturation:
            return {"maturation_disabled": True}

        self.state.current_epoch = epoch
        delta = self.config.integration_rate

        stats = {
            "epoch": epoch,
            "immature_count": 0,
            "maturing_count": 0,
            "mature_count": 0,
        }

        for layer_idx, metadata_dict in self._neuron_metadata.items():
            for neuron_idx, meta in metadata_dict.items():
                meta.update_maturity(delta)

                # Count states
                if meta.state == NeuronState.IMMATURE:
                    stats["immature_count"] += 1
                elif meta.state == NeuronState.MATURING:
                    stats["maturing_count"] += 1
                elif meta.state == NeuronState.MATURE:
                    stats["mature_count"] += 1

        return stats

    def get_neuron_lr_multiplier(
        self, layer_idx: int, neuron_idx: int
    ) -> float:
        """
        Get learning rate multiplier for a specific neuron.

        Immature neurons have enhanced plasticity.

        Args:
            layer_idx: Layer index
            neuron_idx: Neuron index in layer

        Returns:
            Learning rate multiplier
        """
        if layer_idx not in self._neuron_metadata:
            return 1.0

        metadata = self._neuron_metadata[layer_idx]
        if neuron_idx not in metadata:
            return 1.0

        return metadata[neuron_idx].get_lr_multiplier(self.config)

    def get_layer_stats(self, layer_idx: int) -> dict:
        """Get neurogenesis statistics for a layer."""
        if layer_idx not in self._neuron_metadata:
            return {"tracked_neurons": 0}

        metadata = self._neuron_metadata[layer_idx]

        return {
            "tracked_neurons": len(metadata),
            "immature": sum(
                1 for m in metadata.values() if m.state == NeuronState.IMMATURE
            ),
            "maturing": sum(
                1 for m in metadata.values() if m.state == NeuronState.MATURING
            ),
            "mature": sum(
                1 for m in metadata.values() if m.state == NeuronState.MATURE
            ),
            "mean_maturity": (
                float(np.mean([m.maturity for m in metadata.values()]))
                if metadata
                else 0.0
            ),
            "mean_activity": (
                float(np.mean([m.mean_activity for m in metadata.values()]))
                if metadata
                else 0.0
            ),
        }

    def get_stats(self) -> dict:
        """Get global neurogenesis statistics."""
        total_neurons = sum(
            len(metadata) for metadata in self._neuron_metadata.values()
        )

        return {
            "state": self.state.to_dict(),
            "total_tracked_neurons": total_neurons,
            "layers_tracked": len(self._neuron_metadata),
            "config": {
                "birth_rate": self.config.birth_rate,
                "survival_threshold": self.config.survival_threshold,
                "maturation_epochs": self.config.maturation_epochs,
                "max_neurons_per_layer": self.config.max_neurons_per_layer,
            },
        }

    def reset(self) -> None:
        """Reset neurogenesis state."""
        self.state = NeurogenesisState()
        self._neuron_metadata.clear()
        logger.info("Phase 4A: NeurogenesisManager reset")


__all__ = [
    "NeurogenesisManager",
    "NeurogenesisConfig",
    "NeurogenesisState",
    "NeuronMetadata",
    "NeuronState",
]
