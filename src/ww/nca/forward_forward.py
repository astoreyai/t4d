"""
Forward-Forward Algorithm Implementation.

Implements Hinton's Forward-Forward Algorithm (2022) for layer-local learning
without backpropagation. Each layer learns to maximize "goodness" for positive
data and minimize it for negative data.

Theoretical Foundation:
=======================

1. Goodness Function (Hinton, 2022):
   G(h) = sum(h_i^2) for layer activations h
   - High goodness = "positive" (real) data
   - Low goodness = "negative" (fake) data
   - Threshold theta separates positive/negative

2. Local Learning Rule:
   - Positive phase: increase G when seeing real data
   - Negative phase: decrease G when seeing corrupted data
   - No backward pass through network required

3. Biological Plausibility:
   - Each layer has local objective (above/below threshold)
   - Learning uses only pre/post synaptic activity
   - Maps to Hebbian-like plasticity with phase-dependent sign

Integration with World Weaver:
==============================
- Connects to neuromodulator system for learning rate modulation
- Dopamine boosts learning rate on surprise
- ACh gates encoding vs retrieval phase
- NE arousal modulates classification threshold

References:
- Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations
- Lillicrap et al. (2020). Backpropagation and the brain
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class FFPhase(Enum):
    """Phase of Forward-Forward learning."""

    POSITIVE = auto()  # Maximize goodness for real data
    NEGATIVE = auto()  # Minimize goodness for fake data
    INFERENCE = auto()  # Classification only (no learning)


class NegativeMethod(Enum):
    """Method for generating negative samples."""

    NOISE = "noise"  # Add Gaussian noise
    SHUFFLE = "shuffle"  # Randomly permute features
    ADVERSARIAL = "adversarial"  # Gradient ascent on goodness
    HYBRID = "hybrid"  # Mix of above methods
    WRONG_LABEL = "wrong_label"  # Correct data with wrong label


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ForwardForwardConfig:
    """
    Configuration for Forward-Forward layer.

    Parameters based on Hinton (2022):
    - Threshold theta determines positive/negative boundary
    - Goodness margin controls confidence
    - Learning rate typically higher than backprop (0.03-0.1)

    Biological mapping:
    - Threshold ~ homeostatic set point for neural activity
    - Goodness ~ local metabolic cost / energy consumption
    - Learning ~ Hebbian plasticity with phase-dependent sign
    """

    # Layer dimensions
    input_dim: int = 1024
    hidden_dim: int = 512

    # Goodness parameters (Hinton 2022)
    threshold_theta: float = 2.0  # Goodness threshold for classification
    goodness_margin: float = 0.5  # Margin for confident classification
    normalize_goodness: bool = True  # Divide by layer size for comparability

    # Learning parameters
    learning_rate: float = 0.03  # Higher than backprop (Hinton suggests 0.03-0.1)
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Activation function
    activation: str = "relu"  # "relu", "leaky_relu", "gelu"
    leaky_slope: float = 0.01

    # Negative generation
    negative_method: str = "hybrid"  # noise/shuffle/adversarial/hybrid
    noise_scale: float = 0.3
    adversarial_steps: int = 5
    adversarial_lr: float = 0.1

    # Neuromodulation integration
    use_neuromod_gating: bool = True
    da_modulates_lr: bool = True  # Dopamine surprise boosts learning
    ach_modulates_phase: bool = True  # ACh determines encoding vs retrieval
    ne_modulates_threshold: bool = True  # NE arousal affects threshold

    # Biological constraints
    max_weight: float = 2.0  # Synaptic saturation
    min_weight: float = -2.0
    sparsity_target: float = 0.05  # Target activation sparsity
    weight_normalization: bool = True  # Layer norm on weights

    # Layer normalization
    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-5

    # History tracking (bounded - MEM-007 pattern)
    max_history: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.threshold_theta > 0, "threshold must be positive"
        assert 0 < self.learning_rate < 1, "learning_rate must be in (0, 1)"
        assert self.activation in ("relu", "leaky_relu", "gelu"), \
            f"Unknown activation: {self.activation}"


@dataclass
class ForwardForwardState:
    """State of a Forward-Forward layer."""

    # Layer outputs
    activations: np.ndarray = field(
        default_factory=lambda: np.zeros(512, dtype=np.float32)
    )
    pre_activations: np.ndarray = field(
        default_factory=lambda: np.zeros(512, dtype=np.float32)
    )

    # Goodness metrics
    goodness: float = 0.0  # Sum of squared activations
    normalized_goodness: float = 0.0  # Goodness / layer_size

    # Classification
    is_positive: bool = True  # Above threshold
    confidence: float = 0.0  # Distance from threshold (signed)
    probability: float = 0.5  # Sigmoid of goodness - threshold

    # Learning statistics (bounded history - MEM-007 pattern)
    positive_goodness_history: list = field(default_factory=list)
    negative_goodness_history: list = field(default_factory=list)
    accuracy_history: list = field(default_factory=list)
    gradient_norm_history: list = field(default_factory=list)

    # Phase tracking
    current_phase: FFPhase = FFPhase.INFERENCE
    total_positive_samples: int = 0
    total_negative_samples: int = 0
    total_updates: int = 0

    # Weight statistics
    weight_mean: float = 0.0
    weight_std: float = 1.0
    weight_max: float = 0.0

    timestamp: datetime = field(default_factory=datetime.now)

    _max_history: int = 1000  # Bounded history

    def to_dict(self) -> dict:
        """Serialize for logging/persistence."""
        return {
            "goodness": self.goodness,
            "normalized_goodness": self.normalized_goodness,
            "is_positive": self.is_positive,
            "confidence": self.confidence,
            "total_updates": self.total_updates,
            "current_phase": self.current_phase.name,
            "recent_accuracy": (
                float(np.mean(self.accuracy_history[-100:]))
                if self.accuracy_history
                else 0.0
            ),
        }

    def _trim_history(self) -> None:
        """Trim history lists to max_history length."""
        if len(self.positive_goodness_history) > self._max_history:
            self.positive_goodness_history = self.positive_goodness_history[
                -self._max_history :
            ]
        if len(self.negative_goodness_history) > self._max_history:
            self.negative_goodness_history = self.negative_goodness_history[
                -self._max_history :
            ]
        if len(self.accuracy_history) > self._max_history:
            self.accuracy_history = self.accuracy_history[-self._max_history :]
        if len(self.gradient_norm_history) > self._max_history:
            self.gradient_norm_history = self.gradient_norm_history[
                -self._max_history :
            ]


# =============================================================================
# Forward-Forward Layer
# =============================================================================


class ForwardForwardLayer:
    """
    Single Forward-Forward layer with local learning.

    Implements Hinton (2022) FF algorithm:
    1. Forward pass computes activations and goodness
    2. Positive phase: increase goodness for real data
    3. Negative phase: decrease goodness for fake data
    4. No backward pass through network

    Biological interpretation:
    - Goodness ~ local energy (sum of squared activities)
    - Learning ~ Hebbian with polarity determined by phase
    - Threshold ~ homeostatic set point maintained by inhibition

    Hinton's key insight: "The goodness function provides a local
    learning signal that does not require backpropagation."
    """

    def __init__(
        self,
        config: ForwardForwardConfig | None = None,
        layer_idx: int = 0,
        random_seed: int | None = None,
    ):
        """
        Initialize Forward-Forward layer.

        Args:
            config: Layer configuration
            layer_idx: Index in network (for logging)
            random_seed: Random seed for reproducibility
        """
        self.config = config or ForwardForwardConfig()
        self.layer_idx = layer_idx
        self._rng = np.random.default_rng(random_seed)

        # Initialize state
        self.state = ForwardForwardState(
            activations=np.zeros(self.config.hidden_dim, dtype=np.float32),
            pre_activations=np.zeros(self.config.hidden_dim, dtype=np.float32),
        )

        # Initialize weights (Xavier/Glorot)
        self._init_weights()

        # Optional layer normalization parameters
        if self.config.use_layer_norm:
            self.gamma = np.ones(self.config.hidden_dim, dtype=np.float32)
            self.beta = np.zeros(self.config.hidden_dim, dtype=np.float32)

        # Momentum buffers
        self._weight_momentum = np.zeros_like(self.W)
        self._bias_momentum = np.zeros_like(self.b)

        logger.debug(
            f"ForwardForwardLayer[{layer_idx}] initialized: "
            f"{self.config.input_dim} -> {self.config.hidden_dim}"
        )

    def _init_weights(self) -> None:
        """Initialize weights with Xavier initialization."""
        scale = np.sqrt(2.0 / (self.config.input_dim + self.config.hidden_dim))
        self.W = self._rng.normal(
            0, scale, size=(self.config.input_dim, self.config.hidden_dim)
        ).astype(np.float32)
        self.b = np.zeros(self.config.hidden_dim, dtype=np.float32)

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + self.config.layer_norm_eps)
        return self.gamma * normalized + self.beta

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.config.activation == "relu":
            return np.maximum(0, x)
        elif self.config.activation == "leaky_relu":
            return np.where(x > 0, x, self.config.leaky_slope * x)
        elif self.config.activation == "gelu":
            return (
                0.5
                * x
                * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
            )
        else:
            return x

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass computing activations and goodness.

        Args:
            x: Input tensor [batch, input_dim] or [input_dim]
            training: Enable dropout and update statistics

        Returns:
            Activations [batch, hidden_dim] or [hidden_dim]
        """
        x = np.atleast_2d(x).astype(np.float32)

        # Linear transformation
        pre_act = x @ self.W + self.b
        self.state.pre_activations = pre_act.copy()

        # Layer normalization
        if self.config.use_layer_norm:
            pre_act = self._layer_norm(pre_act)

        # Activation function
        h = self._apply_activation(pre_act)

        # Update state
        self.state.activations = h.copy()
        self.state.goodness = self.compute_goodness(h)
        if self.config.normalize_goodness:
            self.state.normalized_goodness = (
                self.state.goodness / self.config.hidden_dim
            )
        else:
            self.state.normalized_goodness = self.state.goodness

        # Classification
        self.state.is_positive, self.state.confidence = self.classify(h)

        # Update weight statistics
        self.state.weight_mean = float(np.mean(self.W))
        self.state.weight_std = float(np.std(self.W))
        self.state.weight_max = float(np.max(np.abs(self.W)))

        return h.squeeze() if h.shape[0] == 1 else h

    def compute_goodness(self, h: np.ndarray) -> float:
        """
        Compute goodness = sum of squared activations.

        Hinton (2022): "The goodness of a layer is simply the sum
        of the squares of the activities of the rectified linear units."

        Args:
            h: Layer activations

        Returns:
            Goodness value (sum of squared activations)
        """
        return float(np.sum(h**2))

    def classify(self, h: np.ndarray) -> tuple[bool, float]:
        """
        Classify as positive/negative based on goodness threshold.

        Args:
            h: Layer activations

        Returns:
            (is_positive, confidence): Classification and signed distance from threshold
        """
        goodness = self.compute_goodness(h)
        threshold = self.config.threshold_theta

        confidence = goodness - threshold
        is_positive = confidence > 0

        # ATOM-P4-6: Clip confidence to prevent exp overflow
        confidence_safe = np.clip(confidence, -500, 500)
        # Probability via sigmoid
        self.state.probability = 1.0 / (1.0 + np.exp(-confidence_safe))

        return is_positive, float(confidence)

    def learn_positive(self, x: np.ndarray, h: np.ndarray) -> dict:
        """
        Update weights to INCREASE goodness for positive sample.

        The local learning rule:
            delta_W = lr * (1 - p) * h * x^T
        where p = sigmoid(goodness - threshold)

        When goodness is below threshold, gradient is strong.
        When goodness is above threshold, gradient weakens.

        Args:
            x: Input activations
            h: Layer output activations

        Returns:
            Training statistics
        """
        x = np.atleast_2d(x).astype(np.float32)
        h = np.atleast_2d(h).astype(np.float32)

        self.state.current_phase = FFPhase.POSITIVE
        lr = self.config.learning_rate
        p = self.state.probability

        # Gradient: increase goodness
        # dG/dW = 2 * h * x^T (for squared activations)
        # Weighted by (1 - p) to reduce updates when already confident
        gradient = (1 - p) * (h.T @ x).T

        # Apply momentum and weight decay
        self._weight_momentum = self.config.momentum * self._weight_momentum + lr * (
            gradient - self.config.weight_decay * self.W
        )

        self.W += self._weight_momentum

        # Enforce weight bounds
        self.W = np.clip(self.W, self.config.min_weight, self.config.max_weight)

        # Update statistics
        self.state.total_positive_samples += 1
        self.state.total_updates += 1
        self.state.positive_goodness_history.append(self.state.goodness)
        self.state.gradient_norm_history.append(float(np.linalg.norm(gradient)))
        self.state._trim_history()

        return {
            "phase": "positive",
            "goodness": self.state.goodness,
            "probability": p,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "lr_effective": lr,
        }

    def learn_negative(self, x: np.ndarray, h: np.ndarray) -> dict:
        """
        Update weights to DECREASE goodness for negative sample.

        The local learning rule:
            delta_W = -lr * p * h * x^T
        where p = sigmoid(goodness - threshold)

        When goodness is above threshold (wrongly classified), gradient is strong.

        Args:
            x: Input activations
            h: Layer output activations

        Returns:
            Training statistics
        """
        x = np.atleast_2d(x).astype(np.float32)
        h = np.atleast_2d(h).astype(np.float32)

        self.state.current_phase = FFPhase.NEGATIVE
        lr = self.config.learning_rate
        p = self.state.probability

        # Gradient: decrease goodness (negative sign)
        gradient = -p * (h.T @ x).T

        # Apply momentum and weight decay
        self._weight_momentum = self.config.momentum * self._weight_momentum + lr * (
            gradient - self.config.weight_decay * self.W
        )

        self.W += self._weight_momentum

        # Enforce weight bounds
        self.W = np.clip(self.W, self.config.min_weight, self.config.max_weight)

        # Update statistics
        self.state.total_negative_samples += 1
        self.state.total_updates += 1
        self.state.negative_goodness_history.append(self.state.goodness)
        self.state.gradient_norm_history.append(float(np.linalg.norm(gradient)))
        self.state._trim_history()

        return {
            "phase": "negative",
            "goodness": self.state.goodness,
            "probability": p,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "lr_effective": lr,
        }

    # =========================================================================
    # Convenience methods (simplified API)
    # =========================================================================

    def train_positive(self, x: np.ndarray, provenance: str | None = None) -> dict:
        """
        Train on positive sample (convenience method).

        Combines forward pass and learning in one call.
        Use this when you don't need separate control over forward/learn phases.

        ATOM-P2-20: Added provenance parameter for input tracking.

        Args:
            x: Input data (positive sample)
            provenance: Optional input origin (e.g., "memory", "sensor", "replay")

        Returns:
            Training statistics from learn_positive
        """
        h = self.forward(x, training=True)
        stats = self.learn_positive(x, h)
        # ATOM-P2-20: Track provenance if provided
        if provenance is not None:
            stats["provenance"] = provenance
        return stats

    def train_negative(self, x: np.ndarray, provenance: str | None = None) -> dict:
        """
        Train on negative sample (convenience method).

        Combines forward pass and learning in one call.
        Use this when you don't need separate control over forward/learn phases.

        ATOM-P2-20: Added provenance parameter for input tracking.

        Args:
            x: Input data (negative sample)
            provenance: Optional input origin (e.g., "noise", "adversarial")

        Returns:
            Training statistics from learn_negative
        """
        h = self.forward(x, training=True)
        stats = self.learn_negative(x, h)
        # ATOM-P2-20: Track provenance if provided
        if provenance is not None:
            stats["provenance"] = provenance
        return stats

    # =========================================================================
    # Phase 4A: Neurogenesis Support
    # =========================================================================

    def add_neuron(self, weights: np.ndarray) -> int:
        """
        Add a new neuron to the layer (activity-dependent neurogenesis).

        Phase 4A: This enables the layer to grow in response to novel patterns,
        mimicking hippocampal adult neurogenesis (Kempermann 2015).

        Args:
            weights: Initial weights for new neuron [input_dim]

        Returns:
            Index of the newly added neuron
        """
        weights = weights.astype(np.float32)
        assert weights.shape == (self.config.input_dim,), \
            f"Weights shape {weights.shape} must match input_dim {self.config.input_dim}"

        # Add column to weight matrix
        self.W = np.column_stack([self.W, weights])

        # Add bias for new neuron
        self.b = np.append(self.b, 0.0)

        # Update layer norm parameters if enabled
        if self.config.use_layer_norm:
            self.gamma = np.append(self.gamma, 1.0)
            self.beta = np.append(self.beta, 0.0)

        # Expand momentum buffers
        self._weight_momentum = np.column_stack([
            self._weight_momentum,
            np.zeros(self.config.input_dim, dtype=np.float32)
        ])
        self._bias_momentum = np.append(self._bias_momentum, 0.0)

        # Update config and state
        new_idx = self.config.hidden_dim
        self.config.hidden_dim += 1

        self.state.activations = np.zeros(self.config.hidden_dim, dtype=np.float32)
        self.state.pre_activations = np.zeros(self.config.hidden_dim, dtype=np.float32)

        logger.debug(
            f"Phase 4A: Neuron added to layer {self.layer_idx} "
            f"(new_count={self.config.hidden_dim})"
        )

        return new_idx

    def remove_neuron(self, neuron_idx: int) -> None:
        """
        Remove a neuron from the layer (activity-dependent pruning).

        Phase 4A: This enables pruning of inactive neurons, maintaining
        computational efficiency while allowing growth (Tashiro 2007).

        Args:
            neuron_idx: Index of neuron to remove
        """
        assert 0 <= neuron_idx < self.config.hidden_dim, \
            f"Neuron index {neuron_idx} out of range [0, {self.config.hidden_dim})"

        # Remove column from weight matrix
        self.W = np.delete(self.W, neuron_idx, axis=1)

        # Remove bias
        self.b = np.delete(self.b, neuron_idx)

        # Update layer norm parameters if enabled
        if self.config.use_layer_norm:
            self.gamma = np.delete(self.gamma, neuron_idx)
            self.beta = np.delete(self.beta, neuron_idx)

        # Update momentum buffers
        self._weight_momentum = np.delete(self._weight_momentum, neuron_idx, axis=1)
        self._bias_momentum = np.delete(self._bias_momentum, neuron_idx)

        # Update config and state
        self.config.hidden_dim -= 1

        self.state.activations = np.zeros(self.config.hidden_dim, dtype=np.float32)
        self.state.pre_activations = np.zeros(self.config.hidden_dim, dtype=np.float32)

        logger.debug(
            f"Phase 4A: Neuron removed from layer {self.layer_idx} "
            f"(remaining={self.config.hidden_dim})"
        )

    def get_stats(self) -> dict:
        """Get layer statistics."""
        return {
            "layer_idx": self.layer_idx,
            "input_dim": self.config.input_dim,
            "hidden_dim": self.config.hidden_dim,
            "total_updates": self.state.total_updates,
            "positive_samples": self.state.total_positive_samples,
            "negative_samples": self.state.total_negative_samples,
            "mean_positive_goodness": (
                float(np.mean(self.state.positive_goodness_history))
                if self.state.positive_goodness_history
                else 0.0
            ),
            "mean_negative_goodness": (
                float(np.mean(self.state.negative_goodness_history))
                if self.state.negative_goodness_history
                else 0.0
            ),
            "weight_mean": self.state.weight_mean,
            "weight_std": self.state.weight_std,
        }

    def reset(self) -> None:
        """Reset layer to initial state."""
        self._init_weights()
        self.state = ForwardForwardState(
            activations=np.zeros(self.config.hidden_dim, dtype=np.float32),
            pre_activations=np.zeros(self.config.hidden_dim, dtype=np.float32),
        )
        self._weight_momentum = np.zeros_like(self.W)
        self._bias_momentum = np.zeros_like(self.b)


# =============================================================================
# Forward-Forward Network
# =============================================================================


class ForwardForwardNetwork:
    """
    Multi-layer Forward-Forward network.

    Each layer learns independently using FF rule:
    - Layer 1: Learns to classify (input + label) as positive/negative
    - Layer 2+: Learns to classify previous layer output as positive/negative

    Architecture follows Hinton (2022):
    "Each layer has its own objective function which is simply to have
    high goodness for positive data and low goodness for negative data."
    """

    def __init__(
        self,
        layer_dims: list[int],
        config: ForwardForwardConfig | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize multi-layer FF network.

        Args:
            layer_dims: List of layer dimensions [input, hidden1, hidden2, ...]
            config: Base configuration (dimensions will be overridden per layer)
            random_seed: Random seed for reproducibility
        """
        self.config = config or ForwardForwardConfig()
        self.layer_dims = layer_dims
        self._rng = np.random.default_rng(random_seed)

        # Create layers
        self.layers: list[ForwardForwardLayer] = []

        for i in range(len(layer_dims) - 1):
            layer_config = ForwardForwardConfig(
                input_dim=layer_dims[i],
                hidden_dim=layer_dims[i + 1],
                threshold_theta=self.config.threshold_theta,
                learning_rate=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                activation=self.config.activation,
                negative_method=self.config.negative_method,
                noise_scale=self.config.noise_scale,
                use_layer_norm=self.config.use_layer_norm,
                max_weight=self.config.max_weight,
                min_weight=self.config.min_weight,
            )
            layer = ForwardForwardLayer(
                layer_config,
                layer_idx=i,
                random_seed=int(self._rng.integers(0, 2**31)),
            )
            self.layers.append(layer)

        # Statistics
        self._total_training_steps = 0

        logger.info(
            f"ForwardForwardNetwork initialized with {len(self.layers)} layers: "
            f"{layer_dims}"
        )

    def forward(
        self, x: np.ndarray, label: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """
        Forward pass through all layers.

        For classification, label is embedded in first layer input
        following Hinton (2022) approach.

        Args:
            x: Input data [batch, input_dim] or [input_dim]
            label: Optional one-hot label to embed

        Returns:
            List of activations from each layer
        """
        x = np.atleast_2d(x).astype(np.float32)

        # Optionally embed label in input (first 10 dims)
        if label is not None:
            x = self._embed_label(x, label)

        activations = []
        current = x

        for layer in self.layers:
            current = layer.forward(current, training=True)
            activations.append(current)

        return activations

    def _embed_label(self, x: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        Embed label into input by replacing first n_classes dimensions.

        Args:
            x: Input data [batch, input_dim]
            label: One-hot label [batch, n_classes] or [n_classes]

        Returns:
            Modified input with label embedded
        """
        label = np.atleast_2d(label).astype(np.float32)
        n_classes = label.shape[-1]

        x = x.copy()
        x[:, :n_classes] = label

        return x

    def train_step(
        self,
        positive_data: np.ndarray,
        negative_data: np.ndarray,
        positive_labels: np.ndarray | None = None,
        negative_labels: np.ndarray | None = None,
    ) -> dict:
        """
        One training step with positive and negative phases.

        Args:
            positive_data: Real data samples
            negative_data: Corrupted/fake samples
            positive_labels: Labels for positive (optional)
            negative_labels: Wrong labels for negative (optional)

        Returns:
            Training statistics per layer
        """
        stats = {"layers": [], "positive_phase": [], "negative_phase": []}

        # Positive phase
        pos_activations = self.forward(positive_data, positive_labels)

        for i, (layer, h) in enumerate(zip(self.layers, pos_activations)):
            if i == 0:
                x = (
                    self._embed_label(
                        np.atleast_2d(positive_data), positive_labels
                    )
                    if positive_labels is not None
                    else np.atleast_2d(positive_data)
                )
            else:
                x = np.atleast_2d(pos_activations[i - 1])

            layer_stats = layer.learn_positive(x, np.atleast_2d(h))
            layer_stats["layer_idx"] = i
            stats["positive_phase"].append(layer_stats)

        # Negative phase
        neg_activations = self.forward(negative_data, negative_labels)

        for i, (layer, h) in enumerate(zip(self.layers, neg_activations)):
            if i == 0:
                x = (
                    self._embed_label(
                        np.atleast_2d(negative_data), negative_labels
                    )
                    if negative_labels is not None
                    else np.atleast_2d(negative_data)
                )
            else:
                x = np.atleast_2d(neg_activations[i - 1])

            layer_stats = layer.learn_negative(x, np.atleast_2d(h))
            layer_stats["layer_idx"] = i
            stats["negative_phase"].append(layer_stats)

        self._total_training_steps += 1
        stats["total_steps"] = self._total_training_steps

        # Compute layer accuracies
        for i, layer in enumerate(self.layers):
            stats["layers"].append(layer.get_stats())

        return stats

    def generate_negative(
        self,
        positive_data: np.ndarray,
        method: str | None = None,
        labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Generate negative samples using configured method.

        Methods:
        - "noise": Add Gaussian noise
        - "shuffle": Randomly permute features
        - "adversarial": Gradient ascent on goodness
        - "hybrid": Mix of above
        - "wrong_label": Correct data with wrong label

        ATOM-P3-17: Method validation added to restrict to safe methods.

        Args:
            positive_data: Real data to corrupt
            method: Override default method
            labels: Labels for wrong_label method

        Returns:
            (negative_data, negative_labels)
        """
        method = method or self.config.negative_method
        positive_data = np.atleast_2d(positive_data).astype(np.float32)

        # ATOM-P3-17: Validate method
        valid_methods = {"noise", "shuffle", "adversarial", "wrong_label", "hybrid"}
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

        if method == "noise":
            noise = self._rng.normal(
                0, self.config.noise_scale, positive_data.shape
            )
            return np.clip(positive_data + noise, 0, 1).astype(np.float32), labels

        elif method == "shuffle":
            # Shuffle features within each sample
            neg = positive_data.copy()
            for i in range(neg.shape[0]):
                self._rng.shuffle(neg[i])
            return neg.astype(np.float32), labels

        elif method == "adversarial":
            # Gradient ascent on first layer's goodness
            neg = positive_data.copy()
            for _ in range(self.config.adversarial_steps):
                h = self.layers[0].forward(neg)
                h = np.atleast_2d(h)
                # Approximate gradient: direction that increases goodness
                grad = 2 * h @ self.layers[0].W.T
                neg = neg + self.config.adversarial_lr * grad
                neg = np.clip(neg, 0, 1)
            return neg.astype(np.float32), labels

        elif method == "wrong_label":
            # Use correct data but wrong labels
            if labels is None:
                return self.generate_negative(positive_data, "noise")
            # Shift labels by random amount
            labels = np.atleast_2d(labels)
            n_classes = labels.shape[-1]
            shift = self._rng.integers(1, n_classes)
            wrong_labels = np.roll(labels, shift, axis=-1)
            return positive_data.astype(np.float32), wrong_labels

        elif method == "hybrid":
            # Mix methods
            batch_size = positive_data.shape[0]
            n_noise = batch_size // 3
            n_shuffle = batch_size // 3
            n_adversarial = batch_size - n_noise - n_shuffle

            if n_noise > 0:
                neg_noise, _ = self.generate_negative(
                    positive_data[:n_noise], "noise"
                )
            else:
                neg_noise = np.array([]).reshape(0, positive_data.shape[1])

            if n_shuffle > 0:
                neg_shuffle, _ = self.generate_negative(
                    positive_data[n_noise : n_noise + n_shuffle], "shuffle"
                )
            else:
                neg_shuffle = np.array([]).reshape(0, positive_data.shape[1])

            if n_adversarial > 0:
                neg_adv, _ = self.generate_negative(
                    positive_data[-n_adversarial:], "adversarial"
                )
            else:
                neg_adv = np.array([]).reshape(0, positive_data.shape[1])

            neg_all = np.concatenate([neg_noise, neg_shuffle, neg_adv], axis=0)
            return neg_all.astype(np.float32), labels

        else:
            raise ValueError(f"Unknown negative method: {method}")

    def infer(self, x: np.ndarray) -> tuple[bool, float]:
        """
        Inference: classify input by layer goodness.

        Returns:
            (is_positive, confidence): Overall classification
        """
        activations = self.forward(x)

        # Sum goodness across layers
        total_goodness = sum(layer.state.goodness for layer in self.layers)

        # Average confidence
        avg_confidence = np.mean([layer.state.confidence for layer in self.layers])

        # Use final layer's classification
        is_positive = self.layers[-1].state.is_positive

        return is_positive, float(avg_confidence)

    def get_stats(self) -> dict:
        """Get network statistics."""
        return {
            "n_layers": len(self.layers),
            "layer_dims": self.layer_dims,
            "total_training_steps": self._total_training_steps,
            "layer_stats": [layer.get_stats() for layer in self.layers],
        }

    def reset(self) -> None:
        """Reset all layers to initial state."""
        for layer in self.layers:
            layer.reset()
        self._total_training_steps = 0


# =============================================================================
# Factory Functions
# =============================================================================


def create_ff_layer(
    input_dim: int = 1024,
    hidden_dim: int = 512,
    learning_rate: float = 0.03,
    threshold: float = 2.0,
    random_seed: int | None = None,
) -> ForwardForwardLayer:
    """
    Factory function to create a Forward-Forward layer.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden/output dimension
        learning_rate: Learning rate
        threshold: Goodness threshold
        random_seed: Random seed

    Returns:
        Configured ForwardForwardLayer
    """
    config = ForwardForwardConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        threshold_theta=threshold,
    )
    return ForwardForwardLayer(config, random_seed=random_seed)


def create_ff_network(
    layer_dims: list[int],
    learning_rate: float = 0.03,
    threshold: float = 2.0,
    random_seed: int | None = None,
) -> ForwardForwardNetwork:
    """
    Factory function to create a Forward-Forward network.

    Args:
        layer_dims: List of layer dimensions
        learning_rate: Learning rate for all layers
        threshold: Goodness threshold for all layers
        random_seed: Random seed

    Returns:
        Configured ForwardForwardNetwork
    """
    config = ForwardForwardConfig(
        learning_rate=learning_rate,
        threshold_theta=threshold,
    )
    return ForwardForwardNetwork(layer_dims, config, random_seed=random_seed)
