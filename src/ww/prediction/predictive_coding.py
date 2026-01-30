"""
Hierarchical Predictive Coding for World Weaver.

Implements Rao & Ballard (1999) / Friston (2005) style predictive coding
where each level generates predictions for the level below and computes
prediction errors from the mismatch.

Key Concepts:
1. Hierarchy: Each level predicts the level below
2. Prediction Error: Mismatch between prediction and actual
3. Precision Weighting: Confidence in predictions vs errors
4. Top-down: Predictions flow from abstract to concrete
5. Bottom-up: Prediction errors flow from concrete to abstract

Architecture:
    Level 3 (Abstract)     ↓ predict ↑ error
    Level 2 (Intermediate) ↓ predict ↑ error
    Level 1 (Concrete)     ↓ predict ↑ error
    Sensory Input

Integration Points:
- DendriticNeuron: Basal=bottom-up, Apical=top-down predictions
- HierarchicalPredictor: Multi-timescale prediction targets
- DopamineSystem: Precision-weighted prediction errors → RPE

References:
- Rao & Ballard (1999): Predictive coding in the visual cortex
- Friston (2005): A theory of cortical responses
- Clark (2013): Whatever next? Predictive brains, situated agents
- Keller & Mrsic-Flogel (2018): Predictive processing in sensory cortex
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PredictionDirection(Enum):
    """Direction of information flow."""
    TOP_DOWN = "top_down"      # Predictions (generative)
    BOTTOM_UP = "bottom_up"    # Prediction errors (inferential)


@dataclass
class PredictiveCodingConfig:
    """Configuration for hierarchical predictive coding.

    Attributes:
        n_levels: Number of hierarchical levels (including input)
        dims: Dimension at each level [input, level1, level2, ...]
        learning_rates: LR at each level (decreasing up hierarchy)
        precision_init: Initial precision weights (confidence)
        update_precision: Whether to learn precision weights
        error_gain: Gain on prediction errors (attention)
    """
    n_levels: int = 3
    dims: list[int] = field(default_factory=lambda: [1024, 512, 256])
    learning_rates: list[float] = field(default_factory=lambda: [0.01, 0.005, 0.001])
    precision_init: float = 1.0
    update_precision: bool = True
    error_gain: float = 1.0
    prediction_gain: float = 1.0


@dataclass
class LevelState:
    """State of a single hierarchical level.

    Attributes:
        level_id: Level index (0 = closest to input)
        representation: Current state estimate
        prediction: Top-down prediction (what we expect)
        error: Prediction error (surprise)
        precision: Confidence in this level's predictions
        timestamp: When last updated
    """
    level_id: int
    representation: np.ndarray
    prediction: np.ndarray | None = None
    error: np.ndarray | None = None
    precision: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def error_magnitude(self) -> float:
        """L2 norm of prediction error."""
        if self.error is None:
            return 0.0
        return float(np.linalg.norm(self.error))

    @property
    def weighted_error(self) -> float:
        """Precision-weighted prediction error."""
        return self.error_magnitude * self.precision


@dataclass
class HierarchyState:
    """Complete state of predictive hierarchy.

    Attributes:
        levels: State at each hierarchical level
        total_error: Sum of precision-weighted errors
        timestamp: When computed
    """
    levels: list[LevelState]
    total_error: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_levels": len(self.levels),
            "total_error": self.total_error,
            "level_errors": [l.error_magnitude for l in self.levels],
            "level_precisions": [l.precision for l in self.levels],
            "timestamp": self.timestamp.isoformat()
        }


class PredictiveLevel:
    """A single level in the predictive coding hierarchy.

    Each level:
    1. Receives predictions from level above (top-down)
    2. Computes prediction error vs. representation
    3. Sends errors to level above (bottom-up)
    4. Generates predictions for level below
    """

    def __init__(
        self,
        level_id: int,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.01,
        precision: float = 1.0
    ):
        """Initialize predictive level.

        Args:
            level_id: Level index in hierarchy
            input_dim: Dimension of input (from below or sensory)
            output_dim: Dimension of this level's representation
            learning_rate: Learning rate for weight updates
            precision: Initial precision (confidence)
        """
        self.level_id = level_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.precision = precision

        # Weights for encoding (bottom-up)
        self.W_encode = np.random.randn(output_dim, input_dim) * 0.01

        # Weights for prediction (top-down)
        self.W_predict = np.random.randn(input_dim, output_dim) * 0.01

        # Current state
        self.representation = np.zeros(output_dim)
        self.prediction = np.zeros(input_dim)
        self.error = np.zeros(input_dim)

        # Statistics
        self._n_updates = 0
        self._cumulative_error = 0.0

    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """
        Encode input from level below (or sensory).

        Args:
            input_data: Input representation

        Returns:
            This level's representation
        """
        # Simple linear encoding with nonlinearity
        self.representation = np.tanh(self.W_encode @ input_data)
        return self.representation

    def predict(self) -> np.ndarray:
        """
        Generate prediction for level below.

        Returns:
            Predicted representation for lower level
        """
        self.prediction = np.tanh(self.W_predict @ self.representation)
        return self.prediction

    def compute_error(
        self,
        target: np.ndarray,
        prediction: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute prediction error.

        Args:
            target: Actual representation from below
            prediction: Optional explicit prediction (else use self.prediction)

        Returns:
            Prediction error vector
        """
        pred = prediction if prediction is not None else self.prediction
        self.error = target - pred
        return self.error

    def update_weights(
        self,
        error_from_above: np.ndarray | None = None,
        error_for_below: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Update weights based on prediction errors.

        Args:
            error_from_above: Error signal from level above (backprop-like)
            error_for_below: Error at our prediction for level below

        Returns:
            Update statistics
        """
        updates = {}

        # Update prediction weights (reduce error at level below)
        if error_for_below is not None:
            # Gradient: prediction error w.r.t. prediction weights
            grad = np.outer(error_for_below, self.representation)
            self.W_predict += self.learning_rate * self.precision * grad
            updates["predict_update_norm"] = float(np.linalg.norm(grad))

        # Update encoding weights (use error from above)
        if error_from_above is not None:
            # Pass error through prediction weights, scale by local error
            backprop = self.W_predict.T @ error_from_above
            grad = np.outer(backprop, self.representation) * (1 - self.representation**2)
            self.W_encode += self.learning_rate * grad
            updates["encode_update_norm"] = float(np.linalg.norm(grad))

        self._n_updates += 1
        self._cumulative_error += self.error_magnitude

        return updates

    @property
    def error_magnitude(self) -> float:
        """L2 norm of prediction error."""
        return float(np.linalg.norm(self.error))

    def get_state(self) -> LevelState:
        """Get current level state."""
        return LevelState(
            level_id=self.level_id,
            representation=self.representation.copy(),
            prediction=self.prediction.copy(),
            error=self.error.copy(),
            precision=self.precision
        )


class PredictiveCodingHierarchy:
    """
    Complete hierarchical predictive coding system.

    Implements Friston-style free energy minimization through prediction
    error minimization at each level of a representational hierarchy.

    Example:
        ```python
        config = PredictiveCodingConfig(
            n_levels=3,
            dims=[1024, 512, 256]
        )
        hierarchy = PredictiveCodingHierarchy(config)

        # Process sensory input
        state = hierarchy.process(sensory_embedding)

        # Get total prediction error (free energy proxy)
        free_energy = state.total_error

        # Learn from errors
        hierarchy.update()

        # Get top-level abstract representation
        abstract = hierarchy.get_top_representation()
        ```
    """

    def __init__(self, config: PredictiveCodingConfig | None = None):
        """
        Initialize predictive coding hierarchy.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or PredictiveCodingConfig()

        # Validate configuration
        if len(self.config.dims) != self.config.n_levels:
            raise ValueError(
                f"dims length ({len(self.config.dims)}) must match "
                f"n_levels ({self.config.n_levels})"
            )

        # Create levels
        self.levels: list[PredictiveLevel] = []
        for i in range(self.config.n_levels - 1):
            input_dim = self.config.dims[i]
            output_dim = self.config.dims[i + 1]
            lr = self.config.learning_rates[min(i, len(self.config.learning_rates) - 1)]

            level = PredictiveLevel(
                level_id=i + 1,  # Level 0 is input
                input_dim=input_dim,
                output_dim=output_dim,
                learning_rate=lr,
                precision=self.config.precision_init
            )
            self.levels.append(level)

        # Track last input for updates
        self._last_input: np.ndarray | None = None
        self._history: list[HierarchyState] = []

        logger.info(
            f"PredictiveCodingHierarchy initialized: "
            f"{self.config.n_levels} levels, dims={self.config.dims}"
        )

    def process(self, sensory_input: np.ndarray) -> HierarchyState:
        """
        Process sensory input through hierarchy.

        1. Bottom-up: Encode input through levels
        2. Top-down: Generate predictions at each level
        3. Compute prediction errors at each level
        4. Return hierarchy state with total error

        Args:
            sensory_input: Input embedding (e.g., BGE-M3 1024-dim)

        Returns:
            Complete hierarchy state with errors at each level
        """
        self._last_input = sensory_input.copy()
        level_states = []

        # Level 0: Input level (no processing)
        level_states.append(LevelState(
            level_id=0,
            representation=sensory_input,
            prediction=None,
            error=None,
            precision=1.0
        ))

        # Bottom-up pass: encode through hierarchy
        current = sensory_input
        for level in self.levels:
            current = level.encode(current)

        # Top-down pass: generate predictions
        for level in reversed(self.levels):
            level.predict()

        # Compute prediction errors
        # Level 1 predicts level 0 (input)
        self.levels[0].compute_error(sensory_input)

        # Higher levels predict lower levels
        for i in range(1, len(self.levels)):
            lower_rep = self.levels[i - 1].representation
            self.levels[i].compute_error(lower_rep)

        # Collect states
        for level in self.levels:
            level_states.append(level.get_state())

        # Compute total precision-weighted error
        total_error = sum(
            level.error_magnitude * level.precision * self.config.error_gain
            for level in self.levels
        )

        state = HierarchyState(
            levels=level_states,
            total_error=total_error
        )

        self._history.append(state)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return state

    def update(self) -> dict[str, Any]:
        """
        Update weights to minimize prediction errors.

        Implements predictive coding learning:
        - Each level adjusts to reduce its prediction error
        - Precision weights can adapt based on error statistics

        Returns:
            Update statistics
        """
        stats = {"level_updates": []}

        # Update each level
        for i, level in enumerate(self.levels):
            # Get error from level above (if exists)
            error_from_above = None
            if i < len(self.levels) - 1:
                error_from_above = self.levels[i + 1].error

            # Get error for level below
            error_for_below = level.error

            # Update weights
            update = level.update_weights(
                error_from_above=error_from_above,
                error_for_below=error_for_below
            )
            stats["level_updates"].append(update)

        # Update precision weights (optional)
        if self.config.update_precision:
            self._update_precision()

        return stats

    def _update_precision(self):
        """Update precision weights based on error statistics."""
        for level in self.levels:
            if level._n_updates > 0:
                avg_error = level._cumulative_error / level._n_updates
                # Higher average error → lower precision (less confident)
                # Exponential moving average
                level.precision = 0.99 * level.precision + 0.01 * (1.0 / (1.0 + avg_error))

    def get_top_representation(self) -> np.ndarray:
        """Get the most abstract (top-level) representation."""
        if not self.levels:
            return np.zeros(self.config.dims[-1])
        return self.levels[-1].representation.copy()

    def get_prediction_for_input(self) -> np.ndarray:
        """Get the top-down prediction for sensory input."""
        if not self.levels:
            return np.zeros(self.config.dims[0])
        return self.levels[0].prediction.copy()

    def get_statistics(self) -> dict[str, Any]:
        """Get hierarchy statistics."""
        return {
            "n_levels": len(self.levels) + 1,  # +1 for input level
            "dims": self.config.dims,
            "level_errors": [l.error_magnitude for l in self.levels],
            "level_precisions": [l.precision for l in self.levels],
            "total_updates": sum(l._n_updates for l in self.levels),
            "history_size": len(self._history),
            "mean_total_error": (
                np.mean([s.total_error for s in self._history])
                if self._history else 0.0
            )
        }

    def compute_dopamine_signal(
        self,
        baseline: float = 0.3,
        gain: float = 0.5
    ) -> float:
        """
        P5.2: Compute dopamine-compatible signal from prediction errors.

        Bridges hierarchical prediction error to DopamineSystem for RPE-based
        learning modulation. High prediction error = high surprise = positive RPE.

        Biological basis: Cortical prediction errors modulate VTA dopamine neurons.
        Precision-weighted errors indicate reliability of surprise signal.

        Args:
            baseline: Expected average error (determines surprise threshold)
            gain: Scaling factor for RPE conversion

        Returns:
            RPE-like signal in range [-1, 1], where:
            - Positive: Surprising (error > baseline)
            - Negative: Expected (error < baseline)
            - Zero: Matches expectations

        References:
            - Schultz et al. (1997): Dopamine neurons encode prediction errors
            - Friston (2005): Precision-weighted prediction errors in free energy
        """
        if not self.levels:
            return 0.0

        # Sum precision-weighted errors across all levels
        total_weighted = sum(
            level.error_magnitude * level.precision
            for level in self.levels
        )

        # Normalize by number of levels
        avg_error = total_weighted / len(self.levels)

        # Convert to RPE: positive if surprising, negative if expected
        rpe = (avg_error - baseline) * gain

        # Clip to [-1, 1] range
        return float(np.clip(rpe, -1.0, 1.0))

    def get_precision_weighted_error(self) -> np.ndarray:
        """
        P5.2: Get precision-weighted prediction error vector.

        Returns error signal suitable for direct injection into learning systems.

        Returns:
            Array of precision-weighted errors per level
        """
        return np.array([
            level.error_magnitude * level.precision
            for level in self.levels
        ])

    def reset(self):
        """Reset all level states."""
        for level in self.levels:
            level.representation = np.zeros(level.output_dim)
            level.prediction = np.zeros(level.input_dim)
            level.error = np.zeros(level.input_dim)
            level.precision = self.config.precision_init
            level._n_updates = 0
            level._cumulative_error = 0.0
        self._history.clear()


# Factory function
def create_predictive_hierarchy(
    n_levels: int = 3,
    input_dim: int = 1024,
    compression_ratio: float = 0.5
) -> PredictiveCodingHierarchy:
    """
    Create a predictive coding hierarchy.

    Args:
        n_levels: Number of levels (including input)
        input_dim: Input dimension (e.g., 1024 for BGE-M3)
        compression_ratio: Reduction factor per level

    Returns:
        Configured PredictiveCodingHierarchy
    """
    dims = [input_dim]
    for _ in range(n_levels - 1):
        dims.append(int(dims[-1] * compression_ratio))

    config = PredictiveCodingConfig(
        n_levels=n_levels,
        dims=dims
    )

    return PredictiveCodingHierarchy(config)
