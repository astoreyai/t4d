"""
Latent Predictor for World Models.

P2-2: MLP that predicts next latent state from context.

Biological Basis:
- Prefrontal cortex predicts future states for planning
- Prediction errors (mismatch) drive learning
- The brain is fundamentally a prediction machine (Friston)

Architecture:
- Input: Context vector [1024] from ContextEncoder
- Hidden: 2-layer MLP with ReLU
- Output: Predicted next embedding [1024]

JEPA/DreamerV3 Insight:
- Predict in embedding space, not raw content
- Avoids "snowball error" of pixel-level prediction
- Prediction error is the learning signal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatentPredictorConfig:
    """Configuration for latent predictor."""

    # Dimensions
    context_dim: int = 1024  # Input context dimension
    hidden_dim: int = 512  # Hidden layer dimension
    output_dim: int = 1024  # Predicted embedding dimension

    # Architecture
    num_hidden_layers: int = 2
    activation: str = "relu"  # "relu", "gelu", "tanh"

    # Regularization
    dropout: float = 0.1
    layer_norm: bool = True
    residual: bool = True  # Add residual connection

    # Learning
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # Prediction
    prediction_horizon: int = 1  # How many steps ahead to predict


@dataclass
class Prediction:
    """A latent prediction result."""

    predicted_embedding: np.ndarray  # [output_dim]
    confidence: float  # [0, 1] based on context quality
    context_summary: np.ndarray  # Input context used
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_embedding": self.predicted_embedding.tolist(),
            "confidence": self.confidence,
            "context_summary": self.context_summary.tolist(),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PredictionError:
    """
    Prediction error for an episode.

    This is the key learning signal: how wrong was our prediction?
    High error = surprising = prioritize for replay.
    """

    episode_id: UUID
    predicted: np.ndarray  # What we predicted
    actual: np.ndarray  # What actually happened
    error_magnitude: float  # L2 distance
    cosine_error: float  # 1 - cosine similarity
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def combined_error(self) -> float:
        """Combined error metric (mean of L2 and cosine)."""
        return (self.error_magnitude + self.cosine_error) / 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode_id": str(self.episode_id),
            "error_magnitude": self.error_magnitude,
            "cosine_error": self.cosine_error,
            "combined_error": self.combined_error,
            "timestamp": self.timestamp.isoformat(),
        }


class LatentPredictor:
    """
    MLP that predicts next latent state from context.

    P2-2: The core world model component.

    Architecture:
        context [1024] → hidden1 [512] → hidden2 [512] → predicted [1024]

    Training:
        - Input: Context from recent episodes
        - Target: Actual next episode embedding
        - Loss: L2 + cosine distance

    Usage:
        predictor = LatentPredictor()
        context = context_encoder.encode(recent_embeddings)
        prediction = predictor.predict(context.context_vector)

        # Later, when we see what actually happened:
        error = predictor.compute_error(prediction, actual_embedding, episode_id)
        predictor.train_step(context.context_vector, actual_embedding)
    """

    def __init__(self, config: LatentPredictorConfig | None = None):
        """
        Initialize latent predictor.

        Args:
            config: Predictor configuration
        """
        self.config = config or LatentPredictorConfig()

        # Initialize weights
        self._init_weights()

        # Momentum buffers for SGD
        self._momentum_buffers: dict[str, np.ndarray] = {}

        # Statistics
        self._total_predictions = 0
        self._total_errors = 0.0
        self._error_history: list[float] = []
        self._max_history = 1000

        logger.info(
            f"LatentPredictor initialized: "
            f"context_dim={self.config.context_dim}, "
            f"hidden_dim={self.config.hidden_dim}, "
            f"num_layers={self.config.num_hidden_layers}"
        )

    def _init_weights(self) -> None:
        """Initialize predictor weights."""
        np.random.seed(43)  # Different seed from encoder

        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []

        # Input layer
        input_dim = self.config.context_dim

        for i in range(self.config.num_hidden_layers):
            output_dim = self.config.hidden_dim
            W = np.random.randn(input_dim, output_dim).astype(np.float32)
            W *= np.sqrt(2.0 / input_dim)  # He initialization
            b = np.zeros(output_dim, dtype=np.float32)

            self._weights.append(W)
            self._biases.append(b)
            input_dim = output_dim

        # Output layer
        W_out = np.random.randn(input_dim, self.config.output_dim).astype(np.float32)
        W_out *= np.sqrt(2.0 / input_dim)
        b_out = np.zeros(self.config.output_dim, dtype=np.float32)

        self._weights.append(W_out)
        self._biases.append(b_out)

        # Layer norm parameters (for each layer)
        if self.config.layer_norm:
            self._ln_gammas = [
                np.ones(self.config.hidden_dim, dtype=np.float32)
                for _ in range(self.config.num_hidden_layers)
            ]
            self._ln_betas = [
                np.zeros(self.config.hidden_dim, dtype=np.float32)
                for _ in range(self.config.num_hidden_layers)
            ]

    def predict(
        self,
        context: np.ndarray,
        return_confidence: bool = True,
    ) -> Prediction:
        """
        Predict next latent state from context.

        Args:
            context: Context vector [context_dim]
            return_confidence: Whether to compute confidence

        Returns:
            Prediction with predicted embedding
        """
        # Forward pass
        x = context.copy()
        activations = [x]

        for i, (W, b) in enumerate(zip(self._weights[:-1], self._biases[:-1])):
            x = x @ W + b

            # Layer norm before activation
            if self.config.layer_norm:
                x = self._layer_norm(x, self._ln_gammas[i], self._ln_betas[i])

            # Activation
            x = self._activation(x)
            activations.append(x)

        # Output layer (no activation, will normalize)
        x = x @ self._weights[-1] + self._biases[-1]

        # Residual connection if enabled
        if self.config.residual and self.config.context_dim == self.config.output_dim:
            x = x + context

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm

        # Compute confidence based on context norm and activation patterns
        confidence = 1.0
        if return_confidence:
            context_norm = np.linalg.norm(context)
            confidence = min(1.0, context_norm / 1.5)  # Normalized contexts have norm ~1

        self._total_predictions += 1

        return Prediction(
            predicted_embedding=x.astype(np.float32),
            confidence=confidence,
            context_summary=context[:32],  # First 32 dims as summary
        )

    def compute_error(
        self,
        prediction: Prediction,
        actual_embedding: np.ndarray,
        episode_id: UUID,
    ) -> PredictionError:
        """
        Compute prediction error for an episode.

        Args:
            prediction: The prediction we made
            actual_embedding: What actually happened
            episode_id: ID of the actual episode

        Returns:
            PredictionError with error metrics
        """
        predicted = prediction.predicted_embedding
        actual = actual_embedding

        # L2 distance
        error_magnitude = float(np.linalg.norm(predicted - actual))

        # Cosine error (1 - similarity)
        dot = np.dot(predicted, actual)
        norm_pred = np.linalg.norm(predicted)
        norm_actual = np.linalg.norm(actual)
        if norm_pred > 0 and norm_actual > 0:
            cosine_sim = dot / (norm_pred * norm_actual)
            cosine_error = 1.0 - cosine_sim
        else:
            cosine_error = 1.0

        # Track statistics
        self._total_errors += (error_magnitude + cosine_error) / 2
        self._error_history.append((error_magnitude + cosine_error) / 2)
        if len(self._error_history) > self._max_history:
            self._error_history = self._error_history[-self._max_history:]

        return PredictionError(
            episode_id=episode_id,
            predicted=predicted,
            actual=actual,
            error_magnitude=error_magnitude,
            cosine_error=float(cosine_error),
        )

    def train_step(
        self,
        context: np.ndarray,
        target: np.ndarray,
        learning_rate: float | None = None,
    ) -> float:
        """
        Train predictor on single example.

        Args:
            context: Input context [context_dim]
            target: Target embedding [output_dim]
            learning_rate: Optional override learning rate

        Returns:
            Loss value
        """
        lr = learning_rate or self.config.learning_rate

        # Forward pass with cached activations
        x = context.copy()
        activations = [x]

        for i, (W, b) in enumerate(zip(self._weights[:-1], self._biases[:-1])):
            x = x @ W + b
            if self.config.layer_norm:
                x = self._layer_norm(x, self._ln_gammas[i], self._ln_betas[i])
            x = self._activation(x)
            activations.append(x)

        # Output
        output = x @ self._weights[-1] + self._biases[-1]
        if self.config.residual and self.config.context_dim == self.config.output_dim:
            output = output + context

        # Normalize
        norm = np.linalg.norm(output)
        if norm > 0:
            output = output / norm

        # Loss: L2 + cosine
        l2_loss = np.sum((output - target) ** 2)
        cosine_loss = 1.0 - np.dot(output, target) / (
            np.linalg.norm(output) * np.linalg.norm(target) + 1e-8
        )
        loss = 0.5 * l2_loss + 0.5 * cosine_loss

        # Backward pass (simplified gradient descent)
        # Gradient of normalized output w.r.t. unnormalized
        grad_output = 2 * (output - target) + 0.5 * (-target / (norm + 1e-8))

        # Output layer gradient
        grad_W_out = np.outer(activations[-1], grad_output)
        grad_b_out = grad_output.copy()

        # Backprop through residual
        if self.config.residual:
            grad_input = grad_output.copy()
        else:
            grad_input = np.zeros_like(context)

        # Backprop through hidden layers
        grad = grad_output @ self._weights[-1].T
        grads_W = [grad_W_out]
        grads_b = [grad_b_out]

        for i in range(self.config.num_hidden_layers - 1, -1, -1):
            # Through activation
            grad = grad * self._activation_grad(activations[i + 1])

            # Gradient for this layer
            grad_W = np.outer(activations[i], grad)
            grad_b = grad.copy()

            grads_W.insert(0, grad_W)
            grads_b.insert(0, grad_b)

            # Propagate gradient
            if i > 0:
                grad = grad @ self._weights[i].T

        # Update weights with momentum
        for i, (grad_W, grad_b) in enumerate(zip(grads_W, grads_b)):
            # Add weight decay
            grad_W = grad_W + self.config.weight_decay * self._weights[i]

            # Momentum
            key_W = f"W_{i}"
            key_b = f"b_{i}"

            if key_W not in self._momentum_buffers:
                self._momentum_buffers[key_W] = np.zeros_like(self._weights[i])
                self._momentum_buffers[key_b] = np.zeros_like(self._biases[i])

            self._momentum_buffers[key_W] = (
                self.config.momentum * self._momentum_buffers[key_W] + grad_W
            )
            self._momentum_buffers[key_b] = (
                self.config.momentum * self._momentum_buffers[key_b] + grad_b
            )

            self._weights[i] -= lr * self._momentum_buffers[key_W]
            self._biases[i] -= lr * self._momentum_buffers[key_b]

        return float(loss)

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.config.activation == "relu":
            return np.maximum(0, x)
        elif self.config.activation == "gelu":
            # Approximate GELU
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif self.config.activation == "tanh":
            return np.tanh(x)
        else:
            return np.maximum(0, x)  # Default ReLU

    def _activation_grad(self, x: np.ndarray) -> np.ndarray:
        """Gradient of activation function."""
        if self.config.activation == "relu":
            return (x > 0).astype(np.float32)
        elif self.config.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        else:
            return (x > 0).astype(np.float32)

    def _layer_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Apply layer normalization."""
        mean = x.mean()
        var = x.var()
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta

    def get_statistics(self) -> dict[str, Any]:
        """Get predictor statistics."""
        return {
            "total_predictions": self._total_predictions,
            "mean_error": self._total_errors / max(1, len(self._error_history)),
            "recent_error": np.mean(self._error_history[-100:]) if self._error_history else 0.0,
            "error_trend": self._compute_error_trend(),
        }

    def _compute_error_trend(self) -> str:
        """Compute error trend (improving, stable, degrading)."""
        if len(self._error_history) < 20:
            return "insufficient_data"

        recent = np.mean(self._error_history[-20:])
        earlier = np.mean(self._error_history[-100:-20]) if len(self._error_history) >= 100 else np.mean(self._error_history[:-20])

        if recent < earlier * 0.9:
            return "improving"
        elif recent > earlier * 1.1:
            return "degrading"
        else:
            return "stable"

    def save_state(self) -> dict[str, Any]:
        """Save predictor state."""
        return {
            "config": {
                "context_dim": self.config.context_dim,
                "hidden_dim": self.config.hidden_dim,
                "output_dim": self.config.output_dim,
                "num_hidden_layers": self.config.num_hidden_layers,
                "activation": self.config.activation,
                "residual": self.config.residual,
            },
            "weights": [w.tolist() for w in self._weights],
            "biases": [b.tolist() for b in self._biases],
            "momentum_buffers": {k: v.tolist() for k, v in self._momentum_buffers.items()},
            "statistics": self.get_statistics(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load predictor state."""
        self._weights = [np.array(w, dtype=np.float32) for w in state["weights"]]
        self._biases = [np.array(b, dtype=np.float32) for b in state["biases"]]
        self._momentum_buffers = {
            k: np.array(v, dtype=np.float32)
            for k, v in state.get("momentum_buffers", {}).items()
        }
