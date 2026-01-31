"""
Context Encoder for Latent Prediction.

P2-1: Encodes recent episode embeddings into a context representation.

Biological Basis:
- Hippocampal CA3 creates compressed representations from sequences
- Working memory holds recent context (~7 items)
- Temporal ordering matters for prediction

Architecture:
- Input: List of episode embeddings [n_context, 1024]
- Processing: Position encoding + attention-weighted aggregation
- Output: Single context vector [1024]

JEPA Insight: Context should capture "where we are" for prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContextEncoderConfig:
    """Configuration for context encoder."""

    # Dimensions
    embedding_dim: int = 1024  # BGE-M3 dimension
    hidden_dim: int = 512  # Internal hidden dimension
    context_dim: int = 1024  # Output context dimension (same as embedding)

    # Context window
    max_context_length: int = 8  # Maximum episodes in context
    use_position_encoding: bool = True  # Add positional information

    # Aggregation
    aggregation: str = "attention"  # "attention", "mean", "last", "lstm"
    attention_heads: int = 4  # For multi-head attention

    # Regularization
    dropout: float = 0.1
    layer_norm: bool = True


@dataclass
class EncodedContext:
    """Result of context encoding."""

    context_vector: np.ndarray  # [context_dim]
    attention_weights: np.ndarray | None  # [n_context] if attention used
    n_episodes: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_vector": self.context_vector.tolist(),
            "attention_weights": self.attention_weights.tolist() if self.attention_weights is not None else None,
            "n_episodes": self.n_episodes,
            "timestamp": self.timestamp.isoformat(),
        }


class ContextEncoder:
    """
    Encodes recent episode embeddings into a context representation.

    P2-1: Context is the "current state" for latent prediction.
    Given where we've been, predict where we're going.

    Architecture:
        episodes [n, 1024] → position encoding → attention → context [1024]

    Usage:
        encoder = ContextEncoder()
        context = encoder.encode(recent_embeddings)
        # context.context_vector is ready for prediction
    """

    def __init__(self, config: ContextEncoderConfig | None = None):
        """
        Initialize context encoder.

        Args:
            config: Encoder configuration
        """
        self.config = config or ContextEncoderConfig()

        # Initialize weights
        self._init_weights()

        logger.info(
            f"ContextEncoder initialized: "
            f"embedding_dim={self.config.embedding_dim}, "
            f"context_dim={self.config.context_dim}, "
            f"aggregation={self.config.aggregation}"
        )

    def _init_weights(self) -> None:
        """Initialize encoder weights."""
        np.random.seed(42)  # Reproducibility

        # Position encoding (learnable)
        if self.config.use_position_encoding:
            self._position_embedding = np.random.randn(
                self.config.max_context_length,
                self.config.embedding_dim
            ).astype(np.float32) * 0.02

        # Input projection
        self._W_in = np.random.randn(
            self.config.embedding_dim,
            self.config.hidden_dim
        ).astype(np.float32) * np.sqrt(2.0 / self.config.embedding_dim)

        # Attention weights (query, key, value projections)
        if self.config.aggregation == "attention":
            head_dim = self.config.hidden_dim // self.config.attention_heads

            self._W_query = np.random.randn(
                self.config.hidden_dim,
                self.config.hidden_dim
            ).astype(np.float32) * np.sqrt(2.0 / self.config.hidden_dim)

            self._W_key = np.random.randn(
                self.config.hidden_dim,
                self.config.hidden_dim
            ).astype(np.float32) * np.sqrt(2.0 / self.config.hidden_dim)

            self._W_value = np.random.randn(
                self.config.hidden_dim,
                self.config.hidden_dim
            ).astype(np.float32) * np.sqrt(2.0 / self.config.hidden_dim)

            # Global query for context aggregation
            self._global_query = np.random.randn(
                self.config.hidden_dim
            ).astype(np.float32) * 0.02

        # Output projection
        self._W_out = np.random.randn(
            self.config.hidden_dim,
            self.config.context_dim
        ).astype(np.float32) * np.sqrt(2.0 / self.config.hidden_dim)

        # Layer norm parameters
        if self.config.layer_norm:
            self._ln_gamma = np.ones(self.config.context_dim, dtype=np.float32)
            self._ln_beta = np.zeros(self.config.context_dim, dtype=np.float32)

    def encode(
        self,
        embeddings: list[np.ndarray] | np.ndarray,
        return_attention: bool = True,
    ) -> EncodedContext:
        """
        Encode episode embeddings into context representation.

        Args:
            embeddings: List of episode embeddings or array [n, 1024]
            return_attention: Whether to return attention weights

        Returns:
            EncodedContext with context vector and metadata
        """
        # Convert to array
        if isinstance(embeddings, list):
            if len(embeddings) == 0:
                # Empty context: return zero vector
                return EncodedContext(
                    context_vector=np.zeros(self.config.context_dim, dtype=np.float32),
                    attention_weights=None,
                    n_episodes=0,
                )
            embeddings = np.stack(embeddings)

        # Ensure 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n_episodes = embeddings.shape[0]

        # Truncate if too long
        if n_episodes > self.config.max_context_length:
            embeddings = embeddings[-self.config.max_context_length:]
            n_episodes = self.config.max_context_length

        # Add position encoding
        if self.config.use_position_encoding:
            positions = self._position_embedding[:n_episodes]
            embeddings = embeddings + positions

        # Input projection with ReLU
        hidden = np.maximum(0, embeddings @ self._W_in)  # [n, hidden_dim]

        # Aggregate based on method
        if self.config.aggregation == "attention":
            context, attention_weights = self._attention_aggregate(hidden)
        elif self.config.aggregation == "mean":
            context = hidden.mean(axis=0)
            attention_weights = np.ones(n_episodes) / n_episodes
        elif self.config.aggregation == "last":
            context = hidden[-1]
            attention_weights = np.zeros(n_episodes)
            attention_weights[-1] = 1.0
        else:
            # Default: mean
            context = hidden.mean(axis=0)
            attention_weights = np.ones(n_episodes) / n_episodes

        # Output projection
        context_out = context @ self._W_out

        # Layer normalization
        if self.config.layer_norm:
            context_out = self._layer_norm(context_out)

        # L2 normalize for cosine similarity compatibility
        norm = np.linalg.norm(context_out)
        if norm > 0:
            context_out = context_out / norm

        return EncodedContext(
            context_vector=context_out.astype(np.float32),
            attention_weights=attention_weights if return_attention else None,
            n_episodes=n_episodes,
        )

    def _attention_aggregate(
        self,
        hidden: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aggregate using attention with global query.

        Args:
            hidden: Hidden states [n, hidden_dim]

        Returns:
            (aggregated_context, attention_weights)
        """
        # Project to Q, K, V
        keys = hidden @ self._W_key  # [n, hidden_dim]
        values = hidden @ self._W_value  # [n, hidden_dim]

        # Global query for aggregation
        query = self._global_query @ self._W_query  # [hidden_dim]

        # Attention scores
        scale = np.sqrt(self.config.hidden_dim)
        scores = (keys @ query) / scale  # [n]

        # Softmax
        scores_max = scores.max()
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / (exp_scores.sum() + 1e-8)

        # Weighted sum of values
        context = (values.T @ attention_weights).T  # [hidden_dim]

        return context, attention_weights

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Apply layer normalization."""
        mean = x.mean()
        var = x.var()
        x_norm = (x - mean) / np.sqrt(var + eps)
        return self._ln_gamma * x_norm + self._ln_beta

    def update_weights(
        self,
        gradient: dict[str, np.ndarray],
        learning_rate: float = 0.001,
    ) -> None:
        """
        Update encoder weights from gradients.

        Args:
            gradient: Dictionary of gradients for each weight
            learning_rate: Learning rate
        """
        if "W_in" in gradient:
            self._W_in -= learning_rate * gradient["W_in"]
        if "W_out" in gradient:
            self._W_out -= learning_rate * gradient["W_out"]
        if "W_query" in gradient and self.config.aggregation == "attention":
            self._W_query -= learning_rate * gradient["W_query"]
        if "W_key" in gradient and self.config.aggregation == "attention":
            self._W_key -= learning_rate * gradient["W_key"]
        if "W_value" in gradient and self.config.aggregation == "attention":
            self._W_value -= learning_rate * gradient["W_value"]
        if "global_query" in gradient and self.config.aggregation == "attention":
            self._global_query -= learning_rate * gradient["global_query"]
        if "position_embedding" in gradient and self.config.use_position_encoding:
            self._position_embedding -= learning_rate * gradient["position_embedding"]

    def save_state(self) -> dict[str, Any]:
        """Save encoder state for persistence."""
        state = {
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "hidden_dim": self.config.hidden_dim,
                "context_dim": self.config.context_dim,
                "max_context_length": self.config.max_context_length,
                "use_position_encoding": self.config.use_position_encoding,
                "aggregation": self.config.aggregation,
                "attention_heads": self.config.attention_heads,
            },
            "W_in": self._W_in.tolist(),
            "W_out": self._W_out.tolist(),
        }

        if self.config.use_position_encoding:
            state["position_embedding"] = self._position_embedding.tolist()

        if self.config.aggregation == "attention":
            state["W_query"] = self._W_query.tolist()
            state["W_key"] = self._W_key.tolist()
            state["W_value"] = self._W_value.tolist()
            state["global_query"] = self._global_query.tolist()

        if self.config.layer_norm:
            state["ln_gamma"] = self._ln_gamma.tolist()
            state["ln_beta"] = self._ln_beta.tolist()

        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """Load encoder state from persistence."""
        self._W_in = np.array(state["W_in"], dtype=np.float32)
        self._W_out = np.array(state["W_out"], dtype=np.float32)

        if "position_embedding" in state:
            self._position_embedding = np.array(state["position_embedding"], dtype=np.float32)

        if "W_query" in state:
            self._W_query = np.array(state["W_query"], dtype=np.float32)
            self._W_key = np.array(state["W_key"], dtype=np.float32)
            self._W_value = np.array(state["W_value"], dtype=np.float32)
            self._global_query = np.array(state["global_query"], dtype=np.float32)

        if "ln_gamma" in state:
            self._ln_gamma = np.array(state["ln_gamma"], dtype=np.float32)
            self._ln_beta = np.array(state["ln_beta"], dtype=np.float32)
