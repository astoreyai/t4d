"""
P5.4: Query-Memory Encoder Separation.

Implements separate encoding pathways for queries and memories,
following insights from Dense Passage Retrieval (DPR) and CA3/CA1
hippocampal specialization.

Biological Basis:
- CA3: Pattern completion (recurrent, attractor dynamics) - memory encoding
- CA1: Output encoding for downstream use - query encoding
- Separate projections allow asymmetric optimization

Architecture:
- Query projection: Optimized for intent extraction
- Memory projection: Optimized for content representation
- Learned through retrieval outcome feedback

References:
- Karpicke & Roediger (2008) "The critical importance of retrieval for learning"
- Kang et al. (2020) "Dense Passage Retrieval for Open-Domain Question Answering"
- Tulving & Thomson (1973) "Encoding specificity principle"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SeparationConfig:
    """Configuration for query-memory separation."""
    embedding_dim: int = 1024
    hidden_dim: int = 256  # Projection bottleneck
    learning_rate: float = 0.01
    momentum: float = 0.9
    regularization: float = 0.001  # L2 weight decay
    initial_scale: float = 0.1  # Initial projection scale


@dataclass
class SeparationStats:
    """Statistics for query-memory separator."""
    query_projections: int = 0
    memory_projections: int = 0
    training_updates: int = 0
    avg_query_norm: float = 1.0
    avg_memory_norm: float = 1.0
    last_loss: float = 0.0
    last_updated: datetime | None = None


class QueryMemorySeparator:
    """
    P5.4: Learned query-memory encoder separation.

    Projects queries and memories through separate learned projections,
    optimizing for asymmetric similarity matching.

    Query path: x -> W_q -> h -> U_q -> y_q
    Memory path: x -> W_m -> h -> U_m -> y_m

    The projections are trained to maximize similarity between
    queries and relevant memories while minimizing similarity
    to irrelevant memories.
    """

    def __init__(self, config: SeparationConfig | None = None):
        """
        Initialize query-memory separator.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or SeparationConfig()
        self.stats = SeparationStats()

        # Initialize projections with orthogonal-like initialization
        dim = self.config.embedding_dim
        hdim = self.config.hidden_dim
        scale = self.config.initial_scale

        # Query path: dim -> hdim -> dim
        self.W_q = np.random.randn(dim, hdim).astype(np.float32) * scale / np.sqrt(dim)
        self.U_q = np.random.randn(hdim, dim).astype(np.float32) * scale / np.sqrt(hdim)

        # Memory path: dim -> hdim -> dim
        self.W_m = np.random.randn(dim, hdim).astype(np.float32) * scale / np.sqrt(dim)
        self.U_m = np.random.randn(hdim, dim).astype(np.float32) * scale / np.sqrt(hdim)

        # Momentum buffers for SGD with momentum
        self._v_Wq = np.zeros_like(self.W_q)
        self._v_Uq = np.zeros_like(self.U_q)
        self._v_Wm = np.zeros_like(self.W_m)
        self._v_Um = np.zeros_like(self.U_m)

        logger.info(
            f"QueryMemorySeparator initialized: "
            f"dim={dim}, hidden={hdim}, "
            f"params={(dim * hdim + hdim * dim) * 2}"
        )

    def project_query(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project embedding through query pathway.

        Uses residual connection for stability:
        y = x + tanh(x @ W_q) @ U_q

        Args:
            embedding: Query embedding [dim] or [batch, dim]

        Returns:
            Projected query embedding
        """
        x = np.atleast_2d(embedding).astype(np.float32)

        # Forward pass with tanh nonlinearity and residual
        h = np.tanh(x @ self.W_q)
        residual = h @ self.U_q
        y = x + residual

        # Normalize to unit sphere
        norms = np.linalg.norm(y, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        y = y / norms

        # Update stats
        self.stats.query_projections += x.shape[0]
        self.stats.avg_query_norm = float(np.mean(norms))

        # Return in original shape
        if embedding.ndim == 1:
            return y[0]
        return y

    def project_memory(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project embedding through memory pathway.

        Uses residual connection for stability:
        y = x + tanh(x @ W_m) @ U_m

        Args:
            embedding: Memory embedding [dim] or [batch, dim]

        Returns:
            Projected memory embedding
        """
        x = np.atleast_2d(embedding).astype(np.float32)

        # Forward pass with tanh nonlinearity and residual
        h = np.tanh(x @ self.W_m)
        residual = h @ self.U_m
        y = x + residual

        # Normalize to unit sphere
        norms = np.linalg.norm(y, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        y = y / norms

        # Update stats
        self.stats.memory_projections += x.shape[0]
        self.stats.avg_memory_norm = float(np.mean(norms))

        if embedding.ndim == 1:
            return y[0]
        return y

    def compute_similarity(
        self,
        query: np.ndarray,
        memories: np.ndarray,
        project: bool = True
    ) -> np.ndarray:
        """
        Compute query-memory similarity with optional projection.

        Args:
            query: Query embedding [dim]
            memories: Memory embeddings [n, dim]
            project: Whether to apply projections

        Returns:
            Similarity scores [n]
        """
        if project:
            q = self.project_query(query)
            m = self.project_memory(memories)
        else:
            q = query
            m = memories

        # Cosine similarity (already normalized if projected)
        return np.dot(m, q)

    def train_step(
        self,
        query: np.ndarray,
        positive: np.ndarray,
        negatives: np.ndarray,
        margin: float = 0.2
    ) -> float:
        """
        Train on triplet (query, positive, negatives) using margin loss.

        Loss = max(0, margin - sim(q, pos) + sim(q, neg))

        Args:
            query: Query embedding [dim]
            positive: Positive (relevant) memory [dim]
            negatives: Negative memories [n, dim]
            margin: Triplet margin

        Returns:
            Training loss
        """
        lr = self.config.learning_rate
        momentum = self.config.momentum
        reg = self.config.regularization

        # Forward pass
        q = np.atleast_2d(query).astype(np.float32)
        pos = np.atleast_2d(positive).astype(np.float32)
        negs = np.atleast_2d(negatives).astype(np.float32)

        # Query projection
        h_q = np.tanh(q @ self.W_q)
        y_q = q + h_q @ self.U_q
        y_q_norm = y_q / (np.linalg.norm(y_q, axis=-1, keepdims=True) + 1e-8)

        # Positive projection
        h_pos = np.tanh(pos @ self.W_m)
        y_pos = pos + h_pos @ self.U_m
        y_pos_norm = y_pos / (np.linalg.norm(y_pos, axis=-1, keepdims=True) + 1e-8)

        # Negative projections
        h_neg = np.tanh(negs @ self.W_m)
        y_neg = negs + h_neg @ self.U_m
        y_neg_norm = y_neg / (np.linalg.norm(y_neg, axis=-1, keepdims=True) + 1e-8)

        # Compute similarities
        sim_pos = float(np.dot(y_q_norm[0], y_pos_norm[0]))
        sim_negs = np.dot(y_neg_norm, y_q_norm[0])

        # Triplet margin loss
        losses = np.maximum(0, margin - sim_pos + sim_negs)
        loss = float(np.mean(losses))

        if loss > 0:
            # Compute gradients (simplified, approximate)
            # dL/dy_q ∝ -y_pos + mean(y_neg) for violating triplets
            violating = losses > 0
            if np.any(violating):
                n_violating = np.sum(violating)
                grad_yq = (-y_pos_norm[0] + np.mean(y_neg_norm[violating], axis=0)) / n_violating

                # Backprop through query path
                # dy/dU_q = h_q^T
                grad_Uq = np.outer(h_q[0], grad_yq)
                # dy/dh_q = U_q @ grad_yq
                grad_hq = self.U_q @ grad_yq
                # dh_q/dW_q = (1 - h_q^2) * q^T (tanh derivative)
                tanh_deriv = 1 - h_q[0] ** 2
                grad_Wq = np.outer(q[0], grad_hq * tanh_deriv)

                # Backprop through memory path (for positive)
                grad_ypos = y_q_norm[0] / n_violating
                grad_Upos = np.outer(h_pos[0], grad_ypos)
                grad_hpos = self.U_m @ grad_ypos
                tanh_deriv_pos = 1 - h_pos[0] ** 2
                grad_Wpos = np.outer(pos[0], grad_hpos * tanh_deriv_pos)

                # Update with momentum and regularization
                self._v_Wq = momentum * self._v_Wq - lr * (grad_Wq + reg * self.W_q)
                self._v_Uq = momentum * self._v_Uq - lr * (grad_Uq + reg * self.U_q)
                self.W_q += self._v_Wq
                self.U_q += self._v_Uq

                self._v_Wm = momentum * self._v_Wm - lr * (grad_Wpos + reg * self.W_m)
                self._v_Um = momentum * self._v_Um - lr * (grad_Upos + reg * self.U_m)
                self.W_m += self._v_Wm
                self.U_m += self._v_Um

        # Update stats
        self.stats.training_updates += 1
        self.stats.last_loss = loss
        self.stats.last_updated = datetime.now()

        return loss

    def get_stats(self) -> dict:
        """Get separator statistics."""
        return {
            "query_projections": self.stats.query_projections,
            "memory_projections": self.stats.memory_projections,
            "training_updates": self.stats.training_updates,
            "avg_query_norm": round(self.stats.avg_query_norm, 4),
            "avg_memory_norm": round(self.stats.avg_memory_norm, 4),
            "last_loss": round(self.stats.last_loss, 4),
            "last_updated": self.stats.last_updated.isoformat() if self.stats.last_updated else None,
        }

    def save_state(self) -> dict:
        """Save separator state for persistence."""
        return {
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "hidden_dim": self.config.hidden_dim,
                "learning_rate": self.config.learning_rate,
                "momentum": self.config.momentum,
                "regularization": self.config.regularization,
            },
            "W_q": self.W_q.tolist(),
            "U_q": self.U_q.tolist(),
            "W_m": self.W_m.tolist(),
            "U_m": self.U_m.tolist(),
            "stats": self.get_stats(),
        }

    def load_state(self, state: dict) -> None:
        """Load separator state from dictionary."""
        self.W_q = np.array(state["W_q"], dtype=np.float32)
        self.U_q = np.array(state["U_q"], dtype=np.float32)
        self.W_m = np.array(state["W_m"], dtype=np.float32)
        self.U_m = np.array(state["U_m"], dtype=np.float32)

        # Reset momentum buffers
        self._v_Wq = np.zeros_like(self.W_q)
        self._v_Uq = np.zeros_like(self.U_q)
        self._v_Wm = np.zeros_like(self.W_m)
        self._v_Um = np.zeros_like(self.U_m)

        logger.info("QueryMemorySeparator state loaded")


# Singleton instance for global use
_separator: QueryMemorySeparator | None = None


def get_query_memory_separator() -> QueryMemorySeparator:
    """Get or create global QueryMemorySeparator instance."""
    global _separator
    if _separator is None:
        _separator = QueryMemorySeparator()
    return _separator


def reset_separator() -> None:
    """Reset global separator (for testing)."""
    global _separator
    _separator = None
