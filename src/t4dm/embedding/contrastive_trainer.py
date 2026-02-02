"""
Contrastive Adapter for T4DM.

Implements learnable projection on top of frozen BGE-M3 embeddings,
enabling the memory system to adapt representations to task-specific needs.

Hinton-inspired principles:
- Forward-forward style local learning (no full backprop through embeddings)
- Contrastive loss for learning meaningful distinctions
- Temperature-scaled softmax for entropy control
- Hard negative mining for efficient learning

References:
- Oord et al. (2018): Contrastive Predictive Coding
- Chen et al. (2020): SimCLR
- Hinton (2022): Forward-Forward Algorithm
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AdapterMode(Enum):
    """Adapter training modes."""
    FROZEN = "frozen"        # Only use frozen embeddings
    ADAPTER = "adapter"      # Train adapter on top of frozen
    FINE_TUNE = "fine_tune"  # Full fine-tuning (expensive)


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive adapter.

    Parameters based on contrastive learning literature:
    - Temperature: 0.07-0.2 typical (lower = harder distinctions)
    - Projection dim: 128-256 for efficiency
    - Hard negative ratio: 0.1-0.3 for mining
    """

    # Architecture
    input_dim: int = 1024          # BGE-M3 dimension
    hidden_dim: int = 512          # Hidden layer size
    output_dim: int = 256          # Projection dimension
    num_layers: int = 2            # MLP depth

    # Training
    learning_rate: float = 1e-4
    temperature: float = 0.1       # InfoNCE temperature
    temperature_learnable: bool = True
    weight_decay: float = 1e-5

    # Hard negative mining
    hard_negative_ratio: float = 0.2
    semi_hard_margin: float = 0.1

    # Regularization
    dropout: float = 0.1
    layer_norm: bool = True

    # Memory-specific
    temporal_weight: float = 0.3   # Weight for temporal contrastive
    semantic_weight: float = 0.7   # Weight for semantic contrastive

    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.temperature <= 1.0, "Temperature must be in (0, 1]"
        assert 0 <= self.hard_negative_ratio <= 1.0


@dataclass
class AdapterStats:
    """Statistics for contrastive adapter training."""

    total_updates: int = 0
    total_samples: int = 0
    total_loss: float = 0.0
    contrastive_loss: float = 0.0
    temporal_loss: float = 0.0
    accuracy: float = 0.0
    hard_negative_ratio: float = 0.0
    temperature: float = 0.1
    last_updated: datetime | None = None

    @property
    def avg_loss(self) -> float:
        """Average loss per update."""
        return self.total_loss / self.total_updates if self.total_updates > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_updates": self.total_updates,
            "total_samples": self.total_samples,
            "avg_loss": round(self.avg_loss, 6),
            "contrastive_loss": round(self.contrastive_loss, 6),
            "temporal_loss": round(self.temporal_loss, 6),
            "accuracy": round(self.accuracy, 4),
            "hard_negative_ratio": round(self.hard_negative_ratio, 4),
            "temperature": round(self.temperature, 4),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class ContrastiveAdapter:
    """
    Learnable projection adapter for frozen embeddings.

    Architecture:
        Frozen BGE-M3 → Linear → LayerNorm → ReLU → Linear → L2 Normalize

    Training:
        - InfoNCE loss for semantic similarity
        - Temporal contrastive for episode ordering
        - Hard negative mining for efficiency

    Example:
        >>> config = ContrastiveConfig()
        >>> adapter = ContrastiveAdapter(config)
        >>> embeddings = np.random.randn(32, 1024)  # Batch of frozen embeddings
        >>> projected = adapter.forward(embeddings)
        >>> print(projected.shape)  # (32, 256)
    """

    def __init__(self, config: ContrastiveConfig | None = None):
        """
        Initialize contrastive adapter.

        Args:
            config: Adapter configuration
        """
        self.config = config or ContrastiveConfig()
        self.stats = AdapterStats()
        self.mode = AdapterMode.ADAPTER

        # Initialize weights (Xavier/Glorot initialization)
        self._init_weights()

        # Learnable temperature
        self._log_temperature = math.log(self.config.temperature)

    def _init_weights(self) -> None:
        """Initialize adapter weights with Xavier initialization."""
        cfg = self.config

        # First layer: input_dim → hidden_dim
        scale1 = np.sqrt(2.0 / (cfg.input_dim + cfg.hidden_dim))
        self.W1 = np.random.randn(cfg.input_dim, cfg.hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(cfg.hidden_dim, dtype=np.float32)

        # Layer norm parameters
        self.gamma1 = np.ones(cfg.hidden_dim, dtype=np.float32)
        self.beta1 = np.zeros(cfg.hidden_dim, dtype=np.float32)

        # Second layer: hidden_dim → output_dim
        scale2 = np.sqrt(2.0 / (cfg.hidden_dim + cfg.output_dim))
        self.W2 = np.random.randn(cfg.hidden_dim, cfg.output_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(cfg.output_dim, dtype=np.float32)

        # Optional third layer for deeper networks
        if cfg.num_layers >= 3:
            scale3 = np.sqrt(2.0 / (cfg.output_dim + cfg.output_dim))
            self.W3 = np.random.randn(cfg.output_dim, cfg.output_dim).astype(np.float32) * scale3
            self.b3 = np.zeros(cfg.output_dim, dtype=np.float32)
        else:
            self.W3 = None
            self.b3 = None

        # Store gradients for optimization
        self._grads: dict[str, np.ndarray] = {}

        # Adam optimizer state
        self._m: dict[str, np.ndarray] = {}  # First moment
        self._v: dict[str, np.ndarray] = {}  # Second moment
        self._t = 0  # Timestep

    @property
    def temperature(self) -> float:
        """Get current temperature (may be learned)."""
        return math.exp(self._log_temperature)

    def forward(self, embeddings: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass through adapter.

        Args:
            embeddings: Input embeddings from frozen encoder [batch, input_dim]
            training: Whether in training mode (enables dropout)

        Returns:
            Projected embeddings [batch, output_dim]
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Layer 1: Linear → LayerNorm → ReLU
        h = embeddings @ self.W1 + self.b1

        if self.config.layer_norm:
            h = self._layer_norm(h, self.gamma1, self.beta1)

        h = self._relu(h)

        if training and self.config.dropout > 0:
            h = self._dropout(h, self.config.dropout)

        # Layer 2: Linear
        out = h @ self.W2 + self.b2

        # Optional Layer 3
        if self.W3 is not None:
            out = self._relu(out)
            if training and self.config.dropout > 0:
                out = self._dropout(out, self.config.dropout)
            out = out @ self.W3 + self.b3

        # L2 normalize for cosine similarity
        out = self._l2_normalize(out)

        return out

    def _layer_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _dropout(self, x: np.ndarray, p: float) -> np.ndarray:
        """Dropout during training."""
        mask = np.random.binomial(1, 1 - p, x.shape).astype(np.float32)
        return x * mask / (1 - p)

    def _l2_normalize(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """L2 normalize vectors."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + eps)

    def compute_similarity(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarities for contrastive learning.

        Args:
            anchors: Anchor embeddings [batch, dim]
            positives: Positive embeddings [batch, dim]
            negatives: Negative embeddings [batch, num_neg, dim] (optional)

        Returns:
            (pos_sim, neg_sim): Positive and negative similarities
        """
        # Positive similarities
        pos_sim = np.sum(anchors * positives, axis=-1)  # [batch]

        # Negative similarities (if provided)
        if negatives is not None:
            # anchors: [batch, dim] → [batch, 1, dim]
            anchors_exp = anchors[:, np.newaxis, :]
            # negatives: [batch, num_neg, dim]
            neg_sim = np.sum(anchors_exp * negatives, axis=-1)  # [batch, num_neg]
        else:
            neg_sim = None

        return pos_sim, neg_sim

    def info_nce_loss(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray
    ) -> tuple[float, float]:
        """
        Compute InfoNCE contrastive loss.

        L = -log(exp(sim(a,p)/τ) / Σ exp(sim(a,n)/τ))

        Args:
            anchors: Anchor embeddings [batch, dim]
            positives: Positive samples [batch, dim]
            negatives: Negative samples [batch, num_neg, dim]

        Returns:
            (loss, accuracy): InfoNCE loss and retrieval accuracy
        """
        batch_size = anchors.shape[0]
        temperature = self.temperature

        # Compute similarities
        pos_sim, neg_sim = self.compute_similarity(anchors, positives, negatives)

        # Scale by temperature
        pos_logits = pos_sim / temperature  # [batch]
        neg_logits = neg_sim / temperature  # [batch, num_neg]

        # Concatenate for softmax: positive is index 0
        all_logits = np.concatenate([
            pos_logits[:, np.newaxis],  # [batch, 1]
            neg_logits                   # [batch, num_neg]
        ], axis=1)  # [batch, 1 + num_neg]

        # Softmax and cross-entropy
        max_logits = np.max(all_logits, axis=1, keepdims=True)
        exp_logits = np.exp(all_logits - max_logits)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        log_probs = all_logits - max_logits - np.log(sum_exp)

        # Loss is negative log prob of positive (index 0)
        loss = -np.mean(log_probs[:, 0])

        # Accuracy: how often positive has highest similarity
        predictions = np.argmax(all_logits, axis=1)
        accuracy = np.mean(predictions == 0)

        return float(loss), float(accuracy)

    def mine_hard_negatives(
        self,
        anchors: np.ndarray,
        candidates: np.ndarray,
        positive_mask: np.ndarray,
        num_negatives: int = 16
    ) -> np.ndarray:
        """
        Mine hard negatives using semi-hard mining.

        Semi-hard negatives satisfy: sim(a, p) > sim(a, n) > sim(a, p) - margin

        Args:
            anchors: Anchor embeddings [batch, dim]
            candidates: Candidate pool [pool_size, dim]
            positive_mask: Boolean mask for positives [batch, pool_size]
            num_negatives: Number of negatives to mine

        Returns:
            Hard negative embeddings [batch, num_negatives, dim]
        """
        batch_size = anchors.shape[0]
        pool_size = candidates.shape[0]

        # Compute all similarities
        sims = anchors @ candidates.T  # [batch, pool_size]

        # Mask out positives with -inf
        sims_masked = np.where(positive_mask, -np.inf, sims)

        # Get positive similarities (max among positives)
        pos_sims = np.where(positive_mask, sims, -np.inf)
        pos_sim = np.max(pos_sims, axis=1, keepdims=True)  # [batch, 1]

        # Semi-hard mining: neg_sim < pos_sim but close
        margin = self.config.semi_hard_margin
        semi_hard_mask = (sims_masked > pos_sim - margin) & (sims_masked < pos_sim)

        # Fall back to hardest if no semi-hard
        has_semi_hard = np.any(semi_hard_mask, axis=1)

        hard_negatives = []
        for i in range(batch_size):
            if has_semi_hard[i]:
                # Sample from semi-hard
                semi_hard_indices = np.where(semi_hard_mask[i])[0]
                if len(semi_hard_indices) >= num_negatives:
                    selected = np.random.choice(semi_hard_indices, num_negatives, replace=False)
                else:
                    # Pad with hardest
                    remaining = num_negatives - len(semi_hard_indices)
                    other_indices = np.where(~semi_hard_mask[i] & ~positive_mask[i])[0]
                    other_sims = sims_masked[i, other_indices]
                    hardest = other_indices[np.argsort(other_sims)[-remaining:]]
                    selected = np.concatenate([semi_hard_indices, hardest])
            else:
                # Use hardest negatives
                neg_indices = np.where(~positive_mask[i])[0]
                neg_sims = sims_masked[i, neg_indices]
                hardest = neg_indices[np.argsort(neg_sims)[-num_negatives:]]
                selected = hardest

            hard_negatives.append(candidates[selected])

        return np.array(hard_negatives)  # [batch, num_negatives, dim]

    def temporal_contrastive_loss(
        self,
        sequence: np.ndarray,
        window_size: int = 3
    ) -> float:
        """
        Temporal contrastive loss for episode ordering.

        Encourages nearby episodes to have similar representations.

        Args:
            sequence: Ordered sequence of embeddings [seq_len, dim]
            window_size: Context window for positive pairs

        Returns:
            Temporal contrastive loss
        """
        seq_len = sequence.shape[0]
        if seq_len < window_size + 1:
            return 0.0

        total_loss = 0.0
        count = 0

        for i in range(seq_len):
            anchor = sequence[i:i+1]  # [1, dim]

            # Positives: within window
            pos_start = max(0, i - window_size)
            pos_end = min(seq_len, i + window_size + 1)
            pos_indices = [j for j in range(pos_start, pos_end) if j != i]

            if not pos_indices:
                continue

            # Sample one positive
            pos_idx = np.random.choice(pos_indices)
            positive = sequence[pos_idx:pos_idx+1]

            # Negatives: outside window
            neg_indices = [j for j in range(seq_len) if j < pos_start or j >= pos_end]
            if len(neg_indices) < 4:
                continue

            num_neg = min(8, len(neg_indices))
            neg_sample = np.random.choice(neg_indices, num_neg, replace=False)
            negatives = sequence[neg_sample][np.newaxis, :]  # [1, num_neg, dim]

            loss, _ = self.info_nce_loss(anchor, positive, negatives)
            total_loss += loss
            count += 1

        return total_loss / count if count > 0 else 0.0

    def update(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray,
        temporal_sequence: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Perform one gradient update step.

        Uses Adam optimizer with computed gradients.

        Args:
            anchors: Frozen anchor embeddings [batch, input_dim]
            positives: Frozen positive embeddings [batch, input_dim]
            negatives: Frozen negative embeddings [batch, num_neg, input_dim]
            temporal_sequence: Optional temporal sequence for ordering loss

        Returns:
            Dictionary of loss components
        """
        self._t += 1
        batch_size = anchors.shape[0]

        # Forward pass
        anchor_proj = self.forward(anchors, training=True)
        pos_proj = self.forward(positives, training=True)

        # Project negatives
        neg_shape = negatives.shape
        neg_flat = negatives.reshape(-1, neg_shape[-1])
        neg_proj_flat = self.forward(neg_flat, training=True)
        neg_proj = neg_proj_flat.reshape(batch_size, -1, self.config.output_dim)

        # Compute InfoNCE loss
        contrastive_loss, accuracy = self.info_nce_loss(anchor_proj, pos_proj, neg_proj)

        # Temporal loss (optional)
        temporal_loss = 0.0
        if temporal_sequence is not None:
            seq_proj = self.forward(temporal_sequence, training=True)
            temporal_loss = self.temporal_contrastive_loss(seq_proj)

        # Combined loss
        total_loss = (
            self.config.semantic_weight * contrastive_loss +
            self.config.temporal_weight * temporal_loss
        )

        # Compute gradients (numerical for now - can upgrade to autograd)
        self._compute_gradients_numerical(
            anchors, positives, negatives,
            contrastive_loss, temporal_loss
        )

        # Adam update
        self._adam_update()

        # Update temperature if learnable
        if self.config.temperature_learnable:
            temp_grad = self._estimate_temperature_gradient(
                anchor_proj, pos_proj, neg_proj
            )
            self._log_temperature -= self.config.learning_rate * temp_grad

        # Update stats
        self.stats.total_updates += 1
        self.stats.total_samples += batch_size
        self.stats.total_loss += total_loss
        self.stats.contrastive_loss = contrastive_loss
        self.stats.temporal_loss = temporal_loss
        self.stats.accuracy = accuracy
        self.stats.temperature = self.temperature
        self.stats.last_updated = datetime.now()

        return {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "temporal_loss": temporal_loss,
            "accuracy": accuracy,
            "temperature": self.temperature,
        }

    def _compute_gradients_numerical(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray,
        contrastive_loss: float,
        temporal_loss: float,
        eps: float = 1e-5
    ) -> None:
        """
        Compute gradients numerically (finite differences).

        Note: This is for correctness - production would use autograd.
        """
        # For efficiency, only compute gradients for W1, W2
        # (Most important parameters)

        # Gradient for W1
        grad_W1 = np.zeros_like(self.W1)
        for i in range(min(10, self.W1.shape[0])):  # Sample subset
            for j in range(min(10, self.W1.shape[1])):
                self.W1[i, j] += eps
                loss_plus = self._eval_loss(anchors, positives, negatives)
                self.W1[i, j] -= 2 * eps
                loss_minus = self._eval_loss(anchors, positives, negatives)
                self.W1[i, j] += eps
                grad_W1[i, j] = (loss_plus - loss_minus) / (2 * eps)

        # Scale gradient (since we only sampled)
        scale = (self.W1.shape[0] * self.W1.shape[1]) / 100
        self._grads["W1"] = grad_W1 * scale

        # Gradient for W2 (similar sampling)
        grad_W2 = np.zeros_like(self.W2)
        for i in range(min(10, self.W2.shape[0])):
            for j in range(min(10, self.W2.shape[1])):
                self.W2[i, j] += eps
                loss_plus = self._eval_loss(anchors, positives, negatives)
                self.W2[i, j] -= 2 * eps
                loss_minus = self._eval_loss(anchors, positives, negatives)
                self.W2[i, j] += eps
                grad_W2[i, j] = (loss_plus - loss_minus) / (2 * eps)

        scale = (self.W2.shape[0] * self.W2.shape[1]) / 100
        self._grads["W2"] = grad_W2 * scale

        # Bias gradients (full computation - smaller)
        grad_b1 = np.zeros_like(self.b1)
        for j in range(self.b1.shape[0]):
            self.b1[j] += eps
            loss_plus = self._eval_loss(anchors, positives, negatives)
            self.b1[j] -= 2 * eps
            loss_minus = self._eval_loss(anchors, positives, negatives)
            self.b1[j] += eps
            grad_b1[j] = (loss_plus - loss_minus) / (2 * eps)
        self._grads["b1"] = grad_b1

        grad_b2 = np.zeros_like(self.b2)
        for j in range(self.b2.shape[0]):
            self.b2[j] += eps
            loss_plus = self._eval_loss(anchors, positives, negatives)
            self.b2[j] -= 2 * eps
            loss_minus = self._eval_loss(anchors, positives, negatives)
            self.b2[j] += eps
            grad_b2[j] = (loss_plus - loss_minus) / (2 * eps)
        self._grads["b2"] = grad_b2

    def _eval_loss(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray
    ) -> float:
        """Evaluate loss with current weights."""
        batch_size = anchors.shape[0]
        anchor_proj = self.forward(anchors, training=False)
        pos_proj = self.forward(positives, training=False)

        neg_shape = negatives.shape
        neg_flat = negatives.reshape(-1, neg_shape[-1])
        neg_proj_flat = self.forward(neg_flat, training=False)
        neg_proj = neg_proj_flat.reshape(batch_size, -1, self.config.output_dim)

        loss, _ = self.info_nce_loss(anchor_proj, pos_proj, neg_proj)
        return loss

    def _adam_update(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        """Apply Adam optimizer update."""
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        for name, param in [("W1", self.W1), ("W2", self.W2), ("b1", self.b1), ("b2", self.b2)]:
            if name not in self._grads:
                continue

            grad = self._grads[name]

            # Add weight decay
            if "W" in name:
                grad = grad + wd * param

            # Initialize moments
            if name not in self._m:
                self._m[name] = np.zeros_like(param)
                self._v[name] = np.zeros_like(param)

            # Update moments
            self._m[name] = beta1 * self._m[name] + (1 - beta1) * grad
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * (grad ** 2)

            # Bias correction
            m_hat = self._m[name] / (1 - beta1 ** self._t)
            v_hat = self._v[name] / (1 - beta2 ** self._t)

            # Update parameter
            if name == "W1":
                self.W1 -= lr * m_hat / (np.sqrt(v_hat) + eps)
            elif name == "W2":
                self.W2 -= lr * m_hat / (np.sqrt(v_hat) + eps)
            elif name == "b1":
                self.b1 -= lr * m_hat / (np.sqrt(v_hat) + eps)
            elif name == "b2":
                self.b2 -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def _estimate_temperature_gradient(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray,
        eps: float = 1e-5
    ) -> float:
        """Estimate gradient w.r.t. log temperature."""
        # Save current
        orig = self._log_temperature

        # Plus
        self._log_temperature = orig + eps
        loss_plus, _ = self.info_nce_loss(anchors, positives, negatives)

        # Minus
        self._log_temperature = orig - eps
        loss_minus, _ = self.info_nce_loss(anchors, positives, negatives)

        # Restore
        self._log_temperature = orig

        return (loss_plus - loss_minus) / (2 * eps)

    def save_weights(self) -> dict[str, Any]:
        """Save adapter weights to dictionary."""
        weights = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "gamma1": self.gamma1.tolist(),
            "beta1": self.beta1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "log_temperature": self._log_temperature,
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "output_dim": self.config.output_dim,
                "num_layers": self.config.num_layers,
            },
            "stats": self.stats.to_dict(),
        }
        if self.W3 is not None:
            weights["W3"] = self.W3.tolist()
            weights["b3"] = self.b3.tolist()
        return weights

    def load_weights(self, weights: dict[str, Any]) -> None:
        """Load adapter weights from dictionary."""
        self.W1 = np.array(weights["W1"], dtype=np.float32)
        self.b1 = np.array(weights["b1"], dtype=np.float32)
        self.gamma1 = np.array(weights["gamma1"], dtype=np.float32)
        self.beta1 = np.array(weights["beta1"], dtype=np.float32)
        self.W2 = np.array(weights["W2"], dtype=np.float32)
        self.b2 = np.array(weights["b2"], dtype=np.float32)
        self._log_temperature = weights["log_temperature"]
        if "W3" in weights:
            self.W3 = np.array(weights["W3"], dtype=np.float32)
            self.b3 = np.array(weights["b3"], dtype=np.float32)

    def get_health_status(self) -> dict[str, Any]:
        """Get adapter health status."""
        return {
            "mode": self.mode.value,
            "temperature": self.temperature,
            "total_updates": self.stats.total_updates,
            "avg_loss": self.stats.avg_loss,
            "accuracy": self.stats.accuracy,
            "healthy": self.stats.accuracy > 0.5 if self.stats.total_updates > 10 else True,
        }


# Factory function
def create_contrastive_adapter(
    input_dim: int = 1024,
    output_dim: int = 256,
    **kwargs
) -> ContrastiveAdapter:
    """
    Create contrastive adapter with specified dimensions.

    Args:
        input_dim: Dimension of frozen embeddings
        output_dim: Dimension of projected embeddings
        **kwargs: Additional config options

    Returns:
        Configured ContrastiveAdapter
    """
    config = ContrastiveConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs
    )
    return ContrastiveAdapter(config)
