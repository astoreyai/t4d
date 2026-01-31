"""
Online Embedding Adapter for World Weaver.

Phase 2C: Low-Rank Adaptation from Retrieval Feedback.

This module implements a numpy-only LoRA-style adapter that learns from
retrieval outcomes without requiring PyTorch. It provides:

1. Low-rank residual adaptation: W' = W + scale * A @ B
2. InfoNCE-style contrastive training from feedback signals
3. Efficient online learning with bounded memory usage

Biological Basis:
- Prefrontal cortex modulates sensory representations for task relevance
- Attention mechanisms gate information flow based on context
- Low-rank adaptation mirrors synaptic plasticity in projection neurons
- Contrastive learning aligns with Hebbian "fire together, wire together"

Neural Plausibility:
- Local learning: each adapter weight updated based on local gradient
- No backprop through multiple layers required
- Can be viewed as single-layer Hebbian update with supervision

Key Differences from embedding/lora_adapter.py:
- Pure numpy implementation (no PyTorch dependency)
- Designed for encoding module integration
- Simpler API focused on online adaptation
- Direct integration with RetrievalFeedbackCollector

References:
- Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models
- Oord et al. (2018): Representation Learning with Contrastive Predictive Coding
- Hinton Analysis: Task-specific adaptation without full fine-tuning

Usage:
    ```python
    from t4dm.encoding.online_adapter import OnlineEmbeddingAdapter, AdapterConfig

    # Create adapter
    config = AdapterConfig(base_dim=1024, adapter_rank=32)
    adapter = OnlineEmbeddingAdapter(config)

    # Adapt embedding
    adapted = adapter.adapt(embedding)

    # Train from feedback
    loss = adapter.train_step(
        query_emb=query,
        positive_embs=[good_result1, good_result2],
        negative_embs=[bad_result1, bad_result2]
    )
    ```
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Security/performance limits
MAX_BATCH_SIZE = 256
MAX_POSITIVES = 32
MAX_NEGATIVES = 64
MAX_TRAINING_HISTORY = 10000


@dataclass
class AdapterConfig:
    """
    Configuration for online embedding adapter.

    Attributes:
        base_dim: Base embedding dimension (e.g., 1024 for BGE-M3)
        adapter_rank: LoRA rank - lower = fewer params, higher = more capacity
        scale: Scaling factor for adaptation (alpha/rank in LoRA paper)
        learning_rate: Step size for gradient updates
        temperature: Temperature for InfoNCE softmax
        weight_decay: L2 regularization strength
        gradient_clip: Maximum gradient norm
        momentum: Momentum coefficient for SGD with momentum
        use_bias: Whether to include bias terms in adapter
        normalize_output: Whether to L2-normalize adapted embeddings
    """
    base_dim: int = 1024
    adapter_rank: int = 32
    scale: float = 0.1
    learning_rate: float = 0.001
    temperature: float = 0.07
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    momentum: float = 0.9
    use_bias: bool = False
    normalize_output: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.base_dim <= 0:
            raise ValueError("base_dim must be positive")
        if self.adapter_rank <= 0:
            raise ValueError("adapter_rank must be positive")
        if self.adapter_rank > self.base_dim:
            raise ValueError("adapter_rank cannot exceed base_dim")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be in (0, 1)")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "base_dim": self.base_dim,
            "adapter_rank": self.adapter_rank,
            "scale": self.scale,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "momentum": self.momentum,
            "use_bias": self.use_bias,
            "normalize_output": self.normalize_output,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AdapterConfig:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AdapterState:
    """Serializable adapter state for persistence."""
    step_count: int = 0
    training_losses: list[float] = field(default_factory=list)
    mean_positive_sim: float = 0.0
    mean_negative_sim: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "step_count": self.step_count,
            "training_losses": self.training_losses[-100:],  # Keep last 100
            "mean_positive_sim": self.mean_positive_sim,
            "mean_negative_sim": self.mean_negative_sim,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> AdapterState:
        """Create from dict."""
        return cls(
            step_count=data.get("step_count", 0),
            training_losses=data.get("training_losses", []),
            mean_positive_sim=data.get("mean_positive_sim", 0.0),
            mean_negative_sim=data.get("mean_negative_sim", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
        )


class OnlineEmbeddingAdapter:
    """
    Train LoRA adapters from retrieval feedback.

    Implements Low-Rank Adaptation (LoRA) for embedding transformation:
        adapted = embedding + scale * (embedding @ A @ B)

    where A is [base_dim, rank] and B is [rank, base_dim].

    This allows task-specific tuning with far fewer parameters than
    full fine-tuning: 2 * rank * dim vs dim^2.

    The adapter learns from contrastive signals:
    - Positive examples: embeddings that led to good retrieval outcomes
    - Negative examples: embeddings that led to poor outcomes

    Training uses InfoNCE loss to pull positives closer and push negatives away.
    """

    def __init__(
        self,
        config: AdapterConfig | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize online embedding adapter.

        Args:
            config: Adapter configuration
            random_seed: Seed for reproducible initialization
        """
        self.config = config or AdapterConfig()
        self._rng = np.random.default_rng(random_seed)
        self.state = AdapterState()

        # LoRA matrices: W' = W + scale * A @ B
        # A: down-projection (base_dim -> rank)
        # B: up-projection (rank -> base_dim)
        # Initialize A with small random, B with zeros (starts as identity)
        self.lora_A = self._rng.normal(
            0, 0.01, size=(self.config.base_dim, self.config.adapter_rank)
        ).astype(np.float32)
        self.lora_B = np.zeros(
            (self.config.adapter_rank, self.config.base_dim),
            dtype=np.float32
        )

        # Optional bias terms
        if self.config.use_bias:
            self.bias_A = np.zeros(self.config.adapter_rank, dtype=np.float32)
            self.bias_B = np.zeros(self.config.base_dim, dtype=np.float32)
        else:
            self.bias_A = None
            self.bias_B = None

        # Momentum buffers for SGD with momentum
        self._momentum_A = np.zeros_like(self.lora_A)
        self._momentum_B = np.zeros_like(self.lora_B)
        if self.config.use_bias:
            self._momentum_bias_A = np.zeros_like(self.bias_A)
            self._momentum_bias_B = np.zeros_like(self.bias_B)

        logger.info(
            f"OnlineEmbeddingAdapter initialized: "
            f"dim={self.config.base_dim}, rank={self.config.adapter_rank}, "
            f"params={self.num_parameters()}"
        )

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        count = self.lora_A.size + self.lora_B.size
        if self.config.use_bias:
            count += self.bias_A.size + self.bias_B.size
        return count

    def adapt(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply learned LoRA adaptation to embedding.

        Computes: adapted = embedding + scale * (embedding @ A @ B)

        Args:
            embedding: Input embedding [dim] or [batch, dim]

        Returns:
            Adapted embedding with same shape
        """
        embedding = np.atleast_2d(embedding).astype(np.float32)
        was_1d = embedding.ndim == 1 or embedding.shape[0] == 1

        # Validate dimension
        if embedding.shape[-1] != self.config.base_dim:
            raise ValueError(
                f"Embedding dimension {embedding.shape[-1]} does not match "
                f"configured base_dim {self.config.base_dim}"
            )

        # LoRA: delta = scale * embedding @ A @ B
        hidden = embedding @ self.lora_A  # [batch, rank]
        if self.bias_A is not None:
            hidden = hidden + self.bias_A

        delta = hidden @ self.lora_B  # [batch, base_dim]
        if self.bias_B is not None:
            delta = delta + self.bias_B

        # Apply residual adaptation
        adapted = embedding + self.config.scale * delta

        # Optional normalization
        if self.config.normalize_output:
            norm = np.linalg.norm(adapted, axis=-1, keepdims=True)
            adapted = adapted / (norm + 1e-8)

        # Restore original shape
        if was_1d:
            adapted = adapted.squeeze(0)

        return adapted

    def train_step(
        self,
        query_emb: np.ndarray,
        positive_embs: list[np.ndarray],
        negative_embs: list[np.ndarray],
    ) -> float:
        """
        Contrastive training step from retrieval feedback.

        Uses InfoNCE-style loss to:
        - Pull query closer to positive embeddings (relevant results)
        - Push query away from negative embeddings (irrelevant results)

        Loss = -log( exp(sim(q, p+) / tau) / sum(exp(sim(q, p) / tau)) )

        where p+ are positives and the sum is over all positives and negatives.

        Args:
            query_emb: Query embedding that was used [dim]
            positive_embs: Embeddings of relevant results
            negative_embs: Embeddings of irrelevant results

        Returns:
            Loss value for this step
        """
        if not positive_embs:
            logger.debug("No positive examples, skipping training step")
            return 0.0

        # Validate and prepare inputs
        query_emb = np.atleast_2d(query_emb).astype(np.float32)
        if query_emb.shape[-1] != self.config.base_dim:
            raise ValueError(f"Query dimension mismatch: {query_emb.shape[-1]} vs {self.config.base_dim}")

        # Limit batch sizes for memory safety
        positive_embs = positive_embs[:MAX_POSITIVES]
        negative_embs = negative_embs[:MAX_NEGATIVES]

        positives = np.array(positive_embs, dtype=np.float32)
        if positives.shape[-1] != self.config.base_dim:
            raise ValueError(f"Positive dimension mismatch: {positives.shape[-1]} vs {self.config.base_dim}")

        # Adapt all embeddings through LoRA
        query_adapted = self.adapt(query_emb)  # [1, dim]
        positives_adapted = np.array([self.adapt(p) for p in positives])  # [n_pos, dim]

        if negative_embs:
            negatives = np.array(negative_embs, dtype=np.float32)
            negatives_adapted = np.array([self.adapt(n) for n in negatives])  # [n_neg, dim]
        else:
            negatives_adapted = np.array([]).reshape(0, self.config.base_dim)

        # Compute similarities (cosine via dot product of normalized vectors)
        query_norm = query_adapted / (np.linalg.norm(query_adapted, axis=-1, keepdims=True) + 1e-8)
        pos_norms = positives_adapted / (np.linalg.norm(positives_adapted, axis=-1, keepdims=True) + 1e-8)

        pos_sims = np.dot(pos_norms, query_norm.T).flatten()  # [n_pos]

        if negatives_adapted.size > 0:
            neg_norms = negatives_adapted / (np.linalg.norm(negatives_adapted, axis=-1, keepdims=True) + 1e-8)
            neg_sims = np.dot(neg_norms, query_norm.T).flatten()  # [n_neg]
            all_sims = np.concatenate([pos_sims, neg_sims])
        else:
            neg_sims = np.array([])
            all_sims = pos_sims

        # InfoNCE loss
        # For each positive, compute log-softmax against all examples
        scaled_sims = all_sims / self.config.temperature
        log_sum_exp = np.log(np.sum(np.exp(scaled_sims - np.max(scaled_sims))) + 1e-8) + np.max(scaled_sims)

        # Average over positives
        pos_scaled = pos_sims / self.config.temperature
        loss = -np.mean(pos_scaled) + log_sum_exp

        # Compute gradients via finite differences (simple but effective for small adapters)
        # For production, would use autograd or manual gradient computation
        grad_A, grad_B = self._compute_gradients(
            query_emb.flatten(), positives, negatives_adapted if negatives_adapted.size > 0 else None
        )

        # Apply weight decay
        grad_A = grad_A + self.config.weight_decay * self.lora_A
        grad_B = grad_B + self.config.weight_decay * self.lora_B

        # Gradient clipping
        grad_A_norm = np.linalg.norm(grad_A)
        grad_B_norm = np.linalg.norm(grad_B)
        if grad_A_norm > self.config.gradient_clip:
            grad_A = grad_A * (self.config.gradient_clip / grad_A_norm)
        if grad_B_norm > self.config.gradient_clip:
            grad_B = grad_B * (self.config.gradient_clip / grad_B_norm)

        # SGD with momentum
        self._momentum_A = self.config.momentum * self._momentum_A - self.config.learning_rate * grad_A
        self._momentum_B = self.config.momentum * self._momentum_B - self.config.learning_rate * grad_B

        self.lora_A = self.lora_A + self._momentum_A
        self.lora_B = self.lora_B + self._momentum_B

        # Update state
        self.state.step_count += 1
        self.state.training_losses.append(float(loss))
        if len(self.state.training_losses) > MAX_TRAINING_HISTORY:
            self.state.training_losses = self.state.training_losses[-MAX_TRAINING_HISTORY:]

        # Update running statistics
        alpha = 0.1
        mean_pos = float(np.mean(pos_sims)) if len(pos_sims) > 0 else 0.0
        mean_neg = float(np.mean(neg_sims)) if len(neg_sims) > 0 else 0.0
        self.state.mean_positive_sim = alpha * mean_pos + (1 - alpha) * self.state.mean_positive_sim
        self.state.mean_negative_sim = alpha * mean_neg + (1 - alpha) * self.state.mean_negative_sim
        self.state.updated_at = datetime.now()

        logger.debug(
            f"OnlineAdapter step {self.state.step_count}: "
            f"loss={loss:.4f}, pos_sim={mean_pos:.3f}, neg_sim={mean_neg:.3f}"
        )

        return float(loss)

    def _compute_gradients(
        self,
        query: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for LoRA matrices using contrastive loss.

        Uses analytical gradients for InfoNCE loss:
        - Gradient w.r.t. A and B pulls positives closer, pushes negatives away

        Args:
            query: Query embedding [dim]
            positives: Positive embeddings [n_pos, dim]
            negatives: Negative embeddings [n_neg, dim] or None

        Returns:
            Tuple of (grad_A, grad_B)
        """
        # Forward pass to get adapted embeddings
        query_2d = query.reshape(1, -1)
        query_adapted = self.adapt(query_2d).flatten()

        # Compute softmax weights over all examples
        all_examples = [positives]
        if negatives is not None and negatives.size > 0:
            all_examples.append(negatives)
        all_embs = np.vstack(all_examples)
        n_pos = len(positives)

        # Adapt all examples
        adapted_all = np.array([self.adapt(e) for e in all_embs])

        # Compute similarities
        query_norm = query_adapted / (np.linalg.norm(query_adapted) + 1e-8)
        all_norms = adapted_all / (np.linalg.norm(adapted_all, axis=-1, keepdims=True) + 1e-8)
        sims = np.dot(all_norms, query_norm)  # [n_all]

        # Softmax weights
        scaled = sims / self.config.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        softmax_weights = exp_scaled / (np.sum(exp_scaled) + 1e-8)

        # Gradient: for positives, we want to increase similarity
        # For negatives, we want to decrease similarity
        # InfoNCE gradient: d_loss/d_sim = softmax_weight - (1/n_pos if positive else 0)
        target_weights = np.zeros(len(all_embs))
        target_weights[:n_pos] = 1.0 / n_pos

        sim_gradients = softmax_weights - target_weights  # [n_all]

        # Chain rule through similarity to adaptation
        # d_sim/d_adapted_query = normalized(all_embs)
        # d_adapted/d_A and d_adapted/d_B

        # Gradient w.r.t query adaptation
        grad_query_adapted = np.sum(
            sim_gradients[:, None] * all_norms / self.config.temperature,
            axis=0
        )

        # Chain rule through LoRA
        # adapted = query + scale * query @ A @ B
        # d_adapted/d_A = scale * outer(query, B^T @ grad)
        # d_adapted/d_B = scale * outer(A^T @ query, grad)

        hidden = query @ self.lora_A  # [rank]
        grad_B = self.config.scale * np.outer(hidden, grad_query_adapted)
        # grad_hidden = grad_query_adapted @ lora_B.T gives shape (rank,)
        grad_A = self.config.scale * np.outer(query, grad_query_adapted @ self.lora_B.T)

        return grad_A.astype(np.float32), grad_B.astype(np.float32)

    def save(self, path: str | Path) -> str:
        """
        Save adapter weights and state.

        Args:
            path: Directory or file path to save to

        Returns:
            Path where saved
        """
        save_path = Path(path)

        # If path is a file, use its parent as directory
        if save_path.suffix:
            save_dir = save_path.parent
            base_name = save_path.stem
        else:
            save_dir = save_path
            base_name = "online_adapter"

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_path = save_dir / f"{base_name}_weights.npz"
        weight_data = {
            "lora_A": self.lora_A,
            "lora_B": self.lora_B,
            "momentum_A": self._momentum_A,
            "momentum_B": self._momentum_B,
        }
        if self.config.use_bias:
            weight_data["bias_A"] = self.bias_A
            weight_data["bias_B"] = self.bias_B
            weight_data["momentum_bias_A"] = self._momentum_bias_A
            weight_data["momentum_bias_B"] = self._momentum_bias_B

        np.savez(weights_path, **weight_data)

        # Save config and state
        state_path = save_dir / f"{base_name}_state.json"
        with open(state_path, "w") as f:
            json.dump({
                "config": self.config.to_dict(),
                "state": self.state.to_dict(),
            }, f, indent=2)

        logger.info(f"OnlineAdapter saved to {save_dir}")
        return str(save_dir)

    def load(self, path: str | Path) -> bool:
        """
        Load adapter weights and state.

        Args:
            path: Directory or file path to load from

        Returns:
            True if loaded successfully
        """
        load_path = Path(path)

        # Determine paths
        if load_path.suffix:
            load_dir = load_path.parent
            base_name = load_path.stem
        else:
            load_dir = load_path
            base_name = "online_adapter"

        weights_path = load_dir / f"{base_name}_weights.npz"
        state_path = load_dir / f"{base_name}_state.json"

        if not weights_path.exists() or not state_path.exists():
            logger.warning(f"Adapter files not found at {load_dir}")
            return False

        # Load weights
        try:
            data = np.load(weights_path)
            self.lora_A = data["lora_A"]
            self.lora_B = data["lora_B"]
            self._momentum_A = data["momentum_A"]
            self._momentum_B = data["momentum_B"]

            if "bias_A" in data and self.config.use_bias:
                self.bias_A = data["bias_A"]
                self.bias_B = data["bias_B"]
                self._momentum_bias_A = data["momentum_bias_A"]
                self._momentum_bias_B = data["momentum_bias_B"]
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return False

        # Load state
        try:
            with open(state_path) as f:
                data = json.load(f)

            # Update config if present
            if "config" in data:
                self.config = AdapterConfig.from_dict(data["config"])

            if "state" in data:
                self.state = AdapterState.from_dict(data["state"])
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

        logger.info(f"OnlineAdapter loaded from {load_dir}: step={self.state.step_count}")
        return True

    def reset(self) -> None:
        """Reset adapter to initial state."""
        # Reinitialize weights
        self.lora_A = self._rng.normal(
            0, 0.01, size=(self.config.base_dim, self.config.adapter_rank)
        ).astype(np.float32)
        self.lora_B = np.zeros(
            (self.config.adapter_rank, self.config.base_dim),
            dtype=np.float32
        )

        if self.config.use_bias:
            self.bias_A = np.zeros(self.config.adapter_rank, dtype=np.float32)
            self.bias_B = np.zeros(self.config.base_dim, dtype=np.float32)

        # Reset momentum
        self._momentum_A = np.zeros_like(self.lora_A)
        self._momentum_B = np.zeros_like(self.lora_B)
        if self.config.use_bias:
            self._momentum_bias_A = np.zeros_like(self.bias_A)
            self._momentum_bias_B = np.zeros_like(self.bias_B)

        # Reset state
        self.state = AdapterState()

        logger.info("OnlineAdapter reset to initial state")

    def get_stats(self) -> dict:
        """Get adapter statistics."""
        stats = {
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
            "num_parameters": self.num_parameters(),
            "parameter_efficiency": (
                self.num_parameters() / (self.config.base_dim ** 2) * 100
            ),
            "weight_norms": {
                "lora_A": float(np.linalg.norm(self.lora_A)),
                "lora_B": float(np.linalg.norm(self.lora_B)),
            },
        }

        if self.state.training_losses:
            recent = self.state.training_losses[-10:]
            stats["recent_avg_loss"] = sum(recent) / len(recent)
            stats["recent_loss_std"] = float(np.std(recent))

        # Separation metric: positive vs negative similarity gap
        if self.state.mean_positive_sim > 0 or self.state.mean_negative_sim > 0:
            stats["similarity_gap"] = self.state.mean_positive_sim - self.state.mean_negative_sim

        return stats

    def get_adaptation_magnitude(self) -> float:
        """
        Compute magnitude of current adaptation.

        Returns how much the adapter modifies embeddings on average.
        Useful for monitoring adaptation strength.

        Returns:
            Average L2 norm of adaptation delta
        """
        # Sample random unit vectors
        n_samples = 100
        samples = self._rng.standard_normal((n_samples, self.config.base_dim))
        samples = samples / np.linalg.norm(samples, axis=-1, keepdims=True)

        # Compute adaptation deltas
        deltas = []
        for sample in samples:
            adapted = self.adapt(sample)
            delta = np.linalg.norm(adapted - sample)
            deltas.append(delta)

        return float(np.mean(deltas))


# Factory function
def create_online_adapter(
    base_dim: int = 1024,
    adapter_rank: int = 32,
    learning_rate: float = 0.001,
    scale: float = 0.1,
    random_seed: int | None = None,
    **kwargs,
) -> OnlineEmbeddingAdapter:
    """
    Factory function for OnlineEmbeddingAdapter.

    Args:
        base_dim: Base embedding dimension
        adapter_rank: LoRA rank
        learning_rate: Learning rate
        scale: Adaptation scale
        random_seed: Random seed
        **kwargs: Additional config options

    Returns:
        Configured OnlineEmbeddingAdapter
    """
    config = AdapterConfig(
        base_dim=base_dim,
        adapter_rank=adapter_rank,
        learning_rate=learning_rate,
        scale=scale,
        **kwargs,
    )
    return OnlineEmbeddingAdapter(config, random_seed=random_seed)


__all__ = [
    "AdapterConfig",
    "AdapterState",
    "OnlineEmbeddingAdapter",
    "create_online_adapter",
]
