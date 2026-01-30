"""
Unified Attention System combining Capsule Routing and Transformer Attention.

Biological Basis:
- Capsule routing: Part-whole relationships, configuration agreement
- Transformer attention: Associative content-addressing, parallel queries
- Fusion enables both compositional and associative retrieval

Mathematical Formulation:
    Attention(Q, K, V) = softmax(A_capsule + A_transformer) * V

    A_capsule[i,j] = exp(-||pose_i @ W_ij - pose_j||_F / tau)
    A_transformer[i,j] = (q_i @ k_j) / sqrt(d_k)
    A_combined = alpha * A_capsule + (1 - alpha) * A_transformer

References:
- Sabour et al. (2017): Dynamic routing between capsules
- Vaswani et al. (2017): Attention is all you need
- Kosiorek et al. (2019): Stacked capsule autoencoders
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UnifiedAttentionConfig:
    """Configuration for unified attention system.

    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        pose_dim: Pose matrix dimension
        capsule_weight: Weight for capsule attention (alpha)
        temperature: Softmax temperature
        use_ff_learning: Use Forward-Forward for learning
        attention_dropout: Dropout rate
    """
    embed_dim: int = 1024
    num_heads: int = 8
    head_dim: int = 64
    pose_dim: int = 4
    capsule_weight: float = 0.5
    adaptive_alpha: bool = False  # If True, alpha adjusts based on agreement
    temperature: float = 1.0
    use_ff_learning: bool = True
    attention_dropout: float = 0.1


class UnifiedAttentionHead:
    """
    Single attention head combining capsule and transformer attention.

    Key innovation: Fuses pose-based agreement (capsules) with
    content-based similarity (transformers) for richer retrieval.
    """

    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        pose_dim: int = 4,
        capsule_weight: float = 0.5,
        adaptive_alpha: bool = False,
        temperature: float = 1.0,
    ):
        """
        Initialize unified attention head.

        Args:
            embed_dim: Input embedding dimension
            head_dim: Dimension of this head's output
            pose_dim: Pose matrix dimension
            capsule_weight: Weight for capsule attention (0-1)
            adaptive_alpha: If True, alpha adjusts based on agreement
            temperature: Softmax temperature
        """
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.pose_dim = pose_dim
        self.capsule_weight = capsule_weight
        self.adaptive_alpha = adaptive_alpha
        self.temperature = temperature

        # Transformer projections
        self.W_q = np.random.randn(head_dim, embed_dim).astype(np.float32) * 0.02
        self.W_k = np.random.randn(head_dim, embed_dim).astype(np.float32) * 0.02
        self.W_v = np.random.randn(head_dim, embed_dim).astype(np.float32) * 0.02

        # Pose extraction
        self.W_pose = np.random.randn(
            pose_dim * pose_dim, embed_dim
        ).astype(np.float32) * 0.02

        # Pose transformation matrices for capsule attention
        # W_ij transforms pose_j for comparison with pose_i
        self.W_transform = np.random.randn(
            pose_dim, pose_dim
        ).astype(np.float32) * 0.1
        self.W_transform = self.W_transform + np.eye(pose_dim)  # Near identity

        # Statistics
        self._attention_history: list = []

        logger.debug(f"UnifiedAttentionHead: embed_dim={embed_dim}, head_dim={head_dim}")

    def extract_pose(self, embedding: np.ndarray) -> np.ndarray:
        """Extract pose matrix from embedding."""
        pose_flat = self.W_pose @ embedding
        pose = pose_flat.reshape(self.pose_dim, self.pose_dim)
        return pose

    def compute_capsule_attention(
        self,
        query_poses: np.ndarray,
        key_poses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute capsule-style attention based on pose agreement.

        A_capsule[i,j] = exp(-||T @ pose_j - pose_i||_F / tau)

        Args:
            query_poses: Query poses [n_queries, pose_dim, pose_dim]
            key_poses: Key poses [n_keys, pose_dim, pose_dim]

        Returns:
            Attention scores [n_queries, n_keys]
        """
        n_queries = query_poses.shape[0]
        n_keys = key_poses.shape[0]

        attention = np.zeros((n_queries, n_keys), dtype=np.float32)

        for i in range(n_queries):
            for j in range(n_keys):
                # Transform key pose
                transformed = self.W_transform @ key_poses[j]

                # Frobenius distance
                diff = query_poses[i] - transformed
                distance = np.linalg.norm(diff, 'fro')

                # Convert to similarity (smaller distance = higher attention)
                attention[i, j] = np.exp(-distance / self.temperature)

        return attention

    def compute_transformer_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
    ) -> np.ndarray:
        """
        Compute standard transformer scaled dot-product attention.

        A_transformer[i,j] = (q_i @ k_j) / sqrt(d_k)

        Args:
            query: Query vectors [n_queries, embed_dim]
            key: Key vectors [n_keys, embed_dim]

        Returns:
            Attention scores [n_queries, n_keys]
        """
        # Project
        Q = query @ self.W_q.T  # [n_queries, head_dim]
        K = key @ self.W_k.T    # [n_keys, head_dim]

        # Scaled dot product
        scale = np.sqrt(self.head_dim)
        attention = (Q @ K.T) / scale

        return attention

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        poses: tuple[np.ndarray, np.ndarray] | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute unified attention.

        Args:
            query: Query embeddings [n_queries, embed_dim]
            key: Key embeddings [n_keys, embed_dim]
            value: Value embeddings [n_keys, embed_dim]
            poses: Optional (query_poses, key_poses) for capsule attention
            mask: Optional attention mask [n_queries, n_keys]

        Returns:
            Tuple of (output, attention_weights)
        """
        query = np.atleast_2d(query)
        key = np.atleast_2d(key)
        value = np.atleast_2d(value)

        # Compute transformer attention
        A_transformer = self.compute_transformer_attention(query, key)

        # Compute capsule attention if poses provided
        if poses is not None:
            query_poses, key_poses = poses
            A_capsule = self.compute_capsule_attention(query_poses, key_poses)
        else:
            # Extract poses from embeddings
            query_poses = np.stack([self.extract_pose(q) for q in query])
            key_poses = np.stack([self.extract_pose(k) for k in key])
            A_capsule = self.compute_capsule_attention(query_poses, key_poses)

        # ATOM-P3-37: Dynamic alpha based on agreement
        alpha = self.capsule_weight
        if self.adaptive_alpha:
            # Mean agreement score from capsule attention
            agreement = np.mean(A_capsule)
            alpha = 1.0 / (1.0 + np.exp(-4.0 * (agreement - 0.5)))

        # Fuse attention scores
        A_combined = (
            alpha * A_capsule +
            (1 - alpha) * A_transformer
        )

        # Apply mask if provided
        if mask is not None:
            A_combined = A_combined * mask + (1 - mask) * (-1e9)

        # Softmax
        A_weights = self._softmax(A_combined / self.temperature)

        # Compute output
        V = value @ self.W_v.T  # [n_keys, head_dim]
        output = A_weights @ V  # [n_queries, head_dim]

        return output, A_weights

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)

    def learn_fusion_weight(
        self,
        utility: float,
        contributions: tuple[float, float],
    ) -> None:
        """
        Learn optimal fusion weight based on contribution utility.

        Args:
            utility: Overall utility of attention output
            contributions: (capsule_contribution, transformer_contribution)
        """
        capsule_contrib, transformer_contrib = contributions

        # Move weight toward more useful component
        if capsule_contrib > transformer_contrib:
            self.capsule_weight = min(0.9, self.capsule_weight + 0.01 * utility)
        else:
            self.capsule_weight = max(0.1, self.capsule_weight - 0.01 * utility)


class UnifiedAttentionSystem:
    """
    Multi-head unified attention system.

    Combines multiple UnifiedAttentionHeads for rich representation
    of both compositional (capsule) and associative (transformer) patterns.
    """

    def __init__(self, config: UnifiedAttentionConfig | None = None):
        """
        Initialize unified attention system.

        Args:
            config: Attention configuration
        """
        self.config = config or UnifiedAttentionConfig()

        # Create attention heads
        self.heads: list[UnifiedAttentionHead] = []
        for _ in range(self.config.num_heads):
            head = UnifiedAttentionHead(
                embed_dim=self.config.embed_dim,
                head_dim=self.config.head_dim,
                pose_dim=self.config.pose_dim,
                capsule_weight=self.config.capsule_weight,
                adaptive_alpha=self.config.adaptive_alpha,
                temperature=self.config.temperature,
            )
            self.heads.append(head)

        # Output projection
        total_head_dim = self.config.num_heads * self.config.head_dim
        self.W_out = np.random.randn(
            self.config.embed_dim, total_head_dim
        ).astype(np.float32) * 0.02

        logger.info(
            f"UnifiedAttentionSystem: {self.config.num_heads} heads, "
            f"capsule_weight={self.config.capsule_weight}"
        )

    def attend(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        poses: tuple[np.ndarray, np.ndarray] | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Multi-head unified attention.

        Args:
            query: Query embeddings [n_queries, embed_dim]
            keys: Key embeddings [n_keys, embed_dim]
            values: Value embeddings [n_keys, embed_dim]
            poses: Optional (query_poses, key_poses)
            mask: Optional attention mask

        Returns:
            Attended output [n_queries, embed_dim]
        """
        head_outputs = []

        for head in self.heads:
            output, _ = head.forward(query, keys, values, poses, mask)
            head_outputs.append(output)

        # Concatenate heads
        concat = np.concatenate(head_outputs, axis=-1)  # [n_queries, total_head_dim]

        # Project to output dimension
        output = concat @ self.W_out.T  # [n_queries, embed_dim]

        return output

    def extract_poses(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Extract pose matrices from embeddings.

        Args:
            embeddings: [n, embed_dim]

        Returns:
            Poses [n, pose_dim, pose_dim]
        """
        embeddings = np.atleast_2d(embeddings)
        poses = np.stack([self.heads[0].extract_pose(e) for e in embeddings])
        return poses

    def get_attention_stats(self) -> dict:
        """Get attention statistics."""
        capsule_weights = [h.capsule_weight for h in self.heads]
        return {
            "num_heads": len(self.heads),
            "mean_capsule_weight": float(np.mean(capsule_weights)),
            "capsule_weights": capsule_weights,
            "temperature": self.config.temperature,
        }


__all__ = [
    "UnifiedAttentionConfig",
    "UnifiedAttentionHead",
    "UnifiedAttentionSystem",
]
