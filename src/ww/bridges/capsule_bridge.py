"""
P6.2: Bridge between CapsuleLayer and Memory Retrieval.

Uses capsule network representations to enhance memory retrieval through
part-whole compositional similarity. Capsules encode entities and their
configurations, enabling richer semantic matching beyond cosine similarity.

Biological Basis:
- Capsules model cortical microcolumns encoding entity configurations
- Routing-by-agreement models binding between neural populations
- Pose transformations capture viewpoint-invariant representations

Integration Flow:
    Query Embedding
         |
    [CapsuleLayer] → query_activations, query_poses
         |
    Memory Embeddings
         |
    [CapsuleLayer] → memory_activations, memory_poses
         |
    [CapsuleRetrievalBridge]
         |
    Capsule Agreement Score → Retrieval Boost
         |
    Enhanced Retrieval Scoring

References:
- Sabour et al. (2017): Dynamic Routing Between Capsules
- Hinton et al. (2018): Matrix capsules with EM routing
- Rae et al. (2016): Scaling Memory-Augmented Neural Networks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class CapsuleLayerProtocol(Protocol):
    """Protocol for capsule layer interface."""

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process embedding through capsule layer.

        Returns:
            Tuple of (activations, poses)
        """
        ...


@dataclass
class CapsuleBridgeConfig:
    """Configuration for Capsule-Retrieval bridge.

    Attributes:
        activation_weight: Weight for activation similarity
        pose_weight: Weight for pose agreement
        agreement_threshold: Minimum agreement for boost
        max_boost: Maximum retrieval score boost
        cache_size: Size of capsule representation cache
        batch_size: Batch size for processing
    """
    activation_weight: float = 0.4
    pose_weight: float = 0.6
    agreement_threshold: float = 0.3
    max_boost: float = 0.3
    cache_size: int = 1000
    batch_size: int = 64


@dataclass
class CapsuleBridgeState:
    """Current state of the bridge.

    Attributes:
        n_queries: Number of queries processed
        n_comparisons: Number of memory comparisons
        mean_boost: Running mean of boost values
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        timestamp: Last update time
    """
    n_queries: int = 0
    n_comparisons: int = 0
    mean_boost: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CapsuleRepresentation:
    """Cached capsule representation for an embedding.

    Attributes:
        activations: Capsule activation magnitudes
        poses: Capsule pose matrices
        timestamp: When computed
    """
    activations: np.ndarray
    poses: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)


class CapsuleRetrievalBridge:
    """
    Bridge connecting CapsuleLayer to memory retrieval scoring.

    Implements P6.2: capsule-enhanced memory retrieval using part-whole
    compositional similarity. Query and memory embeddings are processed
    through capsules, and agreement between capsule representations
    provides a retrieval score boost.

    The bridge provides:
    1. Activation similarity: Do the same capsules fire?
    2. Pose agreement: Are the configurations compatible?
    3. Combined boost: Weighted combination for retrieval enhancement

    Example:
        ```python
        from ww.nca.capsules import CapsuleLayer, CapsuleConfig
        from ww.bridges import CapsuleRetrievalBridge

        # Create capsule layer
        caps_config = CapsuleConfig(input_dim=1024, num_capsules=32)
        capsule_layer = CapsuleLayer(caps_config)

        # Create bridge
        bridge = CapsuleRetrievalBridge(capsule_layer)

        # Get retrieval boost
        query_emb = get_query_embedding()
        memory_embs = [m.embedding for m in memories]

        boosts = bridge.compute_boosts(query_emb, memory_embs)
        enhanced_scores = [s + b for s, b in zip(base_scores, boosts)]
        ```
    """

    def __init__(
        self,
        capsule_layer: CapsuleLayerProtocol | None = None,
        config: CapsuleBridgeConfig | None = None
    ):
        """
        Initialize Capsule-Retrieval bridge.

        Args:
            capsule_layer: CapsuleLayer instance
            config: Bridge configuration
        """
        self.capsule_layer = capsule_layer
        self.config = config or CapsuleBridgeConfig()
        self.state = CapsuleBridgeState()

        # LRU-style cache for capsule representations
        self._cache: dict[int, CapsuleRepresentation] = {}
        self._cache_order: list[int] = []

        logger.info(
            f"P6.2: CapsuleRetrievalBridge initialized "
            f"(act_weight={self.config.activation_weight}, "
            f"pose_weight={self.config.pose_weight})"
        )

    def set_capsule_layer(self, capsule_layer: CapsuleLayerProtocol) -> None:
        """Set or update the capsule layer.

        Args:
            capsule_layer: CapsuleLayer instance
        """
        self.capsule_layer = capsule_layer
        self._cache.clear()
        self._cache_order.clear()
        logger.info("P6.2: Capsule layer updated, cache cleared")

    def _get_capsule_representation(
        self,
        embedding: np.ndarray
    ) -> CapsuleRepresentation:
        """
        Get capsule representation, using cache if available.

        Args:
            embedding: Input embedding

        Returns:
            CapsuleRepresentation with activations and poses
        """
        # Use hash of embedding for cache key
        cache_key = hash(embedding.tobytes())

        if cache_key in self._cache:
            self.state.cache_hits += 1
            return self._cache[cache_key]

        self.state.cache_misses += 1

        # Compute capsule representation
        if self.capsule_layer is not None:
            activations, poses = self.capsule_layer.forward(embedding)
        else:
            # Fallback: mock representation
            activations = np.abs(embedding[:32]) if len(embedding) >= 32 else np.zeros(32)
            poses = np.eye(4, dtype=np.float32)[np.newaxis, :, :].repeat(32, axis=0)

        representation = CapsuleRepresentation(
            activations=activations,
            poses=poses
        )

        # Add to cache with LRU eviction
        self._cache[cache_key] = representation
        self._cache_order.append(cache_key)

        while len(self._cache) > self.config.cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]

        return representation

    def compute_activation_similarity(
        self,
        query_acts: np.ndarray,
        memory_acts: np.ndarray
    ) -> float:
        """
        Compute activation pattern similarity.

        Uses cosine similarity between capsule activation patterns,
        weighted by joint activation strength.

        Args:
            query_acts: Query capsule activations
            memory_acts: Memory capsule activations

        Returns:
            Activation similarity in [0, 1]
        """
        # Normalize activations
        q_norm = np.linalg.norm(query_acts)
        m_norm = np.linalg.norm(memory_acts)

        if q_norm < 1e-8 or m_norm < 1e-8:
            return 0.0

        # Cosine similarity
        cosine_sim = np.dot(query_acts, memory_acts) / (q_norm * m_norm)

        # Weight by joint activation (both should be active)
        joint_activation = np.minimum(query_acts, memory_acts).sum()
        activation_weight = joint_activation / (query_acts.sum() + 1e-8)

        return float(np.clip(cosine_sim * activation_weight, 0.0, 1.0))

    def compute_pose_agreement(
        self,
        query_poses: np.ndarray,
        memory_poses: np.ndarray,
        query_acts: np.ndarray,
        memory_acts: np.ndarray
    ) -> float:
        """
        Compute pose agreement between query and memory.

        Measures how compatible the capsule configurations are,
        weighted by capsule activations (only compare active capsules).

        Args:
            query_poses: Query pose matrices [num_caps, pose_dim, pose_dim]
            memory_poses: Memory pose matrices
            query_acts: Query activations (for weighting)
            memory_acts: Memory activations (for weighting)

        Returns:
            Pose agreement score in [0, 1]
        """
        if len(query_poses) == 0 or len(memory_poses) == 0:
            return 0.0

        num_capsules = len(query_acts)
        total_agreement = 0.0
        total_weight = 0.0

        for i in range(num_capsules):
            # Weight by minimum activation (both must be active)
            weight = min(query_acts[i], memory_acts[i])

            if weight < 0.1:  # Skip inactive capsules
                continue

            # Pose difference (Frobenius norm)
            diff = query_poses[i] - memory_poses[i]
            distance = np.linalg.norm(diff)

            # Convert distance to agreement (exponential decay)
            agreement = np.exp(-distance / 2.0)

            total_agreement += weight * agreement
            total_weight += weight

        if total_weight < 1e-8:
            return 0.0

        return float(total_agreement / total_weight)

    def compute_boost(
        self,
        query_embedding: np.ndarray,
        memory_embedding: np.ndarray
    ) -> float:
        """
        Compute retrieval score boost for a single memory.

        Combines activation similarity and pose agreement into
        a single boost value that enhances retrieval scores.

        Args:
            query_embedding: Query embedding vector
            memory_embedding: Memory embedding vector

        Returns:
            Score boost in [0, max_boost]
        """
        # Get capsule representations
        query_rep = self._get_capsule_representation(query_embedding)
        memory_rep = self._get_capsule_representation(memory_embedding)

        # Compute similarities
        act_sim = self.compute_activation_similarity(
            query_rep.activations,
            memory_rep.activations
        )
        pose_agree = self.compute_pose_agreement(
            query_rep.poses,
            memory_rep.poses,
            query_rep.activations,
            memory_rep.activations
        )

        # Weighted combination
        combined = (
            self.config.activation_weight * act_sim +
            self.config.pose_weight * pose_agree
        )

        # Apply threshold and scale
        if combined < self.config.agreement_threshold:
            boost = 0.0
        else:
            # Scale to max_boost
            boost = (combined - self.config.agreement_threshold) / \
                    (1.0 - self.config.agreement_threshold) * self.config.max_boost

        # Update state
        self.state.n_comparisons += 1
        alpha = 0.01
        self.state.mean_boost = alpha * boost + (1 - alpha) * self.state.mean_boost

        return float(np.clip(boost, 0.0, self.config.max_boost))

    def compute_boosts(
        self,
        query_embedding: np.ndarray,
        memory_embeddings: list[np.ndarray]
    ) -> list[float]:
        """
        Compute retrieval boosts for multiple memories.

        Efficiently processes a batch of memories against a single query.

        Args:
            query_embedding: Query embedding vector
            memory_embeddings: List of memory embedding vectors

        Returns:
            List of score boosts, one per memory
        """
        if not memory_embeddings:
            return []

        self.state.n_queries += 1
        self.state.timestamp = datetime.now()

        # Get query representation once
        query_rep = self._get_capsule_representation(query_embedding)

        boosts = []
        for mem_emb in memory_embeddings:
            mem_rep = self._get_capsule_representation(mem_emb)

            # Compute similarities
            act_sim = self.compute_activation_similarity(
                query_rep.activations,
                mem_rep.activations
            )
            pose_agree = self.compute_pose_agreement(
                query_rep.poses,
                mem_rep.poses,
                query_rep.activations,
                mem_rep.activations
            )

            # Combined boost
            combined = (
                self.config.activation_weight * act_sim +
                self.config.pose_weight * pose_agree
            )

            if combined < self.config.agreement_threshold:
                boost = 0.0
            else:
                boost = (combined - self.config.agreement_threshold) / \
                        (1.0 - self.config.agreement_threshold) * self.config.max_boost

            boosts.append(float(np.clip(boost, 0.0, self.config.max_boost)))

        # Update state
        self.state.n_comparisons += len(memory_embeddings)
        if boosts:
            alpha = 0.1
            mean_batch_boost = np.mean(boosts)
            self.state.mean_boost = alpha * mean_batch_boost + \
                                     (1 - alpha) * self.state.mean_boost

        logger.debug(
            f"P6.2: Computed {len(boosts)} capsule boosts, "
            f"mean={np.mean(boosts):.4f}"
        )

        return boosts

    def get_capsule_features(
        self,
        embedding: np.ndarray
    ) -> dict[str, Any]:
        """
        Get capsule-derived features for an embedding.

        Useful for analysis and visualization.

        Args:
            embedding: Input embedding vector

        Returns:
            Dict with capsule features
        """
        rep = self._get_capsule_representation(embedding)

        # Compute statistics
        active_capsules = (rep.activations > 0.5).sum()
        mean_activation = float(np.mean(rep.activations))
        max_activation = float(np.max(rep.activations))

        # Pose complexity (average Frobenius norm from identity)
        pose_complexity = 0.0
        for pose in rep.poses:
            diff_from_identity = pose - np.eye(pose.shape[0])
            pose_complexity += np.linalg.norm(diff_from_identity)
        pose_complexity /= len(rep.poses)

        return {
            "activations": rep.activations,
            "poses": rep.poses,
            "active_capsules": int(active_capsules),
            "total_capsules": len(rep.activations),
            "mean_activation": mean_activation,
            "max_activation": max_activation,
            "pose_complexity": float(pose_complexity),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get bridge statistics."""
        cache_hit_rate = (
            self.state.cache_hits /
            (self.state.cache_hits + self.state.cache_misses + 1e-8)
        )

        return {
            "n_queries": self.state.n_queries,
            "n_comparisons": self.state.n_comparisons,
            "mean_boost": self.state.mean_boost,
            "cache_size": len(self._cache),
            "cache_hit_rate": cache_hit_rate,
            "config": {
                "activation_weight": self.config.activation_weight,
                "pose_weight": self.config.pose_weight,
                "agreement_threshold": self.config.agreement_threshold,
                "max_boost": self.config.max_boost,
            }
        }

    def clear_cache(self) -> None:
        """Clear the capsule representation cache."""
        self._cache.clear()
        self._cache_order.clear()
        self.state.cache_hits = 0
        self.state.cache_misses = 0
        logger.info("P6.2: Capsule cache cleared")


def create_capsule_bridge(
    capsule_layer: CapsuleLayerProtocol | None = None,
    activation_weight: float = 0.4,
    pose_weight: float = 0.6,
    max_boost: float = 0.3
) -> CapsuleRetrievalBridge:
    """
    Factory function for Capsule-Retrieval bridge.

    Args:
        capsule_layer: Optional CapsuleLayer instance
        activation_weight: Weight for activation similarity
        pose_weight: Weight for pose agreement
        max_boost: Maximum retrieval score boost

    Returns:
        Configured CapsuleRetrievalBridge
    """
    config = CapsuleBridgeConfig(
        activation_weight=activation_weight,
        pose_weight=pose_weight,
        max_boost=max_boost
    )
    return CapsuleRetrievalBridge(
        capsule_layer=capsule_layer,
        config=config
    )
