"""
ClusterIndex - Hierarchical Episode Retrieval for World Weaver.

Phase 1 of HSA-inspired improvements. Provides CA3-like semantic clustering
for O(log n) retrieval instead of O(n) flat search.

Architecture:
- Cluster centroids computed during sleep consolidation
- NE-modulated cluster selection (high arousal = more clusters)
- ACh-mode affects exploration/exploitation balance
- Per-cluster statistics enable learned routing

Complexity Improvement:
- Current: O(n) flat k-NN search
- Hierarchical: O(K + k*n/K) where K=clusters, k=selected
- For 100K episodes, K=500, k=5: ~67x speedup
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusterMeta:
    """Metadata for a semantic cluster."""

    cluster_id: str
    centroid: np.ndarray  # Mean embedding of cluster members
    member_count: int = 0
    member_ids: list[UUID] = field(default_factory=list)

    # Statistics for learned routing
    total_retrievals: int = 0
    successful_retrievals: int = 0
    avg_score: float = 0.5
    last_accessed: datetime | None = None

    # Cluster quality metrics
    variance: float = 0.0  # Intra-cluster variance (lower = tighter)
    coherence: float = 1.0  # Semantic coherence (higher = better)

    # Temporal properties
    created_at: datetime = field(default_factory=datetime.now)
    avg_member_age_days: float = 0.0

    @property
    def success_rate(self) -> float:
        """Retrieval success rate for this cluster."""
        if self.total_retrievals == 0:
            return 0.5  # Prior
        return self.successful_retrievals / self.total_retrievals

    @property
    def priority_score(self) -> float:
        """Combined priority for cluster selection."""
        # Blend success rate, recency, and coherence
        recency_factor = 1.0
        if self.last_accessed:
            hours_since = (datetime.now() - self.last_accessed).total_seconds() / 3600
            recency_factor = np.exp(-0.01 * hours_since)  # Decay over days

        return (
            0.4 * self.success_rate +
            0.3 * recency_factor +
            0.2 * self.coherence +
            0.1 * min(1.0, self.member_count / 100)  # Prefer populated clusters
        )


class ClusterIndex:
    """
    Hierarchical index for semantic clustering of episodes.

    Provides two-stage retrieval:
    1. Select top-k clusters by query similarity
    2. Search within selected clusters only

    Integration Points:
    - SleepConsolidation.rem_phase() → register_cluster()
    - EpisodicMemory.recall() → select_clusters()
    - learn_from_outcome() → record_retrieval_outcome()
    """

    DEFAULT_K = 5  # Default clusters to select
    MIN_CLUSTER_SIZE = 5  # Minimum episodes per cluster
    MAX_CLUSTERS = 1000  # Maximum clusters to maintain

    def __init__(
        self,
        embedding_dim: int = 1024,
        default_k: int = 5,
        similarity_threshold: float = 0.3,
    ):
        """
        Initialize cluster index.

        Args:
            embedding_dim: Dimension of embeddings (BGE-M3 = 1024)
            default_k: Default number of clusters to select
            similarity_threshold: Minimum similarity for cluster inclusion
        """
        self.embedding_dim = embedding_dim
        self.default_k = default_k
        self.similarity_threshold = similarity_threshold

        # Cluster storage
        self.clusters: dict[str, ClusterMeta] = {}
        self.centroid_matrix: np.ndarray | None = None  # [n_clusters, dim]
        self._centroid_ids: list[str] = []  # Mapping index → cluster_id

        # Learning parameters
        self.lr = 0.01
        self.momentum = 0.9

        # Statistics
        self.n_queries = 0
        self.total_clusters_selected = 0

    @property
    def n_clusters(self) -> int:
        """Number of registered clusters."""
        return len(self.clusters)

    def register_cluster(
        self,
        cluster_id: str,
        centroid: np.ndarray,
        member_ids: list[UUID],
        variance: float = 0.0,
        coherence: float = 1.0,
    ) -> ClusterMeta:
        """
        Register a new cluster (called during sleep consolidation).

        Args:
            cluster_id: Unique cluster identifier
            centroid: Mean embedding of cluster members
            member_ids: UUIDs of episodes in this cluster
            variance: Intra-cluster variance
            coherence: Semantic coherence score

        Returns:
            Created ClusterMeta
        """
        # Normalize centroid
        centroid = np.array(centroid, dtype=np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm

        # Ensure correct dimension
        if len(centroid) != self.embedding_dim:
            if len(centroid) > self.embedding_dim:
                centroid = centroid[:self.embedding_dim]
            else:
                centroid = np.pad(centroid, (0, self.embedding_dim - len(centroid)))

        cluster = ClusterMeta(
            cluster_id=cluster_id,
            centroid=centroid,
            member_count=len(member_ids),
            member_ids=member_ids,
            variance=variance,
            coherence=coherence,
        )

        self.clusters[cluster_id] = cluster
        self._rebuild_centroid_matrix()

        logger.debug(
            f"Registered cluster {cluster_id}: "
            f"{len(member_ids)} members, coherence={coherence:.2f}"
        )

        return cluster

    def unregister_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster from the index."""
        if cluster_id not in self.clusters:
            return False

        del self.clusters[cluster_id]
        self._rebuild_centroid_matrix()
        return True

    def _rebuild_centroid_matrix(self) -> None:
        """Rebuild centroid matrix for batch similarity computation."""
        if not self.clusters:
            self.centroid_matrix = None
            self._centroid_ids = []
            return

        self._centroid_ids = list(self.clusters.keys())
        centroids = [self.clusters[cid].centroid for cid in self._centroid_ids]
        self.centroid_matrix = np.vstack(centroids).astype(np.float32)

    def select_clusters(
        self,
        query_embedding: np.ndarray,
        k: int | None = None,
        ne_gain: float = 1.0,
        ach_mode: str = "retrieval",
        min_coverage: float = 0.3,
    ) -> list[tuple[str, float]]:
        """
        Select top-k clusters for a query.

        NE-modulation:
        - High arousal (ne_gain > 1.5) → select more clusters (exploration)
        - Low arousal (ne_gain < 0.7) → select fewer clusters (exploitation)

        ACh-mode effects:
        - "retrieval" mode → prioritize high success rate clusters
        - "encoding" mode → prioritize diverse clusters

        Args:
            query_embedding: Query vector (1024-dim)
            k: Number of clusters to select (None = auto from ne_gain)
            ne_gain: Norepinephrine arousal level (1.0 = baseline)
            ach_mode: Acetylcholine mode ("encoding" or "retrieval")
            min_coverage: Minimum fraction of episodes to cover

        Returns:
            List of (cluster_id, similarity_score) tuples
        """
        if self.centroid_matrix is None or len(self.clusters) == 0:
            return []

        # Normalize query
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if len(query_embedding) > self.embedding_dim:
            query_embedding = query_embedding[:self.embedding_dim]
        elif len(query_embedding) < self.embedding_dim:
            query_embedding = np.pad(query_embedding, (0, self.embedding_dim - len(query_embedding)))

        norm = np.linalg.norm(query_embedding)
        if norm > 1e-8:
            query_embedding = query_embedding / norm

        # Compute similarities with all clusters (batch operation)
        similarities = self.centroid_matrix @ query_embedding

        # Determine k based on NE arousal
        if k is None:
            base_k = self.default_k
            # NE modulation: high arousal = more exploration
            if ne_gain > 1.5:
                k = min(base_k * 2, len(self.clusters))
            elif ne_gain < 0.7:
                k = max(base_k // 2, 1)
            else:
                k = base_k

        # Apply ACh-mode modulation to scores
        if ach_mode == "retrieval":
            # Boost high success rate clusters
            priority_boost = np.array([
                self.clusters[cid].priority_score
                for cid in self._centroid_ids
            ])
            adjusted_scores = 0.7 * similarities + 0.3 * priority_boost
        else:
            # Encoding mode: prioritize diverse/underexplored clusters
            diversity_boost = np.array([
                1.0 - self.clusters[cid].success_rate  # Inverse success = unexplored
                for cid in self._centroid_ids
            ])
            adjusted_scores = 0.6 * similarities + 0.4 * diversity_boost

        # Get top-k indices
        k = min(k, len(self.clusters))
        top_indices = np.argsort(adjusted_scores)[-k:][::-1]

        # Filter by similarity threshold
        selected = []
        total_members = 0
        total_episodes = sum(c.member_count for c in self.clusters.values())

        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= self.similarity_threshold or len(selected) == 0:
                cluster_id = self._centroid_ids[idx]
                selected.append((cluster_id, sim))
                total_members += self.clusters[cluster_id].member_count

                # Check coverage
                if total_episodes > 0 and total_members / total_episodes >= min_coverage:
                    break

        # Update statistics
        self.n_queries += 1
        self.total_clusters_selected += len(selected)

        return selected

    def record_retrieval_outcome(
        self,
        cluster_ids: list[str],
        successful: bool,
        retrieved_scores: dict[str, float] | None = None,
    ) -> None:
        """
        Record retrieval outcome for cluster statistics.

        Args:
            cluster_ids: Clusters that were searched
            successful: Whether retrieval was successful
            retrieved_scores: Per-cluster average scores from retrieval
        """
        for cid in cluster_ids:
            if cid not in self.clusters:
                continue

            cluster = self.clusters[cid]
            cluster.total_retrievals += 1
            cluster.last_accessed = datetime.now()

            if successful:
                cluster.successful_retrievals += 1

            # Update average score with momentum
            if retrieved_scores and cid in retrieved_scores:
                score = retrieved_scores[cid]
                cluster.avg_score = (
                    self.momentum * cluster.avg_score +
                    (1 - self.momentum) * score
                )

    def update_cluster_centroid(
        self,
        cluster_id: str,
        new_embedding: np.ndarray,
        weight: float = 0.1,
    ) -> None:
        """
        Update cluster centroid with new embedding (incremental update).

        Args:
            cluster_id: Cluster to update
            new_embedding: New embedding to incorporate
            weight: Blending weight for new embedding
        """
        if cluster_id not in self.clusters:
            return

        cluster = self.clusters[cluster_id]
        new_embedding = np.array(new_embedding, dtype=np.float32)

        # Ensure correct dimension
        if len(new_embedding) > self.embedding_dim:
            new_embedding = new_embedding[:self.embedding_dim]
        elif len(new_embedding) < self.embedding_dim:
            new_embedding = np.pad(new_embedding, (0, self.embedding_dim - len(new_embedding)))

        # Weighted update
        cluster.centroid = (1 - weight) * cluster.centroid + weight * new_embedding

        # Re-normalize
        norm = np.linalg.norm(cluster.centroid)
        if norm > 1e-8:
            cluster.centroid = cluster.centroid / norm

        # Rebuild matrix for next query
        self._rebuild_centroid_matrix()

    def add_to_cluster(
        self,
        cluster_id: str,
        episode_id: UUID,
        embedding: np.ndarray,
    ) -> bool:
        """
        Add an episode to an existing cluster.

        Args:
            cluster_id: Target cluster
            episode_id: Episode UUID
            embedding: Episode embedding

        Returns:
            True if successfully added
        """
        if cluster_id not in self.clusters:
            return False

        cluster = self.clusters[cluster_id]

        if episode_id not in cluster.member_ids:
            cluster.member_ids.append(episode_id)
            cluster.member_count += 1

            # Update centroid incrementally
            weight = 1.0 / (cluster.member_count + 1)
            self.update_cluster_centroid(cluster_id, embedding, weight)

        return True

    def find_nearest_cluster(
        self,
        embedding: np.ndarray,
        exclude: set[str] | None = None,
    ) -> tuple[str, float] | None:
        """
        Find the nearest cluster to an embedding.

        Args:
            embedding: Query embedding
            exclude: Cluster IDs to exclude

        Returns:
            (cluster_id, similarity) or None
        """
        if self.centroid_matrix is None:
            return None

        # Normalize
        embedding = np.array(embedding, dtype=np.float32)
        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
        elif len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))

        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        # Compute similarities
        similarities = self.centroid_matrix @ embedding

        # Find best non-excluded
        exclude = exclude or set()
        best_idx = None
        best_sim = -1.0

        for idx in np.argsort(similarities)[::-1]:
            cid = self._centroid_ids[idx]
            if cid not in exclude:
                best_idx = idx
                best_sim = float(similarities[idx])
                break

        if best_idx is None:
            return None

        return self._centroid_ids[best_idx], best_sim

    def get_cluster_members(self, cluster_id: str) -> list[UUID]:
        """Get member IDs for a cluster."""
        if cluster_id not in self.clusters:
            return []
        return self.clusters[cluster_id].member_ids

    def get_statistics(self) -> dict:
        """Get index statistics."""
        if not self.clusters:
            return {
                "n_clusters": 0,
                "total_episodes": 0,
                "avg_cluster_size": 0,
                "avg_queries_per_cluster": 0,
            }

        total_episodes = sum(c.member_count for c in self.clusters.values())
        total_retrievals = sum(c.total_retrievals for c in self.clusters.values())

        return {
            "n_clusters": len(self.clusters),
            "total_episodes": total_episodes,
            "avg_cluster_size": total_episodes / len(self.clusters),
            "avg_queries_per_cluster": total_retrievals / len(self.clusters) if self.clusters else 0,
            "avg_success_rate": np.mean([c.success_rate for c in self.clusters.values()]),
            "avg_coherence": np.mean([c.coherence for c in self.clusters.values()]),
            "n_queries": self.n_queries,
            "avg_clusters_per_query": self.total_clusters_selected / max(1, self.n_queries),
        }

    def prune_stale_clusters(
        self,
        max_age_days: float = 30.0,
        min_success_rate: float = 0.1,
    ) -> list[str]:
        """
        Remove stale or low-quality clusters.

        Args:
            max_age_days: Maximum age before pruning
            min_success_rate: Minimum success rate to keep

        Returns:
            List of pruned cluster IDs
        """
        pruned = []
        now = datetime.now()

        for cid, cluster in list(self.clusters.items()):
            age_days = (now - cluster.created_at).total_seconds() / 86400

            # Prune old low-performers
            if (age_days > max_age_days and cluster.success_rate < min_success_rate) or cluster.member_count == 0:
                del self.clusters[cid]
                pruned.append(cid)

        if pruned:
            self._rebuild_centroid_matrix()
            logger.info(f"Pruned {len(pruned)} stale clusters")

        return pruned

    # MEMORY-HIGH-006 FIX: Add save/load methods for cluster persistence
    def save_state(self) -> dict:
        """
        Save cluster index state to dictionary.

        Returns:
            Dictionary with all cluster data
        """
        clusters_data = {}
        for cid, cluster in self.clusters.items():
            clusters_data[cid] = {
                "centroid": cluster.centroid.tolist(),
                "member_count": cluster.member_count,
                "member_ids": [str(mid) for mid in cluster.member_ids],
                "variance": cluster.variance,
                "coherence": cluster.coherence,
                "total_retrievals": cluster.total_retrievals,
                "successful_retrievals": cluster.successful_retrievals,
                "created_at": cluster.created_at.isoformat(),
            }

        return {
            "clusters": clusters_data,
            "embedding_dim": self.embedding_dim,
            "default_k": self.default_k,
            "similarity_threshold": self.similarity_threshold,
            "n_queries": self.n_queries,
            "total_clusters_selected": self.total_clusters_selected,
        }

    def load_state(self, state: dict) -> None:
        """
        Load cluster index state from dictionary.

        Args:
            state: Dictionary from save_state()
        """
        self.n_queries = state.get("n_queries", 0)
        self.total_clusters_selected = state.get("total_clusters_selected", 0)

        clusters_data = state.get("clusters", {})
        for cid, cluster_data in clusters_data.items():
            centroid = np.array(cluster_data["centroid"], dtype=np.float32)
            member_ids = [UUID(mid) for mid in cluster_data["member_ids"]]

            created_at = datetime.now()
            if "created_at" in cluster_data:
                try:
                    created_at = datetime.fromisoformat(cluster_data["created_at"])
                except (ValueError, TypeError):
                    pass

            cluster = ClusterMeta(
                cluster_id=cid,
                centroid=centroid,
                member_count=cluster_data.get("member_count", len(member_ids)),
                member_ids=member_ids,
                variance=cluster_data.get("variance", 0.0),
                coherence=cluster_data.get("coherence", 1.0),
            )
            cluster.total_retrievals = cluster_data.get("total_retrievals", 0)
            cluster.successful_retrievals = cluster_data.get("successful_retrievals", 0)
            cluster.created_at = created_at

            self.clusters[cid] = cluster

        self._rebuild_centroid_matrix()
        logger.info(f"Loaded {len(self.clusters)} clusters from state")

    # ==================== P2.2: ACh-CA3 Completion Strength ====================

    @staticmethod
    def get_completion_strength(ach_mode: str = "balanced") -> float:
        """
        Get pattern completion strength based on ACh mode.

        P2.2 ACh-CA3 Connection (Hasselmo 2006):
        - High ACh (encoding mode): Reduce pattern completion, favor separation
        - Low ACh (retrieval mode): Enhance pattern completion for recall

        Args:
            ach_mode: Acetylcholine mode ("encoding", "balanced", "retrieval")

        Returns:
            Completion strength [0.2, 0.7] where higher = more pattern completion
        """
        if ach_mode == "encoding":
            # Reduce pattern completion, favor separation
            return 0.2
        elif ach_mode == "retrieval":
            # Full pattern completion for best recall
            return 0.7
        else:
            # Balanced mode
            return 0.4

    def search(
        self,
        query_embedding: np.ndarray,
        k: int | None = None,
        ne_gain: float = 1.0,
        ach_mode: str = "balanced",
    ) -> tuple[list[tuple[str, float]], float]:
        """
        Search clusters and return completion strength.

        P2.2: Unified search interface that integrates ACh-modulated
        pattern completion with cluster selection.

        Args:
            query_embedding: Query vector (1024-dim)
            k: Number of clusters to select (None = auto from ne_gain)
            ne_gain: Norepinephrine arousal level (1.0 = baseline)
            ach_mode: Acetylcholine mode ("encoding", "balanced", "retrieval")

        Returns:
            Tuple of:
            - List of (cluster_id, similarity_score) tuples
            - Completion strength to use for pattern completion
        """
        # Select clusters
        selected = self.select_clusters(
            query_embedding=query_embedding,
            k=k,
            ne_gain=ne_gain,
            ach_mode=ach_mode,
        )

        # Get completion strength based on ACh mode
        completion_strength = self.get_completion_strength(ach_mode)

        logger.debug(
            f"Cluster search: ach_mode={ach_mode}, "
            f"completion_strength={completion_strength:.2f}, "
            f"clusters_selected={len(selected)}"
        )

        return selected, completion_strength


# Module exports
__all__ = [
    "ClusterIndex",
    "ClusterMeta",
]
