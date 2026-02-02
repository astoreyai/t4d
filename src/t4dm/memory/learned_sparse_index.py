"""
LearnedSparseIndex - Adaptive Sparse Addressing for T4DM.

Phase 2 of HSA-inspired improvements. Replaces fixed 10% sparsity
with learned, query-dependent sparse addressing.

Architecture:
    q [d=1024]
        │
        ▼
    Shared MLP → h [hidden=256]
        │
        ├─→ Cluster Head → softmax → cluster attention [K]
        ├─→ Feature Head → sigmoid → feature attention [d]
        └─→ Sparsity Gate → sigmoid → sparsity level [1]

Training:
- Online gradient descent from retrieval outcomes
- Neuromodulator guidance (NE for exploration, ACh for mode)
- Per-cluster contribution scores for credit assignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SparseAddressingResult:
    """Result of sparse addressing computation."""

    cluster_attention: np.ndarray  # [K] - attention over clusters
    feature_attention: np.ndarray  # [d] - attention over features
    sparsity_level: float  # Adaptive sparsity (0-1)
    effective_clusters: list[int]  # Indices of active clusters
    query_id: str  # For tracking pending updates

    @property
    def top_k_clusters(self) -> list[int]:
        """Get indices of top-k attended clusters."""
        return list(np.argsort(self.cluster_attention)[::-1])

    def get_sparse_mask(self, threshold: float = 0.1) -> np.ndarray:
        """Get binary mask for feature attention above threshold."""
        return (self.feature_attention > threshold).astype(np.float32)


@dataclass
class PendingUpdate:
    """Pending training update from retrieval outcome."""

    query_id: str
    query_embedding: np.ndarray
    hidden_state: np.ndarray
    cluster_attention: np.ndarray
    feature_attention: np.ndarray
    sparsity_level: float
    timestamp: datetime = field(default_factory=datetime.now)


class LearnedSparseIndex:
    """
    Learned sparse addressing for memory retrieval.

    Replaces fixed 10% sparsity with adaptive, query-dependent addressing.
    Learns to route queries to relevant clusters and features.

    Integration Points:
    - ClusterIndex.select_clusters() uses cluster_attention
    - recall() uses feature_attention for weighted scoring
    - learn_from_outcome() → update() trains the model
    """

    # Default dimensions
    EMBED_DIM = 1024
    HIDDEN_DIM = 256
    MAX_CLUSTERS = 500  # Maximum supported clusters

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        max_clusters: int = 500,
        learning_rate: float = 0.005,
        momentum: float = 0.9,
    ):
        """
        Initialize learned sparse index.

        Args:
            embed_dim: Query embedding dimension
            hidden_dim: Shared MLP hidden dimension
            max_clusters: Maximum clusters to support
            learning_rate: SGD learning rate
            momentum: Momentum for gradient updates
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_clusters = max_clusters
        self.lr = learning_rate
        self.momentum = momentum

        # Shared MLP: query → hidden
        # Xavier initialization
        self.W_shared = np.random.randn(hidden_dim, embed_dim).astype(np.float32) * np.sqrt(2.0 / (embed_dim + hidden_dim))
        self.b_shared = np.zeros(hidden_dim, dtype=np.float32)

        # Cluster head: hidden → cluster attention
        self.W_cluster = np.random.randn(max_clusters, hidden_dim).astype(np.float32) * np.sqrt(2.0 / (hidden_dim + max_clusters))
        self.b_cluster = np.zeros(max_clusters, dtype=np.float32)

        # Feature head: hidden → feature attention
        self.W_feature = np.random.randn(embed_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / (hidden_dim + embed_dim))
        self.b_feature = np.zeros(embed_dim, dtype=np.float32)

        # Sparsity gate: hidden → sparsity level
        self.W_sparsity = np.random.randn(1, hidden_dim).astype(np.float32) * np.sqrt(2.0 / (hidden_dim + 1))
        self.b_sparsity = np.zeros(1, dtype=np.float32)

        # Momentum buffers
        self._velocity_W_shared = np.zeros_like(self.W_shared)
        self._velocity_W_cluster = np.zeros_like(self.W_cluster)
        self._velocity_W_feature = np.zeros_like(self.W_feature)
        self._velocity_W_sparsity = np.zeros_like(self.W_sparsity)

        # Training tracking
        self.n_updates = 0
        self.cold_start_threshold = 50  # Updates before active use

        # Pending updates for deferred training
        self._pending_updates: dict[str, PendingUpdate] = {}
        self._max_pending = 100

        # Statistics
        self.avg_sparsity = 0.1  # Running average sparsity
        self.n_queries = 0

    @property
    def is_warm(self) -> bool:
        """Check if model has enough training data."""
        return self.n_updates >= self.cold_start_threshold

    def forward(
        self,
        query_embedding: np.ndarray,
        n_clusters: int = 0,
        ne_gain: float = 1.0,
        ach_mode: str = "retrieval",
    ) -> SparseAddressingResult:
        """
        Compute sparse addressing for a query.

        Args:
            query_embedding: Query vector [embed_dim]
            n_clusters: Number of actual clusters (0 = use all)
            ne_gain: Norepinephrine arousal level
            ach_mode: Acetylcholine mode ("encoding" or "retrieval")

        Returns:
            SparseAddressingResult with attention distributions
        """
        # Normalize query
        query = np.array(query_embedding, dtype=np.float32)
        if len(query) > self.embed_dim:
            query = query[:self.embed_dim]
        elif len(query) < self.embed_dim:
            query = np.pad(query, (0, self.embed_dim - len(query)))

        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm

        # Shared hidden representation
        hidden = np.tanh(self.W_shared @ query + self.b_shared)

        # Cluster attention (softmax)
        cluster_logits = self.W_cluster @ hidden + self.b_cluster
        if n_clusters > 0:
            # Mask unused clusters
            cluster_logits[n_clusters:] = -1e9

        # Temperature modulation by NE arousal
        # High arousal = lower temperature = sharper attention (exploitation)
        # Low arousal = higher temperature = broader attention (exploration)
        temperature = 1.0 / max(0.5, ne_gain)
        cluster_attention = self._softmax(cluster_logits / temperature)

        # Feature attention (sigmoid)
        feature_logits = self.W_feature @ hidden + self.b_feature
        feature_attention = self._sigmoid(feature_logits)

        # ACh mode modulation
        if ach_mode == "encoding":
            # Encoding mode: broader feature attention
            feature_attention = 0.7 * feature_attention + 0.3 * np.ones_like(feature_attention)
        else:
            # Retrieval mode: sharper feature attention
            feature_attention = np.clip(feature_attention * 1.2, 0, 1)

        # Sparsity level (sigmoid)
        sparsity_logit = self.W_sparsity @ hidden + self.b_sparsity
        base_sparsity = self._sigmoid(sparsity_logit[0])

        # NE modulation of sparsity
        # High arousal = lower sparsity (attend to more)
        # Low arousal = higher sparsity (focus)
        sparsity_level = base_sparsity / max(0.5, ne_gain)
        sparsity_level = np.clip(sparsity_level, 0.01, 0.5)

        # Determine effective clusters (above threshold)
        cluster_threshold = sparsity_level * np.max(cluster_attention)
        effective_clusters = list(np.where(cluster_attention > cluster_threshold)[0])

        # Track for training
        query_id = str(uuid4())

        result = SparseAddressingResult(
            cluster_attention=cluster_attention,
            feature_attention=feature_attention,
            sparsity_level=float(sparsity_level),
            effective_clusters=effective_clusters,
            query_id=query_id,
        )

        # Store pending update
        if len(self._pending_updates) < self._max_pending:
            self._pending_updates[query_id] = PendingUpdate(
                query_id=query_id,
                query_embedding=query,
                hidden_state=hidden,
                cluster_attention=cluster_attention,
                feature_attention=feature_attention,
                sparsity_level=float(sparsity_level),
            )

        self.n_queries += 1
        return result

    def update(
        self,
        query_id: str,
        cluster_rewards: dict[int, float],
        overall_success: bool,
        feature_importance: np.ndarray | None = None,
    ) -> bool:
        """
        Update model from retrieval outcome.

        Args:
            query_id: ID from forward() result
            cluster_rewards: Per-cluster reward signals {cluster_idx: reward}
            overall_success: Whether retrieval was successful
            feature_importance: Optional per-feature importance scores

        Returns:
            True if update was applied
        """
        if query_id not in self._pending_updates:
            return False

        pending = self._pending_updates.pop(query_id)
        query = pending.query_embedding
        hidden = pending.hidden_state
        cluster_att = pending.cluster_attention
        feature_att = pending.feature_attention

        # Target: increase attention on successful clusters
        cluster_target = cluster_att.copy()
        for idx, reward in cluster_rewards.items():
            if idx < len(cluster_target):
                # Shift attention toward rewarded clusters
                cluster_target[idx] += self.lr * (reward - 0.5)

        cluster_target = np.clip(cluster_target, 0.001, 0.999)
        cluster_target = cluster_target / cluster_target.sum()  # Re-normalize

        # Cluster head gradient (cross-entropy style)
        cluster_error = cluster_att - cluster_target
        grad_W_cluster = np.outer(cluster_error, hidden)
        grad_b_cluster = cluster_error

        # Feature head gradient (if importance provided)
        if feature_importance is not None:
            feature_target = np.clip(feature_importance, 0.01, 0.99)
            feature_error = feature_att - feature_target
            grad_W_feature = np.outer(feature_error, hidden)
            grad_b_feature = feature_error
        else:
            # Default: encourage consistent attention
            grad_W_feature = np.zeros_like(self.W_feature)
            grad_b_feature = np.zeros_like(self.b_feature)

        # Sparsity gate gradient
        target_sparsity = 0.1 if overall_success else 0.2  # Success = more focus
        sparsity_error = pending.sparsity_level - target_sparsity
        grad_W_sparsity = sparsity_error * hidden.reshape(1, -1)
        grad_b_sparsity = np.array([sparsity_error])

        # Shared MLP gradient (backprop from all heads)
        d_hidden = (
            self.W_cluster.T @ cluster_error +
            self.W_feature.T @ (feature_att - feature_att.mean()) * 0.1 +  # Regularization
            self.W_sparsity.T.flatten() * sparsity_error
        )
        d_hidden *= (1 - hidden ** 2)  # tanh derivative
        grad_W_shared = np.outer(d_hidden, query)
        grad_b_shared = d_hidden

        # Apply momentum-based updates
        self._velocity_W_cluster = self.momentum * self._velocity_W_cluster - self.lr * grad_W_cluster
        self._velocity_W_feature = self.momentum * self._velocity_W_feature - self.lr * grad_W_feature
        self._velocity_W_sparsity = self.momentum * self._velocity_W_sparsity - self.lr * grad_W_sparsity
        self._velocity_W_shared = self.momentum * self._velocity_W_shared - self.lr * grad_W_shared

        self.W_cluster += self._velocity_W_cluster
        self.W_feature += self._velocity_W_feature
        self.W_sparsity += self._velocity_W_sparsity
        self.W_shared += self._velocity_W_shared

        self.b_cluster -= self.lr * grad_b_cluster
        self.b_feature -= self.lr * grad_b_feature
        self.b_sparsity -= self.lr * grad_b_sparsity
        self.b_shared -= self.lr * grad_b_shared

        # Update statistics
        self.n_updates += 1
        self.avg_sparsity = 0.95 * self.avg_sparsity + 0.05 * pending.sparsity_level

        return True

    def register_pending(
        self,
        query_embedding: np.ndarray,
        cluster_attention: np.ndarray,
        feature_attention: np.ndarray,
        sparsity_level: float,
    ) -> str:
        """
        Register a pending update manually (for external use).

        Returns:
            Query ID for later update
        """
        query_id = str(uuid4())

        # Compute hidden state
        query = np.array(query_embedding, dtype=np.float32)
        if len(query) != self.embed_dim:
            if len(query) > self.embed_dim:
                query = query[:self.embed_dim]
            else:
                query = np.pad(query, (0, self.embed_dim - len(query)))

        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm

        hidden = np.tanh(self.W_shared @ query + self.b_shared)

        self._pending_updates[query_id] = PendingUpdate(
            query_id=query_id,
            query_embedding=query,
            hidden_state=hidden,
            cluster_attention=cluster_attention,
            feature_attention=feature_attention,
            sparsity_level=sparsity_level,
        )

        return query_id

    def get_feature_weighted_score(
        self,
        query_embedding: np.ndarray,
        candidate_embedding: np.ndarray,
    ) -> float:
        """
        Compute feature-weighted similarity score.

        Args:
            query_embedding: Query vector
            candidate_embedding: Candidate memory vector

        Returns:
            Weighted similarity score
        """
        result = self.forward(query_embedding, n_clusters=0)

        query = np.array(query_embedding, dtype=np.float32)
        candidate = np.array(candidate_embedding, dtype=np.float32)

        # Truncate/pad to match
        if len(query) > self.embed_dim:
            query = query[:self.embed_dim]
        elif len(query) < self.embed_dim:
            query = np.pad(query, (0, self.embed_dim - len(query)))

        if len(candidate) > self.embed_dim:
            candidate = candidate[:self.embed_dim]
        elif len(candidate) < self.embed_dim:
            candidate = np.pad(candidate, (0, self.embed_dim - len(candidate)))

        # Feature-weighted dot product
        weighted_query = query * result.feature_attention
        weighted_candidate = candidate * result.feature_attention

        # Normalize
        q_norm = np.linalg.norm(weighted_query)
        c_norm = np.linalg.norm(weighted_candidate)

        if q_norm > 1e-8 and c_norm > 1e-8:
            return float(np.dot(weighted_query, weighted_candidate) / (q_norm * c_norm))

        return 0.0

    def get_statistics(self) -> dict:
        """Get model statistics."""
        return {
            "n_updates": self.n_updates,
            "n_queries": self.n_queries,
            "is_warm": self.is_warm,
            "avg_sparsity": self.avg_sparsity,
            "pending_updates": len(self._pending_updates),
            "cold_start_threshold": self.cold_start_threshold,
        }

    def prune_stale_pending(self, max_age_minutes: float = 60.0) -> int:
        """
        Remove stale pending updates.

        Args:
            max_age_minutes: Maximum age before removal

        Returns:
            Number of pruned entries
        """
        now = datetime.now()
        stale_ids = []

        for query_id, pending in self._pending_updates.items():
            age_minutes = (now - pending.timestamp).total_seconds() / 60
            if age_minutes > max_age_minutes:
                stale_ids.append(query_id)

        for query_id in stale_ids:
            del self._pending_updates[query_id]

        return len(stale_ids)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Stable softmax."""
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (exp_x.sum() + 1e-8)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# Module exports
__all__ = [
    "LearnedSparseIndex",
    "PendingUpdate",
    "SparseAddressingResult",
]
