"""
Generalization Quality Scoring (W3-03).

Compute generalization quality of REM-created prototypes using
silhouette scores. High separation indicates good generalization.

Evidence Base: O'Reilly & Rudy (2001) "Conjunctive Representations in
Learning and Memory: Principles of Cortical and Hippocampal Function"

Key Insight:
    Not all clusters should be generalized. Overlapping clusters
    produce poor prototypes. Silhouette score measures cluster
    separation - only well-separated clusters get prototyped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

import numpy as np
from sklearn.metrics import silhouette_score

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GeneralizationResult:
    """Result of generalization quality scoring.

    Attributes:
        quality: Silhouette score in [-1, 1]. Higher is better.
        should_generalize: Whether to create prototype from this cluster.
        reason: Explanation for the decision.
    """

    quality: float
    should_generalize: bool
    reason: str = ""


@dataclass
class Cluster:
    """A cluster of memory vectors.

    Attributes:
        id: Unique identifier for this cluster.
        vectors: Array of vectors in this cluster [n, dim].
        generalization_quality: Set after scoring.
    """

    id: UUID
    vectors: np.ndarray
    generalization_quality: Optional[float] = None


class GeneralizationQualityScorer:
    """Compute generalization quality of REM-created prototypes.

    Uses silhouette score to measure cluster separation.
    Only well-separated clusters should be generalized into prototypes.

    Silhouette score interpretation:
    - +1: Perfectly separated
    -  0: Overlapping
    - -1: Wrong cluster assignment

    Example:
        >>> scorer = GeneralizationQualityScorer(min_quality=0.3)
        >>> result = scorer.score_cluster(cluster_vectors, all_vectors)
        >>> if result.should_generalize:
        ...     create_prototype(cluster_vectors)
    """

    def __init__(self, min_quality: float = 0.3):
        """Initialize scorer.

        Args:
            min_quality: Minimum silhouette score to recommend generalization.
        """
        self.min_quality = min_quality

    def score_cluster(
        self,
        cluster_vectors: np.ndarray,
        all_vectors: np.ndarray,
    ) -> GeneralizationResult:
        """Compute generalization quality for a cluster.

        Args:
            cluster_vectors: Vectors in the cluster [n, dim].
            all_vectors: All vectors in the dataset [N, dim].

        Returns:
            GeneralizationResult with quality score and recommendation.
        """
        if len(cluster_vectors) < 2:
            return GeneralizationResult(
                quality=0.0,
                should_generalize=False,
                reason="cluster_too_small",
            )

        # Create labels: 1 for cluster members, 0 for others
        labels = np.zeros(len(all_vectors), dtype=int)
        cluster_indices = self._find_indices(cluster_vectors, all_vectors)
        labels[cluster_indices] = 1

        # Need at least 2 labels to compute silhouette
        if len(np.unique(labels)) < 2:
            return GeneralizationResult(
                quality=0.0,
                should_generalize=False,
                reason="insufficient_contrast",
            )

        # Compute silhouette score
        try:
            quality = silhouette_score(all_vectors, labels)
        except ValueError:
            return GeneralizationResult(
                quality=0.0,
                should_generalize=False,
                reason="silhouette_computation_failed",
            )

        should_generalize = quality >= self.min_quality
        reason = "high_separation" if should_generalize else "low_separation"

        return GeneralizationResult(
            quality=float(quality),
            should_generalize=should_generalize,
            reason=reason,
        )

    def filter_clusters_for_prototyping(
        self,
        clusters: list[Cluster],
        all_vectors: np.ndarray,
    ) -> list[Cluster]:
        """Filter clusters to only those suitable for prototyping.

        Args:
            clusters: List of clusters to evaluate.
            all_vectors: All vectors in the dataset.

        Returns:
            List of clusters suitable for prototyping.
        """
        suitable = []

        for cluster in clusters:
            result = self.score_cluster(cluster.vectors, all_vectors)

            if result.should_generalize:
                cluster.generalization_quality = result.quality
                suitable.append(cluster)
            else:
                logger.info(
                    f"Cluster {cluster.id} not suitable for prototyping: "
                    f"{result.reason} (quality={result.quality:.2f})"
                )

        return suitable

    def _find_indices(
        self,
        cluster_vectors: np.ndarray,
        all_vectors: np.ndarray,
    ) -> list[int]:
        """Find indices of cluster vectors in all_vectors.

        Uses row-wise equality check with tolerance.

        Args:
            cluster_vectors: Vectors to find.
            all_vectors: Array to search in.

        Returns:
            List of indices where cluster vectors appear.
        """
        indices = []

        for cv in cluster_vectors:
            # Find closest match
            distances = np.linalg.norm(all_vectors - cv, axis=1)
            idx = np.argmin(distances)
            if distances[idx] < 1e-6:  # Exact match
                indices.append(idx)

        return indices
