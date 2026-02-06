"""
Variational Consolidation Framing (W4-01).

Reframes existing consolidation as variational Expectation-Maximization:
- NREM = E-step: Infer cluster assignments (soft membership)
- REM = M-step: Update cluster prototypes (means/variances)
- PRUNE = Regularization: Remove low-posterior items

Evidence Base: Friston (2010) "The free-energy principle: a unified brain theory?"

Mathematical Foundation:
    Sleep consolidation minimizes variational free energy:

    F = E_q[log q(z) - log p(x,z)]

    where:
    - x = observed episodic memories
    - z = latent cluster assignments
    - q(z) = approximate posterior (inferred during NREM)
    - p(x,z) = joint model (prototypes created during REM)

    NREM (E-step): Update q(z|x) to minimize KL(q||p(z|x))
    REM (M-step): Update p(x|z) prototypes to maximize ELBO
    PRUNE: Remove z's with posterior < threshold (regularization)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VariationalStep(Enum):
    """Variational EM step corresponding to sleep phase."""

    E_STEP = "e_step"  # NREM: infer cluster assignments
    M_STEP = "m_step"  # REM: update prototypes
    REGULARIZATION = "regularization"  # PRUNE: remove low-posterior


@dataclass
class VariationalState:
    """State of variational consolidation.

    Attributes:
        free_energy: Current variational free energy.
        elbo: Evidence lower bound (negative free energy).
        kl_divergence: KL(q||p) between approximate and true posterior.
        log_likelihood: E_q[log p(x|z)] expected log-likelihood.
        step_count: Number of EM iterations.
    """

    free_energy: float
    elbo: float
    kl_divergence: float
    log_likelihood: float
    step_count: int = 0

    @property
    def converged(self) -> bool:
        """Check if converged (ELBO stable)."""
        return self.free_energy < 0.01


@dataclass
class ClusterAssignment:
    """Soft cluster assignment for a memory.

    Attributes:
        memory_id: The memory being assigned.
        cluster_posteriors: Posterior probabilities for each cluster.
        most_likely_cluster: Index of highest posterior cluster.
        assignment_entropy: Uncertainty in assignment.
    """

    memory_id: str
    cluster_posteriors: np.ndarray
    most_likely_cluster: int
    assignment_entropy: float

    @classmethod
    def from_similarities(
        cls,
        memory_id: str,
        similarities: np.ndarray,
        temperature: float = 1.0,
    ) -> "ClusterAssignment":
        """Create soft assignment from similarity scores.

        Args:
            memory_id: ID of the memory.
            similarities: Similarity to each cluster centroid.
            temperature: Softmax temperature (lower = sharper).

        Returns:
            ClusterAssignment with posteriors computed via softmax.
        """
        # Softmax for soft clustering
        scaled = similarities / temperature
        exp_sim = np.exp(scaled - np.max(scaled))  # Numerical stability
        posteriors = exp_sim / exp_sim.sum()

        most_likely = int(np.argmax(posteriors))

        # Entropy: H = -sum(p * log(p))
        entropy = -np.sum(posteriors * np.log(posteriors + 1e-10))

        return cls(
            memory_id=memory_id,
            cluster_posteriors=posteriors,
            most_likely_cluster=most_likely,
            assignment_entropy=entropy,
        )


@dataclass
class ClusterPrototype:
    """Cluster prototype (sufficient statistics).

    Attributes:
        cluster_id: Unique cluster identifier.
        mean: Centroid of cluster members.
        variance: Diagonal variance of cluster.
        member_count: Number of assigned members.
        total_responsibility: Sum of soft assignments.
    """

    cluster_id: int
    mean: np.ndarray
    variance: np.ndarray
    member_count: int
    total_responsibility: float


class VariationalConsolidation:
    """Variational EM wrapper for sleep consolidation.

    Interprets existing consolidation phases as variational inference:
    - NREM (E-step): Compute soft cluster assignments q(z|x)
    - REM (M-step): Update cluster prototypes p(x|z)
    - PRUNE: Remove low-responsibility items

    Example:
        >>> vc = VariationalConsolidation()
        >>> # During NREM
        >>> assignments = vc.e_step(memories, clusters)
        >>> # During REM
        >>> new_clusters = vc.m_step(memories, assignments)
        >>> # Track convergence
        >>> state = vc.compute_state(memories, clusters, assignments)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        posterior_threshold: float = 0.1,
        convergence_threshold: float = 0.001,
    ):
        """Initialize variational consolidation.

        Args:
            temperature: Softmax temperature for E-step.
            posterior_threshold: Minimum posterior for assignment.
            convergence_threshold: ELBO change threshold for convergence.
        """
        self.temperature = temperature
        self.posterior_threshold = posterior_threshold
        self.convergence_threshold = convergence_threshold
        self._previous_elbo: Optional[float] = None

    def e_step(
        self,
        memories: list[np.ndarray],
        clusters: list[ClusterPrototype],
    ) -> list[ClusterAssignment]:
        """E-step: Infer soft cluster assignments.

        Computes q(z|x) for each memory given current cluster prototypes.
        This corresponds to NREM phase identifying which memories
        belong to which clusters.

        Args:
            memories: List of memory embeddings.
            clusters: Current cluster prototypes.

        Returns:
            List of ClusterAssignment with soft posteriors.
        """
        assignments = []

        for i, memory in enumerate(memories):
            # Compute similarity to each cluster
            similarities = np.array([
                self._cluster_similarity(memory, cluster)
                for cluster in clusters
            ])

            assignment = ClusterAssignment.from_similarities(
                memory_id=str(i),
                similarities=similarities,
                temperature=self.temperature,
            )
            assignments.append(assignment)

        return assignments

    def m_step(
        self,
        memories: list[np.ndarray],
        assignments: list[ClusterAssignment],
        n_clusters: int,
    ) -> list[ClusterPrototype]:
        """M-step: Update cluster prototypes.

        Updates p(x|z) parameters given soft assignments.
        This corresponds to REM phase creating abstract prototypes.

        Args:
            memories: List of memory embeddings.
            assignments: Soft cluster assignments from E-step.
            n_clusters: Number of clusters.

        Returns:
            Updated ClusterPrototypes.
        """
        dim = memories[0].shape[0] if memories else 64

        clusters = []
        for k in range(n_clusters):
            # Weighted sum of responsibilities
            responsibilities = np.array([
                a.cluster_posteriors[k] if k < len(a.cluster_posteriors) else 0
                for a in assignments
            ])
            total_resp = responsibilities.sum()

            if total_resp > 1e-8:
                # Weighted mean
                weighted_sum = sum(
                    r * m for r, m in zip(responsibilities, memories)
                )
                mean = weighted_sum / total_resp

                # Weighted variance
                weighted_var = sum(
                    r * (m - mean) ** 2
                    for r, m in zip(responsibilities, memories)
                )
                variance = weighted_var / total_resp + 1e-6
            else:
                mean = np.zeros(dim)
                variance = np.ones(dim)

            clusters.append(ClusterPrototype(
                cluster_id=k,
                mean=mean,
                variance=variance,
                member_count=int((responsibilities > self.posterior_threshold).sum()),
                total_responsibility=total_resp,
            ))

        return clusters

    def regularization_step(
        self,
        assignments: list[ClusterAssignment],
    ) -> list[str]:
        """Regularization: Identify low-posterior items for pruning.

        Items with max posterior below threshold are candidates for
        removal. This corresponds to PRUNE phase.

        Args:
            assignments: Cluster assignments from E-step.

        Returns:
            List of memory_ids to prune.
        """
        to_prune = []

        for assignment in assignments:
            max_posterior = assignment.cluster_posteriors.max()
            if max_posterior < self.posterior_threshold:
                to_prune.append(assignment.memory_id)

        return to_prune

    def compute_state(
        self,
        memories: list[np.ndarray],
        clusters: list[ClusterPrototype],
        assignments: list[ClusterAssignment],
    ) -> VariationalState:
        """Compute variational state (free energy, ELBO).

        Args:
            memories: Memory embeddings.
            clusters: Current prototypes.
            assignments: Current soft assignments.

        Returns:
            VariationalState with free energy decomposition.
        """
        # Log-likelihood: E_q[log p(x|z)]
        log_lik = 0.0
        for memory, assignment in zip(memories, assignments):
            for k, posterior in enumerate(assignment.cluster_posteriors):
                if k < len(clusters) and posterior > 1e-10:
                    # Gaussian log-likelihood
                    diff = memory - clusters[k].mean
                    var = clusters[k].variance
                    ll = -0.5 * np.sum(diff ** 2 / var + np.log(var))
                    log_lik += posterior * ll

        # KL divergence: KL(q||p) using entropy
        kl_div = sum(a.assignment_entropy for a in assignments)

        # ELBO = log_lik - KL
        elbo = log_lik - kl_div

        # Free energy = -ELBO
        free_energy = -elbo

        return VariationalState(
            free_energy=free_energy,
            elbo=elbo,
            kl_divergence=kl_div,
            log_likelihood=log_lik,
        )

    def _cluster_similarity(
        self,
        memory: np.ndarray,
        cluster: ClusterPrototype,
    ) -> float:
        """Compute similarity between memory and cluster.

        Uses negative squared Mahalanobis distance (Gaussian likelihood).
        """
        diff = memory - cluster.mean
        var = cluster.variance + 1e-8
        return -0.5 * np.sum(diff ** 2 / var)
