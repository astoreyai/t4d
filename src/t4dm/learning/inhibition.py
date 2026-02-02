"""
GABA/Glutamate-like Inhibitory Dynamics for T4DM.

Biological Basis:
- GABA neurons provide lateral inhibition in cortex/hippocampus
- Winner-take-all dynamics sharpen neural representations
- Balance of E/I determines network stability and sparsity
- Oscillations (gamma/theta) emerge from E/I balance

Implementation:
- Soft winner-take-all sharpens retrieval rankings
- Lateral inhibition between similar memories
- Sparse activation via competitive dynamics
- Configurable inhibition strength and sparsity

Integration Points:
1. EpisodicMemory.recall(): Apply inhibition to sharpen rankings
2. PatternSeparation: Reinforce sparse activation
3. NeuroSymbolicReasoner.fuse_scores(): Apply after fusion
4. UnifiedMemory: Competitive dynamics across memory systems

References:
- Douglas & Martin (2004): Recurrent excitation in neocortex
- Rolls & Treves (1998): Sparse coding in hippocampus
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InhibitionResult:
    """Result of applying inhibitory dynamics."""

    original_scores: dict[str, float]
    inhibited_scores: dict[str, float]
    winners: list[str]  # IDs that "won" the competition
    sparsity: float  # Fraction of items surviving
    iterations: int  # Convergence iterations
    timestamp: datetime = field(default_factory=datetime.now)


class InhibitoryNetwork:
    """
    Competitive inhibition network inspired by GABA/glutamate balance.

    Implements soft winner-take-all dynamics that:
    1. Sharpen retrieval rankings
    2. Suppress weakly activated memories
    3. Reduce interference between similar items
    4. Create sparse output representations

    The dynamics are:
    - Excitation: Each item's score contributes to activation
    - Inhibition: Active items suppress similar/competing items
    - Convergence: Iterate until stable or max iterations

    Key insight: The brain doesn't return "all memories with score > X".
    It runs a competition where strong patterns suppress weak ones,
    creating sparse, distinct outputs.
    """

    def __init__(
        self,
        inhibition_strength: float = 0.5,
        sparsity_target: float = 0.2,
        similarity_inhibition: bool = True,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
        temperature: float = 1.0
    ):
        """
        Initialize inhibitory network.

        Args:
            inhibition_strength: How strongly winners suppress losers
            sparsity_target: Target fraction of surviving items
            similarity_inhibition: Whether similar items inhibit each other
            max_iterations: Maximum competition iterations
            convergence_threshold: Threshold for early stopping
            temperature: Softmax temperature for competition
        """
        self.inhibition_strength = inhibition_strength
        self.sparsity_target = sparsity_target
        self.similarity_inhibition = similarity_inhibition
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.temperature = temperature

        # History for analysis
        self._history: list[InhibitionResult] = []

    def apply_inhibition(
        self,
        scores: dict[str, float],
        embeddings: dict[str, np.ndarray] | None = None
    ) -> InhibitionResult:
        """
        Apply competitive inhibition to retrieval scores.

        Args:
            scores: Memory ID -> initial retrieval score
            embeddings: Optional Memory ID -> embedding for similarity

        Returns:
            InhibitionResult with sharpened scores
        """
        if not scores:
            return InhibitionResult(
                original_scores={},
                inhibited_scores={},
                winners=[],
                sparsity=0.0,
                iterations=0
            )

        ids = list(scores.keys())
        n = len(ids)

        # Convert to array
        activations = np.array([scores[id_] for id_ in ids], dtype=np.float32)
        original_activations = activations.copy()

        # Compute similarity matrix if embeddings provided
        similarity_matrix = None
        if self.similarity_inhibition and embeddings:
            similarity_matrix = self._compute_similarity_matrix(ids, embeddings)

        # Iterative competition
        final_iteration = 0
        for iteration in range(self.max_iterations):
            final_iteration = iteration
            prev_activations = activations.copy()

            # Softmax for competition weights
            exp_act = np.exp(activations / self.temperature)
            competition_weights = exp_act / (exp_act.sum() + 1e-10)

            # Compute inhibition
            inhibition = np.zeros(n, dtype=np.float32)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue

                    # Base inhibition from competing item j
                    base_inhibit = competition_weights[j] * self.inhibition_strength

                    # Scale by similarity if available
                    if similarity_matrix is not None:
                        base_inhibit *= similarity_matrix[i, j]

                    inhibition[i] += base_inhibit

            # Apply inhibition (bounded)
            activations = activations - inhibition
            activations = np.maximum(activations, 0.0)

            # Normalize to preserve total activation
            if activations.sum() > 0:
                activations = activations * (original_activations.sum() / activations.sum())

            # Check convergence
            delta = np.linalg.norm(activations - prev_activations)
            if delta < self.convergence_threshold:
                break

        # Determine winners (above dynamic threshold)
        if len(activations) > 0 and activations.max() > 0:
            threshold = np.percentile(activations, (1 - self.sparsity_target) * 100)
            winners = [ids[i] for i in range(n) if activations[i] >= threshold]
        else:
            winners = []

        # Convert back to dict
        inhibited_scores = {ids[i]: float(activations[i]) for i in range(n)}

        # Compute sparsity
        active_count = sum(1 for a in activations if a > 0.01)
        sparsity = active_count / n if n > 0 else 0.0

        result = InhibitionResult(
            original_scores=scores.copy(),
            inhibited_scores=inhibited_scores,
            winners=winners,
            sparsity=sparsity,
            iterations=final_iteration + 1
        )

        self._history.append(result)

        logger.debug(
            f"Inhibition: {n} items -> {len(winners)} winners, "
            f"sparsity={sparsity:.2f}, iterations={final_iteration + 1}"
        )

        return result

    def _compute_similarity_matrix(
        self,
        ids: list[str],
        embeddings: dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            ids: List of memory IDs
            embeddings: Memory ID -> embedding

        Returns:
            Similarity matrix [n, n]
        """
        n = len(ids)
        matrix = np.zeros((n, n), dtype=np.float32)

        # Get normalized embeddings
        normalized: dict[str, np.ndarray] = {}
        for id_ in ids:
            if id_ in embeddings:
                emb = np.asarray(embeddings[id_], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    normalized[id_] = emb / norm

        # Compute similarities
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if i == j:
                    matrix[i, j] = 1.0
                elif id_i in normalized and id_j in normalized:
                    matrix[i, j] = float(np.dot(normalized[id_i], normalized[id_j]))

        return matrix

    def apply_lateral_inhibition(
        self,
        target_id: str,
        target_score: float,
        competitors: list[tuple[str, float, float]]  # (id, score, similarity)
    ) -> float:
        """
        Apply lateral inhibition from competitors to target.

        Args:
            target_id: ID of target memory
            target_score: Target's initial score
            competitors: List of (id, score, similarity) tuples

        Returns:
            Inhibited score for target
        """
        inhibition = 0.0

        for comp_id, comp_score, similarity in competitors:
            if comp_id == target_id:
                continue

            # Inhibition proportional to competitor's score and similarity
            inhibition += comp_score * similarity * self.inhibition_strength

        # Apply inhibition (bounded)
        inhibited_score = max(0.0, target_score - inhibition)

        return inhibited_score

    def sharpen_ranking(
        self,
        ranked_results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """
        Sharpen a ranked list by increasing separation.

        Higher-ranked items maintain scores; lower-ranked are suppressed.

        Args:
            ranked_results: List of (id, score) in descending order

        Returns:
            Re-ranked list with sharpened scores
        """
        if not ranked_results:
            return []

        # Convert to arrays
        ids = [r[0] for r in ranked_results]
        scores = np.array([r[1] for r in ranked_results], dtype=np.float32)

        # Compute rank-based suppression
        ranks = np.arange(len(scores))
        suppression = self.inhibition_strength * (ranks / len(scores))

        # Apply suppression
        sharpened = scores * (1 - suppression)
        sharpened = np.maximum(sharpened, 0.0)

        return [(ids[i], float(sharpened[i])) for i in range(len(ids))]

    def compute_sparsity(self, scores: dict[str, float]) -> float:
        """
        Compute current sparsity of score distribution.

        Args:
            scores: Memory ID -> score

        Returns:
            Sparsity [0, 1] where 1 = maximally sparse
        """
        if not scores:
            return 0.0

        values = np.array(list(scores.values()))
        if values.max() == 0:
            return 0.0

        # Normalized entropy as sparsity measure
        probs = values / values.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(values))

        # Low entropy = sparse, high entropy = dense
        sparsity = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        return float(sparsity)

    def get_stats(self) -> dict:
        """Get inhibitory network statistics."""
        if not self._history:
            return {
                "total_applications": 0,
                "avg_sparsity": 0.0,
                "avg_iterations": 0.0,
                "avg_winners": 0.0,
                "config": self.get_config(),
            }

        return {
            "total_applications": len(self._history),
            "avg_sparsity": float(np.mean([r.sparsity for r in self._history])),
            "avg_iterations": float(np.mean([r.iterations for r in self._history])),
            "avg_winners": float(np.mean([len(r.winners) for r in self._history])),
            "config": self.get_config(),
        }

    # ==================== Runtime Configuration Setters ====================

    def set_inhibition_strength(self, strength: float) -> None:
        """
        Set inhibition strength.

        Args:
            strength: How strongly winners suppress losers [0.0, 1.0]
        """
        self.inhibition_strength = float(np.clip(strength, 0.0, 1.0))
        logger.info(f"GABA inhibition_strength set to {self.inhibition_strength}")

    def set_sparsity_target(self, target: float) -> None:
        """
        Set target sparsity.

        Args:
            target: Target fraction of surviving items [0.05, 0.5]
        """
        self.sparsity_target = float(np.clip(target, 0.05, 0.5))
        logger.info(f"GABA sparsity_target set to {self.sparsity_target}")

    def set_temperature(self, temp: float) -> None:
        """
        Set softmax temperature.

        Args:
            temp: Competition temperature [0.1, 5.0]
        """
        self.temperature = float(np.clip(temp, 0.1, 5.0))
        logger.info(f"GABA temperature set to {self.temperature}")

    def set_max_iterations(self, iterations: int) -> None:
        """
        Set maximum competition iterations.

        Args:
            iterations: Max iterations [1, 20]
        """
        self.max_iterations = int(np.clip(iterations, 1, 20))
        logger.info(f"GABA max_iterations set to {self.max_iterations}")

    def set_convergence_threshold(self, threshold: float) -> None:
        """
        Set convergence threshold for early stopping.

        Args:
            threshold: Threshold [0.001, 0.1]
        """
        self.convergence_threshold = float(np.clip(threshold, 0.001, 0.1))
        logger.info(f"GABA convergence_threshold set to {self.convergence_threshold}")

    def set_similarity_inhibition(self, enabled: bool) -> None:
        """
        Enable/disable similarity-based inhibition.

        Args:
            enabled: Whether similar items inhibit each other
        """
        self.similarity_inhibition = bool(enabled)
        logger.info(f"GABA similarity_inhibition set to {self.similarity_inhibition}")

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "inhibition_strength": self.inhibition_strength,
            "sparsity_target": self.sparsity_target,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "similarity_inhibition": self.similarity_inhibition,
        }

    def reset_history(self) -> None:
        """Clear history."""
        self._history.clear()


class SparseRetrieval:
    """
    Sparse retrieval layer that applies GABA-like dynamics.

    Wraps a retrieval function and applies competitive inhibition
    to sharpen the output.
    """

    def __init__(
        self,
        inhibitory_network: InhibitoryNetwork | None = None,
        min_score_threshold: float = 0.1,
        max_results: int = 10
    ):
        """
        Initialize sparse retrieval layer.

        Args:
            inhibitory_network: Network for applying inhibition
            min_score_threshold: Minimum score to return
            max_results: Maximum results after sparsification
        """
        self.inhibitory = inhibitory_network or InhibitoryNetwork()
        self.min_score_threshold = min_score_threshold
        self.max_results = max_results

    def sparsify_results(
        self,
        results: list[tuple[str, float]],
        embeddings: dict[str, np.ndarray] | None = None
    ) -> list[tuple[str, float]]:
        """
        Apply sparse coding to retrieval results.

        Args:
            results: List of (id, score) tuples
            embeddings: Optional embeddings for similarity-based inhibition

        Returns:
            Sparsified results (fewer items, sharper distribution)
        """
        if not results:
            return []

        # Convert to dict
        scores = {r[0]: r[1] for r in results}

        # Apply inhibition
        result = self.inhibitory.apply_inhibition(scores, embeddings)

        # Filter and sort
        filtered = [
            (id_, score)
            for id_, score in result.inhibited_scores.items()
            if score >= self.min_score_threshold
        ]

        # Sort by score descending
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Limit results
        return filtered[:self.max_results]

    def sparsify_dict(
        self,
        scores: dict[str, float],
        embeddings: dict[str, np.ndarray] | None = None
    ) -> dict[str, float]:
        """
        Apply sparse coding to score dictionary.

        Args:
            scores: Memory ID -> score
            embeddings: Optional embeddings for similarity-based inhibition

        Returns:
            Sparsified scores
        """
        if not scores:
            return {}

        result = self.inhibitory.apply_inhibition(scores, embeddings)

        # Filter by threshold
        return {
            id_: score
            for id_, score in result.inhibited_scores.items()
            if score >= self.min_score_threshold
        }


__all__ = [
    "InhibitionResult",
    "InhibitoryNetwork",
    "SparseRetrieval",
]
