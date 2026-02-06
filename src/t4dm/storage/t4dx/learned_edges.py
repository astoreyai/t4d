"""
Learned Edge Importance (W3-02).

Learn importance weights for each edge type during graph traversal.
Edges that lead to relevant results get higher weights.

Evidence Base: Graves et al. (2014) "Neural Turing Machines"

Key Insight:
    Not all edge types are equally useful for retrieval. By learning
    importance weights from feedback, traversal can prioritize paths
    that have historically led to relevant memories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LearnedEdgeImportance(nn.Module):
    """Learn importance weights for each edge type.

    During traversal, weight edges by learned importance.
    Uses exponential activation to ensure positive weights.

    Example:
        >>> edge_importance = LearnedEdgeImportance()
        >>> weight = edge_importance.get_weight("CAUSES")
        >>> # Use in traversal to prioritize edges
    """

    EDGE_TYPES = [
        "CAUSES", "TEMPORAL_BEFORE", "PART_OF", "SIMILAR_TO",
        "CONTRADICTS", "ELABORATES", "EXEMPLIFIES", "GENERALIZES",
        "PRECONDITION", "EFFECT", "ATTRIBUTE", "CONTEXT",
        "REFERENCES", "DERIVES_FROM", "SUPPORTS", "OPPOSES", "NEUTRAL"
    ]

    def __init__(self):
        """Initialize edge importance with uniform weights."""
        super().__init__()
        # Initialize to zero so exp(0) = 1 (uniform)
        self.importance = nn.Embedding(len(self.EDGE_TYPES), 1)
        nn.init.zeros_(self.importance.weight)

        # Build type to index mapping
        self._type_to_idx = {t: i for i, t in enumerate(self.EDGE_TYPES)}

    def get_weight(self, edge_type: str) -> float:
        """Get learned importance weight for edge type.

        Args:
            edge_type: The edge type string.

        Returns:
            Positive weight (exponential of learned parameter).

        Raises:
            ValueError: If edge_type is unknown.
        """
        if edge_type not in self._type_to_idx:
            raise ValueError(f"Unknown edge type: {edge_type}")

        idx = self._type_to_idx[edge_type]
        with torch.no_grad():
            weight = torch.exp(self.importance(torch.tensor(idx))).item()
        return weight

    def forward(self, edge_types: torch.Tensor) -> torch.Tensor:
        """Get importance weights for batch of edge type indices.

        Args:
            edge_types: Tensor of edge type indices [batch].

        Returns:
            Positive weights [batch].
        """
        return torch.exp(self.importance(edge_types)).squeeze(-1)


@dataclass
class TrainerConfig:
    """Configuration for EdgeImportanceTrainer."""
    lr: float = 0.01


class EdgeImportanceTrainer:
    """Train edge importance from retrieval feedback.

    Updates edge weights based on which edges led to relevant results.

    Example:
        >>> trainer = EdgeImportanceTrainer(edge_importance, lr=0.01)
        >>> trainer.train_step(["CAUSES", "SIMILAR_TO"], [0.9, 0.3])
    """

    def __init__(self, edge_importance: LearnedEdgeImportance, lr: float = 0.01):
        """Initialize trainer.

        Args:
            edge_importance: The LearnedEdgeImportance module to train.
            lr: Learning rate.
        """
        self.edge_importance = edge_importance
        self.optimizer = torch.optim.Adam(edge_importance.parameters(), lr=lr)

    def train_step(
        self,
        traversal_edges: list[str],
        relevance: list[float],
    ) -> float:
        """Single training step from feedback.

        Increases weights for edges that led to relevant results,
        decreases for irrelevant ones.

        Args:
            traversal_edges: List of edge types used in traversal.
            relevance: Corresponding relevance scores [0, 1].

        Returns:
            Loss value for this step.
        """
        if not traversal_edges:
            return 0.0

        self.optimizer.zero_grad()

        # Convert to tensors
        edge_indices = torch.tensor([
            self.edge_importance._type_to_idx[e] for e in traversal_edges
        ])
        targets = torch.tensor(relevance, dtype=torch.float32)

        # Get predicted weights (normalized by max for stability)
        weights = self.edge_importance(edge_indices)
        normalized_weights = weights / (weights.max() + 1e-8)

        # MSE loss: push weights toward relevance
        loss = F.mse_loss(normalized_weights, targets)

        loss.backward()
        self.optimizer.step()

        return loss.item()


class TraversalWithLearnedEdges:
    """Traversal that weights edges by learned importance.

    BFS traversal where edge scores are multiplied by learned importance.
    High-importance edges contribute more to result ranking.

    Example:
        >>> traversal = TraversalWithLearnedEdges(engine, edge_importance)
        >>> results = traversal.traverse(start_id, depth=2)
    """

    def __init__(self, engine: Any, edge_importance: LearnedEdgeImportance):
        """Initialize traversal.

        Args:
            engine: Storage engine with get_edges() and get() methods.
            edge_importance: Learned edge importance module.
        """
        self.engine = engine
        self.edge_importance = edge_importance

    def traverse(
        self,
        start_id: UUID,
        depth: int = 2,
        min_weight: float = 0.1,
    ) -> list[Any]:
        """BFS traversal with learned edge weighting.

        Candidate score = base_score * edge_importance

        Args:
            start_id: Starting node ID.
            depth: Maximum traversal depth.
            min_weight: Minimum accumulated weight to include result.

        Returns:
            List of memory items sorted by traversal score.
        """
        visited = {start_id}
        frontier = [(start_id, 1.0)]  # (id, accumulated_weight)
        results = []

        for d in range(depth):
            new_frontier = []

            for node_id, acc_weight in frontier:
                edges = self.engine.get_edges(node_id)

                for edge in edges:
                    if edge.target_id in visited:
                        continue

                    # Apply learned importance
                    edge_weight = self.edge_importance.get_weight(edge.edge_type)
                    new_acc = acc_weight * edge_weight * edge.weight

                    if new_acc >= min_weight:
                        visited.add(edge.target_id)
                        new_frontier.append((edge.target_id, new_acc))

                        target = self.engine.get(edge.target_id)
                        target.traversal_score = new_acc
                        results.append(target)

            frontier = new_frontier

        return sorted(results, key=lambda x: x.traversal_score, reverse=True)
