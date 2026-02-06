"""
Markov Blanket Retrieval (W3-01).

Retrieve memories prioritizing the Markov blanket of the query concept.
The MB provides statistical insulation - memories outside MB are conditionally
independent given MB members.

Evidence Base:
- Pearl (1988) "Probabilistic Reasoning in Intelligent Systems"
- Friston (2010) "The free-energy principle: a unified brain theory?"

Key Insight:
    Markov blanket = parents + children + spouses (co-parents of children)
    p(X | MB(X)) ⊥ p(X | non-MB) - memories outside MB are irrelevant given MB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class MarkovBlanketConfig:
    """Configuration for Markov blanket retrieval.

    Attributes:
        exploration_ratio: Fraction of results from global search (default 0.1).
        edge_type: Edge type defining causal relationships (default "CAUSES").
    """

    exploration_ratio: float = 0.1
    edge_type: str = "CAUSES"


class MarkovBlanketRetriever:
    """Retrieve memories prioritizing Markov blanket of query concept.

    The Markov blanket provides statistical insulation: once you condition
    on MB members, all other nodes become irrelevant. This focuses retrieval
    on truly relevant memories while maintaining some exploration.

    Example:
        >>> retriever = MarkovBlanketRetriever(engine, exploration_ratio=0.1)
        >>> results = retriever.search(query_vector, query_concept=concept_id, k=10)
    """

    def __init__(
        self,
        engine: Any,
        exploration_ratio: float = 0.1,
        edge_type: str = "CAUSES",
    ):
        """Initialize Markov blanket retriever.

        Args:
            engine: Storage engine with traverse() and search() methods.
            exploration_ratio: Fraction of results from global search (0-1).
            edge_type: Edge type for causal relationships.
        """
        self.engine = engine
        self.exploration_ratio = exploration_ratio
        self.edge_type = edge_type

    def search(
        self,
        query_vector: np.ndarray,
        query_concept: Optional[UUID] = None,
        k: int = 10,
    ) -> list[Any]:
        """Retrieve k memories, prioritizing Markov blanket.

        If query_concept is provided, 90% of results come from MB and
        10% from global search (exploration). Otherwise, uses pure
        vector search.

        Args:
            query_vector: Query embedding vector.
            query_concept: Optional concept ID for MB computation.
            k: Number of results to return.

        Returns:
            List of memory items, ranked by relevance.
        """
        if query_concept is None:
            # No concept provided, use pure vector search
            return self.engine.search(query_vector, k)

        # Get Markov blanket
        mb = self._get_markov_blanket(query_concept)

        if not mb:
            # Empty MB, fall back to global
            return self.engine.search(query_vector, k)

        # Retrieve from MB
        mb_k = int(k * (1 - self.exploration_ratio))
        mb_results = self._search_in_set(query_vector, mb, mb_k)

        # Retrieve from global (exploration)
        explore_k = k - len(mb_results)
        global_results = self.engine.search(query_vector, explore_k * 2)

        # Filter out MB items from global
        global_results = [r for r in global_results if r.id not in mb]
        global_results = global_results[:explore_k]

        # Combine
        return mb_results + global_results

    def _get_markov_blanket(self, concept_id: UUID) -> set[UUID]:
        """Get Markov blanket: parents + children + spouses.

        MB(X) = Pa(X) ∪ Ch(X) ∪ {Y : Y ∈ Pa(Z), Z ∈ Ch(X), Y ≠ X}

        Args:
            concept_id: The node to compute MB for.

        Returns:
            Set of UUIDs in the Markov blanket.
        """
        mb = set()

        # Parents: concepts that CAUSE this one
        parents = self.engine.traverse(
            concept_id,
            edge_type=self.edge_type,
            direction="incoming",
            depth=1,
        )
        parent_ids = {p.id for p in parents}
        mb.update(parent_ids)

        # Children: concepts caused by this one
        children = self.engine.traverse(
            concept_id,
            edge_type=self.edge_type,
            direction="outgoing",
            depth=1,
        )
        child_ids = {c.id for c in children}
        mb.update(child_ids)

        # Spouses: other parents of children (co-parents)
        for child in children:
            child_parents = self.engine.traverse(
                child.id,
                edge_type=self.edge_type,
                direction="incoming",
                depth=1,
            )
            for p in child_parents:
                if p.id != concept_id:
                    mb.add(p.id)

        return mb

    def _search_in_set(
        self,
        query_vector: np.ndarray,
        id_set: set[UUID],
        k: int,
    ) -> list[Any]:
        """Search within a specific set of IDs.

        Args:
            query_vector: Query embedding.
            id_set: Set of IDs to search within.
            k: Number of results.

        Returns:
            List of items from id_set, ranked by similarity.
        """
        # Use engine's filtered search if available
        if hasattr(self.engine, "search_filtered"):
            return self.engine.search_filtered(query_vector, list(id_set), k)

        # Otherwise, use regular search and filter
        results = self.engine.search(query_vector, k * 3)
        filtered = [r for r in results if r.id in id_set]
        return filtered[:k]
