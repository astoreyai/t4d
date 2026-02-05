"""
Unified Memory Store with κ-based routing.

Replaces the separate episodic/semantic/procedural stores with a single
unified store where memory type is determined by the κ consolidation gradient.

Key Features:
- Single storage backend (T4DX) for all memory types
- κ-based query routing using query_policies
- Automatic type inference from κ value
- Consolidation-aware retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from t4dm.core.query_policies import (
    EpisodicPolicy,
    ProceduralPolicy,
    QueryFilters,
    SemanticPolicy,
    select_policy,
)
from t4dm.core.unified_memory import UnifiedMemoryItem, convert_to_unified

if TYPE_CHECKING:
    from t4dm.core.types import Episode, Entity, Procedure
    from t4dm.storage.t4dx.engine import T4DXEngine

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from unified store retrieval."""

    items: list[UnifiedMemoryItem]
    scores: list[float]
    query_filters: QueryFilters
    total_candidates: int
    retrieval_time_ms: float


class UnifiedMemoryStore:
    """
    Unified memory store with κ-based routing.

    This store provides a single interface for storing and retrieving
    memories of all types (episodic, semantic, procedural) using the
    κ consolidation gradient for routing and filtering.

    Usage:
        store = UnifiedMemoryStore(engine)

        # Store any memory type
        store.store(episode)
        store.store(entity)
        store.store(procedure)

        # Query with automatic κ routing
        results = store.query("search text", policy="episodic")

        # Or direct κ-range query
        results = store.query_kappa_range(embedding, kappa_min=0.3, kappa_max=0.7)
    """

    def __init__(
        self,
        engine: T4DXEngine | None = None,
        default_policy: str = "episodic",
    ):
        """
        Initialize unified memory store.

        Args:
            engine: T4DX storage engine (creates mock if None)
            default_policy: Default query policy ('episodic', 'semantic', 'procedural')
        """
        self.engine = engine
        self.default_policy = default_policy

        # In-memory fallback for testing
        self._memory: dict[UUID, UnifiedMemoryItem] = {}

        logger.info(f"UnifiedMemoryStore initialized (default_policy={default_policy})")

    def store(
        self,
        item: UnifiedMemoryItem | Episode | Entity | Procedure,
        tau_value: float = 0.5,
    ) -> UUID:
        """
        Store a memory item.

        Accepts any memory type and converts to UnifiedMemoryItem.

        Args:
            item: Memory item to store
            tau_value: τ(t) gate value at encoding time

        Returns:
            UUID of stored item
        """
        # Convert to unified format
        unified = convert_to_unified(item)

        # Set tau value if not already set
        if hasattr(unified, "tau_value") and unified.tau_value == 0.5:
            unified.tau_value = tau_value

        # Store in engine or memory
        if self.engine is not None:
            self._store_to_engine(unified)
        else:
            self._memory[unified.id] = unified

        logger.debug(f"Stored item {unified.id} with κ={unified.kappa:.2f}")
        return unified.id

    def _store_to_engine(self, item: UnifiedMemoryItem) -> None:
        """Store item in T4DX engine."""
        # Convert to T4DX Item format
        from t4dm.storage.t4dx.memory_adapter import memory_item_to_t4dx

        t4dx_item = memory_item_to_t4dx(item)
        self.engine.insert(t4dx_item, edges=[])

    def get(self, item_id: UUID) -> UnifiedMemoryItem | None:
        """
        Get a specific memory item by ID.

        Args:
            item_id: UUID of the item

        Returns:
            UnifiedMemoryItem or None if not found
        """
        if self.engine is not None:
            return self._get_from_engine(item_id)
        return self._memory.get(item_id)

    def _get_from_engine(self, item_id: UUID) -> UnifiedMemoryItem | None:
        """Get item from T4DX engine."""
        from t4dm.storage.t4dx.memory_adapter import t4dx_to_memory_item

        t4dx_item = self.engine.get(item_id.bytes)
        if t4dx_item is None:
            return None
        return t4dx_to_memory_item(t4dx_item)

    def query(
        self,
        query_text: str,
        embedding: list[float] | None = None,
        policy: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Query memories using κ-based policy routing.

        Args:
            query_text: Text query (for embedding if not provided)
            embedding: Pre-computed embedding vector
            policy: Query policy ('episodic', 'semantic', 'procedural', or None for auto)
            limit: Maximum results to return
            **kwargs: Additional filter arguments

        Returns:
            RetrievalResult with matching items
        """
        import time

        start = time.perf_counter()

        # Select policy
        policy_name = policy or self.default_policy
        query_policy = select_policy(policy_name)

        # Get filters from policy
        if isinstance(query_policy, EpisodicPolicy):
            filters = query_policy.filters()
        else:
            filters = query_policy.filters()

        # Override with kwargs
        if "kappa_min" in kwargs:
            filters.kappa_min = kwargs["kappa_min"]
        if "kappa_max" in kwargs:
            filters.kappa_max = kwargs["kappa_max"]
        if "session_id" in kwargs:
            filters.session_id = kwargs["session_id"]

        # Execute query
        if self.engine is not None:
            items, scores = self._query_engine(embedding, filters, limit)
        else:
            items, scores = self._query_memory(query_text, filters, limit)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            items=items,
            scores=scores,
            query_filters=filters,
            total_candidates=len(self._memory) if self.engine is None else -1,
            retrieval_time_ms=elapsed_ms,
        )

    def _query_engine(
        self,
        embedding: list[float] | None,
        filters: QueryFilters,
        limit: int,
    ) -> tuple[list[UnifiedMemoryItem], list[float]]:
        """Query T4DX engine."""
        from t4dm.storage.t4dx.memory_adapter import t4dx_to_memory_item

        # Build T4DX filter dict
        t4dx_filters = {}
        if filters.kappa_min is not None:
            t4dx_filters["kappa_min"] = filters.kappa_min
        if filters.kappa_max is not None:
            t4dx_filters["kappa_max"] = filters.kappa_max
        if filters.item_type is not None:
            t4dx_filters["item_type"] = filters.item_type

        # Search
        results = self.engine.search(
            vector=embedding,
            k=limit,
            filters=t4dx_filters,
        )

        items = [t4dx_to_memory_item(r.item) for r in results]
        scores = [r.score for r in results]

        return items, scores

    def _query_memory(
        self,
        query_text: str,
        filters: QueryFilters,
        limit: int,
    ) -> tuple[list[UnifiedMemoryItem], list[float]]:
        """Query in-memory store (for testing)."""
        results = []

        for item in self._memory.values():
            # Apply filters
            if filters.kappa_min is not None and item.kappa < filters.kappa_min:
                continue
            if filters.kappa_max is not None and item.kappa > filters.kappa_max:
                continue
            if filters.item_type is not None and item.item_type != filters.item_type:
                continue
            if filters.session_id is not None and item.session_id != filters.session_id:
                continue

            # Simple text matching score
            score = self._text_similarity(query_text, item.content)
            results.append((item, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]

        items = [r[0] for r in results]
        scores = [r[1] for r in results]

        return items, scores

    def _text_similarity(self, query: str, content: str) -> float:
        """Simple word overlap similarity."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words or not content_words:
            return 0.0

        overlap = len(query_words & content_words)
        return overlap / max(len(query_words), len(content_words))

    def query_kappa_range(
        self,
        embedding: list[float] | None = None,
        kappa_min: float = 0.0,
        kappa_max: float = 1.0,
        limit: int = 10,
    ) -> RetrievalResult:
        """
        Query memories within a specific κ range.

        Args:
            embedding: Query embedding vector
            kappa_min: Minimum κ value
            kappa_max: Maximum κ value
            limit: Maximum results

        Returns:
            RetrievalResult with matching items
        """
        return self.query(
            query_text="",
            embedding=embedding,
            policy=None,
            limit=limit,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
        )

    def get_consolidation_candidates(
        self,
        kappa_max: float = 0.3,
        min_importance: float = 0.3,
        limit: int = 100,
    ) -> list[UnifiedMemoryItem]:
        """
        Get items that should be consolidated.

        Returns low-κ, high-importance items for NREM replay.

        Args:
            kappa_max: Maximum κ for candidates
            min_importance: Minimum importance score
            limit: Maximum candidates to return

        Returns:
            List of consolidation candidates
        """
        candidates = []

        if self.engine is not None:
            # Use engine scan
            for item in self.engine.scan(
                filters={"kappa_max": kappa_max},
                limit=limit * 2,  # Over-fetch for filtering
            ):
                unified = self._get_from_engine(item.id)
                if unified and unified.importance >= min_importance:
                    candidates.append(unified)
                if len(candidates) >= limit:
                    break
        else:
            # In-memory scan
            for item in self._memory.values():
                if item.kappa <= kappa_max and item.importance >= min_importance:
                    candidates.append(item)
                    if len(candidates) >= limit:
                        break

        # Sort by importance (highest first)
        candidates.sort(key=lambda x: x.importance, reverse=True)
        return candidates[:limit]

    def update_kappa(
        self,
        item_id: UUID,
        delta: float,
        phase: str = "consolidation",
    ) -> bool:
        """
        Update κ for an item.

        Args:
            item_id: Item to update
            delta: Change in κ (positive for consolidation)
            phase: Consolidation phase name

        Returns:
            True if successful
        """
        item = self.get(item_id)
        if item is None:
            return False

        item.update_kappa(delta=delta, phase=phase)

        # Re-store updated item
        if self.engine is not None:
            self._store_to_engine(item)
        else:
            self._memory[item_id] = item

        return True

    def batch_update_kappa(
        self,
        item_ids: list[UUID],
        delta: float,
        phase: str = "consolidation",
    ) -> int:
        """
        Batch update κ for multiple items.

        Args:
            item_ids: Items to update
            delta: Change in κ
            phase: Consolidation phase

        Returns:
            Number of items updated
        """
        updated = 0
        for item_id in item_ids:
            if self.update_kappa(item_id, delta, phase):
                updated += 1
        return updated

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        if self.engine is not None:
            return self.engine.get_stats()

        # In-memory stats
        kappa_dist = {"episodic": 0, "transitional": 0, "semantic": 0}
        for item in self._memory.values():
            if item.kappa < 0.3:
                kappa_dist["episodic"] += 1
            elif item.kappa < 0.7:
                kappa_dist["transitional"] += 1
            else:
                kappa_dist["semantic"] += 1

        return {
            "total_items": len(self._memory),
            "kappa_distribution": kappa_dist,
            "engine": "in_memory",
        }


# Convenience function
def create_unified_store(
    engine: T4DXEngine | None = None,
) -> UnifiedMemoryStore:
    """Create a unified memory store."""
    return UnifiedMemoryStore(engine=engine)


__all__ = [
    "UnifiedMemoryStore",
    "RetrievalResult",
    "create_unified_store",
]
