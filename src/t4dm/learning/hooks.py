"""
Learning hooks for T4DM retrieval.

Provides decorators and functions to hook into retrieval methods
and emit RetrievalEvents for the learning system.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any
from uuid import UUID

from t4dm.learning.collector import EventCollector, get_collector
from t4dm.learning.events import (
    MemoryType,
    RetrievalEvent,
)

logger = logging.getLogger(__name__)

# Global collector instance (lazy-loaded)
_collector: EventCollector | None = None


def get_learning_collector() -> EventCollector | None:
    """Get the global learning collector, initializing if needed."""
    global _collector
    if _collector is None:
        try:
            _collector = get_collector()
            logger.info("Learning collector initialized for retrieval hooks")
        except Exception as e:
            logger.debug(f"Learning collector not available: {e}")
            return None
    return _collector


def emit_retrieval_event(
    query: str,
    memory_type: MemoryType,
    results: list[dict[str, Any]],
    session_id: str = "",
    project: str = "",
    context: str = "",
) -> RetrievalEvent | None:
    """
    Emit a RetrievalEvent from retrieval results.

    Args:
        query: The search query
        memory_type: Type of memory searched
        results: List of result dicts with 'id', 'score', and optionally 'components'
        session_id: Current session ID
        project: Current project context
        context: Optional context string for hash computation

    Returns:
        The created RetrievalEvent or None if collector unavailable
    """
    collector = get_learning_collector()
    if collector is None:
        return None

    try:
        # Extract IDs and scores from results
        retrieved_ids: list[UUID] = []
        retrieval_scores: dict[str, float] = {}
        component_scores: dict[str, dict[str, float]] = {}

        for result in results:
            result_id = result.get("id")
            if result_id:
                try:
                    uid = UUID(result_id) if isinstance(result_id, str) else result_id
                    retrieved_ids.append(uid)
                except (ValueError, TypeError):
                    continue

            # Store score
            score = result.get("score", 0.0)
            if result_id:
                retrieval_scores[str(result_id)[:8]] = score

            # Store component scores if available
            components = result.get("components", {})
            if components and result_id:
                component_scores[str(result_id)[:8]] = {
                    k: float(v) for k, v in components.items()
                    if isinstance(v, (int, float))
                }

        # Record event using collector API
        event = collector.record_retrieval(
            query=query,
            memory_type=memory_type,
            retrieved_ids=retrieved_ids,
            retrieval_scores=retrieval_scores,
            component_scores=component_scores if component_scores else None,
            context=context if context else None,
            session_id=session_id,
            project=project,
        )

        logger.debug(
            f"Emitted RetrievalEvent: query='{query[:50]}', "
            f"type={memory_type.value}, count={len(retrieved_ids)}"
        )

        return event

    except Exception as e:
        logger.error(f"Failed to emit RetrievalEvent: {e}")
        return None


def emit_unified_retrieval_event(
    query: str,
    results_by_type: dict[str, list[dict[str, Any]]],
    session_id: str = "",
    project: str = "",
    context: str = "",
) -> list[RetrievalEvent]:
    """
    Emit RetrievalEvents for unified search results.

    Emits one event per memory type searched.

    Args:
        query: The search query
        results_by_type: Dict mapping memory type to result list
        session_id: Current session ID
        project: Current project context
        context: Optional context string for hash computation

    Returns:
        List of created RetrievalEvents
    """
    events = []

    type_map = {
        "episodic": MemoryType.EPISODIC,
        "semantic": MemoryType.SEMANTIC,
        "procedural": MemoryType.PROCEDURAL,
    }

    for type_name, results in results_by_type.items():
        if results and type_name in type_map:
            event = emit_retrieval_event(
                query=query,
                memory_type=type_map[type_name],
                results=results,
                session_id=session_id,
                project=project,
                context=context,
            )
            if event:
                events.append(event)

    return events


def learning_retrieval(memory_type: MemoryType = MemoryType.EPISODIC):
    """
    Decorator to emit learning events from retrieval methods.

    The decorated function should return a list of ScoredResult or similar
    objects with 'id', 'score' attributes.

    Args:
        memory_type: Type of memory being retrieved

    Usage:
        @learning_retrieval(MemoryType.EPISODIC)
        async def recall(self, query: str, ...) -> list[ScoredResult]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute the retrieval
            results = await func(*args, **kwargs)

            # Try to emit event
            try:
                # Extract query from kwargs or first arg after self
                query = kwargs.get("query") or kwargs.get("task", "")
                if not query and len(args) > 1:
                    query = str(args[1])

                # Extract session_id
                session_id = kwargs.get("session_filter", "") or kwargs.get("session_id", "")

                # Convert results to dicts
                result_dicts = []
                for r in results:
                    if hasattr(r, "item") and hasattr(r, "score"):
                        # ScoredResult
                        result_dicts.append({
                            "id": str(r.item.id) if hasattr(r.item, "id") else "",
                            "score": r.score,
                            "components": getattr(r, "components", {}),
                        })
                    elif isinstance(r, dict):
                        result_dicts.append(r)

                if result_dicts:
                    emit_retrieval_event(
                        query=query,
                        memory_type=memory_type,
                        results=result_dicts,
                        session_id=session_id,
                    )

            except Exception as e:
                logger.debug(f"Learning hook failed (non-fatal): {e}")

            return results

        return wrapper
    return decorator


class RetrievalHookMixin:
    """
    Mixin to add learning hooks to memory services.

    Add this mixin to memory classes to automatically emit
    RetrievalEvents when retrieval methods are called.

    Usage:
        class EpisodicMemoryWithLearning(RetrievalHookMixin, EpisodicMemory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._hook_memory_type = MemoryType.EPISODIC
    """

    _hook_memory_type: MemoryType = MemoryType.EPISODIC
    _hook_enabled: bool = True

    def _emit_hook(
        self,
        query: str,
        results: list,
        session_id: str = "",
    ) -> RetrievalEvent | None:
        """Emit learning event for a retrieval."""
        if not self._hook_enabled:
            return None

        result_dicts = []
        for r in results:
            if hasattr(r, "item") and hasattr(r, "score"):
                result_dicts.append({
                    "id": str(r.item.id) if hasattr(r.item, "id") else "",
                    "score": r.score,
                    "components": getattr(r, "components", {}),
                })
            elif isinstance(r, dict):
                result_dicts.append(r)

        return emit_retrieval_event(
            query=query,
            memory_type=self._hook_memory_type,
            results=result_dicts,
            session_id=session_id,
        )


# Export
__all__ = [
    "RetrievalHookMixin",
    "emit_retrieval_event",
    "emit_unified_retrieval_event",
    "get_learning_collector",
    "learning_retrieval",
]
