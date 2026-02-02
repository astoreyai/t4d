"""
Unified Memory Service for T4DM.

Coordinates cross-memory search across episodic, semantic, and procedural subsystems.
"""

import asyncio
import logging
from uuid import UUID

from opentelemetry.trace import SpanKind

# P7.1: Bridge container for NCA subsystem integration
from t4dm.core.bridge_container import get_bridge_container
from t4dm.core.types import ScoredResult
from t4dm.memory.episodic import EpisodicMemory
from t4dm.memory.procedural import ProceduralMemory
from t4dm.memory.semantic import SemanticMemory
from t4dm.observability.tracing import add_span_attribute, traced

# Lazy import for learning hooks to avoid circular deps
_learning_hooks_available = None

logger = logging.getLogger(__name__)


def _get_learning_hooks():
    """Lazy load learning hooks module."""
    global _learning_hooks_available
    if _learning_hooks_available is None:
        try:
            from t4dm.learning.hooks import emit_unified_retrieval_event
            _learning_hooks_available = emit_unified_retrieval_event
        except ImportError:
            _learning_hooks_available = False
    return _learning_hooks_available if _learning_hooks_available else None


class UnifiedMemoryService:
    """
    Coordinates search across all memory subsystems.

    Provides parallel search, unified relevance ranking, and graph-based
    relationship traversal across episodic, semantic, and procedural memory.
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
    ):
        """
        Initialize unified memory service.

        Args:
            episodic: Episodic memory service instance
            semantic: Semantic memory service instance
            procedural: Procedural memory service instance
        """
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural

        # P7.1: Bridge container for NCA bridge integration
        # NCA bridge provides state-dependent memory operations
        session_id = getattr(episodic, 'session_id', 'default')
        self._bridge_container = get_bridge_container(session_id)

        # Config for NCA-modulated retrieval
        from t4dm.core.config import get_settings
        settings = get_settings()
        self._nca_modulation_enabled = getattr(settings, "nca_modulation_enabled", True)

    @traced("unified.search", kind=SpanKind.INTERNAL)
    async def search(
        self,
        query: str,
        k: int = 10,
        memory_types: list[str] | None = None,
        min_score: float = 0.0,
        session_id: str | None = None,
    ) -> dict:
        """
        Execute parallel search across memory types.

        Searches episodic, semantic, and procedural memory simultaneously using
        asyncio.gather for optimal performance. Results are filtered by minimum
        score and ranked by relevance.

        Args:
            query: Search query text
            k: Maximum results per memory type (default 10)
            memory_types: Filter to specific types (episodic, semantic, procedural)
            min_score: Minimum relevance score threshold (default 0.0)
            session_id: Optional session context for filtering

        Returns:
            Dict with results, by_type breakdown, total_count, and query
        """
        # Default to all memory types
        types = memory_types or ["episodic", "semantic", "procedural"]

        # Build parallel search tasks
        tasks = []
        task_names = []

        if "episodic" in types:
            tasks.append(self.episodic.recall(
                query=query,
                limit=k,
                session_filter=session_id,
            ))
            task_names.append("episodic")

        if "semantic" in types:
            tasks.append(self.semantic.recall(
                query=query,
                limit=k,
                session_filter=session_id,
            ))
            task_names.append("semantic")

        if "procedural" in types:
            tasks.append(self.procedural.recall_skill(
                task=query,
                limit=k,
                session_filter=session_id,
            ))
            task_names.append("procedural")

        # Execute parallel search with exception handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and build response
        all_results = []
        by_type = {}

        for idx, result in enumerate(results):
            memory_type = task_names[idx]

            if isinstance(result, Exception):
                logger.error(f"Error searching {memory_type} memory: {result}")
                by_type[memory_type] = []
                continue

            # Filter by min_score and convert to unified format
            filtered = [r for r in result if r.score >= min_score]

            # Convert to unified format
            for scored_result in filtered:
                unified_result = self._to_unified_result(scored_result, memory_type)
                all_results.append(unified_result)

            # Store by_type breakdown
            by_type[memory_type] = [
                self._to_dict(scored_result, memory_type)
                for scored_result in filtered
            ]

        # Sort all results by score (descending)
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # P7.1: NCA Bridge for state-dependent retrieval modulation
        # Modulates results based on current cognitive state (FOCUS/EXPLORE/REST)
        if self._nca_modulation_enabled and all_results:
            try:
                nca_bridge = self._bridge_container.get_nca_bridge()
                if nca_bridge is not None:
                    # Get current cognitive state
                    cognitive_state = nca_bridge.get_current_cognitive_state()
                    nt_state = nca_bridge.get_current_nt_state()

                    # Track state in observability
                    add_span_attribute("nca.cognitive_state", str(cognitive_state) if cognitive_state else "none")
                    add_span_attribute("nca.nt_state", ",".join(f"{v:.2f}" for v in nt_state[:3]))

                    # Apply cognitive state modulation
                    if cognitive_state is not None:
                        from t4dm.nca.attractors import CognitiveState

                        if cognitive_state == CognitiveState.FOCUS:
                            # In FOCUS: Boost top results, suppress lower ones
                            for i, result in enumerate(all_results):
                                # Concentration: top results get boost, tail gets suppressed
                                boost = max(0.0, 1.0 - (i / len(all_results)) * 0.3)
                                result["score"] *= boost
                                result["nca_boost"] = boost

                        elif cognitive_state == CognitiveState.EXPLORE:
                            # In EXPLORE: Flatten scoring to encourage diversity
                            import random
                            for result in all_results:
                                # Add random exploration noise
                                noise = random.uniform(-0.1, 0.2)
                                result["score"] *= (1.0 + noise)
                                result["nca_boost"] = 1.0 + noise

                        # Re-sort after modulation
                        all_results.sort(key=lambda x: x["score"], reverse=True)

                        logger.debug(
                            f"P7.1: NCA modulation applied - state={cognitive_state.name}"
                        )
            except Exception as e:
                logger.warning(f"NCA modulation failed: {e}")

        # Add observability metrics
        add_span_attribute("search.query", query)
        add_span_attribute("search.total_count", len(all_results))
        add_span_attribute("search.memory_types", ",".join(types))

        logger.info(
            f"Unified search for '{query}': {len(all_results)} results "
            f"(episodic={len(by_type.get('episodic', []))}, "
            f"semantic={len(by_type.get('semantic', []))}, "
            f"procedural={len(by_type.get('procedural', []))})"
        )

        # Emit learning events for retrieval
        emit_fn = _get_learning_hooks()
        if emit_fn and all_results:
            try:
                emit_fn(
                    query=query,
                    results_by_type=by_type,
                    session_id=session_id or "",
                )
            except Exception as e:
                logger.debug(f"Learning hook failed (non-fatal): {e}")

        return {
            "query": query,
            "results": all_results,
            "by_type": by_type,
            "total_count": len(all_results),
            "session_id": session_id,
        }

    @traced("unified.get_related", kind=SpanKind.INTERNAL)
    async def get_related(
        self,
        memory_id: str,
        memory_type: str,
        depth: int = 1,
        session_id: str | None = None,
    ) -> dict:
        """
        Get related memories across all types via graph traversal.

        Uses graph relationships to find connected memories. For episodic
        memories, finds entities extracted from them. For semantic entities,
        finds related entities and episodes that mention them. For procedures,
        finds related skills and episodes where they were used.

        Args:
            memory_id: UUID of the source memory
            memory_type: Type of source memory (episodic, semantic, procedural)
            depth: Graph traversal depth (default 1)
            session_id: Optional session context

        Returns:
            Dict with related memories grouped by type
        """
        memory_uuid = UUID(memory_id)
        related = {
            "source_id": memory_id,
            "source_type": memory_type,
            "depth": depth,
            "related": {
                "episodic": [],
                "semantic": [],
                "procedural": [],
            },
        }

        try:
            if memory_type == "episodic":
                # Find entities extracted from this episode
                related_entities = await self._get_entities_from_episode(
                    memory_uuid,
                    session_id,
                )
                related["related"]["semantic"] = related_entities

            elif memory_type == "semantic":
                # Find related entities via graph relationships
                entity = await self.semantic.get_entity(memory_uuid)
                if entity:
                    # Get graph neighbors
                    neighbors = await self.semantic.graph_store.get_relationships(
                        node_id=memory_id,
                        direction="both",
                    )

                    # Load neighbor entities
                    neighbor_entities = []
                    for neighbor in neighbors[:10]:  # Limit to 10 neighbors
                        neighbor_entity = await self.semantic.get_entity(
                            UUID(neighbor["other_id"])
                        )
                        if neighbor_entity:
                            neighbor_entities.append({
                                "id": str(neighbor_entity.id),
                                "name": neighbor_entity.name,
                                "entity_type": neighbor_entity.entity_type.value,
                                "summary": neighbor_entity.summary,
                                "relationship_weight": neighbor["properties"].get("weight", 0.0),
                            })

                    related["related"]["semantic"] = neighbor_entities

            elif memory_type == "procedural":
                # Find similar procedures
                procedure = await self.procedural.get_procedure(memory_uuid)
                if procedure:
                    similar = await self.procedural.recall_skill(
                        task=procedure.name,
                        limit=5,
                        session_filter=session_id,
                    )

                    related["related"]["procedural"] = [
                        self._to_dict(r, "procedural")
                        for r in similar
                        if str(r.item.id) != memory_id  # Exclude self
                    ]

            add_span_attribute("related.depth", depth)
            add_span_attribute("related.source_type", memory_type)

        except Exception as e:
            logger.error(f"Error finding related memories for {memory_id}: {e}")
            related["error"] = str(e)

        return related

    async def _get_entities_from_episode(
        self,
        episode_id: UUID,
        session_id: str | None,
    ) -> list[dict]:
        """
        Get entities extracted from an episode via graph relationships.

        Args:
            episode_id: Episode UUID
            session_id: Optional session filter

        Returns:
            List of entity dicts
        """
        entities = []

        try:
            # Query graph for EXTRACTED_FROM relationships
            relationships = await self.semantic.graph_store.get_relationships(
                node_id=str(episode_id),
                direction="in",  # Entities point to episodes
            )

            for rel in relationships:
                if rel.get("type") == "EXTRACTED_FROM":
                    entity = await self.semantic.get_entity(UUID(rel["other_id"]))
                    if entity:
                        entities.append({
                            "id": str(entity.id),
                            "name": entity.name,
                            "entity_type": entity.entity_type.value,
                            "summary": entity.summary,
                        })
        except Exception as e:
            logger.warning(f"Failed to get entities from episode {episode_id}: {e}")

        return entities

    def _to_unified_result(self, scored_result: ScoredResult, memory_type: str) -> dict:
        """
        Convert ScoredResult to unified format.

        Args:
            scored_result: ScoredResult from memory subsystem
            memory_type: Type of memory (episodic, semantic, procedural)

        Returns:
            Unified result dict
        """
        item = scored_result.item

        # Extract common fields
        result = {
            "id": str(item.id),
            "memory_type": memory_type,
            "score": scored_result.score,
            "metadata": {
                "components": scored_result.components,
            },
        }

        # Add type-specific fields
        if memory_type == "episodic":
            result["content"] = item.content
            result["metadata"].update({
                "timestamp": item.timestamp.isoformat(),
                "outcome": item.outcome.value,
                "valence": item.emotional_valence,
            })
        elif memory_type == "semantic":
            result["content"] = f"{item.name}: {item.summary}"
            result["metadata"].update({
                "name": item.name,
                "entity_type": item.entity_type.value,
                "summary": item.summary,
            })
        elif memory_type == "procedural":
            result["content"] = f"{item.name} ({item.domain.value})"
            result["metadata"].update({
                "name": item.name,
                "domain": item.domain.value,
                "trigger_pattern": item.trigger_pattern,
                "success_rate": item.success_rate,
                "execution_count": item.execution_count,
            })

        return result

    def _to_dict(self, scored_result: ScoredResult, memory_type: str) -> dict:
        """
        Convert ScoredResult to dict for by_type breakdown.

        Args:
            scored_result: ScoredResult from memory subsystem
            memory_type: Type of memory

        Returns:
            Result dict with full item details
        """
        item = scored_result.item

        base = {
            "id": str(item.id),
            "score": scored_result.score,
            "components": scored_result.components,
        }

        if memory_type == "episodic":
            base.update({
                "content": item.content,
                "timestamp": item.timestamp.isoformat(),
                "outcome": item.outcome.value,
                "valence": item.emotional_valence,
                "session_id": item.session_id,
            })
        elif memory_type == "semantic":
            base.update({
                "name": item.name,
                "entity_type": item.entity_type.value,
                "summary": item.summary,
                "details": item.details,
            })
        elif memory_type == "procedural":
            base.update({
                "name": item.name,
                "domain": item.domain.value,
                "trigger_pattern": item.trigger_pattern,
                "success_rate": item.success_rate,
                "execution_count": item.execution_count,
                "steps": [
                    {
                        "order": step.order,
                        "action": step.action,
                        "tool": step.tool,
                    }
                    for step in item.steps
                ],
            })

        return base


# Factory function
def get_unified_memory_service(
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    procedural: ProceduralMemory,
) -> UnifiedMemoryService:
    """
    Get unified memory service instance.

    Args:
        episodic: Episodic memory service
        semantic: Semantic memory service
        procedural: Procedural memory service

    Returns:
        UnifiedMemoryService instance
    """
    return UnifiedMemoryService(episodic, semantic, procedural)
