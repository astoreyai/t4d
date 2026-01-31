"""
World Weaver API Routes for ccapi.

Provides FastAPI endpoints for WW memory operations that can be
mounted into the ccapi application.

Usage:
    from fastapi import FastAPI
    from t4dm.integration.ccapi_routes import create_ww_router

    app = FastAPI()
    app.include_router(create_ww_router(), prefix="/memory")
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class MemoryStoreRequest(BaseModel):
    """Request to store a memory."""
    content: str = Field(..., description="Memory content")
    memory_type: str = Field("episodic", description="episodic, semantic, or procedural")
    session_id: str | None = Field(None, description="Session identifier")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")

    # Episodic-specific
    outcome: str | None = Field(None, description="SUCCESS, FAILURE, PARTIAL, NEUTRAL")
    emotional_valence: float | None = Field(None, ge=-1.0, le=1.0)

    # Semantic-specific
    entity_name: str | None = None
    entity_type: str | None = None
    summary: str | None = None

    # Procedural-specific
    procedure_name: str | None = None
    domain: str | None = None
    trigger_pattern: str | None = None
    steps: list[dict[str, Any]] | None = None


class MemoryStoreResponse(BaseModel):
    """Response from storing a memory."""
    success: bool
    memory_id: str
    memory_type: str
    message: str


class MemorySearchRequest(BaseModel):
    """Request to search memories."""
    query: str = Field(..., description="Search query")
    memory_types: list[str] | None = Field(
        None,
        description="Types to search: episodic, semantic, procedural"
    )
    limit: int = Field(10, ge=1, le=100)
    min_score: float = Field(0.0, ge=0.0, le=1.0)
    session_id: str | None = None


class MemoryResult(BaseModel):
    """A single memory search result."""
    id: str
    memory_type: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResponse(BaseModel):
    """Response from memory search."""
    query: str
    total_count: int
    results: list[MemoryResult]
    by_type: dict[str, int] = Field(default_factory=dict)


class ContextRequest(BaseModel):
    """Request for memory context."""
    session_id: str = Field(..., description="Session to get context for")
    max_episodes: int = Field(10, ge=1, le=50)
    include_entities: bool = Field(True)
    include_skills: bool = Field(True)


class ContextResponse(BaseModel):
    """Memory context for a session."""
    session_id: str
    episodes: list[MemoryResult]
    entities: list[MemoryResult]
    skills: list[MemoryResult]
    toon_json: str | None = None


class LearningStatsResponse(BaseModel):
    """Learning system statistics."""
    total_experiences: int
    total_retrievals: int
    total_outcomes: int
    avg_success_score: float
    replay_buffer_size: int


class OutcomeRequest(BaseModel):
    """Request to record an outcome."""
    outcome_type: str = Field(..., description="success, partial, failure, neutral")
    success_score: float = Field(..., ge=0.0, le=1.0)
    session_id: str | None = None
    context: str | None = None
    memory_citations: list[str] | None = None
    task_description: str | None = None


class OutcomeResponse(BaseModel):
    """Response from recording an outcome."""
    success: bool
    outcome_id: str
    matched_retrievals: int


# =============================================================================
# Router Factory
# =============================================================================

def create_ww_router(
    session_id: str | None = None,
    enable_learning: bool = True,
) -> APIRouter:
    """
    Create a FastAPI router with WW memory endpoints.

    Args:
        session_id: Default session ID for operations
        enable_learning: Enable learning system

    Returns:
        Configured APIRouter
    """
    router = APIRouter(
        tags=["memory"],
        responses={
            503: {"description": "WW backend not available"},
            500: {"description": "Internal error"},
        },
    )

    # Lazy-loaded WW components
    _ww_memory = None
    _ww_unified = None
    _ww_collector = None

    async def get_memory():
        """Get or create WW memory instance."""
        nonlocal _ww_memory
        if _ww_memory is None:
            from t4dm.integration.ccapi_memory import create_ww_memory
            _ww_memory = create_ww_memory(session_id=session_id or "default")
        return _ww_memory

    async def get_unified():
        """Get unified memory service."""
        nonlocal _ww_unified
        if _ww_unified is None:
            try:
                from t4dm.embedding.bge_m3 import BGEM3Embedder
                from t4dm.memory.episodic import EpisodicMemory
                from t4dm.memory.procedural import ProceduralMemory
                from t4dm.memory.semantic import SemanticMemory
                from t4dm.memory.unified import UnifiedMemoryService
                from t4dm.storage.qdrant_store import QdrantStore

                # Initialize components
                embedder = await BGEM3Embedder.create()
                vector_store = await QdrantStore.create()

                episodic = EpisodicMemory(vector_store=vector_store, embedder=embedder)
                semantic = SemanticMemory(vector_store=vector_store, embedder=embedder)
                procedural = ProceduralMemory(vector_store=vector_store, embedder=embedder)

                _ww_unified = UnifiedMemoryService(episodic, semantic, procedural)
            except Exception as e:
                logger.error(f"Failed to initialize unified memory: {e}")
                raise HTTPException(status_code=503, detail="WW backend not available")
        return _ww_unified

    async def get_collector():
        """Get learning collector."""
        nonlocal _ww_collector
        if _ww_collector is None and enable_learning:
            try:
                from t4dm.learning.collector import get_collector
                _ww_collector = get_collector()
            except Exception as e:
                logger.warning(f"Learning collector not available: {e}")
        return _ww_collector

    # -------------------------------------------------------------------------
    # Memory Endpoints
    # -------------------------------------------------------------------------

    @router.post("/store", response_model=MemoryStoreResponse)
    async def store_memory(request: MemoryStoreRequest):
        """
        Store a memory in World Weaver.

        Supports episodic, semantic, and procedural memory types.
        """
        try:
            unified = await get_unified()

            if request.memory_type == "episodic":
                from t4dm.core.types import Episode, Outcome

                outcome_map = {
                    "SUCCESS": Outcome.SUCCESS,
                    "FAILURE": Outcome.FAILURE,
                    "PARTIAL": Outcome.PARTIAL,
                    "NEUTRAL": Outcome.NEUTRAL,
                }

                episode = Episode(
                    content=request.content,
                    session_id=request.session_id or session_id or "default",
                    outcome=outcome_map.get(request.outcome or "NEUTRAL", Outcome.NEUTRAL),
                    emotional_valence=request.emotional_valence or 0.0,
                    metadata=request.metadata or {},
                )

                await unified.episodic.store(episode)
                return MemoryStoreResponse(
                    success=True,
                    memory_id=str(episode.id),
                    memory_type="episodic",
                    message="Episode stored successfully",
                )

            if request.memory_type == "semantic":
                from t4dm.core.types import Entity, EntityType

                type_map = {
                    "PERSON": EntityType.PERSON,
                    "CONCEPT": EntityType.CONCEPT,
                    "PROJECT": EntityType.PROJECT,
                    "TOOL": EntityType.TOOL,
                    "LOCATION": EntityType.LOCATION,
                }

                entity = Entity(
                    name=request.entity_name or request.content[:50],
                    entity_type=type_map.get(
                        request.entity_type or "CONCEPT",
                        EntityType.CONCEPT
                    ),
                    summary=request.summary or request.content,
                    details=request.content,
                    metadata=request.metadata or {},
                )

                await unified.semantic.store_entity(entity)
                return MemoryStoreResponse(
                    success=True,
                    memory_id=str(entity.id),
                    memory_type="semantic",
                    message="Entity stored successfully",
                )

            if request.memory_type == "procedural":
                from t4dm.core.types import Domain, Procedure, ProcedureStep

                domain_map = {
                    "CODING": Domain.CODING,
                    "RESEARCH": Domain.RESEARCH,
                    "COMMUNICATION": Domain.COMMUNICATION,
                    "GENERAL": Domain.GENERAL,
                }

                steps = []
                if request.steps:
                    for i, step_data in enumerate(request.steps):
                        steps.append(ProcedureStep(
                            order=i,
                            action=step_data.get("action", ""),
                            tool=step_data.get("tool"),
                            parameters=step_data.get("parameters", {}),
                        ))

                procedure = Procedure(
                    name=request.procedure_name or request.content[:50],
                    domain=domain_map.get(request.domain or "GENERAL", Domain.GENERAL),
                    trigger_pattern=request.trigger_pattern or "",
                    steps=steps,
                    metadata=request.metadata or {},
                )

                await unified.procedural.store_procedure(procedure)
                return MemoryStoreResponse(
                    success=True,
                    memory_id=str(procedure.id),
                    memory_type="procedural",
                    message="Procedure stored successfully",
                )

            raise HTTPException(
                status_code=400,
                detail=f"Unknown memory type: {request.memory_type}",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/search", response_model=MemorySearchResponse)
    async def search_memories(request: MemorySearchRequest):
        """
        Search across all memory types.

        Uses hybrid vector + graph retrieval with optional
        memory type filtering.
        """
        try:
            unified = await get_unified()

            result = await unified.search(
                query=request.query,
                k=request.limit,
                memory_types=request.memory_types,
                min_score=request.min_score,
                session_id=request.session_id,
            )

            results = [
                MemoryResult(
                    id=r["id"],
                    memory_type=r["memory_type"],
                    content=r.get("content", ""),
                    score=r["score"],
                    metadata=r.get("metadata", {}),
                )
                for r in result["results"]
            ]

            by_type = {
                k: len(v) for k, v in result.get("by_type", {}).items()
            }

            return MemorySearchResponse(
                query=request.query,
                total_count=result["total_count"],
                results=results,
                by_type=by_type,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/context", response_model=ContextResponse)
    async def get_context(
        sid: str = Query(..., description="Session ID"),
        max_episodes: int = Query(10, ge=1, le=50),
        include_entities: bool = Query(True),
        include_skills: bool = Query(True),
        format: str = Query("full", description="full or toon"),
    ):
        """
        Get memory context for a session.

        Returns recent episodes, related entities, and applicable skills
        for injection into agent context.
        """
        try:
            unified = await get_unified()

            # Get recent episodes
            episodes_result = await unified.episodic.recall(
                query="",
                limit=max_episodes,
                session_filter=sid,
            )

            episodes = [
                MemoryResult(
                    id=str(r.item.id),
                    memory_type="episodic",
                    content=r.item.content,
                    score=r.score,
                    metadata={
                        "timestamp": r.item.timestamp.isoformat(),
                        "outcome": r.item.outcome.value,
                    },
                )
                for r in episodes_result
            ]

            # Get related entities
            entities = []
            if include_entities and episodes:
                # Get entities mentioned in recent episodes
                for ep in episodes[:5]:
                    related = await unified.get_related(
                        memory_id=ep.id,
                        memory_type="episodic",
                    )
                    for ent in related.get("related", {}).get("semantic", []):
                        entities.append(MemoryResult(
                            id=ent["id"],
                            memory_type="semantic",
                            content=f"{ent['name']}: {ent.get('summary', '')}",
                            score=ent.get("relationship_weight", 0.5),
                            metadata=ent,
                        ))

            # Get applicable skills
            skills = []
            if include_skills:
                skills_result = await unified.procedural.recall_skill(
                    task="session context",
                    limit=5,
                    session_filter=sid,
                )
                skills = [
                    MemoryResult(
                        id=str(r.item.id),
                        memory_type="procedural",
                        content=f"{r.item.name} ({r.item.domain.value})",
                        score=r.score,
                        metadata={
                            "trigger": r.item.trigger_pattern,
                            "success_rate": r.item.success_rate,
                        },
                    )
                    for r in skills_result
                ]

            # Optionally convert to ToonJSON
            toon_json = None
            if format == "toon":
                from t4dm.learning.events import ToonJSON
                toon = ToonJSON()
                context_data = {
                    "session_id": sid,
                    "episodes": [e.model_dump() for e in episodes[:5]],
                    "entities": [e.model_dump() for e in entities[:5]],
                    "skills": [s.model_dump() for s in skills[:3]],
                }
                toon_json = toon.encode(context_data)

            return ContextResponse(
                session_id=sid,
                episodes=episodes,
                entities=entities,
                skills=skills,
                toon_json=toon_json,
            )

        except Exception as e:
            logger.error(f"Get context failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------------------------------------------------------
    # Learning Endpoints
    # -------------------------------------------------------------------------

    @router.post("/outcome", response_model=OutcomeResponse)
    async def record_outcome(request: OutcomeRequest):
        """
        Record a task outcome for learning.

        Triggers credit assignment to recent retrievals based on
        the outcome success score.
        """
        collector = await get_collector()
        if collector is None:
            raise HTTPException(status_code=503, detail="Learning not available")

        try:
            from t4dm.learning.events import OutcomeType

            type_map = {
                "success": OutcomeType.SUCCESS,
                "partial": OutcomeType.PARTIAL,
                "failure": OutcomeType.FAILURE,
                "neutral": OutcomeType.NEUTRAL,
            }

            citations = []
            if request.memory_citations:
                for cid in request.memory_citations:
                    try:
                        citations.append(UUID(cid))
                    except ValueError:
                        pass

            event = collector.record_outcome(
                outcome_type=type_map.get(request.outcome_type, OutcomeType.UNKNOWN),
                success_score=request.success_score,
                context=request.context,
                session_id=request.session_id or session_id or "",
                explicit_citations=citations,
                task_description=request.task_description or "",
            )

            # Count matched retrievals
            matched = len(collector.store.get_retrievals_by_context(
                event.context_hash,
                max_age_hours=1.0,
            )) if event.context_hash else 0

            return OutcomeResponse(
                success=True,
                outcome_id=str(event.outcome_id),
                matched_retrievals=matched,
            )

        except Exception as e:
            logger.error(f"Record outcome failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/learning/stats", response_model=LearningStatsResponse)
    async def learning_stats():
        """Get learning system statistics."""
        collector = await get_collector()
        if collector is None:
            raise HTTPException(status_code=503, detail="Learning not available")

        try:
            stats = collector.store.get_stats()
            return LearningStatsResponse(
                total_experiences=stats.get("experiences", 0),
                total_retrievals=stats.get("retrievals", 0),
                total_outcomes=stats.get("outcomes", 0),
                avg_success_score=stats.get("avg_success", 0.5),
                replay_buffer_size=stats.get("replay_buffer", 0),
            )
        except Exception as e:
            logger.error(f"Get stats failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/learning/flush")
    async def flush_learning():
        """Flush pending learning events."""
        collector = await get_collector()
        if collector is None:
            return {"status": "learning not enabled"}

        try:
            collector.flush()
            return {"status": "flushed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    return router


# Convenience function to get router with defaults
def get_ww_router(**kwargs) -> APIRouter:
    """Get WW router with default configuration."""
    return create_ww_router(**kwargs)


__all__ = [
    "ContextRequest",
    "ContextResponse",
    "LearningStatsResponse",
    "MemoryResult",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "MemoryStoreRequest",
    "MemoryStoreResponse",
    "OutcomeRequest",
    "OutcomeResponse",
    "create_ww_router",
    "get_ww_router",
]
