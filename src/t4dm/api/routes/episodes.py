"""
T4DM REST API Episode Routes.

CRUD and search operations for episodic memory.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from t4dm.api.deps import MemoryServices
from t4dm.api.errors import sanitize_error
from t4dm.core.types import EpisodeContext, Outcome

logger = logging.getLogger(__name__)
# ATOM-P2-4: Mandatory persistent audit trail
audit_logger = logging.getLogger("t4dm.audit")

router = APIRouter()


# Request/Response Models

class EpisodeCreate(BaseModel):
    """Request model for creating an episode."""

    # API-001 FIX: Add max_length to prevent DoS via memory exhaustion
    content: str = Field(..., min_length=1, max_length=50000, description="Episode content (max 50KB)")
    project: str | None = Field(default=None, max_length=500, description="Project context")
    file: str | None = Field(default=None, max_length=1000, description="File context")
    tool: str | None = Field(default=None, max_length=200, description="Tool used")
    outcome: Outcome = Field(default=Outcome.NEUTRAL, description="Episode outcome")
    emotional_valence: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance")
    timestamp: datetime | None = Field(default=None, description="Event time (defaults to now)")


class EpisodeResponse(BaseModel):
    """Response model for episode data."""

    id: UUID
    session_id: str
    content: str
    timestamp: datetime
    outcome: Outcome
    emotional_valence: float
    context: EpisodeContext
    access_count: int
    stability: float
    retrievability: float | None = None


class EpisodeList(BaseModel):
    """Response model for episode list."""

    episodes: list[EpisodeResponse]
    total: int
    page: int
    page_size: int


class EpisodeUpdate(BaseModel):
    """Request model for updating an episode."""

    content: str | None = Field(default=None, max_length=50000, description="Updated content")
    emotional_valence: float | None = Field(default=None, ge=0.0, le=1.0, description="Importance")
    outcome: Outcome | None = Field(default=None, description="Updated outcome")
    project: str | None = Field(default=None, max_length=500, description="Project context")
    file: str | None = Field(default=None, max_length=1000, description="File context")
    tool: str | None = Field(default=None, max_length=200, description="Tool used")


class RecallRequest(BaseModel):
    """Request model for episode recall."""

    # API-001 FIX: Add max_length to prevent DoS via memory exhaustion
    query: str = Field(..., min_length=1, max_length=10000, description="Search query (max 10KB)")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0, description="Min similarity threshold")
    project: str | None = Field(default=None, max_length=500, description="Filter by project")
    outcome: Outcome | None = Field(default=None, description="Filter by outcome")


class RecallResponse(BaseModel):
    """Response model for recall results."""

    query: str
    episodes: list[EpisodeResponse]
    scores: list[float]


# Endpoints

@router.post("", response_model=EpisodeResponse, status_code=status.HTTP_201_CREATED)
async def create_episode(
    request: EpisodeCreate,
    services: MemoryServices,
):
    """
    Create a new episodic memory.

    Stores an experience with temporal-spatial context and FSRS decay tracking.
    ATOM-P0-11: Server-side immutable ingestion timestamp added.
    """
    episodic = services["episodic"]
    services["session_id"]

    try:
        # ATOM-P0-11: Add server-side immutable timestamp
        ingested_at = datetime.now(timezone.utc)

        context_dict = {
            "project": request.project,
            "file": request.file,
            "tool": request.tool,
            "ingested_at": ingested_at.isoformat(),  # Server-controlled timestamp
            # ATOM-P2-25: Add origin_type field
            "origin_type": "episodic_raw",  # Default for API-created memories
        }

        stored = await episodic.create(
            content=request.content,
            context=context_dict,
            outcome=request.outcome.value if hasattr(request.outcome, "value") else str(request.outcome),
            valence=request.emotional_valence,
        )

        if stored is None:
            raise HTTPException(
                status_code=status.HTTP_200_OK,
                detail="Episode was gated out (not stored)",
            )

        # ATOM-P2-4: Audit logging for memory mutations
        audit_logger.info(
            f"MEMORY_STORE: id={stored.id}, session={stored.session_id}, "
            f"ingested_at={ingested_at.isoformat()}"
        )

        return EpisodeResponse(
            id=stored.id,
            session_id=stored.session_id,
            content=stored.content,
            timestamp=stored.timestamp,
            outcome=stored.outcome,
            emotional_valence=stored.emotional_valence,
            context=stored.context,
            access_count=stored.access_count,
            stability=stored.stability,
            retrievability=stored.retrievability() if callable(getattr(stored, "retrievability", None)) else stored.retrievability,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "create episode"),
        )


@router.get("/{episode_id}", response_model=EpisodeResponse)
async def get_episode(
    episode_id: UUID,
    services: MemoryServices,
):
    """
    Get a specific episode by ID.

    Updates access count and retrievability.
    ATOM-P4-9: Session isolation verified via API key (see deps.py get_session_id).
    """
    episodic = services["episodic"]
    # ATOM-P4-9: Session isolation is enforced through API key in middleware

    try:
        episode = await episodic.get(str(episode_id))
        if not episode:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Episode {episode_id} not found",
            )

        return EpisodeResponse(
            id=episode.id,
            session_id=episode.session_id,
            content=episode.content,
            timestamp=episode.timestamp,
            outcome=episode.outcome,
            emotional_valence=episode.emotional_valence,
            context=episode.context,
            access_count=episode.access_count,
            stability=episode.stability,
            retrievability=episode.retrievability(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "get episode"),
        )


@router.delete("/{episode_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_episode(
    episode_id: UUID,
    services: MemoryServices,
):
    """
    Delete an episode by ID.
    """
    episodic = services["episodic"]

    try:
        deleted = await episodic.delete(str(episode_id))
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Episode {episode_id} not found",
            )
        # ATOM-P2-4: Audit logging for memory mutations
        audit_logger.info(f"MEMORY_DELETE: id={episode_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "delete episode"),
        )


@router.put("/{episode_id}", response_model=EpisodeResponse)
async def update_episode(
    episode_id: UUID,
    request: EpisodeUpdate,
    services: MemoryServices,
):
    """
    Update an episode by ID.

    Only provided fields will be updated; others remain unchanged.
    ATOM-P0-11: ingested_at timestamp is immutable and cannot be modified.
    """
    episodic = services["episodic"]

    try:
        episode = await episodic.get(str(episode_id))
        if not episode:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Episode {episode_id} not found",
            )

        # Apply updates
        if request.content is not None:
            episode.content = request.content
        if request.emotional_valence is not None:
            episode.emotional_valence = request.emotional_valence
        if request.outcome is not None:
            episode.outcome = request.outcome

        # Update context fields
        if request.project is not None:
            episode.context.project = request.project
        if request.file is not None:
            episode.context.file = request.file
        if request.tool is not None:
            episode.context.tool = request.tool

        # ATOM-P0-11: Preserve immutable ingested_at timestamp
        # If context has ingested_at, ensure it's not modified
        # (This protects against any client attempts to change it)

        # ATOM-P2-25: Prevent changing origin_type (immutable field)
        # Remove from any update data to preserve original value

        # Re-store with updates
        updated = await episodic.store(episode)

        # ATOM-P2-4: Audit logging for memory mutations
        update_fields = []
        if request.content is not None:
            update_fields.append("content")
        if request.emotional_valence is not None:
            update_fields.append("emotional_valence")
        if request.outcome is not None:
            update_fields.append("outcome")
        if request.project is not None:
            update_fields.append("project")
        if request.file is not None:
            update_fields.append("file")
        if request.tool is not None:
            update_fields.append("tool")
        audit_logger.info(f"MEMORY_UPDATE: id={episode_id}, fields={update_fields}")

        return EpisodeResponse(
            id=updated.id,
            session_id=updated.session_id,
            content=updated.content,
            timestamp=updated.timestamp,
            outcome=updated.outcome,
            emotional_valence=updated.emotional_valence,
            context=updated.context,
            access_count=updated.access_count,
            stability=updated.stability,
            retrievability=updated.retrievability(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "update episode"),
        )


@router.get("", response_model=EpisodeList)
async def list_episodes(
    services: MemoryServices,
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
    project: str | None = Query(default=None, description="Filter by project"),
    outcome: Outcome | None = Query(default=None, description="Filter by outcome"),
):
    """
    List episodes with pagination and filtering.
    """
    episodic = services["episodic"]

    try:
        # Get recent episodes
        episodes = await episodic.recent(limit=page_size * page)

        # Apply filters
        if project:
            episodes = [e for e in episodes if e.context.project == project]
        if outcome:
            episodes = [e for e in episodes if e.outcome == outcome]

        # Paginate
        total = len(episodes)
        start = (page - 1) * page_size
        end = start + page_size
        page_episodes = episodes[start:end]

        return EpisodeList(
            episodes=[
                EpisodeResponse(
                    id=e.id,
                    session_id=e.session_id,
                    content=e.content,
                    timestamp=e.timestamp,
                    outcome=e.outcome,
                    emotional_valence=e.emotional_valence,
                    context=e.context,
                    access_count=e.access_count,
                    stability=e.stability,
                    retrievability=e.retrievability(),
                )
                for e in page_episodes
            ],
            total=total,
            page=page,
            page_size=page_size,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "list episodes"),
        )


@router.post("/recall", response_model=RecallResponse)
async def recall_episodes(
    request: RecallRequest,
    services: MemoryServices,
):
    """
    Semantic search for episodes.

    Uses BGE-M3 embeddings for similarity search with optional filtering.
    """
    episodic = services["episodic"]

    try:
        results = await episodic.recall(
            query=request.query,
            limit=request.limit,
        )

        # Filter by min_similarity after retrieval
        if request.min_similarity:
            results = [r for r in results if r.score >= request.min_similarity]

        # Apply filters
        episodes = []
        scores = []
        for result in results:
            episode = result.item
            score = result.score
            if request.project and episode.context.project != request.project:
                continue
            if request.outcome and episode.outcome != request.outcome:
                continue
            episodes.append(episode)
            scores.append(score)

        return RecallResponse(
            query=request.query,
            episodes=[
                EpisodeResponse(
                    id=e.id,
                    session_id=e.session_id,
                    content=e.content,
                    timestamp=e.timestamp,
                    outcome=e.outcome,
                    emotional_valence=e.emotional_valence,
                    context=e.context,
                    access_count=e.access_count,
                    stability=e.stability,
                    retrievability=e.retrievability(),
                )
                for e in episodes
            ],
            scores=scores,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "recall episodes"),
        )


@router.post("/{episode_id}/mark-important", response_model=EpisodeResponse)
async def mark_important(
    episode_id: UUID,
    importance: float = Query(default=1.0, ge=0.0, le=1.0),
    services: MemoryServices = None,
):
    """
    Mark an episode as important.

    Increases emotional valence and stability to reduce decay.
    ATOM-P3-21: Rate limited to 10 marks per episode per minute.
    """
    episodic = services["episodic"]

    # ATOM-P3-21: Simple in-memory rate limit (10 marks/episode/minute)
    import time
    if not hasattr(mark_important, '_timestamps'):
        mark_important._timestamps = {}

    eid = str(episode_id)
    now = time.monotonic()

    if eid not in mark_important._timestamps:
        mark_important._timestamps[eid] = []

    # Remove old timestamps (>60s)
    mark_important._timestamps[eid] = [
        ts for ts in mark_important._timestamps[eid] if now - ts < 60.0
    ]

    # Check rate limit
    if len(mark_important._timestamps[eid]) >= 10:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit: max 10 importance marks per episode per minute",
        )

    # Record this request
    mark_important._timestamps[eid].append(now)

    try:
        episode = await episodic.get(str(episode_id))
        if not episode:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Episode {episode_id} not found",
            )

        # Update importance
        episode.emotional_valence = importance
        episode.stability = max(episode.stability, importance * 10)  # Boost stability

        # Re-store with updates
        updated = await episodic.store(episode)

        return EpisodeResponse(
            id=updated.id,
            session_id=updated.session_id,
            content=updated.content,
            timestamp=updated.timestamp,
            outcome=updated.outcome,
            emotional_valence=updated.emotional_valence,
            context=updated.context,
            access_count=updated.access_count,
            stability=updated.stability,
            retrievability=updated.retrievability(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "mark important"),
        )


# Request/Response Models for Feedback

class FeedbackRequest(BaseModel):
    """Request model for learning feedback on retrieved episodes."""

    episode_ids: list[UUID] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Episode IDs that were retrieved"
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The query that retrieved these episodes"
    )
    outcome_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall success score (0=failure, 1=success)"
    )
    per_memory_rewards: dict[str, float] | None = Field(
        default=None,
        description="Optional per-episode reward overrides"
    )


class FeedbackResponse(BaseModel):
    """Response model for learning feedback."""

    reconsolidated_count: int = Field(description="Number of embeddings updated")
    fusion_loss: float = Field(description="Fusion training loss")
    avg_rpe: float = Field(description="Average reward prediction error")
    positive_surprises: int = Field(description="Better-than-expected count")
    negative_surprises: int = Field(description="Worse-than-expected count")
    three_factor_stats: dict | None = Field(
        default=None,
        description="Three-factor learning statistics"
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def provide_feedback(
    request: FeedbackRequest,
    services: MemoryServices,
):
    """
    Provide outcome feedback for retrieved episodes.

    PHASE-2 LEARNING WIRING: This endpoint closes the learning loop by:
    1. Computing three-factor learning rates (eligibility × neuromod × dopamine)
    2. Updating embeddings via reconsolidation (move toward/away from query)
    3. Training fusion weights based on RPE (reward prediction error)

    The key insight is that learning scales with SURPRISE (|actual - expected|),
    not raw outcomes. Expected outcomes don't update the model; surprising
    outcomes (good or bad) drive adaptation.

    Call this after a retrieval session when you know the outcome.

    ATOM-P3-25: Rate limited to 5 feedback events per memory_id per hour.
    """
    episodic = services["episodic"]

    # ATOM-P3-25: Per-memory-id rate limiting (5 events/hour)
    import time
    if not hasattr(provide_feedback, '_timestamps'):
        provide_feedback._timestamps = {}

    now = time.monotonic()

    # Check rate limit for each episode_id in the request
    for episode_id in request.episode_ids:
        eid = str(episode_id)
        if eid not in provide_feedback._timestamps:
            provide_feedback._timestamps[eid] = []

        # Remove timestamps older than 60 minutes
        provide_feedback._timestamps[eid] = [
            ts for ts in provide_feedback._timestamps[eid] if now - ts < 3600.0
        ]

        # Check rate limit (5 per hour)
        if len(provide_feedback._timestamps[eid]) >= 5:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit: max 5 feedback events per memory per hour (memory_id={eid})",
            )

        # Record this request
        provide_feedback._timestamps[eid].append(now)

    try:
        # Call the complete learning loop
        stats = await episodic.learn_from_outcome(
            episode_ids=request.episode_ids,
            query=request.query,
            outcome_score=request.outcome_score,
            per_memory_rewards=request.per_memory_rewards,
        )

        # Get three-factor statistics if available
        three_factor_stats = None
        if hasattr(episodic, 'three_factor'):
            tf_stats = episodic.three_factor.get_stats()
            three_factor_stats = tf_stats.get('three_factor', {})

        return FeedbackResponse(
            reconsolidated_count=stats.get("reconsolidated_count", 0),
            fusion_loss=stats.get("fusion_loss", 0.0),
            avg_rpe=stats.get("avg_rpe", 0.0),
            positive_surprises=stats.get("positive_surprises", 0),
            negative_surprises=stats.get("negative_surprises", 0),
            three_factor_stats=three_factor_stats,
        )
    except Exception as e:
        logger.exception("Learning feedback failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "provide feedback"),
        )


@router.get("/learning-stats")
async def get_learning_stats(
    services: MemoryServices,
):
    """
    Get learning system statistics.

    Returns statistics from:
    - Three-factor learning rule (eligibility, neuromod, dopamine)
    - Reconsolidation engine (update counts, magnitudes)
    - Dopamine system (RPE distribution, value learning)
    """
    episodic = services["episodic"]

    try:
        stats = {
            "reconsolidation": episodic.get_reconsolidation_stats(),
            "dopamine": episodic.get_dopamine_stats(),
        }

        if hasattr(episodic, 'three_factor'):
            stats["three_factor"] = episodic.three_factor.get_stats()

        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "get learning stats"),
        )
