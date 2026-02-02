"""
T4DM REST API Entity Routes.

CRUD and search operations for semantic memory (entities and relationships).
"""

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from t4dm.api.deps import MemoryServices
from t4dm.api.errors import sanitize_error
from t4dm.core.types import Entity, EntityType, Relationship, RelationType

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models

class EntityCreate(BaseModel):
    """Request model for creating an entity."""

    name: str = Field(..., min_length=1, description="Entity name")
    entity_type: EntityType = Field(..., description="Entity type")
    summary: str = Field(..., min_length=1, description="Short description")
    details: str | None = Field(default=None, description="Extended description")
    source: str | None = Field(default=None, description="Source episode ID or 'user_provided'")


class EntityResponse(BaseModel):
    """Response model for entity data."""

    id: UUID
    name: str
    entity_type: EntityType
    summary: str
    details: str | None
    source: str | None
    stability: float
    access_count: int
    created_at: datetime
    valid_from: datetime
    valid_to: datetime | None


class EntityList(BaseModel):
    """Response model for entity list."""

    entities: list[EntityResponse]
    total: int


class RelationCreate(BaseModel):
    """Request model for creating a relationship."""

    source_id: UUID = Field(..., description="Source entity ID")
    target_id: UUID = Field(..., description="Target entity ID")
    relation_type: RelationType = Field(..., description="Relationship type")
    weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Initial weight")


class RelationResponse(BaseModel):
    """Response model for relationship data."""

    source_id: UUID
    target_id: UUID
    relation_type: RelationType
    weight: float
    co_access_count: int


class EntityUpdate(BaseModel):
    """Request model for updating an entity."""

    name: str | None = Field(default=None, max_length=500, description="Updated name")
    summary: str | None = Field(default=None, max_length=2000, description="Updated summary")
    details: str | None = Field(default=None, max_length=10000, description="Updated details")


class SemanticRecallRequest(BaseModel):
    """Request model for semantic recall."""

    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    entity_types: list[EntityType] | None = Field(default=None, description="Filter by types")


class SpreadActivationRequest(BaseModel):
    """Request model for spreading activation."""

    entity_id: UUID = Field(..., description="Starting entity")
    depth: int = Field(default=2, ge=1, le=5, description="Traversal depth")
    threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Min activation")


class ActivationResponse(BaseModel):
    """Response model for activation results."""

    entities: list[EntityResponse]
    activations: list[float]
    paths: list[list[str]]


# Endpoints

@router.post("", response_model=EntityResponse, status_code=status.HTTP_201_CREATED)
async def create_entity(
    request: EntityCreate,
    services: MemoryServices,
):
    """
    Create a new semantic entity.

    Stores knowledge with ACT-R activation and Hebbian learning support.
    """
    semantic = services["semantic"]

    try:
        entity = Entity(
            name=request.name,
            entity_type=request.entity_type,
            summary=request.summary,
            details=request.details,
            source=request.source or "user_provided",
        )

        stored = await semantic.create_entity(
            name=entity.name,
            entity_type=entity.entity_type,
            summary=entity.summary,
            details=entity.details,
            source=entity.source,
        )

        return EntityResponse(
            id=stored.id,
            name=stored.name,
            entity_type=stored.entity_type,
            summary=stored.summary,
            details=stored.details,
            source=stored.source,
            stability=stored.stability,
            access_count=stored.access_count,
            created_at=stored.created_at,
            valid_from=stored.valid_from,
            valid_to=stored.valid_to,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "create entity"),
        )


@router.get("/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: UUID,
    services: MemoryServices,
):
    """
    Get a specific entity by ID.
    """
    semantic = services["semantic"]

    try:
        entity = await semantic.get_entity(str(entity_id))
        if not entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entity {entity_id} not found",
            )

        return EntityResponse(
            id=entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            summary=entity.summary,
            details=entity.details,
            source=entity.source,
            stability=entity.stability,
            access_count=entity.access_count,
            created_at=entity.created_at,
            valid_from=entity.valid_from,
            valid_to=entity.valid_to,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "get entity"),
        )


@router.put("/{entity_id}", response_model=EntityResponse)
async def update_entity(
    entity_id: UUID,
    request: EntityUpdate,
    services: MemoryServices,
):
    """
    Update an entity by ID.

    Only provided fields will be updated; others remain unchanged.
    For full versioned update with bi-temporal tracking, use /supersede.
    """
    semantic = services["semantic"]

    try:
        entity = await semantic.get_entity(str(entity_id))
        if not entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entity {entity_id} not found",
            )

        # Apply updates
        if request.name is not None:
            entity.name = request.name
        if request.summary is not None:
            entity.summary = request.summary
        if request.details is not None:
            entity.details = request.details

        # Re-store with updates
        updated = await semantic.update_entity(entity)

        return EntityResponse(
            id=updated.id,
            name=updated.name,
            entity_type=updated.entity_type,
            summary=updated.summary,
            details=updated.details,
            source=updated.source,
            stability=updated.stability,
            access_count=updated.access_count,
            created_at=updated.created_at,
            valid_from=updated.valid_from,
            valid_to=updated.valid_to,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "update entity"),
        )


@router.delete("/{entity_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_entity(
    entity_id: UUID,
    services: MemoryServices,
):
    """
    Delete an entity by ID.

    Also removes related relationships.
    """
    semantic = services["semantic"]

    try:
        deleted = await semantic.delete_entity(str(entity_id))
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entity {entity_id} not found",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "delete entity"),
        )


@router.get("", response_model=EntityList)
async def list_entities(
    services: MemoryServices,
    entity_type: EntityType | None = Query(default=None, description="Filter by type"),
    limit: int = Query(default=50, ge=1, le=500, description="Max results"),
):
    """
    List entities with optional type filtering.
    """
    semantic = services["semantic"]

    try:
        if entity_type:
            entities = await semantic.get_entities_by_type(entity_type, limit=limit)
        else:
            entities = await semantic.list_entities(limit=limit)

        return EntityList(
            entities=[
                EntityResponse(
                    id=e.id,
                    name=e.name,
                    entity_type=e.entity_type,
                    summary=e.summary,
                    details=e.details,
                    source=e.source,
                    stability=e.stability,
                    access_count=e.access_count,
                    created_at=e.created_at,
                    valid_from=e.valid_from,
                    valid_to=e.valid_to,
                )
                for e in entities
            ],
            total=len(entities),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "list entities"),
        )


@router.post("/relations", response_model=RelationResponse, status_code=status.HTTP_201_CREATED)
async def create_relation(
    request: RelationCreate,
    services: MemoryServices,
):
    """
    Create a relationship between two entities.

    Supports Hebbian weight strengthening on co-access.
    """
    semantic = services["semantic"]

    try:
        # Verify both entities exist
        source = await semantic.get_entity(str(request.source_id))
        target = await semantic.get_entity(str(request.target_id))

        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source entity {request.source_id} not found",
            )
        if not target:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Target entity {request.target_id} not found",
            )

        relation = Relationship(
            source_id=request.source_id,
            target_id=request.target_id,
            relation_type=request.relation_type,
            weight=request.weight,
        )

        stored = await semantic.create_relationship(
            source_id=str(relation.source_id),
            target_id=str(relation.target_id),
            relation_type=relation.relation_type,
            weight=relation.weight,
        )

        return RelationResponse(
            source_id=stored.source_id,
            target_id=stored.target_id,
            relation_type=stored.relation_type,
            weight=stored.weight,
            co_access_count=stored.co_access_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "create relation"),
        )


@router.post("/recall", response_model=EntityList)
async def semantic_recall(
    request: SemanticRecallRequest,
    services: MemoryServices,
):
    """
    Semantic search for entities.

    Uses BGE-M3 embeddings for similarity search.
    """
    semantic = services["semantic"]

    try:
        results = await semantic.recall(
            query=request.query,
            limit=request.limit,
        )
        # Extract entities from ScoredResult objects
        entities = [r.item for r in results]
        # Filter by entity_types if specified
        if request.entity_types:
            entities = [e for e in entities if e.entity_type in request.entity_types]

        return EntityList(
            entities=[
                EntityResponse(
                    id=e.id,
                    name=e.name,
                    entity_type=e.entity_type,
                    summary=e.summary,
                    details=e.details,
                    source=e.source,
                    stability=e.stability,
                    access_count=e.access_count,
                    created_at=e.created_at,
                    valid_from=e.valid_from,
                    valid_to=e.valid_to,
                )
                for e in entities
            ],
            total=len(entities),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "recall entities"),
        )


@router.post("/spread-activation", response_model=ActivationResponse)
async def spread_activation(
    request: SpreadActivationRequest,
    services: MemoryServices,
):
    """
    Perform spreading activation from an entity.

    Traverses the knowledge graph using ACT-R activation spreading.
    """
    semantic = services["semantic"]

    try:
        # Verify starting entity exists
        start = await semantic.get_entity(str(request.entity_id))
        if not start:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entity {request.entity_id} not found",
            )

        results = await semantic.spread_activation(
            entity_id=str(request.entity_id),
            depth=request.depth,
            threshold=request.threshold,
        )

        entities = []
        activations = []
        paths = []

        for entity, activation, path in results:
            entities.append(EntityResponse(
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                summary=entity.summary,
                details=entity.details,
                source=entity.source,
                stability=entity.stability,
                access_count=entity.access_count,
                created_at=entity.created_at,
                valid_from=entity.valid_from,
                valid_to=entity.valid_to,
            ))
            activations.append(activation)
            paths.append(path)

        return ActivationResponse(
            entities=entities,
            activations=activations,
            paths=paths,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "spread activation"),
        )


@router.post("/{entity_id}/supersede", response_model=EntityResponse)
async def supersede_entity(
    entity_id: UUID,
    request: EntityCreate,
    services: MemoryServices,
):
    """
    Supersede an entity with updated information.

    Creates new version and marks old as invalid (bi-temporal versioning).
    """
    semantic = services["semantic"]

    try:
        old_entity = await semantic.get_entity(str(entity_id))
        if not old_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entity {entity_id} not found",
            )

        # Create new version
        new_entity = Entity(
            name=request.name,
            entity_type=request.entity_type,
            summary=request.summary,
            details=request.details,
            source=request.source or old_entity.source,
        )

        stored = await semantic.supersede(str(entity_id), new_entity)

        return EntityResponse(
            id=stored.id,
            name=stored.name,
            entity_type=stored.entity_type,
            summary=stored.summary,
            details=stored.details,
            source=stored.source,
            stability=stored.stability,
            access_count=stored.access_count,
            created_at=stored.created_at,
            valid_from=stored.valid_from,
            valid_to=stored.valid_to,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "supersede entity"),
        )


# Request/Response Models for Feedback

class EntityFeedbackRequest(BaseModel):
    """Request model for learning feedback on retrieved entities."""

    entity_ids: list[UUID] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Entity IDs that were retrieved"
    )
    outcome_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall success score (0=failure, 1=success)"
    )
    query: str | None = Field(
        default=None,
        max_length=10000,
        description="Optional query string"
    )


class EntityFeedbackResponse(BaseModel):
    """Response model for semantic learning feedback."""

    strengthened_count: int = Field(description="Relationships strengthened")
    weakened_count: int = Field(description="Relationships weakened")
    avg_delta: float = Field(description="Average weight change")


@router.post("/feedback", response_model=EntityFeedbackResponse)
async def provide_entity_feedback(
    request: EntityFeedbackRequest,
    services: MemoryServices,
):
    """
    Provide outcome feedback for retrieved entities.

    PHASE-2 LEARNING WIRING: This endpoint implements reward-modulated
    Hebbian learning for semantic memory relationships:
    - Positive outcome (>0.5): Strengthen relationships between entities
    - Negative outcome (<0.5): Weaken relationships (anti-Hebbian)

    Call this after a retrieval session when you know the outcome.
    """
    semantic = services["semantic"]

    try:
        stats = await semantic.learn_from_outcome(
            entity_ids=request.entity_ids,
            outcome_score=request.outcome_score,
            query=request.query,
        )

        return EntityFeedbackResponse(
            strengthened_count=stats.get("strengthened_count", 0),
            weakened_count=stats.get("weakened_count", 0),
            avg_delta=stats.get("avg_delta", 0.0),
        )
    except Exception as e:
        logger.exception("Semantic learning feedback failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error(e, "provide entity feedback"),
        )
