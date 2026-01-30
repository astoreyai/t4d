"""
World Weaver SDK Data Models.

Simplified models for SDK responses.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class EpisodeContext(BaseModel):
    """Episode context information."""

    project: str | None = None
    file: str | None = None
    tool: str | None = None


class Episode(BaseModel):
    """Episodic memory record."""

    id: UUID
    session_id: str
    content: str
    timestamp: datetime
    outcome: str
    emotional_valence: float
    context: EpisodeContext
    access_count: int
    stability: float
    retrievability: float | None = None


class Entity(BaseModel):
    """Semantic memory entity."""

    id: UUID
    name: str
    entity_type: str
    summary: str
    details: str | None = None
    source: str | None = None
    stability: float
    access_count: int
    created_at: datetime


class Relationship(BaseModel):
    """Relationship between entities."""

    source_id: UUID
    target_id: UUID
    relation_type: str
    weight: float
    co_access_count: int


class Step(BaseModel):
    """Procedure step."""

    order: int
    action: str
    tool: str | None = None
    parameters: dict = Field(default_factory=dict)
    expected_outcome: str | None = None


class Skill(BaseModel):
    """Procedural memory skill."""

    id: UUID
    name: str
    domain: str
    trigger_pattern: str | None = None
    steps: list[Step]
    script: str | None = None
    success_rate: float
    execution_count: int
    last_executed: datetime | None = None
    version: int
    deprecated: bool
    created_at: datetime


class RecallResult(BaseModel):
    """Result from episode recall."""

    query: str
    episodes: list[Episode]
    scores: list[float]


class ActivationResult(BaseModel):
    """Result from spreading activation."""

    entities: list[Entity]
    activations: list[float]
    paths: list[list[str]]


class HealthStatus(BaseModel):
    """Health check result."""

    status: str
    timestamp: str
    version: str
    session_id: str | None = None


class MemoryStats(BaseModel):
    """Memory statistics."""

    session_id: str
    episodic: dict
    semantic: dict
    procedural: dict
