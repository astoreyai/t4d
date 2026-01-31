"""
Unified MemoryItem with continuous κ (kappa) consolidation level.

Replaces discrete episodic/semantic/procedural stores with a single
schema where κ ∈ [0,1] indicates consolidation progress:
  κ=0.0  Raw episodic (just encoded)
  κ~0.15 Replayed (NREM strengthened)
  κ~0.4  Transitional (being abstracted)
  κ~0.85 Semantic concept (REM prototype)
  κ=1.0  Stable knowledge (fully consolidated)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from t4dm.core.types import Entity, Episode, Outcome, Procedure


class MemoryItem(BaseModel):
    """Unified memory item with κ-gradient consolidation."""

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., min_length=1)
    embedding: list[float] = Field(default_factory=list)
    event_time: datetime = Field(default_factory=datetime.now)
    record_time: datetime = Field(default_factory=datetime.now)
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_until: datetime | None = None

    kappa: float = Field(default=0.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    item_type: Literal["episodic", "semantic", "procedural"] = "episodic"

    access_count: int = Field(default=0, ge=0)
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_episode(cls, ep: Episode) -> MemoryItem:
        """Convert an Episode to a MemoryItem (κ near 0)."""
        return cls(
            id=ep.id,
            content=ep.content,
            embedding=ep.embedding or [],
            event_time=ep.timestamp,
            record_time=ep.ingested_at,
            valid_from=ep.ingested_at,
            kappa=0.0,
            importance=ep.emotional_valence,
            item_type="episodic",
            access_count=ep.access_count,
            session_id=ep.session_id,
            metadata={
                "outcome": ep.outcome.value,
                "context": ep.context.model_dump() if ep.context else {},
                "prediction_error": ep.prediction_error,
            },
        )

    @classmethod
    def from_entity(cls, ent: Entity) -> MemoryItem:
        """Convert an Entity to a MemoryItem (κ near 1)."""
        return cls(
            id=ent.id,
            content=f"{ent.name}: {ent.summary}",
            embedding=ent.embedding or [],
            event_time=ent.created_at,
            record_time=ent.created_at,
            valid_from=ent.valid_from,
            valid_until=ent.valid_to,
            kappa=0.85,
            importance=0.7,
            item_type="semantic",
            access_count=ent.access_count,
            metadata={
                "entity_type": ent.entity_type.value,
                "name": ent.name,
                "details": ent.details,
                "source": ent.source,
            },
        )

    @classmethod
    def from_procedure(cls, proc: Procedure) -> MemoryItem:
        """Convert a Procedure to a MemoryItem."""
        return cls(
            id=proc.id,
            content=f"{proc.name}: {proc.script or ''}",
            embedding=proc.embedding or [],
            event_time=proc.created_at,
            record_time=proc.created_at,
            valid_from=proc.created_at,
            kappa=0.5,
            importance=proc.success_rate,
            item_type="procedural",
            access_count=proc.execution_count,
            metadata={
                "domain": proc.domain.value,
                "trigger_pattern": proc.trigger_pattern,
                "success_rate": proc.success_rate,
                "execution_count": proc.execution_count,
                "steps_count": len(proc.steps),
            },
        )

    def to_episode(self) -> Episode:
        """Convert back to Episode (loses κ-specific fields)."""
        ctx_data = self.metadata.get("context", {})
        outcome_str = self.metadata.get("outcome", "neutral")
        from t4dm.core.types import EpisodeContext

        return Episode(
            id=self.id,
            session_id=self.session_id or "unknown",
            content=self.content,
            embedding=self.embedding if self.embedding else None,
            timestamp=self.event_time,
            ingested_at=self.record_time,
            context=EpisodeContext(**ctx_data) if ctx_data else EpisodeContext(),
            outcome=Outcome(outcome_str),
            emotional_valence=self.importance,
            access_count=max(1, self.access_count),
        )
