"""
Shared serialization utilities for World Weaver.

Provides consistent conversion between domain objects and storage formats.
Eliminates ~150 lines of duplicated serialization code across memory modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import UUID

if TYPE_CHECKING:
    from t4dm.core.types import Entity, Episode
    from t4dm.memory.procedural import Procedure

T = TypeVar("T")


class Serializer(ABC, Generic[T]):
    """Abstract base for type-specific serializers."""

    @abstractmethod
    def to_payload(self, obj: T, session_id: str) -> dict[str, Any]:
        """Convert domain object to storage payload."""
        ...

    @abstractmethod
    def from_payload(self, id_str: str, payload: dict[str, Any]) -> T:
        """Reconstruct domain object from storage payload."""
        ...

    @abstractmethod
    def to_graph_props(self, obj: T, session_id: str) -> dict[str, Any]:
        """Convert domain object to graph node properties."""
        ...


class DateTimeSerializer:
    """Shared datetime serialization utilities."""

    @staticmethod
    def to_iso(dt: datetime | None) -> str | None:
        """Convert datetime to ISO string."""
        return dt.isoformat() if dt else None

    @staticmethod
    def from_iso(iso_str: str | None) -> datetime | None:
        """Parse datetime from ISO string."""
        return datetime.fromisoformat(iso_str) if iso_str else None


class EpisodeSerializer(Serializer):
    """Serializer for Episode objects."""

    def to_payload(self, episode: Episode, session_id: str) -> dict[str, Any]:
        """Convert episode to Qdrant payload."""

        return {
            "session_id": session_id,
            "content": episode.content,
            "timestamp": DateTimeSerializer.to_iso(episode.timestamp),
            "ingested_at": DateTimeSerializer.to_iso(episode.ingested_at),
            "context": episode.context.model_dump() if episode.context else {},
            "outcome": episode.outcome.value if episode.outcome else None,
            "emotional_valence": episode.emotional_valence,
            "access_count": episode.access_count,
            "last_accessed": DateTimeSerializer.to_iso(episode.last_accessed),
            "stability": episode.stability,
        }

    def from_payload(self, id_str: str, payload: dict[str, Any]) -> Episode:
        """Reconstruct episode from Qdrant payload."""
        from t4dm.core.types import Episode, EpisodeContext, Outcome

        # Handle datetime fields with defaults
        now = datetime.now()
        timestamp = DateTimeSerializer.from_iso(payload.get("timestamp")) or now
        ingested_at = DateTimeSerializer.from_iso(payload.get("ingested_at")) or now
        last_accessed = DateTimeSerializer.from_iso(payload.get("last_accessed")) or now

        return Episode(
            id=UUID(id_str),
            session_id=payload.get("session_id", ""),
            content=payload.get("content", ""),
            embedding=None,  # Not stored in payload
            timestamp=timestamp,
            ingested_at=ingested_at,
            context=EpisodeContext(**payload.get("context", {})),
            outcome=Outcome(payload["outcome"]) if payload.get("outcome") else Outcome.NEUTRAL,
            emotional_valence=payload.get("emotional_valence", 0.0),
            access_count=max(payload.get("access_count", 1), 1),  # Ensure >= 1
            last_accessed=last_accessed,
            stability=payload.get("stability", 1.0),
        )

    def to_graph_props(self, episode: Episode, session_id: str) -> dict[str, Any]:
        """Convert episode to Neo4j properties."""

        return {
            "id": str(episode.id),
            "sessionId": session_id,
            "content": episode.content[:500] if episode.content else "",  # Truncate for graph
            "timestamp": DateTimeSerializer.to_iso(episode.timestamp),
            "ingestedAt": DateTimeSerializer.to_iso(episode.ingested_at),
            "outcome": episode.outcome.value if episode.outcome else None,
            "emotionalValence": episode.emotional_valence,
            "accessCount": episode.access_count,
            "lastAccessed": DateTimeSerializer.to_iso(episode.last_accessed),
            "stability": episode.stability,
        }


class EntitySerializer(Serializer):
    """Serializer for Entity objects."""

    def to_payload(self, entity: Entity, session_id: str) -> dict[str, Any]:
        """Convert entity to Qdrant payload."""

        return {
            "session_id": session_id,
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "summary": entity.summary,
            "details": entity.details,
            "source": entity.source,
            "stability": entity.stability,
            "access_count": entity.access_count,
            "last_accessed": DateTimeSerializer.to_iso(entity.last_accessed),
            "created_at": DateTimeSerializer.to_iso(entity.created_at),
            "valid_from": DateTimeSerializer.to_iso(entity.valid_from),
            "valid_to": DateTimeSerializer.to_iso(entity.valid_to),
        }

    def from_payload(self, id_str: str, payload: dict[str, Any]) -> Entity:
        """Reconstruct entity from Qdrant payload."""
        from t4dm.core.types import Entity, EntityType

        # Handle datetime fields with defaults
        now = datetime.now()
        last_accessed = DateTimeSerializer.from_iso(payload.get("last_accessed")) or now
        created_at = DateTimeSerializer.from_iso(payload.get("created_at")) or now
        valid_from = DateTimeSerializer.from_iso(payload.get("valid_from")) or now

        return Entity(
            id=UUID(id_str),
            name=payload.get("name", ""),
            entity_type=EntityType(payload.get("entity_type", "CONCEPT")),
            summary=payload.get("summary", ""),
            details=payload.get("details"),
            embedding=None,  # Not stored in payload
            source=payload.get("source"),
            stability=payload.get("stability", 1.0),
            access_count=max(payload.get("access_count", 1), 1),  # Ensure >= 1
            last_accessed=last_accessed,
            created_at=created_at,
            valid_from=valid_from,
            valid_to=DateTimeSerializer.from_iso(payload.get("valid_to")),
        )

    def to_graph_props(self, entity: Entity, session_id: str) -> dict[str, Any]:
        """Convert entity to Neo4j properties."""

        return {
            "id": str(entity.id),
            "sessionId": session_id,
            "name": entity.name,
            "entityType": entity.entity_type.value,
            "summary": entity.summary[:500] if entity.summary else "",  # Truncate for graph
            "details": entity.details[:500] if entity.details else "",  # Truncate for graph
            "source": entity.source or "",
            "stability": entity.stability,
            "accessCount": entity.access_count,
            "lastAccessed": DateTimeSerializer.to_iso(entity.last_accessed),
            "createdAt": DateTimeSerializer.to_iso(entity.created_at),
            "validFrom": DateTimeSerializer.to_iso(entity.valid_from),
            "validTo": DateTimeSerializer.to_iso(entity.valid_to) if entity.valid_to else "",
        }


class ProcedureSerializer(Serializer):
    """Serializer for Procedure objects."""

    def to_payload(self, procedure: Procedure, session_id: str) -> dict[str, Any]:
        """Convert procedure to Qdrant payload."""

        return {
            "session_id": session_id,
            "name": procedure.name,
            "domain": procedure.domain.value,
            "trigger_pattern": procedure.trigger_pattern,
            "steps": [s.model_dump() for s in procedure.steps],
            "script": procedure.script,
            "success_rate": procedure.success_rate,
            "execution_count": procedure.execution_count,
            "last_executed": DateTimeSerializer.to_iso(procedure.last_executed),
            "version": procedure.version,
            "deprecated": procedure.deprecated,
            "consolidated_into": str(procedure.consolidated_into) if procedure.consolidated_into else None,
            "created_at": DateTimeSerializer.to_iso(procedure.created_at),
            "created_from": procedure.created_from,
        }

    def from_payload(self, id_str: str, payload: dict[str, Any]) -> Procedure:
        """Reconstruct procedure from Qdrant payload."""
        from t4dm.core.types import Domain, Procedure, ProcedureStep

        # Handle datetime fields with defaults
        created_at = DateTimeSerializer.from_iso(payload.get("created_at")) or datetime.now()

        return Procedure(
            id=UUID(id_str),
            name=payload.get("name", ""),
            domain=Domain(payload.get("domain", "coding")),
            trigger_pattern=payload.get("trigger_pattern"),
            steps=[ProcedureStep(**s) for s in payload.get("steps", [])],
            script=payload.get("script"),
            embedding=None,  # Not stored in payload
            success_rate=payload.get("success_rate", 1.0),
            execution_count=max(payload.get("execution_count", 1), 1),  # Ensure >= 1
            last_executed=DateTimeSerializer.from_iso(payload.get("last_executed")),
            version=payload.get("version", 1),
            deprecated=payload.get("deprecated", False),
            consolidated_into=UUID(payload["consolidated_into"]) if payload.get("consolidated_into") else None,
            created_at=created_at,
            created_from=payload.get("created_from", "manual"),
        )

    def to_graph_props(self, procedure: Procedure, session_id: str) -> dict[str, Any]:
        """Convert procedure to Neo4j properties."""

        return {
            "id": str(procedure.id),
            "sessionId": session_id,
            "name": procedure.name,
            "domain": procedure.domain.value,
            "triggerPattern": procedure.trigger_pattern or "",
            "script": procedure.script[:500] if procedure.script else "",  # Truncate for graph
            "stepCount": len(procedure.steps) if procedure.steps else 0,
            "successRate": procedure.success_rate,
            "executionCount": procedure.execution_count,
            "lastExecuted": DateTimeSerializer.to_iso(procedure.last_executed) if procedure.last_executed else "",
            "version": procedure.version,
            "deprecated": procedure.deprecated,
            "consolidatedInto": str(procedure.consolidated_into) if procedure.consolidated_into else "",
            "createdAt": DateTimeSerializer.to_iso(procedure.created_at),
            "createdFrom": procedure.created_from,
        }


# Serializer registry
_serializers: dict[str, Serializer] = {}


def get_serializer(type_name: str) -> Serializer:
    """
    Get serializer for given type.

    Args:
        type_name: Type name (episode, entity, procedure)

    Returns:
        Serializer instance

    Raises:
        ValueError: If type_name not registered
    """
    if not _serializers:
        _serializers["episode"] = EpisodeSerializer()
        _serializers["entity"] = EntitySerializer()
        _serializers["procedure"] = ProcedureSerializer()

    if type_name not in _serializers:
        raise ValueError(f"No serializer registered for type: {type_name}")

    return _serializers[type_name]


def register_serializer(type_name: str, serializer: Serializer) -> None:
    """
    Register a custom serializer.

    Args:
        type_name: Type identifier
        serializer: Serializer implementation
    """
    _serializers[type_name] = serializer
