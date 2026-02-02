"""T4DX internal record types and MemoryItem conversion."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


class EdgeType(str, Enum):
    """Edge relationship types stored in T4DX.

    Maps all Neo4j relationship types from the legacy store.
    """

    USES = "USES"
    PRODUCES = "PRODUCES"
    REQUIRES = "REQUIRES"
    CAUSES = "CAUSES"
    PART_OF = "PART_OF"
    SIMILAR_TO = "SIMILAR_TO"
    IMPLEMENTS = "IMPLEMENTS"
    IMPROVES_ON = "IMPROVES_ON"
    CONSOLIDATED_INTO = "CONSOLIDATED_INTO"
    SOURCE_OF = "SOURCE_OF"
    SEQUENCE = "SEQUENCE"
    TEMPORAL_BEFORE = "TEMPORAL_BEFORE"
    TEMPORAL_AFTER = "TEMPORAL_AFTER"
    MERGED_FROM = "MERGED_FROM"
    SUPERSEDES = "SUPERSEDES"
    RELATES_TO = "RELATES_TO"
    HAS_CONTEXT = "HAS_CONTEXT"
    DERIVED_FROM = "DERIVED_FROM"
    DEPENDS_ON = "DEPENDS_ON"


@dataclass(slots=True)
class ItemRecord:
    """Internal record for a memory item stored in T4DX.

    UUID is stored as 16-byte ``bytes`` for fast hashing/comparison.
    """

    id: bytes  # 16-byte UUID
    vector: list[float]
    kappa: float
    importance: float
    event_time: float  # unix timestamp
    record_time: float  # unix timestamp
    valid_from: float  # unix timestamp
    valid_until: float | None  # unix timestamp or None
    item_type: str  # "episodic" | "semantic" | "procedural"
    content: str
    access_count: int
    session_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    spike_trace: dict[str, Any] | None = None
    graph_delta: dict[str, Any] | None = None

    # --- UUID helpers ---

    @staticmethod
    def uuid_to_bytes(u: UUID) -> bytes:
        return u.bytes

    @staticmethod
    def bytes_to_uuid(b: bytes) -> UUID:
        return UUID(bytes=b)

    # --- MemoryItem conversion ---

    @classmethod
    def from_memory_item(cls, mi: Any) -> ItemRecord:
        """Convert a ``MemoryItem`` to an ``ItemRecord``."""
        return cls(
            id=mi.id.bytes,
            vector=list(mi.embedding),
            kappa=mi.kappa,
            importance=mi.importance,
            event_time=mi.event_time.timestamp(),
            record_time=mi.record_time.timestamp(),
            valid_from=mi.valid_from.timestamp(),
            valid_until=mi.valid_until.timestamp() if mi.valid_until else None,
            item_type=mi.item_type,
            content=mi.content,
            access_count=mi.access_count,
            session_id=mi.session_id,
            metadata=dict(mi.metadata),
            spike_trace=dict(mi.spike_trace) if mi.spike_trace else None,
            graph_delta=dict(mi.graph_delta) if mi.graph_delta else None,
        )

    def to_memory_item(self) -> Any:
        """Convert back to ``MemoryItem``."""
        from t4dm.core.memory_item import MemoryItem

        return MemoryItem(
            id=self.bytes_to_uuid(self.id),
            content=self.content,
            embedding=self.vector,
            event_time=datetime.fromtimestamp(self.event_time),
            record_time=datetime.fromtimestamp(self.record_time),
            valid_from=datetime.fromtimestamp(self.valid_from),
            valid_until=(
                datetime.fromtimestamp(self.valid_until)
                if self.valid_until is not None
                else None
            ),
            kappa=self.kappa,
            importance=self.importance,
            item_type=self.item_type,
            access_count=self.access_count,
            session_id=self.session_id,
            metadata=dict(self.metadata),
            spike_trace=dict(self.spike_trace) if self.spike_trace else None,
            graph_delta=dict(self.graph_delta) if self.graph_delta else None,
        )

    # --- Serialisation (JSON-safe dicts) ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id.hex(),
            "vector": self.vector,
            "kappa": self.kappa,
            "importance": self.importance,
            "event_time": self.event_time,
            "record_time": self.record_time,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "item_type": self.item_type,
            "content": self.content,
            "access_count": self.access_count,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "spike_trace": self.spike_trace,
            "graph_delta": self.graph_delta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ItemRecord:
        return cls(
            id=bytes.fromhex(d["id"]),
            vector=d["vector"],
            kappa=d["kappa"],
            importance=d["importance"],
            event_time=d["event_time"],
            record_time=d["record_time"],
            valid_from=d["valid_from"],
            valid_until=d["valid_until"],
            item_type=d["item_type"],
            content=d["content"],
            access_count=d["access_count"],
            session_id=d.get("session_id"),
            metadata=d.get("metadata", {}),
            spike_trace=d.get("spike_trace"),
            graph_delta=d.get("graph_delta"),
        )


@dataclass(slots=True)
class EdgeRecord:
    """Directed weighted edge between two items."""

    source_id: bytes  # 16-byte UUID
    target_id: bytes  # 16-byte UUID
    edge_type: str
    weight: float = 0.1
    created_at: float = 0.0  # unix timestamp
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id.hex(),
            "target_id": self.target_id.hex(),
            "edge_type": self.edge_type,
            "weight": self.weight,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EdgeRecord:
        return cls(
            source_id=bytes.fromhex(d["source_id"]),
            target_id=bytes.fromhex(d["target_id"]),
            edge_type=d["edge_type"],
            weight=d.get("weight", 0.1),
            created_at=d.get("created_at", 0.0),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class SegmentMetadata:
    """Manifest metadata for an on-disk segment."""

    segment_id: int
    item_count: int
    edge_count: int
    time_min: float
    time_max: float
    kappa_min: float
    kappa_max: float
    created_at: float  # unix timestamp
    level: int = 0  # compaction level

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "item_count": self.item_count,
            "edge_count": self.edge_count,
            "time_min": self.time_min,
            "time_max": self.time_max,
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
            "created_at": self.created_at,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SegmentMetadata:
        return cls(
            segment_id=d["segment_id"],
            item_count=d["item_count"],
            edge_count=d["edge_count"],
            time_min=d["time_min"],
            time_max=d["time_max"],
            kappa_min=d["kappa_min"],
            kappa_max=d["kappa_max"],
            created_at=d["created_at"],
            level=d.get("level", 0),
        )
