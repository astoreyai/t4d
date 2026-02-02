"""Async GraphStore protocol adapter over T4DXEngine."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import EdgeRecord, ItemRecord


class T4DXGraphStore:
    """Implements the ``GraphStore`` protocol using T4DXEngine edges.

    Nodes are items; the ``label`` parameter maps to ``item_type``.
    Cypher queries are not supported — ``query()`` raises NotImplementedError.
    """

    def __init__(self, engine: T4DXEngine) -> None:
        self._engine = engine

    async def initialize(self) -> None:
        """No-op: T4DX engine is already started."""
        pass

    async def close(self) -> None:
        """No-op: engine lifecycle managed by storage module."""
        pass

    # --- node operations ---

    async def create_node(
        self,
        label: str,
        properties: dict[str, Any],
    ) -> str:
        import time as _time
        uid_str = properties.get("id", str(UUID(int=0)))
        uid = UUID(uid_str) if isinstance(uid_str, str) else uid_str
        now = _time.time()
        rec = ItemRecord(
            id=uid.bytes,
            vector=properties.get("embedding", []),
            kappa=properties.get("kappa", 0.0),
            importance=properties.get("importance", 0.5),
            event_time=properties.get("event_time", now),
            record_time=properties.get("record_time", now),
            valid_from=properties.get("valid_from", now),
            valid_until=properties.get("valid_until"),
            item_type=label,
            content=properties.get("content", ""),
            access_count=properties.get("access_count", 0),
            session_id=properties.get("session_id"),
            metadata={
                k: v for k, v in properties.items()
                if k not in {
                    "id", "embedding", "kappa", "importance", "event_time",
                    "record_time", "valid_from", "valid_until", "content",
                    "access_count", "session_id",
                }
            },
        )
        self._engine.insert(rec)
        return str(uid)

    async def get_node(
        self,
        node_id: str,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        uid = UUID(node_id)
        rec = self._engine.get(uid.bytes)
        if rec is None:
            return None
        if label is not None and rec.item_type != label:
            return None
        return {
            "id": node_id,
            "label": rec.item_type,
            "content": rec.content,
            "kappa": rec.kappa,
            "importance": rec.importance,
            **rec.metadata,
        }

    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any],
        label: str | None = None,
    ) -> None:
        uid = UUID(node_id)
        self._engine.update_fields(uid.bytes, properties)

    async def delete_node(
        self,
        node_id: str,
        label: str | None = None,
    ) -> None:
        uid = UUID(node_id)
        self._engine.delete(uid.bytes)

    # --- relationship operations ---

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any],
    ) -> None:
        edge = EdgeRecord(
            source_id=UUID(source_id).bytes,
            target_id=UUID(target_id).bytes,
            edge_type=rel_type,
            weight=properties.get("weight", 0.1),
            metadata={k: v for k, v in properties.items() if k != "weight"},
        )
        self._engine.insert_edge(edge)

    async def get_relationships(
        self,
        node_id: str,
        rel_type: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        uid = UUID(node_id)
        edges = self._engine.traverse(uid.bytes, rel_type, direction)
        return [
            {
                "source_id": ItemRecord.bytes_to_uuid(e.source_id).__str__(),
                "target_id": ItemRecord.bytes_to_uuid(e.target_id).__str__(),
                "type": e.edge_type,
                "weight": e.weight,
                **e.metadata,
            }
            for e in edges
        ]

    async def update_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any],
    ) -> None:
        if "weight" in properties:
            uid_s = UUID(source_id)
            uid_t = UUID(target_id)
            # Get current weight to compute delta
            edges = self._engine.traverse(uid_s.bytes, rel_type, "out")
            current = 0.1
            for e in edges:
                if e.target_id == uid_t.bytes:
                    current = e.weight
                    break
            delta = properties["weight"] - current
            self._engine.update_edge_weight(uid_s.bytes, uid_t.bytes, rel_type, delta)

    # --- query operations ---

    async def query(
        self,
        query: str | dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("T4DXGraphStore does not support raw graph queries")

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """BFS shortest path."""
        start = UUID(source_id).bytes
        end = UUID(target_id).bytes

        if start == end:
            return [source_id]

        visited: set[bytes] = {start}
        queue: list[tuple[bytes, list[str]]] = [
            (start, [source_id])
        ]

        for _ in range(max_depth):
            next_queue: list[tuple[bytes, list[str]]] = []
            for node, path in queue:
                edges = self._engine.traverse(node, direction="out")
                for e in edges:
                    neighbor = e.target_id
                    if neighbor in visited:
                        continue
                    nid = ItemRecord.bytes_to_uuid(neighbor).__str__()
                    new_path = path + [nid]
                    if neighbor == end:
                        return new_path
                    visited.add(neighbor)
                    next_queue.append((neighbor, new_path))
            queue = next_queue
            if not queue:
                break

        return None
