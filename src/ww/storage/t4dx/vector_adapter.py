"""Async VectorStore protocol adapter over T4DXEngine."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from ww.storage.t4dx.engine import T4DXEngine
from ww.storage.t4dx.types import ItemRecord


class T4DXVectorStore:
    """Implements the ``VectorStore`` protocol using T4DXEngine.

    Collections are emulated via the ``item_type`` field — the collection
    name maps to item_type. All vectors live in a single engine instance.
    """

    def __init__(self, engine: T4DXEngine) -> None:
        self._engine = engine
        self._collections: dict[str, dict[str, Any]] = {}

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "cosine",
    ) -> None:
        self._collections[name] = {"dimension": dimension, "distance": distance}

    async def add(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        import time

        for id_str, vec, payload in zip(ids, vectors, payloads):
            uid = UUID(id_str)
            now = time.time()
            rec = ItemRecord(
                id=uid.bytes,
                vector=vec,
                kappa=payload.get("kappa", 0.0),
                importance=payload.get("importance", 0.5),
                event_time=payload.get("event_time", now),
                record_time=payload.get("record_time", now),
                valid_from=payload.get("valid_from", now),
                valid_until=payload.get("valid_until"),
                item_type=payload.get("item_type", collection),
                content=payload.get("content", ""),
                access_count=payload.get("access_count", 0),
                session_id=payload.get("session_id"),
                metadata=payload.get("metadata", {}),
            )
            self._engine.insert(rec)

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        kw: dict[str, Any] = {"item_type": collection}
        if filter:
            if "kappa_min" in filter:
                kw["kappa_min"] = filter["kappa_min"]
            if "kappa_max" in filter:
                kw["kappa_max"] = filter["kappa_max"]
            if "time_min" in filter:
                kw["time_min"] = filter["time_min"]
            if "time_max" in filter:
                kw["time_max"] = filter["time_max"]

        raw = self._engine.search(vector, k=limit, **kw)
        results = []
        for rid, score in raw:
            rec = self._engine.get(rid)
            if rec is None:
                continue
            payload = {
                "content": rec.content,
                "kappa": rec.kappa,
                "importance": rec.importance,
                "item_type": rec.item_type,
                "metadata": rec.metadata,
            }
            results.append((ItemRecord.bytes_to_uuid(rid).__str__(), score, payload))
        return results

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        for id_str in ids:
            uid = UUID(id_str)
            self._engine.delete(uid.bytes)

    async def get(
        self,
        collection: str,
        ids: list[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        results = []
        for id_str in ids:
            uid = UUID(id_str)
            rec = self._engine.get(uid.bytes)
            if rec is not None:
                payload = {
                    "content": rec.content,
                    "kappa": rec.kappa,
                    "importance": rec.importance,
                    "item_type": rec.item_type,
                    "metadata": rec.metadata,
                }
                results.append((id_str, payload))
        return results
