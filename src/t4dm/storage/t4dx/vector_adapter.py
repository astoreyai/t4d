"""Async VectorStore protocol adapter over T4DXEngine."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import ItemRecord


class T4DXVectorStore:
    """Implements the ``VectorStore`` protocol using T4DXEngine.

    Collections are emulated via the ``item_type`` field — the collection
    name maps to item_type. All vectors live in a single engine instance.
    """

    def __init__(self, engine: T4DXEngine) -> None:
        self._engine = engine
        self._collections: dict[str, dict[str, Any]] = {}

        # Legacy collection name attributes (used by memory stores)
        self.episodes_collection = "episodes"
        self.entities_collection = "entities"
        self.skills_collection = "skills"
        self.procedures_collection = "procedures"

    async def initialize(self) -> None:
        """No-op: T4DX engine is already started."""
        pass

    async def close(self) -> None:
        """No-op: engine lifecycle managed by storage module."""
        pass

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
                metadata=payload,  # Store full payload for reconstruction
            )
            self._engine.insert(rec)

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        **kwargs: Any,
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

        # Ensure vector is a plain list (not numpy array)
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()

        raw = self._engine.search(vector, k=limit, **kw)
        results = []
        for rid, score in raw:
            if score_threshold is not None and score < score_threshold:
                continue
            rec = self._engine.get(rid)
            if rec is None:
                continue
            # Return full stored payload (metadata contains the original payload)
            payload = dict(rec.metadata) if rec.metadata else {}
            payload.update({
                "content": rec.content,
                "kappa": rec.kappa,
                "importance": rec.importance,
                "item_type": rec.item_type,
            })
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
                payload = dict(rec.metadata) if rec.metadata else {}
                payload.update({
                    "content": rec.content,
                    "kappa": rec.kappa,
                    "importance": rec.importance,
                    "item_type": rec.item_type,
                })
                results.append((id_str, payload))
        return results

    # Hybrid search methods (stubs for compatibility)
    async def _ensure_collection(
        self,
        client: Any,
        collection_name: str,
        hybrid: bool = False,
    ) -> None:
        """Stub: Ensure collection exists (for compatibility)."""
        await self.create_collection(collection_name, dimension=1024)

    async def search_hybrid(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Stub: Hybrid search (falls back to dense-only)."""
        return await self.search(collection, dense_vector, limit, filter)

    async def add_hybrid(
        self,
        collection: str,
        ids: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Stub: Add with hybrid vectors (uses dense-only)."""
        await self.add(collection, ids, dense_vectors, payloads)

    async def ensure_hybrid_collection(self, collection_name: str) -> None:
        """Stub: Ensure hybrid collection exists."""
        await self.create_collection(collection_name, dimension=1024)

    # Additional methods for compatibility with old QdrantStore
    async def update_payload(
        self,
        collection: str,
        point_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Update payload for a single point."""
        uid = UUID(point_id)
        rec = self._engine.get(uid.bytes)
        if rec is not None:
            # Update metadata and other fields
            import time
            updated_rec = ItemRecord(
                id=rec.id,
                vector=rec.vector,
                kappa=payload.get("kappa", rec.kappa),
                importance=payload.get("importance", rec.importance),
                event_time=payload.get("event_time", rec.event_time),
                record_time=rec.record_time,
                valid_from=rec.valid_from,
                valid_until=payload.get("valid_until", rec.valid_until),
                item_type=rec.item_type,
                content=payload.get("content", rec.content),
                access_count=rec.access_count,
                session_id=rec.session_id,
                metadata={**rec.metadata, **payload.get("metadata", {})},
            )
            self._engine.insert(updated_rec)

    async def batch_update_payloads(
        self,
        collection: str,
        updates: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Batch update payloads."""
        for point_id, payload in updates:
            await self.update_payload(collection, point_id, payload)

    async def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Upsert points (alias for add)."""
        await self.add(collection, ids, vectors, payloads)

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: int = 0,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Scroll through collection using engine scan."""
        kw: dict[str, Any] = {"item_type": collection}
        if filter:
            if "kappa_min" in filter:
                kw["kappa_min"] = filter["kappa_min"]
            if "kappa_max" in filter:
                kw["kappa_max"] = filter["kappa_max"]
        all_items = self._engine.scan(**kw)
        page = all_items[offset : offset + limit]
        results = []
        for rec in page:
            uid = ItemRecord.bytes_to_uuid(rec.id).__str__()
            payload = dict(rec.metadata) if rec.metadata else {}
            payload.update({
                "content": rec.content,
                "kappa": rec.kappa,
                "importance": rec.importance,
                "item_type": rec.item_type,
            })
            results.append((uid, payload))
        return results

    async def count(
        self,
        collection: str,
        filter: dict[str, Any] | None = None,
    ) -> int:
        """Count items in collection using engine scan."""
        kw: dict[str, Any] = {"item_type": collection}
        if filter:
            if "kappa_min" in filter:
                kw["kappa_min"] = filter["kappa_min"]
            if "kappa_max" in filter:
                kw["kappa_max"] = filter["kappa_max"]
        return len(self._engine.scan(**kw))

    async def get_with_vectors(
        self,
        collection: str,
        ids: list[str],
    ) -> list[tuple[str, list[float], dict[str, Any]]]:
        """Get points with vectors."""
        results = []
        for id_str in ids:
            uid = UUID(id_str)
            rec = self._engine.get(uid.bytes)
            if rec is not None:
                payload = dict(rec.metadata) if rec.metadata else {}
                payload.update({
                    "content": rec.content,
                    "kappa": rec.kappa,
                    "importance": rec.importance,
                    "item_type": rec.item_type,
                })
                results.append((id_str, rec.vector, payload))
        return results
