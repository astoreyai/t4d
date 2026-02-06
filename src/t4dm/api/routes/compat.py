"""
Mem0-compatible REST API routes.

Provides a compatibility layer matching the Mem0 API JSON structure,
allowing T4DM to serve as a drop-in replacement for Mem0.

Storage backend:
- Uses T4DX embedded storage when available (production)
- Falls back to in-memory dict when T4DX unavailable (testing/standalone)

Endpoints:
    POST   /v1/memories/        - Store a memory
    GET    /v1/memories/search/  - Search memories
    GET    /v1/memories/         - List all memories
    GET    /v1/memories/{id}     - Get memory by ID
    DELETE /v1/memories/{id}     - Delete memory
    PUT    /v1/memories/{id}     - Update memory
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/memories", tags=["Mem0 Compatibility"])


# --- Request / Response Models ---


class MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=50000)
    user_id: str | None = None
    agent_id: str | None = None
    metadata: dict[str, Any] | None = None
    embedding: list[float] | None = Field(None, description="Optional pre-computed embedding")


class MemoryUpdateRequest(BaseModel):
    content: str | None = Field(None, max_length=50000)
    metadata: dict[str, Any] | None = None


class MemoryResponse(BaseModel):
    id: str
    memory: str
    user_id: str | None = None
    agent_id: str | None = None
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    id: str
    memory: str
    user_id: str | None = None
    score: float = 0.0
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Storage Backend ---


class CompatStorageBackend:
    """Abstract interface for Mem0 compat storage."""

    async def store(self, record: dict) -> dict:
        raise NotImplementedError

    async def get(self, memory_id: str) -> dict | None:
        raise NotImplementedError

    async def delete(self, memory_id: str) -> bool:
        raise NotImplementedError

    async def update(self, memory_id: str, updates: dict) -> dict | None:
        raise NotImplementedError

    async def list_all(self, user_id: str | None, limit: int) -> list[dict]:
        raise NotImplementedError

    async def search(self, query: str, user_id: str | None, limit: int, embedding: list[float] | None = None) -> list[tuple[dict, float]]:
        raise NotImplementedError


class InMemoryBackend(CompatStorageBackend):
    """In-memory fallback storage for testing."""

    def __init__(self) -> None:
        self._memories: dict[str, dict[str, Any]] = {}

    async def store(self, record: dict) -> dict:
        self._memories[record["id"]] = record
        return record

    async def get(self, memory_id: str) -> dict | None:
        return self._memories.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    async def update(self, memory_id: str, updates: dict) -> dict | None:
        mem = self._memories.get(memory_id)
        if not mem:
            return None
        mem.update(updates)
        mem["updated_at"] = datetime.now(timezone.utc).isoformat()
        return mem

    async def list_all(self, user_id: str | None, limit: int) -> list[dict]:
        results = []
        for mem in self._memories.values():
            if user_id and mem.get("user_id") != user_id:
                continue
            results.append(mem)
            if len(results) >= limit:
                break
        return results

    async def search(self, query: str, user_id: str | None, limit: int, embedding: list[float] | None = None) -> list[tuple[dict, float]]:
        # Simple substring match
        results = []
        query_lower = query.lower()
        for mem in self._memories.values():
            if user_id and mem.get("user_id") != user_id:
                continue
            content = mem.get("content", "")
            if query_lower in content.lower():
                results.append((mem, 1.0))
                if len(results) >= limit:
                    break
        return results


class T4DXBackend(CompatStorageBackend):
    """T4DX-backed storage for production."""

    def __init__(self) -> None:
        self._engine = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of T4DX engine."""
        if self._initialized:
            return self._engine is not None

        self._initialized = True
        try:
            from t4dm.storage.t4dx.engine import T4DXEngine
            from pathlib import Path

            data_dir = Path(".data/compat")
            data_dir.mkdir(parents=True, exist_ok=True)

            self._engine = T4DXEngine(data_dir=data_dir)
            self._engine.startup()
            logger.info("T4DX backend initialized for Mem0 compat layer")
            return True
        except Exception as e:
            logger.warning(f"T4DX unavailable for compat layer, using in-memory: {e}")
            return False

    async def store(self, record: dict) -> dict:
        if not self._ensure_initialized():
            raise RuntimeError("T4DX not available")

        from uuid import UUID
        from t4dm.storage.t4dx.types import ItemRecord

        # Create embedding if not provided
        embedding = record.get("embedding")
        if embedding is None:
            embedding = self._text_to_embedding(record["content"])

        now = datetime.now(timezone.utc).timestamp()
        item_record = ItemRecord(
            id=UUID(record["id"]).bytes,
            vector=embedding,
            kappa=0.0,  # New memory starts at κ=0
            importance=0.5,
            event_time=now,
            record_time=now,
            valid_from=now,
            valid_until=None,
            item_type="episodic",
            content=record["content"],
            access_count=0,
            session_id=record.get("user_id"),
            metadata={
                "user_id": record.get("user_id"),
                "agent_id": record.get("agent_id"),
                "metadata": record.get("metadata", {}),
                "created_at": record["created_at"],
                "updated_at": record["updated_at"],
            },
        )
        self._engine.insert(item_record)
        return record

    async def get(self, memory_id: str) -> dict | None:
        if not self._ensure_initialized():
            return None

        from uuid import UUID

        item = self._engine.get(UUID(memory_id).bytes)
        if not item:
            return None

        return self._item_to_record(item)

    async def delete(self, memory_id: str) -> bool:
        if not self._ensure_initialized():
            return False

        from uuid import UUID

        id_bytes = UUID(memory_id).bytes
        # Check if item exists before deleting
        item = self._engine.get(id_bytes)
        if not item:
            return False
        self._engine.delete(id_bytes)
        return True

    async def update(self, memory_id: str, updates: dict) -> dict | None:
        if not self._ensure_initialized():
            return None

        from uuid import UUID

        item = self._engine.get(UUID(memory_id).bytes)
        if not item:
            return None

        # Build update dict
        field_updates = {}
        if "content" in updates:
            field_updates["content"] = updates["content"]
        if "metadata" in updates:
            # Merge into existing metadata
            new_meta = item.metadata.copy()
            new_meta["metadata"] = updates["metadata"]
            new_meta["updated_at"] = datetime.now(timezone.utc).isoformat()
            field_updates["metadata"] = new_meta

        if field_updates:
            self._engine.update_fields(UUID(memory_id).bytes, field_updates)

        updated = self._engine.get(UUID(memory_id).bytes)
        return self._item_to_record(updated) if updated else None

    async def list_all(self, user_id: str | None, limit: int) -> list[dict]:
        if not self._ensure_initialized():
            return []

        results = []
        for item in self._engine.scan():  # Get all, filter manually
            meta = item.metadata or {}
            if user_id and meta.get("user_id") != user_id:
                continue
            results.append(self._item_to_record(item))
            if len(results) >= limit:
                break
        return results

    async def search(self, query: str, user_id: str | None, limit: int, embedding: list[float] | None = None) -> list[tuple[dict, float]]:
        if not self._ensure_initialized():
            return []

        # Use provided embedding or generate one
        if embedding is None:
            embedding = self._text_to_embedding(query)

        # T4DX search returns list of (id_bytes, score)
        search_results = self._engine.search(
            query_vector=embedding,
            k=limit * 5,  # Over-fetch for filtering
        )

        filtered = []
        for item_id_bytes, score in search_results:
            item = self._engine.get(item_id_bytes)
            if not item:
                continue
            meta = item.metadata or {}
            if user_id and meta.get("user_id") != user_id:
                continue
            filtered.append((self._item_to_record(item), score))
            if len(filtered) >= limit:
                break
        return filtered

    def _text_to_embedding(self, text: str, dim: int = 256) -> list[float]:
        """Simple hash-based embedding for compat layer (not semantic)."""
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        # Expand hash to desired dimension
        expanded = []
        for i in range(dim):
            byte_idx = i % len(h)
            expanded.append((h[byte_idx] + i) / 256.0 - 0.5)
        return expanded

    def _item_to_record(self, item) -> dict:
        """Convert T4DX ItemRecord to compat record format."""
        from uuid import UUID

        meta = item.metadata or {}
        item_id = UUID(bytes=item.id) if isinstance(item.id, bytes) else item.id
        created = meta.get("created_at") or datetime.fromtimestamp(item.event_time, tz=timezone.utc).isoformat()
        updated = meta.get("updated_at") or created

        return {
            "id": str(item_id),
            "content": item.content,
            "user_id": meta.get("user_id"),
            "agent_id": meta.get("agent_id"),
            "metadata": meta.get("metadata", {}),
            "created_at": created,
            "updated_at": updated,
        }


# --- Storage Singleton ---


_backend: CompatStorageBackend | None = None


def get_storage_backend() -> CompatStorageBackend:
    """Get or create storage backend."""
    global _backend
    if _backend is None:
        # Try T4DX first, fall back to in-memory
        t4dx = T4DXBackend()
        if t4dx._ensure_initialized():
            _backend = t4dx
        else:
            _backend = InMemoryBackend()
            logger.info("Using in-memory backend for Mem0 compat layer")
    return _backend


# --- Helper ---


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_to_response(record: dict) -> MemoryResponse:
    """Convert storage record to API response."""
    return MemoryResponse(
        id=record["id"],
        memory=record.get("content", ""),
        user_id=record.get("user_id"),
        agent_id=record.get("agent_id"),
        created_at=record.get("created_at", _now_iso()),
        updated_at=record.get("updated_at", record.get("created_at", _now_iso())),
        metadata=record.get("metadata", {}),
    )


# --- Endpoints ---


@router.post("/", response_model=MemoryResponse, status_code=201)
async def create_memory(
    req: MemoryCreateRequest,
    backend: CompatStorageBackend = Depends(get_storage_backend),
) -> MemoryResponse:
    """Store a new memory (Mem0-compatible)."""
    memory_id = str(uuid4())
    now = _now_iso()
    record = {
        "id": memory_id,
        "content": req.content,
        "user_id": req.user_id,
        "agent_id": req.agent_id,
        "metadata": req.metadata or {},
        "created_at": now,
        "updated_at": now,
    }
    if req.embedding:
        record["embedding"] = req.embedding

    stored = await backend.store(record)
    return _record_to_response(stored)


@router.get("/search/", response_model=list[MemorySearchResult])
async def search_memories(
    query: str = Query(..., min_length=1, max_length=10000),
    user_id: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
    backend: CompatStorageBackend = Depends(get_storage_backend),
) -> list[MemorySearchResult]:
    """Search memories (Mem0-compatible)."""
    results = await backend.search(query, user_id, limit)
    return [
        MemorySearchResult(
            id=record["id"],
            memory=record.get("content", ""),
            user_id=record.get("user_id"),
            score=score,
            created_at=record.get("created_at", _now_iso()),
            updated_at=record.get("updated_at", _now_iso()),
            metadata=record.get("metadata", {}),
        )
        for record, score in results
    ]


@router.get("/", response_model=list[MemoryResponse])
async def list_memories(
    user_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    backend: CompatStorageBackend = Depends(get_storage_backend),
) -> list[MemoryResponse]:
    """List all memories (Mem0-compatible)."""
    records = await backend.list_all(user_id, limit)
    return [_record_to_response(r) for r in records]


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    backend: CompatStorageBackend = Depends(get_storage_backend),
) -> MemoryResponse:
    """Get a memory by ID (Mem0-compatible)."""
    record = await backend.get(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _record_to_response(record)


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: str,
    backend: CompatStorageBackend = Depends(get_storage_backend),
) -> None:
    """Delete a memory (Mem0-compatible)."""
    deleted = await backend.delete(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")


@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    req: MemoryUpdateRequest,
    backend: CompatStorageBackend = Depends(get_storage_backend),
) -> MemoryResponse:
    """Update a memory (Mem0-compatible)."""
    updates = {}
    if req.content is not None:
        updates["content"] = req.content
    if req.metadata is not None:
        updates["metadata"] = req.metadata

    record = await backend.update(memory_id, updates)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _record_to_response(record)


def get_compat_router() -> APIRouter:
    """Return the Mem0-compatible router for inclusion in the main app."""
    return router
