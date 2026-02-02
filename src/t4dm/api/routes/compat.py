"""
Mem0-compatible REST API routes.

Provides a compatibility layer matching the Mem0 API JSON structure,
allowing T4DM to serve as a drop-in replacement for Mem0.

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
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/memories", tags=["Mem0 Compatibility"])


# --- Request / Response Models ---


class MemoryCreateRequest(BaseModel):
    content: str
    user_id: str | None = None
    agent_id: str | None = None
    metadata: dict[str, Any] | None = None


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
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


# --- Helper ---

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _episode_to_response(data: dict, user_id: str | None = None, agent_id: str | None = None) -> MemoryResponse:
    """Convert an internal episode dict to Mem0-style response."""
    ts = data.get("timestamp") or data.get("created_at") or _now_iso()
    return MemoryResponse(
        id=str(data.get("id", "")),
        memory=data.get("content", ""),
        user_id=user_id or data.get("user_id"),
        agent_id=agent_id or data.get("agent_id"),
        created_at=str(ts),
        updated_at=str(data.get("updated_at", ts)),
        metadata=data.get("context", {}) if isinstance(data.get("context"), dict) else {},
    )


# --- In-memory store for compat layer (delegates to core when available) ---

# We store memories in a simple dict so this compat layer works standalone
# for testing and as a facade. In production, wire to core services.

_memories: dict[str, dict[str, Any]] = {}


@router.post("/", response_model=MemoryResponse, status_code=201)
async def create_memory(req: MemoryCreateRequest) -> MemoryResponse:
    """Store a new memory (Mem0-compatible)."""
    import uuid

    memory_id = str(uuid.uuid4())
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
    _memories[memory_id] = record
    return MemoryResponse(
        id=memory_id,
        memory=req.content,
        user_id=req.user_id,
        agent_id=req.agent_id,
        created_at=now,
        updated_at=now,
        metadata=req.metadata or {},
    )


@router.get("/search/", response_model=list[MemorySearchResult])
async def search_memories(
    query: str = Query(...),
    user_id: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
) -> list[MemorySearchResult]:
    """Search memories (Mem0-compatible)."""
    # Simple substring match for the compat layer
    results = []
    query_lower = query.lower()
    for mem in _memories.values():
        if user_id and mem.get("user_id") != user_id:
            continue
        content = mem.get("content", "")
        if query_lower in content.lower():
            score = 1.0
        else:
            score = 0.0
            continue
        results.append(
            MemorySearchResult(
                id=mem["id"],
                memory=content,
                user_id=mem.get("user_id"),
                score=score,
                created_at=mem["created_at"],
                updated_at=mem["updated_at"],
                metadata=mem.get("metadata", {}),
            )
        )
        if len(results) >= limit:
            break
    return results


@router.get("/", response_model=list[MemoryResponse])
async def list_memories(
    user_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[MemoryResponse]:
    """List all memories (Mem0-compatible)."""
    results = []
    for mem in _memories.values():
        if user_id and mem.get("user_id") != user_id:
            continue
        results.append(
            MemoryResponse(
                id=mem["id"],
                memory=mem.get("content", ""),
                user_id=mem.get("user_id"),
                agent_id=mem.get("agent_id"),
                created_at=mem["created_at"],
                updated_at=mem["updated_at"],
                metadata=mem.get("metadata", {}),
            )
        )
        if len(results) >= limit:
            break
    return results


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str) -> MemoryResponse:
    """Get a memory by ID (Mem0-compatible)."""
    mem = _memories.get(memory_id)
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse(
        id=mem["id"],
        memory=mem.get("content", ""),
        user_id=mem.get("user_id"),
        agent_id=mem.get("agent_id"),
        created_at=mem["created_at"],
        updated_at=mem["updated_at"],
        metadata=mem.get("metadata", {}),
    )


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(memory_id: str) -> None:
    """Delete a memory (Mem0-compatible)."""
    if memory_id not in _memories:
        raise HTTPException(status_code=404, detail="Memory not found")
    del _memories[memory_id]


@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(memory_id: str, req: MemoryUpdateRequest) -> MemoryResponse:
    """Update a memory (Mem0-compatible)."""
    mem = _memories.get(memory_id)
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")
    if req.content is not None:
        mem["content"] = req.content
    if req.metadata is not None:
        mem["metadata"] = req.metadata
    mem["updated_at"] = _now_iso()
    _memories[memory_id] = mem
    return MemoryResponse(
        id=mem["id"],
        memory=mem.get("content", ""),
        user_id=mem.get("user_id"),
        agent_id=mem.get("agent_id"),
        created_at=mem["created_at"],
        updated_at=mem["updated_at"],
        metadata=mem.get("metadata", {}),
    )


def get_compat_router() -> APIRouter:
    """Return the Mem0-compatible router for inclusion in the main app."""
    return router
