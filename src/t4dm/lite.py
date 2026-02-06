"""
T4DM Lite - Minimal in-memory vector store for quick prototyping.

Example:
    from t4dm.lite import Memory
    mem = Memory()
    id1 = mem.store("Python is a programming language")
    results = mem.search("programming")
    mem.delete(id1)

For production use with consolidation and temporal encoding:
    from t4dm import memory
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np


def _hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Generate a deterministic pseudo-embedding from text using SHA256."""
    text_bytes = text.lower().encode("utf-8")
    hash_bytes = hashlib.sha256(text_bytes).digest()

    # Expand hash to desired dimension
    expanded = bytearray()
    for i in range((dim // 32) + 1):
        round_input = hash_bytes + i.to_bytes(4, "little")
        expanded.extend(hashlib.sha256(round_input).digest())

    # Convert bytes to floats in range [-1, 1]
    arr = np.zeros(dim, dtype=np.float32)
    for i in range(dim):
        arr[i] = (expanded[i] / 127.5) - 1.0

    # Normalize to unit length
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


@dataclass
class _MemoryItem:
    """Internal storage for a memory item."""
    id: str
    content: str
    embedding: np.ndarray
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class Memory:
    """
    Minimal in-memory vector store for quick prototyping.

    Example:
        mem = Memory()
        mem.store("hello world")
        results = mem.search("hello")
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        embed_fn: Optional[callable] = None,
    ):
        """Initialize with optional custom embedding function."""
        self._memories: dict[str, _MemoryItem] = {}
        self._embedding_dim = embedding_dim
        self._embed_fn = embed_fn or (lambda t: _hash_embedding(t, embedding_dim))

    def store(
        self,
        content: str,
        id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a memory and return its ID."""
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        memory_id = id or str(uuid.uuid4())
        embedding = self._embed_fn(content)

        self._memories[memory_id] = _MemoryItem(
            id=memory_id,
            content=content,
            embedding=embedding,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )
        return memory_id

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search for similar memories. Returns list sorted by score."""
        if not self._memories:
            return []

        query_embedding = self._embed_fn(query)

        # Compute similarities and sort
        scored: list[tuple[float, _MemoryItem]] = []
        for item in self._memories.values():
            score = _cosine_similarity(query_embedding, item.embedding)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top k
        return [
            {
                "id": item.id,
                "content": item.content,
                "score": score,
                "timestamp": item.timestamp.isoformat(),
                "metadata": item.metadata,
            }
            for score, item in scored[:k]
        ]

    def delete(self, id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        if id in self._memories:
            del self._memories[id]
            return True
        return False

    def get(self, id: str) -> Optional[dict[str, Any]]:
        """Get a single memory by ID."""
        item = self._memories.get(id)
        if item is None:
            return None
        return {
            "id": item.id,
            "content": item.content,
            "timestamp": item.timestamp.isoformat(),
            "metadata": item.metadata,
        }

    def count(self) -> int:
        """Return the number of stored memories."""
        return len(self._memories)

    def clear(self) -> None:
        """Remove all memories."""
        self._memories.clear()


# Module-level convenience functions
_default_memory: Optional[Memory] = None


def _get_default() -> Memory:
    """Get or create the default memory instance."""
    global _default_memory
    if _default_memory is None:
        _default_memory = Memory()
    return _default_memory


def store(content: str, id: Optional[str] = None) -> str:
    """Store a memory using the default instance."""
    return _get_default().store(content, id=id)


def search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Search memories using the default instance."""
    return _get_default().search(query, k=k)


def delete(id: str) -> bool:
    """Delete a memory using the default instance."""
    return _get_default().delete(id)
