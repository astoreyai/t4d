"""
T4DM Simple API - Ultra-simple 3-line memory interface.

Competing with Mem0 for simplicity:
    from t4dm import T4DM
    m = T4DM()
    m.add("learned something")

Uses WorldWeaverClient internally for all HTTP communication.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class T4DM:
    """
    Ultra-simple memory API.

    Example::

        m = T4DM()
        mid = m.add("Python decorators modify functions")
        results = m.search("decorators")
        m.delete(mid)
    """

    def __init__(
        self,
        url: str = "http://localhost:8765",
        api_key: str | None = None,
        user_id: str = "default",
    ):
        from t4dm.sdk.client import WorldWeaverClient

        self._user_id = user_id
        self._api_key = api_key
        self._client = WorldWeaverClient(
            base_url=url,
            session_id=user_id,
            timeout=30.0,
        )
        self._client.connect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store an episodic memory. Returns the episode ID as a string."""
        valence = importance if importance is not None else 0.5
        project = None
        if tags:
            project = ",".join(tags)
        episode = self._client.create_episode(
            content=content,
            project=project,
            outcome="neutral",
            emotional_valence=valence,
        )
        return str(episode.id)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search. Returns list of {id, content, score, timestamp}."""
        result = self._client.recall_episodes(query=query, limit=k)
        items = []
        for episode, score in zip(result.episodes, result.scores):
            items.append(
                {
                    "id": str(episode.id),
                    "content": episode.content,
                    "score": score,
                    "timestamp": episode.timestamp.isoformat(),
                }
            )
        # Client-side time filtering if requested
        if time_range is not None:
            start, end = time_range
            start_iso = start.isoformat()
            end_iso = end.isoformat()
            items = [
                i for i in items if start_iso <= i["timestamp"] <= end_iso
            ]
        return items

    def get(self, memory_id: str) -> dict[str, Any]:
        """Get a single memory by ID."""
        from uuid import UUID

        episode = self._client._request("GET", f"/episodes/{memory_id}")
        return {
            "id": str(episode.get("id", memory_id)),
            "content": episode.get("content", ""),
            "timestamp": episode.get("timestamp", ""),
            "outcome": episode.get("outcome", "neutral"),
            "importance": episode.get("emotional_valence", 0.5),
        }

    def get_all(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """List all memories."""
        data = self._client._request(
            "GET", "/episodes", params={"page": 1, "page_size": limit}
        )
        items = []
        for ep in data.get("episodes", []):
            items.append(
                {
                    "id": str(ep.get("id", "")),
                    "content": ep.get("content", ""),
                    "timestamp": ep.get("timestamp", ""),
                    "outcome": ep.get("outcome", "neutral"),
                    "importance": ep.get("emotional_valence", 0.5),
                }
            )
        return items

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True on success."""
        try:
            self._client._request("DELETE", f"/episodes/{memory_id}")
            return True
        except Exception:
            return False

    def forget(self, query: str) -> int:
        """Search and delete matching memories. Returns count deleted."""
        results = self.search(query, k=100)
        count = 0
        for item in results:
            if self.delete(item["id"]):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> T4DM:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
