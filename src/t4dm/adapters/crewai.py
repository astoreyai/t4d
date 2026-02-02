"""
CrewAI Memory adapter for T4DM.

Provides a CrewAI-compatible memory interface backed by T4DM.

Usage:
    from t4dm.adapters.crewai import T4DMCrewMemory
    memory = T4DMCrewMemory(url="http://localhost:8765")
    memory.save("important fact", metadata={"topic": "ai"}, agent="researcher")
    results = memory.search("ai facts", limit=5)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import crewai  # noqa: F401

    _HAS_CREWAI = True
except ImportError:
    _HAS_CREWAI = False


class T4DMCrewMemory:
    """
    CrewAI-compatible memory adapter backed by T4DM.

    Implements save/search/reset for use as a CrewAI memory backend.
    """

    def __init__(
        self,
        url: str = "http://localhost:8765",
        api_key: str | None = None,
    ):
        from t4dm.sdk.simple import T4DM

        self.t4dm = T4DM(url=url, api_key=api_key)

    def save(
        self,
        value: str,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> None:
        """
        Save a memory value.

        Args:
            value: The content to store.
            metadata: Optional metadata dict.
            agent: Optional agent identifier.
        """
        meta = dict(metadata) if metadata else {}
        if agent:
            meta["agent"] = agent
        self.t4dm.add(content=value, metadata=meta)

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant memories.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.

        Returns:
            List of dicts with 'memory', 'score', and 'metadata' keys.
        """
        results = self.t4dm.search(query=query, k=limit)
        return [
            {
                "memory": r.get("content", ""),
                "score": r.get("score", 0.0),
                "metadata": {},
            }
            for r in results
            if r.get("score", 0.0) >= score_threshold
        ]

    def reset(self) -> None:
        """Reset all memories (best-effort, no bulk delete API)."""
        # The simple SDK doesn't have a reset; use forget with broad query
        try:
            self.t4dm.forget("")
        except Exception:
            pass
