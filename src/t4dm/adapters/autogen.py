"""
T4DM AutoGen Adapter.

Provides a memory interface compatible with Microsoft AutoGen agents.
AutoGen is an optional dependency — imports are guarded with try/except.
"""

from __future__ import annotations

from typing import Any


class T4DMAutoGenMemory:
    """
    AutoGen-compatible memory backed by T4DM.

    Implements the interface expected by AutoGen agents for memory
    query, update, add, clear, and close operations.

    Example::

        from t4dm.adapters.autogen import T4DMAutoGenMemory
        from t4dm.sdk.simple import T4DM

        memory = T4DMAutoGenMemory(t4dm=T4DM())
        result = memory.query("What do I know about decorators?")
    """

    def __init__(self, t4dm=None, k: int = 5):
        self._t4dm = t4dm
        self._k = k

    def query(self, query: str, **kwargs: Any) -> str:
        """Query memory and return relevant context as a string."""
        if self._t4dm is None:
            return ""
        k = kwargs.get("k", self._k)
        results = self._t4dm.search(query, k=k)
        if not results:
            return ""
        return "\n".join(r["content"] for r in results)

    def update_context(self, context: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Update context dict with relevant memories."""
        if self._t4dm is None:
            return context

        # Extract a query from context
        query = ""
        for key in ("input", "query", "message", "content"):
            if key in context:
                query = str(context[key])
                break

        if not query:
            return context

        results = self._t4dm.search(query, k=self._k)
        if results:
            context["memory"] = "\n".join(r["content"] for r in results)
            context["memory_ids"] = [r["id"] for r in results]
        return context

    def add(self, content: str, **kwargs: Any) -> None:
        """Add a memory."""
        if self._t4dm is None:
            return
        self._t4dm.add(content, **{
            k: v for k, v in kwargs.items()
            if k in ("tags", "importance", "metadata")
        })

    def clear(self) -> None:
        """Clear is a no-op (T4DM does not support bulk clear)."""
        pass

    def close(self) -> None:
        """Close the underlying T4DM client."""
        if self._t4dm is not None:
            self._t4dm.close()
