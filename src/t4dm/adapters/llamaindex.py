"""
T4DM LlamaIndex Adapter.

Provides a VectorStore implementation backed by T4DM.
LlamaIndex is an optional dependency — imports are guarded with try/except.
"""

from __future__ import annotations

from typing import Any

try:
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.schema import BaseNode, TextNode

    _HAS_LLAMAINDEX = True
except ImportError:  # pragma: no cover
    _HAS_LLAMAINDEX = False

    class BasePydanticVectorStore:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class VectorStoreQuery:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.query_str = kwargs.get("query_str", "")
            self.similarity_top_k = kwargs.get("similarity_top_k", 5)

    class VectorStoreQueryResult:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.nodes = kwargs.get("nodes", [])
            self.similarities = kwargs.get("similarities", [])
            self.ids = kwargs.get("ids", [])

    class BaseNode:  # type: ignore[no-redef]
        pass

    class TextNode:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.text = kwargs.get("text", "")
            self.id_ = kwargs.get("id_", "")
            self.metadata = kwargs.get("metadata", {})

        def get_content(self, **kwargs) -> str:
            return self.text


def _check_llamaindex() -> None:
    if not _HAS_LLAMAINDEX:
        raise ImportError(
            "llama_index is required for T4DM LlamaIndex adapters. "
            "Install with: pip install llama-index-core"
        )


class T4DMVectorStore(BasePydanticVectorStore):
    """
    LlamaIndex VectorStore backed by T4DM.

    Example::

        from t4dm.adapters.llamaindex import T4DMVectorStore
        from t4dm.sdk.simple import T4DM

        store = T4DMVectorStore(t4dm=T4DM())
        index = VectorStoreIndex.from_vector_store(store)
    """

    stores_text: bool = True

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, t4dm=None, **kwargs):
        _check_llamaindex()
        super().__init__(**kwargs)
        object.__setattr__(self, "_t4dm", t4dm)

    @classmethod
    def class_name(cls) -> str:
        return "T4DMVectorStore"

    @property
    def client(self) -> Any:
        return object.__getattribute__(self, "_t4dm")

    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        """Add nodes to T4DM. Returns list of IDs."""
        t4dm = object.__getattribute__(self, "_t4dm")
        ids = []
        for node in nodes:
            text = node.get_content() if hasattr(node, "get_content") else str(node)
            mid = t4dm.add(text)
            ids.append(mid)
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete a document by ID."""
        t4dm = object.__getattribute__(self, "_t4dm")
        t4dm.delete(ref_doc_id)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the vector store."""
        t4dm = object.__getattribute__(self, "_t4dm")
        query_str = getattr(query, "query_str", "") or ""
        top_k = getattr(query, "similarity_top_k", 5) or 5

        results = t4dm.search(query_str, k=top_k)

        nodes = []
        similarities = []
        ids = []
        for r in results:
            node = TextNode(
                text=r["content"],
                id_=r["id"],
                metadata={"timestamp": r["timestamp"], "source": "t4dm"},
            )
            nodes.append(node)
            similarities.append(r["score"])
            ids.append(r["id"])

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )
