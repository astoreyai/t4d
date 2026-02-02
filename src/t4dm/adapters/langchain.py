"""
T4DM LangChain Adapter.

Provides BaseMemory and BaseRetriever implementations backed by T4DM.
LangChain is an optional dependency — imports are guarded with try/except.
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.memory import BaseMemory
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document

    _HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover
    _HAS_LANGCHAIN = False

    # Provide stubs so the module can be imported without langchain
    class BaseMemory:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class BaseRetriever:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Document:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.page_content = kwargs.get("page_content", "")
            self.metadata = kwargs.get("metadata", {})

    class CallbackManagerForRetrieverRun:  # type: ignore[no-redef]
        pass


def _check_langchain() -> None:
    if not _HAS_LANGCHAIN:
        raise ImportError(
            "langchain_core is required for T4DM LangChain adapters. "
            "Install with: pip install langchain-core"
        )


class T4DMMemory(BaseMemory):
    """
    LangChain BaseMemory backed by T4DM.

    Stores conversation turns as episodic memories and retrieves
    relevant context on each call.

    Example::

        from t4dm.adapters.langchain import T4DMMemory
        from t4dm.sdk.simple import T4DM

        memory = T4DMMemory(t4dm=T4DM(), memory_key="history")
        chain = SomeChain(memory=memory)
    """

    memory_key: str = "history"
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, t4dm=None, **kwargs):
        _check_langchain()
        super().__init__(**kwargs)
        # Store t4dm instance outside pydantic fields
        object.__setattr__(self, "_t4dm", t4dm)

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        t4dm = object.__getattribute__(self, "_t4dm")
        if t4dm is None:
            return {self.memory_key: ""}

        # Use the latest input as query
        query = ""
        if inputs:
            query = str(next(iter(inputs.values())))

        if not query:
            return {self.memory_key: ""}

        results = t4dm.search(query, k=self.k)
        history = "\n".join(r["content"] for r in results)
        return {self.memory_key: history}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        t4dm = object.__getattribute__(self, "_t4dm")
        if t4dm is None:
            return

        input_str = " ".join(str(v) for v in inputs.values())
        output_str = " ".join(str(v) for v in outputs.values())
        t4dm.add(f"Human: {input_str}\nAssistant: {output_str}")

    def clear(self) -> None:
        # T4DM does not support bulk clear; this is a no-op.
        pass


class T4DMRetriever(BaseRetriever):
    """
    LangChain BaseRetriever backed by T4DM semantic search.

    Example::

        retriever = T4DMRetriever(t4dm=T4DM(), k=5)
        docs = retriever.invoke("decorators")
    """

    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, t4dm=None, **kwargs):
        _check_langchain()
        super().__init__(**kwargs)
        object.__setattr__(self, "_t4dm", t4dm)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        t4dm = object.__getattribute__(self, "_t4dm")
        if t4dm is None:
            return []

        results = t4dm.search(query, k=self.k)
        docs = []
        for r in results:
            docs.append(
                Document(
                    page_content=r["content"],
                    metadata={
                        "id": r["id"],
                        "score": r["score"],
                        "timestamp": r["timestamp"],
                        "source": "t4dm",
                    },
                )
            )
        return docs
