"""
T4DM Framework Adapters.

Optional integrations with LangChain, LlamaIndex, and AutoGen.
All imports are lazy — no hard dependencies on any framework.
"""

from t4dm.sdk.simple import T4DM

__all__ = ["T4DM"]

# Lazy imports for framework adapters
def __getattr__(name: str):
    if name == "T4DMMemory":
        from t4dm.adapters.langchain import T4DMMemory
        return T4DMMemory
    if name == "T4DMRetriever":
        from t4dm.adapters.langchain import T4DMRetriever
        return T4DMRetriever
    if name == "T4DMVectorStore":
        from t4dm.adapters.llamaindex import T4DMVectorStore
        return T4DMVectorStore
    if name == "T4DMAutoGenMemory":
        from t4dm.adapters.autogen import T4DMAutoGenMemory
        return T4DMAutoGenMemory
    if name == "T4DMCrewMemory":
        from t4dm.adapters.crewai import T4DMCrewMemory
        return T4DMCrewMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
