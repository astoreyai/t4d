"""Storage providers for T4DM.

All storage is handled by the embedded T4DX engine.
Legacy Neo4j/Qdrant stores have been removed.
"""

import logging
import threading
from pathlib import Path

from t4dm.storage.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    get_circuit_breaker,
    get_storage_circuit_breakers,
    reset_all_circuit_breakers,
)
from t4dm.storage.t4dx import T4DXEngine, T4DXGraphStore, T4DXVectorStore

logger = logging.getLogger(__name__)

# Singleton engine instances per session
_engines: dict[str, T4DXEngine] = {}
_vector_stores: dict[str, T4DXVectorStore] = {}
_graph_stores: dict[str, T4DXGraphStore] = {}
_lock = threading.RLock()


def _get_data_dir() -> Path:
    """Get the T4DX data directory from settings."""
    try:
        from t4dm.core.config import get_settings
        settings = get_settings()
        base = Path(getattr(settings, "data_dir", ".data"))
    except Exception:
        base = Path(".data")
    return base / "t4dx"


def get_engine(session_id: str = "default") -> T4DXEngine:
    """Get or create T4DX engine for session."""
    with _lock:
        if session_id not in _engines:
            data_dir = _get_data_dir() / session_id
            engine = T4DXEngine(data_dir)
            engine.startup()
            _engines[session_id] = engine
            logger.info(f"T4DX engine started for session: {session_id}")
        return _engines[session_id]


def get_vector_store(session_id: str = "default") -> T4DXVectorStore:
    """Get T4DX vector store adapter (VectorStore protocol)."""
    with _lock:
        if session_id not in _vector_stores:
            engine = get_engine(session_id)
            _vector_stores[session_id] = T4DXVectorStore(engine)
        return _vector_stores[session_id]


def get_graph_store(session_id: str = "default") -> T4DXGraphStore:
    """Get T4DX graph store adapter (GraphStore protocol)."""
    with _lock:
        if session_id not in _graph_stores:
            engine = get_engine(session_id)
            _graph_stores[session_id] = T4DXGraphStore(engine)
        return _graph_stores[session_id]


async def close_stores(session_id: str | None = None) -> None:
    """Shut down T4DX engine(s) and clear caches."""
    with _lock:
        if session_id:
            if session_id in _engines:
                _engines[session_id].shutdown()
                del _engines[session_id]
                _vector_stores.pop(session_id, None)
                _graph_stores.pop(session_id, None)
                logger.info(f"T4DX engine closed for session: {session_id}")
        else:
            for sid, engine in _engines.items():
                engine.shutdown()
                logger.info(f"T4DX engine closed for session: {sid}")
            _engines.clear()
            _vector_stores.clear()
            _graph_stores.clear()


def reset_stores() -> None:
    """Reset all store caches (for testing)."""
    with _lock:
        _engines.clear()
        _vector_stores.clear()
        _graph_stores.clear()


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "T4DXEngine",
    "T4DXGraphStore",
    "T4DXVectorStore",
    "close_stores",
    "get_circuit_breaker",
    "get_engine",
    "get_graph_store",
    "get_storage_circuit_breakers",
    "get_vector_store",
    "reset_all_circuit_breakers",
    "reset_stores",
]
