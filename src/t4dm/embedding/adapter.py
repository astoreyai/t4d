"""
Embedding Adapter for World Weaver.

Provides a clean abstraction layer for embedding providers, enabling:
- Easy swapping between providers (BGE-M3, OpenAI, sentence-transformers)
- Consistent interface across the codebase
- Statistics tracking and health monitoring
- Mock implementation for testing

Hinton-inspired: Distributed representations are the foundation of neural memory.
This adapter ensures consistent, high-quality vector representations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBackend(Enum):
    """Available embedding backends."""
    BGE_M3 = "bge_m3"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"
    MOCK = "mock"


@dataclass
class EmbeddingStats:
    """Statistics for embedding provider."""

    total_queries: int = 0
    total_documents: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    last_used: datetime | None = None

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        total = self.total_queries + self.total_documents
        return self.total_latency_ms / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_queries": self.total_queries,
            "total_documents": self.total_documents,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "errors": self.errors,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query text."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...


class EmbeddingAdapter(ABC):
    """
    Abstract base class for embedding adapters.

    Provides consistent interface with statistics tracking.
    """

    def __init__(self, dimension: int = 1024):
        """
        Initialize adapter.

        Args:
            dimension: Embedding dimension
        """
        self._dimension = dimension
        self._stats = EmbeddingStats()
        self._backend: EmbeddingBackend = EmbeddingBackend.MOCK

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    @property
    def backend(self) -> EmbeddingBackend:
        """Return backend type."""
        return self._backend

    @property
    def stats(self) -> EmbeddingStats:
        """Return current statistics."""
        return self._stats

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query text.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...

    async def embed_query_np(self, query: str) -> np.ndarray:
        """
        Embed query and return numpy array.

        Args:
            query: Query text

        Returns:
            Embedding as numpy array
        """
        embedding = await self.embed_query(query)
        return np.array(embedding, dtype=np.float32)

    async def embed_np(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts and return numpy array.

        Args:
            texts: List of texts

        Returns:
            Embeddings as numpy array (shape: [n, dim])
        """
        embeddings = await self.embed(texts)
        return np.array(embeddings, dtype=np.float32)

    def _record_query(self, latency_ms: float, cache_hit: bool = False) -> None:
        """Record query statistics."""
        self._stats.total_queries += 1
        self._stats.total_latency_ms += latency_ms
        self._stats.last_used = datetime.now()
        if cache_hit:
            self._stats.cache_hits += 1
        else:
            self._stats.cache_misses += 1

    def _record_documents(self, count: int, latency_ms: float) -> None:
        """Record document embedding statistics."""
        self._stats.total_documents += count
        self._stats.total_latency_ms += latency_ms
        self._stats.last_used = datetime.now()

    def _record_error(self) -> None:
        """Record an error."""
        self._stats.errors += 1

    def is_healthy(self) -> bool:
        """Check if adapter is healthy."""
        # Healthy if error rate < 10%
        total = self._stats.total_queries + self._stats.total_documents
        if total == 0:
            return True
        error_rate = self._stats.errors / total
        return error_rate < 0.1

    def get_health_status(self) -> dict:
        """Get detailed health status."""
        return {
            "healthy": self.is_healthy(),
            "backend": self._backend.value,
            "dimension": self._dimension,
            "stats": self._stats.to_dict(),
        }

    def clear_stats(self) -> None:
        """Clear statistics."""
        self._stats = EmbeddingStats()


class BGEM3Adapter(EmbeddingAdapter):
    """
    Adapter for BGE-M3 embedding provider.

    Wraps the existing BGEM3Embedding class with adapter interface.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_fp16: bool = True,
        dimension: int = 1024,
    ):
        """
        Initialize BGE-M3 adapter.

        Args:
            model_name: Model name (default: BAAI/bge-m3)
            device: Device for inference
            use_fp16: Use FP16 precision
            dimension: Embedding dimension
        """
        super().__init__(dimension=dimension)
        self._backend = EmbeddingBackend.BGE_M3
        self._provider: Any | None = None
        self._model_name = model_name
        self._device = device
        self._use_fp16 = use_fp16

    def _ensure_provider(self) -> Any:
        """Ensure provider is initialized."""
        if self._provider is None:
            from t4dm.embedding.bge_m3 import BGEM3Embedding

            self._provider = BGEM3Embedding(
                model_name=self._model_name,
                device=self._device,
                use_fp16=self._use_fp16,
            )
        return self._provider

    async def embed_query(self, query: str) -> list[float]:
        """Embed query using BGE-M3 with Redis cache."""
        import time

        # Try cache first (Phase 3A)
        try:
            from t4dm.core.cache import get_cache, hash_text

            cache = await get_cache()
            text_hash = hash_text(query)
            cached = await cache.get_embedding(text_hash)

            if cached is not None:
                self._record_query(0.5, cache_hit=True)  # Cache hit is fast
                return cached.tolist()
        except Exception as e:
            logger.debug(f"Cache lookup failed, continuing: {e}")

        # Cache miss - compute embedding
        start = time.perf_counter()
        try:
            provider = self._ensure_provider()
            result = await provider.embed_query(query)
            latency = (time.perf_counter() - start) * 1000
            self._record_query(latency, cache_hit=False)

            # Cache result (Phase 3A)
            try:
                import numpy as np

                from t4dm.core.cache import get_cache, hash_text

                cache = await get_cache()
                text_hash = hash_text(query)
                embedding_array = np.array(result, dtype=np.float32)
                await cache.cache_embedding(text_hash, embedding_array)
            except Exception as e:
                logger.debug(f"Cache store failed, continuing: {e}")

            return result
        except Exception as e:
            self._record_error()
            logger.error(f"BGE-M3 embed_query failed: {e}")
            raise

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using BGE-M3."""
        import time

        if not texts:
            return []

        start = time.perf_counter()
        try:
            provider = self._ensure_provider()
            result = await provider.embed(texts)
            latency = (time.perf_counter() - start) * 1000
            self._record_documents(len(texts), latency)
            return result
        except Exception as e:
            self._record_error()
            logger.error(f"BGE-M3 embed failed: {e}")
            raise


class MockEmbeddingAdapter(EmbeddingAdapter):
    """
    Mock embedding adapter for testing.

    Generates deterministic embeddings based on text hash.
    """

    def __init__(self, dimension: int = 128, seed: int = 42):
        """
        Initialize mock adapter.

        Args:
            dimension: Embedding dimension
            seed: Random seed for reproducibility
        """
        super().__init__(dimension=dimension)
        self._backend = EmbeddingBackend.MOCK
        self._seed = seed

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text."""
        # Use text hash as seed for reproducibility
        np.random.seed(hash(text) % (2**32))
        emb = np.random.randn(self._dimension).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # Normalize
        return emb.tolist()

    async def embed_query(self, query: str) -> list[float]:
        """Generate mock query embedding."""
        self._record_query(0.1, cache_hit=False)
        return self._generate_embedding(query)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings."""
        if not texts:
            return []
        self._record_documents(len(texts), 0.1 * len(texts))
        return [self._generate_embedding(text) for text in texts]


class CachedEmbeddingAdapter(EmbeddingAdapter):
    """
    Caching wrapper for embedding adapters.

    Adds an additional caching layer with configurable TTL.
    """

    def __init__(
        self,
        adapter: EmbeddingAdapter,
        cache_size: int = 10000,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize cached adapter.

        Args:
            adapter: Underlying adapter
            cache_size: Maximum cache entries
            cache_ttl_seconds: Cache TTL in seconds
        """
        super().__init__(dimension=adapter.dimension)
        self._adapter = adapter
        self._backend = adapter.backend
        self._cache: dict[str, tuple[list[float], datetime]] = {}
        self._cache_size = cache_size
        self._cache_ttl_seconds = cache_ttl_seconds

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key."""
        from hashlib import md5
        return md5(text.encode(), usedforsecurity=False).hexdigest()

    def _is_cached(self, key: str) -> bool:
        """Check if key is cached and not expired."""
        if key not in self._cache:
            return False
        _, timestamp = self._cache[key]
        age = (datetime.now() - timestamp).total_seconds()
        if age > self._cache_ttl_seconds:
            del self._cache[key]
            return False
        return True

    def _evict_if_needed(self) -> None:
        """Evict oldest entry if cache is full."""
        if len(self._cache) >= self._cache_size:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            del self._cache[oldest_key]

    async def embed_query(self, query: str) -> list[float]:
        """Embed query with caching."""
        import time

        key = self._get_cache_key(query)

        if self._is_cached(key):
            self._record_query(0.01, cache_hit=True)
            return self._cache[key][0]

        start = time.perf_counter()
        result = await self._adapter.embed_query(query)
        latency = (time.perf_counter() - start) * 1000

        self._evict_if_needed()
        self._cache[key] = (result, datetime.now())
        self._record_query(latency, cache_hit=False)

        return result

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts (no caching for batch)."""
        return await self._adapter.embed(texts)


# Factory functions

def create_adapter(
    backend: EmbeddingBackend = EmbeddingBackend.BGE_M3,
    dimension: int = 1024,
    **kwargs
) -> EmbeddingAdapter:
    """
    Create embedding adapter for specified backend.

    Args:
        backend: Backend type
        dimension: Embedding dimension
        **kwargs: Backend-specific arguments

    Returns:
        Configured adapter
    """
    if backend == EmbeddingBackend.BGE_M3:
        return BGEM3Adapter(dimension=dimension, **kwargs)
    if backend == EmbeddingBackend.MOCK:
        return MockEmbeddingAdapter(dimension=dimension, **kwargs)
    raise ValueError(f"Unsupported backend: {backend}")


def get_mock_adapter(dimension: int = 128, seed: int = 42) -> MockEmbeddingAdapter:
    """
    Get mock adapter for testing.

    Args:
        dimension: Embedding dimension
        seed: Random seed

    Returns:
        Mock adapter
    """
    return MockEmbeddingAdapter(dimension=dimension, seed=seed)


# Global adapter registry for singleton access

_adapters: dict[str, EmbeddingAdapter] = {}


def get_adapter(name: str = "default") -> EmbeddingAdapter | None:
    """Get adapter by name from registry."""
    return _adapters.get(name)


def register_adapter(adapter: EmbeddingAdapter, name: str = "default") -> None:
    """Register adapter in global registry."""
    _adapters[name] = adapter
    logger.info(f"Registered embedding adapter '{name}' with backend {adapter.backend.value}")


def clear_adapters() -> None:
    """Clear adapter registry."""
    _adapters.clear()


# Utility functions

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Euclidean distance (>= 0)
    """
    return float(np.linalg.norm(a - b))


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length.

    Args:
        embedding: Input embedding

    Returns:
        Normalized embedding
    """
    norm = np.linalg.norm(embedding)
    if norm < 1e-8:
        return embedding
    return embedding / norm
