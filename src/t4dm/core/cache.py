"""
Redis Caching Layer for World Weaver.

Phase 3A: Multi-tier caching for embeddings and search results.

Architecture:
- Primary: Redis for distributed caching
- Fallback: In-memory LRU cache when Redis unavailable
- Graceful degradation: System continues without cache on failure

Tier TTLs:
- Embeddings: 1 hour (expensive to compute)
- Search results: 5 minutes (query-dependent)
- Graph traversal: 10 minutes (semi-static)
"""

import asyncio
import logging
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import md5
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    errors: int = 0
    evictions: int = 0
    last_hit: datetime | None = None
    last_miss: datetime | None = None

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "errors": self.errors,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "last_hit": self.last_hit.isoformat() if self.last_hit else None,
            "last_miss": self.last_miss.isoformat() if self.last_miss else None,
        }


class InMemoryCache:
    """
    In-memory LRU cache fallback.

    Used when Redis is unavailable or for local testing.
    Thread-safe with asyncio lock.
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self._cache: OrderedDict[str, tuple[bytes, datetime]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    async def get(self, key: str) -> bytes | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                self._stats.last_miss = datetime.now()
                return None

            value, expiry = self._cache[key]

            # Check expiration
            if datetime.now() > expiry:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.last_miss = datetime.now()
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            self._stats.last_hit = datetime.now()
            return value

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (None = default)
        """
        async with self._lock:
            ttl = ttl or self._default_ttl
            expiry = datetime.now() + timedelta(seconds=ttl)

            # Evict if at capacity
            if key not in self._cache and len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Remove oldest
                self._stats.evictions += 1

            self._cache[key] = (value, expiry)
            self._cache.move_to_end(key)
            self._stats.sets += 1

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def clear_stats(self) -> None:
        """Clear statistics."""
        self._stats = CacheStats()


class RedisCache:
    """
    Multi-tier Redis caching for embeddings and search results.

    Features:
    - Configurable TTLs per data type
    - Graceful degradation to in-memory cache
    - Statistics tracking
    - Async/await throughout
    """

    # TTL constants (seconds)
    EMBEDDING_TTL = 3600  # 1 hour - expensive to compute
    SEARCH_TTL = 300  # 5 minutes - query-dependent
    GRAPH_TTL = 600  # 10 minutes - semi-static

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "t4dm:",
        fallback_enabled: bool = True,
        fallback_max_size: int = 10000,
    ):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys
            fallback_enabled: Enable in-memory fallback
            fallback_max_size: Max size for fallback cache
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._fallback_enabled = fallback_enabled
        self._redis_client: Any | None = None
        self._fallback_cache = InMemoryCache(max_size=fallback_max_size)
        self._stats = CacheStats()
        self._using_fallback = False
        self._connection_attempts = 0
        self._max_connection_attempts = 3
        self._lock = asyncio.Lock()

    async def _ensure_redis(self) -> Any | None:
        """
        Ensure Redis connection, return None if unavailable.

        Returns:
            Redis client or None if connection failed
        """
        if self._redis_client is not None:
            return self._redis_client

        # Skip if already exhausted attempts
        if self._connection_attempts >= self._max_connection_attempts:
            return None

        async with self._lock:
            # Double-check after acquiring lock
            if self._redis_client is not None:
                return self._redis_client

            try:
                import redis.asyncio as redis

                self._redis_client = redis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=False,  # We handle bytes
                )
                # Test connection
                await self._redis_client.ping()
                logger.info(f"Redis cache connected: {self._redis_url}")
                self._using_fallback = False
                self._connection_attempts = 0
                return self._redis_client

            except ImportError:
                logger.warning(
                    "redis package not installed. Using in-memory fallback. "
                    "Install with: pip install redis"
                )
                self._using_fallback = True
                self._connection_attempts = self._max_connection_attempts
                return None

            except Exception as e:
                self._connection_attempts += 1
                logger.warning(
                    f"Redis connection failed (attempt {self._connection_attempts}/"
                    f"{self._max_connection_attempts}): {e}"
                )

                if self._connection_attempts >= self._max_connection_attempts:
                    logger.warning(
                        "Max Redis connection attempts reached. "
                        "Using in-memory fallback for this session."
                    )
                    self._using_fallback = True

                return None

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self._key_prefix}{key}"

    async def get_embedding(self, text_hash: str) -> np.ndarray | None:
        """
        Get cached embedding.

        Args:
            text_hash: Hash of text (use hash_text helper)

        Returns:
            Embedding array or None if not cached
        """
        key = self._make_key(f"emb:{text_hash}")

        # Try Redis first
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                data = await redis.get(key)
                if data:
                    embedding = pickle.loads(data)
                    self._stats.hits += 1
                    self._stats.last_hit = datetime.now()
                    return embedding

                self._stats.misses += 1
                self._stats.last_miss = datetime.now()
                return None

            except Exception as e:
                logger.error(f"Redis get_embedding error: {e}")
                self._stats.errors += 1
                # Fall through to fallback

        # Fallback to in-memory
        if self._fallback_enabled:
            data = await self._fallback_cache.get(key)
            if data:
                return pickle.loads(data)

        return None

    async def cache_embedding(
        self, text_hash: str, embedding: np.ndarray, ttl: int | None = None
    ) -> None:
        """
        Cache embedding with TTL.

        Args:
            text_hash: Hash of text
            embedding: Embedding array to cache
            ttl: TTL in seconds (None = default EMBEDDING_TTL)
        """
        ttl = ttl or self.EMBEDDING_TTL
        key = self._make_key(f"emb:{text_hash}")
        data = pickle.dumps(embedding)

        # Try Redis first
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.setex(key, ttl, data)
                self._stats.sets += 1
                return
            except Exception as e:
                logger.error(f"Redis cache_embedding error: {e}")
                self._stats.errors += 1
                # Fall through to fallback

        # Fallback to in-memory
        if self._fallback_enabled:
            await self._fallback_cache.set(key, data, ttl)

    async def get_search(self, query_hash: str) -> list | None:
        """
        Get cached search results.

        Args:
            query_hash: Hash of query (use hash_query helper)

        Returns:
            Cached search results or None
        """
        key = self._make_key(f"search:{query_hash}")

        # Try Redis first
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                data = await redis.get(key)
                if data:
                    results = pickle.loads(data)
                    self._stats.hits += 1
                    self._stats.last_hit = datetime.now()
                    return results

                self._stats.misses += 1
                self._stats.last_miss = datetime.now()
                return None

            except Exception as e:
                logger.error(f"Redis get_search error: {e}")
                self._stats.errors += 1
                # Fall through to fallback

        # Fallback to in-memory
        if self._fallback_enabled:
            data = await self._fallback_cache.get(key)
            if data:
                return pickle.loads(data)

        return None

    async def cache_search(
        self, query_hash: str, results: list, ttl: int | None = None
    ) -> None:
        """
        Cache search results with TTL.

        Args:
            query_hash: Hash of query
            results: Search results to cache
            ttl: TTL in seconds (None = default SEARCH_TTL)
        """
        ttl = ttl or self.SEARCH_TTL
        key = self._make_key(f"search:{query_hash}")
        data = pickle.dumps(results)

        # Try Redis first
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.setex(key, ttl, data)
                self._stats.sets += 1
                return
            except Exception as e:
                logger.error(f"Redis cache_search error: {e}")
                self._stats.errors += 1
                # Fall through to fallback

        # Fallback to in-memory
        if self._fallback_enabled:
            await self._fallback_cache.set(key, data, ttl)

    async def get_graph(self, graph_hash: str) -> Any | None:
        """
        Get cached graph traversal result.

        Args:
            graph_hash: Hash of graph query

        Returns:
            Cached graph result or None
        """
        key = self._make_key(f"graph:{graph_hash}")

        # Try Redis first
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                data = await redis.get(key)
                if data:
                    result = pickle.loads(data)
                    self._stats.hits += 1
                    self._stats.last_hit = datetime.now()
                    return result

                self._stats.misses += 1
                self._stats.last_miss = datetime.now()
                return None

            except Exception as e:
                logger.error(f"Redis get_graph error: {e}")
                self._stats.errors += 1
                # Fall through to fallback

        # Fallback to in-memory
        if self._fallback_enabled:
            data = await self._fallback_cache.get(key)
            if data:
                return pickle.loads(data)

        return None

    async def cache_graph(
        self, graph_hash: str, result: Any, ttl: int | None = None
    ) -> None:
        """
        Cache graph traversal result.

        Args:
            graph_hash: Hash of graph query
            result: Graph result to cache
            ttl: TTL in seconds (None = default GRAPH_TTL)
        """
        ttl = ttl or self.GRAPH_TTL
        key = self._make_key(f"graph:{graph_hash}")
        data = pickle.dumps(result)

        # Try Redis first
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.setex(key, ttl, data)
                self._stats.sets += 1
                return
            except Exception as e:
                logger.error(f"Redis cache_graph error: {e}")
                self._stats.errors += 1
                # Fall through to fallback

        # Fallback to in-memory
        if self._fallback_enabled:
            await self._fallback_cache.set(key, data, ttl)

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "emb:*")

        Returns:
            Number of keys deleted
        """
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                full_pattern = self._make_key(pattern)
                cursor = 0
                deleted = 0

                while True:
                    cursor, keys = await redis.scan(cursor, match=full_pattern, count=100)
                    if keys:
                        deleted += await redis.delete(*keys)
                    if cursor == 0:
                        break

                logger.info(f"Invalidated {deleted} keys matching {pattern}")
                return deleted

            except Exception as e:
                logger.error(f"Redis invalidate_pattern error: {e}")
                self._stats.errors += 1

        return 0

    async def clear(self) -> None:
        """Clear all cache entries."""
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.flushdb()
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
                self._stats.errors += 1

        if self._fallback_enabled:
            await self._fallback_cache.clear()

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis_client is not None:
            try:
                await self._redis_client.close()
                logger.info("Redis cache connection closed")
            except Exception as e:
                logger.error(f"Redis close error: {e}")
            finally:
                self._redis_client = None

    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "redis": self._stats.to_dict(),
            "using_fallback": self._using_fallback,
            "connection_attempts": self._connection_attempts,
        }

        if self._fallback_enabled:
            stats["fallback"] = self._fallback_cache.get_stats().to_dict()

        return stats

    def clear_stats(self) -> None:
        """Clear statistics."""
        self._stats = CacheStats()
        if self._fallback_enabled:
            self._fallback_cache.clear_stats()

    def is_healthy(self) -> bool:
        """Check if cache is healthy (Redis connected or fallback working)."""
        # Healthy if Redis is connected
        if self._redis_client is not None:
            return True
        # Healthy if fallback is enabled (even if not yet used)
        return self._fallback_enabled


# =============================================================================
# Helper Functions
# =============================================================================


def hash_text(text: str) -> str:
    """
    Generate hash for text.

    Args:
        text: Text to hash

    Returns:
        MD5 hash hex string
    """
    return md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def hash_query(query: str, **params) -> str:
    """
    Generate hash for query with parameters.

    Args:
        query: Query text
        **params: Additional query parameters

    Returns:
        Hash of query and parameters
    """
    # Sort params for consistent hashing
    param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    combined = f"{query}|{param_str}"
    return md5(combined.encode("utf-8"), usedforsecurity=False).hexdigest()


# =============================================================================
# Global Cache Instance
# =============================================================================

_cache_instance: RedisCache | None = None
_cache_lock = asyncio.Lock()


async def get_cache(
    redis_url: str | None = None, key_prefix: str = "t4dm:"
) -> RedisCache:
    """
    Get or create global cache instance.

    Args:
        redis_url: Redis URL (None = use default)
        key_prefix: Cache key prefix

    Returns:
        RedisCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        async with _cache_lock:
            # Double-check after lock
            if _cache_instance is None:
                redis_url = redis_url or "redis://localhost:6379"
                _cache_instance = RedisCache(
                    redis_url=redis_url, key_prefix=key_prefix
                )

    return _cache_instance


async def close_cache() -> None:
    """Close global cache instance."""
    global _cache_instance

    if _cache_instance is not None:
        await _cache_instance.close()
        _cache_instance = None


def reset_cache() -> None:
    """Reset cache instance (for testing)."""
    global _cache_instance
    _cache_instance = None
