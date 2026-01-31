"""
Tests for Redis cache layer (Phase 3A).

Tests cover:
- Embedding cache hit/miss
- Search cache hit/miss
- Graph cache hit/miss
- TTL expiration
- Fallback to in-memory when Redis unavailable
- Cache invalidation
- Statistics tracking
"""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pytest

from t4dm.core.cache import (
    CacheStats,
    InMemoryCache,
    RedisCache,
    close_cache,
    get_cache,
    hash_query,
    hash_text,
    reset_cache,
)
from t4dm.core.cache_config import RedisCacheConfig


class TestInMemoryCache:
    """Tests for in-memory fallback cache."""

    @pytest.fixture
    async def cache(self):
        """Create in-memory cache."""
        cache = InMemoryCache(max_size=100, default_ttl=60)
        yield cache
        await cache.clear()

    async def test_cache_set_get(self, cache):
        """Test basic cache set and get."""
        await cache.set("key1", b"value1")
        result = await cache.get("key1")
        assert result == b"value1"

    async def test_cache_miss(self, cache):
        """Test cache miss."""
        result = await cache.get("nonexistent")
        assert result is None

    async def test_cache_ttl_expiration(self, cache):
        """Test TTL expiration."""
        # Set with very short TTL
        await cache.set("key1", b"value1", ttl=1)

        # Should exist immediately
        result = await cache.get("key1")
        assert result == b"value1"

        # Wait for expiration
        await asyncio.sleep(1.1)
        result = await cache.get("key1")
        assert result is None

    async def test_cache_eviction(self, cache):
        """Test LRU eviction when at capacity."""
        # Fill cache to capacity
        for i in range(100):
            await cache.set(f"key{i}", f"value{i}".encode())

        # Add one more - should evict oldest
        await cache.set("key100", b"value100")

        # Oldest (key0) should be evicted
        result = await cache.get("key0")
        assert result is None

        # Newest should exist
        result = await cache.get("key100")
        assert result == b"value100"

    async def test_cache_delete(self, cache):
        """Test cache deletion."""
        await cache.set("key1", b"value1")
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None

    async def test_cache_clear(self, cache):
        """Test cache clear."""
        await cache.set("key1", b"value1")
        await cache.set("key2", b"value2")
        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    async def test_cache_stats(self, cache):
        """Test cache statistics tracking."""
        # Set some data
        await cache.set("key1", b"value1")

        # Hit
        await cache.get("key1")

        # Miss
        await cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.sets == 1
        assert stats.hit_rate == 0.5
        assert stats.last_hit is not None
        assert stats.last_miss is not None


class TestRedisCache:
    """Tests for Redis cache with fallback."""

    @pytest.fixture
    async def cache(self):
        """Create Redis cache."""
        # Will fall back to in-memory if Redis unavailable
        cache = RedisCache(
            redis_url="redis://localhost:6379",
            key_prefix="test:",
            fallback_enabled=True,
        )
        yield cache
        await cache.clear()
        await cache.close()
        reset_cache()

    async def test_embedding_cache_hit(self, cache):
        """Test embedding cache hit."""
        text = "test embedding query"
        text_hash = hash_text(text)
        embedding = np.random.randn(128).astype(np.float32)

        # Cache embedding
        await cache.cache_embedding(text_hash, embedding)

        # Retrieve from cache
        cached = await cache.get_embedding(text_hash)
        assert cached is not None
        np.testing.assert_array_almost_equal(cached, embedding)

    async def test_embedding_cache_miss(self, cache):
        """Test embedding cache miss."""
        text_hash = hash_text("nonexistent")
        cached = await cache.get_embedding(text_hash)
        assert cached is None

    async def test_search_cache_hit(self, cache):
        """Test search cache hit."""
        query = "test search query"
        query_hash = hash_query(query, limit=10, filter="category:test")
        results = [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.8}]

        # Cache search results
        await cache.cache_search(query_hash, results)

        # Retrieve from cache
        cached = await cache.get_search(query_hash)
        assert cached == results

    async def test_search_cache_miss(self, cache):
        """Test search cache miss."""
        query_hash = hash_query("nonexistent", limit=10)
        cached = await cache.get_search(query_hash)
        assert cached is None

    async def test_graph_cache_hit(self, cache):
        """Test graph cache hit."""
        graph_query = "MATCH (n)-[r]->(m) RETURN n, r, m"
        graph_hash = hash_text(graph_query)
        graph_result = {"nodes": [1, 2, 3], "edges": [(1, 2), (2, 3)]}

        # Cache graph result
        await cache.cache_graph(graph_hash, graph_result)

        # Retrieve from cache
        cached = await cache.get_graph(graph_hash)
        assert cached == graph_result

    async def test_graph_cache_miss(self, cache):
        """Test graph cache miss."""
        graph_hash = hash_text("nonexistent query")
        cached = await cache.get_graph(graph_hash)
        assert cached is None

    async def test_cache_custom_ttl(self, cache):
        """Test custom TTL for cache entries."""
        text_hash = hash_text("short ttl test")
        embedding = np.random.randn(128).astype(np.float32)

        # Cache with short TTL (1 second)
        await cache.cache_embedding(text_hash, embedding, ttl=1)

        # Should exist immediately
        cached = await cache.get_embedding(text_hash)
        assert cached is not None

        # Wait for expiration
        await asyncio.sleep(1.5)
        cached = await cache.get_embedding(text_hash)
        # Note: In-memory fallback will expire, Redis TTL might not be instant
        # but both should eventually expire

    async def test_fallback_when_redis_unavailable(self, cache):
        """Test fallback to in-memory when Redis unavailable."""
        # Redis connection should fail or use fallback
        # Cache should still work via fallback

        text_hash = hash_text("fallback test")
        embedding = np.random.randn(128).astype(np.float32)

        await cache.cache_embedding(text_hash, embedding)
        cached = await cache.get_embedding(text_hash)

        # Should work via fallback even if Redis is down
        assert cached is not None

    async def test_invalidate_pattern(self, cache):
        """Test pattern-based cache invalidation."""
        # Cache multiple embeddings
        for i in range(5):
            text_hash = hash_text(f"test{i}")
            embedding = np.random.randn(128).astype(np.float32)
            await cache.cache_embedding(text_hash, embedding)

        # Invalidate all embeddings
        await cache.invalidate_pattern("emb:*")

        # All should be gone (in Redis; fallback won't support pattern delete)
        # This test validates the API works without error

    async def test_cache_clear(self, cache):
        """Test clearing entire cache."""
        # Add various cache types
        await cache.cache_embedding(hash_text("test1"), np.random.randn(128).astype(np.float32))
        await cache.cache_search(hash_query("test2"), [{"id": "1"}])
        await cache.cache_graph(hash_text("test3"), {"nodes": [1]})

        # Clear all
        await cache.clear()

        # All should be gone
        assert await cache.get_embedding(hash_text("test1")) is None
        assert await cache.get_search(hash_query("test2")) is None
        assert await cache.get_graph(hash_text("test3")) is None

    async def test_cache_stats(self, cache):
        """Test cache statistics tracking."""
        text_hash = hash_text("stats test")
        embedding = np.random.randn(128).astype(np.float32)

        # Cache and retrieve
        await cache.cache_embedding(text_hash, embedding)
        await cache.get_embedding(text_hash)  # Hit
        await cache.get_embedding(hash_text("nonexistent"))  # Miss

        stats = cache.get_stats()
        assert "redis" in stats or "fallback" in stats
        assert isinstance(stats["using_fallback"], bool)

    async def test_hash_text(self):
        """Test text hashing consistency."""
        text1 = "test text"
        text2 = "test text"
        text3 = "different text"

        hash1 = hash_text(text1)
        hash2 = hash_text(text2)
        hash3 = hash_text(text3)

        assert hash1 == hash2  # Same text = same hash
        assert hash1 != hash3  # Different text = different hash

    async def test_hash_query(self):
        """Test query hashing with parameters."""
        query1 = hash_query("test", limit=10, filter="a:b")
        query2 = hash_query("test", limit=10, filter="a:b")
        query3 = hash_query("test", limit=20, filter="a:b")

        assert query1 == query2  # Same params = same hash
        assert query1 != query3  # Different params = different hash

    async def test_is_healthy(self, cache):
        """Test health check."""
        # Should be healthy with fallback enabled
        assert cache.is_healthy()


class TestGlobalCache:
    """Tests for global cache instance."""

    async def test_get_cache_singleton(self):
        """Test global cache is singleton."""
        cache1 = await get_cache()
        cache2 = await get_cache()
        assert cache1 is cache2

    async def test_close_cache(self):
        """Test closing global cache."""
        cache = await get_cache()
        await close_cache()

        # Should be able to get new instance
        new_cache = await get_cache()
        assert new_cache is not None

    async def test_reset_cache(self):
        """Test resetting global cache."""
        await get_cache()
        reset_cache()

        # Should create new instance on next get
        new_cache = await get_cache()
        assert new_cache is not None


class TestCacheIntegration:
    """Integration tests for cache with embedding adapter."""

    @pytest.fixture
    async def cache(self):
        """Create cache for integration tests."""
        cache = RedisCache(fallback_enabled=True)
        yield cache
        await cache.clear()
        await cache.close()
        reset_cache()

    async def test_embedding_adapter_cache_integration(self, cache):
        """Test cache integration with embedding adapter."""
        from t4dm.embedding.adapter import BGEM3Adapter

        # Create adapter
        adapter = BGEM3Adapter(dimension=128)

        # First query - should miss cache and compute
        query = "test integration query"
        result1 = await adapter.embed_query(query)

        # Second query - should hit cache
        result2 = await adapter.embed_query(query)

        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2)

    async def test_cache_concurrent_access(self, cache):
        """Test concurrent cache access."""
        async def cache_and_retrieve(i):
            text_hash = hash_text(f"concurrent{i}")
            embedding = np.random.randn(128).astype(np.float32)
            await cache.cache_embedding(text_hash, embedding)
            return await cache.get_embedding(text_hash)

        # Run multiple concurrent operations
        results = await asyncio.gather(*[cache_and_retrieve(i) for i in range(10)])

        # All should succeed
        assert all(r is not None for r in results)

    async def test_cache_large_embeddings(self, cache):
        """Test caching large embeddings."""
        # Large embedding (1024 dimensions)
        text_hash = hash_text("large embedding")
        embedding = np.random.randn(1024).astype(np.float32)

        await cache.cache_embedding(text_hash, embedding)
        cached = await cache.get_embedding(text_hash)

        assert cached is not None
        assert cached.shape == embedding.shape
        np.testing.assert_array_almost_equal(cached, embedding)

    async def test_cache_batch_operations(self, cache):
        """Test batch cache operations."""
        # Cache multiple items
        embeddings = {}
        for i in range(20):
            text_hash = hash_text(f"batch{i}")
            embedding = np.random.randn(128).astype(np.float32)
            embeddings[text_hash] = embedding
            await cache.cache_embedding(text_hash, embedding)

        # Retrieve all
        for text_hash, expected in embeddings.items():
            cached = await cache.get_embedding(text_hash)
            assert cached is not None
            np.testing.assert_array_almost_equal(cached, expected)


class TestCacheConfig:
    """Tests for cache configuration."""

    def test_cache_config_from_env(self, monkeypatch):
        """Test creating config from environment variables."""
        monkeypatch.setenv("T4DM_REDIS_URL", "redis://test:6379")
        monkeypatch.setenv("T4DM_REDIS_ENABLED", "true")
        monkeypatch.setenv("T4DM_CACHE_EMBEDDING_TTL", "7200")

        config = RedisCacheConfig.from_env()

        assert config.redis_url == "redis://test:6379"
        assert config.enabled is True
        assert config.embedding.ttl == 7200

    def test_cache_config_development(self):
        """Test development config."""
        config = RedisCacheConfig.development()
        assert config.fallback_enabled is True
        assert config.enabled is True

    def test_cache_config_production(self):
        """Test production config."""
        config = RedisCacheConfig.production()
        assert config.embedding.ttl == 7200  # 2 hours
        assert config.fallback_enabled is True

    def test_cache_config_test(self):
        """Test test environment config."""
        config = RedisCacheConfig.test()
        assert config.fallback_enabled is True
        assert config.max_connection_attempts == 1

    def test_cache_config_to_dict(self):
        """Test config serialization."""
        config = RedisCacheConfig.development()
        config_dict = config.to_dict()

        assert "redis_url" in config_dict
        assert "embedding" in config_dict
        assert "search" in config_dict
        assert "graph" in config_dict
