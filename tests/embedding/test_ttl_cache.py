"""Tests for TTL cache."""
import pytest
import time
import threading
from datetime import datetime

# NOTE: Environment variables are set by conftest.py fixtures, not here.
# Setting them at module level pollutes the environment for integration tests.


class TestTTLCache:
    """Tests for TTLCache class."""

    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        from ww.embedding.bge_m3 import TTLCache

        cache = TTLCache(max_size=10, ttl_seconds=60)
        cache.set("key1", "value1")

        assert cache.get("key1") == "value1"

    def test_cache_miss_returns_none(self):
        """Test that missing keys return None."""
        from ww.embedding.bge_m3 import TTLCache

        cache = TTLCache()
        assert cache.get("nonexistent") is None

    def test_cache_ttl_expiry(self):
        """Test that entries expire after TTL."""
        from ww.embedding.bge_m3 import TTLCache

        cache = TTLCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL
        cache.set("key1", "value1")

        assert cache.get("key1") == "value1"

        time.sleep(0.15)  # Wait for expiry

        assert cache.get("key1") is None

    def test_cache_max_size_eviction(self):
        """Test that oldest entries are evicted when full."""
        from ww.embedding.bge_m3 import TTLCache

        cache = TTLCache(max_size=3, ttl_seconds=60)

        cache.set("key1", "value1")
        time.sleep(0.01)
        cache.set("key2", "value2")
        time.sleep(0.01)
        cache.set("key3", "value3")
        time.sleep(0.01)
        cache.set("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_thread_safety(self):
        """Test concurrent access doesn't cause race conditions."""
        from ww.embedding.bge_m3 import TTLCache

        cache = TTLCache(max_size=1000, ttl_seconds=60)
        errors = []

        def writer():
            for i in range(100):
                cache.set(f"key{i}", f"value{i}")

        def reader():
            for i in range(100):
                try:
                    cache.get(f"key{i}")
                except Exception as e:
                    errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_cache_evict_expired(self):
        """Test manual eviction of expired entries."""
        from ww.embedding.bge_m3 import TTLCache

        cache = TTLCache(max_size=10, ttl_seconds=0.1)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        time.sleep(0.15)  # Wait for expiry

        evicted = cache.evict_expired()

        assert evicted == 3
        assert len(cache) == 0

    def test_cache_stats(self):
        """Test cache statistics."""
        from ww.embedding.bge_m3 import TTLCache

        cache = TTLCache(max_size=10, ttl_seconds=60)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.stats

        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_cache_clear(self):
        """Test cache clearing."""
        from ww.embedding.bge_m3 import TTLCache

        cache = TTLCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert len(cache) == 0
        assert cache.get("key1") is None

    def test_cache_update_existing_key(self):
        """Test that updating an existing key refreshes TTL."""
        from ww.embedding.bge_m3 import TTLCache

        cache = TTLCache(max_size=10, ttl_seconds=0.2)

        cache.set("key1", "value1")
        time.sleep(0.15)  # Almost expired

        # Update the key (should refresh TTL)
        cache.set("key1", "value2")

        # Should still be accessible after original TTL
        time.sleep(0.1)
        assert cache.get("key1") == "value2"

    def test_cache_partial_expiry(self):
        """Test that only expired entries are evicted, not all."""
        from ww.embedding.bge_m3 import TTLCache

        cache = TTLCache(max_size=10, ttl_seconds=0.2)

        cache.set("key1", "value1")
        time.sleep(0.15)
        cache.set("key2", "value2")  # Should not be expired
        time.sleep(0.1)  # key1 expired, key2 still valid

        evicted = cache.evict_expired()

        assert evicted == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"


class TestBGEM3EmbeddingWithCache:
    """Tests for BGEM3Embedding caching."""

    @pytest.fixture
    def mock_model(self):
        from unittest.mock import MagicMock
        import numpy as np

        model = MagicMock()
        # Return different embeddings each time to verify caching
        model.encode.side_effect = lambda texts, **kwargs: {
            "dense_vecs": np.random.rand(len(texts), 1024).astype(np.float32)
        }
        return model

    @pytest.mark.asyncio
    async def test_embed_query_uses_cache(self, mock_model):
        """Test that repeated embeds use cache."""
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model):
            from ww.embedding.bge_m3 import BGEM3Embedding

            provider = BGEM3Embedding(
                embedding_cache_size=100, embedding_cache_ttl=60
            )

            result1 = await provider.embed_query("test text")
            result2 = await provider.embed_query("test text")

            # Should only encode once (cache hit on second call)
            assert mock_model.encode.call_count == 1
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_embed_query_different_texts_no_cache(self, mock_model):
        """Test that different texts are not cached together."""
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model):
            from ww.embedding.bge_m3 import BGEM3Embedding

            provider = BGEM3Embedding(
                embedding_cache_size=100, embedding_cache_ttl=60
            )

            result1 = await provider.embed_query("text one")
            result2 = await provider.embed_query("text two")

            # Should encode twice (different queries)
            assert mock_model.encode.call_count == 2
            assert result1 != result2

    @pytest.mark.asyncio
    async def test_cache_ttl_expiry(self, mock_model):
        """Test that cached embeddings expire after TTL."""
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model):
            from ww.embedding.bge_m3 import BGEM3Embedding

            provider = BGEM3Embedding(
                embedding_cache_size=100, embedding_cache_ttl=0.1  # 100ms
            )

            await provider.embed_query("test text")
            time.sleep(0.15)  # Wait for expiry
            await provider.embed_query("test text")

            # Should encode twice (cache expired)
            assert mock_model.encode.call_count == 2

    def test_cache_stats_accessible(self, mock_model):
        """Test that cache stats are accessible."""
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model):
            from ww.embedding.bge_m3 import BGEM3Embedding

            provider = BGEM3Embedding()

            stats = provider.get_cache_stats()

            assert "hits" in stats
            assert "misses" in stats
            assert "size" in stats
            assert "max_size" in stats
            assert "hit_rate" in stats
            assert "ttl_seconds" in stats

    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_model):
        """Test clearing the cache."""
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model):
            from ww.embedding.bge_m3 import BGEM3Embedding

            provider = BGEM3Embedding()

            await provider.embed_query("test text")
            stats_before = provider.get_cache_stats()
            assert stats_before["size"] == 1

            provider.clear_cache()
            stats_after = provider.get_cache_stats()

            assert stats_after["size"] == 0
            assert stats_after["hits"] == 0
            assert stats_after["misses"] == 0

    @pytest.mark.asyncio
    async def test_evict_expired(self, mock_model):
        """Test manual eviction of expired entries."""
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model):
            from ww.embedding.bge_m3 import BGEM3Embedding

            provider = BGEM3Embedding(
                embedding_cache_size=100, embedding_cache_ttl=0.1
            )

            await provider.embed_query("text1")
            await provider.embed_query("text2")

            time.sleep(0.15)  # Wait for expiry

            evicted = provider.evict_expired()

            assert evicted == 2
            assert provider.get_cache_stats()["size"] == 0

    @pytest.mark.asyncio
    async def test_cache_max_size_eviction(self, mock_model):
        """Test that cache evicts oldest entries when full."""
        from unittest.mock import patch

        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_model):
            from ww.embedding.bge_m3 import BGEM3Embedding

            provider = BGEM3Embedding(
                embedding_cache_size=3, embedding_cache_ttl=60
            )

            # Add 4 entries (should evict oldest)
            await provider.embed_query("text1")
            time.sleep(0.01)
            await provider.embed_query("text2")
            time.sleep(0.01)
            await provider.embed_query("text3")
            time.sleep(0.01)
            await provider.embed_query("text4")

            stats = provider.get_cache_stats()
            assert stats["size"] <= 3  # Should not exceed max size
