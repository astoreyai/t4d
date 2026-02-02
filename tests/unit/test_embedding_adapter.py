"""
Unit tests for T4DM embedding adapter module.

Tests MockEmbeddingAdapter, CachedEmbeddingAdapter, and utility functions.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from t4dm.embedding.adapter import (
    EmbeddingBackend,
    EmbeddingStats,
    MockEmbeddingAdapter,
    CachedEmbeddingAdapter,
    create_adapter,
    get_mock_adapter,
    get_adapter,
    register_adapter,
    clear_adapters,
    cosine_similarity,
    euclidean_distance,
    normalize_embedding,
)


class TestEmbeddingStats:
    """Tests for EmbeddingStats dataclass."""

    def test_default_values(self):
        stats = EmbeddingStats()
        assert stats.total_queries == 0
        assert stats.total_documents == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.errors == 0

    def test_cache_hit_rate_empty(self):
        stats = EmbeddingStats()
        assert stats.cache_hit_rate == 0.0

    def test_cache_hit_rate_calculated(self):
        stats = EmbeddingStats(cache_hits=75, cache_misses=25)
        assert stats.cache_hit_rate == 0.75

    def test_avg_latency_empty(self):
        stats = EmbeddingStats()
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency_calculated(self):
        stats = EmbeddingStats(
            total_queries=10,
            total_documents=10,
            total_latency_ms=200.0
        )
        assert stats.avg_latency_ms == 10.0

    def test_to_dict(self):
        stats = EmbeddingStats(
            total_queries=5,
            cache_hits=3,
            cache_misses=2,
            last_used=datetime(2024, 1, 1, 12, 0, 0)
        )
        result = stats.to_dict()
        assert result["total_queries"] == 5
        assert result["cache_hits"] == 3
        assert result["cache_hit_rate"] == 0.6
        assert "2024-01-01" in result["last_used"]


class TestMockEmbeddingAdapter:
    """Tests for MockEmbeddingAdapter."""

    @pytest.fixture
    def adapter(self):
        return MockEmbeddingAdapter(dimension=128, seed=42)

    def test_creation(self, adapter):
        assert adapter.dimension == 128
        assert adapter.backend == EmbeddingBackend.MOCK

    def test_dimension_property(self, adapter):
        assert adapter.dimension == 128

    @pytest.mark.asyncio
    async def test_embed_query_returns_correct_dimension(self, adapter):
        result = await adapter.embed_query("test query")
        assert len(result) == 128

    @pytest.mark.asyncio
    async def test_embed_query_normalized(self, adapter):
        result = await adapter.embed_query("test query")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_embed_query_deterministic(self, adapter):
        result1 = await adapter.embed_query("same query")
        result2 = await adapter.embed_query("same query")
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_embed_query_different_texts_different_embeddings(self, adapter):
        result1 = await adapter.embed_query("query one")
        result2 = await adapter.embed_query("query two")
        assert result1 != result2

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, adapter):
        result = await adapter.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, adapter):
        texts = ["text one", "text two", "text three"]
        results = await adapter.embed(texts)
        assert len(results) == 3
        for emb in results:
            assert len(emb) == 128
            assert abs(np.linalg.norm(emb) - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_embed_query_np(self, adapter):
        result = await adapter.embed_query_np("test")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (128,)

    @pytest.mark.asyncio
    async def test_embed_np(self, adapter):
        texts = ["a", "b"]
        result = await adapter.embed_np(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 128)

    @pytest.mark.asyncio
    async def test_stats_tracking(self, adapter):
        await adapter.embed_query("q1")
        await adapter.embed_query("q2")
        await adapter.embed(["t1", "t2", "t3"])

        stats = adapter.stats
        assert stats.total_queries == 2
        assert stats.total_documents == 3
        assert stats.last_used is not None

    def test_is_healthy_initial(self, adapter):
        assert adapter.is_healthy() is True

    def test_get_health_status(self, adapter):
        status = adapter.get_health_status()
        assert status["healthy"] is True
        assert status["backend"] == "mock"
        assert status["dimension"] == 128

    def test_clear_stats(self, adapter):
        adapter._stats.total_queries = 100
        adapter.clear_stats()
        assert adapter.stats.total_queries == 0


class TestCachedEmbeddingAdapter:
    """Tests for CachedEmbeddingAdapter."""

    @pytest.fixture
    def base_adapter(self):
        return MockEmbeddingAdapter(dimension=128)

    @pytest.fixture
    def cached_adapter(self, base_adapter):
        return CachedEmbeddingAdapter(
            adapter=base_adapter,
            cache_size=100,
            cache_ttl_seconds=3600
        )

    def test_creation(self, cached_adapter):
        assert cached_adapter.dimension == 128
        assert cached_adapter.backend == EmbeddingBackend.MOCK

    @pytest.mark.asyncio
    async def test_cache_hit(self, cached_adapter):
        # First call - cache miss
        result1 = await cached_adapter.embed_query("test")
        stats1 = cached_adapter.stats
        assert stats1.cache_misses == 1

        # Second call - cache hit
        result2 = await cached_adapter.embed_query("test")
        stats2 = cached_adapter.stats
        assert stats2.cache_hits == 1
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_different_queries(self, cached_adapter):
        await cached_adapter.embed_query("query1")
        await cached_adapter.embed_query("query2")

        stats = cached_adapter.stats
        assert stats.cache_misses == 2

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        base = MockEmbeddingAdapter(dimension=64)
        cached = CachedEmbeddingAdapter(
            adapter=base,
            cache_size=3,
            cache_ttl_seconds=3600
        )

        # Fill cache
        await cached.embed_query("q1")
        await cached.embed_query("q2")
        await cached.embed_query("q3")
        await cached.embed_query("q4")  # Should evict oldest

        # q1 should be evicted
        await cached.embed_query("q1")  # Cache miss
        stats = cached.stats
        assert stats.cache_misses == 5  # 4 initial + 1 re-query

    @pytest.mark.asyncio
    async def test_embed_batch_not_cached(self, cached_adapter):
        # Batch embed doesn't use cache
        result = await cached_adapter.embed(["a", "b"])
        assert len(result) == 2


class TestFactoryFunctions:
    """Tests for adapter factory functions."""

    def test_create_adapter_mock(self):
        adapter = create_adapter(backend=EmbeddingBackend.MOCK, dimension=64)
        assert isinstance(adapter, MockEmbeddingAdapter)
        assert adapter.dimension == 64

    def test_get_mock_adapter(self):
        adapter = get_mock_adapter(dimension=256, seed=123)
        assert adapter.dimension == 256
        assert adapter.backend == EmbeddingBackend.MOCK

    def test_create_adapter_invalid_backend(self):
        with pytest.raises(ValueError):
            create_adapter(backend="invalid")


class TestAdapterRegistry:
    """Tests for global adapter registry."""

    def setup_method(self):
        clear_adapters()

    def teardown_method(self):
        clear_adapters()

    def test_register_and_get(self):
        adapter = MockEmbeddingAdapter(dimension=128)
        register_adapter(adapter, name="test")

        retrieved = get_adapter("test")
        assert retrieved is adapter

    def test_get_nonexistent(self):
        result = get_adapter("nonexistent")
        assert result is None

    def test_register_default(self):
        adapter = MockEmbeddingAdapter()
        register_adapter(adapter)  # Uses "default" name

        retrieved = get_adapter("default")
        assert retrieved is adapter

    def test_clear_adapters(self):
        register_adapter(MockEmbeddingAdapter(), "a")
        register_adapter(MockEmbeddingAdapter(), "b")

        clear_adapters()

        assert get_adapter("a") is None
        assert get_adapter("b") is None


class TestUtilityFunctions:
    """Tests for embedding utility functions."""

    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(a, b) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_opposite(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert abs(cosine_similarity(a, b) + 1.0) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        a = np.zeros(3)
        b = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0

    def test_euclidean_distance_same_point(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(a, b) < 1e-6

    def test_euclidean_distance_unit_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        expected = np.sqrt(2)
        assert abs(euclidean_distance(a, b) - expected) < 1e-6

    def test_normalize_embedding_unit_vector(self):
        emb = np.array([3.0, 4.0, 0.0])  # Length 5
        normalized = normalize_embedding(emb)
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6

    def test_normalize_embedding_already_unit(self):
        emb = np.array([1.0, 0.0, 0.0])
        normalized = normalize_embedding(emb)
        assert np.allclose(emb, normalized)

    def test_normalize_embedding_zero_vector(self):
        emb = np.zeros(3)
        normalized = normalize_embedding(emb)
        assert np.allclose(normalized, emb)  # Returns zero vector unchanged


class TestEmbeddingProviderProtocol:
    """Tests for EmbeddingProvider protocol compliance."""

    @pytest.mark.asyncio
    async def test_mock_adapter_protocol_compliance(self):
        from t4dm.embedding.adapter import EmbeddingProvider

        adapter = MockEmbeddingAdapter(dimension=64)

        # Check protocol methods
        assert isinstance(adapter, EmbeddingProvider)
        assert hasattr(adapter, 'dimension')
        assert hasattr(adapter, 'embed_query')
        assert hasattr(adapter, 'embed')

        # Check methods work
        assert adapter.dimension == 64
        result = await adapter.embed_query("test")
        assert len(result) == 64

    @pytest.mark.asyncio
    async def test_cached_adapter_protocol_compliance(self):
        from t4dm.embedding.adapter import EmbeddingProvider

        base = MockEmbeddingAdapter(dimension=64)
        cached = CachedEmbeddingAdapter(base, cache_size=10)

        assert isinstance(cached, EmbeddingProvider)
        result = await cached.embed_query("test")
        assert len(result) == 64
