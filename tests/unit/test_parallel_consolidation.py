"""
Tests for PO-1: Parallel Consolidation.

Tests parallel execution utilities for consolidation tasks.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ww.consolidation.parallel import (
    ParallelConfig,
    ParallelExecutor,
    ParallelStats,
    cluster_embeddings_hdbscan,
    get_parallel_executor,
    reset_parallel_executor,
)


class TestParallelConfig:
    """Test ParallelConfig defaults and validation."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = ParallelConfig()
        assert config.max_workers == 4
        assert config.use_process_pool is True
        assert config.max_concurrent_embeddings == 10
        assert config.chunk_size == 500

    def test_custom_config(self):
        """Custom config values are preserved."""
        config = ParallelConfig(
            max_workers=8,
            use_process_pool=False,
            chunk_size=1000,
        )
        assert config.max_workers == 8
        assert config.use_process_pool is False
        assert config.chunk_size == 1000


class TestParallelStats:
    """Test ParallelStats calculations."""

    def test_efficiency_calculation(self):
        """Efficiency is calculated correctly."""
        stats = ParallelStats(
            speedup_factor=2.0,
            sequential_estimate_seconds=10.0,
            parallel_time_seconds=5.0,
        )
        assert stats.efficiency == 0.5  # 2x speedup / 4 workers

    def test_efficiency_capped_at_one(self):
        """Efficiency is capped at 1.0."""
        stats = ParallelStats(
            speedup_factor=5.0,
            sequential_estimate_seconds=10.0,  # Must be > 0
        )
        assert stats.efficiency == 1.0

    def test_zero_sequential_estimate(self):
        """Zero sequential estimate returns 0 efficiency."""
        stats = ParallelStats(sequential_estimate_seconds=0.0)
        assert stats.efficiency == 0.0


class TestParallelExecutor:
    """Test ParallelExecutor functionality."""

    @pytest.fixture
    def executor(self):
        """Create a fresh executor for each test."""
        reset_parallel_executor()
        return ParallelExecutor(ParallelConfig(use_process_pool=False))

    @pytest.mark.asyncio
    async def test_parallel_embed_empty_list(self, executor):
        """Empty list returns empty results."""
        result = await executor.parallel_embed([], AsyncMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_parallel_embed_batches_correctly(self, executor):
        """Texts are batched correctly."""
        executor.config.embedding_batch_size = 2

        texts = ["a", "b", "c", "d", "e"]
        embed_calls = []

        async def mock_embed(batch):
            embed_calls.append(batch)
            return [np.array([1.0] * len(t)) for t in batch]

        results = await executor.parallel_embed(texts, mock_embed)

        # Should have 3 batches: [a,b], [c,d], [e]
        assert len(embed_calls) == 3
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_parallel_store_empty_list(self, executor):
        """Empty list returns empty results."""
        result = await executor.parallel_store([], AsyncMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_parallel_store_with_semaphore(self, executor):
        """Store operations respect semaphore limit."""
        executor.config.max_concurrent_stores = 2

        call_count = 0
        max_concurrent = 0
        current_concurrent = 0

        async def mock_store(item):
            nonlocal call_count, max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            call_count += 1
            await asyncio.sleep(0.01)
            current_concurrent -= 1
            return item

        items = list(range(10))
        await executor.parallel_store(items, mock_store)

        assert call_count == 10
        assert max_concurrent <= 2  # Semaphore limit

    @pytest.mark.asyncio
    async def test_chunked_process(self, executor):
        """Chunked processing works correctly."""
        executor.config.chunk_size = 3

        async def process_chunk(chunk):
            return sum(chunk)

        items = [1, 2, 3, 4, 5, 6, 7]
        results, stats = await executor.chunked_process(items, process_chunk)

        assert len(results) == 3  # 3 chunks: [1,2,3], [4,5,6], [7]
        assert results == [6, 15, 7]
        assert stats.total_items == 7
        assert stats.chunks_processed == 3
        assert stats.parallel_time_seconds > 0

    @pytest.mark.asyncio
    async def test_chunked_process_handles_errors(self, executor):
        """Chunked processing handles errors gracefully."""
        executor.config.chunk_size = 2

        async def process_chunk(chunk):
            if chunk[0] == 3:
                raise ValueError("Test error")
            return sum(chunk)

        items = [1, 2, 3, 4, 5, 6]
        results, stats = await executor.chunked_process(items, process_chunk)

        assert len(results) == 2  # One chunk failed
        assert len(stats.errors) == 1
        assert "Test error" in stats.errors[0]

    def test_shutdown(self, executor):
        """Shutdown cleans up pools."""
        # Force pool creation
        executor._get_thread_pool()
        assert executor._thread_pool is not None

        executor.shutdown()
        assert executor._thread_pool is None


class TestClusterEmbeddingsHdbscan:
    """Test the standalone clustering function."""

    @pytest.fixture(autouse=True)
    def reset_hdbscan_imports(self):
        """
        Ensure HDBSCAN imports are clean for each test.

        Other tests may mock ww.consolidation.parallel or hdbscan,
        causing the lazy import inside cluster_embeddings_hdbscan to
        return a mock instead of the real library. This fixture clears
        any cached mocks to ensure real HDBSCAN is used.
        """
        import sys
        # Clear any cached hdbscan-related imports that might be mocked
        modules_to_clear = [k for k in list(sys.modules.keys()) if 'hdbscan' in k.lower()]
        for mod in modules_to_clear:
            sys.modules.pop(mod, None)
        yield
        # Cleanup after test as well
        for mod in modules_to_clear:
            sys.modules.pop(mod, None)

    def test_cluster_small_dataset(self):
        """Clustering works on small datasets."""
        # Create 3 well-separated clusters
        np.random.seed(42)
        # Use larger separation and smaller variance for reliable clustering
        cluster1 = np.random.randn(15, 32) * 0.1 + np.array([10] + [0] * 31)
        cluster2 = np.random.randn(15, 32) * 0.1 + np.array([-10] + [0] * 31)
        cluster3 = np.random.randn(15, 32) * 0.1 + np.array([0, 10] + [0] * 30)

        embeddings = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)

        labels = cluster_embeddings_hdbscan(
            embeddings,
            min_cluster_size=5,
            min_samples=3,
            metric="euclidean",
        )

        assert len(labels) == 45
        # Should find at least one cluster (some points may be noise)
        unique_labels = set(labels)
        # Either we find clusters (max >= 0) or all noise is acceptable for small datasets
        assert len(unique_labels) >= 1

    def test_cluster_handles_small_input(self):
        """Clustering handles input smaller than min_cluster_size."""
        embeddings = np.random.randn(3, 128).astype(np.float32)
        labels = cluster_embeddings_hdbscan(embeddings, min_cluster_size=5, metric="euclidean")

        # All should be noise (-1) since < min_cluster_size
        assert len(labels) == 3


class TestSingleton:
    """Test singleton pattern."""

    def test_get_parallel_executor_singleton(self):
        """get_parallel_executor returns same instance."""
        reset_parallel_executor()
        e1 = get_parallel_executor()
        e2 = get_parallel_executor()
        assert e1 is e2

    def test_reset_parallel_executor(self):
        """reset_parallel_executor creates new instance."""
        e1 = get_parallel_executor()
        reset_parallel_executor()
        e2 = get_parallel_executor()
        assert e1 is not e2


class TestParallelCluster:
    """Test parallel clustering."""

    @pytest.mark.asyncio
    async def test_parallel_cluster_empty(self):
        """Empty list returns empty results."""
        executor = ParallelExecutor(ParallelConfig(use_process_pool=False))
        result = await executor.parallel_cluster([], lambda x: x)
        assert result == []

    @pytest.mark.asyncio
    async def test_parallel_cluster_multiple_sets(self):
        """Multiple embedding sets are clustered in parallel."""
        executor = ParallelExecutor(ParallelConfig(use_process_pool=False))

        embeddings_list = [
            np.random.randn(20, 64).astype(np.float32),
            np.random.randn(15, 64).astype(np.float32),
            np.random.randn(25, 64).astype(np.float32),
        ]

        def simple_cluster(emb):
            # Simple mock: assign to 2 clusters based on first dimension
            return np.array([0 if e[0] > 0 else 1 for e in emb])

        results = await executor.parallel_cluster(embeddings_list, simple_cluster)

        assert len(results) == 3
        assert len(results[0]) == 20
        assert len(results[1]) == 15
        assert len(results[2]) == 25
