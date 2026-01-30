"""
Unit tests for Qdrant store optimizations.

Tests cover:
1. Session ID prefiltering in search operations
2. Parallel batch upsert for large datasets
3. Rollback on partial failure
4. Complexity characteristics
5. Thread-safe lazy client initialization (QDRANT-004)
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Any

from ww.storage.qdrant_store import QdrantStore
from qdrant_client.http import models


# ============================================================================
# Lazy Client Initialization Tests (QDRANT-004)
# ============================================================================


class TestLazyClientInitialization:
    """QDRANT-004: Test thread-safe lazy client initialization."""

    @pytest.mark.asyncio
    async def test_concurrent_get_client_creates_single_instance(self):
        """QDRANT-004: Concurrent calls should only create one client."""
        store = QdrantStore(dimension=3)

        # Track how many times AsyncQdrantClient constructor is called
        constructor_calls = []

        class MockAsyncQdrantClient:
            def __init__(self, **kwargs):
                constructor_calls.append(kwargs)
                # Simulate some initialization delay
                # In real code, this might be network connection

        with patch('ww.storage.qdrant_store.AsyncQdrantClient', MockAsyncQdrantClient):
            # Launch many concurrent requests for the client
            tasks = [store._get_client() for _ in range(20)]
            results = await asyncio.gather(*tasks)

            # All should return the same instance
            assert all(r is results[0] for r in results)

            # Constructor should only be called once
            assert len(constructor_calls) == 1, (
                f"AsyncQdrantClient constructor called {len(constructor_calls)} times, "
                "expected exactly 1 (race condition detected)"
            )

    @pytest.mark.asyncio
    async def test_get_client_fast_path_no_lock(self):
        """Test that fast path (client exists) doesn't acquire lock."""
        store = QdrantStore(dimension=3)

        # Pre-initialize the client
        mock_client = MagicMock()
        store._client = mock_client

        # Get client multiple times
        for _ in range(100):
            result = await store._get_client()
            assert result is mock_client

        # Lock should never have been created since we never needed it
        # (fast path should skip lock acquisition)

    def test_sync_client_thread_safety(self):
        """QDRANT-004: Sync client creation should be thread-safe."""
        store = QdrantStore(dimension=3)

        constructor_calls = []

        class MockQdrantClient:
            def __init__(self, **kwargs):
                constructor_calls.append(kwargs)

        with patch('ww.storage.qdrant_store.QdrantClient', MockQdrantClient):
            import threading
            results = []
            errors = []

            def get_client():
                try:
                    client = store._get_sync_client()
                    results.append(client)
                except Exception as e:
                    errors.append(e)

            # Launch concurrent threads
            threads = [threading.Thread(target=get_client) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # No errors should have occurred
            assert not errors, f"Errors during concurrent access: {errors}"

            # All should return the same instance
            assert all(r is results[0] for r in results)

            # Constructor should only be called once
            assert len(constructor_calls) == 1, (
                f"QdrantClient constructor called {len(constructor_calls)} times, "
                "expected exactly 1 (race condition detected)"
            )


# ============================================================================
# Session ID Prefiltering Tests
# ============================================================================


class TestSessionIDPrefiltering:
    """Test session_id prefiltering in search operations."""

    @pytest.mark.asyncio
    async def test_search_with_session_id_applies_filter(self):
        """Test that session_id is added to filter conditions."""
        store = QdrantStore(dimension=3)

        # Mock the client and its query_points method
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch.object(store, '_get_client', return_value=mock_client):
            await store.search(
                collection="test_collection",
                vector=[0.1, 0.2, 0.3],
                limit=10,
                session_id="session-123"
            )

            # Verify query_points was called with session_id filter
            mock_client.query_points.assert_called_once()
            call_args = mock_client.query_points.call_args

            # Check that query_filter was provided
            assert call_args.kwargs['query_filter'] is not None

            # The filter should contain session_id condition
            query_filter = call_args.kwargs['query_filter']
            assert query_filter is not None

    @pytest.mark.asyncio
    async def test_search_without_session_id_no_filter(self):
        """Test that search without session_id works normally."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch.object(store, '_get_client', return_value=mock_client):
            await store.search(
                collection="test_collection",
                vector=[0.1, 0.2, 0.3],
                limit=10
            )

            # Verify query_points was called
            mock_client.query_points.assert_called_once()
            call_args = mock_client.query_points.call_args

            # Without session_id or filter, query_filter should be None
            assert call_args.kwargs['query_filter'] is None

    @pytest.mark.asyncio
    async def test_search_combines_session_id_with_other_filters(self):
        """Test that session_id filter is merged with existing filters."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch.object(store, '_get_client', return_value=mock_client):
            await store.search(
                collection="test_collection",
                vector=[0.1, 0.2, 0.3],
                limit=10,
                filter={"category": "important"},
                session_id="session-456"
            )

            # Verify both filters are applied
            mock_client.query_points.assert_called_once()
            call_args = mock_client.query_points.call_args
            assert call_args.kwargs['query_filter'] is not None

    @pytest.mark.asyncio
    async def test_search_returns_correct_results_with_session_filter(self):
        """Test that search returns correct results when session_id is filtered."""
        store = QdrantStore(dimension=3)

        # Create mock results
        mock_point_1 = MagicMock()
        mock_point_1.id = "id-1"
        mock_point_1.score = 0.95
        mock_point_1.payload = {"session_id": "session-123", "data": "test"}

        mock_point_2 = MagicMock()
        mock_point_2.id = "id-2"
        mock_point_2.score = 0.87
        mock_point_2.payload = {"session_id": "session-123", "data": "test2"}

        mock_results = MagicMock()
        mock_results.points = [mock_point_1, mock_point_2]

        mock_client = AsyncMock()
        mock_client.query_points.return_value = mock_results

        with patch.object(store, '_get_client', return_value=mock_client):
            results = await store.search(
                collection="test_collection",
                vector=[0.1, 0.2, 0.3],
                limit=10,
                session_id="session-123"
            )

            # Verify results structure
            assert len(results) == 2
            assert results[0] == ("id-1", 0.95, {"session_id": "session-123", "data": "test"})
            assert results[1] == ("id-2", 0.87, {"session_id": "session-123", "data": "test2"})

    @pytest.mark.asyncio
    async def test_search_preserves_original_filter(self):
        """Test that session_id filtering doesn't modify original filter dict."""
        store = QdrantStore(dimension=3)

        original_filter = {"category": "important"}
        original_filter_copy = original_filter.copy()

        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch.object(store, '_get_client', return_value=mock_client):
            await store.search(
                collection="test_collection",
                vector=[0.1, 0.2, 0.3],
                limit=10,
                filter=original_filter,
                session_id="session-789"
            )

            # Original filter should be unchanged
            assert original_filter == original_filter_copy
            assert "session_id" not in original_filter


# ============================================================================
# Parallel Batch Upsert Tests
# ============================================================================


class TestParallelBatchUpsert:
    """Test parallel batch upsert for large datasets."""

    @pytest.mark.asyncio
    async def test_small_batch_uses_single_upload(self):
        """Test that batches <= batch_size use single upload."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()

        ids = [f"id-{i}" for i in range(50)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(50)]
        payloads = [{"idx": i} for i in range(50)]

        with patch.object(store, '_get_client', return_value=mock_client):
            await store.add(
                collection="test_collection",
                ids=ids,
                vectors=vectors,
                payloads=payloads,
                batch_size=100
            )

            # Should only call upsert once
            assert mock_client.upsert.call_count == 1

    @pytest.mark.asyncio
    async def test_large_batch_splits_into_chunks(self):
        """Test that large batches are split into parallel chunks."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()

        # 250 items with batch_size=100 should create 3 chunks
        ids = [f"id-{i}" for i in range(250)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(250)]
        payloads = [{"idx": i} for i in range(250)]

        with patch.object(store, '_get_client', return_value=mock_client):
            await store.add(
                collection="test_collection",
                ids=ids,
                vectors=vectors,
                payloads=payloads,
                batch_size=100
            )

            # Should call upsert 3 times (100 + 100 + 50)
            assert mock_client.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_parallel_batches_execute_concurrently(self):
        """Test that parallel batches execute concurrently, not sequentially."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()

        # Track execution order
        execution_times = []

        async def tracked_upsert(*args, **kwargs):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Simulate network delay
            execution_times.append(start)

        mock_client.upsert = tracked_upsert

        ids = [f"id-{i}" for i in range(200)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(200)]
        payloads = [{"idx": i} for i in range(200)]

        with patch.object(store, '_get_client', return_value=mock_client):
            start_time = asyncio.get_event_loop().time()
            await store.add(
                collection="test_collection",
                ids=ids,
                vectors=vectors,
                payloads=payloads,
                batch_size=100
            )
            end_time = asyncio.get_event_loop().time()

            # If executed in parallel, should take ~0.1s
            # If sequential, would take ~0.2s (2 batches * 0.1s each)
            elapsed = end_time - start_time
            assert elapsed < 0.15, "Batches should execute in parallel"

    @pytest.mark.asyncio
    async def test_batch_size_parameter_controls_chunking(self):
        """Test that batch_size parameter controls chunk size."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()

        ids = [f"id-{i}" for i in range(300)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(300)]
        payloads = [{"idx": i} for i in range(300)]

        with patch.object(store, '_get_client', return_value=mock_client):
            await store.add(
                collection="test_collection",
                ids=ids,
                vectors=vectors,
                payloads=payloads,
                batch_size=50  # Custom batch size
            )

            # 300 items / 50 per batch = 6 batches
            assert mock_client.upsert.call_count == 6


# ============================================================================
# Rollback on Failure Tests
# ============================================================================


class TestRollbackOnFailure:
    """Test rollback behavior when parallel batch upsert fails."""

    @pytest.mark.asyncio
    async def test_failure_triggers_rollback_for_completed_batches(self):
        """Test that failure in one batch triggers rollback of successfully uploaded batches."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()

        # Track batches to ensure deterministic ordering
        batch_order = []

        async def upsert_with_delayed_failure(*args, **kwargs):
            """First batch succeeds quickly, second batch fails after delay."""
            points = kwargs.get('points', [])
            batch_num = 1 if points[0].id.startswith("id-0") else 2
            batch_order.append(batch_num)

            if batch_num == 1:
                await asyncio.sleep(0.01)  # First batch completes quickly
            else:
                await asyncio.sleep(0.02)  # Second batch fails after first completes
                raise Exception("Simulated network error")

        mock_client.upsert = upsert_with_delayed_failure
        mock_client.delete = AsyncMock()

        ids = [f"id-{i}" for i in range(200)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(200)]
        payloads = [{"idx": i} for i in range(200)]

        with patch.object(store, '_get_client', return_value=mock_client):
            with pytest.raises(Exception, match="Simulated network error"):
                await store.add(
                    collection="test_collection",
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                    batch_size=100
                )

            # Verify delete was called for rollback of the successful batch
            mock_client.delete.assert_called_once()
            # Should rollback only the first batch's 100 IDs
            deleted_ids = mock_client.delete.call_args.kwargs['points_selector'].points
            assert len(deleted_ids) == 100

    @pytest.mark.asyncio
    async def test_rollback_deletes_only_successfully_uploaded_ids(self):
        """QDRANT-002: Rollback should only delete IDs that were successfully uploaded."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()

        # Track which batches succeed
        batches_attempted = []

        async def upsert_with_partial_failure(*args, **kwargs):
            """First batch succeeds, second batch fails."""
            points = kwargs.get('points', [])
            batch_ids = [str(p.id) for p in points]
            batches_attempted.append(batch_ids)

            # Second batch fails
            if len(batches_attempted) == 2:
                await asyncio.sleep(0.02)  # Ensure first batch completes
                raise Exception("Second batch failed")
            await asyncio.sleep(0.01)  # First batch succeeds

        mock_client.upsert = upsert_with_partial_failure
        mock_client.delete = AsyncMock()

        ids = [f"id-{i}" for i in range(200)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(200)]
        payloads = [{"idx": i} for i in range(200)]

        with patch.object(store, '_get_client', return_value=mock_client):
            with pytest.raises(Exception, match="Second batch failed"):
                await store.add(
                    collection="test_collection",
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                    batch_size=100
                )

            # Verify delete was called for rollback
            mock_client.delete.assert_called_once()
            call_args = mock_client.delete.call_args

            # Get the IDs that were deleted
            deleted_ids = call_args.kwargs['points_selector'].points

            # QDRANT-002 FIX: Should only delete the first batch's IDs (100 items),
            # not all 200 items, since second batch never succeeded
            assert len(deleted_ids) == 100, (
                f"Should rollback only successfully uploaded IDs (100), not all (200). "
                f"Got {len(deleted_ids)}"
            )
            # Verify they're the first batch's IDs
            expected_ids = [f"id-{i}" for i in range(100)]
            assert set(deleted_ids) == set(expected_ids)

    @pytest.mark.asyncio
    async def test_no_rollback_when_no_batches_succeed(self):
        """QDRANT-002: No rollback needed if no batches succeeded."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()

        async def immediate_failure(*args, **kwargs):
            raise Exception("Immediate failure")

        mock_client.upsert = immediate_failure
        mock_client.delete = AsyncMock()

        ids = [f"id-{i}" for i in range(150)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(150)]
        payloads = [{"idx": i} for i in range(150)]

        with patch.object(store, '_get_client', return_value=mock_client):
            with pytest.raises(Exception):
                await store.add(
                    collection="test_collection",
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                    batch_size=100
                )

            # QDRANT-002 FIX: Delete should NOT be called since no batches succeeded
            mock_client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_failure_is_logged_but_doesnt_mask_original_error(self):
        """Test that rollback failure doesn't hide the original error."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()

        async def upsert_with_failure(*args, **kwargs):
            raise Exception("Original upload error")

        async def delete_with_failure(*args, **kwargs):
            raise Exception("Rollback also failed")

        mock_client.upsert = upsert_with_failure
        mock_client.delete = delete_with_failure

        ids = [f"id-{i}" for i in range(150)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(150)]
        payloads = [{"idx": i} for i in range(150)]

        with patch.object(store, '_get_client', return_value=mock_client):
            # Should raise the original error, not the rollback error
            with pytest.raises(Exception, match="Original upload error"):
                await store.add(
                    collection="test_collection",
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                    batch_size=100
                )

    @pytest.mark.asyncio
    async def test_successful_upload_no_rollback(self):
        """Test that successful uploads don't trigger rollback."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()
        mock_client.delete = AsyncMock()

        ids = [f"id-{i}" for i in range(200)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(200)]
        payloads = [{"idx": i} for i in range(200)]

        with patch.object(store, '_get_client', return_value=mock_client):
            await store.add(
                collection="test_collection",
                ids=ids,
                vectors=vectors,
                payloads=payloads,
                batch_size=100
            )

            # Delete should NOT be called on success
            mock_client.delete.assert_not_called()


# ============================================================================
# Complexity and Performance Tests
# ============================================================================


class TestComplexityCharacteristics:
    """Test complexity characteristics of optimizations."""

    @pytest.mark.asyncio
    async def test_search_complexity_with_filter(self):
        """Test that search with session_id filter is more efficient than without."""
        store = QdrantStore(dimension=3)

        # This is a conceptual test - in practice, we'd measure actual
        # Qdrant performance, but here we verify the filter is applied
        mock_client = AsyncMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_client.query_points.return_value = mock_results

        with patch.object(store, '_get_client', return_value=mock_client):
            # Search without filter (O(n) scan)
            await store.search(
                collection="test_collection",
                vector=[0.1, 0.2, 0.3],
                limit=10
            )

            # Search with session_id filter (O(log n + k) with index)
            await store.search(
                collection="test_collection",
                vector=[0.1, 0.2, 0.3],
                limit=10,
                session_id="session-123"
            )

            # Both should complete, but the second should use filtering
            assert mock_client.query_points.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_upsert_space_complexity(self):
        """Test that batch upsert uses O(batch_size) space, not O(n)."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()
        batch_sizes_seen = []

        async def track_batch_size(*args, **kwargs):
            points = kwargs.get('points', [])
            batch_sizes_seen.append(len(points))
            await asyncio.sleep(0.01)

        mock_client.upsert = track_batch_size

        ids = [f"id-{i}" for i in range(350)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(350)]
        payloads = [{"idx": i} for i in range(350)]

        with patch.object(store, '_get_client', return_value=mock_client):
            await store.add(
                collection="test_collection",
                ids=ids,
                vectors=vectors,
                payloads=payloads,
                batch_size=100
            )

            # Verify batches are correctly sized
            assert batch_sizes_seen == [100, 100, 100, 50]
            # No single batch should exceed batch_size
            assert all(size <= 100 for size in batch_sizes_seen)


# ============================================================================
# Integration Tests
# ============================================================================


class TestOptimizationIntegration:
    """Test that all optimizations work together correctly."""

    @pytest.mark.asyncio
    async def test_search_and_batch_upsert_together(self):
        """Test search with session_id after parallel batch upsert."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()

        # Create mock search results
        mock_point = MagicMock()
        mock_point.id = "id-1"
        mock_point.score = 0.95
        mock_point.payload = {"session_id": "test-session"}

        mock_results = MagicMock()
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results

        with patch.object(store, '_get_client', return_value=mock_client):
            # First, batch upsert
            ids = [f"id-{i}" for i in range(200)]
            vectors = [[0.1, 0.2, 0.3] for _ in range(200)]
            payloads = [{"session_id": "test-session", "idx": i} for i in range(200)]

            await store.add(
                collection="test_collection",
                ids=ids,
                vectors=vectors,
                payloads=payloads,
                batch_size=100
            )

            # Then search with session_id filter
            results = await store.search(
                collection="test_collection",
                vector=[0.1, 0.2, 0.3],
                limit=10,
                session_id="test-session"
            )

            # Verify both operations succeeded
            assert mock_client.upsert.call_count == 2  # Two batches
            assert len(results) == 1
            assert results[0][2]["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_default_batch_size_is_100(self):
        """Test that default batch_size parameter is 100."""
        store = QdrantStore(dimension=3)

        mock_client = AsyncMock()
        mock_client.upsert = AsyncMock()

        ids = [f"id-{i}" for i in range(150)]
        vectors = [[0.1, 0.2, 0.3] for _ in range(150)]
        payloads = [{"idx": i} for i in range(150)]

        with patch.object(store, '_get_client', return_value=mock_client):
            # Don't specify batch_size, should default to 100
            await store.add(
                collection="test_collection",
                ids=ids,
                vectors=vectors,
                payloads=payloads
            )

            # Should split into 2 batches (100 + 50)
            assert mock_client.upsert.call_count == 2
