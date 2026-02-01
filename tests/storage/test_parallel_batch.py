"""Tests for parallel batch operations."""
import pytest
import asyncio
import time
import os
from unittest.mock import AsyncMock, patch, MagicMock

# Set test mode to bypass password validation
os.environ["T4DM_TEST_MODE"] = "true"

pytestmark = pytest.mark.skip(reason="Neo4j removed — replaced by T4DX")


class TestParallelBatchUpdate:
    """Tests for parallel batch update operations."""

    @pytest.mark.asyncio
    async def test_parallel_faster_than_sequential(self):
        """Test that parallel updates are faster than sequential."""
        from t4dm.storage import T4DXVectorStore

        delay_per_update = 0.05  # 50ms per update
        num_updates = 20

        async def delayed_set_payload(*args, **kwargs):
            await asyncio.sleep(delay_per_update)

        mock_client = AsyncMock()
        mock_client.set_payload = delayed_set_payload

        with patch.object(T4DXVectorStore, "_get_client", return_value=mock_client):
            store = T4DXVectorStore()
            store.timeout = 10  # 10 second timeout

            updates = [
                (f"id_{i}", {"field": f"value_{i}"})
                for i in range(num_updates)
            ]

            start = time.time()
            result = await store.batch_update_payloads(
                collection="test",
                updates=updates,
                max_concurrency=10,
            )
            elapsed = time.time() - start

            assert result == num_updates
            # Sequential would take 20 * 0.05 = 1.0s
            # Parallel with 10 concurrency should take ~0.1s
            assert elapsed < 0.5  # Allow some overhead

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self):
        """Test that max_concurrency is respected."""
        from t4dm.storage import T4DXVectorStore

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_set_payload(*args, **kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.02)
            async with lock:
                current_concurrent -= 1

        mock_client = AsyncMock()
        mock_client.set_payload = tracking_set_payload

        with patch.object(T4DXVectorStore, "_get_client", return_value=mock_client):
            store = T4DXVectorStore()
            store.timeout = 10

            updates = [(f"id_{i}", {}) for i in range(50)]

            await store.batch_update_payloads(
                collection="test",
                updates=updates,
                max_concurrency=5,
            )

            assert max_concurrent <= 5

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self):
        """Test that partial failures don't stop other updates."""
        from t4dm.storage import T4DXVectorStore

        call_count = 0

        async def sometimes_failing_update(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every 3rd call fails
                raise RuntimeError("Simulated failure")

        mock_client = AsyncMock()
        mock_client.set_payload = sometimes_failing_update

        with patch.object(T4DXVectorStore, "_get_client", return_value=mock_client):
            store = T4DXVectorStore()
            store.timeout = 10

            updates = [(f"id_{i}", {}) for i in range(9)]

            result = await store.batch_update_payloads(
                collection="test",
                updates=updates,
                max_concurrency=10,
            )

            # 3 failures out of 9 = 6 successes
            assert result == 6
            assert call_count == 9  # All were attempted

    @pytest.mark.asyncio
    async def test_empty_updates_returns_zero(self):
        """Test that empty updates list returns 0."""
        from t4dm.storage import T4DXVectorStore

        store = T4DXVectorStore()
        result = await store.batch_update_payloads(
            collection="test",
            updates=[],
        )

        assert result == 0


class TestParallelBatchDelete:
    """Tests for parallel batch delete operations."""

    @pytest.mark.asyncio
    async def test_parallel_delete(self):
        """Test parallel delete operation."""
        from t4dm.storage import T4DXVectorStore

        deleted_ids = []

        async def tracking_delete(*args, **kwargs):
            points = kwargs.get("points_selector", {})
            if hasattr(points, "points"):
                deleted_ids.extend(points.points)

        mock_client = AsyncMock()
        mock_client.delete = tracking_delete

        with patch.object(T4DXVectorStore, "_get_client", return_value=mock_client):
            store = T4DXVectorStore()
            store.timeout = 10

            ids = [f"id_{i}" for i in range(10)]

            result = await store.batch_delete(
                collection="test",
                ids=ids,
                max_concurrency=5,
            )

            assert result == 10
            assert set(deleted_ids) == set(ids)

    @pytest.mark.asyncio
    async def test_delete_empty_list(self):
        """Test deleting empty list returns 0."""
        from t4dm.storage import T4DXVectorStore

        store = T4DXVectorStore()
        result = await store.batch_delete(
            collection="test",
            ids=[],
        )

        assert result == 0


class TestConfigParameters:
    """Tests for batch configuration parameters."""

    def test_batch_concurrency_config(self):
        """Test batch_max_concurrency is configurable."""
        from t4dm.core.config import Settings

        settings = Settings(batch_max_concurrency=20)
        assert settings.batch_max_concurrency == 20

    def test_batch_concurrency_validation(self):
        """Test batch_max_concurrency validation."""
        from t4dm.core.config import Settings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(batch_max_concurrency=0)

        with pytest.raises(ValidationError):
            Settings(batch_max_concurrency=101)
