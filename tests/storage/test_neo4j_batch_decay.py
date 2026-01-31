"""Tests for batch Hebbian decay operations."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from t4dm.storage.neo4j_store import Neo4jStore


class TestBatchDecayRelationships:
    """Test suite for batch_decay_relationships."""

    @pytest.fixture
    def mock_store(self):
        """Create Neo4j store with mocked query method."""
        store = Neo4jStore.__new__(Neo4jStore)
        store.query = AsyncMock()
        return store

    @pytest.mark.asyncio
    async def test_batch_decay_returns_counts(self, mock_store):
        """Test that batch decay returns correct counts."""
        mock_store.query = AsyncMock(side_effect=[
            [{"decayed": 10}],  # Decay query result
            [{"pruned": 3}],    # Prune query result
        ])

        result = await mock_store.batch_decay_relationships(
            stale_days=30,
            decay_rate=0.01,
            min_weight=0.01,
        )

        assert result["decayed"] == 10
        assert result["pruned"] == 3

    @pytest.mark.asyncio
    async def test_batch_decay_uses_two_queries(self, mock_store):
        """Test that only 2 queries are executed."""
        mock_store.query = AsyncMock(side_effect=[
            [{"decayed": 0}],  # Decay query result
            [{"pruned": 0}],   # Prune query result
        ])

        await mock_store.batch_decay_relationships()

        # Should be exactly 2 calls: decay + prune
        assert mock_store.query.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_decay_passes_session_filter(self, mock_store):
        """Test session_id is passed to queries."""
        mock_store.query = AsyncMock(side_effect=[
            [{"decayed": 0}],  # Decay query result
            [{"pruned": 0}],   # Prune query result
        ])

        await mock_store.batch_decay_relationships(session_id="test-session")

        # Check both calls include session_id
        for call in mock_store.query.call_args_list:
            params = call.args[1] if len(call.args) > 1 else call.kwargs.get("parameters", {})
            assert params.get("session_id") == "test-session"

    @pytest.mark.asyncio
    async def test_batch_decay_handles_empty_result(self, mock_store):
        """Test handling of no stale relationships."""
        mock_store.query = AsyncMock(return_value=[])

        result = await mock_store.batch_decay_relationships()

        assert result["decayed"] == 0
        assert result["pruned"] == 0

    @pytest.mark.asyncio
    async def test_batch_decay_uses_correct_cutoff(self, mock_store):
        """Test that cutoff timestamp is calculated correctly."""
        mock_store.query = AsyncMock(side_effect=[
            [{"decayed": 0}],  # Decay query result
            [{"pruned": 0}],   # Prune query result
        ])

        stale_days = 45
        before_call = datetime.now() - timedelta(days=stale_days)

        await mock_store.batch_decay_relationships(stale_days=stale_days)

        # Check that cutoff is approximately correct
        params = mock_store.query.call_args_list[0].args[1]
        cutoff_str = params["cutoff"]
        cutoff_dt = datetime.fromisoformat(cutoff_str)

        # Allow 1 second tolerance for execution time
        assert abs((cutoff_dt - before_call).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_batch_decay_propagates_exceptions(self, mock_store):
        """Test that database exceptions are propagated."""
        mock_store.query = AsyncMock(side_effect=Exception("Database error"))

        with pytest.raises(Exception, match="Database error"):
            await mock_store.batch_decay_relationships()


class TestCountStaleRelationships:
    """Test suite for count_stale_relationships."""

    @pytest.fixture
    def mock_store(self):
        store = Neo4jStore.__new__(Neo4jStore)
        store.query = AsyncMock()
        return store

    @pytest.mark.asyncio
    async def test_count_returns_integer(self, mock_store):
        mock_store.query = AsyncMock(return_value=[{"count": 42}])

        result = await mock_store.count_stale_relationships(stale_days=30)

        assert result == 42

    @pytest.mark.asyncio
    async def test_count_handles_empty(self, mock_store):
        mock_store.query = AsyncMock(return_value=[])

        result = await mock_store.count_stale_relationships(stale_days=30)

        assert result == 0

    @pytest.mark.asyncio
    async def test_count_uses_correct_cutoff(self, mock_store):
        """Test that cutoff timestamp is calculated correctly."""
        mock_store.query = AsyncMock(return_value=[{"count": 0}])

        stale_days = 60
        before_call = datetime.now() - timedelta(days=stale_days)

        await mock_store.count_stale_relationships(stale_days=stale_days)

        # Check that cutoff is approximately correct
        params = mock_store.query.call_args.args[1]
        cutoff_str = params["cutoff"]
        cutoff_dt = datetime.fromisoformat(cutoff_str)

        # Allow 1 second tolerance for execution time
        assert abs((cutoff_dt - before_call).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_count_passes_session_filter(self, mock_store):
        """Test session_id is passed to query."""
        mock_store.query = AsyncMock(return_value=[{"count": 0}])

        await mock_store.count_stale_relationships(
            stale_days=30,
            session_id="test-session"
        )

        params = mock_store.query.call_args.args[1]
        assert params["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_count_uses_single_query(self, mock_store):
        """Test that only 1 query is executed."""
        mock_store.query = AsyncMock(return_value=[{"count": 5}])

        await mock_store.count_stale_relationships()

        assert mock_store.query.call_count == 1


class TestBatchDecayIntegration:
    """Integration tests for batch decay (requires actual Neo4j instance)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_decay_with_real_database(self):
        """Test batch decay against actual Neo4j instance."""
        # This test requires a running Neo4j instance
        # Skip in unit test runs
        pytest.skip("Integration test - requires Neo4j instance")

        from t4dm.storage.neo4j_store import get_neo4j_store

        store = get_neo4j_store("test-session")
        await store.initialize()

        try:
            # Create test relationships
            # ... setup code ...

            # Test batch decay
            result = await store.batch_decay_relationships(
                stale_days=1,
                decay_rate=0.1,
                min_weight=0.01,
            )

            assert "decayed" in result
            assert "pruned" in result

        finally:
            await store.close()
