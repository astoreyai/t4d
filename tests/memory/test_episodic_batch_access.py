"""Tests for batch access update functionality."""
import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from t4dm.memory.episodic import EpisodicMemory
from t4dm.core.types import Episode


class TestBatchAccessUpdate:
    """Test _batch_update_access method."""

    @pytest.fixture
    def mock_episodic(self):
        """Create episodic memory with mocked stores."""
        episodic = EpisodicMemory.__new__(EpisodicMemory)
        episodic.vector_store = MagicMock()
        episodic.vector_store.episodes_collection = "ww_episodes"
        episodic.vector_store.get = AsyncMock(return_value=[])
        episodic.vector_store.batch_update_payloads = AsyncMock(return_value=0)
        episodic.graph_store = MagicMock()
        episodic.embedding = MagicMock()
        return episodic

    @pytest.fixture
    def sample_payload(self):
        """Create sample episode payload."""
        return {
            "content": "Test episode",
            "timestamp": datetime.now().isoformat(),
            "ingested_at": datetime.now().isoformat(),
            "stability": 1.0,
            "last_accessed": (datetime.now() - timedelta(days=1)).isoformat(),
            "access_count": 5,
            "session_id": "test",
            "context": {},
            "outcome": "neutral",
            "emotional_valence": 0.5,
        }

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero(self, mock_episodic):
        """Empty episode list returns 0."""
        result = await mock_episodic._batch_update_access([])
        assert result == 0
        mock_episodic.vector_store.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetches_episodes_by_id(self, mock_episodic, sample_payload):
        """Fetches episodes using provided IDs."""
        ids = [uuid4(), uuid4()]
        mock_episodic.vector_store.get = AsyncMock(return_value=[
            (str(ids[0]), sample_payload),
            (str(ids[1]), sample_payload),
        ])

        await mock_episodic._batch_update_access(ids)

        mock_episodic.vector_store.get.assert_called_once()
        call_args = mock_episodic.vector_store.get.call_args
        assert str(ids[0]) in call_args.kwargs["ids"]
        assert str(ids[1]) in call_args.kwargs["ids"]

    @pytest.mark.asyncio
    async def test_updates_stability_on_success(self, mock_episodic, sample_payload):
        """Stability increases on successful recall."""
        episode_id = uuid4()
        mock_episodic.vector_store.get = AsyncMock(return_value=[
            (str(episode_id), sample_payload),
        ])
        mock_episodic.vector_store.batch_update_payloads = AsyncMock(return_value=1)

        await mock_episodic._batch_update_access([episode_id], success=True)

        call_args = mock_episodic.vector_store.batch_update_payloads.call_args
        updates = call_args.kwargs["updates"]

        # Stability should increase
        new_stability = updates[0][1]["stability"]
        assert new_stability > sample_payload["stability"]

    @pytest.mark.asyncio
    async def test_decreases_stability_on_failure(self, mock_episodic, sample_payload):
        """Stability decreases on failed recall."""
        episode_id = uuid4()
        mock_episodic.vector_store.get = AsyncMock(return_value=[
            (str(episode_id), sample_payload),
        ])
        mock_episodic.vector_store.batch_update_payloads = AsyncMock(return_value=1)

        await mock_episodic._batch_update_access([episode_id], success=False)

        call_args = mock_episodic.vector_store.batch_update_payloads.call_args
        updates = call_args.kwargs["updates"]

        # Stability should decrease (0.8x)
        new_stability = updates[0][1]["stability"]
        assert new_stability == sample_payload["stability"] * 0.8

    @pytest.mark.asyncio
    async def test_increments_access_count(self, mock_episodic, sample_payload):
        """Access count is incremented."""
        episode_id = uuid4()
        mock_episodic.vector_store.get = AsyncMock(return_value=[
            (str(episode_id), sample_payload),
        ])
        mock_episodic.vector_store.batch_update_payloads = AsyncMock(return_value=1)

        await mock_episodic._batch_update_access([episode_id])

        call_args = mock_episodic.vector_store.batch_update_payloads.call_args
        updates = call_args.kwargs["updates"]

        assert updates[0][1]["access_count"] == sample_payload["access_count"] + 1

    @pytest.mark.asyncio
    async def test_updates_last_accessed(self, mock_episodic, sample_payload):
        """Last accessed timestamp is updated."""
        episode_id = uuid4()
        mock_episodic.vector_store.get = AsyncMock(return_value=[
            (str(episode_id), sample_payload),
        ])
        mock_episodic.vector_store.batch_update_payloads = AsyncMock(return_value=1)

        before = datetime.now()
        await mock_episodic._batch_update_access([episode_id])
        after = datetime.now()

        call_args = mock_episodic.vector_store.batch_update_payloads.call_args
        updates = call_args.kwargs["updates"]

        last_accessed = datetime.fromisoformat(updates[0][1]["last_accessed"])
        assert before <= last_accessed <= after

    @pytest.mark.asyncio
    async def test_handles_missing_episodes(self, mock_episodic):
        """Handles episodes not found gracefully."""
        mock_episodic.vector_store.get = AsyncMock(return_value=[])

        result = await mock_episodic._batch_update_access([uuid4()])

        assert result == 0
        mock_episodic.vector_store.batch_update_payloads.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_update_error(self, mock_episodic, sample_payload):
        """Handles update errors gracefully."""
        episode_id = uuid4()
        mock_episodic.vector_store.get = AsyncMock(return_value=[
            (str(episode_id), sample_payload),
        ])
        mock_episodic.vector_store.batch_update_payloads = AsyncMock(
            side_effect=Exception("Update failed")
        )

        # Should not raise
        result = await mock_episodic._batch_update_access([episode_id])
        assert result == 0

    @pytest.mark.asyncio
    async def test_batch_with_multiple_episodes(self, mock_episodic, sample_payload):
        """Batch processes multiple episodes correctly."""
        ids = [uuid4() for _ in range(5)]
        payloads = [
            {**sample_payload, "stability": 1.0 + i * 0.1}
            for i in range(5)
        ]

        mock_episodic.vector_store.get = AsyncMock(return_value=[
            (str(episode_id), payload)
            for episode_id, payload in zip(ids, payloads)
        ])
        mock_episodic.vector_store.batch_update_payloads = AsyncMock(return_value=5)

        result = await mock_episodic._batch_update_access(ids, success=True)

        assert result == 5
        call_args = mock_episodic.vector_store.batch_update_payloads.call_args
        updates = call_args.kwargs["updates"]

        # Should have 5 updates
        assert len(updates) == 5

        # All should have increased stability
        for i, (id_str, update_payload) in enumerate(updates):
            assert update_payload["stability"] > payloads[i]["stability"]
            assert update_payload["access_count"] == payloads[i]["access_count"] + 1
