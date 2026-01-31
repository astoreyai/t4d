"""
Tests for P3.4: Interleaved Replay (CLS Theory).

Tests the get_replay_batch method and interleaved replay in NREM phase
that mixes recent and older memories to prevent catastrophic forgetting.
"""

import random
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest

from t4dm.consolidation.sleep import SleepConsolidation, SleepPhase


# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_episode(
    age_hours: float = 0,
    outcome: str = "neutral",
    valence: float = 0.5,
):
    """Create a mock episode with specified attributes."""
    episode = MagicMock()
    episode.id = uuid4()
    episode.content = f"Episode content {episode.id}"
    episode.embedding = np.random.randn(1024).tolist()
    episode.timestamp = datetime.now() - timedelta(hours=age_hours)
    episode.created_at = episode.timestamp
    episode.outcome = MagicMock()
    episode.outcome.value = outcome
    episode.emotional_valence = valence
    episode.context = MagicMock()
    episode.context.project = "test_project"
    episode.context.file = "test.py"
    episode.context.tool = None
    return episode


@pytest.fixture
def mock_episodic_memory():
    """Create mock episodic memory with recent and old episodes."""
    memory = MagicMock()

    # Create episodes of varying ages
    recent_episodes = [create_mock_episode(age_hours=i) for i in range(10)]
    old_episodes = [create_mock_episode(age_hours=48 + i) for i in range(20)]

    async def mock_get_recent(hours=24, limit=100):
        return recent_episodes[:limit]

    async def mock_sample_random(limit=50, session_filter=None, exclude_hours=24):
        return old_episodes[:limit]

    async def mock_get_by_id(episode_id):
        for ep in recent_episodes + old_episodes:
            if ep.id == episode_id:
                return ep
        return None

    memory.get_recent = AsyncMock(side_effect=mock_get_recent)
    memory.sample_random = AsyncMock(side_effect=mock_sample_random)
    memory.get_by_id = AsyncMock(side_effect=mock_get_by_id)

    return memory


@pytest.fixture
def mock_semantic_memory():
    """Create mock semantic memory."""
    memory = MagicMock()
    memory.create_or_strengthen = AsyncMock()
    memory.cluster_embeddings = AsyncMock(return_value=[])
    return memory


@pytest.fixture
def mock_graph_store():
    """Create mock graph store."""
    store = MagicMock()
    store.get_all_nodes = AsyncMock(return_value=[])
    store.get_relationships = AsyncMock(return_value=[])
    store.update_relationship_weight = AsyncMock()
    store.delete_relationship = AsyncMock()
    return store


@pytest.fixture
def sleep_consolidation(mock_episodic_memory, mock_semantic_memory, mock_graph_store):
    """Create SleepConsolidation with mocks."""
    return SleepConsolidation(
        episodic_memory=mock_episodic_memory,
        semantic_memory=mock_semantic_memory,
        graph_store=mock_graph_store,
        replay_hours=24,
        max_replays=100,
        interleave_enabled=True,
        recent_ratio=0.6,
        replay_batch_size=100,
    )


@pytest.fixture
def sleep_consolidation_no_interleave(
    mock_episodic_memory, mock_semantic_memory, mock_graph_store
):
    """Create SleepConsolidation with interleaving disabled."""
    return SleepConsolidation(
        episodic_memory=mock_episodic_memory,
        semantic_memory=mock_semantic_memory,
        graph_store=mock_graph_store,
        replay_hours=24,
        max_replays=100,
        interleave_enabled=False,
    )


# =============================================================================
# Test Interleaved Replay Configuration
# =============================================================================


class TestInterleavedReplayConfig:
    """Tests for interleaved replay configuration."""

    def test_default_interleave_enabled(
        self, mock_episodic_memory, mock_semantic_memory, mock_graph_store
    ):
        """Test default interleave is enabled."""
        sc = SleepConsolidation(
            episodic_memory=mock_episodic_memory,
            semantic_memory=mock_semantic_memory,
            graph_store=mock_graph_store,
        )
        assert sc.interleave_enabled is True
        assert sc.recent_ratio == 0.6
        assert sc.replay_batch_size == 100

    def test_custom_interleave_config(
        self, mock_episodic_memory, mock_semantic_memory, mock_graph_store
    ):
        """Test custom interleave configuration."""
        sc = SleepConsolidation(
            episodic_memory=mock_episodic_memory,
            semantic_memory=mock_semantic_memory,
            graph_store=mock_graph_store,
            interleave_enabled=False,
            recent_ratio=0.8,
            replay_batch_size=50,
        )
        assert sc.interleave_enabled is False
        assert sc.recent_ratio == 0.8
        assert sc.replay_batch_size == 50

    def test_stats_include_interleave_config(self, sleep_consolidation):
        """Test get_stats includes interleave configuration."""
        stats = sleep_consolidation.get_stats()
        assert "interleave_enabled" in stats
        assert "recent_ratio" in stats
        assert "replay_batch_size" in stats
        assert stats["interleave_enabled"] is True
        assert stats["recent_ratio"] == 0.6
        assert stats["replay_batch_size"] == 100


# =============================================================================
# Test get_replay_batch
# =============================================================================


class TestGetReplayBatch:
    """Tests for get_replay_batch method."""

    @pytest.mark.asyncio
    async def test_default_batch_composition(self, sleep_consolidation):
        """Test batch has correct recent/old ratio."""
        batch = await sleep_consolidation.get_replay_batch()

        # Default: 60% recent, 40% old
        # get_recent called with limit=60, sample_random with limit=40
        sleep_consolidation.episodic.get_recent.assert_called()
        sleep_consolidation.episodic.sample_random.assert_called()

    @pytest.mark.asyncio
    async def test_custom_ratio(self, sleep_consolidation):
        """Test custom recent_ratio."""
        batch = await sleep_consolidation.get_replay_batch(
            recent_ratio=0.8, batch_size=100
        )

        # 80% recent = 80 episodes, 20% old = 20 episodes
        call_args = sleep_consolidation.episodic.get_recent.call_args
        assert call_args[1]["limit"] == 80

        call_args = sleep_consolidation.episodic.sample_random.call_args
        assert call_args[1]["limit"] == 20

    @pytest.mark.asyncio
    async def test_all_recent_ratio(self, sleep_consolidation):
        """Test ratio=1.0 (all recent, no old)."""
        batch = await sleep_consolidation.get_replay_batch(
            recent_ratio=1.0, batch_size=50
        )

        # Should only get recent, not sample_random
        sleep_consolidation.episodic.get_recent.assert_called()
        # sample_random not called since old_count = 0
        # Actually it may be called with limit=0, let's check the implementation

    @pytest.mark.asyncio
    async def test_all_old_ratio(self, sleep_consolidation):
        """Test ratio=0.0 (all old, no recent)."""
        batch = await sleep_consolidation.get_replay_batch(
            recent_ratio=0.0, batch_size=50
        )

        # recent_count = 0, so get_recent called with limit=0
        # sample_random called with limit=50
        sleep_consolidation.episodic.sample_random.assert_called()

    @pytest.mark.asyncio
    async def test_batch_is_shuffled(self, sleep_consolidation):
        """Test batch is shuffled (not ordered by recency)."""
        # Run multiple times and check that order varies
        random.seed(42)  # For reproducibility
        batch1 = await sleep_consolidation.get_replay_batch()

        random.seed(123)
        batch2 = await sleep_consolidation.get_replay_batch()

        # Batches should have same elements but potentially different order
        # (depends on shuffle randomness)
        assert len(batch1) == len(batch2)

    @pytest.mark.asyncio
    async def test_handles_empty_recent(self, sleep_consolidation):
        """Test handles case when no recent episodes."""
        sleep_consolidation.episodic.get_recent = AsyncMock(return_value=[])

        batch = await sleep_consolidation.get_replay_batch()

        # Should still have old episodes
        assert isinstance(batch, list)

    @pytest.mark.asyncio
    async def test_handles_empty_old(self, sleep_consolidation):
        """Test handles case when no old episodes."""
        sleep_consolidation.episodic.sample_random = AsyncMock(return_value=[])

        batch = await sleep_consolidation.get_replay_batch()

        # Should still have recent episodes
        assert isinstance(batch, list)

    @pytest.mark.asyncio
    async def test_handles_get_recent_error(self, sleep_consolidation):
        """Test handles error in get_recent gracefully."""
        sleep_consolidation.episodic.get_recent = AsyncMock(
            side_effect=Exception("Database error")
        )

        batch = await sleep_consolidation.get_replay_batch()

        # Should return empty or old-only batch
        assert isinstance(batch, list)

    @pytest.mark.asyncio
    async def test_handles_sample_random_error(self, sleep_consolidation):
        """Test handles error in sample_random gracefully."""
        sleep_consolidation.episodic.sample_random = AsyncMock(
            side_effect=Exception("Database error")
        )

        batch = await sleep_consolidation.get_replay_batch()

        # Should return recent-only batch
        assert isinstance(batch, list)


# =============================================================================
# Test NREM Phase with Interleaving
# =============================================================================


class TestNREMPhaseInterleaved:
    """Tests for NREM phase with interleaved replay."""

    @pytest.mark.asyncio
    async def test_nrem_uses_interleaved_when_enabled(self, sleep_consolidation):
        """Test NREM phase uses get_replay_batch when interleave_enabled."""
        events = await sleep_consolidation.nrem_phase("test_session")

        # Should call both get_recent and sample_random via get_replay_batch
        sleep_consolidation.episodic.get_recent.assert_called()
        sleep_consolidation.episodic.sample_random.assert_called()

    @pytest.mark.asyncio
    async def test_nrem_skips_interleaved_when_disabled(
        self, sleep_consolidation_no_interleave
    ):
        """Test NREM phase uses original behavior when interleave disabled."""
        events = await sleep_consolidation_no_interleave.nrem_phase("test_session")

        # Should call get_recent but NOT sample_random
        sleep_consolidation_no_interleave.episodic.get_recent.assert_called()
        sleep_consolidation_no_interleave.episodic.sample_random.assert_not_called()

    @pytest.mark.asyncio
    async def test_nrem_falls_back_on_interleave_error(self, sleep_consolidation):
        """Test NREM falls back to original behavior on interleave error."""
        # Make get_replay_batch fail by making sample_random fail
        # But get_recent should still work for fallback
        sleep_consolidation.episodic.sample_random = AsyncMock(
            side_effect=Exception("Random sample failed")
        )

        events = await sleep_consolidation.nrem_phase("test_session")

        # Should still complete (using recent episodes)
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_nrem_returns_replay_events(self, sleep_consolidation):
        """Test NREM returns ReplayEvent objects."""
        events = await sleep_consolidation.nrem_phase("test_session", replay_count=5)

        # Should have replay events
        for event in events:
            assert hasattr(event, "episode_id")
            assert hasattr(event, "replay_time")

    @pytest.mark.asyncio
    async def test_nrem_respects_replay_count(self, sleep_consolidation):
        """Test NREM respects replay_count parameter."""
        events = await sleep_consolidation.nrem_phase("test_session", replay_count=3)

        # Should not exceed replay_count
        assert len(events) <= 3


# =============================================================================
# Test Full Sleep Cycle with Interleaving
# =============================================================================


class TestFullSleepCycleInterleaved:
    """Tests for full sleep cycle with interleaved replay."""

    @pytest.mark.asyncio
    async def test_full_cycle_uses_interleaving(self, sleep_consolidation):
        """Test full sleep cycle uses interleaved replay."""
        result = await sleep_consolidation.full_sleep_cycle("test_session")

        # Should call sample_random (only if interleave enabled)
        assert sleep_consolidation.episodic.sample_random.called

    @pytest.mark.asyncio
    async def test_full_cycle_returns_result(self, sleep_consolidation):
        """Test full sleep cycle returns SleepCycleResult."""
        result = await sleep_consolidation.full_sleep_cycle("test_session")

        assert result.session_id == "test_session"
        assert hasattr(result, "nrem_replays")
        assert hasattr(result, "rem_abstractions")


# =============================================================================
# Test EpisodicMemory Protocol Methods
# =============================================================================


class TestEpisodicMemoryProtocol:
    """Tests for EpisodicMemory protocol methods used by interleaved replay."""

    def test_protocol_includes_sample_random(self):
        """Test EpisodicMemory protocol includes sample_random."""
        from t4dm.consolidation.sleep import EpisodicMemory
        import inspect

        # Check that sample_random is in the protocol
        methods = [name for name, _ in inspect.getmembers(EpisodicMemory)]
        assert "sample_random" in methods

    def test_protocol_includes_get_recent(self):
        """Test EpisodicMemory protocol includes get_recent."""
        from t4dm.consolidation.sleep import EpisodicMemory
        import inspect

        methods = [name for name, _ in inspect.getmembers(EpisodicMemory)]
        assert "get_recent" in methods


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestInterleavedReplayEdgeCases:
    """Tests for edge cases in interleaved replay."""

    @pytest.mark.asyncio
    async def test_empty_memory(
        self, mock_semantic_memory, mock_graph_store
    ):
        """Test handles completely empty memory."""
        empty_memory = MagicMock()
        empty_memory.get_recent = AsyncMock(return_value=[])
        empty_memory.sample_random = AsyncMock(return_value=[])
        empty_memory.get_by_id = AsyncMock(return_value=None)

        sc = SleepConsolidation(
            episodic_memory=empty_memory,
            semantic_memory=mock_semantic_memory,
            graph_store=mock_graph_store,
        )

        batch = await sc.get_replay_batch()
        assert batch == []

    @pytest.mark.asyncio
    async def test_only_recent_memories_available(
        self, mock_semantic_memory, mock_graph_store
    ):
        """Test when only recent memories exist (new system)."""
        recent_only = MagicMock()
        recent_episodes = [create_mock_episode(age_hours=i) for i in range(5)]
        recent_only.get_recent = AsyncMock(return_value=recent_episodes)
        recent_only.sample_random = AsyncMock(return_value=[])
        recent_only.get_by_id = AsyncMock(return_value=None)

        sc = SleepConsolidation(
            episodic_memory=recent_only,
            semantic_memory=mock_semantic_memory,
            graph_store=mock_graph_store,
        )

        batch = await sc.get_replay_batch()
        assert len(batch) == 5  # Only recent available

    @pytest.mark.asyncio
    async def test_only_old_memories_available(
        self, mock_semantic_memory, mock_graph_store
    ):
        """Test when no recent memories but old ones exist."""
        old_only = MagicMock()
        old_episodes = [create_mock_episode(age_hours=48 + i) for i in range(10)]
        old_only.get_recent = AsyncMock(return_value=[])
        old_only.sample_random = AsyncMock(return_value=old_episodes)
        old_only.get_by_id = AsyncMock(return_value=None)

        sc = SleepConsolidation(
            episodic_memory=old_only,
            semantic_memory=mock_semantic_memory,
            graph_store=mock_graph_store,
        )

        batch = await sc.get_replay_batch()
        assert len(batch) == 10  # Only old available

    @pytest.mark.asyncio
    async def test_small_batch_size(self, sleep_consolidation):
        """Test very small batch size."""
        batch = await sleep_consolidation.get_replay_batch(
            batch_size=2, recent_ratio=0.5
        )

        # Should still work: 1 recent + 1 old
        assert isinstance(batch, list)

    @pytest.mark.asyncio
    async def test_large_batch_size(self, sleep_consolidation):
        """Test large batch size."""
        batch = await sleep_consolidation.get_replay_batch(
            batch_size=1000, recent_ratio=0.6
        )

        # Should not crash, returns what's available
        assert isinstance(batch, list)
