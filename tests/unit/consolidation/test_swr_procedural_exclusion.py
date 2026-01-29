"""
Tests for hippocampal SWR replay procedural memory exclusion.

Biological basis: Procedural memories (motor skills, habits) consolidate
via basal ganglia MSN pathways, not hippocampal sharp-wave ripples.
This test verifies that procedural memories are correctly filtered out
from hippocampal replay batches.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from ww.consolidation.sleep import SleepConsolidation


# Mock episode classes for testing
class MockEpisode:
    """Mock episode with memory_type attribute."""

    def __init__(self, memory_type="episodic", content="test"):
        self.id = uuid4()
        self.memory_type = memory_type
        self.content = content
        self.created_at = datetime.now()
        self.embedding = [0.1] * 128


@pytest.fixture
def mock_episodic():
    """Mock episodic memory service."""
    mock = AsyncMock()
    mock.get_recent = AsyncMock(return_value=[])
    mock.sample_random = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_semantic():
    """Mock semantic memory service."""
    return AsyncMock()


@pytest.fixture
def mock_graph_store():
    """Mock graph store."""
    return AsyncMock()


@pytest.fixture
def sleep_consolidation(mock_episodic, mock_semantic, mock_graph_store):
    """Create SleepConsolidation instance for testing."""
    return SleepConsolidation(
        episodic_memory=mock_episodic,
        semantic_memory=mock_semantic,
        graph_store=mock_graph_store,
        replay_hours=24,
        max_replays=100,
    )


@pytest.mark.asyncio
async def test_get_replay_batch_excludes_procedural_memories(
    sleep_consolidation, mock_episodic
):
    """Test that procedural memories are excluded from hippocampal SWR replay batch."""
    # Create mixed batch of memories
    recent_memories = [
        MockEpisode(memory_type="episodic", content="episodic 1"),
        MockEpisode(memory_type="procedural", content="procedural 1"),
        MockEpisode(memory_type="semantic", content="semantic 1"),
        MockEpisode(memory_type="episodic", content="episodic 2"),
    ]
    old_memories = [
        MockEpisode(memory_type="procedural", content="procedural 2"),
        MockEpisode(memory_type="semantic", content="semantic 2"),
        MockEpisode(memory_type="episodic", content="episodic 3"),
    ]

    # Configure mocks to return our test data
    mock_episodic.get_recent = AsyncMock(return_value=recent_memories)
    mock_episodic.sample_random = AsyncMock(return_value=old_memories)

    # Get replay batch
    batch = await sleep_consolidation.get_replay_batch(
        recent_ratio=0.6, batch_size=10
    )

    # Verify procedural memories are excluded
    memory_types = [getattr(m, "memory_type", "episodic") for m in batch]
    assert "procedural" not in memory_types, "Procedural memories should be excluded"

    # Verify episodic and semantic memories are included
    assert "episodic" in memory_types, "Episodic memories should be included"
    assert "semantic" in memory_types, "Semantic memories should be included"

    # Verify correct count (7 total - 2 procedural = 5)
    assert len(batch) == 5, f"Expected 5 memories, got {len(batch)}"


@pytest.mark.asyncio
async def test_get_replay_batch_with_only_procedural_memories(
    sleep_consolidation, mock_episodic
):
    """Test that replay batch is empty when only procedural memories are available."""
    # Create batch with only procedural memories
    recent_memories = [
        MockEpisode(memory_type="procedural", content="procedural 1"),
        MockEpisode(memory_type="procedural", content="procedural 2"),
    ]
    old_memories = [
        MockEpisode(memory_type="procedural", content="procedural 3"),
    ]

    # Configure mocks
    mock_episodic.get_recent = AsyncMock(return_value=recent_memories)
    mock_episodic.sample_random = AsyncMock(return_value=old_memories)

    # Get replay batch
    batch = await sleep_consolidation.get_replay_batch(
        recent_ratio=0.6, batch_size=10
    )

    # Verify batch is empty
    assert len(batch) == 0, "Batch should be empty when only procedural memories exist"


@pytest.mark.asyncio
async def test_get_replay_batch_with_missing_memory_type_attribute(
    sleep_consolidation, mock_episodic
):
    """Test that memories without memory_type attribute default to episodic."""

    # Create mock episode without memory_type attribute
    class LegacyEpisode:
        """Legacy episode without memory_type attribute."""

        def __init__(self, content):
            self.id = uuid4()
            self.content = content
            self.created_at = datetime.now()
            self.embedding = [0.1] * 128

    recent_memories = [
        LegacyEpisode(content="legacy 1"),
        MockEpisode(memory_type="procedural", content="procedural 1"),
        LegacyEpisode(content="legacy 2"),
    ]

    # Configure mocks
    mock_episodic.get_recent = AsyncMock(return_value=recent_memories)
    mock_episodic.sample_random = AsyncMock(return_value=[])

    # Get replay batch
    batch = await sleep_consolidation.get_replay_batch(
        recent_ratio=1.0, batch_size=10
    )

    # Verify legacy episodes are included (default to episodic)
    assert len(batch) == 2, "Legacy episodes without memory_type should be included"

    # Verify procedural is excluded
    memory_types = [getattr(m, "memory_type", "episodic") for m in batch]
    assert "procedural" not in memory_types


@pytest.mark.asyncio
async def test_get_replay_batch_preserves_episodic_and_semantic(
    sleep_consolidation, mock_episodic
):
    """Test that episodic and semantic memories pass through filter unchanged."""
    recent_memories = [
        MockEpisode(memory_type="episodic", content="episodic 1"),
        MockEpisode(memory_type="episodic", content="episodic 2"),
        MockEpisode(memory_type="semantic", content="semantic 1"),
        MockEpisode(memory_type="semantic", content="semantic 2"),
    ]

    # Configure mocks
    mock_episodic.get_recent = AsyncMock(return_value=recent_memories)
    mock_episodic.sample_random = AsyncMock(return_value=[])

    # Get replay batch
    batch = await sleep_consolidation.get_replay_batch(
        recent_ratio=1.0, batch_size=10
    )

    # Verify all memories are included
    assert len(batch) == 4, "All episodic/semantic memories should be included"

    # Verify memory types
    memory_types = [getattr(m, "memory_type", "episodic") for m in batch]
    assert memory_types.count("episodic") == 2
    assert memory_types.count("semantic") == 2


@pytest.mark.asyncio
async def test_get_procedural_replay_batch_returns_empty_list(sleep_consolidation):
    """Test that get_procedural_replay_batch returns empty list (placeholder)."""
    # Call procedural replay method
    batch = await sleep_consolidation.get_procedural_replay_batch(batch_size=10)

    # Verify returns empty list
    assert batch == [], "Procedural replay batch should return empty list (placeholder)"
    assert isinstance(batch, list), "Should return a list"


@pytest.mark.asyncio
async def test_get_procedural_replay_batch_accepts_batch_size_parameter(
    sleep_consolidation,
):
    """Test that get_procedural_replay_batch accepts batch_size parameter."""
    # Call with different batch sizes (should all return empty for now)
    batch_10 = await sleep_consolidation.get_procedural_replay_batch(batch_size=10)
    batch_50 = await sleep_consolidation.get_procedural_replay_batch(batch_size=50)
    batch_none = await sleep_consolidation.get_procedural_replay_batch()

    assert batch_10 == []
    assert batch_50 == []
    assert batch_none == []


@pytest.mark.asyncio
async def test_integration_replay_batch_biological_correctness(
    sleep_consolidation, mock_episodic
):
    """
    Integration test: verify biological correctness of memory routing.

    Hippocampal SWR replay should only process declarative memories
    (episodic and semantic), while procedural memories should route
    through basal ganglia (future implementation).
    """
    # Create realistic memory mix
    memories = [
        MockEpisode(memory_type="episodic", content="Met John at cafe"),
        MockEpisode(memory_type="episodic", content="Finished project report"),
        MockEpisode(memory_type="semantic", content="Python is a programming language"),
        MockEpisode(memory_type="semantic", content="Cafes serve coffee"),
        MockEpisode(memory_type="procedural", content="Typing skill improvement"),
        MockEpisode(memory_type="procedural", content="Walking pattern adjustment"),
    ]

    # Configure mocks
    mock_episodic.get_recent = AsyncMock(return_value=memories)
    mock_episodic.sample_random = AsyncMock(return_value=[])

    # Get hippocampal replay batch
    hippocampal_batch = await sleep_consolidation.get_replay_batch(
        recent_ratio=1.0, batch_size=10
    )

    # Get procedural replay batch (striatal pathway - not yet implemented)
    striatal_batch = await sleep_consolidation.get_procedural_replay_batch(
        batch_size=10
    )

    # Verify hippocampal batch contains only declarative memories
    hippocampal_types = [
        getattr(m, "memory_type", "episodic") for m in hippocampal_batch
    ]
    assert all(
        t in ["episodic", "semantic"] for t in hippocampal_types
    ), "Hippocampal replay should only contain declarative memories"

    # Verify correct counts
    assert len(hippocampal_batch) == 4, "Should have 4 declarative memories"
    assert (
        len(striatal_batch) == 0
    ), "Striatal pathway not yet implemented (placeholder)"

    # Future: striatal_batch should contain the 2 procedural memories
    # when striatal MSN pathway is implemented
