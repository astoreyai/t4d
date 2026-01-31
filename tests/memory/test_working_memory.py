"""
Tests for working memory buffer.

Tests WorkingMemoryItem, WorkingMemory, EvictionEvent, and ItemState classes.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock

from t4dm.memory.working_memory import (
    ItemState,
    WorkingMemoryItem,
    EvictionEvent,
    WorkingMemory,
)


# =============================================================================
# Test ItemState Enum
# =============================================================================


class TestItemState:
    """Tests for ItemState enum."""

    def test_active_state(self):
        """Test ACTIVE state."""
        assert ItemState.ACTIVE.value == "active"

    def test_decaying_state(self):
        """Test DECAYING state."""
        assert ItemState.DECAYING.value == "decaying"

    def test_evicted_state(self):
        """Test EVICTED state."""
        assert ItemState.EVICTED.value == "evicted"

    def test_consolidated_state(self):
        """Test CONSOLIDATED state."""
        assert ItemState.CONSOLIDATED.value == "consolidated"


# =============================================================================
# Test WorkingMemoryItem
# =============================================================================


class TestWorkingMemoryItem:
    """Tests for WorkingMemoryItem dataclass."""

    def test_default_creation(self):
        """Test default item creation."""
        item = WorkingMemoryItem(content="test data")
        assert item.content == "test data"
        assert item.priority == 0.5
        assert item.access_count == 0
        assert item.state == ItemState.ACTIVE

    def test_custom_priority(self):
        """Test item with custom priority."""
        item = WorkingMemoryItem(content="high priority", priority=0.9)
        assert item.priority == 0.9

    def test_touch_updates_access(self):
        """Test touch() updates access count and timestamp."""
        item = WorkingMemoryItem(content="test")
        old_accessed = item.last_accessed
        assert item.access_count == 0

        item.touch()
        assert item.access_count == 1
        assert item.last_accessed >= old_accessed

        item.touch()
        assert item.access_count == 2

    def test_age_seconds(self):
        """Test age_seconds property."""
        item = WorkingMemoryItem(content="test")
        # Just created, should be very young
        assert item.age_seconds >= 0
        assert item.age_seconds < 1.0

    def test_idle_seconds(self):
        """Test idle_seconds property."""
        item = WorkingMemoryItem(content="test")
        # Just created, idle time should be minimal
        assert item.idle_seconds >= 0
        assert item.idle_seconds < 1.0

    def test_metadata(self):
        """Test metadata field."""
        item = WorkingMemoryItem(
            content="test",
            metadata={"key": "value", "num": 42},
        )
        assert item.metadata["key"] == "value"
        assert item.metadata["num"] == 42

    def test_state_transitions(self):
        """Test state field can be modified."""
        item = WorkingMemoryItem(content="test")
        assert item.state == ItemState.ACTIVE

        item.state = ItemState.DECAYING
        assert item.state == ItemState.DECAYING

        item.state = ItemState.EVICTED
        assert item.state == ItemState.EVICTED


# =============================================================================
# Test EvictionEvent
# =============================================================================


class TestEvictionEvent:
    """Tests for EvictionEvent dataclass."""

    def test_creation(self):
        """Test eviction event creation."""
        event = EvictionEvent(
            item_id=uuid4(),
            content="evicted content",
            reason="capacity_exceeded",
            priority=0.2,
            age_seconds=120.5,
            consolidated=True,
        )
        assert event.content == "evicted content"
        assert event.reason == "capacity_exceeded"
        assert event.priority == 0.2
        assert event.consolidated is True

    def test_not_consolidated(self):
        """Test eviction without consolidation."""
        event = EvictionEvent(
            item_id=uuid4(),
            content="lost content",
            reason="low_priority",
            priority=0.05,
            age_seconds=60.0,
            consolidated=False,
        )
        assert event.consolidated is False


# =============================================================================
# Test WorkingMemory
# =============================================================================


class TestWorkingMemory:
    """Tests for WorkingMemory class."""

    @pytest.fixture
    def wm(self):
        """Create working memory instance."""
        return WorkingMemory(capacity=4)

    def test_initialization(self, wm):
        """Test default initialization."""
        assert wm.capacity == 4
        assert wm.decay_rate == 0.1
        assert wm.min_priority == 0.1

    def test_custom_capacity(self):
        """Test custom capacity."""
        wm = WorkingMemory(capacity=7)
        assert wm.capacity == 7

    @pytest.mark.asyncio
    async def test_load_item(self, wm):
        """Test loading item to working memory."""
        item = await wm.load("test content", priority=0.8)
        assert item is not None
        assert item.content == "test content"
        assert item.priority == 0.8

        items = wm.peek_all()
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_load_multiple_items(self, wm):
        """Test loading multiple items."""
        await wm.load("item1", priority=0.5)
        await wm.load("item2", priority=0.6)
        await wm.load("item3", priority=0.7)

        items = wm.peek_all()
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_retrieve_by_id(self, wm):
        """Test retrieving item by ID."""
        item = await wm.load("findme", priority=0.5)
        retrieved = wm.retrieve(item_id=item.id)

        assert retrieved is not None
        assert retrieved.content == "findme"

    def test_retrieve_nonexistent(self, wm):
        """Test retrieving nonexistent item."""
        item = wm.retrieve(item_id=uuid4())
        assert item is None

    @pytest.mark.asyncio
    async def test_retrieve_by_index(self, wm):
        """Test retrieving item by index."""
        await wm.load("first", priority=0.5)
        await wm.load("second", priority=0.6)

        item = wm.retrieve(index=0)
        assert item is not None
        assert item.content == "first"

        item = wm.retrieve(index=1)
        assert item.content == "second"

    def test_retrieve_invalid_index(self, wm):
        """Test retrieving with invalid index."""
        item = wm.retrieve(index=100)
        assert item is None

        item = wm.retrieve(index=-1)
        assert item is None

    @pytest.mark.asyncio
    async def test_remove_item(self, wm):
        """Test removing item."""
        item = await wm.load("removeme", priority=0.5)
        removed = await wm.remove(item.id)

        assert removed is not None
        assert removed.content == "removeme"
        assert removed.state == ItemState.EVICTED

        retrieved = wm.retrieve(item_id=item.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, wm):
        """Test removing nonexistent item."""
        removed = await wm.remove(uuid4())
        assert removed is None

    @pytest.mark.asyncio
    async def test_clear(self, wm):
        """Test clearing working memory."""
        await wm.load("item1")
        await wm.load("item2")

        items = wm.peek_all()
        assert len(items) == 2

        cleared = await wm.clear()
        assert len(cleared) == 2

        items = wm.peek_all()
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_capacity_limit(self, wm):
        """Test capacity limit triggers eviction."""
        # Add items up to and beyond capacity
        for i in range(5):
            await wm.load(f"item{i}", priority=0.5)

        items = wm.peek_all()
        # Should not exceed capacity
        assert len(items) <= wm.capacity

    @pytest.mark.asyncio
    async def test_priority_based_eviction(self):
        """Test low priority items evicted first."""
        wm = WorkingMemory(capacity=2)

        await wm.load("low", priority=0.2)
        await wm.load("high", priority=0.9)
        await wm.load("medium", priority=0.5)  # Triggers eviction

        items = wm.peek_all()
        contents = [item.content for item in items]

        # High priority should be kept
        assert "high" in contents

    @pytest.mark.asyncio
    async def test_update_priority(self, wm):
        """Test updating item priority."""
        item = await wm.load("updateme", priority=0.3)

        result = wm.update_priority(item.id, 0.9)
        assert result is True

        retrieved = wm.retrieve(item_id=item.id)
        assert retrieved.priority == 0.9

    @pytest.mark.asyncio
    async def test_update_priority_nonexistent(self, wm):
        """Test updating priority of nonexistent item."""
        result = wm.update_priority(uuid4(), 0.9)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_full_property(self, wm):
        """Test is_full property."""
        assert wm.is_full is False

        for i in range(4):
            await wm.load(f"item{i}")

        assert wm.is_full is True

    @pytest.mark.asyncio
    async def test_get_most_attended(self, wm):
        """Test getting highest priority item."""
        await wm.load("low", priority=0.2)
        await wm.load("high", priority=0.9)
        await wm.load("medium", priority=0.5)

        item = wm.get_most_attended()
        assert item.content == "high"
        assert item.priority == 0.9

    @pytest.mark.asyncio
    async def test_get_least_attended(self, wm):
        """Test getting lowest priority item."""
        await wm.load("low", priority=0.2)
        await wm.load("high", priority=0.9)
        await wm.load("medium", priority=0.5)

        item = wm.get_least_attended()
        assert item.content == "low"
        assert item.priority == 0.2

    def test_get_most_attended_empty(self, wm):
        """Test get_most_attended on empty buffer."""
        item = wm.get_most_attended()
        assert item is None

    def test_get_least_attended_empty(self, wm):
        """Test get_least_attended on empty buffer."""
        item = wm.get_least_attended()
        assert item is None

    @pytest.mark.asyncio
    async def test_get_by_priority_descending(self, wm):
        """Test getting items sorted by priority (descending)."""
        await wm.load("low", priority=0.2)
        await wm.load("high", priority=0.9)
        await wm.load("medium", priority=0.5)

        sorted_items = wm.get_by_priority(descending=True)
        assert sorted_items[0].content == "high"
        assert sorted_items[1].content == "medium"
        assert sorted_items[2].content == "low"

    @pytest.mark.asyncio
    async def test_get_by_priority_ascending(self, wm):
        """Test getting items sorted by priority (ascending)."""
        await wm.load("low", priority=0.2)
        await wm.load("high", priority=0.9)
        await wm.load("medium", priority=0.5)

        sorted_items = wm.get_by_priority(descending=False)
        assert sorted_items[0].content == "low"
        assert sorted_items[1].content == "medium"
        assert sorted_items[2].content == "high"

    @pytest.mark.asyncio
    async def test_load_with_metadata(self, wm):
        """Test loading item with metadata."""
        item = await wm.load(
            "with meta",
            priority=0.5,
            metadata={"source": "test", "id": 42},
        )
        assert item.metadata["source"] == "test"
        assert item.metadata["id"] == 42


# =============================================================================
# Test WorkingMemory with Consolidation
# =============================================================================


class TestWorkingMemoryConsolidation:
    """Tests for working memory consolidation to episodic."""

    @pytest.fixture
    def mock_episodic(self):
        """Create mock episodic memory."""
        episodic = MagicMock()
        episodic.create = AsyncMock(return_value=MagicMock(id=uuid4()))
        return episodic

    @pytest.mark.asyncio
    async def test_remove_with_consolidate(self, mock_episodic):
        """Test removing with consolidation flag."""
        wm = WorkingMemory(
            capacity=4,
            consolidation_threshold=0.3,
            episodic_memory=mock_episodic,
        )

        item = await wm.load("consolidate me", priority=0.5)
        removed = await wm.remove(item.id, consolidate=True)

        assert removed is not None
        # Consolidation should have been attempted

    @pytest.mark.asyncio
    async def test_clear_with_consolidate(self, mock_episodic):
        """Test clearing with consolidation."""
        wm = WorkingMemory(
            capacity=4,
            consolidation_threshold=0.3,
            episodic_memory=mock_episodic,
        )

        await wm.load("item1", priority=0.5)
        await wm.load("item2", priority=0.6)

        cleared = await wm.clear(consolidate_all=True)
        assert len(cleared) == 2
