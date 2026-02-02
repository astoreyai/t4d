"""
Unit tests for T4DM working memory module.

Tests WorkingMemory buffer with capacity limits and attention-based eviction.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from t4dm.memory.working_memory import (
    WorkingMemory,
    WorkingMemoryItem,
    ItemState,
    EvictionEvent,
    AttentionalBlink,
    create_working_memory,
)


class MockEpisodicMemory:
    """Mock episodic memory for testing consolidation."""

    def __init__(self):
        self.created_episodes = []

    async def create(
        self,
        content: str,
        context=None,
        outcome="neutral",
        valence=0.5
    ):
        self.created_episodes.append({
            "content": content,
            "context": context,
            "outcome": outcome,
            "valence": valence
        })
        return MagicMock(id=uuid4())


class TestWorkingMemoryItem:
    """Tests for WorkingMemoryItem dataclass."""

    def test_creation_defaults(self):
        item = WorkingMemoryItem(content="test")
        assert item.content == "test"
        assert item.priority == 0.5
        assert item.access_count == 0
        assert item.state == ItemState.ACTIVE

    def test_creation_custom(self):
        item = WorkingMemoryItem(
            content="test",
            priority=0.8,
            metadata={"key": "value"}
        )
        assert item.priority == 0.8
        assert item.metadata["key"] == "value"

    def test_touch_updates_access(self):
        item = WorkingMemoryItem(content="test")
        original_access = item.last_accessed

        # Small delay
        import time
        time.sleep(0.01)

        item.touch()
        assert item.access_count == 1
        assert item.last_accessed > original_access

    def test_age_seconds(self):
        item = WorkingMemoryItem(content="test")
        import time
        time.sleep(0.01)
        assert item.age_seconds > 0

    def test_idle_seconds(self):
        item = WorkingMemoryItem(content="test")
        import time
        time.sleep(0.01)
        assert item.idle_seconds > 0

        item.touch()
        assert item.idle_seconds < 0.01


class TestEvictionEvent:
    """Tests for EvictionEvent dataclass."""

    def test_creation(self):
        event = EvictionEvent(
            item_id=uuid4(),
            content="test content",
            reason="capacity",
            priority=0.3,
            age_seconds=10.5,
            consolidated=True
        )
        assert event.reason == "capacity"
        assert event.consolidated


class TestWorkingMemory:
    """Tests for WorkingMemory buffer."""

    @pytest.fixture
    def buffer(self):
        return WorkingMemory(capacity=4, decay_rate=0.0)

    @pytest.fixture
    def buffer_with_episodic(self):
        episodic = MockEpisodicMemory()
        return WorkingMemory(
            capacity=4,
            decay_rate=0.0,
            episodic_memory=episodic
        ), episodic

    def test_creation_default(self):
        wm = WorkingMemory()
        assert wm.capacity == 4
        assert wm.size == 0
        assert not wm.is_full

    def test_creation_custom(self):
        wm = WorkingMemory(
            capacity=7,
            decay_rate=0.2,
            min_priority=0.2,
            consolidation_threshold=0.5
        )
        assert wm.capacity == 7
        assert wm.consolidation_threshold == 0.5

    @pytest.mark.asyncio
    async def test_load_single_item(self, buffer):
        item = await buffer.load("content1", priority=0.7)

        assert buffer.size == 1
        assert item.content == "content1"
        assert item.priority == 0.7

    @pytest.mark.asyncio
    async def test_load_with_metadata(self, buffer):
        item = await buffer.load(
            "content",
            priority=0.5,
            metadata={"key": "value"}
        )

        assert item.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_load_clamps_priority(self, buffer):
        item_high = await buffer.load("test", priority=1.5)
        assert item_high.priority == 1.0

        item_low = await buffer.load("test", priority=-0.5)
        assert item_low.priority == 0.0

    @pytest.mark.asyncio
    async def test_load_respects_capacity(self, buffer):
        for i in range(5):
            await buffer.load(f"content{i}", priority=0.5)

        assert buffer.size == 4  # Capacity limit

    @pytest.mark.asyncio
    async def test_load_evicts_lowest_priority(self, buffer):
        await buffer.load("low", priority=0.1)
        await buffer.load("medium", priority=0.5)
        await buffer.load("high", priority=0.9)
        await buffer.load("medium2", priority=0.6)

        # Add one more - should evict "low"
        await buffer.load("new", priority=0.7)

        contents = [item.content for item in buffer.peek_all()]
        assert "low" not in contents
        assert "new" in contents

    @pytest.mark.asyncio
    async def test_retrieve_by_id(self, buffer):
        item = await buffer.load("test")

        retrieved = buffer.retrieve(item_id=item.id)
        assert retrieved is not None
        assert retrieved.content == "test"
        assert retrieved.access_count == 1  # Touched on retrieve

    @pytest.mark.asyncio
    async def test_retrieve_by_index(self, buffer):
        await buffer.load("first")
        await buffer.load("second")

        first = buffer.retrieve(index=0)
        second = buffer.retrieve(index=1)

        assert first.content == "first"
        assert second.content == "second"

    @pytest.mark.asyncio
    async def test_retrieve_not_found(self, buffer):
        assert buffer.retrieve(item_id=uuid4()) is None
        assert buffer.retrieve(index=100) is None

    @pytest.mark.asyncio
    async def test_update_priority(self, buffer):
        item = await buffer.load("test", priority=0.5)

        success = buffer.update_priority(item.id, 0.9)
        assert success
        assert item.priority == 0.9

    @pytest.mark.asyncio
    async def test_update_priority_not_found(self, buffer):
        success = buffer.update_priority(uuid4(), 0.5)
        assert not success

    @pytest.mark.asyncio
    async def test_peek_all(self, buffer):
        await buffer.load("a")
        await buffer.load("b")
        await buffer.load("c")

        items = buffer.peek_all()
        assert len(items) == 3
        contents = [i.content for i in items]
        assert "a" in contents
        assert "b" in contents
        assert "c" in contents

    @pytest.mark.asyncio
    async def test_get_by_priority(self, buffer):
        await buffer.load("low", priority=0.2)
        await buffer.load("high", priority=0.9)
        await buffer.load("medium", priority=0.5)

        sorted_items = buffer.get_by_priority(descending=True)

        assert sorted_items[0].content == "high"
        assert sorted_items[-1].content == "low"

    @pytest.mark.asyncio
    async def test_get_most_attended(self, buffer):
        await buffer.load("low", priority=0.2)
        await buffer.load("high", priority=0.9)

        most = buffer.get_most_attended()
        assert most.content == "high"

    @pytest.mark.asyncio
    async def test_get_least_attended(self, buffer):
        await buffer.load("low", priority=0.2)
        await buffer.load("high", priority=0.9)

        least = buffer.get_least_attended()
        assert least.content == "low"

    @pytest.mark.asyncio
    async def test_get_most_attended_empty(self, buffer):
        assert buffer.get_most_attended() is None

    @pytest.mark.asyncio
    async def test_remove(self, buffer):
        item = await buffer.load("test")
        assert buffer.size == 1

        removed = await buffer.remove(item.id)
        assert removed is not None
        assert removed.content == "test"
        assert buffer.size == 0

    @pytest.mark.asyncio
    async def test_remove_not_found(self, buffer):
        removed = await buffer.remove(uuid4())
        assert removed is None

    @pytest.mark.asyncio
    async def test_remove_with_consolidation(self, buffer_with_episodic):
        buffer, episodic = buffer_with_episodic

        item = await buffer.load("important content", priority=0.8)
        await buffer.remove(item.id, consolidate=True)

        assert len(episodic.created_episodes) == 1
        assert "important content" in episodic.created_episodes[0]["content"]

    @pytest.mark.asyncio
    async def test_clear(self, buffer):
        await buffer.load("a")
        await buffer.load("b")
        assert buffer.size == 2

        cleared = await buffer.clear()
        assert len(cleared) == 2
        assert buffer.size == 0

    @pytest.mark.asyncio
    async def test_clear_with_consolidation(self, buffer_with_episodic):
        buffer, episodic = buffer_with_episodic

        await buffer.load("a", priority=0.5)
        await buffer.load("b", priority=0.6)

        await buffer.clear(consolidate_all=True)

        assert len(episodic.created_episodes) == 2

    @pytest.mark.asyncio
    async def test_rehearse_all(self, buffer):
        item1 = await buffer.load("a")
        item2 = await buffer.load("b")

        import time
        time.sleep(0.01)

        rehearsed = await buffer.rehearse()

        assert len(rehearsed) == 2
        for item in rehearsed:
            assert item.access_count >= 1

    @pytest.mark.asyncio
    async def test_rehearse_specific(self, buffer):
        item1 = await buffer.load("a")
        item2 = await buffer.load("b")

        rehearsed = await buffer.rehearse(item_id=item1.id)

        assert len(rehearsed) == 1
        assert rehearsed[0].id == item1.id

    @pytest.mark.asyncio
    async def test_decay_reduces_attention(self):
        buffer = WorkingMemory(capacity=4, decay_rate=1.0)  # Fast decay

        await buffer.load("test", priority=0.5)

        # Wait for decay
        import time
        time.sleep(0.1)

        # Apply decay
        buffer._apply_decay()

        # Attention should have decayed
        assert buffer._attention_weights[0] < 0.5

    @pytest.mark.asyncio
    async def test_consolidate_decaying(self):
        buffer = WorkingMemory(
            capacity=4,
            decay_rate=10.0,  # Very fast decay
            min_priority=0.4,
            consolidation_threshold=0.1,
            episodic_memory=MockEpisodicMemory()
        )

        await buffer.load("test", priority=0.5)

        # Wait for significant decay
        import time
        time.sleep(0.1)
        buffer._apply_decay()  # Force decay

        # Item should be in decaying state
        events = await buffer.consolidate_decaying()

        # Decaying items should be consolidated/evicted
        assert buffer.size == 0

    def test_size_property(self, buffer):
        assert buffer.size == 0

    @pytest.mark.asyncio
    async def test_is_full_property(self, buffer):
        for i in range(4):
            await buffer.load(f"item{i}")

        assert buffer.is_full

    @pytest.mark.asyncio
    async def test_available_capacity_property(self, buffer):
        assert buffer.available_capacity == 4

        await buffer.load("a")
        assert buffer.available_capacity == 3

    @pytest.mark.asyncio
    async def test_get_stats(self, buffer):
        await buffer.load("a", priority=0.6)
        await buffer.load("b", priority=0.8)

        stats = buffer.get_stats()

        assert stats["capacity"] == 4
        assert stats["current_size"] == 2
        assert stats["available"] == 2
        assert stats["total_loads"] == 2

    @pytest.mark.asyncio
    async def test_eviction_history(self, buffer):
        # Fill buffer
        for i in range(4):
            await buffer.load(f"item{i}", priority=0.5)

        # Force eviction
        await buffer.load("new", priority=0.6)

        history = buffer.get_eviction_history()
        assert len(history) >= 1
        assert history[-1].reason == "capacity"


class TestAttentionalBlink:
    """Tests for AttentionalBlink."""

    def test_creation(self):
        blink = AttentionalBlink(
            blink_duration_ms=500,
            capacity_reduction=0.5
        )
        assert blink.capacity_reduction == 0.5

    def test_trigger(self):
        blink = AttentionalBlink(blink_duration_ms=100)

        blink.trigger()
        assert blink.is_blinking()

    def test_blink_expires(self):
        blink = AttentionalBlink(blink_duration_ms=10)

        blink.trigger()
        assert blink.is_blinking()

        import time
        time.sleep(0.02)  # Wait for blink to expire

        assert not blink.is_blinking()

    def test_get_effective_capacity_normal(self):
        blink = AttentionalBlink()

        capacity = blink.get_effective_capacity(4)
        assert capacity == 4

    def test_get_effective_capacity_during_blink(self):
        blink = AttentionalBlink(
            blink_duration_ms=1000,
            capacity_reduction=0.5
        )

        blink.trigger()

        capacity = blink.get_effective_capacity(4)
        assert capacity == 2  # 4 * 0.5


class TestCreateWorkingMemory:
    """Tests for factory function."""

    def test_create_basic(self):
        wm = create_working_memory()
        assert wm.capacity == 4

    def test_create_custom(self):
        episodic = MockEpisodicMemory()
        wm = create_working_memory(
            capacity=7,
            episodic_memory=episodic,
            decay_rate=0.2
        )
        assert wm.capacity == 7
        assert wm.episodic is episodic


class TestIntegration:
    """Integration tests for working memory."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        episodic = MockEpisodicMemory()
        buffer = WorkingMemory(
            capacity=3,
            decay_rate=0.0,
            consolidation_threshold=0.4,
            episodic_memory=episodic
        )

        # Load items
        item1 = await buffer.load("task1", priority=0.8)
        item2 = await buffer.load("task2", priority=0.3)
        item3 = await buffer.load("task3", priority=0.6)

        assert buffer.is_full

        # Access high-priority item
        buffer.retrieve(item_id=item1.id)
        assert item1.access_count == 1

        # Add new item - should evict lowest priority (task2)
        item4 = await buffer.load("task4", priority=0.7)

        contents = [i.content for i in buffer.peek_all()]
        assert "task2" not in contents
        assert "task4" in contents

        # Check consolidation of evicted item
        # task2 had priority 0.3 < 0.4 threshold, so not consolidated
        assert len(episodic.created_episodes) == 0

        # Evict higher priority item
        await buffer.load("task5", priority=0.9)
        await buffer.load("task6", priority=0.95)

        # Now should have consolidated some
        history = buffer.get_eviction_history()
        consolidated_count = sum(1 for e in history if e.consolidated)
        assert consolidated_count >= 1

    @pytest.mark.asyncio
    async def test_priority_ordering_maintained(self):
        buffer = WorkingMemory(capacity=4, decay_rate=0.0)

        await buffer.load("low", priority=0.1)
        await buffer.load("high", priority=0.9)
        await buffer.load("medium", priority=0.5)

        sorted_items = buffer.get_by_priority(descending=True)

        priorities = [i.priority for i in sorted_items]
        assert priorities == sorted(priorities, reverse=True)
