"""
Working Memory Buffer for World Weaver.

Addresses Hinton critique: All memories persist immediately - no transient buffer.
Implements a capacity-limited working memory with attention-based maintenance.

Biological Basis:
- Prefrontal cortex maintains ~4 items through sustained activity
- Attention refreshes items to prevent decay
- Unrehearsed items decay and are lost or consolidated
- Buffer serves as a staging area before long-term storage

Key Properties:
1. Limited capacity (default 4 items, based on Cowan's magical number)
2. Priority-based attention allocation
3. Decay over time for unattended items
4. Automatic eviction when capacity exceeded
5. Consolidation pathway to episodic memory

Implementation:
- Working memory items have content, priority, and timestamp
- Attention weights decay exponentially over time
- Low-priority items are evicted first
- Evicted items can optionally be consolidated to episodic memory
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar
from uuid import UUID, uuid4

import numpy as np

logger = logging.getLogger(__name__)

# RACE-007 FIX: Per-instance async locks for capacity management
_wm_locks: dict[int, asyncio.Lock] = {}


T = TypeVar("T")


class ItemState(Enum):
    """State of a working memory item."""
    ACTIVE = "active"
    DECAYING = "decaying"
    EVICTED = "evicted"
    CONSOLIDATED = "consolidated"


@dataclass
class WorkingMemoryItem(Generic[T]):
    """Item in working memory buffer."""

    id: UUID = field(default_factory=uuid4)
    content: T = None
    priority: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    state: ItemState = ItemState.ACTIVE
    metadata: dict = field(default_factory=dict)

    def touch(self) -> None:
        """Mark as accessed, refreshing attention."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        """Time since creation in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Time since last access in seconds."""
        return (datetime.now() - self.last_accessed).total_seconds()


@dataclass
class EvictionEvent:
    """Record of an eviction from working memory."""

    item_id: UUID
    content: Any
    reason: str
    priority: float
    age_seconds: float
    consolidated: bool
    eviction_time: datetime = field(default_factory=datetime.now)


class EpisodicMemory(Protocol):
    """Protocol for episodic memory consolidation."""

    async def create(
        self,
        content: str,
        context: dict | None = None,
        outcome: str = "neutral",
        valence: float = 0.5
    ) -> Any:
        """Store episode."""
        ...


class WorkingMemory(Generic[T]):
    """
    Capacity-limited working memory buffer.

    Maintains a small set of active items with attention-based prioritization.
    Items decay over time and are evicted when capacity is exceeded.

    Features:
    - Fixed capacity (default 4 items)
    - Priority-based attention weights
    - Exponential decay of attention
    - Automatic eviction of low-priority items
    - Optional consolidation of evicted items to episodic memory
    """

    def __init__(
        self,
        capacity: int = 4,
        decay_rate: float = 0.1,
        min_priority: float = 0.1,
        consolidation_threshold: float = 0.3,
        episodic_memory: EpisodicMemory | None = None
    ):
        """
        Initialize working memory buffer.

        Args:
            capacity: Maximum items to hold (default 4, based on Cowan's limit)
            decay_rate: Exponential decay rate per second
            min_priority: Minimum priority before eviction
            consolidation_threshold: Priority threshold for consolidation
            episodic_memory: Optional episodic memory for consolidation
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.min_priority = min_priority
        self.consolidation_threshold = consolidation_threshold
        self.episodic = episodic_memory

        # Buffer and attention weights
        self._buffer: list[WorkingMemoryItem[T]] = []
        self._attention_weights: list[float] = []

        # History tracking - MEM-009 FIX: Bound history size
        self._eviction_history: list[EvictionEvent] = []
        self._max_history_size = 10000
        self._total_loads: int = 0
        self._total_evictions: int = 0

        # RACE-007 FIX: Async lock for capacity management
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Get or create async lock for this instance."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def load(
        self,
        content: T,
        priority: float = 0.5,
        metadata: dict | None = None
    ) -> WorkingMemoryItem[T]:
        """
        Load item into working memory.

        If buffer is at capacity, evicts lowest priority item first.

        RACE-007 FIX: Uses async lock to prevent concurrent loads from
        exceeding capacity.

        Args:
            content: Content to store
            priority: Attention priority [0, 1]
            metadata: Optional metadata

        Returns:
            Created WorkingMemoryItem
        """
        # RACE-007 FIX: Lock during capacity check and modification
        async with self._get_lock():
            self._total_loads += 1

            # Apply decay before checking capacity
            self._apply_decay()

            # Evict if at capacity
            if len(self._buffer) >= self.capacity:
                await self._evict_lowest()

            # Create new item
            item = WorkingMemoryItem[T](
                content=content,
                priority=np.clip(priority, 0.0, 1.0),
                metadata=metadata or {}
            )

            self._buffer.append(item)
            self._attention_weights.append(item.priority)

            logger.debug(
                f"Loaded item {item.id} with priority {priority:.2f}, "
                f"buffer size: {len(self._buffer)}"
            )

            return item

    def retrieve(
        self,
        item_id: UUID | None = None,
        index: int | None = None
    ) -> WorkingMemoryItem[T] | None:
        """
        Retrieve item from working memory.

        Touching the item refreshes its attention.

        Args:
            item_id: Item ID to retrieve
            index: Index in buffer (alternative to item_id)

        Returns:
            Item if found, None otherwise
        """
        if index is not None:
            if 0 <= index < len(self._buffer):
                item = self._buffer[index]
                item.touch()
                self._attention_weights[index] = item.priority
                return item
            return None

        if item_id is not None:
            for i, item in enumerate(self._buffer):
                if item.id == item_id:
                    item.touch()
                    self._attention_weights[i] = item.priority
                    return item

        return None

    def update_priority(
        self,
        item_id: UUID,
        new_priority: float
    ) -> bool:
        """
        Update item priority.

        Args:
            item_id: Item to update
            new_priority: New priority value

        Returns:
            True if found and updated
        """
        for i, item in enumerate(self._buffer):
            if item.id == item_id:
                item.priority = np.clip(new_priority, 0.0, 1.0)
                self._attention_weights[i] = item.priority
                return True
        return False

    def peek_all(self) -> list[WorkingMemoryItem[T]]:
        """
        Get all items without modifying state.

        Returns:
            List of all items in buffer
        """
        return list(self._buffer)

    def get_by_priority(
        self,
        descending: bool = True
    ) -> list[WorkingMemoryItem[T]]:
        """
        Get items sorted by priority.

        Args:
            descending: Sort highest first if True

        Returns:
            Sorted list of items
        """
        indexed = list(zip(self._attention_weights, self._buffer))
        indexed.sort(key=lambda x: x[0], reverse=descending)
        return [item for _, item in indexed]

    def get_most_attended(self) -> WorkingMemoryItem[T] | None:
        """Get highest priority item."""
        if not self._buffer:
            return None
        max_idx = int(np.argmax(self._attention_weights))
        return self._buffer[max_idx]

    def get_least_attended(self) -> WorkingMemoryItem[T] | None:
        """Get lowest priority item (eviction candidate)."""
        if not self._buffer:
            return None
        min_idx = int(np.argmin(self._attention_weights))
        return self._buffer[min_idx]

    async def remove(
        self,
        item_id: UUID,
        consolidate: bool = False
    ) -> WorkingMemoryItem[T] | None:
        """
        Remove item from working memory.

        Args:
            item_id: Item to remove
            consolidate: Whether to consolidate to episodic memory

        Returns:
            Removed item if found
        """
        for i, item in enumerate(self._buffer):
            if item.id == item_id:
                self._buffer.pop(i)
                self._attention_weights.pop(i)
                item.state = ItemState.EVICTED

                if consolidate and self.episodic:
                    await self._consolidate_item(item)

                return item
        return None

    async def clear(
        self,
        consolidate_all: bool = False
    ) -> list[WorkingMemoryItem[T]]:
        """
        Clear all items from working memory.

        Args:
            consolidate_all: Whether to consolidate all items

        Returns:
            List of cleared items
        """
        cleared = list(self._buffer)

        if consolidate_all and self.episodic:
            for item in cleared:
                await self._consolidate_item(item)

        self._buffer.clear()
        self._attention_weights.clear()

        return cleared

    def _apply_decay(self) -> None:
        """Apply exponential decay to attention weights."""
        now = datetime.now()

        for i, item in enumerate(self._buffer):
            # Decay based on idle time
            idle_time = (now - item.last_accessed).total_seconds()
            decay_factor = np.exp(-self.decay_rate * idle_time)

            # Apply decay to attention weight
            self._attention_weights[i] = item.priority * decay_factor

            # Update item state
            if self._attention_weights[i] < self.min_priority:
                item.state = ItemState.DECAYING

    async def _evict_lowest(self) -> WorkingMemoryItem[T] | None:
        """Evict lowest priority item."""
        if not self._buffer:
            return None

        # Find minimum attention weight
        min_idx = int(np.argmin(self._attention_weights))
        item = self._buffer.pop(min_idx)
        priority = self._attention_weights.pop(min_idx)

        item.state = ItemState.EVICTED
        self._total_evictions += 1

        # Decide whether to consolidate
        consolidated = False
        if (self.episodic and priority >= self.consolidation_threshold):
            try:
                await self._consolidate_item(item)
                consolidated = True
                item.state = ItemState.CONSOLIDATED
            except Exception as e:
                logger.warning(f"Failed to consolidate evicted item: {e}")

        # Record eviction
        event = EvictionEvent(
            item_id=item.id,
            content=item.content,
            reason="capacity",
            priority=priority,
            age_seconds=item.age_seconds,
            consolidated=consolidated
        )
        self._eviction_history.append(event)

        # MEM-009 FIX: Trim history if over limit
        if len(self._eviction_history) > self._max_history_size:
            self._eviction_history = self._eviction_history[-self._max_history_size:]

        logger.debug(
            f"Evicted item {item.id} (priority={priority:.2f}, "
            f"consolidated={consolidated})"
        )

        return item

    async def _consolidate_item(
        self,
        item: WorkingMemoryItem[T]
    ) -> None:
        """Consolidate item to episodic memory."""
        if not self.episodic:
            return

        # Convert content to string if needed
        content = item.content
        if not isinstance(content, str):
            content = str(content)

        await self.episodic.create(
            content=content,
            context=item.metadata,
            outcome="neutral",
            valence=item.priority
        )

        logger.debug(f"Consolidated item {item.id} to episodic memory")

    async def rehearse(
        self,
        item_id: UUID | None = None
    ) -> list[WorkingMemoryItem[T]]:
        """
        Rehearse items to refresh attention.

        If item_id is None, rehearses all items.

        Args:
            item_id: Specific item to rehearse (optional)

        Returns:
            List of rehearsed items
        """
        rehearsed = []

        if item_id is not None:
            for i, item in enumerate(self._buffer):
                if item.id == item_id:
                    item.touch()
                    self._attention_weights[i] = item.priority
                    rehearsed.append(item)
                    break
        else:
            for i, item in enumerate(self._buffer):
                item.touch()
                self._attention_weights[i] = item.priority
                rehearsed.append(item)

        return rehearsed

    async def consolidate_decaying(self) -> list[EvictionEvent]:
        """
        Consolidate all decaying items.

        Returns:
            List of eviction events for consolidated items
        """
        events = []
        decaying = [
            (i, item) for i, item in enumerate(self._buffer)
            if item.state == ItemState.DECAYING
        ]

        # Remove in reverse order to maintain indices
        for i, item in sorted(decaying, key=lambda x: x[0], reverse=True):
            priority = self._attention_weights[i]
            self._buffer.pop(i)
            self._attention_weights.pop(i)

            consolidated = False
            if self.episodic and priority >= self.consolidation_threshold:
                try:
                    await self._consolidate_item(item)
                    consolidated = True
                    item.state = ItemState.CONSOLIDATED
                except Exception as e:
                    logger.warning(f"Failed to consolidate: {e}")

            event = EvictionEvent(
                item_id=item.id,
                content=item.content,
                reason="decay",
                priority=priority,
                age_seconds=item.age_seconds,
                consolidated=consolidated
            )
            events.append(event)
            self._eviction_history.append(event)

        # MEM-009 FIX: Trim history if over limit
        if len(self._eviction_history) > self._max_history_size:
            self._eviction_history = self._eviction_history[-self._max_history_size:]

        return events

    def get_eviction_history(
        self,
        limit: int = 100
    ) -> list[EvictionEvent]:
        """Get recent eviction history."""
        return self._eviction_history[-limit:]

    @property
    def size(self) -> int:
        """Current number of items in buffer."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Whether buffer is at capacity."""
        return len(self._buffer) >= self.capacity

    @property
    def available_capacity(self) -> int:
        """Remaining capacity."""
        return max(0, self.capacity - len(self._buffer))

    def get_stats(self) -> dict:
        """
        Get working memory statistics.

        Returns:
            Dict with buffer stats
        """
        active_priorities = []
        for i, item in enumerate(self._buffer):
            if item.state == ItemState.ACTIVE:
                active_priorities.append(self._attention_weights[i])

        return {
            "capacity": self.capacity,
            "current_size": len(self._buffer),
            "available": self.available_capacity,
            "total_loads": self._total_loads,
            "total_evictions": self._total_evictions,
            "eviction_rate": (
                self._total_evictions / self._total_loads
                if self._total_loads > 0 else 0.0
            ),
            "avg_priority": float(np.mean(active_priorities)) if active_priorities else 0.0,
            "active_count": len(active_priorities),
            "decaying_count": sum(1 for i in self._buffer if i.state == ItemState.DECAYING)
        }


class AttentionalBlink:
    """
    Simulates attentional blink - reduced capacity after attention-demanding events.

    When attention is consumed by a high-priority event, subsequent items
    have reduced processing capacity for a brief period.
    """

    def __init__(
        self,
        blink_duration_ms: int = 500,
        capacity_reduction: float = 0.5
    ):
        """
        Initialize attentional blink.

        Args:
            blink_duration_ms: Duration of reduced capacity
            capacity_reduction: Factor to reduce capacity by during blink
        """
        self.blink_duration = timedelta(milliseconds=blink_duration_ms)
        self.capacity_reduction = capacity_reduction
        self._blink_start: datetime | None = None

    def trigger(self) -> None:
        """Trigger an attentional blink."""
        self._blink_start = datetime.now()

    def is_blinking(self) -> bool:
        """Check if currently in blink period."""
        if self._blink_start is None:
            return False
        return datetime.now() - self._blink_start < self.blink_duration

    def get_effective_capacity(self, base_capacity: int) -> int:
        """Get capacity during blink."""
        if self.is_blinking():
            return max(1, int(base_capacity * self.capacity_reduction))
        return base_capacity


# Factory function
def create_working_memory(
    capacity: int = 4,
    episodic_memory: EpisodicMemory | None = None,
    **kwargs
) -> WorkingMemory:
    """
    Create a working memory buffer.

    Args:
        capacity: Maximum items (default 4)
        episodic_memory: Optional episodic memory for consolidation
        **kwargs: Additional arguments

    Returns:
        Configured WorkingMemory instance
    """
    return WorkingMemory(
        capacity=capacity,
        episodic_memory=episodic_memory,
        **kwargs
    )
