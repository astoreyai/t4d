"""P4-01: Sleep scheduler — triggers compaction based on pressure model.

Triggers T4DX compaction after N items in MemTable, T seconds idle,
or when adenosine pressure exceeds threshold.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class SleepTrigger(str, Enum):
    MEMTABLE_FULL = "memtable_full"
    IDLE_TIMEOUT = "idle_timeout"
    ADENOSINE_PRESSURE = "adenosine_pressure"
    MANUAL = "manual"


@dataclass
class SleepSchedulerConfig:
    """Configuration for sleep scheduling."""

    memtable_threshold: int = 5000
    idle_timeout_seconds: float = 300.0
    adenosine_threshold: float = 0.8
    adenosine_accumulation_rate: float = 0.01  # per insert
    adenosine_decay_rate: float = 0.005  # per second idle
    check_interval_seconds: float = 30.0
    enabled: bool = True


@dataclass
class SleepSchedulerState:
    """Observable state of the sleep scheduler."""

    adenosine: float = 0.0
    last_activity: float = field(default_factory=time.time)
    last_sleep: float = 0.0
    total_sleeps: int = 0
    total_items_since_sleep: int = 0
    running: bool = False


class SleepScheduler:
    """Triggers T4DX compaction based on biological sleep pressure model.

    Adenosine accumulates with activity (inserts) and decays with idle time.
    When pressure exceeds threshold, a sleep cycle is triggered.
    """

    def __init__(
        self,
        cfg: SleepSchedulerConfig | None = None,
        on_sleep: Callable[..., Coroutine] | None = None,
    ) -> None:
        self.cfg = cfg or SleepSchedulerConfig()
        self._on_sleep = on_sleep
        self._state = SleepSchedulerState()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    @property
    def state(self) -> SleepSchedulerState:
        return self._state

    def notify_activity(self, item_count: int = 1) -> SleepTrigger | None:
        """Called on each insert to accumulate pressure. Returns trigger if threshold hit."""
        self._state.last_activity = time.time()
        self._state.total_items_since_sleep += item_count
        self._state.adenosine = min(
            1.0,
            self._state.adenosine + self.cfg.adenosine_accumulation_rate * item_count,
        )

        if self._state.total_items_since_sleep >= self.cfg.memtable_threshold:
            return SleepTrigger.MEMTABLE_FULL
        if self._state.adenosine >= self.cfg.adenosine_threshold:
            return SleepTrigger.ADENOSINE_PRESSURE
        return None

    def _decay_adenosine(self) -> None:
        """Decay adenosine based on idle time."""
        now = time.time()
        idle = now - self._state.last_activity
        decay = self.cfg.adenosine_decay_rate * idle
        self._state.adenosine = max(0.0, self._state.adenosine - decay)

    def check(self) -> SleepTrigger | None:
        """Check if sleep should be triggered."""
        if not self.cfg.enabled:
            return None

        self._decay_adenosine()

        now = time.time()
        idle = now - self._state.last_activity

        if self._state.total_items_since_sleep >= self.cfg.memtable_threshold:
            return SleepTrigger.MEMTABLE_FULL
        if idle >= self.cfg.idle_timeout_seconds and self._state.total_items_since_sleep > 0:
            return SleepTrigger.IDLE_TIMEOUT
        if self._state.adenosine >= self.cfg.adenosine_threshold:
            return SleepTrigger.ADENOSINE_PRESSURE
        return None

    async def _trigger_sleep(self, reason: SleepTrigger) -> None:
        """Execute sleep cycle and reset counters."""
        logger.info("Sleep triggered: %s (adenosine=%.3f, items=%d)",
                     reason, self._state.adenosine, self._state.total_items_since_sleep)
        if self._on_sleep:
            await self._on_sleep(reason)
        self._state.last_sleep = time.time()
        self._state.total_sleeps += 1
        self._state.total_items_since_sleep = 0
        self._state.adenosine = 0.0

    async def run(self) -> None:
        """Background loop checking for sleep triggers."""
        self._state.running = True
        self._stop.clear()
        try:
            while not self._stop.is_set():
                trigger = self.check()
                if trigger is not None:
                    await self._trigger_sleep(trigger)
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.cfg.check_interval_seconds,
                    )
                    break  # stop was set
                except asyncio.TimeoutError:
                    pass
        finally:
            self._state.running = False

    async def start(self) -> None:
        """Start background scheduler."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        """Stop background scheduler."""
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    def reset(self) -> None:
        """Reset all state."""
        self._state = SleepSchedulerState()
