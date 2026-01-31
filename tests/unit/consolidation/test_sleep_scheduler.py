"""Tests for sleep scheduler (P4-01)."""

import asyncio
import time

import pytest

from t4dm.consolidation.sleep_scheduler import (
    SleepScheduler,
    SleepSchedulerConfig,
    SleepTrigger,
)


class TestSleepScheduler:
    def test_adenosine_accumulates(self):
        s = SleepScheduler(SleepSchedulerConfig(adenosine_threshold=0.5))
        s.notify_activity(10)
        assert s.state.adenosine > 0

    def test_memtable_trigger(self):
        s = SleepScheduler(SleepSchedulerConfig(memtable_threshold=5))
        trigger = s.notify_activity(5)
        assert trigger == SleepTrigger.MEMTABLE_FULL

    def test_adenosine_trigger(self):
        cfg = SleepSchedulerConfig(
            adenosine_threshold=0.1, adenosine_accumulation_rate=0.2,
        )
        s = SleepScheduler(cfg)
        trigger = s.notify_activity(1)
        assert trigger == SleepTrigger.ADENOSINE_PRESSURE

    def test_no_trigger_below_threshold(self):
        s = SleepScheduler(SleepSchedulerConfig(memtable_threshold=100))
        trigger = s.notify_activity(1)
        assert trigger is None

    def test_check_idle_timeout(self):
        cfg = SleepSchedulerConfig(idle_timeout_seconds=0.0)
        s = SleepScheduler(cfg)
        s.notify_activity(1)
        trigger = s.check()
        assert trigger == SleepTrigger.IDLE_TIMEOUT

    def test_disabled(self):
        s = SleepScheduler(SleepSchedulerConfig(enabled=False))
        s.notify_activity(100)
        assert s.check() is None

    def test_reset(self):
        s = SleepScheduler()
        s.notify_activity(10)
        s.reset()
        assert s.state.adenosine == 0.0
        assert s.state.total_items_since_sleep == 0

    @pytest.mark.asyncio
    async def test_trigger_sleep_callback(self):
        called = []

        async def on_sleep(trigger):
            called.append(trigger)

        cfg = SleepSchedulerConfig(memtable_threshold=3)
        s = SleepScheduler(cfg, on_sleep=on_sleep)
        s.notify_activity(3)
        await s._trigger_sleep(SleepTrigger.MEMTABLE_FULL)
        assert len(called) == 1
        assert s.state.total_sleeps == 1
        assert s.state.total_items_since_sleep == 0
