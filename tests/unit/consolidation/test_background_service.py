"""Tests for background consolidation service (P4-07)."""

import asyncio

import pytest

from ww.consolidation.background_service import BackgroundConsolidationService
from ww.consolidation.sleep_scheduler import SleepSchedulerConfig, SleepTrigger
from ww.consolidation.sleep_cycle_v2 import SleepCycleV2Config
from tests.unit.consolidation.conftest import make_item


class TestBackgroundService:
    @pytest.mark.asyncio
    async def test_manual_consolidate(self, engine):
        for _ in range(10):
            engine.insert(make_item(kappa=0.1, importance=0.5))

        svc = BackgroundConsolidationService(
            engine,
            scheduler_cfg=SleepSchedulerConfig(enabled=False),
            cycle_cfg=SleepCycleV2Config(num_cycles=1, enable_reinjection=False),
        )
        result = await svc.consolidate()
        assert result is not None
        assert len(svc.results) == 1

    @pytest.mark.asyncio
    async def test_notify_insert(self, engine):
        svc = BackgroundConsolidationService(
            engine,
            scheduler_cfg=SleepSchedulerConfig(enabled=False),
        )
        svc.notify_insert(5)
        assert svc.scheduler.state.total_items_since_sleep == 5

    @pytest.mark.asyncio
    async def test_start_stop(self, engine):
        svc = BackgroundConsolidationService(
            engine,
            scheduler_cfg=SleepSchedulerConfig(
                enabled=True, check_interval_seconds=0.1,
            ),
        )
        await svc.start()
        assert svc.is_running
        await asyncio.sleep(0.2)
        await svc.stop()
        assert not svc.is_running
