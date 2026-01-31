"""P4-07: Background consolidation service — async with graceful shutdown."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ww.consolidation.sleep_cycle_v2 import SleepCycleV2, SleepCycleV2Config, SleepCycleV2Result
from ww.consolidation.sleep_scheduler import SleepScheduler, SleepSchedulerConfig, SleepTrigger
from ww.storage.t4dx.engine import T4DXEngine

logger = logging.getLogger(__name__)


class BackgroundConsolidationService:
    """Async background service for T4DX-based consolidation.

    Runs SleepScheduler in background, triggers SleepCycleV2 on demand.
    Thread-safe T4DX access via single-writer model.
    """

    def __init__(
        self,
        engine: T4DXEngine,
        spiking_stack: Any = None,
        scheduler_cfg: SleepSchedulerConfig | None = None,
        cycle_cfg: SleepCycleV2Config | None = None,
    ) -> None:
        self.engine = engine
        self.spiking = spiking_stack
        self._cycle = SleepCycleV2(engine, spiking_stack, cycle_cfg)
        self._lock = asyncio.Lock()
        self._results: list[SleepCycleV2Result] = []
        self._running = False

        self._scheduler = SleepScheduler(
            cfg=scheduler_cfg,
            on_sleep=self._on_sleep_trigger,
        )

    async def _on_sleep_trigger(self, trigger: SleepTrigger) -> None:
        """Called by scheduler when sleep is triggered."""
        await self.consolidate(trigger)

    async def consolidate(self, trigger: SleepTrigger = SleepTrigger.MANUAL) -> SleepCycleV2Result:
        """Run a consolidation cycle (thread-safe)."""
        async with self._lock:
            logger.info("Starting consolidation (trigger=%s)", trigger)
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._cycle.run)
            self._results.append(result)
            return result

    def notify_insert(self, count: int = 1) -> None:
        """Notify scheduler of new inserts."""
        self._scheduler.notify_activity(count)

    async def start(self) -> None:
        """Start background scheduler."""
        self._running = True
        await self._scheduler.start()
        logger.info("Background consolidation service started")

    async def stop(self) -> None:
        """Graceful shutdown: flush memtable, then stop."""
        logger.info("Stopping background consolidation service")
        await self._scheduler.stop()

        # Final flush
        if not self.engine._memtable.is_empty:
            self.engine.flush()

        self._running = False
        logger.info("Background consolidation service stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def results(self) -> list[SleepCycleV2Result]:
        return list(self._results)

    @property
    def scheduler(self) -> SleepScheduler:
        return self._scheduler
