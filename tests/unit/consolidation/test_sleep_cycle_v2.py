"""Tests for full sleep cycle v2 (P4-06)."""

import pytest

from tests.unit.consolidation.conftest import make_item
from ww.consolidation.sleep_cycle_v2 import SleepCycleV2, SleepCycleV2Config
from ww.spiking.cortical_stack import CorticalStack


class TestSleepCycleV2:
    def test_full_cycle(self, populated_engine):
        engine, items = populated_engine
        cfg = SleepCycleV2Config(num_cycles=1, enable_reinjection=False)
        cycle = SleepCycleV2(engine, spiking_stack=None, cfg=cfg)
        result = cycle.run()

        assert len(result.nrem_results) == 1
        assert len(result.rem_results) == 1
        assert result.prune_result is not None
        assert result.duration_seconds > 0

    def test_with_spiking(self, populated_engine):
        engine, items = populated_engine
        stack = CorticalStack(dim=32, num_blocks=1, num_heads=4)
        cfg = SleepCycleV2Config(num_cycles=1, enable_reinjection=True)
        cycle = SleepCycleV2(engine, stack, cfg)
        result = cycle.run()

        assert result.total_replayed >= 0  # may replay items
        assert result.duration_seconds > 0

    def test_multiple_cycles(self, populated_engine):
        engine, items = populated_engine
        cfg = SleepCycleV2Config(num_cycles=3, enable_reinjection=False)
        cycle = SleepCycleV2(engine, cfg=cfg)
        result = cycle.run()

        assert len(result.nrem_results) == 3
        assert len(result.rem_results) == 3
