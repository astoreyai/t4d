"""Tests for spike reinjection (P4-05)."""

import torch
import pytest

from tests.unit.consolidation.conftest import make_item
from ww.consolidation.spike_reinjection import SpikeReinjection, ReinjectionConfig
from ww.spiking.cortical_stack import CorticalStack
from ww.storage.t4dx.types import EdgeRecord


class TestSpikeReinjection:
    def test_select_for_replay(self, engine):
        for i in range(10):
            engine.insert(make_item(kappa=0.1, importance=0.3 + i * 0.05))
        stack = CorticalStack(dim=32, num_blocks=1, num_heads=4)
        reinj = SpikeReinjection(engine, stack)
        ids = reinj.select_for_replay(max_items=5)
        assert len(ids) <= 5

    def test_replay_updates_kappa(self, engine):
        item = make_item(kappa=0.1, importance=0.8)
        engine.insert(item)
        edge = EdgeRecord(
            source_id=item.id, target_id=item.id, edge_type="SELF", weight=0.5,
        )
        engine.insert_edge(edge)

        stack = CorticalStack(dim=32, num_blocks=1, num_heads=4)
        reinj = SpikeReinjection(engine, stack, ReinjectionConfig(kappa_increment=0.1))
        result = reinj.replay([item.id])

        assert result.items_replayed == 1
        assert result.kappa_updated == 1
        rec = engine.get(item.id)
        assert rec.kappa == pytest.approx(0.2, abs=0.01)

    def test_replay_missing_item(self, engine):
        stack = CorticalStack(dim=32, num_blocks=1, num_heads=4)
        reinj = SpikeReinjection(engine, stack)
        result = reinj.replay([b"\x00" * 16])
        assert result.items_replayed == 0
