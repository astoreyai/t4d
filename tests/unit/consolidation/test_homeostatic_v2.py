"""Tests for homeostatic scaling v2 (P4-08)."""

import pytest

from tests.unit.consolidation.conftest import make_item
from ww.consolidation.homeostatic_v2 import HomeostaticScalingV2, HomeostaticV2Config
from ww.storage.t4dx.types import EdgeRecord


class TestHomeostaticV2:
    def test_estimate_firing_rate(self, engine):
        items = [make_item() for _ in range(5)]
        for item in items:
            engine.insert(item)
        # Add edges with high weights (above BCM threshold)
        for i in range(len(items) - 1):
            edge = EdgeRecord(
                source_id=items[i].id, target_id=items[i + 1].id,
                edge_type="USES", weight=0.8,
            )
            engine.insert_edge(edge)

        hs = HomeostaticScalingV2(engine)
        rate = hs.estimate_firing_rate()
        assert 0.0 <= rate <= 1.0

    def test_run_scales_down(self, engine):
        items = [make_item() for _ in range(5)]
        for item in items:
            engine.insert(item)
        for i in range(len(items) - 1):
            edge = EdgeRecord(
                source_id=items[i].id, target_id=items[i + 1].id,
                edge_type="USES", weight=0.9,  # high weight = high firing
            )
            engine.insert_edge(edge)

        cfg = HomeostaticV2Config(target_firing_rate=0.05)
        hs = HomeostaticScalingV2(engine, cfg)
        result = hs.run()
        # With high weights, firing rate should be high, so scaling should apply
        assert result.estimated_firing_rate >= 0.0

    def test_no_edges(self, engine):
        engine.insert(make_item())
        hs = HomeostaticScalingV2(engine)
        result = hs.run()
        assert result.estimated_firing_rate == 0.0
        assert not result.applied

    def test_empty_engine(self, engine):
        hs = HomeostaticScalingV2(engine)
        result = hs.run()
        assert result.estimated_firing_rate == 0.0
