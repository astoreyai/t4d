"""Tests for NREM phase (P4-02)."""

import pytest

from tests.unit.consolidation.conftest import make_item
from t4dm.consolidation.nrem_phase import NREMConfig, NREMPhase


class TestNREMPhase:
    def test_run_boosts_kappa(self, populated_engine):
        engine, items = populated_engine
        cfg = NREMConfig(kappa_ceiling=0.3, min_importance=0.0, replay_passes=1)
        nrem = NREMPhase(engine, spiking_stack=None, cfg=cfg)
        result = nrem.run()

        assert result.replayed > 0
        assert result.kappa_updated > 0

        # Check kappa was boosted
        for item in items:
            if item.kappa < 0.3:
                rec = engine.get(item.id)
                if rec:
                    assert rec.kappa >= item.kappa

    def test_run_no_candidates(self, engine):
        # Insert only high-kappa items
        for _ in range(3):
            engine.insert(make_item(kappa=0.9))
        nrem = NREMPhase(engine, cfg=NREMConfig(kappa_ceiling=0.1))
        result = nrem.run()
        assert result.replayed == 0

    def test_run_with_edges(self, populated_engine):
        engine, items = populated_engine
        cfg = NREMConfig(kappa_ceiling=0.3, min_importance=0.0, replay_passes=1)
        nrem = NREMPhase(engine, spiking_stack=None, cfg=cfg)
        result = nrem.run()
        # Edges can't be strengthened without spiking stack doing replay
        # but κ should still be updated
        assert result.kappa_updated > 0
