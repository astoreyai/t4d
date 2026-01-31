"""Tests for prune phase (P4-04)."""

import pytest

from tests.unit.consolidation.conftest import make_item
from ww.consolidation.prune_phase import PruneConfig, PrunePhase


class TestPrunePhase:
    def test_prune_low_kappa_low_importance(self, engine):
        # Insert items: some low-kappa+low-importance, some high
        for _ in range(3):
            engine.insert(make_item(kappa=0.01, importance=0.05))
        for _ in range(3):
            engine.insert(make_item(kappa=0.8, importance=0.9))

        # Flush to segments so prune has something to work on
        engine.flush()

        cfg = PruneConfig(kappa_threshold=0.05, importance_threshold=0.1)
        prune = PrunePhase(engine, cfg)
        result = prune.run()
        assert result.tombstoned >= 3

    def test_prune_nothing(self, engine):
        for _ in range(3):
            engine.insert(make_item(kappa=0.5, importance=0.5))
        cfg = PruneConfig(kappa_threshold=0.01)
        prune = PrunePhase(engine, cfg)
        result = prune.run()
        assert result.tombstoned == 0
