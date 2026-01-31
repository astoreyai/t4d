"""Tests for REM phase (P4-03)."""

import numpy as np
import pytest

from tests.unit.consolidation.conftest import make_item
from ww.consolidation.rem_phase import REMConfig, REMPhase


class TestREMPhase:
    def test_creates_prototypes(self, engine):
        # Insert similar items in kappa range [0.3, 0.7]
        np.random.seed(42)
        base_vec = np.random.randn(32).tolist()
        for i in range(5):
            item = make_item(kappa=0.5, content=f"similar-{i}")
            # Make vectors similar
            noise = (np.random.randn(32) * 0.1).tolist()
            item.vector = [b + n for b, n in zip(base_vec, noise)]
            engine.insert(item)

        cfg = REMConfig(kappa_min=0.3, kappa_max=0.7, min_cluster_size=3, similarity_threshold=0.5)
        rem = REMPhase(engine, cfg)
        result = rem.run()

        assert result.candidates_scanned == 5
        assert result.prototypes_created >= 1

    def test_no_candidates(self, engine):
        # Only low-kappa items
        for _ in range(3):
            engine.insert(make_item(kappa=0.1))
        rem = REMPhase(engine, REMConfig(kappa_min=0.5))
        result = rem.run()
        assert result.prototypes_created == 0

    def test_too_few_for_cluster(self, engine):
        engine.insert(make_item(kappa=0.5))
        rem = REMPhase(engine, REMConfig(min_cluster_size=3))
        result = rem.run()
        assert result.prototypes_created == 0
