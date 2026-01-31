"""Tests for T4DX Compactor."""

import numpy as np
import pytest

from tests.unit.storage.conftest import make_edge, make_item
from t4dm.storage.t4dx.compactor import Compactor
from t4dm.storage.t4dx.global_index import GlobalIndex
from t4dm.storage.t4dx.memtable import MemTable
from t4dm.storage.t4dx.segment import SegmentReader


class TestCompactor:
    def _setup(self, tmp_path):
        mt = MemTable()
        gi = GlobalIndex()
        segments: dict[int, SegmentReader] = {}
        compactor = Compactor(tmp_path, mt, segments, gi)
        return compactor, mt, gi, segments

    def test_flush_creates_segment(self, tmp_path):
        compactor, mt, gi, segments = self._setup(tmp_path)
        items = [make_item(dim=4) for _ in range(3)]
        for i in items:
            mt.insert(i)
        edge = make_edge(items[0], items[1])
        mt.insert_edge(edge)

        sid = compactor.flush()
        assert sid is not None
        assert sid in segments
        assert mt.is_empty

    def test_flush_empty(self, tmp_path):
        compactor, mt, gi, segments = self._setup(tmp_path)
        assert compactor.flush() is None

    def test_nrem_compact(self, tmp_path):
        compactor, mt, gi, segments = self._setup(tmp_path)

        # Create two L0 segments
        for _ in range(2):
            for _ in range(3):
                mt.insert(make_item(dim=4))
            compactor.flush()

        assert len(segments) == 2
        new_sid = compactor.nrem_compact()
        assert new_sid is not None
        assert len(segments) == 1  # merged into one

    def test_nrem_compact_boosts_kappa(self, tmp_path):
        compactor, mt, gi, segments = self._setup(tmp_path)

        item = make_item(dim=4, kappa=0.1)
        mt.insert(item)
        compactor.flush()

        item2 = make_item(dim=4, kappa=0.2)
        mt.insert(item2)
        compactor.flush()

        new_sid = compactor.nrem_compact(kappa_boost=0.1)
        reader = segments[new_sid]
        for d in reader.items:
            assert d["kappa"] >= 0.2  # boosted by 0.1

    def test_rem_compact(self, tmp_path):
        compactor, mt, gi, segments = self._setup(tmp_path)

        # Insert items above kappa threshold
        for _ in range(5):
            mt.insert(make_item(dim=4, kappa=0.5))
        compactor.flush()

        new_sid = compactor.rem_compact(kappa_threshold=0.4)
        assert new_sid is not None
        reader = segments[new_sid]
        assert reader.metadata.item_count == 1  # one prototype

    def test_prune_removes_tombstoned(self, tmp_path):
        compactor, mt, gi, segments = self._setup(tmp_path)

        items = [make_item(dim=4) for _ in range(3)]
        for i in items:
            mt.insert(i)
        compactor.flush()

        gi.tombstone(items[0].id.hex())
        removed = compactor.prune()
        assert removed >= 1
