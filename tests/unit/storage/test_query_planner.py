"""Tests for T4DX QueryPlanner."""

import numpy as np
import pytest

from tests.unit.storage.conftest import make_item
from ww.storage.t4dx.global_index import GlobalIndex
from ww.storage.t4dx.memtable import MemTable
from ww.storage.t4dx.query_planner import QueryPlanner
from ww.storage.t4dx.segment import SegmentBuilder, SegmentReader


class TestQueryPlanner:
    def _setup(self, tmp_path, items=None):
        mt = MemTable()
        gi = GlobalIndex()
        segments = {}

        if items:
            # Write items to a segment
            seg_dir = tmp_path / "seg_000000"
            builder = SegmentBuilder(seg_dir, 0)
            builder.write(items, [])
            reader = SegmentReader(seg_dir)
            segments[0] = reader
            gi.register_segment(0, reader.manifest)
            for rec in items:
                gi.register_item(rec.id.hex(), 0)

        return QueryPlanner(mt, segments, gi), mt, gi

    def test_get_from_memtable(self, tmp_path):
        planner, mt, gi = self._setup(tmp_path)
        item = make_item(dim=4)
        mt.insert(item)
        assert planner.get(item.id) is item

    def test_get_from_segment(self, tmp_path):
        items = [make_item(dim=4)]
        planner, mt, gi = self._setup(tmp_path, items=items)
        rec = planner.get(items[0].id)
        assert rec is not None
        assert rec.content == items[0].content

    def test_get_tombstoned(self, tmp_path):
        items = [make_item(dim=4)]
        planner, mt, gi = self._setup(tmp_path, items=items)
        gi.tombstone(items[0].id.hex())
        assert planner.get(items[0].id) is None

    def test_search_merges(self, tmp_path):
        seg_items = [make_item(dim=4)]
        planner, mt, gi = self._setup(tmp_path, items=seg_items)

        mem_item = make_item(dim=4)
        mt.insert(mem_item)

        q = np.array(seg_items[0].vector, dtype=np.float32)
        results = planner.search(q, k=10)
        ids = {r[0] for r in results}
        assert seg_items[0].id in ids
        assert mem_item.id in ids

    def test_scan_filters(self, tmp_path):
        items = [
            make_item(dim=4, kappa=0.1, item_type="episodic"),
            make_item(dim=4, kappa=0.9, item_type="semantic"),
        ]
        planner, mt, gi = self._setup(tmp_path, items=items)
        results = planner.scan(kappa_min=0.5)
        assert len(results) == 1
        assert results[0].kappa >= 0.5
