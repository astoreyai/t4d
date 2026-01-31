"""Tests for T4DXEngine."""

import numpy as np
import pytest

from tests.unit.storage.conftest import make_item
from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import EdgeRecord


class TestT4DXEngine:
    def _engine(self, tmp_path):
        e = T4DXEngine(tmp_path / "data", flush_threshold=100)
        e.startup()
        return e

    def test_insert_and_get(self, tmp_path):
        e = self._engine(tmp_path)
        item = make_item(dim=4)
        e.insert(item)
        rec = e.get(item.id)
        assert rec is not None
        assert rec.content == item.content
        e.shutdown()

    def test_delete(self, tmp_path):
        e = self._engine(tmp_path)
        item = make_item(dim=4)
        e.insert(item)
        e.delete(item.id)
        assert e.get(item.id) is None
        e.shutdown()

    def test_search(self, tmp_path):
        e = self._engine(tmp_path)
        item = make_item(dim=4)
        item.vector = [1.0, 0.0, 0.0, 0.0]
        e.insert(item)

        results = e.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0][0] == item.id
        e.shutdown()

    def test_update_fields(self, tmp_path):
        e = self._engine(tmp_path)
        item = make_item(dim=4)
        e.insert(item)
        e.update_fields(item.id, {"kappa": 0.9})
        rec = e.get(item.id)
        assert rec.kappa == 0.9
        e.shutdown()

    def test_traverse(self, tmp_path):
        e = self._engine(tmp_path)
        i1 = make_item(dim=4)
        i2 = make_item(dim=4)
        e.insert(i1)
        e.insert(i2)
        edge = EdgeRecord(source_id=i1.id, target_id=i2.id, edge_type="USES", weight=0.5)
        e.insert_edge(edge)

        edges = e.traverse(i1.id, direction="out")
        assert len(edges) == 1
        e.shutdown()

    def test_scan(self, tmp_path):
        e = self._engine(tmp_path)
        e.insert(make_item(dim=4, kappa=0.1))
        e.insert(make_item(dim=4, kappa=0.9))
        results = e.scan(kappa_min=0.5)
        assert len(results) == 1
        e.shutdown()

    def test_auto_flush(self, tmp_path):
        e = T4DXEngine(tmp_path / "data", flush_threshold=3)
        e.startup()
        for _ in range(5):
            e.insert(make_item(dim=4))
        assert e.segment_count >= 1
        e.shutdown()

    def test_wal_replay(self, tmp_path):
        data_dir = tmp_path / "data"
        e = T4DXEngine(data_dir, flush_threshold=100)
        e.startup()
        item = make_item(dim=4)
        e.insert(item)
        # Don't flush - just close WAL
        e._wal.close()
        e._started = False

        # Reopen - WAL should replay
        e2 = T4DXEngine(data_dir, flush_threshold=100)
        e2.startup()
        rec = e2.get(item.id)
        assert rec is not None
        assert rec.content == item.content
        e2.shutdown()

    def test_shutdown_flushes(self, tmp_path):
        e = self._engine(tmp_path)
        e.insert(make_item(dim=4))
        assert e.memtable_count == 1
        e.shutdown()
        # Reopen - should find item in segment
        e2 = T4DXEngine(tmp_path / "data", flush_threshold=100)
        e2.startup()
        assert e2.segment_count >= 1
        e2.shutdown()

    def test_batch_scale_weights(self, tmp_path):
        e = self._engine(tmp_path)
        i1 = make_item(dim=4)
        i2 = make_item(dim=4)
        e.insert_edge(EdgeRecord(source_id=i1.id, target_id=i2.id, edge_type="USES", weight=0.8))
        e.batch_scale_weights(0.5)
        edges = e.traverse(i1.id, direction="out")
        assert edges[0].weight == pytest.approx(0.4)
        e.shutdown()
