"""Tests for T4DX MemTable."""

import numpy as np
import pytest

from tests.unit.storage.conftest import make_edge, make_item
from t4dm.storage.t4dx.memtable import MemTable


class TestMemTable:
    def test_insert_and_get(self):
        mt = MemTable()
        item = make_item()
        mt.insert(item)
        assert mt.item_count == 1
        assert mt.get(item.id) is item

    def test_delete(self):
        mt = MemTable()
        item = make_item()
        mt.insert(item)
        mt.delete(item.id)
        assert mt.get(item.id) is None

    def test_update_fields_in_memory(self):
        mt = MemTable()
        item = make_item()
        mt.insert(item)
        mt.update_fields(item.id, {"kappa": 0.9})
        assert mt.get(item.id).kappa == 0.9

    def test_update_fields_overlay(self):
        mt = MemTable()
        fake_id = b"\x00" * 16
        mt.update_fields(fake_id, {"kappa": 0.5})
        _, _, overlays, _, _ = mt.flush()
        assert len(overlays) == 1
        assert overlays[0].fields["kappa"] == 0.5

    def test_search_cosine(self):
        mt = MemTable()
        item = make_item(dim=4)
        item.vector = [1.0, 0.0, 0.0, 0.0]
        mt.insert(item)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = mt.search(query, k=1)
        assert len(results) == 1
        assert results[0][0] == item.id
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_search_with_filters(self):
        mt = MemTable()
        i1 = make_item(kappa=0.1, item_type="episodic")
        i2 = make_item(kappa=0.9, item_type="semantic")
        mt.insert(i1)
        mt.insert(i2)

        q = np.array(i2.vector, dtype=np.float32)
        results = mt.search(q, k=10, kappa_min=0.5)
        assert all(mt.get(r[0]).kappa >= 0.5 for r in results)

    def test_edge_operations(self):
        mt = MemTable()
        i1 = make_item()
        i2 = make_item()
        mt.insert(i1)
        mt.insert(i2)

        edge = make_edge(i1, i2)
        mt.insert_edge(edge)
        assert mt.edge_count == 1

        edges = mt.get_edges(i1.id, direction="out")
        assert len(edges) == 1

        mt.delete_edge(i1.id, i2.id, "USES")
        assert mt.edge_count == 0

    def test_update_edge_weight(self):
        mt = MemTable()
        i1 = make_item()
        i2 = make_item()
        edge = make_edge(i1, i2, weight=0.5)
        mt.insert_edge(edge)
        mt.update_edge_weight(i1.id, i2.id, "USES", 0.2)
        assert mt._edges[0].weight == pytest.approx(0.7)

    def test_batch_scale_weights(self):
        mt = MemTable()
        i1 = make_item()
        i2 = make_item()
        edge = make_edge(i1, i2, weight=0.5)
        mt.insert_edge(edge)
        mt.batch_scale_weights(0.5)
        assert mt._edges[0].weight == pytest.approx(0.25)

    def test_flush_drains(self):
        mt = MemTable()
        mt.insert(make_item())
        assert not mt.is_empty
        mt.flush()
        assert mt.is_empty

    def test_search_empty(self):
        mt = MemTable()
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert mt.search(q) == []
