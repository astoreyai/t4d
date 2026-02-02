"""Tests for KappaIndex secondary index."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from t4dm.storage.t4dx.kappa_index import KappaIndex


def _uid() -> bytes:
    return uuid.uuid4().bytes


class TestKappaIndex:
    def test_add_and_query_range(self):
        idx = KappaIndex()
        ids = [_uid() for _ in range(5)]
        kappas = [0.1, 0.3, 0.5, 0.7, 0.9]
        for item_id, k in zip(ids, kappas):
            idx.add(item_id, k)

        result = idx.query_range(0.2, 0.6)
        assert set(result) == {ids[1], ids[2]}

    def test_query_above(self):
        idx = KappaIndex()
        ids = [_uid() for _ in range(3)]
        for item_id, k in zip(ids, [0.2, 0.5, 0.8]):
            idx.add(item_id, k)
        result = idx.query_above(0.5)
        assert set(result) == {ids[1], ids[2]}

    def test_query_below(self):
        idx = KappaIndex()
        ids = [_uid() for _ in range(3)]
        for item_id, k in zip(ids, [0.2, 0.5, 0.8]):
            idx.add(item_id, k)
        result = idx.query_below(0.5)
        assert result == [ids[0]]

    def test_remove(self):
        idx = KappaIndex()
        a, b = _uid(), _uid()
        idx.add(a, 0.5)
        idx.add(b, 0.5)
        idx.remove(a)
        assert idx.query_range(0.0, 1.0) == [b]
        assert len(idx) == 1

    def test_update(self):
        idx = KappaIndex()
        a = _uid()
        idx.add(a, 0.3)
        idx.update(a, 0.8)
        assert idx.query_range(0.0, 0.5) == []
        assert idx.query_range(0.7, 0.9) == [a]

    def test_add_duplicate_replaces(self):
        idx = KappaIndex()
        a = _uid()
        idx.add(a, 0.3)
        idx.add(a, 0.7)
        assert len(idx) == 1
        assert idx.query_range(0.6, 0.8) == [a]

    def test_save_load(self, tmp_path: Path):
        idx = KappaIndex()
        ids = [_uid() for _ in range(3)]
        for item_id, k in zip(ids, [0.1, 0.5, 0.9]):
            idx.add(item_id, k)

        path = tmp_path / "kappa.json"
        idx.save(path)

        idx2 = KappaIndex()
        idx2.load(path)
        assert len(idx2) == 3
        assert idx2.query_range(0.0, 1.0) == idx.query_range(0.0, 1.0)

    def test_load_nonexistent(self, tmp_path: Path):
        idx = KappaIndex()
        idx.load(tmp_path / "nope.json")
        assert len(idx) == 0

    def test_empty_queries(self):
        idx = KappaIndex()
        assert idx.query_range(0.0, 1.0) == []
        assert idx.query_above(0.5) == []
        assert idx.query_below(0.5) == []

    def test_remove_nonexistent(self):
        idx = KappaIndex()
        idx.remove(_uid())  # should not raise
