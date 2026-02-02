"""Tests for BitemporalQuery."""

from __future__ import annotations

import time
import uuid

import pytest

from t4dm.storage.t4dx.bitemporal import BitemporalQuery
from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import ItemRecord


def _make_record(
    event_time: float = 1000.0,
    record_time: float = 2000.0,
    valid_from: float = 1000.0,
    valid_until: float | None = None,
) -> ItemRecord:
    return ItemRecord(
        id=uuid.uuid4().bytes,
        vector=[1.0, 0.0, 0.0],
        kappa=0.5,
        importance=0.5,
        event_time=event_time,
        record_time=record_time,
        valid_from=valid_from,
        valid_until=valid_until,
        item_type="episodic",
        content="test",
        access_count=0,
        session_id=None,
    )


class TestBitemporalQuery:
    def test_as_of_filters_by_record_time(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        r1 = _make_record(record_time=100.0)
        r2 = _make_record(record_time=200.0)
        r3 = _make_record(record_time=300.0)
        for r in [r1, r2, r3]:
            engine.insert(r)

        result = BitemporalQuery().as_of(150.0).execute(engine)
        assert len(result) == 1
        assert result[0].id == r1.id

        engine.shutdown()

    def test_valid_at_filters_by_validity(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        r1 = _make_record(valid_from=100.0, valid_until=200.0)
        r2 = _make_record(valid_from=150.0, valid_until=300.0)
        r3 = _make_record(valid_from=250.0, valid_until=None)
        for r in [r1, r2, r3]:
            engine.insert(r)

        result = BitemporalQuery().valid_at(175.0).execute(engine)
        ids = {r.id for r in result}
        assert r1.id in ids
        assert r2.id in ids
        assert r3.id not in ids

        engine.shutdown()

    def test_valid_at_open_ended(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        r1 = _make_record(valid_from=100.0, valid_until=None)
        engine.insert(r1)

        result = BitemporalQuery().valid_at(999999.0).execute(engine)
        assert len(result) == 1

        engine.shutdown()

    def test_between_filters_overlap(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        r1 = _make_record(valid_from=100.0, valid_until=200.0)
        r2 = _make_record(valid_from=250.0, valid_until=350.0)
        r3 = _make_record(valid_from=150.0, valid_until=None)
        for r in [r1, r2, r3]:
            engine.insert(r)

        result = BitemporalQuery().between(180.0, 260.0).execute(engine)
        ids = {r.id for r in result}
        assert r1.id in ids  # overlaps at [180, 200)
        assert r2.id in ids  # overlaps at [250, 260)
        assert r3.id in ids  # open-ended, starts at 150

        engine.shutdown()

    def test_combined_as_of_and_valid_at(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        r1 = _make_record(record_time=100.0, valid_from=50.0, valid_until=200.0)
        r2 = _make_record(record_time=300.0, valid_from=50.0, valid_until=200.0)
        for r in [r1, r2]:
            engine.insert(r)

        result = BitemporalQuery().as_of(150.0).valid_at(100.0).execute(engine)
        assert len(result) == 1
        assert result[0].id == r1.id

        engine.shutdown()

    def test_empty_result(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()
        result = BitemporalQuery().as_of(100.0).execute(engine)
        assert result == []
        engine.shutdown()
