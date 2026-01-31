"""Tests for T4DX types: ItemRecord, EdgeRecord, SegmentMetadata."""

import time
import uuid

import pytest

from t4dm.storage.t4dx.types import EdgeRecord, EdgeType, ItemRecord, SegmentMetadata


class TestItemRecord:
    def test_uuid_round_trip(self):
        u = uuid.uuid4()
        b = ItemRecord.uuid_to_bytes(u)
        assert len(b) == 16
        assert ItemRecord.bytes_to_uuid(b) == u

    def test_to_dict_from_dict_round_trip(self):
        now = time.time()
        rec = ItemRecord(
            id=uuid.uuid4().bytes,
            vector=[1.0, 2.0, 3.0],
            kappa=0.5,
            importance=0.8,
            event_time=now,
            record_time=now,
            valid_from=now,
            valid_until=None,
            item_type="episodic",
            content="hello",
            access_count=3,
            session_id="s1",
            metadata={"key": "val"},
        )
        d = rec.to_dict()
        rec2 = ItemRecord.from_dict(d)
        assert rec2.id == rec.id
        assert rec2.vector == rec.vector
        assert rec2.kappa == rec.kappa
        assert rec2.content == rec.content
        assert rec2.metadata == rec.metadata

    def test_memory_item_round_trip(self):
        from t4dm.core.memory_item import MemoryItem

        mi = MemoryItem(content="test", embedding=[1.0, 2.0], kappa=0.3)
        rec = ItemRecord.from_memory_item(mi)
        assert rec.kappa == 0.3
        assert rec.content == "test"

        mi2 = rec.to_memory_item()
        assert mi2.content == "test"
        assert mi2.kappa == pytest.approx(0.3)
        assert mi2.id == mi.id


class TestEdgeRecord:
    def test_round_trip(self):
        e = EdgeRecord(
            source_id=uuid.uuid4().bytes,
            target_id=uuid.uuid4().bytes,
            edge_type="USES",
            weight=0.7,
        )
        d = e.to_dict()
        e2 = EdgeRecord.from_dict(d)
        assert e2.source_id == e.source_id
        assert e2.weight == e.weight


class TestSegmentMetadata:
    def test_round_trip(self):
        m = SegmentMetadata(
            segment_id=0,
            item_count=10,
            edge_count=5,
            time_min=1000.0,
            time_max=2000.0,
            kappa_min=0.0,
            kappa_max=0.8,
            created_at=time.time(),
            level=1,
        )
        d = m.to_dict()
        m2 = SegmentMetadata.from_dict(d)
        assert m2.segment_id == 0
        assert m2.level == 1
        assert m2.item_count == 10


class TestEdgeType:
    def test_values(self):
        assert EdgeType.USES.value == "USES"
        assert EdgeType.CONSOLIDATED_INTO.value == "CONSOLIDATED_INTO"
