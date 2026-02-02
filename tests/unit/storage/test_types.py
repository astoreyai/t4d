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

    def test_spike_trace_and_graph_delta_round_trip(self):
        from t4dm.core.memory_item import MemoryItem

        trace = {"neuron_ids": [1, 2], "rate": 0.5}
        delta = {"added": [("a", "b")]}
        mi = MemoryItem(
            content="x", spike_trace=trace, graph_delta=delta,
        )
        rec = ItemRecord.from_memory_item(mi)
        assert rec.spike_trace == trace
        assert rec.graph_delta == delta

        mi2 = rec.to_memory_item()
        assert mi2.spike_trace == trace
        assert mi2.graph_delta == delta

    def test_spike_trace_none_round_trip(self):
        from t4dm.core.memory_item import MemoryItem

        mi = MemoryItem(content="x")
        rec = ItemRecord.from_memory_item(mi)
        assert rec.spike_trace is None
        assert rec.graph_delta is None
        mi2 = rec.to_memory_item()
        assert mi2.spike_trace is None
        assert mi2.graph_delta is None

    def test_dict_round_trip_with_spike_trace(self):
        now = time.time()
        rec = ItemRecord(
            id=uuid.uuid4().bytes, vector=[1.0], kappa=0.5,
            importance=0.8, event_time=now, record_time=now,
            valid_from=now, valid_until=None, item_type="episodic",
            content="hi", access_count=0, session_id=None,
            spike_trace={"k": 1}, graph_delta={"e": []},
        )
        d = rec.to_dict()
        assert d["spike_trace"] == {"k": 1}
        rec2 = ItemRecord.from_dict(d)
        assert rec2.spike_trace == {"k": 1}
        assert rec2.graph_delta == {"e": []}


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

    def test_created_at_default(self):
        e = EdgeRecord(
            source_id=uuid.uuid4().bytes,
            target_id=uuid.uuid4().bytes,
            edge_type="CAUSES",
        )
        assert e.created_at == 0.0

    def test_created_at_round_trip(self):
        now = time.time()
        e = EdgeRecord(
            source_id=uuid.uuid4().bytes,
            target_id=uuid.uuid4().bytes,
            edge_type="CAUSES",
            created_at=now,
        )
        d = e.to_dict()
        assert d["created_at"] == now
        e2 = EdgeRecord.from_dict(d)
        assert e2.created_at == now


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

    def test_all_legacy_types_present(self):
        expected = {
            "USES", "PRODUCES", "REQUIRES", "CAUSES", "PART_OF",
            "SIMILAR_TO", "IMPLEMENTS", "IMPROVES_ON", "CONSOLIDATED_INTO",
            "SOURCE_OF", "SEQUENCE", "TEMPORAL_BEFORE", "TEMPORAL_AFTER",
            "MERGED_FROM", "SUPERSEDES", "RELATES_TO", "HAS_CONTEXT",
            "DERIVED_FROM", "DEPENDS_ON",
        }
        actual = {e.value for e in EdgeType}
        assert expected <= actual
