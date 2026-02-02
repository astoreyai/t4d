"""Tests for ProvenanceTracer."""

from __future__ import annotations

import uuid

import pytest

from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.provenance import ProvenanceNode, ProvenanceTracer
from t4dm.storage.t4dx.types import EdgeRecord, EdgeType, ItemRecord


def _make_record(item_id: bytes | None = None) -> ItemRecord:
    return ItemRecord(
        id=item_id or uuid.uuid4().bytes,
        vector=[1.0, 0.0, 0.0],
        kappa=0.5,
        importance=0.5,
        event_time=1000.0,
        record_time=2000.0,
        valid_from=1000.0,
        valid_until=None,
        item_type="episodic",
        content="test",
        access_count=0,
        session_id=None,
    )


def _make_edge(src: bytes, tgt: bytes, etype: str) -> EdgeRecord:
    return EdgeRecord(
        source_id=src,
        target_id=tgt,
        edge_type=etype,
        weight=1.0,
        created_at=1000.0,
    )


class TestProvenanceTracer:
    def test_forward_trace_single_hop(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        a, b = _make_record(), _make_record()
        engine.insert(a)
        engine.insert(b)
        engine.insert_edge(_make_edge(a.id, b.id, EdgeType.DERIVED_FROM.value))

        tracer = ProvenanceTracer(engine)
        result = tracer.forward_trace(a.id, max_depth=5)
        assert len(result) == 1
        assert result[0].item_id == b.id
        assert result[0].depth == 1
        assert result[0].parent_id == a.id

        engine.shutdown()

    def test_backward_trace(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        a, b = _make_record(), _make_record()
        engine.insert(a)
        engine.insert(b)
        engine.insert_edge(_make_edge(a.id, b.id, EdgeType.SOURCE_OF.value))

        tracer = ProvenanceTracer(engine)
        result = tracer.backward_trace(b.id, max_depth=5)
        assert len(result) == 1
        assert result[0].item_id == a.id

        engine.shutdown()

    def test_multi_hop_chain(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        records = [_make_record() for _ in range(4)]
        for r in records:
            engine.insert(r)
        for i in range(3):
            engine.insert_edge(
                _make_edge(records[i].id, records[i + 1].id, EdgeType.DERIVED_FROM.value)
            )

        tracer = ProvenanceTracer(engine)
        result = tracer.forward_trace(records[0].id, max_depth=10)
        assert len(result) == 3
        assert [n.depth for n in result] == [1, 2, 3]

        engine.shutdown()

    def test_max_depth_limits(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        records = [_make_record() for _ in range(5)]
        for r in records:
            engine.insert(r)
        for i in range(4):
            engine.insert_edge(
                _make_edge(records[i].id, records[i + 1].id, EdgeType.DERIVED_FROM.value)
            )

        tracer = ProvenanceTracer(engine)
        result = tracer.forward_trace(records[0].id, max_depth=2)
        assert len(result) == 2

        engine.shutdown()

    def test_ignores_non_provenance_edges(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        a, b = _make_record(), _make_record()
        engine.insert(a)
        engine.insert(b)
        engine.insert_edge(_make_edge(a.id, b.id, EdgeType.SIMILAR_TO.value))

        tracer = ProvenanceTracer(engine)
        result = tracer.forward_trace(a.id)
        assert len(result) == 0

        engine.shutdown()

    def test_lineage_graph(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        a, b, c = _make_record(), _make_record(), _make_record()
        for r in [a, b, c]:
            engine.insert(r)
        engine.insert_edge(_make_edge(a.id, b.id, EdgeType.DERIVED_FROM.value))
        engine.insert_edge(_make_edge(b.id, c.id, EdgeType.MERGED_FROM.value))

        tracer = ProvenanceTracer(engine)
        graph = tracer.lineage_graph(b.id)
        assert isinstance(graph, dict)
        # Forward from b -> c, backward from b -> a
        all_targets = []
        for entries in graph.values():
            for entry in entries:
                all_targets.append(entry["target"])
        assert c.id.hex() in all_targets or a.id.hex() in all_targets

        engine.shutdown()

    def test_no_provenance_edges(self, tmp_path):
        engine = T4DXEngine(tmp_path / "data")
        engine.startup()

        a = _make_record()
        engine.insert(a)

        tracer = ProvenanceTracer(engine)
        assert tracer.forward_trace(a.id) == []
        assert tracer.backward_trace(a.id) == []
        assert tracer.lineage_graph(a.id) == {}

        engine.shutdown()
