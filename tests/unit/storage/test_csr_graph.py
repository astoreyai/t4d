"""Tests for CSRGraph."""

import uuid
from pathlib import Path

import pytest

from t4dm.storage.t4dx.csr_graph import CSRGraph
from t4dm.storage.t4dx.types import EdgeRecord


def _uid() -> bytes:
    return uuid.uuid4().bytes


def _edge(src: bytes, tgt: bytes, etype: str = "RELATES_TO", weight: float = 0.5) -> EdgeRecord:
    return EdgeRecord(source_id=src, target_id=tgt, edge_type=etype, weight=weight)


class TestCSRGraph:
    def test_empty_graph(self):
        g = CSRGraph()
        assert len(g) == 0
        assert g.neighbors(b"\x00" * 16) == []
        assert g.get_edge(b"\x00" * 16, b"\x01" * 16) is None

    def test_from_edges_empty(self):
        g = CSRGraph.from_edges([])
        assert len(g) == 0

    def test_from_edges_basic(self):
        a, b, c = _uid(), _uid(), _uid()
        edges = [_edge(a, b), _edge(a, c), _edge(b, c)]
        g = CSRGraph.from_edges(edges)
        assert len(g) == 3

    def test_neighbors_outgoing(self):
        a, b, c = _uid(), _uid(), _uid()
        edges = [_edge(a, b), _edge(a, c)]
        g = CSRGraph.from_edges(edges)
        out = g.neighbors(a, direction="outgoing")
        target_ids = {nid for nid, _ in out}
        assert target_ids == {b, c}

    def test_neighbors_incoming(self):
        a, b, c = _uid(), _uid(), _uid()
        edges = [_edge(a, c), _edge(b, c)]
        g = CSRGraph.from_edges(edges)
        inc = g.neighbors(c, direction="incoming")
        source_ids = {nid for nid, _ in inc}
        assert source_ids == {a, b}

    def test_neighbors_both(self):
        a, b, c = _uid(), _uid(), _uid()
        edges = [_edge(a, b), _edge(c, a)]
        g = CSRGraph.from_edges(edges)
        both = g.neighbors(a, direction="both")
        assert len(both) == 2

    def test_neighbors_unknown_node(self):
        a, b = _uid(), _uid()
        g = CSRGraph.from_edges([_edge(a, b)])
        assert g.neighbors(_uid()) == []

    def test_get_edge_found(self):
        a, b = _uid(), _uid()
        e = _edge(a, b, weight=0.7)
        g = CSRGraph.from_edges([e])
        found = g.get_edge(a, b)
        assert found is not None
        assert found.weight == pytest.approx(0.7)

    def test_get_edge_not_found(self):
        a, b, c = _uid(), _uid(), _uid()
        g = CSRGraph.from_edges([_edge(a, b)])
        assert g.get_edge(a, c) is None
        assert g.get_edge(b, a) is None  # wrong direction

    def test_save_load_roundtrip(self, tmp_path: Path):
        a, b, c = _uid(), _uid(), _uid()
        edges = [_edge(a, b, weight=0.3), _edge(b, c, weight=0.8)]
        g = CSRGraph.from_edges(edges)

        save_path = tmp_path / "csr_graph.npz"
        g.save(save_path)

        g2 = CSRGraph.load(save_path)
        assert len(g2) == 2
        assert g2.get_edge(a, b) is not None
        assert g2.get_edge(a, b).weight == pytest.approx(0.3)
        assert g2.get_edge(b, c).weight == pytest.approx(0.8)
        out = g2.neighbors(a, direction="outgoing")
        assert len(out) == 1

    def test_save_load_empty(self, tmp_path: Path):
        g = CSRGraph.from_edges([])
        save_path = tmp_path / "empty.npz"
        g.save(save_path)
        g2 = CSRGraph.load(save_path)
        assert len(g2) == 0
