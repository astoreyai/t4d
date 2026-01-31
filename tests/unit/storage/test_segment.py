"""Tests for T4DX Segment (Builder + Reader)."""

import numpy as np
import pytest

from tests.unit.storage.conftest import make_edge, make_item
from ww.storage.t4dx.segment import SegmentBuilder, SegmentReader


class TestSegmentBuilder:
    def test_write_creates_files(self, tmp_path):
        seg_dir = tmp_path / "seg_000000"
        builder = SegmentBuilder(seg_dir, segment_id=0)
        items = [make_item(dim=4) for _ in range(3)]
        edges = [make_edge(items[0], items[1])]
        meta = builder.write(items, edges)

        assert (seg_dir / "vectors.npy").exists()
        assert (seg_dir / "items.json").exists()
        assert (seg_dir / "edges.json").exists()
        assert (seg_dir / "manifest.json").exists()
        assert meta.item_count == 3
        assert meta.edge_count == 1

    def test_write_empty_raises(self, tmp_path):
        builder = SegmentBuilder(tmp_path / "empty", segment_id=0)
        with pytest.raises(ValueError):
            builder.write([], [])


class TestSegmentReader:
    def _build_segment(self, tmp_path, items=None, edges=None):
        items = items or [make_item(dim=4) for _ in range(5)]
        edges = edges or [make_edge(items[0], items[1])]
        seg_dir = tmp_path / "seg_000000"
        builder = SegmentBuilder(seg_dir, segment_id=0)
        builder.write(items, edges)
        return SegmentReader(seg_dir), items, edges

    def test_get(self, tmp_path):
        reader, items, _ = self._build_segment(tmp_path)
        rec = reader.get(items[0].id)
        assert rec is not None
        assert rec.content == items[0].content

    def test_get_missing(self, tmp_path):
        reader, _, _ = self._build_segment(tmp_path)
        assert reader.get(b"\x00" * 16) is None

    def test_search(self, tmp_path):
        reader, items, _ = self._build_segment(tmp_path)
        q = np.array(items[0].vector, dtype=np.float32)
        results = reader.search(q, k=2)
        assert len(results) <= 2
        assert results[0][0] == items[0].id  # best match is self

    def test_scan(self, tmp_path):
        items = [
            make_item(dim=4, kappa=0.1),
            make_item(dim=4, kappa=0.9),
        ]
        reader, _, _ = self._build_segment(tmp_path, items=items, edges=[])
        results = reader.scan(kappa_min=0.5)
        assert len(results) == 1
        assert results[0].kappa >= 0.5

    def test_traverse(self, tmp_path):
        reader, items, edges = self._build_segment(tmp_path)
        results = reader.traverse(items[0].id, direction="out")
        assert len(results) == 1
        assert results[0].edge_type == "USES"

    def test_metadata(self, tmp_path):
        reader, _, _ = self._build_segment(tmp_path)
        meta = reader.metadata
        assert meta.item_count == 5
        assert meta.segment_id == 0
