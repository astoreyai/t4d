"""Tests for T4DX GlobalIndex."""

import uuid

from ww.storage.t4dx.global_index import GlobalIndex
from ww.storage.t4dx.types import EdgeRecord


class TestGlobalIndex:
    def test_register_and_locate(self):
        gi = GlobalIndex()
        hex_id = uuid.uuid4().hex
        gi.register_item(hex_id, 0)
        assert gi.locate(hex_id) == 0
        assert gi.is_live(hex_id)

    def test_tombstone(self):
        gi = GlobalIndex()
        hex_id = uuid.uuid4().hex
        gi.register_item(hex_id, 0)
        gi.tombstone(hex_id)
        assert not gi.is_live(hex_id)
        assert gi.locate(hex_id) is None

    def test_segment_tracking(self):
        gi = GlobalIndex()
        sid = gi.next_segment_id()
        assert sid == 0
        gi.register_segment(sid, {"test": True})
        assert sid in gi.segments
        gi.remove_segment(sid)
        assert sid not in gi.segments

    def test_cross_edges(self):
        gi = GlobalIndex()
        src = uuid.uuid4().bytes
        tgt = uuid.uuid4().bytes
        edge = EdgeRecord(source_id=src, target_id=tgt, edge_type="USES")
        gi.add_cross_edge(edge)
        results = gi.get_cross_edges(src, direction="out")
        assert len(results) == 1

    def test_save_load(self, tmp_path):
        gi = GlobalIndex()
        hex_id = uuid.uuid4().hex
        gi.register_item(hex_id, 0)
        gi.tombstone(uuid.uuid4().hex)
        gi.register_segment(0, {"test": True})

        path = tmp_path / "gi.json"
        gi.save(path)

        gi2 = GlobalIndex()
        gi2.load(path)
        assert gi2.locate(hex_id) == 0
        assert 0 in gi2.segments

    def test_load_nonexistent(self, tmp_path):
        gi = GlobalIndex()
        gi.load(tmp_path / "nope.json")  # should not raise
        assert gi.id_map == {}

    def test_next_segment_id_monotonic(self):
        gi = GlobalIndex()
        assert gi.next_segment_id() == 0
        assert gi.next_segment_id() == 1
        assert gi.next_segment_id() == 2
