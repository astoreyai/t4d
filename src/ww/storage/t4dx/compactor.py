"""Compactor: flush memtable, NREM/REM/PRUNE compaction.

LSM compaction = biological memory consolidation:
  flush()        = working memory → episodic store
  nrem_compact() = merge segments + κ boost + STDP
  rem_compact()  = cluster + prototype creation
  prune()        = GC tombstoned + low-κ items
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from ww.storage.t4dx.global_index import GlobalIndex
from ww.storage.t4dx.memtable import MemTable
from ww.storage.t4dx.segment import SegmentBuilder, SegmentReader
from ww.storage.t4dx.types import EdgeRecord, ItemRecord


class Compactor:
    """Manages segment lifecycle: flush, compact, prune."""

    def __init__(
        self,
        data_dir: Path,
        memtable: MemTable,
        segments: dict[int, SegmentReader],
        global_index: GlobalIndex,
    ) -> None:
        self._data_dir = data_dir
        self._memtable = memtable
        self._segments = segments
        self._global_index = global_index

    def _segment_dir(self, segment_id: int) -> Path:
        return self._data_dir / f"seg_{segment_id:06d}"

    def flush(self) -> int | None:
        """Flush memtable to a new L0 segment. Returns segment_id or None if empty."""
        items, edges, overlays, deltas, deleted = self._memtable.flush()

        # Apply tombstones
        for d in deleted:
            self._global_index.tombstone(d.hex())

        if not items:
            return None

        sid = self._global_index.next_segment_id()
        seg_dir = self._segment_dir(sid)
        builder = SegmentBuilder(seg_dir, sid, level=0)
        item_list = list(items.values())
        meta = builder.write(item_list, edges)

        # Register in global index
        reader = SegmentReader(seg_dir)
        self._segments[sid] = reader
        self._global_index.register_segment(sid, reader.manifest)
        for rec in item_list:
            self._global_index.register_item(rec.id.hex(), sid)

        return sid

    def nrem_compact(
        self,
        segment_ids: list[int] | None = None,
        kappa_boost: float = 0.05,
    ) -> int | None:
        """NREM compaction: merge segments, boost κ, remove tombstoned.

        If segment_ids is None, merges all L0 segments.
        Returns new segment_id or None if nothing to compact.
        """
        if segment_ids is None:
            segment_ids = [
                sid for sid, reader in self._segments.items()
                if reader.metadata.level == 0
            ]

        if len(segment_ids) < 2:
            return None

        merged_items: dict[bytes, ItemRecord] = {}
        merged_edges: list[EdgeRecord] = []

        for sid in segment_ids:
            reader = self._segments[sid]
            id_list = reader.manifest["id_list"]
            for i, d in enumerate(reader.items):
                hex_id = id_list[i]
                if hex_id in self._global_index.tombstones:
                    continue
                dd = dict(d)
                dd["vector"] = reader.vectors[i].tolist()
                dd["id"] = hex_id
                rec = ItemRecord.from_dict(dd)
                # κ boost
                rec.kappa = min(1.0, rec.kappa + kappa_boost)
                merged_items[rec.id] = rec

            for ed in reader.edges:
                merged_edges.append(EdgeRecord.from_dict(ed))

        if not merged_items:
            # Remove empty source segments
            for sid in segment_ids:
                self._remove_segment(sid)
            return None

        new_sid = self._global_index.next_segment_id()
        max_level = max(
            self._segments[s].metadata.level for s in segment_ids if s in self._segments
        )
        seg_dir = self._segment_dir(new_sid)
        builder = SegmentBuilder(seg_dir, new_sid, level=max_level + 1)
        meta = builder.write(list(merged_items.values()), merged_edges)

        reader = SegmentReader(seg_dir)
        self._segments[new_sid] = reader
        self._global_index.register_segment(new_sid, reader.manifest)
        for rec in merged_items.values():
            self._global_index.register_item(rec.id.hex(), new_sid)

        # Remove source segments
        for sid in segment_ids:
            self._remove_segment(sid)

        return new_sid

    def rem_compact(
        self,
        kappa_threshold: float = 0.4,
        prototype_kappa: float = 0.85,
    ) -> int | None:
        """REM compaction: cluster items above kappa_threshold, create prototypes.

        Simple centroid-based clustering using cosine similarity.
        Returns new segment_id with prototypes or None.
        """
        candidates: list[ItemRecord] = []
        for sid, reader in list(self._segments.items()):
            if reader.metadata.kappa_min < kappa_threshold:
                continue
            for rec in reader.scan(kappa_min=kappa_threshold):
                if rec.id.hex() not in self._global_index.tombstones:
                    candidates.append(rec)

        if len(candidates) < 2:
            return None

        # Simple single-centroid prototype
        vecs = np.array([c.vector for c in candidates], dtype=np.float32)
        centroid = vecs.mean(axis=0).tolist()
        contents = [c.content for c in candidates]
        prototype_content = f"[prototype] {contents[0][:100]}... (+{len(candidates)-1} more)"

        import uuid
        proto = ItemRecord(
            id=uuid.uuid4().bytes,
            vector=centroid,
            kappa=prototype_kappa,
            importance=max(c.importance for c in candidates),
            event_time=max(c.event_time for c in candidates),
            record_time=time.time(),
            valid_from=min(c.valid_from for c in candidates),
            valid_until=None,
            item_type="semantic",
            content=prototype_content,
            access_count=sum(c.access_count for c in candidates),
            session_id=None,
            metadata={"source_count": len(candidates), "source_ids": [c.id.hex() for c in candidates]},
        )

        new_sid = self._global_index.next_segment_id()
        seg_dir = self._segment_dir(new_sid)
        builder = SegmentBuilder(seg_dir, new_sid, level=2)
        builder.write([proto], [])

        reader = SegmentReader(seg_dir)
        self._segments[new_sid] = reader
        self._global_index.register_segment(new_sid, reader.manifest)
        self._global_index.register_item(proto.id.hex(), new_sid)

        return new_sid

    def prune(self, max_age_seconds: float | None = None, min_kappa: float = 0.0) -> int:
        """Remove tombstoned items and optionally items below thresholds. Returns removed count."""
        removed = 0

        for sid in list(self._segments.keys()):
            reader = self._segments[sid]
            id_list = reader.manifest["id_list"]
            keep_items: list[ItemRecord] = []
            keep_edges: list[EdgeRecord] = []

            now = time.time()
            for i, d in enumerate(reader.items):
                hex_id = id_list[i]
                if hex_id in self._global_index.tombstones:
                    removed += 1
                    continue
                if max_age_seconds is not None and (now - d["event_time"]) > max_age_seconds:
                    self._global_index.tombstone(hex_id)
                    removed += 1
                    continue
                if d["kappa"] < min_kappa:
                    self._global_index.tombstone(hex_id)
                    removed += 1
                    continue
                dd = dict(d)
                dd["vector"] = reader.vectors[i].tolist()
                dd["id"] = hex_id
                keep_items.append(ItemRecord.from_dict(dd))

            if not keep_items:
                self._remove_segment(sid)
                continue

            if len(keep_items) < reader.metadata.item_count:
                # Edges: keep only those with both endpoints alive
                live_ids = {r.id.hex() for r in keep_items}
                for ed in reader.edges:
                    if ed["source_id"] in live_ids and ed["target_id"] in live_ids:
                        keep_edges.append(EdgeRecord.from_dict(ed))

                # Rewrite segment
                self._remove_segment(sid)
                new_sid = self._global_index.next_segment_id()
                seg_dir = self._segment_dir(new_sid)
                builder = SegmentBuilder(seg_dir, new_sid, level=reader.metadata.level)
                builder.write(keep_items, keep_edges)
                new_reader = SegmentReader(seg_dir)
                self._segments[new_sid] = new_reader
                self._global_index.register_segment(new_sid, new_reader.manifest)
                for rec in keep_items:
                    self._global_index.register_item(rec.id.hex(), new_sid)

        # Clear tombstones for items no longer in any segment
        self._global_index.tombstones = {
            t for t in self._global_index.tombstones
            if any(
                t in (self._segments[s].manifest.get("id_list", []))
                for s in self._segments
            )
        }

        return removed

    def _remove_segment(self, sid: int) -> None:
        self._segments.pop(sid, None)
        self._global_index.remove_segment(sid)
        seg_dir = self._segment_dir(sid)
        if seg_dir.exists():
            shutil.rmtree(seg_dir)
