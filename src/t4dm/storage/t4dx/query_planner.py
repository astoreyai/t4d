"""Query planner: prune segments by time/kappa, search memtable + segments, merge."""

from __future__ import annotations

from typing import Any

import numpy as np

from t4dm.storage.t4dx.global_index import GlobalIndex
from t4dm.storage.t4dx.memtable import MemTable
from t4dm.storage.t4dx.segment import SegmentReader
from t4dm.storage.t4dx.types import ItemRecord


class QueryPlanner:
    """Plans and executes queries across memtable and segments."""

    def __init__(
        self,
        memtable: MemTable,
        segments: dict[int, SegmentReader],
        global_index: GlobalIndex,
    ) -> None:
        self._memtable = memtable
        self._segments = segments
        self._global_index = global_index

    def _prune_segments(
        self,
        time_min: float | None = None,
        time_max: float | None = None,
        kappa_min: float | None = None,
        kappa_max: float | None = None,
    ) -> list[int]:
        """Return segment IDs that might contain matching items."""
        result = []
        for sid, reader in self._segments.items():
            meta = reader.metadata
            # Time range overlap check
            if time_min is not None and meta.time_max < time_min:
                continue
            if time_max is not None and meta.time_min > time_max:
                continue
            # Kappa range overlap check
            if kappa_min is not None and meta.kappa_max < kappa_min:
                continue
            if kappa_max is not None and meta.kappa_min > kappa_max:
                continue
            result.append(sid)
        return result

    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        time_min: float | None = None,
        time_max: float | None = None,
        kappa_min: float | None = None,
        kappa_max: float | None = None,
        item_type: str | None = None,
    ) -> list[tuple[bytes, float]]:
        """Search memtable + relevant segments, merge top-k results."""
        q = np.array(query_vector, dtype=np.float32)

        # Search memtable
        all_results = self._memtable.search(
            q, k=k,
            time_min=time_min, time_max=time_max,
            kappa_min=kappa_min, kappa_max=kappa_max,
            item_type=item_type,
        )

        # Search pruned segments
        candidate_sids = self._prune_segments(time_min, time_max, kappa_min, kappa_max)
        for sid in candidate_sids:
            reader = self._segments[sid]
            seg_results = reader.search(
                q, k=k,
                time_min=time_min, time_max=time_max,
                kappa_min=kappa_min, kappa_max=kappa_max,
                item_type=item_type,
            )
            all_results.extend(seg_results)

        # Filter tombstoned
        tombstones = self._global_index.tombstones
        all_results = [
            (rid, score) for rid, score in all_results
            if rid.hex() not in tombstones
        ]

        # Deduplicate: keep highest score per id
        best: dict[bytes, float] = {}
        for rid, score in all_results:
            if rid not in best or score > best[rid]:
                best[rid] = score

        # Sort and limit
        sorted_results = sorted(best.items(), key=lambda x: -x[1])
        return sorted_results[:k]

    def get(self, item_id: bytes) -> ItemRecord | None:
        """Get a single item by ID from memtable or segments."""
        hex_id = item_id.hex()
        if hex_id in self._global_index.tombstones:
            return None

        # Check memtable first
        rec = self._memtable.get(item_id)
        if rec is not None:
            return rec

        # Check segment
        seg_id = self._global_index.locate(hex_id)
        if seg_id is not None and seg_id in self._segments:
            return self._segments[seg_id].get(item_id)

        return None

    def scan(
        self,
        time_min: float | None = None,
        time_max: float | None = None,
        kappa_min: float | None = None,
        kappa_max: float | None = None,
        item_type: str | None = None,
    ) -> list[ItemRecord]:
        """Scan all items matching filters."""
        tombstones = self._global_index.tombstones
        results: list[ItemRecord] = []

        # Memtable scan
        for rec in self._memtable._items.values():
            if rec.id.hex() in tombstones:
                continue
            if time_min is not None and rec.event_time < time_min:
                continue
            if time_max is not None and rec.event_time > time_max:
                continue
            if kappa_min is not None and rec.kappa < kappa_min:
                continue
            if kappa_max is not None and rec.kappa > kappa_max:
                continue
            if item_type is not None and rec.item_type != item_type:
                continue
            results.append(rec)

        # Segment scan
        candidate_sids = self._prune_segments(time_min, time_max, kappa_min, kappa_max)
        for sid in candidate_sids:
            reader = self._segments[sid]
            seg_items = reader.scan(
                time_min=time_min, time_max=time_max,
                kappa_min=kappa_min, kappa_max=kappa_max,
                item_type=item_type,
            )
            for rec in seg_items:
                if rec.id.hex() not in tombstones:
                    results.append(rec)

        return results
