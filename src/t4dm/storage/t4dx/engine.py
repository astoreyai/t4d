"""T4DXEngine: 9 public operations, startup/shutdown, auto-flush."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from t4dm.storage.t4dx.compactor import Compactor
from t4dm.storage.t4dx.global_index import GlobalIndex
from t4dm.storage.t4dx.kappa_index import KappaIndex
from t4dm.storage.t4dx.memtable import MemTable
from t4dm.storage.t4dx.query_planner import QueryPlanner
from t4dm.storage.t4dx.segment import SegmentReader
from t4dm.storage.t4dx.types import EdgeRecord, ItemRecord
from t4dm.storage.t4dx.wal import OpType, WAL, WALWriter

logger = logging.getLogger(__name__)

_DEFAULT_FLUSH_THRESHOLD = 1000


class T4DXEngine:
    """Embedded LSM-style storage engine with 9 public operations.

    Operations: INSERT, GET, SEARCH, UPDATE_FIELDS, UPDATE_EDGE_WEIGHT,
    TRAVERSE, SCAN, DELETE, BATCH_SCALE_WEIGHTS.
    """

    def __init__(
        self,
        data_dir: str | Path,
        flush_threshold: int = _DEFAULT_FLUSH_THRESHOLD,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._flush_threshold = flush_threshold

        self._lock = threading.RLock()
        self._memtable = MemTable()
        self._global_index = GlobalIndex()
        self._segments: dict[int, SegmentReader] = {}
        self._wal = WAL(self._data_dir / "wal.jsonl")
        self._compactor = Compactor(
            self._data_dir, self._memtable, self._segments, self._global_index,
        )
        self._kappa_index = KappaIndex()
        self._started = False
        self._batch_writer: WALWriter | None = None

    # --- lifecycle ---

    def startup(self) -> None:
        """Open WAL, load global index, load segments, replay WAL."""
        self._wal.open()
        gi_path = self._data_dir / "global_index.json"
        self._global_index.load(gi_path)

        # Load existing segments
        for sid, manifest in list(self._global_index.segments.items()):
            seg_dir = self._data_dir / f"seg_{sid:06d}"
            if seg_dir.exists():
                self._segments[sid] = SegmentReader(seg_dir)
            else:
                self._global_index.remove_segment(sid)

        # Replay WAL
        entries = self._wal.replay()
        for entry in entries:
            self._apply_wal_entry(entry)

        # Load kappa index
        ki_path = self._data_dir / "kappa_index.json"
        self._kappa_index.load(ki_path)

        self._started = True
        logger.info("T4DXEngine started with %d segments", len(self._segments))

    def shutdown(self) -> None:
        """Flush memtable, save global index, close WAL."""
        if not self._memtable.is_empty:
            self._compactor.flush()
        self._global_index.save(self._data_dir / "global_index.json")
        self._kappa_index.save(self._data_dir / "kappa_index.json")
        self._wal.close()
        self._started = False
        logger.info("T4DXEngine shutdown complete")

    # --- 9 public operations ---

    def insert(self, record: ItemRecord) -> None:
        """INSERT: add an item to the memtable."""
        with self._lock:
            self._wal.append(OpType.INSERT, {"item": record.to_dict()})
            self._memtable.insert(record)
            self._kappa_index.add(record.id, record.kappa)
            self._maybe_flush()

    def get(self, item_id: bytes) -> ItemRecord | None:
        """GET: retrieve a single item by 16-byte UUID."""
        with self._lock:
            planner = self._make_planner()
            return planner.get(item_id)

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
        """SEARCH: vector similarity search with filters."""
        with self._lock:
            planner = self._make_planner()
            return planner.search(
                query_vector, k=k,
                time_min=time_min, time_max=time_max,
                kappa_min=kappa_min, kappa_max=kappa_max,
                item_type=item_type,
            )

    def update_fields(self, item_id: bytes, fields: dict[str, Any]) -> None:
        """UPDATE_FIELDS: overlay field mutations."""
        with self._lock:
            self._wal.append(OpType.UPDATE_FIELDS, {
                "item_id": item_id.hex(), "fields": fields,
            })
            self._memtable.update_fields(item_id, fields)
            if "kappa" in fields:
                self._kappa_index.update(item_id, fields["kappa"])

    def update_edge_weight(
        self,
        source_id: bytes,
        target_id: bytes,
        edge_type: str,
        weight_delta: float,
    ) -> None:
        """UPDATE_EDGE_WEIGHT: Hebbian weight update."""
        with self._lock:
            self._wal.append(OpType.UPDATE_EDGE_WEIGHT, {
                "source_id": source_id.hex(),
                "target_id": target_id.hex(),
                "edge_type": edge_type,
                "weight_delta": weight_delta,
            })
            self._memtable.update_edge_weight(source_id, target_id, edge_type, weight_delta)

    def traverse(
        self,
        node_id: bytes,
        edge_type: str | None = None,
        direction: str = "both",
    ) -> list[EdgeRecord]:
        """TRAVERSE: get edges for a node across memtable + segments + cross-edges."""
        with self._lock:
            results = self._memtable.get_edges(node_id, edge_type, direction)

            # Check segments
            hex_id = node_id.hex()
            for sid, reader in self._segments.items():
                results.extend(reader.traverse(node_id, edge_type, direction))

            # Cross-edges
            results.extend(
                self._global_index.get_cross_edges(node_id, edge_type, direction)
            )

            return results

    def scan(
        self,
        time_min: float | None = None,
        time_max: float | None = None,
        kappa_min: float | None = None,
        kappa_max: float | None = None,
        item_type: str | None = None,
    ) -> list[ItemRecord]:
        """SCAN: filtered iteration over all items."""
        with self._lock:
            planner = self._make_planner()
            return planner.scan(
                time_min=time_min, time_max=time_max,
                kappa_min=kappa_min, kappa_max=kappa_max,
                item_type=item_type,
            )

    def delete(self, item_id: bytes) -> None:
        """DELETE: tombstone an item."""
        with self._lock:
            self._wal.append(OpType.DELETE, {"item_id": item_id.hex()})
            self._memtable.delete(item_id)
            self._global_index.tombstone(item_id.hex())
            self._kappa_index.remove(item_id)

    def batch_scale_weights(self, factor: float) -> None:
        """BATCH_SCALE_WEIGHTS: scale all edge weights in memtable (uses batch WAL)."""
        with self._lock:
            self._wal_append(OpType.BATCH_SCALE_WEIGHTS, {"factor": factor})
            self._memtable.batch_scale_weights(factor)

    # --- edge insert (used by graph adapter) ---

    def insert_edge(self, edge: EdgeRecord) -> None:
        """Insert an edge into the memtable."""
        with self._lock:
            self._wal.append(OpType.INSERT_EDGE, {"edge": edge.to_dict()})
            self._memtable.insert_edge(edge)

    def delete_edge(self, source_id: bytes, target_id: bytes, edge_type: str) -> None:
        """Delete an edge."""
        with self._lock:
            self._wal.append(OpType.DELETE_EDGE, {
                "source_id": source_id.hex(),
                "target_id": target_id.hex(),
                "edge_type": edge_type,
            })
            self._memtable.delete_edge(source_id, target_id, edge_type)

    # --- kappa queries ---

    def query_by_kappa(self, kappa_min: float, kappa_max: float) -> list[bytes]:
        """Return item IDs with kappa in [kappa_min, kappa_max] via secondary index."""
        with self._lock:
            return self._kappa_index.query_range(kappa_min, kappa_max)

    # --- compaction ---

    def flush(self) -> int | None:
        """Force flush memtable to segment."""
        with self._lock:
            sid = self._compactor.flush()
            if sid is not None:
                self._wal.truncate()
            return sid

    def nrem_compact(self, **kwargs: Any) -> int | None:
        with self._lock:
            return self._compactor.nrem_compact(**kwargs)

    def rem_compact(self, **kwargs: Any) -> int | None:
        with self._lock:
            return self._compactor.rem_compact(**kwargs)

    def prune(self, **kwargs: Any) -> int:
        with self._lock:
            return self._compactor.prune(**kwargs)

    # --- batch WAL mode ---

    def begin_batch(self) -> None:
        """Start batch mode: WAL entries accumulate, fsync once at end_batch()."""
        self._batch_writer = WALWriter(self._wal)
        self._batch_writer.__enter__()

    def end_batch(self) -> None:
        """End batch mode and fsync all accumulated WAL entries."""
        if self._batch_writer is not None:
            self._batch_writer.__exit__(None, None, None)
            self._batch_writer = None

    def _wal_append(self, op: OpType, payload: dict[str, Any]) -> None:
        """Append to WAL, respecting batch mode."""
        if self._batch_writer is not None:
            self._batch_writer.append(op, payload)
        else:
            self._wal.append(op, payload)

    # --- internals ---

    def _make_planner(self) -> QueryPlanner:
        return QueryPlanner(self._memtable, self._segments, self._global_index)

    def _maybe_flush(self) -> None:
        if self._memtable.item_count >= self._flush_threshold:
            self.flush()

    def _apply_wal_entry(self, entry: dict[str, Any]) -> None:
        """Replay a single WAL entry into the memtable."""
        op = entry.get("op")
        if op == OpType.INSERT.value:
            rec = ItemRecord.from_dict(entry["item"])
            self._memtable.insert(rec)
        elif op == OpType.DELETE.value:
            item_id = bytes.fromhex(entry["item_id"])
            self._memtable.delete(item_id)
            self._global_index.tombstone(entry["item_id"])
        elif op == OpType.UPDATE_FIELDS.value:
            item_id = bytes.fromhex(entry["item_id"])
            self._memtable.update_fields(item_id, entry["fields"])
        elif op == OpType.INSERT_EDGE.value:
            edge = EdgeRecord.from_dict(entry["edge"])
            self._memtable.insert_edge(edge)
        elif op == OpType.DELETE_EDGE.value:
            self._memtable.delete_edge(
                bytes.fromhex(entry["source_id"]),
                bytes.fromhex(entry["target_id"]),
                entry["edge_type"],
            )
        elif op == OpType.UPDATE_EDGE_WEIGHT.value:
            self._memtable.update_edge_weight(
                bytes.fromhex(entry["source_id"]),
                bytes.fromhex(entry["target_id"]),
                entry["edge_type"],
                entry["weight_delta"],
            )
        elif op == OpType.BATCH_SCALE_WEIGHTS.value:
            self._memtable.batch_scale_weights(entry["factor"])

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    @property
    def memtable_count(self) -> int:
        return self._memtable.item_count
