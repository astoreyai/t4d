"""In-memory buffer (MemTable) for T4DX.

Holds unflushed items, edges, field overlays and edge deltas.
Provides brute-force numpy cosine search over buffered vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from t4dm.storage.t4dx.types import EdgeRecord, ItemRecord


@dataclass
class FieldOverlay:
    """Pending field update for an item."""

    item_id: bytes
    fields: dict[str, Any]


@dataclass
class EdgeDelta:
    """Pending edge weight delta."""

    source_id: bytes
    target_id: bytes
    edge_type: str
    weight_delta: float


class MemTable:
    """In-memory write buffer with brute-force vector search."""

    def __init__(self) -> None:
        self._items: dict[bytes, ItemRecord] = {}
        self._edges: list[EdgeRecord] = []
        self._field_overlays: list[FieldOverlay] = []
        self._edge_deltas: list[EdgeDelta] = []
        self._deleted_ids: set[bytes] = set()

    # --- properties ---

    @property
    def item_count(self) -> int:
        return len(self._items)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def is_empty(self) -> bool:
        return (
            not self._items
            and not self._edges
            and not self._field_overlays
            and not self._edge_deltas
            and not self._deleted_ids
        )

    # --- mutations ---

    def insert(self, record: ItemRecord) -> None:
        self._items[record.id] = record

    def delete(self, item_id: bytes) -> None:
        self._items.pop(item_id, None)
        self._deleted_ids.add(item_id)

    def update_fields(self, item_id: bytes, fields: dict[str, Any]) -> None:
        if item_id in self._items:
            rec = self._items[item_id]
            for k, v in fields.items():
                if hasattr(rec, k):
                    object.__setattr__(rec, k, v)
        else:
            self._field_overlays.append(FieldOverlay(item_id=item_id, fields=fields))

    def insert_edge(self, edge: EdgeRecord) -> None:
        self._edges.append(edge)

    def delete_edge(self, source_id: bytes, target_id: bytes, edge_type: str) -> None:
        self._edges = [
            e
            for e in self._edges
            if not (
                e.source_id == source_id
                and e.target_id == target_id
                and e.edge_type == edge_type
            )
        ]

    def update_edge_weight(
        self, source_id: bytes, target_id: bytes, edge_type: str, weight_delta: float
    ) -> None:
        # Try in-memory first
        for e in self._edges:
            if (
                e.source_id == source_id
                and e.target_id == target_id
                and e.edge_type == edge_type
            ):
                e.weight = max(0.0, min(1.0, e.weight + weight_delta))
                return
        self._edge_deltas.append(
            EdgeDelta(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight_delta=weight_delta,
            )
        )

    def batch_scale_weights(self, factor: float) -> None:
        for e in self._edges:
            e.weight = max(0.0, min(1.0, e.weight * factor))

    # --- reads ---

    def get(self, item_id: bytes) -> ItemRecord | None:
        if item_id in self._deleted_ids:
            return None
        return self._items.get(item_id)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        time_min: float | None = None,
        time_max: float | None = None,
        kappa_min: float | None = None,
        kappa_max: float | None = None,
        item_type: str | None = None,
    ) -> list[tuple[bytes, float]]:
        """Brute-force cosine similarity search. Returns (id, score) pairs."""
        candidates: list[ItemRecord] = []
        for rec in self._items.values():
            if rec.id in self._deleted_ids:
                continue
            if not rec.vector:
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
            candidates.append(rec)

        if not candidates:
            return []

        vecs = np.array([c.vector for c in candidates], dtype=np.float32)
        q = query.astype(np.float32)
        # cosine similarity
        norms = np.linalg.norm(vecs, axis=1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        norms = np.where(norms == 0, 1.0, norms)
        sims = (vecs @ q) / (norms * q_norm)

        top_k = min(k, len(candidates))
        if top_k >= len(candidates):
            indices = np.argsort(-sims)
        else:
            indices = np.argpartition(-sims, top_k)[:top_k]
            indices = indices[np.argsort(-sims[indices])]

        return [(candidates[i].id, float(sims[i])) for i in indices]

    def get_edges(
        self,
        node_id: bytes,
        edge_type: str | None = None,
        direction: str = "both",
    ) -> list[EdgeRecord]:
        results = []
        for e in self._edges:
            if edge_type and e.edge_type != edge_type:
                continue
            if direction == "out" and e.source_id == node_id:
                results.append(e)
            elif direction == "in" and e.target_id == node_id:
                results.append(e)
            elif direction == "both" and (
                e.source_id == node_id or e.target_id == node_id
            ):
                results.append(e)
        return results

    # --- flush ---

    def flush(self) -> tuple[
        dict[bytes, ItemRecord],
        list[EdgeRecord],
        list[FieldOverlay],
        list[EdgeDelta],
        set[bytes],
    ]:
        """Drain all buffered state and return it."""
        items = self._items
        edges = self._edges
        overlays = self._field_overlays
        deltas = self._edge_deltas
        deleted = self._deleted_ids

        self._items = {}
        self._edges = []
        self._field_overlays = []
        self._edge_deltas = []
        self._deleted_ids = set()

        return items, edges, overlays, deltas, deleted
