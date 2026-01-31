"""On-disk segment: SegmentBuilder writes, SegmentReader reads.

Each segment directory contains:
  - vectors.npy   (N × D float32 matrix, mmap-able)
  - items.json    (list of ItemRecord dicts, vectors excluded)
  - edges.json    (list of EdgeRecord dicts)
  - manifest.json (SegmentMetadata dict + id_list ordering)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ww.storage.t4dx.types import EdgeRecord, ItemRecord, SegmentMetadata


class SegmentBuilder:
    """Writes a new immutable segment to disk."""

    def __init__(self, segment_dir: Path, segment_id: int, level: int = 0) -> None:
        self._dir = segment_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._segment_id = segment_id
        self._level = level

    def write(
        self,
        items: list[ItemRecord],
        edges: list[EdgeRecord],
    ) -> SegmentMetadata:
        if not items:
            raise ValueError("Cannot write empty segment")

        dim = len(items[0].vector) if items[0].vector else 0
        vectors = np.zeros((len(items), dim), dtype=np.float32)
        id_list: list[str] = []
        items_data: list[dict[str, Any]] = []

        for i, rec in enumerate(items):
            if rec.vector:
                vectors[i] = rec.vector
            id_list.append(rec.id.hex())
            d = rec.to_dict()
            d.pop("vector", None)  # stored in .npy
            items_data.append(d)

        np.save(str(self._dir / "vectors.npy"), vectors)

        with open(self._dir / "items.json", "w") as f:
            json.dump(items_data, f, separators=(",", ":"))

        edges_data = [e.to_dict() for e in edges]
        with open(self._dir / "edges.json", "w") as f:
            json.dump(edges_data, f, separators=(",", ":"))

        times = [r.event_time for r in items]
        kappas = [r.kappa for r in items]

        import time as _time

        meta = SegmentMetadata(
            segment_id=self._segment_id,
            item_count=len(items),
            edge_count=len(edges),
            time_min=min(times),
            time_max=max(times),
            kappa_min=min(kappas),
            kappa_max=max(kappas),
            created_at=_time.time(),
            level=self._level,
        )

        manifest = {**meta.to_dict(), "id_list": id_list, "dim": dim}
        with open(self._dir / "manifest.json", "w") as f:
            json.dump(manifest, f, separators=(",", ":"))

        return meta


class SegmentReader:
    """Read-only view over an on-disk segment."""

    def __init__(self, segment_dir: Path) -> None:
        self._dir = segment_dir
        self._vectors: np.ndarray | None = None
        self._items: list[dict[str, Any]] | None = None
        self._edges: list[dict[str, Any]] | None = None
        self._manifest: dict[str, Any] | None = None
        self._id_index: dict[str, int] | None = None

    # --- lazy loading ---

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            with open(self._dir / "manifest.json") as f:
                self._manifest = json.load(f)
        return self._manifest

    @property
    def metadata(self) -> SegmentMetadata:
        return SegmentMetadata.from_dict(self.manifest)

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = np.load(
                str(self._dir / "vectors.npy"), mmap_mode="r"
            )
        return self._vectors

    @property
    def items(self) -> list[dict[str, Any]]:
        if self._items is None:
            with open(self._dir / "items.json") as f:
                self._items = json.load(f)
        return self._items

    @property
    def edges(self) -> list[dict[str, Any]]:
        if self._edges is None:
            with open(self._dir / "edges.json") as f:
                self._edges = json.load(f)
        return self._edges

    @property
    def id_index(self) -> dict[str, int]:
        """Map hex-id -> row index for O(1) lookup."""
        if self._id_index is None:
            self._id_index = {
                h: i for i, h in enumerate(self.manifest["id_list"])
            }
        return self._id_index

    # --- reads ---

    def get(self, item_id: bytes) -> ItemRecord | None:
        hex_id = item_id.hex()
        idx = self.id_index.get(hex_id)
        if idx is None:
            return None
        d = dict(self.items[idx])
        d["vector"] = self.vectors[idx].tolist()
        d["id"] = hex_id
        return ItemRecord.from_dict(d)

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
        """Brute-force cosine search over segment vectors."""
        vecs = self.vectors
        if len(vecs) == 0:
            return []

        # Build filter mask
        mask = np.ones(len(vecs), dtype=bool)
        items_list = self.items

        for i, d in enumerate(items_list):
            if time_min is not None and d["event_time"] < time_min:
                mask[i] = False
            elif time_max is not None and d["event_time"] > time_max:
                mask[i] = False
            elif kappa_min is not None and d["kappa"] < kappa_min:
                mask[i] = False
            elif kappa_max is not None and d["kappa"] > kappa_max:
                mask[i] = False
            elif item_type is not None and d["item_type"] != item_type:
                mask[i] = False

        filtered_idx = np.where(mask)[0]
        if len(filtered_idx) == 0:
            return []

        filtered_vecs = np.array(vecs[filtered_idx], dtype=np.float32)
        q = query.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        norms = np.linalg.norm(filtered_vecs, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        sims = (filtered_vecs @ q) / (norms * q_norm)

        top_k = min(k, len(filtered_idx))
        if top_k >= len(filtered_idx):
            local_indices = np.argsort(-sims)
        else:
            local_indices = np.argpartition(-sims, top_k)[:top_k]
            local_indices = local_indices[np.argsort(-sims[local_indices])]

        id_list = self.manifest["id_list"]
        results = []
        for li in local_indices:
            orig_idx = filtered_idx[li]
            item_id_bytes = bytes.fromhex(id_list[orig_idx])
            results.append((item_id_bytes, float(sims[li])))
        return results

    def scan(
        self,
        time_min: float | None = None,
        time_max: float | None = None,
        kappa_min: float | None = None,
        kappa_max: float | None = None,
        item_type: str | None = None,
    ) -> list[ItemRecord]:
        """Scan items matching filters."""
        results = []
        id_list = self.manifest["id_list"]
        for i, d in enumerate(self.items):
            if time_min is not None and d["event_time"] < time_min:
                continue
            if time_max is not None and d["event_time"] > time_max:
                continue
            if kappa_min is not None and d["kappa"] < kappa_min:
                continue
            if kappa_max is not None and d["kappa"] > kappa_max:
                continue
            if item_type is not None and d["item_type"] != item_type:
                continue
            dd = dict(d)
            dd["vector"] = self.vectors[i].tolist()
            dd["id"] = id_list[i]
            results.append(ItemRecord.from_dict(dd))
        return results

    def traverse(
        self,
        node_id: bytes,
        edge_type: str | None = None,
        direction: str = "both",
    ) -> list[EdgeRecord]:
        hex_id = node_id.hex()
        results = []
        for d in self.edges:
            if edge_type and d["edge_type"] != edge_type:
                continue
            if direction == "out" and d["source_id"] == hex_id:
                results.append(EdgeRecord.from_dict(d))
            elif direction == "in" and d["target_id"] == hex_id:
                results.append(EdgeRecord.from_dict(d))
            elif direction == "both" and (
                d["source_id"] == hex_id or d["target_id"] == hex_id
            ):
                results.append(EdgeRecord.from_dict(d))
        return results
