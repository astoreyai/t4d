"""Global in-memory index: id→segment mapping, tombstones, cross-edges, manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from t4dm.storage.t4dx.types import EdgeRecord


class GlobalIndex:
    """Tracks which segment owns each item, tombstones, and cross-segment edges."""

    def __init__(self) -> None:
        # id (hex) -> segment_id
        self.id_map: dict[str, int] = {}
        # set of tombstoned hex ids
        self.tombstones: set[str] = set()
        # segment_id -> manifest dict
        self.segments: dict[int, dict[str, Any]] = {}
        # cross-segment edges (edges spanning different segments)
        self.cross_edges: list[dict[str, Any]] = []
        # monotonic segment counter
        self._next_segment_id: int = 0

    def next_segment_id(self) -> int:
        sid = self._next_segment_id
        self._next_segment_id += 1
        return sid

    # --- item tracking ---

    def register_item(self, hex_id: str, segment_id: int) -> None:
        self.id_map[hex_id] = segment_id
        self.tombstones.discard(hex_id)

    def tombstone(self, hex_id: str) -> None:
        self.tombstones.add(hex_id)
        self.id_map.pop(hex_id, None)

    def is_live(self, hex_id: str) -> bool:
        return hex_id not in self.tombstones and hex_id in self.id_map

    def locate(self, hex_id: str) -> int | None:
        if hex_id in self.tombstones:
            return None
        return self.id_map.get(hex_id)

    # --- segment tracking ---

    def register_segment(self, segment_id: int, manifest: dict[str, Any]) -> None:
        self.segments[segment_id] = manifest
        if segment_id >= self._next_segment_id:
            self._next_segment_id = segment_id + 1

    def remove_segment(self, segment_id: int) -> None:
        self.segments.pop(segment_id, None)
        to_remove = [k for k, v in self.id_map.items() if v == segment_id]
        for k in to_remove:
            del self.id_map[k]

    # --- cross edges ---

    def add_cross_edge(self, edge: EdgeRecord) -> None:
        self.cross_edges.append(edge.to_dict())

    def get_cross_edges(
        self,
        node_id: bytes,
        edge_type: str | None = None,
        direction: str = "both",
    ) -> list[EdgeRecord]:
        hex_id = node_id.hex()
        results = []
        for d in self.cross_edges:
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

    # --- persistence ---

    def save(self, path: Path) -> None:
        data = {
            "id_map": self.id_map,
            "tombstones": list(self.tombstones),
            "segments": {str(k): v for k, v in self.segments.items()},
            "cross_edges": self.cross_edges,
            "next_segment_id": self._next_segment_id,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self.id_map = data.get("id_map", {})
        self.tombstones = set(data.get("tombstones", []))
        self.segments = {int(k): v for k, v in data.get("segments", {}).items()}
        self.cross_edges = data.get("cross_edges", [])
        self._next_segment_id = data.get("next_segment_id", 0)
