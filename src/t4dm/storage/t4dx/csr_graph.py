"""Compressed Sparse Row graph for fast directed edge traversal."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any

import numpy as np

from t4dm.storage.t4dx.types import EdgeRecord


class CSRGraph:
    """CSR representation of directed edges for O(degree) neighbor lookup.

    Stores outgoing edges in CSR format. For incoming edge queries, a
    transposed CSR is built alongside.
    """

    def __init__(self) -> None:
        self._node_ids: list[bytes] = []  # sorted unique node IDs
        self._node_index: dict[bytes, int] = {}  # node_id -> position in node_ids
        # Outgoing CSR
        self._out_indptr: np.ndarray = np.zeros(1, dtype=np.int64)
        self._out_indices: np.ndarray = np.zeros(0, dtype=np.int64)
        self._out_edges: list[EdgeRecord] = []
        # Incoming CSR
        self._in_indptr: np.ndarray = np.zeros(1, dtype=np.int64)
        self._in_indices: np.ndarray = np.zeros(0, dtype=np.int64)
        self._in_edges: list[EdgeRecord] = []

    @classmethod
    def from_edges(cls, edges: list[EdgeRecord]) -> CSRGraph:
        """Build a CSR graph from an edge list."""
        g = cls()
        if not edges:
            return g

        # Collect unique node IDs
        node_set: set[bytes] = set()
        for e in edges:
            node_set.add(e.source_id)
            node_set.add(e.target_id)
        g._node_ids = sorted(node_set)
        g._node_index = {nid: i for i, nid in enumerate(g._node_ids)}
        n = len(g._node_ids)

        # Build outgoing CSR
        g._out_edges, g._out_indptr, g._out_indices = _build_csr(
            edges, n, g._node_index, key_fn=lambda e: e.source_id, col_fn=lambda e: e.target_id,
        )

        # Build incoming CSR
        g._in_edges, g._in_indptr, g._in_indices = _build_csr(
            edges, n, g._node_index, key_fn=lambda e: e.target_id, col_fn=lambda e: e.source_id,
        )

        return g

    def neighbors(
        self, node_id: bytes, direction: str = "outgoing",
    ) -> list[tuple[bytes, EdgeRecord]]:
        """Return neighbors and their edge records."""
        idx = self._node_index.get(node_id)
        if idx is None:
            return []

        results: list[tuple[bytes, EdgeRecord]] = []
        if direction in ("outgoing", "both"):
            start, end = int(self._out_indptr[idx]), int(self._out_indptr[idx + 1])
            for j in range(start, end):
                col = int(self._out_indices[j])
                results.append((self._node_ids[col], self._out_edges[j]))
        if direction in ("incoming", "both"):
            start, end = int(self._in_indptr[idx]), int(self._in_indptr[idx + 1])
            for j in range(start, end):
                col = int(self._in_indices[j])
                results.append((self._node_ids[col], self._in_edges[j]))
        return results

    def get_edge(self, source: bytes, target: bytes) -> EdgeRecord | None:
        """Lookup a specific directed edge."""
        src_idx = self._node_index.get(source)
        tgt_idx = self._node_index.get(target)
        if src_idx is None or tgt_idx is None:
            return None
        start, end = int(self._out_indptr[src_idx]), int(self._out_indptr[src_idx + 1])
        for j in range(start, end):
            if int(self._out_indices[j]) == tgt_idx:
                return self._out_edges[j]
        return None

    def save(self, path: Path) -> None:
        """Persist CSR arrays to an npz file + edges JSON sidecar."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            node_ids=np.array([nid.hex() for nid in self._node_ids], dtype="U32"),
            out_indptr=self._out_indptr,
            out_indices=self._out_indices,
            in_indptr=self._in_indptr,
            in_indices=self._in_indices,
        )
        # Save edge data as JSON sidecar
        edges_path = path.with_suffix(".edges.json")
        out_edges_data = [e.to_dict() for e in self._out_edges]
        in_edges_data = [e.to_dict() for e in self._in_edges]
        with open(edges_path, "w") as f:
            json.dump({"out": out_edges_data, "in": in_edges_data}, f, separators=(",", ":"))

    @classmethod
    def load(cls, path: Path) -> CSRGraph:
        """Load a CSR graph from disk."""
        import json

        g = cls()
        data = np.load(str(path), allow_pickle=False)
        node_hex = data["node_ids"]
        g._node_ids = [bytes.fromhex(str(h)) for h in node_hex]
        g._node_index = {nid: i for i, nid in enumerate(g._node_ids)}
        g._out_indptr = data["out_indptr"].astype(np.int64)
        g._out_indices = data["out_indices"].astype(np.int64)
        g._in_indptr = data["in_indptr"].astype(np.int64)
        g._in_indices = data["in_indices"].astype(np.int64)

        edges_path = path.with_suffix(".edges.json")
        with open(edges_path) as f:
            edges_data = json.load(f)
        g._out_edges = [EdgeRecord.from_dict(d) for d in edges_data["out"]]
        g._in_edges = [EdgeRecord.from_dict(d) for d in edges_data["in"]]
        return g

    def __len__(self) -> int:
        return len(self._out_edges)


def _build_csr(
    edges: list[EdgeRecord],
    n: int,
    node_index: dict[bytes, int],
    key_fn: Any,
    col_fn: Any,
) -> tuple[list[EdgeRecord], np.ndarray, np.ndarray]:
    """Build CSR arrays for one direction."""
    # Sort edges by row key
    sorted_edges = sorted(edges, key=lambda e: node_index[key_fn(e)])
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.zeros(len(sorted_edges), dtype=np.int64)
    for i, e in enumerate(sorted_edges):
        row = node_index[key_fn(e)]
        col = node_index[col_fn(e)]
        indptr[row + 1] += 1
        indices[i] = col
    # Cumulative sum for indptr
    np.cumsum(indptr, out=indptr)
    return sorted_edges, indptr, indices
