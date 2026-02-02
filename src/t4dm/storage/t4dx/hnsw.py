"""HNSW vector index for T4DX segments.

Uses hnswlib if available, otherwise falls back to brute-force numpy cosine search.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class HNSWIndex:
    """HNSW approximate nearest-neighbor index with hnswlib/brute-force fallback."""

    def __init__(
        self,
        dim: int,
        max_elements: int = 10000,
        ef_construction: int = 200,
        M: int = 16,
    ) -> None:
        self._dim = dim
        self._max_elements = max_elements
        self._ef_construction = ef_construction
        self._M = M
        self._ids: list[bytes] = []
        self._id_to_idx: dict[bytes, int] = {}
        self._vectors: np.ndarray | None = None
        self._index = None  # hnswlib index or None
        self._use_hnswlib = False

        try:
            import hnswlib

            self._index = hnswlib.Index(space="cosine", dim=dim)
            self._index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=M,
            )
            self._index.set_ef(max(ef_construction, 50))
            self._use_hnswlib = True
        except ImportError:
            pass

    def add(self, ids: list[bytes], vectors: np.ndarray) -> None:
        """Add vectors with corresponding byte IDs."""
        if len(ids) == 0:
            return
        vecs = vectors.astype(np.float32)
        start = len(self._ids)
        int_ids = list(range(start, start + len(ids)))

        for i, bid in enumerate(ids):
            self._id_to_idx[bid] = start + i
        self._ids.extend(ids)

        if self._vectors is None:
            self._vectors = vecs.copy()
        else:
            self._vectors = np.vstack([self._vectors, vecs])

        if self._use_hnswlib and self._index is not None:
            # Resize if needed
            needed = start + len(ids)
            if needed > self._max_elements:
                self._index.resize_index(needed)
                self._max_elements = needed
            self._index.add_items(vecs, np.array(int_ids, dtype=np.int64))

    def search(self, query: np.ndarray, k: int = 10) -> tuple[list[bytes], list[float]]:
        """Search for k nearest neighbors. Returns (ids, distances)."""
        if len(self._ids) == 0:
            return [], []

        k = min(k, len(self._ids))
        q = query.astype(np.float32).reshape(1, -1)

        if self._use_hnswlib and self._index is not None:
            labels, distances = self._index.knn_query(q, k=k)
            result_ids = [self._ids[int(idx)] for idx in labels[0]]
            # hnswlib cosine space returns 1 - cosine_sim
            result_dists = [float(d) for d in distances[0]]
            return result_ids, result_dists

        # Brute-force fallback
        return self._brute_force_search(q[0], k)

    def _brute_force_search(
        self, query: np.ndarray, k: int
    ) -> tuple[list[bytes], list[float]]:
        """Brute-force cosine distance search."""
        assert self._vectors is not None
        vecs = self._vectors
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return [], []
        norms = np.linalg.norm(vecs, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        sims = (vecs @ query) / (norms * q_norm)
        # Convert to distance: 1 - cosine_sim (matching hnswlib convention)
        dists = 1.0 - sims

        if k >= len(dists):
            indices = np.argsort(dists)
        else:
            indices = np.argpartition(dists, k)[:k]
            indices = indices[np.argsort(dists[indices])]

        result_ids = [self._ids[i] for i in indices]
        result_dists = [float(dists[i]) for i in indices]
        return result_ids, result_dists

    def save(self, path: Path) -> None:
        """Persist index to disk."""
        path = Path(path)
        if self._use_hnswlib and self._index is not None:
            self._index.save_index(str(path))
            # Save id mapping alongside
            meta_path = path.with_suffix(".ids.npy")
            id_array = np.array([bid.hex() for bid in self._ids], dtype="U32")
            np.save(str(meta_path), id_array)
        else:
            # Save brute-force data
            np.savez(
                str(path),
                vectors=self._vectors if self._vectors is not None else np.array([]),
                ids=np.array([bid.hex() for bid in self._ids], dtype="U32"),
            )

    @classmethod
    def load(cls, path: Path, dim: int) -> HNSWIndex:
        """Load index from disk."""
        path = Path(path)
        instance = cls(dim=dim)

        meta_path = path.with_suffix(".ids.npy")
        if meta_path.exists() and instance._use_hnswlib and instance._index is not None:
            # Load hnswlib index
            id_array = np.load(str(meta_path), allow_pickle=False)
            instance._ids = [bytes.fromhex(h) for h in id_array]
            instance._id_to_idx = {bid: i for i, bid in enumerate(instance._ids)}
            if len(instance._ids) > 0:
                instance._index.resize_index(max(len(instance._ids), instance._max_elements))
                instance._index.load_index(str(path), max_elements=len(instance._ids))
            return instance

        # Try npz fallback
        npz_path = path if path.suffix == ".npz" else Path(str(path) + ".npz")
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=False)
            ids_hex = data["ids"]
            vecs = data["vectors"]
            instance._ids = [bytes.fromhex(str(h)) for h in ids_hex]
            instance._id_to_idx = {bid: i for i, bid in enumerate(instance._ids)}
            if len(vecs.shape) == 2 and vecs.shape[0] > 0:
                instance._vectors = vecs.astype(np.float32)
                if instance._use_hnswlib and instance._index is not None:
                    int_ids = np.arange(len(instance._ids), dtype=np.int64)
                    instance._index.resize_index(max(len(instance._ids), instance._max_elements))
                    instance._index.add_items(instance._vectors, int_ids)
            return instance

        return instance

    def __len__(self) -> int:
        return len(self._ids)
