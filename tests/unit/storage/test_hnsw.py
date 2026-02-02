"""Tests for HNSWIndex."""

import uuid

import numpy as np
import pytest

from t4dm.storage.t4dx.hnsw import HNSWIndex


def _rand_ids(n: int) -> list[bytes]:
    return [uuid.uuid4().bytes for _ in range(n)]


def _rand_vecs(n: int, dim: int = 32) -> np.ndarray:
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class TestHNSWIndex:
    def test_add_and_len(self):
        idx = HNSWIndex(dim=16, max_elements=100)
        assert len(idx) == 0
        ids = _rand_ids(10)
        vecs = _rand_vecs(10, 16)
        idx.add(ids, vecs)
        assert len(idx) == 10

    def test_search_returns_correct_nearest(self):
        dim = 32
        idx = HNSWIndex(dim=dim, max_elements=100)
        ids = _rand_ids(50)
        vecs = _rand_vecs(50, dim)
        idx.add(ids, vecs)

        # Query with first vector should return itself as closest
        result_ids, result_dists = idx.search(vecs[0], k=1)
        assert len(result_ids) == 1
        assert result_ids[0] == ids[0]
        assert result_dists[0] < 0.01  # distance ~ 0 for identical vector

    def test_search_k_larger_than_count(self):
        dim = 8
        idx = HNSWIndex(dim=dim, max_elements=100)
        ids = _rand_ids(3)
        vecs = _rand_vecs(3, dim)
        idx.add(ids, vecs)

        result_ids, result_dists = idx.search(vecs[0], k=10)
        assert len(result_ids) == 3

    def test_search_empty(self):
        idx = HNSWIndex(dim=8)
        ids, dists = idx.search(np.zeros(8, dtype=np.float32), k=5)
        assert ids == []
        assert dists == []

    def test_save_load(self, tmp_path):
        dim = 16
        idx = HNSWIndex(dim=dim, max_elements=100)
        ids = _rand_ids(20)
        vecs = _rand_vecs(20, dim)
        idx.add(ids, vecs)

        save_path = tmp_path / "hnsw.bin"
        idx.save(save_path)

        loaded = HNSWIndex.load(save_path, dim=dim)
        assert len(loaded) == 20

        # Search should return same results
        r_ids, r_dists = loaded.search(vecs[0], k=1)
        assert r_ids[0] == ids[0]

    def test_multiple_adds(self):
        dim = 8
        idx = HNSWIndex(dim=dim, max_elements=100)
        ids1 = _rand_ids(5)
        vecs1 = _rand_vecs(5, dim)
        idx.add(ids1, vecs1)

        ids2 = _rand_ids(5)
        vecs2 = _rand_vecs(5, dim)
        idx.add(ids2, vecs2)

        assert len(idx) == 10
        r_ids, _ = idx.search(vecs2[0], k=1)
        assert r_ids[0] == ids2[0]


class TestHNSWBruteForceFallback:
    """Test brute-force path directly."""

    def test_brute_force_search(self):
        dim = 16
        idx = HNSWIndex(dim=dim)
        # Force brute-force by disabling hnswlib
        idx._use_hnswlib = False
        idx._index = None

        ids = _rand_ids(20)
        vecs = _rand_vecs(20, dim)
        idx.add(ids, vecs)

        r_ids, r_dists = idx.search(vecs[0], k=3)
        assert len(r_ids) == 3
        assert r_ids[0] == ids[0]
        assert r_dists[0] < 0.01

    def test_brute_force_save_load(self, tmp_path):
        dim = 8
        idx = HNSWIndex(dim=dim)
        idx._use_hnswlib = False
        idx._index = None

        ids = _rand_ids(10)
        vecs = _rand_vecs(10, dim)
        idx.add(ids, vecs)

        path = tmp_path / "hnsw.bin"
        idx.save(path)

        loaded = HNSWIndex.load(path, dim=dim)
        assert len(loaded) == 10
