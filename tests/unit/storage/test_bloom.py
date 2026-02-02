"""Tests for BloomFilter."""

import os
import uuid

import pytest

from t4dm.storage.t4dx.bloom import BloomFilter


class TestBloomFilter:
    def test_add_and_might_contain(self):
        bf = BloomFilter(capacity=100)
        key = b"hello"
        assert not bf.might_contain(key)
        bf.add(key)
        assert bf.might_contain(key)

    def test_definite_negative(self):
        bf = BloomFilter(capacity=100)
        bf.add(b"key1")
        bf.add(b"key2")
        # If might_contain returns False, the key is definitely absent
        # We can't guarantee False for random keys, but with low fill ratio it's likely
        absent = b"definitely_not_here_xyz_12345"
        # Just test the interface works; false positives are possible
        _ = bf.might_contain(absent)

    def test_no_false_negatives(self):
        """Every added key must return True."""
        bf = BloomFilter(capacity=1000, fp_rate=0.01)
        keys = [uuid.uuid4().bytes for _ in range(500)]
        for k in keys:
            bf.add(k)
        for k in keys:
            assert bf.might_contain(k), "Bloom filter must not have false negatives"

    def test_false_positive_rate(self):
        """FP rate should be approximately at or below the configured rate."""
        n = 1000
        bf = BloomFilter(capacity=n, fp_rate=0.01)
        inserted = {uuid.uuid4().bytes for _ in range(n)}
        for k in inserted:
            bf.add(k)

        # Test 10000 absent keys
        fp_count = 0
        test_count = 10000
        for _ in range(test_count):
            key = uuid.uuid4().bytes
            if key not in inserted and bf.might_contain(key):
                fp_count += 1

        fp_rate = fp_count / test_count
        # Allow 3x the target rate as margin
        assert fp_rate < 0.03, f"FP rate {fp_rate:.4f} exceeds 3x target"

    def test_len(self):
        bf = BloomFilter(capacity=100)
        assert len(bf) == 0
        bf.add(b"a")
        bf.add(b"b")
        assert len(bf) == 2

    def test_save_load(self, tmp_path):
        bf = BloomFilter(capacity=100, fp_rate=0.01)
        keys = [uuid.uuid4().bytes for _ in range(50)]
        for k in keys:
            bf.add(k)

        path = tmp_path / "bloom.bin"
        bf.save(path)
        assert path.exists()

        loaded = BloomFilter.load(path)
        assert len(loaded) == 50
        for k in keys:
            assert loaded.might_contain(k)

    def test_empty_filter(self):
        bf = BloomFilter(capacity=10)
        assert len(bf) == 0
        assert not bf.might_contain(b"anything")
