"""Bloom filter for fast negative lookups in T4DX segments.

Uses hashlib-based hashing (no external deps).
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path


class BloomFilter:
    """Bit-array bloom filter using SHA-256 derived hashes."""

    def __init__(self, capacity: int = 10000, fp_rate: float = 0.01) -> None:
        self._capacity = capacity
        self._fp_rate = fp_rate
        # Optimal size: m = -n*ln(p) / (ln2)^2
        self._size = max(64, int(-capacity * math.log(fp_rate) / (math.log(2) ** 2)))
        # Optimal hash count: k = (m/n) * ln2
        self._num_hashes = max(1, int((self._size / capacity) * math.log(2)))
        self._bits = bytearray(math.ceil(self._size / 8))
        self._count = 0

    def _hashes(self, key: bytes) -> list[int]:
        """Generate hash positions using double-hashing from SHA-256."""
        digest = hashlib.sha256(key).digest()
        h1 = struct.unpack_from("<Q", digest, 0)[0]
        h2 = struct.unpack_from("<Q", digest, 8)[0]
        return [(h1 + i * h2) % self._size for i in range(self._num_hashes)]

    def add(self, key: bytes) -> None:
        """Add a key to the filter."""
        for pos in self._hashes(key):
            self._bits[pos >> 3] |= 1 << (pos & 7)
        self._count += 1

    def might_contain(self, key: bytes) -> bool:
        """Check if key might be in the set. False means definitely not present."""
        for pos in self._hashes(key):
            if not (self._bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def save(self, path: Path) -> None:
        """Save bloom filter to disk."""
        path = Path(path)
        header = struct.pack("<QQQ", self._size, self._num_hashes, self._count)
        path.write_bytes(header + bytes(self._bits))

    @classmethod
    def load(cls, path: Path) -> BloomFilter:
        """Load bloom filter from disk."""
        path = Path(path)
        data = path.read_bytes()
        size, num_hashes, count = struct.unpack_from("<QQQ", data, 0)
        bf = cls.__new__(cls)
        bf._size = size
        bf._num_hashes = num_hashes
        bf._count = count
        bf._capacity = count or 1
        bf._fp_rate = 0.01
        bf._bits = bytearray(data[24:])
        return bf

    def __len__(self) -> int:
        return self._count
