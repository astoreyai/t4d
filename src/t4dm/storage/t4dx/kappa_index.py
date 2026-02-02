"""Secondary index for kappa-range queries using sorted bisect."""

from __future__ import annotations

import bisect
import json
from pathlib import Path


class KappaIndex:
    """Maintains a sorted index of (kappa, item_id) pairs for O(log N) range queries."""

    def __init__(self) -> None:
        # Sorted list of (kappa, item_id_hex) tuples
        self._entries: list[tuple[float, str]] = []
        # Reverse lookup: item_id_hex -> kappa (for update/remove)
        self._id_to_kappa: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, item_id: bytes, kappa: float) -> None:
        """Add an item to the index."""
        hex_id = item_id.hex()
        if hex_id in self._id_to_kappa:
            self.remove(item_id)
        self._id_to_kappa[hex_id] = kappa
        bisect.insort(self._entries, (kappa, hex_id))

    def remove(self, item_id: bytes) -> None:
        """Remove an item from the index."""
        hex_id = item_id.hex()
        kappa = self._id_to_kappa.pop(hex_id, None)
        if kappa is None:
            return
        idx = bisect.bisect_left(self._entries, (kappa, hex_id))
        while idx < len(self._entries):
            if self._entries[idx] == (kappa, hex_id):
                self._entries.pop(idx)
                return
            if self._entries[idx][0] > kappa:
                break
            idx += 1

    def update(self, item_id: bytes, new_kappa: float) -> None:
        """Update the kappa value for an item."""
        self.remove(item_id)
        self.add(item_id, new_kappa)

    def query_range(self, kappa_min: float, kappa_max: float) -> list[bytes]:
        """Return item IDs with kappa in [kappa_min, kappa_max]. O(log N + result size)."""
        lo = bisect.bisect_left(self._entries, (kappa_min,))
        hi = bisect.bisect_right(self._entries, (kappa_max, "\xff" * 64))
        return [bytes.fromhex(e[1]) for e in self._entries[lo:hi]]

    def query_above(self, threshold: float) -> list[bytes]:
        """Return item IDs with kappa >= threshold."""
        lo = bisect.bisect_left(self._entries, (threshold,))
        return [bytes.fromhex(e[1]) for e in self._entries[lo:]]

    def query_below(self, threshold: float) -> list[bytes]:
        """Return item IDs with kappa < threshold."""
        hi = bisect.bisect_left(self._entries, (threshold,))
        return [bytes.fromhex(e[1]) for e in self._entries[:hi]]

    def save(self, path: Path) -> None:
        """Persist index to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"entries": [[k, h] for k, h in self._entries]}
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

    def load(self, path: Path) -> None:
        """Load index from JSON."""
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self._entries = [(e[0], e[1]) for e in data.get("entries", [])]
        self._id_to_kappa = {h: k for k, h in self._entries}
