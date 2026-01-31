"""Write-Ahead Log for T4DX — JSON-lines format with fsync on flush."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any


class OpType(str, Enum):
    INSERT = "INSERT"
    DELETE = "DELETE"
    UPDATE_FIELDS = "UPDATE_FIELDS"
    INSERT_EDGE = "INSERT_EDGE"
    DELETE_EDGE = "DELETE_EDGE"
    UPDATE_EDGE_WEIGHT = "UPDATE_EDGE_WEIGHT"
    BATCH_SCALE_WEIGHTS = "BATCH_SCALE_WEIGHTS"


class WAL:
    """Append-only JSON-lines write-ahead log."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd: int | None = None

    # --- lifecycle ---

    def open(self) -> None:
        if self._fd is not None:
            return
        self._fd = os.open(
            str(self._path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    # --- write ---

    def append(self, op: OpType, payload: dict[str, Any]) -> None:
        """Append a single op to the WAL and fsync."""
        if self._fd is None:
            self.open()
        entry = {"op": op.value, **payload}
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        os.write(self._fd, line.encode())  # type: ignore[arg-type]
        os.fsync(self._fd)  # type: ignore[arg-type]

    # --- replay ---

    def replay(self) -> list[dict[str, Any]]:
        """Read all entries. Skips corrupted lines."""
        if not self._path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip corrupt
        return entries

    # --- truncate ---

    def truncate(self) -> None:
        """Clear the WAL after a successful flush."""
        self.close()
        if self._path.exists():
            self._path.unlink()
        self.open()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def size(self) -> int:
        return self._path.stat().st_size if self._path.exists() else 0
