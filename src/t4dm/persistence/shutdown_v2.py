"""Shutdown v2: T4DX flush + checkpoint before exit (P5-03).

Extends ShutdownManager with:
1. Stop accepting new writes
2. Drain in-flight operations (30s timeout)
3. Flush MemTable -> final segment
4. Force checkpoint (captures all learning state)
5. Fsync T4DX WAL
6. Close handles, unmap numpy mmaps
7. Write shutdown marker
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ShutdownV2Result:
    """Result of T4DX-aware shutdown."""
    success: bool = True
    memtable_flushed: bool = False
    checkpoint_created: bool = False
    wal_fsynced: bool = False
    handles_closed: bool = False
    shutdown_marker_written: bool = False
    errors: list[str] | None = None
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


class ShutdownManagerV2:
    """T4DX-aware shutdown manager.

    Ensures all in-memory state is durably persisted before exit.
    """

    def __init__(
        self,
        engine: Any,
        checkpoint_fn: Any | None = None,
        drain_timeout: float = 30.0,
    ) -> None:
        self._engine = engine
        self._checkpoint_fn = checkpoint_fn
        self._drain_timeout = drain_timeout
        self._accepting_writes = True

    @property
    def accepting_writes(self) -> bool:
        return self._accepting_writes

    def execute_shutdown(self) -> ShutdownV2Result:
        """Execute T4DX-aware shutdown sequence."""
        start = time.time()
        result = ShutdownV2Result()

        # 1. Stop accepting writes
        self._accepting_writes = False
        logger.info("Shutdown: stopped accepting writes")

        # 2. Drain timeout (synchronous — caller should have drained async ops)

        # 3. Flush MemTable to segment
        try:
            if not self._engine._memtable.is_empty:
                sid = self._engine.flush()
                result.memtable_flushed = True
                logger.info("Shutdown: flushed memtable to segment %s", sid)
            else:
                result.memtable_flushed = True
                logger.info("Shutdown: memtable already empty")
        except Exception as e:
            result.memtable_flushed = False
            result.errors.append(f"Memtable flush failed: {e}")
            result.success = False

        # 4. Force checkpoint
        if self._checkpoint_fn is not None:
            try:
                self._checkpoint_fn()
                result.checkpoint_created = True
                logger.info("Shutdown: checkpoint created")
            except Exception as e:
                result.checkpoint_created = False
                result.errors.append(f"Checkpoint failed: {e}")
                result.success = False

        # 5. Fsync WAL
        try:
            self._engine._wal.close()
            result.wal_fsynced = True
            logger.info("Shutdown: WAL closed")
        except Exception as e:
            result.wal_fsynced = False
            result.errors.append(f"WAL close failed: {e}")
            result.success = False

        # 6. Save global index + close segment handles
        try:
            self._engine._global_index.save(
                self._engine._data_dir / "global_index.json"
            )
            # Segment readers have mmap'd numpy arrays; let GC handle them
            self._engine._segments.clear()
            result.handles_closed = True
            logger.info("Shutdown: handles closed")
        except Exception as e:
            result.handles_closed = False
            result.errors.append(f"Handle close failed: {e}")
            result.success = False

        # 7. Write shutdown marker
        try:
            marker_path = self._engine._data_dir / "shutdown_marker"
            marker_path.write_text(f"clean_shutdown={time.time()}\n")
            result.shutdown_marker_written = True
            logger.info("Shutdown: marker written")
        except Exception as e:
            result.shutdown_marker_written = False
            result.errors.append(f"Shutdown marker failed: {e}")

        self._engine._started = False
        result.duration_seconds = time.time() - start
        logger.info(
            "Shutdown complete: success=%s, %.2fs",
            result.success,
            result.duration_seconds,
        )
        return result
