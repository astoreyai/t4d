"""
T4DX Storage Engine Metrics Visualization for T4DM.

Visualizes T4DX embedded storage engine health:
- Storage overview (memtable size, segment count, total items)
- Compaction timeline (flush, NREM, REM, PRUNE events)
- Write amplification ratio

Author: Claude Opus 4.5
Date: 2026-02-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CompactionType(Enum):
    """Types of compaction events."""
    FLUSH = "flush"
    NREM = "nrem"
    REM = "rem"
    PRUNE = "prune"


@dataclass
class T4DXSnapshot:
    """Snapshot of T4DX storage engine state."""

    memtable_size: int
    segment_count: int
    total_items: int
    total_edges: int
    wal_size: int
    last_flush_time: float
    last_compact_time: float
    tombstone_count: int
    timestamp: float = 0.0


@dataclass
class CompactionEvent:
    """Record of a compaction event."""

    timestamp: float
    compaction_type: CompactionType
    items_processed: int = 0
    bytes_written: int = 0
    duration_s: float = 0.0


class T4DXMetricsVisualizer:
    """
    Visualizes T4DX storage engine health metrics.

    Tracks memtable size, segment count, total items, compaction events,
    and write amplification over time.
    """

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self._snapshots: list[T4DXSnapshot] = []
        self._compaction_events: list[CompactionEvent] = []
        self._logical_bytes_written: int = 0
        self._physical_bytes_written: int = 0
        logger.info("T4DXMetricsVisualizer initialized")

    def record_snapshot(self, snapshot: T4DXSnapshot) -> None:
        """Append a snapshot to history."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

    def record_compaction(self, event: CompactionEvent) -> None:
        """Record a compaction event."""
        self._compaction_events.append(event)
        self._physical_bytes_written += event.bytes_written
        if len(self._compaction_events) > self.window_size:
            self._compaction_events.pop(0)

    def record_write(self, logical_bytes: int) -> None:
        """Record logical bytes written (for write amplification)."""
        self._logical_bytes_written += logical_bytes

    def plot_storage_overview(self, fig: Any = None) -> Any:
        """Multi-panel: memtable size, segment count, total items over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if fig is None:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        else:
            axes = fig.subplots(3, 1, sharex=True)

        if not self._snapshots:
            for ax in axes:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
            return fig

        times = [s.timestamp for s in self._snapshots]
        memtable = [s.memtable_size for s in self._snapshots]
        segments = [s.segment_count for s in self._snapshots]
        items = [s.total_items for s in self._snapshots]

        axes[0].plot(times, memtable, color="steelblue", linewidth=1.5)
        axes[0].set_ylabel("MemTable Size")
        axes[0].set_title("T4DX Storage Overview")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times, segments, color="darkorange", linewidth=1.5)
        axes[1].set_ylabel("Segment Count")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(times, items, color="seagreen", linewidth=1.5)
        axes[2].set_ylabel("Total Items")
        axes[2].set_xlabel("Time")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_compaction_timeline(self, ax: Any = None) -> Any:
        """When flush/NREM/REM/PRUNE events occurred."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        if not self._compaction_events:
            ax.text(0.5, 0.5, "No compaction events", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        type_colors = {
            CompactionType.FLUSH: "steelblue",
            CompactionType.NREM: "seagreen",
            CompactionType.REM: "darkorange",
            CompactionType.PRUNE: "firebrick",
        }
        type_offsets = {
            CompactionType.FLUSH: 0,
            CompactionType.NREM: 1,
            CompactionType.REM: 2,
            CompactionType.PRUNE: 3,
        }

        for ctype in CompactionType:
            events = [e for e in self._compaction_events if e.compaction_type == ctype]
            if events:
                times = [e.timestamp for e in events]
                ax.eventplot(
                    [times],
                    lineoffsets=type_offsets[ctype],
                    colors=type_colors[ctype],
                    label=ctype.value.upper(),
                )

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["FLUSH", "NREM", "REM", "PRUNE"])
        ax.set_xlabel("Time")
        ax.set_title("Compaction Timeline")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")

        return ax

    def plot_write_amplification(self, ax: Any = None) -> Any:
        """Ratio of total bytes written vs logical bytes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if not self._compaction_events:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        # Cumulative physical bytes over time
        times = []
        cum_physical = []
        running = 0
        for e in self._compaction_events:
            running += e.bytes_written
            times.append(e.timestamp)
            cum_physical.append(running)

        ax.plot(times, cum_physical, color="firebrick", linewidth=1.5,
                label="Physical bytes written")

        if self._logical_bytes_written > 0:
            ax.axhline(y=self._logical_bytes_written, color="steelblue",
                       linestyle="--", label="Logical bytes written")
            ratio = cum_physical[-1] / self._logical_bytes_written if cum_physical else 0
            ax.set_title(f"Write Amplification (ratio: {ratio:.2f}x)")
        else:
            ax.set_title("Write Amplification")

        ax.set_xlabel("Time")
        ax.set_ylabel("Bytes Written")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def export_data(self) -> dict:
        """Export metrics data for external analysis."""
        return {
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "memtable_size": s.memtable_size,
                    "segment_count": s.segment_count,
                    "total_items": s.total_items,
                    "total_edges": s.total_edges,
                    "wal_size": s.wal_size,
                    "tombstone_count": s.tombstone_count,
                }
                for s in self._snapshots
            ],
            "compaction_events": [
                {
                    "timestamp": e.timestamp,
                    "type": e.compaction_type.value,
                    "items_processed": e.items_processed,
                    "bytes_written": e.bytes_written,
                    "duration_s": e.duration_s,
                }
                for e in self._compaction_events
            ],
            "write_amplification": {
                "logical_bytes": self._logical_bytes_written,
                "physical_bytes": self._physical_bytes_written,
                "ratio": (
                    self._physical_bytes_written / self._logical_bytes_written
                    if self._logical_bytes_written > 0
                    else 0.0
                ),
            },
            "meta": {
                "window_size": self.window_size,
                "total_snapshots": len(self._snapshots),
                "total_compaction_events": len(self._compaction_events),
            },
        }


__all__ = [
    "T4DXSnapshot",
    "T4DXMetricsVisualizer",
    "CompactionEvent",
    "CompactionType",
]
