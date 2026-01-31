"""SnapshotAggregator — rolling windows over observation events."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from ww.t4dv.events import (
    ConsolidationEvent,
    NeuromodEvent,
    ObservationEvent,
    SpikeEvent,
    StorageEvent,
)


class SnapshotAggregator:
    """Maintains rolling windows (1s, 10s, 60s) of observation events.

    Call ``ingest(event)`` for each event.  ``get_dashboard_state()``
    returns a JSON-serialisable dict suitable for WebSocket streaming.
    """

    def __init__(self) -> None:
        self._spike_events: deque[tuple[float, SpikeEvent]] = deque(maxlen=10_000)
        self._storage_events: deque[tuple[float, StorageEvent]] = deque(maxlen=10_000)
        self._consolidation_events: deque[tuple[float, ConsolidationEvent]] = deque(maxlen=1_000)
        self._neuromod_events: deque[tuple[float, NeuromodEvent]] = deque(maxlen=5_000)
        self._last_state: dict[str, Any] = {}

    def ingest(self, event: ObservationEvent) -> None:
        now = time.monotonic()
        if isinstance(event, SpikeEvent):
            self._spike_events.append((now, event))
        elif isinstance(event, StorageEvent):
            self._storage_events.append((now, event))
        elif isinstance(event, ConsolidationEvent):
            self._consolidation_events.append((now, event))
        elif isinstance(event, NeuromodEvent):
            self._neuromod_events.append((now, event))

    def get_dashboard_state(self) -> dict[str, Any]:
        now = time.monotonic()
        state: dict[str, Any] = {
            "timestamp": time.time(),
            "spiking": self._aggregate_spikes(now),
            "storage": self._aggregate_storage(now),
            "consolidation": self._aggregate_consolidation(now),
            "neuromod": self._aggregate_neuromod(now),
        }
        self._last_state = state
        return state

    def _aggregate_spikes(self, now: float) -> dict[str, Any]:
        windows: dict[str, Any] = {}
        for label, secs in [("1s", 1), ("10s", 10), ("60s", 60)]:
            cutoff = now - secs
            recent = [(t, e) for t, e in self._spike_events if t >= cutoff]
            if recent:
                rates = [e.firing_rate for _, e in recent]
                windows[label] = {
                    "count": len(recent),
                    "mean_firing_rate": sum(rates) / len(rates),
                    "mean_pe": sum(e.prediction_error for _, e in recent) / len(recent),
                    "mean_goodness": sum(e.goodness for _, e in recent) / len(recent),
                }
            else:
                windows[label] = {"count": 0}

        # Per-block latest
        block_latest: dict[int, dict] = {}
        for _, e in reversed(list(self._spike_events)):
            if e.block_index not in block_latest:
                block_latest[e.block_index] = {
                    "firing_rate": e.firing_rate,
                    "pe": e.prediction_error,
                    "goodness": e.goodness,
                }
            if len(block_latest) >= 6:
                break

        return {"windows": windows, "blocks": block_latest}

    def _aggregate_storage(self, now: float) -> dict[str, Any]:
        windows: dict[str, Any] = {}
        for label, secs in [("1s", 1), ("10s", 10), ("60s", 60)]:
            cutoff = now - secs
            recent = [(t, e) for t, e in self._storage_events if t >= cutoff]
            op_counts: dict[str, int] = defaultdict(int)
            total_ms = 0.0
            for _, e in recent:
                op_counts[e.operation] += 1
                total_ms += e.duration_ms
            windows[label] = {
                "ops_total": len(recent),
                "ops_by_type": dict(op_counts),
                "total_duration_ms": round(total_ms, 2),
            }

        latest = self._storage_events[-1][1] if self._storage_events else None
        return {
            "windows": windows,
            "segment_count": latest.segment_count if latest else 0,
            "memtable_count": latest.memtable_count if latest else 0,
        }

    def _aggregate_consolidation(self, now: float) -> dict[str, Any]:
        recent = [(t, e) for t, e in self._consolidation_events if t >= now - 60]
        timeline = [
            {
                "phase": e.phase,
                "items": e.items_processed,
                "segments_merged": e.segments_merged,
                "t": round(now - t, 1),
            }
            for t, e in recent
        ]
        return {"recent_60s": timeline}

    def _aggregate_neuromod(self, now: float) -> dict[str, Any]:
        series: dict[str, list[float]] = {"da": [], "ne": [], "ach": [], "serotonin": []}
        cutoff = now - 60
        for t, e in self._neuromod_events:
            if t >= cutoff:
                series["da"].append(e.da)
                series["ne"].append(e.ne)
                series["ach"].append(e.ach)
                series["serotonin"].append(e.serotonin)
        latest = self._neuromod_events[-1][1] if self._neuromod_events else None
        return {
            "series": series,
            "latest": {
                "da": latest.da if latest else 0.5,
                "ne": latest.ne if latest else 0.5,
                "ach": latest.ach if latest else 0.5,
                "serotonin": latest.serotonin if latest else 0.5,
            },
        }
