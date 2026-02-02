"""Tests for T4DX metrics visualization."""

import pytest
import numpy as np

from t4dm.visualization.t4dx_metrics import (
    T4DXMetricsVisualizer,
    T4DXSnapshot,
    CompactionEvent,
    CompactionType,
)


class TestT4DXSnapshot:
    """Test T4DXSnapshot dataclass."""

    def test_create_snapshot(self):
        snapshot = T4DXSnapshot(
            memtable_size=100,
            segment_count=5,
            total_items=1000,
            total_edges=500,
            wal_size=2048,
            last_flush_time=1.0,
            last_compact_time=2.0,
            tombstone_count=10,
            timestamp=3.0,
        )
        assert snapshot.memtable_size == 100
        assert snapshot.total_items == 1000


class TestT4DXMetricsVisualizer:
    """Test T4DXMetricsVisualizer class."""

    def _make_snapshot(self, ts=0.0, items=100):
        return T4DXSnapshot(
            memtable_size=50,
            segment_count=3,
            total_items=items,
            total_edges=20,
            wal_size=1024,
            last_flush_time=ts - 1,
            last_compact_time=ts - 2,
            tombstone_count=5,
            timestamp=ts,
        )

    def test_initialization(self):
        viz = T4DXMetricsVisualizer()
        assert viz.window_size == 500

    def test_record_snapshot(self):
        viz = T4DXMetricsVisualizer()
        viz.record_snapshot(self._make_snapshot())
        assert len(viz._snapshots) == 1

    def test_window_size_limit(self):
        viz = T4DXMetricsVisualizer(window_size=5)
        for i in range(10):
            viz.record_snapshot(self._make_snapshot(ts=float(i)))
        assert len(viz._snapshots) == 5

    def test_plot_storage_overview(self):
        viz = T4DXMetricsVisualizer()
        for i in range(10):
            viz.record_snapshot(self._make_snapshot(ts=float(i), items=100 + i * 10))
        result = viz.plot_storage_overview()
        assert result is not None

    def test_plot_storage_overview_empty(self):
        viz = T4DXMetricsVisualizer()
        result = viz.plot_storage_overview()
        assert result is not None

    def test_plot_compaction_timeline(self):
        viz = T4DXMetricsVisualizer()
        for i, ctype in enumerate([CompactionType.FLUSH, CompactionType.NREM, CompactionType.REM]):
            viz.record_compaction(CompactionEvent(
                timestamp=float(i), compaction_type=ctype,
                items_processed=10, bytes_written=1024,
            ))
        result = viz.plot_compaction_timeline()
        assert result is not None

    def test_plot_compaction_timeline_empty(self):
        viz = T4DXMetricsVisualizer()
        result = viz.plot_compaction_timeline()
        assert result is not None

    def test_plot_write_amplification(self):
        viz = T4DXMetricsVisualizer()
        viz.record_write(1000)
        for i in range(5):
            viz.record_compaction(CompactionEvent(
                timestamp=float(i), compaction_type=CompactionType.FLUSH,
                bytes_written=500,
            ))
        result = viz.plot_write_amplification()
        assert result is not None

    def test_plot_write_amplification_empty(self):
        viz = T4DXMetricsVisualizer()
        result = viz.plot_write_amplification()
        assert result is not None

    def test_export_data(self):
        viz = T4DXMetricsVisualizer()
        viz.record_snapshot(self._make_snapshot(ts=1.0))
        viz.record_compaction(CompactionEvent(
            timestamp=1.0, compaction_type=CompactionType.FLUSH,
            bytes_written=512,
        ))
        viz.record_write(256)
        data = viz.export_data()
        assert isinstance(data, dict)
        assert "snapshots" in data
        assert "compaction_events" in data
        assert "write_amplification" in data
        assert len(data["snapshots"]) == 1
        assert len(data["compaction_events"]) == 1

    def test_export_data_empty(self):
        viz = T4DXMetricsVisualizer()
        data = viz.export_data()
        assert isinstance(data, dict)
        assert len(data["snapshots"]) == 0
