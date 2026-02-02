"""Tests for persistence_state visualization module, including T4DX additions."""

import matplotlib
matplotlib.use("Agg")

import pytest
from datetime import datetime
from pathlib import Path

from t4dm.visualization.persistence_state import (
    CheckpointInfo,
    PersistenceMetrics,
    PersistenceVisualizer,
    T4DXPersistenceInfo,
    WALSegmentInfo,
    plot_checkpoint_history,
    plot_durability_dashboard,
    plot_t4dx_state,
    plot_wal_timeline,
)


class TestT4DXPersistenceInfo:
    def test_defaults(self):
        info = T4DXPersistenceInfo()
        assert info.memtable_items == 0
        assert info.segment_count == 0
        assert info.kappa_index_size == 0

    def test_custom_values(self):
        info = T4DXPersistenceInfo(
            memtable_items=100,
            segment_count=5,
            wal_entries=50,
            last_flush_lsn=42,
            kappa_index_size=200,
        )
        assert info.memtable_items == 100
        assert info.last_flush_lsn == 42


class TestPersistenceVisualizerT4DX:
    def _make_visualizer_with_t4dx(self) -> PersistenceVisualizer:
        viz = PersistenceVisualizer()
        viz.record_t4dx_state(T4DXPersistenceInfo(
            memtable_items=50, segment_count=3, wal_entries=20,
            last_flush_lsn=100, kappa_index_size=150,
        ))
        return viz

    def test_record_t4dx_state(self):
        viz = self._make_visualizer_with_t4dx()
        assert viz.current_t4dx is not None
        assert viz.current_t4dx.memtable_items == 50

    def test_current_t4dx_none(self):
        viz = PersistenceVisualizer()
        assert viz.current_t4dx is None

    def test_t4dx_max_history(self):
        viz = PersistenceVisualizer(max_history=2)
        for i in range(5):
            viz.record_t4dx_state(T4DXPersistenceInfo(memtable_items=i))
        assert len(viz.t4dx_history) == 2
        assert viz.current_t4dx.memtable_items == 4

    def test_plot_t4dx_state_no_data(self):
        viz = PersistenceVisualizer()
        ax = viz.plot_t4dx_state()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_t4dx_state_single(self):
        viz = self._make_visualizer_with_t4dx()
        ax = viz.plot_t4dx_state()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_t4dx_state_timeseries(self):
        viz = PersistenceVisualizer()
        for i in range(5):
            viz.record_t4dx_state(T4DXPersistenceInfo(
                memtable_items=i * 10, segment_count=i, wal_entries=i * 5,
            ))
        ax = viz.plot_t4dx_state()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_t4dx_state_standalone(self):
        viz = self._make_visualizer_with_t4dx()
        ax = plot_t4dx_state(viz)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_dashboard_with_t4dx(self):
        viz = PersistenceVisualizer()
        viz.update_metrics(PersistenceMetrics(
            current_lsn=100, checkpoint_lsn=80,
            operations_since_checkpoint=20,
            wal_total_size_bytes=1024 * 1024,
            recovery_mode="normal",
        ))
        viz.record_t4dx_state(T4DXPersistenceInfo(
            memtable_items=50, segment_count=3, wal_entries=20,
            last_flush_lsn=100, kappa_index_size=150,
        ))
        fig = plot_durability_dashboard(viz, interactive=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_dashboard_without_t4dx(self):
        viz = PersistenceVisualizer()
        viz.update_metrics(PersistenceMetrics(
            current_lsn=100, recovery_mode="normal",
        ))
        fig = plot_durability_dashboard(viz, interactive=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")


class TestExistingPersistenceFunctionality:
    """Verify existing functionality still works."""

    def test_wal_segment_recording(self):
        viz = PersistenceVisualizer()
        viz.record_wal_segment(WALSegmentInfo(
            segment_number=1, path=Path("/tmp/wal.001"),
            size_bytes=1024, min_lsn=0, max_lsn=10,
            created_at=datetime.now(), entry_count=10,
        ))
        assert len(viz.wal_segments) == 1

    def test_checkpoint_recording(self):
        viz = PersistenceVisualizer()
        viz.record_checkpoint(CheckpointInfo(
            lsn=100, timestamp=datetime.now(),
            size_bytes=2048, components=["memory", "index"],
        ))
        assert len(viz.checkpoints) == 1

    def test_metrics_update(self):
        viz = PersistenceVisualizer()
        viz.update_metrics(PersistenceMetrics(current_lsn=42))
        assert viz.current_metrics.current_lsn == 42
