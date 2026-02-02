"""
Tests for Persistence State Visualization module.

Tests WAL, checkpoint, and durability state visualization.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from t4dm.visualization.persistence_state import (
    WALSegmentInfo,
    CheckpointInfo,
    PersistenceMetrics,
    PersistenceVisualizer,
    plot_wal_timeline,
    plot_durability_dashboard,
    plot_checkpoint_history,
)


class TestWALSegmentInfo:
    """Tests for WALSegmentInfo dataclass."""

    def test_create_segment_info(self):
        """Test creating a WAL segment info."""
        now = datetime.now()
        segment = WALSegmentInfo(
            segment_number=1,
            path=Path("/wal/segment_001.wal"),
            size_bytes=1024 * 1024,
            min_lsn=100,
            max_lsn=200,
            created_at=now,
            entry_count=50,
        )

        assert segment.segment_number == 1
        assert segment.path == Path("/wal/segment_001.wal")
        assert segment.size_bytes == 1024 * 1024
        assert segment.min_lsn == 100
        assert segment.max_lsn == 200
        assert segment.entry_count == 50


class TestCheckpointInfo:
    """Tests for CheckpointInfo dataclass."""

    def test_create_checkpoint_info(self):
        """Test creating a checkpoint info."""
        now = datetime.now()
        checkpoint = CheckpointInfo(
            lsn=500,
            timestamp=now,
            size_bytes=10 * 1024 * 1024,
            components=["memory", "embeddings", "graphs"],
            duration_seconds=5.5,
        )

        assert checkpoint.lsn == 500
        assert checkpoint.size_bytes == 10 * 1024 * 1024
        assert len(checkpoint.components) == 3
        assert checkpoint.duration_seconds == 5.5

    def test_checkpoint_default_duration(self):
        """Test checkpoint with default duration."""
        checkpoint = CheckpointInfo(
            lsn=100,
            timestamp=datetime.now(),
            size_bytes=1024,
            components=["test"],
        )
        assert checkpoint.duration_seconds == 0.0


class TestPersistenceMetrics:
    """Tests for PersistenceMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating persistence metrics."""
        metrics = PersistenceMetrics(
            current_lsn=1000,
            checkpoint_lsn=900,
            operations_since_checkpoint=100,
            wal_segment_count=5,
            wal_total_size_bytes=50 * 1024 * 1024,
            checkpoint_count=10,
            last_checkpoint_age_seconds=300.0,
            recovery_mode="normal",
        )

        assert metrics.current_lsn == 1000
        assert metrics.checkpoint_lsn == 900
        assert metrics.operations_since_checkpoint == 100
        assert metrics.recovery_mode == "normal"

    def test_metrics_defaults(self):
        """Test metrics with default values."""
        metrics = PersistenceMetrics()
        assert metrics.current_lsn == 0
        assert metrics.checkpoint_lsn == 0
        assert metrics.operations_since_checkpoint == 0
        assert metrics.recovery_mode == "unknown"


class TestPersistenceVisualizer:
    """Tests for PersistenceVisualizer class."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return PersistenceVisualizer(max_history=100)

    def test_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer.max_history == 100
        assert visualizer.wal_segments == []
        assert visualizer.checkpoints == []
        assert visualizer.metrics_history == []
        assert visualizer.lsn_timeline == []

    def test_record_wal_segment(self, visualizer):
        """Test recording WAL segment."""
        segment = WALSegmentInfo(
            segment_number=1,
            path=Path("/wal/seg.wal"),
            size_bytes=1024,
            min_lsn=0,
            max_lsn=100,
            created_at=datetime.now(),
            entry_count=10,
        )
        visualizer.record_wal_segment(segment)
        assert len(visualizer.wal_segments) == 1

    def test_record_multiple_wal_segments(self, visualizer):
        """Test recording multiple WAL segments."""
        for i in range(10):
            segment = WALSegmentInfo(
                segment_number=i,
                path=Path(f"/wal/seg_{i}.wal"),
                size_bytes=1024 * (i + 1),
                min_lsn=i * 100,
                max_lsn=(i + 1) * 100,
                created_at=datetime.now(),
                entry_count=10,
            )
            visualizer.record_wal_segment(segment)
        assert len(visualizer.wal_segments) == 10

    def test_wal_segment_history_limit(self, visualizer):
        """Test WAL segment history limit."""
        for i in range(150):
            segment = WALSegmentInfo(
                segment_number=i,
                path=Path(f"/wal/seg_{i}.wal"),
                size_bytes=1024,
                min_lsn=i * 100,
                max_lsn=(i + 1) * 100,
                created_at=datetime.now(),
                entry_count=10,
            )
            visualizer.record_wal_segment(segment)
        assert len(visualizer.wal_segments) == 100

    def test_record_checkpoint(self, visualizer):
        """Test recording checkpoint."""
        checkpoint = CheckpointInfo(
            lsn=500,
            timestamp=datetime.now(),
            size_bytes=10 * 1024,
            components=["memory"],
            duration_seconds=1.0,
        )
        visualizer.record_checkpoint(checkpoint)
        assert len(visualizer.checkpoints) == 1

    def test_checkpoint_history_limit(self, visualizer):
        """Test checkpoint history limit."""
        for i in range(150):
            checkpoint = CheckpointInfo(
                lsn=i * 100,
                timestamp=datetime.now(),
                size_bytes=1024,
                components=["test"],
            )
            visualizer.record_checkpoint(checkpoint)
        assert len(visualizer.checkpoints) == 100

    def test_update_metrics(self, visualizer):
        """Test updating metrics."""
        metrics = PersistenceMetrics(
            current_lsn=1000,
            operations_since_checkpoint=50,
        )
        visualizer.update_metrics(metrics)
        assert len(visualizer.metrics_history) == 1
        assert len(visualizer.lsn_timeline) == 1

    def test_metrics_history_limit(self, visualizer):
        """Test metrics history limit."""
        for i in range(150):
            metrics = PersistenceMetrics(current_lsn=i * 100)
            visualizer.update_metrics(metrics)
        assert len(visualizer.metrics_history) == 100
        assert len(visualizer.lsn_timeline) == 100

    def test_current_metrics_empty(self, visualizer):
        """Test current_metrics when empty."""
        assert visualizer.current_metrics is None

    def test_current_metrics(self, visualizer):
        """Test getting current metrics."""
        metrics1 = PersistenceMetrics(current_lsn=100)
        metrics2 = PersistenceMetrics(current_lsn=200)
        visualizer.update_metrics(metrics1)
        visualizer.update_metrics(metrics2)

        current = visualizer.current_metrics
        assert current is not None
        assert current.current_lsn == 200

    def test_get_wal_size_over_time(self, visualizer):
        """Test getting WAL size over time."""
        for i in range(5):
            segment = WALSegmentInfo(
                segment_number=i,
                path=Path(f"/wal/seg_{i}.wal"),
                size_bytes=1024 * (i + 1),
                min_lsn=i * 100,
                max_lsn=(i + 1) * 100,
                created_at=datetime.now() + timedelta(seconds=i),
                entry_count=10,
            )
            visualizer.record_wal_segment(segment)

        timestamps, sizes = visualizer.get_wal_size_over_time()
        assert len(timestamps) == 5
        assert len(sizes) == 5
        assert sizes == [1024, 2048, 3072, 4096, 5120]

    def test_get_lsn_over_time(self, visualizer):
        """Test getting LSN over time."""
        for i in range(5):
            metrics = PersistenceMetrics(current_lsn=i * 100)
            visualizer.update_metrics(metrics)

        timestamps, lsns = visualizer.get_lsn_over_time()
        assert len(timestamps) == 5
        assert len(lsns) == 5
        assert lsns == [0, 100, 200, 300, 400]

    def test_get_checkpoint_timeline(self, visualizer):
        """Test getting checkpoint timeline."""
        for i in range(5):
            checkpoint = CheckpointInfo(
                lsn=i * 100,
                timestamp=datetime.now() + timedelta(seconds=i),
                size_bytes=1024,
                components=["test"],
            )
            visualizer.record_checkpoint(checkpoint)

        timestamps, lsns = visualizer.get_checkpoint_timeline()
        assert len(timestamps) == 5
        assert len(lsns) == 5
        assert lsns == [0, 100, 200, 300, 400]


class TestPersistenceVisualizerIntegration:
    """Integration tests for PersistenceVisualizer."""

    def test_full_workflow(self):
        """Test complete visualization workflow."""
        visualizer = PersistenceVisualizer(max_history=1000)

        # Record WAL segments
        for i in range(10):
            segment = WALSegmentInfo(
                segment_number=i,
                path=Path(f"/wal/seg_{i}.wal"),
                size_bytes=1024 * 1024 * (i + 1),
                min_lsn=i * 1000,
                max_lsn=(i + 1) * 1000,
                created_at=datetime.now() + timedelta(minutes=i),
                entry_count=100,
            )
            visualizer.record_wal_segment(segment)

        # Record checkpoints
        for i in range(3):
            checkpoint = CheckpointInfo(
                lsn=i * 3000,
                timestamp=datetime.now() + timedelta(minutes=i * 3),
                size_bytes=10 * 1024 * 1024,
                components=["memory", "embeddings"],
                duration_seconds=5.0 + i,
            )
            visualizer.record_checkpoint(checkpoint)

        # Record metrics
        for i in range(20):
            metrics = PersistenceMetrics(
                current_lsn=i * 500,
                checkpoint_lsn=(i // 5) * 2500,
                operations_since_checkpoint=i % 5 * 100,
                wal_segment_count=min(10, i // 2 + 1),
                wal_total_size_bytes=i * 1024 * 1024,
                last_checkpoint_age_seconds=float((i % 5) * 60),
                recovery_mode="normal",
            )
            visualizer.update_metrics(metrics)

        # Verify data
        assert len(visualizer.wal_segments) == 10
        assert len(visualizer.checkpoints) == 3
        assert len(visualizer.metrics_history) == 20

        # Verify timeline data
        timestamps, lsns = visualizer.get_lsn_over_time()
        assert len(timestamps) == 20


class TestPlotWalTimeline:
    """Tests for plot_wal_timeline function."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with WAL data."""
        visualizer = PersistenceVisualizer()
        for i in range(5):
            segment = WALSegmentInfo(
                segment_number=i,
                path=Path(f"/wal/seg_{i}.wal"),
                size_bytes=1024 * 1024,
                min_lsn=i * 100,
                max_lsn=(i + 1) * 100,
                created_at=datetime.now() + timedelta(seconds=i),
                entry_count=10,
            )
            visualizer.record_wal_segment(segment)
            metrics = PersistenceMetrics(current_lsn=i * 100)
            visualizer.update_metrics(metrics)
        return visualizer

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_wal_timeline_matplotlib(self, mock_tight, mock_subplots, populated_visualizer):
        """Test matplotlib WAL timeline."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        plot_wal_timeline(populated_visualizer, interactive=False)

        mock_subplots.assert_called_once()

    def test_wal_timeline_plotly(self, populated_visualizer):
        """Test plotly WAL timeline."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("Plotly not available")

        result = plot_wal_timeline(populated_visualizer, interactive=True)
        assert result is not None


class TestPlotDurabilityDashboard:
    """Tests for plot_durability_dashboard function."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with metrics."""
        visualizer = PersistenceVisualizer()
        metrics = PersistenceMetrics(
            current_lsn=10000,
            checkpoint_lsn=9500,
            operations_since_checkpoint=500,
            wal_segment_count=10,
            wal_total_size_bytes=100 * 1024 * 1024,
            last_checkpoint_age_seconds=120.0,
            recovery_mode="normal",
        )
        visualizer.update_metrics(metrics)
        return visualizer

    def test_dashboard_empty_visualizer(self):
        """Test dashboard with empty visualizer."""
        visualizer = PersistenceVisualizer()
        result = plot_durability_dashboard(visualizer, interactive=False)
        assert result is None

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_dashboard_matplotlib(self, mock_tight, mock_subplots, populated_visualizer):
        """Test matplotlib dashboard."""
        mock_fig = MagicMock()
        # Create properly shaped array with MagicMock axes
        mock_axes = MagicMock()
        mock_axes.__getitem__ = MagicMock(return_value=MagicMock())
        mock_subplots.return_value = (mock_fig, mock_axes)

        result = plot_durability_dashboard(populated_visualizer, interactive=False)

        mock_subplots.assert_called_once()

    def test_dashboard_plotly(self, populated_visualizer):
        """Test plotly dashboard."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("Plotly not available")

        result = plot_durability_dashboard(populated_visualizer, interactive=True)
        assert result is not None

    def test_dashboard_high_ops_color(self):
        """Test dashboard with high operations count (warning color)."""
        visualizer = PersistenceVisualizer()
        metrics = PersistenceMetrics(
            operations_since_checkpoint=6000,  # High - should be red
            last_checkpoint_age_seconds=700.0,  # High - should be red
        )
        visualizer.update_metrics(metrics)

        # Just verify it doesn't crash with these values
        try:
            import plotly.graph_objects as go
            result = plot_durability_dashboard(visualizer, interactive=True)
            assert result is not None
        except ImportError:
            pytest.skip("Plotly not available")


class TestPlotCheckpointHistory:
    """Tests for plot_checkpoint_history function."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with checkpoint data."""
        visualizer = PersistenceVisualizer()
        for i in range(5):
            checkpoint = CheckpointInfo(
                lsn=i * 1000,
                timestamp=datetime.now() + timedelta(minutes=i * 5),
                size_bytes=1024 * 1024 * (i + 1),
                components=["memory", "embeddings"],
                duration_seconds=1.0 + i * 0.5,
            )
            visualizer.record_checkpoint(checkpoint)
        return visualizer

    @patch("t4dm.visualization.persistence_state.logger")
    def test_checkpoint_history_empty(self, mock_logger):
        """Test checkpoint history with empty data."""
        visualizer = PersistenceVisualizer()
        result = plot_checkpoint_history(visualizer)
        assert result is None
        mock_logger.warning.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_checkpoint_history_matplotlib(self, mock_tight, mock_subplots, populated_visualizer):
        """Test matplotlib checkpoint history."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        result = plot_checkpoint_history(populated_visualizer, interactive=False)

        mock_subplots.assert_called_once()

    def test_checkpoint_history_plotly(self, populated_visualizer):
        """Test plotly checkpoint history."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("Plotly not available")

        result = plot_checkpoint_history(populated_visualizer, interactive=True)
        assert result is not None


class TestPersistenceStateEdgeCases:
    """Edge case tests for persistence state module."""

    def test_empty_visualizer_timeline(self):
        """Test getting timelines from empty visualizer."""
        visualizer = PersistenceVisualizer()

        timestamps, sizes = visualizer.get_wal_size_over_time()
        assert timestamps == []
        assert sizes == []

        timestamps, lsns = visualizer.get_lsn_over_time()
        assert timestamps == []
        assert lsns == []

        timestamps, lsns = visualizer.get_checkpoint_timeline()
        assert timestamps == []
        assert lsns == []

    def test_single_segment(self):
        """Test with single WAL segment."""
        visualizer = PersistenceVisualizer()
        segment = WALSegmentInfo(
            segment_number=0,
            path=Path("/wal/seg.wal"),
            size_bytes=1024,
            min_lsn=0,
            max_lsn=100,
            created_at=datetime.now(),
            entry_count=10,
        )
        visualizer.record_wal_segment(segment)

        timestamps, sizes = visualizer.get_wal_size_over_time()
        assert len(timestamps) == 1
        assert sizes == [1024]

    def test_single_checkpoint(self):
        """Test with single checkpoint."""
        visualizer = PersistenceVisualizer()
        checkpoint = CheckpointInfo(
            lsn=100,
            timestamp=datetime.now(),
            size_bytes=1024,
            components=["test"],
        )
        visualizer.record_checkpoint(checkpoint)

        timestamps, lsns = visualizer.get_checkpoint_timeline()
        assert len(timestamps) == 1
        assert lsns == [100]

    def test_very_small_history(self):
        """Test with very small max_history."""
        visualizer = PersistenceVisualizer(max_history=2)

        for i in range(10):
            segment = WALSegmentInfo(
                segment_number=i,
                path=Path(f"/wal/seg_{i}.wal"),
                size_bytes=1024,
                min_lsn=i * 100,
                max_lsn=(i + 1) * 100,
                created_at=datetime.now(),
                entry_count=10,
            )
            visualizer.record_wal_segment(segment)

        assert len(visualizer.wal_segments) == 2
        assert visualizer.wal_segments[0].segment_number == 8
        assert visualizer.wal_segments[1].segment_number == 9
