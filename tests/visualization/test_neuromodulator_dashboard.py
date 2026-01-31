"""Tests for neuromodulator state visualization module."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from t4dm.visualization.neuromodulator_state import (
    NeuromodulatorSnapshot,
    NeuromodulatorDashboard,
)


class TestNeuromodulatorSnapshot:
    """Tests for NeuromodulatorSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Create snapshot with all fields."""
        snapshot = NeuromodulatorSnapshot(
            timestamp=datetime.now(),
            dopamine_rpe=0.5,
            norepinephrine_gain=1.2,
            acetylcholine_mode="encoding",
            serotonin_mood=0.7,
            gaba_sparsity=0.3,
        )
        assert snapshot.dopamine_rpe == 0.5
        assert snapshot.norepinephrine_gain == 1.2
        assert snapshot.acetylcholine_mode == "encoding"
        assert snapshot.serotonin_mood == 0.7
        assert snapshot.gaba_sparsity == 0.3

    def test_snapshot_timestamp(self):
        """Snapshot preserves timestamp."""
        now = datetime.now()
        snapshot = NeuromodulatorSnapshot(
            timestamp=now,
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            gaba_sparsity=0.5,
        )
        assert snapshot.timestamp == now


class TestNeuromodulatorDashboard:
    """Tests for NeuromodulatorDashboard class."""

    @pytest.fixture
    def dashboard(self):
        """Create dashboard instance."""
        return NeuromodulatorDashboard(window_size=10)

    def test_initialization(self, dashboard):
        """Test initialization."""
        assert dashboard.window_size == 10
        assert len(dashboard._snapshots) == 0

    def test_initialization_defaults(self):
        """Test default initialization."""
        dashboard = NeuromodulatorDashboard()
        assert dashboard.window_size == 1000

    def test_record_state(self, dashboard):
        """Record state adds snapshot."""
        dashboard.record_state(
            dopamine_rpe=0.5,
            norepinephrine_gain=1.2,
            acetylcholine_mode="encoding",
            serotonin_mood=0.7,
            gaba_sparsity=0.3,
        )
        assert len(dashboard._snapshots) == 1

    def test_record_state_maintains_window(self, dashboard):
        """Window size is maintained."""
        for i in range(15):
            dashboard.record_state(
                dopamine_rpe=0.1 * i,
                norepinephrine_gain=1.0 + 0.1 * i,
                acetylcholine_mode="balanced",
                serotonin_mood=0.5,
                gaba_sparsity=0.5,
            )
        assert len(dashboard._snapshots) == 10

    def test_record_state_preserves_order(self, dashboard):
        """Snapshots are in chronological order."""
        for i in range(5):
            dashboard.record_state(
                dopamine_rpe=0.1 * i,
                norepinephrine_gain=1.0,
                acetylcholine_mode="balanced",
                serotonin_mood=0.5,
                gaba_sparsity=0.5,
            )

        # Oldest first
        assert dashboard._snapshots[0].dopamine_rpe == 0.0
        # Newest last
        assert dashboard._snapshots[-1].dopamine_rpe == 0.4

    def test_get_trace_data_empty(self, dashboard):
        """Get trace data from empty dashboard."""
        traces = dashboard.get_trace_data()
        assert traces == {}

    def test_get_trace_data(self, dashboard):
        """Get trace data with snapshots."""
        dashboard.record_state(
            dopamine_rpe=0.5,
            norepinephrine_gain=1.2,
            acetylcholine_mode="encoding",
            serotonin_mood=0.7,
            gaba_sparsity=0.3,
        )
        dashboard.record_state(
            dopamine_rpe=0.6,
            norepinephrine_gain=1.3,
            acetylcholine_mode="retrieval",
            serotonin_mood=0.8,
            gaba_sparsity=0.4,
        )

        traces = dashboard.get_trace_data()

        assert "dopamine_rpe" in traces
        assert "norepinephrine_gain" in traces
        assert "serotonin_mood" in traces
        assert "gaba_sparsity" in traces

        # Each trace is (timestamps, values)
        timestamps, values = traces["dopamine_rpe"]
        assert len(timestamps) == 2
        assert len(values) == 2
        assert values[0] == 0.5
        assert values[1] == 0.6

    def test_get_mode_distribution_empty(self, dashboard):
        """Get mode distribution from empty dashboard."""
        dist = dashboard.get_mode_distribution()
        assert dist == {}

    def test_get_mode_distribution(self, dashboard):
        """Get mode distribution."""
        dashboard.record_state(0.0, 1.0, "encoding", 0.5, 0.5)
        dashboard.record_state(0.0, 1.0, "encoding", 0.5, 0.5)
        dashboard.record_state(0.0, 1.0, "retrieval", 0.5, 0.5)
        dashboard.record_state(0.0, 1.0, "balanced", 0.5, 0.5)

        dist = dashboard.get_mode_distribution()

        assert dist["encoding"] == 2
        assert dist["retrieval"] == 1
        assert dist["balanced"] == 1

    def test_get_current_state_empty(self, dashboard):
        """Get current state from empty dashboard."""
        state = dashboard.get_current_state()
        assert state is None

    def test_get_current_state(self, dashboard):
        """Get current state returns most recent."""
        dashboard.record_state(0.1, 1.0, "encoding", 0.5, 0.5)
        dashboard.record_state(0.2, 1.1, "retrieval", 0.6, 0.6)
        dashboard.record_state(0.3, 1.2, "balanced", 0.7, 0.7)

        state = dashboard.get_current_state()

        assert state is not None
        assert state.dopamine_rpe == 0.3
        assert state.norepinephrine_gain == 1.2
        assert state.acetylcholine_mode == "balanced"

    def test_get_statistics_empty(self, dashboard):
        """Get statistics from empty dashboard."""
        stats = dashboard.get_statistics()
        assert stats == {}

    def test_get_statistics(self, dashboard):
        """Get statistics with data."""
        dashboard.record_state(0.1, 1.0, "encoding", 0.5, 0.3)
        dashboard.record_state(0.3, 1.2, "retrieval", 0.7, 0.5)
        dashboard.record_state(0.5, 1.4, "balanced", 0.9, 0.7)

        stats = dashboard.get_statistics()

        assert "dopamine" in stats
        assert "norepinephrine" in stats
        assert "serotonin" in stats
        assert "gaba" in stats

        # Check dopamine stats
        da_stats = stats["dopamine"]
        assert da_stats["mean"] == pytest.approx(0.3, abs=0.01)
        assert da_stats["min"] == 0.1
        assert da_stats["max"] == 0.5

    def test_get_statistics_std(self, dashboard):
        """Statistics include standard deviation."""
        # Add values with known std
        values = [0.0, 0.5, 1.0]
        for v in values:
            dashboard.record_state(v, 1.0, "balanced", 0.5, 0.5)

        stats = dashboard.get_statistics()
        # std of [0, 0.5, 1] is approx 0.408
        assert stats["dopamine"]["std"] == pytest.approx(0.408, abs=0.01)


class TestNeuromodulatorDashboardEdgeCases:
    """Edge case tests for NeuromodulatorDashboard."""

    def test_negative_rpe(self):
        """Handle negative dopamine RPE."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(-0.5, 1.0, "balanced", 0.5, 0.5)
        assert dashboard.get_current_state().dopamine_rpe == -0.5

    def test_extreme_values(self):
        """Handle extreme values."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(
            dopamine_rpe=-1.0,
            norepinephrine_gain=0.5,
            acetylcholine_mode="encoding",
            serotonin_mood=0.0,
            gaba_sparsity=1.0,
        )

        state = dashboard.get_current_state()
        assert state.dopamine_rpe == -1.0
        assert state.norepinephrine_gain == 0.5
        assert state.serotonin_mood == 0.0
        assert state.gaba_sparsity == 1.0

    def test_mode_transitions(self):
        """Track mode transitions."""
        dashboard = NeuromodulatorDashboard()

        modes = ["encoding", "balanced", "retrieval", "balanced", "encoding"]
        for mode in modes:
            dashboard.record_state(0.0, 1.0, mode, 0.5, 0.5)

        dist = dashboard.get_mode_distribution()
        assert dist["encoding"] == 2
        assert dist["balanced"] == 2
        assert dist["retrieval"] == 1

    def test_single_snapshot(self):
        """Statistics work with single snapshot."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(0.5, 1.0, "balanced", 0.5, 0.5)

        stats = dashboard.get_statistics()

        # With single value, std should be 0
        assert stats["dopamine"]["std"] == 0.0
        assert stats["dopamine"]["mean"] == 0.5
        assert stats["dopamine"]["min"] == 0.5
        assert stats["dopamine"]["max"] == 0.5

    def test_trace_data_timestamps(self):
        """Trace data timestamps are in order."""
        dashboard = NeuromodulatorDashboard()

        for i in range(5):
            dashboard.record_state(0.1 * i, 1.0, "balanced", 0.5, 0.5)

        traces = dashboard.get_trace_data()
        timestamps, _ = traces["dopamine_rpe"]

        # Timestamps should be increasing
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1]

    def test_large_dataset(self):
        """Handle large dataset efficiently."""
        dashboard = NeuromodulatorDashboard(window_size=10000)

        for i in range(1000):
            dashboard.record_state(
                dopamine_rpe=np.sin(i * 0.1),
                norepinephrine_gain=1.0 + 0.2 * np.cos(i * 0.05),
                acetylcholine_mode=["encoding", "balanced", "retrieval"][i % 3],
                serotonin_mood=0.5 + 0.3 * np.sin(i * 0.02),
                gaba_sparsity=0.5,
            )

        assert len(dashboard._snapshots) == 1000

        stats = dashboard.get_statistics()
        assert "dopamine" in stats
        assert -1 <= stats["dopamine"]["min"] <= 1
        assert -1 <= stats["dopamine"]["max"] <= 1


class TestNeuromodulatorVisualization:
    """Tests for visualization functions."""

    @pytest.fixture
    def dashboard_with_data(self):
        """Create dashboard with sample data."""
        dashboard = NeuromodulatorDashboard(window_size=100)
        for i in range(20):
            dashboard.record_state(
                dopamine_rpe=0.1 * np.sin(i),
                norepinephrine_gain=1.0 + 0.2 * np.cos(i),
                acetylcholine_mode=["encoding", "balanced", "retrieval"][i % 3],
                serotonin_mood=0.5 + 0.1 * i % 0.5,
                gaba_sparsity=0.5,
            )
        return dashboard

    def test_trace_data_structure(self, dashboard_with_data):
        """Trace data has correct structure."""
        traces = dashboard_with_data.get_trace_data()

        for name, (timestamps, values) in traces.items():
            assert len(timestamps) == len(values)
            assert len(timestamps) == 20

    def test_mode_distribution_complete(self, dashboard_with_data):
        """Mode distribution accounts for all snapshots."""
        dist = dashboard_with_data.get_mode_distribution()

        total = sum(dist.values())
        assert total == 20  # All snapshots accounted for

    def test_statistics_bounds(self, dashboard_with_data):
        """Statistics are within expected bounds."""
        stats = dashboard_with_data.get_statistics()

        # Mean should be between min and max
        for modulator in ["dopamine", "norepinephrine", "serotonin", "gaba"]:
            s = stats[modulator]
            assert s["min"] <= s["mean"] <= s["max"]
            assert s["std"] >= 0
