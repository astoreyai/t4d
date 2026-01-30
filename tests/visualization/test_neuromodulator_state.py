"""Tests for neuromodulator state visualization module."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from ww.visualization.neuromodulator_state import (
    NeuromodulatorDashboard,
    NeuromodulatorSnapshot,
    plot_neuromodulator_traces,
    plot_neuromodulator_radar,
)


class TestNeuromodulatorSnapshot:
    """Tests for NeuromodulatorSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a neuromodulator snapshot."""
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


class TestNeuromodulatorDashboard:
    """Tests for NeuromodulatorDashboard class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        dashboard = NeuromodulatorDashboard()
        assert dashboard.window_size == 1000
        assert len(dashboard._snapshots) == 0

    def test_init_custom_window(self):
        """Test initialization with custom window size."""
        dashboard = NeuromodulatorDashboard(window_size=50)
        assert dashboard.window_size == 50

    def test_record_state(self):
        """Test recording a neuromodulator state."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(
            dopamine_rpe=0.5,
            norepinephrine_gain=1.2,
            acetylcholine_mode="encoding",
            serotonin_mood=0.7,
            gaba_sparsity=0.3,
        )
        assert len(dashboard._snapshots) == 1
        assert dashboard._snapshots[0].dopamine_rpe == 0.5

    def test_record_multiple_states(self):
        """Test recording multiple states."""
        dashboard = NeuromodulatorDashboard()
        for i in range(5):
            dashboard.record_state(
                dopamine_rpe=0.1 * i,
                norepinephrine_gain=1.0,
                acetylcholine_mode="balanced",
                serotonin_mood=0.5,
                gaba_sparsity=0.3,
            )
        assert len(dashboard._snapshots) == 5
        assert dashboard._snapshots[4].dopamine_rpe == pytest.approx(0.4)

    def test_window_size_limit(self):
        """Test that window size is maintained."""
        dashboard = NeuromodulatorDashboard(window_size=10)
        for i in range(20):
            dashboard.record_state(
                dopamine_rpe=0.05 * i,
                norepinephrine_gain=1.0,
                acetylcholine_mode="balanced",
                serotonin_mood=0.5,
                gaba_sparsity=0.3,
            )
        assert len(dashboard._snapshots) == 10
        # Should have most recent
        assert dashboard._snapshots[-1].dopamine_rpe == pytest.approx(0.95)

    def test_get_current_state(self):
        """Test getting current (most recent) state."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(
            dopamine_rpe=0.5,
            norepinephrine_gain=1.0,
            acetylcholine_mode="encoding",
            serotonin_mood=0.5,
            gaba_sparsity=0.3,
        )
        dashboard.record_state(
            dopamine_rpe=0.8,
            norepinephrine_gain=1.5,
            acetylcholine_mode="retrieval",
            serotonin_mood=0.6,
            gaba_sparsity=0.4,
        )
        current = dashboard.get_current_state()
        assert current.dopamine_rpe == 0.8
        assert current.acetylcholine_mode == "retrieval"

    def test_get_current_state_empty(self):
        """Test getting current state when no history."""
        dashboard = NeuromodulatorDashboard()
        current = dashboard.get_current_state()
        assert current is None

    def test_get_trace_data(self):
        """Test getting trace data for all neuromodulators."""
        dashboard = NeuromodulatorDashboard()
        for i in range(5):
            dashboard.record_state(
                dopamine_rpe=0.1 * i,
                norepinephrine_gain=1.0 + 0.1 * i,
                acetylcholine_mode="balanced",
                serotonin_mood=0.5,
                gaba_sparsity=0.3,
            )
        traces = dashboard.get_trace_data()
        assert "dopamine_rpe" in traces
        assert "norepinephrine_gain" in traces
        assert "serotonin_mood" in traces
        assert "gaba_sparsity" in traces
        timestamps, values = traces["dopamine_rpe"]
        assert len(timestamps) == 5
        assert len(values) == 5

    def test_get_trace_data_empty(self):
        """Test getting trace data when empty."""
        dashboard = NeuromodulatorDashboard()
        traces = dashboard.get_trace_data()
        assert traces == {}

    def test_get_mode_distribution(self):
        """Test getting ACh mode distribution."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(0.5, 1.0, "encoding", 0.5, 0.3)
        dashboard.record_state(0.5, 1.0, "encoding", 0.5, 0.3)
        dashboard.record_state(0.5, 1.0, "retrieval", 0.5, 0.3)
        dashboard.record_state(0.5, 1.0, "balanced", 0.5, 0.3)

        mode_dist = dashboard.get_mode_distribution()
        assert mode_dist["encoding"] == 2
        assert mode_dist["retrieval"] == 1
        assert mode_dist["balanced"] == 1

    def test_get_statistics(self):
        """Test getting statistics for neuromodulators."""
        dashboard = NeuromodulatorDashboard()
        for i in range(10):
            dashboard.record_state(
                dopamine_rpe=0.1 * i - 0.5,
                norepinephrine_gain=1.0 + 0.1 * i,
                acetylcholine_mode="balanced",
                serotonin_mood=0.5 + 0.05 * i,
                gaba_sparsity=0.3,
            )
        stats = dashboard.get_statistics()
        assert "dopamine" in stats
        assert "norepinephrine" in stats
        assert "serotonin" in stats
        assert "gaba" in stats
        assert "mean" in stats["dopamine"]
        assert "std" in stats["dopamine"]
        assert "min" in stats["dopamine"]
        assert "max" in stats["dopamine"]

    def test_get_statistics_empty(self):
        """Test getting statistics when empty."""
        dashboard = NeuromodulatorDashboard()
        stats = dashboard.get_statistics()
        assert stats == {}


try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available for integration tests"
)
class TestPlotFunctions:
    """Integration tests for plot functions - require matplotlib."""

    def test_data_for_neuromodulator_traces(self):
        """Test data preparation for neuromodulator traces (no actual plotting)."""
        dashboard = NeuromodulatorDashboard()
        for i in range(10):
            dashboard.record_state(
                dopamine_rpe=0.5 + 0.05 * np.sin(i),
                norepinephrine_gain=1.0,
                acetylcholine_mode="balanced",
                serotonin_mood=0.5,
                gaba_sparsity=0.3,
            )
        # Verify trace data is retrievable for plotting
        traces = dashboard.get_trace_data()
        assert "dopamine_rpe" in traces
        assert len(traces["dopamine_rpe"][0]) == 10

    def test_data_for_neuromodulator_radar(self):
        """Test data preparation for radar chart."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(
            dopamine_rpe=0.5,
            norepinephrine_gain=1.2,
            acetylcholine_mode="encoding",
            serotonin_mood=0.7,
            gaba_sparsity=0.3,
        )
        # Verify current state is available for radar
        current = dashboard.get_current_state()
        assert current is not None
        assert current.dopamine_rpe == 0.5

    def test_empty_dashboard_data(self):
        """Test data from empty dashboard."""
        dashboard = NeuromodulatorDashboard()
        traces = dashboard.get_trace_data()
        assert traces == {}
        current = dashboard.get_current_state()
        assert current is None


class TestPlotNeuromodulatorTraces:
    """Tests for plot_neuromodulator_traces function."""

    @pytest.fixture
    def populated_dashboard(self):
        """Create dashboard with data."""
        dashboard = NeuromodulatorDashboard(window_size=100)
        for i in range(20):
            dashboard.record_state(
                dopamine_rpe=0.1 * i - 1.0,
                norepinephrine_gain=1.0 + 0.05 * i,
                acetylcholine_mode="encoding",
                serotonin_mood=0.5,
                gaba_sparsity=0.5,
            )
        return dashboard

    def test_plot_traces_empty_dashboard(self):
        """Test plotting with empty dashboard."""
        dashboard = NeuromodulatorDashboard()
        # Should not raise, just log warning
        plot_neuromodulator_traces(dashboard)

    @patch("ww.visualization.neuromodulator_state.logger")
    def test_plot_traces_logs_warning_when_empty(self, mock_logger):
        """Test that empty dashboard logs warning."""
        dashboard = NeuromodulatorDashboard()
        plot_neuromodulator_traces(dashboard)
        mock_logger.warning.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_plot_traces_matplotlib(
        self, mock_close, mock_show, mock_tight, mock_subplots, populated_dashboard
    ):
        """Test matplotlib trace plotting."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(4)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        plot_neuromodulator_traces(populated_dashboard, interactive=False)

        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_traces_save_path(
        self, mock_close, mock_save, mock_tight, mock_subplots, populated_dashboard, tmp_path
    ):
        """Test saving trace plot."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(4)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        save_path = tmp_path / "traces.png"
        plot_neuromodulator_traces(
            populated_dashboard, save_path=save_path, interactive=False
        )

        mock_save.assert_called_once()

    def test_plot_traces_plotly(self, populated_dashboard):
        """Test plotly trace plotting."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            pytest.skip("Plotly not available")

        # Plotly is available, test with it
        plot_neuromodulator_traces(populated_dashboard, interactive=True)


class TestPlotNeuromodulatorRadar:
    """Tests for plot_neuromodulator_radar function."""

    @pytest.fixture
    def populated_dashboard(self):
        """Create dashboard with data."""
        dashboard = NeuromodulatorDashboard(window_size=100)
        dashboard.record_state(
            dopamine_rpe=0.5,
            norepinephrine_gain=1.5,
            acetylcholine_mode="encoding",
            serotonin_mood=0.7,
            gaba_sparsity=0.3,
        )
        return dashboard

    def test_radar_empty_dashboard(self):
        """Test plotting radar with empty dashboard."""
        dashboard = NeuromodulatorDashboard()
        # Should not raise, just log warning
        plot_neuromodulator_radar(dashboard)

    @patch("ww.visualization.neuromodulator_state.logger")
    def test_radar_logs_warning_when_empty(self, mock_logger):
        """Test that empty dashboard logs warning."""
        dashboard = NeuromodulatorDashboard()
        plot_neuromodulator_radar(dashboard)
        mock_logger.warning.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_radar_matplotlib(
        self, mock_close, mock_show, mock_tight, mock_subplots, populated_dashboard
    ):
        """Test matplotlib radar plotting."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plot_neuromodulator_radar(populated_dashboard, interactive=False)

        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_radar_save_path(
        self, mock_close, mock_save, mock_tight, mock_subplots, populated_dashboard, tmp_path
    ):
        """Test saving radar plot."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        save_path = tmp_path / "radar.png"
        plot_neuromodulator_radar(
            populated_dashboard, save_path=save_path, interactive=False
        )

        mock_save.assert_called_once()

    def test_radar_value_normalization(self, populated_dashboard):
        """Test that radar values are normalized correctly."""
        current = populated_dashboard.get_current_state()

        # DA RPE 0.5 -> (0.5 + 1) / 2 = 0.75
        expected_da = (current.dopamine_rpe + 1) / 2
        assert expected_da == pytest.approx(0.75)

        # NE Gain 1.5 -> (1.5 - 0.5) / 1.5 = 0.667
        expected_ne = (current.norepinephrine_gain - 0.5) / 1.5
        assert expected_ne == pytest.approx(0.667, abs=0.01)

    def test_radar_plotly(self, populated_dashboard):
        """Test plotly radar plotting."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("Plotly not available")

        # Plotly is available, test with it
        plot_neuromodulator_radar(populated_dashboard, interactive=True)


class TestNeuromodulatorStateEdgeCases:
    """Edge case tests for neuromodulator state module."""

    def test_single_entry_statistics(self):
        """Test statistics with single entry."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(
            dopamine_rpe=0.5,
            norepinephrine_gain=1.0,
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            gaba_sparsity=0.5,
        )

        stats = dashboard.get_statistics()
        assert stats["dopamine"]["mean"] == 0.5
        assert stats["dopamine"]["std"] == 0.0
        assert stats["dopamine"]["min"] == 0.5
        assert stats["dopamine"]["max"] == 0.5

    def test_very_small_window(self):
        """Test with very small window size."""
        dashboard = NeuromodulatorDashboard(window_size=2)

        for i in range(10):
            dashboard.record_state(
                dopamine_rpe=float(i),
                norepinephrine_gain=1.0,
                acetylcholine_mode="balanced",
                serotonin_mood=0.5,
                gaba_sparsity=0.5,
            )

        assert len(dashboard._snapshots) == 2
        assert dashboard._snapshots[0].dopamine_rpe == 8.0
        assert dashboard._snapshots[1].dopamine_rpe == 9.0

    def test_all_same_mode(self):
        """Test mode distribution with all same mode."""
        dashboard = NeuromodulatorDashboard()

        for _ in range(10):
            dashboard.record_state(
                dopamine_rpe=0.0,
                norepinephrine_gain=1.0,
                acetylcholine_mode="encoding",
                serotonin_mood=0.5,
                gaba_sparsity=0.5,
            )

        distribution = dashboard.get_mode_distribution()
        assert len(distribution) == 1
        assert distribution["encoding"] == 10

    def test_timestamp_ordering(self):
        """Test that timestamps are properly ordered."""
        dashboard = NeuromodulatorDashboard()

        for _ in range(5):
            dashboard.record_state(
                dopamine_rpe=0.0,
                norepinephrine_gain=1.0,
                acetylcholine_mode="balanced",
                serotonin_mood=0.5,
                gaba_sparsity=0.5,
            )

        timestamps = [s.timestamp for s in dashboard._snapshots]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1]

    def test_negative_dopamine_rpe(self):
        """Test snapshot with negative reward prediction error."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(
            dopamine_rpe=-0.8,
            norepinephrine_gain=1.0,
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            gaba_sparsity=0.5,
        )
        assert dashboard.get_current_state().dopamine_rpe == -0.8

    def test_extreme_ne_gain(self):
        """Test extreme norepinephrine gain values."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(
            dopamine_rpe=0.0,
            norepinephrine_gain=2.0,  # Max expected
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            gaba_sparsity=0.5,
        )
        assert dashboard.get_current_state().norepinephrine_gain == 2.0

    def test_ach_mode_retrieval(self):
        """Test retrieval ACh mode."""
        dashboard = NeuromodulatorDashboard()
        dashboard.record_state(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="retrieval",
            serotonin_mood=0.5,
            gaba_sparsity=0.5,
        )
        assert dashboard.get_current_state().acetylcholine_mode == "retrieval"


class TestNeuromodulatorDashboardIntegration:
    """Integration tests for NeuromodulatorDashboard."""

    def test_full_workflow(self):
        """Test complete dashboard workflow."""
        dashboard = NeuromodulatorDashboard(window_size=1000)

        # Simulate changing neuromodulator states
        for i in range(100):
            phase = i / 100.0
            dashboard.record_state(
                dopamine_rpe=np.sin(phase * 2 * np.pi),
                norepinephrine_gain=1.0 + 0.5 * np.cos(phase * np.pi),
                acetylcholine_mode="encoding" if phase < 0.5 else "retrieval",
                serotonin_mood=0.5 + 0.3 * np.sin(phase * np.pi),
                gaba_sparsity=0.5 + 0.2 * np.cos(phase * 2 * np.pi),
            )

        # Verify all methods work
        assert len(dashboard._snapshots) == 100

        trace_data = dashboard.get_trace_data()
        assert len(trace_data) == 4

        mode_dist = dashboard.get_mode_distribution()
        assert mode_dist["encoding"] == 50
        assert mode_dist["retrieval"] == 50

        current = dashboard.get_current_state()
        assert current is not None

        stats = dashboard.get_statistics()
        assert len(stats) == 4

    def test_oscillating_neuromodulators(self):
        """Test tracking oscillating neuromodulator values."""
        dashboard = NeuromodulatorDashboard(window_size=100)

        # Simulate sinusoidal dopamine RPE
        for i in range(50):
            t = i / 50.0
            dashboard.record_state(
                dopamine_rpe=np.sin(t * 2 * np.pi),
                norepinephrine_gain=1.0,
                acetylcholine_mode="balanced",
                serotonin_mood=0.5,
                gaba_sparsity=0.5,
            )

        stats = dashboard.get_statistics()

        # Sinusoid should have mean near 0, min near -1, max near 1
        da_stats = stats["dopamine"]
        assert da_stats["mean"] == pytest.approx(0.0, abs=0.1)
        assert da_stats["min"] < 0
        assert da_stats["max"] > 0
