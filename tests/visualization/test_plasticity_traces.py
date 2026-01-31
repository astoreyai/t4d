"""Tests for plasticity traces visualization module."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from t4dm.visualization.plasticity_traces import (
    PlasticityTracer,
    WeightUpdate,
    plot_bcm_curve,
    plot_weight_changes,
    plot_ltp_ltd_distribution,
)


class TestWeightUpdate:
    """Tests for WeightUpdate dataclass."""

    def test_create_update(self):
        """Test creating a weight update record."""
        update = WeightUpdate(
            timestamp=datetime.now(),
            source_id="neuron_1",
            target_id="neuron_2",
            old_weight=0.5,
            new_weight=0.6,
            update_type="ltp",
            activation_level=0.7,
        )
        assert update.source_id == "neuron_1"
        assert update.target_id == "neuron_2"
        assert update.old_weight == 0.5
        assert update.new_weight == 0.6
        assert update.update_type == "ltp"

    def test_update_default_activation(self):
        """Test update with default activation level."""
        update = WeightUpdate(
            timestamp=datetime.now(),
            source_id="s1",
            target_id="t1",
            old_weight=0.5,
            new_weight=0.6,
            update_type="ltp",
        )
        assert update.activation_level == 0.0


class TestPlasticityTracer:
    """Tests for PlasticityTracer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        tracer = PlasticityTracer()
        assert tracer.max_updates == 10000
        assert len(tracer._updates) == 0

    def test_init_custom(self):
        """Test initialization with custom max_updates."""
        tracer = PlasticityTracer(max_updates=100)
        assert tracer.max_updates == 100

    def test_record_update(self):
        """Test recording a weight update."""
        tracer = PlasticityTracer()
        tracer.record_update(
            source_id="s1",
            target_id="t1",
            old_weight=0.5,
            new_weight=0.6,
            update_type="ltp",
            activation_level=0.7,
        )
        assert len(tracer._updates) == 1
        assert tracer._updates[0].new_weight - tracer._updates[0].old_weight == pytest.approx(0.1)

    def test_record_multiple_updates(self):
        """Test recording multiple weight updates."""
        tracer = PlasticityTracer()
        for i in range(5):
            tracer.record_update(
                source_id=f"s{i}",
                target_id=f"t{i}",
                old_weight=0.5,
                new_weight=0.5 + 0.1 * i,
                update_type="ltp" if i % 2 == 0 else "ltd",
            )
        assert len(tracer._updates) == 5

    def test_max_updates_limit(self):
        """Test that max_updates limit is maintained."""
        tracer = PlasticityTracer(max_updates=5)
        for i in range(10):
            tracer.record_update(
                source_id=f"s{i}",
                target_id="t1",
                old_weight=0.5,
                new_weight=0.6,
                update_type="ltp",
            )
        assert len(tracer._updates) == 5
        # Should have most recent
        assert tracer._updates[-1].source_id == "s9"

    def test_get_weight_trajectory(self):
        """Test getting weight trajectory for specific synapse."""
        tracer = PlasticityTracer()
        tracer.record_update("s1", "t1", 0.5, 0.6, "ltp")
        tracer.record_update("s2", "t2", 0.5, 0.4, "ltd")
        tracer.record_update("s1", "t1", 0.6, 0.7, "ltp")
        tracer.record_update("s1", "t1", 0.7, 0.65, "ltd")

        timestamps, weights = tracer.get_weight_trajectory("s1", "t1")
        assert len(timestamps) == 3
        assert len(weights) == 3
        assert weights == [0.6, 0.7, 0.65]

    def test_get_weight_trajectory_empty(self):
        """Test getting trajectory for non-existent synapse."""
        tracer = PlasticityTracer()
        tracer.record_update("s1", "t1", 0.5, 0.6, "ltp")

        timestamps, weights = tracer.get_weight_trajectory("s2", "t2")
        assert len(timestamps) == 0
        assert len(weights) == 0

    def test_get_ltp_ltd_distribution(self):
        """Test getting LTP/LTD magnitude distribution."""
        tracer = PlasticityTracer()
        tracer.record_update("s1", "t1", 0.5, 0.6, "ltp")  # +0.1
        tracer.record_update("s2", "t2", 0.5, 0.7, "ltp")  # +0.2
        tracer.record_update("s3", "t3", 0.5, 0.4, "ltd")  # -0.1
        tracer.record_update("s4", "t4", 0.5, 0.3, "ltd")  # -0.2

        ltp_mags, ltd_mags = tracer.get_ltp_ltd_distribution()
        assert len(ltp_mags) == 2
        assert len(ltd_mags) == 2
        # Check magnitudes using numpy any/isclose for array comparison
        assert any(np.isclose(m, 0.1, atol=0.001) for m in ltp_mags)
        assert any(np.isclose(m, 0.2, atol=0.001) for m in ltp_mags)

    def test_get_bcm_curve_data(self):
        """Test getting BCM curve data."""
        tracer = PlasticityTracer()
        for i in range(10):
            activation = 0.1 * i
            change = activation * (activation - 0.5)  # BCM-like
            tracer.record_update(
                source_id=f"s{i}",
                target_id=f"t{i}",
                old_weight=0.5,
                new_weight=0.5 + change,
                update_type="ltp" if change > 0 else "ltd",
                activation_level=activation,
            )

        activations, changes = tracer.get_bcm_curve_data()
        assert len(activations) == 10
        assert len(changes) == 10

    def test_get_update_counts_by_type(self):
        """Test getting update counts by type."""
        tracer = PlasticityTracer()
        tracer.record_update("s1", "t1", 0.5, 0.6, "ltp")
        tracer.record_update("s2", "t2", 0.5, 0.7, "ltp")
        tracer.record_update("s3", "t3", 0.5, 0.4, "ltd")
        tracer.record_update("s4", "t4", 0.5, 0.5, "homeostatic")

        counts = tracer.get_update_counts_by_type()
        assert counts["ltp"] == 2
        assert counts["ltd"] == 1
        assert counts["homeostatic"] == 1

    def test_get_timeline_data(self):
        """Test getting timeline data."""
        tracer = PlasticityTracer()
        for i in range(20):
            tracer.record_update(
                source_id=f"s{i}",
                target_id=f"t{i}",
                old_weight=0.5,
                new_weight=0.55 if i % 2 == 0 else 0.45,
                update_type="ltp" if i % 2 == 0 else "ltd",
            )

        bin_times, type_counts = tracer.get_timeline_data()
        # Should have some bins
        assert "ltp" in type_counts or "ltd" in type_counts

    def test_get_timeline_data_empty(self):
        """Test timeline with no data."""
        tracer = PlasticityTracer()
        bin_times, type_counts = tracer.get_timeline_data()
        assert bin_times == []
        assert type_counts == {}


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

    def test_data_for_bcm_curve(self):
        """Test data preparation for BCM curve (no actual plotting)."""
        tracer = PlasticityTracer()
        for i in range(20):
            tracer.record_update(
                source_id=f"s{i}",
                target_id=f"t{i}",
                old_weight=0.5,
                new_weight=0.5 + 0.05 * np.random.randn(),
                update_type="ltp" if np.random.rand() > 0.5 else "ltd",
                activation_level=np.random.rand(),
            )
        # Verify data is retrievable for plotting
        activations, changes = tracer.get_bcm_curve_data()
        assert len(activations) == len(changes)

    def test_data_for_weight_changes(self):
        """Test data preparation for weight changes plot."""
        tracer = PlasticityTracer()
        for i in range(20):
            tracer.record_update(
                source_id=f"s{i % 5}",
                target_id=f"t{i % 5}",
                old_weight=0.5,
                new_weight=0.5 + 0.05 * np.random.randn(),
                update_type="ltp" if np.random.rand() > 0.5 else "ltd",
            )
        bin_times, type_counts = tracer.get_timeline_data()
        # Should have data for plotting
        assert isinstance(type_counts, dict)

    def test_data_for_ltp_ltd_distribution(self):
        """Test data preparation for LTP/LTD distribution plot."""
        tracer = PlasticityTracer()
        for i in range(30):
            delta = 0.1 * (np.random.rand() - 0.3)
            tracer.record_update(
                source_id=f"s{i}",
                target_id=f"t{i}",
                old_weight=0.5,
                new_weight=0.5 + delta,
                update_type="ltp" if delta > 0 else "ltd",
            )
        ltp_mags, ltd_mags = tracer.get_ltp_ltd_distribution()
        # Should have distributions
        assert len(ltp_mags) > 0 or len(ltd_mags) > 0

    def test_empty_tracer_data(self):
        """Test data from empty tracer."""
        tracer = PlasticityTracer()
        activations, changes = tracer.get_bcm_curve_data()
        assert len(activations) == 0
        bin_times, type_counts = tracer.get_timeline_data()
        assert bin_times == []


class TestPlotBcmCurve:
    """Tests for plot_bcm_curve function."""

    @pytest.fixture
    def populated_tracer(self):
        """Create tracer with BCM data."""
        tracer = PlasticityTracer()
        for i in range(20):
            activation = 0.05 * i
            change = activation * (activation - 0.5)  # BCM-like
            tracer.record_update(
                source_id=f"s{i}",
                target_id=f"t{i}",
                old_weight=0.5,
                new_weight=0.5 + change,
                update_type="ltp" if change > 0 else "ltd",
                activation_level=activation,
            )
        return tracer

    def test_bcm_curve_empty_tracer(self):
        """Test plotting with empty tracer."""
        tracer = PlasticityTracer()
        # Should not raise, just log warning
        plot_bcm_curve(tracer)

    @patch("t4dm.visualization.plasticity_traces.logger")
    def test_bcm_curve_logs_warning_when_empty(self, mock_logger):
        """Test that empty tracer logs warning."""
        tracer = PlasticityTracer()
        plot_bcm_curve(tracer)
        mock_logger.warning.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_bcm_curve_matplotlib(
        self, mock_close, mock_show, mock_colorbar, mock_tight, mock_subplots, populated_tracer
    ):
        """Test matplotlib BCM curve plotting."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.scatter = MagicMock(return_value=MagicMock())
        mock_subplots.return_value = (mock_fig, mock_ax)

        plot_bcm_curve(populated_tracer, interactive=False)

        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_bcm_curve_save_path(
        self, mock_close, mock_save, mock_colorbar, mock_tight, mock_subplots, populated_tracer, tmp_path
    ):
        """Test saving BCM curve plot."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.scatter = MagicMock(return_value=MagicMock())
        mock_subplots.return_value = (mock_fig, mock_ax)

        save_path = tmp_path / "bcm.png"
        plot_bcm_curve(populated_tracer, save_path=save_path, interactive=False)

        mock_save.assert_called_once()

    def test_bcm_curve_plotly(self, populated_tracer):
        """Test plotly BCM curve plotting."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("Plotly not available")

        plot_bcm_curve(populated_tracer, interactive=True)


class TestPlotWeightChanges:
    """Tests for plot_weight_changes function."""

    @pytest.fixture
    def populated_tracer(self):
        """Create tracer with timeline data."""
        tracer = PlasticityTracer()
        import time
        for i in range(20):
            tracer.record_update(
                source_id=f"s{i}",
                target_id=f"t{i}",
                old_weight=0.5,
                new_weight=0.55 if i % 2 == 0 else 0.45,
                update_type="ltp" if i % 2 == 0 else "ltd",
            )
            time.sleep(0.01)  # Small delay for timestamp differentiation
        return tracer

    def test_weight_changes_empty_tracer(self):
        """Test plotting with empty tracer."""
        tracer = PlasticityTracer()
        # Should not raise, just log warning
        plot_weight_changes(tracer)

    @patch("t4dm.visualization.plasticity_traces.logger")
    def test_weight_changes_logs_warning_when_empty(self, mock_logger):
        """Test that empty tracer logs warning."""
        tracer = PlasticityTracer()
        plot_weight_changes(tracer)
        mock_logger.warning.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_weight_changes_matplotlib(
        self, mock_close, mock_show, mock_tight, mock_xticks, mock_subplots, populated_tracer
    ):
        """Test matplotlib weight changes plotting."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plot_weight_changes(populated_tracer, interactive=False)

        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_weight_changes_save_path(
        self, mock_close, mock_save, mock_tight, mock_xticks, mock_subplots, populated_tracer, tmp_path
    ):
        """Test saving weight changes plot."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        save_path = tmp_path / "changes.png"
        plot_weight_changes(populated_tracer, save_path=save_path, interactive=False)

        mock_save.assert_called_once()

    def test_weight_changes_plotly(self, populated_tracer):
        """Test plotly weight changes plotting."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("Plotly not available")

        plot_weight_changes(populated_tracer, interactive=True)


class TestPlotLtpLtdDistribution:
    """Tests for plot_ltp_ltd_distribution function."""

    @pytest.fixture
    def populated_tracer(self):
        """Create tracer with LTP/LTD data."""
        tracer = PlasticityTracer()
        for i in range(30):
            delta = 0.1 * (np.random.rand() - 0.3)
            tracer.record_update(
                source_id=f"s{i}",
                target_id=f"t{i}",
                old_weight=0.5,
                new_weight=0.5 + delta,
                update_type="ltp" if delta > 0 else "ltd",
            )
        return tracer

    def test_ltp_ltd_empty_tracer(self):
        """Test plotting with empty tracer."""
        tracer = PlasticityTracer()
        # Should not raise, just log warning
        plot_ltp_ltd_distribution(tracer)

    @patch("t4dm.visualization.plasticity_traces.logger")
    def test_ltp_ltd_logs_warning_when_empty(self, mock_logger):
        """Test that empty tracer logs warning."""
        tracer = PlasticityTracer()
        plot_ltp_ltd_distribution(tracer)
        mock_logger.warning.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_ltp_ltd_matplotlib(
        self, mock_close, mock_show, mock_tight, mock_subplots, populated_tracer
    ):
        """Test matplotlib LTP/LTD distribution plotting."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        plot_ltp_ltd_distribution(populated_tracer, interactive=False)

        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_ltp_ltd_save_path(
        self, mock_close, mock_save, mock_tight, mock_subplots, populated_tracer, tmp_path
    ):
        """Test saving LTP/LTD distribution plot."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        save_path = tmp_path / "ltp_ltd.png"
        plot_ltp_ltd_distribution(populated_tracer, save_path=save_path, interactive=False)

        mock_save.assert_called_once()

    def test_ltp_ltd_plotly(self, populated_tracer):
        """Test plotly LTP/LTD distribution plotting."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            pytest.skip("Plotly not available")

        plot_ltp_ltd_distribution(populated_tracer, interactive=True)


class TestPlasticityTracerEdgeCases:
    """Edge case tests for PlasticityTracer."""

    def test_homeostatic_update(self):
        """Test recording homeostatic updates."""
        tracer = PlasticityTracer()
        tracer.record_update(
            source_id="s1",
            target_id="t1",
            old_weight=0.5,
            new_weight=0.5,  # No change
            update_type="homeostatic",
        )
        counts = tracer.get_update_counts_by_type()
        assert counts["homeostatic"] == 1

    def test_ltp_ltd_distribution_with_only_ltp(self):
        """Test distribution with only LTP updates."""
        tracer = PlasticityTracer()
        for i in range(10):
            tracer.record_update(f"s{i}", f"t{i}", 0.5, 0.6 + 0.01 * i, "ltp")

        ltp_mags, ltd_mags = tracer.get_ltp_ltd_distribution()
        assert len(ltp_mags) == 10
        assert len(ltd_mags) == 0

    def test_ltp_ltd_distribution_with_only_ltd(self):
        """Test distribution with only LTD updates."""
        tracer = PlasticityTracer()
        for i in range(10):
            tracer.record_update(f"s{i}", f"t{i}", 0.5, 0.4 - 0.01 * i, "ltd")

        ltp_mags, ltd_mags = tracer.get_ltp_ltd_distribution()
        assert len(ltp_mags) == 0
        assert len(ltd_mags) == 10

    def test_very_small_max_updates(self):
        """Test with very small max_updates limit."""
        tracer = PlasticityTracer(max_updates=2)
        for i in range(10):
            tracer.record_update(f"s{i}", "t1", 0.5, 0.6, "ltp")

        assert len(tracer._updates) == 2
        assert tracer._updates[0].source_id == "s8"
        assert tracer._updates[1].source_id == "s9"

    def test_weight_trajectory_multiple_synapses(self):
        """Test getting trajectory with multiple synapses."""
        tracer = PlasticityTracer()
        # Record updates for multiple synapses
        for i in range(5):
            tracer.record_update("s1", "t1", 0.5 + 0.05 * i, 0.55 + 0.05 * i, "ltp")
            tracer.record_update("s2", "t2", 0.4 + 0.05 * i, 0.45 + 0.05 * i, "ltp")

        ts1, w1 = tracer.get_weight_trajectory("s1", "t1")
        ts2, w2 = tracer.get_weight_trajectory("s2", "t2")

        assert len(ts1) == 5
        assert len(ts2) == 5
        assert w1 != w2  # Different trajectories

    def test_bcm_curve_data_empty_tracer(self):
        """Test BCM curve data from empty tracer."""
        tracer = PlasticityTracer()
        activations, changes = tracer.get_bcm_curve_data()
        assert len(activations) == 0
        assert len(changes) == 0

    def test_timeline_single_update(self):
        """Test timeline data with single update."""
        tracer = PlasticityTracer()
        tracer.record_update("s1", "t1", 0.5, 0.6, "ltp")

        bin_times, type_counts = tracer.get_timeline_data()
        # Should have at least one bin
        assert len(bin_times) >= 1
        assert "ltp" in type_counts
