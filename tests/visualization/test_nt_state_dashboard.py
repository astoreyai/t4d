"""
Tests for NT State Dashboard Visualization.

Tests the core functionality of the NT state dashboard:
- State recording and snapshot creation
- Homeostatic setpoint deviation tracking
- Receptor saturation computation (Michaelis-Menten)
- Cross-NT correlation matrix
- Temporal autocorrelation analysis
- Opponent process dynamics
- Alert generation
"""

import numpy as np
import pytest
from datetime import datetime

from t4dm.visualization.nt_state_dashboard import (
    NTStateDashboard,
    NTSnapshot,
    NTStatistics,
    NT_LABELS,
    NT_COLORS,
    HOMEOSTATIC_SETPOINTS,
    RECEPTOR_KM,
)


class TestNTStateDashboard:
    """Tests for NTStateDashboard class."""

    def test_init_default(self):
        """Test default initialization."""
        dashboard = NTStateDashboard()
        assert dashboard.window_size == 1000
        assert dashboard.alert_deviation == 0.3
        assert len(dashboard._snapshots) == 0
        assert np.allclose(dashboard.setpoints, HOMEOSTATIC_SETPOINTS)

    def test_init_custom(self):
        """Test custom initialization."""
        custom_setpoints = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        dashboard = NTStateDashboard(
            window_size=100,
            homeostatic_setpoints=custom_setpoints,
            alert_deviation=0.2,
        )
        assert dashboard.window_size == 100
        assert dashboard.alert_deviation == 0.2
        assert np.allclose(dashboard.setpoints, custom_setpoints)

    def test_record_state_basic(self):
        """Test basic state recording."""
        dashboard = NTStateDashboard()
        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        snapshot = dashboard.record_state(nt_state)

        assert isinstance(snapshot, NTSnapshot)
        assert len(dashboard._snapshots) == 1
        assert np.allclose(snapshot.nt_state, nt_state)

    def test_record_state_with_firing_rates(self):
        """Test state recording with firing rates."""
        dashboard = NTStateDashboard()
        nt_state = np.array([0.6, 0.4, 0.5, 0.7, 0.5, 0.5])

        snapshot = dashboard.record_state(
            nt_state,
            vta_firing_rate=5.0,
            raphe_firing_rate=2.5,
        )

        assert snapshot.vta_firing_rate == 5.0
        assert snapshot.raphe_firing_rate == 2.5

    def test_record_state_window_limit(self):
        """Test that snapshots respect window size."""
        dashboard = NTStateDashboard(window_size=10)

        for i in range(20):
            nt_state = np.random.rand(6)
            dashboard.record_state(nt_state)

        assert len(dashboard._snapshots) == 10
        assert len(dashboard._nt_history) == 10

    def test_setpoint_deviation_computation(self):
        """Test setpoint deviation is computed correctly."""
        dashboard = NTStateDashboard()

        # State above setpoints
        high_state = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
        snapshot = dashboard.record_state(high_state)

        expected_deviation = high_state - dashboard.setpoints
        assert np.allclose(snapshot.setpoint_deviation, expected_deviation)

    def test_receptor_saturation_computation(self):
        """Test Michaelis-Menten saturation computation."""
        dashboard = NTStateDashboard()

        # At Km, saturation should be 0.5
        nt_state = RECEPTOR_KM.copy()
        snapshot = dashboard.record_state(nt_state)

        assert np.allclose(snapshot.receptor_saturation, 0.5, atol=0.01)

    def test_ei_balance_computation(self):
        """Test E/I balance computation."""
        dashboard = NTStateDashboard()

        # GABA = Glu -> E/I = 1.0
        balanced_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        snapshot = dashboard.record_state(balanced_state)
        assert abs(snapshot.ei_balance - 1.0) < 0.01

        dashboard.clear_history()

        # GABA > Glu -> E/I > 1.0
        inhibitory_state = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.4])
        snapshot = dashboard.record_state(inhibitory_state)
        assert snapshot.ei_balance > 1.0

    def test_arousal_index_computation(self):
        """Test arousal index computation."""
        dashboard = NTStateDashboard()

        # High DA, NE, Glu -> high arousal
        high_arousal = np.array([1.0, 0.5, 0.5, 1.0, 0.5, 1.0])
        snapshot = dashboard.record_state(high_arousal)
        assert snapshot.arousal_index > 0.7

        dashboard.clear_history()

        # Low DA, NE, Glu -> low arousal
        low_arousal = np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.0])
        snapshot = dashboard.record_state(low_arousal)
        assert snapshot.arousal_index < 0.2

    def test_get_current_state(self):
        """Test getting current NT state."""
        dashboard = NTStateDashboard()

        nt_state = np.array([0.6, 0.4, 0.5, 0.7, 0.3, 0.8])
        dashboard.record_state(nt_state)

        current = dashboard.get_current_state()
        assert np.allclose(current, nt_state)

    def test_get_current_state_empty(self):
        """Test getting current state with no data."""
        dashboard = NTStateDashboard()
        assert dashboard.get_current_state() is None

    def test_get_current_saturation(self):
        """Test getting current receptor saturation."""
        dashboard = NTStateDashboard()

        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        dashboard.record_state(nt_state)

        sat = dashboard.get_current_saturation()
        assert sat is not None
        assert len(sat) == 6
        assert all(0 <= s <= 1 for s in sat)

    def test_get_nt_traces(self):
        """Test NT time series retrieval."""
        dashboard = NTStateDashboard()

        for i in range(10):
            nt_state = np.random.rand(6)
            dashboard.record_state(nt_state)

        traces = dashboard.get_nt_traces()

        assert len(traces) == 6
        for label in NT_LABELS:
            assert label in traces
            timestamps, values = traces[label]
            assert len(timestamps) == 10
            assert len(values) == 10
            assert all(isinstance(t, datetime) for t in timestamps)

    def test_get_nt_traces_empty(self):
        """Test NT traces with no data."""
        dashboard = NTStateDashboard()
        traces = dashboard.get_nt_traces()
        assert traces == {}

    def test_get_deviation_traces(self):
        """Test deviation traces retrieval."""
        dashboard = NTStateDashboard()

        for _ in range(10):
            dashboard.record_state(np.random.rand(6))

        traces = dashboard.get_deviation_traces()

        assert len(traces) == 6
        for label in NT_LABELS:
            timestamps, values = traces[label]
            assert len(timestamps) == 10
            assert len(values) == 10

    def test_get_saturation_traces(self):
        """Test saturation traces retrieval."""
        dashboard = NTStateDashboard()

        for _ in range(10):
            dashboard.record_state(np.random.rand(6))

        traces = dashboard.get_saturation_traces()

        assert len(traces) == 6
        for label in NT_LABELS:
            timestamps, values = traces[label]
            assert len(timestamps) == 10
            assert all(0 <= v <= 1 for v in values)

    def test_get_ei_balance_trace(self):
        """Test E/I balance trace retrieval."""
        dashboard = NTStateDashboard()

        for _ in range(10):
            dashboard.record_state(np.random.rand(6))

        timestamps, values = dashboard.get_ei_balance_trace()

        assert len(timestamps) == 10
        assert len(values) == 10
        assert all(v > 0 for v in values)

    def test_get_arousal_trace(self):
        """Test arousal trace retrieval."""
        dashboard = NTStateDashboard()

        for _ in range(10):
            dashboard.record_state(np.random.rand(6))

        timestamps, values = dashboard.get_arousal_trace()

        assert len(timestamps) == 10
        assert len(values) == 10
        assert all(0 <= v <= 1 for v in values)

    def test_get_firing_rate_traces(self):
        """Test firing rate traces retrieval."""
        dashboard = NTStateDashboard()

        for i in range(10):
            dashboard.record_state(
                np.random.rand(6),
                vta_firing_rate=5.0 + i,
                raphe_firing_rate=2.0 + i * 0.5,
            )

        traces = dashboard.get_firing_rate_traces()

        assert "VTA" in traces
        assert "Raphe" in traces
        assert len(traces["VTA"][1]) == 10
        assert len(traces["Raphe"][1]) == 10

    def test_compute_statistics(self):
        """Test statistics computation."""
        dashboard = NTStateDashboard()

        for _ in range(20):
            dashboard.record_state(np.random.rand(6))

        stats = dashboard.compute_statistics()

        assert isinstance(stats, NTStatistics)
        assert stats.mean.shape == (6,)
        assert stats.std.shape == (6,)
        assert stats.min.shape == (6,)
        assert stats.max.shape == (6,)
        assert stats.correlation_matrix.shape == (6, 6)
        assert stats.autocorrelation.shape == (6,)

    def test_compute_statistics_window(self):
        """Test statistics with window parameter."""
        dashboard = NTStateDashboard()

        # Record 100 samples
        for _ in range(100):
            dashboard.record_state(np.random.rand(6))

        # Compute stats for last 10
        stats = dashboard.compute_statistics(window=10)

        assert isinstance(stats, NTStatistics)
        # Stats should be based on 10 samples

    def test_compute_statistics_empty(self):
        """Test statistics with no data."""
        dashboard = NTStateDashboard()
        stats = dashboard.compute_statistics()

        assert np.allclose(stats.mean, 0)
        assert np.allclose(stats.correlation_matrix, np.eye(6))

    def test_get_correlation_matrix(self):
        """Test correlation matrix retrieval."""
        dashboard = NTStateDashboard()

        # Add correlated data
        for i in range(50):
            da = np.random.rand()
            nt_state = np.array([da, da * 0.8 + 0.1, 0.5, 0.5, 0.5, 0.5])
            dashboard.record_state(nt_state)

        corr = dashboard.get_correlation_matrix()

        assert corr.shape == (6, 6)
        # DA and 5-HT should be positively correlated
        assert corr[0, 1] > 0.5

    def test_get_autocorrelation(self):
        """Test autocorrelation retrieval."""
        dashboard = NTStateDashboard()

        # Add smooth trajectory (high autocorrelation)
        for i in range(50):
            t = i / 50
            nt_state = np.array([
                0.5 + 0.3 * np.sin(2 * np.pi * t),
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ])
            dashboard.record_state(nt_state)

        autocorr = dashboard.get_autocorrelation()

        assert len(autocorr) == 6
        # DA should have high autocorrelation (smooth sine)
        assert autocorr[0] > 0.8

    def test_compute_saturation_curve(self):
        """Test saturation curve computation."""
        dashboard = NTStateDashboard()

        conc, sat = dashboard.compute_saturation_curve(0)  # DA

        assert len(conc) == 100
        assert len(sat) == 100
        assert conc[0] == 0
        assert conc[-1] == 1
        # At Km, saturation = 0.5
        km_idx = np.argmin(np.abs(conc - dashboard.km[0]))
        assert abs(sat[km_idx] - 0.5) < 0.05

    def test_get_all_saturation_curves(self):
        """Test all saturation curves retrieval."""
        dashboard = NTStateDashboard()

        curves = dashboard.get_all_saturation_curves()

        assert len(curves) == 6
        for label in NT_LABELS:
            assert label in curves
            conc, sat = curves[label]
            assert len(conc) == 100
            assert len(sat) == 100

    def test_get_opponent_processes(self):
        """Test opponent process dynamics retrieval."""
        dashboard = NTStateDashboard()

        for _ in range(10):
            dashboard.record_state(np.random.rand(6))

        processes = dashboard.get_opponent_processes()

        assert "DA_5HT_ratio" in processes
        assert "Glu_GABA_ratio" in processes
        assert "NE_ACh_balance" in processes

        for key, (timestamps, values) in processes.items():
            assert len(timestamps) == 10
            assert len(values) == 10

    def test_get_opponent_processes_empty(self):
        """Test opponent processes with no data."""
        dashboard = NTStateDashboard()
        processes = dashboard.get_opponent_processes()
        assert processes == {}

    def test_alerts_high_deviation(self):
        """Test alerts for high setpoint deviation."""
        dashboard = NTStateDashboard(alert_deviation=0.2)

        # DA way above setpoint
        high_da = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.5])
        dashboard.record_state(high_da)

        alerts = dashboard.get_alerts()
        assert any("DA" in a and "HIGH" in a for a in alerts)

    def test_alerts_low_deviation(self):
        """Test alerts for low setpoint deviation."""
        dashboard = NTStateDashboard(alert_deviation=0.2)

        # 5-HT way below setpoint
        low_5ht = np.array([0.5, 0.1, 0.5, 0.5, 0.5, 0.5])
        dashboard.record_state(low_5ht)

        alerts = dashboard.get_alerts()
        assert any("5-HT" in a and "LOW" in a for a in alerts)

    def test_alerts_ei_imbalance_excitatory(self):
        """Test E/I imbalance alert (excitatory-dominant)."""
        dashboard = NTStateDashboard()

        # Very low GABA, high Glu
        ei_imbalance = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.8])
        dashboard.record_state(ei_imbalance)

        alerts = dashboard.get_alerts()
        assert any("E/I IMBALANCE" in a and "Excitatory" in a for a in alerts)

    def test_alerts_ei_imbalance_inhibitory(self):
        """Test E/I imbalance alert (inhibitory-dominant)."""
        dashboard = NTStateDashboard()

        # Very high GABA, low Glu
        ei_imbalance = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.3])
        dashboard.record_state(ei_imbalance)

        alerts = dashboard.get_alerts()
        assert any("E/I IMBALANCE" in a and "Inhibitory" in a for a in alerts)

    def test_alerts_high_arousal(self):
        """Test high arousal alert."""
        dashboard = NTStateDashboard()

        # High DA, NE, Glu
        high_arousal = np.array([0.95, 0.5, 0.5, 0.95, 0.5, 0.95])
        dashboard.record_state(high_arousal)

        alerts = dashboard.get_alerts()
        assert any("HIGH AROUSAL" in a for a in alerts)

    def test_alerts_low_arousal(self):
        """Test low arousal alert."""
        dashboard = NTStateDashboard()

        # Low DA, NE, Glu
        low_arousal = np.array([0.05, 0.5, 0.5, 0.05, 0.5, 0.05])
        dashboard.record_state(low_arousal)

        alerts = dashboard.get_alerts()
        assert any("LOW AROUSAL" in a for a in alerts)

    def test_export_data(self):
        """Test data export."""
        dashboard = NTStateDashboard()

        for _ in range(10):
            dashboard.record_state(np.random.rand(6))

        data = dashboard.export_data()

        assert "current_state" in data
        assert "setpoints" in data
        assert "statistics" in data
        assert "nt_labels" in data
        assert "nt_colors" in data
        assert "alerts" in data
        assert "n_samples" in data
        assert data["n_samples"] == 10

    def test_clear_history(self):
        """Test history clearing."""
        dashboard = NTStateDashboard()

        for _ in range(10):
            dashboard.record_state(np.random.rand(6))

        dashboard.clear_history()

        assert len(dashboard._snapshots) == 0
        assert len(dashboard._nt_history) == 0
        assert len(dashboard._active_alerts) == 0


class TestNTSnapshot:
    """Tests for NTSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test snapshot creation."""
        snapshot = NTSnapshot(
            timestamp=datetime.now(),
            nt_state=np.array([0.5] * 6),
            setpoint_deviation=np.array([0.1] * 6),
            receptor_saturation=np.array([0.6] * 6),
            ei_balance=1.0,
            arousal_index=0.5,
        )

        assert snapshot.ei_balance == 1.0
        assert snapshot.arousal_index == 0.5

    def test_snapshot_with_firing_rates(self):
        """Test snapshot with firing rates."""
        snapshot = NTSnapshot(
            timestamp=datetime.now(),
            nt_state=np.array([0.5] * 6),
            setpoint_deviation=np.array([0.0] * 6),
            receptor_saturation=np.array([0.5] * 6),
            ei_balance=1.0,
            arousal_index=0.5,
            vta_firing_rate=5.0,
            raphe_firing_rate=2.5,
        )

        assert snapshot.vta_firing_rate == 5.0
        assert snapshot.raphe_firing_rate == 2.5


class TestNTStatistics:
    """Tests for NTStatistics dataclass."""

    def test_statistics_creation(self):
        """Test statistics creation."""
        stats = NTStatistics(
            mean=np.array([0.5] * 6),
            std=np.array([0.1] * 6),
            min=np.array([0.3] * 6),
            max=np.array([0.7] * 6),
            correlation_matrix=np.eye(6),
            autocorrelation=np.array([0.8] * 6),
        )

        assert stats.mean.shape == (6,)
        assert stats.correlation_matrix.shape == (6, 6)


class TestConstants:
    """Tests for module constants."""

    def test_nt_labels(self):
        """Test NT labels."""
        assert len(NT_LABELS) == 6
        assert NT_LABELS == ["DA", "5-HT", "ACh", "NE", "GABA", "Glu"]

    def test_nt_colors(self):
        """Test NT colors."""
        assert len(NT_COLORS) == 6
        for label in NT_LABELS:
            assert label in NT_COLORS
            assert NT_COLORS[label].startswith("#")

    def test_homeostatic_setpoints(self):
        """Test homeostatic setpoints."""
        assert len(HOMEOSTATIC_SETPOINTS) == 6
        assert all(0 <= s <= 1 for s in HOMEOSTATIC_SETPOINTS)

    def test_receptor_km(self):
        """Test receptor Km values."""
        assert len(RECEPTOR_KM) == 6
        assert all(0 < km < 1 for km in RECEPTOR_KM)


class TestWithNeurotransmitterState:
    """Tests with actual NeurotransmitterState."""

    def test_record_from_nt_state(self):
        """Test recording from NeurotransmitterState object."""
        from t4dm.nca.neural_field import NeurotransmitterState

        dashboard = NTStateDashboard()
        nt_state = NeurotransmitterState(
            dopamine=0.6,
            serotonin=0.4,
            acetylcholine=0.5,
            norepinephrine=0.7,
            gaba=0.5,
            glutamate=0.5,
        )

        snapshot = dashboard.record_from_nt_state(nt_state)

        assert abs(snapshot.nt_state[0] - 0.6) < 1e-5  # DA
        assert abs(snapshot.nt_state[1] - 0.4) < 1e-5  # 5-HT
        assert abs(snapshot.nt_state[3] - 0.7) < 1e-5  # NE

    def test_record_from_nt_state_with_firing_rates(self):
        """Test recording from NeurotransmitterState with firing rates."""
        from t4dm.nca.neural_field import NeurotransmitterState

        dashboard = NTStateDashboard()
        nt_state = NeurotransmitterState()

        snapshot = dashboard.record_from_nt_state(
            nt_state,
            vta_firing_rate=4.5,
            raphe_firing_rate=3.0,
        )

        assert snapshot.vta_firing_rate == 4.5
        assert snapshot.raphe_firing_rate == 3.0


class TestIntegration:
    """Integration tests."""

    def test_full_trajectory_analysis(self):
        """Test full trajectory recording and analysis."""
        dashboard = NTStateDashboard()

        # Simulate trajectory with trends
        for i in range(100):
            t = i / 100
            nt_state = np.array([
                0.5 + 0.2 * np.sin(2 * np.pi * t),      # DA oscillates
                0.5 + 0.1 * t,                          # 5-HT trends up
                0.5,                                     # ACh constant
                0.5 + 0.2 * np.cos(2 * np.pi * t),      # NE oscillates
                0.5,                                     # GABA constant
                0.5,                                     # Glu constant
            ])
            dashboard.record_state(nt_state)

        # Verify statistics
        stats = dashboard.compute_statistics()
        assert stats.mean[0] > 0.4 and stats.mean[0] < 0.6  # DA near 0.5
        assert stats.mean[1] > 0.5  # 5-HT above 0.5 (trending up)

        # Verify autocorrelation
        autocorr = dashboard.get_autocorrelation()
        assert autocorr[0] > 0.8  # DA is smooth (sine)
        assert autocorr[2] < 0.3  # ACh is constant (noise)

        # Verify export
        data = dashboard.export_data()
        assert data["n_samples"] == 100

    def test_opponent_process_tracking(self):
        """Test opponent process dynamics tracking."""
        dashboard = NTStateDashboard()

        # Simulate reward-seeking behavior (high DA, low 5-HT)
        for i in range(50):
            nt_state = np.array([
                0.8,  # High DA
                0.3,  # Low 5-HT
                0.5,
                0.5,
                0.4,  # Low GABA
                0.6,  # High Glu
            ])
            dashboard.record_state(nt_state)

        processes = dashboard.get_opponent_processes()

        # DA/5-HT ratio should be high
        _, da_5ht = processes["DA_5HT_ratio"]
        assert all(r > 2.0 for r in da_5ht)

        # Glu/GABA ratio should be high
        _, glu_gaba = processes["Glu_GABA_ratio"]
        assert all(r > 1.0 for r in glu_gaba)


class TestPlotNTChannels:
    """Tests for plot_nt_channels function."""

    @pytest.fixture
    def populated_dashboard(self):
        """Create dashboard with data."""
        dashboard = NTStateDashboard()
        for i in range(20):
            t = i / 20
            nt_state = np.array([
                0.5 + 0.2 * np.sin(2 * np.pi * t),
                0.5 + 0.1 * np.cos(2 * np.pi * t),
                0.5,
                0.5 + 0.15 * np.sin(np.pi * t),
                0.5,
                0.5,
            ])
            dashboard.record_state(nt_state)
        return dashboard

    def test_plot_nt_channels(self, populated_dashboard):
        """Test NT channels plotting."""
        from t4dm.visualization.nt_state_dashboard import plot_nt_channels
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        fig = plot_nt_channels(populated_dashboard)
        assert fig is not None
        plt.close("all")

    def test_plot_nt_channels_with_setpoints(self, populated_dashboard):
        """Test NT channels with setpoint lines."""
        from t4dm.visualization.nt_state_dashboard import plot_nt_channels
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        fig = plot_nt_channels(populated_dashboard, show_setpoints=True)
        assert fig is not None
        plt.close("all")

    def test_plot_nt_channels_empty(self):
        """Test NT channels with no data."""
        from t4dm.visualization.nt_state_dashboard import plot_nt_channels
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        dashboard = NTStateDashboard()
        result = plot_nt_channels(dashboard)
        assert result is None
        plt.close("all")


class TestPlotDeviationHeatmap:
    """Tests for plot_deviation_heatmap function."""

    @pytest.fixture
    def populated_dashboard(self):
        """Create dashboard with deviation data."""
        dashboard = NTStateDashboard()
        for i in range(50):
            # Vary NT states to create deviations
            nt_state = np.array([
                0.5 + 0.3 * np.sin(i / 10),
                0.5 - 0.2 * np.cos(i / 10),
                0.5,
                0.4 + 0.2 * np.sin(i / 5),
                0.5,
                0.5,
            ])
            dashboard.record_state(nt_state)
        return dashboard

    def test_plot_deviation_heatmap(self, populated_dashboard):
        """Test deviation heatmap plotting."""
        from t4dm.visualization.nt_state_dashboard import plot_deviation_heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_deviation_heatmap(populated_dashboard)
        assert ax is not None
        plt.close("all")

    def test_plot_deviation_heatmap_with_window(self, populated_dashboard):
        """Test deviation heatmap with window."""
        from t4dm.visualization.nt_state_dashboard import plot_deviation_heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_deviation_heatmap(populated_dashboard, window=20)
        assert ax is not None
        plt.close("all")

    def test_plot_deviation_heatmap_empty(self):
        """Test deviation heatmap with no data."""
        from t4dm.visualization.nt_state_dashboard import plot_deviation_heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        dashboard = NTStateDashboard()
        ax = plot_deviation_heatmap(dashboard)
        assert ax is not None  # Returns ax with "No data" text
        plt.close("all")


class TestPlotSaturationCurves:
    """Tests for plot_saturation_curves function."""

    @pytest.fixture
    def populated_dashboard(self):
        """Create dashboard with data."""
        dashboard = NTStateDashboard()
        for _ in range(10):
            dashboard.record_state(np.random.rand(6))
        return dashboard

    def test_plot_saturation_curves(self, populated_dashboard):
        """Test saturation curves plotting."""
        from t4dm.visualization.nt_state_dashboard import plot_saturation_curves
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_saturation_curves(populated_dashboard)
        assert ax is not None
        plt.close("all")

    def test_plot_saturation_curves_show_current(self, populated_dashboard):
        """Test saturation curves with current marker."""
        from t4dm.visualization.nt_state_dashboard import plot_saturation_curves
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_saturation_curves(populated_dashboard, show_current=True)
        assert ax is not None
        plt.close("all")

    def test_plot_saturation_curves_no_current(self, populated_dashboard):
        """Test saturation curves without current marker."""
        from t4dm.visualization.nt_state_dashboard import plot_saturation_curves
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_saturation_curves(populated_dashboard, show_current=False)
        assert ax is not None
        plt.close("all")


class TestPlotCorrelationMatrix:
    """Tests for plot_correlation_matrix function."""

    @pytest.fixture
    def populated_dashboard(self):
        """Create dashboard with correlated data."""
        dashboard = NTStateDashboard()
        for i in range(50):
            da = np.random.rand()
            nt_state = np.array([da, da * 0.8 + 0.1, 0.5, 0.5, 0.5, 0.5])
            dashboard.record_state(nt_state)
        return dashboard

    def test_plot_correlation_matrix(self, populated_dashboard):
        """Test correlation matrix plotting."""
        from t4dm.visualization.nt_state_dashboard import plot_correlation_matrix
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_correlation_matrix(populated_dashboard)
        assert ax is not None
        plt.close("all")


class TestPlotAutocorrelation:
    """Tests for plot_autocorrelation function."""

    @pytest.fixture
    def populated_dashboard(self):
        """Create dashboard with smooth trajectory."""
        dashboard = NTStateDashboard()
        for i in range(50):
            t = i / 50
            nt_state = np.array([
                0.5 + 0.3 * np.sin(2 * np.pi * t),
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ])
            dashboard.record_state(nt_state)
        return dashboard

    def test_plot_autocorrelation(self, populated_dashboard):
        """Test autocorrelation plotting."""
        from t4dm.visualization.nt_state_dashboard import plot_autocorrelation
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_autocorrelation(populated_dashboard)
        assert ax is not None
        plt.close("all")


class TestPlotOpponentProcesses:
    """Tests for plot_opponent_processes function."""

    @pytest.fixture
    def populated_dashboard(self):
        """Create dashboard with opponent process dynamics."""
        dashboard = NTStateDashboard()
        for i in range(30):
            nt_state = np.array([
                0.6 + 0.2 * np.sin(i / 10),  # DA
                0.4 + 0.1 * np.cos(i / 10),  # 5-HT
                0.5,                          # ACh
                0.5 + 0.15 * np.sin(i / 5),  # NE
                0.5,                          # GABA
                0.5,                          # Glu
            ])
            dashboard.record_state(nt_state)
        return dashboard

    def test_plot_opponent_processes(self, populated_dashboard):
        """Test opponent processes plotting."""
        from t4dm.visualization.nt_state_dashboard import plot_opponent_processes
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        fig = plot_opponent_processes(populated_dashboard)
        assert fig is not None
        plt.close("all")

    def test_plot_opponent_processes_empty(self):
        """Test opponent processes with no data."""
        from t4dm.visualization.nt_state_dashboard import plot_opponent_processes
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        dashboard = NTStateDashboard()
        result = plot_opponent_processes(dashboard)
        assert result is None
        plt.close("all")


class TestCreateNTDashboard:
    """Tests for create_nt_dashboard function."""

    @pytest.fixture
    def populated_dashboard(self):
        """Create dashboard with comprehensive data."""
        dashboard = NTStateDashboard()
        for i in range(100):
            t = i / 100
            nt_state = np.array([
                0.5 + 0.2 * np.sin(2 * np.pi * t),
                0.5 + 0.1 * np.cos(2 * np.pi * t),
                0.5 + 0.05 * np.sin(4 * np.pi * t),
                0.5 + 0.15 * np.sin(np.pi * t),
                0.5,
                0.5 + 0.1 * np.cos(np.pi * t),
            ])
            dashboard.record_state(nt_state)
        return dashboard

    def test_create_nt_dashboard(self, populated_dashboard):
        """Test full dashboard creation."""
        from t4dm.visualization.nt_state_dashboard import create_nt_dashboard
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        fig = create_nt_dashboard(populated_dashboard)
        assert fig is not None
        plt.close("all")

    def test_create_nt_dashboard_custom_size(self, populated_dashboard):
        """Test dashboard with custom figure size."""
        from t4dm.visualization.nt_state_dashboard import create_nt_dashboard
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        fig = create_nt_dashboard(populated_dashboard, figsize=(12, 10))
        assert fig is not None
        plt.close("all")

    def test_create_nt_dashboard_with_alerts(self):
        """Test dashboard with active alerts."""
        from t4dm.visualization.nt_state_dashboard import create_nt_dashboard
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        dashboard = NTStateDashboard(alert_deviation=0.1)
        for i in range(50):
            # Create states that trigger alerts
            nt_state = np.array([0.9, 0.1, 0.5, 0.9, 0.3, 0.8])
            dashboard.record_state(nt_state)

        fig = create_nt_dashboard(dashboard)
        assert fig is not None
        assert len(dashboard.get_alerts()) > 0
        plt.close("all")


class TestNTDashboardEdgeCases:
    """Edge case tests for NT state dashboard visualization."""

    def test_single_sample(self):
        """Test with single sample."""
        from t4dm.visualization.nt_state_dashboard import (
            plot_nt_channels,
            plot_deviation_heatmap,
            create_nt_dashboard,
        )
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        dashboard = NTStateDashboard()
        dashboard.record_state(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))

        fig = plot_nt_channels(dashboard)
        assert fig is not None
        plt.close("all")

        ax = plot_deviation_heatmap(dashboard)
        assert ax is not None
        plt.close("all")

        fig = create_nt_dashboard(dashboard)
        assert fig is not None
        plt.close("all")

    def test_extreme_values(self):
        """Test with extreme NT values."""
        from t4dm.visualization.nt_state_dashboard import create_nt_dashboard
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        dashboard = NTStateDashboard()
        for _ in range(20):
            nt_state = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            dashboard.record_state(nt_state)

        fig = create_nt_dashboard(dashboard)
        assert fig is not None
        plt.close("all")

    def test_constant_values(self):
        """Test with constant NT values."""
        from t4dm.visualization.nt_state_dashboard import (
            plot_correlation_matrix,
            plot_autocorrelation,
        )
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        dashboard = NTStateDashboard()
        for _ in range(30):
            nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            dashboard.record_state(nt_state)

        ax = plot_correlation_matrix(dashboard)
        assert ax is not None
        plt.close("all")

        ax = plot_autocorrelation(dashboard)
        assert ax is not None
        plt.close("all")

    def test_near_zero_denominator(self):
        """Test E/I balance with near-zero Glu."""
        dashboard = NTStateDashboard()
        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.001])
        snapshot = dashboard.record_state(nt_state)

        # Should not raise division by zero
        assert np.isfinite(snapshot.ei_balance)
        assert snapshot.ei_balance > 1.0  # High GABA / tiny Glu

    def test_all_plots_with_firing_rates(self):
        """Test plots with firing rate data."""
        from t4dm.visualization.nt_state_dashboard import create_nt_dashboard
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        dashboard = NTStateDashboard()
        for i in range(50):
            dashboard.record_state(
                np.random.rand(6),
                vta_firing_rate=5.0 + np.random.rand(),
                raphe_firing_rate=2.5 + np.random.rand(),
            )

        fig = create_nt_dashboard(dashboard)
        assert fig is not None
        plt.close("all")
