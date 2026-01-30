"""
Tests for Coupling Dynamics Visualization.

Tests the core functionality of the coupling dynamics visualizer:
- State recording and snapshot creation
- Spectral radius tracking (stability monitoring)
- E/I balance computation
- Eligibility trace visualization
- Bounds violation detection
- Learning event recording
"""

import numpy as np
import pytest
from datetime import datetime

from ww.visualization.coupling_dynamics import (
    CouplingDynamicsVisualizer,
    CouplingSnapshot,
    LearningEvent,
    NT_LABELS,
)


class TestCouplingDynamicsVisualizer:
    """Tests for CouplingDynamicsVisualizer class."""

    def test_init_default(self):
        """Test default initialization."""
        viz = CouplingDynamicsVisualizer()
        assert viz.window_size == 1000
        assert viz.alert_spectral_radius == 0.95
        assert viz.alert_ei_imbalance == 0.5
        assert len(viz._snapshots) == 0
        assert len(viz._learning_events) == 0

    def test_init_custom(self):
        """Test custom initialization."""
        viz = CouplingDynamicsVisualizer(
            window_size=100,
            alert_spectral_radius=0.8,
            alert_ei_imbalance=0.3,
        )
        assert viz.window_size == 100
        assert viz.alert_spectral_radius == 0.8
        assert viz.alert_ei_imbalance == 0.3

    def test_record_state_basic(self):
        """Test basic state recording."""
        viz = CouplingDynamicsVisualizer()
        K = np.random.randn(6, 6) * 0.1

        snapshot = viz.record_state(K=K)

        assert isinstance(snapshot, CouplingSnapshot)
        assert len(viz._snapshots) == 1
        assert np.allclose(snapshot.K, K)

    def test_record_state_with_eligibility(self):
        """Test state recording with eligibility trace."""
        viz = CouplingDynamicsVisualizer()
        K = np.random.randn(6, 6) * 0.1
        eligibility = np.random.rand(6, 6) * 0.5

        snapshot = viz.record_state(K=K, eligibility_trace=eligibility)

        assert snapshot.eligibility_entropy >= 0
        assert snapshot.eligibility_entropy <= 1

    def test_record_state_with_gradient(self):
        """Test state recording with gradient."""
        viz = CouplingDynamicsVisualizer()
        K = np.random.randn(6, 6) * 0.1
        gradient = np.random.randn(6, 6) * 0.01

        snapshot = viz.record_state(K=K, gradient=gradient)

        assert snapshot.gradient_norm > 0

    def test_record_state_window_limit(self):
        """Test that snapshots respect window size."""
        viz = CouplingDynamicsVisualizer(window_size=10)

        for i in range(20):
            K = np.random.randn(6, 6) * 0.1
            viz.record_state(K=K)

        assert len(viz._snapshots) == 10
        assert len(viz._K_history) == 10

    def test_spectral_radius_computation(self):
        """Test spectral radius is computed correctly."""
        viz = CouplingDynamicsVisualizer()

        # Create matrix with known spectral radius
        K = np.diag([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        snapshot = viz.record_state(K=K)

        assert abs(snapshot.spectral_radius - 0.5) < 1e-6

    def test_spectral_radius_stability(self):
        """Test stable vs unstable detection."""
        viz = CouplingDynamicsVisualizer()

        # Stable matrix (spectral radius < 1)
        K_stable = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        viz.record_state(K=K_stable)
        assert viz.is_stable()

        # Unstable matrix (spectral radius > 1)
        viz.clear_history()
        K_unstable = np.diag([1.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        viz.record_state(K=K_unstable)
        assert not viz.is_stable()

    def test_ei_balance_computation(self):
        """Test E/I balance is computed correctly."""
        viz = CouplingDynamicsVisualizer()

        K = np.zeros((6, 6))
        K[4, 5] = -0.3  # GABA -> Glu (inhibitory)
        K[5, 4] = -0.2  # Glu -> GABA (inhibitory)

        snapshot = viz.record_state(K=K)

        assert abs(snapshot.ei_balance - (-0.5)) < 1e-6

    def test_get_spectral_radius_trace(self):
        """Test spectral radius trace retrieval."""
        viz = CouplingDynamicsVisualizer()

        for i in range(10):
            K = np.diag([0.1 * i] * 6)
            viz.record_state(K=K)

        timestamps, radii = viz.get_spectral_radius_trace()

        assert len(timestamps) == 10
        assert len(radii) == 10
        assert all(isinstance(t, datetime) for t in timestamps)
        # Radii should increase
        assert radii[-1] > radii[0]

    def test_get_spectral_radius_trace_empty(self):
        """Test spectral radius trace with no data."""
        viz = CouplingDynamicsVisualizer()
        timestamps, radii = viz.get_spectral_radius_trace()
        assert timestamps == []
        assert radii == []

    def test_get_ei_balance_trace(self):
        """Test E/I balance trace retrieval."""
        viz = CouplingDynamicsVisualizer()

        for i in range(10):
            K = np.zeros((6, 6))
            K[4, 5] = -0.1 * (i + 1)
            K[5, 4] = -0.05 * (i + 1)
            viz.record_state(K=K)

        timestamps, balances = viz.get_ei_balance_trace()

        assert len(timestamps) == 10
        assert len(balances) == 10
        # All should be negative (inhibitory-dominant)
        assert all(b < 0 for b in balances)

    def test_get_current_K(self):
        """Test getting current coupling matrix."""
        viz = CouplingDynamicsVisualizer()

        K = np.random.randn(6, 6) * 0.1
        viz.record_state(K=K)

        current = viz.get_current_K()
        assert np.allclose(current, K)

    def test_get_current_K_empty(self):
        """Test getting current K with no data."""
        viz = CouplingDynamicsVisualizer()
        assert viz.get_current_K() is None

    def test_get_K_bounds(self):
        """Test getting K bounds."""
        viz = CouplingDynamicsVisualizer()

        K_min, K_max = viz.get_K_bounds()

        assert K_min.shape == (6, 6)
        assert K_max.shape == (6, 6)
        assert np.all(K_min <= K_max)

    def test_get_K_normalized(self):
        """Test normalized K computation."""
        viz = CouplingDynamicsVisualizer()

        K = np.zeros((6, 6))
        viz.record_state(K=K)

        K_norm = viz.get_K_normalized()
        assert K_norm is not None
        # All values should be in [-1, 1]
        assert np.all(K_norm >= -1)
        assert np.all(K_norm <= 1)

    def test_get_coupling_change_rate(self):
        """Test coupling change rate computation."""
        viz = CouplingDynamicsVisualizer()

        # Record sequence of changing K matrices
        for i in range(10):
            K = np.ones((6, 6)) * 0.1 * i
            viz.record_state(K=K)

        rate = viz.get_coupling_change_rate(window=5)

        assert rate.shape == (6, 6)
        # All entries should have same change rate
        assert np.allclose(rate, 0.1, atol=1e-5)

    def test_get_coupling_change_rate_insufficient_data(self):
        """Test change rate with insufficient data."""
        viz = CouplingDynamicsVisualizer()
        rate = viz.get_coupling_change_rate()
        assert np.allclose(rate, 0)

    def test_get_eigenvalue_evolution(self):
        """Test eigenvalue evolution retrieval."""
        viz = CouplingDynamicsVisualizer()

        for _ in range(5):
            K = np.random.randn(6, 6) * 0.1
            viz.record_state(K=K)

        evolution = viz.get_eigenvalue_evolution()

        assert "max_real" in evolution
        assert "min_real" in evolution
        assert "spectral_radius" in evolution
        assert len(evolution["spectral_radius"]) == 5

    def test_get_current_eigenvalues(self):
        """Test getting current eigenvalues."""
        viz = CouplingDynamicsVisualizer()

        K = np.diag([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        viz.record_state(K=K)

        eigenvalues = viz.get_current_eigenvalues()
        assert len(eigenvalues) == 6
        # For diagonal matrix, eigenvalues are diagonal entries
        assert set(np.round(eigenvalues, 2)) == {0.5, 0.4, 0.3, 0.2, 0.1, 0.05}

    def test_get_current_eigenvalues_empty(self):
        """Test getting eigenvalues with no data."""
        viz = CouplingDynamicsVisualizer()
        eigenvalues = viz.get_current_eigenvalues()
        assert len(eigenvalues) == 0

    def test_get_ei_components(self):
        """Test E/I component breakdown."""
        viz = CouplingDynamicsVisualizer()

        K = np.zeros((6, 6))
        K[4, 5] = -0.3  # GABA->Glu
        K[5, 4] = -0.2  # Glu->GABA
        K[0, 1] = 0.5   # DA->5-HT (positive)
        viz.record_state(K=K)

        components = viz.get_ei_components()

        assert "gaba_to_glu" in components
        assert "glu_to_gaba" in components
        assert abs(components["gaba_to_glu"] - (-0.3)) < 1e-6
        assert abs(components["glu_to_gaba"] - (-0.2)) < 1e-6

    def test_get_ei_components_empty(self):
        """Test E/I components with no data."""
        viz = CouplingDynamicsVisualizer()
        components = viz.get_ei_components()
        assert components == {}

    def test_eligibility_entropy_computation(self):
        """Test eligibility entropy is computed correctly."""
        viz = CouplingDynamicsVisualizer()

        # Uniform eligibility -> max entropy
        K = np.zeros((6, 6))
        uniform_elig = np.ones((6, 6))
        snapshot_uniform = viz.record_state(K=K, eligibility_trace=uniform_elig)

        viz.clear_history()

        # Concentrated eligibility -> low entropy
        concentrated_elig = np.zeros((6, 6))
        concentrated_elig[0, 0] = 1.0
        snapshot_concentrated = viz.record_state(K=K, eligibility_trace=concentrated_elig)

        assert snapshot_uniform.eligibility_entropy > snapshot_concentrated.eligibility_entropy

    def test_get_eligibility_entropy_trace(self):
        """Test eligibility entropy trace retrieval."""
        viz = CouplingDynamicsVisualizer()

        for _ in range(5):
            K = np.zeros((6, 6))
            elig = np.random.rand(6, 6)
            viz.record_state(K=K, eligibility_trace=elig)

        timestamps, entropies = viz.get_eligibility_entropy_trace()

        assert len(timestamps) == 5
        assert len(entropies) == 5
        assert all(0 <= e <= 1 for e in entropies)

    def test_get_credit_flow_empty(self):
        """Test credit flow with no eligibility data."""
        viz = CouplingDynamicsVisualizer()
        flow = viz.get_credit_flow()
        assert flow == {}

    def test_get_learning_activity_trace(self):
        """Test learning activity trace retrieval."""
        viz = CouplingDynamicsVisualizer()

        for i in range(10):
            K = np.zeros((6, 6))
            gradient = np.random.randn(6, 6) * 0.01 * (i + 1)
            viz.record_state(K=K, gradient=gradient)

        timestamps, norms = viz.get_learning_activity_trace()

        assert len(timestamps) == 10
        assert len(norms) == 10
        # Norms should increase
        assert norms[-1] > norms[0]

    def test_get_frobenius_norm_trace(self):
        """Test Frobenius norm trace retrieval."""
        viz = CouplingDynamicsVisualizer()

        for i in range(5):
            K = np.ones((6, 6)) * 0.1 * (i + 1)
            viz.record_state(K=K)

        timestamps, norms = viz.get_frobenius_norm_trace()

        assert len(timestamps) == 5
        assert len(norms) == 5
        # Norms should increase
        assert norms[-1] > norms[0]

    def test_record_learning_event(self):
        """Test learning event recording."""
        viz = CouplingDynamicsVisualizer()

        K_before = np.zeros((6, 6))
        K_after = np.ones((6, 6)) * 0.1
        gradient = np.random.randn(6, 6) * 0.01

        event = viz.record_learning_event(
            update_type="rpe",
            gradient=gradient,
            K_before=K_before,
            K_after=K_after,
            rpe=0.5,
        )

        assert isinstance(event, LearningEvent)
        assert event.update_type == "rpe"
        assert event.rpe == 0.5
        assert event.coupling_change > 0
        assert len(viz._learning_events) == 1

    def test_record_learning_event_window_limit(self):
        """Test learning event window limit."""
        viz = CouplingDynamicsVisualizer(window_size=5)

        for i in range(10):
            viz.record_learning_event(
                update_type="energy",
                gradient=np.random.randn(6, 6),
                K_before=np.zeros((6, 6)),
                K_after=np.zeros((6, 6)),
            )

        assert len(viz._learning_events) == 5

    def test_get_learning_summary(self):
        """Test learning summary retrieval."""
        viz = CouplingDynamicsVisualizer()

        # Add RPE events
        for _ in range(3):
            viz.record_learning_event(
                update_type="rpe",
                gradient=np.random.randn(6, 6),
                K_before=np.zeros((6, 6)),
                K_after=np.ones((6, 6)) * 0.1,
                rpe=0.5,
            )

        # Add energy events
        for _ in range(2):
            viz.record_learning_event(
                update_type="energy",
                gradient=np.random.randn(6, 6),
                K_before=np.zeros((6, 6)),
                K_after=np.ones((6, 6)) * 0.1,
            )

        summary = viz.get_learning_summary()

        assert summary["total_updates"] == 5
        assert summary["rpe_updates"] == 3
        assert summary["energy_updates"] == 2

    def test_get_learning_summary_empty(self):
        """Test learning summary with no events."""
        viz = CouplingDynamicsVisualizer()
        summary = viz.get_learning_summary()
        assert summary["total_updates"] == 0

    def test_get_stability_metrics(self):
        """Test stability metrics retrieval."""
        viz = CouplingDynamicsVisualizer()

        for _ in range(20):
            K = np.diag([0.5] * 6)
            viz.record_state(K=K)

        metrics = viz.get_stability_metrics()

        assert "current_spectral_radius" in metrics
        assert "mean_spectral_radius" in metrics
        assert "is_stable" in metrics
        assert "stability_margin" in metrics
        assert metrics["is_stable"] is True

    def test_get_stability_metrics_empty(self):
        """Test stability metrics with no data."""
        viz = CouplingDynamicsVisualizer()
        metrics = viz.get_stability_metrics()
        assert metrics == {}

    def test_alerts_spectral_radius(self):
        """Test spectral radius alerts."""
        viz = CouplingDynamicsVisualizer(alert_spectral_radius=0.5)

        # Below threshold - no alert
        K = np.diag([0.3] * 6)
        viz.record_state(K=K)
        alerts = viz.get_alerts()
        assert not any("STABILITY" in a for a in alerts)

        # Above threshold - alert
        viz.clear_history()
        K = np.diag([0.8] * 6)
        viz.record_state(K=K)
        alerts = viz.get_alerts()
        assert any("STABILITY" in a for a in alerts)

    def test_alerts_ei_balance(self):
        """Test E/I balance alerts."""
        viz = CouplingDynamicsVisualizer(alert_ei_imbalance=0.3)

        # Good E/I balance - no alert
        K = np.zeros((6, 6))
        K[4, 5] = -0.5
        K[5, 4] = -0.3
        viz.record_state(K=K)
        alerts = viz.get_alerts()
        assert not any("E/I BALANCE" in a for a in alerts)

        # Bad E/I balance - alert
        viz.clear_history()
        K = np.zeros((6, 6))
        K[4, 5] = 0.1
        K[5, 4] = 0.2
        viz.record_state(K=K)
        alerts = viz.get_alerts()
        assert any("E/I BALANCE" in a for a in alerts)

    def test_export_data(self):
        """Test data export."""
        viz = CouplingDynamicsVisualizer()

        for _ in range(10):
            K = np.random.randn(6, 6) * 0.1
            viz.record_state(K=K)

        data = viz.export_data()

        assert "coupling_matrix" in data
        assert "eligibility_trace" in data
        assert "spectral_radius" in data
        assert "ei_balance" in data
        assert "stability_metrics" in data
        assert "learning_summary" in data
        assert "alerts" in data
        assert "change_rate" in data

    def test_clear_history(self):
        """Test history clearing."""
        viz = CouplingDynamicsVisualizer()

        for _ in range(10):
            K = np.random.randn(6, 6) * 0.1
            viz.record_state(K=K)
            viz.record_learning_event(
                update_type="test",
                gradient=np.random.randn(6, 6),
                K_before=np.zeros((6, 6)),
                K_after=K,
            )

        viz.clear_history()

        assert len(viz._snapshots) == 0
        assert len(viz._learning_events) == 0
        assert len(viz._K_history) == 0
        assert len(viz._active_alerts) == 0


class TestCouplingSnapshot:
    """Tests for CouplingSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test snapshot creation."""
        snapshot = CouplingSnapshot(
            timestamp=datetime.now(),
            K=np.zeros((6, 6)),
            frobenius_norm=0.5,
            spectral_radius=0.4,
            max_eigenvalue_real=0.3,
            min_eigenvalue_real=-0.1,
            ei_balance=-0.5,
            bounds_violations=0,
            gradient_norm=0.01,
            eligibility_entropy=0.8,
        )

        assert snapshot.spectral_radius == 0.4
        assert snapshot.ei_balance == -0.5


class TestLearningEvent:
    """Tests for LearningEvent dataclass."""

    def test_learning_event_creation(self):
        """Test learning event creation."""
        event = LearningEvent(
            timestamp=datetime.now(),
            update_type="rpe",
            gradient_norm=0.01,
            coupling_change=0.05,
            rpe=0.5,
        )

        assert event.update_type == "rpe"
        assert event.rpe == 0.5

    def test_learning_event_no_rpe(self):
        """Test learning event without RPE."""
        event = LearningEvent(
            timestamp=datetime.now(),
            update_type="energy",
            gradient_norm=0.01,
            coupling_change=0.02,
        )

        assert event.rpe is None


class TestWithCoupling:
    """Tests with actual LearnableCoupling."""

    def test_with_learnable_coupling(self):
        """Test with LearnableCoupling instance."""
        from ww.nca.coupling import LearnableCoupling

        coupling = LearnableCoupling()
        viz = CouplingDynamicsVisualizer(coupling=coupling)

        snapshot = viz.record_state()

        assert snapshot.K.shape == (6, 6)
        assert np.isfinite(snapshot.spectral_radius)
        assert np.isfinite(snapshot.frobenius_norm)

    def test_bounds_violation_detection(self):
        """Test bounds violation detection with coupling."""
        from ww.nca.coupling import LearnableCoupling

        coupling = LearnableCoupling()
        viz = CouplingDynamicsVisualizer(coupling=coupling)

        # Record state - coupling matrix is initialized to biological values
        snapshot = viz.record_state()

        assert isinstance(snapshot.bounds_violations, int)
        assert snapshot.bounds_violations >= 0

    def test_eligibility_from_coupling(self):
        """Test eligibility trace from coupling."""
        from ww.nca.coupling import LearnableCoupling

        coupling = LearnableCoupling()
        viz = CouplingDynamicsVisualizer(coupling=coupling)

        # Simulate some activity to build eligibility with RPE
        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        for _ in range(5):
            rpe = np.random.randn() * 0.1
            coupling.update_with_eligibility(nt_state, rpe=rpe)

        trace = viz.get_eligibility_trace()
        assert trace is not None
        assert trace.shape == (6, 6)

    def test_get_k_bounds_from_coupling(self):
        """Test getting bounds from coupling."""
        from ww.nca.coupling import LearnableCoupling

        coupling = LearnableCoupling()
        viz = CouplingDynamicsVisualizer(coupling=coupling)

        K_min, K_max = viz.get_K_bounds()

        assert K_min.shape == (6, 6)
        assert K_max.shape == (6, 6)
        # Verify biologically valid structure
        assert np.all(K[4, :4] <= 0 for K in [coupling.K])  # GABA inhibitory


class TestNTLabels:
    """Tests for NT label constants."""

    def test_nt_labels(self):
        """Test NT labels are correct."""
        assert len(NT_LABELS) == 6
        assert NT_LABELS == ["DA", "5-HT", "ACh", "NE", "GABA", "Glu"]


class TestIntegration:
    """Integration tests with full NCA components."""

    def test_full_integration(self):
        """Test with full NCA integration."""
        from ww.nca.coupling import LearnableCoupling
        from ww.nca.energy import EnergyLandscape

        coupling = LearnableCoupling()
        viz = CouplingDynamicsVisualizer(coupling=coupling)

        # Simulate learning trajectory
        for i in range(20):
            nt_state = np.random.rand(6) * 0.5 + 0.25
            rpe = np.random.randn() * 0.1
            coupling.update_with_eligibility(nt_state, rpe=rpe)

            # Record with simulated gradient
            gradient = np.random.randn(6, 6) * 0.01
            viz.record_state(gradient=gradient)

        # Verify all components work
        metrics = viz.get_stability_metrics()
        assert "current_spectral_radius" in metrics

        data = viz.export_data()
        assert len(data["spectral_radius"]["values"]) == 20

    def test_learning_with_rpe(self):
        """Test learning with RPE updates."""
        from ww.nca.coupling import LearnableCoupling

        coupling = LearnableCoupling()
        viz = CouplingDynamicsVisualizer(coupling=coupling)

        for i in range(5):
            K_before = coupling.K.copy()

            # Simulate learning - update_with_eligibility handles everything
            nt_state = np.random.rand(6) * 0.5 + 0.25
            rpe = np.random.randn() * 0.1
            coupling.update_with_eligibility(nt_state, rpe=rpe)

            K_after = coupling.K.copy()

            viz.record_learning_event(
                update_type="rpe",
                gradient=np.zeros((6, 6)),  # Simplified - gradient tracking is internal
                K_before=K_before,
                K_after=K_after,
                rpe=rpe,
            )

        summary = viz.get_learning_summary()
        assert summary["rpe_updates"] == 5


class TestPlotCouplingHeatmap:
    """Tests for plot_coupling_heatmap function."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with data."""
        viz = CouplingDynamicsVisualizer()
        for _ in range(10):
            K = np.random.randn(6, 6) * 0.1
            viz.record_state(K=K)
        return viz

    def test_plot_coupling_heatmap(self, populated_visualizer):
        """Test coupling heatmap plotting."""
        from ww.visualization.coupling_dynamics import plot_coupling_heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_coupling_heatmap(populated_visualizer)
        assert ax is not None
        plt.close("all")

    def test_plot_coupling_heatmap_with_bounds(self, populated_visualizer):
        """Test heatmap with bounds indicators."""
        from ww.visualization.coupling_dynamics import plot_coupling_heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_coupling_heatmap(populated_visualizer, show_bounds=True)
        assert ax is not None
        plt.close("all")

    def test_plot_coupling_heatmap_with_values(self, populated_visualizer):
        """Test heatmap with value annotations."""
        from ww.visualization.coupling_dynamics import plot_coupling_heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_coupling_heatmap(populated_visualizer, show_values=True)
        assert ax is not None
        plt.close("all")

    def test_plot_coupling_heatmap_empty(self):
        """Test heatmap with no data."""
        from ww.visualization.coupling_dynamics import plot_coupling_heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        ax = plot_coupling_heatmap(viz)
        assert ax is not None  # Shows "No data"
        plt.close("all")


class TestPlotEligibilityHeatmap:
    """Tests for plot_eligibility_heatmap function."""

    @pytest.fixture
    def visualizer_with_coupling(self):
        """Create visualizer with LearnableCoupling."""
        from ww.nca.coupling import LearnableCoupling

        coupling = LearnableCoupling()
        viz = CouplingDynamicsVisualizer(coupling=coupling)

        # Generate some eligibility
        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        for _ in range(5):
            rpe = np.random.randn() * 0.1
            coupling.update_with_eligibility(nt_state, rpe=rpe)

        return viz

    def test_plot_eligibility_heatmap(self, visualizer_with_coupling):
        """Test eligibility heatmap plotting."""
        from ww.visualization.coupling_dynamics import plot_eligibility_heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_eligibility_heatmap(visualizer_with_coupling)
        assert ax is not None
        plt.close("all")

    def test_plot_eligibility_heatmap_no_coupling(self):
        """Test eligibility heatmap without coupling."""
        from ww.visualization.coupling_dynamics import plot_eligibility_heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        ax = plot_eligibility_heatmap(viz)
        assert ax is not None  # Shows "No eligibility data"
        plt.close("all")


class TestPlotSpectralRadiusTimeline:
    """Tests for plot_spectral_radius_timeline function."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with spectral data."""
        viz = CouplingDynamicsVisualizer()
        for i in range(20):
            K = np.diag([0.3 + 0.02 * i] * 6)
            viz.record_state(K=K)
        return viz

    def test_plot_spectral_radius_timeline(self, populated_visualizer):
        """Test spectral radius timeline plotting."""
        from ww.visualization.coupling_dynamics import plot_spectral_radius_timeline
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_spectral_radius_timeline(populated_visualizer)
        assert ax is not None
        plt.close("all")

    def test_plot_spectral_radius_with_threshold(self, populated_visualizer):
        """Test spectral radius timeline with threshold line."""
        from ww.visualization.coupling_dynamics import plot_spectral_radius_timeline
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_spectral_radius_timeline(populated_visualizer, show_threshold=True)
        assert ax is not None
        plt.close("all")

    def test_plot_spectral_radius_empty(self):
        """Test spectral radius timeline with no data."""
        from ww.visualization.coupling_dynamics import plot_spectral_radius_timeline
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        ax = plot_spectral_radius_timeline(viz)
        assert ax is not None  # Shows "No data"
        plt.close("all")


class TestPlotEIBalanceTimeline:
    """Tests for plot_ei_balance_timeline function."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with E/I data."""
        viz = CouplingDynamicsVisualizer()
        for i in range(20):
            K = np.zeros((6, 6))
            K[4, 5] = -0.3 - 0.01 * i  # GABA->Glu
            K[5, 4] = -0.2 - 0.01 * i  # Glu->GABA
            viz.record_state(K=K)
        return viz

    def test_plot_ei_balance_timeline(self, populated_visualizer):
        """Test E/I balance timeline plotting."""
        from ww.visualization.coupling_dynamics import plot_ei_balance_timeline
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_ei_balance_timeline(populated_visualizer)
        assert ax is not None
        plt.close("all")

    def test_plot_ei_balance_empty(self):
        """Test E/I balance timeline with no data."""
        from ww.visualization.coupling_dynamics import plot_ei_balance_timeline
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        ax = plot_ei_balance_timeline(viz)
        assert ax is not None  # Shows "No data"
        plt.close("all")


class TestPlotEigenvalueSpectrum:
    """Tests for plot_eigenvalue_spectrum function."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with eigenvalue data."""
        viz = CouplingDynamicsVisualizer()
        K = np.random.randn(6, 6) * 0.3
        viz.record_state(K=K)
        return viz

    def test_plot_eigenvalue_spectrum(self, populated_visualizer):
        """Test eigenvalue spectrum plotting."""
        from ww.visualization.coupling_dynamics import plot_eigenvalue_spectrum
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ax = plot_eigenvalue_spectrum(populated_visualizer)
        assert ax is not None
        plt.close("all")

    def test_plot_eigenvalue_spectrum_empty(self):
        """Test eigenvalue spectrum with no data."""
        from ww.visualization.coupling_dynamics import plot_eigenvalue_spectrum
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        ax = plot_eigenvalue_spectrum(viz)
        assert ax is not None  # Shows "No data"
        plt.close("all")


class TestCreateCouplingDashboard:
    """Tests for create_coupling_dashboard function."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with comprehensive data."""
        from ww.nca.coupling import LearnableCoupling

        coupling = LearnableCoupling()
        viz = CouplingDynamicsVisualizer(coupling=coupling)

        for i in range(30):
            nt_state = np.random.rand(6) * 0.5 + 0.25
            rpe = np.random.randn() * 0.1
            coupling.update_with_eligibility(nt_state, rpe=rpe)
            gradient = np.random.randn(6, 6) * 0.01
            viz.record_state(gradient=gradient)

        return viz

    def test_create_coupling_dashboard(self, populated_visualizer):
        """Test full dashboard creation."""
        from ww.visualization.coupling_dynamics import create_coupling_dashboard
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        fig = create_coupling_dashboard(populated_visualizer)
        assert fig is not None
        plt.close("all")

    def test_create_coupling_dashboard_custom_size(self, populated_visualizer):
        """Test dashboard with custom figure size."""
        from ww.visualization.coupling_dynamics import create_coupling_dashboard
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        fig = create_coupling_dashboard(populated_visualizer, figsize=(12, 10))
        assert fig is not None
        plt.close("all")

    def test_create_coupling_dashboard_with_alerts(self):
        """Test dashboard with active alerts."""
        from ww.visualization.coupling_dynamics import create_coupling_dashboard
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer(alert_spectral_radius=0.3)
        for _ in range(20):
            K = np.diag([0.5] * 6)
            viz.record_state(K=K)

        fig = create_coupling_dashboard(viz)
        assert fig is not None
        assert len(viz.get_alerts()) > 0
        plt.close("all")


class TestCouplingDynamicsEdgeCases:
    """Edge case tests for coupling dynamics visualization."""

    def test_single_sample(self):
        """Test with single sample."""
        from ww.visualization.coupling_dynamics import (
            plot_coupling_heatmap,
            plot_spectral_radius_timeline,
            create_coupling_dashboard,
        )
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        K = np.random.randn(6, 6) * 0.1
        viz.record_state(K=K)

        ax = plot_coupling_heatmap(viz)
        assert ax is not None
        plt.close("all")

        ax = plot_spectral_radius_timeline(viz)
        assert ax is not None
        plt.close("all")

        fig = create_coupling_dashboard(viz)
        assert fig is not None
        plt.close("all")

    def test_extreme_eigenvalues(self):
        """Test with extreme eigenvalues."""
        from ww.visualization.coupling_dynamics import plot_eigenvalue_spectrum
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        # Create matrix with large eigenvalues
        K = np.diag([1.5, -1.2, 0.8, -0.5, 0.3, 0.1])
        viz.record_state(K=K)

        ax = plot_eigenvalue_spectrum(viz)
        assert ax is not None
        plt.close("all")

    def test_diagonal_matrix(self):
        """Test with diagonal coupling matrix."""
        from ww.visualization.coupling_dynamics import (
            plot_coupling_heatmap,
            plot_eigenvalue_spectrum,
        )
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        K = np.diag([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        viz.record_state(K=K)

        ax = plot_coupling_heatmap(viz)
        assert ax is not None
        plt.close("all")

        ax = plot_eigenvalue_spectrum(viz)
        assert ax is not None
        plt.close("all")

    def test_zero_matrix(self):
        """Test with zero coupling matrix."""
        from ww.visualization.coupling_dynamics import create_coupling_dashboard
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        K = np.zeros((6, 6))
        viz.record_state(K=K)

        fig = create_coupling_dashboard(viz)
        assert fig is not None
        plt.close("all")

    def test_constant_spectral_radius(self):
        """Test timeline with constant spectral radius."""
        from ww.visualization.coupling_dynamics import plot_spectral_radius_timeline
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        for _ in range(20):
            K = np.diag([0.5] * 6)
            viz.record_state(K=K)

        ax = plot_spectral_radius_timeline(viz)
        assert ax is not None
        plt.close("all")

    def test_all_negative_ei_balance(self):
        """Test with strongly inhibitory E/I balance."""
        from ww.visualization.coupling_dynamics import plot_ei_balance_timeline
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        viz = CouplingDynamicsVisualizer()
        for _ in range(20):
            K = np.zeros((6, 6))
            K[4, 5] = -0.8
            K[5, 4] = -0.6
            viz.record_state(K=K)

        ax = plot_ei_balance_timeline(viz)
        assert ax is not None
        plt.close("all")
