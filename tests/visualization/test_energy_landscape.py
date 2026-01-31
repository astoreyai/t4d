"""
Tests for Energy Landscape Visualization.

Tests the core functionality of the energy landscape visualizer:
- State recording and trajectory tracking
- PCA projection (6D → 2D)
- Energy surface computation
- Gradient field computation
- Attractor basin visualization
- Metrics and analysis
"""

import numpy as np
import pytest
from datetime import datetime

from t4dm.visualization.energy_landscape import (
    EnergyLandscapeVisualizer,
    EnergySnapshot,
    TrajectoryPoint,
)


class TestEnergyLandscapeVisualizer:
    """Tests for EnergyLandscapeVisualizer class."""

    def test_init_default(self):
        """Test default initialization."""
        viz = EnergyLandscapeVisualizer()
        assert viz.window_size == 1000
        assert viz.grid_resolution == 50
        assert len(viz._trajectory) == 0
        assert len(viz._snapshots) == 0

    def test_init_custom(self):
        """Test custom initialization."""
        viz = EnergyLandscapeVisualizer(window_size=100, grid_resolution=25)
        assert viz.window_size == 100
        assert viz.grid_resolution == 25

    def test_record_state_basic(self):
        """Test basic state recording."""
        viz = EnergyLandscapeVisualizer()
        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        snapshot = viz.record_state(nt_state)

        assert isinstance(snapshot, EnergySnapshot)
        assert len(viz._trajectory) == 1
        assert len(viz._snapshots) == 1
        assert np.allclose(snapshot.nt_state, nt_state)

    def test_record_state_with_cognitive_state(self):
        """Test state recording with cognitive state label."""
        viz = EnergyLandscapeVisualizer()
        nt_state = np.array([0.7, 0.4, 0.5, 0.8, 0.3, 0.5])

        snapshot = viz.record_state(nt_state, cognitive_state="ALERT")

        assert len(viz._trajectory) == 1
        assert viz._trajectory[0].cognitive_state == "ALERT"

    def test_record_state_window_limit(self):
        """Test that trajectory respects window size."""
        viz = EnergyLandscapeVisualizer(window_size=10)

        for i in range(20):
            nt_state = np.random.rand(6)
            viz.record_state(nt_state)

        assert len(viz._trajectory) == 10
        assert len(viz._snapshots) == 10

    def test_compute_pca_projection_default(self):
        """Test default PCA projection with insufficient data."""
        viz = EnergyLandscapeVisualizer()

        proj_matrix, proj_mean = viz.compute_pca_projection()

        assert proj_matrix.shape == (2, 6)
        assert proj_mean.shape == (6,)

    def test_compute_pca_projection_with_data(self):
        """Test PCA projection with trajectory data."""
        viz = EnergyLandscapeVisualizer()

        # Add diverse trajectory points
        for _ in range(20):
            nt_state = np.random.rand(6)
            viz.record_state(nt_state)

        proj_matrix, proj_mean = viz.compute_pca_projection()

        assert proj_matrix.shape == (2, 6)
        assert proj_mean.shape == (6,)
        # Principal components should be orthogonal
        dot_product = np.dot(proj_matrix[0], proj_matrix[1])
        assert abs(dot_product) < 0.1  # Near-orthogonal

    def test_project_to_2d(self):
        """Test 6D to 2D projection."""
        viz = EnergyLandscapeVisualizer()

        # Add some data for PCA
        for _ in range(10):
            viz.record_state(np.random.rand(6))

        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        point_2d = viz.project_to_2d(nt_state)

        assert point_2d.shape == (2,)
        assert np.isfinite(point_2d).all()

    def test_project_trajectory(self):
        """Test trajectory projection."""
        viz = EnergyLandscapeVisualizer()

        for _ in range(15):
            viz.record_state(np.random.rand(6))

        traj_2d = viz.project_trajectory()

        assert traj_2d.shape == (15, 2)
        assert np.isfinite(traj_2d).all()

    def test_project_trajectory_empty(self):
        """Test trajectory projection with no data."""
        viz = EnergyLandscapeVisualizer()
        traj_2d = viz.project_trajectory()
        assert traj_2d.shape == (0, 2)

    def test_compute_energy_surface(self):
        """Test energy surface computation."""
        viz = EnergyLandscapeVisualizer(grid_resolution=10)

        # Add data for projection
        for _ in range(5):
            viz.record_state(np.random.rand(6))

        X, Y, E = viz.compute_energy_surface(grid_range=(-1, 1))

        assert X.shape == (10, 10)
        assert Y.shape == (10, 10)
        assert E.shape == (10, 10)
        assert np.isfinite(E).all()

    def test_compute_energy_surface_caching(self):
        """Test that energy surface is cached."""
        viz = EnergyLandscapeVisualizer(grid_resolution=10)

        for _ in range(5):
            viz.record_state(np.random.rand(6))

        # First computation
        X1, Y1, E1 = viz.compute_energy_surface()

        # Second computation should use cache
        X2, Y2, E2 = viz.compute_energy_surface()

        assert np.array_equal(E1, E2)

    def test_compute_gradient_field(self):
        """Test gradient field computation."""
        viz = EnergyLandscapeVisualizer(grid_resolution=20)

        for _ in range(5):
            viz.record_state(np.random.rand(6))

        X, Y, U, V = viz.compute_gradient_field(subsample=4)

        assert X.shape == (5, 5)  # 20 / 4 = 5
        assert Y.shape == (5, 5)
        assert U.shape == (5, 5)
        assert V.shape == (5, 5)
        assert np.isfinite(U).all()
        assert np.isfinite(V).all()

    def test_get_attractor_positions_no_manager(self):
        """Test attractor positions without state manager."""
        viz = EnergyLandscapeVisualizer()
        positions = viz.get_attractor_positions()
        assert positions == {}

    def test_classify_basin_no_manager(self):
        """Test basin classification without state manager."""
        viz = EnergyLandscapeVisualizer()
        basin = viz.classify_basin(np.array([0.5] * 6))
        assert basin is None

    def test_get_energy_trace(self):
        """Test energy trace retrieval."""
        viz = EnergyLandscapeVisualizer()

        for _ in range(10):
            viz.record_state(np.random.rand(6))

        timestamps, energies = viz.get_energy_trace()

        assert len(timestamps) == 10
        assert len(energies) == 10
        assert all(isinstance(t, datetime) for t in timestamps)
        assert all(isinstance(e, float) for e in energies)

    def test_get_energy_trace_empty(self):
        """Test energy trace with no data."""
        viz = EnergyLandscapeVisualizer()
        timestamps, energies = viz.get_energy_trace()
        assert timestamps == []
        assert energies == []

    def test_get_gradient_norm_trace(self):
        """Test gradient norm trace retrieval."""
        viz = EnergyLandscapeVisualizer()

        for _ in range(10):
            viz.record_state(np.random.rand(6))

        timestamps, norms = viz.get_gradient_norm_trace()

        assert len(timestamps) == 10
        assert len(norms) == 10

    def test_get_basin_occupancy_empty(self):
        """Test basin occupancy with no data."""
        viz = EnergyLandscapeVisualizer()
        occupancy = viz.get_basin_occupancy()
        assert occupancy == {}

    def test_get_energy_components(self):
        """Test energy component breakdown."""
        viz = EnergyLandscapeVisualizer()

        for _ in range(10):
            viz.record_state(np.random.rand(6))

        components = viz.get_energy_components()

        assert "total" in components
        assert "hopfield" in components
        assert "boundary" in components
        assert "attractor" in components
        assert len(components["total"]) == 10

    def test_get_stability_metrics(self):
        """Test stability metrics computation."""
        viz = EnergyLandscapeVisualizer()

        for _ in range(20):
            viz.record_state(np.random.rand(6))

        metrics = viz.get_stability_metrics()

        assert "mean_energy" in metrics
        assert "energy_variance" in metrics
        assert "mean_gradient_norm" in metrics
        assert "basin_transitions" in metrics
        assert "transition_rate" in metrics
        assert "energy_trend" in metrics

    def test_get_stability_metrics_insufficient_data(self):
        """Test stability metrics with insufficient data."""
        viz = EnergyLandscapeVisualizer()
        viz.record_state(np.random.rand(6))  # Only 1 point

        metrics = viz.get_stability_metrics()
        assert metrics == {}

    def test_export_data(self):
        """Test data export for external rendering."""
        viz = EnergyLandscapeVisualizer(grid_resolution=10)

        for _ in range(10):
            viz.record_state(np.random.rand(6))

        data = viz.export_data()

        assert "surface" in data
        assert "gradient_field" in data
        assert "trajectory" in data
        assert "attractors" in data
        assert "metrics" in data
        assert "basin_occupancy" in data

    def test_clear_history(self):
        """Test history clearing."""
        viz = EnergyLandscapeVisualizer()

        for _ in range(10):
            viz.record_state(np.random.rand(6))

        viz.clear_history()

        assert len(viz._trajectory) == 0
        assert len(viz._snapshots) == 0
        assert viz._energy_surface is None


class TestEnergySnapshot:
    """Tests for EnergySnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test snapshot creation."""
        snapshot = EnergySnapshot(
            timestamp=datetime.now(),
            nt_state=np.array([0.5] * 6),
            total_energy=-1.0,
            hopfield_energy=-0.5,
            boundary_energy=0.1,
            attractor_energy=-0.6,
            gradient_norm=0.1,
            nearest_attractor="FOCUS",
            attractor_distance=0.15,
        )

        assert snapshot.total_energy == -1.0
        assert snapshot.nearest_attractor == "FOCUS"


class TestTrajectoryPoint:
    """Tests for TrajectoryPoint dataclass."""

    def test_trajectory_point_creation(self):
        """Test trajectory point creation."""
        point = TrajectoryPoint(
            timestamp=datetime.now(),
            nt_state=np.array([0.5] * 6),
            energy=-1.0,
            cognitive_state="ALERT",
        )

        assert point.energy == -1.0
        assert point.cognitive_state == "ALERT"


class TestWithCoupling:
    """Tests with actual coupling matrix."""

    def test_energy_from_coupling(self):
        """Test energy computation using coupling matrix."""
        from t4dm.nca.coupling import LearnableCoupling

        coupling = LearnableCoupling()
        viz = EnergyLandscapeVisualizer(coupling=coupling)

        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        snapshot = viz.record_state(nt_state)

        # Energy should be computed from coupling
        assert snapshot.hopfield_energy != 0.0
        assert np.isfinite(snapshot.total_energy)

    def test_gradient_with_coupling(self):
        """Test gradient computation with coupling."""
        from t4dm.nca.coupling import LearnableCoupling

        coupling = LearnableCoupling()
        viz = EnergyLandscapeVisualizer(coupling=coupling, grid_resolution=10)

        for _ in range(5):
            viz.record_state(np.random.rand(6))

        X, Y, U, V = viz.compute_gradient_field(subsample=2)

        # Gradients should be non-zero with coupling
        assert not np.allclose(U, 0)
        assert not np.allclose(V, 0)


class TestWithAttractors:
    """Tests with attractor state manager."""

    def test_with_state_manager(self):
        """Test with full state manager."""
        from t4dm.nca.attractors import AttractorBasin, CognitiveState, StateTransitionManager

        # Create state manager with default attractors
        manager = StateTransitionManager()

        viz = EnergyLandscapeVisualizer(state_manager=manager)

        # Record state near FOCUS attractor
        focus_center = np.array([0.5, 0.5, 0.8, 0.5, 0.4, 0.7])
        snapshot = viz.record_state(focus_center)

        assert snapshot.nearest_attractor == "FOCUS"
        assert snapshot.attractor_distance < 0.2

    def test_attractor_positions(self):
        """Test attractor position retrieval."""
        from t4dm.nca.attractors import StateTransitionManager

        manager = StateTransitionManager()
        viz = EnergyLandscapeVisualizer(state_manager=manager)

        # Need some trajectory for PCA
        for _ in range(10):
            viz.record_state(np.random.rand(6))

        positions = viz.get_attractor_positions()

        assert len(positions) > 0
        for name, (x, y, r) in positions.items():
            assert isinstance(name, str)
            assert np.isfinite(x)
            assert np.isfinite(y)
            assert r > 0

    def test_basin_classification(self):
        """Test basin classification with attractors."""
        from t4dm.nca.attractors import StateTransitionManager

        manager = StateTransitionManager()
        viz = EnergyLandscapeVisualizer(state_manager=manager)

        # State at ALERT center
        alert_state = np.array([0.7, 0.4, 0.5, 0.8, 0.3, 0.5])
        basin = viz.classify_basin(alert_state)
        assert basin == "ALERT"

        # State at REST center
        rest_state = np.array([0.3, 0.7, 0.4, 0.2, 0.7, 0.3])
        basin = viz.classify_basin(rest_state)
        assert basin == "REST"

    def test_basin_occupancy(self):
        """Test basin occupancy computation."""
        from t4dm.nca.attractors import StateTransitionManager

        manager = StateTransitionManager()
        viz = EnergyLandscapeVisualizer(state_manager=manager)

        # Record states in different basins
        for _ in range(5):
            viz.record_state(np.array([0.7, 0.4, 0.5, 0.8, 0.3, 0.5]))  # ALERT
        for _ in range(5):
            viz.record_state(np.array([0.3, 0.7, 0.4, 0.2, 0.7, 0.3]))  # REST

        occupancy = viz.get_basin_occupancy()

        assert "ALERT" in occupancy
        assert "REST" in occupancy
        assert abs(occupancy["ALERT"] - 0.5) < 0.1
        assert abs(occupancy["REST"] - 0.5) < 0.1


class TestIntegrationWithEnergyLandscape:
    """Integration tests with full EnergyLandscape module."""

    def test_full_integration(self):
        """Test with full energy landscape integration."""
        from t4dm.nca.coupling import LearnableCoupling
        from t4dm.nca.attractors import StateTransitionManager
        from t4dm.nca.energy import EnergyLandscape

        coupling = LearnableCoupling()
        manager = StateTransitionManager()
        energy = EnergyLandscape(coupling=coupling, state_manager=manager)

        viz = EnergyLandscapeVisualizer(
            energy_landscape=energy,
            coupling=coupling,
            state_manager=manager,
            grid_resolution=15,
        )

        # Record trajectory
        for _ in range(20):
            nt_state = np.random.rand(6) * 0.5 + 0.25  # Stay in [0.25, 0.75]
            viz.record_state(nt_state)

        # Verify all components work
        X, Y, E = viz.compute_energy_surface()
        assert E.shape == (15, 15)

        metrics = viz.get_stability_metrics()
        assert "mean_energy" in metrics

        data = viz.export_data()
        assert len(data["trajectory"]) == 20
