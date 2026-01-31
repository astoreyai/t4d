"""
Tests for StabilityMonitor visualization module.

Comprehensive tests for:
- Jacobian eigenvalue tracking
- Lyapunov exponent estimation
- Bifurcation detection
- Convergence monitoring
- Oscillation detection
- All plot functions
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from t4dm.visualization.stability_monitor import (
    StabilityMonitor,
    StabilitySnapshot,
    StabilityType,
    BifurcationEvent,
    plot_eigenvalue_spectrum,
    plot_stability_timeline,
    plot_lyapunov_timeline,
    plot_eigenvalue_evolution,
    plot_bifurcation_diagram,
    plot_oscillation_metrics,
    create_stability_dashboard,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def monitor():
    """Create basic stability monitor."""
    return StabilityMonitor(window_size=100)


@pytest.fixture
def stable_jacobian():
    """Jacobian with all negative eigenvalues (stable node)."""
    return np.array([
        [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.4, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.6, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.7],
    ], dtype=np.float32)


@pytest.fixture
def unstable_jacobian():
    """Jacobian with some positive eigenvalues (unstable)."""
    return np.array([
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
    ], dtype=np.float32)


@pytest.fixture
def oscillatory_jacobian():
    """Jacobian with complex eigenvalues (stable focus)."""
    # Create rotation matrix component for oscillatory behavior
    omega = 2.0  # Angular frequency
    sigma = -0.1  # Damping
    J = np.zeros((6, 6), dtype=np.float32)
    # First pair: oscillatory
    J[0, 0] = sigma
    J[0, 1] = omega
    J[1, 0] = -omega
    J[1, 1] = sigma
    # Rest: stable real
    J[2, 2] = -0.3
    J[3, 3] = -0.4
    J[4, 4] = -0.2
    J[5, 5] = -0.5
    return J


@pytest.fixture
def saddle_jacobian():
    """Jacobian with mixed sign eigenvalues (saddle)."""
    return np.array([
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.4, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.6, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.7],
    ], dtype=np.float32)


@pytest.fixture
def bifurcation_jacobian():
    """Jacobian with near-zero eigenvalue."""
    return np.array([
        [1e-9, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.4, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.6, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.7],
    ], dtype=np.float32)


@pytest.fixture
def sample_nt_state():
    """Sample 6D NT state."""
    return np.array([0.5, 0.4, 0.6, 0.3, 0.5, 0.5], dtype=np.float32)


# =============================================================================
# StabilityType Tests
# =============================================================================


class TestStabilityType:
    """Tests for StabilityType enum."""

    def test_all_types_defined(self):
        """Verify all stability types exist."""
        assert hasattr(StabilityType, "STABLE_NODE")
        assert hasattr(StabilityType, "STABLE_FOCUS")
        assert hasattr(StabilityType, "UNSTABLE_NODE")
        assert hasattr(StabilityType, "UNSTABLE_FOCUS")
        assert hasattr(StabilityType, "SADDLE")
        assert hasattr(StabilityType, "CENTER")
        assert hasattr(StabilityType, "BIFURCATION")

    def test_types_unique(self):
        """Each type should have unique value."""
        values = [t.value for t in StabilityType]
        assert len(values) == len(set(values))

    def test_types_have_names(self):
        """Types should have string names."""
        for t in StabilityType:
            assert isinstance(t.name, str)
            assert len(t.name) > 0


# =============================================================================
# StabilitySnapshot Tests
# =============================================================================


class TestStabilitySnapshot:
    """Tests for StabilitySnapshot dataclass."""

    def test_create_snapshot(self):
        """Test basic snapshot creation."""
        now = datetime.now()
        eigenvalues = np.array([-0.5, -0.3, -0.2, -0.4, -0.6, -0.1], dtype=complex)

        snapshot = StabilitySnapshot(
            timestamp=now,
            nt_state=np.array([0.5] * 6),
            eigenvalues=eigenvalues,
            max_real_eigenvalue=-0.1,
            min_real_eigenvalue=-0.6,
            spectral_abscissa=-0.1,
            stability_type=StabilityType.STABLE_NODE,
            is_stable=True,
            stability_margin=0.1,
            lyapunov_exponent=-0.1,
            is_chaotic=False,
            has_oscillations=False,
            oscillation_frequency=0.0,
            damping_ratio=0.0,
            energy=1.0,
            energy_gradient_norm=0.5,
            convergence_rate=0.1,
        )

        assert snapshot.is_stable
        assert snapshot.stability_type == StabilityType.STABLE_NODE
        assert snapshot.stability_margin == 0.1
        assert not snapshot.is_chaotic

    def test_oscillatory_snapshot(self):
        """Test snapshot with oscillations."""
        # Complex eigenvalues
        eigenvalues = np.array(
            [-0.1 + 2j, -0.1 - 2j, -0.3, -0.4, -0.5, -0.2],
            dtype=complex
        )

        snapshot = StabilitySnapshot(
            timestamp=datetime.now(),
            nt_state=np.array([0.5] * 6),
            eigenvalues=eigenvalues,
            max_real_eigenvalue=-0.1,
            min_real_eigenvalue=-0.5,
            spectral_abscissa=-0.1,
            stability_type=StabilityType.STABLE_FOCUS,
            is_stable=True,
            stability_margin=0.1,
            lyapunov_exponent=-0.1,
            is_chaotic=False,
            has_oscillations=True,
            oscillation_frequency=2.0 / (2 * np.pi),
            damping_ratio=0.05,
            energy=1.0,
            energy_gradient_norm=0.5,
            convergence_rate=0.1,
        )

        assert snapshot.has_oscillations
        assert snapshot.oscillation_frequency > 0
        assert snapshot.stability_type == StabilityType.STABLE_FOCUS


# =============================================================================
# BifurcationEvent Tests
# =============================================================================


class TestBifurcationEvent:
    """Tests for BifurcationEvent dataclass."""

    def test_create_bifurcation(self):
        """Test basic bifurcation event creation."""
        event = BifurcationEvent(
            timestamp=datetime.now(),
            nt_state=np.array([0.5] * 6),
            old_type=StabilityType.STABLE_NODE,
            new_type=StabilityType.SADDLE,
            eigenvalue_crossing=0.01,
            parameter_sensitivity=0.05,
        )

        assert event.old_type == StabilityType.STABLE_NODE
        assert event.new_type == StabilityType.SADDLE
        assert abs(event.eigenvalue_crossing - 0.01) < 1e-6


# =============================================================================
# StabilityMonitor Tests
# =============================================================================


class TestStabilityMonitor:
    """Tests for StabilityMonitor class."""

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.window_size == 100
        assert monitor.tau == 0.1
        assert len(monitor._snapshots) == 0
        assert len(monitor._bifurcations) == 0

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        monitor = StabilityMonitor(
            window_size=50,
            tau=0.2,
            alert_stability_margin=0.2,
            alert_lyapunov=0.05,
        )
        assert monitor.window_size == 50
        assert monitor.tau == 0.2
        assert monitor.alert_stability_margin == 0.2
        assert monitor.alert_lyapunov == 0.05

    def test_record_stable_state(self, monitor, sample_nt_state, stable_jacobian):
        """Test recording stable state."""
        snapshot = monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        assert snapshot.is_stable
        assert snapshot.stability_type == StabilityType.STABLE_NODE
        assert snapshot.max_real_eigenvalue < 0
        assert snapshot.stability_margin > 0
        assert not snapshot.is_chaotic
        assert len(monitor._snapshots) == 1

    def test_record_unstable_state(self, monitor, sample_nt_state, unstable_jacobian):
        """Test recording unstable state."""
        snapshot = monitor.record_state(sample_nt_state, jacobian=unstable_jacobian)

        assert not snapshot.is_stable
        assert snapshot.max_real_eigenvalue > 0
        assert snapshot.stability_margin > 0  # Distance from zero

    def test_record_oscillatory_state(self, monitor, sample_nt_state, oscillatory_jacobian):
        """Test recording oscillatory state."""
        snapshot = monitor.record_state(sample_nt_state, jacobian=oscillatory_jacobian)

        assert snapshot.is_stable  # Negative real parts
        assert snapshot.has_oscillations
        assert snapshot.oscillation_frequency > 0
        assert snapshot.damping_ratio != 0
        assert snapshot.stability_type == StabilityType.STABLE_FOCUS

    def test_record_saddle_state(self, monitor, sample_nt_state, saddle_jacobian):
        """Test recording saddle point state."""
        snapshot = monitor.record_state(sample_nt_state, jacobian=saddle_jacobian)

        assert not snapshot.is_stable
        assert snapshot.stability_type == StabilityType.SADDLE

    def test_record_bifurcation_state(self, monitor, sample_nt_state, bifurcation_jacobian):
        """Test recording near-bifurcation state."""
        snapshot = monitor.record_state(sample_nt_state, jacobian=bifurcation_jacobian)

        assert snapshot.stability_type == StabilityType.BIFURCATION

    def test_window_size_limit(self, monitor, sample_nt_state, stable_jacobian):
        """Test window size limiting."""
        for _ in range(150):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        assert len(monitor._snapshots) == 100
        assert len(monitor._eigenvalue_history) == 100

    def test_default_jacobian_computation(self, monitor, sample_nt_state):
        """Test default Jacobian when no coupling provided."""
        snapshot = monitor.record_state(sample_nt_state)

        # Default is -I/tau diagonal
        assert snapshot.is_stable
        assert snapshot.max_real_eigenvalue < 0

    def test_classify_stability_stable_node(self, monitor):
        """Test stability classification for stable node."""
        eigenvalues = np.array([-0.5, -0.3, -0.2, -0.4, -0.6, -0.1])
        stability_type = monitor._classify_stability(eigenvalues)
        assert stability_type == StabilityType.STABLE_NODE

    def test_classify_stability_stable_focus(self, monitor):
        """Test stability classification for stable focus."""
        eigenvalues = np.array([-0.1 + 2j, -0.1 - 2j, -0.3, -0.4, -0.5, -0.2])
        stability_type = monitor._classify_stability(eigenvalues)
        assert stability_type == StabilityType.STABLE_FOCUS

    def test_classify_stability_unstable_node(self, monitor):
        """Test stability classification for unstable node."""
        eigenvalues = np.array([0.5, 0.3, 0.2, 0.4, 0.6, 0.1])
        stability_type = monitor._classify_stability(eigenvalues)
        assert stability_type == StabilityType.UNSTABLE_NODE

    def test_classify_stability_unstable_focus(self, monitor):
        """Test stability classification for unstable focus."""
        eigenvalues = np.array([0.1 + 2j, 0.1 - 2j, 0.3, 0.4, 0.5, 0.2])
        stability_type = monitor._classify_stability(eigenvalues)
        assert stability_type == StabilityType.UNSTABLE_FOCUS

    def test_classify_stability_saddle(self, monitor):
        """Test stability classification for saddle."""
        eigenvalues = np.array([0.5, -0.3, -0.2, -0.4, -0.6, -0.1])
        stability_type = monitor._classify_stability(eigenvalues)
        assert stability_type == StabilityType.SADDLE

    def test_classify_stability_center(self, monitor):
        """Test stability classification for center."""
        eigenvalues = np.array([0j, 0j, 0j, 0j, 0j, 0j])
        stability_type = monitor._classify_stability(eigenvalues)
        # Pure zero eigenvalues classify as bifurcation
        assert stability_type == StabilityType.BIFURCATION

    def test_classify_stability_bifurcation(self, monitor):
        """Test stability classification near bifurcation."""
        eigenvalues = np.array([1e-10, -0.3, -0.2, -0.4, -0.6, -0.1])
        stability_type = monitor._classify_stability(eigenvalues)
        assert stability_type == StabilityType.BIFURCATION


# =============================================================================
# Bifurcation Detection Tests
# =============================================================================


class TestBifurcationDetection:
    """Tests for bifurcation detection."""

    def test_detect_bifurcation(self, monitor, sample_nt_state, stable_jacobian, saddle_jacobian):
        """Test bifurcation detection on stability type change."""
        # Start stable
        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)
        assert len(monitor._bifurcations) == 0

        # Transition to saddle
        monitor.record_state(sample_nt_state, jacobian=saddle_jacobian)
        assert len(monitor._bifurcations) == 1

        event = monitor._bifurcations[0]
        assert event.old_type == StabilityType.STABLE_NODE
        assert event.new_type == StabilityType.SADDLE

    def test_no_bifurcation_same_type(self, monitor, sample_nt_state, stable_jacobian):
        """Test no bifurcation when type stays same."""
        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)
        monitor.record_state(sample_nt_state, jacobian=stable_jacobian * 0.9)
        monitor.record_state(sample_nt_state, jacobian=stable_jacobian * 0.8)

        assert len(monitor._bifurcations) == 0

    def test_bifurcation_count_limit(self, monitor, sample_nt_state, stable_jacobian, saddle_jacobian):
        """Test bifurcation count limiting."""
        # Create many bifurcations
        for i in range(120):
            J = stable_jacobian if i % 2 == 0 else saddle_jacobian
            monitor.record_state(sample_nt_state, jacobian=J)

        # Should be capped at 100
        assert len(monitor._bifurcations) <= 100


# =============================================================================
# Alert Tests
# =============================================================================


class TestAlerts:
    """Tests for alert system."""

    def test_unstable_alert(self, monitor, sample_nt_state, unstable_jacobian):
        """Test alert on unstable state."""
        monitor.record_state(sample_nt_state, jacobian=unstable_jacobian)
        alerts = monitor.get_alerts()

        assert len(alerts) > 0
        assert any("UNSTABLE" in a for a in alerts)

    def test_marginal_stability_alert(self, sample_nt_state):
        """Test alert for marginal stability."""
        # Set high threshold
        monitor = StabilityMonitor(alert_stability_margin=0.5)

        # Jacobian with small negative eigenvalue
        J = np.diag([-0.1, -0.3, -0.4, -0.5, -0.6, -0.7])
        monitor.record_state(sample_nt_state, jacobian=J)

        alerts = monitor.get_alerts()
        assert any("MARGINAL" in a for a in alerts)

    def test_chaotic_alert(self, sample_nt_state):
        """Test alert for chaotic behavior."""
        monitor = StabilityMonitor(alert_lyapunov=0.01)

        # Jacobian with small positive eigenvalue
        J = np.diag([0.05, -0.3, -0.4, -0.5, -0.6, -0.7])
        monitor.record_state(sample_nt_state, jacobian=J)

        alerts = monitor.get_alerts()
        assert any("CHAOTIC" in a for a in alerts) or any("UNSTABLE" in a for a in alerts)

    def test_bifurcation_alert(self, monitor, sample_nt_state, bifurcation_jacobian):
        """Test alert at bifurcation point."""
        monitor.record_state(sample_nt_state, jacobian=bifurcation_jacobian)
        alerts = monitor.get_alerts()

        assert any("BIFURCATION" in a for a in alerts)

    def test_no_alert_stable(self, monitor, sample_nt_state, stable_jacobian):
        """Test no alerts for stable state."""
        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)
        alerts = monitor.get_alerts()

        assert len(alerts) == 0


# =============================================================================
# Time Series Access Tests
# =============================================================================


class TestTimeSeriesAccess:
    """Tests for time series data access."""

    def test_get_eigenvalue_traces(self, monitor, sample_nt_state, stable_jacobian):
        """Test eigenvalue trace retrieval."""
        for _ in range(10):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        traces = monitor.get_eigenvalue_traces()
        assert "max_real" in traces
        assert "min_real" in traces
        assert "spectral_abscissa" in traces

        timestamps, values = traces["max_real"]
        assert len(timestamps) == 10
        assert len(values) == 10
        assert all(v < 0 for v in values)

    def test_get_stability_margin_trace(self, monitor, sample_nt_state, stable_jacobian):
        """Test stability margin trace retrieval."""
        for _ in range(5):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        timestamps, margins = monitor.get_stability_margin_trace()
        assert len(timestamps) == 5
        assert len(margins) == 5
        assert all(m > 0 for m in margins)

    def test_get_lyapunov_trace(self, monitor, sample_nt_state, stable_jacobian):
        """Test Lyapunov trace retrieval."""
        for _ in range(5):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        timestamps, lyapunov = monitor.get_lyapunov_trace()
        assert len(timestamps) == 5
        assert len(lyapunov) == 5

    def test_get_oscillation_traces(self, monitor, sample_nt_state, oscillatory_jacobian):
        """Test oscillation trace retrieval."""
        for _ in range(5):
            monitor.record_state(sample_nt_state, jacobian=oscillatory_jacobian)

        traces = monitor.get_oscillation_traces()
        assert "frequency" in traces
        assert "damping_ratio" in traces

        _, frequencies = traces["frequency"]
        assert len(frequencies) == 5
        assert all(f > 0 for f in frequencies)

    def test_get_convergence_trace(self, monitor, sample_nt_state, stable_jacobian):
        """Test convergence rate trace retrieval."""
        for _ in range(5):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        timestamps, rates = monitor.get_convergence_trace()
        assert len(timestamps) == 5
        assert all(r > 0 for r in rates)  # Positive convergence for stable

    def test_get_energy_trace(self, monitor, sample_nt_state, stable_jacobian):
        """Test energy trace retrieval."""
        for _ in range(5):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian, energy=1.0)

        timestamps, energies = monitor.get_energy_trace()
        assert len(timestamps) == 5
        assert all(e == 1.0 for e in energies)

    def test_empty_traces(self, monitor):
        """Test trace retrieval with no data."""
        traces = monitor.get_eigenvalue_traces()
        assert traces == {}

        timestamps, margins = monitor.get_stability_margin_trace()
        assert timestamps == []
        assert margins == []


# =============================================================================
# Eigenvalue Evolution Tests
# =============================================================================


class TestEigenvalueEvolution:
    """Tests for eigenvalue evolution tracking."""

    def test_get_eigenvalue_evolution_real(self, monitor, sample_nt_state, stable_jacobian):
        """Test real part evolution retrieval."""
        for _ in range(10):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        evolution = monitor.get_eigenvalue_evolution(component="real")
        assert evolution.shape == (10, 6)
        assert np.all(evolution < 0)  # All negative for stable

    def test_get_eigenvalue_evolution_imag(self, monitor, sample_nt_state, oscillatory_jacobian):
        """Test imaginary part evolution retrieval."""
        for _ in range(10):
            monitor.record_state(sample_nt_state, jacobian=oscillatory_jacobian)

        evolution = monitor.get_eigenvalue_evolution(component="imag")
        assert evolution.shape == (10, 6)
        assert np.any(np.abs(evolution) > 0)  # Some non-zero imaginary

    def test_get_eigenvalue_evolution_abs(self, monitor, sample_nt_state, stable_jacobian):
        """Test absolute value evolution retrieval."""
        for _ in range(10):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        evolution = monitor.get_eigenvalue_evolution(component="abs")
        assert evolution.shape == (10, 6)
        assert np.all(evolution >= 0)  # All non-negative

    def test_get_eigenvalue_trajectory(self, monitor, sample_nt_state, stable_jacobian):
        """Test dominant eigenvalue trajectory."""
        for _ in range(10):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        trajectory = monitor.get_eigenvalue_trajectory()
        assert len(trajectory) == 10
        assert all(isinstance(p, tuple) and len(p) == 2 for p in trajectory)

    def test_empty_evolution(self, monitor):
        """Test evolution retrieval with no data."""
        evolution = monitor.get_eigenvalue_evolution()
        assert evolution.shape == (0, 6)

        trajectory = monitor.get_eigenvalue_trajectory()
        assert trajectory == []


# =============================================================================
# State Access Tests
# =============================================================================


class TestStateAccess:
    """Tests for current state access methods."""

    def test_get_current_snapshot(self, monitor, sample_nt_state, stable_jacobian):
        """Test getting current snapshot."""
        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)
        snapshot = monitor.get_current_snapshot()

        assert snapshot is not None
        assert snapshot.is_stable

    def test_get_current_snapshot_empty(self, monitor):
        """Test getting snapshot with no data."""
        snapshot = monitor.get_current_snapshot()
        assert snapshot is None

    def test_is_stable(self, monitor, sample_nt_state, stable_jacobian):
        """Test is_stable method."""
        assert monitor.is_stable()  # Default True when empty

        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)
        assert monitor.is_stable()

    def test_get_stability_type(self, monitor, sample_nt_state, stable_jacobian):
        """Test get_stability_type method."""
        assert monitor.get_stability_type() is None  # Empty

        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)
        assert monitor.get_stability_type() == StabilityType.STABLE_NODE

    def test_get_current_eigenvalues(self, monitor, sample_nt_state, stable_jacobian):
        """Test get_current_eigenvalues method."""
        assert monitor.get_current_eigenvalues() is None

        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)
        eigenvalues = monitor.get_current_eigenvalues()
        assert eigenvalues is not None
        assert len(eigenvalues) == 6


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for stability statistics computation."""

    def test_compute_stability_statistics(self, monitor, sample_nt_state, stable_jacobian):
        """Test comprehensive statistics computation."""
        for _ in range(20):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        stats = monitor.compute_stability_statistics()

        assert "current_stability_margin" in stats
        assert "mean_stability_margin" in stats
        assert "min_stability_margin" in stats
        assert "margin_trend" in stats
        assert "current_lyapunov" in stats
        assert "mean_lyapunov" in stats
        assert "max_lyapunov" in stats
        assert "stable_fraction" in stats
        assert "bifurcation_count" in stats
        assert "current_type" in stats
        assert "is_stable" in stats
        assert "is_chaotic" in stats

        assert stats["stable_fraction"] == 1.0
        assert stats["is_stable"]
        assert not stats["is_chaotic"]

    def test_empty_statistics(self, monitor):
        """Test statistics with no data."""
        stats = monitor.compute_stability_statistics()
        assert stats == {}

    def test_stability_type_distribution(self, monitor, sample_nt_state, stable_jacobian, saddle_jacobian):
        """Test stability type distribution."""
        # 5 stable states
        for _ in range(5):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        # 3 saddle states
        for _ in range(3):
            monitor.record_state(sample_nt_state, jacobian=saddle_jacobian)

        dist = monitor.get_stability_type_distribution()
        assert dist["STABLE_NODE"] == 5
        assert dist["SADDLE"] == 3


# =============================================================================
# Export and Clear Tests
# =============================================================================


class TestExportAndClear:
    """Tests for data export and history clearing."""

    def test_export_data(self, monitor, sample_nt_state, stable_jacobian):
        """Test comprehensive data export."""
        for _ in range(10):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        data = monitor.export_data()

        assert "current_state" in data
        assert "traces" in data
        assert "statistics" in data
        assert "bifurcations" in data
        assert "type_distribution" in data
        assert "alerts" in data
        assert "n_samples" in data

        assert data["n_samples"] == 10

    def test_export_empty(self, monitor):
        """Test export with no data."""
        data = monitor.export_data()
        assert data["n_samples"] == 0

    def test_clear_history(self, monitor, sample_nt_state, stable_jacobian):
        """Test clearing history."""
        for _ in range(10):
            monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        assert len(monitor._snapshots) == 10

        monitor.clear_history()

        assert len(monitor._snapshots) == 0
        assert len(monitor._bifurcations) == 0
        assert len(monitor._eigenvalue_history) == 0
        assert len(monitor._active_alerts) == 0
        assert monitor._prev_stability_type is None


# =============================================================================
# Plot Function Tests
# =============================================================================


class TestPlotFunctions:
    """Tests for plot functions."""

    @pytest.fixture
    def populated_monitor(self, monitor, sample_nt_state, stable_jacobian, oscillatory_jacobian):
        """Create monitor with data for plotting."""
        for i in range(20):
            J = stable_jacobian if i % 2 == 0 else oscillatory_jacobian
            monitor.record_state(sample_nt_state, jacobian=J)
        return monitor

    def test_plot_eigenvalue_spectrum(self, populated_monitor):
        """Test eigenvalue spectrum plotting."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_eigenvalue_spectrum(populated_monitor, ax=ax)
        assert result is not None
        plt.close(fig)

    def test_plot_eigenvalue_spectrum_empty(self, monitor):
        """Test eigenvalue spectrum with no data."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_eigenvalue_spectrum(monitor, ax=ax)
        assert result is not None
        plt.close(fig)

    def test_plot_stability_timeline(self, populated_monitor):
        """Test stability timeline plotting."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_stability_timeline(populated_monitor, ax=ax)
        assert result is not None
        plt.close(fig)

    def test_plot_lyapunov_timeline(self, populated_monitor):
        """Test Lyapunov timeline plotting."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_lyapunov_timeline(populated_monitor, ax=ax)
        assert result is not None
        plt.close(fig)

    def test_plot_eigenvalue_evolution(self, populated_monitor):
        """Test eigenvalue evolution plotting."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_eigenvalue_evolution(populated_monitor, ax=ax)
        assert result is not None
        plt.close(fig)

    def test_plot_bifurcation_diagram(self, populated_monitor):
        """Test bifurcation diagram plotting."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = plot_bifurcation_diagram(populated_monitor, ax=ax)
        assert result is not None
        plt.close(fig)

    def test_plot_oscillation_metrics(self, populated_monitor):
        """Test oscillation metrics plotting."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_oscillation_metrics(populated_monitor)
        assert fig is not None
        plt.close(fig)

    def test_create_stability_dashboard(self, populated_monitor):
        """Test full dashboard creation."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = create_stability_dashboard(populated_monitor)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Integration with NCA Components
# =============================================================================


class TestIntegrationWithNCA:
    """Tests for integration with actual NCA components."""

    def test_with_learnable_coupling(self):
        """Test integration with LearnableCoupling if available."""
        try:
            from t4dm.nca.coupling import LearnableCoupling

            coupling = LearnableCoupling()  # Uses default config
            monitor = StabilityMonitor(coupling=coupling)

            nt_state = np.array([0.5, 0.4, 0.6, 0.3, 0.5, 0.5], dtype=np.float32)
            snapshot = monitor.record_state(nt_state)

            assert snapshot is not None
            assert len(snapshot.eigenvalues) == 6
        except ImportError:
            pytest.skip("LearnableCoupling not available")

    def test_with_energy_landscape(self):
        """Test integration with EnergyLandscape if available."""
        try:
            from t4dm.nca.energy import EnergyLandscape

            energy = EnergyLandscape()
            monitor = StabilityMonitor(energy_landscape=energy)

            nt_state = np.array([0.5, 0.4, 0.6, 0.3, 0.5, 0.5], dtype=np.float32)
            snapshot = monitor.record_state(nt_state)

            assert snapshot is not None
            assert snapshot.energy != 0.0
        except ImportError:
            pytest.skip("EnergyLandscape not available")


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_small_eigenvalues(self, monitor, sample_nt_state):
        """Test with very small eigenvalues."""
        J = np.diag([1e-12, -1e-11, -1e-10, -1e-9, -1e-8, -1e-7])
        snapshot = monitor.record_state(sample_nt_state, jacobian=J)

        # Should classify as bifurcation due to near-zero values
        assert snapshot.stability_type == StabilityType.BIFURCATION

    def test_large_eigenvalues(self, monitor, sample_nt_state):
        """Test with large eigenvalues."""
        J = np.diag([-100, -200, -150, -300, -250, -180])
        snapshot = monitor.record_state(sample_nt_state, jacobian=J)

        assert snapshot.is_stable
        assert snapshot.convergence_rate > 0

    def test_mixed_eigenvalue_magnitudes(self, monitor, sample_nt_state):
        """Test with widely varying eigenvalue magnitudes."""
        J = np.diag([-0.001, -100, -0.01, -10, -0.1, -1])
        snapshot = monitor.record_state(sample_nt_state, jacobian=J)

        assert snapshot.is_stable
        assert snapshot.max_real_eigenvalue == pytest.approx(-0.001, rel=1e-3)

    def test_all_same_eigenvalue(self, monitor, sample_nt_state):
        """Test with degenerate eigenvalues."""
        J = np.diag([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
        snapshot = monitor.record_state(sample_nt_state, jacobian=J)

        assert snapshot.is_stable
        assert snapshot.max_real_eigenvalue == pytest.approx(-0.5, rel=1e-5)

    def test_nan_handling(self, monitor, sample_nt_state):
        """Test handling of NaN values."""
        # Create Jacobian that might produce numerical issues
        J = np.zeros((6, 6))
        snapshot = monitor.record_state(sample_nt_state, jacobian=J)

        # Zero matrix has all zero eigenvalues -> bifurcation
        assert snapshot.stability_type == StabilityType.BIFURCATION

    def test_rapid_state_changes(self, monitor, sample_nt_state, stable_jacobian, unstable_jacobian):
        """Test rapid alternating state changes."""
        for i in range(100):
            J = stable_jacobian if i % 2 == 0 else unstable_jacobian
            monitor.record_state(sample_nt_state, jacobian=J)

        # Should have detected many bifurcations
        assert monitor.get_bifurcation_count() > 10


# =============================================================================
# Enhanced Anomaly Detection Tests (P0-3)
# =============================================================================


class TestExcitotoxicityDetection:
    """Tests for excitotoxicity detection."""

    def test_detect_glutamate_surge(self, monitor):
        """Test detection of high glutamate levels."""
        # NT order: [DA, 5-HT, ACh, NE, GABA, Glu]
        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.9])  # High Glu
        alerts = monitor.detect_excitotoxicity(nt_state)

        assert len(alerts) > 0
        assert any("EXCITOTOXICITY" in a for a in alerts)

    def test_detect_ei_imbalance(self, monitor):
        """Test detection of E/I imbalance."""
        # High Glu, low GABA = high E/I ratio
        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.5])  # GABA=0.1, Glu=0.5
        alerts = monitor.detect_excitotoxicity(nt_state)

        assert len(alerts) > 0
        assert any("E/I IMBALANCE" in a or "GABA DEPLETION" in a for a in alerts)

    def test_detect_gaba_depletion(self, monitor):
        """Test detection of GABA depletion."""
        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.3])  # Low GABA
        alerts = monitor.detect_excitotoxicity(nt_state)

        assert len(alerts) > 0
        assert any("GABA DEPLETION" in a for a in alerts)

    def test_no_excitotoxicity_healthy(self, monitor):
        """Test no alerts for healthy state."""
        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # Balanced
        alerts = monitor.detect_excitotoxicity(nt_state)

        assert len(alerts) == 0

    def test_excitotoxicity_short_array(self, monitor):
        """Test handling of short arrays."""
        nt_state = np.array([0.5, 0.5, 0.5])  # Only 3 elements
        alerts = monitor.detect_excitotoxicity(nt_state)

        assert len(alerts) == 0  # Should handle gracefully


class TestForgettingRiskDetection:
    """Tests for catastrophic forgetting risk detection."""

    def test_detect_ca3_near_capacity(self, monitor):
        """Test detection of CA3 near capacity."""
        hippocampus_state = {
            "ca3_patterns": list(range(920)),  # 920/1000 = 92%
            "ca3_max_patterns": 1000,
        }
        alerts = monitor.detect_forgetting_risk(hippocampus_state)

        assert len(alerts) > 0
        assert any("CA3 NEAR CAPACITY" in a for a in alerts)

    def test_detect_ca3_critical_capacity(self, monitor):
        """Test detection of CA3 critical capacity."""
        hippocampus_state = {
            "ca3_patterns": list(range(960)),  # 960/1000 = 96%
            "ca3_max_patterns": 1000,
        }
        alerts = monitor.detect_forgetting_risk(hippocampus_state)

        assert len(alerts) > 0
        assert any("CA3 CRITICAL CAPACITY" in a for a in alerts)

    def test_detect_pattern_overlap(self, monitor):
        """Test detection of pattern overlap."""
        hippocampus_state = {
            "pattern_similarity": 0.65,
        }
        alerts = monitor.detect_forgetting_risk(hippocampus_state)

        assert len(alerts) > 0
        assert any("PATTERN OVERLAP" in a for a in alerts)

    def test_detect_severe_pattern_overlap(self, monitor):
        """Test detection of severe pattern overlap."""
        hippocampus_state = {
            "pattern_similarity": 0.75,
        }
        alerts = monitor.detect_forgetting_risk(hippocampus_state)

        assert len(alerts) > 0
        assert any("SEVERE PATTERN OVERLAP" in a for a in alerts)

    def test_detect_memory_collision(self, monitor):
        """Test detection of memory collision."""
        hippocampus_state = {
            "collision_rate": 0.15,
        }
        alerts = monitor.detect_forgetting_risk(hippocampus_state)

        assert len(alerts) > 0
        assert any("MEMORY COLLISION" in a for a in alerts)

    def test_no_forgetting_risk_healthy(self, monitor):
        """Test no alerts for healthy state."""
        hippocampus_state = {
            "ca3_patterns": list(range(500)),  # 50% capacity
            "ca3_max_patterns": 1000,
            "pattern_similarity": 0.3,
        }
        alerts = monitor.detect_forgetting_risk(hippocampus_state)

        assert len(alerts) == 0

    def test_forgetting_risk_empty_dict(self, monitor):
        """Test handling of empty dict."""
        alerts = monitor.detect_forgetting_risk({})
        assert len(alerts) == 0


class TestNTDepletionDetection:
    """Tests for neuromodulator depletion detection."""

    def test_detect_da_depletion(self, monitor):
        """Test detection of dopamine depletion."""
        nt_state = np.array([0.1, 0.5, 0.5, 0.5, 0.5, 0.5])  # Low DA
        alerts = monitor.detect_nt_depletion(nt_state)

        assert len(alerts) > 0
        assert any("DA DEPLETION" in a for a in alerts)

    def test_detect_ne_depletion(self, monitor):
        """Test detection of norepinephrine depletion."""
        nt_state = np.array([0.5, 0.5, 0.5, 0.1, 0.5, 0.5])  # Low NE
        alerts = monitor.detect_nt_depletion(nt_state)

        assert len(alerts) > 0
        assert any("NE DEPLETION" in a for a in alerts)

    def test_detect_ach_depletion(self, monitor):
        """Test detection of acetylcholine depletion."""
        nt_state = np.array([0.5, 0.5, 0.15, 0.5, 0.5, 0.5])  # Low ACh
        alerts = monitor.detect_nt_depletion(nt_state)

        assert len(alerts) > 0
        assert any("ACh DEPLETION" in a for a in alerts)

    def test_detect_5ht_depletion(self, monitor):
        """Test detection of serotonin depletion."""
        nt_state = np.array([0.5, 0.1, 0.5, 0.5, 0.5, 0.5])  # Low 5-HT
        alerts = monitor.detect_nt_depletion(nt_state)

        assert len(alerts) > 0
        assert any("5-HT DEPLETION" in a for a in alerts)

    def test_detect_da_surge(self, monitor):
        """Test detection of dopamine surge."""
        nt_state = np.array([0.95, 0.5, 0.5, 0.5, 0.5, 0.5])  # High DA
        alerts = monitor.detect_nt_depletion(nt_state)

        assert len(alerts) > 0
        assert any("DA SURGE" in a for a in alerts)

    def test_detect_ne_surge(self, monitor):
        """Test detection of norepinephrine surge."""
        nt_state = np.array([0.5, 0.5, 0.5, 0.9, 0.5, 0.5])  # High NE
        alerts = monitor.detect_nt_depletion(nt_state)

        assert len(alerts) > 0
        assert any("NE SURGE" in a for a in alerts)

    def test_no_depletion_healthy(self, monitor):
        """Test no alerts for healthy state."""
        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        alerts = monitor.detect_nt_depletion(nt_state)

        assert len(alerts) == 0

    def test_multiple_depletions(self, monitor):
        """Test detection of multiple simultaneous depletions."""
        nt_state = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5])  # All low
        alerts = monitor.detect_nt_depletion(nt_state)

        assert len(alerts) >= 4  # DA, 5-HT, ACh, NE


class TestOscillationPathologyDetection:
    """Tests for oscillation pathology detection."""

    def test_detect_high_frequency_oscillation(self, monitor, sample_nt_state):
        """Test detection of seizure-like high frequency oscillation."""
        # Create Jacobian with very high imaginary parts
        omega = 700  # Very high frequency
        sigma = -0.1
        J = np.zeros((6, 6), dtype=np.float32)
        J[0, 0] = sigma
        J[0, 1] = omega
        J[1, 0] = -omega
        J[1, 1] = sigma
        J[2, 2] = -0.3
        J[3, 3] = -0.4
        J[4, 4] = -0.2
        J[5, 5] = -0.5

        monitor.record_state(sample_nt_state, jacobian=J)
        alerts = monitor.detect_oscillation_pathology(sample_nt_state)

        assert len(alerts) > 0
        assert any("HIGH FREQUENCY" in a for a in alerts)

    def test_detect_undamped_oscillation(self, monitor, sample_nt_state):
        """Test detection of undamped oscillation."""
        # Create Jacobian with very small damping
        omega = 10
        sigma = -0.001  # Very small damping
        J = np.zeros((6, 6), dtype=np.float32)
        J[0, 0] = sigma
        J[0, 1] = omega
        J[1, 0] = -omega
        J[1, 1] = sigma
        J[2, 2] = -0.3
        J[3, 3] = -0.4
        J[4, 4] = -0.2
        J[5, 5] = -0.5

        monitor.record_state(sample_nt_state, jacobian=J)
        alerts = monitor.detect_oscillation_pathology(sample_nt_state)

        assert len(alerts) > 0
        assert any("UNDAMPED" in a for a in alerts)

    def test_no_oscillation_pathology_healthy(self, monitor, sample_nt_state, stable_jacobian):
        """Test no alerts for healthy oscillations."""
        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)
        alerts = monitor.detect_oscillation_pathology(sample_nt_state)

        assert len(alerts) == 0


class TestComprehensiveAnomalyDetection:
    """Tests for comprehensive anomaly detection."""

    def test_detect_anomalies_all_categories(self, monitor, sample_nt_state, stable_jacobian):
        """Test detect_anomalies returns all categories."""
        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        anomalies = monitor.detect_anomalies(
            nt_state=sample_nt_state,
            hippocampus_state={"ca3_patterns": [], "ca3_max_patterns": 1000}
        )

        assert "stability" in anomalies
        assert "excitotoxicity" in anomalies
        assert "forgetting" in anomalies
        assert "depletion" in anomalies
        assert "oscillation" in anomalies

    def test_detect_anomalies_uses_last_state(self, monitor, sample_nt_state, stable_jacobian):
        """Test detect_anomalies uses last recorded state."""
        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        # Call without nt_state - should use last recorded
        anomalies = monitor.detect_anomalies()

        assert "stability" in anomalies

    def test_get_anomaly_summary_healthy(self, monitor, sample_nt_state, stable_jacobian):
        """Test anomaly summary for healthy state."""
        monitor.record_state(sample_nt_state, jacobian=stable_jacobian)

        summary = monitor.get_anomaly_summary(nt_state=sample_nt_state)

        assert summary["total_alerts"] == 0
        assert summary["critical"] is False
        assert len(summary["all_alerts"]) == 0

    def test_get_anomaly_summary_critical(self, monitor, sample_nt_state, unstable_jacobian):
        """Test anomaly summary detects critical state."""
        # Create critical state: unstable + excitotoxic
        nt_critical = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.9])  # High Glu, low GABA
        monitor.record_state(nt_critical, jacobian=unstable_jacobian)

        summary = monitor.get_anomaly_summary(nt_state=nt_critical)

        assert summary["total_alerts"] > 0
        assert summary["critical"] is True  # UNSTABLE or EXCITOTOXICITY triggers critical

    def test_anomaly_summary_counts_by_category(self, monitor, sample_nt_state):
        """Test anomaly summary has correct category counts."""
        # Create state with NT depletion
        nt_depleted = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5])

        # Stable Jacobian
        J = np.diag([-0.5, -0.4, -0.3, -0.6, -0.2, -0.7])
        monitor.record_state(nt_depleted, jacobian=J)

        summary = monitor.get_anomaly_summary(nt_state=nt_depleted)

        assert "by_category" in summary
        assert summary["by_category"]["depletion"] >= 4  # DA, 5-HT, ACh, NE


# =============================================================================
# Module Exports Test
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_all_exports_available(self):
        """Test all __all__ exports are accessible."""
        from t4dm.visualization import stability_monitor

        for name in stability_monitor.__all__:
            assert hasattr(stability_monitor, name)

    def test_main_exports_in_package(self):
        """Test main exports in package __init__."""
        from t4dm.visualization import (
            StabilityMonitor,
            StabilitySnapshot,
            StabilityType,
            BifurcationEvent,
            plot_eigenvalue_spectrum,
            plot_stability_timeline,
            plot_lyapunov_timeline,
            plot_eigenvalue_evolution,
            plot_bifurcation_diagram,
            plot_oscillation_metrics,
            create_stability_dashboard,
        )

        assert StabilityMonitor is not None
        assert StabilitySnapshot is not None
        assert StabilityType is not None
        assert BifurcationEvent is not None
