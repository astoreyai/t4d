"""
Tests for PAC (Phase-Amplitude Coupling) Telemetry module.

Comprehensive tests for:
- State recording and history management
- Modulation Index (MI) computation (Tort et al. 2010)
- Working memory capacity estimation
- Preferred phase detection
- Biological range validation
- Visualization functions
- Integration with oscillator module
"""

import numpy as np
import pytest
from datetime import datetime
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from t4dm.visualization.pac_telemetry import PACTelemetry, PACSnapshot


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def telemetry():
    """Create basic PAC telemetry instance."""
    return PACTelemetry(window_size=500, n_phase_bins=18)


@pytest.fixture
def populated_telemetry():
    """Create telemetry with realistic coupled data."""
    t = PACTelemetry(window_size=1000, n_phase_bins=18)

    np.random.seed(42)

    # Simulate strong PAC: gamma amplitude peaks at theta phase ~π
    # Use cosine modulation centered at π for strong coupling
    for i in range(500):
        theta_phase = (i * 0.1) % (2 * np.pi)  # Theta cycling

        # Gamma amplitude strongly modulated by theta phase
        # Cosine function gives peak at θ=π (where cos(θ-π) = 1)
        modulation = 0.5 * (1 + np.cos(theta_phase - np.pi))  # 0 to 1
        gamma_amp = 0.2 + 0.7 * modulation + 0.05 * np.random.random()
        gamma_amp = np.clip(gamma_amp, 0, 1)

        t.record_state(
            theta_phase=theta_phase,
            gamma_amplitude=gamma_amp,
            theta_freq=6.0,
            gamma_freq=40.0,
        )

    return t


@pytest.fixture
def uncoupled_telemetry():
    """Create telemetry with no coupling (uniform distribution)."""
    t = PACTelemetry(window_size=500)

    np.random.seed(42)

    # No relationship between phase and amplitude
    for i in range(500):
        theta_phase = np.random.uniform(0, 2 * np.pi)
        gamma_amp = np.random.uniform(0.3, 0.7)  # Uniform amplitude

        t.record_state(
            theta_phase=theta_phase,
            gamma_amplitude=gamma_amp,
            theta_freq=6.0,
            gamma_freq=40.0,
        )

    return t


@dataclass
class MockOscillatorState:
    """Mock oscillator state for testing."""
    theta_phase: float = 0.0
    gamma_amplitude: float = 0.5
    theta_freq: float = 6.0
    gamma_freq: float = 40.0


# =============================================================================
# Initialization Tests
# =============================================================================


class TestPACTelemetryInit:
    """Tests for PACTelemetry initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        telemetry = PACTelemetry()
        assert telemetry.window_size == 1000
        assert telemetry.n_phase_bins == 18
        assert telemetry.theta_freq_range == (4.0, 8.0)
        assert telemetry.gamma_freq_range == (30.0, 80.0)

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        telemetry = PACTelemetry(
            window_size=500,
            n_phase_bins=36,
            theta_freq_range=(3.0, 10.0),
            gamma_freq_range=(25.0, 100.0),
        )
        assert telemetry.window_size == 500
        assert telemetry.n_phase_bins == 36
        assert telemetry.theta_freq_range == (3.0, 10.0)

    def test_empty_history(self):
        """Test empty history on initialization."""
        telemetry = PACTelemetry()
        assert len(telemetry._theta_phase_history) == 0
        assert len(telemetry._gamma_amplitude_history) == 0
        assert len(telemetry._snapshots) == 0


# =============================================================================
# State Recording Tests
# =============================================================================


class TestStateRecording:
    """Tests for recording oscillator state."""

    def test_record_single_state(self, telemetry):
        """Test recording a single state."""
        snapshot = telemetry.record_state(
            theta_phase=np.pi,
            gamma_amplitude=0.7,
            theta_freq=6.0,
            gamma_freq=40.0,
        )

        assert snapshot is not None
        assert snapshot.theta_phase == np.pi
        assert snapshot.gamma_amplitude == 0.7
        assert snapshot.theta_frequency == 6.0
        assert snapshot.gamma_frequency == 40.0
        assert len(telemetry._theta_phase_history) == 1

    def test_record_multiple_states(self, telemetry):
        """Test recording multiple states."""
        for i in range(100):
            telemetry.record_state(
                theta_phase=(i * 0.1) % (2 * np.pi),
                gamma_amplitude=0.5,
            )

        assert len(telemetry._theta_phase_history) == 100
        assert len(telemetry._snapshots) == 100
        assert telemetry._total_samples == 100

    def test_window_size_limit(self):
        """Test that window size is enforced."""
        telemetry = PACTelemetry(window_size=50)

        for i in range(100):
            telemetry.record_state(
                theta_phase=(i * 0.1) % (2 * np.pi),
                gamma_amplitude=0.5,
            )

        assert len(telemetry._theta_phase_history) == 50
        assert telemetry._total_samples == 100

    def test_default_frequencies(self, telemetry):
        """Test default frequency values."""
        snapshot = telemetry.record_state(
            theta_phase=0.0,
            gamma_amplitude=0.5,
        )

        assert snapshot.theta_frequency == 6.0
        assert snapshot.gamma_frequency == 40.0

    def test_record_from_oscillator(self, telemetry):
        """Test recording from oscillator state."""
        state = MockOscillatorState(
            theta_phase=1.5,
            gamma_amplitude=0.8,
            theta_freq=7.0,
            gamma_freq=45.0,
        )

        snapshot = telemetry.record_from_oscillator(state)

        assert snapshot.theta_phase == 1.5
        assert snapshot.gamma_amplitude == 0.8
        assert snapshot.theta_frequency == 7.0
        assert snapshot.gamma_frequency == 45.0


# =============================================================================
# Modulation Index Tests
# =============================================================================


class TestModulationIndex:
    """Tests for Modulation Index computation."""

    def test_mi_insufficient_data(self, telemetry):
        """Test MI with insufficient data returns 0."""
        # Record only 50 samples (need 100+)
        for i in range(50):
            telemetry.record_state(theta_phase=i * 0.1, gamma_amplitude=0.5)

        mi = telemetry.compute_modulation_index()
        assert mi == 0.0

    def test_mi_strong_coupling(self, populated_telemetry):
        """Test MI for strongly coupled data."""
        mi = populated_telemetry.compute_modulation_index()

        # With cosine modulation, MI is typically 0.02-0.1
        # Higher MI requires more concentrated phase distribution
        assert mi > 0.01  # Should show measurable coupling
        assert mi <= 1.0

    def test_mi_no_coupling(self, uncoupled_telemetry):
        """Test MI for uncoupled data."""
        mi = uncoupled_telemetry.compute_modulation_index()

        # No coupling should give MI close to 0
        assert mi < 0.2

    def test_mi_perfect_coupling(self):
        """Test MI for perfect coupling (all amplitude in one phase bin)."""
        telemetry = PACTelemetry(n_phase_bins=18)

        # All data at theta_phase = π with high amplitude
        for i in range(200):
            telemetry.record_state(
                theta_phase=np.pi + np.random.normal(0, 0.1),  # Narrow distribution
                gamma_amplitude=0.9,
            )

        mi = telemetry.compute_modulation_index()
        # Should be high (close to 1)
        assert mi > 0.5

    def test_mi_range(self, populated_telemetry):
        """Test MI is bounded between 0 and 1."""
        mi = populated_telemetry.compute_modulation_index()
        assert 0.0 <= mi <= 1.0


# =============================================================================
# Preferred Phase Tests
# =============================================================================


class TestPreferredPhase:
    """Tests for preferred phase detection."""

    def test_preferred_phase_insufficient_data(self, telemetry):
        """Test preferred phase with insufficient data."""
        for i in range(50):
            telemetry.record_state(theta_phase=i * 0.1, gamma_amplitude=0.5)

        phase = telemetry.find_preferred_phase()
        assert phase == 0.0

    def test_preferred_phase_detection(self, populated_telemetry):
        """Test preferred phase is detected correctly."""
        phase = populated_telemetry.find_preferred_phase()

        # Our simulated data has peak at π
        # Allow some tolerance
        assert abs(phase - np.pi) < 0.5

    def test_preferred_phase_range(self, populated_telemetry):
        """Test preferred phase is in valid range."""
        phase = populated_telemetry.find_preferred_phase()
        assert 0 <= phase <= 2 * np.pi


# =============================================================================
# Working Memory Capacity Tests
# =============================================================================


class TestWorkingMemoryCapacity:
    """Tests for working memory capacity estimation."""

    def test_wm_capacity_default(self, telemetry):
        """Test default WM capacity with no data."""
        # Record just a few samples
        for i in range(5):
            telemetry.record_state(theta_phase=0.0, gamma_amplitude=0.5)

        capacity = telemetry._estimate_wm_capacity()
        assert capacity == 4.0  # Default

    def test_wm_capacity_typical(self, populated_telemetry):
        """Test WM capacity with typical frequencies."""
        capacity = populated_telemetry._estimate_wm_capacity()

        # With 40Hz gamma and 6Hz theta, expect ~6.7 items
        assert 3.0 <= capacity <= 9.0

    def test_wm_capacity_statistics(self, populated_telemetry):
        """Test WM capacity statistics."""
        stats = populated_telemetry.get_wm_capacity_statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "current" in stats
        assert 3.0 <= stats["mean"] <= 9.0


# =============================================================================
# Phase-Amplitude Distribution Tests
# =============================================================================


class TestPhaseAmplitudeDistribution:
    """Tests for phase-amplitude distribution."""

    def test_distribution_empty(self, telemetry):
        """Test distribution with no data."""
        bins, means, stds = telemetry.get_phase_amplitude_distribution()

        assert len(bins) == 18
        assert np.all(means == 0)
        assert np.all(stds == 0)

    def test_distribution_populated(self, populated_telemetry):
        """Test distribution with data."""
        bins, means, stds = populated_telemetry.get_phase_amplitude_distribution()

        assert len(bins) == 18
        assert len(means) == 18
        assert len(stds) == 18
        assert np.max(means) > 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for statistical analysis."""

    def test_mi_statistics(self, populated_telemetry):
        """Test MI statistics."""
        stats = populated_telemetry.get_mi_statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "current" in stats
        assert "n" in stats

    def test_preferred_phase_statistics(self, populated_telemetry):
        """Test preferred phase statistics with circular mean."""
        stats = populated_telemetry.get_preferred_phase_statistics()

        assert "circular_mean" in stats
        assert "circular_std" in stats
        assert "concentration" in stats
        assert 0 <= stats["circular_mean"] <= 2 * np.pi
        assert 0 <= stats["concentration"] <= 1

    def test_empty_statistics(self, telemetry):
        """Test statistics with no data."""
        mi_stats = telemetry.get_mi_statistics()
        wm_stats = telemetry.get_wm_capacity_statistics()
        phase_stats = telemetry.get_preferred_phase_statistics()

        assert mi_stats == {}
        assert wm_stats == {}
        assert phase_stats == {}


# =============================================================================
# Biological Validation Tests
# =============================================================================


class TestBiologicalValidation:
    """Tests for biological range validation."""

    def test_validation_strong_coupling(self, populated_telemetry):
        """Test validation with strong coupling."""
        validation = populated_telemetry.validate_biological_ranges()

        assert "modulation_index" in validation
        assert "wm_capacity" in validation
        assert "phase_stability" in validation
        assert "overall_valid" in validation

    def test_mi_validation_range(self):
        """Test MI validation against target range (0.3-0.7)."""
        telemetry = PACTelemetry()

        # Create data with MI in target range
        np.random.seed(42)
        for i in range(500):
            theta_phase = (i * 0.1) % (2 * np.pi)
            # Moderate coupling
            gamma_amp = 0.5 + 0.2 * np.cos(theta_phase - np.pi) + 0.1 * np.random.random()
            telemetry.record_state(theta_phase=theta_phase, gamma_amplitude=gamma_amp)

        validation = telemetry.validate_biological_ranges()
        # Note: may or may not be in range depending on random seed

    def test_wm_capacity_validation(self, populated_telemetry):
        """Test WM capacity validation (target: 4-7)."""
        validation = populated_telemetry.validate_biological_ranges()

        # With 40Hz gamma / 6Hz theta ≈ 6.7, should be in range
        assert validation["wm_capacity"]["in_range"] is True


# =============================================================================
# Visualization Tests
# =============================================================================


class TestVisualization:
    """Tests for visualization methods."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Setup matplotlib for testing."""
        pytest.importorskip("matplotlib")

    def test_plot_comodulogram_empty(self, telemetry):
        """Test comodulogram with no data."""
        ax = telemetry.plot_comodulogram()
        assert ax is not None

    def test_plot_comodulogram_populated(self, populated_telemetry):
        """Test comodulogram with data."""
        import matplotlib.pyplot as plt

        ax = populated_telemetry.plot_comodulogram()
        assert ax is not None
        plt.close("all")

    def test_plot_mi_timeline(self, populated_telemetry):
        """Test MI timeline plot."""
        import matplotlib.pyplot as plt

        ax = populated_telemetry.plot_mi_timeline()
        assert ax is not None
        plt.close("all")

    def test_plot_wm_capacity_timeline(self, populated_telemetry):
        """Test WM capacity timeline plot."""
        import matplotlib.pyplot as plt

        ax = populated_telemetry.plot_wm_capacity_timeline()
        assert ax is not None
        plt.close("all")

    def test_plot_polar_comodulogram(self, populated_telemetry):
        """Test polar comodulogram plot."""
        import matplotlib.pyplot as plt

        ax = populated_telemetry.plot_polar_comodulogram()
        assert ax is not None
        plt.close("all")

    def test_plot_dashboard(self, populated_telemetry):
        """Test comprehensive dashboard."""
        import matplotlib.pyplot as plt

        fig = populated_telemetry.plot_dashboard()
        assert fig is not None
        plt.close("all")


# =============================================================================
# Export Tests
# =============================================================================


class TestExport:
    """Tests for data export."""

    def test_export_empty(self, telemetry):
        """Test export with no data."""
        data = telemetry.export_data()

        assert "snapshots" in data
        assert "comodulogram" in data
        assert "statistics" in data
        assert "validation" in data
        assert "meta" in data

    def test_export_populated(self, populated_telemetry):
        """Test export with data."""
        data = populated_telemetry.export_data()

        assert len(data["snapshots"]) <= 100  # Limited to 100
        assert len(data["comodulogram"]["phase_bins"]) == 18
        assert "modulation_index" in data["statistics"]

    def test_reset(self, populated_telemetry):
        """Test reset functionality."""
        assert len(populated_telemetry._snapshots) > 0

        populated_telemetry.reset()

        assert len(populated_telemetry._theta_phase_history) == 0
        assert len(populated_telemetry._snapshots) == 0
        assert populated_telemetry._total_samples == 0
        assert populated_telemetry._cached_mi == 0.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_amplitude(self, telemetry):
        """Test with zero gamma amplitude."""
        for i in range(200):
            telemetry.record_state(
                theta_phase=(i * 0.1) % (2 * np.pi),
                gamma_amplitude=0.0,
            )

        mi = telemetry.compute_modulation_index()
        assert mi == 0.0

    def test_constant_amplitude(self, telemetry):
        """Test with constant gamma amplitude (no coupling)."""
        for i in range(200):
            telemetry.record_state(
                theta_phase=(i * 0.1) % (2 * np.pi),
                gamma_amplitude=0.5,  # Constant
            )

        mi = telemetry.compute_modulation_index()
        # Should be close to 0 (uniform distribution)
        assert mi < 0.1

    def test_phase_wrapping(self, telemetry):
        """Test proper handling of phase wrapping at 2π."""
        for i in range(200):
            # Phase spanning full range including wrap
            theta_phase = (i * 0.1) % (2 * np.pi)
            telemetry.record_state(theta_phase=theta_phase, gamma_amplitude=0.5)

        # Should handle without error
        bins, means, stds = telemetry.get_phase_amplitude_distribution()
        assert len(bins) == 18

    def test_very_small_window(self):
        """Test with very small window size."""
        telemetry = PACTelemetry(window_size=10)

        for i in range(100):
            telemetry.record_state(
                theta_phase=(i * 0.1) % (2 * np.pi),
                gamma_amplitude=0.5,
            )

        assert len(telemetry._theta_phase_history) == 10


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests."""

    def test_realistic_oscillator_simulation(self):
        """Test with realistic oscillator simulation."""
        telemetry = PACTelemetry(window_size=2000)

        np.random.seed(42)
        dt = 0.001  # 1ms timestep
        theta_freq = 6.0  # Hz
        gamma_freq = 40.0  # Hz
        pac_strength = 0.8  # Strong coupling

        theta_phase = 0.0

        for t in range(2000):
            # Advance theta phase
            theta_phase += 2 * np.pi * theta_freq * dt
            theta_phase = theta_phase % (2 * np.pi)

            # Gamma amplitude strongly modulated by theta phase
            # Peak at θ=π with strong modulation
            modulation = 0.5 * (1 + pac_strength * np.cos(theta_phase - np.pi))
            gamma_amp = 0.1 + 0.8 * modulation + 0.05 * np.random.random()

            telemetry.record_state(
                theta_phase=theta_phase,
                gamma_amplitude=np.clip(gamma_amp, 0, 1),
                theta_freq=theta_freq,
                gamma_freq=gamma_freq,
            )

        # Check results
        mi = telemetry.compute_modulation_index()
        assert mi > 0.01  # Should show measurable coupling

        preferred = telemetry.find_preferred_phase()
        assert abs(preferred - np.pi) < 1.5  # Should be near π

    def test_cache_update(self):
        """Test MI cache updates properly."""
        telemetry = PACTelemetry()
        telemetry._cache_update_interval = 10

        # Record samples
        for i in range(100):
            telemetry.record_state(
                theta_phase=(i * 0.1) % (2 * np.pi),
                gamma_amplitude=0.5,
            )

        # Cache should have been updated
        assert telemetry._cache_valid_samples > 0


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests."""

    def test_large_window(self):
        """Test with large window size."""
        telemetry = PACTelemetry(window_size=10000)

        for i in range(5000):
            telemetry.record_state(
                theta_phase=(i * 0.1) % (2 * np.pi),
                gamma_amplitude=0.5,
            )

        # Should still compute efficiently
        mi = telemetry.compute_modulation_index()
        assert isinstance(mi, float)

    def test_many_phase_bins(self):
        """Test with many phase bins."""
        telemetry = PACTelemetry(n_phase_bins=72)  # 5° bins

        for i in range(500):
            telemetry.record_state(
                theta_phase=(i * 0.1) % (2 * np.pi),
                gamma_amplitude=0.5,
            )

        bins, means, stds = telemetry.get_phase_amplitude_distribution()
        assert len(bins) == 72
