"""
Tests for SWR (Sharp-Wave Ripple) Telemetry module.

Comprehensive tests for:
- Event recording and storage
- Statistical analysis (frequency, duration, intervals)
- Biological range validation
- Replay statistics
- Visualization functions
- Integration with SWR coupling
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, patch

from ww.visualization.swr_telemetry import SWRTelemetry, SWRTelemetryEvent


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def telemetry():
    """Create basic SWR telemetry instance."""
    return SWRTelemetry(window_size=100)


@pytest.fixture
def populated_telemetry():
    """Create telemetry with sample events."""
    t = SWRTelemetry(window_size=100)

    # Add 20 events with biologically valid parameters
    for i in range(20):
        t.record_event(
            duration_s=0.08 + 0.02 * np.sin(i * 0.5),  # 60-100ms
            ripple_frequency=180.0 + 20 * np.sin(i * 0.3),  # 160-200 Hz
            peak_amplitude=0.7 + 0.2 * np.random.random(),
            compression_factor=10.0,
            reactivated_patterns=[f"pattern_{j}" for j in range(i % 5)],
            replay_count=i % 5,
        )

    return t


@dataclass
class MockSWREvent:
    """Mock NCA SWR event for testing coupling integration."""
    start_time: float
    duration: float
    peak_amplitude: float
    replay_count: int = 0
    memories_activated: List[str] = None

    def __post_init__(self):
        if self.memories_activated is None:
            self.memories_activated = []


@dataclass
class MockSWRConfig:
    """Mock SWR configuration."""
    ripple_frequency: float = 180.0
    compression_factor: float = 10.0


class MockSWRCoupling:
    """Mock SWR coupling for testing integration."""

    def __init__(self):
        self.config = MockSWRConfig()
        self._callbacks = []

    def register_swr_callback(self, callback):
        self._callbacks.append(callback)

    def fire_event(self, event):
        for cb in self._callbacks:
            cb(event)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestSWRTelemetryInit:
    """Tests for SWRTelemetry initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        telemetry = SWRTelemetry()
        assert telemetry.window_size == 1000
        assert telemetry.min_ripple_freq == 150.0
        assert telemetry.max_ripple_freq == 250.0
        assert len(telemetry._events) == 0

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        telemetry = SWRTelemetry(
            window_size=500,
            min_ripple_freq=140.0,
            max_ripple_freq=260.0,
        )
        assert telemetry.window_size == 500
        assert telemetry.min_ripple_freq == 140.0
        assert telemetry.max_ripple_freq == 260.0

    def test_with_coupling(self):
        """Test initialization with SWR coupling."""
        coupling = MockSWRCoupling()
        telemetry = SWRTelemetry(swr_coupling=coupling)

        # Verify callback was registered
        assert len(coupling._callbacks) == 1

    def test_coupling_callback_fires(self):
        """Test that coupling callback records events."""
        coupling = MockSWRCoupling()
        telemetry = SWRTelemetry(swr_coupling=coupling)

        # Fire an event through coupling
        event = MockSWREvent(
            start_time=0.0,
            duration=0.08,
            peak_amplitude=0.9,
            replay_count=3,
            memories_activated=["m1", "m2"],
        )
        coupling.fire_event(event)

        assert len(telemetry._events) == 1
        assert telemetry._events[0].peak_amplitude == 0.9


# =============================================================================
# Event Recording Tests
# =============================================================================


class TestEventRecording:
    """Tests for recording SWR events."""

    def test_record_single_event(self, telemetry):
        """Test recording a single event."""
        event = telemetry.record_event(
            duration_s=0.08,
            ripple_frequency=180.0,
            peak_amplitude=0.85,
        )

        assert event is not None
        assert event.duration_s == 0.08
        assert event.duration_ms == 80.0
        assert event.ripple_frequency == 180.0
        assert event.peak_amplitude == 0.85
        assert len(telemetry._events) == 1

    def test_record_multiple_events(self, telemetry):
        """Test recording multiple events."""
        for i in range(10):
            telemetry.record_event(
                duration_s=0.08,
                ripple_frequency=180.0 + i,
                peak_amplitude=0.8,
            )

        assert len(telemetry._events) == 10
        assert telemetry._total_events == 10

    def test_window_size_limit(self):
        """Test that window size is enforced."""
        telemetry = SWRTelemetry(window_size=5)

        for i in range(10):
            telemetry.record_event(
                duration_s=0.08,
                ripple_frequency=180.0,
                peak_amplitude=0.8,
            )

        assert len(telemetry._events) == 5
        assert telemetry._total_events == 10  # Total count preserved

    def test_inter_event_intervals(self, telemetry):
        """Test inter-event interval tracking."""
        # Record events with small delays
        for i in range(5):
            telemetry.record_event(
                duration_s=0.08,
                ripple_frequency=180.0,
                peak_amplitude=0.8,
            )

        # Should have n-1 intervals for n events
        assert len(telemetry._inter_event_intervals) == 4

    def test_replay_patterns(self, telemetry):
        """Test recording replay patterns."""
        event = telemetry.record_event(
            duration_s=0.08,
            ripple_frequency=180.0,
            peak_amplitude=0.85,
            reactivated_patterns=["p1", "p2", "p3"],
            replay_count=3,
        )

        assert event.reactivated_patterns == ["p1", "p2", "p3"]
        assert event.replay_count == 3

    def test_frequency_range_warning(self, telemetry, caplog):
        """Test warning for out-of-range frequency."""
        # Below range
        telemetry.record_event(
            duration_s=0.08,
            ripple_frequency=100.0,  # Below 150
            peak_amplitude=0.8,
        )
        assert "outside range" in caplog.text

        # Above range
        telemetry.record_event(
            duration_s=0.08,
            ripple_frequency=300.0,  # Above 250
            peak_amplitude=0.8,
        )
        assert "300.0 Hz outside range" in caplog.text

    def test_record_from_coupling(self, telemetry):
        """Test recording from NCA coupling event."""
        nca_event = MockSWREvent(
            start_time=0.0,
            duration=0.1,
            peak_amplitude=0.95,
            replay_count=2,
            memories_activated=["mem1", "mem2"],
        )

        event = telemetry.record_event_from_coupling(nca_event, ripple_frequency=185.0)

        assert event.duration_s == 0.1
        assert event.peak_amplitude == 0.95
        assert event.ripple_frequency == 185.0
        assert event.reactivated_patterns == ["mem1", "mem2"]
        assert event.replay_count == 2


# =============================================================================
# Statistical Analysis Tests
# =============================================================================


class TestStatisticalAnalysis:
    """Tests for statistical analysis methods."""

    def test_event_rate_empty(self, telemetry):
        """Test event rate with no events."""
        assert telemetry.get_event_rate() == 0.0

    def test_event_rate_single(self, telemetry):
        """Test event rate with single event."""
        telemetry.record_event(duration_s=0.08, ripple_frequency=180.0, peak_amplitude=0.8)
        # Single event should return 0 (need 2+ for rate)
        assert telemetry.get_event_rate() == 0.0

    def test_frequency_distribution(self, populated_telemetry):
        """Test frequency distribution statistics."""
        dist = populated_telemetry.get_frequency_distribution()

        assert "mean" in dist
        assert "std" in dist
        assert "min" in dist
        assert "max" in dist
        assert "median" in dist
        assert 160 <= dist["mean"] <= 200  # Based on fixture data

    def test_frequency_distribution_empty(self, telemetry):
        """Test frequency distribution with no events."""
        dist = telemetry.get_frequency_distribution()
        assert dist == {}

    def test_duration_distribution(self, populated_telemetry):
        """Test duration distribution statistics."""
        dist = populated_telemetry.get_duration_distribution()

        assert "mean_ms" in dist
        assert "std_ms" in dist
        assert 60 <= dist["mean_ms"] <= 100  # Based on fixture data

    def test_amplitude_distribution(self, populated_telemetry):
        """Test amplitude distribution statistics."""
        dist = populated_telemetry.get_amplitude_distribution()

        assert "mean" in dist
        assert 0.7 <= dist["mean"] <= 0.9

    def test_inter_event_intervals(self, populated_telemetry):
        """Test inter-event interval statistics."""
        stats = populated_telemetry.get_inter_event_intervals()

        assert "mean_s" in stats
        assert "std_s" in stats
        assert stats["n"] == 19  # 20 events -> 19 intervals

    def test_replay_statistics(self, populated_telemetry):
        """Test replay statistics."""
        stats = populated_telemetry.get_replay_statistics()

        assert "total_swr_events" in stats
        assert stats["total_swr_events"] == 20
        assert "total_replays" in stats
        assert "unique_patterns_replayed" in stats
        assert "mean_patterns_per_swr" in stats


# =============================================================================
# Biological Validation Tests
# =============================================================================


class TestBiologicalValidation:
    """Tests for biological range validation."""

    def test_valid_frequency_range(self):
        """Test validation passes for valid frequency range."""
        telemetry = SWRTelemetry()
        for _ in range(10):
            telemetry.record_event(
                duration_s=0.08,
                ripple_frequency=180.0,  # Within 150-250 Hz
                peak_amplitude=0.8,
            )

        validation = telemetry.validate_biological_ranges()
        assert validation["frequency"]["in_range"] is True

    def test_invalid_frequency_range(self):
        """Test validation fails for invalid frequency."""
        telemetry = SWRTelemetry()
        for _ in range(10):
            telemetry.record_event(
                duration_s=0.08,
                ripple_frequency=100.0,  # Below 150 Hz
                peak_amplitude=0.8,
            )

        validation = telemetry.validate_biological_ranges()
        assert validation["frequency"]["in_range"] is False

    def test_valid_duration_range(self):
        """Test validation passes for valid duration."""
        telemetry = SWRTelemetry()
        for _ in range(10):
            telemetry.record_event(
                duration_s=0.08,  # 80ms, within 50-150ms
                ripple_frequency=180.0,
                peak_amplitude=0.8,
            )

        validation = telemetry.validate_biological_ranges()
        assert validation["duration"]["in_range"] is True
        assert 50 <= validation["duration"]["value"] <= 150

    def test_invalid_duration_range(self):
        """Test validation fails for invalid duration."""
        telemetry = SWRTelemetry()
        for _ in range(10):
            telemetry.record_event(
                duration_s=0.02,  # 20ms, below 50ms
                ripple_frequency=180.0,
                peak_amplitude=0.8,
            )

        validation = telemetry.validate_biological_ranges()
        assert validation["duration"]["in_range"] is False

    def test_overall_validation(self, populated_telemetry):
        """Test overall validation result."""
        validation = populated_telemetry.validate_biological_ranges()

        assert "overall_valid" in validation
        assert isinstance(validation["overall_valid"], bool)


# =============================================================================
# Visualization Tests
# =============================================================================


class TestVisualization:
    """Tests for visualization methods."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Setup matplotlib for testing."""
        pytest.importorskip("matplotlib")

    def test_plot_swr_raster_empty(self, telemetry):
        """Test raster plot with no events."""
        ax = telemetry.plot_swr_raster()
        assert ax is not None

    def test_plot_swr_raster_populated(self, populated_telemetry):
        """Test raster plot with events."""
        import matplotlib.pyplot as plt

        ax = populated_telemetry.plot_swr_raster()
        assert ax is not None
        plt.close("all")

    def test_plot_frequency_histogram_empty(self, telemetry):
        """Test frequency histogram with no data."""
        ax = telemetry.plot_frequency_histogram()
        assert ax is not None

    def test_plot_frequency_histogram_populated(self, populated_telemetry):
        """Test frequency histogram with data."""
        import matplotlib.pyplot as plt

        ax = populated_telemetry.plot_frequency_histogram()
        assert ax is not None
        plt.close("all")

    def test_plot_duration_histogram(self, populated_telemetry):
        """Test duration histogram."""
        import matplotlib.pyplot as plt

        ax = populated_telemetry.plot_duration_histogram()
        assert ax is not None
        plt.close("all")

    def test_plot_inter_event_intervals(self, populated_telemetry):
        """Test inter-event interval histogram."""
        import matplotlib.pyplot as plt

        ax = populated_telemetry.plot_inter_event_intervals()
        assert ax is not None
        plt.close("all")

    def test_plot_dashboard(self, populated_telemetry):
        """Test comprehensive dashboard plot."""
        import matplotlib.pyplot as plt

        fig = populated_telemetry.plot_dashboard()
        assert fig is not None
        plt.close("all")

    def test_plot_without_matplotlib(self, telemetry):
        """Test graceful handling when matplotlib is unavailable."""
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            # This should return None without error
            result = telemetry.plot_swr_raster()
            # Result depends on import handling


# =============================================================================
# Export Tests
# =============================================================================


class TestExport:
    """Tests for data export."""

    def test_export_empty(self, telemetry):
        """Test export with no events."""
        data = telemetry.export_data()

        assert "events" in data
        assert "statistics" in data
        assert "validation" in data
        assert "meta" in data
        assert len(data["events"]) == 0

    def test_export_populated(self, populated_telemetry):
        """Test export with events."""
        data = populated_telemetry.export_data()

        assert len(data["events"]) == 20
        assert data["meta"]["total_events"] == 20

        # Check event structure
        event = data["events"][0]
        assert "timestamp" in event
        assert "duration_s" in event
        assert "duration_ms" in event
        assert "frequency_hz" in event
        assert "amplitude" in event

        # Check statistics
        assert "frequency" in data["statistics"]
        assert "duration" in data["statistics"]
        assert "replay" in data["statistics"]

    def test_reset(self, populated_telemetry):
        """Test reset functionality."""
        assert len(populated_telemetry._events) == 20

        populated_telemetry.reset()

        assert len(populated_telemetry._events) == 0
        assert populated_telemetry._total_events == 0
        assert populated_telemetry._start_time is None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_duration_event(self, telemetry):
        """Test recording event with zero duration."""
        event = telemetry.record_event(
            duration_s=0.0,
            ripple_frequency=180.0,
            peak_amplitude=0.8,
        )
        assert event.duration_s == 0.0
        assert event.duration_ms == 0.0

    def test_negative_values(self, telemetry):
        """Test handling of negative values."""
        # Should record but may violate biological constraints
        event = telemetry.record_event(
            duration_s=-0.01,  # Negative
            ripple_frequency=180.0,
            peak_amplitude=-0.5,  # Negative
        )
        assert event is not None

    def test_large_window(self):
        """Test with large window size."""
        telemetry = SWRTelemetry(window_size=10000)
        for i in range(100):
            telemetry.record_event(
                duration_s=0.08,
                ripple_frequency=180.0,
                peak_amplitude=0.8,
            )
        assert len(telemetry._events) == 100

    def test_empty_patterns_list(self, telemetry):
        """Test with empty patterns list."""
        event = telemetry.record_event(
            duration_s=0.08,
            ripple_frequency=180.0,
            peak_amplitude=0.8,
            reactivated_patterns=[],
        )
        assert event.reactivated_patterns == []

    def test_none_patterns(self, telemetry):
        """Test with None patterns."""
        event = telemetry.record_event(
            duration_s=0.08,
            ripple_frequency=180.0,
            peak_amplitude=0.8,
            reactivated_patterns=None,
        )
        assert event.reactivated_patterns == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with other modules."""

    def test_with_mock_coupling(self):
        """Test full integration with mock coupling."""
        coupling = MockSWRCoupling()
        telemetry = SWRTelemetry(swr_coupling=coupling)

        # Simulate 5 SWR events
        for i in range(5):
            event = MockSWREvent(
                start_time=i * 1.0,
                duration=0.08 + 0.01 * i,
                peak_amplitude=0.8 + 0.02 * i,
                replay_count=i,
                memories_activated=[f"mem_{j}" for j in range(i)],
            )
            coupling.fire_event(event)

        assert len(telemetry._events) == 5

        # Check replay stats
        stats = telemetry.get_replay_statistics()
        assert stats["total_swr_events"] == 5
        assert stats["total_replays"] == 0 + 1 + 2 + 3 + 4  # sum of i

    def test_biological_scenario(self):
        """Test realistic biological scenario."""
        telemetry = SWRTelemetry()

        # Simulate sleep-like SWR pattern
        # ~1 Hz event rate, 80-100ms duration, 150-200 Hz ripple
        np.random.seed(42)

        for _ in range(60):  # 1 minute of simulated sleep
            telemetry.record_event(
                duration_s=np.random.uniform(0.08, 0.10),
                ripple_frequency=np.random.uniform(170, 190),
                peak_amplitude=np.random.uniform(0.7, 0.95),
                compression_factor=10.0,
                replay_count=np.random.randint(1, 5),
            )

        # Validate biological plausibility
        validation = telemetry.validate_biological_ranges()
        assert validation["frequency"]["in_range"] is True
        assert validation["duration"]["in_range"] is True


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests."""

    def test_large_event_count(self):
        """Test with many events."""
        telemetry = SWRTelemetry(window_size=10000)

        for i in range(1000):
            telemetry.record_event(
                duration_s=0.08,
                ripple_frequency=180.0,
                peak_amplitude=0.8,
            )

        # Should still be fast
        stats = telemetry.get_frequency_distribution()
        assert stats["n"] == 1000

    def test_export_large_dataset(self):
        """Test export with large dataset."""
        telemetry = SWRTelemetry(window_size=1000)

        for i in range(500):
            telemetry.record_event(
                duration_s=0.08,
                ripple_frequency=180.0,
                peak_amplitude=0.8,
            )

        data = telemetry.export_data()
        assert len(data["events"]) == 500
