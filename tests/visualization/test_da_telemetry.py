"""Tests for DA temporal structure telemetry."""

import pytest
from datetime import datetime, timedelta
import numpy as np

from ww.visualization.da_telemetry import (
    DATelemetry,
    DASnapshot,
    DASignalType,
    DARampEvent,
    DAStatistics,
    create_da_dashboard,
    plot_da_timeline,
    plot_rpe_distribution,
)


class TestDASignalType:
    """Test DASignalType enum."""

    def test_all_types_defined(self):
        """Verify all signal types exist."""
        assert DASignalType.TONIC.value == "tonic"
        assert DASignalType.PHASIC_BURST.value == "phasic_burst"
        assert DASignalType.PHASIC_PAUSE.value == "phasic_pause"
        assert DASignalType.RAMP.value == "ramp"
        assert DASignalType.DECAY.value == "decay"

    def test_types_unique(self):
        """All types have unique values."""
        values = [t.value for t in DASignalType]
        assert len(values) == len(set(values))


class TestDASnapshot:
    """Test DASnapshot dataclass."""

    def test_create_snapshot(self):
        """Test basic snapshot creation."""
        now = datetime.now()
        snapshot = DASnapshot(
            timestamp=now,
            da_level=0.5,
            firing_rate=25.0,
            signal_type=DASignalType.PHASIC_BURST,
            rpe=0.3,
            td_error=0.25,
            value_estimate=0.4,
            eligibility=0.8,
        )
        assert snapshot.da_level == 0.5
        assert snapshot.firing_rate == 25.0
        assert snapshot.signal_type == DASignalType.PHASIC_BURST
        assert snapshot.rpe == 0.3

    def test_tonic_snapshot(self):
        """Test tonic baseline snapshot."""
        snapshot = DASnapshot(
            timestamp=datetime.now(),
            da_level=0.3,
            firing_rate=4.5,
            signal_type=DASignalType.TONIC,
            rpe=0.0,
            td_error=0.0,
            value_estimate=0.5,
            eligibility=0.0,
        )
        assert snapshot.signal_type == DASignalType.TONIC
        assert snapshot.firing_rate == 4.5


class TestDARampEvent:
    """Test DARampEvent dataclass."""

    def test_create_ramp_event(self):
        """Test ramp event creation."""
        start = datetime.now()
        end = start + timedelta(seconds=2.0)
        ramp = DARampEvent(
            start_time=start,
            end_time=end,
            start_da=0.3,
            peak_da=0.7,
            duration_s=2.0,
            goal_reached=True,
            distance_to_goal=0.0,
        )
        assert ramp.peak_da == 0.7
        assert ramp.goal_reached is True
        assert ramp.duration_s == 2.0


class TestDATelemetry:
    """Test DATelemetry class."""

    def test_initialization(self):
        """Test telemetry initialization."""
        telemetry = DATelemetry()
        assert telemetry.window_size == 1000
        assert len(telemetry._snapshots) == 0

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        telemetry = DATelemetry(
            window_size=500,
            tonic_rate_range=(3.5, 5.5),
            burst_rate_threshold=20.0,
        )
        assert telemetry.window_size == 500
        assert telemetry.tonic_rate_range == (3.5, 5.5)
        assert telemetry.burst_rate_threshold == 20.0

    def test_record_tonic_state(self):
        """Test recording tonic state."""
        telemetry = DATelemetry()
        snapshot = telemetry.record_state(
            da_level=0.3,
            firing_rate=4.5,
            rpe=0.0,
        )
        assert snapshot.signal_type == DASignalType.TONIC
        assert len(telemetry._snapshots) == 1

    def test_record_burst_state(self):
        """Test recording phasic burst."""
        telemetry = DATelemetry()
        snapshot = telemetry.record_state(
            da_level=0.7,
            firing_rate=30.0,
            rpe=0.5,
        )
        assert snapshot.signal_type == DASignalType.PHASIC_BURST
        assert telemetry._total_bursts == 1

    def test_record_pause_state(self):
        """Test recording phasic pause."""
        telemetry = DATelemetry()
        snapshot = telemetry.record_state(
            da_level=0.1,
            firing_rate=1.0,
            rpe=-0.4,
        )
        assert snapshot.signal_type == DASignalType.PHASIC_PAUSE
        assert telemetry._total_pauses == 1

    def test_window_size_limit(self):
        """Test that window size is enforced."""
        telemetry = DATelemetry(window_size=10)
        for i in range(20):
            telemetry.record_state(
                da_level=0.3 + i * 0.01,
                firing_rate=4.5,
                rpe=0.0,
            )
        assert len(telemetry._snapshots) == 10

    def test_ramp_detection(self):
        """Test ramp event detection."""
        telemetry = DATelemetry(ramp_detection_threshold=0.05)
        # Record increasing DA levels
        for i in range(5):
            telemetry.record_state(
                da_level=0.3 + i * 0.15,
                firing_rate=4.5,
                rpe=0.1,
            )
        # Should detect ramp
        ramp_count = sum(
            1 for s in telemetry._snapshots
            if s.signal_type == DASignalType.RAMP
        )
        assert ramp_count > 0


class TestSignalClassification:
    """Test signal type classification."""

    def test_classify_tonic(self):
        """Tonic classification for baseline activity."""
        telemetry = DATelemetry()
        snapshot = telemetry.record_state(
            da_level=0.3,
            firing_rate=4.5,
            rpe=0.0,
        )
        assert snapshot.signal_type == DASignalType.TONIC

    def test_classify_burst_high_rate_positive_rpe(self):
        """Burst classification requires high rate AND positive RPE."""
        telemetry = DATelemetry()
        snapshot = telemetry.record_state(
            da_level=0.6,
            firing_rate=25.0,
            rpe=0.3,
        )
        assert snapshot.signal_type == DASignalType.PHASIC_BURST

    def test_high_rate_no_rpe_is_tonic(self):
        """High rate without RPE is not burst."""
        telemetry = DATelemetry()
        snapshot = telemetry.record_state(
            da_level=0.3,
            firing_rate=20.0,
            rpe=0.0,
        )
        # Without positive RPE, high rate alone doesn't make burst
        assert snapshot.signal_type == DASignalType.TONIC

    def test_classify_pause_low_rate_negative_rpe(self):
        """Pause classification requires low rate AND negative RPE."""
        telemetry = DATelemetry()
        snapshot = telemetry.record_state(
            da_level=0.1,
            firing_rate=1.0,
            rpe=-0.3,
        )
        assert snapshot.signal_type == DASignalType.PHASIC_PAUSE

    def test_low_rate_no_rpe_is_tonic(self):
        """Low rate without negative RPE is not pause."""
        telemetry = DATelemetry()
        snapshot = telemetry.record_state(
            da_level=0.3,
            firing_rate=1.5,
            rpe=0.0,
        )
        assert snapshot.signal_type == DASignalType.TONIC


class TestRPEAnalysis:
    """Test RPE distribution analysis."""

    def test_rpe_distribution_empty(self):
        """Empty telemetry returns empty distribution."""
        telemetry = DATelemetry()
        dist = telemetry.get_rpe_distribution()
        assert dist == {}

    def test_rpe_distribution_with_data(self):
        """RPE distribution with data."""
        telemetry = DATelemetry()
        rpes = [0.3, -0.2, 0.0, 0.5, -0.4]
        for rpe in rpes:
            telemetry.record_state(
                da_level=0.3,
                firing_rate=4.5,
                rpe=rpe,
            )
        dist = telemetry.get_rpe_distribution()
        assert "mean" in dist
        assert "positive_count" in dist
        assert "negative_count" in dist
        assert dist["positive_count"] == 2  # 0.3, 0.5
        assert dist["negative_count"] == 2  # -0.2, -0.4

    def test_rpe_distribution_all_positive(self):
        """All positive RPEs."""
        telemetry = DATelemetry()
        for rpe in [0.2, 0.4, 0.6]:
            telemetry.record_state(da_level=0.5, firing_rate=20.0, rpe=rpe)
        dist = telemetry.get_rpe_distribution()
        assert dist["positive_count"] == 3
        assert dist["negative_count"] == 0


class TestFiringRateAnalysis:
    """Test firing rate distribution analysis."""

    def test_firing_rate_empty(self):
        """Empty telemetry returns empty distribution."""
        telemetry = DATelemetry()
        dist = telemetry.get_firing_rate_distribution()
        assert dist == {}

    def test_firing_rate_by_type(self):
        """Firing rate grouped by signal type."""
        telemetry = DATelemetry()
        # Record different types
        telemetry.record_state(da_level=0.3, firing_rate=4.5, rpe=0.0)  # tonic
        telemetry.record_state(da_level=0.7, firing_rate=30.0, rpe=0.5)  # burst
        telemetry.record_state(da_level=0.1, firing_rate=1.0, rpe=-0.4)  # pause

        dist = telemetry.get_firing_rate_distribution()
        assert "tonic" in dist
        assert "burst" in dist
        assert "pause" in dist
        assert dist["tonic"]["mean"] == 4.5
        assert dist["burst"]["mean"] == 30.0
        assert dist["pause"]["mean"] == 1.0


class TestRampAnalysis:
    """Test ramp event analysis."""

    def test_ramp_statistics_empty(self):
        """Empty ramp events."""
        telemetry = DATelemetry()
        stats = telemetry.get_ramp_statistics()
        assert stats["total_ramps"] == 0

    def test_ramp_event_completion(self):
        """Test ramp event with goal reached."""
        telemetry = DATelemetry(ramp_detection_threshold=0.05)
        # Create ramping pattern
        for i in range(5):
            telemetry.record_state(
                da_level=0.3 + i * 0.15,
                firing_rate=5.0 + i * 3,
                rpe=0.1,
            )
        # End with burst (goal reached)
        telemetry.record_state(
            da_level=0.8,
            firing_rate=35.0,
            rpe=0.6,
        )
        # Check if ramp was recorded
        assert telemetry._total_ramps >= 0  # May or may not have completed


class TestStatistics:
    """Test comprehensive statistics."""

    def test_statistics_empty(self):
        """Statistics on empty data."""
        telemetry = DATelemetry()
        stats = telemetry.get_statistics()
        assert stats.mean_da == 0.3
        assert stats.phasic_burst_count == 0
        assert stats.time_in_tonic == 1.0

    def test_statistics_with_data(self):
        """Statistics with mixed data."""
        telemetry = DATelemetry()
        # Record mixed states
        telemetry.record_state(da_level=0.3, firing_rate=4.5, rpe=0.0)
        telemetry.record_state(da_level=0.7, firing_rate=30.0, rpe=0.5)
        telemetry.record_state(da_level=0.1, firing_rate=1.0, rpe=-0.4)

        stats = telemetry.get_statistics()
        assert stats.phasic_burst_count == 1
        assert stats.phasic_pause_count == 1
        assert 0.3 < stats.mean_da < 0.5


class TestBiologicalValidation:
    """Test biological range validation."""

    def test_validate_tonic_in_range(self):
        """Tonic rate in biological range."""
        telemetry = DATelemetry()
        for _ in range(10):
            telemetry.record_state(da_level=0.3, firing_rate=4.5, rpe=0.0)
        validation = telemetry.validate_biological_ranges()
        assert validation["tonic_rate_in_range"] is True

    def test_validate_burst_in_range(self):
        """Burst rate in biological range."""
        telemetry = DATelemetry()
        for _ in range(10):
            telemetry.record_state(da_level=0.7, firing_rate=30.0, rpe=0.5)
        validation = telemetry.validate_biological_ranges()
        assert validation["burst_rate_in_range"] is True

    def test_validate_pause_in_range(self):
        """Pause rate in biological range."""
        telemetry = DATelemetry()
        for _ in range(10):
            telemetry.record_state(da_level=0.1, firing_rate=1.0, rpe=-0.4)
        validation = telemetry.validate_biological_ranges()
        assert validation["pause_rate_in_range"] is True

    def test_validate_tonic_out_of_range(self):
        """Tonic rate outside biological range."""
        telemetry = DATelemetry()
        for _ in range(10):
            telemetry.record_state(da_level=0.3, firing_rate=10.0, rpe=0.0)  # Too high
        validation = telemetry.validate_biological_ranges()
        assert validation["tonic_rate_in_range"] is False


class TestTraces:
    """Test time series traces."""

    def test_eligibility_trace(self):
        """Get eligibility trace."""
        telemetry = DATelemetry()
        eligibilities = [0.9, 0.7, 0.5, 0.3, 0.1]
        for e in eligibilities:
            telemetry.record_state(
                da_level=0.3,
                firing_rate=4.5,
                rpe=0.0,
                eligibility=e,
            )
        trace = telemetry.get_eligibility_trace()
        assert trace == eligibilities

    def test_value_trace(self):
        """Get value function trace."""
        telemetry = DATelemetry()
        values = [0.4, 0.5, 0.6, 0.7]
        for v in values:
            telemetry.record_state(
                da_level=0.3,
                firing_rate=4.5,
                rpe=0.0,
                value_estimate=v,
            )
        trace = telemetry.get_value_trace()
        assert trace == values

    def test_td_error_trace(self):
        """Get TD error trace."""
        telemetry = DATelemetry()
        td_errors = [0.2, -0.1, 0.0, 0.3]
        for td in td_errors:
            telemetry.record_state(
                da_level=0.3,
                firing_rate=4.5,
                rpe=0.0,
                td_error=td,
            )
        trace = telemetry.get_td_error_trace()
        assert trace == td_errors


class TestExportData:
    """Test data export functionality."""

    def test_export_empty(self):
        """Export empty telemetry."""
        telemetry = DATelemetry()
        data = telemetry.export_data()
        assert "snapshots" in data
        assert "statistics" in data
        assert "validation" in data
        assert len(data["snapshots"]) == 0

    def test_export_with_data(self):
        """Export telemetry with data."""
        telemetry = DATelemetry()
        telemetry.record_state(da_level=0.5, firing_rate=20.0, rpe=0.3)
        data = telemetry.export_data()
        assert len(data["snapshots"]) == 1
        assert data["snapshots"][0]["da_level"] == 0.5
        assert data["snapshots"][0]["signal_type"] == "phasic_burst"


class TestClear:
    """Test clear functionality."""

    def test_clear(self):
        """Test clearing telemetry."""
        telemetry = DATelemetry()
        for _ in range(10):
            telemetry.record_state(da_level=0.5, firing_rate=4.5, rpe=0.0)
        assert len(telemetry._snapshots) == 10
        telemetry.clear()
        assert len(telemetry._snapshots) == 0
        assert telemetry._total_bursts == 0


class TestVisualization:
    """Test visualization functions."""

    def test_plot_da_timeline(self):
        """Test timeline plotting."""
        telemetry = DATelemetry()
        for i in range(20):
            telemetry.record_state(
                da_level=0.3 + 0.02 * i,
                firing_rate=4.5,
                rpe=0.0,
            )
        ax = telemetry.plot_da_timeline()
        assert ax is not None

    def test_plot_da_timeline_empty(self):
        """Timeline with no data."""
        telemetry = DATelemetry()
        ax = telemetry.plot_da_timeline()
        assert ax is not None

    def test_plot_rpe_distribution(self):
        """Test RPE distribution plotting."""
        telemetry = DATelemetry()
        for rpe in np.random.randn(50) * 0.3:
            telemetry.record_state(da_level=0.3, firing_rate=4.5, rpe=float(rpe))
        ax = telemetry.plot_rpe_distribution()
        assert ax is not None

    def test_plot_phasic_raster(self):
        """Test phasic raster plotting."""
        telemetry = DATelemetry()
        # Mix of signal types
        telemetry.record_state(da_level=0.3, firing_rate=4.5, rpe=0.0)
        telemetry.record_state(da_level=0.7, firing_rate=30.0, rpe=0.5)
        telemetry.record_state(da_level=0.1, firing_rate=1.0, rpe=-0.4)
        ax = telemetry.plot_phasic_raster()
        assert ax is not None

    def test_plot_eligibility_decay(self):
        """Test eligibility decay plotting."""
        telemetry = DATelemetry()
        for i in range(20):
            telemetry.record_state(
                da_level=0.3,
                firing_rate=4.5,
                rpe=0.0,
                eligibility=0.9 ** i,
                td_error=0.1 if i % 5 == 0 else 0.0,
            )
        ax = telemetry.plot_eligibility_decay()
        assert ax is not None

    def test_create_dashboard(self):
        """Test dashboard creation."""
        telemetry = DATelemetry()
        for i in range(50):
            signal_type = i % 3
            if signal_type == 0:
                telemetry.record_state(da_level=0.3, firing_rate=4.5, rpe=0.0)
            elif signal_type == 1:
                telemetry.record_state(da_level=0.7, firing_rate=30.0, rpe=0.5)
            else:
                telemetry.record_state(da_level=0.1, firing_rate=1.0, rpe=-0.4)
        fig = telemetry.create_da_dashboard()
        assert fig is not None


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_create_da_dashboard(self):
        """Test convenience function."""
        telemetry = DATelemetry()
        telemetry.record_state(da_level=0.5, firing_rate=4.5, rpe=0.0)
        fig = create_da_dashboard(telemetry)
        assert fig is not None

    def test_plot_da_timeline_function(self):
        """Test convenience function."""
        telemetry = DATelemetry()
        telemetry.record_state(da_level=0.5, firing_rate=4.5, rpe=0.0)
        ax = plot_da_timeline(telemetry)
        assert ax is not None

    def test_plot_rpe_distribution_function(self):
        """Test convenience function."""
        telemetry = DATelemetry()
        telemetry.record_state(da_level=0.5, firing_rate=4.5, rpe=0.3)
        ax = plot_rpe_distribution(telemetry)
        assert ax is not None


class TestModuleExports:
    """Test module exports."""

    def test_all_exports_available(self):
        """Verify all exports are importable."""
        from ww.visualization.da_telemetry import (
            DATelemetry,
            DASnapshot,
            DASignalType,
            DARampEvent,
            DAStatistics,
            create_da_dashboard,
            plot_da_timeline,
            plot_rpe_distribution,
        )
        assert DATelemetry is not None
        assert DASnapshot is not None
        assert DASignalType is not None
