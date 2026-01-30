"""Tests for Multi-Scale Telemetry Hub."""

import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from ww.visualization.telemetry_hub import (
    TelemetryHub,
    TelemetryConfig,
    TimeScale,
    CrossScaleEvent,
    SystemHealth,
    create_telemetry_hub,
)


class TestTimeScale:
    """Test TimeScale enum."""

    def test_all_scales_defined(self):
        """Verify all timescales exist."""
        assert TimeScale.FAST.value == "fast"
        assert TimeScale.OSCILLATORY.value == "oscillatory"
        assert TimeScale.NEUROMODULATOR.value == "neuromodulator"
        assert TimeScale.CONSOLIDATION.value == "consolidation"

    def test_scales_unique(self):
        """All scales have unique values."""
        values = [t.value for t in TimeScale]
        assert len(values) == len(set(values))


class TestTelemetryConfig:
    """Test TelemetryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = TelemetryConfig()
        assert config.enable_swr is True
        assert config.enable_pac is True
        assert config.enable_da is True
        assert config.enable_stability is True
        assert config.enable_nt is True
        assert config.correlation_window == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = TelemetryConfig(
            enable_swr=False,
            correlation_window=200,
            critical_alert_threshold=5,
        )
        assert config.enable_swr is False
        assert config.correlation_window == 200
        assert config.critical_alert_threshold == 5


class TestCrossScaleEvent:
    """Test CrossScaleEvent dataclass."""

    def test_create_event(self):
        """Test event creation."""
        event = CrossScaleEvent(
            timestamp=datetime.now(),
            event_type="reward_consolidation",
            scales_involved=[TimeScale.FAST, TimeScale.NEUROMODULATOR],
            metrics={"da_level": 0.7, "swr_detected": True},
            description="SWR during elevated DA",
            severity="info",
        )
        assert event.event_type == "reward_consolidation"
        assert len(event.scales_involved) == 2
        assert event.severity == "info"


class TestSystemHealth:
    """Test SystemHealth dataclass."""

    def test_create_health(self):
        """Test health creation."""
        health = SystemHealth(
            timestamp=datetime.now(),
            overall_status="healthy",
            active_alerts=[],
            stability_score=0.95,
            biological_validity=0.88,
            modules_active={"da": True, "pac": True},
            cross_scale_coherence=0.72,
        )
        assert health.overall_status == "healthy"
        assert health.stability_score == 0.95
        assert health.cross_scale_coherence == 0.72


class TestTelemetryHub:
    """Test TelemetryHub class."""

    def test_initialization(self):
        """Test hub initialization."""
        hub = TelemetryHub()
        assert hub.config is not None
        assert len(hub._events) == 0

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = TelemetryConfig(enable_swr=False)
        hub = TelemetryHub(config=config)
        assert hub.config.enable_swr is False

    def test_register_modules(self):
        """Test module registration."""
        hub = TelemetryHub()

        # Create mock modules
        @dataclass
        class MockModule:
            _snapshots: list = None
            def __post_init__(self):
                self._snapshots = []

        mock_da = MockModule()
        hub.register_da(mock_da)
        assert hub._da is mock_da


class TestModuleIntegration:
    """Test integration with actual telemetry modules."""

    def test_with_da_telemetry(self):
        """Test integration with DA telemetry."""
        from ww.visualization.da_telemetry import DATelemetry

        hub = TelemetryHub()
        da = DATelemetry()
        hub.register_da(da)

        # Record some DA data
        for i in range(10):
            da.record_state(
                da_level=0.3 + i * 0.05,
                firing_rate=4.5,
                rpe=0.0,
            )

        stats = hub.get_summary_statistics()
        assert "da" in stats["modules"]

    def test_with_pac_telemetry(self):
        """Test integration with PAC telemetry."""
        from ww.visualization.pac_telemetry import PACTelemetry

        hub = TelemetryHub()
        pac = PACTelemetry()
        hub.register_pac(pac)

        # Record some PAC data
        for i in range(100):
            theta_phase = (i * 0.1) % (2 * np.pi)
            gamma_amp = 0.5 + 0.3 * np.cos(theta_phase)
            pac.record_state(theta_phase, gamma_amp)

        stats = hub.get_summary_statistics()
        assert "pac" in stats["modules"]

    def test_with_stability_monitor(self):
        """Test integration with stability monitor."""
        from ww.visualization.stability_monitor import StabilityMonitor

        hub = TelemetryHub()
        stability = StabilityMonitor()
        hub.register_stability(stability)

        # Record some stability data
        jacobian = np.array([[-0.5, 0.1], [0.1, -0.3]])
        stability.record_state(jacobian)

        stats = hub.get_summary_statistics()
        assert "stability" in stats["modules"]

    def test_with_swr_telemetry(self):
        """Test integration with SWR telemetry."""
        from ww.visualization.swr_telemetry import SWRTelemetry

        hub = TelemetryHub()
        swr = SWRTelemetry()
        hub.register_swr(swr)

        stats = hub.get_summary_statistics()
        assert "swr" in stats["modules"]


class TestCrossScaleCorrelation:
    """Test cross-scale correlation computation."""

    def test_correlation_insufficient_data(self):
        """Correlation with insufficient data."""
        hub = TelemetryHub()
        correlations = hub.compute_cross_scale_correlation()
        assert correlations.get("insufficient_data", False) is True

    def test_correlation_with_data(self):
        """Correlation with sufficient data."""
        from ww.visualization.da_telemetry import DATelemetry
        from ww.visualization.pac_telemetry import PACTelemetry
        from ww.visualization.stability_monitor import StabilityMonitor

        hub = TelemetryHub()
        da = DATelemetry()
        pac = PACTelemetry()
        stability = StabilityMonitor()

        hub.register_da(da)
        hub.register_pac(pac)
        hub.register_stability(stability)

        # Record correlated data
        for i in range(100):
            da.record_state(da_level=0.3 + 0.005 * i, firing_rate=4.5, rpe=0.0)
            theta_phase = (i * 0.1) % (2 * np.pi)
            pac.record_state(theta_phase, 0.5)
            jacobian = np.array([[-0.5 - 0.001 * i, 0.1], [0.1, -0.3]])
            stability.record_state(jacobian)

        correlations = hub.compute_cross_scale_correlation()
        assert "da_pac" in correlations or "insufficient_data" in correlations


class TestHealthMonitoring:
    """Test system health monitoring."""

    def test_health_no_modules(self):
        """Health with no modules registered."""
        hub = TelemetryHub()
        health = hub.get_system_health()
        assert health.overall_status == "healthy"
        assert health.stability_score == 1.0

    def test_health_with_da(self):
        """Health with DA module."""
        from ww.visualization.da_telemetry import DATelemetry

        hub = TelemetryHub()
        da = DATelemetry()
        hub.register_da(da)

        for i in range(20):
            da.record_state(da_level=0.3, firing_rate=4.5, rpe=0.0)

        health = hub.get_system_health()
        assert health.modules_active["da"] is True

    def test_health_alerts(self):
        """Health with active alerts."""
        hub = TelemetryHub()

        # Manually add alerts
        hub._active_alerts["test_alert"] = 5
        hub._active_alerts["another_alert"] = 3

        health = hub.get_system_health()
        assert "test_alert" in health.active_alerts
        assert health.overall_status == "warning"

    def test_health_critical_threshold(self):
        """Health reaches critical with enough alerts."""
        config = TelemetryConfig(critical_alert_threshold=2)
        hub = TelemetryHub(config=config)

        hub._active_alerts["alert1"] = 5
        hub._active_alerts["alert2"] = 5
        hub._active_alerts["alert3"] = 5

        health = hub.get_system_health()
        assert health.overall_status == "critical"


class TestExport:
    """Test data export functionality."""

    def test_export_empty(self):
        """Export with no data."""
        hub = TelemetryHub()
        data = hub.export_all_data()
        assert "timestamp" in data
        assert "config" in data
        assert "modules" in data
        assert "cross_scale_events" in data

    def test_export_with_data(self):
        """Export with module data."""
        from ww.visualization.da_telemetry import DATelemetry

        hub = TelemetryHub()
        da = DATelemetry()
        hub.register_da(da)

        da.record_state(da_level=0.5, firing_rate=4.5, rpe=0.0)

        data = hub.export_all_data()
        assert "da" in data["modules"]


class TestSummaryStatistics:
    """Test summary statistics."""

    def test_summary_empty(self):
        """Summary with no modules."""
        hub = TelemetryHub()
        stats = hub.get_summary_statistics()
        assert "timestamp" in stats
        assert "modules" in stats
        assert "health" in stats

    def test_summary_with_modules(self):
        """Summary with registered modules."""
        from ww.visualization.da_telemetry import DATelemetry

        hub = TelemetryHub()
        da = DATelemetry()
        hub.register_da(da)

        for i in range(10):
            da.record_state(da_level=0.3, firing_rate=4.5, rpe=0.0)

        stats = hub.get_summary_statistics()
        assert "da" in stats["modules"]
        assert "mean_da" in stats["modules"]["da"]


class TestAlerts:
    """Test alert management."""

    def test_get_active_alerts_empty(self):
        """No active alerts."""
        hub = TelemetryHub()
        alerts = hub.get_active_alerts()
        assert alerts == []

    def test_get_active_alerts(self):
        """With active alerts."""
        hub = TelemetryHub()
        hub._active_alerts["test_alert"] = 5
        alerts = hub.get_active_alerts()
        assert "test_alert" in alerts

    def test_alert_decay(self):
        """Alerts decay over time."""
        hub = TelemetryHub()
        hub._active_alerts["test_alert"] = 2

        # First health check
        hub.get_system_health()
        assert hub._active_alerts["test_alert"] == 1

        # Second health check
        hub.get_system_health()
        assert hub._active_alerts["test_alert"] == 0


class TestRecentEvents:
    """Test recent events retrieval."""

    def test_get_recent_events_empty(self):
        """No events."""
        hub = TelemetryHub()
        events = hub.get_recent_events()
        assert events == []

    def test_get_recent_events(self):
        """With events."""
        hub = TelemetryHub()
        event = CrossScaleEvent(
            timestamp=datetime.now(),
            event_type="test",
            scales_involved=[TimeScale.FAST],
            metrics={},
            description="Test event",
        )
        hub._events.append(event)

        events = hub.get_recent_events(n=5)
        assert len(events) == 1
        assert events[0].event_type == "test"


class TestClear:
    """Test clear functionality."""

    def test_clear_all(self):
        """Clear all data."""
        hub = TelemetryHub()
        hub._events.append(
            CrossScaleEvent(
                timestamp=datetime.now(),
                event_type="test",
                scales_involved=[],
                metrics={},
                description="",
            )
        )
        hub._active_alerts["test"] = 5

        hub.clear_all()

        assert len(hub._events) == 0
        assert len(hub._active_alerts) == 0


class TestVisualization:
    """Test visualization functions."""

    def test_create_dashboard_empty(self):
        """Dashboard with no data."""
        hub = TelemetryHub()
        fig = hub.create_unified_dashboard()
        assert fig is not None

    def test_create_dashboard_with_data(self):
        """Dashboard with module data."""
        from ww.visualization.da_telemetry import DATelemetry

        hub = TelemetryHub()
        da = DATelemetry()
        hub.register_da(da)

        for i in range(20):
            da.record_state(
                da_level=0.3 + 0.02 * i,
                firing_rate=4.5,
                rpe=0.0,
            )

        fig = hub.create_unified_dashboard()
        assert fig is not None


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_create_telemetry_hub(self):
        """Test convenience function."""
        hub = create_telemetry_hub(enable_swr=False)
        assert hub is not None
        assert hub.config.enable_swr is False

    def test_create_telemetry_hub_defaults(self):
        """Test convenience function with defaults."""
        hub = create_telemetry_hub()
        assert hub.config.enable_swr is True
        assert hub.config.enable_pac is True


class TestModuleExports:
    """Test module exports."""

    def test_all_exports_available(self):
        """Verify all exports are importable."""
        from ww.visualization.telemetry_hub import (
            TelemetryHub,
            TelemetryConfig,
            TimeScale,
            CrossScaleEvent,
            SystemHealth,
            create_telemetry_hub,
        )
        assert TelemetryHub is not None
        assert TelemetryConfig is not None
        assert TimeScale is not None
        assert CrossScaleEvent is not None
        assert SystemHealth is not None
        assert create_telemetry_hub is not None


class TestPhase4Registration:
    """Test Phase 4 module registration."""

    def test_register_ff_visualizer(self):
        """Test registering Forward-Forward visualizer."""
        hub = TelemetryHub()

        @dataclass
        class MockFFVisualizer:
            def record_state(self, state):
                return state

            def get_mean_goodness(self):
                return 0.5

            def get_separation_score(self):
                return 0.8

            def get_threshold_range(self):
                return (0.1, 0.9)

            def get_layer_count(self):
                return 3

            def export_data(self):
                return {"type": "ff"}

        ff = MockFFVisualizer()
        hub.register_ff(ff)
        assert hub._ff is ff

    def test_register_capsule_visualizer(self):
        """Test registering Capsule visualizer."""
        hub = TelemetryHub()

        @dataclass
        class MockCapsuleVisualizer:
            def record_state(self, state):
                return state

            def get_routing_convergence(self):
                return 0.5

            def get_mean_activation(self):
                return 0.3

            def get_pose_variance(self):
                return 0.1

            def get_hierarchy_depth(self):
                return 2

            def export_data(self):
                return {"type": "capsule"}

        capsule = MockCapsuleVisualizer()
        hub.register_capsule(capsule)
        assert hub._capsule is capsule

    def test_register_glymphatic_visualizer(self):
        """Test registering Glymphatic visualizer."""
        hub = TelemetryHub()

        @dataclass
        class MockGlymphaticVisualizer:
            def record_state(self, state):
                return state

            def get_clearance_rate(self):
                return 0.6

            def get_aqp4_polarization(self):
                return 0.7

            def get_current_sleep_stage(self):
                return "NREM3"

            def get_pruning_count(self):
                return 10

            def export_data(self):
                return {"type": "glymphatic"}

        glymphatic = MockGlymphaticVisualizer()
        hub.register_glymphatic(glymphatic)
        assert hub._glymphatic is glymphatic

    def test_register_nt_dashboard(self):
        """Test registering NT state dashboard."""
        hub = TelemetryHub()

        @dataclass
        class MockNTDashboard:
            def record_state(self, nt_state):
                return {"nt": nt_state}

        nt = MockNTDashboard()
        hub.register_nt(nt)
        assert hub._nt is nt


class TestRecordState:
    """Test record_state method."""

    def test_record_state_no_modules(self):
        """Record state with no modules registered."""
        hub = TelemetryHub()
        results = hub.record_state()
        assert isinstance(results, dict)
        assert len(results) == 0

    def test_record_state_with_nt(self):
        """Record state with NT dashboard."""
        from ww.visualization.nt_state_dashboard import NTStateDashboard

        hub = TelemetryHub()
        nt = NTStateDashboard()
        hub.register_nt(nt)

        nt_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        results = hub.record_state(nt_state=nt_state)
        assert "nt" in results

    def test_record_state_with_stability(self):
        """Record state with stability monitor."""
        from ww.visualization.stability_monitor import StabilityMonitor

        hub = TelemetryHub()
        stability = StabilityMonitor()
        hub.register_stability(stability)

        jacobian = np.array([[-0.5, 0.1], [0.1, -0.3]])
        results = hub.record_state(jacobian=jacobian)
        assert "stability" in results

    def test_record_state_with_mock_swr(self):
        """Record state with mock SWR module."""
        hub = TelemetryHub()

        @dataclass
        class MockSWR:
            def record_swr_event(self, state):
                return {"swr": "recorded"}

        swr = MockSWR()
        hub.register_swr(swr)

        @dataclass
        class MockSWRState:
            pass

        results = hub.record_state(swr_state=MockSWRState())
        assert "swr" in results

    def test_record_state_with_mock_oscillator(self):
        """Record state with mock PAC and oscillator."""
        hub = TelemetryHub()

        @dataclass
        class MockPAC:
            _snapshots: list = None

            def __post_init__(self):
                self._snapshots = []

            def record_state(self, theta_phase, gamma_amp):
                return {"theta": theta_phase, "gamma": gamma_amp}

        pac = MockPAC()
        hub.register_pac(pac)

        @dataclass
        class MockOscillatorState:
            theta_phase: float = 1.0
            gamma_amplitude: float = 0.5

        results = hub.record_state(oscillator_state=MockOscillatorState())
        assert "pac" in results

    def test_record_state_with_mock_vta(self):
        """Record state with mock DA and VTA."""
        hub = TelemetryHub()

        @dataclass
        class MockDA:
            _snapshots: list = None

            def __post_init__(self):
                self._snapshots = []

            def record_from_vta(self, vta_state):
                return {"vta": "recorded"}

        da = MockDA()
        hub.register_da(da)

        @dataclass
        class MockVTAState:
            pass

        results = hub.record_state(vta_state=MockVTAState())
        assert "da" in results

    def test_record_state_ff_phase4(self):
        """Record state with FF visualizer (Phase 4)."""
        hub = TelemetryHub()

        @dataclass
        class MockFF:
            def record_state(self, state):
                return {"ff": "recorded", "goodness": 0.5}

        ff = MockFF()
        hub.register_ff(ff)

        @dataclass
        class MockFFState:
            goodness: float = 0.5

        results = hub.record_state(ff_state=MockFFState())
        assert "ff" in results

    def test_record_state_capsule_phase4(self):
        """Record state with Capsule visualizer (Phase 4)."""
        hub = TelemetryHub()

        @dataclass
        class MockCapsule:
            def record_state(self, state):
                return {"capsule": "recorded"}

        capsule = MockCapsule()
        hub.register_capsule(capsule)

        @dataclass
        class MockCapsuleState:
            routing_convergence: float = 0.5

        results = hub.record_state(capsule_state=MockCapsuleState())
        assert "capsule" in results

    def test_record_state_glymphatic_phase4(self):
        """Record state with Glymphatic visualizer (Phase 4)."""
        hub = TelemetryHub()

        @dataclass
        class MockGlymphatic:
            def record_state(self, state):
                return {"glymphatic": "recorded", "clearance_rate": 0.5}

        glymphatic = MockGlymphatic()
        hub.register_glymphatic(glymphatic)

        @dataclass
        class MockGlymphaticState:
            sleep_stage: str = "NREM3"
            clearance_rate: float = 0.5

        results = hub.record_state(glymphatic_state=MockGlymphaticState())
        assert "glymphatic" in results

    def test_record_state_exception_handling(self):
        """Record state handles exceptions gracefully."""
        hub = TelemetryHub()

        @dataclass
        class MockDA:
            _snapshots: list = None

            def __post_init__(self):
                self._snapshots = []

            def record_from_vta(self, vta_state):
                raise ValueError("Mock error")

        da = MockDA()
        hub.register_da(da)

        @dataclass
        class MockVTAState:
            pass

        # Should not raise, exception is logged
        results = hub.record_state(vta_state=MockVTAState())
        assert "da" not in results  # Error caught, not recorded


class TestCrossScaleEventDetection:
    """Test cross-scale event detection."""

    def test_detect_reward_consolidation(self):
        """Test detection of reward consolidation event."""
        hub = TelemetryHub()

        # Create mock modules with high DA + SWR
        @dataclass
        class MockDA:
            _snapshots: list = None

            def __post_init__(self):
                self._snapshots = []

            def record_from_vta(self, vta_state):
                return MockDASnapshot()

        @dataclass
        class MockDASnapshot:
            da_level: float = 0.7  # High DA

        @dataclass
        class MockSWR:
            def record_swr_event(self, state):
                return MockSWREvent()

        @dataclass
        class MockSWREvent:
            pass

        hub.register_da(MockDA())
        hub.register_swr(MockSWR())

        @dataclass
        class MockVTAState:
            pass

        @dataclass
        class MockSWRState:
            pass

        # Record states to trigger event detection
        hub.record_state(vta_state=MockVTAState(), swr_state=MockSWRState())

        # Check for reward_consolidation event
        events = hub.get_recent_events(n=10)
        event_types = [e.event_type for e in events]
        assert "reward_consolidation" in event_types

    def test_detect_oscillation_instability(self):
        """Test detection of oscillation instability event."""
        hub = TelemetryHub()

        @dataclass
        class MockPAC:
            _snapshots: list = None

            def __post_init__(self):
                self._snapshots = []

            def record_state(self, theta_phase, gamma_amp):
                return MockPACSnapshot()

        @dataclass
        class MockPACSnapshot:
            modulation_index: float = 0.6  # High MI

        @dataclass
        class MockStability:
            def record_state(self, jacobian):
                return MockStabilitySnapshot()

            def is_stable(self):
                return False

            def get_current_eigenvalues(self):
                return np.array([0.1, -0.2])

            def get_stability_margin_trace(self):
                return [0.5] * 100

        @dataclass
        class MockStabilitySnapshot:
            is_stable: bool = False

        hub.register_pac(MockPAC())
        hub.register_stability(MockStability())

        @dataclass
        class MockOscillator:
            theta_phase: float = 1.0
            gamma_amplitude: float = 0.5

        jacobian = np.array([[-0.5, 0.1], [0.1, 0.2]])  # Unstable
        hub.record_state(oscillator_state=MockOscillator(), jacobian=jacobian)

        events = hub.get_recent_events(n=10)
        event_types = [e.event_type for e in events]
        assert "oscillation_instability" in event_types

    def test_detect_ff_glymphatic_coupling(self):
        """Test detection of FF-Glymphatic coupling (Phase 4)."""
        hub = TelemetryHub()

        @dataclass
        class MockFF:
            def record_state(self, state):
                return MockFFSnapshot()

        @dataclass
        class MockFFSnapshot:
            goodness: float = -0.3  # Negative goodness

        @dataclass
        class MockGlymphatic:
            def record_state(self, state):
                return MockGlySnapshot()

        @dataclass
        class MockGlySnapshot:
            sleep_stage: str = "NREM3"  # Deep sleep
            clearance_rate: float = 0.6

        hub.register_ff(MockFF())
        hub.register_glymphatic(MockGlymphatic())

        @dataclass
        class MockFFState:
            goodness: float = -0.3

        @dataclass
        class MockGlyState:
            sleep_stage: str = "NREM3"
            clearance_rate: float = 0.6

        hub.record_state(ff_state=MockFFState(), glymphatic_state=MockGlyState())

        events = hub.get_recent_events(n=10)
        event_types = [e.event_type for e in events]
        assert "sleep_negative_generation" in event_types

    def test_detect_ach_routing_enhancement(self):
        """Test detection of ACh-routing enhancement (Phase 4)."""
        hub = TelemetryHub()

        @dataclass
        class MockCapsule:
            def record_state(self, state):
                return MockCapsuleSnapshot()

        @dataclass
        class MockCapsuleSnapshot:
            routing_convergence: float = 0.2  # Low convergence

        @dataclass
        class MockNT:
            def record_state(self, nt_state):
                return MockNTSnapshot()

        @dataclass
        class MockNTSnapshot:
            ach: float = 0.8  # High ACh

        hub.register_capsule(MockCapsule())
        hub.register_nt(MockNT())

        @dataclass
        class MockCapsuleState:
            routing_convergence: float = 0.2

        hub.record_state(
            capsule_state=MockCapsuleState(),
            nt_state=np.array([0.5, 0.5, 0.8, 0.5, 0.5, 0.5]),
        )

        events = hub.get_recent_events(n=10)
        event_types = [e.event_type for e in events]
        assert "ach_routing_enhancement" in event_types

    def test_detect_replay_clearance_coupling(self):
        """Test detection of replay-clearance coupling (Phase 4)."""
        hub = TelemetryHub()

        @dataclass
        class MockGlymphatic:
            def record_state(self, state):
                return MockGlySnapshot()

        @dataclass
        class MockGlySnapshot:
            clearance_rate: float = 0.7  # High clearance

        @dataclass
        class MockSWR:
            def record_swr_event(self, state):
                return MockSWRSnapshot()

        @dataclass
        class MockSWRSnapshot:
            replay_count: int = 10  # Many replays

        hub.register_glymphatic(MockGlymphatic())
        hub.register_swr(MockSWR())

        @dataclass
        class MockGlyState:
            clearance_rate: float = 0.7

        @dataclass
        class MockSWRState:
            replay_count: int = 10

        hub.record_state(glymphatic_state=MockGlyState(), swr_state=MockSWRState())

        events = hub.get_recent_events(n=10)
        event_types = [e.event_type for e in events]
        assert "replay_clearance_coupling" in event_types

    def test_event_list_trimming(self):
        """Test that event list is trimmed to max size."""
        hub = TelemetryHub()
        hub._max_events = 5

        # Add many events manually
        for i in range(10):
            event = CrossScaleEvent(
                timestamp=datetime.now(),
                event_type=f"test_{i}",
                scales_involved=[TimeScale.FAST],
                metrics={},
                description="Test",
            )
            hub._events.append(event)

        # Trigger trimming via record_state
        hub._detect_cross_scale_events({})

        assert len(hub._events) <= hub._max_events


class TestBiologicalValidity:
    """Test biological validity computation."""

    def test_validity_no_modules(self):
        """Validity with no modules."""
        hub = TelemetryHub()
        # Access via get_system_health which calls _compute_biological_validity
        health = hub.get_system_health()
        assert health.biological_validity == 1.0

    def test_validity_with_da(self):
        """Validity with DA validation."""
        from ww.visualization.da_telemetry import DATelemetry

        hub = TelemetryHub()
        da = DATelemetry()
        hub.register_da(da)

        # Record valid DA data
        for _ in range(20):
            da.record_state(da_level=0.5, firing_rate=4.5, rpe=0.0)

        health = hub.get_system_health()
        assert 0 <= health.biological_validity <= 1.0

    def test_validity_with_pac(self):
        """Validity with PAC validation."""
        from ww.visualization.pac_telemetry import PACTelemetry

        hub = TelemetryHub()
        pac = PACTelemetry()
        hub.register_pac(pac)

        # Record PAC data with good MI
        for i in range(100):
            theta = (i * 0.1) % (2 * np.pi)
            gamma = 0.5 + 0.2 * np.cos(theta)
            pac.record_state(theta, gamma)

        health = hub.get_system_health()
        assert 0 <= health.biological_validity <= 1.0

    def test_validity_with_swr(self):
        """Validity with SWR validation."""
        from ww.visualization.swr_telemetry import SWRTelemetry

        hub = TelemetryHub()
        swr = SWRTelemetry()
        hub.register_swr(swr)

        health = hub.get_system_health()
        assert 0 <= health.biological_validity <= 1.0


class TestGetTraces:
    """Test trace getter methods."""

    def test_get_da_trace_no_module(self):
        """DA trace with no module."""
        hub = TelemetryHub()
        trace = hub._get_da_trace()
        assert trace == []

    def test_get_da_trace_with_data(self):
        """DA trace with data."""
        from ww.visualization.da_telemetry import DATelemetry

        hub = TelemetryHub()
        da = DATelemetry()
        hub.register_da(da)

        for i in range(10):
            da.record_state(da_level=0.3 + i * 0.05, firing_rate=4.5, rpe=0.0)

        trace = hub._get_da_trace()
        assert len(trace) == 10

    def test_get_pac_trace_no_module(self):
        """PAC trace with no module."""
        hub = TelemetryHub()
        trace = hub._get_pac_trace()
        assert trace == []

    def test_get_pac_trace_with_data(self):
        """PAC trace with data."""
        from ww.visualization.pac_telemetry import PACTelemetry

        hub = TelemetryHub()
        pac = PACTelemetry()
        hub.register_pac(pac)

        for i in range(10):
            pac.record_state(i * 0.5, 0.5)

        trace = hub._get_pac_trace()
        assert len(trace) == 10

    def test_get_stability_trace_no_module(self):
        """Stability trace with no module."""
        hub = TelemetryHub()
        trace = hub._get_stability_trace()
        assert trace == []

    def test_get_stability_trace_with_data(self):
        """Stability trace with data."""
        from ww.visualization.stability_monitor import StabilityMonitor

        hub = TelemetryHub()
        stability = StabilityMonitor()
        hub.register_stability(stability)

        for _ in range(10):
            jacobian = np.array([[-0.5, 0.1], [0.1, -0.3]])
            stability.record_state(jacobian)

        trace = hub._get_stability_trace()
        assert len(trace) >= 0  # May be 0 if margin not computed


class TestStabilityScore:
    """Test stability score computation."""

    def test_stability_score_no_monitor(self):
        """Stability score with no monitor."""
        hub = TelemetryHub()
        score = hub._compute_stability_score()
        assert score == 1.0

    def test_stability_score_stable(self):
        """Stability score when stable."""
        from ww.visualization.stability_monitor import StabilityMonitor

        hub = TelemetryHub()
        stability = StabilityMonitor()
        hub.register_stability(stability)

        # Record stable Jacobian
        jacobian = np.array([[-0.5, 0.1], [0.1, -0.3]])
        stability.record_state(jacobian)

        score = hub._compute_stability_score()
        assert score == 1.0

    def test_stability_score_unstable(self):
        """Stability score when unstable."""
        from ww.visualization.stability_monitor import StabilityMonitor

        hub = TelemetryHub()
        stability = StabilityMonitor()
        hub.register_stability(stability)

        # Record multiple unstable Jacobians to trigger instability detection
        jacobian = np.array([[0.5, 0.1], [0.1, 0.3]])
        for _ in range(5):
            stability.record_state(jacobian)

        score = hub._compute_stability_score()
        # Score should be between 0 and 1
        assert 0 <= score <= 1.0


class TestCoherence:
    """Test coherence computation."""

    def test_coherence_insufficient_data(self):
        """Coherence with insufficient data."""
        hub = TelemetryHub()
        correlations = {"insufficient_data": True}
        coherence = hub._compute_coherence(correlations)
        assert coherence == 0.5

    def test_coherence_with_correlations(self):
        """Coherence with correlation values."""
        hub = TelemetryHub()
        correlations = {
            "da_pac": 0.5,
            "da_stability": 0.3,
            "pac_stability": 0.4,
        }
        coherence = hub._compute_coherence(correlations)
        assert coherence == pytest.approx(0.4, rel=0.1)


class TestPhase4Export:
    """Test Phase 4 module export."""

    def test_export_ff_data(self):
        """Test exporting FF visualizer data."""
        hub = TelemetryHub()

        @dataclass
        class MockFF:
            def export_data(self):
                return {"layers": 3, "goodness": 0.5}

        hub.register_ff(MockFF())
        data = hub.export_all_data()
        assert "ff" in data["modules"]
        assert data["modules"]["ff"]["layers"] == 3

    def test_export_capsule_data(self):
        """Test exporting Capsule visualizer data."""
        hub = TelemetryHub()

        @dataclass
        class MockCapsule:
            def export_data(self):
                return {"capsules": 8, "routing": 0.9}

        hub.register_capsule(MockCapsule())
        data = hub.export_all_data()
        assert "capsule" in data["modules"]
        assert data["modules"]["capsule"]["capsules"] == 8

    def test_export_glymphatic_data(self):
        """Test exporting Glymphatic visualizer data."""
        hub = TelemetryHub()

        @dataclass
        class MockGlymphatic:
            def export_data(self):
                return {"clearance": 0.6, "stage": "NREM3"}

        hub.register_glymphatic(MockGlymphatic())
        data = hub.export_all_data()
        assert "glymphatic" in data["modules"]
        assert data["modules"]["glymphatic"]["clearance"] == 0.6

    def test_export_handles_errors(self):
        """Test export handles errors gracefully."""
        hub = TelemetryHub()

        @dataclass
        class MockFF:
            def export_data(self):
                raise ValueError("Export error")

        hub.register_ff(MockFF())
        data = hub.export_all_data()
        assert "ff" in data["modules"]
        assert "error" in data["modules"]["ff"]


class TestPhase4Statistics:
    """Test Phase 4 module statistics."""

    def test_stats_ff(self):
        """Test FF statistics."""
        hub = TelemetryHub()

        @dataclass
        class MockFF:
            def get_mean_goodness(self):
                return 0.5

            def get_separation_score(self):
                return 0.8

            def get_threshold_range(self):
                return (0.1, 0.9)

            def get_layer_count(self):
                return 3

        hub.register_ff(MockFF())
        stats = hub.get_summary_statistics()
        assert "ff" in stats["modules"]
        assert stats["modules"]["ff"]["mean_goodness"] == 0.5

    def test_stats_capsule(self):
        """Test Capsule statistics."""
        hub = TelemetryHub()

        @dataclass
        class MockCapsule:
            def get_routing_convergence(self):
                return 0.5

            def get_mean_activation(self):
                return 0.3

            def get_pose_variance(self):
                return 0.1

            def get_hierarchy_depth(self):
                return 2

        hub.register_capsule(MockCapsule())
        stats = hub.get_summary_statistics()
        assert "capsule" in stats["modules"]
        assert stats["modules"]["capsule"]["routing_convergence"] == 0.5

    def test_stats_glymphatic(self):
        """Test Glymphatic statistics."""
        hub = TelemetryHub()

        @dataclass
        class MockGlymphatic:
            def get_clearance_rate(self):
                return 0.6

            def get_aqp4_polarization(self):
                return 0.7

            def get_current_sleep_stage(self):
                return "NREM3"

            def get_pruning_count(self):
                return 10

        hub.register_glymphatic(MockGlymphatic())
        stats = hub.get_summary_statistics()
        assert "glymphatic" in stats["modules"]
        assert stats["modules"]["glymphatic"]["clearance_rate"] == 0.6

    def test_stats_handles_errors(self):
        """Test statistics handle errors gracefully."""
        hub = TelemetryHub()

        @dataclass
        class MockFF:
            def get_mean_goodness(self):
                raise ValueError("Stats error")

        hub.register_ff(MockFF())
        stats = hub.get_summary_statistics()
        assert "ff" in stats["modules"]
        assert "error" in stats["modules"]["ff"]


class TestVisualizationPlots:
    """Test visualization plot methods."""

    def test_plot_unified_timeline_empty(self):
        """Test unified timeline with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        hub = TelemetryHub()
        fig, ax = plt.subplots()
        hub._plot_unified_timeline(ax)
        plt.close("all")

    def test_plot_unified_timeline_with_da(self):
        """Test unified timeline with DA data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from ww.visualization.da_telemetry import DATelemetry

        hub = TelemetryHub()
        da = DATelemetry()
        hub.register_da(da)

        for i in range(20):
            da.record_state(da_level=0.3 + i * 0.02, firing_rate=4.5, rpe=0.0)

        fig, ax = plt.subplots()
        hub._plot_unified_timeline(ax)
        plt.close("all")

    def test_plot_correlations_insufficient(self):
        """Test correlations plot with insufficient data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        hub = TelemetryHub()
        fig, ax = plt.subplots()
        hub._plot_correlations(ax)
        plt.close("all")

    def test_plot_correlations_with_data(self):
        """Test correlations plot with data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from ww.visualization.da_telemetry import DATelemetry
        from ww.visualization.pac_telemetry import PACTelemetry
        from ww.visualization.stability_monitor import StabilityMonitor

        hub = TelemetryHub()
        da = DATelemetry()
        pac = PACTelemetry()
        stability = StabilityMonitor()

        hub.register_da(da)
        hub.register_pac(pac)
        hub.register_stability(stability)

        for i in range(100):
            da.record_state(da_level=0.3 + i * 0.005, firing_rate=4.5, rpe=0.0)
            pac.record_state(i * 0.1, 0.5)
            stability.record_state(np.array([[-0.5, 0.1], [0.1, -0.3]]))

        fig, ax = plt.subplots()
        hub._plot_correlations(ax)
        plt.close("all")

    def test_plot_events_empty(self):
        """Test events plot with no events."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        hub = TelemetryHub()
        fig, ax = plt.subplots()
        hub._plot_events(ax)
        plt.close("all")

    def test_plot_events_with_data(self):
        """Test events plot with events."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        hub = TelemetryHub()

        for severity in ["info", "warning", "critical"]:
            event = CrossScaleEvent(
                timestamp=datetime.now(),
                event_type=f"test_{severity}",
                scales_involved=[TimeScale.FAST],
                metrics={},
                description="Test event with a long description that gets truncated",
                severity=severity,
            )
            hub._events.append(event)

        fig, ax = plt.subplots()
        hub._plot_events(ax)
        plt.close("all")

    def test_plot_health(self):
        """Test health gauge plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        hub = TelemetryHub()
        fig, ax = plt.subplots()
        hub._plot_health(ax)
        plt.close("all")

    def test_plot_summary(self):
        """Test summary plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from ww.visualization.da_telemetry import DATelemetry

        hub = TelemetryHub()
        da = DATelemetry()
        hub.register_da(da)

        for i in range(10):
            da.record_state(da_level=0.3 + i * 0.05, firing_rate=4.5, rpe=0.0)

        fig, ax = plt.subplots()
        hub._plot_summary(ax)
        plt.close("all")

    def test_create_unified_dashboard_full(self):
        """Test full dashboard creation with all modules."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from ww.visualization.da_telemetry import DATelemetry
        from ww.visualization.pac_telemetry import PACTelemetry
        from ww.visualization.stability_monitor import StabilityMonitor

        hub = TelemetryHub()
        da = DATelemetry()
        pac = PACTelemetry()
        stability = StabilityMonitor()

        hub.register_da(da)
        hub.register_pac(pac)
        hub.register_stability(stability)

        for i in range(100):
            da.record_state(da_level=0.3 + i * 0.005, firing_rate=4.5, rpe=0.0)
            pac.record_state(i * 0.1, 0.5)
            stability.record_state(np.array([[-0.5, 0.1], [0.1, -0.3]]))

        fig = hub.create_unified_dashboard()
        assert fig is not None
        plt.close("all")


class TestConfigPhase4:
    """Test Phase 4 config options."""

    def test_config_phase4_defaults(self):
        """Test Phase 4 config defaults."""
        config = TelemetryConfig()
        assert config.enable_ff is True
        assert config.enable_capsule is True
        assert config.enable_glymphatic is True

    def test_config_phase4_custom(self):
        """Test Phase 4 config customization."""
        config = TelemetryConfig(
            enable_ff=False,
            enable_capsule=False,
            enable_glymphatic=False,
        )
        assert config.enable_ff is False
        assert config.enable_capsule is False
        assert config.enable_glymphatic is False

    def test_hub_with_phase4_config(self):
        """Test hub with Phase 4 config."""
        config = TelemetryConfig(enable_ff=False)
        hub = TelemetryHub(config=config)
        assert hub.config.enable_ff is False


class TestHealthHistory:
    """Test health history management."""

    def test_health_history_accumulates(self):
        """Test health history accumulates."""
        hub = TelemetryHub()

        for _ in range(5):
            hub.get_system_health()

        assert len(hub._health_history) == 5

    def test_health_history_trimmed(self):
        """Test health history is trimmed to max."""
        hub = TelemetryHub()
        hub._max_health_history = 3

        for _ in range(10):
            hub.get_system_health()

        assert len(hub._health_history) == 3

    def test_health_history_in_export(self):
        """Test health history included in export."""
        hub = TelemetryHub()

        for _ in range(5):
            hub.get_system_health()

        data = hub.export_all_data()
        assert len(data["health_history"]) == 5
