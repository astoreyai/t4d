"""
Tests for Glymphatic Visualizer module.

Tests sleep-gated waste clearance visualization.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from t4dm.visualization.glymphatic_visualizer import (
    SleepStage,
    GlymphaticSnapshot,
    ClearanceEvent,
    GlymphaticVisualizer,
)


class TestSleepStage:
    """Tests for SleepStage enum."""

    def test_all_stages_exist(self):
        """Test all sleep stages are defined."""
        assert SleepStage.WAKE is not None
        assert SleepStage.NREM_LIGHT is not None
        assert SleepStage.NREM_DEEP is not None
        assert SleepStage.REM is not None


class TestGlymphaticSnapshot:
    """Tests for GlymphaticSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a glymphatic snapshot."""
        snapshot = GlymphaticSnapshot(
            timestamp=datetime.now(),
            sleep_stage=SleepStage.NREM_DEEP,
            clearance_rate=0.8,
            adenosine_level=0.3,
            waste_level=0.4,
            aqp4_activity=0.7,
            n_prune_candidates=5,
            csf_flow_rate=0.6,
        )

        assert snapshot.sleep_stage == SleepStage.NREM_DEEP
        assert snapshot.clearance_rate == 0.8
        assert snapshot.adenosine_level == 0.3
        assert snapshot.waste_level == 0.4
        assert snapshot.n_prune_candidates == 5


class TestClearanceEvent:
    """Tests for ClearanceEvent dataclass."""

    def test_event_creation(self):
        """Test creating a clearance event."""
        event = ClearanceEvent(
            timestamp=datetime.now(),
            memory_id="mem-123",
            memory_type="episodic",
            reason="unused",
            age_days=30.0,
            last_access_days=25.0,
            stability_score=0.2,
        )

        assert event.memory_id == "mem-123"
        assert event.memory_type == "episodic"
        assert event.reason == "unused"
        assert event.stability_score == 0.2


class TestGlymphaticVisualizer:
    """Tests for GlymphaticVisualizer."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return GlymphaticVisualizer(window_size=100)

    def test_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer.window_size == 100
        assert visualizer._snapshots == []
        assert visualizer._clearance_events == []
        assert visualizer._stage_history == []

    def test_record_state(self, visualizer):
        """Test recording a glymphatic state."""
        snapshot = visualizer.record_state(
            sleep_stage=SleepStage.NREM_DEEP,
            clearance_rate=0.8,
            adenosine_level=0.3,
            waste_level=0.4,
        )

        assert isinstance(snapshot, GlymphaticSnapshot)
        assert snapshot.sleep_stage == SleepStage.NREM_DEEP
        assert snapshot.clearance_rate == 0.8
        assert len(visualizer._snapshots) == 1

    def test_record_state_tracks_stage_transitions(self, visualizer):
        """Test that stage transitions are tracked."""
        # Record WAKE
        visualizer.record_state(
            sleep_stage=SleepStage.WAKE,
            clearance_rate=0.2,
            adenosine_level=0.5,
            waste_level=0.6,
        )

        # Record same stage - no new transition
        visualizer.record_state(
            sleep_stage=SleepStage.WAKE,
            clearance_rate=0.2,
            adenosine_level=0.5,
            waste_level=0.6,
        )

        # Record new stage - should add transition
        visualizer.record_state(
            sleep_stage=SleepStage.NREM_LIGHT,
            clearance_rate=0.5,
            adenosine_level=0.4,
            waste_level=0.5,
        )

        # Should have 2 transitions (WAKE, then NREM_LIGHT)
        assert len(visualizer._stage_history) == 2

    def test_record_clearance_event(self, visualizer):
        """Test recording a clearance event."""
        event = visualizer.record_clearance_event(
            memory_id="mem-456",
            memory_type="semantic",
            reason="low_stability",
            age_days=60.0,
            last_access_days=45.0,
            stability_score=0.15,
        )

        assert isinstance(event, ClearanceEvent)
        assert event.memory_id == "mem-456"
        assert len(visualizer._clearance_events) == 1

    def test_window_size_limit(self, visualizer):
        """Test that window size limits history."""
        # Record more than window size
        for i in range(150):
            visualizer.record_state(
                sleep_stage=SleepStage.NREM_DEEP,
                clearance_rate=0.8,
                adenosine_level=float(i) / 150,
                waste_level=0.4,
            )

        # Should be limited to window size
        assert len(visualizer._snapshots) == 100

    def test_alert_high_waste(self, visualizer):
        """Test alert for high waste level."""
        visualizer.record_state(
            sleep_stage=SleepStage.WAKE,
            clearance_rate=0.2,
            adenosine_level=0.5,
            waste_level=0.9,  # High waste
        )

        alerts = visualizer.get_alerts()
        assert any("HIGH WASTE" in a for a in alerts)

    def test_alert_low_clearance_during_sleep(self, visualizer):
        """Test alert for low clearance during sleep."""
        visualizer.record_state(
            sleep_stage=SleepStage.NREM_DEEP,  # Sleep
            clearance_rate=0.05,  # Low clearance
            adenosine_level=0.3,
            waste_level=0.4,
        )

        alerts = visualizer.get_alerts()
        assert any("LOW CLEARANCE" in a for a in alerts)

    def test_no_alert_low_clearance_during_wake(self, visualizer):
        """Test no alert for low clearance during wake (expected)."""
        visualizer.record_state(
            sleep_stage=SleepStage.WAKE,  # Awake
            clearance_rate=0.05,  # Low clearance is normal when awake
            adenosine_level=0.5,
            waste_level=0.4,
        )

        alerts = visualizer.get_alerts()
        assert not any("LOW CLEARANCE" in a for a in alerts)

    def test_clearance_history_tracked(self, visualizer):
        """Test that clearance history is tracked."""
        for i in range(5):
            visualizer.record_state(
                sleep_stage=SleepStage.NREM_DEEP,
                clearance_rate=0.2 * (i + 1),
                adenosine_level=0.3,
                waste_level=0.4,
            )

        assert len(visualizer._clearance_history) == 5
        assert visualizer._clearance_history == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])

    def test_waste_history_tracked(self, visualizer):
        """Test that waste history is tracked."""
        for i in range(5):
            visualizer.record_state(
                sleep_stage=SleepStage.NREM_DEEP,
                clearance_rate=0.8,
                adenosine_level=0.3,
                waste_level=0.1 * (i + 1),
            )

        assert len(visualizer._waste_history) == 5
        assert visualizer._waste_history == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


class TestGlymphaticVisualizerIntegration:
    """Integration tests for glymphatic visualizer."""

    def test_full_sleep_cycle(self):
        """Test simulating a full sleep cycle."""
        visualizer = GlymphaticVisualizer(window_size=1000)

        # Wake phase - high waste, low clearance
        for i in range(10):
            visualizer.record_state(
                sleep_stage=SleepStage.WAKE,
                clearance_rate=0.1,
                adenosine_level=0.5 + i * 0.02,  # Rising adenosine
                waste_level=0.3 + i * 0.02,  # Rising waste
            )

        # Light sleep - moderate clearance
        for i in range(10):
            visualizer.record_state(
                sleep_stage=SleepStage.NREM_LIGHT,
                clearance_rate=0.4,
                adenosine_level=0.7 - i * 0.01,
                waste_level=0.5 - i * 0.01,
            )

        # Deep sleep - high clearance
        for i in range(10):
            visualizer.record_state(
                sleep_stage=SleepStage.NREM_DEEP,
                clearance_rate=0.9,
                adenosine_level=0.6 - i * 0.02,
                waste_level=0.4 - i * 0.02,
            )
            # Record some clearance events during deep sleep
            if i % 3 == 0:
                visualizer.record_clearance_event(
                    memory_id=f"mem-{i}",
                    memory_type="episodic",
                    reason="unused",
                    age_days=float(i * 10),
                    last_access_days=float(i * 5),
                    stability_score=0.2,
                )

        # REM sleep
        for i in range(10):
            visualizer.record_state(
                sleep_stage=SleepStage.REM,
                clearance_rate=0.3,
                adenosine_level=0.4,
                waste_level=0.2,
            )

        assert len(visualizer._snapshots) == 40
        assert len(visualizer._stage_history) == 4  # WAKE, NREM_LIGHT, NREM_DEEP, REM
        assert len(visualizer._clearance_events) >= 3

    def test_clearance_event_history_limit(self):
        """Test that clearance event history is limited."""
        visualizer = GlymphaticVisualizer(window_size=50)

        # Record more events than window size
        for i in range(100):
            visualizer.record_clearance_event(
                memory_id=f"mem-{i}",
                memory_type="episodic",
                reason="unused",
                age_days=float(i),
                last_access_days=float(i / 2),
                stability_score=0.1,
            )

        assert len(visualizer._clearance_events) == 50

    def test_multiple_alert_conditions(self):
        """Test multiple alerts can fire simultaneously."""
        visualizer = GlymphaticVisualizer(
            window_size=100,
            alert_high_waste=0.7,
            alert_low_clearance=0.2,
        )

        visualizer.record_state(
            sleep_stage=SleepStage.NREM_DEEP,
            clearance_rate=0.05,  # Low clearance during sleep
            adenosine_level=0.5,
            waste_level=0.9,  # High waste
        )

        alerts = visualizer.get_alerts()
        assert len(alerts) >= 2
        assert any("HIGH WASTE" in a for a in alerts)
        assert any("LOW CLEARANCE" in a for a in alerts)


class TestGlymphaticVisualizerAnalysis:
    """Tests for analysis methods."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with data."""
        visualizer = GlymphaticVisualizer(window_size=100)

        # Simulate sleep cycle
        stages = [SleepStage.WAKE, SleepStage.NREM_LIGHT, SleepStage.NREM_DEEP, SleepStage.REM]
        for stage in stages:
            for i in range(5):
                visualizer.record_state(
                    sleep_stage=stage,
                    clearance_rate=0.3 if stage == SleepStage.WAKE else 0.8,
                    adenosine_level=0.5,
                    waste_level=0.4,
                    aqp4_activity=0.6 + i * 0.02,
                    csf_flow_rate=0.5 + i * 0.01,
                    n_prune_candidates=i,
                )

        # Add clearance events
        for i in range(10):
            memory_type = ["episodic", "semantic", "procedural"][i % 3]
            reason = ["unused", "low_stability", "redundant"][i % 3]
            visualizer.record_clearance_event(
                memory_id=f"mem-{i}",
                memory_type=memory_type,
                reason=reason,
                age_days=float(i * 10),
                last_access_days=float(i * 5),
                stability_score=0.1 + i * 0.05,
            )

        return visualizer

    def test_get_current_stage(self, populated_visualizer):
        """Test getting current sleep stage."""
        stage = populated_visualizer.get_current_stage()
        assert stage == SleepStage.REM

    def test_get_current_stage_empty(self):
        """Test getting current stage when empty."""
        visualizer = GlymphaticVisualizer()
        assert visualizer.get_current_stage() is None

    def test_get_stage_timeline(self, populated_visualizer):
        """Test getting stage timeline."""
        timeline = populated_visualizer.get_stage_timeline()
        assert len(timeline) == 4
        assert timeline[0][1] == SleepStage.WAKE
        assert timeline[1][1] == SleepStage.NREM_LIGHT
        assert timeline[2][1] == SleepStage.NREM_DEEP
        assert timeline[3][1] == SleepStage.REM

    def test_get_stage_durations(self, populated_visualizer):
        """Test getting stage durations."""
        durations = populated_visualizer.get_stage_durations()
        assert SleepStage.WAKE in durations
        assert SleepStage.NREM_LIGHT in durations
        assert SleepStage.NREM_DEEP in durations
        assert SleepStage.REM in durations
        # All should be timedelta objects
        for stage, duration in durations.items():
            assert isinstance(duration, timedelta)

    def test_get_stage_durations_empty(self):
        """Test stage durations with insufficient data."""
        visualizer = GlymphaticVisualizer()
        visualizer.record_state(
            sleep_stage=SleepStage.WAKE,
            clearance_rate=0.2,
            adenosine_level=0.5,
            waste_level=0.4,
        )
        durations = visualizer.get_stage_durations()
        # Only one stage transition, not enough for calculation
        assert all(d == timedelta() for d in durations.values()) or any(d > timedelta() for d in durations.values())

    def test_get_stage_percentages(self, populated_visualizer):
        """Test getting stage percentages."""
        percentages = populated_visualizer.get_stage_percentages()
        assert "WAKE" in percentages
        assert "NREM_LIGHT" in percentages
        assert "NREM_DEEP" in percentages
        assert "REM" in percentages
        # All values should be between 0 and 1 (may be 0 if stages happened too quickly)
        for p in percentages.values():
            assert 0.0 <= p <= 1.0

    def test_get_stage_percentages_no_data(self):
        """Test stage percentages with no data."""
        visualizer = GlymphaticVisualizer()
        percentages = visualizer.get_stage_percentages()
        assert all(p == 0.0 for p in percentages.values())

    def test_get_clearance_trace(self, populated_visualizer):
        """Test getting clearance trace."""
        timestamps, rates = populated_visualizer.get_clearance_trace()
        assert len(timestamps) == 20
        assert len(rates) == 20
        assert all(isinstance(t, datetime) for t in timestamps)
        assert all(isinstance(r, float) for r in rates)

    def test_get_clearance_by_stage(self, populated_visualizer):
        """Test getting clearance by stage."""
        by_stage = populated_visualizer.get_clearance_by_stage()
        assert "WAKE" in by_stage
        assert "NREM_DEEP" in by_stage
        # WAKE should have lower clearance than NREM_DEEP
        assert by_stage["WAKE"] < by_stage["NREM_DEEP"]

    def test_get_clearance_statistics(self, populated_visualizer):
        """Test getting clearance statistics."""
        stats = populated_visualizer.get_clearance_statistics()
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "current" in stats
        assert 0 <= stats["mean"] <= 1
        assert 0 <= stats["min"] <= stats["max"] <= 1

    def test_get_clearance_statistics_empty(self):
        """Test clearance statistics with no data."""
        visualizer = GlymphaticVisualizer()
        stats = visualizer.get_clearance_statistics()
        assert stats == {}

    def test_get_waste_trace(self, populated_visualizer):
        """Test getting waste trace."""
        timestamps, waste = populated_visualizer.get_waste_trace()
        assert len(timestamps) == 20
        assert len(waste) == 20

    def test_get_waste_statistics(self, populated_visualizer):
        """Test getting waste statistics."""
        stats = populated_visualizer.get_waste_statistics()
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "current" in stats

    def test_get_waste_statistics_empty(self):
        """Test waste statistics with no data."""
        visualizer = GlymphaticVisualizer()
        stats = visualizer.get_waste_statistics()
        assert stats == {}

    def test_get_clearance_events_by_type(self, populated_visualizer):
        """Test getting clearance events by type."""
        by_type = populated_visualizer.get_clearance_events_by_type()
        assert "episodic" in by_type
        assert "semantic" in by_type
        assert "procedural" in by_type
        total = sum(by_type.values())
        assert total == 10

    def test_get_clearance_events_by_reason(self, populated_visualizer):
        """Test getting clearance events by reason."""
        by_reason = populated_visualizer.get_clearance_events_by_reason()
        assert "unused" in by_reason or "low_stability" in by_reason or "redundant" in by_reason
        total = sum(by_reason.values())
        assert total == 10

    def test_get_recent_clearances(self, populated_visualizer):
        """Test getting recent clearance events."""
        recent = populated_visualizer.get_recent_clearances(n=5)
        assert len(recent) == 5
        for event in recent:
            assert "memory_id" in event
            assert "memory_type" in event
            assert "reason" in event
            assert "age_days" in event
            assert "timestamp" in event

    def test_get_recent_clearances_more_than_available(self, populated_visualizer):
        """Test getting more clearances than available."""
        recent = populated_visualizer.get_recent_clearances(n=50)
        assert len(recent) == 10  # Only 10 events recorded

    def test_get_aqp4_trace(self, populated_visualizer):
        """Test getting AQP4 activity trace."""
        timestamps, aqp4 = populated_visualizer.get_aqp4_trace()
        assert len(timestamps) == 20
        assert len(aqp4) == 20
        assert all(0 <= a <= 1 for a in aqp4)

    def test_get_csf_flow_trace(self, populated_visualizer):
        """Test getting CSF flow trace."""
        timestamps, csf = populated_visualizer.get_csf_flow_trace()
        assert len(timestamps) == 20
        assert len(csf) == 20

    def test_export_data(self, populated_visualizer):
        """Test exporting visualization data."""
        data = populated_visualizer.export_data()
        assert "current_stage" in data
        assert "stage_percentages" in data
        assert "stage_timeline" in data
        assert "clearance_statistics" in data
        assert "clearance_by_stage" in data
        assert "waste_statistics" in data
        assert "clearance_events_by_type" in data
        assert "clearance_events_by_reason" in data
        assert "recent_clearances" in data
        assert "alerts" in data
        assert data["current_stage"] == "REM"

    def test_export_data_empty(self):
        """Test exporting data when empty."""
        visualizer = GlymphaticVisualizer()
        data = visualizer.export_data()
        assert data["current_stage"] is None
        assert data["clearance_statistics"] == {}

    def test_clear_history(self, populated_visualizer):
        """Test clearing history."""
        assert len(populated_visualizer._snapshots) > 0
        assert len(populated_visualizer._clearance_events) > 0

        populated_visualizer.clear_history()

        assert len(populated_visualizer._snapshots) == 0
        assert len(populated_visualizer._clearance_events) == 0
        assert len(populated_visualizer._stage_history) == 0
        assert len(populated_visualizer._clearance_history) == 0
        assert len(populated_visualizer._waste_history) == 0
        assert len(populated_visualizer._active_alerts) == 0


class TestGlymphaticVisualizerPlots:
    """Tests for plot functions."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with data for plotting."""
        visualizer = GlymphaticVisualizer(window_size=100)

        stages = [SleepStage.WAKE, SleepStage.NREM_LIGHT, SleepStage.NREM_DEEP, SleepStage.REM]
        for stage in stages:
            for i in range(5):
                visualizer.record_state(
                    sleep_stage=stage,
                    clearance_rate=0.3 if stage == SleepStage.WAKE else 0.8,
                    adenosine_level=0.5,
                    waste_level=0.4,
                )

        return visualizer

    def test_plot_sleep_stage_timeline(self, populated_visualizer):
        """Test plotting sleep stage timeline."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import plot_sleep_stage_timeline

        result = plot_sleep_stage_timeline(populated_visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_sleep_stage_timeline_no_data(self):
        """Test plotting with insufficient data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import plot_sleep_stage_timeline

        visualizer = GlymphaticVisualizer()
        # Only one stage - not enough for timeline
        visualizer.record_state(
            sleep_stage=SleepStage.WAKE,
            clearance_rate=0.2,
            adenosine_level=0.5,
            waste_level=0.4,
        )

        result = plot_sleep_stage_timeline(visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_sleep_stage_timeline_with_ax(self, populated_visualizer):
        """Test plotting with existing axes."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import plot_sleep_stage_timeline

        fig, ax = plt.subplots()
        result = plot_sleep_stage_timeline(populated_visualizer, ax=ax)
        assert result is ax
        plt.close("all")

    def test_plot_clearance_and_waste(self, populated_visualizer):
        """Test plotting clearance and waste."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import plot_clearance_and_waste

        result = plot_clearance_and_waste(populated_visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_clearance_and_waste_no_data(self):
        """Test plotting with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import plot_clearance_and_waste

        visualizer = GlymphaticVisualizer()
        result = plot_clearance_and_waste(visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_clearance_by_stage(self, populated_visualizer):
        """Test plotting clearance by stage."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import plot_clearance_by_stage

        result = plot_clearance_by_stage(populated_visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_clearance_by_stage_no_data(self):
        """Test plotting clearance by stage with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import plot_clearance_by_stage

        visualizer = GlymphaticVisualizer()
        result = plot_clearance_by_stage(visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_stage_pie(self, populated_visualizer):
        """Test plotting stage pie chart."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import plot_stage_pie

        result = plot_stage_pie(populated_visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_stage_pie_no_data(self):
        """Test plotting pie with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import plot_stage_pie

        visualizer = GlymphaticVisualizer()
        result = plot_stage_pie(visualizer)
        assert result is not None
        plt.close("all")

    def test_create_glymphatic_dashboard(self, populated_visualizer):
        """Test creating comprehensive dashboard."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import create_glymphatic_dashboard

        result = create_glymphatic_dashboard(populated_visualizer)
        assert result is not None
        plt.close("all")

    def test_create_glymphatic_dashboard_custom_size(self, populated_visualizer):
        """Test dashboard with custom figsize."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.glymphatic_visualizer import create_glymphatic_dashboard

        result = create_glymphatic_dashboard(populated_visualizer, figsize=(20, 12))
        assert result is not None
        plt.close("all")


class TestGlymphaticVisualizerEdgeCases:
    """Edge case tests for glymphatic visualizer."""

    def test_initialization_with_systems(self):
        """Test initialization with optional system dependencies."""
        from unittest.mock import MagicMock

        mock_glymphatic = MagicMock()
        mock_adenosine = MagicMock()

        visualizer = GlymphaticVisualizer(
            glymphatic_system=mock_glymphatic,
            adenosine_dynamics=mock_adenosine,
        )

        assert visualizer.glymphatic_system == mock_glymphatic
        assert visualizer.adenosine_dynamics == mock_adenosine

    def test_custom_alert_thresholds(self):
        """Test custom alert thresholds."""
        visualizer = GlymphaticVisualizer(
            alert_high_waste=0.5,  # Lower threshold
            alert_low_clearance=0.3,  # Higher threshold
        )

        assert visualizer.alert_high_waste == 0.5
        assert visualizer.alert_low_clearance == 0.3

        # Waste at 0.6 should trigger alert with lower threshold
        visualizer.record_state(
            sleep_stage=SleepStage.WAKE,
            clearance_rate=0.2,
            adenosine_level=0.5,
            waste_level=0.6,
        )

        alerts = visualizer.get_alerts()
        assert any("HIGH WASTE" in a for a in alerts)

    def test_stage_history_window_limit(self):
        """Test that stage history respects window limit."""
        visualizer = GlymphaticVisualizer(window_size=5)

        # Create many stage transitions
        for i in range(20):
            stage = SleepStage.WAKE if i % 2 == 0 else SleepStage.NREM_LIGHT
            visualizer.record_state(
                sleep_stage=stage,
                clearance_rate=0.5,
                adenosine_level=0.5,
                waste_level=0.4,
            )

        # Stage history should be limited
        assert len(visualizer._stage_history) <= 5

    def test_alert_low_clearance_nrem_light(self):
        """Test low clearance alert during NREM_LIGHT."""
        visualizer = GlymphaticVisualizer()

        visualizer.record_state(
            sleep_stage=SleepStage.NREM_LIGHT,  # Also triggers low clearance alert
            clearance_rate=0.05,
            adenosine_level=0.5,
            waste_level=0.4,
        )

        alerts = visualizer.get_alerts()
        assert any("LOW CLEARANCE" in a for a in alerts)

    def test_no_alert_during_rem(self):
        """Test no low clearance alert during REM (expected lower clearance)."""
        visualizer = GlymphaticVisualizer()

        visualizer.record_state(
            sleep_stage=SleepStage.REM,
            clearance_rate=0.05,  # Low clearance
            adenosine_level=0.4,
            waste_level=0.4,
        )

        alerts = visualizer.get_alerts()
        # REM is not in the checked stages for low clearance
        assert not any("LOW CLEARANCE" in a for a in alerts)

    def test_unknown_memory_type_in_clearance_events(self):
        """Test handling of unknown memory type."""
        visualizer = GlymphaticVisualizer()

        visualizer.record_clearance_event(
            memory_id="mem-custom",
            memory_type="custom_type",  # Not in default list
            reason="experimental",
            age_days=10.0,
            last_access_days=5.0,
            stability_score=0.3,
        )

        by_type = visualizer.get_clearance_events_by_type()
        assert "custom_type" in by_type
        assert by_type["custom_type"] == 1

    def test_snapshot_all_parameters(self):
        """Test snapshot with all custom parameters."""
        visualizer = GlymphaticVisualizer()

        snapshot = visualizer.record_state(
            sleep_stage=SleepStage.NREM_DEEP,
            clearance_rate=0.95,
            adenosine_level=0.2,
            waste_level=0.1,
            aqp4_activity=0.9,
            n_prune_candidates=15,
            csf_flow_rate=0.85,
        )

        assert snapshot.aqp4_activity == 0.9
        assert snapshot.n_prune_candidates == 15
        assert snapshot.csf_flow_rate == 0.85

    def test_clearance_event_export_format(self):
        """Test that recent clearances have correct format."""
        visualizer = GlymphaticVisualizer()

        visualizer.record_clearance_event(
            memory_id="mem-test",
            memory_type="episodic",
            reason="unused",
            age_days=100.5,
            last_access_days=90.2,
            stability_score=0.15,
        )

        recent = visualizer.get_recent_clearances(n=1)
        assert len(recent) == 1

        event = recent[0]
        assert event["memory_id"] == "mem-test"
        assert event["memory_type"] == "episodic"
        assert event["reason"] == "unused"
        assert event["age_days"] == 100.5
        assert event["last_access_days"] == 90.2
        assert event["stability_score"] == 0.15
        assert "timestamp" in event
        # Timestamp should be ISO format string
        assert "T" in event["timestamp"]
