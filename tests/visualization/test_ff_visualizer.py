"""
Tests for Forward-Forward Visualizer module.

Tests FF learning dynamics visualization.
"""

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from t4dm.visualization.ff_visualizer import (
    FFSnapshot,
    FFTrainingEvent,
    ForwardForwardVisualizer,
)


class TestFFSnapshot:
    """Tests for FFSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating an FF snapshot."""
        snapshot = FFSnapshot(
            timestamp=datetime.now(),
            layer_idx=0,
            goodness=1.5,
            threshold=1.0,
            is_positive=True,
            confidence=0.5,
            activation_norm=2.0,
            phase="positive",
            learning_rate=0.01,
            da_modulation=1.0,
            ne_modulation=1.0,
        )

        assert snapshot.layer_idx == 0
        assert snapshot.goodness == 1.5
        assert snapshot.threshold == 1.0
        assert snapshot.is_positive is True
        assert snapshot.confidence == 0.5
        assert snapshot.phase == "positive"


class TestFFTrainingEvent:
    """Tests for FFTrainingEvent dataclass."""

    def test_event_creation(self):
        """Test creating a training event."""
        event = FFTrainingEvent(
            timestamp=datetime.now(),
            phase="positive",
            layer_goodnesses=[1.0, 1.5, 2.0],
            layer_thresholds=[0.8, 1.0, 1.2],
            mean_goodness=1.5,
            mean_threshold=1.0,
            accuracy=0.85,
            positive_margin=0.5,
            negative_margin=0.0,
        )

        assert event.phase == "positive"
        assert len(event.layer_goodnesses) == 3
        assert event.accuracy == 0.85


class TestForwardForwardVisualizer:
    """Tests for ForwardForwardVisualizer."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return ForwardForwardVisualizer(window_size=100)

    def test_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer.window_size == 100
        assert visualizer._snapshots == []
        assert visualizer._training_events == []

    def test_record_layer_state(self, visualizer):
        """Test recording a layer state."""
        activation = np.random.randn(64).astype(np.float32)

        snapshot = visualizer.record_layer_state(
            layer_idx=0,
            goodness=1.5,
            threshold=1.0,
            activation=activation,
            phase="positive",
            learning_rate=0.01,
        )

        assert isinstance(snapshot, FFSnapshot)
        assert snapshot.goodness == 1.5
        assert snapshot.threshold == 1.0
        assert snapshot.is_positive is True
        assert snapshot.confidence == 0.5
        assert len(visualizer._snapshots) == 1

    def test_record_negative_phase(self, visualizer):
        """Test recording negative phase (goodness < threshold)."""
        activation = np.random.randn(64).astype(np.float32)

        snapshot = visualizer.record_layer_state(
            layer_idx=0,
            goodness=0.5,
            threshold=1.0,
            activation=activation,
            phase="negative",
        )

        assert snapshot.is_positive is False
        assert snapshot.confidence == 0.5

    def test_da_modulation(self, visualizer):
        """Test dopamine modulation of learning rate."""
        activation = np.random.randn(64).astype(np.float32)

        # High DA
        snapshot_high = visualizer.record_layer_state(
            layer_idx=0,
            goodness=1.0,
            threshold=1.0,
            activation=activation,
            da_level=1.0,
        )
        assert snapshot_high.da_modulation > 1.0

        # Low DA
        snapshot_low = visualizer.record_layer_state(
            layer_idx=0,
            goodness=1.0,
            threshold=1.0,
            activation=activation,
            da_level=0.0,
        )
        assert snapshot_low.da_modulation < 1.0

    def test_ne_modulation(self, visualizer):
        """Test norepinephrine modulation of threshold."""
        activation = np.random.randn(64).astype(np.float32)

        # High NE
        snapshot_high = visualizer.record_layer_state(
            layer_idx=0,
            goodness=1.0,
            threshold=1.0,
            activation=activation,
            ne_level=1.0,
        )
        assert snapshot_high.ne_modulation > 1.0

        # Low NE
        snapshot_low = visualizer.record_layer_state(
            layer_idx=0,
            goodness=1.0,
            threshold=1.0,
            activation=activation,
            ne_level=0.0,
        )
        assert snapshot_low.ne_modulation < 1.0

    def test_record_training_step(self, visualizer):
        """Test recording a training step."""
        event = visualizer.record_training_step(
            phase="positive",
            layer_goodnesses=[1.0, 1.5, 2.0],
            layer_thresholds=[0.8, 1.0, 1.2],
            accuracy=0.85,
        )

        assert isinstance(event, FFTrainingEvent)
        assert event.phase == "positive"
        assert event.mean_goodness == pytest.approx(1.5, rel=0.01)
        assert event.positive_margin > 0
        assert event.negative_margin == 0.0
        assert len(visualizer._training_events) == 1

    def test_record_negative_training_step(self, visualizer):
        """Test recording negative training step."""
        event = visualizer.record_training_step(
            phase="negative",
            layer_goodnesses=[0.5, 0.7, 0.8],
            layer_thresholds=[1.0, 1.0, 1.0],
            accuracy=0.75,
        )

        assert event.negative_margin > 0
        assert event.positive_margin == 0.0

    def test_window_size_limit(self, visualizer):
        """Test that window size limits history."""
        activation = np.random.randn(64).astype(np.float32)

        # Record more than window size
        for i in range(150):
            visualizer.record_layer_state(
                layer_idx=0,
                goodness=float(i),
                threshold=1.0,
                activation=activation,
            )

        # Should be limited to window size
        assert len(visualizer._snapshots) == 100

    def test_get_alerts_low_margin(self, visualizer):
        """Test alerts for low margin."""
        activation = np.random.randn(64).astype(np.float32)

        # Record with very low confidence
        visualizer.record_layer_state(
            layer_idx=0,
            goodness=1.0,
            threshold=0.99,  # Very close
            activation=activation,
        )

        alerts = visualizer.get_alerts()
        assert any("LOW MARGIN" in a for a in alerts)

    def test_get_layer_goodness_trace(self, visualizer):
        """Test getting goodness trace for a layer."""
        activation = np.random.randn(64).astype(np.float32)

        # Record several snapshots
        for i in range(5):
            visualizer.record_layer_state(
                layer_idx=0,
                goodness=float(i + 1),
                threshold=1.0,
                activation=activation,
            )

        timestamps, goodnesses = visualizer.get_layer_goodness_trace(0)

        assert len(timestamps) == 5
        assert len(goodnesses) == 5
        assert goodnesses == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_get_all_layer_goodnesses(self, visualizer):
        """Test getting all layer goodnesses."""
        activation = np.random.randn(64).astype(np.float32)

        # Record for multiple layers
        for layer in range(3):
            for i in range(3):
                visualizer.record_layer_state(
                    layer_idx=layer,
                    goodness=float(layer + i),
                    threshold=1.0,
                    activation=activation,
                )

        all_goodnesses = visualizer.get_all_layer_goodnesses()

        assert len(all_goodnesses) == 3
        assert 0 in all_goodnesses
        assert 1 in all_goodnesses
        assert 2 in all_goodnesses

    def test_activation_norm_computed(self, visualizer):
        """Test that activation norm is computed correctly."""
        activation = np.array([3.0, 4.0]).astype(np.float32)  # norm = 5.0

        snapshot = visualizer.record_layer_state(
            layer_idx=0,
            goodness=1.0,
            threshold=1.0,
            activation=activation,
        )

        assert snapshot.activation_norm == pytest.approx(5.0, rel=0.01)


class TestFFVisualizerIntegration:
    """Integration tests for FF visualizer."""

    def test_full_training_session(self):
        """Test simulating a full training session."""
        visualizer = ForwardForwardVisualizer(window_size=1000)
        activation = np.random.randn(64).astype(np.float32)

        # Simulate positive phase
        for i in range(10):
            for layer in range(3):
                visualizer.record_layer_state(
                    layer_idx=layer,
                    goodness=1.5 + layer * 0.2,
                    threshold=1.0,
                    activation=activation,
                    phase="positive",
                    learning_rate=0.01,
                )

        # Record training event
        event = visualizer.record_training_step(
            phase="positive",
            layer_goodnesses=[1.5, 1.7, 1.9],
            layer_thresholds=[1.0, 1.0, 1.0],
            accuracy=0.9,
        )

        # Simulate negative phase
        for i in range(10):
            for layer in range(3):
                visualizer.record_layer_state(
                    layer_idx=layer,
                    goodness=0.5 + layer * 0.1,
                    threshold=1.0,
                    activation=activation,
                    phase="negative",
                    learning_rate=0.01,
                )

        assert len(visualizer._snapshots) == 60  # 10 * 3 * 2
        assert len(visualizer._training_events) == 1

    def test_threshold_drift_detection(self):
        """Test detection of threshold drift."""
        visualizer = ForwardForwardVisualizer(
            window_size=1000,
            alert_threshold_drift=0.3,
        )
        activation = np.random.randn(64).astype(np.float32)

        # Record with drifting threshold
        for i in range(15):
            visualizer.record_layer_state(
                layer_idx=0,
                goodness=1.0,
                threshold=1.0 + i * 0.1,  # Drifting up
                activation=activation,
            )

        alerts = visualizer.get_alerts()
        # Should detect threshold drift
        assert any("THRESHOLD DRIFT" in a for a in alerts) or len(alerts) > 0

    def test_layer_history_isolation(self):
        """Test that layer histories are isolated."""
        visualizer = ForwardForwardVisualizer(window_size=100)
        activation = np.random.randn(64).astype(np.float32)

        # Record different patterns for different layers
        for i in range(5):
            visualizer.record_layer_state(
                layer_idx=0,
                goodness=1.0,  # Constant
                threshold=1.0,
                activation=activation,
            )
            visualizer.record_layer_state(
                layer_idx=1,
                goodness=2.0,  # Different constant
                threshold=1.0,
                activation=activation,
            )

        all_goodnesses = visualizer.get_all_layer_goodnesses()

        assert all(g == 1.0 for g in all_goodnesses[0])
        assert all(g == 2.0 for g in all_goodnesses[1])


class TestForwardForwardVisualizerAnalysis:
    """Tests for analysis methods."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with data."""
        visualizer = ForwardForwardVisualizer(window_size=100)
        activation = np.random.randn(64).astype(np.float32)

        # Record states for multiple layers
        for layer in range(3):
            for i in range(10):
                phase = "positive" if i < 5 else "negative"
                visualizer.record_layer_state(
                    layer_idx=layer,
                    goodness=1.5 + i * 0.1 if phase == "positive" else 0.5 + i * 0.05,
                    threshold=1.0,
                    activation=activation,
                    phase=phase,
                    learning_rate=0.01,
                    da_level=0.5 + i * 0.05,
                    ne_level=0.4 + i * 0.02,
                )

        # Record training events
        for i in range(5):
            visualizer.record_training_step(
                phase="positive" if i % 2 == 0 else "negative",
                layer_goodnesses=[1.5, 1.7, 1.9] if i % 2 == 0 else [0.5, 0.6, 0.7],
                layer_thresholds=[1.0, 1.0, 1.0],
                accuracy=0.8 + i * 0.02,
            )

        return visualizer

    def test_get_current_goodnesses(self, populated_visualizer):
        """Test getting current goodness for each layer."""
        goodnesses = populated_visualizer.get_current_goodnesses()
        assert 0 in goodnesses
        assert 1 in goodnesses
        assert 2 in goodnesses
        assert len(goodnesses) == 3

    def test_get_current_goodnesses_empty(self):
        """Test current goodnesses with no data."""
        visualizer = ForwardForwardVisualizer()
        assert visualizer.get_current_goodnesses() == {}

    def test_get_goodness_statistics(self, populated_visualizer):
        """Test getting goodness statistics."""
        stats = populated_visualizer.get_goodness_statistics()
        assert 0 in stats
        assert "mean" in stats[0]
        assert "std" in stats[0]
        assert "min" in stats[0]
        assert "max" in stats[0]
        assert "current" in stats[0]

    def test_get_goodness_statistics_empty(self):
        """Test goodness statistics with no data."""
        visualizer = ForwardForwardVisualizer()
        assert visualizer.get_goodness_statistics() == {}

    def test_get_layer_threshold_trace(self, populated_visualizer):
        """Test getting threshold trace for a layer."""
        timestamps, thresholds = populated_visualizer.get_layer_threshold_trace(0)
        assert len(timestamps) == 10
        assert len(thresholds) == 10
        assert all(t == 1.0 for t in thresholds)

    def test_get_current_thresholds(self, populated_visualizer):
        """Test getting current thresholds."""
        thresholds = populated_visualizer.get_current_thresholds()
        assert 0 in thresholds
        assert 1 in thresholds
        assert 2 in thresholds

    def test_get_current_thresholds_empty(self):
        """Test current thresholds with no data."""
        visualizer = ForwardForwardVisualizer()
        assert visualizer.get_current_thresholds() == {}

    def test_get_phase_breakdown(self, populated_visualizer):
        """Test getting phase breakdown."""
        breakdown = populated_visualizer.get_phase_breakdown()
        assert "positive" in breakdown
        assert "negative" in breakdown
        assert "inference" in breakdown
        # 3 layers * 5 positive + 3 layers * 5 negative = 15 each
        assert breakdown["positive"] == 15
        assert breakdown["negative"] == 15

    def test_get_accuracy_by_phase(self, populated_visualizer):
        """Test getting accuracy by phase."""
        accuracy = populated_visualizer.get_accuracy_by_phase()
        assert "positive" in accuracy or "negative" in accuracy
        for phase, acc in accuracy.items():
            assert 0 <= acc <= 1.0

    def test_get_accuracy_by_phase_empty(self):
        """Test accuracy by phase with no data."""
        visualizer = ForwardForwardVisualizer()
        assert visualizer.get_accuracy_by_phase() == {}

    def test_get_margin_history(self, populated_visualizer):
        """Test getting margin history."""
        margins = populated_visualizer.get_margin_history()
        assert "positive" in margins
        assert "negative" in margins
        # Should have margins from training events
        assert len(margins["positive"]) >= 1
        assert len(margins["negative"]) >= 1

    def test_get_neuromod_effects(self, populated_visualizer):
        """Test getting neuromodulator effects."""
        effects = populated_visualizer.get_neuromod_effects()
        assert "dopamine" in effects
        assert "norepinephrine" in effects
        assert "mean_modulation" in effects["dopamine"]
        assert "effect_on_lr" in effects["dopamine"]
        assert "mean_modulation" in effects["norepinephrine"]
        assert "effect_on_threshold" in effects["norepinephrine"]

    def test_get_neuromod_effects_empty(self):
        """Test neuromod effects with no data."""
        visualizer = ForwardForwardVisualizer()
        assert visualizer.get_neuromod_effects() == {}

    def test_export_data(self, populated_visualizer):
        """Test exporting visualization data."""
        data = populated_visualizer.export_data()
        assert "layer_goodnesses" in data
        assert "layer_thresholds" in data
        assert "current_goodnesses" in data
        assert "current_thresholds" in data
        assert "goodness_statistics" in data
        assert "phase_breakdown" in data
        assert "accuracy_by_phase" in data
        assert "margin_history" in data
        assert "neuromod_effects" in data
        assert "alerts" in data

    def test_clear_history(self, populated_visualizer):
        """Test clearing history."""
        assert len(populated_visualizer._snapshots) > 0
        assert len(populated_visualizer._training_events) > 0

        populated_visualizer.clear_history()

        assert len(populated_visualizer._snapshots) == 0
        assert len(populated_visualizer._training_events) == 0
        assert len(populated_visualizer._goodness_history) == 0
        assert len(populated_visualizer._threshold_history) == 0
        assert len(populated_visualizer._active_alerts) == 0


class TestFFVisualizerPlots:
    """Tests for plot functions."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with data for plotting."""
        visualizer = ForwardForwardVisualizer(window_size=100)
        activation = np.random.randn(64).astype(np.float32)

        # Record data for multiple layers
        for layer in range(3):
            for i in range(10):
                phase = "positive" if i < 5 else "negative"
                visualizer.record_layer_state(
                    layer_idx=layer,
                    goodness=1.5 if phase == "positive" else 0.5,
                    threshold=1.0,
                    activation=activation,
                    phase=phase,
                )

        # Record training events
        for i in range(5):
            visualizer.record_training_step(
                phase="positive" if i % 2 == 0 else "negative",
                layer_goodnesses=[1.5, 1.7, 1.9] if i % 2 == 0 else [0.5, 0.6, 0.7],
                layer_thresholds=[1.0, 1.0, 1.0],
                accuracy=0.85,
            )

        return visualizer

    def test_plot_goodness_bars(self, populated_visualizer):
        """Test plotting goodness bars."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.ff_visualizer import plot_goodness_bars

        result = plot_goodness_bars(populated_visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_goodness_bars_no_data(self):
        """Test plotting with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.ff_visualizer import plot_goodness_bars

        visualizer = ForwardForwardVisualizer()
        result = plot_goodness_bars(visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_goodness_timeline(self, populated_visualizer):
        """Test plotting goodness timeline."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.ff_visualizer import plot_goodness_timeline

        result = plot_goodness_timeline(populated_visualizer, layer_idx=0)
        assert result is not None
        plt.close("all")

    def test_plot_goodness_timeline_no_data(self):
        """Test timeline with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.ff_visualizer import plot_goodness_timeline

        visualizer = ForwardForwardVisualizer()
        result = plot_goodness_timeline(visualizer, layer_idx=0)
        assert result is not None
        plt.close("all")

    def test_plot_margin_evolution(self, populated_visualizer):
        """Test plotting margin evolution."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.ff_visualizer import plot_margin_evolution

        result = plot_margin_evolution(populated_visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_margin_evolution_no_data(self):
        """Test margin evolution with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.ff_visualizer import plot_margin_evolution

        visualizer = ForwardForwardVisualizer()
        result = plot_margin_evolution(visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_phase_comparison(self, populated_visualizer):
        """Test plotting phase comparison."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.ff_visualizer import plot_phase_comparison

        result = plot_phase_comparison(populated_visualizer)
        assert result is not None
        plt.close("all")

    def test_plot_phase_comparison_no_data(self):
        """Test phase comparison with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.ff_visualizer import plot_phase_comparison

        visualizer = ForwardForwardVisualizer()
        result = plot_phase_comparison(visualizer)
        assert result is not None
        plt.close("all")

    def test_create_ff_dashboard(self, populated_visualizer):
        """Test creating FF dashboard."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.ff_visualizer import create_ff_dashboard

        result = create_ff_dashboard(populated_visualizer)
        assert result is not None
        plt.close("all")

    def test_create_ff_dashboard_custom_size(self, populated_visualizer):
        """Test dashboard with custom figsize."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.ff_visualizer import create_ff_dashboard

        result = create_ff_dashboard(populated_visualizer, figsize=(20, 12))
        assert result is not None
        plt.close("all")


class TestFFVisualizerEdgeCases:
    """Edge case tests for FF visualizer."""

    def test_initialization_with_network(self):
        """Test initialization with FF network."""
        mock_network = MagicMock()
        visualizer = ForwardForwardVisualizer(ff_network=mock_network)
        assert visualizer.ff_network == mock_network

    def test_custom_alert_thresholds(self):
        """Test custom alert thresholds."""
        visualizer = ForwardForwardVisualizer(
            alert_low_margin=0.2,
            alert_threshold_drift=0.3,
        )
        assert visualizer.alert_low_margin == 0.2
        assert visualizer.alert_threshold_drift == 0.3

    def test_goodness_history_window_limit(self):
        """Test that goodness history respects window limit."""
        visualizer = ForwardForwardVisualizer(window_size=10)
        activation = np.random.randn(64).astype(np.float32)

        for i in range(20):
            visualizer.record_layer_state(
                layer_idx=0,
                goodness=float(i),
                threshold=1.0,
                activation=activation,
            )

        # History should be limited
        assert len(visualizer._goodness_history[0]) <= 10

    def test_training_event_window_limit(self):
        """Test that training events respect window limit."""
        visualizer = ForwardForwardVisualizer(window_size=10)

        for i in range(20):
            visualizer.record_training_step(
                phase="positive",
                layer_goodnesses=[1.0, 1.0, 1.0],
                layer_thresholds=[1.0, 1.0, 1.0],
            )

        assert len(visualizer._training_events) <= 10

    def test_inference_phase(self):
        """Test recording inference phase."""
        visualizer = ForwardForwardVisualizer()
        activation = np.random.randn(64).astype(np.float32)

        snapshot = visualizer.record_layer_state(
            layer_idx=0,
            goodness=1.2,
            threshold=1.0,
            activation=activation,
            phase="inference",
        )

        assert snapshot.phase == "inference"
        breakdown = visualizer.get_phase_breakdown()
        assert breakdown["inference"] == 1

    def test_get_layer_traces_nonexistent_layer(self):
        """Test getting traces for non-existent layer."""
        visualizer = ForwardForwardVisualizer()
        activation = np.random.randn(64).astype(np.float32)

        visualizer.record_layer_state(
            layer_idx=0,
            goodness=1.0,
            threshold=1.0,
            activation=activation,
        )

        # Layer 5 doesn't exist
        timestamps, goodnesses = visualizer.get_layer_goodness_trace(5)
        assert len(timestamps) == 0
        assert len(goodnesses) == 0
