"""
Tests for Capsule Visualizer module.

Tests visualization of capsule network dynamics.
"""

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from t4dm.visualization.capsule_visualizer import (
    CapsuleSnapshot,
    RoutingEvent,
    CapsuleVisualizer,
)


class TestCapsuleSnapshot:
    """Tests for CapsuleSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a capsule snapshot."""
        snapshot = CapsuleSnapshot(
            timestamp=datetime.now(),
            layer_idx=0,
            n_capsules=16,
            capsule_lengths=np.random.rand(16),
            pose_vectors=np.random.randn(16, 8),
        )

        assert snapshot.layer_idx == 0
        assert snapshot.n_capsules == 16
        assert snapshot.routing_iterations == 3  # default

    def test_snapshot_with_routing(self):
        """Test snapshot with routing coefficients."""
        snapshot = CapsuleSnapshot(
            timestamp=datetime.now(),
            layer_idx=1,
            n_capsules=8,
            capsule_lengths=np.random.rand(8),
            pose_vectors=np.random.randn(8, 4),
            routing_coefficients=np.random.rand(8, 8),
            routing_iterations=5,
        )

        assert snapshot.routing_coefficients.shape == (8, 8)
        assert snapshot.routing_iterations == 5

    def test_snapshot_with_nt_modulation(self):
        """Test snapshot with neuromodulator info."""
        snapshot = CapsuleSnapshot(
            timestamp=datetime.now(),
            layer_idx=0,
            n_capsules=4,
            capsule_lengths=np.random.rand(4),
            pose_vectors=np.random.randn(4, 4),
            nt_modulation={"da_routing_temp": 0.5},
        )

        assert "da_routing_temp" in snapshot.nt_modulation


class TestRoutingEvent:
    """Tests for RoutingEvent dataclass."""

    def test_event_creation(self):
        """Test creating a routing event."""
        event = RoutingEvent(
            timestamp=datetime.now(),
            iteration=1,
            source_layer=0,
            target_layer=1,
            agreement_scores=np.random.rand(8, 16),
            coefficient_entropy=0.5,
            dominant_routes=[(0, 3, 0.8), (2, 7, 0.7)],
        )

        assert event.iteration == 1
        assert event.source_layer == 0
        assert event.target_layer == 1
        assert event.coefficient_entropy == 0.5
        assert len(event.dominant_routes) == 2


class TestCapsuleVisualizer:
    """Tests for CapsuleVisualizer."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return CapsuleVisualizer(window_size=100)

    def test_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer.window_size == 100
        assert visualizer.alert_low_agreement == 0.3
        assert visualizer.alert_high_entropy == 0.9
        assert visualizer._snapshots == []
        assert visualizer._routing_events == []

    def test_initialization_with_network(self):
        """Test initialization with capsule network."""
        mock_network = MagicMock()
        vis = CapsuleVisualizer(capsule_network=mock_network)
        assert vis.capsule_network is mock_network

    def test_record_layer_state(self, visualizer):
        """Test recording layer state."""
        lengths = np.random.rand(8)
        poses = np.random.randn(8, 4)

        snapshot = visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=lengths,
            pose_vectors=poses,
        )

        assert isinstance(snapshot, CapsuleSnapshot)
        assert snapshot.layer_idx == 0
        assert snapshot.n_capsules == 8
        assert len(visualizer._snapshots) == 1

    def test_record_layer_state_with_nt(self, visualizer):
        """Test recording with neuromodulator levels."""
        lengths = np.random.rand(8)
        poses = np.random.randn(8, 4)

        snapshot = visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=lengths,
            pose_vectors=poses,
            da_level=0.8,
            ne_level=0.6,
            ach_level=0.7,
            sht_level=0.4,
        )

        assert "da_routing_temp" in snapshot.nt_modulation
        assert "ne_squash_thresh" in snapshot.nt_modulation
        assert "ach_mode" in snapshot.nt_modulation
        assert "sht_patience" in snapshot.nt_modulation

    def test_record_layer_state_with_routing(self, visualizer):
        """Test recording with routing coefficients."""
        lengths = np.random.rand(8)
        poses = np.random.randn(8, 4)
        routing = np.random.rand(8, 8)

        snapshot = visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=lengths,
            pose_vectors=poses,
            routing_coefficients=routing,
            routing_iterations=5,
        )

        assert snapshot.routing_coefficients is not None
        assert snapshot.routing_iterations == 5

    def test_window_size_limit(self, visualizer):
        """Test that window size limits history."""
        for i in range(150):
            visualizer.record_layer_state(
                layer_idx=0,
                capsule_lengths=np.random.rand(8),
                pose_vectors=np.random.randn(8, 4),
            )

        assert len(visualizer._snapshots) == 100

    def test_record_routing_iteration(self, visualizer):
        """Test recording routing iteration."""
        predictions = np.random.randn(8, 16, 4)
        outputs = np.random.randn(16, 4)
        coefficients = np.random.rand(8, 16)

        event = visualizer.record_routing_iteration(
            iteration=1,
            source_layer=0,
            target_layer=1,
            predictions=predictions,
            outputs=outputs,
            coefficients=coefficients,
        )

        assert isinstance(event, RoutingEvent)
        assert event.iteration == 1
        assert len(visualizer._routing_events) == 1
        assert event.coefficient_entropy >= 0.0  # Entropy is non-negative

    def test_record_routing_iteration_2d_predictions(self, visualizer):
        """Test routing iteration with 2D predictions."""
        predictions = np.random.randn(8, 16)
        outputs = np.random.randn(16, 4)
        coefficients = np.random.rand(8, 16)

        event = visualizer.record_routing_iteration(
            iteration=1,
            source_layer=0,
            target_layer=1,
            predictions=predictions,
            outputs=outputs,
            coefficients=coefficients,
        )

        assert event.agreement_scores.shape == (8, 16)

    def test_get_alerts_low_confidence(self, visualizer):
        """Test alerts for low confidence detection."""
        # Record with low probabilities
        visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=np.full(8, 0.1),  # All low
            pose_vectors=np.random.randn(8, 4),
        )

        alerts = visualizer.get_alerts()
        assert any("LOW CONFIDENCE" in a for a in alerts)

    def test_get_alerts_high_entropy(self, visualizer):
        """Test alerts for high routing entropy."""
        # Record some routing events with high entropy
        for i in range(15):
            # Uniform coefficients = high entropy
            coefficients = np.ones((8, 16)) / 128

            visualizer.record_routing_iteration(
                iteration=i,
                source_layer=0,
                target_layer=0,
                predictions=np.random.randn(8, 16, 4),
                outputs=np.random.randn(16, 4),
                coefficients=coefficients,
            )

        # Record a snapshot for this layer
        visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=np.random.rand(8),
            pose_vectors=np.random.randn(8, 4),
        )

        alerts = visualizer.get_alerts()
        assert any("DIFFUSE ROUTING" in a for a in alerts)

    def test_get_current_probabilities(self, visualizer):
        """Test getting current probabilities."""
        lengths = np.array([0.1, 0.5, 0.8, 0.3])
        visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=lengths,
            pose_vectors=np.random.randn(4, 4),
        )

        probs = visualizer.get_current_probabilities(layer_idx=0)
        np.testing.assert_array_equal(probs, lengths)

    def test_get_current_probabilities_empty(self, visualizer):
        """Test probabilities when no data."""
        probs = visualizer.get_current_probabilities(layer_idx=0)
        assert len(probs) == 0

    def test_get_top_entities(self, visualizer):
        """Test getting top entities by probability."""
        lengths = np.array([0.1, 0.5, 0.8, 0.3, 0.9])
        visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=lengths,
            pose_vectors=np.random.randn(5, 4),
        )

        top = visualizer.get_top_entities(layer_idx=0, n=3)
        assert len(top) == 3
        assert top[0][0] == 4  # Index of 0.9
        assert top[0][1] == pytest.approx(0.9)

    def test_get_probability_history(self, visualizer):
        """Test getting probability history."""
        for i in range(5):
            visualizer.record_layer_state(
                layer_idx=0,
                capsule_lengths=np.full(4, float(i) * 0.1),
                pose_vectors=np.random.randn(4, 4),
            )

        history = visualizer.get_probability_history(layer_idx=0)
        assert len(history) == 5

    def test_get_current_poses(self, visualizer):
        """Test getting current poses."""
        poses = np.random.randn(4, 8)
        visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=np.random.rand(4),
            pose_vectors=poses,
        )

        current_poses = visualizer.get_current_poses(layer_idx=0)
        np.testing.assert_array_equal(current_poses, poses)

    def test_get_pose_variance(self, visualizer):
        """Test pose variance computation."""
        for i in range(5):
            visualizer.record_layer_state(
                layer_idx=0,
                capsule_lengths=np.random.rand(4),
                pose_vectors=np.random.randn(4, 4) * (i + 1),
            )

        variance = visualizer.get_pose_variance(layer_idx=0)
        assert variance > 0

    def test_get_pose_variance_insufficient_data(self, visualizer):
        """Test pose variance with insufficient data."""
        visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=np.random.rand(4),
            pose_vectors=np.random.randn(4, 4),
        )

        variance = visualizer.get_pose_variance(layer_idx=0)
        assert variance == 0.0

    def test_get_pose_alignment(self, visualizer):
        """Test pose alignment computation."""
        # Record similar poses
        for i in range(5):
            poses = np.random.randn(4, 4) + i * 0.01  # Small drift
            visualizer.record_layer_state(
                layer_idx=0,
                capsule_lengths=np.random.rand(4),
                pose_vectors=poses,
            )

        alignment = visualizer.get_pose_alignment(layer_idx=0)
        # Cosine similarity can be negative, so check it's a valid value
        assert -1.0 <= alignment <= 1.0

    def test_get_routing_matrix(self, visualizer):
        """Test getting routing matrix."""
        routing = np.random.rand(8, 8)
        visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=np.random.rand(8),
            pose_vectors=np.random.randn(8, 4),
            routing_coefficients=routing,
        )

        matrix = visualizer.get_routing_matrix(layer_idx=0)
        np.testing.assert_array_equal(matrix, routing)

    def test_get_routing_matrix_none(self, visualizer):
        """Test routing matrix when none recorded."""
        matrix = visualizer.get_routing_matrix(layer_idx=0)
        assert matrix is None

    def test_get_routing_entropy_history(self, visualizer):
        """Test getting routing entropy history."""
        for i in range(5):
            visualizer.record_routing_iteration(
                iteration=i,
                source_layer=0,
                target_layer=1,
                predictions=np.random.randn(8, 16, 4),
                outputs=np.random.randn(16, 4),
                coefficients=np.random.rand(8, 16),
            )

        entropy_history = visualizer.get_routing_entropy_history()
        assert len(entropy_history) == 5
        assert all(e >= 0.0 for e in entropy_history)  # Entropy is non-negative

    def test_get_dominant_routes_summary(self, visualizer):
        """Test getting dominant routes summary."""
        coefficients = np.zeros((4, 4))
        coefficients[0, 2] = 0.9  # Strong route 0->2
        coefficients[1, 3] = 0.8  # Strong route 1->3

        visualizer.record_routing_iteration(
            iteration=0,
            source_layer=0,
            target_layer=1,
            predictions=np.random.randn(4, 4, 4),
            outputs=np.random.randn(4, 4),
            coefficients=coefficients,
        )

        summary = visualizer.get_dominant_routes_summary()
        assert (0, 2) in summary or (1, 3) in summary

    def test_get_nt_modulation_summary(self, visualizer):
        """Test neuromodulator summary."""
        for i in range(10):
            visualizer.record_layer_state(
                layer_idx=0,
                capsule_lengths=np.random.rand(4),
                pose_vectors=np.random.randn(4, 4),
                da_level=0.7,
                ne_level=0.5,
                ach_level=0.6,
                sht_level=0.4,
            )

        summary = visualizer.get_nt_modulation_summary()
        assert "da_routing_temp" in summary
        assert "ne_squash_thresh" in summary
        assert "ach_mode" in summary
        assert "sht_patience" in summary

    def test_get_nt_modulation_summary_empty(self, visualizer):
        """Test summary with no data."""
        summary = visualizer.get_nt_modulation_summary()
        assert summary == {}

    def test_export_data(self, visualizer):
        """Test exporting visualization data."""
        # Record some data
        for i in range(5):
            visualizer.record_layer_state(
                layer_idx=0,
                capsule_lengths=np.random.rand(4),
                pose_vectors=np.random.randn(4, 4),
                routing_coefficients=np.random.rand(4, 4),
            )
            visualizer.record_routing_iteration(
                iteration=i,
                source_layer=0,
                target_layer=0,
                predictions=np.random.randn(4, 4, 4),
                outputs=np.random.randn(4, 4),
                coefficients=np.random.rand(4, 4),
            )

        data = visualizer.export_data()

        assert "layers" in data
        assert "routing_entropy_history" in data
        assert "dominant_routes" in data
        assert "nt_modulation" in data
        assert "alerts" in data

    def test_clear_history(self, visualizer):
        """Test clearing history."""
        for i in range(10):
            visualizer.record_layer_state(
                layer_idx=0,
                capsule_lengths=np.random.rand(4),
                pose_vectors=np.random.randn(4, 4),
            )

        assert len(visualizer._snapshots) == 10

        visualizer.clear_history()

        assert len(visualizer._snapshots) == 0
        assert len(visualizer._routing_events) == 0
        assert len(visualizer._length_history) == 0
        assert len(visualizer._pose_history) == 0


class TestCapsuleVisualizerIntegration:
    """Integration tests for capsule visualizer."""

    def test_full_capsule_monitoring(self):
        """Test complete capsule monitoring workflow."""
        visualizer = CapsuleVisualizer(window_size=1000)

        # Simulate forward pass through layers
        n_layers = 3
        n_capsules = [8, 16, 4]

        for iter_num in range(50):
            for layer_idx in range(n_layers):
                # Simulate layer output
                lengths = np.random.rand(n_capsules[layer_idx])
                poses = np.random.randn(n_capsules[layer_idx], 8)

                if layer_idx > 0:
                    # Routing from previous layer
                    routing = np.random.rand(
                        n_capsules[layer_idx - 1],
                        n_capsules[layer_idx],
                    )
                    # Normalize routing
                    routing = routing / routing.sum(axis=1, keepdims=True)

                    # Record routing iteration
                    visualizer.record_routing_iteration(
                        iteration=iter_num,
                        source_layer=layer_idx - 1,
                        target_layer=layer_idx,
                        predictions=np.random.randn(
                            n_capsules[layer_idx - 1],
                            n_capsules[layer_idx],
                            8,
                        ),
                        outputs=poses,
                        coefficients=routing,
                    )
                else:
                    routing = None

                # Record layer state
                visualizer.record_layer_state(
                    layer_idx=layer_idx,
                    capsule_lengths=lengths,
                    pose_vectors=poses,
                    routing_coefficients=routing if layer_idx > 0 else None,
                    da_level=0.5 + 0.3 * np.sin(iter_num / 10),
                    ne_level=0.5,
                )

        # Verify data collection
        assert len(visualizer._snapshots) > 0
        assert len(visualizer._routing_events) > 0

        # Export and verify
        data = visualizer.export_data()
        assert len(data["layers"]) == n_layers
        assert len(data["routing_entropy_history"]) > 0

    def test_alert_system(self):
        """Test that alert system detects issues."""
        visualizer = CapsuleVisualizer(
            window_size=100,
            alert_low_agreement=0.3,
            alert_high_entropy=0.8,
        )

        # Create conditions for alerts
        # 1. Low confidence
        visualizer.record_layer_state(
            layer_idx=0,
            capsule_lengths=np.full(8, 0.1),  # All low
            pose_vectors=np.random.randn(8, 4),
        )

        alerts = visualizer.get_alerts()
        assert len(alerts) >= 1

    def test_layer_history_isolation(self):
        """Test that layer histories are isolated."""
        visualizer = CapsuleVisualizer(window_size=100)

        # Record different patterns for different layers
        for i in range(10):
            visualizer.record_layer_state(
                layer_idx=0,
                capsule_lengths=np.full(4, 0.5),
                pose_vectors=np.random.randn(4, 4),
            )
            visualizer.record_layer_state(
                layer_idx=1,
                capsule_lengths=np.full(8, 0.7),
                pose_vectors=np.random.randn(8, 4),
            )

        # Check histories are separate
        history_0 = visualizer.get_probability_history(layer_idx=0)
        history_1 = visualizer.get_probability_history(layer_idx=1)

        assert len(history_0) == 10
        assert len(history_1) == 10
        assert history_0[0].shape != history_1[0].shape


class TestCapsulePlotFunctions:
    """Tests for capsule plot functions."""

    @pytest.fixture
    def populated_visualizer(self):
        """Create visualizer with capsule data for plotting."""
        from t4dm.visualization.capsule_visualizer import (
            plot_entity_probabilities,
            plot_routing_heatmap,
            plot_pose_vectors,
            plot_routing_entropy,
            create_capsule_dashboard,
        )

        visualizer = CapsuleVisualizer(window_size=100)

        # Record multiple layer states
        for i in range(10):
            # Layer 0
            lengths_0 = np.random.rand(8) * 0.8 + 0.1  # 0.1 to 0.9
            poses_0 = np.random.randn(8, 4)
            routing_0 = np.random.rand(8, 16)
            routing_0 = routing_0 / routing_0.sum(axis=1, keepdims=True)

            visualizer.record_layer_state(
                layer_idx=0,
                capsule_lengths=lengths_0,
                pose_vectors=poses_0,
                routing_coefficients=routing_0,
                da_level=0.5 + 0.2 * np.sin(i / 5),
                ne_level=0.6,
                ach_level=0.55,
                sht_level=0.4,
            )

            # Layer 1 (output)
            lengths_1 = np.random.rand(4) * 0.9 + 0.05
            poses_1 = np.random.randn(4, 4)

            visualizer.record_layer_state(
                layer_idx=-1,
                capsule_lengths=lengths_1,
                pose_vectors=poses_1,
            )

            # Record routing iteration
            visualizer.record_routing_iteration(
                iteration=i,
                source_layer=0,
                target_layer=1,
                predictions=np.random.randn(8, 4, 4),
                outputs=poses_1,
                coefficients=routing_0[:, :4],
            )

        return visualizer

    def test_plot_entity_probabilities(self, populated_visualizer):
        """Test plotting entity probabilities."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import plot_entity_probabilities

        ax = plot_entity_probabilities(populated_visualizer, layer_idx=0)
        assert ax is not None
        plt.close("all")

    def test_plot_entity_probabilities_output_layer(self, populated_visualizer):
        """Test plotting entity probabilities for output layer."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import plot_entity_probabilities

        ax = plot_entity_probabilities(populated_visualizer, layer_idx=-1)
        assert ax is not None
        plt.close("all")

    def test_plot_entity_probabilities_empty(self):
        """Test plotting entity probabilities with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import plot_entity_probabilities

        empty_vis = CapsuleVisualizer()
        ax = plot_entity_probabilities(empty_vis, layer_idx=0)
        assert ax is not None
        plt.close("all")

    def test_plot_routing_heatmap(self, populated_visualizer):
        """Test plotting routing heatmap."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import plot_routing_heatmap

        ax = plot_routing_heatmap(populated_visualizer, layer_idx=0)
        assert ax is not None
        plt.close("all")

    def test_plot_routing_heatmap_no_data(self):
        """Test routing heatmap with no routing data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import plot_routing_heatmap

        vis = CapsuleVisualizer()
        vis.record_layer_state(
            layer_idx=0,
            capsule_lengths=np.random.rand(4),
            pose_vectors=np.random.randn(4, 4),
            # No routing_coefficients
        )

        ax = plot_routing_heatmap(vis, layer_idx=0)
        assert ax is not None
        plt.close("all")

    def test_plot_pose_vectors(self, populated_visualizer):
        """Test plotting pose vectors."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import plot_pose_vectors

        ax = plot_pose_vectors(populated_visualizer, layer_idx=0)
        assert ax is not None
        plt.close("all")

    def test_plot_pose_vectors_empty(self):
        """Test plotting pose vectors with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import plot_pose_vectors

        empty_vis = CapsuleVisualizer()
        ax = plot_pose_vectors(empty_vis, layer_idx=0)
        assert ax is not None
        plt.close("all")

    def test_plot_pose_vectors_1d_pose(self):
        """Test plotting pose vectors with 1D poses."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import plot_pose_vectors

        vis = CapsuleVisualizer()
        vis.record_layer_state(
            layer_idx=0,
            capsule_lengths=np.random.rand(4),
            pose_vectors=np.random.randn(4, 1),  # 1D poses
        )

        ax = plot_pose_vectors(vis, layer_idx=0)
        assert ax is not None
        plt.close("all")

    def test_plot_routing_entropy(self, populated_visualizer):
        """Test plotting routing entropy."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import plot_routing_entropy

        ax = plot_routing_entropy(populated_visualizer)
        assert ax is not None
        plt.close("all")

    def test_plot_routing_entropy_empty(self):
        """Test plotting routing entropy with no data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import plot_routing_entropy

        empty_vis = CapsuleVisualizer()
        ax = plot_routing_entropy(empty_vis)
        assert ax is not None
        plt.close("all")

    def test_create_capsule_dashboard(self, populated_visualizer):
        """Test creating capsule dashboard."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import create_capsule_dashboard

        fig = create_capsule_dashboard(populated_visualizer, layer_idx=0)
        assert fig is not None
        plt.close("all")

    def test_create_capsule_dashboard_output_layer(self, populated_visualizer):
        """Test creating capsule dashboard for output layer."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import create_capsule_dashboard

        fig = create_capsule_dashboard(populated_visualizer, layer_idx=-1)
        assert fig is not None
        plt.close("all")

    def test_create_capsule_dashboard_with_alerts(self):
        """Test dashboard with alerts present."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from t4dm.visualization.capsule_visualizer import create_capsule_dashboard

        vis = CapsuleVisualizer(alert_low_agreement=0.3, alert_high_entropy=0.5)

        # Create data that triggers alerts
        vis.record_layer_state(
            layer_idx=0,
            capsule_lengths=np.full(8, 0.1),  # Low confidence
            pose_vectors=np.random.randn(8, 4),
            routing_coefficients=np.random.rand(8, 8),
        )

        # Record routing with high entropy
        uniform_routing = np.ones((8, 8)) / 64
        vis.record_routing_iteration(
            iteration=0,
            source_layer=0,
            target_layer=0,
            predictions=np.random.randn(8, 8, 4),
            outputs=np.random.randn(8, 4),
            coefficients=uniform_routing,
        )

        fig = create_capsule_dashboard(vis, layer_idx=0)
        assert fig is not None
        plt.close("all")


class TestCapsuleVisualizerEdgeCases:
    """Edge case tests for capsule visualizer."""

    def test_very_large_capsule_layer(self):
        """Test with very large capsule layer."""
        vis = CapsuleVisualizer()
        n_capsules = 256

        vis.record_layer_state(
            layer_idx=0,
            capsule_lengths=np.random.rand(n_capsules),
            pose_vectors=np.random.randn(n_capsules, 16),
        )

        probs = vis.get_current_probabilities(layer_idx=0)
        assert len(probs) == n_capsules

    def test_zero_routing_coefficients(self):
        """Test with zero routing coefficients."""
        vis = CapsuleVisualizer()

        vis.record_routing_iteration(
            iteration=0,
            source_layer=0,
            target_layer=1,
            predictions=np.random.randn(4, 4, 4),
            outputs=np.random.randn(4, 4),
            coefficients=np.zeros((4, 4)),
        )

        entropy = vis.get_routing_entropy_history()
        assert len(entropy) == 1
        assert entropy[0] == 0.0

    def test_single_dominant_route(self):
        """Test with single dominant route."""
        vis = CapsuleVisualizer()

        coefficients = np.zeros((4, 4))
        coefficients[0, 0] = 1.0  # Single dominant route

        vis.record_routing_iteration(
            iteration=0,
            source_layer=0,
            target_layer=1,
            predictions=np.random.randn(4, 4, 4),
            outputs=np.random.randn(4, 4),
            coefficients=coefficients,
        )

        routes = vis.get_dominant_routes_summary()
        assert (0, 0) in routes

    def test_multiple_layers_same_recording(self):
        """Test recording multiple layers in same timestep."""
        vis = CapsuleVisualizer()

        # Record 3 layers
        for layer in range(3):
            vis.record_layer_state(
                layer_idx=layer,
                capsule_lengths=np.random.rand(4 * (layer + 1)),
                pose_vectors=np.random.randn(4 * (layer + 1), 4),
            )

        # Check all layers recorded
        for layer in range(3):
            probs = vis.get_current_probabilities(layer_idx=layer)
            assert len(probs) == 4 * (layer + 1)
