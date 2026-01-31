"""Tests for activation heatmap visualization module."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from t4dm.visualization.activation_heatmap import (
    ActivationHeatmap,
    ActivationSnapshot,
    plot_activation_heatmap,
    plot_activation_timeline,
)


class TestActivationSnapshot:
    """Tests for ActivationSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating an activation snapshot."""
        snapshot = ActivationSnapshot(
            timestamp=datetime.now(),
            episodic_activations=np.array([0.1, 0.5, 0.9]),
            semantic_activations=np.array([0.2, 0.8]),
            neuromod_state={"DA": 0.7, "NE": 0.5, "ACh": 0.6},
            memory_ids=["ep1", "ep2", "ep3", "sem1", "sem2"],
        )
        assert snapshot.episodic_activations.shape == (3,)
        assert snapshot.semantic_activations.shape == (2,)
        assert len(snapshot.memory_ids) == 5

    def test_snapshot_neuromod_access(self):
        """Test accessing neuromodulator state from snapshot."""
        snapshot = ActivationSnapshot(
            timestamp=datetime.now(),
            episodic_activations=np.array([0.5]),
            semantic_activations=np.array([0.5]),
            neuromod_state={"DA": 1.2, "NE": 0.8, "ACh": 0.4, "5-HT": 0.6, "GABA": 0.3},
            memory_ids=["m1", "m2"],
        )
        assert snapshot.neuromod_state["DA"] == 1.2
        assert snapshot.neuromod_state["GABA"] == 0.3


class TestActivationHeatmap:
    """Tests for ActivationHeatmap class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        heatmap = ActivationHeatmap()
        assert heatmap.window_size == 100
        assert heatmap.max_memories_tracked == 50
        assert len(heatmap._snapshots) == 0

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        heatmap = ActivationHeatmap(window_size=50, max_memories_tracked=20)
        assert heatmap.window_size == 50
        assert heatmap.max_memories_tracked == 20

    def test_record_snapshot(self):
        """Test recording a single snapshot."""
        heatmap = ActivationHeatmap(window_size=10)
        heatmap.record_snapshot(
            episodic_activations={"ep1": 0.5, "ep2": 0.8},
            semantic_activations={"sem1": 0.3},
            neuromod_state={"DA": 0.7},
        )
        assert len(heatmap._snapshots) == 1
        assert heatmap._snapshots[0].episodic_activations.shape == (2,)

    def test_record_multiple_snapshots(self):
        """Test recording multiple snapshots."""
        heatmap = ActivationHeatmap(window_size=10)
        for i in range(5):
            heatmap.record_snapshot(
                episodic_activations={f"ep{i}": 0.1 * i},
                semantic_activations={},
                neuromod_state={"DA": 0.5},
            )
        assert len(heatmap._snapshots) == 5

    def test_window_size_limit(self):
        """Test that window size is maintained."""
        heatmap = ActivationHeatmap(window_size=5)
        for i in range(10):
            heatmap.record_snapshot(
                episodic_activations={f"ep{i}": 0.1},
                semantic_activations={},
                neuromod_state={"DA": 0.5},
            )
        assert len(heatmap._snapshots) == 5

    def test_get_activation_matrix_episodic(self):
        """Test getting activation matrix for episodic memories."""
        heatmap = ActivationHeatmap(window_size=10)
        for i in range(3):
            heatmap.record_snapshot(
                episodic_activations={"ep1": 0.1 * i, "ep2": 0.2 * i},
                semantic_activations={},
                neuromod_state={"DA": 0.5},
            )
        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert matrix.shape[0] == 3  # 3 timesteps
        assert len(timestamps) == 3

    def test_get_activation_matrix_semantic(self):
        """Test getting activation matrix for semantic memories."""
        heatmap = ActivationHeatmap(window_size=10)
        heatmap.record_snapshot(
            episodic_activations={},
            semantic_activations={"sem1": 0.5, "sem2": 0.7, "sem3": 0.9},
            neuromod_state={"DA": 0.5},
        )
        matrix, ids, timestamps = heatmap.get_activation_matrix("semantic")
        assert len(timestamps) == 1

    def test_empty_activation_matrix(self):
        """Test getting activation matrix when no snapshots recorded."""
        heatmap = ActivationHeatmap()
        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert len(timestamps) == 0

    def test_get_neuromod_timeline(self):
        """Test getting neuromodulator timeline."""
        heatmap = ActivationHeatmap()
        for i in range(5):
            heatmap.record_snapshot(
                episodic_activations={},
                semantic_activations={},
                neuromod_state={"DA": 0.1 * i, "NE": 0.5, "ACh": 0.3},
            )
        matrix, mod_names, timestamps = heatmap.get_neuromod_timeline()
        assert "DA" in mod_names
        assert "NE" in mod_names
        assert matrix.shape[0] == 5  # 5 timesteps
        assert len(timestamps) == 5


try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE,
    reason="matplotlib not available for integration tests"
)
class TestPlotFunctions:
    """Integration tests for plot functions - require matplotlib."""

    def test_data_for_activation_heatmap(self):
        """Test data preparation for activation heatmap (no actual plotting)."""
        heatmap = ActivationHeatmap()
        for i in range(3):
            heatmap.record_snapshot(
                episodic_activations={"ep1": 0.5, "ep2": 0.3},
                semantic_activations={},
                neuromod_state={"DA": 0.7},
            )
        # Verify data is retrievable for plotting
        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert matrix.shape[0] == 3
        assert len(timestamps) == 3

    def test_data_for_activation_timeline(self):
        """Test data preparation for activation timeline."""
        heatmap = ActivationHeatmap()
        for i in range(5):
            heatmap.record_snapshot(
                episodic_activations={"ep1": 0.1 * i},
                semantic_activations={},
                neuromod_state={"DA": 0.5 + 0.1 * i, "NE": 0.6},
            )
        # Verify neuromodulator timeline data is available
        matrix, mod_names, timestamps = heatmap.get_neuromod_timeline()
        assert len(timestamps) == 5
        assert "DA" in mod_names

    def test_empty_data_handling(self):
        """Test data retrieval with no data recorded."""
        heatmap = ActivationHeatmap()
        # Should handle gracefully
        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert len(timestamps) == 0
        nm_matrix, mod_names, nm_timestamps = heatmap.get_neuromod_timeline()
        assert len(nm_timestamps) == 0


class TestActivationHeatmapAnalysis:
    """Extended tests for ActivationHeatmap analysis methods."""

    @pytest.fixture
    def populated_heatmap(self):
        """Create heatmap with mixed activations."""
        heatmap = ActivationHeatmap(window_size=20, max_memories_tracked=10)
        for i in range(10):
            heatmap.record_snapshot(
                episodic_activations={
                    "episode_a": 0.1 * i,
                    "episode_b": 0.8 - 0.05 * i,
                    "episode_c": 0.5
                },
                semantic_activations={
                    "entity_x": 0.3 + 0.05 * i,
                    "entity_y": 0.6
                },
                neuromod_state={
                    "DA": 0.5 + 0.02 * i,
                    "NE": 0.4 - 0.01 * i,
                    "ACh": 0.6,
                    "5-HT": 0.3,
                    "GABA": 0.5
                }
            )
        return heatmap

    def test_activation_matrix_values(self, populated_heatmap):
        """Test activation matrix contains expected values."""
        matrix, ids, timestamps = populated_heatmap.get_activation_matrix("episodic")

        assert matrix.shape[0] == 10  # 10 timesteps
        assert len(ids) > 0
        assert len(timestamps) == 10

        # Values should be in reasonable range
        assert np.all(matrix >= 0)
        assert np.all(matrix <= 1)

    def test_semantic_matrix_values(self, populated_heatmap):
        """Test semantic activation matrix."""
        matrix, ids, timestamps = populated_heatmap.get_activation_matrix("semantic")

        assert matrix.shape[0] == 10  # 10 timesteps
        assert len(timestamps) == 10

    def test_neuromod_timeline_values(self, populated_heatmap):
        """Test neuromodulator timeline values."""
        matrix, mod_names, timestamps = populated_heatmap.get_neuromod_timeline()

        assert matrix.shape[0] == 10  # 10 timesteps
        assert len(mod_names) == 5  # DA, NE, ACh, 5-HT, GABA
        assert "DA" in mod_names
        assert "GABA" in mod_names

    def test_max_memories_tracked_limit(self):
        """Test max_memories_tracked is respected."""
        heatmap = ActivationHeatmap(window_size=100, max_memories_tracked=5)

        # Record with more memories than limit
        for i in range(10):
            eps = {f"ep{j}": 0.5 for j in range(20)}
            heatmap.record_snapshot(
                episodic_activations=eps,
                semantic_activations={},
                neuromod_state={"DA": 0.5}
            )

        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        # Should be limited to max_memories_tracked
        assert len(ids) <= 5

    def test_sliding_window_behavior(self):
        """Test sliding window removes old snapshots."""
        heatmap = ActivationHeatmap(window_size=5)

        for i in range(20):
            heatmap.record_snapshot(
                episodic_activations={f"ep_{i}": float(i)},
                semantic_activations={},
                neuromod_state={"DA": 0.5}
            )

        # Only last 5 should remain
        assert len(heatmap._snapshots) == 5

        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert matrix.shape[0] == 5


class TestPlotActivationHeatmap:
    """Tests for plot_activation_heatmap function."""

    @pytest.fixture
    def heatmap_with_data(self):
        """Create heatmap with data for plotting."""
        heatmap = ActivationHeatmap(window_size=20)
        for i in range(10):
            heatmap.record_snapshot(
                episodic_activations={
                    "mem_001": 0.1 * i,
                    "mem_002": 0.9 - 0.08 * i,
                    "mem_003": 0.5
                },
                semantic_activations={
                    "ent_A": 0.4 + 0.05 * i,
                    "ent_B": 0.7
                },
                neuromod_state={
                    "DA": 0.6,
                    "NE": 0.4
                }
            )
        return heatmap

    def test_plot_activation_heatmap_episodic(self, heatmap_with_data):
        """Test plotting episodic activation heatmap."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_activation_heatmap(heatmap_with_data, memory_type="episodic")
        plt.close("all")

    def test_plot_activation_heatmap_semantic(self, heatmap_with_data):
        """Test plotting semantic activation heatmap."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_activation_heatmap(heatmap_with_data, memory_type="semantic")
        plt.close("all")

    def test_plot_activation_heatmap_save(self, heatmap_with_data, tmp_path):
        """Test saving activation heatmap to file."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        save_path = tmp_path / "activation_heatmap.png"
        plot_activation_heatmap(
            heatmap_with_data,
            memory_type="episodic",
            save_path=save_path
        )
        assert save_path.exists()
        plt.close("all")

    def test_plot_activation_heatmap_empty(self):
        """Test plotting empty heatmap logs warning."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        empty_heatmap = ActivationHeatmap()
        # Should log warning but not raise
        plot_activation_heatmap(empty_heatmap, memory_type="episodic")
        plt.close("all")


class TestPlotActivationTimeline:
    """Tests for plot_activation_timeline function."""

    @pytest.fixture
    def heatmap_with_neuromod(self):
        """Create heatmap with neuromodulator data."""
        heatmap = ActivationHeatmap(window_size=50)
        for i in range(20):
            heatmap.record_snapshot(
                episodic_activations={"ep1": 0.5},
                semantic_activations={},
                neuromod_state={
                    "DA": 0.4 + 0.02 * i,
                    "NE": 0.6 - 0.015 * i,
                    "ACh": 0.5 + 0.01 * np.sin(i),
                    "5-HT": 0.35,
                    "GABA": 0.45
                }
            )
        return heatmap

    def test_plot_activation_timeline(self, heatmap_with_neuromod):
        """Test plotting neuromodulator timeline."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_activation_timeline(heatmap_with_neuromod)
        plt.close("all")

    def test_plot_activation_timeline_save(self, heatmap_with_neuromod, tmp_path):
        """Test saving neuromodulator timeline to file."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        save_path = tmp_path / "neuromod_timeline.png"
        plot_activation_timeline(heatmap_with_neuromod, save_path=save_path)
        assert save_path.exists()
        plt.close("all")

    def test_plot_activation_timeline_empty(self):
        """Test plotting empty timeline logs warning."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        empty_heatmap = ActivationHeatmap()
        # Should log warning but not raise
        plot_activation_timeline(empty_heatmap)
        plt.close("all")


class TestActivationHeatmapEdgeCases:
    """Edge case tests for activation heatmap."""

    def test_single_snapshot(self):
        """Test with single snapshot."""
        heatmap = ActivationHeatmap()
        heatmap.record_snapshot(
            episodic_activations={"single_mem": 0.99},
            semantic_activations={},
            neuromod_state={"DA": 0.5}
        )

        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert matrix.shape[0] == 1
        assert len(ids) == 1

    def test_many_memories_few_snapshots(self):
        """Test many memories with few snapshots."""
        heatmap = ActivationHeatmap(max_memories_tracked=100)
        eps = {f"memory_{i:04d}": np.random.random() for i in range(100)}
        heatmap.record_snapshot(
            episodic_activations=eps,
            semantic_activations={},
            neuromod_state={"DA": 0.5}
        )

        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert matrix.shape[0] == 1
        assert len(ids) <= 100

    def test_zero_activations(self):
        """Test with zero activation values."""
        heatmap = ActivationHeatmap()
        heatmap.record_snapshot(
            episodic_activations={"zero_ep": 0.0},
            semantic_activations={"zero_sem": 0.0},
            neuromod_state={"DA": 0.0, "NE": 0.0}
        )

        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert np.all(matrix == 0)

    def test_high_activations(self):
        """Test with high activation values."""
        heatmap = ActivationHeatmap()
        heatmap.record_snapshot(
            episodic_activations={"high_ep": 1.0},
            semantic_activations={"high_sem": 0.999},
            neuromod_state={"DA": 1.5, "NE": 2.0}  # Can exceed 1.0
        )

        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert np.max(matrix) == 1.0

    def test_long_memory_ids(self):
        """Test with very long memory IDs."""
        heatmap = ActivationHeatmap()
        long_id = "memory_" + "a" * 500
        heatmap.record_snapshot(
            episodic_activations={long_id: 0.5},
            semantic_activations={},
            neuromod_state={"DA": 0.5}
        )

        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert long_id in ids

    def test_special_characters_in_ids(self):
        """Test memory IDs with special characters."""
        heatmap = ActivationHeatmap()
        heatmap.record_snapshot(
            episodic_activations={
                "ep:with:colons": 0.5,
                "ep/with/slashes": 0.6,
                "ep.with.dots": 0.7
            },
            semantic_activations={},
            neuromod_state={"DA": 0.5}
        )

        matrix, ids, timestamps = heatmap.get_activation_matrix("episodic")
        assert "ep:with:colons" in ids or any("colon" in i for i in ids)

    def test_plot_many_memories(self):
        """Test plotting with many memories."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        heatmap = ActivationHeatmap(max_memories_tracked=30)
        for i in range(5):
            eps = {f"memory_{j:02d}": np.random.random() for j in range(30)}
            heatmap.record_snapshot(
                episodic_activations=eps,
                semantic_activations={},
                neuromod_state={"DA": 0.5, "NE": 0.4}
            )

        plot_activation_heatmap(heatmap, memory_type="episodic")
        plt.close("all")

    def test_plot_many_timesteps(self):
        """Test plotting with many timesteps."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        heatmap = ActivationHeatmap(window_size=100)
        for i in range(50):
            heatmap.record_snapshot(
                episodic_activations={"ep1": 0.5, "ep2": 0.6},
                semantic_activations={},
                neuromod_state={"DA": 0.5 + 0.01 * i}
            )

        plot_activation_heatmap(heatmap, memory_type="episodic")
        plot_activation_timeline(heatmap)
        plt.close("all")
