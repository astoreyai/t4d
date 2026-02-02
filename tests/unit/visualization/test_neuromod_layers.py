"""Tests for neuromod_layers visualization module."""

import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np

from t4dm.visualization.neuromod_layers import (
    NeuromodLayerSnapshot,
    NeuromodLayerVisualizer,
    NT_NAMES,
    NT_COLORS,
    NT_LAYER_MAPPING,
)


def _make_snapshot(num_blocks: int = 6, timestamp: float = 0.0) -> NeuromodLayerSnapshot:
    return NeuromodLayerSnapshot(
        da_per_layer=[float(np.random.rand()) for _ in range(num_blocks)],
        ne_per_layer=[float(np.random.rand()) for _ in range(num_blocks)],
        ach_per_layer=[float(np.random.rand()) for _ in range(num_blocks)],
        sht_per_layer=[float(np.random.rand()) for _ in range(num_blocks)],
        num_blocks=num_blocks,
        timestamp=timestamp,
    )


class TestNeuromodLayerSnapshot:
    def test_creation(self):
        s = _make_snapshot()
        assert s.num_blocks == 6
        assert len(s.da_per_layer) == 6

    def test_constants(self):
        assert len(NT_NAMES) == 4
        assert "DA" in NT_COLORS
        assert "ACh" in NT_LAYER_MAPPING


class TestNeuromodLayerVisualizer:
    def test_record_snapshot(self):
        viz = NeuromodLayerVisualizer()
        viz.record_snapshot(_make_snapshot())
        assert viz.snapshot_count == 1

    def test_max_history(self):
        viz = NeuromodLayerVisualizer(max_history=3)
        for i in range(5):
            viz.record_snapshot(_make_snapshot(timestamp=float(i)))
        assert viz.snapshot_count == 3

    def test_plot_nt_heatmap_no_data(self):
        viz = NeuromodLayerVisualizer()
        ax = viz.plot_nt_heatmap()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_nt_heatmap_with_data(self):
        viz = NeuromodLayerVisualizer()
        viz.record_snapshot(_make_snapshot())
        ax = viz.plot_nt_heatmap()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_nt_per_block_no_data(self):
        viz = NeuromodLayerVisualizer()
        ax = viz.plot_nt_per_block()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_nt_per_block_with_data(self):
        viz = NeuromodLayerVisualizer()
        viz.record_snapshot(_make_snapshot())
        ax = viz.plot_nt_per_block()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_nt_evolution_no_data(self):
        viz = NeuromodLayerVisualizer()
        axes = viz.plot_nt_evolution()
        assert axes is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_nt_evolution_with_data(self):
        viz = NeuromodLayerVisualizer()
        for i in range(10):
            viz.record_snapshot(_make_snapshot(timestamp=float(i)))
        axes = viz.plot_nt_evolution()
        assert axes is not None
        assert len(axes) == 4
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_export_data_empty(self):
        viz = NeuromodLayerVisualizer()
        data = viz.export_data()
        assert data["snapshot_count"] == 0
        assert data["snapshots"] == []
        assert data["nt_names"] == NT_NAMES

    def test_export_data_with_snapshots(self):
        viz = NeuromodLayerVisualizer()
        viz.record_snapshot(_make_snapshot(timestamp=1.0))
        viz.record_snapshot(_make_snapshot(timestamp=2.0))
        data = viz.export_data()
        assert data["snapshot_count"] == 2
        assert len(data["snapshots"]) == 2
        assert "da_per_layer" in data["snapshots"][0]
        assert "layer_mapping" in data
