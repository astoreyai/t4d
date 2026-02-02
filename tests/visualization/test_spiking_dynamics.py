"""Tests for spiking dynamics visualization."""

import pytest
import numpy as np

from t4dm.visualization.spiking_dynamics import (
    SpikingDynamicsVisualizer,
    SpikingSnapshot,
)


class TestSpikingSnapshot:
    """Test SpikingSnapshot dataclass."""

    def test_create_snapshot(self):
        snapshot = SpikingSnapshot(
            membrane_potentials=np.array([0.5, 0.3, 0.8]),
            spike_mask=np.array([False, False, True]),
            thalamic_gate=np.array([0.9, 0.1, 0.5]),
            apical_error=np.array([0.01, -0.02, 0.03]),
            block_index=0,
            timestamp=1.0,
        )
        assert snapshot.block_index == 0
        assert np.sum(snapshot.spike_mask) == 1


class TestSpikingDynamicsVisualizer:
    """Test SpikingDynamicsVisualizer class."""

    def _make_snapshot(self, n=16, ts=0.0, block=0):
        rng = np.random.RandomState(int(ts * 10))
        return SpikingSnapshot(
            membrane_potentials=rng.randn(n).astype(np.float32),
            spike_mask=rng.rand(n) > 0.7,
            thalamic_gate=rng.rand(n).astype(np.float32),
            apical_error=rng.randn(n).astype(np.float32) * 0.1,
            block_index=block,
            timestamp=ts,
        )

    def test_initialization(self):
        viz = SpikingDynamicsVisualizer()
        assert viz.window_size == 500

    def test_record_snapshot(self):
        viz = SpikingDynamicsVisualizer()
        viz.record_snapshot(self._make_snapshot())
        assert len(viz._snapshots) == 1

    def test_window_size_limit(self):
        viz = SpikingDynamicsVisualizer(window_size=5)
        for i in range(10):
            viz.record_snapshot(self._make_snapshot(ts=float(i)))
        assert len(viz._snapshots) == 5

    def test_plot_spike_raster(self):
        viz = SpikingDynamicsVisualizer()
        for i in range(20):
            viz.record_snapshot(self._make_snapshot(ts=float(i)))
        result = viz.plot_spike_raster()
        assert result is not None

    def test_plot_spike_raster_empty(self):
        viz = SpikingDynamicsVisualizer()
        result = viz.plot_spike_raster()
        assert result is not None

    def test_plot_membrane_potentials(self):
        viz = SpikingDynamicsVisualizer()
        for i in range(10):
            viz.record_snapshot(self._make_snapshot(ts=float(i)))
        result = viz.plot_membrane_potentials()
        assert result is not None

    def test_plot_membrane_potentials_empty(self):
        viz = SpikingDynamicsVisualizer()
        result = viz.plot_membrane_potentials()
        assert result is not None

    def test_plot_thalamic_gate_state(self):
        viz = SpikingDynamicsVisualizer()
        viz.record_snapshot(self._make_snapshot())
        result = viz.plot_thalamic_gate_state()
        assert result is not None

    def test_plot_thalamic_gate_state_empty(self):
        viz = SpikingDynamicsVisualizer()
        result = viz.plot_thalamic_gate_state()
        assert result is not None

    def test_plot_apical_modulation(self):
        viz = SpikingDynamicsVisualizer()
        viz.record_snapshot(self._make_snapshot(n=64))
        result = viz.plot_apical_modulation()
        assert result is not None

    def test_plot_apical_modulation_empty(self):
        viz = SpikingDynamicsVisualizer()
        result = viz.plot_apical_modulation()
        assert result is not None

    def test_create_spiking_dashboard(self):
        viz = SpikingDynamicsVisualizer()
        for i in range(10):
            viz.record_snapshot(self._make_snapshot(ts=float(i)))
        result = viz.create_spiking_dashboard()
        assert result is not None

    def test_export_data(self):
        viz = SpikingDynamicsVisualizer()
        viz.record_snapshot(self._make_snapshot(ts=1.0))
        data = viz.export_data()
        assert isinstance(data, dict)
        assert "snapshots" in data
        assert "meta" in data
        assert len(data["snapshots"]) == 1
        assert "spike_count" in data["snapshots"][0]
        assert "mean_membrane" in data["snapshots"][0]

    def test_export_data_empty(self):
        viz = SpikingDynamicsVisualizer()
        data = viz.export_data()
        assert isinstance(data, dict)
        assert len(data["snapshots"]) == 0
