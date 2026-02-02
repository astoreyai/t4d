"""Tests for oscillator_injection visualization module."""

import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np

from t4dm.visualization.oscillator_injection import (
    OscillatorSnapshot,
    OscillatorInjectionVisualizer,
    OSC_NAMES,
    OSC_COLORS,
)


def _make_snapshot(timestamp: float = 0.0) -> OscillatorSnapshot:
    return OscillatorSnapshot(
        theta_phase=np.random.uniform(0, 2 * np.pi),
        gamma_phase=np.random.uniform(0, 2 * np.pi),
        delta_phase=np.random.uniform(0, 2 * np.pi),
        bias_values=np.random.randn(64),
        timestamp=timestamp,
    )


class TestOscillatorSnapshot:
    def test_creation(self):
        s = _make_snapshot()
        assert 0 <= s.theta_phase <= 2 * np.pi
        assert len(s.bias_values) == 64

    def test_constants(self):
        assert len(OSC_NAMES) == 3
        assert "theta" in OSC_COLORS


class TestOscillatorInjectionVisualizer:
    def test_record_snapshot(self):
        viz = OscillatorInjectionVisualizer()
        viz.record_snapshot(_make_snapshot())
        assert viz.snapshot_count == 1

    def test_max_history(self):
        viz = OscillatorInjectionVisualizer(max_history=3)
        for i in range(5):
            viz.record_snapshot(_make_snapshot(timestamp=float(i)))
        assert viz.snapshot_count == 3

    def test_plot_phase_polar_no_data(self):
        viz = OscillatorInjectionVisualizer()
        ax = viz.plot_phase_polar()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_phase_polar_with_data(self):
        viz = OscillatorInjectionVisualizer()
        viz.record_snapshot(_make_snapshot())
        ax = viz.plot_phase_polar()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_bias_timeline_no_data(self):
        viz = OscillatorInjectionVisualizer()
        ax = viz.plot_bias_timeline()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_bias_timeline_with_data(self):
        viz = OscillatorInjectionVisualizer()
        for i in range(10):
            viz.record_snapshot(_make_snapshot(timestamp=float(i)))
        ax = viz.plot_bias_timeline()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_phase_coupling_no_data(self):
        viz = OscillatorInjectionVisualizer()
        ax = viz.plot_phase_coupling()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_phase_coupling_with_data(self):
        viz = OscillatorInjectionVisualizer()
        for i in range(20):
            viz.record_snapshot(_make_snapshot(timestamp=float(i)))
        ax = viz.plot_phase_coupling()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_export_data_empty(self):
        viz = OscillatorInjectionVisualizer()
        data = viz.export_data()
        assert data["snapshot_count"] == 0
        assert data["oscillator_names"] == OSC_NAMES

    def test_export_data_with_snapshots(self):
        viz = OscillatorInjectionVisualizer()
        viz.record_snapshot(_make_snapshot(timestamp=1.0))
        data = viz.export_data()
        assert data["snapshot_count"] == 1
        assert "theta_phase" in data["snapshots"][0]
        assert "bias_mean" in data["snapshots"][0]
