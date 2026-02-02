"""Tests for kappa gradient visualization."""

import pytest
import numpy as np

from t4dm.visualization.kappa_gradient import (
    KappaGradientVisualizer,
    KappaSnapshot,
    KAPPA_BANDS,
    KAPPA_COLORS,
)


class TestKappaSnapshot:
    """Test KappaSnapshot dataclass."""

    def test_create_snapshot(self):
        snapshot = KappaSnapshot(
            kappa_values=[0.1, 0.3, 0.5, 0.9],
            item_count=4,
            timestamp=1.0,
            level_counts={0: 2, 1: 1},
        )
        assert snapshot.item_count == 4
        assert len(snapshot.kappa_values) == 4

    def test_default_level_counts(self):
        snapshot = KappaSnapshot(
            kappa_values=[0.5], item_count=1, timestamp=0.0,
        )
        assert snapshot.level_counts == {}


class TestKappaGradientVisualizer:
    """Test KappaGradientVisualizer class."""

    def _make_snapshot(self, kappa_values, ts=0.0, levels=None):
        return KappaSnapshot(
            kappa_values=kappa_values,
            item_count=len(kappa_values),
            timestamp=ts,
            level_counts=levels or {},
        )

    def test_initialization(self):
        viz = KappaGradientVisualizer()
        assert viz.window_size == 500
        assert len(viz._snapshots) == 0

    def test_record_snapshot(self):
        viz = KappaGradientVisualizer()
        viz.record_snapshot(self._make_snapshot([0.1, 0.5, 0.9]))
        assert len(viz._snapshots) == 1

    def test_window_size_limit(self):
        viz = KappaGradientVisualizer(window_size=5)
        for i in range(10):
            viz.record_snapshot(self._make_snapshot([0.5], ts=float(i)))
        assert len(viz._snapshots) == 5

    def test_plot_kappa_distribution(self):
        viz = KappaGradientVisualizer()
        viz.record_snapshot(self._make_snapshot(
            [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.9, 0.95]
        ))
        result = viz.plot_kappa_distribution()
        assert result is not None

    def test_plot_kappa_distribution_empty(self):
        viz = KappaGradientVisualizer()
        result = viz.plot_kappa_distribution()
        assert result is not None

    def test_plot_kappa_flow(self):
        viz = KappaGradientVisualizer()
        for i in range(10):
            viz.record_snapshot(self._make_snapshot(
                [0.05 + i * 0.05, 0.3 + i * 0.02, 0.7, 0.95],
                ts=float(i),
            ))
        result = viz.plot_kappa_flow()
        assert result is not None

    def test_plot_kappa_flow_empty(self):
        viz = KappaGradientVisualizer()
        result = viz.plot_kappa_flow()
        assert result is not None

    def test_plot_level_distribution(self):
        viz = KappaGradientVisualizer()
        viz.record_snapshot(self._make_snapshot(
            [0.5], ts=0.0, levels={0: 5, 1: 3, 2: 1},
        ))
        result = viz.plot_level_distribution()
        assert result is not None

    def test_plot_level_distribution_empty(self):
        viz = KappaGradientVisualizer()
        result = viz.plot_level_distribution()
        assert result is not None

    def test_export_data(self):
        viz = KappaGradientVisualizer()
        viz.record_snapshot(self._make_snapshot([0.1, 0.5, 0.9], ts=1.0))
        data = viz.export_data()
        assert isinstance(data, dict)
        assert "snapshots" in data
        assert "bands" in data
        assert "meta" in data
        assert len(data["snapshots"]) == 1
        assert "band_counts" in data["snapshots"][0]

    def test_export_data_empty(self):
        viz = KappaGradientVisualizer()
        data = viz.export_data()
        assert isinstance(data, dict)
        assert len(data["snapshots"]) == 0


class TestKappaBands:
    """Test kappa band constants."""

    def test_bands_defined(self):
        assert "episodic" in KAPPA_BANDS
        assert "replayed" in KAPPA_BANDS
        assert "transitional" in KAPPA_BANDS
        assert "semantic" in KAPPA_BANDS

    def test_colors_defined(self):
        assert len(KAPPA_COLORS) == len(KAPPA_BANDS)
