"""Tests for Qwen metrics visualization."""

import pytest
import numpy as np

from t4dm.visualization.qwen_metrics import (
    QwenMetricsVisualizer,
    QwenSnapshot,
)


class TestQwenSnapshot:
    """Test QwenSnapshot dataclass."""

    def test_create_snapshot(self):
        snapshot = QwenSnapshot(
            hidden_state_norm=1.5,
            projection_norm=0.8,
            lora_weight_norms={"q_proj": 0.3, "v_proj": 0.25},
            residual_blend_alpha=0.6,
            block_index=2,
            timestamp=1.0,
        )
        assert snapshot.hidden_state_norm == 1.5
        assert "q_proj" in snapshot.lora_weight_norms

    def test_defaults(self):
        snapshot = QwenSnapshot(
            hidden_state_norm=1.0,
            projection_norm=0.5,
        )
        assert snapshot.lora_weight_norms == {}
        assert snapshot.residual_blend_alpha == 0.5
        assert snapshot.block_index == 0


class TestQwenMetricsVisualizer:
    """Test QwenMetricsVisualizer class."""

    def _make_snapshot(self, ts=0.0):
        return QwenSnapshot(
            hidden_state_norm=1.0 + ts * 0.01,
            projection_norm=0.5 + ts * 0.005,
            lora_weight_norms={"q_proj": 0.3, "v_proj": 0.25, "k_proj": 0.28},
            residual_blend_alpha=0.5 + ts * 0.01,
            block_index=0,
            timestamp=ts,
        )

    def test_initialization(self):
        viz = QwenMetricsVisualizer()
        assert viz.window_size == 500

    def test_record_snapshot(self):
        viz = QwenMetricsVisualizer()
        viz.record_snapshot(self._make_snapshot())
        assert len(viz._snapshots) == 1

    def test_window_size_limit(self):
        viz = QwenMetricsVisualizer(window_size=5)
        for i in range(10):
            viz.record_snapshot(self._make_snapshot(ts=float(i)))
        assert len(viz._snapshots) == 5

    def test_plot_hidden_state_norms(self):
        viz = QwenMetricsVisualizer()
        for i in range(10):
            viz.record_snapshot(self._make_snapshot(ts=float(i)))
        result = viz.plot_hidden_state_norms()
        assert result is not None

    def test_plot_hidden_state_norms_empty(self):
        viz = QwenMetricsVisualizer()
        result = viz.plot_hidden_state_norms()
        assert result is not None

    def test_plot_lora_weight_norms(self):
        viz = QwenMetricsVisualizer()
        viz.record_snapshot(self._make_snapshot())
        result = viz.plot_lora_weight_norms()
        assert result is not None

    def test_plot_lora_weight_norms_empty(self):
        viz = QwenMetricsVisualizer()
        result = viz.plot_lora_weight_norms()
        assert result is not None

    def test_plot_lora_weight_norms_no_lora_data(self):
        viz = QwenMetricsVisualizer()
        viz.record_snapshot(QwenSnapshot(
            hidden_state_norm=1.0, projection_norm=0.5,
        ))
        result = viz.plot_lora_weight_norms()
        assert result is not None

    def test_plot_residual_blend(self):
        viz = QwenMetricsVisualizer()
        for i in range(10):
            viz.record_snapshot(self._make_snapshot(ts=float(i)))
        result = viz.plot_residual_blend()
        assert result is not None

    def test_plot_residual_blend_empty(self):
        viz = QwenMetricsVisualizer()
        result = viz.plot_residual_blend()
        assert result is not None

    def test_export_data(self):
        viz = QwenMetricsVisualizer()
        viz.record_snapshot(self._make_snapshot(ts=1.0))
        data = viz.export_data()
        assert isinstance(data, dict)
        assert "snapshots" in data
        assert "meta" in data
        assert len(data["snapshots"]) == 1
        assert "hidden_state_norm" in data["snapshots"][0]
        assert "lora_weight_norms" in data["snapshots"][0]

    def test_export_data_empty(self):
        viz = QwenMetricsVisualizer()
        data = viz.export_data()
        assert isinstance(data, dict)
        assert len(data["snapshots"]) == 0
