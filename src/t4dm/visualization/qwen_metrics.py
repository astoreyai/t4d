"""
Qwen Adapter Metrics Visualization for T4DM.

Visualizes Qwen 3B + QLoRA adapter metrics:
- Hidden state norms over time
- LoRA weight norms per adapter (grouped bar chart)
- Residual blend alpha over time

Author: Claude Opus 4.5
Date: 2026-02-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QwenSnapshot:
    """Snapshot of Qwen adapter state."""

    hidden_state_norm: float
    projection_norm: float
    lora_weight_norms: dict[str, float] = field(default_factory=dict)
    residual_blend_alpha: float = 0.5
    block_index: int = 0
    timestamp: float = 0.0


class QwenMetricsVisualizer:
    """
    Visualizes Qwen adapter metrics.

    Tracks hidden state norms, LoRA weight norms, and residual
    blend factors over time for monitoring training health.
    """

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self._snapshots: list[QwenSnapshot] = []
        logger.info("QwenMetricsVisualizer initialized")

    def record_snapshot(self, snapshot: QwenSnapshot) -> None:
        """Append a snapshot to history."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

    def plot_hidden_state_norms(self, ax: Any = None) -> Any:
        """Line chart of hidden state and projection norms over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        times = [s.timestamp for s in self._snapshots]
        hidden_norms = [s.hidden_state_norm for s in self._snapshots]
        proj_norms = [s.projection_norm for s in self._snapshots]

        ax.plot(times, hidden_norms, color="steelblue", linewidth=1.5,
                label="Hidden state norm")
        ax.plot(times, proj_norms, color="darkorange", linewidth=1.5,
                label="Projection norm")

        ax.set_xlabel("Time")
        ax.set_ylabel("L2 Norm")
        ax.set_title("Hidden State and Projection Norms")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_lora_weight_norms(self, ax: Any = None) -> Any:
        """Grouped bar chart of LoRA weight norms per adapter."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        # Use latest snapshot
        norms = self._snapshots[-1].lora_weight_norms
        if not norms:
            ax.text(0.5, 0.5, "No LoRA data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        names = sorted(norms.keys())
        values = [norms[n] for n in names]

        ax.bar(names, values, color="mediumpurple", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Adapter")
        ax.set_ylabel("Weight Norm")
        ax.set_title("LoRA Weight Norms")
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate labels if many adapters
        if len(names) > 4:
            ax.tick_params(axis="x", rotation=45)

        return ax

    def plot_residual_blend(self, ax: Any = None) -> Any:
        """Alpha blending factor over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        times = [s.timestamp for s in self._snapshots]
        alphas = [s.residual_blend_alpha for s in self._snapshots]

        ax.plot(times, alphas, color="seagreen", linewidth=1.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Midpoint")
        ax.set_xlabel("Time")
        ax.set_ylabel("Blend Alpha")
        ax.set_title("Residual Blend Factor")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def export_data(self) -> dict:
        """Export Qwen metrics for external analysis."""
        return {
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "block_index": s.block_index,
                    "hidden_state_norm": s.hidden_state_norm,
                    "projection_norm": s.projection_norm,
                    "lora_weight_norms": s.lora_weight_norms,
                    "residual_blend_alpha": s.residual_blend_alpha,
                }
                for s in self._snapshots
            ],
            "meta": {
                "window_size": self.window_size,
                "total_snapshots": len(self._snapshots),
            },
        }


__all__ = [
    "QwenSnapshot",
    "QwenMetricsVisualizer",
]
