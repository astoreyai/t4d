"""
Per-layer neuromodulator injection visualization for spiking cortical blocks.

Visualizes how DA, NE, ACh, and 5-HT are injected into each cortical block,
following the laminar mapping from neuromod_bus.py:
  DA  -> L2/3 + L5 (stages 3, 4)
  NE  -> L5 (stages 2, 6)
  ACh -> L1/L4 (stage 1)
  5-HT -> L5/6 (stage 5)

Author: Claude Opus 4.5
Date: 2026-02-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NT_NAMES = ["DA", "NE", "ACh", "5-HT"]
NT_COLORS = {"DA": "#e74c3c", "NE": "#3498db", "ACh": "#2ecc71", "5-HT": "#9b59b6"}

# Laminar mapping: which stages each NT targets
NT_LAYER_MAPPING = {
    "DA": [2, 3],   # stages 3, 4 (0-indexed: 2, 3)
    "NE": [1, 5],   # stages 2, 6
    "ACh": [0],      # stage 1
    "5-HT": [4],     # stage 5
}


@dataclass
class NeuromodLayerSnapshot:
    """Snapshot of per-layer neuromodulator injection magnitudes."""

    da_per_layer: list[float]
    ne_per_layer: list[float]
    ach_per_layer: list[float]
    sht_per_layer: list[float]
    num_blocks: int
    timestamp: float


class NeuromodLayerVisualizer:
    """
    Visualizes per-layer neuromodulator injection into spiking cortical blocks.

    Records snapshots over time and provides heatmap, bar chart, and
    evolution plots for the four neuromodulators across block layers.
    """

    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self._snapshots: list[NeuromodLayerSnapshot] = []

    def record_snapshot(self, snapshot: NeuromodLayerSnapshot) -> None:
        """Record a neuromodulator layer snapshot."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_history:
            self._snapshots.pop(0)

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def _latest_matrix(self) -> np.ndarray | None:
        """Get latest snapshot as 4 x num_blocks matrix (DA, NE, ACh, 5-HT)."""
        if not self._snapshots:
            return None
        s = self._snapshots[-1]
        return np.array([
            s.da_per_layer,
            s.ne_per_layer,
            s.ach_per_layer,
            s.sht_per_layer,
        ])

    def plot_nt_heatmap(self, ax=None) -> Any:
        """
        4 x N heatmap (NT x block layer) showing injection magnitudes.

        Rows: DA, NE, ACh, 5-HT
        Columns: Block 0 .. Block N-1
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        matrix = self._latest_matrix()
        if matrix is None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return ax

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        num_blocks = matrix.shape[1]
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_yticks(range(4))
        ax.set_yticklabels(NT_NAMES)
        ax.set_xticks(range(num_blocks))
        ax.set_xticklabels([f"Block {i}" for i in range(num_blocks)])
        ax.set_title("Neuromodulator Injection Heatmap")
        plt.colorbar(im, ax=ax, label="Injection Magnitude")

        # Annotate cells
        for i in range(4):
            for j in range(num_blocks):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

        return ax

    def plot_nt_per_block(self, ax=None) -> Any:
        """Grouped bar chart: per block showing 4 NT values."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        matrix = self._latest_matrix()
        if matrix is None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return ax

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        num_blocks = matrix.shape[1]
        x = np.arange(num_blocks)
        width = 0.2

        for i, nt_name in enumerate(NT_NAMES):
            ax.bar(x + i * width, matrix[i], width, label=nt_name,
                   color=NT_COLORS[nt_name], edgecolor="black", linewidth=0.5)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f"Block {j}" for j in range(num_blocks)])
        ax.set_ylabel("Injection Magnitude")
        ax.set_title("Neuromodulator Levels per Block")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        return ax

    def plot_nt_evolution(self, axes=None) -> Any:
        """
        4 subplots, one per NT, showing per-layer traces over time.

        Each subplot has one line per block layer.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if not self._snapshots:
            if axes is None:
                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            for ax_item, nt in zip(axes, NT_NAMES):
                ax_item.text(0.5, 0.5, "No data", ha="center", va="center")
                ax_item.set_ylabel(nt)
            return axes

        if axes is None:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        timestamps = [s.timestamp for s in self._snapshots]
        num_blocks = self._snapshots[0].num_blocks

        # Build time series per NT
        nt_keys = ["da_per_layer", "ne_per_layer", "ach_per_layer", "sht_per_layer"]

        for ax_item, nt_name, key in zip(axes, NT_NAMES, nt_keys):
            for block_idx in range(num_blocks):
                values = [getattr(s, key)[block_idx] for s in self._snapshots]
                ax_item.plot(timestamps, values, label=f"Block {block_idx}", linewidth=1.5)
            ax_item.set_ylabel(nt_name)
            ax_item.legend(loc="upper right", fontsize=7, ncol=num_blocks)
            ax_item.grid(True, alpha=0.3)

        axes[0].set_title("Neuromodulator Evolution per Layer")
        axes[-1].set_xlabel("Time")

        return axes

    def export_data(self) -> dict:
        """Export all recorded data as JSON-serializable dict."""
        return {
            "snapshot_count": len(self._snapshots),
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "num_blocks": s.num_blocks,
                    "da_per_layer": s.da_per_layer,
                    "ne_per_layer": s.ne_per_layer,
                    "ach_per_layer": s.ach_per_layer,
                    "sht_per_layer": s.sht_per_layer,
                }
                for s in self._snapshots
            ],
            "nt_names": NT_NAMES,
            "layer_mapping": NT_LAYER_MAPPING,
        }


__all__ = [
    "NeuromodLayerSnapshot",
    "NeuromodLayerVisualizer",
    "NT_NAMES",
    "NT_COLORS",
    "NT_LAYER_MAPPING",
]
