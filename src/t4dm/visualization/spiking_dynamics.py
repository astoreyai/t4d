"""
Spiking Cortical Block Dynamics Visualization for T4DM.

Visualizes spiking neural network internals:
- Spike raster (dot plot of spike times per neuron)
- Membrane potential heatmap
- Thalamic gate state
- Apical modulation (prediction error distribution)
- Composite dashboard

Author: Claude Opus 4.5
Date: 2026-02-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpikingSnapshot:
    """Snapshot of spiking cortical block state."""

    membrane_potentials: np.ndarray  # (num_neurons,)
    spike_mask: np.ndarray  # (num_neurons,) boolean
    thalamic_gate: np.ndarray  # (dim,)
    apical_error: np.ndarray  # (dim,)
    block_index: int
    timestamp: float


class SpikingDynamicsVisualizer:
    """
    Visualizes spiking cortical block internals.

    Tracks membrane potentials, spike patterns, thalamic gating,
    and apical prediction errors over time.
    """

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self._snapshots: list[SpikingSnapshot] = []
        logger.info("SpikingDynamicsVisualizer initialized")

    def record_snapshot(self, snapshot: SpikingSnapshot) -> None:
        """Append a snapshot to history."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

    def plot_spike_raster(self, ax: Any = None) -> Any:
        """Dot plot of spike times per neuron."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        spike_times = []
        spike_neurons = []
        for s in self._snapshots:
            neuron_ids = np.where(s.spike_mask)[0]
            for nid in neuron_ids:
                spike_times.append(s.timestamp)
                spike_neurons.append(nid)

        if spike_times:
            ax.scatter(spike_times, spike_neurons, s=1, c="black", marker="|")

        ax.set_xlabel("Time")
        ax.set_ylabel("Neuron Index")
        ax.set_title("Spike Raster")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_membrane_potentials(self, ax: Any = None) -> Any:
        """Heatmap of membrane voltages across neurons over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        # Build matrix: (num_neurons, num_timesteps)
        potentials = np.column_stack([s.membrane_potentials for s in self._snapshots])
        times = [s.timestamp for s in self._snapshots]

        im = ax.imshow(
            potentials, aspect="auto", cmap="RdBu_r",
            extent=[times[0], times[-1], potentials.shape[0] - 0.5, -0.5],
            interpolation="nearest",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Neuron Index")
        ax.set_title("Membrane Potentials")
        plt.colorbar(im, ax=ax, label="Voltage")

        return ax

    def plot_thalamic_gate_state(self, ax: Any = None) -> Any:
        """Bar chart of gate values per dimension (latest snapshot)."""
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

        gate = self._snapshots[-1].thalamic_gate
        dims = np.arange(len(gate))

        ax.bar(dims, gate, color="mediumpurple", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Gate Value")
        ax.set_title(f"Thalamic Gate State (block {self._snapshots[-1].block_index})")
        ax.grid(True, alpha=0.3, axis="y")

        return ax

    def plot_apical_modulation(self, ax: Any = None) -> Any:
        """Prediction error distribution from latest snapshot."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        error = self._snapshots[-1].apical_error

        ax.hist(error, bins=30, color="coral", edgecolor="black", alpha=0.8)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5)
        ax.axvline(x=float(np.mean(error)), color="red", linestyle="--",
                   label=f"Mean: {np.mean(error):.3f}")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Count")
        ax.set_title("Apical Modulation (Prediction Error)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def create_spiking_dashboard(self, fig: Any = None) -> Any:
        """Composite figure with all spiking visualizations."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if fig is None:
            fig = plt.figure(figsize=(16, 12))

        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax_raster = fig.add_subplot(gs[0, 0])
        self.plot_spike_raster(ax=ax_raster)

        ax_membrane = fig.add_subplot(gs[0, 1])
        self.plot_membrane_potentials(ax=ax_membrane)

        ax_gate = fig.add_subplot(gs[1, 0])
        self.plot_thalamic_gate_state(ax=ax_gate)

        ax_apical = fig.add_subplot(gs[1, 1])
        self.plot_apical_modulation(ax=ax_apical)

        block_idx = self._snapshots[-1].block_index if self._snapshots else "?"
        fig.suptitle(
            f"Spiking Cortical Block {block_idx} Dashboard",
            fontsize=14, fontweight="bold",
        )

        return fig

    def export_data(self) -> dict:
        """Export spiking dynamics data for external analysis."""
        snapshots_data = []
        for s in self._snapshots:
            spike_count = int(np.sum(s.spike_mask))
            snapshots_data.append({
                "timestamp": s.timestamp,
                "block_index": s.block_index,
                "spike_count": spike_count,
                "spike_rate": spike_count / len(s.spike_mask) if len(s.spike_mask) > 0 else 0.0,
                "mean_membrane": float(np.mean(s.membrane_potentials)),
                "mean_thalamic_gate": float(np.mean(s.thalamic_gate)),
                "mean_apical_error": float(np.mean(s.apical_error)),
                "std_apical_error": float(np.std(s.apical_error)),
            })
        return {
            "snapshots": snapshots_data,
            "meta": {
                "window_size": self.window_size,
                "total_snapshots": len(self._snapshots),
            },
        }


__all__ = [
    "SpikingSnapshot",
    "SpikingDynamicsVisualizer",
]
