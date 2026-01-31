"""
Activation heatmap visualization for World Weaver.

Visualizes activation patterns across different memory types
(episodic, semantic, procedural) and neuromodulatory systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActivationSnapshot:
    """Snapshot of system activations at a point in time."""

    timestamp: datetime
    episodic_activations: np.ndarray  # (N,) activation values
    semantic_activations: np.ndarray  # (M,) activation values
    neuromod_state: dict[str, float]  # DA/NE/ACh/5-HT/GABA values
    memory_ids: list[str]  # IDs for activated memories


class ActivationHeatmap:
    """
    Tracks and visualizes activation patterns across memory types.

    Maintains a sliding window of activation snapshots and provides
    methods to visualize temporal dynamics.
    """

    def __init__(
        self,
        window_size: int = 100,
        max_memories_tracked: int = 50
    ):
        """
        Initialize activation heatmap tracker.

        Args:
            window_size: Number of timesteps to track
            max_memories_tracked: Maximum memories to show in heatmap
        """
        self.window_size = window_size
        self.max_memories_tracked = max_memories_tracked
        self._snapshots: list[ActivationSnapshot] = []

    def record_snapshot(
        self,
        episodic_activations: dict[str, float],
        semantic_activations: dict[str, float],
        neuromod_state: dict[str, float]
    ) -> None:
        """
        Record activation snapshot at current time.

        Args:
            episodic_activations: Memory ID -> activation value
            semantic_activations: Entity ID -> activation value
            neuromod_state: Neuromodulator name -> value
        """
        # Convert to arrays
        ep_ids = list(episodic_activations.keys())
        ep_vals = np.array([episodic_activations[k] for k in ep_ids])

        sem_ids = list(semantic_activations.keys())
        sem_vals = np.array([semantic_activations[k] for k in sem_ids])

        snapshot = ActivationSnapshot(
            timestamp=datetime.now(),
            episodic_activations=ep_vals,
            semantic_activations=sem_vals,
            neuromod_state=neuromod_state,
            memory_ids=ep_ids + sem_ids
        )

        self._snapshots.append(snapshot)

        # Maintain window
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

    def get_activation_matrix(
        self,
        memory_type: str = "episodic"
    ) -> tuple[np.ndarray, list[str], list[datetime]]:
        """
        Get activation matrix for heatmap plotting.

        Args:
            memory_type: "episodic" or "semantic"

        Returns:
            Tuple of (matrix, memory_ids, timestamps)
            Matrix shape: (time_steps, n_memories)
        """
        if not self._snapshots:
            return np.array([[]]), [], []

        # Collect all unique memory IDs
        all_ids = set()
        for snap in self._snapshots:
            all_ids.update(snap.memory_ids)

        # Limit to most recently active
        all_ids = list(all_ids)[:self.max_memories_tracked]

        # Build matrix
        matrix = []
        timestamps = []

        for snap in self._snapshots:
            row = np.zeros(len(all_ids))

            if memory_type == "episodic":
                activations = dict(zip(snap.memory_ids[:len(snap.episodic_activations)],
                                     snap.episodic_activations))
            else:
                activations = dict(zip(snap.memory_ids[len(snap.episodic_activations):],
                                     snap.semantic_activations))

            for i, mem_id in enumerate(all_ids):
                if mem_id in activations:
                    row[i] = activations[mem_id]

            matrix.append(row)
            timestamps.append(snap.timestamp)

        return np.array(matrix), all_ids, timestamps

    def get_neuromod_timeline(self) -> tuple[np.ndarray, list[str], list[datetime]]:
        """
        Get neuromodulator timeline for plotting.

        Returns:
            Tuple of (matrix, modulator_names, timestamps)
            Matrix shape: (time_steps, n_modulators)
        """
        if not self._snapshots:
            return np.array([[]]), [], []

        # Get all neuromodulator names
        mod_names = list(self._snapshots[0].neuromod_state.keys())

        # Build matrix
        matrix = []
        timestamps = []

        for snap in self._snapshots:
            row = [snap.neuromod_state.get(name, 0.0) for name in mod_names]
            matrix.append(row)
            timestamps.append(snap.timestamp)

        return np.array(matrix), mod_names, timestamps


def plot_activation_heatmap(
    tracker: ActivationHeatmap,
    memory_type: str = "episodic",
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot activation heatmap.

    Args:
        tracker: ActivationHeatmap instance
        memory_type: "episodic" or "semantic"
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    matrix, memory_ids, timestamps = tracker.get_activation_matrix(memory_type)

    if matrix.size == 0:
        logger.warning("No activation data to plot")
        return

    if interactive:
        try:
            import plotly.graph_objects as go

            fig = go.Figure(data=go.Heatmap(
                z=matrix.T,
                x=[t.strftime("%H:%M:%S") for t in timestamps],
                y=[f"{mid[:8]}..." for mid in memory_ids],
                colorscale="Viridis",
                colorbar=dict(title="Activation")
            ))

            fig.update_layout(
                title=f"{memory_type.capitalize()} Memory Activation Heatmap",
                xaxis_title="Time",
                yaxis_title="Memory ID",
                height=max(400, len(memory_ids) * 20)
            )

            if save_path:
                fig.write_html(str(save_path))
            else:
                fig.show()

        except ImportError:
            logger.warning("Plotly not available, falling back to matplotlib")
            interactive = False

    if not interactive:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, max(6, len(memory_ids) * 0.3)))

            im = ax.imshow(
                matrix.T,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest"
            )

            # Format x-axis (time)
            n_ticks = min(10, len(timestamps))
            tick_indices = np.linspace(0, len(timestamps) - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([timestamps[i].strftime("%H:%M:%S") for i in tick_indices],
                              rotation=45, ha="right")

            # Format y-axis (memory IDs)
            ax.set_yticks(range(len(memory_ids)))
            ax.set_yticklabels([f"{mid[:8]}..." for mid in memory_ids])

            ax.set_xlabel("Time")
            ax.set_ylabel("Memory ID")
            ax.set_title(f"{memory_type.capitalize()} Memory Activation Heatmap")

            plt.colorbar(im, ax=ax, label="Activation")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)  # MEM-006 FIX: Close figure to prevent memory leak

        except ImportError:
            logger.error("Neither plotly nor matplotlib available for plotting")


def plot_activation_timeline(
    tracker: ActivationHeatmap,
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot neuromodulator activation timeline.

    Args:
        tracker: ActivationHeatmap instance
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    matrix, mod_names, timestamps = tracker.get_neuromod_timeline()

    if matrix.size == 0:
        logger.warning("No neuromodulator data to plot")
        return

    if interactive:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            for i, name in enumerate(mod_names):
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=matrix[:, i],
                    mode="lines",
                    name=name,
                    line=dict(width=2)
                ))

            fig.update_layout(
                title="Neuromodulator Timeline",
                xaxis_title="Time",
                yaxis_title="Level",
                hovermode="x unified",
                height=500
            )

            if save_path:
                fig.write_html(str(save_path))
            else:
                fig.show()

        except ImportError:
            logger.warning("Plotly not available, falling back to matplotlib")
            interactive = False

    if not interactive:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))

            for i, name in enumerate(mod_names):
                ax.plot(timestamps, matrix[:, i], label=name, linewidth=2)

            ax.set_xlabel("Time")
            ax.set_ylabel("Level")
            ax.set_title("Neuromodulator Timeline")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)  # MEM-006 FIX: Close figure to prevent memory leak

        except ImportError:
            logger.error("Neither plotly nor matplotlib available for plotting")


__all__ = [
    "ActivationHeatmap",
    "ActivationSnapshot",
    "plot_activation_heatmap",
    "plot_activation_timeline",
]
