"""
Neuromodulator state visualization for T4DM.

Provides dashboard and timeline visualizations of neuromodulatory systems:
- Dopamine (DA): Reward prediction error
- Norepinephrine (NE): Arousal and novelty
- Acetylcholine (ACh): Encoding/retrieval mode
- Serotonin (5-HT): Long-term value tracking
- GABA: Inhibitory dynamics and sparsity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NeuromodulatorSnapshot:
    """Snapshot of neuromodulator states."""

    timestamp: datetime
    dopamine_rpe: float
    norepinephrine_gain: float
    acetylcholine_mode: str  # "encoding", "balanced", "retrieval"
    serotonin_mood: float
    gaba_sparsity: float


class NeuromodulatorDashboard:
    """
    Tracks and visualizes neuromodulator dynamics over time.

    Maintains a history of neuromodulator states for analysis
    of system dynamics and emergent behaviors.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize neuromodulator dashboard.

        Args:
            window_size: Number of timesteps to track
        """
        self.window_size = window_size
        self._snapshots: list[NeuromodulatorSnapshot] = []

    def record_state(
        self,
        dopamine_rpe: float,
        norepinephrine_gain: float,
        acetylcholine_mode: str,
        serotonin_mood: float,
        gaba_sparsity: float
    ) -> None:
        """
        Record current neuromodulator state.

        Args:
            dopamine_rpe: Reward prediction error [-1, 1]
            norepinephrine_gain: Arousal gain [0.5, 2.0]
            acetylcholine_mode: Mode ("encoding", "balanced", "retrieval")
            serotonin_mood: Mood level [0, 1]
            gaba_sparsity: Sparsity level [0, 1]
        """
        snapshot = NeuromodulatorSnapshot(
            timestamp=datetime.now(),
            dopamine_rpe=dopamine_rpe,
            norepinephrine_gain=norepinephrine_gain,
            acetylcholine_mode=acetylcholine_mode,
            serotonin_mood=serotonin_mood,
            gaba_sparsity=gaba_sparsity
        )

        self._snapshots.append(snapshot)

        # Maintain window
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

    def get_trace_data(self) -> dict[str, tuple]:
        """
        Get time series data for all modulators.

        Returns:
            Dict mapping modulator name -> (timestamps, values)
        """
        if not self._snapshots:
            return {}

        timestamps = [s.timestamp for s in self._snapshots]

        return {
            "dopamine_rpe": (timestamps, [s.dopamine_rpe for s in self._snapshots]),
            "norepinephrine_gain": (timestamps, [s.norepinephrine_gain for s in self._snapshots]),
            "serotonin_mood": (timestamps, [s.serotonin_mood for s in self._snapshots]),
            "gaba_sparsity": (timestamps, [s.gaba_sparsity for s in self._snapshots])
        }

    def get_mode_distribution(self) -> dict[str, int]:
        """
        Get distribution of ACh modes.

        Returns:
            Dict mapping mode -> count
        """
        mode_counts = {}
        for snap in self._snapshots:
            mode = snap.acetylcholine_mode
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        return mode_counts

    def get_current_state(self) -> NeuromodulatorSnapshot | None:
        """Get most recent neuromodulator state."""
        return self._snapshots[-1] if self._snapshots else None

    def get_statistics(self) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for all modulators.

        Returns:
            Dict mapping modulator -> {mean, std, min, max}
        """
        if not self._snapshots:
            return {}

        da_vals = [s.dopamine_rpe for s in self._snapshots]
        ne_vals = [s.norepinephrine_gain for s in self._snapshots]
        ht_vals = [s.serotonin_mood for s in self._snapshots]
        gaba_vals = [s.gaba_sparsity for s in self._snapshots]

        def stats(vals):
            return {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals))
            }

        return {
            "dopamine": stats(da_vals),
            "norepinephrine": stats(ne_vals),
            "serotonin": stats(ht_vals),
            "gaba": stats(gaba_vals)
        }


def plot_neuromodulator_traces(
    dashboard: NeuromodulatorDashboard,
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot time series of all neuromodulators.

    Args:
        dashboard: NeuromodulatorDashboard instance
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    trace_data = dashboard.get_trace_data()

    if not trace_data:
        logger.warning("No neuromodulator data to plot")
        return

    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Create subplots for each modulator
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=("Dopamine RPE", "Norepinephrine Gain",
                              "Serotonin Mood", "GABA Sparsity"),
                vertical_spacing=0.08
            )

            colors = {
                "dopamine_rpe": "blue",
                "norepinephrine_gain": "red",
                "serotonin_mood": "green",
                "gaba_sparsity": "purple"
            }

            for idx, (name, (timestamps, values)) in enumerate(trace_data.items(), 1):
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=values,
                        mode="lines",
                        name=name,
                        line=dict(color=colors.get(name, "black"), width=2)
                    ),
                    row=idx, col=1
                )

            fig.update_xaxes(title_text="Time", row=4, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="Gain", row=2, col=1)
            fig.update_yaxes(title_text="Mood", row=3, col=1)
            fig.update_yaxes(title_text="Sparsity", row=4, col=1)

            fig.update_layout(
                title="Neuromodulator Timeline",
                showlegend=False,
                height=800
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

            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

            colors = {
                "dopamine_rpe": "blue",
                "norepinephrine_gain": "red",
                "serotonin_mood": "green",
                "gaba_sparsity": "purple"
            }

            titles = {
                "dopamine_rpe": "Dopamine RPE",
                "norepinephrine_gain": "Norepinephrine Gain",
                "serotonin_mood": "Serotonin Mood",
                "gaba_sparsity": "GABA Sparsity"
            }

            for idx, (name, (timestamps, values)) in enumerate(trace_data.items()):
                ax = axes[idx]
                ax.plot(timestamps, values, color=colors.get(name, "black"), linewidth=2)
                ax.set_ylabel(titles.get(name, name))
                ax.grid(True, alpha=0.3)

                # Add reference lines
                if name == "dopamine_rpe":
                    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
                elif name == "norepinephrine_gain":
                    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)

            axes[-1].set_xlabel("Time")
            fig.suptitle("Neuromodulator Timeline", fontsize=14)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)  # MEM-006 FIX: Close figure to prevent memory leak

        except ImportError:
            logger.error("Neither plotly nor matplotlib available for plotting")


def plot_neuromodulator_radar(
    dashboard: NeuromodulatorDashboard,
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot radar chart of current neuromodulator state.

    Args:
        dashboard: NeuromodulatorDashboard instance
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    current = dashboard.get_current_state()

    if not current:
        logger.warning("No current state to plot")
        return

    # Normalize values to [0, 1] for radar chart
    values = {
        "DA RPE": (current.dopamine_rpe + 1) / 2,  # [-1, 1] -> [0, 1]
        "NE Gain": (current.norepinephrine_gain - 0.5) / 1.5,  # [0.5, 2.0] -> [0, 1]
        "5-HT Mood": current.serotonin_mood,  # Already [0, 1]
        "GABA Sparsity": current.gaba_sparsity,  # Already [0, 1]
        "ACh Encoding": 1.0 if current.acetylcholine_mode == "encoding" else 0.0
    }

    if interactive:
        try:
            import plotly.graph_objects as go

            categories = list(values.keys())
            vals = list(values.values())

            fig = go.Figure(data=go.Scatterpolar(
                r=vals + [vals[0]],  # Close the loop
                theta=categories + [categories[0]],
                fill="toself",
                line=dict(color="blue", width=2)
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Current Neuromodulator State",
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

            categories = list(values.keys())
            vals = list(values.values())

            # Number of variables
            N = len(categories)

            # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            vals += vals[:1]  # Close the loop
            angles += angles[:1]

            # Initialize plot
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

            # Draw one axis per variable and add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            # Draw ylabels
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])

            # Plot data
            ax.plot(angles, vals, linewidth=2, linestyle="solid", color="blue")
            ax.fill(angles, vals, alpha=0.3, color="blue")

            ax.set_title("Current Neuromodulator State", pad=20)
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
    "NeuromodulatorDashboard",
    "NeuromodulatorSnapshot",
    "plot_neuromodulator_radar",
    "plot_neuromodulator_traces",
]
