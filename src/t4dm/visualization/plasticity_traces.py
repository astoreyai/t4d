"""
Plasticity trace visualization for T4DM.

Visualizes synaptic weight changes over time including:
- BCM learning curves
- LTP/LTD distributions
- Homeostatic scaling events
- Weight trajectory plots
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeightUpdate:
    """Record of a synaptic weight update."""

    timestamp: datetime
    source_id: str
    target_id: str
    old_weight: float
    new_weight: float
    update_type: str  # "ltp", "ltd", "homeostatic"
    activation_level: float = 0.0


class PlasticityTracer:
    """
    Tracks and visualizes synaptic plasticity dynamics.

    Records weight changes from LTP/LTD and homeostatic mechanisms,
    enabling analysis of learning dynamics.
    """

    def __init__(self, max_updates: int = 10000):
        """
        Initialize plasticity tracer.

        Args:
            max_updates: Maximum weight updates to track
        """
        self.max_updates = max_updates
        self._updates: list[WeightUpdate] = []

    def record_update(
        self,
        source_id: str,
        target_id: str,
        old_weight: float,
        new_weight: float,
        update_type: str,
        activation_level: float = 0.0
    ) -> None:
        """
        Record a weight update event.

        Args:
            source_id: Presynaptic entity ID
            target_id: Postsynaptic entity ID
            old_weight: Weight before update
            new_weight: Weight after update
            update_type: Type of plasticity ("ltp", "ltd", "homeostatic")
            activation_level: Activation level that triggered update
        """
        update = WeightUpdate(
            timestamp=datetime.now(),
            source_id=source_id,
            target_id=target_id,
            old_weight=old_weight,
            new_weight=new_weight,
            update_type=update_type,
            activation_level=activation_level
        )

        self._updates.append(update)

        # Maintain size limit
        if len(self._updates) > self.max_updates:
            self._updates.pop(0)

    def get_weight_trajectory(
        self,
        source_id: str,
        target_id: str
    ) -> tuple[list[datetime], list[float]]:
        """
        Get weight trajectory for a specific synapse.

        Args:
            source_id: Presynaptic entity ID
            target_id: Postsynaptic entity ID

        Returns:
            Tuple of (timestamps, weights)
        """
        timestamps = []
        weights = []

        for update in self._updates:
            if update.source_id == source_id and update.target_id == target_id:
                timestamps.append(update.timestamp)
                weights.append(update.new_weight)

        return timestamps, weights

    def get_ltp_ltd_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get distribution of LTP and LTD magnitudes.

        Returns:
            Tuple of (ltp_magnitudes, ltd_magnitudes)
        """
        ltp_mags = []
        ltd_mags = []

        for update in self._updates:
            delta = update.new_weight - update.old_weight

            if update.update_type == "ltp" and delta > 0:
                ltp_mags.append(delta)
            elif update.update_type == "ltd" and delta < 0:
                ltd_mags.append(abs(delta))

        return np.array(ltp_mags), np.array(ltd_mags)

    def get_bcm_curve_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get data for BCM learning curve visualization.

        Returns:
            Tuple of (activation_levels, weight_changes)
        """
        activations = []
        changes = []

        for update in self._updates:
            if update.update_type in ("ltp", "ltd"):
                activations.append(update.activation_level)
                changes.append(update.new_weight - update.old_weight)

        return np.array(activations), np.array(changes)

    def get_update_counts_by_type(self) -> dict[str, int]:
        """
        Get counts of updates by type.

        Returns:
            Dict mapping update_type -> count
        """
        counts = {}
        for update in self._updates:
            counts[update.update_type] = counts.get(update.update_type, 0) + 1
        return counts

    def get_timeline_data(
        self,
        bin_size_minutes: int = 5
    ) -> tuple[list[datetime], dict[str, list[int]]]:
        """
        Get binned timeline of plasticity events.

        Args:
            bin_size_minutes: Size of time bins in minutes

        Returns:
            Tuple of (bin_timestamps, type_counts)
        """
        if not self._updates:
            return [], {}

        # Create time bins
        start_time = self._updates[0].timestamp
        end_time = self._updates[-1].timestamp
        duration = (end_time - start_time).total_seconds() / 60  # minutes

        n_bins = max(1, int(duration / bin_size_minutes))

        # Initialize bins
        from datetime import timedelta
        bin_times = [start_time + timedelta(minutes=i * bin_size_minutes)
                    for i in range(n_bins + 1)]

        update_types = list(set(u.update_type for u in self._updates))
        type_counts = {utype: [0] * n_bins for utype in update_types}

        # Fill bins
        for update in self._updates:
            elapsed = (update.timestamp - start_time).total_seconds() / 60
            bin_idx = min(int(elapsed / bin_size_minutes), n_bins - 1)
            type_counts[update.update_type][bin_idx] += 1

        return bin_times[:-1], type_counts


def plot_bcm_curve(
    tracer: PlasticityTracer,
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot BCM learning curve (weight change vs activation).

    Args:
        tracer: PlasticityTracer instance
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    activations, changes = tracer.get_bcm_curve_data()

    if len(activations) == 0:
        logger.warning("No BCM data to plot")
        return

    if interactive:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Scatter plot of actual data
            fig.add_trace(go.Scatter(
                x=activations,
                y=changes,
                mode="markers",
                marker=dict(
                    size=4,
                    color=changes,
                    colorscale="RdBu",
                    showscale=True,
                    colorbar=dict(title="Weight Change")
                ),
                name="Observed"
            ))

            # Theoretical BCM curve
            x_theory = np.linspace(0, 1, 100)
            theta_m = 0.5  # Modification threshold
            y_theory = x_theory * (x_theory - theta_m)

            fig.add_trace(go.Scatter(
                x=x_theory,
                y=y_theory,
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name="Theoretical BCM"
            ))

            fig.update_layout(
                title="BCM Learning Curve",
                xaxis_title="Activation Level",
                yaxis_title="Weight Change",
                hovermode="closest",
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

            fig, ax = plt.subplots(figsize=(10, 6))

            # Scatter plot
            scatter = ax.scatter(
                activations,
                changes,
                c=changes,
                cmap="RdBu",
                alpha=0.6,
                s=20
            )

            # Theoretical BCM curve
            x_theory = np.linspace(0, 1, 100)
            theta_m = 0.5
            y_theory = x_theory * (x_theory - theta_m)
            ax.plot(x_theory, y_theory, "k--", linewidth=2, label="Theoretical BCM")

            ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
            ax.axvline(x=theta_m, color="gray", linestyle=":", linewidth=1,
                      label=f"θ_m = {theta_m}")

            ax.set_xlabel("Activation Level")
            ax.set_ylabel("Weight Change")
            ax.set_title("BCM Learning Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.colorbar(scatter, ax=ax, label="Weight Change")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)  # MEM-006 FIX: Close figure to prevent memory leak

        except ImportError:
            logger.error("Neither plotly nor matplotlib available for plotting")


def plot_weight_changes(
    tracer: PlasticityTracer,
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot timeline of weight change events.

    Args:
        tracer: PlasticityTracer instance
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    bin_times, type_counts = tracer.get_timeline_data()

    if not bin_times:
        logger.warning("No timeline data to plot")
        return

    if interactive:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            for update_type, counts in type_counts.items():
                fig.add_trace(go.Bar(
                    x=bin_times,
                    y=counts,
                    name=update_type.upper()
                ))

            fig.update_layout(
                title="Synaptic Weight Changes Over Time",
                xaxis_title="Time",
                yaxis_title="Number of Updates",
                barmode="stack",
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

            # Stacked bar chart
            bottom = np.zeros(len(bin_times))
            for update_type, counts in type_counts.items():
                ax.bar(bin_times, counts, label=update_type.upper(),
                      bottom=bottom, alpha=0.8)
                bottom += np.array(counts)

            ax.set_xlabel("Time")
            ax.set_ylabel("Number of Updates")
            ax.set_title("Synaptic Weight Changes Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)  # MEM-006 FIX: Close figure to prevent memory leak

        except ImportError:
            logger.error("Neither plotly nor matplotlib available for plotting")


def plot_ltp_ltd_distribution(
    tracer: PlasticityTracer,
    save_path: Path | None = None,
    interactive: bool = False
) -> None:
    """
    Plot distribution of LTP and LTD magnitudes.

    Args:
        tracer: PlasticityTracer instance
        save_path: Optional path to save figure
        interactive: If True, use plotly; else matplotlib
    """
    ltp_mags, ltd_mags = tracer.get_ltp_ltd_distribution()

    if len(ltp_mags) == 0 and len(ltd_mags) == 0:
        logger.warning("No LTP/LTD data to plot")
        return

    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("LTP Magnitudes", "LTD Magnitudes")
            )

            # LTP histogram
            if len(ltp_mags) > 0:
                fig.add_trace(
                    go.Histogram(x=ltp_mags, name="LTP", marker_color="green"),
                    row=1, col=1
                )

            # LTD histogram
            if len(ltd_mags) > 0:
                fig.add_trace(
                    go.Histogram(x=ltd_mags, name="LTD", marker_color="red"),
                    row=1, col=2
                )

            fig.update_xaxes(title_text="Weight Change Magnitude", row=1, col=1)
            fig.update_xaxes(title_text="Weight Change Magnitude", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)

            fig.update_layout(
                title="LTP/LTD Distribution",
                showlegend=False,
                height=400
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

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # LTP histogram
            if len(ltp_mags) > 0:
                ax1.hist(ltp_mags, bins=30, color="green", alpha=0.7, edgecolor="black")
                ax1.set_xlabel("Weight Change Magnitude")
                ax1.set_ylabel("Count")
                ax1.set_title(f"LTP Magnitudes (n={len(ltp_mags)})")
                ax1.grid(True, alpha=0.3)

            # LTD histogram
            if len(ltd_mags) > 0:
                ax2.hist(ltd_mags, bins=30, color="red", alpha=0.7, edgecolor="black")
                ax2.set_xlabel("Weight Change Magnitude")
                ax2.set_ylabel("Count")
                ax2.set_title(f"LTD Magnitudes (n={len(ltd_mags)})")
                ax2.grid(True, alpha=0.3)

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
    "PlasticityTracer",
    "WeightUpdate",
    "plot_bcm_curve",
    "plot_ltp_ltd_distribution",
    "plot_weight_changes",
]
