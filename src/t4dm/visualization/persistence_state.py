"""
Persistence State Visualization.

Visualizations for WAL, checkpoints, and system durability state.

Features:
- WAL segment timeline
- Checkpoint history
- LSN progression
- Recovery status dashboard
- Real-time durability metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WALSegmentInfo:
    """Information about a WAL segment."""
    segment_number: int
    path: Path
    size_bytes: int
    min_lsn: int
    max_lsn: int
    created_at: datetime
    entry_count: int


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    lsn: int
    timestamp: datetime
    size_bytes: int
    components: list[str]
    duration_seconds: float = 0.0


@dataclass
class PersistenceMetrics:
    """Current persistence metrics for visualization."""
    current_lsn: int = 0
    checkpoint_lsn: int = 0
    operations_since_checkpoint: int = 0
    wal_segment_count: int = 0
    wal_total_size_bytes: int = 0
    checkpoint_count: int = 0
    last_checkpoint_age_seconds: float = 0.0
    recovery_mode: str = "unknown"


@dataclass
class T4DXPersistenceInfo:
    """T4DX embedded storage engine persistence metrics."""
    memtable_items: int = 0
    segment_count: int = 0
    wal_entries: int = 0
    last_flush_lsn: int = 0
    kappa_index_size: int = 0


class PersistenceVisualizer:
    """
    Visualizer for persistence layer state.

    Tracks WAL segments, checkpoints, and recovery status for visualization.

    Usage:
        visualizer = PersistenceVisualizer()

        # Record WAL segment info
        visualizer.record_wal_segment(WALSegmentInfo(...))

        # Record checkpoint
        visualizer.record_checkpoint(CheckpointInfo(...))

        # Update metrics
        visualizer.update_metrics(PersistenceMetrics(...))

        # Visualize
        plot_wal_timeline(visualizer)
        plot_checkpoint_history(visualizer)
        plot_durability_dashboard(visualizer)
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize visualizer.

        Args:
            max_history: Maximum history entries to keep
        """
        self.max_history = max_history
        self.wal_segments: list[WALSegmentInfo] = []
        self.checkpoints: list[CheckpointInfo] = []
        self.metrics_history: list[tuple[datetime, PersistenceMetrics]] = []
        self.lsn_timeline: list[tuple[datetime, int]] = []
        self.t4dx_history: list[tuple[datetime, T4DXPersistenceInfo]] = []

    def record_wal_segment(self, segment: WALSegmentInfo) -> None:
        """Record WAL segment information."""
        self.wal_segments.append(segment)
        if len(self.wal_segments) > self.max_history:
            self.wal_segments.pop(0)

    def record_checkpoint(self, checkpoint: CheckpointInfo) -> None:
        """Record checkpoint information."""
        self.checkpoints.append(checkpoint)
        if len(self.checkpoints) > self.max_history:
            self.checkpoints.pop(0)

    def update_metrics(self, metrics: PersistenceMetrics) -> None:
        """Update current metrics."""
        now = datetime.now()
        self.metrics_history.append((now, metrics))
        self.lsn_timeline.append((now, metrics.current_lsn))

        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        if len(self.lsn_timeline) > self.max_history:
            self.lsn_timeline.pop(0)

    @property
    def current_metrics(self) -> PersistenceMetrics | None:
        """Get most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1][1]
        return None

    def get_wal_size_over_time(self) -> tuple[list[datetime], list[int]]:
        """Get WAL size progression over time."""
        timestamps = []
        sizes = []
        for segment in self.wal_segments:
            timestamps.append(segment.created_at)
            sizes.append(segment.size_bytes)
        return timestamps, sizes

    def get_lsn_over_time(self) -> tuple[list[datetime], list[int]]:
        """Get LSN progression over time."""
        timestamps = [t for t, _ in self.lsn_timeline]
        lsns = [lsn for _, lsn in self.lsn_timeline]
        return timestamps, lsns

    def get_checkpoint_timeline(self) -> tuple[list[datetime], list[int]]:
        """Get checkpoint LSNs over time."""
        timestamps = [cp.timestamp for cp in self.checkpoints]
        lsns = [cp.lsn for cp in self.checkpoints]
        return timestamps, lsns


    def record_t4dx_state(self, info: T4DXPersistenceInfo) -> None:
        """Record T4DX storage engine state."""
        now = datetime.now()
        self.t4dx_history.append((now, info))
        if len(self.t4dx_history) > self.max_history:
            self.t4dx_history.pop(0)

    @property
    def current_t4dx(self) -> T4DXPersistenceInfo | None:
        """Get most recent T4DX info."""
        if self.t4dx_history:
            return self.t4dx_history[-1][1]
        return None

    def plot_t4dx_state(self, ax=None) -> Any:
        """
        Plot T4DX storage engine state: memtable fill, segment count, WAL size.

        Shows a 3-bar summary of the current T4DX state, or time series if
        multiple snapshots are available.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        if not self.t4dx_history:
            ax.text(0.5, 0.5, "No T4DX data", ha="center", va="center")
            return ax

        if len(self.t4dx_history) < 3:
            # Bar chart of latest state
            info = self.t4dx_history[-1][1]
            labels = ["MemTable\nItems", "Segments", "WAL\nEntries",
                       "Flush LSN", "Kappa\nIndex"]
            values = [info.memtable_items, info.segment_count, info.wal_entries,
                      info.last_flush_lsn, info.kappa_index_size]
            colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
            ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_title("T4DX Storage Engine State")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            # Time series
            times = [t for t, _ in self.t4dx_history]
            ax.plot(times, [i.memtable_items for _, i in self.t4dx_history],
                    label="MemTable Items", color="#3498db")
            ax.plot(times, [i.segment_count for _, i in self.t4dx_history],
                    label="Segments", color="#e74c3c")
            ax.plot(times, [i.wal_entries for _, i in self.t4dx_history],
                    label="WAL Entries", color="#2ecc71")
            ax.set_xlabel("Time")
            ax.set_ylabel("Count")
            ax.set_title("T4DX Storage Engine Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)

        return ax


def plot_t4dx_state(
    visualizer: PersistenceVisualizer,
    ax=None,
) -> Any:
    """Plot T4DX storage engine state."""
    return visualizer.plot_t4dx_state(ax=ax)


def plot_wal_timeline(
    visualizer: PersistenceVisualizer,
    interactive: bool = True,
    save_path: Path | None = None,
) -> Any:
    """
    Plot WAL segment timeline.

    Shows segment sizes and LSN ranges over time.
    """
    if interactive:
        return _plot_wal_timeline_plotly(visualizer, save_path)
    return _plot_wal_timeline_matplotlib(visualizer, save_path)


def _plot_wal_timeline_plotly(
    visualizer: PersistenceVisualizer,
    save_path: Path | None = None,
) -> Any:
    """Interactive WAL timeline using plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("plotly not installed, falling back to matplotlib")
        return _plot_wal_timeline_matplotlib(visualizer, save_path)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("WAL Segment Sizes", "LSN Progression"),
        vertical_spacing=0.15,
    )

    # WAL segment sizes
    segments = visualizer.wal_segments
    if segments:
        fig.add_trace(
            go.Bar(
                x=[s.segment_number for s in segments],
                y=[s.size_bytes / 1024 / 1024 for s in segments],  # MB
                name="Segment Size (MB)",
                marker_color="steelblue",
                hovertemplate="Segment %{x}<br>Size: %{y:.2f} MB<br>LSN: %{customdata[0]}-%{customdata[1]}",
                customdata=[[s.min_lsn, s.max_lsn] for s in segments],
            ),
            row=1, col=1,
        )

    # LSN progression
    timestamps, lsns = visualizer.get_lsn_over_time()
    if timestamps:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=lsns,
                mode="lines",
                name="Current LSN",
                line=dict(color="green", width=2),
            ),
            row=2, col=1,
        )

    # Checkpoint markers
    cp_times, cp_lsns = visualizer.get_checkpoint_timeline()
    if cp_times:
        fig.add_trace(
            go.Scatter(
                x=cp_times,
                y=cp_lsns,
                mode="markers",
                name="Checkpoints",
                marker=dict(color="red", size=10, symbol="diamond"),
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title="WAL and Checkpoint Timeline",
        height=600,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Segment Number", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Size (MB)", row=1, col=1)
    fig.update_yaxes(title_text="LSN", row=2, col=1)

    if save_path:
        fig.write_html(str(save_path))

    return fig


def _plot_wal_timeline_matplotlib(
    visualizer: PersistenceVisualizer,
    save_path: Path | None = None,
) -> Any:
    """Static WAL timeline using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # WAL segment sizes
    segments = visualizer.wal_segments
    if segments:
        segment_nums = [s.segment_number for s in segments]
        sizes_mb = [s.size_bytes / 1024 / 1024 for s in segments]
        axes[0].bar(segment_nums, sizes_mb, color="steelblue")
        axes[0].set_xlabel("Segment Number")
        axes[0].set_ylabel("Size (MB)")
        axes[0].set_title("WAL Segment Sizes")

    # LSN progression
    timestamps, lsns = visualizer.get_lsn_over_time()
    if timestamps:
        axes[1].plot(timestamps, lsns, "g-", linewidth=2, label="Current LSN")

    # Checkpoint markers
    cp_times, cp_lsns = visualizer.get_checkpoint_timeline()
    if cp_times:
        axes[1].scatter(cp_times, cp_lsns, c="red", s=100, marker="D", label="Checkpoints", zorder=5)

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("LSN")
    axes[1].set_title("LSN Progression")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_durability_dashboard(
    visualizer: PersistenceVisualizer,
    interactive: bool = True,
    save_path: Path | None = None,
) -> Any:
    """
    Plot durability metrics dashboard.

    Shows current state, checkpoint freshness, and recovery risk.
    """
    if interactive:
        return _plot_dashboard_plotly(visualizer, save_path)
    return _plot_dashboard_matplotlib(visualizer, save_path)


def _plot_dashboard_plotly(
    visualizer: PersistenceVisualizer,
    save_path: Path | None = None,
) -> Any:
    """Interactive dashboard using plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return _plot_dashboard_matplotlib(visualizer, save_path)

    metrics = visualizer.current_metrics
    if not metrics:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "indicator"}],
        ],
        subplot_titles=("Current LSN", "Ops Since Checkpoint", "Checkpoint Age", "WAL Size"),
    )

    # Current LSN
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=metrics.current_lsn,
            title={"text": "Current LSN"},
            number={"font": {"size": 40}},
        ),
        row=1, col=1,
    )

    # Operations since checkpoint
    ops_color = "green" if metrics.operations_since_checkpoint < 1000 else (
        "orange" if metrics.operations_since_checkpoint < 5000 else "red"
    )
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=metrics.operations_since_checkpoint,
            title={"text": "Uncommitted Ops"},
            delta={"reference": 1000, "relative": False},
            number={"font": {"color": ops_color, "size": 40}},
        ),
        row=1, col=2,
    )

    # Checkpoint age
    age_color = "green" if metrics.last_checkpoint_age_seconds < 300 else (
        "orange" if metrics.last_checkpoint_age_seconds < 600 else "red"
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=metrics.last_checkpoint_age_seconds,
            title={"text": "Checkpoint Age (s)"},
            number={"font": {"color": age_color, "size": 40}},
        ),
        row=2, col=1,
    )

    # WAL size
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=metrics.wal_total_size_bytes / 1024 / 1024,
            title={"text": "WAL Size (MB)"},
            number={"font": {"size": 40}, "suffix": " MB"},
        ),
        row=2, col=2,
    )

    fig.update_layout(
        title=f"Durability Dashboard - {metrics.recovery_mode.upper()}",
        height=500,
    )

    if save_path:
        fig.write_html(str(save_path))

    return fig


def _plot_dashboard_matplotlib(
    visualizer: PersistenceVisualizer,
    save_path: Path | None = None,
) -> Any:
    """Static dashboard using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    metrics = visualizer.current_metrics
    if not metrics:
        return None

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # Style for metric boxes
    dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black")

    # Current LSN
    axes[0, 0].text(
        0.5, 0.5, f"{metrics.current_lsn:,}",
        ha="center", va="center", fontsize=24, weight="bold",
        transform=axes[0, 0].transAxes,
    )
    axes[0, 0].set_title("Current LSN", fontsize=14)
    axes[0, 0].axis("off")

    # Operations since checkpoint
    color = "green" if metrics.operations_since_checkpoint < 1000 else (
        "orange" if metrics.operations_since_checkpoint < 5000 else "red"
    )
    axes[0, 1].text(
        0.5, 0.5, f"{metrics.operations_since_checkpoint:,}",
        ha="center", va="center", fontsize=24, weight="bold", color=color,
        transform=axes[0, 1].transAxes,
    )
    axes[0, 1].set_title("Uncommitted Ops", fontsize=14)
    axes[0, 1].axis("off")

    # Checkpoint age
    color = "green" if metrics.last_checkpoint_age_seconds < 300 else (
        "orange" if metrics.last_checkpoint_age_seconds < 600 else "red"
    )
    axes[1, 0].text(
        0.5, 0.5, f"{metrics.last_checkpoint_age_seconds:.0f}s",
        ha="center", va="center", fontsize=24, weight="bold", color=color,
        transform=axes[1, 0].transAxes,
    )
    axes[1, 0].set_title("Checkpoint Age", fontsize=14)
    axes[1, 0].axis("off")

    # WAL size
    axes[1, 1].text(
        0.5, 0.5, f"{metrics.wal_total_size_bytes / 1024 / 1024:.1f} MB",
        ha="center", va="center", fontsize=24, weight="bold",
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].set_title("WAL Size", fontsize=14)
    axes[1, 1].axis("off")

    # T4DX panel (bottom row)
    t4dx = visualizer.current_t4dx
    if t4dx:
        labels = ["MemTable", "Segments", "WAL Ent.", "Kappa Idx"]
        values = [t4dx.memtable_items, t4dx.segment_count,
                  t4dx.wal_entries, t4dx.kappa_index_size]
        bar_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
        axes[2, 0].bar(labels, values, color=bar_colors, edgecolor="black", linewidth=0.5)
        axes[2, 0].set_title("T4DX Engine State", fontsize=14)
        axes[2, 0].grid(True, alpha=0.3, axis="y")

        axes[2, 1].text(
            0.5, 0.5, f"Flush LSN: {t4dx.last_flush_lsn:,}",
            ha="center", va="center", fontsize=20, weight="bold",
            transform=axes[2, 1].transAxes,
        )
        axes[2, 1].set_title("T4DX Last Flush", fontsize=14)
        axes[2, 1].axis("off")
    else:
        for col in range(2):
            axes[2, col].text(
                0.5, 0.5, "No T4DX data",
                ha="center", va="center", fontsize=14,
                transform=axes[2, col].transAxes,
            )
            axes[2, col].axis("off")

    fig.suptitle(f"Durability Dashboard - {metrics.recovery_mode.upper()}", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_checkpoint_history(
    visualizer: PersistenceVisualizer,
    interactive: bool = True,
    save_path: Path | None = None,
) -> Any:
    """
    Plot checkpoint history.

    Shows checkpoint sizes, durations, and frequency.
    """
    checkpoints = visualizer.checkpoints
    if not checkpoints:
        logger.warning("No checkpoint data available")
        return None

    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Checkpoint Size", "Checkpoint Duration"),
            )

            times = [cp.timestamp for cp in checkpoints]
            sizes = [cp.size_bytes / 1024 for cp in checkpoints]  # KB
            durations = [cp.duration_seconds for cp in checkpoints]

            fig.add_trace(
                go.Scatter(x=times, y=sizes, mode="lines+markers", name="Size (KB)"),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=times, y=durations, mode="lines+markers", name="Duration (s)"),
                row=2, col=1,
            )

            fig.update_layout(title="Checkpoint History", height=500)

            if save_path:
                fig.write_html(str(save_path))

            return fig

        except ImportError:
            pass

    # Matplotlib fallback
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        times = [cp.timestamp for cp in checkpoints]
        sizes = [cp.size_bytes / 1024 for cp in checkpoints]
        durations = [cp.duration_seconds for cp in checkpoints]

        axes[0].plot(times, sizes, "b-o")
        axes[0].set_ylabel("Size (KB)")
        axes[0].set_title("Checkpoint Size")

        axes[1].plot(times, durations, "r-o")
        axes[1].set_ylabel("Duration (s)")
        axes[1].set_title("Checkpoint Duration")
        axes[1].set_xlabel("Time")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig

    except ImportError:
        return None
