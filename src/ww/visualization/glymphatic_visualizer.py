"""
Glymphatic Clearance Visualization for World Weaver NCA.

CompBio-inspired visualization of sleep-gated waste clearance:
- Sleep stage timeline (NREM/REM/Wake)
- Clearance rate over time
- Waste accumulation gradient
- AQP4 channel activity
- Memory pruning candidates

This is CRITICAL for understanding:
- How sleep stages affect memory clearance
- Which memories are candidates for pruning
- Glymphatic system efficiency
- Relationship between sleep pressure and clearance

References:
- Xie et al. (2013) - Glymphatic system and sleep
- Borbély (1982) - Two-process model of sleep regulation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ww.nca.adenosine import AdenosineDynamics
    from ww.nca.glymphatic import GlymphaticSystem

logger = logging.getLogger(__name__)


class SleepStage(Enum):
    """Sleep stages for glymphatic activity."""

    WAKE = auto()
    NREM_LIGHT = auto()  # N1/N2
    NREM_DEEP = auto()   # N3/SWS
    REM = auto()


@dataclass
class GlymphaticSnapshot:
    """Snapshot of glymphatic system state."""

    timestamp: datetime
    sleep_stage: SleepStage
    clearance_rate: float  # 0.0 to 1.0
    adenosine_level: float  # Sleep pressure
    waste_level: float  # Accumulated metabolic waste
    aqp4_activity: float  # Aquaporin-4 channel activity
    n_prune_candidates: int  # Memories marked for pruning
    csf_flow_rate: float  # Cerebrospinal fluid flow


@dataclass
class ClearanceEvent:
    """Record of a memory clearance event."""

    timestamp: datetime
    memory_id: str
    memory_type: str  # "episodic", "semantic", "procedural"
    reason: str  # "unused", "low_stability", "redundant"
    age_days: float
    last_access_days: float
    stability_score: float


class GlymphaticVisualizer:
    """
    Visualizes glymphatic clearance dynamics.

    Tracks sleep stages, clearance rates, waste accumulation,
    and memory pruning during sleep.
    """

    def __init__(
        self,
        glymphatic_system: GlymphaticSystem | None = None,
        adenosine_dynamics: AdenosineDynamics | None = None,
        window_size: int = 1000,
        alert_high_waste: float = 0.8,
        alert_low_clearance: float = 0.1,
    ):
        """
        Initialize glymphatic visualizer.

        Args:
            glymphatic_system: GlymphaticSystem instance to monitor
            adenosine_dynamics: AdenosineDynamics for sleep pressure
            window_size: Number of snapshots to retain
            alert_high_waste: Alert if waste level too high
            alert_low_clearance: Alert if clearance rate too low during sleep
        """
        self.glymphatic_system = glymphatic_system
        self.adenosine_dynamics = adenosine_dynamics
        self.window_size = window_size
        self.alert_high_waste = alert_high_waste
        self.alert_low_clearance = alert_low_clearance

        # History tracking
        self._snapshots: list[GlymphaticSnapshot] = []
        self._clearance_events: list[ClearanceEvent] = []
        self._stage_history: list[tuple[datetime, SleepStage]] = []
        self._clearance_history: list[float] = []
        self._waste_history: list[float] = []

        # Alerts
        self._active_alerts: list[str] = []

        logger.info("GlymphaticVisualizer initialized")

    def record_state(
        self,
        sleep_stage: SleepStage,
        clearance_rate: float,
        adenosine_level: float,
        waste_level: float,
        aqp4_activity: float = 0.5,
        n_prune_candidates: int = 0,
        csf_flow_rate: float = 0.5,
    ) -> GlymphaticSnapshot:
        """
        Record current glymphatic state.

        Args:
            sleep_stage: Current sleep stage
            clearance_rate: Clearance efficiency (0-1)
            adenosine_level: Sleep pressure (0-1)
            waste_level: Metabolic waste (0-1)
            aqp4_activity: AQP4 channel activity (0-1)
            n_prune_candidates: Number of memories to prune
            csf_flow_rate: CSF flow (0-1)

        Returns:
            GlymphaticSnapshot with all metrics
        """
        now = datetime.now()

        snapshot = GlymphaticSnapshot(
            timestamp=now,
            sleep_stage=sleep_stage,
            clearance_rate=clearance_rate,
            adenosine_level=adenosine_level,
            waste_level=waste_level,
            aqp4_activity=aqp4_activity,
            n_prune_candidates=n_prune_candidates,
            csf_flow_rate=csf_flow_rate,
        )

        # Store snapshot
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

        # Store stage transitions
        if not self._stage_history or self._stage_history[-1][1] != sleep_stage:
            self._stage_history.append((now, sleep_stage))
            if len(self._stage_history) > self.window_size:
                self._stage_history.pop(0)

        # Store clearance and waste history
        self._clearance_history.append(clearance_rate)
        self._waste_history.append(waste_level)
        if len(self._clearance_history) > self.window_size:
            self._clearance_history.pop(0)
            self._waste_history.pop(0)

        self._check_alerts(snapshot)

        return snapshot

    def record_clearance_event(
        self,
        memory_id: str,
        memory_type: str,
        reason: str,
        age_days: float,
        last_access_days: float,
        stability_score: float,
    ) -> ClearanceEvent:
        """
        Record a memory clearance event.

        Args:
            memory_id: ID of cleared memory
            memory_type: Type (episodic, semantic, procedural)
            reason: Why it was cleared
            age_days: Age in days
            last_access_days: Days since last access
            stability_score: Memory stability (0-1)

        Returns:
            ClearanceEvent record
        """
        event = ClearanceEvent(
            timestamp=datetime.now(),
            memory_id=memory_id,
            memory_type=memory_type,
            reason=reason,
            age_days=age_days,
            last_access_days=last_access_days,
            stability_score=stability_score,
        )

        self._clearance_events.append(event)
        if len(self._clearance_events) > self.window_size:
            self._clearance_events.pop(0)

        return event

    def _check_alerts(self, snapshot: GlymphaticSnapshot) -> None:
        """Check for alert conditions."""
        self._active_alerts.clear()

        if snapshot.waste_level > self.alert_high_waste:
            self._active_alerts.append(
                f"HIGH WASTE: Waste level {snapshot.waste_level:.2f} > {self.alert_high_waste}"
            )

        # Low clearance during sleep is concerning
        if snapshot.sleep_stage in (SleepStage.NREM_DEEP, SleepStage.NREM_LIGHT):
            if snapshot.clearance_rate < self.alert_low_clearance:
                self._active_alerts.append(
                    f"LOW CLEARANCE: Clearance {snapshot.clearance_rate:.2f} "
                    f"during {snapshot.sleep_stage.name}"
                )

    def get_alerts(self) -> list[str]:
        """Get current active alerts."""
        return self._active_alerts.copy()

    # -------------------------------------------------------------------------
    # Sleep Stage Analysis
    # -------------------------------------------------------------------------

    def get_current_stage(self) -> SleepStage | None:
        """Get current sleep stage."""
        if self._snapshots:
            return self._snapshots[-1].sleep_stage
        return None

    def get_stage_timeline(self) -> list[tuple[datetime, SleepStage]]:
        """Get timeline of sleep stage transitions."""
        return self._stage_history.copy()

    def get_stage_durations(self) -> dict[SleepStage, timedelta]:
        """Get total duration spent in each sleep stage."""
        durations = {stage: timedelta() for stage in SleepStage}

        if len(self._stage_history) < 2:
            return durations

        for i in range(len(self._stage_history) - 1):
            start_time, stage = self._stage_history[i]
            end_time = self._stage_history[i + 1][0]
            durations[stage] += end_time - start_time

        # Add current stage duration
        if self._stage_history:
            last_time, last_stage = self._stage_history[-1]
            durations[last_stage] += datetime.now() - last_time

        return durations

    def get_stage_percentages(self) -> dict[str, float]:
        """Get percentage of time in each sleep stage."""
        durations = self.get_stage_durations()
        total_seconds = sum(d.total_seconds() for d in durations.values())

        if total_seconds < 1:
            return {stage.name: 0.0 for stage in SleepStage}

        return {
            stage.name: d.total_seconds() / total_seconds
            for stage, d in durations.items()
        }

    # -------------------------------------------------------------------------
    # Clearance Analysis
    # -------------------------------------------------------------------------

    def get_clearance_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of clearance rate."""
        timestamps = [s.timestamp for s in self._snapshots]
        rates = [s.clearance_rate for s in self._snapshots]
        return timestamps, rates

    def get_clearance_by_stage(self) -> dict[str, float]:
        """Get mean clearance rate for each sleep stage."""
        by_stage = {stage.name: [] for stage in SleepStage}

        for s in self._snapshots:
            by_stage[s.sleep_stage.name].append(s.clearance_rate)

        return {
            stage: float(np.mean(rates)) if rates else 0.0
            for stage, rates in by_stage.items()
        }

    def get_clearance_statistics(self) -> dict[str, float]:
        """Get clearance rate statistics."""
        if not self._clearance_history:
            return {}

        return {
            "mean": float(np.mean(self._clearance_history)),
            "std": float(np.std(self._clearance_history)),
            "min": float(np.min(self._clearance_history)),
            "max": float(np.max(self._clearance_history)),
            "current": self._clearance_history[-1],
        }

    # -------------------------------------------------------------------------
    # Waste Analysis
    # -------------------------------------------------------------------------

    def get_waste_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of waste level."""
        timestamps = [s.timestamp for s in self._snapshots]
        waste = [s.waste_level for s in self._snapshots]
        return timestamps, waste

    def get_waste_statistics(self) -> dict[str, float]:
        """Get waste level statistics."""
        if not self._waste_history:
            return {}

        return {
            "mean": float(np.mean(self._waste_history)),
            "std": float(np.std(self._waste_history)),
            "min": float(np.min(self._waste_history)),
            "max": float(np.max(self._waste_history)),
            "current": self._waste_history[-1],
        }

    # -------------------------------------------------------------------------
    # Memory Clearance Analysis
    # -------------------------------------------------------------------------

    def get_clearance_events_by_type(self) -> dict[str, int]:
        """Get count of clearance events by memory type."""
        by_type = {"episodic": 0, "semantic": 0, "procedural": 0}
        for event in self._clearance_events:
            by_type[event.memory_type] = by_type.get(event.memory_type, 0) + 1
        return by_type

    def get_clearance_events_by_reason(self) -> dict[str, int]:
        """Get count of clearance events by reason."""
        by_reason = {}
        for event in self._clearance_events:
            by_reason[event.reason] = by_reason.get(event.reason, 0) + 1
        return by_reason

    def get_recent_clearances(self, n: int = 10) -> list[dict]:
        """Get N most recent clearance events."""
        recent = self._clearance_events[-n:]
        return [
            {
                "memory_id": e.memory_id,
                "memory_type": e.memory_type,
                "reason": e.reason,
                "age_days": e.age_days,
                "last_access_days": e.last_access_days,
                "stability_score": e.stability_score,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in recent
        ]

    # -------------------------------------------------------------------------
    # AQP4 and CSF Analysis
    # -------------------------------------------------------------------------

    def get_aqp4_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of AQP4 activity."""
        timestamps = [s.timestamp for s in self._snapshots]
        aqp4 = [s.aqp4_activity for s in self._snapshots]
        return timestamps, aqp4

    def get_csf_flow_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of CSF flow rate."""
        timestamps = [s.timestamp for s in self._snapshots]
        csf = [s.csf_flow_rate for s in self._snapshots]
        return timestamps, csf

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_data(self) -> dict:
        """Export visualization data for external rendering."""
        return {
            "current_stage": self.get_current_stage().name if self.get_current_stage() else None,
            "stage_percentages": self.get_stage_percentages(),
            "stage_timeline": [
                {"timestamp": t.isoformat(), "stage": s.name}
                for t, s in self._stage_history
            ],
            "clearance_statistics": self.get_clearance_statistics(),
            "clearance_by_stage": self.get_clearance_by_stage(),
            "waste_statistics": self.get_waste_statistics(),
            "clearance_events_by_type": self.get_clearance_events_by_type(),
            "clearance_events_by_reason": self.get_clearance_events_by_reason(),
            "recent_clearances": self.get_recent_clearances(),
            "alerts": self.get_alerts(),
        }

    def clear_history(self) -> None:
        """Clear all history."""
        self._snapshots.clear()
        self._clearance_events.clear()
        self._stage_history.clear()
        self._clearance_history.clear()
        self._waste_history.clear()
        self._active_alerts.clear()


# =============================================================================
# Standalone Plot Functions
# =============================================================================


def plot_sleep_stage_timeline(
    visualizer: GlymphaticVisualizer,
    ax=None,
):
    """
    Plot sleep stage timeline as colored bands.

    Args:
        visualizer: GlymphaticVisualizer instance
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 3))

    timeline = visualizer.get_stage_timeline()

    if len(timeline) < 2:
        ax.text(0.5, 0.5, "No stage data", ha="center", va="center")
        return ax

    # Stage colors
    colors = {
        SleepStage.WAKE: "#FFD700",       # Gold
        SleepStage.NREM_LIGHT: "#87CEEB",  # Sky blue
        SleepStage.NREM_DEEP: "#4169E1",   # Royal blue
        SleepStage.REM: "#9370DB",          # Medium purple
    }

    t0 = timeline[0][0]

    for i in range(len(timeline)):
        start_time, stage = timeline[i]
        if i + 1 < len(timeline):
            end_time = timeline[i + 1][0]
        else:
            end_time = datetime.now()

        x_start = (start_time - t0).total_seconds() / 60  # Minutes
        width = (end_time - start_time).total_seconds() / 60

        rect = Rectangle((x_start, 0), width, 1,
                          facecolor=colors[stage], edgecolor="black", linewidth=0.5)
        ax.add_patch(rect)

    ax.set_xlim(0, (datetime.now() - t0).total_seconds() / 60)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("")
    ax.set_title("Sleep Stage Timeline")
    ax.set_yticks([])

    # Legend
    handles = [Rectangle((0, 0), 1, 1, facecolor=colors[s], edgecolor="black")
               for s in SleepStage]
    labels = [s.name for s in SleepStage]
    ax.legend(handles, labels, loc="upper right", ncol=4)

    return ax


def plot_clearance_and_waste(
    visualizer: GlymphaticVisualizer,
    ax=None,
):
    """
    Plot clearance rate and waste level over time.

    Args:
        visualizer: GlymphaticVisualizer instance
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    timestamps_c, clearance = visualizer.get_clearance_trace()
    timestamps_w, waste = visualizer.get_waste_trace()

    if not timestamps_c:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    t0 = timestamps_c[0]
    t_minutes_c = [(t - t0).total_seconds() / 60 for t in timestamps_c]
    t_minutes_w = [(t - t0).total_seconds() / 60 for t in timestamps_w]

    ax.plot(t_minutes_c, clearance, "g-", linewidth=1.5, label="Clearance Rate")
    ax.plot(t_minutes_w, waste, "r-", linewidth=1.5, label="Waste Level")

    ax.axhline(y=visualizer.alert_high_waste, color="r",
               linestyle="--", alpha=0.5, label=f"Waste threshold ({visualizer.alert_high_waste})")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Level (0-1)")
    ax.set_title("Glymphatic Clearance vs Waste Accumulation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    return ax


def plot_clearance_by_stage(
    visualizer: GlymphaticVisualizer,
    ax=None,
):
    """
    Plot mean clearance rate by sleep stage.

    Args:
        visualizer: GlymphaticVisualizer instance
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    by_stage = visualizer.get_clearance_by_stage()

    if not any(by_stage.values()):
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    stages = list(by_stage.keys())
    values = [by_stage[s] for s in stages]

    colors = ["#FFD700", "#87CEEB", "#4169E1", "#9370DB"]
    ax.bar(stages, values, color=colors)

    ax.set_xlabel("Sleep Stage")
    ax.set_ylabel("Mean Clearance Rate")
    ax.set_title("Clearance Efficiency by Sleep Stage")
    ax.grid(True, alpha=0.3, axis="y")

    # Expected pattern annotation
    ax.axhline(y=0.9, color="g", linestyle=":", alpha=0.5, label="Target NREM_DEEP")
    ax.legend()

    return ax


def plot_stage_pie(
    visualizer: GlymphaticVisualizer,
    ax=None,
):
    """
    Plot pie chart of time spent in each sleep stage.

    Args:
        visualizer: GlymphaticVisualizer instance
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    percentages = visualizer.get_stage_percentages()

    if not any(percentages.values()):
        ax.text(0.5, 0.5, "No stage data", ha="center", va="center")
        return ax

    labels = list(percentages.keys())
    sizes = [percentages[l] * 100 for l in labels]
    colors = ["#FFD700", "#87CEEB", "#4169E1", "#9370DB"]

    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, wedgeprops={"edgecolor": "black"})
    ax.set_title("Sleep Stage Distribution")

    return ax


def create_glymphatic_dashboard(
    visualizer: GlymphaticVisualizer,
    figsize: tuple[int, int] = (16, 10),
):
    """
    Create comprehensive glymphatic system dashboard.

    Args:
        visualizer: GlymphaticVisualizer instance
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig)

    # Sleep stage timeline (full width top)
    ax_timeline = fig.add_subplot(gs[0, :])
    plot_sleep_stage_timeline(visualizer, ax=ax_timeline)

    # Clearance and waste
    ax_clearance = fig.add_subplot(gs[1, :2])
    plot_clearance_and_waste(visualizer, ax=ax_clearance)

    # Stage pie
    ax_pie = fig.add_subplot(gs[1, 2])
    plot_stage_pie(visualizer, ax=ax_pie)

    # Clearance by stage
    ax_by_stage = fig.add_subplot(gs[2, :2])
    plot_clearance_by_stage(visualizer, ax=ax_by_stage)

    # Statistics text box
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis("off")

    clearance_stats = visualizer.get_clearance_statistics()
    waste_stats = visualizer.get_waste_statistics()
    by_type = visualizer.get_clearance_events_by_type()
    by_reason = visualizer.get_clearance_events_by_reason()
    alerts = visualizer.get_alerts()

    text_lines = [
        "Glymphatic Statistics",
        "=" * 30,
    ]

    if clearance_stats:
        text_lines.extend([
            f"Clearance: {clearance_stats.get('mean', 0):.2f} "
            f"(current: {clearance_stats.get('current', 0):.2f})",
        ])

    if waste_stats:
        text_lines.extend([
            f"Waste: {waste_stats.get('mean', 0):.2f} "
            f"(current: {waste_stats.get('current', 0):.2f})",
        ])

    text_lines.extend([
        "",
        "Cleared Memories:",
    ])
    for mem_type, count in by_type.items():
        text_lines.append(f"  {mem_type}: {count}")

    text_lines.extend([
        "",
        "Clearance Reasons:",
    ])
    for reason, count in by_reason.items():
        text_lines.append(f"  {reason}: {count}")

    if alerts:
        text_lines.extend(["", "ALERTS:", "-" * 20])
        text_lines.extend(alerts[:3])

    text = "\n".join(text_lines)
    ax_stats.text(0.05, 0.95, text, fontsize=9, family="monospace",
                  verticalalignment="top", transform=ax_stats.transAxes)

    plt.tight_layout()
    return fig


__all__ = [
    "GlymphaticVisualizer",
    "GlymphaticSnapshot",
    "ClearanceEvent",
    "SleepStage",
    "plot_sleep_stage_timeline",
    "plot_clearance_and_waste",
    "plot_clearance_by_stage",
    "plot_stage_pie",
    "create_glymphatic_dashboard",
]
