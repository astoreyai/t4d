"""
NT State Dashboard for T4DM NCA.

Comprehensive visualization of 6-channel neuromodulator dynamics:
- Real-time NT state display with biological color coding
- Homeostatic setpoint deviation indicators
- Receptor saturation curves (Michaelis-Menten)
- Cross-NT correlation matrix
- Temporal autocorrelation analysis

This provides ESSENTIAL observability into the NT state space
for understanding system behavior and debugging pathologies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from t4dm.nca.neural_field import NeurotransmitterState

logger = logging.getLogger(__name__)

# NT labels and biological color coding
NT_LABELS = ["DA", "5-HT", "ACh", "NE", "GABA", "Glu"]
NT_COLORS = {
    "DA": "#FF6B6B",      # Red - reward/motivation
    "5-HT": "#4ECDC4",    # Teal - mood/patience
    "ACh": "#45B7D1",     # Blue - attention/encoding
    "NE": "#FFA07A",      # Orange - arousal
    "GABA": "#9B59B6",    # Purple - inhibition
    "Glu": "#2ECC71",     # Green - excitation
}

# Biological homeostatic setpoints (normalized to [0, 1])
HOMEOSTATIC_SETPOINTS = np.array([
    0.5,   # DA: tonic level
    0.5,   # 5-HT: baseline serotonin
    0.5,   # ACh: balanced encoding/retrieval
    0.4,   # NE: moderate arousal
    0.5,   # GABA: E/I balance
    0.5,   # Glu: E/I balance
], dtype=np.float32)

# Receptor saturation Km values (Michaelis-Menten half-saturation)
RECEPTOR_KM = np.array([
    0.3,   # DA: D1/D2 receptors
    0.4,   # 5-HT: 5-HT1A/2A receptors
    0.25,  # ACh: mAChR/nAChR
    0.35,  # NE: alpha/beta adrenergic
    0.4,   # GABA: GABA-A/B receptors
    0.3,   # Glu: NMDA/AMPA receptors
], dtype=np.float32)


@dataclass
class NTSnapshot:
    """Complete NT state snapshot with derived metrics."""

    timestamp: datetime
    nt_state: np.ndarray  # [6] NT concentrations

    # Derived metrics
    setpoint_deviation: np.ndarray  # [6] deviation from homeostatic target
    receptor_saturation: np.ndarray  # [6] receptor occupancy [0, 1]
    ei_balance: float  # GABA/Glu ratio
    arousal_index: float  # Combined NE + DA + Glu measure

    # Optional firing rates (if available)
    vta_firing_rate: float | None = None
    raphe_firing_rate: float | None = None


@dataclass
class NTStatistics:
    """Statistical summary of NT dynamics."""

    mean: np.ndarray  # [6] mean concentrations
    std: np.ndarray  # [6] standard deviations
    min: np.ndarray  # [6] minimum values
    max: np.ndarray  # [6] maximum values
    correlation_matrix: np.ndarray  # [6, 6] cross-NT correlations
    autocorrelation: np.ndarray  # [6] temporal autocorrelation (lag-1)


class NTStateDashboard:
    """
    Comprehensive NT state monitoring dashboard.

    Tracks all 6 neuromodulators with biological context:
    - Real-time concentration tracking
    - Homeostatic regulation monitoring
    - Receptor dynamics modeling
    - Cross-system correlations
    - Temporal dynamics analysis
    """

    def __init__(
        self,
        window_size: int = 1000,
        homeostatic_setpoints: np.ndarray | None = None,
        receptor_km: np.ndarray | None = None,
        alert_deviation: float = 0.3,
    ):
        """
        Initialize NT state dashboard.

        Args:
            window_size: Number of snapshots to retain
            homeostatic_setpoints: Custom setpoints (default: biological)
            receptor_km: Custom receptor Km values (default: biological)
            alert_deviation: Threshold for setpoint deviation alerts
        """
        self.window_size = window_size
        self.setpoints = homeostatic_setpoints if homeostatic_setpoints is not None else HOMEOSTATIC_SETPOINTS.copy()
        self.km = receptor_km if receptor_km is not None else RECEPTOR_KM.copy()
        self.alert_deviation = alert_deviation

        # History tracking
        self._snapshots: list[NTSnapshot] = []
        self._nt_history: list[np.ndarray] = []  # Raw NT arrays for fast computation

        # Alert state
        self._active_alerts: list[str] = []

        logger.info("NTStateDashboard initialized")

    def record_state(
        self,
        nt_state: np.ndarray,
        vta_firing_rate: float | None = None,
        raphe_firing_rate: float | None = None,
    ) -> NTSnapshot:
        """
        Record NT state snapshot.

        Args:
            nt_state: 6D NT concentration vector [DA, 5-HT, ACh, NE, GABA, Glu]
            vta_firing_rate: Optional VTA DA neuron firing rate (Hz)
            raphe_firing_rate: Optional Raphe 5-HT neuron firing rate (Hz)

        Returns:
            NTSnapshot with all computed metrics
        """
        now = datetime.now()
        nt = np.asarray(nt_state, dtype=np.float32)

        # Compute derived metrics
        setpoint_deviation = nt - self.setpoints
        receptor_saturation = self._compute_saturation(nt)
        ei_balance = nt[4] / max(nt[5], 0.01)  # GABA / Glu
        arousal_index = 0.4 * nt[0] + 0.4 * nt[3] + 0.2 * nt[5]  # DA + NE + Glu

        snapshot = NTSnapshot(
            timestamp=now,
            nt_state=nt.copy(),
            setpoint_deviation=setpoint_deviation,
            receptor_saturation=receptor_saturation,
            ei_balance=float(ei_balance),
            arousal_index=float(arousal_index),
            vta_firing_rate=vta_firing_rate,
            raphe_firing_rate=raphe_firing_rate,
        )

        # Store
        self._snapshots.append(snapshot)
        self._nt_history.append(nt.copy())

        # Maintain window
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)
            self._nt_history.pop(0)

        # Check alerts
        self._check_alerts(snapshot)

        return snapshot

    def record_from_nt_state(
        self,
        state: NeurotransmitterState,
        vta_firing_rate: float | None = None,
        raphe_firing_rate: float | None = None,
    ) -> NTSnapshot:
        """
        Record from NeurotransmitterState object.

        Args:
            state: NeurotransmitterState instance
            vta_firing_rate: Optional VTA firing rate
            raphe_firing_rate: Optional Raphe firing rate

        Returns:
            NTSnapshot
        """
        nt_array = np.array([
            state.dopamine,
            state.serotonin,
            state.acetylcholine,
            state.norepinephrine,
            state.gaba,
            state.glutamate,
        ], dtype=np.float32)

        return self.record_state(nt_array, vta_firing_rate, raphe_firing_rate)

    def _compute_saturation(self, nt: np.ndarray) -> np.ndarray:
        """
        Compute receptor saturation using Michaelis-Menten kinetics.

        saturation = [NT] / ([NT] + Km)
        """
        return nt / (nt + self.km)

    def _check_alerts(self, snapshot: NTSnapshot) -> None:
        """Check for alert conditions."""
        self._active_alerts.clear()

        # Check setpoint deviations
        for i, (dev, label) in enumerate(zip(snapshot.setpoint_deviation, NT_LABELS)):
            if abs(dev) > self.alert_deviation:
                direction = "HIGH" if dev > 0 else "LOW"
                self._active_alerts.append(
                    f"{label}: {direction} ({snapshot.nt_state[i]:.2f}, target {self.setpoints[i]:.2f})"
                )

        # Check E/I balance
        if snapshot.ei_balance < 0.5:
            self._active_alerts.append(
                f"E/I IMBALANCE: Excitatory-dominant (ratio={snapshot.ei_balance:.2f})"
            )
        elif snapshot.ei_balance > 2.0:
            self._active_alerts.append(
                f"E/I IMBALANCE: Inhibitory-dominant (ratio={snapshot.ei_balance:.2f})"
            )

        # Check arousal extremes
        if snapshot.arousal_index > 0.8:
            self._active_alerts.append(
                f"HIGH AROUSAL: Index={snapshot.arousal_index:.2f}"
            )
        elif snapshot.arousal_index < 0.2:
            self._active_alerts.append(
                f"LOW AROUSAL: Index={snapshot.arousal_index:.2f}"
            )

    def get_alerts(self) -> list[str]:
        """Get current active alerts."""
        return self._active_alerts.copy()

    # -------------------------------------------------------------------------
    # Current State Access
    # -------------------------------------------------------------------------

    def get_current_state(self) -> np.ndarray | None:
        """Get current NT state vector."""
        if self._snapshots:
            return self._snapshots[-1].nt_state.copy()
        return None

    def get_current_snapshot(self) -> NTSnapshot | None:
        """Get most recent snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def get_current_saturation(self) -> np.ndarray | None:
        """Get current receptor saturation levels."""
        if self._snapshots:
            return self._snapshots[-1].receptor_saturation.copy()
        return None

    def get_current_deviation(self) -> np.ndarray | None:
        """Get current setpoint deviation."""
        if self._snapshots:
            return self._snapshots[-1].setpoint_deviation.copy()
        return None

    # -------------------------------------------------------------------------
    # Time Series Access
    # -------------------------------------------------------------------------

    def get_nt_traces(self) -> dict[str, tuple[list[datetime], list[float]]]:
        """
        Get time series for each NT.

        Returns:
            Dict mapping NT label -> (timestamps, values)
        """
        if not self._snapshots:
            return {}

        timestamps = [s.timestamp for s in self._snapshots]

        traces = {}
        for i, label in enumerate(NT_LABELS):
            values = [float(s.nt_state[i]) for s in self._snapshots]
            traces[label] = (timestamps, values)

        return traces

    def get_deviation_traces(self) -> dict[str, tuple[list[datetime], list[float]]]:
        """
        Get setpoint deviation time series for each NT.

        Returns:
            Dict mapping NT label -> (timestamps, deviation_values)
        """
        if not self._snapshots:
            return {}

        timestamps = [s.timestamp for s in self._snapshots]

        traces = {}
        for i, label in enumerate(NT_LABELS):
            values = [float(s.setpoint_deviation[i]) for s in self._snapshots]
            traces[label] = (timestamps, values)

        return traces

    def get_saturation_traces(self) -> dict[str, tuple[list[datetime], list[float]]]:
        """
        Get receptor saturation time series for each NT.

        Returns:
            Dict mapping NT label -> (timestamps, saturation_values)
        """
        if not self._snapshots:
            return {}

        timestamps = [s.timestamp for s in self._snapshots]

        traces = {}
        for i, label in enumerate(NT_LABELS):
            values = [float(s.receptor_saturation[i]) for s in self._snapshots]
            traces[label] = (timestamps, values)

        return traces

    def get_ei_balance_trace(self) -> tuple[list[datetime], list[float]]:
        """Get E/I balance time series."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        values = [s.ei_balance for s in self._snapshots]
        return timestamps, values

    def get_arousal_trace(self) -> tuple[list[datetime], list[float]]:
        """Get arousal index time series."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        values = [s.arousal_index for s in self._snapshots]
        return timestamps, values

    def get_firing_rate_traces(self) -> dict[str, tuple[list[datetime], list[float]]]:
        """Get VTA and Raphe firing rate traces."""
        if not self._snapshots:
            return {}

        timestamps = [s.timestamp for s in self._snapshots]

        traces = {}

        vta_rates = [s.vta_firing_rate for s in self._snapshots if s.vta_firing_rate is not None]
        if vta_rates:
            vta_times = [s.timestamp for s in self._snapshots if s.vta_firing_rate is not None]
            traces["VTA"] = (vta_times, vta_rates)

        raphe_rates = [s.raphe_firing_rate for s in self._snapshots if s.raphe_firing_rate is not None]
        if raphe_rates:
            raphe_times = [s.timestamp for s in self._snapshots if s.raphe_firing_rate is not None]
            traces["Raphe"] = (raphe_times, raphe_rates)

        return traces

    # -------------------------------------------------------------------------
    # Statistical Analysis
    # -------------------------------------------------------------------------

    def compute_statistics(self, window: int | None = None) -> NTStatistics:
        """
        Compute comprehensive statistics over recent history.

        Args:
            window: Number of recent snapshots (default: all)

        Returns:
            NTStatistics with mean, std, correlations, autocorrelation
        """
        if not self._nt_history:
            return NTStatistics(
                mean=np.zeros(6),
                std=np.zeros(6),
                min=np.zeros(6),
                max=np.ones(6),
                correlation_matrix=np.eye(6),
                autocorrelation=np.zeros(6),
            )

        # Get window of data
        if window is not None:
            data = np.array(self._nt_history[-window:])
        else:
            data = np.array(self._nt_history)

        # Basic statistics
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)

        # Cross-NT correlation matrix
        if len(data) > 1:
            correlation_matrix = np.corrcoef(data.T)
            # Handle NaN (constant columns)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        else:
            correlation_matrix = np.eye(6)

        # Temporal autocorrelation (lag-1)
        autocorrelation = np.zeros(6)
        if len(data) > 2:
            for i in range(6):
                series = data[:, i]
                if np.std(series) > 1e-10:
                    autocorrelation[i] = np.corrcoef(series[:-1], series[1:])[0, 1]

        return NTStatistics(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            min=min_vals.astype(np.float32),
            max=max_vals.astype(np.float32),
            correlation_matrix=correlation_matrix.astype(np.float32),
            autocorrelation=np.nan_to_num(autocorrelation).astype(np.float32),
        )

    def get_correlation_matrix(self) -> np.ndarray:
        """Get cross-NT correlation matrix."""
        stats = self.compute_statistics()
        return stats.correlation_matrix

    def get_autocorrelation(self) -> np.ndarray:
        """Get temporal autocorrelation for each NT."""
        stats = self.compute_statistics()
        return stats.autocorrelation

    # -------------------------------------------------------------------------
    # Receptor Saturation Analysis
    # -------------------------------------------------------------------------

    def compute_saturation_curve(
        self,
        nt_index: int,
        concentration_range: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute receptor saturation curve for a specific NT.

        Args:
            nt_index: Index of NT (0=DA, 1=5-HT, etc.)
            concentration_range: Range of concentrations (default: 0-1)

        Returns:
            (concentrations, saturation_values)
        """
        if concentration_range is None:
            concentration_range = np.linspace(0, 1, 100)

        km = self.km[nt_index]
        saturation = concentration_range / (concentration_range + km)

        return concentration_range, saturation

    def get_all_saturation_curves(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Get saturation curves for all NTs."""
        curves = {}
        for i, label in enumerate(NT_LABELS):
            curves[label] = self.compute_saturation_curve(i)
        return curves

    # -------------------------------------------------------------------------
    # Opponent Process Analysis
    # -------------------------------------------------------------------------

    def get_opponent_processes(self) -> dict[str, tuple[list[datetime], list[float]]]:
        """
        Get opponent process dynamics.

        Returns:
            Dict with:
            - "DA_5HT_ratio": Reward-seeking vs impulse control
            - "Glu_GABA_ratio": Excitation vs inhibition
            - "NE_ACh_balance": Arousal vs attention mode
        """
        if not self._snapshots:
            return {}

        timestamps = [s.timestamp for s in self._snapshots]

        # DA/5-HT ratio (reward vs patience)
        da_5ht = []
        for s in self._snapshots:
            ratio = s.nt_state[0] / max(s.nt_state[1], 0.01)
            da_5ht.append(float(ratio))

        # Glu/GABA ratio (E/I balance)
        glu_gaba = []
        for s in self._snapshots:
            ratio = s.nt_state[5] / max(s.nt_state[4], 0.01)
            glu_gaba.append(float(ratio))

        # NE - ACh (arousal vs attention mode)
        ne_ach = []
        for s in self._snapshots:
            diff = s.nt_state[3] - s.nt_state[2]
            ne_ach.append(float(diff))

        return {
            "DA_5HT_ratio": (timestamps, da_5ht),
            "Glu_GABA_ratio": (timestamps, glu_gaba),
            "NE_ACh_balance": (timestamps, ne_ach),
        }

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_data(self) -> dict:
        """Export all visualization data."""
        current = self.get_current_snapshot()
        stats = self.compute_statistics()

        return {
            "current_state": {
                "nt_values": current.nt_state.tolist() if current else None,
                "deviation": current.setpoint_deviation.tolist() if current else None,
                "saturation": current.receptor_saturation.tolist() if current else None,
                "ei_balance": current.ei_balance if current else None,
                "arousal_index": current.arousal_index if current else None,
            },
            "setpoints": self.setpoints.tolist(),
            "statistics": {
                "mean": stats.mean.tolist(),
                "std": stats.std.tolist(),
                "correlation_matrix": stats.correlation_matrix.tolist(),
                "autocorrelation": stats.autocorrelation.tolist(),
            },
            "nt_labels": NT_LABELS,
            "nt_colors": NT_COLORS,
            "alerts": self.get_alerts(),
            "n_samples": len(self._snapshots),
        }

    def clear_history(self) -> None:
        """Clear all history."""
        self._snapshots.clear()
        self._nt_history.clear()
        self._active_alerts.clear()


# =============================================================================
# Standalone Plot Functions
# =============================================================================


def plot_nt_channels(
    dashboard: NTStateDashboard,
    ax=None,
    show_setpoints: bool = True,
):
    """
    Plot 6-channel NT state as stacked time series.

    Args:
        dashboard: NTStateDashboard instance
        ax: Matplotlib axes (creates 6 subplots if None)
        show_setpoints: Show homeostatic setpoint lines

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    traces = dashboard.get_nt_traces()
    if not traces:
        logger.warning("No data to plot")
        return None

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

    for i, label in enumerate(NT_LABELS):
        ax = axes[i]
        timestamps, values = traces[label]

        # Convert to relative seconds
        t0 = timestamps[0]
        t_seconds = [(t - t0).total_seconds() for t in timestamps]

        # Plot NT trace
        color = NT_COLORS[label]
        ax.plot(t_seconds, values, color=color, linewidth=1.5, label=label)
        ax.fill_between(t_seconds, 0, values, color=color, alpha=0.2)

        # Setpoint line
        if show_setpoints:
            setpoint = dashboard.setpoints[i]
            ax.axhline(y=setpoint, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("6-Channel NT State", fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def plot_deviation_heatmap(
    dashboard: NTStateDashboard,
    ax=None,
    window: int = 100,
):
    """
    Plot setpoint deviation as heatmap over time.

    Args:
        dashboard: NTStateDashboard instance
        ax: Matplotlib axes
        window: Number of recent samples to show

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    if len(dashboard._snapshots) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    # Get recent deviations
    recent = dashboard._snapshots[-window:]
    deviations = np.array([s.setpoint_deviation for s in recent]).T  # [6, T]

    # Plot heatmap
    im = ax.imshow(
        deviations,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        origin="upper",
    )

    plt.colorbar(im, ax=ax, label="Deviation from Setpoint")

    ax.set_yticks(range(6))
    ax.set_yticklabels(NT_LABELS)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Neuromodulator")
    ax.set_title("Homeostatic Deviation Over Time")

    return ax


def plot_saturation_curves(
    dashboard: NTStateDashboard,
    ax=None,
    show_current: bool = True,
):
    """
    Plot receptor saturation curves for all NTs.

    Args:
        dashboard: NTStateDashboard instance
        ax: Matplotlib axes
        show_current: Mark current concentration on curves

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    curves = dashboard.get_all_saturation_curves()
    current = dashboard.get_current_state()

    for i, label in enumerate(NT_LABELS):
        conc, sat = curves[label]
        color = NT_COLORS[label]
        ax.plot(conc, sat, color=color, linewidth=2, label=label)

        # Mark current position
        if show_current and current is not None:
            current_conc = current[i]
            current_sat = current_conc / (current_conc + dashboard.km[i])
            ax.scatter([current_conc], [current_sat], color=color, s=80, zorder=5,
                      edgecolors="black", linewidths=1.5)

    ax.set_xlabel("NT Concentration")
    ax.set_ylabel("Receptor Saturation")
    ax.set_title("Receptor Saturation Curves (Michaelis-Menten)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def plot_correlation_matrix(
    dashboard: NTStateDashboard,
    ax=None,
):
    """
    Plot cross-NT correlation matrix.

    Args:
        dashboard: NTStateDashboard instance
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
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    corr = dashboard.get_correlation_matrix()

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(NT_LABELS)
    ax.set_yticklabels(NT_LABELS)

    # Annotate values
    for i in range(6):
        for j in range(6):
            val = corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   fontsize=9, color=color)

    ax.set_title("Cross-NT Correlation Matrix")

    return ax


def plot_autocorrelation(
    dashboard: NTStateDashboard,
    ax=None,
):
    """
    Plot temporal autocorrelation for each NT.

    Args:
        dashboard: NTStateDashboard instance
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    autocorr = dashboard.get_autocorrelation()
    colors = [NT_COLORS[label] for label in NT_LABELS]

    bars = ax.bar(NT_LABELS, autocorr, color=colors, edgecolor="black", linewidth=1.5)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=1)
    ax.set_ylabel("Autocorrelation (lag-1)")
    ax.set_title("Temporal Persistence of NT Dynamics")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate values
    for bar, val in zip(bars, autocorr):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
               f"{val:.2f}", ha="center", fontsize=9)

    return ax


def plot_opponent_processes(
    dashboard: NTStateDashboard,
    ax=None,
):
    """
    Plot opponent process dynamics (DA/5-HT, Glu/GABA).

    Args:
        dashboard: NTStateDashboard instance
        ax: Matplotlib axes (creates 3 subplots if None)

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    processes = dashboard.get_opponent_processes()
    if not processes:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    titles = {
        "DA_5HT_ratio": "DA/5-HT Ratio (Reward vs Patience)",
        "Glu_GABA_ratio": "Glu/GABA Ratio (E/I Balance)",
        "NE_ACh_balance": "NE - ACh (Arousal vs Attention)",
    }

    colors = {
        "DA_5HT_ratio": "#FF6B6B",
        "Glu_GABA_ratio": "#2ECC71",
        "NE_ACh_balance": "#FFA07A",
    }

    for i, (key, (timestamps, values)) in enumerate(processes.items()):
        ax = axes[i]

        t0 = timestamps[0]
        t_seconds = [(t - t0).total_seconds() for t in timestamps]

        ax.plot(t_seconds, values, color=colors[key], linewidth=1.5)

        # Reference line
        if "ratio" in key:
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        else:
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_ylabel(titles[key], fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Opponent Process Dynamics", fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def create_nt_dashboard(
    dashboard: NTStateDashboard,
    figsize: tuple[int, int] = (16, 14),
):
    """
    Create comprehensive NT state dashboard.

    Args:
        dashboard: NTStateDashboard instance
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
    gs = GridSpec(4, 3, figure=fig, height_ratios=[2, 1.5, 1, 1])

    # 6-channel traces (top row, spanning 2 columns)
    ax_traces = fig.add_subplot(gs[0, :2])
    traces = dashboard.get_nt_traces()
    if traces:
        for i, label in enumerate(NT_LABELS):
            timestamps, values = traces[label]
            if timestamps:
                t0 = timestamps[0]
                t_seconds = [(t - t0).total_seconds() for t in timestamps]
                ax_traces.plot(t_seconds, values, color=NT_COLORS[label],
                             linewidth=1.5, label=label)
        ax_traces.legend(loc="upper right", ncol=3)
        ax_traces.set_xlabel("Time (s)")
        ax_traces.set_ylabel("Concentration")
        ax_traces.set_title("6-Channel NT State")
        ax_traces.set_ylim(0, 1)
        ax_traces.grid(True, alpha=0.3)

    # Current state bar chart
    ax_current = fig.add_subplot(gs[0, 2])
    current = dashboard.get_current_state()
    if current is not None:
        colors = [NT_COLORS[label] for label in NT_LABELS]
        ax_current.bar(NT_LABELS, current, color=colors, edgecolor="black")
        ax_current.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
        ax_current.set_ylabel("Concentration")
        ax_current.set_title("Current State")
        ax_current.set_ylim(0, 1)

    # Deviation heatmap
    ax_dev = fig.add_subplot(gs[1, :2])
    plot_deviation_heatmap(dashboard, ax=ax_dev, window=100)

    # Saturation curves
    ax_sat = fig.add_subplot(gs[1, 2])
    plot_saturation_curves(dashboard, ax=ax_sat)

    # Correlation matrix
    ax_corr = fig.add_subplot(gs[2, 0])
    plot_correlation_matrix(dashboard, ax=ax_corr)

    # Autocorrelation
    ax_auto = fig.add_subplot(gs[2, 1:])
    plot_autocorrelation(dashboard, ax=ax_auto)

    # E/I balance and arousal
    ax_ei = fig.add_subplot(gs[3, :2])
    timestamps_ei, ei_values = dashboard.get_ei_balance_trace()
    timestamps_ar, ar_values = dashboard.get_arousal_trace()
    if timestamps_ei:
        t0 = timestamps_ei[0]
        t_seconds = [(t - t0).total_seconds() for t in timestamps_ei]
        ax_ei.plot(t_seconds, ei_values, color="#9B59B6", linewidth=1.5, label="E/I Balance")
        ax_ei.plot(t_seconds, ar_values, color="#FFA07A", linewidth=1.5, label="Arousal")
        ax_ei.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax_ei.legend()
        ax_ei.set_xlabel("Time (s)")
        ax_ei.set_ylabel("Value")
        ax_ei.set_title("E/I Balance & Arousal")
        ax_ei.grid(True, alpha=0.3)

    # Metrics text box
    ax_metrics = fig.add_subplot(gs[3, 2])
    ax_metrics.axis("off")

    stats = dashboard.compute_statistics()
    alerts = dashboard.get_alerts()
    snapshot = dashboard.get_current_snapshot()

    text_lines = [
        "NT Statistics",
        "-" * 30,
    ]

    for i, label in enumerate(NT_LABELS):
        text_lines.append(f"{label}: {stats.mean[i]:.2f} +/- {stats.std[i]:.2f}")

    if snapshot:
        text_lines.extend([
            "",
            "Current Metrics",
            "-" * 30,
            f"E/I Balance: {snapshot.ei_balance:.2f}",
            f"Arousal Index: {snapshot.arousal_index:.2f}",
        ])

    if alerts:
        text_lines.extend(["", "Alerts", "-" * 30])
        text_lines.extend(alerts[:4])

    text = "\n".join(text_lines)
    ax_metrics.text(0.05, 0.95, text, fontsize=9, family="monospace",
                   verticalalignment="top", transform=ax_metrics.transAxes)

    plt.tight_layout()
    return fig


__all__ = [
    "NTStateDashboard",
    "NTSnapshot",
    "NTStatistics",
    "NT_LABELS",
    "NT_COLORS",
    "HOMEOSTATIC_SETPOINTS",
    "RECEPTOR_KM",
    "plot_nt_channels",
    "plot_deviation_heatmap",
    "plot_saturation_curves",
    "plot_correlation_matrix",
    "plot_autocorrelation",
    "plot_opponent_processes",
    "create_nt_dashboard",
]
