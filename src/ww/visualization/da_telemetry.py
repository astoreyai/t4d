"""
Dopamine Temporal Structure Telemetry for World Weaver NCA.

Visualizes dopamine dynamics with biological temporal structure:
- Phasic bursts (20-40 Hz) for positive RPE
- Phasic pauses (0-2 Hz) for negative RPE
- Tonic baseline (4-5 Hz) for motivation
- RPE temporal dynamics (ramping, prediction error)
- TD error traces and eligibility decay

Biological References:
- Schultz et al. (1997): Dopamine neurons encode prediction error
- Cohen et al. (2012): Dopamine ramps during goal-directed navigation
- Steinberg et al. (2013): Causal link between DA and TD error
- Hamid et al. (2016): Mesolimbic DA signals value

Author: Claude Opus 4.5
Date: 2026-01-01
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ww.learning.dopamine import RewardPredictionError
    from ww.nca.vta import VTAState

logger = logging.getLogger(__name__)


class DASignalType(Enum):
    """Classification of dopamine signal types."""

    TONIC = "tonic"  # Baseline sustained firing
    PHASIC_BURST = "phasic_burst"  # Positive RPE burst
    PHASIC_PAUSE = "phasic_pause"  # Negative RPE pause
    RAMP = "ramp"  # Gradual increase toward expected reward
    DECAY = "decay"  # Return to baseline after phasic event


@dataclass
class DASnapshot:
    """Snapshot of dopamine system state."""

    timestamp: datetime
    da_level: float  # Current DA concentration [0, 1]
    firing_rate: float  # Hz (tonic ~4-5, burst ~20-40)
    signal_type: DASignalType
    rpe: float  # Reward prediction error
    td_error: float  # Temporal difference error
    value_estimate: float  # V(s) current state value
    eligibility: float  # Eligibility trace level


@dataclass
class DARampEvent:
    """Record of dopamine ramping event."""

    start_time: datetime
    end_time: datetime | None = None
    start_da: float = 0.3
    peak_da: float = 0.3
    duration_s: float = 0.0
    goal_reached: bool = False
    distance_to_goal: float = 1.0  # Normalized distance


@dataclass
class DAStatistics:
    """Statistical summary of DA dynamics."""

    mean_da: float
    std_da: float
    mean_firing_rate: float
    phasic_burst_count: int
    phasic_pause_count: int
    ramp_count: int
    mean_rpe: float
    mean_td_error: float
    time_in_burst: float  # Fraction of time
    time_in_pause: float
    time_in_tonic: float


class DATelemetry:
    """
    Dopamine temporal structure telemetry and analysis.

    Tracks DA signaling patterns with emphasis on:
    1. Phasic vs tonic firing mode distribution
    2. RPE temporal structure (bursts, pauses, ramps)
    3. TD error dynamics for learning
    4. Eligibility trace visualization
    5. Value function evolution

    Biological Validation:
    - Burst rate should be 20-40 Hz for positive RPE
    - Pause rate should be 0-2 Hz for negative RPE
    - Tonic rate should be 4-5 Hz at baseline
    - RPE should follow Schultz et al. (1997) patterns
    """

    def __init__(
        self,
        window_size: int = 1000,
        tonic_rate_range: tuple[float, float] = (3.0, 6.0),
        burst_rate_threshold: float = 15.0,
        pause_rate_threshold: float = 2.0,
        ramp_detection_threshold: float = 0.1,
    ):
        """
        Initialize DA telemetry.

        Args:
            window_size: Number of snapshots to retain
            tonic_rate_range: Expected tonic firing rate range (Hz)
            burst_rate_threshold: Min rate to classify as burst (Hz)
            pause_rate_threshold: Max rate to classify as pause (Hz)
            ramp_detection_threshold: Min slope for ramp detection
        """
        self.window_size = window_size
        self.tonic_rate_range = tonic_rate_range
        self.burst_rate_threshold = burst_rate_threshold
        self.pause_rate_threshold = pause_rate_threshold
        self.ramp_detection_threshold = ramp_detection_threshold

        # History
        self._snapshots: list[DASnapshot] = []
        self._ramp_events: list[DARampEvent] = []
        self._current_ramp: DARampEvent | None = None

        # Running statistics
        self._total_bursts = 0
        self._total_pauses = 0
        self._total_ramps = 0

        logger.info("DATelemetry initialized")

    def record_state(
        self,
        da_level: float,
        firing_rate: float,
        rpe: float = 0.0,
        td_error: float = 0.0,
        value_estimate: float = 0.5,
        eligibility: float = 0.0,
    ) -> DASnapshot:
        """
        Record current DA system state.

        Args:
            da_level: Current DA concentration [0, 1]
            firing_rate: Current firing rate (Hz)
            rpe: Reward prediction error
            td_error: TD error (δ)
            value_estimate: Current value function estimate
            eligibility: Current eligibility trace level

        Returns:
            DASnapshot with classified signal type
        """
        now = datetime.now()

        # Classify signal type
        signal_type = self._classify_signal(
            da_level, firing_rate, rpe
        )

        snapshot = DASnapshot(
            timestamp=now,
            da_level=da_level,
            firing_rate=firing_rate,
            signal_type=signal_type,
            rpe=rpe,
            td_error=td_error,
            value_estimate=value_estimate,
            eligibility=eligibility,
        )

        # Track phasic events
        if signal_type == DASignalType.PHASIC_BURST:
            self._total_bursts += 1
        elif signal_type == DASignalType.PHASIC_PAUSE:
            self._total_pauses += 1

        # Detect ramping
        self._update_ramp_detection(snapshot)

        # Store snapshot
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

        return snapshot

    def record_from_vta(
        self,
        vta_state: VTAState,
    ) -> DASnapshot:
        """
        Record state from VTA circuit.

        Args:
            vta_state: VTAState from vta.py

        Returns:
            DASnapshot
        """
        return self.record_state(
            da_level=vta_state.current_da,
            firing_rate=vta_state.current_rate,
            rpe=vta_state.last_rpe,
            td_error=vta_state.td_error,
            value_estimate=vta_state.value_estimate,
            eligibility=vta_state.eligibility,
        )

    def record_rpe_event(
        self,
        rpe_record: RewardPredictionError,
        da_level: float,
        firing_rate: float,
    ) -> DASnapshot:
        """
        Record RPE event from DopamineSystem.

        Args:
            rpe_record: RewardPredictionError from dopamine.py
            da_level: Current DA level
            firing_rate: Current firing rate

        Returns:
            DASnapshot
        """
        return self.record_state(
            da_level=da_level,
            firing_rate=firing_rate,
            rpe=rpe_record.rpe,
            td_error=rpe_record.rpe,  # RPE approximates TD error
            value_estimate=rpe_record.expected,
            eligibility=0.0,
        )

    def _classify_signal(
        self,
        da_level: float,
        firing_rate: float,
        rpe: float,
    ) -> DASignalType:
        """Classify DA signal type based on firing rate and RPE."""
        # Check for phasic burst (high rate, positive RPE)
        if firing_rate >= self.burst_rate_threshold and rpe > 0.05:
            return DASignalType.PHASIC_BURST

        # Check for phasic pause (low rate, negative RPE)
        if firing_rate <= self.pause_rate_threshold and rpe < -0.05:
            return DASignalType.PHASIC_PAUSE

        # Check for ramping (gradual increase)
        if len(self._snapshots) >= 3:
            recent_da = [s.da_level for s in self._snapshots[-3:]]
            slope = (recent_da[-1] - recent_da[0]) / 3.0
            if slope > self.ramp_detection_threshold:
                return DASignalType.RAMP
            elif slope < -self.ramp_detection_threshold:
                return DASignalType.DECAY

        # Default to tonic
        return DASignalType.TONIC

    def _update_ramp_detection(self, snapshot: DASnapshot) -> None:
        """Track ramping events (Cohen et al. 2012)."""
        if snapshot.signal_type == DASignalType.RAMP:
            if self._current_ramp is None:
                # Start new ramp
                self._current_ramp = DARampEvent(
                    start_time=snapshot.timestamp,
                    start_da=snapshot.da_level,
                )
            else:
                # Update current ramp
                self._current_ramp.peak_da = max(
                    self._current_ramp.peak_da, snapshot.da_level
                )
        else:
            # End current ramp if any
            if self._current_ramp is not None:
                self._current_ramp.end_time = snapshot.timestamp
                self._current_ramp.duration_s = (
                    snapshot.timestamp - self._current_ramp.start_time
                ).total_seconds()
                self._current_ramp.goal_reached = (
                    snapshot.signal_type == DASignalType.PHASIC_BURST
                )
                self._ramp_events.append(self._current_ramp)
                self._total_ramps += 1
                self._current_ramp = None

                # Trim ramp history
                if len(self._ramp_events) > 100:
                    self._ramp_events = self._ramp_events[-100:]

    # -------------------------------------------------------------------------
    # Analysis Methods
    # -------------------------------------------------------------------------

    def get_statistics(self) -> DAStatistics:
        """Compute comprehensive DA statistics."""
        if not self._snapshots:
            return DAStatistics(
                mean_da=0.3,
                std_da=0.0,
                mean_firing_rate=4.5,
                phasic_burst_count=0,
                phasic_pause_count=0,
                ramp_count=0,
                mean_rpe=0.0,
                mean_td_error=0.0,
                time_in_burst=0.0,
                time_in_pause=0.0,
                time_in_tonic=1.0,
            )

        da_levels = [s.da_level for s in self._snapshots]
        rates = [s.firing_rate for s in self._snapshots]
        rpes = [s.rpe for s in self._snapshots]
        td_errors = [s.td_error for s in self._snapshots]
        types = [s.signal_type for s in self._snapshots]

        n_total = len(types)
        n_burst = sum(1 for t in types if t == DASignalType.PHASIC_BURST)
        n_pause = sum(1 for t in types if t == DASignalType.PHASIC_PAUSE)
        n_tonic = sum(1 for t in types if t == DASignalType.TONIC)

        return DAStatistics(
            mean_da=float(np.mean(da_levels)),
            std_da=float(np.std(da_levels)),
            mean_firing_rate=float(np.mean(rates)),
            phasic_burst_count=self._total_bursts,
            phasic_pause_count=self._total_pauses,
            ramp_count=self._total_ramps,
            mean_rpe=float(np.mean(rpes)),
            mean_td_error=float(np.mean(td_errors)),
            time_in_burst=n_burst / n_total,
            time_in_pause=n_pause / n_total,
            time_in_tonic=n_tonic / n_total,
        )

    def get_rpe_distribution(self) -> dict:
        """Get RPE value distribution."""
        if not self._snapshots:
            return {}

        rpes = [s.rpe for s in self._snapshots]
        positive = [r for r in rpes if r > 0.05]
        negative = [r for r in rpes if r < -0.05]

        return {
            "mean": float(np.mean(rpes)),
            "std": float(np.std(rpes)),
            "min": float(np.min(rpes)),
            "max": float(np.max(rpes)),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "neutral_count": len(rpes) - len(positive) - len(negative),
            "positive_mean": float(np.mean(positive)) if positive else 0.0,
            "negative_mean": float(np.mean(negative)) if negative else 0.0,
        }

    def get_firing_rate_distribution(self) -> dict:
        """Get firing rate distribution by signal type."""
        if not self._snapshots:
            return {}

        rates_by_type: dict[str, list[float]] = {
            "tonic": [],
            "burst": [],
            "pause": [],
            "ramp": [],
        }

        for s in self._snapshots:
            if s.signal_type == DASignalType.TONIC:
                rates_by_type["tonic"].append(s.firing_rate)
            elif s.signal_type == DASignalType.PHASIC_BURST:
                rates_by_type["burst"].append(s.firing_rate)
            elif s.signal_type == DASignalType.PHASIC_PAUSE:
                rates_by_type["pause"].append(s.firing_rate)
            elif s.signal_type == DASignalType.RAMP:
                rates_by_type["ramp"].append(s.firing_rate)

        result = {}
        for signal_type, rates in rates_by_type.items():
            if rates:
                result[signal_type] = {
                    "mean": float(np.mean(rates)),
                    "std": float(np.std(rates)),
                    "count": len(rates),
                }
        return result

    def get_ramp_statistics(self) -> dict:
        """Analyze ramping events (Cohen et al. 2012 style)."""
        if not self._ramp_events:
            return {
                "total_ramps": 0,
                "goal_reached_fraction": 0.0,
            }

        durations = [r.duration_s for r in self._ramp_events if r.duration_s > 0]
        amplitudes = [r.peak_da - r.start_da for r in self._ramp_events]
        goal_reached = [r for r in self._ramp_events if r.goal_reached]

        return {
            "total_ramps": len(self._ramp_events),
            "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
            "mean_amplitude": float(np.mean(amplitudes)) if amplitudes else 0.0,
            "goal_reached_fraction": len(goal_reached) / len(self._ramp_events),
        }

    def get_eligibility_trace(self) -> list[float]:
        """Get eligibility trace over time."""
        return [s.eligibility for s in self._snapshots]

    def get_value_trace(self) -> list[float]:
        """Get value function estimates over time."""
        return [s.value_estimate for s in self._snapshots]

    def get_td_error_trace(self) -> list[float]:
        """Get TD error trace."""
        return [s.td_error for s in self._snapshots]

    def validate_biological_ranges(self) -> dict:
        """
        Validate against experimental literature.

        Schultz et al. (1997) patterns:
        - Tonic: 4-5 Hz
        - Burst: 20-40 Hz, 100-200 ms duration
        - Pause: 0-2 Hz, 200-400 ms duration
        """
        stats = self.get_statistics()
        rate_dist = self.get_firing_rate_distribution()

        validation = {
            "tonic_rate_in_range": False,
            "burst_rate_in_range": False,
            "pause_rate_in_range": False,
            "overall_valid": False,
        }

        # Check tonic rate (3-6 Hz acceptable)
        if "tonic" in rate_dist:
            tonic_mean = rate_dist["tonic"]["mean"]
            validation["tonic_rate_in_range"] = (
                self.tonic_rate_range[0] <= tonic_mean <= self.tonic_rate_range[1]
            )
            validation["tonic_rate_hz"] = tonic_mean

        # Check burst rate (15-50 Hz acceptable)
        if "burst" in rate_dist:
            burst_mean = rate_dist["burst"]["mean"]
            validation["burst_rate_in_range"] = 15.0 <= burst_mean <= 50.0
            validation["burst_rate_hz"] = burst_mean

        # Check pause rate (0-3 Hz acceptable)
        if "pause" in rate_dist:
            pause_mean = rate_dist["pause"]["mean"]
            validation["pause_rate_in_range"] = 0.0 <= pause_mean <= 3.0
            validation["pause_rate_hz"] = pause_mean

        # Overall validity
        checks = [
            validation.get("tonic_rate_in_range", True),
            validation.get("burst_rate_in_range", True),
            validation.get("pause_rate_in_range", True),
        ]
        validation["overall_valid"] = all(checks)

        return validation

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_da_timeline(self, ax=None):
        """
        Plot DA concentration and firing rate over time.

        Mimics in vivo voltammetry + electrophysiology recordings.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        if not self._snapshots:
            ax[0].text(0.5, 0.5, "No data", ha="center", va="center")
            return ax

        # Time axis
        t0 = self._snapshots[0].timestamp
        times = [(s.timestamp - t0).total_seconds() for s in self._snapshots]
        da_levels = [s.da_level for s in self._snapshots]
        rates = [s.firing_rate for s in self._snapshots]
        types = [s.signal_type for s in self._snapshots]

        # Color-code by signal type
        colors = []
        for t in types:
            if t == DASignalType.PHASIC_BURST:
                colors.append("red")
            elif t == DASignalType.PHASIC_PAUSE:
                colors.append("blue")
            elif t == DASignalType.RAMP:
                colors.append("orange")
            else:
                colors.append("gray")

        # Top: DA concentration
        ax[0].scatter(times, da_levels, c=colors, s=10, alpha=0.7)
        ax[0].axhline(y=0.3, color="green", linestyle="--", alpha=0.5, label="Tonic baseline")
        ax[0].set_ylabel("DA Concentration")
        ax[0].set_title("Dopamine Temporal Structure")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # Bottom: Firing rate
        ax[1].scatter(times, rates, c=colors, s=10, alpha=0.7)
        ax[1].axhline(y=4.5, color="green", linestyle="--", alpha=0.5, label="Tonic rate")
        ax[1].axhline(y=self.burst_rate_threshold, color="red", linestyle=":", alpha=0.5)
        ax[1].axhline(y=self.pause_rate_threshold, color="blue", linestyle=":", alpha=0.5)
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Firing Rate (Hz)")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        return ax

    def plot_rpe_distribution(self, ax=None):
        """Plot RPE histogram with Schultz-style visualization."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return ax

        rpes = [s.rpe for s in self._snapshots]

        # Histogram with color-coded regions
        bins = np.linspace(-1, 1, 41)
        n, bins_out, patches = ax.hist(
            rpes, bins=bins, edgecolor="black", alpha=0.7
        )

        # Color bars by value
        for i, (patch, b) in enumerate(zip(patches, bins_out[:-1])):
            if b > 0.05:
                patch.set_facecolor("red")
            elif b < -0.05:
                patch.set_facecolor("blue")
            else:
                patch.set_facecolor("gray")

        ax.axvline(x=0, color="black", linestyle="-", linewidth=2)
        ax.axvline(x=0.05, color="red", linestyle="--", alpha=0.5)
        ax.axvline(x=-0.05, color="blue", linestyle="--", alpha=0.5)

        ax.set_xlabel("Reward Prediction Error (δ)")
        ax.set_ylabel("Count")
        ax.set_title("RPE Distribution (Schultz et al. 1997 style)")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_phasic_raster(self, ax=None):
        """Plot phasic events as raster (burst/pause times)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return ax

        t0 = self._snapshots[0].timestamp
        burst_times = []
        pause_times = []
        ramp_times = []

        for s in self._snapshots:
            t = (s.timestamp - t0).total_seconds()
            if s.signal_type == DASignalType.PHASIC_BURST:
                burst_times.append(t)
            elif s.signal_type == DASignalType.PHASIC_PAUSE:
                pause_times.append(t)
            elif s.signal_type == DASignalType.RAMP:
                ramp_times.append(t)

        # Raster plot
        if burst_times:
            ax.eventplot([burst_times], lineoffsets=2, colors="red", label="Burst")
        if pause_times:
            ax.eventplot([pause_times], lineoffsets=1, colors="blue", label="Pause")
        if ramp_times:
            ax.eventplot([ramp_times], lineoffsets=0, colors="orange", label="Ramp")

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Ramp", "Pause", "Burst"])
        ax.set_xlabel("Time (s)")
        ax.set_title("Phasic DA Events")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")

        return ax

    def plot_eligibility_decay(self, ax=None):
        """Plot eligibility trace decay (TD(λ) visualization)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return ax

        t0 = self._snapshots[0].timestamp
        times = [(s.timestamp - t0).total_seconds() for s in self._snapshots]
        eligibility = [s.eligibility for s in self._snapshots]
        td_errors = [s.td_error for s in self._snapshots]

        ax.fill_between(times, eligibility, alpha=0.3, color="purple", label="Eligibility")
        ax.plot(times, eligibility, color="purple", linewidth=2)

        # Overlay TD errors as stems
        ax2 = ax.twinx()
        ax2.stem(times, td_errors, linefmt="r-", markerfmt="ro", basefmt=" ", label="TD Error")
        ax2.set_ylabel("TD Error (δ)", color="red")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Eligibility Trace", color="purple")
        ax.set_title("Eligibility Trace Decay (TD(λ))")
        ax.grid(True, alpha=0.3)

        return ax

    def create_da_dashboard(self, fig=None):
        """Create comprehensive DA telemetry dashboard."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if fig is None:
            fig = plt.figure(figsize=(16, 12))

        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Top row: Timeline (spans both columns)
        ax_timeline = fig.add_subplot(gs[0, :])
        if self._snapshots:
            t0 = self._snapshots[0].timestamp
            times = [(s.timestamp - t0).total_seconds() for s in self._snapshots]
            da_levels = [s.da_level for s in self._snapshots]
            ax_timeline.plot(times, da_levels, "k-", linewidth=1)
            ax_timeline.axhline(y=0.3, color="green", linestyle="--", alpha=0.5)
            ax_timeline.set_ylabel("DA Level")
            ax_timeline.set_title("Dopamine Timeline")
            ax_timeline.grid(True, alpha=0.3)

        # Middle left: RPE distribution
        ax_rpe = fig.add_subplot(gs[1, 0])
        self.plot_rpe_distribution(ax=ax_rpe)

        # Middle right: Phasic raster
        ax_raster = fig.add_subplot(gs[1, 1])
        self.plot_phasic_raster(ax=ax_raster)

        # Bottom left: Firing rate by type
        ax_rates = fig.add_subplot(gs[2, 0])
        rate_dist = self.get_firing_rate_distribution()
        if rate_dist:
            types = list(rate_dist.keys())
            means = [rate_dist[t]["mean"] for t in types]
            colors = {"tonic": "gray", "burst": "red", "pause": "blue", "ramp": "orange"}
            bar_colors = [colors.get(t, "gray") for t in types]
            ax_rates.bar(types, means, color=bar_colors, edgecolor="black")
            ax_rates.set_ylabel("Mean Firing Rate (Hz)")
            ax_rates.set_title("Firing Rate by Signal Type")
            ax_rates.grid(True, alpha=0.3, axis="y")

        # Bottom right: Statistics text
        ax_stats = fig.add_subplot(gs[2, 1])
        ax_stats.axis("off")
        stats = self.get_statistics()
        validation = self.validate_biological_ranges()
        text = (
            f"DA Statistics\n"
            f"─────────────────\n"
            f"Mean DA: {stats.mean_da:.3f} ± {stats.std_da:.3f}\n"
            f"Mean Rate: {stats.mean_firing_rate:.1f} Hz\n"
            f"Bursts: {stats.phasic_burst_count}\n"
            f"Pauses: {stats.phasic_pause_count}\n"
            f"Ramps: {stats.ramp_count}\n"
            f"Mean RPE: {stats.mean_rpe:.3f}\n\n"
            f"Validation\n"
            f"─────────────────\n"
            f"Tonic valid: {'✓' if validation.get('tonic_rate_in_range', False) else '✗'}\n"
            f"Burst valid: {'✓' if validation.get('burst_rate_in_range', False) else '✗'}\n"
            f"Pause valid: {'✓' if validation.get('pause_rate_in_range', False) else '✗'}"
        )
        ax_stats.text(
            0.1, 0.9, text, transform=ax_stats.transAxes,
            fontsize=10, fontfamily="monospace", verticalalignment="top"
        )

        fig.suptitle("Dopamine Telemetry Dashboard", fontsize=14, fontweight="bold")

        return fig

    def export_data(self) -> dict:
        """Export telemetry data for external analysis."""
        return {
            "snapshots": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "da_level": s.da_level,
                    "firing_rate": s.firing_rate,
                    "signal_type": s.signal_type.value,
                    "rpe": s.rpe,
                    "td_error": s.td_error,
                    "value_estimate": s.value_estimate,
                    "eligibility": s.eligibility,
                }
                for s in self._snapshots
            ],
            "statistics": {
                "general": self.get_statistics().__dict__
                if self._snapshots
                else {},
                "rpe": self.get_rpe_distribution(),
                "firing_rates": self.get_firing_rate_distribution(),
                "ramps": self.get_ramp_statistics(),
            },
            "validation": self.validate_biological_ranges(),
        }

    def clear(self) -> None:
        """Clear all history."""
        self._snapshots.clear()
        self._ramp_events.clear()
        self._current_ramp = None
        self._total_bursts = 0
        self._total_pauses = 0
        self._total_ramps = 0


# Convenience functions
def create_da_dashboard(telemetry: DATelemetry):
    """Create DA telemetry dashboard."""
    return telemetry.create_da_dashboard()


def plot_da_timeline(telemetry: DATelemetry, ax=None):
    """Plot DA timeline."""
    return telemetry.plot_da_timeline(ax=ax)


def plot_rpe_distribution(telemetry: DATelemetry, ax=None):
    """Plot RPE distribution."""
    return telemetry.plot_rpe_distribution(ax=ax)


__all__ = [
    "DATelemetry",
    "DASnapshot",
    "DASignalType",
    "DARampEvent",
    "DAStatistics",
    "create_da_dashboard",
    "plot_da_timeline",
    "plot_rpe_distribution",
]
