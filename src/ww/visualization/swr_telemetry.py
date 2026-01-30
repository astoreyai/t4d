"""
SWR Telemetry for World Weaver NCA.

Visualizes sharp-wave ripple events for consolidation monitoring:
- Ripple frequency (150-250 Hz)
- Sharp-wave amplitude (CA3 burst)
- Replay sequences (reactivated patterns)
- Inter-event intervals
- Temporal compression factor

Biological References:
- Buzsaki (2015): Hippocampal Sharp Wave-Ripple
- Foster & Wilson (2006): Reverse replay of behavioural sequences
- Girardeau et al. (2009): Selective suppression impairs consolidation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ww.nca.swr_coupling import SWREvent as NCASwrEvent
    from ww.nca.swr_coupling import SWRNeuralFieldCoupling

logger = logging.getLogger(__name__)


@dataclass
class SWRTelemetryEvent:
    """Single SWR event record for telemetry."""

    timestamp: datetime
    duration_s: float  # seconds (typically 0.05-0.15)
    ripple_frequency: float  # Hz (150-250 range)
    peak_amplitude: float  # 0-1 normalized
    compression_factor: float  # Replay speed (typically 10x)
    reactivated_patterns: list[str] = field(default_factory=list)
    replay_count: int = 0
    phase_at_record: str = "RIPPLING"  # QUIESCENT, INITIATING, RIPPLING, TERMINATING

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_s * 1000


class SWRTelemetry:
    """
    Sharp-wave ripple telemetry and analysis.

    Tracks SWR events for consolidation monitoring.
    Validates against experimental literature (Buzsaki lab data).
    """

    def __init__(
        self,
        swr_coupling: SWRNeuralFieldCoupling | None = None,
        window_size: int = 1000,
        min_ripple_freq: float = 150.0,
        max_ripple_freq: float = 250.0,
    ):
        """
        Initialize SWR telemetry.

        Args:
            swr_coupling: SWRNeuralFieldCoupling module (from nca/)
            window_size: Number of events to retain
            min_ripple_freq: Minimum ripple frequency (Hz)
            max_ripple_freq: Maximum ripple frequency (Hz)
        """
        self.swr_coupling = swr_coupling
        self.window_size = window_size
        self.min_ripple_freq = min_ripple_freq
        self.max_ripple_freq = max_ripple_freq

        # Event history
        self._events: list[SWRTelemetryEvent] = []
        self._inter_event_intervals: list[float] = []

        # Statistics
        self._total_events = 0
        self._total_replay_time = 0.0
        self._start_time: datetime | None = None

        # Register callback if coupling provided
        if self.swr_coupling is not None:
            self.swr_coupling.register_swr_callback(self._on_swr_complete)

        logger.info(
            f"SWRTelemetry initialized: freq_range=[{min_ripple_freq}, {max_ripple_freq}] Hz"
        )

    def _on_swr_complete(self, nca_event: NCASwrEvent) -> None:
        """Callback when SWR coupling completes an event."""
        self.record_event_from_coupling(nca_event)

    def record_event_from_coupling(
        self,
        nca_event: NCASwrEvent,
        ripple_frequency: float | None = None,
    ) -> SWRTelemetryEvent:
        """
        Record an SWR event from the NCA coupling module.

        Args:
            nca_event: SWREvent from swr_coupling module
            ripple_frequency: Override ripple frequency (else use config)

        Returns:
            SWRTelemetryEvent
        """
        now = datetime.now()

        if self._start_time is None:
            self._start_time = now

        # Get ripple frequency from coupling config or use default
        freq = ripple_frequency
        if freq is None and self.swr_coupling is not None:
            freq = self.swr_coupling.config.ripple_frequency
        freq = freq or 180.0

        # Get compression factor
        compression = 10.0
        if self.swr_coupling is not None:
            compression = self.swr_coupling.config.compression_factor

        # Create telemetry event
        event = SWRTelemetryEvent(
            timestamp=now,
            duration_s=nca_event.duration,
            ripple_frequency=freq,
            peak_amplitude=nca_event.peak_amplitude,
            compression_factor=compression,
            reactivated_patterns=list(nca_event.memories_activated),
            replay_count=nca_event.replay_count,
        )

        return self._store_event(event)

    def record_event(
        self,
        duration_s: float,
        ripple_frequency: float,
        peak_amplitude: float,
        compression_factor: float = 10.0,
        reactivated_patterns: list[str] | None = None,
        replay_count: int = 0,
    ) -> SWRTelemetryEvent:
        """
        Manually record an SWR event.

        Args:
            duration_s: Event duration in seconds
            ripple_frequency: Ripple frequency in Hz
            peak_amplitude: Peak amplitude (0-1)
            compression_factor: Temporal compression factor
            reactivated_patterns: List of replayed pattern IDs
            replay_count: Number of patterns replayed

        Returns:
            SWRTelemetryEvent
        """
        now = datetime.now()

        if self._start_time is None:
            self._start_time = now

        event = SWRTelemetryEvent(
            timestamp=now,
            duration_s=duration_s,
            ripple_frequency=ripple_frequency,
            peak_amplitude=peak_amplitude,
            compression_factor=compression_factor,
            reactivated_patterns=reactivated_patterns or [],
            replay_count=replay_count,
        )

        return self._store_event(event)

    def _store_event(self, event: SWRTelemetryEvent) -> SWRTelemetryEvent:
        """Store event and update statistics."""
        # Validate frequency range
        if not (self.min_ripple_freq <= event.ripple_frequency <= self.max_ripple_freq):
            logger.warning(
                f"SWR frequency {event.ripple_frequency:.1f} Hz outside range "
                f"[{self.min_ripple_freq}, {self.max_ripple_freq}]"
            )

        # Compute inter-event interval
        if self._events:
            interval = (event.timestamp - self._events[-1].timestamp).total_seconds()
            self._inter_event_intervals.append(interval)

        # Store event
        self._events.append(event)
        self._total_events += 1
        self._total_replay_time += event.duration_s

        # Maintain window
        if len(self._events) > self.window_size:
            self._events.pop(0)
        if len(self._inter_event_intervals) > self.window_size:
            self._inter_event_intervals.pop(0)

        logger.debug(
            f"SWR recorded: freq={event.ripple_frequency:.0f}Hz, "
            f"dur={event.duration_ms:.1f}ms, amp={event.peak_amplitude:.2f}"
        )

        return event

    # -------------------------------------------------------------------------
    # Analysis Methods
    # -------------------------------------------------------------------------

    def get_event_rate(self, window_seconds: float = 60.0) -> float:
        """
        Compute SWR event rate (events/second).

        Args:
            window_seconds: Time window for rate calculation

        Returns:
            Event rate (Hz)
        """
        if len(self._events) < 2:
            return 0.0

        now = datetime.now()
        recent_events = [
            e for e in self._events
            if (now - e.timestamp).total_seconds() <= window_seconds
        ]

        if not recent_events:
            return 0.0

        # Use actual time span of events
        if len(recent_events) >= 2:
            span = (recent_events[-1].timestamp - recent_events[0].timestamp).total_seconds()
            if span > 0:
                return (len(recent_events) - 1) / span

        return len(recent_events) / window_seconds

    def get_frequency_distribution(self) -> dict:
        """Get statistics on ripple frequency distribution."""
        if not self._events:
            return {}

        frequencies = np.array([e.ripple_frequency for e in self._events])

        return {
            "mean": float(np.mean(frequencies)),
            "std": float(np.std(frequencies)),
            "min": float(np.min(frequencies)),
            "max": float(np.max(frequencies)),
            "median": float(np.median(frequencies)),
            "n": len(frequencies),
        }

    def get_duration_distribution(self) -> dict:
        """Get statistics on SWR duration (in milliseconds)."""
        if not self._events:
            return {}

        durations = np.array([e.duration_ms for e in self._events])

        return {
            "mean_ms": float(np.mean(durations)),
            "std_ms": float(np.std(durations)),
            "min_ms": float(np.min(durations)),
            "max_ms": float(np.max(durations)),
            "median_ms": float(np.median(durations)),
            "n": len(durations),
        }

    def get_amplitude_distribution(self) -> dict:
        """Get statistics on peak amplitude."""
        if not self._events:
            return {}

        amplitudes = np.array([e.peak_amplitude for e in self._events])

        return {
            "mean": float(np.mean(amplitudes)),
            "std": float(np.std(amplitudes)),
            "min": float(np.min(amplitudes)),
            "max": float(np.max(amplitudes)),
            "median": float(np.median(amplitudes)),
        }

    def get_inter_event_intervals(self) -> dict:
        """Get statistics on inter-SWR intervals."""
        if not self._inter_event_intervals:
            return {}

        intervals = np.array(self._inter_event_intervals)

        return {
            "mean_s": float(np.mean(intervals)),
            "std_s": float(np.std(intervals)),
            "min_s": float(np.min(intervals)),
            "max_s": float(np.max(intervals)),
            "median_s": float(np.median(intervals)),
            "n": len(intervals),
        }

    def get_replay_statistics(self) -> dict:
        """Analyze replay content."""
        if not self._events:
            return {}

        # Count unique patterns replayed
        all_patterns: list[str] = []
        total_replays = 0
        for event in self._events:
            all_patterns.extend(event.reactivated_patterns)
            total_replays += event.replay_count

        unique_patterns = len(set(all_patterns))

        return {
            "total_swr_events": len(self._events),
            "total_replays": total_replays,
            "unique_patterns_replayed": unique_patterns,
            "mean_patterns_per_swr": total_replays / len(self._events) if self._events else 0,
            "total_replay_time_s": self._total_replay_time,
            "mean_compression_factor": float(np.mean([e.compression_factor for e in self._events])),
        }

    def validate_biological_ranges(self) -> dict:
        """
        Validate against experimental literature.

        Biological targets:
        - Frequency: 150-250 Hz (Buzsaki, 2015)
        - Duration: 50-150 ms (Buzsaki, 2015)
        - Event rate: ~0.5-2 Hz during sleep (Girardeau et al. 2009)

        Returns:
            Dict with validation results
        """
        validation: dict[str, Any] = {
            "frequency": {"in_range": False, "value": None, "target": "150-250 Hz"},
            "duration": {"in_range": False, "value": None, "target": "50-150 ms"},
            "event_rate": {"in_range": False, "value": None, "target": "0.5-2 Hz"},
            "overall_valid": False,
        }

        freq_dist = self.get_frequency_distribution()
        dur_dist = self.get_duration_distribution()

        # Frequency: 150-250 Hz
        if freq_dist:
            mean_freq = freq_dist["mean"]
            validation["frequency"]["value"] = mean_freq
            validation["frequency"]["in_range"] = 150 <= mean_freq <= 250

        # Duration: 50-150 ms
        if dur_dist:
            mean_dur = dur_dist["mean_ms"]
            validation["duration"]["value"] = mean_dur
            validation["duration"]["in_range"] = 50 <= mean_dur <= 150

        # Event rate: 0.5-2 Hz during sleep
        event_rate = self.get_event_rate(window_seconds=60.0)
        validation["event_rate"]["value"] = event_rate
        validation["event_rate"]["in_range"] = 0.5 <= event_rate <= 2.0

        # Overall validation
        validation["overall_valid"] = (
            validation["frequency"]["in_range"] and
            validation["duration"]["in_range"] and
            validation["event_rate"]["in_range"]
        )

        return validation

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_swr_raster(self, ax: Any = None) -> Any:
        """
        Plot SWR events as raster (time vs frequency).

        Args:
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
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        if not self._events:
            ax.text(0.5, 0.5, "No SWR events", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Ripple Frequency (Hz)")
            return ax

        # Convert timestamps to relative seconds
        t0 = self._events[0].timestamp
        times = [(e.timestamp - t0).total_seconds() for e in self._events]
        frequencies = [e.ripple_frequency for e in self._events]
        durations = [e.duration_ms for e in self._events]

        # Scatter plot: time vs frequency, size = duration
        scatter = ax.scatter(
            times, frequencies, s=np.array(durations) * 2,
            c=durations, cmap="viridis", alpha=0.7,
            edgecolors="black", linewidths=0.5
        )

        # Mark biological range
        ax.axhspan(150, 250, alpha=0.1, color="green", label="Biological range (150-250 Hz)")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Ripple Frequency (Hz)")
        ax.set_title("SWR Events Over Time")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label="Duration (ms)")

        return ax

    def plot_frequency_histogram(self, ax: Any = None) -> Any:
        """Plot ripple frequency distribution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if not self._events:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        frequencies = [e.ripple_frequency for e in self._events]

        ax.hist(frequencies, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        ax.axvline(x=150, color="green", linestyle="--", linewidth=1.5, label="Min (150 Hz)")
        ax.axvline(x=250, color="red", linestyle="--", linewidth=1.5, label="Max (250 Hz)")
        ax.axvline(x=np.mean(frequencies), color="black", linestyle="-",
                   linewidth=2, label=f"Mean ({np.mean(frequencies):.0f} Hz)")

        ax.set_xlabel("Ripple Frequency (Hz)")
        ax.set_ylabel("Count")
        ax.set_title("Ripple Frequency Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_duration_histogram(self, ax: Any = None) -> Any:
        """Plot SWR duration distribution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if not self._events:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        durations = [e.duration_ms for e in self._events]

        ax.hist(durations, bins=20, color="salmon", edgecolor="black", alpha=0.7)
        ax.axvline(x=50, color="green", linestyle="--", linewidth=1.5, label="Min (50 ms)")
        ax.axvline(x=150, color="red", linestyle="--", linewidth=1.5, label="Max (150 ms)")
        ax.axvline(x=np.mean(durations), color="black", linestyle="-",
                   linewidth=2, label=f"Mean ({np.mean(durations):.0f} ms)")

        ax.set_xlabel("Duration (ms)")
        ax.set_ylabel("Count")
        ax.set_title("SWR Duration Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_inter_event_intervals(self, ax: Any = None) -> Any:
        """Plot inter-event interval distribution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if not self._inter_event_intervals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        intervals = self._inter_event_intervals

        ax.hist(intervals, bins=30, color="lightgreen", edgecolor="black", alpha=0.7)
        ax.axvline(x=np.mean(intervals), color="black", linestyle="-",
                   linewidth=2, label=f"Mean ({np.mean(intervals):.2f} s)")

        ax.set_xlabel("Inter-Event Interval (s)")
        ax.set_ylabel("Count")
        ax.set_title("Inter-SWR Interval Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_dashboard(self, figsize: tuple = (14, 10)) -> Any:
        """
        Create a comprehensive SWR telemetry dashboard.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        self.plot_swr_raster(axes[0, 0])
        self.plot_frequency_histogram(axes[0, 1])
        self.plot_duration_histogram(axes[1, 0])
        self.plot_inter_event_intervals(axes[1, 1])

        # Add validation summary
        validation = self.validate_biological_ranges()
        status = "VALID" if validation["overall_valid"] else "INVALID"
        fig.suptitle(f"SWR Telemetry Dashboard (Biological Validation: {status})",
                     fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_data(self) -> dict:
        """Export telemetry data for external analysis."""
        return {
            "events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "duration_s": e.duration_s,
                    "duration_ms": e.duration_ms,
                    "frequency_hz": e.ripple_frequency,
                    "amplitude": e.peak_amplitude,
                    "compression_factor": e.compression_factor,
                    "replay_count": e.replay_count,
                    "patterns": e.reactivated_patterns,
                }
                for e in self._events
            ],
            "statistics": {
                "frequency": self.get_frequency_distribution(),
                "duration": self.get_duration_distribution(),
                "amplitude": self.get_amplitude_distribution(),
                "intervals": self.get_inter_event_intervals(),
                "replay": self.get_replay_statistics(),
            },
            "validation": self.validate_biological_ranges(),
            "meta": {
                "total_events": self._total_events,
                "window_size": self.window_size,
                "freq_range": [self.min_ripple_freq, self.max_ripple_freq],
            },
        }

    def reset(self) -> None:
        """Reset telemetry state."""
        self._events.clear()
        self._inter_event_intervals.clear()
        self._total_events = 0
        self._total_replay_time = 0.0
        self._start_time = None
        logger.info("SWRTelemetry reset")


__all__ = ["SWRTelemetry", "SWRTelemetryEvent"]
