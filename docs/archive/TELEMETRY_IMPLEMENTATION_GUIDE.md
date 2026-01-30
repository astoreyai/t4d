# World Weaver NCA Telemetry: Implementation Guide
**Author**: Claude Sonnet 4.5 (CompBio Agent)
**Date**: 2026-01-01
**Sprint Goal**: Implement P0 (critical) telemetry enhancements

---

## Quick Reference: Priority Matrix

| Enhancement | Priority | Bio Fidelity | Effort | Impact | Files to Create |
|-------------|----------|--------------|--------|--------|-----------------|
| SWR Telemetry | **P0** | 92/100 | 1 week | High | `swr_telemetry.py` |
| PAC Visualization | **P0** | 90/100 | 1 week | High | `pac_telemetry.py` |
| Enhanced Anomaly Detection | **P0** | 90/100 | 3 days | Critical | Update `stability_monitor.py` |
| DA Temporal Structure | **P1** | 88/100 | 3 days | Medium | `da_telemetry.py` |
| Multi-Scale Hub | **P1** | 85/100 | 2-3 weeks | High | `telemetry_hub.py` |
| Validation Framework | **P1** | 95/100 | 2 weeks | High | `validation/` directory |

---

## P0-1: SWR Telemetry (1 week)

### Biological Rationale
Sharp-wave ripples (SWRs) are **150-250 Hz oscillations** in hippocampal CA1 during offline states (sleep, quiet rest). They drive memory consolidation by replaying recent experiences at 10x speed (Foster & Wilson, 2006).

### Implementation

**File**: `/mnt/projects/ww/src/ww/visualization/swr_telemetry.py`

```python
"""
SWR Telemetry for World Weaver NCA.

Visualizes sharp-wave ripple events for consolidation monitoring:
- Ripple frequency (150-250 Hz)
- Sharp-wave amplitude (CA3 burst)
- Replay sequences (reactivated patterns)
- Inter-event intervals
- Temporal compression factor

Biological References:
- Buzsáki (2015): Hippocampal Sharp Wave-Ripple
- Foster & Wilson (2006): Reverse replay of behavioural sequences
- Girardeau et al. (2009): Selective suppression impairs consolidation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ww.nca.swr_coupling import SWRState, SWRPhase

logger = logging.getLogger(__name__)


@dataclass
class SWREvent:
    """Single SWR event record."""

    timestamp: datetime
    duration: float  # seconds (typically 50-150 ms)
    ripple_frequency: float  # Hz (150-250 range)
    sharp_wave_amplitude: float  # Arbitrary units
    compression_factor: float  # Replay speed (typically 10x)
    reactivated_patterns: List[int]  # Which memories replayed
    hippocampal_activity: float  # CA3/CA1 activity level
    phase: str  # QUIESCENT, INITIATING, RIPPLING, TERMINATING


class SWRTelemetry:
    """
    Sharp-wave ripple telemetry and analysis.

    Tracks SWR events for consolidation monitoring.
    Validates against experimental literature (Buzsáki lab data).
    """

    def __init__(
        self,
        swr_coupling: Optional["SWRCoupling"] = None,
        window_size: int = 1000,
        min_ripple_freq: float = 150.0,
        max_ripple_freq: float = 250.0,
    ):
        """
        Initialize SWR telemetry.

        Args:
            swr_coupling: SWRCoupling module (from nca/)
            window_size: Number of events to retain
            min_ripple_freq: Minimum ripple frequency (Hz)
            max_ripple_freq: Maximum ripple frequency (Hz)
        """
        self.swr_coupling = swr_coupling
        self.window_size = window_size
        self.min_ripple_freq = min_ripple_freq
        self.max_ripple_freq = max_ripple_freq

        # Event history
        self._events: List[SWREvent] = []
        self._inter_event_intervals: List[float] = []

        # Statistics
        self._total_events = 0
        self._total_replay_time = 0.0

        logger.info("SWRTelemetry initialized")

    def record_swr_event(
        self,
        swr_state: "SWRState",
        reactivated_patterns: Optional[List[int]] = None,
    ) -> Optional[SWREvent]:
        """
        Record an SWR event.

        Args:
            swr_state: Current SWR state from swr_coupling module
            reactivated_patterns: List of pattern IDs that were replayed

        Returns:
            SWREvent if event is valid, None otherwise
        """
        now = datetime.now()

        # Only record during RIPPLING phase
        if swr_state.phase != "RIPPLING":
            return None

        # Create event
        event = SWREvent(
            timestamp=now,
            duration=swr_state.current_duration,
            ripple_frequency=swr_state.ripple_frequency,
            sharp_wave_amplitude=swr_state.current_amplitude,
            compression_factor=swr_state.compression_factor,
            reactivated_patterns=reactivated_patterns or [],
            hippocampal_activity=swr_state.hippocampal_activity,
            phase=swr_state.phase.name if hasattr(swr_state.phase, 'name') else str(swr_state.phase),
        )

        # Validate frequency range
        if not (self.min_ripple_freq <= event.ripple_frequency <= self.max_ripple_freq):
            logger.warning(
                f"SWR frequency {event.ripple_frequency:.1f} Hz outside range "
                f"[{self.min_ripple_freq}, {self.max_ripple_freq}]"
            )

        # Compute inter-event interval
        if self._events:
            interval = (now - self._events[-1].timestamp).total_seconds()
            self._inter_event_intervals.append(interval)

        # Store event
        self._events.append(event)
        self._total_events += 1
        self._total_replay_time += event.duration

        # Maintain window
        if len(self._events) > self.window_size:
            self._events.pop(0)
        if len(self._inter_event_intervals) > self.window_size:
            self._inter_event_intervals.pop(0)

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

        return len(recent_events) / window_seconds

    def get_frequency_distribution(self) -> dict:
        """Get statistics on ripple frequency distribution."""
        if not self._events:
            return {}

        frequencies = [e.ripple_frequency for e in self._events]

        return {
            "mean": float(np.mean(frequencies)),
            "std": float(np.std(frequencies)),
            "min": float(np.min(frequencies)),
            "max": float(np.max(frequencies)),
            "median": float(np.median(frequencies)),
        }

    def get_duration_distribution(self) -> dict:
        """Get statistics on SWR duration."""
        if not self._events:
            return {}

        durations = [e.duration * 1000 for e in self._events]  # Convert to ms

        return {
            "mean_ms": float(np.mean(durations)),
            "std_ms": float(np.std(durations)),
            "min_ms": float(np.min(durations)),
            "max_ms": float(np.max(durations)),
        }

    def get_inter_event_intervals(self) -> dict:
        """Get statistics on inter-SWR intervals."""
        if not self._inter_event_intervals:
            return {}

        return {
            "mean_s": float(np.mean(self._inter_event_intervals)),
            "std_s": float(np.std(self._inter_event_intervals)),
            "min_s": float(np.min(self._inter_event_intervals)),
            "max_s": float(np.max(self._inter_event_intervals)),
        }

    def get_replay_statistics(self) -> dict:
        """Analyze replay content."""
        if not self._events:
            return {}

        # Count unique patterns replayed
        all_patterns = []
        for event in self._events:
            all_patterns.extend(event.reactivated_patterns)

        unique_patterns = len(set(all_patterns))
        total_replays = len(all_patterns)

        return {
            "total_swr_events": len(self._events),
            "total_replays": total_replays,
            "unique_patterns_replayed": unique_patterns,
            "mean_patterns_per_swr": total_replays / len(self._events) if self._events else 0,
            "total_replay_time_s": self._total_replay_time,
        }

    def validate_biological_ranges(self) -> dict:
        """
        Validate against experimental literature.

        Returns:
            Dict with validation results
        """
        stats = {}

        freq_dist = self.get_frequency_distribution()
        dur_dist = self.get_duration_distribution()

        # Frequency: 150-250 Hz (Buzsáki, 2015)
        if freq_dist:
            stats["frequency_in_range"] = (
                150 <= freq_dist["mean"] <= 250
            )
            stats["frequency_mean"] = freq_dist["mean"]

        # Duration: 50-150 ms (Buzsáki, 2015)
        if dur_dist:
            stats["duration_in_range"] = (
                50 <= dur_dist["mean_ms"] <= 150
            )
            stats["duration_mean_ms"] = dur_dist["mean_ms"]

        # Event rate: ~1 Hz during sleep (Girardeau et al. 2009)
        event_rate = self.get_event_rate(window_seconds=60.0)
        stats["event_rate_hz"] = event_rate
        stats["event_rate_in_range"] = 0.5 <= event_rate <= 2.0

        return stats

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_swr_raster(self, ax=None):
        """
        Plot SWR events as raster (time vs event).

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
            ax.text(0.5, 0.5, "No SWR events", ha="center", va="center")
            return ax

        # Convert timestamps to relative seconds
        t0 = self._events[0].timestamp
        times = [(e.timestamp - t0).total_seconds() for e in self._events]
        frequencies = [e.ripple_frequency for e in self._events]
        durations = [e.duration * 1000 for e in self._events]  # ms

        # Scatter plot: time vs frequency, size = duration
        scatter = ax.scatter(
            times, frequencies, s=np.array(durations) * 2,
            c=durations, cmap="viridis", alpha=0.7, edgecolors="black", linewidths=0.5
        )

        # Mark biological range
        ax.axhspan(150, 250, alpha=0.1, color="green", label="Biological range")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Ripple Frequency (Hz)")
        ax.set_title("SWR Events Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label="Duration (ms)")

        return ax

    def plot_frequency_histogram(self, ax=None):
        """Plot ripple frequency distribution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if not self._events:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return ax

        frequencies = [e.ripple_frequency for e in self._events]

        ax.hist(frequencies, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        ax.axvline(x=150, color="green", linestyle="--", label="Min (150 Hz)")
        ax.axvline(x=250, color="red", linestyle="--", label="Max (250 Hz)")
        ax.axvline(x=np.mean(frequencies), color="black", linestyle="-", linewidth=2, label="Mean")

        ax.set_xlabel("Ripple Frequency (Hz)")
        ax.set_ylabel("Count")
        ax.set_title("Ripple Frequency Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def export_data(self) -> dict:
        """Export telemetry data for external analysis."""
        return {
            "events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "duration": e.duration,
                    "frequency": e.ripple_frequency,
                    "amplitude": e.sharp_wave_amplitude,
                    "patterns": e.reactivated_patterns,
                }
                for e in self._events
            ],
            "statistics": {
                "frequency": self.get_frequency_distribution(),
                "duration": self.get_duration_distribution(),
                "intervals": self.get_inter_event_intervals(),
                "replay": self.get_replay_statistics(),
            },
            "validation": self.validate_biological_ranges(),
        }


__all__ = ["SWRTelemetry", "SWREvent"]
```

### Testing

**File**: `/mnt/projects/ww/tests/visualization/test_swr_telemetry.py`

```python
"""Tests for SWR telemetry."""

import pytest
from datetime import datetime
from dataclasses import dataclass
from ww.visualization.swr_telemetry import SWRTelemetry, SWREvent


@dataclass
class MockSWRState:
    """Mock SWR state for testing."""
    phase: str
    current_duration: float
    ripple_frequency: float
    current_amplitude: float
    compression_factor: float
    hippocampal_activity: float


def test_swr_telemetry_init():
    """Test SWR telemetry initialization."""
    telemetry = SWRTelemetry()
    assert len(telemetry._events) == 0


def test_record_swr_event():
    """Test recording SWR event."""
    telemetry = SWRTelemetry()

    state = MockSWRState(
        phase="RIPPLING",
        current_duration=0.08,  # 80 ms
        ripple_frequency=180.0,
        current_amplitude=1.5,
        compression_factor=10.0,
        hippocampal_activity=0.7,
    )

    event = telemetry.record_swr_event(state, reactivated_patterns=[1, 3, 5])

    assert event is not None
    assert event.ripple_frequency == 180.0
    assert event.duration == 0.08
    assert len(event.reactivated_patterns) == 3


def test_frequency_validation():
    """Test biological frequency range validation."""
    telemetry = SWRTelemetry()

    # Record 10 events in valid range
    for i in range(10):
        state = MockSWRState(
            phase="RIPPLING",
            current_duration=0.08,
            ripple_frequency=180.0 + i * 5,  # 180-225 Hz
            current_amplitude=1.0,
            compression_factor=10.0,
            hippocampal_activity=0.7,
        )
        telemetry.record_swr_event(state)

    validation = telemetry.validate_biological_ranges()
    assert validation["frequency_in_range"] is True
    assert 150 <= validation["frequency_mean"] <= 250


def test_duration_validation():
    """Test biological duration range validation."""
    telemetry = SWRTelemetry()

    # Record events with valid durations (50-150 ms)
    for i in range(10):
        state = MockSWRState(
            phase="RIPPLING",
            current_duration=0.08,  # 80 ms
            ripple_frequency=180.0,
            current_amplitude=1.0,
            compression_factor=10.0,
            hippocampal_activity=0.7,
        )
        telemetry.record_swr_event(state)

    validation = telemetry.validate_biological_ranges()
    assert validation["duration_in_range"] is True


def test_replay_statistics():
    """Test replay pattern analysis."""
    telemetry = SWRTelemetry()

    # Record events with different replay patterns
    patterns_list = [[1, 2], [3], [1, 2, 4], [5, 6]]

    for patterns in patterns_list:
        state = MockSWRState(
            phase="RIPPLING",
            current_duration=0.08,
            ripple_frequency=180.0,
            current_amplitude=1.0,
            compression_factor=10.0,
            hippocampal_activity=0.7,
        )
        telemetry.record_swr_event(state, reactivated_patterns=patterns)

    stats = telemetry.get_replay_statistics()
    assert stats["total_swr_events"] == 4
    assert stats["unique_patterns_replayed"] == 6  # {1,2,3,4,5,6}
    assert stats["total_replays"] == 9  # Sum of pattern lengths
```

---

## P0-2: PAC Telemetry (1 week)

### Biological Rationale
Phase-amplitude coupling (PAC) between theta (4-8 Hz) and gamma (30-80 Hz) is the neural code for working memory (Lisman & Jensen, 2013). The number of gamma cycles nested in each theta cycle = working memory capacity.

### Implementation

**File**: `/mnt/projects/ww/src/ww/visualization/pac_telemetry.py`

```python
"""
Phase-Amplitude Coupling (PAC) Telemetry for World Weaver NCA.

Measures theta-gamma coupling for working memory capacity estimation.

Biological References:
- Tort et al. (2010): Measuring phase-amplitude coupling
- Lisman & Jensen (2013): Theta-gamma neural code
- Canolty & Knight (2010): High gamma power is phase-locked to theta phase
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from ww.nca.oscillators import OscillatorState

logger = logging.getLogger(__name__)


@dataclass
class PACSnapshot:
    """Snapshot of phase-amplitude coupling."""

    timestamp: datetime
    theta_phase: float  # radians (0-2π)
    gamma_amplitude: float
    modulation_index: float  # Tort et al. 2010 MI
    preferred_phase: float  # Phase where gamma peaks
    working_memory_capacity: float  # Estimated items (gamma cycles / theta)


class PACTelemetry:
    """
    Phase-amplitude coupling telemetry.

    Tracks theta-gamma coupling for working memory monitoring.
    """

    def __init__(
        self,
        oscillator_state: Optional["OscillatorState"] = None,
        window_size: int = 1000,
        n_phase_bins: int = 18,  # Tort et al. use 18 bins (20° each)
    ):
        """
        Initialize PAC telemetry.

        Args:
            oscillator_state: Oscillator state module
            window_size: Number of snapshots to retain
            n_phase_bins: Number of phase bins for MI calculation
        """
        self.oscillator_state = oscillator_state
        self.window_size = window_size
        self.n_phase_bins = n_phase_bins

        # History
        self._snapshots: list[PACSnapshot] = []
        self._theta_phase_history: list[float] = []
        self._gamma_amplitude_history: list[float] = []

        logger.info("PACTelemetry initialized")

    def record_state(
        self,
        theta_phase: float,
        gamma_amplitude: float,
    ) -> PACSnapshot:
        """
        Record current PAC state.

        Args:
            theta_phase: Theta phase (radians, 0-2π)
            gamma_amplitude: Gamma band amplitude

        Returns:
            PACSnapshot with computed metrics
        """
        now = datetime.now()

        # Store history
        self._theta_phase_history.append(theta_phase)
        self._gamma_amplitude_history.append(gamma_amplitude)

        # Maintain window
        if len(self._theta_phase_history) > self.window_size:
            self._theta_phase_history.pop(0)
            self._gamma_amplitude_history.pop(0)

        # Compute modulation index
        mi = self.compute_modulation_index()

        # Find preferred phase
        preferred_phase = self.find_preferred_phase()

        # Estimate working memory capacity
        wm_capacity = self.estimate_wm_capacity()

        snapshot = PACSnapshot(
            timestamp=now,
            theta_phase=theta_phase,
            gamma_amplitude=gamma_amplitude,
            modulation_index=mi,
            preferred_phase=preferred_phase,
            working_memory_capacity=wm_capacity,
        )

        self._snapshots.append(snapshot)

        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

        return snapshot

    def compute_modulation_index(self) -> float:
        """
        Compute Tort et al. (2010) modulation index.

        MI = (H - Hmax) / Hmax
        where H = entropy of amplitude distribution across phase bins

        Returns:
            Modulation index (0 = no coupling, 1 = perfect coupling)
        """
        if len(self._theta_phase_history) < 100:
            return 0.0

        phases = np.array(self._theta_phase_history)
        amplitudes = np.array(self._gamma_amplitude_history)

        # Bin phases (0-2π into n_phase_bins)
        phase_bins = np.linspace(0, 2 * np.pi, self.n_phase_bins + 1)
        bin_indices = np.digitize(phases, phase_bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_phase_bins - 1)

        # Mean amplitude per phase bin
        mean_amplitudes = np.zeros(self.n_phase_bins)
        for i in range(self.n_phase_bins):
            mask = bin_indices == i
            if np.any(mask):
                mean_amplitudes[i] = np.mean(amplitudes[mask])

        # Normalize to probability distribution
        p = mean_amplitudes / mean_amplitudes.sum() if mean_amplitudes.sum() > 0 else mean_amplitudes

        # Compute entropy
        p_nonzero = p[p > 0]
        H = -np.sum(p_nonzero * np.log(p_nonzero))

        # Maximum entropy (uniform distribution)
        H_max = np.log(self.n_phase_bins)

        # Modulation index
        mi = (H_max - H) / H_max if H_max > 0 else 0.0

        return float(mi)

    def find_preferred_phase(self) -> float:
        """
        Find theta phase where gamma amplitude is maximal.

        Returns:
            Preferred phase (radians)
        """
        if len(self._theta_phase_history) < 100:
            return 0.0

        phases = np.array(self._theta_phase_history)
        amplitudes = np.array(self._gamma_amplitude_history)

        # Bin phases
        phase_bins = np.linspace(0, 2 * np.pi, self.n_phase_bins + 1)
        bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        bin_indices = np.digitize(phases, phase_bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_phase_bins - 1)

        # Mean amplitude per bin
        mean_amplitudes = np.zeros(self.n_phase_bins)
        for i in range(self.n_phase_bins):
            mask = bin_indices == i
            if np.any(mask):
                mean_amplitudes[i] = np.mean(amplitudes[mask])

        # Find peak
        peak_bin = np.argmax(mean_amplitudes)
        preferred_phase = bin_centers[peak_bin]

        return float(preferred_phase)

    def estimate_wm_capacity(self) -> float:
        """
        Estimate working memory capacity from gamma/theta ratio.

        Lisman & Jensen (2013): Capacity ≈ gamma_freq / theta_freq

        Returns:
            Estimated number of items (4-7 typical)
        """
        # Estimate frequencies from history
        if len(self._theta_phase_history) < 100:
            return 4.0  # Default

        # Theta frequency from phase derivative
        phases = np.array(self._theta_phase_history[-100:])
        phase_diff = np.diff(phases)

        # Unwrap phase (handle 2π discontinuities)
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

        # Mean phase increment per sample → frequency
        # Assuming dt = 0.1s (10 Hz sampling)
        dt = 0.1
        theta_freq = np.mean(phase_diff) / (2 * np.pi * dt)

        # Gamma frequency (assume 40 Hz default from oscillators)
        gamma_freq = 40.0

        # Capacity = gamma / theta
        capacity = gamma_freq / max(theta_freq, 1.0)

        # Clip to realistic range (4-7 items)
        capacity = np.clip(capacity, 3.0, 9.0)

        return float(capacity)

    def plot_comodulogram(self, ax=None):
        """
        Plot phase-amplitude comodulogram.

        Shows gamma amplitude as function of theta phase.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if len(self._theta_phase_history) < 100:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
            return ax

        phases = np.array(self._theta_phase_history)
        amplitudes = np.array(self._gamma_amplitude_history)

        # Bin and average
        phase_bins = np.linspace(0, 2 * np.pi, self.n_phase_bins + 1)
        bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        bin_indices = np.digitize(phases, phase_bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_phase_bins - 1)

        mean_amplitudes = np.zeros(self.n_phase_bins)
        std_amplitudes = np.zeros(self.n_phase_bins)

        for i in range(self.n_phase_bins):
            mask = bin_indices == i
            if np.any(mask):
                mean_amplitudes[i] = np.mean(amplitudes[mask])
                std_amplitudes[i] = np.std(amplitudes[mask])

        # Plot
        ax.plot(bin_centers, mean_amplitudes, "b-", linewidth=2)
        ax.fill_between(
            bin_centers,
            mean_amplitudes - std_amplitudes,
            mean_amplitudes + std_amplitudes,
            alpha=0.3,
            color="blue",
        )

        # Mark preferred phase
        preferred_phase = self.find_preferred_phase()
        ax.axvline(x=preferred_phase, color="red", linestyle="--", linewidth=2, label=f"Preferred phase: {preferred_phase:.2f} rad")

        ax.set_xlabel("Theta Phase (radians)")
        ax.set_ylabel("Gamma Amplitude")
        ax.set_title(f"Phase-Amplitude Coupling (MI = {self.compute_modulation_index():.3f})")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def export_data(self) -> dict:
        """Export PAC data."""
        current = self._snapshots[-1] if self._snapshots else None

        return {
            "current": {
                "modulation_index": current.modulation_index if current else 0.0,
                "preferred_phase": current.preferred_phase if current else 0.0,
                "wm_capacity": current.working_memory_capacity if current else 4.0,
            },
            "history": {
                "theta_phase": self._theta_phase_history[-100:],
                "gamma_amplitude": self._gamma_amplitude_history[-100:],
            },
        }


__all__ = ["PACTelemetry", "PACSnapshot"]
```

---

## P0-3: Enhanced Anomaly Detection (3 days)

### Update `stability_monitor.py`

Add to `/mnt/projects/ww/src/ww/visualization/stability_monitor.py`:

```python
# Add to StabilityMonitor class:

def detect_excitotoxicity(self, nt_state: np.ndarray) -> list[str]:
    """
    Detect excitotoxic conditions.

    Args:
        nt_state: 6D NT state [DA, 5-HT, ACh, NE, GABA, Glu]

    Returns:
        List of alert strings
    """
    alerts = []

    glutamate = nt_state[5]
    gaba = nt_state[4]

    # Glutamate surge
    if glutamate > 0.85:
        alerts.append(f"EXCITOTOXICITY: Glu={glutamate:.2f} > 0.85")

    # E/I imbalance
    ei_ratio = glutamate / max(gaba, 0.01)
    if ei_ratio > 3.0:
        alerts.append(f"E/I RATIO: {ei_ratio:.2f} > 3.0 (seizure-like)")

    # GABA depletion
    if gaba < 0.2:
        alerts.append(f"GABA DEPLETION: {gaba:.2f} < 0.2")

    return alerts

def detect_forgetting_risk(self, hippocampus_state: dict) -> list[str]:
    """
    Detect catastrophic forgetting risk.

    Args:
        hippocampus_state: Dict with CA3 pattern info

    Returns:
        List of alert strings
    """
    alerts = []

    # CA3 capacity
    if "ca3_patterns" in hippocampus_state:
        n_patterns = len(hippocampus_state["ca3_patterns"])
        max_patterns = hippocampus_state.get("ca3_max_patterns", 1000)
        capacity_fraction = n_patterns / max_patterns

        if capacity_fraction > 0.9:
            alerts.append(f"CA3 NEAR CAPACITY: {capacity_fraction*100:.1f}%")

    # Pattern overlap
    if "pattern_similarity" in hippocampus_state:
        avg_similarity = hippocampus_state["pattern_similarity"]
        if avg_similarity > 0.6:
            alerts.append(f"PATTERN OVERLAP: {avg_similarity:.2f} > 0.6 (interference)")

    return alerts
```

---

## Timeline & Testing

### Week 1: SWR Telemetry
- Days 1-2: Implement `swr_telemetry.py`
- Day 3: Write tests
- Day 4: Integration with `swr_coupling.py`
- Day 5: Validation against Buzsáki data

### Week 2: PAC Telemetry
- Days 1-2: Implement `pac_telemetry.py`
- Day 3: Write tests
- Day 4: Integration with `oscillators.py`
- Day 5: Validation (MI should be 0.3-0.7)

### Week 3: Enhanced Anomaly Detection
- Days 1-2: Add excitotoxicity detection
- Day 3: Add forgetting risk detection
- Days 4-5: Testing & validation

---

## Validation Checklist

### SWR Telemetry
- [ ] Frequency in 150-250 Hz range
- [ ] Duration in 50-150 ms range
- [ ] Event rate ~1 Hz during "sleep" state
- [ ] Compression factor = 10x
- [ ] Replay patterns tracked

### PAC Telemetry
- [ ] Modulation index 0.3-0.7 (strong PAC)
- [ ] Preferred phase stable across windows
- [ ] WM capacity 4-7 items
- [ ] Comodulogram shows peak at preferred phase

### Anomaly Detection
- [ ] Detects Glu > 0.85 (excitotoxicity)
- [ ] Detects E/I > 3.0 (seizure-like)
- [ ] Detects CA3 capacity > 90% (forgetting risk)
- [ ] Alerts logged and visualized

---

**Next Steps**: After P0 completion, proceed to P1 (Multi-Scale Hub, DA Temporal Structure, Validation Framework).

END OF IMPLEMENTATION GUIDE
