"""
Phase-Amplitude Coupling (PAC) Telemetry for T4DM NCA.

Measures theta-gamma coupling for working memory capacity estimation.

Biological Basis:
- Theta (4-8 Hz) phase modulates gamma (30-80 Hz) amplitude
- Number of gamma cycles per theta cycle = working memory capacity (4-7 items)
- Modulation Index (MI) quantifies coupling strength (Tort et al. 2010)
- Strong PAC indicates efficient information binding

Key Metrics:
- Modulation Index (MI): Entropy-based coupling measure (0-1)
- Preferred Phase: Theta phase where gamma amplitude peaks
- Working Memory Capacity: Estimated items from gamma/theta ratio
- Comodulogram: 2D visualization of phase-amplitude relationship

Biological References:
- Tort et al. (2010): Measuring phase-amplitude coupling
- Lisman & Jensen (2013): Theta-gamma neural code
- Canolty & Knight (2010): High gamma power is phase-locked to theta phase
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from t4dm.nca.oscillators import FrequencyBandGenerator, OscillatorState

logger = logging.getLogger(__name__)


@dataclass
class PACSnapshot:
    """Snapshot of phase-amplitude coupling state."""

    timestamp: datetime
    theta_phase: float  # radians (0-2π)
    gamma_amplitude: float  # normalized (0-1)
    modulation_index: float  # Tort et al. 2010 MI (0-1)
    preferred_phase: float  # Phase where gamma peaks (radians)
    working_memory_capacity: float  # Estimated items (4-7 typical)
    theta_frequency: float  # Current theta frequency (Hz)
    gamma_frequency: float  # Current gamma frequency (Hz)


class PACTelemetry:
    """
    Phase-amplitude coupling telemetry.

    Tracks theta-gamma coupling for working memory monitoring.
    Implements Tort et al. (2010) Modulation Index algorithm.
    """

    def __init__(
        self,
        oscillator: FrequencyBandGenerator | None = None,
        window_size: int = 1000,
        n_phase_bins: int = 18,  # Tort et al. use 18 bins (20° each)
        theta_freq_range: tuple[float, float] = (4.0, 8.0),
        gamma_freq_range: tuple[float, float] = (30.0, 80.0),
    ):
        """
        Initialize PAC telemetry.

        Args:
            oscillator: FrequencyBandGenerator from nca/oscillators
            window_size: Number of samples to retain for MI calculation
            n_phase_bins: Number of phase bins for MI (18 = 20° bins)
            theta_freq_range: Theta band frequency range (Hz)
            gamma_freq_range: Gamma band frequency range (Hz)
        """
        self.oscillator = oscillator
        self.window_size = window_size
        self.n_phase_bins = n_phase_bins
        self.theta_freq_range = theta_freq_range
        self.gamma_freq_range = gamma_freq_range

        # History buffers for MI calculation
        self._theta_phase_history: list[float] = []
        self._gamma_amplitude_history: list[float] = []
        self._theta_freq_history: list[float] = []
        self._gamma_freq_history: list[float] = []

        # Snapshot history
        self._snapshots: list[PACSnapshot] = []
        self._max_snapshots = 1000

        # Running statistics
        self._total_samples = 0
        self._start_time: datetime | None = None

        # Cached MI computation (avoid recomputing every sample)
        self._cached_mi: float = 0.0
        self._cached_preferred_phase: float = 0.0
        self._cache_valid_samples: int = 0
        self._cache_update_interval: int = 50  # Recompute every N samples

        logger.info(
            f"PACTelemetry initialized: {n_phase_bins} phase bins, "
            f"theta={theta_freq_range}Hz, gamma={gamma_freq_range}Hz"
        )

    def record_state(
        self,
        theta_phase: float,
        gamma_amplitude: float,
        theta_freq: float | None = None,
        gamma_freq: float | None = None,
    ) -> PACSnapshot:
        """
        Record current oscillator state.

        Args:
            theta_phase: Theta phase (radians, 0-2π)
            gamma_amplitude: Gamma band amplitude (normalized 0-1)
            theta_freq: Current theta frequency (Hz)
            gamma_freq: Current gamma frequency (Hz)

        Returns:
            PACSnapshot with computed metrics
        """
        now = datetime.now()

        if self._start_time is None:
            self._start_time = now

        # Default frequencies
        theta_freq = theta_freq or 6.0
        gamma_freq = gamma_freq or 40.0

        # Store in history
        self._theta_phase_history.append(theta_phase)
        self._gamma_amplitude_history.append(gamma_amplitude)
        self._theta_freq_history.append(theta_freq)
        self._gamma_freq_history.append(gamma_freq)
        self._total_samples += 1

        # Maintain window size
        if len(self._theta_phase_history) > self.window_size:
            self._theta_phase_history.pop(0)
            self._gamma_amplitude_history.pop(0)
            self._theta_freq_history.pop(0)
            self._gamma_freq_history.pop(0)

        # Update MI cache if needed
        samples_since_update = self._total_samples - self._cache_valid_samples
        if samples_since_update >= self._cache_update_interval:
            self._update_mi_cache()

        # Compute working memory capacity
        wm_capacity = self._estimate_wm_capacity()

        # Create snapshot
        snapshot = PACSnapshot(
            timestamp=now,
            theta_phase=theta_phase,
            gamma_amplitude=gamma_amplitude,
            modulation_index=self._cached_mi,
            preferred_phase=self._cached_preferred_phase,
            working_memory_capacity=wm_capacity,
            theta_frequency=theta_freq,
            gamma_frequency=gamma_freq,
        )

        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

        return snapshot

    def record_from_oscillator(
        self,
        state: OscillatorState,
    ) -> PACSnapshot:
        """
        Record state directly from oscillator module.

        Args:
            state: OscillatorState from nca/oscillators

        Returns:
            PACSnapshot
        """
        return self.record_state(
            theta_phase=state.theta_phase,
            gamma_amplitude=state.gamma_amplitude,
            theta_freq=state.theta_freq,
            gamma_freq=state.gamma_freq,
        )

    def _update_mi_cache(self) -> None:
        """Update cached modulation index and preferred phase."""
        self._cached_mi = self.compute_modulation_index()
        self._cached_preferred_phase = self.find_preferred_phase()
        self._cache_valid_samples = self._total_samples

    # -------------------------------------------------------------------------
    # Core PAC Algorithms
    # -------------------------------------------------------------------------

    def compute_modulation_index(self) -> float:
        """
        Compute Tort et al. (2010) modulation index.

        MI = (H_max - H) / H_max

        where H is the entropy of the amplitude distribution across phase bins.
        MI = 0: no coupling (uniform distribution)
        MI = 1: perfect coupling (all amplitude in one bin)

        Returns:
            Modulation index (0-1)
        """
        if len(self._theta_phase_history) < 100:
            return 0.0

        phases = np.array(self._theta_phase_history)
        amplitudes = np.array(self._gamma_amplitude_history)

        # Bin phases into n_phase_bins (0 to 2π)
        phase_bins = np.linspace(0, 2 * np.pi, self.n_phase_bins + 1)
        bin_indices = np.digitize(phases, phase_bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_phase_bins - 1)

        # Compute mean amplitude per phase bin
        mean_amplitudes = np.zeros(self.n_phase_bins)
        for i in range(self.n_phase_bins):
            mask = bin_indices == i
            if np.any(mask):
                mean_amplitudes[i] = np.mean(amplitudes[mask])

        # Normalize to probability distribution
        total = mean_amplitudes.sum()
        if total <= 0:
            return 0.0

        p = mean_amplitudes / total

        # Compute Shannon entropy
        p_nonzero = p[p > 0]
        H = -np.sum(p_nonzero * np.log(p_nonzero))

        # Maximum entropy (uniform distribution)
        H_max = np.log(self.n_phase_bins)

        # Modulation index
        if H_max <= 0:
            return 0.0

        mi = (H_max - H) / H_max
        return float(np.clip(mi, 0, 1))

    def find_preferred_phase(self) -> float:
        """
        Find theta phase where gamma amplitude is maximal.

        Returns:
            Preferred phase (radians, 0-2π)
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
        peak_bin = int(np.argmax(mean_amplitudes))
        return float(bin_centers[peak_bin])

    def _estimate_wm_capacity(self) -> float:
        """
        Estimate working memory capacity from gamma/theta ratio.

        Lisman & Jensen (2013): Capacity ≈ gamma_freq / theta_freq

        Returns:
            Estimated number of items (typically 4-7)
        """
        if len(self._theta_freq_history) < 10:
            return 4.0  # Default

        # Use recent frequency estimates
        theta_freq = np.mean(self._theta_freq_history[-100:])
        gamma_freq = np.mean(self._gamma_freq_history[-100:])

        # Capacity = number of gamma cycles per theta cycle
        if theta_freq <= 0:
            return 4.0

        capacity = gamma_freq / theta_freq

        # Clip to realistic range (3-9 items)
        return float(np.clip(capacity, 3.0, 9.0))

    def get_phase_amplitude_distribution(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get phase-binned amplitude distribution for comodulogram.

        Returns:
            Tuple of (bin_centers, mean_amplitudes, std_amplitudes)
        """
        if len(self._theta_phase_history) < 10:
            bins = np.linspace(0, 2 * np.pi, self.n_phase_bins)
            return bins, np.zeros(self.n_phase_bins), np.zeros(self.n_phase_bins)

        phases = np.array(self._theta_phase_history)
        amplitudes = np.array(self._gamma_amplitude_history)

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

        return bin_centers, mean_amplitudes, std_amplitudes

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_mi_statistics(self) -> dict:
        """Get modulation index statistics over snapshots."""
        if not self._snapshots:
            return {}

        mis = np.array([s.modulation_index for s in self._snapshots])

        return {
            "mean": float(np.mean(mis)),
            "std": float(np.std(mis)),
            "min": float(np.min(mis)),
            "max": float(np.max(mis)),
            "current": float(mis[-1]) if len(mis) > 0 else 0.0,
            "n": len(mis),
        }

    def get_wm_capacity_statistics(self) -> dict:
        """Get working memory capacity statistics."""
        if not self._snapshots:
            return {}

        caps = np.array([s.working_memory_capacity for s in self._snapshots])

        return {
            "mean": float(np.mean(caps)),
            "std": float(np.std(caps)),
            "min": float(np.min(caps)),
            "max": float(np.max(caps)),
            "current": float(caps[-1]) if len(caps) > 0 else 4.0,
        }

    def get_preferred_phase_statistics(self) -> dict:
        """Get preferred phase statistics."""
        if not self._snapshots:
            return {}

        phases = np.array([s.preferred_phase for s in self._snapshots])

        # Circular mean for phase
        sin_mean = np.mean(np.sin(phases))
        cos_mean = np.mean(np.cos(phases))
        circular_mean = np.arctan2(sin_mean, cos_mean)
        if circular_mean < 0:
            circular_mean += 2 * np.pi

        # Circular std (concentration)
        R = np.sqrt(sin_mean**2 + cos_mean**2)
        circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.pi

        return {
            "circular_mean": float(circular_mean),
            "circular_std": float(circular_std),
            "concentration": float(R),  # Higher = more consistent phase
            "current": float(phases[-1]) if len(phases) > 0 else 0.0,
        }

    def validate_biological_ranges(self) -> dict:
        """
        Validate PAC metrics against biological literature.

        Targets:
        - MI: 0.3-0.7 for strong PAC (Tort et al. 2010)
        - WM Capacity: 4-7 items (Miller's 7±2)
        - Preferred phase stability: concentration > 0.5

        Returns:
            Dict with validation results
        """
        validation: dict[str, Any] = {
            "modulation_index": {"in_range": False, "value": None, "target": "0.3-0.7"},
            "wm_capacity": {"in_range": False, "value": None, "target": "4-7 items"},
            "phase_stability": {"in_range": False, "value": None, "target": ">0.5 concentration"},
            "overall_valid": False,
        }

        mi_stats = self.get_mi_statistics()
        wm_stats = self.get_wm_capacity_statistics()
        phase_stats = self.get_preferred_phase_statistics()

        # MI validation
        if mi_stats:
            mean_mi = mi_stats["mean"]
            validation["modulation_index"]["value"] = mean_mi
            validation["modulation_index"]["in_range"] = 0.3 <= mean_mi <= 0.7

        # WM capacity validation
        if wm_stats:
            mean_wm = wm_stats["mean"]
            validation["wm_capacity"]["value"] = mean_wm
            validation["wm_capacity"]["in_range"] = 4.0 <= mean_wm <= 7.0

        # Phase stability validation
        if phase_stats:
            concentration = phase_stats["concentration"]
            validation["phase_stability"]["value"] = concentration
            validation["phase_stability"]["in_range"] = concentration > 0.5

        # Overall
        validation["overall_valid"] = (
            validation["modulation_index"]["in_range"] and
            validation["wm_capacity"]["in_range"] and
            validation["phase_stability"]["in_range"]
        )

        return validation

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_comodulogram(self, ax: Any = None) -> Any:
        """
        Plot phase-amplitude comodulogram.

        Shows gamma amplitude as function of theta phase.

        Args:
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        bin_centers, mean_amps, std_amps = self.get_phase_amplitude_distribution()

        if np.all(mean_amps == 0):
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        # Plot mean with std shading
        ax.plot(bin_centers, mean_amps, "b-", linewidth=2, label="Mean amplitude")
        ax.fill_between(
            bin_centers,
            mean_amps - std_amps,
            mean_amps + std_amps,
            alpha=0.3,
            color="blue",
        )

        # Mark preferred phase
        preferred = self._cached_preferred_phase
        ax.axvline(x=preferred, color="red", linestyle="--", linewidth=2,
                   label=f"Preferred phase: {preferred:.2f} rad")

        ax.set_xlabel("Theta Phase (radians)")
        ax.set_ylabel("Gamma Amplitude")
        ax.set_title(f"Phase-Amplitude Coupling (MI = {self._cached_mi:.3f})")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_mi_timeline(self, ax: Any = None) -> Any:
        """
        Plot modulation index over time.

        Args:
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        t0 = self._snapshots[0].timestamp
        times = [(s.timestamp - t0).total_seconds() for s in self._snapshots]
        mis = [s.modulation_index for s in self._snapshots]

        ax.plot(times, mis, "b-", linewidth=1.5)

        # Mark biological range
        ax.axhspan(0.3, 0.7, alpha=0.1, color="green", label="Target range (0.3-0.7)")
        ax.axhline(y=np.mean(mis), color="red", linestyle="--",
                   label=f"Mean: {np.mean(mis):.3f}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Modulation Index")
        ax.set_title("PAC Modulation Index Over Time")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_wm_capacity_timeline(self, ax: Any = None) -> Any:
        """
        Plot working memory capacity estimate over time.

        Args:
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        t0 = self._snapshots[0].timestamp
        times = [(s.timestamp - t0).total_seconds() for s in self._snapshots]
        caps = [s.working_memory_capacity for s in self._snapshots]

        ax.plot(times, caps, "g-", linewidth=1.5)

        # Mark Miller's 7±2 range
        ax.axhspan(4, 7, alpha=0.1, color="blue", label="Miller's range (4-7)")
        ax.axhline(y=np.mean(caps), color="red", linestyle="--",
                   label=f"Mean: {np.mean(caps):.1f}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Working Memory Capacity (items)")
        ax.set_title("Estimated Working Memory Capacity Over Time")
        ax.set_ylim(2, 10)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_polar_comodulogram(self, ax: Any = None) -> Any:
        """
        Plot phase-amplitude coupling on polar axes.

        Args:
            ax: Matplotlib polar axes

        Returns:
            Matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='polar'))

        bin_centers, mean_amps, _ = self.get_phase_amplitude_distribution()

        if np.all(mean_amps == 0):
            return ax

        # Normalize amplitudes for visibility
        if np.max(mean_amps) > 0:
            norm_amps = mean_amps / np.max(mean_amps)
        else:
            norm_amps = mean_amps

        # Close the polar plot
        bin_centers = np.append(bin_centers, bin_centers[0])
        norm_amps = np.append(norm_amps, norm_amps[0])

        ax.plot(bin_centers, norm_amps, "b-", linewidth=2)
        ax.fill(bin_centers, norm_amps, alpha=0.3, color="blue")

        # Mark preferred phase
        preferred = self._cached_preferred_phase
        ax.axvline(x=preferred, color="red", linestyle="--", linewidth=2)

        ax.set_title(f"PAC Polar Plot (MI = {self._cached_mi:.3f})")

        return ax

    def plot_dashboard(self, figsize: tuple = (14, 10)) -> Any:
        """
        Create comprehensive PAC telemetry dashboard.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        fig = plt.figure(figsize=figsize)

        # Create grid: 2x2 with polar in bottom right
        ax1 = fig.add_subplot(2, 2, 1)  # Comodulogram
        ax2 = fig.add_subplot(2, 2, 2)  # MI timeline
        ax3 = fig.add_subplot(2, 2, 3)  # WM capacity
        ax4 = fig.add_subplot(2, 2, 4, projection='polar')  # Polar

        self.plot_comodulogram(ax1)
        self.plot_mi_timeline(ax2)
        self.plot_wm_capacity_timeline(ax3)
        self.plot_polar_comodulogram(ax4)

        # Add validation summary
        validation = self.validate_biological_ranges()
        status = "VALID" if validation["overall_valid"] else "INVALID"
        fig.suptitle(f"PAC Telemetry Dashboard (Biological Validation: {status})",
                     fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_data(self) -> dict:
        """Export telemetry data for external analysis."""
        bin_centers, mean_amps, std_amps = self.get_phase_amplitude_distribution()

        return {
            "snapshots": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "theta_phase": s.theta_phase,
                    "gamma_amplitude": s.gamma_amplitude,
                    "modulation_index": s.modulation_index,
                    "preferred_phase": s.preferred_phase,
                    "wm_capacity": s.working_memory_capacity,
                    "theta_freq": s.theta_frequency,
                    "gamma_freq": s.gamma_frequency,
                }
                for s in self._snapshots[-100:]  # Last 100 for export
            ],
            "comodulogram": {
                "phase_bins": bin_centers.tolist(),
                "mean_amplitude": mean_amps.tolist(),
                "std_amplitude": std_amps.tolist(),
            },
            "statistics": {
                "modulation_index": self.get_mi_statistics(),
                "wm_capacity": self.get_wm_capacity_statistics(),
                "preferred_phase": self.get_preferred_phase_statistics(),
            },
            "validation": self.validate_biological_ranges(),
            "meta": {
                "total_samples": self._total_samples,
                "window_size": self.window_size,
                "n_phase_bins": self.n_phase_bins,
                "theta_range": self.theta_freq_range,
                "gamma_range": self.gamma_freq_range,
            },
        }

    def reset(self) -> None:
        """Reset telemetry state."""
        self._theta_phase_history.clear()
        self._gamma_amplitude_history.clear()
        self._theta_freq_history.clear()
        self._gamma_freq_history.clear()
        self._snapshots.clear()
        self._total_samples = 0
        self._start_time = None
        self._cached_mi = 0.0
        self._cached_preferred_phase = 0.0
        self._cache_valid_samples = 0
        logger.info("PACTelemetry reset")


__all__ = ["PACTelemetry", "PACSnapshot"]
