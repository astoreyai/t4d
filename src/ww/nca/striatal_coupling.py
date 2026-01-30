"""
DA-ACh Striatal Coupling for Biologically Plausible Learning.

Biological Basis (2025 Nature Neuroscience):
- ACh acts as "axonal brake" on DA release in striatum
- ~100ms phase lag creates traveling wave dynamics
- Bidirectional coupling essential for habit formation
- Anticorrelation (r < -0.3) observed in limbic striatum

This module implements the delayed, bidirectional coupling between
dopamine and acetylcholine systems that is critical for:
1. Habit formation and extinction
2. Reward-based learning gating
3. Attention-reward interaction

Core Dynamics:
    dDA/dt = ... - K_ach_da * ACh(t - tau)  # ACh inhibits DA with delay
    dACh/dt = ... + K_da_ach * DA(t - tau)  # DA excites ACh with delay

where tau ≈ 100ms is the phase lag between systems.

References:
- Threlfell et al. (2012): Striatal dopamine release is triggered by
  synchronized activity in cholinergic interneurons. Neuron.
- Cachope et al. (2012): Selective activation of cholinergic interneurons
  enhances accumbal phasic dopamine release. Neuron.
- 2025 Nature Neuroscience: ACh-DA traveling waves in striatum
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DAACHCouplingConfig:
    """Configuration for DA-ACh striatal coupling."""

    # Coupling strengths (from CompBio audit)
    k_ach_to_da: float = -0.25  # ACh inhibits DA (axonal brake)
    k_da_to_ach: float = 0.15   # DA facilitates ACh

    # Phase lag parameters
    phase_lag_ms: float = 100.0  # ~100ms delay (from 2025 literature)
    dt: float = 1.0              # Time step in ms for delay buffer

    # Delay buffer size (phase_lag_ms / dt)
    @property
    def delay_steps(self) -> int:
        return max(1, int(self.phase_lag_ms / self.dt))

    # Spatial specificity
    striatal_regions: list = field(default_factory=lambda: ["limbic", "motor", "associative"])
    limbic_coupling_scale: float = 1.2   # Stronger in limbic (emotion/reward)
    motor_coupling_scale: float = 0.8    # Weaker in motor (habit)
    associative_coupling_scale: float = 1.0  # Baseline in associative

    # Oscillatory parameters
    enable_oscillation: bool = True
    oscillation_freq_hz: float = 4.0  # Theta-band modulation
    oscillation_amplitude: float = 0.1

    # Learning modulation
    rpe_modulates_coupling: bool = True  # RPE can strengthen/weaken coupling
    rpe_coupling_lr: float = 0.001       # Learning rate for RPE modulation

    # Validation thresholds (from audit)
    target_anticorrelation: float = -0.3  # Correlation should be < -0.3


@dataclass
class DAACHState:
    """State of DA-ACh coupling at a single time point."""
    da_level: float
    ach_level: float
    da_to_ach_effect: float
    ach_to_da_effect: float
    correlation: float
    phase_difference: float  # In radians
    timestamp_ms: float


class DAACHCoupling:
    """
    Bidirectional DA-ACh coupling with phase lag.

    Implements the striatal "traveling wave" dynamics where:
    - ACh leads DA by ~100ms
    - ACh inhibits DA release (axonal brake)
    - DA facilitates ACh (positive feedback with delay)

    This creates anticorrelated oscillations essential for:
    - Learning gating (ACh high = encode, DA high = reinforce)
    - Habit formation (gradual shift from goal-directed to habitual)
    - Reward prediction (DA phasic responses gated by tonic ACh)

    Usage:
        coupling = DAACHCoupling(config)

        # Each timestep:
        da_effect, ach_effect = coupling.compute_coupling(da_level, ach_level)

        # Apply effects to NT dynamics
        da_new = da + dt * (... + da_effect)
        ach_new = ach + dt * (... + ach_effect)
    """

    def __init__(self, config: DAACHCouplingConfig | None = None):
        """
        Initialize DA-ACh coupling.

        Args:
            config: Coupling configuration
        """
        self.config = config or DAACHCouplingConfig()

        # Delay buffers for phase lag (circular buffers)
        buffer_size = self.config.delay_steps + 1
        self._da_buffer: deque = deque([0.5] * buffer_size, maxlen=buffer_size)
        self._ach_buffer: deque = deque([0.5] * buffer_size, maxlen=buffer_size)

        # History for correlation computation
        self._da_history: list[float] = []
        self._ach_history: list[float] = []
        self._max_history = 1000

        # Coupling strength modulation (can be learned)
        self._k_ach_to_da = self.config.k_ach_to_da
        self._k_da_to_ach = self.config.k_da_to_ach

        # Oscillator phase
        self._phase = 0.0

        # Statistics
        self._step_count = 0
        self._correlation_history: list[float] = []

        logger.info(
            f"DAACHCoupling initialized: "
            f"lag={self.config.phase_lag_ms}ms, "
            f"k_ach_da={self._k_ach_to_da}, k_da_ach={self._k_da_to_ach}"
        )

    def compute_coupling(
        self,
        da_level: float,
        ach_level: float,
        region: str = "limbic",
        time_ms: float | None = None
    ) -> tuple[float, float]:
        """
        Compute bidirectional coupling effects with phase lag.

        Args:
            da_level: Current dopamine level [0, 1]
            ach_level: Current acetylcholine level [0, 1]
            region: Striatal region for coupling scale
            time_ms: Current time in ms (for oscillation)

        Returns:
            (effect_on_da, effect_on_ach) - coupling contributions
        """
        # Get regional coupling scale
        if region == "limbic":
            scale = self.config.limbic_coupling_scale
        elif region == "motor":
            scale = self.config.motor_coupling_scale
        else:
            scale = self.config.associative_coupling_scale

        # Get delayed values from buffers
        delayed_da = self._da_buffer[0]    # Oldest value = most delayed
        delayed_ach = self._ach_buffer[0]

        # Compute coupling effects with delay
        # ACh(t-tau) inhibits DA(t)
        effect_on_da = self._k_ach_to_da * scale * delayed_ach

        # DA(t-tau) facilitates ACh(t)
        effect_on_ach = self._k_da_to_ach * scale * delayed_da

        # Add oscillatory modulation (theta-band)
        if self.config.enable_oscillation:
            if time_ms is not None:
                self._phase = 2 * np.pi * self.config.oscillation_freq_hz * time_ms / 1000.0
            else:
                self._phase += 2 * np.pi * self.config.oscillation_freq_hz * self.config.dt / 1000.0

            osc = self.config.oscillation_amplitude * np.sin(self._phase)
            # DA and ACh oscillate in antiphase
            effect_on_da += osc
            effect_on_ach -= osc  # Antiphase

        # Update buffers with current values
        self._da_buffer.append(da_level)
        self._ach_buffer.append(ach_level)

        # Track history for correlation
        self._da_history.append(da_level)
        self._ach_history.append(ach_level)
        if len(self._da_history) > self._max_history:
            self._da_history = self._da_history[-self._max_history:]
            self._ach_history = self._ach_history[-self._max_history:]

        self._step_count += 1

        return float(effect_on_da), float(effect_on_ach)

    def compute_coupling_field(
        self,
        da_field: np.ndarray,
        ach_field: np.ndarray,
        region_map: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute coupling effects across spatial field.

        Args:
            da_field: Dopamine spatial field [grid_size, ...]
            ach_field: Acetylcholine spatial field [grid_size, ...]
            region_map: Optional map of regions (0=limbic, 1=motor, 2=assoc)

        Returns:
            (da_coupling_field, ach_coupling_field)
        """
        # Flatten for vectorized computation
        da_flat = da_field.flatten()
        ach_flat = ach_field.flatten()

        # Initialize output
        da_effects = np.zeros_like(da_flat)
        ach_effects = np.zeros_like(ach_flat)

        # Compute point-wise (could be optimized with vectorization)
        for i in range(len(da_flat)):
            region = "limbic"  # Default
            if region_map is not None:
                region_idx = int(region_map.flatten()[i])
                region = ["limbic", "motor", "associative"][region_idx % 3]

            da_eff, ach_eff = self.compute_coupling(
                da_flat[i], ach_flat[i], region
            )
            da_effects[i] = da_eff
            ach_effects[i] = ach_eff

        return da_effects.reshape(da_field.shape), ach_effects.reshape(ach_field.shape)

    def update_from_rpe(self, rpe: float) -> None:
        """
        Update coupling strengths based on reward prediction error.

        Positive RPE strengthens DA->ACh (reward enhances attention)
        Negative RPE strengthens ACh->DA inhibition (uncertainty enhances braking)

        Args:
            rpe: Reward prediction error [-1, 1]
        """
        if not self.config.rpe_modulates_coupling:
            return

        lr = self.config.rpe_coupling_lr

        if rpe > 0:
            # Positive surprise: strengthen DA->ACh facilitation
            self._k_da_to_ach += lr * rpe
            self._k_da_to_ach = np.clip(self._k_da_to_ach, 0.05, 0.5)
        else:
            # Negative surprise: strengthen ACh->DA inhibition
            self._k_ach_to_da -= lr * abs(rpe)  # Make more negative
            self._k_ach_to_da = np.clip(self._k_ach_to_da, -0.5, -0.1)

        logger.debug(
            f"RPE={rpe:.3f} updated coupling: "
            f"k_da_ach={self._k_da_to_ach:.3f}, k_ach_da={self._k_ach_to_da:.3f}"
        )

    def get_correlation(self, window: int = 100) -> float:
        """
        Compute DA-ACh correlation over recent history.

        Should be negative (anticorrelated) for proper function.

        Args:
            window: Number of recent samples to use

        Returns:
            Pearson correlation coefficient
        """
        if len(self._da_history) < 10:
            return 0.0

        da = np.array(self._da_history[-window:])
        ach = np.array(self._ach_history[-window:])

        # Handle constant signals
        if np.std(da) < 1e-6 or np.std(ach) < 1e-6:
            return 0.0

        corr = np.corrcoef(da, ach)[0, 1]

        # Track correlation history
        self._correlation_history.append(corr)
        if len(self._correlation_history) > 1000:
            self._correlation_history = self._correlation_history[-1000:]

        return float(corr)

    def get_phase_difference(self, window: int = 100) -> float:
        """
        Compute phase difference between DA and ACh oscillations.

        Uses Hilbert transform to extract instantaneous phase.

        Args:
            window: Number of samples for analysis

        Returns:
            Phase difference in radians (positive = ACh leads)
        """
        if len(self._da_history) < window:
            return 0.0

        try:
            from scipy.signal import hilbert

            da = np.array(self._da_history[-window:])
            ach = np.array(self._ach_history[-window:])

            # Remove mean
            da = da - np.mean(da)
            ach = ach - np.mean(ach)

            # Hilbert transform for instantaneous phase
            da_analytic = hilbert(da)
            ach_analytic = hilbert(ach)

            da_phase = np.angle(da_analytic)
            ach_phase = np.angle(ach_analytic)

            # Mean phase difference
            phase_diff = np.mean(ach_phase - da_phase)

            # Wrap to [-pi, pi]
            phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

            return float(phase_diff)

        except ImportError:
            # Fallback: estimate from cross-correlation lag
            return self._estimate_phase_from_xcorr(window)

    def _estimate_phase_from_xcorr(self, window: int = 100) -> float:
        """Estimate phase difference from cross-correlation."""
        if len(self._da_history) < window:
            return 0.0

        da = np.array(self._da_history[-window:])
        ach = np.array(self._ach_history[-window:])

        # Normalize
        da = (da - np.mean(da)) / (np.std(da) + 1e-6)
        ach = (ach - np.mean(ach)) / (np.std(ach) + 1e-6)

        # Cross-correlation
        xcorr = np.correlate(da, ach, mode='full')
        lags = np.arange(-len(da) + 1, len(da))

        # Find peak
        peak_idx = np.argmax(xcorr)
        peak_lag = lags[peak_idx]

        # Convert lag to phase (assuming oscillation period from config)
        period_samples = 1000.0 / (self.config.oscillation_freq_hz * self.config.dt)
        phase_diff = 2 * np.pi * peak_lag / period_samples

        return float(phase_diff)

    def validate_anticorrelation(self) -> dict:
        """
        Validate that DA-ACh dynamics meet biological criteria.

        Returns:
            Validation results with pass/fail status
        """
        corr = self.get_correlation()
        phase = self.get_phase_difference()

        # Expected phase lag in radians for ~100ms at theta
        # At 4Hz (250ms period), 100ms = 0.4 * 2*pi ≈ 2.5 rad
        expected_phase_lag = 2 * np.pi * self.config.phase_lag_ms / 1000.0 * self.config.oscillation_freq_hz

        results = {
            "correlation": corr,
            "target_correlation": self.config.target_anticorrelation,
            "correlation_pass": corr < self.config.target_anticorrelation,
            "phase_difference_rad": phase,
            "expected_phase_lag_rad": expected_phase_lag,
            "phase_lag_error_rad": abs(phase - expected_phase_lag),
            "phase_pass": abs(phase - expected_phase_lag) < 0.5,  # Within 0.5 rad
            "k_ach_to_da": self._k_ach_to_da,
            "k_da_to_ach": self._k_da_to_ach,
            "samples": len(self._da_history),
        }

        results["overall_pass"] = results["correlation_pass"] and results["phase_pass"]

        return results

    def get_current_state(self) -> DAACHState:
        """Get current coupling state."""
        corr = self.get_correlation() if len(self._da_history) > 10 else 0.0
        phase = self.get_phase_difference() if len(self._da_history) > 50 else 0.0

        da = self._da_history[-1] if self._da_history else 0.5
        ach = self._ach_history[-1] if self._ach_history else 0.5

        return DAACHState(
            da_level=da,
            ach_level=ach,
            da_to_ach_effect=self._k_da_to_ach * self._da_buffer[0],
            ach_to_da_effect=self._k_ach_to_da * self._ach_buffer[0],
            correlation=corr,
            phase_difference=phase,
            timestamp_ms=self._step_count * self.config.dt
        )

    def get_stats(self) -> dict:
        """Get coupling statistics."""
        stats = {
            "step_count": self._step_count,
            "k_ach_to_da": self._k_ach_to_da,
            "k_da_to_ach": self._k_da_to_ach,
            "phase_lag_ms": self.config.phase_lag_ms,
            "delay_steps": self.config.delay_steps,
            "history_size": len(self._da_history),
        }

        if len(self._da_history) > 10:
            stats["current_correlation"] = self.get_correlation()
            stats["mean_da"] = float(np.mean(self._da_history[-100:]))
            stats["mean_ach"] = float(np.mean(self._ach_history[-100:]))
            stats["std_da"] = float(np.std(self._da_history[-100:]))
            stats["std_ach"] = float(np.std(self._ach_history[-100:]))

        if self._correlation_history:
            stats["mean_correlation"] = float(np.mean(self._correlation_history))
            stats["correlation_trend"] = (
                float(np.mean(self._correlation_history[-50:]) -
                      np.mean(self._correlation_history[:50]))
                if len(self._correlation_history) > 100 else 0.0
            )

        return stats

    def reset(self) -> None:
        """Reset coupling state."""
        buffer_size = self.config.delay_steps + 1
        self._da_buffer = deque([0.5] * buffer_size, maxlen=buffer_size)
        self._ach_buffer = deque([0.5] * buffer_size, maxlen=buffer_size)
        self._da_history = []
        self._ach_history = []
        self._k_ach_to_da = self.config.k_ach_to_da
        self._k_da_to_ach = self.config.k_da_to_ach
        self._phase = 0.0
        self._step_count = 0
        self._correlation_history = []


__all__ = [
    "DAACHCoupling",
    "DAACHCouplingConfig",
    "DAACHState",
]
