"""
Substantia Nigra pars compacta (SNc) Dopamine Circuit for T4DM.

Biological Basis:
- SNc contains dopamine neurons projecting to striatum (nigrostriatal pathway)
- SNc DA modulates motor action selection and procedural learning
- Distinct from VTA: SNc → motor circuits, VTA → limbic/reward circuits
- SNc firing encodes action vigor and movement initiation
- Two firing modes:
  - Tonic: baseline 4-6 Hz, sets motor readiness
  - Phasic: bursts for action initiation, pauses for action suppression

Architecture:
- Receives cortical motor signals and striatal feedback
- Produces phasic DA bursts for action selection
- Connects to striatal D1 (go) and D2 (no-go) pathways
- Integrates with procedural memory consolidation

Integration:
- StrialMSN: SNc DA modulates D1/D2 pathway balance
- ProceduralMemory: SNc signals action value for procedural learning
- NeuralFieldSolver: inject DA for motor-related modulation

References:
- Schultz (2007): SNc dopamine in motor control
- Howe & Dombeck (2016): SNc phasic activity in motor learning
- Surmeier et al. (2007): SNc-striatal DA transmission
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SNcFiringMode(Enum):
    """SNc dopamine neuron firing modes."""
    TONIC = "tonic"         # Baseline sustained firing (4-6 Hz)
    PHASIC_BURST = "burst"  # Action initiation (20-40 Hz)
    PHASIC_PAUSE = "pause"  # Action suppression (0-2 Hz)


@dataclass
class SNcConfig:
    """Configuration for SNc circuit dynamics."""

    # Tonic firing parameters
    tonic_rate: float = 5.0        # Hz, baseline firing rate
    tonic_da_level: float = 0.4    # Baseline DA concentration [0, 1]

    # Phasic dynamics
    burst_peak_rate: float = 35.0  # Hz, during action initiation
    burst_duration: float = 0.15   # seconds
    pause_duration: float = 0.25   # seconds, during action suppression

    # Motor signal -> DA conversion
    motor_to_da_gain: float = 0.6  # How much motor signal changes DA

    # Decay dynamics
    tau_decay: float = 0.15        # 150ms time constant for exponential decay

    # Motor vigor scaling
    vigor_min: float = 0.3         # Minimum action vigor
    vigor_max: float = 1.0         # Maximum action vigor

    # Biological constraints
    min_da: float = 0.05           # Minimum DA (even during pause)
    max_da: float = 0.95           # Maximum DA (saturation)
    refractory_period: float = 0.08  # Min time between phasic events


@dataclass
class SNcState:
    """Current state of SNc circuit."""

    firing_mode: SNcFiringMode = SNcFiringMode.TONIC
    current_da: float = 0.4        # Current DA concentration
    current_rate: float = 5.0      # Current firing rate (Hz)

    # Motor tracking
    last_motor_signal: float = 0.0
    action_vigor: float = 0.5      # Current action vigor [0, 1]

    # Timing
    time_since_phasic: float = 1.0  # Seconds since last burst/pause
    phasic_remaining: float = 0.0   # Remaining phasic event duration

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "firing_mode": self.firing_mode.value,
            "current_da": self.current_da,
            "current_rate": self.current_rate,
            "last_motor_signal": self.last_motor_signal,
            "action_vigor": self.action_vigor,
        }


class SubstantiaNigraCircuit:
    """
    Substantia Nigra pars compacta dopamine circuit.

    Models the dynamics of SNc dopamine neurons that:
    1. Maintain tonic DA levels for motor readiness
    2. Generate phasic bursts for action initiation
    3. Generate phasic pauses for action suppression
    4. Modulate striatal D1/D2 pathways for action selection

    Key innovation: Motor-specific dopamine separate from VTA reward DA,
    enabling procedural learning independent of reward valuation.
    """

    def __init__(self, config: SNcConfig | None = None):
        """
        Initialize SNc circuit.

        Args:
            config: SNc configuration parameters
        """
        self.config = config or SNcConfig()
        self.state = SNcState(
            current_da=self.config.tonic_da_level,
            current_rate=self.config.tonic_rate
        )

        # Track history
        self._da_history: list[float] = []
        self._motor_history: list[float] = []
        self._max_history = 1000

        logger.info(
            f"SNcCircuit initialized: tonic_da={self.config.tonic_da_level}, "
            f"tonic_rate={self.config.tonic_rate}Hz"
        )

    # =========================================================================
    # Core Motor Signal Processing
    # =========================================================================

    def process_motor_signal(self, motor_signal: float, dt: float = 0.1) -> float:
        """
        Process motor command signal to generate dopamine response.

        Converts motor intention/execution to DA concentration change.

        Args:
            motor_signal: Motor command strength [-1, 1]
                          positive = initiate action
                          negative = suppress action
            dt: Timestep (seconds)

        Returns:
            New DA concentration
        """
        # Store signal
        self.state.last_motor_signal = motor_signal
        self._motor_history.append(motor_signal)
        if len(self._motor_history) > self._max_history:
            self._motor_history = self._motor_history[-self._max_history:]

        # Determine firing mode based on motor signal
        if abs(motor_signal) < 0.05:
            # Near-zero signal: maintain tonic
            self._to_tonic_mode(dt)
        elif motor_signal > 0.05:
            # Positive signal: burst for action initiation
            self._to_burst_mode(motor_signal, dt)
        else:
            # Negative signal: pause for action suppression
            self._to_pause_mode(motor_signal, dt)

        # Update action vigor based on DA level
        self._update_action_vigor()

        return self.state.current_da

    def _to_tonic_mode(self, dt: float) -> None:
        """
        Transition to/maintain tonic firing.

        Uses exponential decay to baseline with tau_decay time constant.
        """
        self.state.firing_mode = SNcFiringMode.TONIC
        self.state.current_rate = self.config.tonic_rate

        # Exponential decay to tonic DA level
        da_target = self.config.tonic_da_level
        da_level = self.state.current_da
        self.state.current_da = da_target + (da_level - da_target) * np.exp(-dt / self.config.tau_decay)

        # Update timing
        self.state.time_since_phasic += dt
        self.state.phasic_remaining = 0.0

    def _to_burst_mode(self, motor_signal: float, dt: float) -> None:
        """Generate phasic burst for action initiation."""
        # Check refractory period
        if self.state.time_since_phasic < self.config.refractory_period:
            self._to_tonic_mode(dt)
            return

        self.state.firing_mode = SNcFiringMode.PHASIC_BURST

        # Scale firing rate with motor signal magnitude
        rate_scale = min(motor_signal / 0.5, 1.0)  # Saturate at 0.5 signal
        self.state.current_rate = (
            self.config.tonic_rate +
            (self.config.burst_peak_rate - self.config.tonic_rate) * rate_scale
        )

        # Increase DA proportional to motor signal
        da_increase = motor_signal * self.config.motor_to_da_gain
        self.state.current_da = np.clip(
            self.state.current_da + da_increase,
            self.config.min_da,
            self.config.max_da
        )

        # Set phasic timing
        self.state.time_since_phasic = 0.0
        self.state.phasic_remaining = self.config.burst_duration

    def _to_pause_mode(self, motor_signal: float, dt: float) -> None:
        """Generate phasic pause for action suppression."""
        # Check refractory period
        if self.state.time_since_phasic < self.config.refractory_period:
            self._to_tonic_mode(dt)
            return

        self.state.firing_mode = SNcFiringMode.PHASIC_PAUSE

        # Reduce firing rate proportional to negative motor signal
        rate_scale = min(abs(motor_signal) / 0.5, 1.0)
        self.state.current_rate = self.config.tonic_rate * (1.0 - 0.8 * rate_scale)

        # Decrease DA proportional to signal magnitude
        da_decrease = abs(motor_signal) * self.config.motor_to_da_gain
        self.state.current_da = np.clip(
            self.state.current_da - da_decrease,
            self.config.min_da,
            self.config.max_da
        )

        # Set phasic timing
        self.state.time_since_phasic = 0.0
        self.state.phasic_remaining = self.config.pause_duration

    def _update_action_vigor(self) -> None:
        """Update action vigor based on current DA level."""
        # Linear mapping from DA to vigor
        da_normalized = (self.state.current_da - self.config.min_da) / (
            self.config.max_da - self.config.min_da
        )
        self.state.action_vigor = (
            self.config.vigor_min +
            (self.config.vigor_max - self.config.vigor_min) * da_normalized
        )

    def step(self, dt: float = 0.1) -> None:
        """
        Advance SNc dynamics by one timestep without new motor signal.

        Handles decay back to tonic state.
        ATOM-P3-13: Send DA to striatum if connected.

        Args:
            dt: Timestep in seconds
        """
        self.state.time_since_phasic += dt

        # Decay phasic effects
        if self.state.phasic_remaining > 0:
            self.state.phasic_remaining -= dt
            if self.state.phasic_remaining <= 0:
                self._to_tonic_mode(dt)
        else:
            # Passive decay to tonic
            self._to_tonic_mode(dt)

        # ATOM-P3-13: Send DA to striatum if connected
        if hasattr(self, '_striatum') and self._striatum is not None:
            if hasattr(self._striatum, 'set_dopamine_level'):
                self._striatum.set_dopamine_level(self.state.current_da)
            elif hasattr(self._striatum, 'receive_da'):
                self._striatum.receive_da(self.state.current_da)

        # Track history
        self._da_history.append(self.state.current_da)
        if len(self._da_history) > self._max_history:
            self._da_history = self._da_history[-self._max_history:]

    # =========================================================================
    # Integration Methods
    # =========================================================================

    def receive_cortical_input(self, cortical_signal: float) -> None:
        """
        Receive cortical motor command.

        Args:
            cortical_signal: Cortical motor planning signal [0, 1]
        """
        # Convert to motor signal (centered at 0)
        motor_signal = (cortical_signal - 0.5) * 2.0
        self.process_motor_signal(motor_signal)

    def emit_da(self) -> float:
        """
        Emit current DA level for striatal targets.

        Returns:
            DA concentration [0, 1]
        """
        return self.state.current_da

    def get_da_for_neural_field(self) -> float:
        """
        Get DA level for injection into NeuralFieldSolver.

        Returns:
            DA concentration [0, 1]
        """
        return self.state.current_da

    def get_action_vigor(self) -> float:
        """
        Get current action vigor signal.

        Returns:
            Action vigor [0, 1]
        """
        return self.state.action_vigor

    def connect_to_striatum(self, striatum=None) -> None:
        """
        Connect SNc dopamine output to striatal MSNs.

        ATOM-P3-13: Implemented connection pattern for SNc→striatum projection.

        Args:
            striatum: StriatalMSN instance or compatible object with receive_da() method
        """
        self._striatum = striatum
        logger.info("SNc connected to striatum")

    # =========================================================================
    # Statistics and State
    # =========================================================================

    def get_stats(self) -> dict:
        """Get SNc circuit statistics."""
        stats = {
            "firing_mode": self.state.firing_mode.value,
            "current_da": self.state.current_da,
            "current_rate": self.state.current_rate,
            "action_vigor": self.state.action_vigor,
            "last_motor_signal": self.state.last_motor_signal,
        }

        if self._da_history:
            stats["avg_da"] = float(np.mean(self._da_history))
        if self._motor_history:
            stats["avg_motor"] = float(np.mean(self._motor_history))

        return stats

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = SNcState(
            current_da=self.config.tonic_da_level,
            current_rate=self.config.tonic_rate
        )
        self._da_history.clear()
        self._motor_history.clear()
        logger.info("SNcCircuit reset")

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "state": self.state.to_dict(),
            "config": {
                "tonic_da_level": self.config.tonic_da_level,
                "tonic_rate": self.config.tonic_rate,
                "motor_to_da_gain": self.config.motor_to_da_gain,
            }
        }

    def load_state(self, saved: dict) -> None:
        """Load state from persistence."""
        if "state" in saved:
            s = saved["state"]
            self.state.current_da = s.get("current_da", self.config.tonic_da_level)
            self.state.current_rate = s.get("current_rate", self.config.tonic_rate)
            self.state.action_vigor = s.get("action_vigor", 0.5)


def create_snc_circuit(
    tonic_da: float = 0.4,
    motor_gain: float = 0.6,
) -> SubstantiaNigraCircuit:
    """
    Factory function to create SNc circuit with common configurations.

    Args:
        tonic_da: Baseline DA level
        motor_gain: Motor signal to DA conversion gain

    Returns:
        Configured SubstantiaNigraCircuit
    """
    config = SNcConfig(
        tonic_da_level=tonic_da,
        motor_to_da_gain=motor_gain,
    )
    return SubstantiaNigraCircuit(config)


__all__ = [
    "SubstantiaNigraCircuit",
    "SNcConfig",
    "SNcState",
    "SNcFiringMode",
    "create_snc_circuit",
]
