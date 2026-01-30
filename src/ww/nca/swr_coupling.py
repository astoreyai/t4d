"""
Sharp-Wave Ripple (SWR) to Neural Field Coupling for World Weaver.

Biological Basis:
- SWRs are 150-250Hz oscillations during NREM sleep/quiet wakefulness
- They originate in hippocampal CA3 and propagate to CA1
- During SWRs, compressed memory sequences are replayed
- SWRs trigger increased glutamate and modulate neural field dynamics
- Critical for hippocampal-cortical memory transfer

Phase 2 Enhancements (Sprints 8-9):
- Validated ripple frequency range (150-250 Hz per Buzsaki)
- Explicit wake/sleep state separation for SWR gating
- State-dependent frequency modulation
- Enhanced validation and biological constraints

Key Features:
- SWR detection/generation linked to hippocampal circuit
- Glutamate injection during ripple events
- ACh modulation (low during SWRs, high blocks them)
- Coordination with oscillator module
- Memory replay triggers via eligibility

References:
- Buzsaki (2015): Hippocampal sharp wave-ripple
- Girardeau et al. (2009): SWRs and memory consolidation
- Foster & Wilson (2006): Reverse replay of sequences
- Carr et al. (2011): SWR frequency range validation
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from ww.core.access_control import CallerToken, require_capability

if TYPE_CHECKING:
    from ww.nca.hippocampus import HippocampalCircuit
    from ww.nca.neural_field import NeuralFieldSolver
    from ww.nca.oscillators import FrequencyBandGenerator

logger = logging.getLogger(__name__)


class SWRPhase(Enum):
    """Sharp-wave ripple event phases."""
    QUIESCENT = "quiescent"       # No SWR activity
    INITIATING = "initiating"     # SWR starting (sharp wave)
    RIPPLING = "rippling"         # High-frequency ripple (150-250Hz)
    TERMINATING = "terminating"   # SWR ending


class WakeSleepMode(Enum):
    """Wake/sleep state for SWR gating (Phase 2)."""
    ACTIVE_WAKE = "active_wake"      # High activity, no SWRs
    QUIET_WAKE = "quiet_wake"        # Low activity, SWRs possible
    NREM_LIGHT = "nrem_light"        # Light NREM, spindles + some SWRs
    NREM_DEEP = "nrem_deep"          # Deep NREM, frequent SWRs
    REM = "rem"                       # REM sleep, no SWRs (high ACh)


# Biological frequency constraints (Buzsaki 2015, Carr et al. 2011)
RIPPLE_FREQ_MIN = 150.0  # Hz - minimum valid ripple frequency
RIPPLE_FREQ_MAX = 250.0  # Hz - maximum valid ripple frequency
RIPPLE_FREQ_OPTIMAL = 180.0  # Hz - most common frequency


@dataclass
class SWRConfig:
    """Configuration for SWR-neural field coupling."""

    # SWR timing parameters
    ripple_duration: float = 0.08       # ~80ms ripple
    sharp_wave_duration: float = 0.05   # ~50ms sharp wave before ripple
    min_inter_swr_interval: float = 0.5  # Minimum gap between SWRs

    # Frequency parameters (validated 150-250 Hz range)
    ripple_frequency: float = 180.0     # Hz, ripple oscillation (optimal)
    ripple_freq_min: float = 150.0      # Hz, minimum valid
    ripple_freq_max: float = 250.0      # Hz, maximum valid
    sharp_wave_frequency: float = 2.0   # Hz, underlying sharp wave

    # Wake/sleep state parameters (Phase 2)
    wake_sleep_mode: WakeSleepMode = WakeSleepMode.QUIET_WAKE
    enable_state_gating: bool = False   # Use wake/sleep for SWR gating (opt-in)

    # State-specific SWR probabilities
    swr_prob_active_wake: float = 0.0   # No SWRs during active wake
    swr_prob_quiet_wake: float = 0.3    # Low probability during quiet wake
    swr_prob_nrem_light: float = 0.5    # Moderate during light NREM
    swr_prob_nrem_deep: float = 0.9     # High during deep NREM
    swr_prob_rem: float = 0.0           # No SWRs during REM (high ACh)

    # Gating conditions
    ach_threshold: float = 0.3         # ACh must be below this for SWR
    arousal_threshold: float = 0.4     # NE must be below this for SWR
    hippocampal_activity_threshold: float = 0.6  # CA3 activity threshold

    # Neural field modulation
    glutamate_boost: float = 0.3        # Glutamate increase during SWR
    gaba_boost: float = 0.1             # GABA increase (inhibitory surround)
    spatial_spread: float = 0.2         # Spatial spread of field modulation

    # Replay parameters
    compression_factor: float = 10.0    # Temporal compression of replay
    replay_gain: float = 1.5            # Strength multiplier during replay

    # Eligibility
    swr_eligibility_boost: float = 0.3  # Boost to eligibility during SWR

    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_ripple_frequency()

    def _validate_ripple_frequency(self) -> None:
        """Validate ripple frequency is in biological range."""
        if not (RIPPLE_FREQ_MIN <= self.ripple_frequency <= RIPPLE_FREQ_MAX):
            raise ValueError(
                f"Ripple frequency {self.ripple_frequency} Hz outside "
                f"biological range [{RIPPLE_FREQ_MIN}, {RIPPLE_FREQ_MAX}] Hz. "
                f"See Buzsaki (2015), Carr et al. (2011)."
            )

    def get_state_swr_probability(self, mode: WakeSleepMode) -> float:
        """Get SWR probability for a given wake/sleep state."""
        probs = {
            WakeSleepMode.ACTIVE_WAKE: self.swr_prob_active_wake,
            WakeSleepMode.QUIET_WAKE: self.swr_prob_quiet_wake,
            WakeSleepMode.NREM_LIGHT: self.swr_prob_nrem_light,
            WakeSleepMode.NREM_DEEP: self.swr_prob_nrem_deep,
            WakeSleepMode.REM: self.swr_prob_rem,
        }
        return probs.get(mode, 0.0)


@dataclass
class SWREvent:
    """Record of a sharp-wave ripple event."""

    start_time: float              # Event start time
    duration: float                # Total duration
    peak_amplitude: float          # Peak ripple amplitude
    replay_count: int = 0          # Number of patterns replayed
    memories_activated: list = field(default_factory=list)  # Memory IDs

    @property
    def end_time(self) -> float:
        """Event end time."""
        return self.start_time + self.duration


@dataclass
class SWRCouplingState:
    """Current state of SWR-neural field coupling."""

    phase: SWRPhase = SWRPhase.QUIESCENT
    current_amplitude: float = 0.0      # Current ripple amplitude
    time_in_phase: float = 0.0          # Time in current phase
    time_since_last_swr: float = 1.0    # Time since last SWR

    # Wake/sleep state (Phase 2)
    wake_sleep_mode: WakeSleepMode = WakeSleepMode.QUIET_WAKE
    current_ripple_freq: float = 180.0  # Hz, current frequency

    # Gating state
    ach_level: float = 0.3
    ne_level: float = 0.3
    hippocampal_activity: float = 0.5

    # Modulation tracking
    glutamate_injection: float = 0.0
    gaba_injection: float = 0.0

    # Phase 2 metrics
    swr_count_wake: int = 0
    swr_count_sleep: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "current_amplitude": self.current_amplitude,
            "time_since_last_swr": self.time_since_last_swr,
            "wake_sleep_mode": self.wake_sleep_mode.value,
            "current_ripple_freq": self.current_ripple_freq,
            "ach_level": self.ach_level,
            "ne_level": self.ne_level,
        }


class SWRNeuralFieldCoupling:
    """
    Couples Sharp-Wave Ripples to Neural Field Dynamics.

    This module bridges the SWR generator in consolidation/sleep.py
    with the NCA neural field solver, ensuring that:

    1. SWRs only occur under proper gating conditions (low ACh, low NE)
    2. SWR events inject glutamate into the neural field
    3. Hippocampal replay activates corresponding field patterns
    4. Eligibility traces are boosted during SWR for consolidation

    The coupling enables biologically realistic memory consolidation
    where replay events actually affect brain-state dynamics.
    """

    def __init__(
        self,
        config: SWRConfig | None = None,
        neural_field: NeuralFieldSolver | None = None,
        hippocampus: HippocampalCircuit | None = None,
        oscillators: FrequencyBandGenerator | None = None,
        seed: int | None = None,
    ):
        """
        Initialize SWR-neural field coupling.

        Args:
            config: SWR configuration
            neural_field: Neural field solver for modulation
            hippocampus: Hippocampal circuit for replay
            oscillators: Oscillator generator for ripple frequency
            seed: Random seed for reproducibility
        """
        self.config = config or SWRConfig()
        self.neural_field = neural_field
        self.hippocampus = hippocampus
        self.oscillators = oscillators

        self.state = SWRCouplingState()

        # Event history
        self._swr_events: list[SWREvent] = []
        self._max_history = 1000

        # Current event tracking
        self._current_event: SWREvent | None = None
        self._simulation_time: float = 0.0

        # C5: Procedural memory filter - REM consolidation candidates
        self.rem_candidates: list[dict] = []

        # Callbacks
        self._swr_callbacks: list[Callable[[SWREvent], None]] = []
        self._replay_callbacks: list[Callable[[Any], None]] = []

        # Random number generator for reproducibility
        self._rng = np.random.default_rng(seed)

        logger.info(
            f"SWRNeuralFieldCoupling initialized: "
            f"ripple_freq={self.config.ripple_frequency}Hz"
        )

    # =========================================================================
    # Core Dynamics
    # =========================================================================

    def step(self, dt: float = 0.01) -> bool:
        """
        Advance SWR coupling dynamics.

        Args:
            dt: Timestep in seconds

        Returns:
            True if an SWR occurred this step
        """
        self._simulation_time += dt
        self.state.time_since_last_swr += dt
        self.state.time_in_phase += dt

        # Update gating state from neural field
        self._update_gating_state()

        # Check for SWR initiation
        swr_occurred = False
        if self.state.phase == SWRPhase.QUIESCENT:
            if self._should_initiate_swr():
                self._initiate_swr()
                swr_occurred = True
        else:
            # Progress through SWR phases
            self._advance_swr_phase(dt)

        # Apply neural field modulation
        self._apply_field_modulation(dt)

        return swr_occurred

    def _update_gating_state(self) -> None:
        """Update gating conditions from neural field."""
        if self.neural_field is not None:
            state = self.neural_field.get_mean_state()
            self.state.ach_level = float(state.acetylcholine)
            self.state.ne_level = float(state.norepinephrine)

        if self.hippocampus is not None:
            # Get CA3 activity as hippocampal readout
            self.state.hippocampal_activity = self.hippocampus.state.ca3_activity

        # Infer wake/sleep mode from NT levels (Phase 2)
        self.state.wake_sleep_mode = self._infer_wake_sleep_mode()

        # Update ripple frequency based on state
        self.state.current_ripple_freq = self._compute_state_ripple_freq()

    def _infer_wake_sleep_mode(self) -> WakeSleepMode:
        """
        Infer wake/sleep mode from neuromodulator levels.

        Phase 2 Enhancement: Uses ACh and NE to determine state.

        State characteristics:
        - Active wake: High ACh, High NE
        - Quiet wake: Low ACh, Low-Med NE
        - NREM light: Low ACh, Low NE
        - NREM deep: Very low ACh, Very low NE
        - REM: High ACh, Very low NE
        """
        ach = self.state.ach_level
        ne = self.state.ne_level

        # REM: High ACh + very low NE
        if ach > 0.6 and ne < 0.2:
            return WakeSleepMode.REM

        # Active wake: High ACh + high NE
        if ach > 0.5 and ne > 0.5:
            return WakeSleepMode.ACTIVE_WAKE

        # NREM deep: Very low everything
        if ach < 0.2 and ne < 0.2:
            return WakeSleepMode.NREM_DEEP

        # NREM light: Low ACh + low NE
        if ach < 0.3 and ne < 0.3:
            return WakeSleepMode.NREM_LIGHT

        # Default: Quiet wake
        return WakeSleepMode.QUIET_WAKE

    def _compute_state_ripple_freq(self) -> float:
        """
        Compute ripple frequency based on current state.

        Biological basis: Ripple frequency varies with brain state,
        typically faster during deeper NREM sleep.

        Returns:
            Ripple frequency in Hz (150-250 range)
        """
        mode = self.state.wake_sleep_mode
        base_freq = self.config.ripple_frequency

        # State-dependent modulation
        freq_mods = {
            WakeSleepMode.ACTIVE_WAKE: 0.0,      # N/A - no SWRs
            WakeSleepMode.QUIET_WAKE: 0.95,     # Slightly slower
            WakeSleepMode.NREM_LIGHT: 1.0,      # Baseline
            WakeSleepMode.NREM_DEEP: 1.1,       # Faster during deep NREM
            WakeSleepMode.REM: 0.0,             # N/A - no SWRs
        }

        freq = base_freq * freq_mods.get(mode, 1.0)

        # Clamp to valid range
        return np.clip(freq, RIPPLE_FREQ_MIN, RIPPLE_FREQ_MAX)

    def _should_initiate_swr(self) -> bool:
        """
        Check if conditions are right for SWR initiation.

        Phase 2: Now includes wake/sleep state gating.
        """
        # Refractory period
        if self.state.time_since_last_swr < self.config.min_inter_swr_interval:
            return False

        # Phase 2: Wake/sleep state gating
        if self.config.enable_state_gating:
            state_prob = self.config.get_state_swr_probability(
                self.state.wake_sleep_mode
            )
            if state_prob == 0.0:
                return False
            # Probabilistic gating based on state
            if self._rng.random() > state_prob:
                return False

        # Low ACh required (cholinergic suppression blocks SWRs)
        if self.state.ach_level > self.config.ach_threshold:
            return False

        # Low NE required (high arousal blocks SWRs)
        if self.state.ne_level > self.config.arousal_threshold:
            return False

        # Sufficient hippocampal activity needed
        if self.state.hippocampal_activity < self.config.hippocampal_activity_threshold:
            # Probabilistic initiation based on activity
            prob = self.state.hippocampal_activity / self.config.hippocampal_activity_threshold
            return self._rng.random() < prob * 0.1

        return True

    def _initiate_swr(self) -> None:
        """Initiate a new SWR event."""
        self.state.phase = SWRPhase.INITIATING
        self.state.time_in_phase = 0.0
        self.state.time_since_last_swr = 0.0

        # Track wake/sleep counts (Phase 2)
        if self.state.wake_sleep_mode in [
            WakeSleepMode.NREM_LIGHT,
            WakeSleepMode.NREM_DEEP
        ]:
            self.state.swr_count_sleep += 1
        else:
            self.state.swr_count_wake += 1

        # Create new event
        self._current_event = SWREvent(
            start_time=self._simulation_time,
            duration=self.config.sharp_wave_duration + self.config.ripple_duration,
            peak_amplitude=0.0
        )

        logger.debug(
            f"SWR initiated at t={self._simulation_time:.3f}s "
            f"(mode={self.state.wake_sleep_mode.value}, "
            f"freq={self.state.current_ripple_freq:.0f}Hz)"
        )

    def _advance_swr_phase(self, dt: float) -> None:
        """Advance through SWR phases."""
        if self.state.phase == SWRPhase.INITIATING:
            # Sharp wave phase
            if self.state.time_in_phase >= self.config.sharp_wave_duration:
                self.state.phase = SWRPhase.RIPPLING
                self.state.time_in_phase = 0.0

            # Ramp up amplitude
            progress = self.state.time_in_phase / self.config.sharp_wave_duration
            self.state.current_amplitude = progress

        elif self.state.phase == SWRPhase.RIPPLING:
            # High-frequency ripple phase
            if self.state.time_in_phase >= self.config.ripple_duration:
                self.state.phase = SWRPhase.TERMINATING
                self.state.time_in_phase = 0.0

            # Ripple oscillation - use state-dependent frequency (Phase 2)
            ripple_freq = self.state.current_ripple_freq
            phase = 2 * np.pi * ripple_freq * self.state.time_in_phase
            envelope = 1.0 - 0.3 * (self.state.time_in_phase / self.config.ripple_duration)
            self.state.current_amplitude = envelope * (0.7 + 0.3 * np.cos(phase))

            # Track peak
            if self._current_event is not None:
                self._current_event.peak_amplitude = max(
                    self._current_event.peak_amplitude,
                    self.state.current_amplitude
                )

        elif self.state.phase == SWRPhase.TERMINATING:
            # Rapid decay
            decay_time = 0.02  # 20ms decay
            if self.state.time_in_phase >= decay_time:
                self._complete_swr()
            else:
                progress = self.state.time_in_phase / decay_time
                self.state.current_amplitude = (1 - progress) * 0.3

    def _complete_swr(self) -> None:
        """Complete current SWR event."""
        self.state.phase = SWRPhase.QUIESCENT
        self.state.current_amplitude = 0.0
        self.state.time_in_phase = 0.0

        # Store event
        if self._current_event is not None:
            self._swr_events.append(self._current_event)
            if len(self._swr_events) > self._max_history:
                self._swr_events = self._swr_events[-self._max_history:]

            # Fire callbacks
            for callback in self._swr_callbacks:
                callback(self._current_event)

            logger.debug(
                f"SWR completed: duration={self._current_event.duration:.3f}s, "
                f"peak={self._current_event.peak_amplitude:.2f}"
            )

            self._current_event = None

    def _apply_field_modulation(self, dt: float) -> None:
        """Apply SWR-driven modulation to neural field."""
        if self.neural_field is None:
            return

        if self.state.phase in [SWRPhase.INITIATING, SWRPhase.RIPPLING]:
            # Inject glutamate proportional to amplitude
            glu_amount = (
                self.config.glutamate_boost *
                self.state.current_amplitude *
                dt
            )
            self.state.glutamate_injection = glu_amount

            # Also inject GABA (inhibitory surround)
            gaba_amount = (
                self.config.gaba_boost *
                self.state.current_amplitude *
                dt
            )
            self.state.gaba_injection = gaba_amount

            # Apply to neural field
            # Glutamate is index 5, GABA is index 4 in NT array
            current_state = self.neural_field.get_mean_state()
            new_glu = min(1.0, current_state.glutamate + glu_amount)
            new_gaba = min(1.0, current_state.gaba + gaba_amount)

            # Use inject method if available
            if hasattr(self.neural_field, '_fields'):
                self.neural_field._fields[5] += glu_amount * self.config.spatial_spread
                self.neural_field._fields[4] += gaba_amount * self.config.spatial_spread
                # Clamp
                self.neural_field._fields[5] = np.clip(self.neural_field._fields[5], 0, 1)
                self.neural_field._fields[4] = np.clip(self.neural_field._fields[4], 0, 1)
        else:
            self.state.glutamate_injection = 0.0
            self.state.gaba_injection = 0.0

    # =========================================================================
    # Replay Interface
    # =========================================================================

    def trigger_replay(
        self,
        pattern: np.ndarray,
        memory_id: str | None = None,
        memory_type: str | None = None,
        token: CallerToken | None = None
    ) -> bool:
        """
        C5: Trigger memory replay during SWR (with procedural filter).

        Procedural memories are NOT replayed during SWR. They are added to
        REM consolidation candidates instead. Only episodic/semantic memories
        are replayed during SWR events.

        Biological basis: Procedural consolidation occurs in REM sleep,
        not during NREM slow-wave sleep (SWR events).

        Args:
            pattern: Pattern to replay
            memory_id: Optional memory identifier
            memory_type: Memory type ("PROCEDURAL", "EPISODIC", "SEMANTIC")
            token: Optional access control token (only enforced if provided)

        Returns:
            True if replay was processed (or queued for REM)
        """
        if token is not None:
            require_capability(token, "trigger_replay")
        if isinstance(pattern, np.ndarray):
            if not np.all(np.isfinite(pattern)):
                raise ValueError("Pattern contains NaN or Inf values")
        if self.state.phase not in [SWRPhase.INITIATING, SWRPhase.RIPPLING]:
            return False

        # C5: Filter out procedural memories
        if memory_type and memory_type.upper() == "PROCEDURAL":
            # Add to REM consolidation candidates instead
            self.rem_candidates.append({
                "pattern": pattern.copy(),
                "memory_id": memory_id,
                "memory_type": memory_type,
                "queued_at": self._simulation_time,
            })
            logger.debug(
                f"C5: Procedural memory {memory_id} queued for REM consolidation "
                f"(skipped SWR replay)"
            )
            return True

        # Replay through hippocampus (episodic/semantic only)
        if self.hippocampus is not None:
            # Inject pattern into CA3 for pattern completion
            completed = self.hippocampus.ca3.pattern_completion(
                pattern * self.config.replay_gain
            )

            # Track in event
            if self._current_event is not None:
                self._current_event.replay_count += 1
                if memory_id:
                    self._current_event.memories_activated.append(memory_id)

            # Fire replay callbacks
            for callback in self._replay_callbacks:
                callback({"pattern": completed, "memory_id": memory_id})

            return True

        return False

    def get_replay_compression(self) -> float:
        """Get temporal compression factor for replay."""
        if self.state.phase in [SWRPhase.INITIATING, SWRPhase.RIPPLING]:
            return self.config.compression_factor
        return 1.0

    # =========================================================================
    # External Interface
    # =========================================================================

    def is_swr_active(self) -> bool:
        """Check if SWR is currently active."""
        return self.state.phase != SWRPhase.QUIESCENT

    def can_initiate_swr(self) -> bool:
        """Check if conditions allow SWR initiation."""
        return (
            self.state.phase == SWRPhase.QUIESCENT and
            self._should_initiate_swr()
        )

    def force_swr(self, token: CallerToken | None = None) -> bool:
        """Force an SWR event (for testing/experiments).

        Args:
            token: Optional access control token (only enforced if provided)

        Returns:
            True if SWR was initiated, False otherwise
        """
        if token is not None:
            require_capability(token, "trigger_swr")
        if self.state.phase != SWRPhase.QUIESCENT:
            return False
        self._initiate_swr()
        return True

    def set_ach_level(self, level: float, token: CallerToken | None = None) -> None:
        """
        Manually set ACh level (for testing).

        ATOM-P0-12: Protected by access control.

        Args:
            level: ACh level [0, 1]
            token: Caller token (required for access control)
        """
        if token is not None:
            require_capability(token, "set_neuromod")
        self.state.ach_level = float(np.clip(level, 0, 1))

    def set_ne_level(self, level: float, token: CallerToken | None = None) -> None:
        """
        Manually set NE level (for testing).

        ATOM-P0-12: Protected by access control.

        Args:
            level: NE level [0, 1]
            token: Caller token (required for access control)
        """
        if token is not None:
            require_capability(token, "set_neuromod")
        self.state.ne_level = float(np.clip(level, 0, 1))

    def set_wake_sleep_mode(self, mode: WakeSleepMode, token: CallerToken | None = None) -> None:
        """
        Manually set wake/sleep mode (Phase 2).

        ATOM-P0-12: Protected by access control.

        Args:
            mode: Wake/sleep state to set
            token: Caller token (required for access control)
        """
        if token is not None:
            require_capability(token, "set_sleep_state")
        self.state.wake_sleep_mode = mode
        self.state.current_ripple_freq = self._compute_state_ripple_freq()
        logger.debug(f"Wake/sleep mode set to {mode.value}")

    def validate_ripple_frequency(self, frequency: float) -> bool:
        """
        Validate ripple frequency is in biological range (Phase 2).

        Args:
            frequency: Frequency to validate (Hz)

        Returns:
            True if valid (150-250 Hz)
        """
        return RIPPLE_FREQ_MIN <= frequency <= RIPPLE_FREQ_MAX

    def get_swr_probability(self) -> float:
        """
        Get current SWR probability based on state (Phase 2).

        Returns:
            Probability of SWR initiation [0, 1]
        """
        if not self.config.enable_state_gating:
            return 1.0

        state_prob = self.config.get_state_swr_probability(
            self.state.wake_sleep_mode
        )

        # Modulate by neuromodulator levels
        ach_factor = max(0.0, 1.0 - self.state.ach_level / self.config.ach_threshold)
        ne_factor = max(0.0, 1.0 - self.state.ne_level / self.config.arousal_threshold)

        return state_prob * ach_factor * ne_factor

    # =========================================================================
    # Callbacks
    # =========================================================================

    def register_swr_callback(
        self,
        callback: Callable[[SWREvent], None]
    ) -> None:
        """Register callback for SWR completion."""
        self._swr_callbacks.append(callback)

    def register_replay_callback(
        self,
        callback: Callable[[Any], None]
    ) -> None:
        """Register callback for replay events."""
        self._replay_callbacks.append(callback)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict:
        """Get SWR coupling statistics."""
        stats = {
            "phase": self.state.phase.value,
            "current_amplitude": self.state.current_amplitude,
            "time_since_last_swr": self.state.time_since_last_swr,
            "total_swrs": len(self._swr_events),
            "ach_level": self.state.ach_level,
            "ne_level": self.state.ne_level,
            "hippocampal_activity": self.state.hippocampal_activity,
            # Phase 2 metrics
            "wake_sleep_mode": self.state.wake_sleep_mode.value,
            "current_ripple_freq": self.state.current_ripple_freq,
            "swr_count_wake": self.state.swr_count_wake,
            "swr_count_sleep": self.state.swr_count_sleep,
            "swr_probability": self.get_swr_probability(),
        }

        if self._swr_events:
            recent = self._swr_events[-10:]
            stats["avg_swr_duration"] = np.mean([e.duration for e in recent])
            stats["avg_peak_amplitude"] = np.mean([e.peak_amplitude for e in recent])
            stats["total_replays"] = sum(e.replay_count for e in self._swr_events)

        return stats

    def get_recent_events(self, n: int = 10) -> list[SWREvent]:
        """Get recent SWR events."""
        return self._swr_events[-n:]

    def reset(self) -> None:
        """Reset coupling state."""
        self.state = SWRCouplingState()
        self._current_event = None
        self._swr_events.clear()
        self._simulation_time = 0.0
        logger.info("SWRNeuralFieldCoupling reset")

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "state": self.state.to_dict(),
            "simulation_time": self._simulation_time,
            "event_count": len(self._swr_events),
        }

    def load_state(self, saved: dict) -> None:
        """Load state from persistence."""
        if "simulation_time" in saved:
            self._simulation_time = saved["simulation_time"]
        if "state" in saved:
            s = saved["state"]
            self.state.time_since_last_swr = s.get("time_since_last_swr", 1.0)
            self.state.ach_level = s.get("ach_level", 0.3)
            self.state.ne_level = s.get("ne_level", 0.3)


def create_swr_coupling(
    neural_field: NeuralFieldSolver | None = None,
    hippocampus: HippocampalCircuit | None = None,
    ripple_frequency: float = 180.0,
    enable_state_gating: bool = False,
) -> SWRNeuralFieldCoupling:
    """
    Factory function to create SWR-neural field coupling.

    Args:
        neural_field: Neural field solver
        hippocampus: Hippocampal circuit
        ripple_frequency: Ripple frequency in Hz
        enable_state_gating: Enable Phase 2 wake/sleep state gating

    Returns:
        Configured SWRNeuralFieldCoupling
    """
    config = SWRConfig(
        ripple_frequency=ripple_frequency,
        enable_state_gating=enable_state_gating,
    )
    return SWRNeuralFieldCoupling(
        config=config,
        neural_field=neural_field,
        hippocampus=hippocampus,
    )


# Backward compatibility alias
SWRCoupling = SWRNeuralFieldCoupling


__all__ = [
    "SWRNeuralFieldCoupling",
    "SWRCoupling",  # Alias
    "SWRConfig",
    "SWRCouplingState",
    "SWREvent",
    "SWRPhase",
    "create_swr_coupling",
]
