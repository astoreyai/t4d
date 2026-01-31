"""
Frequency band oscillators for NCA.

Implements biologically-plausible neural oscillations:
1. Theta (4-8 Hz): Hippocampal rhythm, memory encoding, ACh-dependent
2. Gamma (30-100 Hz): Local cortical processing, E/I balance dependent
3. Beta (13-30 Hz): Motor/cognitive control, DA-dependent
4. Phase-Amplitude Coupling (PAC): Theta phase modulates gamma amplitude

Hinton Learning Connection:
- Theta phase determines encoding vs retrieval mode (Forward-Forward phases)
- Gamma nesting provides sequence indexing (4-8 items per theta cycle)
- Learnable PAC strength = meta-learning capacity

References:
- Buzsáki & Draguhn (2004): Neuronal oscillations in cortical networks
- Lisman & Jensen (2013): Theta-gamma neural code
- Canolty & Knight (2010): Cross-frequency coupling
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class OscillationBand(Enum):
    """Neural oscillation frequency bands."""
    DELTA = "delta"      # 0.5-4 Hz (slow-wave sleep)
    THETA = "theta"      # 4-8 Hz (memory, navigation)
    ALPHA = "alpha"      # 8-13 Hz (relaxed wakefulness)
    BETA = "beta"        # 13-30 Hz (motor, cognitive)
    GAMMA_LOW = "gamma_low"    # 30-60 Hz (local processing)
    GAMMA_HIGH = "gamma_high"  # 60-100 Hz (attention binding)


class CognitivePhase(Enum):
    """Theta-phase cognitive modes (Hinton's Forward-Forward connection)."""
    ENCODING = "encoding"      # Theta phase 0-π: positive data, high plasticity
    RETRIEVAL = "retrieval"    # Theta phase π-2π: pattern completion


@dataclass
class OscillatorConfig:
    """Configuration for frequency band oscillators.

    Parameters based on neuroscience literature:
    - Theta: 6 Hz typical in humans (4-8 Hz range)
    - Gamma: 40 Hz typical (30-100 Hz range)
    - PAC modulation index: 0.2-0.5 typical
    """

    # Theta oscillator parameters
    theta_freq_hz: float = 6.0          # Center frequency
    theta_freq_range: tuple[float, float] = (4.0, 8.0)
    theta_amplitude: float = 0.3         # Baseline amplitude
    theta_ach_sensitivity: float = 0.5   # ACh increases theta power

    # Gamma oscillator parameters
    gamma_freq_hz: float = 40.0         # Center frequency
    gamma_freq_range: tuple[float, float] = (30.0, 80.0)
    gamma_amplitude: float = 0.2         # Baseline amplitude
    gamma_ei_sensitivity: float = 0.4    # E/I balance affects gamma

    # Alpha oscillator parameters (8-13 Hz)
    alpha_freq_hz: float = 10.0         # Center frequency
    alpha_freq_range: tuple[float, float] = (8.0, 13.0)
    alpha_amplitude: float = 0.25        # Baseline amplitude
    alpha_ne_sensitivity: float = -0.4   # NE SUPPRESSES alpha (negative)

    # Beta oscillator parameters
    beta_freq_hz: float = 20.0          # Center frequency
    beta_amplitude: float = 0.15
    beta_da_sensitivity: float = 0.3     # DA modulates beta

    # Phase-amplitude coupling
    pac_strength: float = 0.4           # Modulation index (learnable)
    pac_preferred_phase: float = 0.0    # Theta phase for max gamma (radians)
    pac_learning_rate: float = 0.01     # PAC adaptation rate
    pac_n_bins: int = 18                # Number of phase bins for MI calculation

    # Integration
    dt_ms: float = 1.0                  # Timestep in ms

    # Cognitive phase thresholds
    encoding_phase_start: float = 0.0   # 0 radians
    encoding_phase_end: float = np.pi   # π radians

    def __post_init__(self):
        """Validate configuration."""
        assert self.theta_freq_range[0] <= self.theta_freq_hz <= self.theta_freq_range[1]
        assert self.gamma_freq_range[0] <= self.gamma_freq_hz <= self.gamma_freq_range[1]
        assert 0 <= self.pac_strength <= 1.0


class SleepState(Enum):
    """Sleep-wake state for delta oscillator."""
    AWAKE = "awake"
    LIGHT_SLEEP = "light_sleep"  # NREM stage 1-2
    DEEP_SLEEP = "deep_sleep"    # NREM stage 3-4 (slow-wave)
    REM = "rem"


@dataclass
class OscillatorState:
    """Current state of oscillator system."""

    # Phase accumulators (radians, 0 to 2π)
    theta_phase: float = 0.0
    alpha_phase: float = 0.0
    gamma_phase: float = 0.0
    beta_phase: float = 0.0
    delta_phase: float = 0.0

    # Current amplitudes
    theta_amplitude: float = 0.3
    alpha_amplitude: float = 0.25
    gamma_amplitude: float = 0.2
    beta_amplitude: float = 0.15
    delta_amplitude: float = 0.0  # Only active during sleep

    # Instantaneous frequencies (can vary)
    theta_freq: float = 6.0
    alpha_freq: float = 10.0
    gamma_freq: float = 40.0
    beta_freq: float = 20.0
    delta_freq: float = 1.5

    # Cognitive mode
    cognitive_phase: CognitivePhase = CognitivePhase.ENCODING

    # Sleep state
    sleep_state: SleepState = SleepState.AWAKE
    in_up_state: bool = False  # Delta up-state for consolidation

    # PAC metrics
    current_pac: float = 0.0  # Instantaneous modulation

    # Statistics
    theta_power: float = 0.0
    gamma_power: float = 0.0
    delta_power: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize state."""
        return {
            "theta_phase": self.theta_phase,
            "alpha_phase": self.alpha_phase,
            "gamma_phase": self.gamma_phase,
            "beta_phase": self.beta_phase,
            "delta_phase": self.delta_phase,
            "theta_amplitude": self.theta_amplitude,
            "alpha_amplitude": self.alpha_amplitude,
            "gamma_amplitude": self.gamma_amplitude,
            "delta_amplitude": self.delta_amplitude,
            "cognitive_phase": self.cognitive_phase.value,
            "sleep_state": self.sleep_state.value,
            "in_up_state": self.in_up_state,
            "current_pac": self.current_pac,
        }


class ThetaOscillator:
    """
    Theta band oscillator (4-8 Hz).

    Biological basis:
    - Generated by medial septum cholinergic neurons
    - Modulated by ACh levels (higher ACh = stronger theta)
    - Phase codes temporal/spatial information
    - Critical for episodic memory encoding
    """

    def __init__(self, config: OscillatorConfig):
        self.config = config
        self.phase = 0.0
        self.freq = config.theta_freq_hz
        self.amplitude = config.theta_amplitude

    def step(self, ach_level: float, dt_ms: float) -> tuple[float, float]:
        """
        Advance theta oscillator by one timestep.

        Args:
            ach_level: Acetylcholine level (0-1), modulates theta power
            dt_ms: Timestep in milliseconds

        Returns:
            (theta_output, phase): Current oscillation value and phase
        """
        # ACh modulates theta amplitude (higher ACh = stronger theta)
        ach_mod = 1.0 + self.config.theta_ach_sensitivity * (ach_level - 0.5)
        self.amplitude = self.config.theta_amplitude * ach_mod

        # ACh also slightly increases theta frequency
        freq_mod = 1.0 + 0.1 * (ach_level - 0.5)
        self.freq = self.config.theta_freq_hz * freq_mod
        self.freq = np.clip(self.freq, *self.config.theta_freq_range)

        # Phase advance: Δφ = 2π * f * Δt
        dt_sec = dt_ms / 1000.0
        phase_increment = 2 * np.pi * self.freq * dt_sec
        self.phase = (self.phase + phase_increment) % (2 * np.pi)

        # Output: amplitude * sin(phase)
        output = self.amplitude * np.sin(self.phase)

        return output, self.phase

    def get_cognitive_phase(self) -> CognitivePhase:
        """Determine cognitive mode from theta phase."""
        if 0 <= self.phase < np.pi:
            return CognitivePhase.ENCODING
        else:
            return CognitivePhase.RETRIEVAL


class GammaOscillator:
    """
    Gamma band oscillator (30-100 Hz).

    Biological basis:
    - Generated by fast-spiking GABAergic interneurons (PING model)
    - Frequency set by E/I balance (more inhibition = faster)
    - Amplitude modulated by theta phase (PAC)
    - Critical for local information processing and attention
    """

    def __init__(self, config: OscillatorConfig):
        self.config = config
        self.phase = 0.0
        self.freq = config.gamma_freq_hz
        self.amplitude = config.gamma_amplitude

    def step(
        self,
        glu_level: float,
        gaba_level: float,
        theta_phase: float,
        pac_strength: float,
        dt_ms: float
    ) -> tuple[float, float]:
        """
        Advance gamma oscillator by one timestep.

        Args:
            glu_level: Glutamate level (0-1), excitatory drive
            gaba_level: GABA level (0-1), inhibitory drive
            theta_phase: Current theta phase for PAC modulation
            pac_strength: Phase-amplitude coupling strength
            dt_ms: Timestep in milliseconds

        Returns:
            (gamma_output, phase): Current oscillation value and phase
        """
        # E/I balance affects gamma frequency
        # More inhibition (higher GABA) = faster gamma (tighter interneuron loop)
        ei_balance = glu_level - gaba_level  # Positive = E-dominant
        freq_mod = 1.0 - self.config.gamma_ei_sensitivity * ei_balance
        self.freq = self.config.gamma_freq_hz * freq_mod
        self.freq = np.clip(self.freq, *self.config.gamma_freq_range)

        # Base amplitude from activity level
        activity = 0.5 * (glu_level + gaba_level)
        base_amp = self.config.gamma_amplitude * (0.5 + activity)

        # Phase-amplitude coupling: theta phase modulates gamma amplitude
        # Max gamma at preferred theta phase (typically theta peak)
        pac_modulation = 1.0 + pac_strength * np.cos(
            theta_phase - self.config.pac_preferred_phase
        )
        self.amplitude = base_amp * pac_modulation

        # Phase advance
        dt_sec = dt_ms / 1000.0
        phase_increment = 2 * np.pi * self.freq * dt_sec
        self.phase = (self.phase + phase_increment) % (2 * np.pi)

        # Output
        output = self.amplitude * np.sin(self.phase)

        return output, self.phase


class DeltaOscillator:
    """
    Delta band oscillator (0.5-4 Hz) for slow-wave sleep.

    Biological basis:
    - Generated during NREM stage 3-4 (slow-wave sleep)
    - Cortical origin with thalamocortical involvement
    - Modulated by adenosine levels (high adenosine = stronger delta)
    - Up-states (~200-500ms) trigger memory consolidation
    - Down-states (~200-500ms) allow synaptic homeostasis

    Hinton connection:
    - Delta up-states = "positive phase" for consolidation
    - Down-states = synaptic downscaling (regularization)
    - Coordinates with SWR for memory replay

    References:
    - Steriade et al. (1993): Slow oscillation in neocortex
    - Tononi & Cirelli (2006): Synaptic homeostasis hypothesis
    - Marshall et al. (2006): Delta boosting enhances memory
    """

    def __init__(self, config: OscillatorConfig):
        self.config = config
        self.phase = 0.0
        self.freq = 1.5  # Center of delta range
        self.amplitude = 0.4  # Strong during sleep

        # Delta-specific parameters
        self.freq_range = (0.5, 4.0)
        self.adenosine_sensitivity = 0.6  # Adenosine increases delta
        self.up_state_threshold = 0.3  # Phase threshold for up-state

        # State tracking
        self._in_up_state = False
        self._up_state_duration_ms = 0.0
        self._down_state_duration_ms = 0.0

    def step(
        self,
        adenosine_level: float,
        sleep_depth: float,
        dt_ms: float
    ) -> tuple[float, float]:
        """
        Advance delta oscillator by one timestep.

        Args:
            adenosine_level: Adenosine concentration (0-1), increases with wakefulness
            sleep_depth: Sleep depth (0=awake, 1=deep NREM)
            dt_ms: Timestep in milliseconds

        Returns:
            (delta_output, phase): Current oscillation value and phase
        """
        # Delta only active during sleep
        if sleep_depth < 0.3:
            self.amplitude = 0.05  # Minimal delta when awake
            self._in_up_state = False
            return 0.0, self.phase

        # Adenosine pressure increases delta power
        # High adenosine (accumulated during wake) = stronger delta
        adenosine_mod = 1.0 + self.adenosine_sensitivity * (adenosine_level - 0.3)
        adenosine_mod = max(0.5, adenosine_mod)

        # Sleep depth directly affects amplitude
        sleep_mod = 0.3 + 0.7 * sleep_depth

        self.amplitude = 0.4 * adenosine_mod * sleep_mod

        # Frequency slightly varies with depth (deeper = slower)
        self.freq = 2.5 - 1.5 * sleep_depth  # 2.5 Hz light → 1.0 Hz deep
        self.freq = np.clip(self.freq, *self.freq_range)

        # Phase advance
        dt_sec = dt_ms / 1000.0
        phase_increment = 2 * np.pi * self.freq * dt_sec
        self.phase = (self.phase + phase_increment) % (2 * np.pi)

        # Determine up-state vs down-state
        output = self.amplitude * np.sin(self.phase)
        prev_up_state = self._in_up_state
        self._in_up_state = output > self.up_state_threshold * self.amplitude

        # Track state durations
        if self._in_up_state:
            self._up_state_duration_ms += dt_ms
            if not prev_up_state:
                self._down_state_duration_ms = 0.0
        else:
            self._down_state_duration_ms += dt_ms
            if prev_up_state:
                self._up_state_duration_ms = 0.0

        return output, self.phase

    def is_up_state(self) -> bool:
        """Check if currently in up-state (consolidation window)."""
        return self._in_up_state

    def get_consolidation_gate(self) -> float:
        """
        Get consolidation gating signal.

        Returns 1.0 during up-state, 0.0 during down-state.
        Used to gate hippocampal-cortical transfer.
        """
        return 1.0 if self._in_up_state else 0.0

    def get_downscaling_signal(self) -> float:
        """
        Get synaptic downscaling signal.

        Active during down-states for synaptic homeostasis.
        """
        if self._in_up_state:
            return 0.0
        # Stronger downscaling in sustained down-states
        sustained = min(1.0, self._down_state_duration_ms / 300.0)
        return sustained * 0.1  # 10% max downscaling rate


class BetaOscillator:
    """
    Beta band oscillator (13-30 Hz).

    Biological basis:
    - Associated with motor control and cognitive processing
    - Modulated by dopamine (higher DA = stronger beta)
    - Suppressed during movement, elevated during holding
    - Important for top-down control
    """

    def __init__(self, config: OscillatorConfig):
        self.config = config
        self.phase = 0.0
        self.freq = config.beta_freq_hz
        self.amplitude = config.beta_amplitude

    def step(self, da_level: float, dt_ms: float) -> tuple[float, float]:
        """
        Advance beta oscillator by one timestep.

        Args:
            da_level: Dopamine level (0-1), modulates beta power
            dt_ms: Timestep in milliseconds

        Returns:
            (beta_output, phase): Current oscillation value and phase
        """
        # DA modulates beta amplitude
        da_mod = 1.0 + self.config.beta_da_sensitivity * (da_level - 0.5)
        self.amplitude = self.config.beta_amplitude * da_mod

        # Phase advance
        dt_sec = dt_ms / 1000.0
        phase_increment = 2 * np.pi * self.freq * dt_sec
        self.phase = (self.phase + phase_increment) % (2 * np.pi)

        # Output
        output = self.amplitude * np.sin(self.phase)

        return output, self.phase


class AlphaOscillator:
    """
    Alpha band oscillator (8-13 Hz).

    Biological basis:
    - Generated by thalamo-cortical circuits
    - Dominant during relaxed wakefulness, eyes closed
    - SUPPRESSED by norepinephrine (arousal reduces alpha)
    - Represents "idling" or inhibitory state
    - Increases in cortical areas NOT being attended

    References:
    - Klimesch (2012): Alpha oscillations and attention
    - Jensen & Mazaheri (2010): Alpha inhibition hypothesis
    - Sara (2009): LC-NE modulation of cortical alpha
    """

    def __init__(self, config: OscillatorConfig):
        self.config = config
        self.phase = 0.0
        self.freq = config.alpha_freq_hz
        self.amplitude = config.alpha_amplitude

    def step(
        self,
        ne_level: float,
        attention_level: float,
        dt_ms: float
    ) -> tuple[float, float]:
        """
        Advance alpha oscillator by one timestep.

        Args:
            ne_level: Norepinephrine level (0-1), SUPPRESSES alpha
            attention_level: Attention/arousal (0-1), suppresses alpha
            dt_ms: Timestep in milliseconds

        Returns:
            (alpha_output, phase): Current oscillation value and phase
        """
        # NE suppresses alpha (negative sensitivity in config)
        # High NE = low alpha (alert/aroused state)
        # Low NE = high alpha (relaxed/idling state)
        ne_mod = 1.0 + self.config.alpha_ne_sensitivity * (ne_level - 0.3)
        ne_mod = max(0.1, ne_mod)  # Don't go below 10%

        # Attention also suppresses alpha (inverse relationship)
        # High attention = processing = low alpha in attended area
        attention_mod = 1.0 - 0.4 * attention_level
        attention_mod = max(0.2, attention_mod)

        self.amplitude = self.config.alpha_amplitude * ne_mod * attention_mod

        # Slight frequency variation with arousal
        # Lower arousal -> slightly slower alpha (drowsy)
        arousal = 0.5 * (ne_level + attention_level)
        freq_mod = 0.9 + 0.2 * arousal  # 0.9-1.1x
        self.freq = self.config.alpha_freq_hz * freq_mod
        self.freq = np.clip(self.freq, *self.config.alpha_freq_range)

        # Phase advance
        dt_sec = dt_ms / 1000.0
        phase_increment = 2 * np.pi * self.freq * dt_sec
        self.phase = (self.phase + phase_increment) % (2 * np.pi)

        # Output
        output = self.amplitude * np.sin(self.phase)

        return output, self.phase

    def get_inhibition_level(self) -> float:
        """
        Get cortical inhibition level from alpha amplitude.

        Higher alpha = more inhibition (idling/suppression).
        """
        # Normalize to [0, 1]
        return min(1.0, self.amplitude / self.config.alpha_amplitude)


class PhaseAmplitudeCoupling:
    """
    Theta-gamma phase-amplitude coupling (PAC).

    Implements the theta-gamma neural code:
    - Theta phase determines WHEN gamma bursts occur
    - Gamma amplitude determines WHAT information is processed
    - ~4-8 gamma cycles nest within each theta cycle

    Hinton connection:
    - PAC strength is LEARNABLE (meta-learning)
    - Higher PAC = better sequence memory
    - Lower PAC = more flexibility, less structure
    """

    def __init__(self, config: OscillatorConfig):
        self.config = config
        self.strength = config.pac_strength
        self.preferred_phase = config.pac_preferred_phase

        # History for computing modulation index
        self._theta_phases: deque = deque(maxlen=1000)
        self._gamma_amplitudes: deque = deque(maxlen=1000)

        # Learnable parameters
        self._strength_history: deque = deque(maxlen=100)

    def compute_modulation(
        self,
        theta_phase: float,
        base_gamma_amplitude: float
    ) -> float:
        """
        Compute gamma amplitude modulation from theta phase.

        Args:
            theta_phase: Current theta phase (radians)
            base_gamma_amplitude: Unmodulated gamma amplitude

        Returns:
            Modulated gamma amplitude
        """
        # Cosine modulation centered on preferred phase
        modulation = 1.0 + self.strength * np.cos(
            theta_phase - self.preferred_phase
        )

        modulated_amplitude = base_gamma_amplitude * modulation

        # Store for MI calculation
        self._theta_phases.append(theta_phase)
        self._gamma_amplitudes.append(modulated_amplitude)

        return modulated_amplitude

    def compute_modulation_index(self) -> float:
        """
        Compute modulation index (MI) from recent history.

        MI measures strength of phase-amplitude coupling:
        - MI = 0: No coupling
        - MI > 0.3: Strong coupling (typical in hippocampus)

        Uses Kullback-Leibler divergence method (Tort et al., 2010).
        """
        if len(self._theta_phases) < 100:
            return 0.0

        phases = np.array(self._theta_phases)
        amplitudes = np.array(self._gamma_amplitudes)

        # Bin phases into N bins (configurable)
        n_bins = self.config.pac_n_bins
        phase_bins = np.linspace(0, 2*np.pi, n_bins + 1)

        # Mean amplitude per phase bin
        mean_amps = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (phases >= phase_bins[i]) & (phases < phase_bins[i+1])
            if np.sum(mask) > 0:
                mean_amps[i] = np.mean(amplitudes[mask])
            else:
                mean_amps[i] = np.mean(amplitudes)  # Default

        # Normalize to probability distribution
        if np.sum(mean_amps) > 0:
            p = mean_amps / np.sum(mean_amps)
        else:
            return 0.0

        # Uniform distribution
        q = np.ones(n_bins) / n_bins

        # KL divergence (with small epsilon to avoid log(0))
        eps = 1e-10
        kl = np.sum(p * np.log((p + eps) / (q + eps)))

        # Normalize to [0, 1]
        mi = kl / np.log(n_bins)

        return float(np.clip(mi, 0, 1))

    def update_strength(self, reward_signal: float):
        """
        Update PAC strength based on task performance (meta-learning).

        Args:
            reward_signal: Positive = increase PAC, negative = decrease
        """
        delta = self.config.pac_learning_rate * reward_signal
        self.strength = np.clip(self.strength + delta, 0.1, 0.9)
        self._strength_history.append(self.strength)

    def get_gamma_slots(self) -> int:
        """
        Get number of gamma cycles per theta cycle.

        This determines working memory capacity:
        - ~4-8 gamma cycles = 4-8 items in working memory (Miller's 7±2)
        """
        gamma_freq = 40.0  # Typical gamma
        theta_freq = 6.0   # Typical theta
        return int(gamma_freq / theta_freq)

    # ATOM-P4-13: Cross-frequency coupling beyond PAC (stub methods)
    def compute_ppc(self):
        """Phase-phase coupling (planned)."""
        raise NotImplementedError("Phase-phase coupling not yet implemented")

    def compute_aac(self):
        """Amplitude-amplitude coupling (planned)."""
        raise NotImplementedError("Amplitude-amplitude coupling not yet implemented")


class FrequencyBandGenerator:
    """
    Unified frequency band oscillation generator.

    Orchestrates theta, gamma, and beta oscillators with
    proper phase relationships and neuromodulator coupling.

    Example:
        >>> config = OscillatorConfig()
        >>> gen = FrequencyBandGenerator(config)
        >>> outputs = gen.step(ach=0.6, da=0.5, glu=0.5, gaba=0.4)
        >>> print(outputs['theta'], outputs['cognitive_phase'])
    """

    def __init__(self, config: OscillatorConfig | None = None):
        self.config = config or OscillatorConfig()

        # Initialize oscillators
        self.theta = ThetaOscillator(self.config)
        self.alpha = AlphaOscillator(self.config)
        self.gamma = GammaOscillator(self.config)
        self.beta = BetaOscillator(self.config)
        self.delta = DeltaOscillator(self.config)  # Sleep oscillator
        self.pac = PhaseAmplitudeCoupling(self.config)

        # State
        self.state = OscillatorState()
        self._step_count = 0

        # Sleep-wake state tracking
        self._sleep_depth = 0.0  # 0=awake, 1=deep sleep
        self._adenosine_level = 0.3  # Accumulates during wake

        # History for spectral analysis
        self._theta_history: deque = deque(maxlen=2000)
        self._alpha_history: deque = deque(maxlen=2000)
        self._gamma_history: deque = deque(maxlen=2000)
        self._beta_history: deque = deque(maxlen=2000)
        self._delta_history: deque = deque(maxlen=2000)

    def step(
        self,
        ach_level: float = 0.5,
        da_level: float = 0.5,
        ne_level: float = 0.3,
        glu_level: float = 0.5,
        gaba_level: float = 0.5,
        attention_level: float = 0.5,
        sleep_depth: float = 0.0,
        adenosine_level: float | None = None,
        dt_ms: float | None = None
    ) -> dict[str, float]:
        """
        Advance all oscillators by one timestep.

        Args:
            ach_level: Acetylcholine (modulates theta)
            da_level: Dopamine (modulates beta)
            ne_level: Norepinephrine (suppresses alpha)
            glu_level: Glutamate (affects gamma frequency)
            gaba_level: GABA (affects gamma frequency)
            attention_level: Attention/arousal level (suppresses alpha)
            sleep_depth: Sleep depth (0=awake, 1=deep NREM)
            adenosine_level: Adenosine level (auto-tracked if None)
            dt_ms: Timestep in ms (uses config default if None)

        Returns:
            Dict with oscillator outputs and phase information
        """
        dt = dt_ms or self.config.dt_ms

        # Track sleep-wake state
        self._sleep_depth = sleep_depth
        if adenosine_level is not None:
            self._adenosine_level = adenosine_level

        # Determine sleep state
        if sleep_depth < 0.3:
            self.state.sleep_state = SleepState.AWAKE
        elif sleep_depth < 0.6:
            self.state.sleep_state = SleepState.LIGHT_SLEEP
        else:
            self.state.sleep_state = SleepState.DEEP_SLEEP

        # Step theta (provides phase for PAC)
        theta_out, theta_phase = self.theta.step(ach_level, dt)

        # Step delta (only active during sleep)
        delta_out, delta_phase = self.delta.step(
            self._adenosine_level, sleep_depth, dt
        )

        # Step alpha (suppressed by NE and attention)
        alpha_out, alpha_phase = self.alpha.step(ne_level, attention_level, dt)

        # Step gamma (modulated by theta phase via PAC)
        gamma_out, gamma_phase = self.gamma.step(
            glu_level, gaba_level,
            theta_phase, self.pac.strength,
            dt
        )

        # Step beta
        beta_out, beta_phase = self.beta.step(da_level, dt)

        # Update PAC modulation tracking
        self.pac.compute_modulation(theta_phase, self.gamma.amplitude)

        # Update state
        self.state.theta_phase = theta_phase
        self.state.alpha_phase = alpha_phase
        self.state.gamma_phase = gamma_phase
        self.state.beta_phase = beta_phase
        self.state.delta_phase = delta_phase
        self.state.theta_amplitude = self.theta.amplitude
        self.state.alpha_amplitude = self.alpha.amplitude
        self.state.gamma_amplitude = self.gamma.amplitude
        self.state.beta_amplitude = self.beta.amplitude
        self.state.delta_amplitude = self.delta.amplitude
        self.state.theta_freq = self.theta.freq
        self.state.alpha_freq = self.alpha.freq
        self.state.gamma_freq = self.gamma.freq
        self.state.delta_freq = self.delta.freq
        self.state.cognitive_phase = self.theta.get_cognitive_phase()
        self.state.in_up_state = self.delta.is_up_state()

        # Track history
        self._theta_history.append(theta_out)
        self._alpha_history.append(alpha_out)
        self._gamma_history.append(gamma_out)
        self._beta_history.append(beta_out)
        self._delta_history.append(delta_out)

        self._step_count += 1

        return {
            "theta": theta_out,
            "alpha": alpha_out,
            "gamma": gamma_out,
            "beta": beta_out,
            "delta": delta_out,
            "theta_phase": theta_phase,
            "alpha_phase": alpha_phase,
            "gamma_phase": gamma_phase,
            "beta_phase": beta_phase,
            "delta_phase": delta_phase,
            "cognitive_phase": self.state.cognitive_phase.value,
            "sleep_state": self.state.sleep_state.value,
            "in_up_state": self.state.in_up_state,
            "consolidation_gate": self.delta.get_consolidation_gate(),
            "downscaling_signal": self.delta.get_downscaling_signal(),
            "pac_strength": self.pac.strength,
            "alpha_inhibition": self.alpha.get_inhibition_level(),
        }

    def compute_oscillation_field(
        self,
        ach_field: np.ndarray,
        da_field: np.ndarray,
        ne_field: np.ndarray,
        glu_field: np.ndarray,
        gaba_field: np.ndarray,
        attention_field: np.ndarray | None = None,
        dt_ms: float | None = None
    ) -> dict[str, np.ndarray]:
        """
        Compute oscillations across spatial field.

        Args:
            ach_field: Spatial ACh concentration
            da_field: Spatial DA concentration
            ne_field: Spatial NE concentration
            glu_field: Spatial Glu concentration
            gaba_field: Spatial GABA concentration
            attention_field: Spatial attention map (optional)
            dt_ms: Timestep

        Returns:
            Dict with spatial oscillation fields
        """
        dt = dt_ms or self.config.dt_ms

        if attention_field is None:
            attention_field = np.ones_like(ach_field) * 0.5

        # Use mean levels for phase tracking (global phase)
        mean_ach = float(np.mean(ach_field))
        mean_da = float(np.mean(da_field))
        mean_ne = float(np.mean(ne_field))
        mean_glu = float(np.mean(glu_field))
        mean_gaba = float(np.mean(gaba_field))
        mean_attention = float(np.mean(attention_field))

        # Step global oscillators
        outputs = self.step(mean_ach, mean_da, mean_ne, mean_glu, mean_gaba, mean_attention, dt)

        # Create spatial modulation based on local NT levels
        # Theta amplitude varies with local ACh
        ach_mod = 1.0 + self.config.theta_ach_sensitivity * (ach_field - 0.5)
        theta_field = outputs["theta"] * ach_mod

        # Alpha amplitude varies inversely with local NE and attention
        # Low NE / low attention = high alpha (idling)
        ne_mod = 1.0 + self.config.alpha_ne_sensitivity * (ne_field - 0.3)
        ne_mod = np.maximum(0.1, ne_mod)
        attention_mod = 1.0 - 0.4 * attention_field
        attention_mod = np.maximum(0.2, attention_mod)
        alpha_field = outputs["alpha"] * ne_mod * attention_mod

        # Gamma amplitude varies with local activity
        activity = 0.5 * (glu_field + gaba_field)
        gamma_field = outputs["gamma"] * (0.5 + activity)

        # Beta amplitude varies with local DA
        da_mod = 1.0 + self.config.beta_da_sensitivity * (da_field - 0.5)
        beta_field = outputs["beta"] * da_mod

        return {
            "theta": theta_field.astype(np.float32),
            "alpha": alpha_field.astype(np.float32),
            "gamma": gamma_field.astype(np.float32),
            "beta": beta_field.astype(np.float32),
            "theta_phase": outputs["theta_phase"],
            "alpha_phase": outputs["alpha_phase"],
            "gamma_phase": outputs["gamma_phase"],
            "cognitive_phase": outputs["cognitive_phase"],
        }

    def get_encoding_signal(self) -> float:
        """
        Get encoding strength based on theta phase.

        Returns value in [0, 1]:
        - 1.0 at peak encoding phase (theta peak)
        - 0.0 at retrieval phase (theta trough)

        Used to modulate plasticity in Forward-Forward style.
        """
        phase = self.state.theta_phase
        # Cosine with max at phase 0 (encoding peak)
        encoding = 0.5 * (1.0 + np.cos(phase))
        return float(encoding)

    def get_retrieval_signal(self) -> float:
        """
        Get retrieval strength based on theta phase.

        Complementary to encoding signal.
        """
        return 1.0 - self.get_encoding_signal()

    def get_working_memory_capacity(self) -> int:
        """
        Estimate working memory capacity from gamma/theta ratio.

        Based on Lisman & Jensen (2013): WM items = gamma_cycles / theta_cycle
        """
        return self.pac.get_gamma_slots()

    def compute_spectral_power(self) -> dict[str, float]:
        """
        Compute power in each frequency band from recent history.
        """
        if len(self._theta_history) < 100:
            return {"theta": 0.0, "alpha": 0.0, "gamma": 0.0, "beta": 0.0, "delta": 0.0}

        theta_arr = np.array(self._theta_history)
        alpha_arr = np.array(self._alpha_history)
        gamma_arr = np.array(self._gamma_history)
        beta_arr = np.array(self._beta_history)
        delta_arr = np.array(self._delta_history)

        return {
            "theta": float(np.var(theta_arr)),
            "alpha": float(np.var(alpha_arr)),
            "gamma": float(np.var(gamma_arr)),
            "beta": float(np.var(beta_arr)),
            "delta": float(np.var(delta_arr)),
        }

    def get_modulation_index(self) -> float:
        """Get theta-gamma PAC modulation index."""
        return self.pac.compute_modulation_index()

    def update_pac_from_reward(self, reward: float):
        """
        Update PAC strength based on task performance.

        This implements meta-learning: the system learns
        how strongly to couple theta and gamma based on
        whether tight coupling improves performance.
        """
        self.pac.update_strength(reward)

    def reset(self):
        """Reset all oscillators to initial state."""
        self.theta.phase = 0.0
        self.alpha.phase = 0.0
        self.gamma.phase = 0.0
        self.beta.phase = 0.0
        self.delta.phase = 0.0
        self.delta._in_up_state = False
        self._sleep_depth = 0.0
        self._adenosine_level = 0.3
        self.state = OscillatorState()
        self._step_count = 0
        self._theta_history.clear()
        self._alpha_history.clear()
        self._gamma_history.clear()
        self._beta_history.clear()
        self._delta_history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive oscillator statistics."""
        power = self.compute_spectral_power()
        return {
            "step_count": self._step_count,
            "theta_freq": self.state.theta_freq,
            "alpha_freq": self.state.alpha_freq,
            "gamma_freq": self.state.gamma_freq,
            "delta_freq": self.state.delta_freq,
            "theta_amplitude": self.state.theta_amplitude,
            "alpha_amplitude": self.state.alpha_amplitude,
            "gamma_amplitude": self.state.gamma_amplitude,
            "delta_amplitude": self.state.delta_amplitude,
            "theta_power": power["theta"],
            "alpha_power": power["alpha"],
            "gamma_power": power["gamma"],
            "beta_power": power["beta"],
            "delta_power": power["delta"],
            "pac_strength": self.pac.strength,
            "modulation_index": self.get_modulation_index(),
            "cognitive_phase": self.state.cognitive_phase.value,
            "sleep_state": self.state.sleep_state.value,
            "in_up_state": self.state.in_up_state,
            "wm_capacity": self.get_working_memory_capacity(),
            "alpha_inhibition": self.alpha.get_inhibition_level(),
        }

    def validate_oscillations(self) -> dict[str, Any]:
        """
        Validate oscillation properties against biological criteria.
        """
        stats = self.get_stats()

        results = {
            # Theta should be in 4-8 Hz range
            "theta_freq_valid": 4.0 <= stats["theta_freq"] <= 8.0,

            # Alpha should be in 8-13 Hz range
            "alpha_freq_valid": 8.0 <= stats["alpha_freq"] <= 13.0,

            # Gamma should be in 30-80 Hz range
            "gamma_freq_valid": 30.0 <= stats["gamma_freq"] <= 80.0,

            # Delta should be in 0.5-4 Hz range
            "delta_freq_valid": 0.5 <= stats["delta_freq"] <= 4.0,

            # PAC should show some coupling
            "pac_present": stats["modulation_index"] > 0.1 if self._step_count > 500 else True,

            # Working memory capacity should be reasonable
            "wm_capacity_valid": 4 <= stats["wm_capacity"] <= 10,

            # Should have non-zero power after running
            "oscillations_active": stats["theta_power"] > 0 if self._step_count > 100 else True,

            # Alpha inhibition should be bounded
            "alpha_inhibition_valid": 0.0 <= stats["alpha_inhibition"] <= 1.0,

            # Sleep state should be valid
            "sleep_state_valid": stats["sleep_state"] in ["awake", "light_sleep", "deep_sleep", "rem"],
        }

        results["all_pass"] = all(results.values())
        return results


# Backward compatibility alias
OscillatorBank = FrequencyBandGenerator
