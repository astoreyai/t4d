"""
Locus Coeruleus Norepinephrine Circuit for World Weaver.

Sprint 4 (P2): Implements LC phasic/tonic firing modes for arousal modulation.
Phase 2: Adds surprise-driven NE with uncertainty signaling.

Biological Basis:
- LC is the primary source of brain norepinephrine (NE)
- Tonic mode: Low sustained firing (0.5-5 Hz) for baseline arousal
- Phasic mode: Brief bursts (10-20 Hz) for salient/novel stimuli
- NE modulates signal-to-noise ratio across cortex
- Yerkes-Dodson: Optimal performance at intermediate arousal
- Phase 2: LC phasic encodes surprise/uncertainty (unexpected outcomes)

Key Features:
1. Explicit tonic/phasic firing modes
2. Arousal-performance (inverted U) relationship
3. NE release and reuptake dynamics
4. Alpha-2 autoreceptor negative feedback
5. Stress/CRH input integration
6. Phase 2: Surprise-driven phasic bursts (prediction error signaling)
7. Phase 2: Uncertainty tracking for meta-cognitive control

References:
- Aston-Jones & Cohen (2005): Adaptive gain theory
- Sara (2009): LC-NE system and cognitive function
- Berridge & Waterhouse (2003): LC-NE modulation of cortex
- Dayan & Yu (2006): Uncertainty, neuromodulation and attention
- Payzan-LeNestour et al. (2013): LC encodes unexpected uncertainty
- Nassar et al. (2012): Rational regulation of learning by surprise
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class LCFiringMode(Enum):
    """Locus coeruleus firing modes."""
    QUIESCENT = "quiescent"      # Very low activity (sleep)
    TONIC_LOW = "tonic_low"      # Low tonic (drowsy/relaxed)
    TONIC_OPTIMAL = "tonic_optimal"  # Optimal tonic (focused)
    TONIC_HIGH = "tonic_high"    # High tonic (stressed/anxious)
    PHASIC = "phasic"            # Phasic burst (salient event)


@dataclass
class LCConfig:
    """Configuration for locus coeruleus dynamics."""

    # Firing rate parameters (Hz)
    quiescent_rate: float = 0.1      # Sleep state
    tonic_low_rate: float = 1.0      # Drowsy
    tonic_optimal_rate: float = 3.0  # Alert/focused
    tonic_high_rate: float = 5.0     # Stressed
    phasic_peak_rate: float = 15.0   # Burst peak

    # Mode thresholds (arousal level)
    quiescent_threshold: float = 0.1
    tonic_low_threshold: float = 0.3
    tonic_optimal_threshold: float = 0.5
    tonic_high_threshold: float = 0.7

    # NE dynamics
    ne_release_per_spike: float = 0.03   # NE release per spike
    ne_reuptake_rate: float = 0.15       # NET reuptake rate
    ne_baseline: float = 0.3             # Baseline NE level

    # Phasic burst parameters
    phasic_duration: float = 0.3         # Burst duration (s)
    phasic_threshold: float = 0.6        # Salience threshold for phasic
    phasic_refractory: float = 0.5       # Refractory period (s)

    # Alpha-2 autoreceptor (negative feedback)
    autoreceptor_sensitivity: float = 0.4
    autoreceptor_ec50: float = 0.5
    autoreceptor_hill: float = 1.5

    # Yerkes-Dodson parameters
    optimal_arousal: float = 0.6         # Arousal for peak performance
    arousal_sensitivity: float = 2.0     # Width of inverted-U

    # Time constants
    tau_firing: float = 0.1              # Firing rate smoothing (s)
    tau_ne: float = 0.3                  # NE concentration (s)

    # Stress/CRH input
    crh_gain: float = 0.5                # CRH effect on firing


@dataclass
class LCState:
    """Current state of locus coeruleus."""

    mode: LCFiringMode = LCFiringMode.TONIC_OPTIMAL
    firing_rate: float = 3.0            # Current firing rate (Hz)
    ne_level: float = 0.3               # Extracellular NE [0, 1]

    # Autoreceptor state
    autoreceptor_inhibition: float = 0.0

    # Phasic state
    in_phasic: bool = False
    phasic_time_remaining: float = 0.0
    time_since_phasic: float = 1.0

    # Input tracking
    arousal_drive: float = 0.5          # Internal arousal signal
    salience_input: float = 0.0         # External salience
    stress_input: float = 0.0           # CRH/stress input

    # Performance modulation
    gain_modulation: float = 1.0        # Yerkes-Dodson gain
    pfc_gain: float = 1.0               # PFC-modulated phasic responsiveness

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "firing_rate": self.firing_rate,
            "ne_level": self.ne_level,
            "autoreceptor_inhibition": self.autoreceptor_inhibition,
            "in_phasic": self.in_phasic,
            "arousal_drive": self.arousal_drive,
            "gain_modulation": self.gain_modulation,
        }


# =============================================================================
# Phase 2: Surprise Model (Uncertainty Signaling)
# =============================================================================


@dataclass
class SurpriseConfig:
    """
    Configuration for surprise-driven NE signaling.

    Based on Dayan & Yu (2006), Payzan-LeNestour et al. (2013):
    - LC phasic encodes unexpected uncertainty (surprise)
    - High prediction error → phasic burst → increased learning rate
    - Sustained uncertainty → elevated tonic → exploration mode
    """

    # Prediction error tracking
    prediction_error_window: int = 50      # Window for error history
    baseline_uncertainty: float = 0.2      # Baseline expected uncertainty

    # Surprise thresholds (unsigned prediction error)
    surprise_threshold_low: float = 0.3    # Below this: routine
    surprise_threshold_high: float = 0.7   # Above this: high surprise

    # Phasic triggering from surprise
    surprise_phasic_gain: float = 1.5      # How much surprise amplifies phasic
    surprise_decay_rate: float = 0.1       # Decay of surprise signal

    # Uncertainty estimation (running variance)
    uncertainty_alpha: float = 0.1         # EMA factor for uncertainty
    uncertainty_gain: float = 0.5          # How much uncertainty raises tonic

    # Learning rate modulation
    learning_rate_min: float = 0.01        # Minimum learning rate
    learning_rate_max: float = 0.3         # Maximum learning rate
    learning_rate_sensitivity: float = 2.0  # Sensitivity to surprise

    # Change point detection (Nassar et al. 2012)
    change_point_threshold: float = 0.8    # Threshold for change detection
    hazard_rate: float = 0.05              # Prior probability of change


@dataclass
class SurpriseState:
    """Current state of surprise/uncertainty model."""

    # Current surprise level (unsigned prediction error)
    surprise: float = 0.0                  # Current surprise [0, 1]
    cumulative_surprise: float = 0.0       # EMA of recent surprise

    # Uncertainty estimation
    estimated_uncertainty: float = 0.2     # Expected variance in outcomes
    unexpected_uncertainty: float = 0.0    # Deviation from expected variance

    # Prediction tracking
    last_prediction: float = 0.0           # Last prediction made
    last_outcome: float = 0.0              # Last observed outcome
    prediction_error: float = 0.0          # Signed prediction error

    # Change point detection
    change_point_probability: float = 0.0  # P(change point just occurred)
    run_length: int = 0                    # Steps since last inferred change

    # Learning rate modulation
    adaptive_learning_rate: float = 0.1    # Current recommended learning rate

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "surprise": self.surprise,
            "cumulative_surprise": self.cumulative_surprise,
            "estimated_uncertainty": self.estimated_uncertainty,
            "unexpected_uncertainty": self.unexpected_uncertainty,
            "prediction_error": self.prediction_error,
            "change_point_probability": self.change_point_probability,
            "adaptive_learning_rate": self.adaptive_learning_rate,
        }


class SurpriseModel:
    """
    Surprise-driven norepinephrine model for uncertainty signaling.

    Implements the theory that LC-NE encodes unexpected uncertainty:

    1. Expected Uncertainty: Known variability in the environment
       - Does NOT trigger LC phasic (it's anticipated)
       - Modulates tonic baseline

    2. Unexpected Uncertainty: Violation of predictions (surprise)
       - DOES trigger LC phasic burst
       - Signals need to update models
       - Modulates learning rate

    Key insight (Dayan & Yu 2006):
    - Expected uncertainty → increase tonic → broader attention
    - Unexpected uncertainty → phasic burst → model update

    References:
        - Dayan & Yu (2006): Uncertainty, neuromodulation and attention
        - Payzan-LeNestour et al. (2013): LC encodes unexpected uncertainty
        - Nassar et al. (2012): Rational regulation of learning by surprise
    """

    def __init__(self, config: SurpriseConfig | None = None):
        """Initialize surprise model."""
        self.config = config or SurpriseConfig()
        self.state = SurpriseState()

        # History buffers
        self._prediction_errors: list[float] = []
        self._outcomes: list[float] = []

        logger.debug("SurpriseModel initialized")

    def observe_prediction(self, prediction: float) -> None:
        """
        Record a prediction before observing outcome.

        Args:
            prediction: Predicted value/probability
        """
        self.state.last_prediction = float(np.clip(prediction, -10, 10))

    def observe_outcome(self, outcome: float) -> float:
        """
        Observe actual outcome and compute surprise.

        Args:
            outcome: Observed value/probability

        Returns:
            Surprise magnitude [0, 1]
        """
        self.state.last_outcome = float(np.clip(outcome, -10, 10))

        # Compute signed prediction error
        self.state.prediction_error = self.state.last_outcome - self.state.last_prediction

        # Compute unsigned surprise (magnitude)
        raw_surprise = abs(self.state.prediction_error)
        self.state.surprise = float(np.clip(raw_surprise, 0.0, 1.0))

        # Store in history
        self._prediction_errors.append(self.state.prediction_error)
        self._outcomes.append(self.state.last_outcome)

        # Trim history
        max_len = self.config.prediction_error_window
        if len(self._prediction_errors) > max_len:
            self._prediction_errors = self._prediction_errors[-max_len:]
            self._outcomes = self._outcomes[-max_len:]

        # Update uncertainty estimate
        self._update_uncertainty()

        # Update change point probability
        self._update_change_point()

        # Compute adaptive learning rate
        self._update_learning_rate()

        # Update cumulative surprise (EMA)
        alpha = self.config.surprise_decay_rate
        self.state.cumulative_surprise = (
            alpha * self.state.surprise +
            (1 - alpha) * self.state.cumulative_surprise
        )

        return self.state.surprise

    def _update_uncertainty(self) -> None:
        """Update uncertainty estimate from prediction error variance."""
        if len(self._prediction_errors) < 3:
            return

        # Compute running variance of prediction errors
        errors = np.array(self._prediction_errors[-20:])
        variance = float(np.var(errors))

        # EMA update of expected uncertainty
        alpha = self.config.uncertainty_alpha
        old_uncertainty = self.state.estimated_uncertainty
        self.state.estimated_uncertainty = (
            alpha * variance + (1 - alpha) * old_uncertainty
        )

        # Unexpected uncertainty = deviation from expected
        # (i.e., variance is higher than usual)
        if old_uncertainty > 0:
            unexpected = max(0, variance - old_uncertainty) / old_uncertainty
            self.state.unexpected_uncertainty = float(np.clip(unexpected, 0, 1))
        else:
            self.state.unexpected_uncertainty = 0.0

    def _update_change_point(self) -> None:
        """
        Update change point probability (Nassar et al. 2012).

        Uses simple heuristic: high surprise → likely change point.
        """
        # High surprise increases change point probability
        if self.state.surprise > self.config.change_point_threshold:
            # Likely change point
            self.state.change_point_probability = min(
                0.9,
                self.state.change_point_probability + 0.3
            )
            self.state.run_length = 0
        else:
            # Decay change point probability
            self.state.change_point_probability *= 0.9
            self.state.run_length += 1

        # Add hazard rate baseline
        self.state.change_point_probability = max(
            self.config.hazard_rate,
            self.state.change_point_probability
        )

    def _update_learning_rate(self) -> None:
        """
        Compute adaptive learning rate from surprise.

        High surprise → high learning rate (update model quickly).
        Low surprise → low learning rate (trust current model).
        """
        # Base learning rate scales with surprise
        surprise_factor = self.state.surprise ** self.config.learning_rate_sensitivity

        # Also consider change point probability
        change_factor = self.state.change_point_probability

        # Combined factor
        combined = max(surprise_factor, change_factor)

        # Interpolate between min and max learning rate
        self.state.adaptive_learning_rate = (
            self.config.learning_rate_min +
            (self.config.learning_rate_max - self.config.learning_rate_min) * combined
        )

    def should_trigger_phasic(self) -> bool:
        """
        Check if surprise warrants a phasic burst.

        Returns:
            True if surprise exceeds threshold
        """
        return self.state.surprise > self.config.surprise_threshold_high

    def get_phasic_magnitude(self) -> float:
        """
        Get magnitude for phasic burst based on surprise.

        Returns:
            Phasic magnitude [0, 1]
        """
        if self.state.surprise < self.config.surprise_threshold_low:
            return 0.0

        # Scale between thresholds
        norm_surprise = (
            self.state.surprise - self.config.surprise_threshold_low
        ) / (self.config.surprise_threshold_high - self.config.surprise_threshold_low)

        return float(np.clip(
            norm_surprise * self.config.surprise_phasic_gain,
            0.0,
            1.0
        ))

    def get_tonic_modulation(self) -> float:
        """
        Get tonic modulation from expected uncertainty.

        High expected uncertainty → increase tonic → exploration.

        Returns:
            Tonic modulation factor [0, 1]
        """
        return float(np.clip(
            self.state.estimated_uncertainty * self.config.uncertainty_gain,
            0.0,
            0.5  # Cap at 50% tonic increase
        ))

    def reset(self) -> None:
        """Reset surprise model state."""
        self.state = SurpriseState()
        self._prediction_errors.clear()
        self._outcomes.clear()

    def get_stats(self) -> dict:
        """Get surprise model statistics."""
        return {
            "surprise": self.state.surprise,
            "cumulative_surprise": self.state.cumulative_surprise,
            "estimated_uncertainty": self.state.estimated_uncertainty,
            "unexpected_uncertainty": self.state.unexpected_uncertainty,
            "prediction_error": self.state.prediction_error,
            "change_point_probability": self.state.change_point_probability,
            "run_length": self.state.run_length,
            "adaptive_learning_rate": self.state.adaptive_learning_rate,
            "error_history_length": len(self._prediction_errors),
        }


class LocusCoeruleus:
    """
    Locus Coeruleus norepinephrine circuit.

    Models the LC-NE system with explicit tonic and phasic modes:

    Tonic Mode:
    - Sustained low-frequency firing (0.5-5 Hz)
    - Sets baseline arousal/alertness
    - Modulates signal-to-noise across cortex
    - Yerkes-Dodson: optimal performance at intermediate tonic

    Phasic Mode:
    - Brief high-frequency bursts (10-20 Hz)
    - Triggered by salient/novel stimuli
    - Resets attentional focus
    - Followed by refractory period

    Key insight (Aston-Jones & Cohen 2005):
    - High tonic + low phasic = exploration (broad attention)
    - Low tonic + high phasic = exploitation (focused attention)

    Usage:
        lc = LocusCoeruleus()
        lc.set_arousal_drive(0.6)  # Alert state

        # Salient stimulus arrives
        if lc.trigger_phasic(salience=0.8):
            logger.info("Phasic burst triggered!")

        lc.step(dt=0.01)
        ne = lc.get_ne_for_neural_field()
    """

    def __init__(
        self,
        config: LCConfig | None = None,
        surprise_config: SurpriseConfig | None = None,
        seed: int | None = None,
    ):
        """
        Initialize locus coeruleus.

        Args:
            config: LC configuration parameters
            surprise_config: Surprise model configuration (Phase 2)
            seed: Random seed for phasic decision (ATOM-P4-10)
        """
        self.config = config or LCConfig()
        self.state = LCState(
            firing_rate=self.config.tonic_optimal_rate,
            ne_level=self.config.ne_baseline,
        )

        # Phase 2: Surprise model integration
        self.surprise = SurpriseModel(surprise_config)

        # ATOM-P4-10: Seeded RNG for reproducible phasic decisions
        self._rng = np.random.default_rng(seed)

        # Callbacks
        self._ne_callbacks: list[Callable[[float], None]] = []

        # History
        self._firing_history: list[float] = []
        self._ne_history: list[float] = []
        self._max_history = 1000

        # Simulation time
        self._simulation_time = 0.0

        # ATOM-P3-10: NE arousal ceiling with habituation
        self._high_arousal_duration = 0.0
        self._habituation_factor = 1.0

        logger.info(
            f"LocusCoeruleus initialized: tonic_optimal={self.config.tonic_optimal_rate}Hz, "
            f"surprise_model=enabled, seed={seed}"
        )

    # =========================================================================
    # Core Dynamics
    # =========================================================================

    def step(self, dt: float = 0.01) -> float:
        """
        Advance LC dynamics by one timestep.

        Args:
            dt: Timestep in seconds

        Returns:
            Current NE level
        """
        # ATOM-P2-1: Input validation at NCA layer boundary
        from t4dm.core.validation import ValidationError
        if not np.isfinite(dt):
            raise ValidationError("dt", "Contains NaN or Inf values")

        self._simulation_time += dt

        # 1. Update phasic state
        self._update_phasic_state(dt)

        # 2. Compute autoreceptor inhibition
        self._update_autoreceptor()

        # 3. Update firing rate
        self._update_firing_rate(dt)

        # 4. Update NE concentration
        self._update_ne(dt)

        # 5. Compute Yerkes-Dodson gain
        self._update_gain_modulation()

        # 6. Classify mode
        self._classify_mode()

        # Track history
        self._firing_history.append(self.state.firing_rate)
        self._ne_history.append(self.state.ne_level)
        if len(self._firing_history) > self._max_history:
            self._firing_history = self._firing_history[-self._max_history:]
            self._ne_history = self._ne_history[-self._max_history:]

        # Fire callbacks
        for callback in self._ne_callbacks:
            callback(self.state.ne_level)

        return self.state.ne_level

    def _update_phasic_state(self, dt: float) -> None:
        """Update phasic burst timing."""
        self.state.time_since_phasic += dt

        if self.state.in_phasic:
            self.state.phasic_time_remaining -= dt
            if self.state.phasic_time_remaining <= 0:
                self.state.in_phasic = False
                self.state.phasic_time_remaining = 0.0

    def _update_autoreceptor(self) -> None:
        """
        Compute alpha-2 autoreceptor inhibition.

        High NE -> autoreceptor activation -> reduced firing
        """
        ne = self.state.ne_level
        ec50 = self.config.autoreceptor_ec50
        n = self.config.autoreceptor_hill

        # Hill function
        if ne > 0:
            hill = (ne ** n) / (ec50 ** n + ne ** n)
        else:
            hill = 0.0

        self.state.autoreceptor_inhibition = (
            self.config.autoreceptor_sensitivity * hill
        )

    def _update_firing_rate(self, dt: float) -> None:
        """Update LC neuron firing rate."""
        if self.state.in_phasic:
            # Phasic burst - high firing
            target_rate = self.config.phasic_peak_rate
        else:
            # Tonic firing based on arousal drive
            target_rate = self._compute_tonic_rate()

        # Apply autoreceptor inhibition
        target_rate *= (1 - self.state.autoreceptor_inhibition)

        # Apply stress/CRH modulation
        target_rate *= (1 + self.config.crh_gain * self.state.stress_input)

        # Clamp to valid range
        target_rate = np.clip(
            target_rate,
            self.config.quiescent_rate,
            self.config.phasic_peak_rate
        )

        # Smooth update
        alpha = dt / self.config.tau_firing
        self.state.firing_rate += alpha * (target_rate - self.state.firing_rate)
        self.state.firing_rate = float(np.clip(
            self.state.firing_rate,
            self.config.quiescent_rate,
            self.config.phasic_peak_rate
        ))

    def _compute_tonic_rate(self) -> float:
        """Compute tonic firing rate from arousal drive."""
        arousal = self.state.arousal_drive

        if arousal < self.config.quiescent_threshold:
            return self.config.quiescent_rate
        elif arousal < self.config.tonic_low_threshold:
            # Interpolate quiescent -> low
            t = (arousal - self.config.quiescent_threshold) / (
                self.config.tonic_low_threshold - self.config.quiescent_threshold
            )
            return self.config.quiescent_rate + t * (
                self.config.tonic_low_rate - self.config.quiescent_rate
            )
        elif arousal < self.config.tonic_optimal_threshold:
            # Interpolate low -> optimal
            t = (arousal - self.config.tonic_low_threshold) / (
                self.config.tonic_optimal_threshold - self.config.tonic_low_threshold
            )
            return self.config.tonic_low_rate + t * (
                self.config.tonic_optimal_rate - self.config.tonic_low_rate
            )
        elif arousal < self.config.tonic_high_threshold:
            # Interpolate optimal -> high
            t = (arousal - self.config.tonic_optimal_threshold) / (
                self.config.tonic_high_threshold - self.config.tonic_optimal_threshold
            )
            return self.config.tonic_optimal_rate + t * (
                self.config.tonic_high_rate - self.config.tonic_optimal_rate
            )
        else:
            return self.config.tonic_high_rate

    def _update_ne(self, dt: float) -> None:
        """Update extracellular NE concentration."""
        # Release proportional to firing
        release = self.config.ne_release_per_spike * self.state.firing_rate * dt

        # Reuptake (first-order)
        reuptake = self.config.ne_reuptake_rate * self.state.ne_level * dt

        # Net change
        d_ne = release - reuptake

        # Smooth update
        alpha = dt / self.config.tau_ne
        self.state.ne_level += alpha * d_ne

        # Clamp
        self.state.ne_level = float(np.clip(self.state.ne_level, 0.0, 1.0))

        # ATOM-P3-10: Arousal ceiling with habituation
        # Track high arousal duration and apply habituation
        if self.state.ne_level > 0.8:
            self._high_arousal_duration += dt
            if self._high_arousal_duration > 60.0:  # 60 sim-seconds
                self._habituation_factor *= 0.99
        else:
            self._high_arousal_duration = 0.0
            self._habituation_factor = min(1.0, self._habituation_factor + 0.001)

        # Apply habituation factor
        self.state.ne_level *= self._habituation_factor

    def _update_gain_modulation(self) -> None:
        """
        Compute Yerkes-Dodson performance modulation.

        Inverted-U: optimal performance at intermediate arousal.
        """
        arousal = self.state.ne_level
        optimal = self.config.optimal_arousal
        sensitivity = self.config.arousal_sensitivity

        # Gaussian around optimal arousal
        deviation = arousal - optimal
        self.state.gain_modulation = float(np.exp(
            -sensitivity * deviation * deviation
        ))

    def _classify_mode(self) -> None:
        """Classify current LC firing mode."""
        if self.state.in_phasic:
            self.state.mode = LCFiringMode.PHASIC
        elif self.state.arousal_drive < self.config.quiescent_threshold:
            self.state.mode = LCFiringMode.QUIESCENT
        elif self.state.arousal_drive < self.config.tonic_low_threshold:
            self.state.mode = LCFiringMode.TONIC_LOW
        elif self.state.arousal_drive < self.config.tonic_high_threshold:
            self.state.mode = LCFiringMode.TONIC_OPTIMAL
        else:
            self.state.mode = LCFiringMode.TONIC_HIGH

    # =========================================================================
    # Phasic Control
    # =========================================================================

    def trigger_phasic(self, salience: float = 1.0) -> bool:
        """
        Attempt to trigger a phasic burst.

        Args:
            salience: Salience of stimulus [0, 1]

        Returns:
            True if phasic burst was triggered
        """
        # Check refractory period
        if self.state.time_since_phasic < self.config.phasic_refractory:
            return False

        # Check salience threshold
        if salience < self.config.phasic_threshold:
            # ATOM-P4-10: Use seeded RNG for reproducible probabilistic trigger
            prob = salience / self.config.phasic_threshold
            if self._rng.random() > prob:
                return False

        # Trigger phasic burst
        self.state.in_phasic = True
        self.state.phasic_time_remaining = self.config.phasic_duration
        self.state.time_since_phasic = 0.0
        self.state.salience_input = salience

        logger.debug(f"LC phasic burst triggered at t={self._simulation_time:.3f}")
        return True

    def is_phasic_active(self) -> bool:
        """Check if currently in phasic mode."""
        return self.state.in_phasic

    def can_trigger_phasic(self) -> bool:
        """Check if phasic can be triggered (not in refractory)."""
        return self.state.time_since_phasic >= self.config.phasic_refractory

    # =========================================================================
    # External Inputs
    # =========================================================================

    def set_arousal_drive(self, arousal: float) -> None:
        """
        Set internal arousal drive.

        Args:
            arousal: Arousal level [0, 1]
        """
        self.state.arousal_drive = float(np.clip(arousal, 0, 1))

    def set_stress_input(self, stress: float) -> None:
        """
        Set stress/CRH input.

        Args:
            stress: Stress level [0, 1]
        """
        self.state.stress_input = float(np.clip(stress, 0, 1))

    def receive_salience_signal(self, salience: float) -> bool:
        """
        Receive salience signal that may trigger phasic.

        Args:
            salience: Salience magnitude [0, 1]

        Returns:
            True if phasic was triggered
        """
        self.state.salience_input = float(np.clip(salience, 0, 1))
        return self.trigger_phasic(salience)

    def receive_pfc_modulation(self, pfc_signal: float) -> None:
        """
        Receive prefrontal cortex modulation signal.

        Biological basis:
        - PFC → LC glutamatergic projection modulates LC firing mode
        - High PFC activity → tonic mode (focused attention)
        - Low PFC activity → phasic mode (exploratory attention)
        - Implements Aston-Jones & Cohen (2005) adaptive gain theory

        Args:
            pfc_signal: PFC activity level [0, 1]
        """
        pfc_signal = float(np.clip(pfc_signal, 0, 1))

        # High PFC → shift toward tonic mode (focused attention)
        # Low PFC → allow phasic bursts (exploratory mode)

        # Modulate arousal drive based on PFC
        # High PFC increases baseline arousal (more tonic)
        arousal_shift = 0.3 * (pfc_signal - 0.5)  # ±0.15 shift
        self.state.arousal_drive = float(np.clip(
            self.state.arousal_drive + arousal_shift,
            0.0,
            1.0
        ))

        # PFC modulates phasic responsiveness (top-down control)
        # High PFC → reduced phasic gain (less distractible, more focused)
        # Low PFC → increased phasic gain (more distractible, more exploratory)
        # Gain ranges from 0.5 (high PFC, focused) to 1.0 (low PFC, exploratory)
        self.state.pfc_gain = float(np.clip(
            1.0 - 0.5 * pfc_signal,  # High PFC (1.0) → gain 0.5, Low PFC (0.0) → gain 1.0
            0.5,
            1.0
        ))

    # =========================================================================
    # Integration Methods
    # =========================================================================

    def register_ne_callback(self, callback: Callable[[float], None]) -> None:
        """Register callback for NE changes."""
        self._ne_callbacks.append(callback)

    def get_ne_for_neural_field(self) -> float:
        """Get NE level for injection into neural field."""
        return self.state.ne_level

    def get_gain_modulation(self) -> float:
        """Get Yerkes-Dodson performance gain."""
        return self.state.gain_modulation

    def get_exploration_exploitation_bias(self) -> float:
        """
        Get exploration vs exploitation bias.

        High tonic -> exploration (0)
        Low tonic with phasic -> exploitation (1)

        Returns:
            Exploitation bias [0, 1]
        """
        # Higher tonic = more exploration
        tonic_factor = 1 - (self.state.arousal_drive / self.config.tonic_high_threshold)

        # Phasic = exploitation
        if self.state.in_phasic:
            return 1.0  # Full exploitation during phasic

        return float(np.clip(tonic_factor, 0, 1))

    def connect_to_hippocampus(self, novelty_signal: float) -> None:
        """
        Receive novelty signal from hippocampus.

        Novelty can trigger phasic responses.

        Args:
            novelty_signal: Hippocampal novelty [0, 1]
        """
        if novelty_signal > 0.5:
            # High novelty may trigger phasic
            self.receive_salience_signal(novelty_signal)

    def connect_to_amygdala(self, threat_signal: float) -> None:
        """
        Receive threat signal from amygdala.

        Threat increases arousal and may trigger phasic.

        Args:
            threat_signal: Amygdala threat [0, 1]
        """
        # Threat raises arousal
        self.state.arousal_drive = min(1.0, self.state.arousal_drive + 0.3 * threat_signal)

        if threat_signal > 0.7:
            self.trigger_phasic(threat_signal)

    # =========================================================================
    # Phase 2: Surprise-Driven NE API
    # =========================================================================

    def observe_prediction_outcome(
        self,
        prediction: float,
        outcome: float,
    ) -> tuple[float, bool]:
        """
        Observe a prediction-outcome pair and compute surprise.

        Phase 2 addition: Implements surprise-driven phasic triggering.

        Args:
            prediction: Predicted value
            outcome: Observed value

        Returns:
            (surprise_magnitude, phasic_triggered)
        """
        # Record prediction
        self.surprise.observe_prediction(prediction)

        # Observe outcome and compute surprise
        surprise = self.surprise.observe_outcome(outcome)

        # Apply uncertainty-based tonic modulation
        tonic_mod = self.surprise.get_tonic_modulation()
        self.state.arousal_drive = min(
            1.0,
            self.state.arousal_drive + tonic_mod
        )

        # Check if surprise triggers phasic
        phasic_triggered = False
        if self.surprise.should_trigger_phasic():
            magnitude = self.surprise.get_phasic_magnitude()
            phasic_triggered = self.trigger_phasic(magnitude)

        return surprise, phasic_triggered

    def get_surprise_level(self) -> float:
        """
        Get current surprise level.

        Returns:
            Surprise magnitude [0, 1]
        """
        return self.surprise.state.surprise

    def get_uncertainty(self) -> float:
        """
        Get estimated environmental uncertainty.

        Returns:
            Estimated uncertainty
        """
        return self.surprise.state.estimated_uncertainty

    def get_unexpected_uncertainty(self) -> float:
        """
        Get unexpected uncertainty (surprise in variance).

        Returns:
            Unexpected uncertainty [0, 1]
        """
        return self.surprise.state.unexpected_uncertainty

    def get_change_point_probability(self) -> float:
        """
        Get probability that a change point just occurred.

        Based on Nassar et al. (2012).

        Returns:
            Change point probability [0, 1]
        """
        return self.surprise.state.change_point_probability

    def get_adaptive_learning_rate(self) -> float:
        """
        Get surprise-modulated learning rate.

        High surprise → high learning rate.

        Returns:
            Recommended learning rate
        """
        return self.surprise.state.adaptive_learning_rate

    def should_update_model(self) -> bool:
        """
        Check if model should be updated based on surprise/change detection.

        Returns:
            True if high surprise or change point detected
        """
        return (
            self.surprise.should_trigger_phasic() or
            self.surprise.state.change_point_probability > 0.5
        )

    # =========================================================================
    # Statistics and State
    # =========================================================================

    def get_stats(self) -> dict:
        """Get LC statistics."""
        stats = {
            "mode": self.state.mode.value,
            "firing_rate": self.state.firing_rate,
            "ne_level": self.state.ne_level,
            "arousal_drive": self.state.arousal_drive,
            "gain_modulation": self.state.gain_modulation,
            "autoreceptor_inhibition": self.state.autoreceptor_inhibition,
            "in_phasic": self.state.in_phasic,
            "time_since_phasic": self.state.time_since_phasic,
            "simulation_time": self._simulation_time,
        }

        if self._firing_history:
            stats["avg_firing"] = float(np.mean(self._firing_history))
            stats["avg_ne"] = float(np.mean(self._ne_history))

        # Phase 2: Add surprise model stats
        stats["surprise"] = self.surprise.get_stats()

        return stats

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = LCState(
            firing_rate=self.config.tonic_optimal_rate,
            ne_level=self.config.ne_baseline,
        )
        self._firing_history.clear()
        self._ne_history.clear()
        self._simulation_time = 0.0
        self.surprise.reset()  # Phase 2: Reset surprise model
        logger.info("LocusCoeruleus reset")

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "state": self.state.to_dict(),
            "simulation_time": self._simulation_time,
        }

    def load_state(self, saved: dict) -> None:
        """Load state from persistence."""
        if "simulation_time" in saved:
            self._simulation_time = saved["simulation_time"]
        if "state" in saved:
            s = saved["state"]
            self.state.firing_rate = s.get("firing_rate", self.config.tonic_optimal_rate)
            self.state.ne_level = s.get("ne_level", self.config.ne_baseline)
            self.state.arousal_drive = s.get("arousal_drive", 0.5)


def create_locus_coeruleus(
    tonic_optimal_rate: float = 3.0,
    phasic_peak_rate: float = 15.0,
    optimal_arousal: float = 0.6,
) -> LocusCoeruleus:
    """
    Factory function to create locus coeruleus.

    Args:
        tonic_optimal_rate: Optimal tonic firing rate (Hz)
        phasic_peak_rate: Peak phasic firing rate (Hz)
        optimal_arousal: Optimal arousal for Yerkes-Dodson

    Returns:
        Configured LocusCoeruleus
    """
    config = LCConfig(
        tonic_optimal_rate=tonic_optimal_rate,
        phasic_peak_rate=phasic_peak_rate,
        optimal_arousal=optimal_arousal,
    )
    return LocusCoeruleus(config)


__all__ = [
    # Core Locus Coeruleus
    "LocusCoeruleus",
    "LCConfig",
    "LCState",
    "LCFiringMode",
    "create_locus_coeruleus",
    # Phase 2: Surprise Model
    "SurpriseModel",
    "SurpriseConfig",
    "SurpriseState",
]
