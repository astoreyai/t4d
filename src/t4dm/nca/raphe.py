"""
Raphe Nucleus Serotonin Circuit for T4DM.

Biological Basis:
- Dorsal Raphe Nucleus (DRN) is primary source of brain serotonin
- DRN neurons have 5-HT1A somatodendritic autoreceptors
- Autoreceptors provide negative feedback: high 5-HT → reduced firing
- This creates homeostatic regulation of serotonin levels
- DRN projects widely (cortex, hippocampus, amygdala, striatum)

Key Features:
- Autoreceptor-mediated negative feedback
- Tonic firing (~1-5 Hz) with state-dependent modulation
- Desensitization of autoreceptors over time (SSRI mechanism)
- Integration with ww.learning.SerotoninSystem
- Phase 2: Patience model with temporal discounting (Doya 2002)

References:
- Blier & de Montigny (1987): 5-HT1A autoreceptor desensitization
- Celada et al. (2001): Control of DRN activity
- Hajos et al. (2007): DRN firing patterns
- Doya (2002): Metalearning and neuromodulation
- Miyazaki et al. (2014): Serotonin and patience for future rewards
- Schweighofer et al. (2008): Low serotonin and impulsive choice
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from t4dm.nca.vta import VTACircuit

logger = logging.getLogger(__name__)


class RapheState(Enum):
    """Raphe nucleus activity states."""
    QUIESCENT = "quiescent"    # Near-zero firing (sleep)
    TONIC = "tonic"            # Regular ~2-3 Hz firing (wake)
    ELEVATED = "elevated"      # Increased firing (stress/arousal)
    SUPPRESSED = "suppressed"  # Autoreceptor-inhibited


@dataclass
class RapheConfig:
    """Configuration for raphe nucleus dynamics."""

    # Baseline firing
    baseline_rate: float = 2.5        # Hz, tonic DRN firing
    max_rate: float = 8.0             # Hz, maximum firing
    min_rate: float = 0.1             # Hz, minimum (not zero)

    # 5-HT1A autoreceptor parameters
    autoreceptor_sensitivity: float = 0.5   # Inhibition strength
    autoreceptor_ec50: float = 0.4          # 5-HT level for half-max inhibition
    autoreceptor_hill: float = 2.0          # Hill coefficient (cooperativity)

    # Desensitization dynamics (like SSRI effect)
    desensitization_rate: float = 0.01      # Per-step desensitization
    resensitization_rate: float = 0.001     # Per-step recovery
    max_desensitization: float = 0.8        # Maximum desensitization

    # 5-HT release dynamics
    release_per_spike: float = 0.02         # 5-HT release per spike
    reuptake_rate: float = 0.1              # Per-step reuptake
    diffusion_rate: float = 0.05            # Spatial spread

    # Homeostatic setpoint
    setpoint: float = 0.4                   # Target 5-HT level
    setpoint_gain: float = 0.5              # Feedback strength

    # Temporal dynamics
    tau_firing: float = 0.2                 # Firing rate time constant (s)
    tau_5ht: float = 0.5                    # 5-HT decay time constant (s)


@dataclass
class RapheNucleusState:
    """Current state of raphe nucleus."""

    state: RapheState = RapheState.TONIC
    firing_rate: float = 2.5              # Current firing (Hz)
    extracellular_5ht: float = 0.4        # 5-HT concentration [0, 1]

    # Autoreceptor state
    autoreceptor_inhibition: float = 0.0  # Current inhibition [0, 1]
    autoreceptor_sensitivity: float = 1.0  # Current sensitivity [0, 1]

    # Input tracking
    stress_input: float = 0.0             # Stress/arousal input
    reward_input: float = 0.0             # Reward modulation

    # Homeostatic error
    setpoint_error: float = 0.0           # Current - setpoint

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "firing_rate": self.firing_rate,
            "extracellular_5ht": self.extracellular_5ht,
            "autoreceptor_inhibition": self.autoreceptor_inhibition,
            "autoreceptor_sensitivity": self.autoreceptor_sensitivity,
            "setpoint_error": self.setpoint_error,
        }


# =============================================================================
# Phase 2: Patience Model (Temporal Discounting)
# =============================================================================


@dataclass
class PatienceConfig:
    """
    Configuration for serotonin-based patience/temporal discounting.

    Based on Doya (2002) and Miyazaki et al. (2014):
    - Serotonin modulates the temporal discount rate (gamma)
    - High 5-HT → lower discount rate → preference for delayed rewards
    - Low 5-HT → higher discount rate → preference for immediate rewards
    """

    # Discount rate parameters
    gamma_min: float = 0.8      # Minimum discount factor (low 5-HT, impatient)
    gamma_max: float = 0.99     # Maximum discount factor (high 5-HT, patient)
    gamma_slope: float = 2.0    # Sensitivity of gamma to 5-HT

    # Temporal horizon (in time steps)
    horizon_min: float = 3.0    # Minimum planning horizon (impatient)
    horizon_max: float = 50.0   # Maximum planning horizon (patient)

    # Wait/don't-wait decision thresholds
    wait_threshold: float = 0.5     # 5-HT level above which waiting is favored
    impulsivity_threshold: float = 0.25  # Below this, strong impulsivity

    # Temporal decay of waiting benefit
    wait_decay_rate: float = 0.1    # How fast waiting benefit decays

    # Integration with reward
    reward_patience_boost: float = 0.1   # Anticipated reward increases patience
    punishment_patience_drop: float = 0.2  # Anticipated punishment decreases patience


@dataclass
class PatienceState:
    """Current state of patience model."""

    discount_rate: float = 0.95     # Current gamma (temporal discount)
    temporal_horizon: float = 20.0  # Current planning horizon (steps)
    wait_signal: float = 0.5        # Tendency to wait [0=impulsive, 1=patient]
    impulsivity: float = 0.5        # Impulsivity level [0=patient, 1=impulsive]

    # Temporal integration
    cumulative_wait_time: float = 0.0    # How long currently waiting
    expected_delay: float = 0.0          # Expected delay to reward
    expected_reward_magnitude: float = 0.0  # Expected reward value

    # Decision metrics
    immediate_value: float = 0.0    # Value of acting now
    delayed_value: float = 0.0      # Discounted value of waiting

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "discount_rate": self.discount_rate,
            "temporal_horizon": self.temporal_horizon,
            "wait_signal": self.wait_signal,
            "impulsivity": self.impulsivity,
            "cumulative_wait_time": self.cumulative_wait_time,
            "expected_delay": self.expected_delay,
        }


class PatienceModel:
    """
    Serotonin-dependent patience model for temporal discounting.

    Implements the Doya (2002) hypothesis that serotonin encodes the
    time scale of reward prediction (temporal discount rate).

    Key equation:
        gamma = gamma_min + (gamma_max - gamma_min) * sigmoid(slope * (5HT - 0.5))

    Higher 5-HT → higher gamma → more patient → longer horizon planning.

    References:
        - Doya (2002): Metalearning and neuromodulation
        - Miyazaki et al. (2014): Serotonin and patience for future rewards
        - Schweighofer et al. (2008): Low serotonin and impulsive choice
    """

    def __init__(self, config: PatienceConfig | None = None):
        """Initialize patience model."""
        self.config = config or PatienceConfig()
        self.state = PatienceState()

        logger.debug("PatienceModel initialized")

    def update(self, serotonin_level: float, dt: float = 0.1) -> PatienceState:
        """
        Update patience model based on current serotonin level.

        Args:
            serotonin_level: Current 5-HT level [0, 1]
            dt: Timestep in seconds

        Returns:
            Updated patience state
        """
        # 1. Compute discount rate (gamma) from 5-HT
        self.state.discount_rate = self._compute_discount_rate(serotonin_level)

        # 2. Compute temporal horizon
        self.state.temporal_horizon = self._compute_horizon(serotonin_level)

        # 3. Compute wait signal
        self.state.wait_signal = self._compute_wait_signal(serotonin_level)

        # 4. Compute impulsivity (inverse of patience)
        self.state.impulsivity = 1.0 - self.state.wait_signal

        # 5. Update cumulative wait time if waiting
        if self.state.wait_signal > 0.5:
            self.state.cumulative_wait_time += dt
        else:
            # Reset if acting impulsively
            self.state.cumulative_wait_time *= 0.9  # Decay rather than reset

        return self.state

    def _compute_discount_rate(self, serotonin: float) -> float:
        """
        Compute temporal discount rate from serotonin level.

        gamma = gamma_min + (gamma_max - gamma_min) * sigmoid(slope * (5HT - 0.5))

        Args:
            serotonin: 5-HT level [0, 1]

        Returns:
            Discount rate gamma [gamma_min, gamma_max]
        """
        # Centered sigmoid
        x = self.config.gamma_slope * (serotonin - 0.5)
        sigmoid = 1.0 / (1.0 + np.exp(-x))

        gamma = self.config.gamma_min + (
            self.config.gamma_max - self.config.gamma_min
        ) * sigmoid

        return float(gamma)

    def _compute_horizon(self, serotonin: float) -> float:
        """
        Compute effective planning horizon from serotonin level.

        Horizon ≈ 1 / (1 - gamma) in discounted sum sense.

        Args:
            serotonin: 5-HT level [0, 1]

        Returns:
            Planning horizon in time steps
        """
        # Linear interpolation based on serotonin
        horizon = self.config.horizon_min + (
            self.config.horizon_max - self.config.horizon_min
        ) * serotonin

        return float(horizon)

    def _compute_wait_signal(self, serotonin: float) -> float:
        """
        Compute wait/don't-wait signal.

        Above wait_threshold: tendency to wait increases
        Below impulsivity_threshold: strong impulsivity

        Args:
            serotonin: 5-HT level [0, 1]

        Returns:
            Wait signal [0=act now, 1=wait]
        """
        if serotonin < self.config.impulsivity_threshold:
            # Very low 5-HT: strong impulsivity
            wait = serotonin / self.config.impulsivity_threshold * 0.3
        elif serotonin < self.config.wait_threshold:
            # Moderate: gradual increase
            norm = (serotonin - self.config.impulsivity_threshold) / (
                self.config.wait_threshold - self.config.impulsivity_threshold
            )
            wait = 0.3 + norm * 0.2
        else:
            # High 5-HT: patience increases
            excess = serotonin - self.config.wait_threshold
            wait = 0.5 + excess * 1.0  # Scale to reach 1.0 at 5-HT=1.0

        return float(np.clip(wait, 0.0, 1.0))

    def evaluate_wait_decision(
        self,
        immediate_reward: float,
        delayed_reward: float,
        delay_steps: int,
        serotonin_level: float,
    ) -> tuple[bool, float]:
        """
        Evaluate whether to wait for delayed reward or act immediately.

        Args:
            immediate_reward: Value available now
            delayed_reward: Value available after delay
            delay_steps: Steps until delayed reward
            serotonin_level: Current 5-HT level

        Returns:
            (should_wait, expected_value_difference)
        """
        # Update state
        self.update(serotonin_level)

        # Compute discounted delayed value
        gamma = self.state.discount_rate
        discounted_delayed = delayed_reward * (gamma ** delay_steps)

        # Store for analysis
        self.state.immediate_value = immediate_reward
        self.state.delayed_value = discounted_delayed
        self.state.expected_delay = float(delay_steps)
        self.state.expected_reward_magnitude = delayed_reward

        # Decision: wait if discounted delayed > immediate
        value_diff = discounted_delayed - immediate_reward

        # Apply wait signal as bias
        # High 5-HT (high wait_signal) tips balance toward waiting
        biased_diff = value_diff + (self.state.wait_signal - 0.5) * 0.2

        should_wait = biased_diff > 0

        return should_wait, float(value_diff)

    def get_temporal_value(
        self,
        reward_magnitude: float,
        delay_steps: int,
    ) -> float:
        """
        Get temporally discounted value of a future reward.

        Args:
            reward_magnitude: Size of future reward
            delay_steps: Steps until reward

        Returns:
            Discounted present value
        """
        return float(reward_magnitude * (self.state.discount_rate ** delay_steps))

    def reset(self) -> None:
        """Reset patience state."""
        self.state = PatienceState()

    def get_stats(self) -> dict:
        """Get patience model statistics."""
        return {
            "discount_rate": self.state.discount_rate,
            "temporal_horizon": self.state.temporal_horizon,
            "wait_signal": self.state.wait_signal,
            "impulsivity": self.state.impulsivity,
            "cumulative_wait_time": self.state.cumulative_wait_time,
            "immediate_value": self.state.immediate_value,
            "delayed_value": self.state.delayed_value,
        }


class RapheNucleus:
    """
    Dorsal Raphe Nucleus serotonin circuit.

    Models the dynamics of DRN serotonergic neurons with:
    1. Tonic firing regulated by 5-HT1A autoreceptors
    2. Negative feedback: high 5-HT → autoreceptor activation → reduced firing
    3. Homeostatic regulation toward setpoint
    4. Desensitization (like chronic SSRI effect)

    Key equation:
        firing = baseline * (1 - autoreceptor_inhibition) * (1 + inputs)
        inhibition = sensitivity * hill_function(5-HT)
    """

    def __init__(
        self,
        config: RapheConfig | None = None,
        patience_config: PatienceConfig | None = None,
    ):
        """
        Initialize raphe nucleus.

        Args:
            config: Raphe configuration parameters
            patience_config: Patience model configuration (Phase 2)
        """
        self.config = config or RapheConfig()
        self.state = RapheNucleusState(
            firing_rate=self.config.baseline_rate,
            extracellular_5ht=self.config.setpoint,
            autoreceptor_sensitivity=1.0,
        )

        # Phase 2: Patience model integration
        self.patience = PatienceModel(patience_config)

        # Callbacks for integration
        self._5ht_callbacks: list[Callable[[float], None]] = []

        # History for analysis
        self._firing_history: list[float] = []
        self._5ht_history: list[float] = []
        self._max_history = 1000

        # P2.1 Autowiring: Reference to VTA for automatic RPE modulation
        self._vta: VTACircuit | None = None

        logger.info(
            f"RapheNucleus initialized: baseline={self.config.baseline_rate}Hz, "
            f"setpoint={self.config.setpoint}, patience_model=enabled"
        )

    # =========================================================================
    # Core Dynamics
    # =========================================================================

    def step(self, dt: float = 0.1) -> float:
        """
        Advance raphe dynamics by one timestep.
        P2.1: Auto-calls VTA RPE modulation if wired.

        Args:
            dt: Timestep in seconds

        Returns:
            Current 5-HT level
        """
        # P2.1: Auto-receive VTA RPE signal if wired
        if self._vta is not None:
            self.connect_to_vta(self._vta.state.last_rpe)

        # 1. Compute autoreceptor inhibition
        self._update_autoreceptor_inhibition()

        # 2. Compute homeostatic drive
        homeostatic_drive = self._compute_homeostatic_drive()

        # 3. Update firing rate
        self._update_firing_rate(homeostatic_drive, dt)

        # 4. Update 5-HT concentration
        self._update_5ht(dt)

        # 5. Update desensitization
        self._update_desensitization(dt)

        # 6. Update state classification
        self._classify_state()

        # Track history
        self._firing_history.append(self.state.firing_rate)
        self._5ht_history.append(self.state.extracellular_5ht)
        if len(self._firing_history) > self._max_history:
            self._firing_history = self._firing_history[-self._max_history:]
            self._5ht_history = self._5ht_history[-self._max_history:]

        # 7. Update patience model (Phase 2)
        self.patience.update(self.state.extracellular_5ht, dt)

        # Fire callbacks
        for callback in self._5ht_callbacks:
            callback(self.state.extracellular_5ht)

        return self.state.extracellular_5ht

    def _update_autoreceptor_inhibition(self) -> None:
        """
        Compute 5-HT1A autoreceptor-mediated inhibition.

        Uses Hill function: inhibition = max_inhib * (5HT^n / (EC50^n + 5HT^n))
        """
        ec50 = self.config.autoreceptor_ec50
        n = self.config.autoreceptor_hill
        ht = self.state.extracellular_5ht

        # Hill function
        if ht > 0:
            hill = (ht ** n) / (ec50 ** n + ht ** n)
        else:
            hill = 0.0

        # Apply sensitivity (desensitization reduces this)
        raw_inhibition = (
            self.config.autoreceptor_sensitivity *
            self.state.autoreceptor_sensitivity *
            hill
        )

        self.state.autoreceptor_inhibition = float(np.clip(raw_inhibition, 0, 1))

    def _compute_homeostatic_drive(self) -> float:
        """
        Compute drive to return 5-HT to setpoint.

        Returns:
            Homeostatic drive (positive = increase firing)
        """
        error = self.config.setpoint - self.state.extracellular_5ht
        self.state.setpoint_error = error

        # Proportional control
        drive = self.config.setpoint_gain * error

        return drive

    def _update_firing_rate(self, homeostatic_drive: float, dt: float) -> None:
        """
        Update DRN firing rate.

        firing = baseline * (1 - inhibition) * (1 + stress) * (1 + homeostatic)
        """
        # Base firing modulated by autoreceptor inhibition
        effective_base = self.config.baseline_rate * (
            1 - self.state.autoreceptor_inhibition
        )

        # External inputs (stress increases, reward modulates)
        input_factor = 1.0 + self.state.stress_input - 0.2 * self.state.reward_input

        # Homeostatic adjustment
        homeostatic_factor = 1.0 + homeostatic_drive

        # Target firing rate
        target_rate = effective_base * input_factor * homeostatic_factor

        # Clamp to bounds
        target_rate = np.clip(
            target_rate,
            self.config.min_rate,
            self.config.max_rate
        )

        # Smooth update (low-pass filter)
        alpha = dt / self.config.tau_firing
        self.state.firing_rate += alpha * (target_rate - self.state.firing_rate)
        self.state.firing_rate = float(np.clip(
            self.state.firing_rate,
            self.config.min_rate,
            self.config.max_rate
        ))

    def _update_5ht(self, dt: float) -> None:
        """
        Update extracellular 5-HT concentration.

        d[5-HT]/dt = release - reuptake - diffusion
        """
        # Release proportional to firing
        release = self.config.release_per_spike * self.state.firing_rate * dt

        # Reuptake (first-order)
        reuptake = self.config.reuptake_rate * self.state.extracellular_5ht * dt

        # Net change
        d_5ht = release - reuptake

        # Smooth update
        alpha = dt / self.config.tau_5ht
        self.state.extracellular_5ht += alpha * d_5ht

        # Clamp to [0, 1]
        self.state.extracellular_5ht = float(np.clip(
            self.state.extracellular_5ht, 0.0, 1.0
        ))

    def _update_desensitization(self, dt: float) -> None:
        """
        Update autoreceptor desensitization.

        Chronic high 5-HT → desensitization (reduced sensitivity)
        Low 5-HT → resensitization
        """
        # Desensitization driven by 5-HT above setpoint
        if self.state.extracellular_5ht > self.config.setpoint:
            excess = self.state.extracellular_5ht - self.config.setpoint
            desens_rate = self.config.desensitization_rate * excess
        else:
            desens_rate = 0.0

        # Resensitization always occurs
        resens_rate = self.config.resensitization_rate

        # Net change
        d_sensitivity = resens_rate - desens_rate

        # Update
        new_sens = self.state.autoreceptor_sensitivity + d_sensitivity * dt

        # Clamp
        min_sens = 1 - self.config.max_desensitization
        self.state.autoreceptor_sensitivity = float(np.clip(new_sens, min_sens, 1.0))

    def _classify_state(self) -> None:
        """Classify current raphe state."""
        rate = self.state.firing_rate
        inhib = self.state.autoreceptor_inhibition

        if rate < 0.5:
            self.state.state = RapheState.QUIESCENT
        elif inhib > 0.5:
            self.state.state = RapheState.SUPPRESSED
        elif rate > self.config.baseline_rate * 1.5:
            self.state.state = RapheState.ELEVATED
        else:
            self.state.state = RapheState.TONIC

    # =========================================================================
    # P2.1: Autowiring
    # =========================================================================

    def set_vta(self, vta: VTACircuit) -> None:
        """
        P2.1: Set VTA circuit for automatic RPE modulation.

        After calling this, step() will automatically receive RPE signals
        from the VTA on each timestep.

        Args:
            vta: VTACircuit instance to wire for bidirectional modulation
        """
        self._vta = vta
        logger.debug("P2.1: RapheNucleus VTA autowiring enabled")

    # =========================================================================
    # External Inputs
    # =========================================================================

    def set_stress_input(self, stress: float) -> None:
        """
        Set stress/arousal input.

        Args:
            stress: Stress level [0, 1]
        """
        self.state.stress_input = float(np.clip(stress, 0, 1))

    def set_reward_input(self, reward: float) -> None:
        """
        Set reward modulation.

        Args:
            reward: Reward signal [0, 1]
        """
        self.state.reward_input = float(np.clip(reward, 0, 1))

    def inject_5ht(self, amount: float) -> None:
        """
        Inject external 5-HT (like drug effect).

        Args:
            amount: Amount to add [0, 1]
        """
        self.state.extracellular_5ht = float(np.clip(
            self.state.extracellular_5ht + amount, 0, 1
        ))

    def block_reuptake(self, block_fraction: float) -> None:
        """
        Block 5-HT reuptake (SSRI-like effect).

        Args:
            block_fraction: Fraction of reuptake blocked [0, 1]
        """
        # Temporarily reduce reuptake rate
        self.config.reuptake_rate *= (1 - block_fraction)

    # =========================================================================
    # Integration Methods
    # =========================================================================

    def register_5ht_callback(self, callback: Callable[[float], None]) -> None:
        """Register callback for 5-HT changes."""
        self._5ht_callbacks.append(callback)

    def get_5ht_for_neural_field(self) -> float:
        """Get 5-HT level for injection into neural field."""
        return self.state.extracellular_5ht

    def get_mood_modulation(self) -> float:
        """
        Get mood modulation factor for SerotoninSystem integration.

        Returns:
            Mood factor [0, 1] based on 5-HT level
        """
        # Higher 5-HT → higher mood
        return self.state.extracellular_5ht

    def connect_to_vta(self, vta_rpe: float) -> None:
        """
        Receive VTA RPE signal.

        Positive RPE can suppress DRN (reward reduces need for patience).

        Args:
            vta_rpe: VTA reward prediction error
        """
        # Positive RPE reduces DRN activity
        self.state.reward_input = max(0, vta_rpe)

    def receive_pfc_modulation(self, pfc_signal: float) -> None:
        """
        Receive prefrontal cortex modulation signal.

        Biological basis:
        - mPFC (medial PFC) → DRN glutamatergic projection
        - PFC activation dampens stress-induced 5-HT response
        - Implements top-down emotional regulation
        - High PFC → reduced stress reactivity

        Args:
            pfc_signal: PFC activity level [0, 1]
        """
        pfc_signal = float(np.clip(pfc_signal, 0, 1))

        # High PFC dampens stress response
        # This represents cognitive reappraisal / emotional regulation
        stress_damping = pfc_signal * 0.4  # Up to 40% stress reduction
        self.state.stress_input = float(np.clip(
            self.state.stress_input * (1 - stress_damping),
            0.0,
            1.0
        ))

        # Also slightly increases baseline 5-HT (mood regulation)
        # PFC-mediated mood stabilization
        mood_boost = 0.05 * pfc_signal
        self.state.extracellular_5ht = float(np.clip(
            self.state.extracellular_5ht + mood_boost,
            0.0,
            1.0
        ))

    # =========================================================================
    # P2.1: Raphe → VTA Inhibition
    # =========================================================================

    def get_vta_inhibition(self) -> float:
        """
        Get inhibitory signal for VTA dopamine neurons.

        Biological basis:
        - 5-HT inhibits VTA DA neurons via 5-HT2C receptors
        - High serotonin → reduced dopamine (opponent process)
        - This balances reward-seeking with impulse control

        Returns:
            Inhibition strength [0, 1] for VTA
        """
        # Base inhibition proportional to 5-HT level
        # Centered around setpoint: above setpoint = net inhibition
        ht_excess = self.state.extracellular_5ht - self.config.setpoint
        inhibition = 0.3 * ht_excess  # Scale factor

        # Always some baseline inhibition when 5-HT > 0.3
        if self.state.extracellular_5ht > 0.3:
            inhibition += 0.1 * (self.state.extracellular_5ht - 0.3)

        return float(np.clip(inhibition, 0.0, 0.5))

    def get_patience_signal(self) -> float:
        """
        Get patience/waiting signal for downstream targets.

        Phase 2: Now uses comprehensive patience model with temporal discounting.
        Higher 5-HT → more patience, less impulsive.

        Returns:
            Patience signal [0, 1]
        """
        return self.patience.state.wait_signal

    # =========================================================================
    # Phase 2: Temporal Discounting API
    # =========================================================================

    def get_discount_rate(self) -> float:
        """
        Get current temporal discount rate (gamma).

        Phase 2 addition based on Doya (2002).

        Returns:
            Discount rate [gamma_min, gamma_max]
        """
        return self.patience.state.discount_rate

    def get_temporal_horizon(self) -> float:
        """
        Get current effective planning horizon.

        Returns:
            Planning horizon in time steps
        """
        return self.patience.state.temporal_horizon

    def get_impulsivity(self) -> float:
        """
        Get current impulsivity level (inverse of patience).

        Returns:
            Impulsivity [0=patient, 1=impulsive]
        """
        return self.patience.state.impulsivity

    def evaluate_wait_decision(
        self,
        immediate_reward: float,
        delayed_reward: float,
        delay_steps: int,
    ) -> tuple[bool, float]:
        """
        Evaluate whether to wait for delayed reward or act immediately.

        Uses current 5-HT level to compute temporal discounting.

        Args:
            immediate_reward: Value available now
            delayed_reward: Value available after delay
            delay_steps: Steps until delayed reward

        Returns:
            (should_wait, expected_value_difference)
        """
        return self.patience.evaluate_wait_decision(
            immediate_reward,
            delayed_reward,
            delay_steps,
            self.state.extracellular_5ht,
        )

    def get_temporal_value(
        self,
        reward_magnitude: float,
        delay_steps: int,
    ) -> float:
        """
        Get temporally discounted value of a future reward.

        Args:
            reward_magnitude: Size of future reward
            delay_steps: Steps until reward

        Returns:
            Discounted present value
        """
        return self.patience.get_temporal_value(reward_magnitude, delay_steps)

    # =========================================================================
    # Statistics and State
    # =========================================================================

    def get_stats(self) -> dict:
        """Get raphe nucleus statistics."""
        stats = {
            "state": self.state.state.value,
            "firing_rate": self.state.firing_rate,
            "extracellular_5ht": self.state.extracellular_5ht,
            "autoreceptor_inhibition": self.state.autoreceptor_inhibition,
            "autoreceptor_sensitivity": self.state.autoreceptor_sensitivity,
            "setpoint_error": self.state.setpoint_error,
            "stress_input": self.state.stress_input,
        }

        if self._firing_history:
            stats["avg_firing"] = float(np.mean(self._firing_history))
            stats["avg_5ht"] = float(np.mean(self._5ht_history))

        # Phase 2: Add patience model stats
        patience_stats = self.patience.get_stats()
        stats["patience"] = patience_stats

        return stats

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = RapheNucleusState(
            firing_rate=self.config.baseline_rate,
            extracellular_5ht=self.config.setpoint,
            autoreceptor_sensitivity=1.0,
        )
        self._firing_history.clear()
        self._5ht_history.clear()
        self.patience.reset()  # Phase 2: Reset patience model
        logger.info("RapheNucleus reset")

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "state": self.state.to_dict(),
            "config": {
                "baseline_rate": self.config.baseline_rate,
                "setpoint": self.config.setpoint,
                "autoreceptor_sensitivity": self.config.autoreceptor_sensitivity,
            }
        }

    def load_state(self, saved: dict) -> None:
        """Load state from persistence."""
        if "state" in saved:
            s = saved["state"]
            self.state.firing_rate = s.get("firing_rate", self.config.baseline_rate)
            self.state.extracellular_5ht = s.get("extracellular_5ht", self.config.setpoint)
            self.state.autoreceptor_sensitivity = s.get("autoreceptor_sensitivity", 1.0)


def create_raphe_nucleus(
    baseline_rate: float = 2.5,
    setpoint: float = 0.4,
    autoreceptor_sensitivity: float = 0.5,
) -> RapheNucleus:
    """
    Factory function to create raphe nucleus.

    Args:
        baseline_rate: Baseline firing rate (Hz)
        setpoint: Homeostatic 5-HT setpoint
        autoreceptor_sensitivity: Autoreceptor inhibition strength

    Returns:
        Configured RapheNucleus
    """
    config = RapheConfig(
        baseline_rate=baseline_rate,
        setpoint=setpoint,
        autoreceptor_sensitivity=autoreceptor_sensitivity,
    )
    return RapheNucleus(config)


__all__ = [
    # Core Raphe Nucleus
    "RapheNucleus",
    "RapheConfig",
    "RapheNucleusState",
    "RapheState",
    "create_raphe_nucleus",
    # Phase 2: Patience Model
    "PatienceModel",
    "PatienceConfig",
    "PatienceState",
]
