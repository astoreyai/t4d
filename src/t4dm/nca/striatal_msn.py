"""
D1/D2 Medium Spiny Neuron Populations for Striatal Action Selection.

Biological Basis:
- D1 MSNs: Express D1 receptors → "direct pathway" → promotes action (GO)
- D2 MSNs: Express D2 receptors → "indirect pathway" → inhibits action (NO-GO)
- Dopamine oppositely modulates: excites D1, inhibits D2
- Competition creates GO/NO-GO decision making
- Key for reinforcement learning: DA boosts D1 plasticity (LTP), D2 (LTD)
- TANs (Tonically Active Neurons): Cholinergic interneurons that pause during learning

Architecture:
    Cortex → Striatum (D1 + D2 MSNs + TANs) → Output
                ↓ DA from VTA/SNc
           D1 (GO) vs D2 (NO-GO) competition
           TANs pause marks "when" reinforcement occurred

Key Features:
1. D1/D2 receptor binding kinetics
2. Opponent process dynamics (GO vs NO-GO)
3. Dopamine-modulated plasticity
4. Action selection via winner-take-all
5. Habit formation (D1 strength increases with practice)
6. Fix 2: TAN pause mechanism (Aosaki et al. 1994)

References:
- Surmeier et al. (2007): D1/D2 receptor physiology
- Hikida et al. (2010): Direct/indirect pathway functions
- Gerfen & Surmeier (2011): Modulation of striatal MSNs
- Aosaki et al. (1994): Cholinergic interneuron pause responses
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ActionState(Enum):
    """Action selection state."""
    UNDECIDED = "undecided"      # No clear winner
    GO = "go"                     # D1 pathway dominant
    NO_GO = "no_go"              # D2 pathway dominant
    COMPETING = "competing"       # Close competition


class TANState(Enum):
    """Tonically Active Neuron (cholinergic interneuron) state."""
    ACTIVE = "active"        # Normal tonic firing
    PAUSED = "paused"        # Paused during learning signal


@dataclass
class MSNConfig:
    """Configuration for D1/D2 MSN populations."""

    # Population sizes (relative)
    d1_population_size: int = 100
    d2_population_size: int = 100

    # D1 receptor parameters
    d1_affinity: float = 0.3         # DA concentration for half-max binding (K_d)
    d1_efficacy: float = 0.8         # Maximum D1 activation
    d1_hill: float = 1.5             # Hill coefficient

    # D2 receptor parameters
    d2_affinity: float = 0.1         # D2 has higher affinity than D1 (K_d)
    d2_efficacy: float = 0.7         # Maximum D2 inhibition
    d2_hill: float = 1.2             # Hill coefficient

    def __post_init__(self) -> None:
        """
        ATOM-P3-27: Validate Hill function K_d parameters.
        """
        if self.d1_affinity <= 0:
            raise ValueError("D1 K_d (d1_affinity) must be positive")
        if self.d2_affinity <= 0:
            raise ValueError("D2 K_d (d2_affinity) must be positive")

    # Baseline activities (without DA modulation)
    d1_baseline: float = 0.2
    d2_baseline: float = 0.3         # D2 slightly higher at rest

    # Time constants
    tau_d1: float = 0.05             # 50ms activation time constant
    tau_d2: float = 0.05             # 50ms activation time constant
    tau_da_binding: float = 0.02     # 20ms DA binding kinetics

    # Competition parameters
    lateral_inhibition: float = 0.3   # Mutual inhibition strength (base)
    # ATOM-P3-26: Asymmetric lateral inhibition strengths (Taverna 2008)
    d2_to_d1_strength: float = 1.3    # D2→D1 stronger (Taverna 2008)
    d1_to_d2_strength: float = 0.7    # D1→D2 weaker
    decision_threshold: float = 0.6   # Activity threshold for decision
    competition_gain: float = 2.0     # Softmax temperature for selection

    # Plasticity parameters (DA-modulated)
    d1_ltp_rate: float = 0.01        # D1 LTP when DA high
    d1_ltd_rate: float = 0.005       # D1 LTD when DA low
    d2_ltp_rate: float = 0.005       # D2 LTP when DA low
    d2_ltd_rate: float = 0.01        # D2 LTD when DA high

    # Habit formation
    habit_formation_rate: float = 0.001   # Slow shift toward habitual
    habit_threshold: float = 0.8          # D1 strength for habitual

    # Fix 2: TAN parameters (Aosaki et al. 1994)
    # ATOM-P4-2: TAN pause duration now configurable
    tan_pause_ms: float = 200.0           # Pause duration in milliseconds (default 200ms)
    tan_pause_threshold: float = 0.3      # Reward surprise threshold for pause
    tan_baseline_ach: float = 0.5         # Baseline ACh during tonic firing
    tan_pause_ach: float = 0.1            # ACh level during pause

    @property
    def tan_pause_duration(self) -> float:
        """TAN pause duration in seconds (computed from ms config)."""
        return self.tan_pause_ms / 1000.0


@dataclass
class TANPopulationState:
    """State of TAN (cholinergic interneuron) population."""

    state: TANState = TANState.ACTIVE
    ach_level: float = 0.5            # Current ACh output [0, 1]
    pause_remaining: float = 0.0      # Remaining pause duration (seconds)
    last_pause_trigger: float = 0.0   # Last RPE that triggered pause

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "ach_level": self.ach_level,
            "pause_remaining": self.pause_remaining,
            "last_pause_trigger": self.last_pause_trigger,
        }


@dataclass
class MSNPopulationState:
    """State of MSN populations."""

    # D1 (direct/GO) pathway
    d1_activity: float = 0.2          # Current D1 MSN activity [0, 1]
    d1_receptor_occupancy: float = 0.0  # D1 receptor bound fraction
    d1_synaptic_strength: float = 1.0   # Learned D1 pathway weight

    # D2 (indirect/NO-GO) pathway
    d2_activity: float = 0.3          # Current D2 MSN activity [0, 1]
    d2_receptor_occupancy: float = 0.0  # D2 receptor bound fraction
    d2_synaptic_strength: float = 1.0   # Learned D2 pathway weight

    # Action selection
    action_state: ActionState = ActionState.UNDECIDED
    go_probability: float = 0.5       # Softmax probability of GO
    competition_margin: float = 0.0   # D1 - D2 difference

    # Inputs
    cortical_input: float = 0.0       # Glutamatergic input from cortex
    dopamine_level: float = 0.5       # Current DA level
    ach_level: float = 0.5            # Current ACh level (from TANs)
    gaba_level: float = 0.5           # GABA level from neural field (modulates lateral inhibition)

    # Statistics
    habit_strength: float = 0.0       # Degree of habituation [0, 1]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "d1_activity": self.d1_activity,
            "d2_activity": self.d2_activity,
            "d1_receptor_occupancy": self.d1_receptor_occupancy,
            "d2_receptor_occupancy": self.d2_receptor_occupancy,
            "action_state": self.action_state.value,
            "go_probability": self.go_probability,
            "competition_margin": self.competition_margin,
            "habit_strength": self.habit_strength,
        }


class CholinergicInterneuron:
    """
    Tonically Active Neurons (TANs) - Cholinergic interneurons in striatum.

    Fix 2: Implements TAN pause mechanism (Aosaki et al. 1994):
    - TANs fire tonically (~5 Hz) releasing acetylcholine
    - Upon reward surprise (unexpected reward or RPE > threshold), TANs pause for ~200ms
    - This pause marks "when" reinforcement occurred
    - The pause timing is critical for credit assignment in learning

    Biological basis:
    - TANs comprise ~1-2% of striatal neurons but have extensive arborization
    - They receive inputs from midbrain DA neurons and thalamus
    - The pause is triggered by salient/unexpected events
    - During pause, ACh drops, allowing enhanced plasticity in MSNs

    References:
    - Aosaki et al. (1994): Temporal and spatial characteristics of tonically active neurons
    - Cragg (2006): Meaningful silences: how dopamine listens to the ACh pause
    - Morris et al. (2004): Coincident but distinct messages of midbrain DA and striatal TANs
    """

    def __init__(self, config: MSNConfig | None = None):
        """
        Initialize TAN population.

        Args:
            config: Configuration (uses MSN config for TAN parameters)
        """
        self.config = config or MSNConfig()
        self.state = TANPopulationState(ach_level=self.config.tan_baseline_ach)

        # History tracking
        self._pause_history: list[float] = []
        self._max_history = 1000

        logger.info(
            f"CholinergicInterneuron (TAN) initialized: "
            f"pause_duration={self.config.tan_pause_duration}s, "
            f"threshold={self.config.tan_pause_threshold}"
        )

    def process_reward_surprise(self, rpe: float, dt: float = 0.01) -> float:
        """
        Process reward prediction error and potentially trigger pause.

        Fix 2: Implements the TAN pause response to unexpected rewards.
        When RPE exceeds threshold, TANs pause for 200ms.

        Args:
            rpe: Reward prediction error (surprise magnitude)
            dt: Timestep in seconds

        Returns:
            Current ACh level after processing
        """
        # Check if surprise is large enough to trigger pause
        if abs(rpe) > self.config.tan_pause_threshold:
            # Check if not already paused (no double-triggering)
            if self.state.pause_remaining <= 0:
                self._trigger_pause(rpe)

        # Update pause state
        self._update_pause_state(dt)

        return self.state.ach_level

    def _trigger_pause(self, rpe: float) -> None:
        """
        Trigger TAN pause response.

        Args:
            rpe: RPE that triggered the pause
        """
        self.state.state = TANState.PAUSED
        self.state.pause_remaining = self.config.tan_pause_duration
        self.state.ach_level = self.config.tan_pause_ach
        self.state.last_pause_trigger = rpe

        self._pause_history.append(rpe)
        if len(self._pause_history) > self._max_history:
            self._pause_history = self._pause_history[-self._max_history:]

        logger.debug(
            f"TAN pause triggered: RPE={rpe:.3f}, "
            f"duration={self.config.tan_pause_duration}s"
        )

    def _update_pause_state(self, dt: float) -> None:
        """
        Update pause state over time.

        Args:
            dt: Timestep in seconds
        """
        if self.state.pause_remaining > 0:
            self.state.pause_remaining -= dt

            if self.state.pause_remaining <= 0:
                # Resume tonic firing
                self.state.state = TANState.ACTIVE
                self.state.ach_level = self.config.tan_baseline_ach
                self.state.pause_remaining = 0.0
                logger.debug("TAN resumed tonic firing")

    def get_ach_level(self) -> float:
        """
        Get current ACh output level.

        Returns:
            ACh concentration [0, 1]
        """
        return self.state.ach_level

    def is_paused(self) -> bool:
        """Check if TAN is currently paused."""
        return self.state.state == TANState.PAUSED

    def step(self, dt: float = 0.01) -> None:
        """
        Step TAN dynamics forward.

        Args:
            dt: Timestep in seconds
        """
        self._update_pause_state(dt)

    def get_stats(self) -> dict:
        """Get TAN statistics."""
        return {
            "state": self.state.state.value,
            "ach_level": self.state.ach_level,
            "pause_remaining": self.state.pause_remaining,
            "n_pauses": len(self._pause_history),
            "avg_pause_trigger": (
                float(np.mean([abs(x) for x in self._pause_history]))
                if self._pause_history else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset TAN to initial state."""
        self.state = TANPopulationState(ach_level=self.config.tan_baseline_ach)
        self._pause_history.clear()


class StriatalMSN:
    """
    D1/D2 Medium Spiny Neuron populations for action selection.

    Models the striatal GO/NO-GO decision making circuit:
    - D1 MSNs (direct pathway): Activated by DA → promotes action
    - D2 MSNs (indirect pathway): Inhibited by DA → suppresses action
    - Lateral inhibition creates winner-take-all dynamics
    - DA modulates plasticity for reinforcement learning
    - Fix 2: TANs provide temporal credit assignment via pause mechanism

    Key equations:
        D1_occupancy = DA^n / (K_d1^n + DA^n)    # Hill function binding
        D2_occupancy = DA^n / (K_d2^n + DA^n)

        D1_activity = baseline + efficacy * occupancy * cortical_input * strength
        D2_activity = baseline - efficacy * occupancy + cortical_input * strength

        GO = sigmoid(D1 - D2 - threshold)

    Usage:
        msn = StriatalMSN(config)
        msn.set_cortical_input(0.7)
        msn.set_dopamine_level(0.6)  # From VTA

        action = msn.step(dt=0.01)
        if msn.state.action_state == ActionState.GO:
            execute_action()
    """

    def __init__(self, config: MSNConfig | None = None):
        """
        Initialize MSN populations.

        Args:
            config: MSN configuration parameters
        """
        self.config = config or MSNConfig()
        self.state = MSNPopulationState()

        # Fix 2: Add TAN population
        self.tan = CholinergicInterneuron(self.config)

        # History for analysis
        self._d1_history: list[float] = []
        self._d2_history: list[float] = []
        self._decision_history: list[ActionState] = []
        self._max_history = 1000

        # Callbacks
        self._action_callbacks: list[Callable[[ActionState], None]] = []

        # Simulation time
        self._simulation_time = 0.0

        logger.info(
            f"StriatalMSN initialized: "
            f"D1 pop={self.config.d1_population_size}, "
            f"D2 pop={self.config.d2_population_size}, "
            f"TAN pause enabled"
        )

    # =========================================================================
    # Core Dynamics
    # =========================================================================

    def step(self, dt: float = 0.01) -> ActionState:
        """
        Advance MSN dynamics by one timestep.

        Args:
            dt: Timestep in seconds

        Returns:
            Current action state (GO/NO_GO/UNDECIDED/COMPETING)
        """
        # 0. Step TANs (updates ACh level)
        self.tan.step(dt)
        self.state.ach_level = self.tan.get_ach_level()

        # 1. Update receptor binding
        self._update_receptor_binding(dt)

        # 2. Update D1/D2 activities
        self._update_d1_activity(dt)
        self._update_d2_activity(dt)

        # 3. Apply lateral inhibition
        self._apply_lateral_inhibition(dt)

        # 4. Compute action selection
        self._compute_action_selection()

        # 5. Update plasticity (if action taken)
        if self.state.action_state in [ActionState.GO, ActionState.NO_GO]:
            self._update_plasticity(dt)

        # 6. Update habit strength
        self._update_habit_strength(dt)

        # Track history
        self._d1_history.append(self.state.d1_activity)
        self._d2_history.append(self.state.d2_activity)
        self._decision_history.append(self.state.action_state)
        if len(self._d1_history) > self._max_history:
            self._d1_history = self._d1_history[-self._max_history:]
            self._d2_history = self._d2_history[-self._max_history:]
            self._decision_history = self._decision_history[-self._max_history:]

        self._simulation_time += dt

        # Fire callbacks if action decided
        if self.state.action_state in [ActionState.GO, ActionState.NO_GO]:
            for callback in self._action_callbacks:
                callback(self.state.action_state)

        return self.state.action_state

    def _update_receptor_binding(self, dt: float) -> None:
        """
        Update D1/D2 receptor occupancy using Hill kinetics.

        D1 has lower affinity (activated only by high DA)
        D2 has higher affinity (activated by basal DA)
        """
        da = self.state.dopamine_level

        # D1 binding (lower affinity, higher DA needed)
        k_d1 = self.config.d1_affinity
        n_d1 = self.config.d1_hill
        if da > 0:
            d1_target = (da ** n_d1) / (k_d1 ** n_d1 + da ** n_d1)
        else:
            d1_target = 0.0

        # D2 binding (higher affinity, responds to lower DA)
        k_d2 = self.config.d2_affinity
        n_d2 = self.config.d2_hill
        if da > 0:
            d2_target = (da ** n_d2) / (k_d2 ** n_d2 + da ** n_d2)
        else:
            d2_target = 0.0

        # Smooth update (binding kinetics)
        alpha = dt / self.config.tau_da_binding
        self.state.d1_receptor_occupancy += alpha * (
            d1_target - self.state.d1_receptor_occupancy
        )
        self.state.d2_receptor_occupancy += alpha * (
            d2_target - self.state.d2_receptor_occupancy
        )

        # Clamp to [0, 1]
        self.state.d1_receptor_occupancy = float(np.clip(
            self.state.d1_receptor_occupancy, 0, 1
        ))
        self.state.d2_receptor_occupancy = float(np.clip(
            self.state.d2_receptor_occupancy, 0, 1
        ))

    def _update_d1_activity(self, dt: float) -> None:
        """
        Update D1 MSN (direct/GO pathway) activity.

        D1 MSNs are EXCITED by dopamine:
        - DA binding → Gs-coupled → cAMP increase → enhanced excitability
        - This promotes action execution

        Fix 2: TAN pause enhances D1 response (low ACh removes inhibition)
        """
        # Base cortical drive (glutamatergic)
        cortical_drive = self.state.cortical_input * self.state.d1_synaptic_strength

        # D1 receptor effect: DA EXCITES D1 MSNs
        d1_da_effect = (
            self.config.d1_efficacy *
            self.state.d1_receptor_occupancy
        )

        # Fix 2: TAN pause effect - low ACh enhances D1 plasticity
        # When TANs pause (ACh drops), D1 MSNs become more responsive
        ach_modulation = 1.0 - (self.state.ach_level - self.config.tan_pause_ach) / (
            self.config.tan_baseline_ach - self.config.tan_pause_ach
        )
        ach_modulation = np.clip(ach_modulation, 0, 1) * 0.3  # Max 30% boost

        # Target activity
        target = (
            self.config.d1_baseline +
            cortical_drive * (1 + d1_da_effect + ach_modulation)
        )
        target = np.clip(target, 0, 1)

        # Smooth update
        alpha = dt / self.config.tau_d1
        self.state.d1_activity += alpha * (target - self.state.d1_activity)
        self.state.d1_activity = float(np.clip(self.state.d1_activity, 0, 1))

    def _update_d2_activity(self, dt: float) -> None:
        """
        Update D2 MSN (indirect/NO-GO pathway) activity.

        D2 MSNs are INHIBITED by dopamine:
        - DA binding → Gi-coupled → cAMP decrease → reduced excitability
        - This suppresses action inhibition (disinhibition)

        Fix 2: High ACh (tonic TANs) enhances D2
        """
        # Base cortical drive (glutamatergic)
        cortical_drive = self.state.cortical_input * self.state.d2_synaptic_strength

        # D2 receptor effect: DA INHIBITS D2 MSNs
        d2_da_effect = (
            self.config.d2_efficacy *
            self.state.d2_receptor_occupancy
        )

        # Fix 2: High ACh (tonic TANs) enhances D2 activity
        # When TANs fire tonically (high ACh), D2 pathway is strengthened
        ach_modulation = (self.state.ach_level - self.config.tan_pause_ach) / (
            self.config.tan_baseline_ach - self.config.tan_pause_ach
        )
        ach_modulation = np.clip(ach_modulation, 0, 1) * 0.2  # Max 20% boost

        # Target activity (DA inhibits, so subtract)
        target = (
            self.config.d2_baseline +
            cortical_drive * (1 - d2_da_effect + ach_modulation)
        )
        target = np.clip(target, 0, 1)

        # Smooth update
        alpha = dt / self.config.tau_d2
        self.state.d2_activity += alpha * (target - self.state.d2_activity)
        self.state.d2_activity = float(np.clip(self.state.d2_activity, 0, 1))

    def _apply_lateral_inhibition(self, dt: float) -> None:
        """
        Apply GABA-mediated mutual inhibition between D1 and D2 populations.

        Creates winner-take-all dynamics for action selection.

        Biological basis: Striatal spiny stellate GABA interneurons mediate
        lateral inhibition. The inhibition strength is modulated by the
        local GABA concentration from the neural field.

        P1-3 Fix: Lateral inhibition is now GABA-mediated rather than
        a fixed parameter, making it responsive to neuromodulator dynamics.

        ATOM-P3-26: Asymmetric inhibition strengths (D2→D1 stronger than D1→D2).
        """
        # GABA modulates the effective inhibition strength
        # Higher GABA → stronger lateral inhibition (winner-take-all sharpening)
        # Lower GABA → weaker inhibition (more parallel activation possible)
        gaba = self.state.gaba_level
        gaba_efficacy = 0.5 + gaba  # Range [0.5, 1.5] for GABA in [0, 1]

        base_inhibition = self.config.lateral_inhibition * gaba_efficacy

        # ATOM-P3-26: Asymmetric lateral inhibition (Taverna 2008)
        # D2→D1 is stronger than D1→D2
        d1_inhibition = base_inhibition * self.config.d2_to_d1_strength * self.state.d2_activity
        d2_inhibition = base_inhibition * self.config.d1_to_d2_strength * self.state.d1_activity

        # Apply inhibition (fast GABA kinetics, ~10ms)
        self.state.d1_activity -= d1_inhibition * dt * 10
        self.state.d2_activity -= d2_inhibition * dt * 10

        # Clamp
        self.state.d1_activity = float(np.clip(self.state.d1_activity, 0, 1))
        self.state.d2_activity = float(np.clip(self.state.d2_activity, 0, 1))

    def _compute_action_selection(self) -> None:
        """
        Compute action selection from D1/D2 competition.

        Uses softmax-like selection with threshold.
        """
        d1 = self.state.d1_activity
        d2 = self.state.d2_activity

        # Competition margin
        margin = d1 - d2
        self.state.competition_margin = margin

        # Softmax probability of GO
        beta = self.config.competition_gain
        exp_d1 = np.exp(beta * d1)
        exp_d2 = np.exp(beta * d2)
        self.state.go_probability = exp_d1 / (exp_d1 + exp_d2)

        # Threshold-based decision
        threshold = self.config.decision_threshold

        if d1 > threshold and d1 > d2 * 1.2:  # D1 dominant by 20%
            self.state.action_state = ActionState.GO
        elif d2 > threshold and d2 > d1 * 1.2:  # D2 dominant by 20%
            self.state.action_state = ActionState.NO_GO
        elif abs(margin) < 0.1:  # Close competition
            self.state.action_state = ActionState.COMPETING
        else:
            self.state.action_state = ActionState.UNDECIDED

    def _update_plasticity(self, dt: float) -> None:
        """
        Update synaptic strengths based on DA-modulated plasticity.

        High DA → D1 LTP, D2 LTD (reinforces GO)
        Low DA → D1 LTD, D2 LTP (reinforces NO-GO)
        """
        da = self.state.dopamine_level

        # D1 plasticity
        if da > 0.6:  # High DA → D1 LTP
            d1_delta = self.config.d1_ltp_rate * (da - 0.5) * dt
        elif da < 0.4:  # Low DA → D1 LTD
            d1_delta = -self.config.d1_ltd_rate * (0.5 - da) * dt
        else:
            d1_delta = 0.0

        # D2 plasticity (opposite of D1)
        if da < 0.4:  # Low DA → D2 LTP
            d2_delta = self.config.d2_ltp_rate * (0.5 - da) * dt
        elif da > 0.6:  # High DA → D2 LTD
            d2_delta = -self.config.d2_ltd_rate * (da - 0.5) * dt
        else:
            d2_delta = 0.0

        # Update strengths
        self.state.d1_synaptic_strength += d1_delta
        self.state.d2_synaptic_strength += d2_delta

        # Clamp to reasonable bounds
        self.state.d1_synaptic_strength = float(np.clip(
            self.state.d1_synaptic_strength, 0.1, 3.0
        ))
        self.state.d2_synaptic_strength = float(np.clip(
            self.state.d2_synaptic_strength, 0.1, 3.0
        ))

    def _update_habit_strength(self, dt: float) -> None:
        """
        Update habit strength (transition from goal-directed to habitual).

        Habits form when D1 pathway becomes dominant through practice.
        ATOM-P4-3: Use sigmoid saturation curve instead of linear accumulation.
        """
        # Compute raw strength (linear accumulation component)
        if self.state.d1_synaptic_strength > self.config.habit_threshold:
            raw_delta = self.config.habit_formation_rate * dt
        else:
            # Decay habit when D1 is weak
            raw_delta = -self.config.habit_formation_rate * 0.1 * dt

        # ATOM-P4-3: Apply sigmoid saturation instead of linear clipping
        # strength = max_strength / (1 + exp(-raw_strength / scale))
        # This creates smoother saturation at high habit strengths
        raw_strength = self.state.habit_strength + raw_delta
        max_strength = 1.0
        scale = 0.2  # Saturation sharpness
        self.state.habit_strength = float(
            max_strength / (1.0 + np.exp(-raw_strength / scale))
        )

    # =========================================================================
    # External Inputs
    # =========================================================================

    def set_cortical_input(self, input_level: float) -> None:
        """
        Set cortical glutamatergic input to striatum.

        Args:
            input_level: Cortical activity [0, 1]
        """
        self.state.cortical_input = float(np.clip(input_level, 0, 1))

    def set_dopamine_level(self, da_level: float) -> None:
        """
        Set dopamine level (from VTA/SNc).

        Args:
            da_level: Dopamine concentration [0, 1]
        """
        self.state.dopamine_level = float(np.clip(da_level, 0, 1))

    def set_ach_level(self, ach_level: float) -> None:
        """
        Set acetylcholine level (from TANs).

        Note: This is overridden by internal TAN dynamics in step().
        Use process_reward_surprise() to trigger TAN pause instead.

        Args:
            ach_level: ACh concentration [0, 1]
        """
        self.state.ach_level = float(np.clip(ach_level, 0, 1))

    def set_gaba_level(self, gaba_level: float) -> None:
        """
        Set GABA level from neural field.

        GABA modulates lateral inhibition strength between D1/D2 populations.
        Higher GABA → stronger winner-take-all competition.
        Lower GABA → more parallel pathway activation.

        P1-3: Connects striatal inhibition to neural field GABA dynamics.

        Args:
            gaba_level: GABA concentration [0, 1]
        """
        self.state.gaba_level = float(np.clip(gaba_level, 0, 1))

    def apply_rpe(self, rpe: float, dt: float = 0.01) -> None:
        """
        Apply reward prediction error for rapid plasticity.

        Fix 2: Also triggers TAN pause if RPE is large enough.

        Positive RPE → strengthen GO pathway, trigger TAN pause
        Negative RPE → strengthen NO-GO pathway, trigger TAN pause

        Args:
            rpe: Reward prediction error [-1, 1]
            dt: Timestep for TAN dynamics
        """
        # Fix 2: Process RPE through TANs (may trigger pause)
        self.tan.process_reward_surprise(rpe, dt)

        # Update MSN synaptic strengths
        if rpe > 0:
            # Positive surprise → strengthen D1
            self.state.d1_synaptic_strength += 0.05 * rpe
            self.state.d2_synaptic_strength -= 0.02 * rpe
        else:
            # Negative surprise → strengthen D2
            self.state.d2_synaptic_strength += 0.05 * abs(rpe)
            self.state.d1_synaptic_strength -= 0.02 * abs(rpe)

        # Clamp
        self.state.d1_synaptic_strength = float(np.clip(
            self.state.d1_synaptic_strength, 0.1, 3.0
        ))
        self.state.d2_synaptic_strength = float(np.clip(
            self.state.d2_synaptic_strength, 0.1, 3.0
        ))

    # =========================================================================
    # Callbacks and Integration
    # =========================================================================

    def register_action_callback(
        self,
        callback: Callable[[ActionState], None]
    ) -> None:
        """Register callback for action decisions."""
        self._action_callbacks.append(callback)

    def get_go_signal(self) -> float:
        """
        Get GO signal strength for downstream processing.

        Returns:
            GO signal [0, 1], high = execute action
        """
        if self.state.action_state == ActionState.GO:
            return self.state.d1_activity
        elif self.state.action_state == ActionState.NO_GO:
            return 0.0
        else:
            return self.state.go_probability * self.state.d1_activity

    def get_no_go_signal(self) -> float:
        """
        Get NO-GO signal strength for downstream processing.

        Returns:
            NO-GO signal [0, 1], high = inhibit action
        """
        if self.state.action_state == ActionState.NO_GO:
            return self.state.d2_activity
        elif self.state.action_state == ActionState.GO:
            return 0.0
        else:
            return (1 - self.state.go_probability) * self.state.d2_activity

    def is_habitual(self) -> bool:
        """Check if behavior has become habitual."""
        return self.state.habit_strength > 0.8

    def get_action_values(self) -> tuple[float, float]:
        """
        Get action values (Q-values) from pathway strengths.

        Returns:
            (Q_go, Q_no_go) tuple
        """
        q_go = self.state.d1_activity * self.state.d1_synaptic_strength
        q_no_go = self.state.d2_activity * self.state.d2_synaptic_strength
        return (q_go, q_no_go)

    # =========================================================================
    # Statistics and State
    # =========================================================================

    def get_stats(self) -> dict:
        """Get MSN population statistics."""
        stats = {
            "d1_activity": self.state.d1_activity,
            "d2_activity": self.state.d2_activity,
            "d1_receptor_occupancy": self.state.d1_receptor_occupancy,
            "d2_receptor_occupancy": self.state.d2_receptor_occupancy,
            "d1_synaptic_strength": self.state.d1_synaptic_strength,
            "d2_synaptic_strength": self.state.d2_synaptic_strength,
            "action_state": self.state.action_state.value,
            "go_probability": self.state.go_probability,
            "competition_margin": self.state.competition_margin,
            "habit_strength": self.state.habit_strength,
            "dopamine_level": self.state.dopamine_level,
            "cortical_input": self.state.cortical_input,
            "simulation_time": self._simulation_time,
        }

        # Fix 2: Add TAN statistics
        stats.update(self.tan.get_stats())

        if self._d1_history:
            stats["mean_d1"] = float(np.mean(self._d1_history))
            stats["mean_d2"] = float(np.mean(self._d2_history))
            stats["go_ratio"] = sum(
                1 for s in self._decision_history if s == ActionState.GO
            ) / len(self._decision_history)

        return stats

    def reset(self) -> None:
        """Reset MSN populations to initial state."""
        self.state = MSNPopulationState()
        self.tan.reset()
        self._d1_history.clear()
        self._d2_history.clear()
        self._decision_history.clear()
        self._simulation_time = 0.0
        logger.info("StriatalMSN reset")

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "state": self.state.to_dict(),
            "tan_state": self.tan.state.to_dict(),
            "d1_synaptic_strength": self.state.d1_synaptic_strength,
            "d2_synaptic_strength": self.state.d2_synaptic_strength,
            "habit_strength": self.state.habit_strength,
        }

    def load_state(self, saved: dict) -> None:
        """Load state from persistence."""
        if "state" in saved:
            s = saved["state"]
            self.state.d1_activity = s.get("d1_activity", 0.2)
            self.state.d2_activity = s.get("d2_activity", 0.3)
        if "d1_synaptic_strength" in saved:
            self.state.d1_synaptic_strength = saved["d1_synaptic_strength"]
        if "d2_synaptic_strength" in saved:
            self.state.d2_synaptic_strength = saved["d2_synaptic_strength"]
        if "habit_strength" in saved:
            self.state.habit_strength = saved["habit_strength"]
        if "tan_state" in saved:
            ts = saved["tan_state"]
            self.tan.state.ach_level = ts.get("ach_level", 0.5)
            self.tan.state.pause_remaining = ts.get("pause_remaining", 0.0)


def create_striatal_msn(
    d1_baseline: float = 0.2,
    d2_baseline: float = 0.3,
    lateral_inhibition: float = 0.3,
) -> StriatalMSN:
    """
    Factory function to create striatal MSN populations.

    Args:
        d1_baseline: D1 MSN baseline activity
        d2_baseline: D2 MSN baseline activity
        lateral_inhibition: Mutual inhibition strength

    Returns:
        Configured StriatalMSN
    """
    config = MSNConfig(
        d1_baseline=d1_baseline,
        d2_baseline=d2_baseline,
        lateral_inhibition=lateral_inhibition,
    )
    return StriatalMSN(config)


__all__ = [
    "StriatalMSN",
    "MSNConfig",
    "MSNPopulationState",
    "ActionState",
    "CholinergicInterneuron",
    "TANState",
    "TANPopulationState",
    "create_striatal_msn",
]
