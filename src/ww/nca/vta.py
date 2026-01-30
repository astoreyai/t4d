"""
Ventral Tegmental Area (VTA) Dopamine Circuit for World Weaver.

Biological Basis:
- VTA contains dopamine neurons projecting to limbic (NAcc) and cortical (PFC) areas
- Dopamine neurons encode Reward Prediction Error (RPE): firing = reward - expectation
- Two firing modes:
  - Tonic: baseline 4-5 Hz, sets DA tone for motivation/exploration
  - Phasic: bursts (20-40 Hz) for positive RPE, pauses for negative RPE

Architecture:
- Receives reward signals and expectation estimates
- Computes RPE using temporal difference-like computation
- Produces phasic DA bursts that modulate learning
- Connects to NeuralFieldSolver DA channel

Integration:
- NeuralFieldSolver: inject_rpe() modifies DA concentration
- HippocampalCircuit: novelty signals can trigger DA responses
- LearnableCoupling: RPE modulates coupling plasticity via eligibility
- P7.4: Direct coupling updates from RPE signals

References:
- Grace & Bunney (1984): Tonic and phasic firing patterns
- Bayer & Glimcher (2005): RPE-to-DA conversion gain
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ww.nca.coupling import LearnableCoupling
    from ww.nca.neural_field import NeurotransmitterState
    from ww.nca.raphe import RapheNucleus

logger = logging.getLogger(__name__)


class VTAFiringMode(Enum):
    """VTA dopamine neuron firing modes."""
    TONIC = "tonic"         # Baseline sustained firing (4-5 Hz)
    PHASIC_BURST = "burst"  # Reward > expectation (20-40 Hz)
    PHASIC_PAUSE = "pause"  # Reward < expectation (0-2 Hz)


@dataclass(frozen=True)
class VTAConfig:
    """Configuration for VTA circuit dynamics."""

    # Tonic firing parameters
    tonic_rate: float = 4.5        # Hz, baseline firing rate
    tonic_da_level: float = 0.3    # Baseline DA concentration [0, 1]

    # Phasic dynamics
    burst_peak_rate: float = 30.0  # Hz, during positive RPE
    burst_duration: float = 0.2    # seconds
    pause_duration: float = 0.3    # seconds, during negative RPE

    # RPE -> DA conversion
    rpe_to_da_gain: float = 0.5    # How much 1.0 RPE changes DA

    # Fix 1: Exponential decay (Grace & Bunney 1984)
    tau_decay: float = 0.2         # 200ms time constant for exponential decay

    # Temporal difference parameters
    discount_gamma: float = 0.95   # Future reward discounting
    td_lambda: float = 0.9         # Eligibility trace decay (TD(λ))

    # Value learning
    value_learning_rate: float = 0.1
    prediction_horizon: int = 10   # Steps ahead for value estimation

    # Biological constraints
    min_da: float = 0.05           # Minimum DA (even during pause)
    max_da: float = 0.95           # Maximum DA (saturation)
    refractory_period: float = 0.1  # Min time between phasic events


@dataclass
class VTAState:
    """Current state of VTA circuit."""

    firing_mode: VTAFiringMode = VTAFiringMode.TONIC
    current_da: float = 0.3        # Current DA concentration
    current_rate: float = 4.5      # Current firing rate (Hz)

    # RPE tracking
    last_rpe: float = 0.0
    cumulative_rpe: float = 0.0    # Running RPE for eligibility

    # PFC modulation (dlPFC modulates tonic DA, stored in state not config)
    tonic_modulation: float = 0.0  # Additive modulation to tonic_da_level

    # Value function state
    value_estimate: float = 0.5    # V(s) - expected future reward
    td_error: float = 0.0          # δ = r + γV(s') - V(s)

    # Timing
    time_since_phasic: float = 1.0  # Seconds since last burst/pause
    phasic_remaining: float = 0.0   # Remaining phasic event duration

    # Eligibility trace for TD(λ)
    eligibility: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "firing_mode": self.firing_mode.value,
            "current_da": self.current_da,
            "current_rate": self.current_rate,
            "last_rpe": self.last_rpe,
            "value_estimate": self.value_estimate,
            "td_error": self.td_error,
        }


class VTACircuit:
    """
    Ventral Tegmental Area dopamine circuit.

    Models the dynamics of VTA dopamine neurons that:
    1. Maintain tonic DA levels for baseline motivation
    2. Generate phasic bursts for positive prediction errors
    3. Generate phasic pauses for negative prediction errors
    4. Connect to downstream targets (NAcc, PFC, neural field)

    Key innovation: Integrates TD learning with neural field dynamics,
    providing biologically plausible RPE computation.
    """

    def __init__(
        self,
        config: VTAConfig | None = None,
        coupling: LearnableCoupling | None = None,
        enable_coupling_updates: bool = True,
    ):
        """
        Initialize VTA circuit.

        Args:
            config: VTA configuration parameters
            coupling: P7.4: LearnableCoupling for RPE-driven updates
            enable_coupling_updates: P7.4: Enable coupling matrix updates
        """
        self.config = config or VTAConfig()
        self.state = VTAState(
            current_da=self.config.tonic_da_level,
            current_rate=self.config.tonic_rate
        )

        # P7.4: Coupling for RPE-based updates
        self._coupling = coupling
        self._enable_coupling_updates = enable_coupling_updates
        self._current_nt_state: NeurotransmitterState | None = None

        # Value function for states (simple tabular for now)
        self._value_table: dict[str, float] = {}

        # Trace history for TD(λ)
        self._trace_history: list[float] = []
        self._reward_history: list[float] = []
        self._max_history = 1000

        # Callbacks for downstream integration
        self._da_callbacks: list[Callable[[float, float], None]] = []

        # P2.1 Autowiring: Reference to raphe nucleus for automatic modulation
        self._raphe: RapheNucleus | None = None

        # ATOM-P3-9: DA homeostatic regulation
        self._da_history: list[float] = []
        self._homeostatic_target = 0.3

        logger.info(
            f"VTACircuit initialized: tonic_da={self.config.tonic_da_level}, "
            f"rpe_gain={self.config.rpe_to_da_gain}, "
            f"tau_decay={self.config.tau_decay}s, "
            f"coupling_updates={enable_coupling_updates}"
        )

    def set_coupling(self, coupling: LearnableCoupling) -> None:
        """
        P7.4: Set coupling matrix for RPE-driven updates.

        Args:
            coupling: LearnableCoupling instance to update during RPE
        """
        self._coupling = coupling
        logger.debug("P7.4: VTACircuit coupling connected")

    def set_raphe(self, raphe: RapheNucleus) -> None:
        """
        P2.1: Set raphe nucleus for automatic serotonin modulation.

        After calling this, step() will automatically receive 5-HT inhibition
        from the raphe on each timestep.

        Args:
            raphe: RapheNucleus instance to wire for bidirectional modulation
        """
        self._raphe = raphe
        logger.debug("P2.1: VTACircuit raphe autowiring enabled")

    def set_nt_state(self, nt_state: NeurotransmitterState) -> None:
        """
        P7.4: Set current NT state for coupling context.

        Args:
            nt_state: Current 6-NT state from NeuralFieldSolver
        """
        self._current_nt_state = nt_state

    # =========================================================================
    # Core RPE Computation
    # =========================================================================

    def compute_td_error(
        self,
        reward: float,
        current_state: str,
        next_state: str | None = None,
        terminal: bool = False
    ) -> float:
        """
        Compute temporal difference error.

        δ = r + γV(s') - V(s)

        This is the dopamine signal: positive δ = reward > expected.

        Args:
            reward: Immediate reward [0, 1]
            current_state: State identifier (for value lookup)
            next_state: Next state (if known)
            terminal: Whether episode ended

        Returns:
            TD error (RPE)
        """
        # Get value estimates
        v_current = self._get_value(current_state)

        if terminal:
            v_next = 0.0
        elif next_state is not None:
            v_next = self._get_value(next_state)
        else:
            # Bootstrap from current value
            v_next = v_current

        # TD error: δ = r + γV(s') - V(s)
        td_error = reward + self.config.discount_gamma * v_next - v_current

        # Update eligibility trace
        self._update_eligibility(td_error)

        # Store for state update
        self.state.td_error = td_error
        self.state.last_rpe = td_error

        return td_error

    def compute_rpe_from_outcome(
        self,
        actual_outcome: float,
        expected_outcome: float
    ) -> float:
        """
        Compute simple RPE from outcome comparison.

        δ = actual - expected

        Args:
            actual_outcome: What happened [0, 1]
            expected_outcome: What was predicted [0, 1]

        Returns:
            RPE
        """
        rpe = actual_outcome - expected_outcome
        self.state.last_rpe = rpe
        self._update_eligibility(rpe)
        return rpe

    def _update_eligibility(self, td_error: float) -> None:
        """
        Update eligibility trace for TD(λ).

        e(t) = λ * γ * e(t-1) + 1

        Traces mark "what led to this" for credit assignment.
        """
        # Decay existing trace
        self.state.eligibility *= (
            self.config.td_lambda * self.config.discount_gamma
        )
        # Add current contribution
        self.state.eligibility += 1.0
        # Clip for stability
        self.state.eligibility = min(self.state.eligibility, 10.0)

        # Cumulative RPE for downstream use
        self.state.cumulative_rpe = (
            0.9 * self.state.cumulative_rpe + 0.1 * td_error
        )

    def _get_value(self, state: str) -> float:
        """Get value estimate for state."""
        return self._value_table.get(state, 0.5)

    # ATOM-P2-5: Value table accessor methods
    def update_value_estimate(self, key: str, value: float) -> None:
        """
        Update value estimate for a state key.

        Args:
            key: State identifier
            value: Value estimate [0, 1]
        """
        self._value_table[key] = float(np.clip(value, 0.0, 1.0))

    def get_value_estimate(self, key: str, default: float = 0.5) -> float:
        """
        Get value estimate for a state key.

        Args:
            key: State identifier
            default: Default value if key not found

        Returns:
            Value estimate [0, 1]
        """
        return self._value_table.get(key, default)

    def update_value(
        self,
        state: str,
        td_error: float
    ) -> float:
        """
        Update value estimate based on TD error.

        V(s) ← V(s) + α * δ * e(s)

        Args:
            state: State to update
            td_error: Computed TD error

        Returns:
            New value estimate
        """
        current = self._get_value(state)

        # TD update with eligibility
        update = (
            self.config.value_learning_rate *
            td_error *
            self.state.eligibility
        )
        new_value = np.clip(current + update, 0.0, 1.0)

        self._value_table[state] = new_value
        self.state.value_estimate = new_value

        # Limit table size
        if len(self._value_table) > 10000:
            # Remove oldest entries (simple FIFO approximation)
            keys = list(self._value_table.keys())
            for k in keys[:1000]:
                del self._value_table[k]

        return new_value

    # =========================================================================
    # Dopamine Dynamics
    # =========================================================================

    def process_rpe(self, rpe: float, dt: float = 0.1) -> float:
        """
        Process RPE to generate dopamine response.

        Converts prediction error to DA concentration change.

        Args:
            rpe: Reward prediction error [-1, 1]
            dt: Timestep (seconds)

        Returns:
            New DA concentration
        """
        # ATOM-P2-1: Input validation at NCA layer boundary
        from ww.core.validation import ValidationError
        if isinstance(rpe, np.ndarray) and not np.all(np.isfinite(rpe)):
            raise ValidationError("rpe", "Contains NaN or Inf values")
        if not np.isfinite(rpe):
            raise ValidationError("rpe", "Contains NaN or Inf values")
        if not np.isfinite(dt):
            raise ValidationError("dt", "Contains NaN or Inf values")

        # Store RPE
        self.state.last_rpe = rpe
        self._reward_history.append(rpe)
        if len(self._reward_history) > self._max_history:
            self._reward_history = self._reward_history[-self._max_history:]

        # Determine firing mode based on RPE
        if abs(rpe) < 0.05:
            # Near-zero RPE: maintain tonic
            self._to_tonic_mode(dt)
        elif rpe > 0.05:
            # Positive RPE: burst
            self._to_burst_mode(rpe, dt)
        else:
            # Negative RPE: pause
            self._to_pause_mode(rpe, dt)

        # Notify callbacks
        for callback in self._da_callbacks:
            callback(self.state.current_da, rpe)

        # P7.4: Update coupling matrix based on RPE
        if self._enable_coupling_updates and self._coupling is not None:
            try:
                # Get current NT state for coupling context
                if self._current_nt_state is not None:
                    nt_array = self._current_nt_state.to_array()
                else:
                    # Use default baseline state
                    nt_array = np.array([
                        self.state.current_da,  # DA from current state
                        0.5,  # 5-HT baseline
                        0.5,  # ACh baseline
                        0.5,  # NE baseline
                        0.5,  # GABA baseline
                        0.5,  # Glu baseline
                    ], dtype=np.float32)

                # Update coupling: positive RPE strengthens, negative weakens
                # Use eligibility trace for temporal credit assignment
                # Note: update_from_rpe expects (nt_state, rpe, eligibility)
                # eligibility must be a 6x6 matrix or None (uses internal)
                self._coupling.update_from_rpe(
                    nt_state=nt_array,
                    rpe=rpe,
                    eligibility=None,  # Use coupling's internal eligibility trace
                )
                logger.debug(
                    f"P7.4: Coupling updated via RPE={rpe:.3f}, "
                    f"eligibility={self.state.eligibility:.3f}"
                )
            except Exception as e:
                logger.warning(f"P7.4: Coupling RPE update failed: {e}")

        return self.state.current_da

    def _to_tonic_mode(self, dt: float) -> None:
        """
        Transition to/maintain tonic firing.

        Fix 1: Uses exponential decay (Grace & Bunney 1984) instead of linear decay.
        DA level decays exponentially to baseline with tau_decay time constant.
        """
        self.state.firing_mode = VTAFiringMode.TONIC

        # ATOM-P2-8: D2 autoreceptor feedback (Bhatt et al. 1998)
        # High DA suppresses VTA firing rate via D2 autoreceptors
        d2_inhibition = self.state.current_da ** 2  # Hill-like function
        effective_rate = self.config.tonic_rate * (1.0 - 0.3 * d2_inhibition)
        self.state.current_rate = effective_rate

        # Fix 1: Exponential decay (Grace & Bunney 1984)
        # da_level = da_target + (da_level - da_target) * exp(-dt / tau_decay)
        # Effective tonic DA includes PFC modulation from state
        effective_tonic = self.config.tonic_da_level + self.state.tonic_modulation
        da_level = self.state.current_da
        self.state.current_da = effective_tonic + (da_level - effective_tonic) * np.exp(-dt / self.config.tau_decay)

        # Update timing
        self.state.time_since_phasic += dt
        self.state.phasic_remaining = 0.0

    def _to_burst_mode(self, rpe: float, dt: float) -> None:
        """Generate phasic burst for positive RPE."""
        # Check refractory period
        if self.state.time_since_phasic < self.config.refractory_period:
            self._to_tonic_mode(dt)
            return

        self.state.firing_mode = VTAFiringMode.PHASIC_BURST

        # Scale firing rate with RPE magnitude
        rate_scale = min(rpe / 0.5, 1.0)  # Saturate at 0.5 RPE
        self.state.current_rate = (
            self.config.tonic_rate +
            (self.config.burst_peak_rate - self.config.tonic_rate) * rate_scale
        )

        # Increase DA proportional to RPE
        da_increase = rpe * self.config.rpe_to_da_gain
        self.state.current_da = np.clip(
            self.state.current_da + da_increase,
            self.config.min_da,
            self.config.max_da
        )

        # Set phasic timing
        self.state.time_since_phasic = 0.0
        self.state.phasic_remaining = self.config.burst_duration

    def _to_pause_mode(self, rpe: float, dt: float) -> None:
        """Generate phasic pause for negative RPE."""
        # Check refractory period
        if self.state.time_since_phasic < self.config.refractory_period:
            self._to_tonic_mode(dt)
            return

        self.state.firing_mode = VTAFiringMode.PHASIC_PAUSE

        # Reduce firing rate proportional to negative RPE
        rate_scale = min(abs(rpe) / 0.5, 1.0)
        self.state.current_rate = self.config.tonic_rate * (1.0 - 0.8 * rate_scale)

        # Decrease DA proportional to RPE magnitude
        da_decrease = abs(rpe) * self.config.rpe_to_da_gain
        self.state.current_da = np.clip(
            self.state.current_da - da_decrease,
            self.config.min_da,
            self.config.max_da
        )

        # Set phasic timing
        self.state.time_since_phasic = 0.0
        self.state.phasic_remaining = self.config.pause_duration

    def step(self, dt: float = 0.1) -> None:
        """
        Advance VTA dynamics by one timestep without new RPE.

        Handles decay back to tonic state.
        P2.1: Auto-calls raphe inhibition if wired.
        ATOM-P3-9: Homeostatic regulation of DA levels.

        Args:
            dt: Timestep in seconds
        """
        self.state.time_since_phasic += dt

        # ATOM-P3-9: DA homeostatic mechanism
        self._da_history.append(self.state.current_da)
        if len(self._da_history) > 1000:
            self._da_history = self._da_history[-1000:]
            mean_da = np.mean(self._da_history)
            if mean_da > self._homeostatic_target * 1.2:
                # Reduce tonic rate to bring DA back to target
                self.state.current_rate *= 0.995

        # P2.1: Auto-receive serotonin inhibition if raphe is wired
        if self._raphe is not None:
            inhibition = self._raphe.get_vta_inhibition()
            self.receive_serotonin_inhibition(inhibition, apply_to_da=True)

        # Decay phasic effects
        if self.state.phasic_remaining > 0:
            self.state.phasic_remaining -= dt
            if self.state.phasic_remaining <= 0:
                self._to_tonic_mode(dt)
        else:
            # Passive decay to tonic
            self._to_tonic_mode(dt)

    # =========================================================================
    # Integration Methods
    # =========================================================================

    def register_da_callback(
        self,
        callback: Callable[[float, float], None]
    ) -> None:
        """
        Register callback for DA changes.

        Callback receives (da_level, rpe) when DA changes.

        Args:
            callback: Function to call on DA change

        Raises:
            ValueError: If callback limit (100) exceeded
        """
        # ATOM-P3-8: Callback registration limit
        if len(self._da_callbacks) >= 100:
            raise ValueError("Max 100 callbacks")
        self._da_callbacks.append(callback)

    def get_da_for_neural_field(self) -> float:
        """
        Get DA level for injection into NeuralFieldSolver.

        Returns:
            DA concentration [0, 1]
        """
        return self.state.current_da

    def get_rpe_for_coupling(self) -> tuple[float, float]:
        """
        Get RPE and eligibility for LearnableCoupling update.

        Returns:
            (rpe, eligibility) tuple
        """
        return self.state.last_rpe, self.state.eligibility

    def connect_to_hippocampus(
        self,
        novelty_signal: float,
        novelty_weight: float = 0.3
    ) -> float:
        """
        Integrate hippocampal novelty signal.

        Novelty can trigger DA responses even without explicit reward.

        Args:
            novelty_signal: Hippocampal novelty [0, 1]
            novelty_weight: How much novelty drives DA

        Returns:
            Modified RPE incorporating novelty
        """
        # Novelty contributes to RPE (novel = unexpected = positive RPE)
        novelty_rpe = (novelty_signal - 0.5) * novelty_weight

        # Combine with existing RPE
        combined_rpe = self.state.last_rpe + novelty_rpe

        return combined_rpe

    # =========================================================================
    # P2.1: Raphe → VTA Inhibition
    # =========================================================================

    def receive_serotonin_inhibition(
        self,
        inhibition: float,
        apply_to_da: bool = True
    ) -> float:
        """
        Receive inhibitory signal from raphe nucleus.

        Biological basis:
        - 5-HT2C receptors on VTA DA neurons mediate inhibition
        - High serotonin reduces dopamine signaling
        - This creates opponent process: 5-HT dampens reward pursuit

        Args:
            inhibition: Inhibition strength from raphe [0, 1]
            apply_to_da: Whether to immediately apply to DA level

        Returns:
            Resulting DA level after inhibition
        """
        # Scale inhibition effect
        da_reduction = inhibition * 0.3  # Max 30% reduction from 5-HT

        if apply_to_da:
            self.state.current_da = np.clip(
                self.state.current_da - da_reduction,
                self.config.min_da,
                self.config.max_da
            )

        return self.state.current_da

    def connect_to_raphe(
        self,
        raphe: RapheNucleus,
        bidirectional: bool = True
    ) -> None:
        """
        Establish bidirectional connection with raphe nucleus.

        Args:
            raphe: RapheNucleus instance
            bidirectional: Also send RPE to raphe
        """
        # Get 5-HT inhibition
        inhibition = raphe.get_vta_inhibition()
        self.receive_serotonin_inhibition(inhibition)

        # Send RPE to raphe (modulates patience)
        if bidirectional:
            raphe.connect_to_vta(self.state.last_rpe)

    def receive_pfc_modulation(
        self,
        pfc_signal: float,
        context: str = "goal"
    ) -> None:
        """
        Receive prefrontal cortex modulation signal.

        Biological basis:
        - dlPFC (dorsolateral) → VTA modulates goal-directed behavior
        - vmPFC (ventromedial) → VTA modulates value-based decisions
        - PFC provides top-down control of dopamine release

        dlPFC context:
        - High PFC signal → increased tonic DA (sustained motivation)
        - Enhances goal-directed behavior

        vmPFC context:
        - Biases expected reward baseline
        - Modulates value computations

        Args:
            pfc_signal: PFC modulation strength [0, 1]
            context: "goal" for dlPFC or "value" for vmPFC
        """
        pfc_signal = float(np.clip(pfc_signal, 0, 1))

        if context == "goal":
            # dlPFC modulates tonic DA firing rate — stored in STATE not CONFIG
            tonic_mod_delta = 0.2 * pfc_signal  # Up to 20% increase
            self.state.tonic_modulation = min(0.2, self.state.tonic_modulation + tonic_mod_delta * 0.01)
        elif context == "value":
            # vmPFC biases expected reward baseline
            # Shifts value estimate toward PFC prediction
            value_bias = 0.1 * (pfc_signal - 0.5)  # ±0.05 shift
            self.state.value_estimate = np.clip(
                self.state.value_estimate + value_bias,
                0.0,
                1.0
            )
        else:
            logger.warning(f"Unknown PFC context: {context}")

    def process_rpe_with_serotonin(
        self,
        rpe: float,
        serotonin_level: float,
        dt: float = 0.1
    ) -> float:
        """
        Process RPE with serotonin modulation.

        High 5-HT dampens the dopamine response to reward.

        Args:
            rpe: Reward prediction error
            serotonin_level: Current 5-HT level [0, 1]
            dt: Timestep

        Returns:
            DA level after modulated response
        """
        # Serotonin dampens RPE effect (opponent process)
        # High 5-HT → patience → reduced impulsive DA response
        dampening = 1.0 - 0.4 * serotonin_level  # 0.6-1.0 multiplier
        modulated_rpe = rpe * dampening

        # Process with dampened RPE
        return self.process_rpe(modulated_rpe, dt)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> dict:
        """Get VTA circuit statistics."""
        return {
            "firing_mode": self.state.firing_mode.value,
            "current_da": self.state.current_da,
            "current_rate": self.state.current_rate,
            "last_rpe": self.state.last_rpe,
            "cumulative_rpe": self.state.cumulative_rpe,
            "eligibility": self.state.eligibility,
            "value_estimate": self.state.value_estimate,
            "n_states_tracked": len(self._value_table),
            "avg_reward": (
                float(np.mean(self._reward_history))
                if self._reward_history else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset VTA to initial state."""
        self.state = VTAState(
            current_da=self.config.tonic_da_level,
            current_rate=self.config.tonic_rate,
            tonic_modulation=0.0
        )
        self._trace_history.clear()
        self._reward_history.clear()
        logger.info("VTACircuit reset to initial state")

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "state": self.state.to_dict(),
            "value_table": self._value_table.copy(),
            "config": {
                "tonic_da_level": self.config.tonic_da_level,
                "rpe_to_da_gain": self.config.rpe_to_da_gain,
                "discount_gamma": self.config.discount_gamma,
                "td_lambda": self.config.td_lambda,
            }
        }

    def load_state(self, saved: dict) -> None:
        """Load state from persistence."""
        if "value_table" in saved:
            self._value_table = saved["value_table"]
        if "state" in saved:
            state_dict = saved["state"]
            self.state.current_da = state_dict.get(
                "current_da", self.config.tonic_da_level
            )
            self.state.current_rate = state_dict.get(
                "current_rate", self.config.tonic_rate
            )
            self.state.value_estimate = state_dict.get("value_estimate", 0.5)


def create_vta_circuit(
    tonic_da: float = 0.3,
    rpe_gain: float = 0.5,
    td_lambda: float = 0.9
) -> VTACircuit:
    """
    Factory function to create VTA circuit with common configurations.

    Args:
        tonic_da: Baseline DA level
        rpe_gain: RPE to DA conversion gain
        td_lambda: Eligibility trace decay

    Returns:
        Configured VTACircuit
    """
    config = VTAConfig(
        tonic_da_level=tonic_da,
        rpe_to_da_gain=rpe_gain,
        td_lambda=td_lambda
    )
    return VTACircuit(config)


# Backward compatibility aliases
VTADopamineCircuit = VTACircuit
VTA = VTACircuit


__all__ = [
    "VTACircuit",
    "VTADopamineCircuit",  # Alias
    "VTA",  # Alias
    "VTAConfig",
    "VTAState",
    "VTAFiringMode",
    "create_vta_circuit",
]
