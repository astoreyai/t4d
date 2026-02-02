"""
Adenosine Sleep-Wake Dynamics for T4DM NCA.

Implements biologically-inspired adenosine accumulation and sleep pressure
as per Borbély's two-process model of sleep regulation.

Biological Basis:
- Adenosine is a metabolic byproduct (ATP → ADP → AMP → Adenosine)
- Accumulates during wake as neurons expend energy
- Acts on A1 receptors (inhibitory) and A2A receptors in basal forebrain
- Creates "sleep pressure" (Process S in two-process model)
- Cleared during sleep by astrocytic enzymes (adenosine kinase, deaminase)
- Caffeine blocks adenosine receptors, reducing sleep pressure perception

Two-Process Model (Borbély, 1982):
- Process S: Sleep homeostasis (adenosine-driven)
- Process C: Circadian rhythm (not implemented here - use oscillators)
- Sleep onset when S exceeds upper threshold
- Wake onset when S drops below lower threshold

Integration with NCA:
- Adenosine modulates NT dynamics (reduces DA, NE release)
- High adenosine → reduced cognitive performance
- Sleep clears adenosine → restored function
- Connects to consolidation/sleep.py for actual sleep processing

References:
- Porkka-Heiskanen et al. (1997) - Adenosine accumulation during wake
- Basheer et al. (2004) - Adenosine and sleep homeostasis
- Borbély & Achermann (1999) - Two-process model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SleepWakeState(Enum):
    """Sleep-wake states based on adenosine levels."""
    WAKE_ALERT = "wake_alert"           # Low adenosine, high performance
    WAKE_DROWSY = "wake_drowsy"         # Rising adenosine, declining performance
    WAKE_EXHAUSTED = "wake_exhausted"   # High adenosine, severely impaired
    SLEEP_LIGHT = "sleep_light"         # NREM stage 1-2
    SLEEP_DEEP = "sleep_deep"           # NREM stage 3-4 (slow-wave)
    SLEEP_REM = "sleep_rem"             # REM sleep


class SleepStage(Enum):
    """Detailed sleep stages (C1 enhancement)."""
    WAKE = "wake"       # Awake
    N1 = "n1"           # NREM Stage 1 (transition)
    N2 = "n2"           # NREM Stage 2 (spindles)
    N3 = "n3"           # NREM Stage 3 (slow-wave)
    REM = "rem"         # REM sleep


@dataclass
class AdenosineConfig:
    """Configuration for adenosine dynamics.

    All time constants are in hours for biological plausibility.
    Default values based on human sleep research.
    """
    # Accumulation parameters
    baseline_level: float = 0.1          # Baseline adenosine (normalized 0-1)
    max_level: float = 1.0               # Maximum adenosine level
    accumulation_rate: float = 0.04      # Rate per hour during wake (~16h to max)

    # C1: Sleep stage machine parameters
    ultradian_cycle_minutes: float = 90.0  # 90-minute sleep cycle
    enable_stage_machine: bool = True    # Enable state machine transitions

    # Clearance parameters (during sleep)
    clearance_rate_light: float = 0.08   # Clearance rate in light sleep
    clearance_rate_deep: float = 0.15    # Clearance rate in deep sleep (faster)
    clearance_rate_rem: float = 0.05     # Clearance rate in REM (slower)

    # Sleep pressure thresholds
    sleep_onset_threshold: float = 0.7   # Adenosine level triggering sleep need
    wake_threshold: float = 0.2          # Adenosine level allowing wake
    drowsy_threshold: float = 0.4        # Threshold for drowsy state
    exhausted_threshold: float = 0.85    # Threshold for exhausted state

    # Receptor dynamics
    a1_sensitivity: float = 1.0          # A1 receptor sensitivity (inhibitory)
    a2a_sensitivity: float = 0.8         # A2A receptor sensitivity
    receptor_adaptation_rate: float = 0.01  # Receptor downregulation rate

    # Caffeine effects (optional antagonist)
    caffeine_half_life_hours: float = 5.0   # Caffeine clearance half-life
    caffeine_block_efficacy: float = 0.7    # Max receptor blocking

    # NT modulation strengths
    da_suppression: float = 0.3          # How much adenosine suppresses DA
    ne_suppression: float = 0.4          # How much adenosine suppresses NE
    ach_suppression: float = 0.2         # How much adenosine suppresses ACh
    gaba_potentiation: float = 0.3       # How much adenosine potentiates GABA

    # Astrocyte-mediated clearance
    astrocyte_clearance_boost: float = 1.5  # Boost from active astrocytes


@dataclass
class AdenosineState:
    """Current state of adenosine system."""
    level: float = 0.1                   # Current adenosine level (0-1)
    sleep_pressure: float = 0.0          # Accumulated sleep pressure
    wake_duration_hours: float = 0.0     # Hours since last sleep
    sleep_duration_hours: float = 0.0    # Hours of current sleep
    caffeine_level: float = 0.0          # Current caffeine level (0-1)
    receptor_sensitivity: float = 1.0    # Current receptor sensitivity
    state: SleepWakeState = SleepWakeState.WAKE_ALERT
    last_update: datetime = field(default_factory=datetime.now)

    # Performance metrics
    cognitive_efficiency: float = 1.0    # 0-1, how well cognition works
    consolidation_need: float = 0.0      # 0-1, how much consolidation needed

    # C1: Sleep stage machine state
    sleep_stage: SleepStage = SleepStage.WAKE
    ultradian_phase: float = 0.0         # Phase in ultradian cycle (0-1)
    time_in_stage_minutes: float = 0.0   # Time in current stage

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "sleep_pressure": self.sleep_pressure,
            "wake_duration_hours": self.wake_duration_hours,
            "sleep_duration_hours": self.sleep_duration_hours,
            "caffeine_level": self.caffeine_level,
            "receptor_sensitivity": self.receptor_sensitivity,
            "state": self.state.value,
            "cognitive_efficiency": self.cognitive_efficiency,
            "consolidation_need": self.consolidation_need,
        }


class AdenosineDynamics:
    """
    Adenosine-based sleep-wake regulation.

    Models the homeostatic sleep drive (Process S) through:
    1. Adenosine accumulation during wake
    2. Adenosine clearance during sleep
    3. Receptor adaptation and caffeine effects
    4. NT modulation based on adenosine levels

    Usage:
        dynamics = AdenosineDynamics()

        # During wake, call periodically
        state = dynamics.step_wake(dt_hours=1.0, activity_level=0.8)

        # Check if sleep needed
        if dynamics.should_sleep():
            # Trigger consolidation
            dynamics.enter_sleep()

        # During sleep
        state = dynamics.step_sleep(dt_hours=0.5, sleep_phase="deep")

        # Get NT modulation
        modulation = dynamics.get_nt_modulation()
    """

    def __init__(
        self,
        config: AdenosineConfig | None = None,
        initial_state: AdenosineState | None = None
    ):
        """
        Initialize adenosine dynamics.

        Args:
            config: Configuration parameters
            initial_state: Initial state (defaults to rested wake)
        """
        self.config = config or AdenosineConfig()
        self.state = initial_state or AdenosineState(
            level=self.config.baseline_level,
            state=SleepWakeState.WAKE_ALERT
        )

        # History for analysis
        self._level_history: list[tuple[datetime, float]] = []
        self._state_transitions: list[tuple[datetime, SleepWakeState, SleepWakeState]] = []

        # Astrocyte integration
        self._astrocyte_activity: float = 0.0

        # C7: Glymphatic coupling
        self._glymphatic: Any | None = None
        self._glymphatic_modulation: float = 1.0  # Modulates accumulation rate

    def step_wake(
        self,
        dt_hours: float,
        activity_level: float = 0.5,
        caffeine_dose: float = 0.0
    ) -> AdenosineState:
        """
        Update adenosine during wake period.

        Args:
            dt_hours: Time step in hours
            activity_level: Cognitive activity level 0-1 (higher = faster accumulation)
            caffeine_dose: Caffeine dose 0-1 (0 = none, 1 = strong coffee)

        Returns:
            Updated state
        """
        # Accumulate adenosine based on activity
        # Higher activity = more ATP consumption = more adenosine
        effective_rate = self.config.accumulation_rate * (0.5 + 0.5 * activity_level)

        # C7: Modulate accumulation by glymphatic clearance
        # High glymphatic flow → reduced accumulation
        effective_rate *= self._glymphatic_modulation

        # Apply accumulation
        delta = effective_rate * dt_hours
        self.state.level = min(self.config.max_level, self.state.level + delta)

        # Update caffeine (decay + new dose)
        if caffeine_dose > 0:
            self.state.caffeine_level = min(1.0, self.state.caffeine_level + caffeine_dose)

        # Caffeine decay (exponential)
        decay_factor = np.exp(-np.log(2) * dt_hours / self.config.caffeine_half_life_hours)
        self.state.caffeine_level *= decay_factor

        # Receptor adaptation (chronic high adenosine reduces sensitivity)
        if self.state.level > self.config.drowsy_threshold:
            adaptation = self.config.receptor_adaptation_rate * dt_hours
            self.state.receptor_sensitivity = max(0.5, self.state.receptor_sensitivity - adaptation)
        else:
            # Recovery when adenosine is low
            recovery = self.config.receptor_adaptation_rate * dt_hours * 0.5
            self.state.receptor_sensitivity = min(1.0, self.state.receptor_sensitivity + recovery)

        # Update wake duration
        self.state.wake_duration_hours += dt_hours
        self.state.sleep_duration_hours = 0.0

        # Compute effective sleep pressure (adenosine * receptor sensitivity - caffeine block)
        caffeine_block = self.state.caffeine_level * self.config.caffeine_block_efficacy
        effective_adenosine = self.state.level * self.state.receptor_sensitivity
        self.state.sleep_pressure = max(0, effective_adenosine - caffeine_block)

        # Update cognitive efficiency (inverse of sleep pressure)
        self.state.cognitive_efficiency = max(0.1, 1.0 - self.state.sleep_pressure * 0.8)

        # Update consolidation need (how much work has accumulated)
        self.state.consolidation_need = min(1.0, self.state.consolidation_need + 0.02 * dt_hours)

        # Determine state
        old_state = self.state.state
        self._update_wake_state()

        if old_state != self.state.state:
            self._state_transitions.append((datetime.now(), old_state, self.state.state))

        # Record history
        self.state.last_update = datetime.now()
        self._level_history.append((self.state.last_update, self.state.level))

        return self.state

    def step_sleep(
        self,
        dt_hours: float,
        sleep_phase: str = "deep"
    ) -> AdenosineState:
        """
        Update adenosine during sleep period.

        Args:
            dt_hours: Time step in hours
            sleep_phase: "light", "deep", or "rem"

        Returns:
            Updated state
        """
        # Determine clearance rate based on phase
        if sleep_phase == "deep":
            clearance_rate = self.config.clearance_rate_deep
            new_state = SleepWakeState.SLEEP_DEEP
        elif sleep_phase == "rem":
            clearance_rate = self.config.clearance_rate_rem
            new_state = SleepWakeState.SLEEP_REM
        else:  # light
            clearance_rate = self.config.clearance_rate_light
            new_state = SleepWakeState.SLEEP_LIGHT

        # Astrocyte boost to clearance
        if self._astrocyte_activity > 0:
            clearance_rate *= (1 + (self.config.astrocyte_clearance_boost - 1) * self._astrocyte_activity)

        # Exponential decay of adenosine
        decay_factor = np.exp(-clearance_rate * dt_hours)
        self.state.level = max(
            self.config.baseline_level,
            self.state.level * decay_factor
        )

        # Caffeine also clears during sleep
        caffeine_decay = np.exp(-np.log(2) * dt_hours / self.config.caffeine_half_life_hours)
        self.state.caffeine_level *= caffeine_decay

        # Receptor sensitivity recovers during sleep
        recovery = self.config.receptor_adaptation_rate * dt_hours * 2.0  # Faster during sleep
        self.state.receptor_sensitivity = min(1.0, self.state.receptor_sensitivity + recovery)

        # Update durations
        self.state.sleep_duration_hours += dt_hours

        # Update sleep pressure
        self.state.sleep_pressure = self.state.level * self.state.receptor_sensitivity

        # Cognitive efficiency not relevant during sleep, but restore it
        self.state.cognitive_efficiency = min(1.0, 1.0 - self.state.sleep_pressure * 0.5)

        # Consolidation happens during sleep
        consolidation_per_hour = 0.15 if sleep_phase == "deep" else 0.08
        self.state.consolidation_need = max(0, self.state.consolidation_need - consolidation_per_hour * dt_hours)

        # Update state
        old_state = self.state.state
        self.state.state = new_state

        if old_state != self.state.state:
            self._state_transitions.append((datetime.now(), old_state, self.state.state))

        # Record
        self.state.last_update = datetime.now()
        self._level_history.append((self.state.last_update, self.state.level))

        return self.state

    def _update_wake_state(self) -> None:
        """Update wake state based on current levels."""
        if self.state.sleep_pressure >= self.config.exhausted_threshold:
            self.state.state = SleepWakeState.WAKE_EXHAUSTED
        elif self.state.sleep_pressure >= self.config.drowsy_threshold:
            self.state.state = SleepWakeState.WAKE_DROWSY
        else:
            self.state.state = SleepWakeState.WAKE_ALERT

    def should_sleep(self) -> bool:
        """
        Check if sleep should be initiated.

        Returns:
            True if adenosine exceeds sleep onset threshold
        """
        return self.state.sleep_pressure >= self.config.sleep_onset_threshold

    def can_wake(self) -> bool:
        """
        Check if wake is possible after sleep.

        Returns:
            True if adenosine below wake threshold
        """
        return self.state.level <= self.config.wake_threshold

    def enter_sleep(self) -> None:
        """Transition to sleep state."""
        self.state.wake_duration_hours = 0.0
        self.state.state = SleepWakeState.SLEEP_LIGHT
        self._state_transitions.append((
            datetime.now(),
            SleepWakeState.WAKE_DROWSY,  # Assumed prior state
            SleepWakeState.SLEEP_LIGHT
        ))

    def exit_sleep(self) -> None:
        """Transition from sleep to wake."""
        self.state.sleep_duration_hours = 0.0
        self._update_wake_state()
        self._state_transitions.append((
            datetime.now(),
            self.state.state,
            SleepWakeState.WAKE_ALERT
        ))
        self.state.state = SleepWakeState.WAKE_ALERT

    def get_nt_modulation(self) -> dict[str, float]:
        """
        Get neurotransmitter modulation based on adenosine levels.

        Adenosine inhibits wake-promoting NTs and potentiates GABA.

        Returns:
            Dict of modulation factors (multiply with baseline release)
        """
        # Effective adenosine action
        effective = self.state.level * self.state.receptor_sensitivity
        caffeine_block = self.state.caffeine_level * self.config.caffeine_block_efficacy
        net_adenosine = max(0, effective - caffeine_block)

        # Compute modulation factors
        # High adenosine suppresses DA, NE, ACh and potentiates GABA
        return {
            "da": 1.0 - net_adenosine * self.config.da_suppression,
            "ne": 1.0 - net_adenosine * self.config.ne_suppression,
            "ach": 1.0 - net_adenosine * self.config.ach_suppression,
            "gaba": 1.0 + net_adenosine * self.config.gaba_potentiation,
            "5ht": 1.0 - net_adenosine * 0.1,  # Mild serotonin suppression
        }

    def get_consolidation_signal(self) -> float:
        """
        Get signal indicating consolidation need.

        Higher values indicate more urgent consolidation need.
        Useful for triggering SleepConsolidation.

        Returns:
            Consolidation urgency 0-1
        """
        # Combine adenosine level with accumulated consolidation need
        adenosine_signal = self.state.level / self.config.max_level
        return (adenosine_signal + self.state.consolidation_need) / 2

    def set_astrocyte_activity(self, activity: float) -> None:
        """
        Set astrocyte activity level for clearance modulation.

        Astrocytes clear adenosine via adenosine kinase.
        Higher activity = faster clearance during sleep.

        Args:
            activity: Activity level 0-1
        """
        self._astrocyte_activity = np.clip(activity, 0, 1)

    def add_caffeine(self, dose: float) -> None:
        """
        Add caffeine dose (simulates coffee/tea consumption).

        Args:
            dose: Dose level 0-1 (0.3 = weak tea, 0.7 = strong coffee)
        """
        self.state.caffeine_level = min(1.0, self.state.caffeine_level + dose)

    def get_sleep_debt(self) -> float:
        """
        Calculate accumulated sleep debt.

        Returns:
            Sleep debt in equivalent hours
        """
        # Based on how far above baseline and duration
        excess = max(0, self.state.level - self.config.baseline_level)
        return excess * self.state.wake_duration_hours

    def get_optimal_sleep_duration(self) -> float:
        """
        Estimate optimal sleep duration to clear adenosine.

        Returns:
            Recommended sleep hours
        """
        if self.state.level <= self.config.wake_threshold:
            return 0.0

        # Solve exponential decay for time to reach wake threshold
        # level * exp(-rate * t) = threshold
        # t = -ln(threshold/level) / rate
        avg_clearance = (
            self.config.clearance_rate_deep * 0.5 +
            self.config.clearance_rate_light * 0.3 +
            self.config.clearance_rate_rem * 0.2
        )

        target = max(self.config.baseline_level, self.config.wake_threshold)
        if self.state.level <= target:
            return 0.0

        return -np.log(target / self.state.level) / avg_clearance

    def reset(self) -> None:
        """Reset to well-rested state."""
        self.state = AdenosineState(
            level=self.config.baseline_level,
            state=SleepWakeState.WAKE_ALERT,
            cognitive_efficiency=1.0,
            receptor_sensitivity=1.0
        )
        self._level_history.clear()
        self._state_transitions.clear()

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "state": self.state.to_dict(),
            "should_sleep": self.should_sleep(),
            "can_wake": self.can_wake(),
            "sleep_debt_hours": self.get_sleep_debt(),
            "optimal_sleep_hours": self.get_optimal_sleep_duration(),
            "consolidation_signal": self.get_consolidation_signal(),
            "nt_modulation": self.get_nt_modulation(),
            "history_length": len(self._level_history),
            "transitions": len(self._state_transitions),
        }

    # -------------------------------------------------------------------------
    # C7: Glymphatic Coupling
    # -------------------------------------------------------------------------

    def connect_glymphatic(self, glymphatic: Any) -> None:
        """
        C7: Connect to glymphatic system for bidirectional coupling.

        During high glymphatic clearance (high flow), adenosine accumulation
        is reduced. During high adenosine, glymphatic pressure signal increases.

        Biological basis:
        - Glymphatic system clears adenosine during sleep (Xie et al. 2013)
        - High adenosine drives sleep need, which activates glymphatic flow
        - Bidirectional feedback loop

        Args:
            glymphatic: GlymphaticSystem instance
        """
        self._glymphatic = glymphatic

    def _update_glymphatic_coupling(self) -> None:
        """
        Update glymphatic modulation of adenosine accumulation.

        Called internally during step_wake/step_sleep.
        """
        if self._glymphatic is None:
            self._glymphatic_modulation = 1.0
            return

        # Get glymphatic clearance rate
        clearance_rate = self._glymphatic.state.clearance_rate

        # High clearance → reduced accumulation
        # clearance_rate = 0.7 (deep NREM) → modulation = 0.3
        # clearance_rate = 0.1 (wake) → modulation = 0.9
        self._glymphatic_modulation = 1.0 - clearance_rate * 0.7


class SleepPressureIntegrator:
    """
    Integrates adenosine dynamics with NCA neural field.

    Connects AdenosineDynamics to:
    1. NeuralFieldSolver - modulates NT dynamics
    2. SleepConsolidation - triggers consolidation cycles
    3. Oscillators - affects theta/gamma power
    """

    def __init__(
        self,
        adenosine: AdenosineDynamics,
        consolidation_threshold: float = 0.6
    ):
        """
        Initialize integrator.

        Args:
            adenosine: AdenosineDynamics instance
            consolidation_threshold: Threshold for triggering consolidation
        """
        self.adenosine = adenosine
        self.consolidation_threshold = consolidation_threshold
        self._last_consolidation: datetime | None = None
        self._min_consolidation_interval = timedelta(hours=4)

    def check_consolidation_needed(self) -> bool:
        """
        Check if consolidation should be triggered.

        Returns:
            True if consolidation needed and allowed
        """
        if not self.adenosine.should_sleep():
            return False

        signal = self.adenosine.get_consolidation_signal()
        if signal < self.consolidation_threshold:
            return False

        # Check minimum interval
        if self._last_consolidation is not None:
            elapsed = datetime.now() - self._last_consolidation
            if elapsed < self._min_consolidation_interval:
                return False

        return True

    def record_consolidation(self) -> None:
        """Record that consolidation occurred."""
        self._last_consolidation = datetime.now()
        self.adenosine.enter_sleep()

    def apply_to_neural_field(
        self,
        nt_state: NeurotransmitterState  # noqa: F821
    ) -> NeurotransmitterState:  # noqa: F821
        """
        Apply adenosine modulation to NT state.

        Args:
            nt_state: Current neurotransmitter state

        Returns:
            Modulated NT state
        """
        modulation = self.adenosine.get_nt_modulation()

        # Create modulated state (assuming NeurotransmitterState has these fields)
        # This is a template - actual implementation depends on NeurotransmitterState structure
        if hasattr(nt_state, 'da'):
            nt_state.da = nt_state.da * modulation["da"]
        if hasattr(nt_state, 'ne'):
            nt_state.ne = nt_state.ne * modulation["ne"]
        if hasattr(nt_state, 'ach'):
            nt_state.ach = nt_state.ach * modulation["ach"]
        if hasattr(nt_state, 'gaba'):
            nt_state.gaba = nt_state.gaba * modulation["gaba"]
        if hasattr(nt_state, 'serotonin') or hasattr(nt_state, '_5ht'):
            attr = 'serotonin' if hasattr(nt_state, 'serotonin') else '_5ht'
            setattr(nt_state, attr, getattr(nt_state, attr) * modulation["5ht"])

        return nt_state

    def get_oscillator_modulation(self) -> dict[str, float]:
        """
        Get oscillator power modulation based on sleep pressure.

        High sleep pressure reduces theta/gamma coherence.

        Returns:
            Dict of oscillator modulation factors
        """
        pressure = self.adenosine.state.sleep_pressure

        return {
            "theta_power": max(0.3, 1.0 - pressure * 0.5),
            "gamma_power": max(0.2, 1.0 - pressure * 0.6),
            "beta_power": max(0.4, 1.0 - pressure * 0.4),
            "pac_strength": max(0.3, 1.0 - pressure * 0.5),
        }

    def simulate_day(
        self,
        wake_hours: float = 16.0,
        sleep_hours: float = 8.0,
        activity_pattern: list[float] | None = None
    ) -> list[dict]:
        """
        Simulate a full day of adenosine dynamics.

        Args:
            wake_hours: Hours of wake
            sleep_hours: Hours of sleep
            activity_pattern: Hourly activity levels (defaults to typical pattern)

        Returns:
            List of hourly states
        """
        if activity_pattern is None:
            # Typical circadian activity pattern
            activity_pattern = [
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  # Morning ramp
                0.9, 0.9, 0.8, 0.7, 0.6, 0.5,  # Midday
                0.6, 0.7, 0.6, 0.5              # Afternoon/evening
            ]

        results = []

        # Wake period
        for hour in range(int(wake_hours)):
            activity = activity_pattern[hour % len(activity_pattern)]
            state = self.adenosine.step_wake(dt_hours=1.0, activity_level=activity)
            results.append({
                "hour": hour,
                "phase": "wake",
                "state": state.to_dict(),
                "should_sleep": self.adenosine.should_sleep()
            })

        # Sleep period
        self.adenosine.enter_sleep()
        sleep_phases = ["light", "deep", "deep", "rem"] * 2  # ~90min cycles
        for hour in range(int(sleep_hours)):
            phase = sleep_phases[hour % len(sleep_phases)]
            state = self.adenosine.step_sleep(dt_hours=1.0, sleep_phase=phase)
            results.append({
                "hour": int(wake_hours) + hour,
                "phase": f"sleep_{phase}",
                "state": state.to_dict(),
                "can_wake": self.adenosine.can_wake()
            })

        self.adenosine.exit_sleep()

        return results


# Convenience functions
def create_adenosine_system(
    baseline: float = 0.1,
    accumulation_rate: float = 0.04
) -> AdenosineDynamics:
    """
    Create an adenosine dynamics system with custom parameters.

    Args:
        baseline: Baseline adenosine level
        accumulation_rate: Hourly accumulation rate

    Returns:
        Configured AdenosineDynamics instance
    """
    config = AdenosineConfig(
        baseline_level=baseline,
        accumulation_rate=accumulation_rate
    )
    return AdenosineDynamics(config)


def compute_sleep_need(
    wake_hours: float,
    activity_level: float = 0.6
) -> float:
    """
    Compute sleep need based on wake duration and activity.

    Args:
        wake_hours: Hours awake
        activity_level: Average activity level 0-1

    Returns:
        Recommended sleep hours
    """
    # Base need increases with wake duration
    base_need = wake_hours / 2  # Roughly 8h sleep for 16h wake

    # Adjust for activity
    activity_factor = 0.8 + 0.4 * activity_level

    return base_need * activity_factor


# Backward compatibility alias
AdenosineSystem = AdenosineDynamics


# =============================================================================
# C1: Sleep Stage State Machine
# =============================================================================


class SleepStateMachine:
    """
    C1: Sleep stage state machine driven by adenosine + ultradian timer.

    Transitions between sleep stages (WAKE, N1, N2, N3, REM) based on:
    1. Adenosine pressure (drives WAKE → N1 → N2 → N3)
    2. Ultradian 90-minute cycle timer (drives stage progression)
    3. Consolidation needs (gates REM entry)

    Biological basis:
    - NREM (N1→N2→N3) dominated by high sleep pressure
    - REM occurs in later cycles when pressure lower
    - Spindles in N2/N3, SWR in N2/N3, procedural consolidation in REM

    Usage:
        state_machine = SleepStateMachine(adenosine_dynamics)
        stage = state_machine.current_stage  # SleepStage enum
        spindle_allowed = stage in [SleepStage.N2, SleepStage.N3]
    """

    def __init__(
        self,
        adenosine: AdenosineDynamics,
        ultradian_cycle_minutes: float = 90.0
    ):
        """
        Initialize sleep stage machine.

        Args:
            adenosine: AdenosineDynamics instance for pressure signal
            ultradian_cycle_minutes: Duration of full sleep cycle
        """
        self.adenosine = adenosine
        self.ultradian_cycle_minutes = ultradian_cycle_minutes
        self._cycle_start_time: float = 0.0
        self._total_time_minutes: float = 0.0

        # Glymphatic connection (C7)
        self._glymphatic: Any | None = None

    @property
    def current_stage(self) -> SleepStage:
        """Get current sleep stage."""
        return self.adenosine.state.sleep_stage

    def step(self, dt_minutes: float) -> SleepStage:
        """
        Advance stage machine.

        Args:
            dt_minutes: Time step in minutes

        Returns:
            Current sleep stage after update
        """
        self._total_time_minutes += dt_minutes
        self.adenosine.state.time_in_stage_minutes += dt_minutes

        # Update ultradian phase
        cycle_elapsed = (self._total_time_minutes - self._cycle_start_time)
        self.adenosine.state.ultradian_phase = (
            cycle_elapsed / self.ultradian_cycle_minutes
        ) % 1.0

        # Determine transitions
        pressure = self.adenosine.state.sleep_pressure
        current = self.adenosine.state.sleep_stage
        phase = self.adenosine.state.ultradian_phase

        new_stage = self._determine_stage_transition(current, pressure, phase)

        if new_stage != current:
            self.adenosine.state.sleep_stage = new_stage
            self.adenosine.state.time_in_stage_minutes = 0.0

            # Reset cycle on WAKE → N1 transition
            if current == SleepStage.WAKE and new_stage == SleepStage.N1:
                self._cycle_start_time = self._total_time_minutes

        return new_stage

    def _determine_stage_transition(
        self,
        current: SleepStage,
        pressure: float,
        ultradian_phase: float
    ) -> SleepStage:
        """
        Determine next sleep stage based on current state and signals.

        Transition rules (simplified):
        - WAKE: If pressure > 0.7 → N1
        - N1: After 5-10 min → N2
        - N2: If pressure > 0.5 and phase < 0.5 → N3, else after 20 min → REM
        - N3: After 20-40 min → N2 (lighter) or REM if phase > 0.6
        - REM: After 10-30 min → N2 or WAKE if pressure < 0.3

        Args:
            current: Current stage
            pressure: Adenosine pressure [0, 1]
            ultradian_phase: Phase in cycle [0, 1]

        Returns:
            Next sleep stage
        """
        time_in_stage = self.adenosine.state.time_in_stage_minutes

        if current == SleepStage.WAKE:
            # Transition to sleep onset (N1) when pressure high
            if pressure > 0.7:
                return SleepStage.N1
            return SleepStage.WAKE

        elif current == SleepStage.N1:
            # Brief transition stage → N2 after 5-10 min
            if time_in_stage > 5.0:
                return SleepStage.N2
            return SleepStage.N1

        elif current == SleepStage.N2:
            # N2 → N3 if high pressure and early in cycle
            if pressure > 0.5 and ultradian_phase < 0.4 and time_in_stage > 10:
                return SleepStage.N3
            # N2 → REM if later in cycle and time elapsed
            if ultradian_phase > 0.6 and time_in_stage > 20:
                return SleepStage.REM
            return SleepStage.N2

        elif current == SleepStage.N3:
            # N3 → N2 after deep sleep period (pressure clears)
            if pressure < 0.4 and time_in_stage > 20:
                return SleepStage.N2
            # N3 → REM if late in cycle
            if ultradian_phase > 0.7 and time_in_stage > 30:
                return SleepStage.REM
            return SleepStage.N3

        elif current == SleepStage.REM:
            # REM → WAKE if pressure very low
            if pressure < 0.2 and time_in_stage > 10:
                return SleepStage.WAKE
            # REM → N2 to start new cycle
            if time_in_stage > 20 and ultradian_phase < 0.2:
                return SleepStage.N2
            return SleepStage.REM

        return current

    def gate_spindle_generation(self) -> bool:
        """
        Gate spindle generation by sleep stage.

        Spindles occur in N2 and N3 (NREM stages).

        Returns:
            True if spindles should be generated
        """
        return self.current_stage in [SleepStage.N2, SleepStage.N3]

    def gate_swr_generation(self) -> bool:
        """
        Gate SWR generation by sleep stage.

        SWRs occur in N2 and N3 (not REM, not wake).

        Returns:
            True if SWRs should be generated
        """
        return self.current_stage in [SleepStage.N2, SleepStage.N3]

    def gate_procedural_consolidation(self) -> bool:
        """
        Gate procedural consolidation by sleep stage.

        Procedural memories consolidate in REM.

        Returns:
            True if procedural consolidation should occur
        """
        return self.current_stage == SleepStage.REM

    def reset(self) -> None:
        """Reset state machine to wake."""
        self.adenosine.state.sleep_stage = SleepStage.WAKE
        self.adenosine.state.ultradian_phase = 0.0
        self.adenosine.state.time_in_stage_minutes = 0.0
        self._cycle_start_time = 0.0
        self._total_time_minutes = 0.0
