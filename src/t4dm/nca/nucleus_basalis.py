"""
Nucleus Basalis of Meynert (NBM) Acetylcholine Circuit for World Weaver.

Biological Basis:
- NBM is the primary source of cortical acetylcholine (ACh)
- NBM projects widely to neocortex, modulating attention and cortical plasticity
- ACh release is driven by salience/arousal signals
- High ACh → enhanced cortical processing, sharper attention, better encoding
- NBM degeneration in Alzheimer's disease leads to cognitive decline

Key Features:
- Salience-driven ACh release
- Baseline cholinergic tone for cortical readiness
- Phasic ACh bursts for attentional capture
- Modulates cortical plasticity and memory encoding strength

Architecture:
- Receives salience signals from amygdala, LC (norepinephrine)
- Produces tonic and phasic ACh release
- Projects to all cortical regions
- Integrates with attention and memory systems

Integration:
- NeuralFieldSolver: inject ACh for cortical modulation
- MemoryEncoding: ACh signals enhance encoding strength
- Attention: ACh sharpens attentional focus

References:
- Mesulam et al. (1983): NBM cortical cholinergic projections
- Sarter et al. (2005): ACh and attention
- Hasselmo (2006): ACh and memory encoding/retrieval modes
- Ballinger et al. (2016): NBM salience encoding
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class NBMState(Enum):
    """NBM activity states."""
    QUIESCENT = "quiescent"    # Low activity (sleep/low arousal)
    TONIC = "tonic"            # Baseline cortical ACh tone
    PHASIC = "phasic"          # Salience-driven burst
    ELEVATED = "elevated"      # Sustained high arousal


@dataclass
class NBMConfig:
    """Configuration for NBM dynamics."""

    # Baseline ACh release
    baseline_ach: float = 0.3        # Baseline ACh level [0, 1]
    tonic_release_rate: float = 0.05  # Tonic ACh release per step

    # Phasic dynamics
    phasic_peak_ach: float = 0.9     # Peak ACh during salience burst
    phasic_duration: float = 0.3     # seconds
    phasic_refractory_ms: float = 100.0  # ATOM-P2-6: Refractory period in milliseconds

    # Salience -> ACh conversion
    salience_to_ach_gain: float = 0.7  # How much salience drives ACh
    salience_threshold: float = 0.3    # Minimum salience for phasic response

    # Cortical target modulation
    cortical_targets: list[str] | None = None  # List of target regions

    # ACh dynamics
    tau_ach: float = 0.4              # ACh decay time constant (s)
    reuptake_rate: float = 0.15       # ACh reuptake/degradation rate

    # Biological constraints
    min_ach: float = 0.05             # Minimum ACh (baseline tone)
    max_ach: float = 0.95             # Maximum ACh (saturation)

    # State transitions
    quiescent_threshold: float = 0.15  # Below this = quiescent
    elevated_threshold: float = 0.6    # Above this = elevated


@dataclass
class NBMCircuitState:
    """Current state of NBM circuit."""

    state: NBMState = NBMState.TONIC
    ach_level: float = 0.3            # Current ACh concentration
    activation: float = 0.5           # Current NBM activation [0, 1]

    # Input tracking
    salience_signal: float = 0.0      # Current salience input
    arousal_signal: float = 0.5       # Current arousal level

    # Phasic state
    in_phasic_burst: bool = False
    phasic_remaining: float = 0.0     # Remaining burst duration

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "state": self.state.value,
            "ach_level": self.ach_level,
            "activation": self.activation,
            "salience_signal": self.salience_signal,
            "arousal_signal": self.arousal_signal,
        }


class NucleusBasalisCircuit:
    """
    Nucleus Basalis of Meynert acetylcholine circuit.

    Models the dynamics of NBM cholinergic neurons that:
    1. Maintain tonic ACh for baseline cortical readiness
    2. Generate phasic ACh bursts for salient events
    3. Modulate cortical attention and plasticity
    4. Gate memory encoding strength

    Key innovation: Salience-driven cortical modulation, bridging
    bottom-up attention (salience) with top-down cortical processing.
    """

    def __init__(
        self,
        config: NBMConfig | None = None,
        cortical_targets: list[str] | None = None,
    ):
        """
        Initialize NBM circuit.

        Args:
            config: NBM configuration parameters
            cortical_targets: List of cortical region names to modulate
        """
        self.config = config or NBMConfig()

        # Set cortical targets
        if cortical_targets is not None:
            self.config.cortical_targets = cortical_targets
        elif self.config.cortical_targets is None:
            # Default cortical targets
            self.config.cortical_targets = [
                "PFC", "Motor", "Sensory", "Parietal", "Temporal", "Hippocampus"
            ]

        self.state = NBMCircuitState(
            ach_level=self.config.baseline_ach,
            activation=0.5,
        )

        # Track history
        self._ach_history: list[float] = []
        self._salience_history: list[float] = []
        self._max_history = 1000

        # ATOM-P0-13: Rate-limit phasic bursts (max 10/minute)
        from collections import deque
        self._phasic_burst_times: deque = deque(maxlen=10)

        # ATOM-P2-6: Track last phasic time for refractory period
        self._last_phasic_time: float | None = None

        # ATOM-P3-12: Homeostatic desensitization tracking
        self._high_ach_duration = 0.0
        self._desensitization_factor = 1.0

        logger.info(
            f"NBMCircuit initialized: baseline_ach={self.config.baseline_ach}, "
            f"targets={len(self.config.cortical_targets)}"
        )

    # =========================================================================
    # Core ACh Dynamics
    # =========================================================================

    def process_salience(self, salience: float, dt: float = 0.1, token: "CallerToken | None" = None) -> float:
        """
        Process salience signal to generate ACh response.

        High salience triggers phasic ACh burst for attentional capture.

        ATOM-P0-13: Protected by access control with rate limiting.

        Args:
            salience: Salience strength [0, 1]
            dt: Timestep (seconds)
            token: Caller token (required for access control)

        Returns:
            New ACh concentration
        """
        # ATOM-P0-13: Access control
        if token is not None:
            from t4dm.core.access_control import require_capability
            require_capability(token, "submit_salience")

        # ATOM-P2-1: Input validation at NCA layer boundary
        from t4dm.core.validation import ValidationError
        if not np.isfinite(salience):
            raise ValidationError("salience", "Contains NaN or Inf values")
        if not np.isfinite(dt):
            raise ValidationError("dt", "Contains NaN or Inf values")

        # ATOM-P0-13: Clamp salience to valid range
        salience = float(np.clip(salience, 0.0, 1.0))

        # Store salience
        self.state.salience_signal = salience
        self._salience_history.append(self.state.salience_signal)
        if len(self._salience_history) > self._max_history:
            self._salience_history = self._salience_history[-self._max_history:]

        # Check for phasic burst threshold
        if salience > self.config.salience_threshold and not self.state.in_phasic_burst:
            # ATOM-P0-13: Rate-limit phasic bursts (max 10/minute)
            import time
            now = time.monotonic()
            # Clear old entries outside 60s window
            while self._phasic_burst_times and now - self._phasic_burst_times[0] > 60.0:
                self._phasic_burst_times.popleft()

            if len(self._phasic_burst_times) >= 10:
                logger.warning("ATOM-P0-13: Phasic burst rate limit exceeded (10/min)")
                # Silently skip burst - don't trigger
            else:
                # Record burst time and trigger
                self._phasic_burst_times.append(now)
                self._trigger_phasic_burst(salience)

        # Update ACh level
        self._update_ach_level(dt)

        # Update state classification
        self._classify_state()

        return self.state.ach_level

    def _trigger_phasic_burst(self, salience: float) -> None:
        """Trigger phasic ACh burst for salient event."""
        # ATOM-P2-6: Check refractory period
        import time
        now = time.monotonic()
        if self._last_phasic_time is not None:
            elapsed_ms = (now - self._last_phasic_time) * 1000
            if elapsed_ms < self.config.phasic_refractory_ms:
                # Skip during refractory period
                return
        self._last_phasic_time = now

        self.state.in_phasic_burst = True
        self.state.phasic_remaining = self.config.phasic_duration

        # Boost ACh proportional to salience
        ach_boost = salience * self.config.salience_to_ach_gain
        self.state.ach_level = float(np.clip(
            self.state.ach_level + ach_boost,
            self.config.min_ach,
            self.config.max_ach
        ))

        # Increase activation
        self.state.activation = float(np.clip(
            self.state.activation + 0.3 * salience,
            0.0,
            1.0
        ))

    def _update_ach_level(self, dt: float) -> None:
        """
        Update ACh concentration.

        d[ACh]/dt = release - reuptake

        ATOM-P3-12: Homeostatic desensitization - sustained high ACh > 0.7 for >30s
        triggers receptor desensitization to prevent overstimulation.
        """
        # ATOM-P3-12: Track high ACh duration for desensitization
        if self.state.ach_level > 0.7:
            self._high_ach_duration += dt
            if self._high_ach_duration > 30.0:
                self._desensitization_factor *= 0.99
                self._desensitization_factor = max(0.5, self._desensitization_factor)
        else:
            self._high_ach_duration = 0.0
            self._desensitization_factor = min(1.0, self._desensitization_factor + 0.001)

        # Compute target ACh based on state
        if self.state.in_phasic_burst:
            # During burst, maintain high ACh
            target_ach = self.config.phasic_peak_ach
        else:
            # Otherwise, tonic release based on arousal
            target_ach = self.config.baseline_ach + (
                0.2 * self.state.arousal_signal
            )

        # Tonic release
        release = self.config.tonic_release_rate * dt

        # Reuptake (proportional to current level)
        reuptake = self.config.reuptake_rate * self.state.ach_level * dt

        # Net change
        d_ach = release - reuptake

        # Smooth update toward target
        alpha = dt / self.config.tau_ach
        self.state.ach_level += alpha * (target_ach - self.state.ach_level) + d_ach

        # ATOM-P3-12: Apply desensitization factor to effective ACh
        self.state.ach_level = float(np.clip(
            self.state.ach_level,
            self.config.min_ach,
            self.config.max_ach
        ))

    def _classify_state(self) -> None:
        """Classify current NBM state."""
        ach = self.state.ach_level

        if ach < self.config.quiescent_threshold:
            self.state.state = NBMState.QUIESCENT
        elif self.state.in_phasic_burst:
            self.state.state = NBMState.PHASIC
        elif ach > self.config.elevated_threshold:
            self.state.state = NBMState.ELEVATED
        else:
            self.state.state = NBMState.TONIC

    def step(self, dt: float = 0.1) -> None:
        """
        Advance NBM dynamics by one timestep.

        Handles phasic burst decay and tonic ACh maintenance.

        Args:
            dt: Timestep in seconds
        """
        # Handle phasic burst duration
        if self.state.in_phasic_burst:
            self.state.phasic_remaining -= dt
            if self.state.phasic_remaining <= 0:
                self.state.in_phasic_burst = False
                self.state.phasic_remaining = 0.0

        # Update ACh level (tonic decay)
        self._update_ach_level(dt)

        # Decay activation gradually
        self.state.activation *= 0.95

        # Update state classification
        self._classify_state()

        # Track history
        self._ach_history.append(self.state.ach_level)
        if len(self._ach_history) > self._max_history:
            self._ach_history = self._ach_history[-self._max_history:]

    # =========================================================================
    # Input Methods
    # =========================================================================

    def receive_salience(self, salience: float) -> None:
        """
        Receive salience signal (e.g., from amygdala).

        Args:
            salience: Salience strength [0, 1]
        """
        self.process_salience(salience)

    def set_arousal(self, arousal: float) -> None:
        """
        Set arousal level (e.g., from LC norepinephrine).

        Args:
            arousal: Arousal level [0, 1]
        """
        self.state.arousal_signal = float(np.clip(arousal, 0, 1))

    # =========================================================================
    # Output Methods
    # =========================================================================

    def emit_ach(self) -> float:
        """
        Emit current ACh level for cortical targets.

        ATOM-P3-12: Returns desensitization-modulated effective ACh level.

        Returns:
            Effective ACh concentration [0, 1] after desensitization
        """
        return self.state.ach_level * self._desensitization_factor

    def get_ach_for_neural_field(self) -> float:
        """
        Get ACh level for injection into NeuralFieldSolver.

        Returns:
            ACh concentration [0, 1]
        """
        return self.state.ach_level

    def get_attention_modulation(self) -> float:
        """
        Get attention modulation signal.

        High ACh → sharper attention, better signal-to-noise.
        ATOM-P3-11: Combines tonic and phasic components for attention.

        Returns:
            Attention modulation [0, 1]
        """
        # Attention benefits from both tonic level and phasic bursts
        phasic_component = 1.0 if self.state.in_phasic_burst else 0.0
        return self.state.ach_level * 0.7 + 0.3 * phasic_component

    def get_encoding_modulation(self) -> float:
        """
        Get memory encoding modulation signal.

        High ACh → enhanced memory encoding (Hasselmo 2006).
        ATOM-P3-11: Pure tonic ACh level for encoding strength.

        Returns:
            Encoding strength modulation [0, 1]
        """
        # ACh enhances encoding, suppresses retrieval
        return self.state.ach_level

    def get_plasticity_gate(self) -> float:
        """
        Get cortical plasticity gating signal.

        High ACh → increased cortical plasticity.
        ATOM-P3-11: Plasticity requires both tonic and phasic ACh via sigmoid gate.

        Returns:
            Plasticity gate [0, 1]
        """
        # Plasticity depends on both tonic and phasic ACh
        phasic_component = 1.0 if self.state.in_phasic_burst else 0.0
        sigmoid = 1.0 / (1.0 + np.exp(-5 * (phasic_component - 0.3)))
        return self.state.ach_level * sigmoid

    def get_cognitive_mode(self) -> str:
        """
        ATOM-P1-11: Get current cognitive mode based on ACh level.

        Based on Hasselmo 2006: ACh differentially modulates
        encoding vs retrieval in hippocampal circuits.

        Returns:
            "ENCODING" if ACh > 0.6
            "RETRIEVAL" if ACh < 0.4
            "TRANSITIONAL" otherwise
        """
        ach = self.state.ach_level
        if ach > 0.6:
            return "ENCODING"
        if ach < 0.4:
            return "RETRIEVAL"
        return "TRANSITIONAL"

    # =========================================================================
    # Statistics and State
    # =========================================================================

    def get_stats(self) -> dict:
        """Get NBM circuit statistics."""
        stats = {
            "state": self.state.state.value,
            "ach_level": self.state.ach_level,
            "activation": self.state.activation,
            "salience_signal": self.state.salience_signal,
            "arousal_signal": self.state.arousal_signal,
            "in_phasic_burst": self.state.in_phasic_burst,
            "n_targets": len(self.config.cortical_targets or []),
        }

        if self._ach_history:
            stats["avg_ach"] = float(np.mean(self._ach_history))
        if self._salience_history:
            stats["avg_salience"] = float(np.mean(self._salience_history))

        return stats

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = NBMCircuitState(
            ach_level=self.config.baseline_ach,
            activation=0.5,
        )
        self._ach_history.clear()
        self._salience_history.clear()
        logger.info("NBMCircuit reset")

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "state": self.state.to_dict(),
            "config": {
                "baseline_ach": self.config.baseline_ach,
                "salience_to_ach_gain": self.config.salience_to_ach_gain,
            }
        }

    def load_state(self, saved: dict) -> None:
        """Load state from persistence."""
        if "state" in saved:
            s = saved["state"]
            self.state.ach_level = s.get("ach_level", self.config.baseline_ach)
            self.state.activation = s.get("activation", 0.5)
            self.state.salience_signal = s.get("salience_signal", 0.0)


def create_nbm_circuit(
    baseline_ach: float = 0.3,
    salience_gain: float = 0.7,
    cortical_targets: list[str] | None = None,
) -> NucleusBasalisCircuit:
    """
    Factory function to create NBM circuit with common configurations.

    Args:
        baseline_ach: Baseline ACh level
        salience_gain: Salience to ACh conversion gain
        cortical_targets: List of cortical regions to target

    Returns:
        Configured NucleusBasalisCircuit
    """
    config = NBMConfig(
        baseline_ach=baseline_ach,
        salience_to_ach_gain=salience_gain,
        cortical_targets=cortical_targets,
    )
    return NucleusBasalisCircuit(config)


__all__ = [
    "NucleusBasalisCircuit",
    "NBMConfig",
    "NBMCircuitState",
    "NBMState",
    "create_nbm_circuit",
]
