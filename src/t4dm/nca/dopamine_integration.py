"""
Dopamine System Integration for World Weaver NCA.

Connects all dopamine-related components:
1. VTACircuit: Biologically realistic DA neuron dynamics
2. DopamineSystem (ww.learning): Memory-level RPE tracking
3. NeuralFieldSolver: DA concentration in neural field
4. HippocampalCircuit: Novelty-driven DA modulation
5. LearnableCoupling: Eligibility-modulated plasticity

Key Insight: Dopamine serves multiple roles:
- Reward signal (RPE) for learning
- Motivation/arousal modulation
- Novelty detection
- Memory encoding/retrieval gating

This module unifies these into a coherent system.

References:
- Schultz et al. (1997): A neural substrate of prediction and reward
- Lisman & Grace (2005): Hippocampal-VTA loop and novelty-dependent memory
- Adcock et al. (2006): Reward-motivated learning and mesolimbic activation
- Shohamy & Adcock (2010): Dopamine and adaptive memory
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from t4dm.core.access_control import CallerToken
    from t4dm.learning.dopamine import DopamineSystem
    from t4dm.nca.coupling import LearnableCoupling
    from t4dm.nca.hippocampus import HippocampalCircuit
    from t4dm.nca.neural_field import NeuralFieldSolver
    from t4dm.nca.vta import VTACircuit

logger = logging.getLogger(__name__)


class DelayBuffer:
    """
    Delay buffer for dopamine arrival timing.

    Implements biological delay between RPE computation and eligibility credit
    application. In real brains, dopamine release takes ~100-1000ms to reach
    synapses after reward signal.

    Reference: Schultz et al. (1997) - DA neuron firing precedes dopamine arrival
    """

    def __init__(self, delay_ms: float = 200.0):
        """
        Initialize delay buffer.

        Args:
            delay_ms: Delay in milliseconds (default 200ms per Schultz 1997)
        """
        self.delay_ms = delay_ms
        self.delay_seconds = delay_ms / 1000.0
        # Buffer stores (timestamp, value) tuples
        self._buffer: deque[tuple[float, float]] = deque()

    def enqueue(self, value: float, sim_time: float) -> None:
        """
        Add a value to the delay buffer.

        Args:
            value: RPE or DA value to buffer
            sim_time: Simulation time in seconds when the value was generated
        """
        self._buffer.append((sim_time, value))

    def dequeue_ready(self, sim_time: float) -> list[float]:
        """
        Dequeue all values whose delay period has elapsed.

        Args:
            sim_time: Current simulation time in seconds

        Returns:
            List of values ready for release
        """
        ready_values = []

        while self._buffer:
            ts, value = self._buffer[0]
            elapsed_seconds = sim_time - ts

            if elapsed_seconds >= self.delay_seconds:
                # This value is ready to release
                self._buffer.popleft()
                ready_values.append(value)
            else:
                # Not ready yet, stop checking
                break

        return ready_values

    def clear(self) -> None:
        """Clear all buffered values."""
        self._buffer.clear()

    def size(self) -> int:
        """Get number of buffered values."""
        return len(self._buffer)


@dataclass
class DopamineIntegrationConfig:
    """Configuration for dopamine integration."""

    # VTA -> NeuralField coupling
    vta_to_field_gain: float = 0.5        # How much VTA DA affects field DA
    field_spatial_spread: float = 0.3     # Spatial spread of DA injection

    # Hippocampus -> VTA coupling
    novelty_to_rpe_weight: float = 0.3    # How much novelty contributes to RPE
    novelty_threshold: float = 0.5        # Minimum novelty for VTA response

    # DopamineSystem -> VTA coupling
    memory_rpe_weight: float = 0.7        # Weight of memory-based RPE
    sync_expectations: bool = True         # Sync value estimates between systems

    # Eligibility integration
    use_coupling_eligibility: bool = True  # Use LearnableCoupling traces
    eligibility_window: float = 0.5        # Seconds for eligibility accumulation

    # Temporal dynamics
    integration_dt: float = 0.1           # Default timestep
    max_da_change_rate: float = 0.5       # Max DA change per step
    da_arrival_delay_ms: float = 200.0    # Dopamine arrival delay in milliseconds (Schultz 1997)


@dataclass
class IntegratedDAState:
    """Combined dopamine state across all systems."""

    # VTA state
    vta_da: float = 0.3
    vta_rpe: float = 0.0
    vta_firing_rate: float = 4.5

    # Neural field state
    field_da: float = 0.3

    # Memory system state
    memory_rpe: float = 0.0
    memory_expected: float = 0.5

    # Hippocampal contributions
    novelty_signal: float = 0.5
    novelty_rpe: float = 0.0

    # Combined
    integrated_rpe: float = 0.0
    integrated_da: float = 0.3

    # Eligibility
    coupling_eligibility: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vta_da": self.vta_da,
            "vta_rpe": self.vta_rpe,
            "field_da": self.field_da,
            "memory_rpe": self.memory_rpe,
            "novelty_signal": self.novelty_signal,
            "integrated_rpe": self.integrated_rpe,
            "integrated_da": self.integrated_da,
        }


class DopamineIntegration:
    """
    Unified dopamine system integration.

    Connects:
    - VTACircuit (biological DA dynamics)
    - DopamineSystem (memory-level RPE)
    - NeuralFieldSolver (DA concentration field)
    - HippocampalCircuit (novelty signals)
    - LearnableCoupling (eligibility traces)

    The integration follows this flow:
    1. Memory outcome arrives
    2. DopamineSystem computes memory-level RPE
    3. HippocampalCircuit provides novelty signal
    4. VTACircuit integrates both into biological DA response
    5. DA is injected into NeuralFieldSolver
    6. LearnableCoupling is updated via eligibility traces
    """

    def __init__(
        self,
        config: DopamineIntegrationConfig | None = None,
        vta: VTACircuit | None = None,
        neural_field: NeuralFieldSolver | None = None,
        hippocampus: HippocampalCircuit | None = None,
        coupling: LearnableCoupling | None = None,
        dopamine_system: DopamineSystem | None = None,
    ):
        """
        Initialize dopamine integration.

        Args:
            config: Integration configuration
            vta: VTA dopamine circuit
            neural_field: Neural field solver
            hippocampus: Hippocampal circuit for novelty
            coupling: Learnable coupling for plasticity
            dopamine_system: Memory-level dopamine system
        """
        self.config = config or DopamineIntegrationConfig()

        # Component references
        self.vta = vta
        self.neural_field = neural_field
        self.hippocampus = hippocampus
        self.coupling = coupling
        self.dopamine_system = dopamine_system

        # State tracking
        self.state = IntegratedDAState()
        self._last_memory_id: UUID | None = None
        self._accumulated_eligibility: float = 0.0
        self._sim_time: float = 0.0  # Simulation time in seconds

        # Event callbacks
        self._rpe_callbacks: list[Callable[[float], None]] = []
        self._da_callbacks: list[Callable[[float], None]] = []

        # History for analysis
        self._rpe_history: list[float] = []
        self._da_history: list[float] = []
        self._max_history = 1000

        # Dopamine arrival delay buffer
        self._da_delay_buffer = DelayBuffer(delay_ms=self.config.da_arrival_delay_ms)

        # Rate limiting for reward signals: max 10 outcomes per memory_id per minute
        self._outcome_timestamps: dict[str, deque] = {}

        # Wire up VTA callback if available
        if self.vta is not None:
            self.vta.register_da_callback(self._on_vta_da_change)

        logger.info(
            f"DopamineIntegration initialized with {self.config.da_arrival_delay_ms}ms DA delay (Schultz 1997)"
        )

    # =========================================================================
    # Core Integration Methods
    # =========================================================================

    def process_memory_outcome(
        self,
        memory_id: UUID,
        actual_outcome: float,
        pattern: np.ndarray | None = None,
        token: CallerToken | None = None,
    ) -> IntegratedDAState:
        """
        Process a memory outcome through the integrated dopamine system.

        This is the main entry point for learning signals.

        Args:
            memory_id: ID of memory that was used
            actual_outcome: Actual outcome [0, 1]
            pattern: Memory pattern (for hippocampal novelty)
            token: Caller token with "submit_reward" capability (optional)

        Returns:
            Updated integrated state

        Raises:
            ValueError: If rate limit exceeded or validation fails
        """
        # Security: Validate caller capability
        if token is not None:
            from t4dm.core.access_control import require_capability
            require_capability(token, "submit_reward")

        # Security: Clamp outcome to valid range [-1.0, 1.0]
        actual_outcome = float(np.clip(actual_outcome, -1.0, 1.0))

        # Security: Rate limit outcomes per memory_id
        mid = str(memory_id)
        now = time.monotonic()
        if mid not in self._outcome_timestamps:
            self._outcome_timestamps[mid] = deque(maxlen=10)
        ts_deque = self._outcome_timestamps[mid]

        # Remove timestamps older than 60 seconds
        while ts_deque and now - ts_deque[0] > 60.0:
            ts_deque.popleft()

        # Check rate limit: max 10 outcomes per memory_id per minute
        if len(ts_deque) >= 10:
            raise ValueError(
                f"Rate limit exceeded: max 10 outcomes per memory_id per minute (memory_id={mid})"
            )

        # Record this outcome timestamp
        ts_deque.append(now)

        # 1. Compute memory-level RPE
        memory_rpe = self._compute_memory_rpe(memory_id, actual_outcome)
        self.state.memory_rpe = memory_rpe

        # 2. Compute novelty-based RPE
        novelty_rpe = self._compute_novelty_rpe(pattern)
        self.state.novelty_rpe = novelty_rpe

        # 3. Integrate RPEs for VTA
        integrated_rpe = self._integrate_rpes(memory_rpe, novelty_rpe)
        self.state.integrated_rpe = integrated_rpe

        # 4. Process through VTA
        vta_da = self._process_vta(integrated_rpe)
        self.state.vta_da = vta_da

        # 5. Inject into neural field
        self._inject_to_neural_field(vta_da, integrated_rpe)

        # 6. Update coupling with eligibility
        self._update_coupling(integrated_rpe)

        # 7. Update combined state
        self.state.integrated_da = vta_da
        self._last_memory_id = memory_id

        # Track history
        self._rpe_history.append(integrated_rpe)
        self._da_history.append(vta_da)
        if len(self._rpe_history) > self._max_history:
            self._rpe_history = self._rpe_history[-self._max_history:]
            self._da_history = self._da_history[-self._max_history:]

        # Fire callbacks
        for callback in self._rpe_callbacks:
            callback(integrated_rpe)
        for callback in self._da_callbacks:
            callback(vta_da)

        return self.state

    def _compute_memory_rpe(
        self,
        memory_id: UUID,
        actual_outcome: float
    ) -> float:
        """Compute RPE from memory system."""
        if self.dopamine_system is None:
            return actual_outcome - 0.5  # Simple surprise

        rpe_result = self.dopamine_system.compute_rpe(memory_id, actual_outcome)

        # Update expectations
        self.dopamine_system.update_expectations(memory_id, actual_outcome)
        self.state.memory_expected = self.dopamine_system.get_expected_value(memory_id)

        return rpe_result.rpe

    def _compute_novelty_rpe(
        self,
        pattern: np.ndarray | None
    ) -> float:
        """Compute RPE contribution from hippocampal novelty."""
        if self.hippocampus is None or pattern is None:
            self.state.novelty_signal = 0.5
            return 0.0

        # Get novelty from hippocampus
        novelty = self.hippocampus.compute_novelty(pattern)
        self.state.novelty_signal = novelty

        # Convert to RPE (novelty above threshold = positive surprise)
        if novelty > self.config.novelty_threshold:
            novelty_rpe = (novelty - self.config.novelty_threshold) * 2
        else:
            novelty_rpe = 0.0

        return novelty_rpe * self.config.novelty_to_rpe_weight

    def _integrate_rpes(
        self,
        memory_rpe: float,
        novelty_rpe: float
    ) -> float:
        """Integrate RPE signals from different sources."""
        # Weighted combination
        integrated = (
            memory_rpe * self.config.memory_rpe_weight +
            novelty_rpe * (1 - self.config.memory_rpe_weight)
        )

        # Clip for stability
        return float(np.clip(integrated, -1.0, 1.0))

    def _process_vta(self, rpe: float) -> float:
        """Process RPE through VTA circuit."""
        if self.vta is None:
            # Simple DA dynamics without VTA
            self.state.vta_rpe = rpe
            da_change = rpe * self.config.vta_to_field_gain
            new_da = np.clip(
                self.state.vta_da + da_change,
                0.05, 0.95
            )
            return float(new_da)

        # Use VTA circuit
        da = self.vta.process_rpe(rpe, dt=self.config.integration_dt)
        self.state.vta_rpe = self.vta.state.last_rpe
        self.state.vta_firing_rate = self.vta.state.current_rate

        return da

    def _inject_to_neural_field(self, da: float, rpe: float) -> None:
        """Inject DA into neural field."""
        if self.neural_field is None:
            return

        # Compute DA change
        current_field_da = self._get_field_da()
        target_da = da * self.config.vta_to_field_gain + current_field_da * (
            1 - self.config.vta_to_field_gain
        )

        # Rate-limit change
        da_change = target_da - current_field_da
        da_change = np.clip(
            da_change,
            -self.config.max_da_change_rate,
            self.config.max_da_change_rate
        )

        # Inject using neural field method
        self.neural_field.inject_rpe(
            rpe=rpe,
            magnitude_scale=abs(da_change)
        )

        self.state.field_da = self._get_field_da()

    def _get_field_da(self) -> float:
        """Get current DA from neural field."""
        if self.neural_field is None:
            return 0.3
        state = self.neural_field.get_mean_state()
        return float(state.dopamine)

    def _update_coupling(self, rpe: float) -> None:
        """Enqueue RPE into delay buffer for later coupling update."""
        if self.coupling is None:
            return

        # Enqueue current RPE into delay buffer with simulation time
        self._da_delay_buffer.enqueue(rpe, self._sim_time)

    def _process_delayed_coupling_updates(self) -> None:
        """Process delayed dopamine signals for coupling updates."""
        if self.coupling is None:
            return

        # Dequeue delayed RPE values that are ready
        delayed_rpes = self._da_delay_buffer.dequeue_ready(self._sim_time)

        if not delayed_rpes:
            # No delayed signals ready yet
            return

        # Get eligibility from coupling
        if self.config.use_coupling_eligibility:
            eligibility = self.coupling.get_eligibility_trace()
            self.state.coupling_eligibility = float(np.mean(np.abs(eligibility)))
        else:
            eligibility = None

        # Get current NT state from field
        if self.neural_field is not None:
            nt_state = self.neural_field.get_mean_state()
        else:
            from t4dm.nca.neural_field import NeurotransmitterState
            nt_state = NeurotransmitterState()

        # Apply delayed RPE values to coupling
        for delayed_rpe in delayed_rpes:
            self.coupling.update_from_rpe(nt_state, delayed_rpe, eligibility)

    def _on_vta_da_change(self, da: float, rpe: float) -> None:
        """Callback when VTA DA changes."""
        # Update state
        self.state.vta_da = da
        self.state.vta_rpe = rpe

    # =========================================================================
    # Eligibility Trace Management
    # =========================================================================

    def accumulate_eligibility(self, pattern: np.ndarray | None = None) -> None:
        """
        Accumulate eligibility traces for upcoming learning.

        Call this during behavior to mark what led to outcomes.

        Args:
            pattern: Current activity pattern
        """
        if self.coupling is not None:
            # Get current NT state
            if self.neural_field is not None:
                nt_state = self.neural_field.get_mean_state()
            else:
                from t4dm.nca.neural_field import NeurotransmitterState
                nt_state = NeurotransmitterState()

            self.coupling.accumulate_eligibility(nt_state)

        # Also accumulate in hippocampus if available
        if self.hippocampus is not None and pattern is not None:
            self.hippocampus.process_input(pattern)

    def reset_eligibility(self) -> None:
        """Reset all eligibility traces."""
        if self.coupling is not None:
            self.coupling.reset_eligibility()
        self._accumulated_eligibility = 0.0

    # =========================================================================
    # Value Function Synchronization
    # =========================================================================

    def sync_value_estimates(self, memory_id: UUID) -> None:
        """
        Synchronize value estimates between DopamineSystem and VTA.

        Args:
            memory_id: Memory to sync
        """
        if not self.config.sync_expectations:
            return

        if self.dopamine_system is not None and self.vta is not None:
            # Get value from DopamineSystem
            expected = self.dopamine_system.get_expected_value(memory_id)

            # Set in VTA (using memory_id as state key)
            state_key = str(memory_id)
            self.vta._value_table[state_key] = expected

    def get_combined_value(self, memory_id: UUID) -> float:
        """
        Get combined value estimate from all systems.

        Args:
            memory_id: Memory to get value for

        Returns:
            Combined expected value
        """
        values = []

        if self.dopamine_system is not None:
            values.append(self.dopamine_system.get_expected_value(memory_id))

        if self.vta is not None:
            values.append(self.vta._get_value(str(memory_id)))

        if values:
            return float(np.mean(values))
        return 0.5

    # =========================================================================
    # Temporal Dynamics
    # =========================================================================

    def step(self, dt: float | None = None) -> None:
        """
        Step the integrated system forward.

        Args:
            dt: Timestep (uses config default if None)
        """
        dt = dt or self.config.integration_dt

        # Increment simulation time
        self._sim_time += dt

        # Process delayed dopamine signals for coupling updates
        self._process_delayed_coupling_updates()

        # Step VTA
        if self.vta is not None:
            self.vta.step(dt)
            self.state.vta_da = self.vta.state.current_da

        # Step neural field
        if self.neural_field is not None:
            self.neural_field.step(dt=dt)
            self.state.field_da = self._get_field_da()

        # Accumulate eligibility if enabled
        if self.config.use_coupling_eligibility:
            self.accumulate_eligibility()

    # =========================================================================
    # Callbacks and Events
    # =========================================================================

    def register_rpe_callback(self, callback: Callable[[float], None]) -> None:
        """
        Register callback for RPE events.

        ATOM-P3-15: Limited to 100 callbacks maximum.
        """
        if len(self._rpe_callbacks) >= 100:
            raise ValueError("Max 100 RPE callbacks")
        self._rpe_callbacks.append(callback)

    def register_da_callback(self, callback: Callable[[float], None]) -> None:
        """
        Register callback for DA changes.

        ATOM-P3-15: Limited to 100 callbacks maximum.
        """
        if len(self._da_callbacks) >= 100:
            raise ValueError("Max 100 DA callbacks")
        self._da_callbacks.append(callback)

    # =========================================================================
    # Statistics and State
    # =========================================================================

    def get_stats(self) -> dict:
        """Get integration statistics."""
        stats = {
            "integrated_rpe": self.state.integrated_rpe,
            "integrated_da": self.state.integrated_da,
            "vta_da": self.state.vta_da,
            "field_da": self.state.field_da,
            "memory_rpe": self.state.memory_rpe,
            "novelty_signal": self.state.novelty_signal,
            "coupling_eligibility": self.state.coupling_eligibility,
        }

        if self._rpe_history:
            stats["avg_rpe"] = float(np.mean(self._rpe_history))
            stats["rpe_std"] = float(np.std(self._rpe_history))
        if self._da_history:
            stats["avg_da"] = float(np.mean(self._da_history))

        return stats

    def get_state(self) -> IntegratedDAState:
        """Get current integrated state."""
        return self.state

    def reset(self) -> None:
        """Reset integration to initial state."""
        self.state = IntegratedDAState()
        self._rpe_history.clear()
        self._da_history.clear()
        self.reset_eligibility()
        self._sim_time = 0.0

        if self.vta is not None:
            self.vta.reset()

        logger.info("DopamineIntegration reset")

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "state": self.state.to_dict(),
            "rpe_history": self._rpe_history[-100:],  # Last 100
            "da_history": self._da_history[-100:],
        }

    def load_state(self, saved: dict) -> None:
        """Load state from persistence."""
        if "state" in saved:
            state_dict = saved["state"]
            self.state.integrated_rpe = state_dict.get("integrated_rpe", 0.0)
            self.state.integrated_da = state_dict.get("integrated_da", 0.3)
        if "rpe_history" in saved:
            self._rpe_history = saved["rpe_history"]
        if "da_history" in saved:
            self._da_history = saved["da_history"]


def create_dopamine_integration(
    vta: VTACircuit | None = None,
    neural_field: NeuralFieldSolver | None = None,
    hippocampus: HippocampalCircuit | None = None,
    coupling: LearnableCoupling | None = None,
    dopamine_system: DopamineSystem | None = None,
) -> DopamineIntegration:
    """
    Factory function to create dopamine integration.

    Args:
        vta: VTA circuit (created if None)
        neural_field: Neural field (created if None)
        hippocampus: Hippocampal circuit (optional)
        coupling: Learnable coupling (optional)
        dopamine_system: Memory dopamine system (optional)

    Returns:
        Configured DopamineIntegration
    """
    # Create VTA if needed
    if vta is None:
        from t4dm.nca.vta import VTACircuit
        vta = VTACircuit()

    # Create neural field if needed
    if neural_field is None:
        from t4dm.nca.neural_field import NeuralFieldSolver
        neural_field = NeuralFieldSolver()

    return DopamineIntegration(
        vta=vta,
        neural_field=neural_field,
        hippocampus=hippocampus,
        coupling=coupling,
        dopamine_system=dopamine_system,
    )


__all__ = [
    "DelayBuffer",
    "DopamineIntegration",
    "DopamineIntegrationConfig",
    "IntegratedDAState",
    "create_dopamine_integration",
]
