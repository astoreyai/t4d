"""
Eligibility Trace System for temporal credit assignment.

Biological inspiration: Synaptic tagging, dopamine eligibility windows.

Eligibility traces enable credit assignment across time, allowing rewards
received after a delay to reinforce earlier memories/actions that led to
the outcome. This is essential for learning in delayed reward scenarios.

Key concepts:
- Traces decay exponentially over time
- Activity increases trace strength
- Reward is multiplied by trace to assign credit
- Implements TD(λ)-style temporal credit assignment
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Security limits
MAX_TRACES = 10000  # Maximum number of concurrent traces
MAX_TRACE_VALUE = 100.0  # Maximum trace value to prevent overflow
MAX_MEMORY_ID_LENGTH = 256
MAX_ACTIVITY = 10.0
MAX_REWARD = 1000.0
MAX_DT = 86400.0  # 24 hours


@dataclass
class EligibilityConfig:
    """Configuration for eligibility traces."""
    decay: float = 0.95          # Per-step decay factor (λ)
    tau_trace: float = 20.0      # Time constant in seconds
    a_plus: float = 0.005        # LTP learning rate
    a_minus: float = 0.00525     # LTD learning rate (slightly higher for stability)
    min_trace: float = 1e-4      # Minimum trace before cleanup
    max_traces: int = 10000      # Maximum concurrent traces


@dataclass
class TraceEntry:
    """Entry for a single eligibility trace."""
    memory_id: str
    value: float
    last_update: float
    total_activations: int = 0


class EligibilityTrace:
    """
    Exponentially decaying trace for temporal credit assignment.

    Biological inspiration: Synaptic tagging, dopamine eligibility windows.

    The trace for a memory increases when that memory is activated,
    then decays exponentially. When a reward arrives, credit is
    assigned proportional to trace strength, implementing TD(λ)-style
    temporal credit assignment.

    Features:
    - Exponential decay with configurable time constant
    - Accumulating traces (repeated activation strengthens)
    - Reward assignment proportional to trace
    - Automatic cleanup of weak traces
    """

    def __init__(
        self,
        decay: float = 0.95,
        tau_trace: float = 20.0,
        a_plus: float = 0.005,
        a_minus: float = 0.00525,
        min_trace: float = 1e-4,
        max_traces: int = 10000
    ):
        """
        Initialize eligibility trace system.

        Args:
            decay: Per-step decay factor (λ)
            tau_trace: Time constant in seconds
            a_plus: LTP learning rate for trace increment
            a_minus: LTD learning rate (slight asymmetry for stability)
            min_trace: Minimum trace value before cleanup
            max_traces: Maximum number of concurrent traces
        """
        # Validation
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if tau_trace <= 0:
            raise ValueError(f"tau_trace must be positive, got {tau_trace}")
        if max_traces > MAX_TRACES:
            raise ValueError(f"max_traces ({max_traces}) exceeds MAX_TRACES ({MAX_TRACES})")

        self.decay = decay
        self.tau_trace = tau_trace
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.min_trace = min_trace
        self.max_traces = max_traces

        self.traces: dict[str, TraceEntry] = {}
        self._last_step_time = time.time()
        self._lock = threading.RLock()

        # Statistics
        self._total_credits_assigned = 0.0
        self._total_updates = 0

    def update(self, memory_id: str, activity: float = 1.0):
        """
        Update trace for memory activation.

        Trace increases with activity using STDP-like rule.

        Args:
            memory_id: ID of activated memory
            activity: Activity level (default 1.0 for full activation)
        """
        # Input validation
        if len(memory_id) > MAX_MEMORY_ID_LENGTH:
            raise ValueError(f"memory_id length ({len(memory_id)}) exceeds MAX_MEMORY_ID_LENGTH ({MAX_MEMORY_ID_LENGTH})")
        if not memory_id.isprintable():
            raise ValueError("memory_id must contain only printable characters")
        if not np.isfinite(activity):
            raise ValueError(f"activity must be finite, got {activity}")
        if activity < 0:
            raise ValueError(f"activity must be non-negative, got {activity}")
        if activity > MAX_ACTIVITY:
            raise ValueError(f"activity ({activity}) exceeds MAX_ACTIVITY ({MAX_ACTIVITY})")

        with self._lock:
            current_time = time.time()

            if memory_id in self.traces:
                entry = self.traces[memory_id]
                # Apply decay since last update
                elapsed = current_time - entry.last_update
                decay_factor = np.exp(-elapsed / self.tau_trace)
                entry.value *= decay_factor
                # Add new activity
                entry.value = min(entry.value + self.a_plus * activity, MAX_TRACE_VALUE)
                entry.last_update = current_time
                entry.total_activations += 1
            else:
                # Check capacity
                if len(self.traces) >= self.max_traces:
                    self._evict_weakest()

                self.traces[memory_id] = TraceEntry(
                    memory_id=memory_id,
                    value=self.a_plus * activity,
                    last_update=current_time,
                    total_activations=1
                )

            self._total_updates += 1

    def step(self, dt: float | None = None):
        """
        Decay all traces by time step.

        Args:
            dt: Time delta in seconds (auto-computed if None)
        """
        with self._lock:
            current_time = time.time()
            if dt is None:
                dt = current_time - self._last_step_time
            else:
                # Input validation for dt
                if not np.isfinite(dt):
                    raise ValueError(f"dt must be finite, got {dt}")
                if dt < 0:
                    raise ValueError(f"dt must be non-negative, got {dt}")
                dt = min(dt, MAX_DT)
            self._last_step_time = current_time

            decay_factor = np.exp(-dt / self.tau_trace)

            # Decay and cleanup
            # LOGIC-009 FIX: Update last_update to prevent double decay.
            # Without this, subsequent update() calls would re-apply decay
            # from the original last_update time, causing double decay.
            to_remove = []
            for memory_id, entry in self.traces.items():
                entry.value *= decay_factor
                entry.last_update = current_time  # Record that decay was applied
                if entry.value < self.min_trace:
                    to_remove.append(memory_id)

            for memory_id in to_remove:
                del self.traces[memory_id]

    def assign_credit(self, reward: float) -> dict[str, float]:
        """
        Assign reward credit to all active traces.

        Credit = reward × trace_value

        This implements temporal credit assignment:
        - Recent, strong activations get more credit
        - Older activations (decayed traces) get less credit

        Args:
            reward: Reward signal to distribute

        Returns:
            Dict mapping memory_id to credit amount
        """
        # Input validation
        if not np.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")
        reward = np.clip(reward, -MAX_REWARD, MAX_REWARD)

        with self._lock:
            credits = {}

            for memory_id, entry in self.traces.items():
                credit = reward * entry.value
                credits[memory_id] = credit
                self._total_credits_assigned += abs(credit)

            return credits

    def assign_credit_with_decay(
        self,
        reward: float,
        apply_decay: bool = True
    ) -> dict[str, float]:
        """
        Assign credit and optionally apply decay based on sign.

        Positive rewards strengthen traces (consolidate).
        Negative rewards weaken traces (anti-Hebbian).

        Args:
            reward: Reward signal
            apply_decay: Whether to apply LTD decay after negative rewards

        Returns:
            Credit assignments
        """
        with self._lock:
            credits = self.assign_credit(reward)

            # Optional: negative rewards can weaken traces
            if apply_decay and reward < 0:
                for memory_id in credits:
                    if memory_id in self.traces:
                        self.traces[memory_id].value *= (1 - self.a_minus)

            return credits

    def get_all_active(self, threshold: float = 0.01) -> dict[str, float]:
        """
        Get all traces above threshold.

        Args:
            threshold: Minimum trace value to include

        Returns:
            Dict mapping memory_id to trace value
        """
        with self._lock:
            return {
                entry.memory_id: entry.value
                for entry in self.traces.values()
                if entry.value >= threshold
            }

    def get_trace(self, memory_id: str) -> float:
        """Get current trace value for a memory."""
        with self._lock:
            if memory_id not in self.traces:
                return 0.0
            return self.traces[memory_id].value

    def clear(self):
        """Clear all traces."""
        with self._lock:
            self.traces.clear()

    def _evict_weakest(self):
        """Evict weakest trace to make room."""
        with self._lock:
            if not self.traces:
                return

            # Find weakest
            weakest = min(self.traces.values(), key=lambda e: e.value)
            del self.traces[weakest.memory_id]

    @property
    def count(self) -> int:
        """Number of active traces."""
        with self._lock:
            return len(self.traces)

    def get_stats(self) -> dict[str, Any]:
        """Get trace statistics."""
        with self._lock:
            if not self.traces:
                return {
                    "count": 0,
                    "total_updates": self._total_updates,
                    "total_credits_assigned": self._total_credits_assigned
                }

            values = [e.value for e in self.traces.values()]
            return {
                "count": len(self.traces),
                "mean_trace": np.mean(values),
                "max_trace": max(values),
                "min_trace": min(values),
                "total_updates": self._total_updates,
                "total_credits_assigned": self._total_credits_assigned
            }


class LayeredEligibilityTrace(EligibilityTrace):
    """
    Multi-layer eligibility trace with different time constants.

    Implements multiple trace buffers with different decay rates,
    capturing both short-term and long-term credit assignment.

    Similar to how synaptic tagging has early and late phases.
    """

    def __init__(
        self,
        fast_tau: float = 5.0,    # Fast trace (seconds)
        slow_tau: float = 60.0,    # Slow trace (minutes)
        fast_weight: float = 0.7,  # Weight for fast trace
        **kwargs
    ):
        """
        Initialize layered trace system.

        Args:
            fast_tau: Time constant for fast trace
            slow_tau: Time constant for slow trace
            fast_weight: Weight for fast trace in combined credit
        """
        super().__init__(**kwargs)

        self.fast_tau = fast_tau
        self.slow_tau = slow_tau
        self.fast_weight = fast_weight
        self.slow_weight = 1.0 - fast_weight

        # Separate trace stores
        self.fast_traces: dict[str, float] = {}
        self.slow_traces: dict[str, float] = {}

    def update(self, memory_id: str, activity: float = 1.0):
        """Update both fast and slow traces."""
        # Input validation
        if len(memory_id) > MAX_MEMORY_ID_LENGTH:
            raise ValueError(f"memory_id length ({len(memory_id)}) exceeds MAX_MEMORY_ID_LENGTH ({MAX_MEMORY_ID_LENGTH})")
        if not memory_id.isprintable():
            raise ValueError("memory_id must contain only printable characters")
        if not np.isfinite(activity):
            raise ValueError(f"activity must be finite, got {activity}")
        if activity < 0:
            raise ValueError(f"activity must be non-negative, got {activity}")
        if activity > MAX_ACTIVITY:
            raise ValueError(f"activity ({activity}) exceeds MAX_ACTIVITY ({MAX_ACTIVITY})")

        with self._lock:
            # Check capacity before adding new entry
            all_ids = set(self.fast_traces.keys()) | set(self.slow_traces.keys())
            if memory_id not in all_ids and len(all_ids) >= self.max_traces:
                self._evict_weakest_layered()

            # Update fast trace
            current = self.fast_traces.get(memory_id, 0.0)
            self.fast_traces[memory_id] = min(
                current + self.a_plus * activity, MAX_TRACE_VALUE
            )

            # Update slow trace
            current = self.slow_traces.get(memory_id, 0.0)
            self.slow_traces[memory_id] = min(
                current + self.a_plus * activity * 0.5,  # Slower accumulation
                MAX_TRACE_VALUE
            )

            self._total_updates += 1

    def step(self, dt: float | None = None):
        """Decay both trace layers."""
        with self._lock:
            current_time = time.time()
            if dt is None:
                dt = current_time - self._last_step_time
            else:
                # Input validation for dt
                if not np.isfinite(dt):
                    raise ValueError(f"dt must be finite, got {dt}")
                if dt < 0:
                    raise ValueError(f"dt must be non-negative, got {dt}")
                dt = min(dt, MAX_DT)
            self._last_step_time = current_time

            # Decay fast traces
            fast_decay = np.exp(-dt / self.fast_tau)
            for mid in list(self.fast_traces.keys()):
                self.fast_traces[mid] *= fast_decay
                if self.fast_traces[mid] < self.min_trace:
                    del self.fast_traces[mid]

            # Decay slow traces
            slow_decay = np.exp(-dt / self.slow_tau)
            for mid in list(self.slow_traces.keys()):
                self.slow_traces[mid] *= slow_decay
                if self.slow_traces[mid] < self.min_trace:
                    del self.slow_traces[mid]

    def assign_credit(self, reward: float) -> dict[str, float]:
        """Assign credit using weighted combination of traces."""
        # Input validation
        if not np.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")
        reward = np.clip(reward, -MAX_REWARD, MAX_REWARD)

        with self._lock:
            credits = {}

            # Combine all memory IDs
            all_ids = set(self.fast_traces.keys()) | set(self.slow_traces.keys())

            for memory_id in all_ids:
                fast_val = self.fast_traces.get(memory_id, 0.0)
                slow_val = self.slow_traces.get(memory_id, 0.0)
                combined = self.fast_weight * fast_val + self.slow_weight * slow_val
                credit = reward * combined
                credits[memory_id] = credit
                self._total_credits_assigned += abs(credit)

            return credits

    def get_all_active(self, threshold: float = 0.01) -> dict[str, float]:
        """Get combined traces above threshold."""
        with self._lock:
            all_ids = set(self.fast_traces.keys()) | set(self.slow_traces.keys())
            result = {}

            for memory_id in all_ids:
                fast_val = self.fast_traces.get(memory_id, 0.0)
                slow_val = self.slow_traces.get(memory_id, 0.0)
                combined = self.fast_weight * fast_val + self.slow_weight * slow_val
                if combined >= threshold:
                    result[memory_id] = combined

            return result

    @property
    def count(self) -> int:
        """Total unique memories with active traces."""
        with self._lock:
            return len(set(self.fast_traces.keys()) | set(self.slow_traces.keys()))

    def clear(self):
        """Clear all traces."""
        with self._lock:
            self.fast_traces.clear()
            self.slow_traces.clear()

    def _evict_weakest_layered(self):
        """Evict weakest combined trace to make room."""
        with self._lock:
            all_ids = set(self.fast_traces.keys()) | set(self.slow_traces.keys())
            if not all_ids:
                return

            # Find memory with weakest combined trace
            weakest_id = None
            weakest_value = float("inf")

            for memory_id in all_ids:
                fast_val = self.fast_traces.get(memory_id, 0.0)
                slow_val = self.slow_traces.get(memory_id, 0.0)
                combined = self.fast_weight * fast_val + self.slow_weight * slow_val

                if combined < weakest_value:
                    weakest_value = combined
                    weakest_id = memory_id

            # Remove from both traces
            if weakest_id:
                self.fast_traces.pop(weakest_id, None)
                self.slow_traces.pop(weakest_id, None)
