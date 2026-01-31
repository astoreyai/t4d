"""
Phase 11: Learning Trace Demo API.

Provides visualization of learning dynamics:
- Eligibility traces (synaptic tags)
- STDP timing windows
- Three-factor learning (pre, post, neuromodulator)
- BCM metaplasticity threshold

Biological basis:
- Gerstner et al. (2018): Three-factor learning rules
- Bi & Poo (1998): STDP timing dependence
- Bienenstock et al. (1982): BCM sliding threshold
- Frey & Morris (1997): Synaptic tagging and capture
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

learning_router = APIRouter()


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class STDPWindow(BaseModel):
    """STDP timing window parameters."""

    tau_plus: float = Field(20.0, description="LTP time constant (ms)")
    tau_minus: float = Field(20.0, description="LTD time constant (ms)")
    a_plus: float = Field(0.1, description="LTP amplitude")
    a_minus: float = Field(0.12, description="LTD amplitude")
    current_delta_t: float = Field(0.0, description="Current spike timing delta (ms)")
    current_weight_change: float = Field(0.0, description="Resulting weight change")


class EligibilityTrace(BaseModel):
    """Eligibility trace (synaptic tag) state."""

    trace_id: str
    synapse_pair: tuple[int, int] = Field(..., description="(pre, post) neuron indices")
    trace_value: float = Field(0.0, description="Current trace amplitude")
    decay_tau: float = Field(1000.0, description="Decay time constant (ms)")
    created_at: datetime
    last_spike: datetime | None = None
    tag_type: str = Field("early", description="early/late protein tag")


class ThreeFactorState(BaseModel):
    """Three-factor learning state."""

    pre_activity: float = Field(0.0, description="Presynaptic activity")
    post_activity: float = Field(0.0, description="Postsynaptic activity")
    modulator_signal: float = Field(0.0, description="Neuromodulator (DA/ACh)")
    eligibility: float = Field(0.0, description="Eligibility trace")
    weight_change: float = Field(0.0, description="Computed dw")
    learning_rule: str = Field("hebbian", description="hebbian/anti_hebbian/reward_modulated")


class BCMState(BaseModel):
    """BCM metaplasticity state."""

    theta_m: float = Field(0.5, description="Sliding threshold")
    mean_activity: float = Field(0.5, description="Recent mean postsynaptic activity")
    integration_tau: float = Field(10000.0, description="Threshold integration time (ms)")
    ltp_zone: bool = Field(False, description="Above threshold (LTP)")
    ltd_zone: bool = Field(False, description="Below threshold (LTD)")


class LearningDashboardState(BaseModel):
    """Complete learning trace dashboard state."""

    timestamp: datetime = Field(default_factory=datetime.now)
    stdp: STDPWindow = Field(default_factory=STDPWindow)
    bcm: BCMState = Field(default_factory=BCMState)
    three_factor: ThreeFactorState = Field(default_factory=ThreeFactorState)

    # Active traces
    active_traces: int = Field(0, description="Number of active eligibility traces")
    total_weight_updates: int = Field(0, description="Total weight updates this session")

    # Visualization data
    stdp_curve_x: list[float] = Field(default_factory=list, description="Delta t values")
    stdp_curve_y: list[float] = Field(default_factory=list, description="Weight change values")
    weight_history: list[float] = Field(default_factory=list, description="Weight trajectory")


class SpikeEventRequest(BaseModel):
    """Request to simulate spike event."""

    pre_time: float = Field(..., description="Presynaptic spike time (ms)")
    post_time: float = Field(..., description="Postsynaptic spike time (ms)")
    modulator_level: float = Field(0.5, ge=0, le=1, description="Neuromodulator level")


class SpikeEventResponse(BaseModel):
    """Response from spike event simulation."""

    delta_t: float
    stdp_change: float
    three_factor_change: float
    eligibility_created: bool
    trace_id: str | None


class TraceDecayRequest(BaseModel):
    """Request to simulate trace decay."""

    time_elapsed_ms: float = Field(100.0, description="Time elapsed in ms")


# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------

class LearningStateManager:
    """Manages learning trace demo state."""

    def __init__(self):
        self._state = LearningDashboardState()
        self._traces: list[EligibilityTrace] = []
        self._weight = 0.5  # Demo synapse weight
        self._last_update = datetime.now()

        # Pre-compute STDP curve for visualization
        self._compute_stdp_curve()

    def _compute_stdp_curve(self):
        """Compute STDP curve for visualization."""
        delta_ts = np.linspace(-100, 100, 201)
        weight_changes = []

        for dt in delta_ts:
            if dt > 0:
                # Post after pre -> LTP
                dw = self._state.stdp.a_plus * np.exp(-dt / self._state.stdp.tau_plus)
            else:
                # Pre after post -> LTD
                dw = -self._state.stdp.a_minus * np.exp(dt / self._state.stdp.tau_minus)

            weight_changes.append(float(dw))

        self._state.stdp_curve_x = delta_ts.tolist()
        self._state.stdp_curve_y = weight_changes

    def get_state(self) -> LearningDashboardState:
        """Get current learning state."""
        self._update_dynamics()
        return self._state

    def get_traces(self) -> list[EligibilityTrace]:
        """Get active eligibility traces."""
        return self._traces

    def simulate_spike_event(
        self,
        pre_time: float,
        post_time: float,
        modulator_level: float = 0.5,
    ) -> SpikeEventResponse:
        """Simulate a spike event and compute learning."""
        delta_t = post_time - pre_time

        # STDP weight change
        if delta_t > 0:
            stdp_change = self._state.stdp.a_plus * np.exp(-delta_t / self._state.stdp.tau_plus)
        else:
            stdp_change = -self._state.stdp.a_minus * np.exp(delta_t / self._state.stdp.tau_minus)

        # Update STDP state
        self._state.stdp.current_delta_t = delta_t
        self._state.stdp.current_weight_change = float(stdp_change)

        # Three-factor learning: eligibility * modulator
        # Eligibility from STDP, gated by modulator
        eligibility = abs(stdp_change)
        three_factor_change = eligibility * (modulator_level - 0.5) * 2 * np.sign(stdp_change)

        # Update three-factor state
        self._state.three_factor.pre_activity = 1.0 if pre_time <= post_time else 0.5
        self._state.three_factor.post_activity = 1.0 if post_time >= pre_time else 0.5
        self._state.three_factor.modulator_signal = modulator_level
        self._state.three_factor.eligibility = float(eligibility)
        self._state.three_factor.weight_change = float(three_factor_change)

        # Create eligibility trace
        import uuid
        trace_id = str(uuid.uuid4())[:8]
        trace = EligibilityTrace(
            trace_id=trace_id,
            synapse_pair=(0, 1),
            trace_value=float(eligibility),
            created_at=datetime.now(),
            last_spike=datetime.now(),
            tag_type="early" if eligibility < 0.5 else "late",
        )
        self._traces.append(trace)
        if len(self._traces) > 50:
            self._traces = self._traces[-50:]

        self._state.active_traces = len(self._traces)

        # Apply weight change
        self._weight = np.clip(self._weight + three_factor_change * 0.1, 0.0, 1.0)
        self._state.weight_history.append(float(self._weight))
        if len(self._state.weight_history) > 100:
            self._state.weight_history = self._state.weight_history[-100:]

        self._state.total_weight_updates += 1

        # Update BCM threshold
        self._update_bcm(self._state.three_factor.post_activity)

        return SpikeEventResponse(
            delta_t=delta_t,
            stdp_change=float(stdp_change),
            three_factor_change=float(three_factor_change),
            eligibility_created=True,
            trace_id=trace_id,
        )

    def decay_traces(self, time_elapsed_ms: float):
        """Decay all eligibility traces."""
        for trace in self._traces:
            decay = np.exp(-time_elapsed_ms / trace.decay_tau)
            trace.trace_value *= decay

        # Remove expired traces
        self._traces = [t for t in self._traces if t.trace_value > 0.01]
        self._state.active_traces = len(self._traces)

    def _update_bcm(self, post_activity: float):
        """Update BCM sliding threshold."""
        bcm = self._state.bcm

        # Integrate mean activity
        alpha = 0.1  # Integration rate
        bcm.mean_activity = (1 - alpha) * bcm.mean_activity + alpha * post_activity

        # Sliding threshold (BCM rule: theta = <y^2>)
        bcm.theta_m = bcm.mean_activity ** 2 + 0.1  # Add baseline

        # Determine zone
        bcm.ltp_zone = post_activity > bcm.theta_m
        bcm.ltd_zone = post_activity < bcm.theta_m and post_activity > 0

    def _update_dynamics(self):
        """Update trace decay over time."""
        now = datetime.now()
        dt_ms = (now - self._last_update).total_seconds() * 1000
        self._last_update = now

        if dt_ms > 10:
            self.decay_traces(dt_ms)

        self._state.timestamp = now

    def reset(self):
        """Reset learning state."""
        self._state = LearningDashboardState()
        self._traces.clear()
        self._weight = 0.5
        self._last_update = datetime.now()
        self._compute_stdp_curve()


# Global state manager
_learning_state = LearningStateManager()


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@learning_router.get(
    "/state",
    response_model=LearningDashboardState,
    summary="Get learning dashboard state",
    description="Get current STDP, BCM, and three-factor learning states",
)
async def get_learning_state():
    """Get current learning dashboard state."""
    return _learning_state.get_state()


@learning_router.post(
    "/spike",
    response_model=SpikeEventResponse,
    summary="Simulate spike event",
    description="Simulate pre/post spike pair and compute learning",
)
async def simulate_spike(request: SpikeEventRequest):
    """
    Simulate spike event.

    Demonstrates:
    - STDP: Timing-dependent plasticity (Bi & Poo, 1998)
    - Three-factor: Eligibility * Modulator (Gerstner, 2018)
    - Synaptic tagging: Early/late protein synthesis tags
    """
    return _learning_state.simulate_spike_event(
        pre_time=request.pre_time,
        post_time=request.post_time,
        modulator_level=request.modulator_level,
    )


@learning_router.post(
    "/decay",
    summary="Simulate trace decay",
    description="Advance time and decay eligibility traces",
)
async def decay_traces(request: TraceDecayRequest):
    """Decay eligibility traces."""
    _learning_state.decay_traces(request.time_elapsed_ms)
    return {
        "time_elapsed_ms": request.time_elapsed_ms,
        "active_traces": _learning_state._state.active_traces,
    }


@learning_router.get(
    "/traces",
    summary="Get eligibility traces",
    description="Get all active eligibility traces",
)
async def get_traces():
    """Get active eligibility traces."""
    traces = _learning_state.get_traces()
    return {
        "traces": [t.model_dump() for t in traces],
        "count": len(traces),
    }


@learning_router.get(
    "/stdp-curve",
    summary="Get STDP curve",
    description="Get pre-computed STDP timing curve for visualization",
)
async def get_stdp_curve():
    """Get STDP timing curve."""
    state = _learning_state.get_state()
    return {
        "delta_t": state.stdp_curve_x,
        "weight_change": state.stdp_curve_y,
        "tau_plus": state.stdp.tau_plus,
        "tau_minus": state.stdp.tau_minus,
    }


@learning_router.get(
    "/bcm",
    response_model=BCMState,
    summary="Get BCM state",
    description="Get BCM metaplasticity threshold state",
)
async def get_bcm_state():
    """Get BCM metaplasticity state."""
    _learning_state._update_dynamics()
    return _learning_state._state.bcm


@learning_router.post(
    "/reset",
    summary="Reset learning state",
    description="Reset all learning traces and weights",
)
async def reset_learning_state():
    """Reset learning state."""
    _learning_state.reset()
    return {"status": "reset", "message": "Learning state reset"}


__all__ = ["learning_router"]
