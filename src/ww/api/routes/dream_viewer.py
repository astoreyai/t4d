"""
Phase 11: Dream Viewer Demo API.

Provides visualization of sleep consolidation processes:
- SWR (Sharp Wave Ripple) replay during NREM
- Memory recombination during REM
- Glymphatic clearance during deep NREM
- Memory pruning and stabilization

Biological basis:
- Stickgold (2005): NREM/REM ratio ~75:25
- Buzsaki (2015): SWR replay at 1-4x speed
- Xie et al. (2013): Glymphatic clearance during sleep
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

dream_router = APIRouter()


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class ReplaySequence(BaseModel):
    """A memory replay sequence during sleep."""
    sequence_id: str
    memory_ids: list[str]
    replay_speed: float = Field(2.0, description="Replay speed (1-4x original)")
    phase: str = Field("nrem", description="Sleep phase: nrem/rem")
    timestamp: datetime


class DreamViewerState(BaseModel):
    """Current state of the dream viewer."""

    # Sleep phase
    current_phase: str = Field("wake", description="Phase: wake/nrem_light/nrem_deep/rem")
    phase_progress: float = Field(0.0, description="Progress through current phase (0-1)")
    cycle_number: int = Field(0, description="Current sleep cycle")

    # Adenosine dynamics
    adenosine_level: float = Field(0.3, description="Sleep pressure (0-1)")
    clearance_rate: float = Field(0.0, description="Glymphatic clearance rate")

    # Replay events
    active_replays: int = Field(0, description="Number of active replay sequences")
    replay_count_total: int = Field(0, description="Total replays this session")

    # Prediction error
    mean_prediction_error: float = Field(0.0, description="Mean prediction error")

    # Glymphatic
    glymphatic_flow_rate: float = Field(0.0, description="Flow rate (arbitrary units)")
    aqp4_activity: float = Field(0.0, description="AQP4 channel activity (0-1)")
    waste_clearance: float = Field(0.0, description="Cumulative waste cleared")


class StartSleepRequest(BaseModel):
    """Request to start simulated sleep."""
    duration_cycles: int = Field(2, description="Number of sleep cycles (1-5)")
    initial_adenosine: float = Field(0.7, description="Initial adenosine level (0-1)")


class StartSleepResponse(BaseModel):
    """Response from starting sleep."""
    success: bool
    session_id: str
    expected_duration_ms: int
    phases_planned: list[str]


class ReplayHistoryResponse(BaseModel):
    """Replay event history."""
    replays: list[ReplaySequence]
    total_count: int
    phase_breakdown: dict


# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------

class DreamStateManager:
    """Manages demo state for dream viewer."""

    def __init__(self):
        self._state = DreamViewerState()
        self._replays: list[ReplaySequence] = []
        self._sleep_task: asyncio.Task | None = None
        self._session_id: str | None = None
        self._running = False

    def get_state(self) -> DreamViewerState:
        return self._state

    def get_replays(self) -> list[ReplaySequence]:
        return self._replays[-50:]  # Last 50 replays

    async def start_sleep(
        self,
        duration_cycles: int = 2,
        initial_adenosine: float = 0.7
    ) -> StartSleepResponse:
        """Start simulated sleep session."""
        if self._running:
            raise HTTPException(status_code=400, detail="Sleep already in progress")

        import uuid
        self._session_id = str(uuid.uuid4())
        self._state.adenosine_level = initial_adenosine
        self._running = True

        # Plan phases
        phases = []
        for cycle in range(duration_cycles):
            phases.extend(["nrem_light", "nrem_deep", "rem"])

        # Start background task
        self._sleep_task = asyncio.create_task(
            self._run_sleep_simulation(duration_cycles)
        )

        return StartSleepResponse(
            success=True,
            session_id=self._session_id,
            expected_duration_ms=duration_cycles * 10000,  # 10s per cycle for demo
            phases_planned=phases,
        )

    async def _run_sleep_simulation(self, cycles: int):
        """Run sleep simulation in background."""
        try:
            for cycle in range(cycles):
                self._state.cycle_number = cycle + 1

                # NREM Light (~50% of cycle)
                await self._simulate_phase("nrem_light", duration=2.0)

                # NREM Deep (~25% of cycle)
                await self._simulate_phase("nrem_deep", duration=1.5)

                # REM (~25% of cycle)
                await self._simulate_phase("rem", duration=1.5)

            # Return to wake
            self._state.current_phase = "wake"
            self._state.phase_progress = 0.0

        finally:
            self._running = False

    async def _simulate_phase(self, phase: str, duration: float):
        """Simulate a single sleep phase."""
        self._state.current_phase = phase
        steps = int(duration * 10)

        for step in range(steps):
            if not self._running:
                return

            self._state.phase_progress = step / steps

            # Phase-specific dynamics
            if phase == "nrem_deep":
                # High glymphatic activity
                self._state.glymphatic_flow_rate = 0.8 + np.random.uniform(-0.1, 0.1)
                self._state.aqp4_activity = 0.9
                self._state.clearance_rate = 0.7
                self._state.waste_clearance += 0.01

                # SWR replays
                if np.random.rand() < 0.3:
                    self._generate_replay("nrem")

            elif phase == "nrem_light":
                self._state.glymphatic_flow_rate = 0.4
                self._state.aqp4_activity = 0.5
                self._state.clearance_rate = 0.3

                if np.random.rand() < 0.2:
                    self._generate_replay("nrem")

            elif phase == "rem":
                # Low glymphatic, high recombination
                self._state.glymphatic_flow_rate = 0.1
                self._state.aqp4_activity = 0.2
                self._state.clearance_rate = 0.1

                if np.random.rand() < 0.4:
                    self._generate_replay("rem")

            # Adenosine clearance
            self._state.adenosine_level = max(
                0.1,
                self._state.adenosine_level - 0.005
            )

            await asyncio.sleep(0.1)

    def _generate_replay(self, phase: str):
        """Generate a replay event."""
        import uuid

        replay = ReplaySequence(
            sequence_id=str(uuid.uuid4()),
            memory_ids=[str(uuid.uuid4()) for _ in range(np.random.randint(2, 6))],
            replay_speed=np.random.uniform(1.5, 4.0) if phase == "nrem" else 1.0,
            phase=phase,
            timestamp=datetime.now(),
        )

        self._replays.append(replay)
        if len(self._replays) > 200:
            self._replays = self._replays[-200:]

        self._state.active_replays = min(5, self._state.active_replays + 1)
        self._state.replay_count_total += 1

        # Decay active replays
        asyncio.get_event_loop().call_later(
            1.0,
            lambda: setattr(
                self._state,
                'active_replays',
                max(0, self._state.active_replays - 1)
            )
        )

    def stop_sleep(self):
        """Stop sleep simulation."""
        self._running = False
        if self._sleep_task and not self._sleep_task.done():
            self._sleep_task.cancel()
        self._state.current_phase = "wake"


# Global state manager
_dream_state = DreamStateManager()


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@dream_router.get(
    "/state",
    response_model=DreamViewerState,
    summary="Get dream viewer state",
    description="Get current state of sleep consolidation visualization",
)
async def get_dream_state():
    """Get current dream viewer state."""
    return _dream_state.get_state()


@dream_router.post(
    "/start",
    response_model=StartSleepResponse,
    summary="Start sleep simulation",
    description="Start simulated sleep for consolidation visualization",
)
async def start_sleep(request: StartSleepRequest):
    """
    Start sleep simulation.

    Simulates sleep cycles with:
    - NREM light phase
    - NREM deep phase (high SWR, glymphatic)
    - REM phase (memory recombination)
    """
    return await _dream_state.start_sleep(
        duration_cycles=min(request.duration_cycles, 5),
        initial_adenosine=request.initial_adenosine,
    )


@dream_router.post(
    "/stop",
    summary="Stop sleep simulation",
    description="Wake up and stop the sleep simulation",
)
async def stop_sleep():
    """Stop sleep simulation."""
    _dream_state.stop_sleep()
    return {"status": "awake", "message": "Sleep simulation stopped"}


@dream_router.get(
    "/replays",
    response_model=ReplayHistoryResponse,
    summary="Get replay history",
    description="Get history of memory replay events during sleep",
)
async def get_replays():
    """Get replay event history."""
    replays = _dream_state.get_replays()

    # Count by phase
    phase_breakdown = {"nrem": 0, "rem": 0}
    for r in replays:
        if r.phase in phase_breakdown:
            phase_breakdown[r.phase] += 1

    return ReplayHistoryResponse(
        replays=replays,
        total_count=len(replays),
        phase_breakdown=phase_breakdown,
    )


__all__ = ["dream_router"]
