"""
Phase 11: Neurotransmitter Dashboard Demo API.

Provides real-time visualization of neuromodulatory states:
- DA (Dopamine): Reward prediction error, motivation
- 5-HT (Serotonin): Patience, long-term value
- ACh (Acetylcholine): Attention, learning rate
- NE (Norepinephrine): Arousal, uncertainty
- GABA: Inhibition, stability
- Glu (Glutamate): Excitation, plasticity

Biological basis:
- Schultz (1997): DA encodes RPE
- Doya (2002): Neuromodulator-metacontrol mapping
- Yu & Dayan (2005): ACh/NE uncertainty signaling
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

nt_router = APIRouter()


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class NeurotransmitterLevels(BaseModel):
    """Current levels of all neurotransmitters."""

    dopamine: float = Field(0.5, ge=0, le=1, description="DA: reward/motivation")
    serotonin: float = Field(0.5, ge=0, le=1, description="5-HT: patience/mood")
    acetylcholine: float = Field(0.5, ge=0, le=1, description="ACh: attention/learning")
    norepinephrine: float = Field(0.3, ge=0, le=1, description="NE: arousal/uncertainty")
    gaba: float = Field(0.4, ge=0, le=1, description="GABA: inhibition")
    glutamate: float = Field(0.5, ge=0, le=1, description="Glu: excitation")


class ReceptorSaturation(BaseModel):
    """Receptor saturation using Michaelis-Menten kinetics."""

    d1_saturation: float = Field(0.5, description="D1 receptor (direct pathway)")
    d2_saturation: float = Field(0.5, description="D2 receptor (indirect pathway)")
    alpha1_saturation: float = Field(0.3, description="Alpha-1 adrenergic")
    beta_saturation: float = Field(0.3, description="Beta adrenergic")
    m1_saturation: float = Field(0.5, description="M1 muscarinic (ACh)")
    nmda_saturation: float = Field(0.5, description="NMDA (Glu)")
    gaba_a_saturation: float = Field(0.4, description="GABA-A")


class CognitiveMode(BaseModel):
    """Current cognitive mode derived from NT balance."""

    mode: str = Field("balanced", description="explore/exploit/encode/retrieve/rest")
    confidence: float = Field(0.5, description="Mode classification confidence")
    exploration_drive: float = Field(0.5, description="Exploration tendency")
    exploitation_drive: float = Field(0.5, description="Exploitation tendency")
    learning_rate_mod: float = Field(1.0, description="Learning rate modifier")
    attention_gain: float = Field(1.0, description="Attention gain factor")


class NTDashboardState(BaseModel):
    """Complete NT dashboard state."""

    timestamp: datetime = Field(default_factory=datetime.now)
    levels: NeurotransmitterLevels = Field(default_factory=NeurotransmitterLevels)
    receptors: ReceptorSaturation = Field(default_factory=ReceptorSaturation)
    cognitive_mode: CognitiveMode = Field(default_factory=CognitiveMode)

    # Dynamics
    da_rpe: float = Field(0.0, description="Current RPE signal")
    ach_uncertainty: float = Field(0.0, description="Expected uncertainty")
    ne_surprise: float = Field(0.0, description="Unexpected uncertainty")

    # Time series (last 50 samples)
    da_history: list[float] = Field(default_factory=list)
    ach_history: list[float] = Field(default_factory=list)


class NTTraceEntry(BaseModel):
    """Single trace entry for NT dynamics."""

    timestamp: datetime
    nt_type: str
    value: float
    event: str | None = None


class InjectNTRequest(BaseModel):
    """Request to inject NT for demonstration."""

    nt_type: str = Field(..., description="NT type: da/5ht/ach/ne/gaba/glu")
    amount: float = Field(0.3, ge=-1, le=1, description="Injection amount (-1 to 1)")
    event_type: str | None = Field(None, description="Associated event type")


class InjectNTResponse(BaseModel):
    """Response from NT injection."""

    success: bool
    new_level: float
    cognitive_mode: str
    receptor_effects: dict


# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------

class NTStateManager:
    """Manages NT dashboard demo state."""

    # Michaelis-Menten parameters (Km values)
    KM_D1 = 0.3
    KM_D2 = 0.5
    KM_ALPHA1 = 0.4
    KM_BETA = 0.35
    KM_M1 = 0.4
    KM_NMDA = 0.45
    KM_GABA_A = 0.35

    def __init__(self):
        self._state = NTDashboardState()
        self._traces: list[NTTraceEntry] = []
        self._last_update = datetime.now()

    def get_state(self) -> NTDashboardState:
        """Get current dashboard state."""
        self._update_dynamics()
        return self._state

    def get_traces(self, limit: int = 100) -> list[NTTraceEntry]:
        """Get NT trace history."""
        return self._traces[-limit:]

    def inject_nt(
        self,
        nt_type: str,
        amount: float,
        event_type: str | None = None,
    ) -> InjectNTResponse:
        """Inject NT and observe effects."""
        nt_map = {
            "da": "dopamine",
            "5ht": "serotonin",
            "ach": "acetylcholine",
            "ne": "norepinephrine",
            "gaba": "gaba",
            "glu": "glutamate",
        }

        if nt_type not in nt_map:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown NT type: {nt_type}. Use: {list(nt_map.keys())}"
            )

        field_name = nt_map[nt_type]
        current = getattr(self._state.levels, field_name)
        new_level = np.clip(current + amount, 0.0, 1.0)
        setattr(self._state.levels, field_name, new_level)

        # Record trace
        self._traces.append(NTTraceEntry(
            timestamp=datetime.now(),
            nt_type=nt_type,
            value=new_level,
            event=event_type,
        ))
        if len(self._traces) > 500:
            self._traces = self._traces[-500:]

        # Update receptor saturation
        self._update_receptors()

        # Update cognitive mode
        self._update_cognitive_mode()

        return InjectNTResponse(
            success=True,
            new_level=float(new_level),
            cognitive_mode=self._state.cognitive_mode.mode,
            receptor_effects=self._get_receptor_dict(),
        )

    def _update_dynamics(self):
        """Update NT dynamics over time."""
        now = datetime.now()
        dt = (now - self._last_update).total_seconds()
        self._last_update = now

        if dt < 0.1:
            return

        # Natural decay toward baseline
        decay_rate = 0.1 * dt
        levels = self._state.levels

        levels.dopamine = levels.dopamine + (0.5 - levels.dopamine) * decay_rate
        levels.serotonin = levels.serotonin + (0.5 - levels.serotonin) * decay_rate
        levels.acetylcholine = levels.acetylcholine + (0.5 - levels.acetylcholine) * decay_rate
        levels.norepinephrine = levels.norepinephrine + (0.3 - levels.norepinephrine) * decay_rate
        levels.gaba = levels.gaba + (0.4 - levels.gaba) * decay_rate
        levels.glutamate = levels.glutamate + (0.5 - levels.glutamate) * decay_rate

        # Add small noise for realism
        noise_scale = 0.02
        levels.dopamine = np.clip(levels.dopamine + np.random.normal(0, noise_scale), 0, 1)
        levels.serotonin = np.clip(levels.serotonin + np.random.normal(0, noise_scale), 0, 1)
        levels.acetylcholine = np.clip(levels.acetylcholine + np.random.normal(0, noise_scale), 0, 1)
        levels.norepinephrine = np.clip(levels.norepinephrine + np.random.normal(0, noise_scale), 0, 1)

        # Update history
        self._state.da_history.append(float(levels.dopamine))
        self._state.ach_history.append(float(levels.acetylcholine))
        if len(self._state.da_history) > 50:
            self._state.da_history = self._state.da_history[-50:]
        if len(self._state.ach_history) > 50:
            self._state.ach_history = self._state.ach_history[-50:]

        # Update derived states
        self._update_receptors()
        self._update_cognitive_mode()
        self._update_signals()
        self._state.timestamp = now

    def _michaelis_menten(self, concentration: float, km: float) -> float:
        """Compute receptor saturation via Michaelis-Menten."""
        return concentration / (km + concentration)

    def _update_receptors(self):
        """Update receptor saturation levels."""
        levels = self._state.levels
        receptors = self._state.receptors

        receptors.d1_saturation = self._michaelis_menten(levels.dopamine, self.KM_D1)
        receptors.d2_saturation = self._michaelis_menten(levels.dopamine, self.KM_D2)
        receptors.alpha1_saturation = self._michaelis_menten(levels.norepinephrine, self.KM_ALPHA1)
        receptors.beta_saturation = self._michaelis_menten(levels.norepinephrine, self.KM_BETA)
        receptors.m1_saturation = self._michaelis_menten(levels.acetylcholine, self.KM_M1)
        receptors.nmda_saturation = self._michaelis_menten(levels.glutamate, self.KM_NMDA)
        receptors.gaba_a_saturation = self._michaelis_menten(levels.gaba, self.KM_GABA_A)

    def _update_cognitive_mode(self):
        """Determine cognitive mode from NT balance."""
        levels = self._state.levels
        mode = self._state.cognitive_mode

        # Doya (2002) mapping
        # DA: gain/vigor, 5-HT: discount rate, ACh: learning rate, NE: exploration

        # Exploration vs exploitation
        mode.exploration_drive = (
            0.4 * levels.norepinephrine +
            0.3 * (1 - levels.dopamine) +
            0.3 * levels.acetylcholine
        )
        mode.exploitation_drive = (
            0.5 * levels.dopamine +
            0.3 * levels.serotonin +
            0.2 * (1 - levels.norepinephrine)
        )

        # Learning rate modifier (ACh-driven)
        mode.learning_rate_mod = 0.5 + levels.acetylcholine

        # Attention gain (ACh/NE)
        mode.attention_gain = 0.5 + 0.3 * levels.acetylcholine + 0.2 * levels.norepinephrine

        # Mode classification
        if levels.norepinephrine > 0.7:
            detected_mode = "explore"
            confidence = levels.norepinephrine
        elif levels.dopamine > 0.7 and levels.serotonin > 0.5:
            detected_mode = "exploit"
            confidence = levels.dopamine
        elif levels.acetylcholine > 0.7:
            detected_mode = "encode"
            confidence = levels.acetylcholine
        elif levels.dopamine > 0.6 and levels.acetylcholine > 0.5:
            detected_mode = "retrieve"
            confidence = (levels.dopamine + levels.acetylcholine) / 2
        elif levels.gaba > 0.6 and levels.norepinephrine < 0.3:
            detected_mode = "rest"
            confidence = levels.gaba
        else:
            detected_mode = "balanced"
            confidence = 0.5

        mode.mode = detected_mode
        mode.confidence = float(np.clip(confidence, 0, 1))

    def _update_signals(self):
        """Update derived signals (RPE, uncertainty)."""
        levels = self._state.levels

        # RPE approximation (DA deviation from baseline)
        self._state.da_rpe = (levels.dopamine - 0.5) * 2

        # Expected uncertainty (ACh-driven)
        self._state.ach_uncertainty = levels.acetylcholine * 0.8

        # Unexpected uncertainty (NE-driven)
        self._state.ne_surprise = levels.norepinephrine * 0.9

    def _get_receptor_dict(self) -> dict:
        """Get receptor saturation as dict."""
        r = self._state.receptors
        return {
            "d1": float(r.d1_saturation),
            "d2": float(r.d2_saturation),
            "alpha1": float(r.alpha1_saturation),
            "beta": float(r.beta_saturation),
            "m1": float(r.m1_saturation),
            "nmda": float(r.nmda_saturation),
            "gaba_a": float(r.gaba_a_saturation),
        }

    def reset(self):
        """Reset to baseline state."""
        self._state = NTDashboardState()
        self._traces.clear()
        self._last_update = datetime.now()


# Global state manager
_nt_state = NTStateManager()


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@nt_router.get(
    "/state",
    response_model=NTDashboardState,
    summary="Get NT dashboard state",
    description="Get current neuromodulator levels and cognitive mode",
)
async def get_nt_state():
    """Get current NT dashboard state."""
    return _nt_state.get_state()


@nt_router.get(
    "/traces",
    summary="Get NT traces",
    description="Get history of NT level changes",
)
async def get_nt_traces(limit: int = 100):
    """Get NT trace history."""
    traces = _nt_state.get_traces(limit)
    return {
        "traces": [t.model_dump() for t in traces],
        "count": len(traces),
    }


@nt_router.post(
    "/inject",
    response_model=InjectNTResponse,
    summary="Inject NT",
    description="Inject neurotransmitter and observe effects",
)
async def inject_nt(request: InjectNTRequest):
    """
    Inject NT for demonstration.

    Demonstrates Doya (2002) neuromodulator-metacontrol mapping:
    - DA injection: Increases exploitation, reward sensitivity
    - NE injection: Increases exploration, arousal
    - ACh injection: Increases learning rate, attention
    - 5-HT injection: Increases patience, long-term focus
    """
    return _nt_state.inject_nt(
        nt_type=request.nt_type,
        amount=request.amount,
        event_type=request.event_type,
    )


@nt_router.post(
    "/reset",
    summary="Reset NT state",
    description="Reset all NT levels to baseline",
)
async def reset_nt_state():
    """Reset NT state to baseline."""
    _nt_state.reset()
    return {"status": "reset", "message": "NT levels reset to baseline"}


@nt_router.get(
    "/receptors",
    response_model=ReceptorSaturation,
    summary="Get receptor saturation",
    description="Get current receptor saturation via Michaelis-Menten kinetics",
)
async def get_receptor_saturation():
    """Get receptor saturation levels."""
    _nt_state._update_dynamics()
    return _nt_state._state.receptors


@nt_router.get(
    "/mode",
    response_model=CognitiveMode,
    summary="Get cognitive mode",
    description="Get current cognitive mode derived from NT balance",
)
async def get_cognitive_mode():
    """Get cognitive mode classification."""
    _nt_state._update_dynamics()
    return _nt_state._state.cognitive_mode


__all__ = ["nt_router"]
