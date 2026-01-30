"""
Astrocyte glial layer for NCA.

Implements tripartite synapse model with:
1. EAAT-2 (GLT-1): Glutamate reuptake (~90% of synaptic clearance)
2. GAT-3: GABA reuptake (modulates inhibitory tone)
3. Gliotransmission: Astrocyte-released glutamate, D-serine, ATP
4. Calcium dynamics: Slow astrocyte calcium waves

Biology:
- Astrocytes wrap ~100,000 synapses each
- EAAT-2 prevents excitotoxicity (glutamate buildup = cell death)
- GAT-3 regulates tonic inhibition
- Gliotransmitter release modulates synaptic strength
- Astrocyte dysfunction implicated in epilepsy, ALS, Alzheimer's

References:
- Araque et al. (2014) Gliotransmitters travel in time and space
- Volterra & Meldolesi (2005) Astrocytes, from brain glue to communication elements
- Murphy-Royal et al. (2017) Surface diffusion of astrocytic glutamate transporters
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class AstrocyteState(Enum):
    """Astrocyte activation states."""
    QUIESCENT = "quiescent"      # Low activity, baseline reuptake
    ACTIVATED = "activated"      # Elevated Ca2+, enhanced reuptake
    REACTIVE = "reactive"        # Pathological, impaired function


@dataclass
class AstrocyteConfig:
    """Configuration for astrocyte layer.

    Parameters tuned from literature:
    - EAAT-2 Km ~20-50 µM glutamate (Trotti et al. 1997)
    - GAT-3 Km ~10-20 µM GABA (Borden 1996)
    - Astrocyte Ca2+ waves: 10-20 µm/s, period ~20-60s (Cornell-Bell et al. 1990)
    """

    # EAAT-2 glutamate transporter parameters (Michaelis-Menten kinetics)
    # Trotti et al. (1997): Km ~25µM, Vmax ~1.5 nmol/mg/min
    eaat2_vmax: float = 0.8         # Max reuptake rate (normalized)
    eaat2_km: float = 0.3           # Michaelis constant (normalized, ~30µM)
    eaat2_baseline: float = 0.1     # Baseline reuptake even at low [Glu]

    # GAT-3 GABA transporter parameters
    gat3_vmax: float = 0.5          # Max GABA reuptake rate
    gat3_km: float = 0.2            # Michaelis constant
    gat3_baseline: float = 0.05     # Baseline GABA clearance

    # Gliotransmission parameters
    gliotx_threshold: float = 0.6   # Ca2+ threshold for gliotransmitter release
    gliotx_glutamate: float = 0.15  # Glutamate release amplitude
    gliotx_dserine: float = 0.1     # D-serine release (NMDA potentiation)
    gliotx_atp: float = 0.08        # ATP release (-> adenosine)

    # Calcium dynamics
    ca_rise_rate: float = 0.1       # Ca2+ rise rate on activation
    ca_decay_rate: float = 0.02     # Ca2+ decay rate (slow, ~seconds)
    ca_threshold: float = 0.4       # Threshold for state transition
    ca_wave_speed: float = 0.05     # Spatial propagation rate

    # Metabolic coupling
    lactate_production: float = 0.1  # Lactate shuttle to neurons
    glycogen_buffer: float = 1.0     # Energy buffer capacity

    # Pathology thresholds (Arundine & Tymianski 2003)
    excitotoxicity_threshold: float = 100.0  # µM glutamate causing damage (normalized: 0.9)
    reactive_threshold: float = 0.8         # Ca2+ level for reactive state

    # Timing
    dt: float = 1.0  # Integration timestep (ms)


@dataclass
class AstrocyteLayerState:
    """Current state of astrocyte layer."""

    calcium: float = 0.1            # Intracellular Ca2+ (0-1)
    glutamate_buffered: float = 0.0 # Glu taken up, not yet metabolized
    gaba_buffered: float = 0.0      # GABA taken up
    glycogen: float = 1.0           # Energy reserve (0-1)
    activation_state: AstrocyteState = AstrocyteState.QUIESCENT

    # Gliotransmitter release state
    last_release_time: float = 0.0
    release_refractory: bool = False

    # Running statistics
    total_glu_cleared: float = 0.0
    total_gaba_cleared: float = 0.0
    excitotoxicity_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize state."""
        return {
            "calcium": self.calcium,
            "glutamate_buffered": self.glutamate_buffered,
            "gaba_buffered": self.gaba_buffered,
            "glycogen": self.glycogen,
            "activation_state": self.activation_state.value,
            "total_glu_cleared": self.total_glu_cleared,
            "total_gaba_cleared": self.total_gaba_cleared,
            "excitotoxicity_events": self.excitotoxicity_events,
        }


class AstrocyteLayer:
    """
    Astrocyte glial layer implementing tripartite synapse dynamics.

    Key functions:
    1. Glutamate clearance via EAAT-2 (prevents excitotoxicity)
    2. GABA clearance via GAT-3 (modulates inhibition)
    3. Gliotransmitter release (glutamate, D-serine, ATP)
    4. Metabolic support (lactate shuttle)

    Example:
        >>> astro = AstrocyteLayer()
        >>> glu_cleared, gaba_cleared = astro.compute_reuptake(glu=0.7, gaba=0.4)
        >>> gliotx = astro.compute_gliotransmission()
    """

    def __init__(self, config: AstrocyteConfig | None = None):
        self.config = config or AstrocyteConfig()
        self.state = AstrocyteLayerState()
        self._step_count = 0

        # History for analysis
        self._ca_history: deque = deque(maxlen=1000)
        self._glu_history: deque = deque(maxlen=1000)
        self._gaba_history: deque = deque(maxlen=1000)

    def compute_reuptake(
        self,
        glutamate: float,
        gaba: float,
        activity_level: float = 0.5
    ) -> tuple[float, float]:
        """
        Compute neurotransmitter reuptake via astrocyte transporters.

        Uses Michaelis-Menten kinetics for both EAAT-2 and GAT-3.

        Args:
            glutamate: Extracellular glutamate level (0-1)
            gaba: Extracellular GABA level (0-1)
            activity_level: Neural activity (modulates transporter expression)

        Returns:
            (glutamate_cleared, gaba_cleared): Amount removed from synapse
        """
        cfg = self.config

        # Activity-dependent transporter upregulation
        # High activity -> more transporters (homeostatic)
        activity_mod = 1.0 + 0.3 * (activity_level - 0.5)

        # State-dependent efficiency
        state_mod = self._get_state_efficiency()

        # EAAT-2: Michaelis-Menten glutamate reuptake
        # V = Vmax * [S] / (Km + [S])
        glu_reuptake = (
            cfg.eaat2_vmax * activity_mod * state_mod *
            glutamate / (cfg.eaat2_km + glutamate + 1e-10)
        )
        glu_reuptake = max(glu_reuptake, cfg.eaat2_baseline * glutamate)

        # GAT-3: Michaelis-Menten GABA reuptake
        gaba_reuptake = (
            cfg.gat3_vmax * activity_mod * state_mod *
            gaba / (cfg.gat3_km + gaba + 1e-10)
        )
        gaba_reuptake = max(gaba_reuptake, cfg.gat3_baseline * gaba)

        # Update internal buffers
        self.state.glutamate_buffered += glu_reuptake * 0.1  # Slow metabolism
        self.state.gaba_buffered += gaba_reuptake * 0.1

        # Metabolize buffered transmitters (convert to glutamine, etc.)
        self.state.glutamate_buffered *= 0.95
        self.state.gaba_buffered *= 0.95

        # Track totals
        self.state.total_glu_cleared += glu_reuptake
        self.state.total_gaba_cleared += gaba_reuptake

        # Check for excitotoxicity
        if glutamate > cfg.excitotoxicity_threshold:
            self.state.excitotoxicity_events += 1

        # Update calcium based on activity
        self._update_calcium(glutamate, activity_level)

        # Update histories
        self._glu_history.append(glutamate)
        self._gaba_history.append(gaba)
        self._ca_history.append(self.state.calcium)

        self._step_count += 1

        return glu_reuptake, gaba_reuptake

    def compute_reuptake_field(
        self,
        glutamate_field: np.ndarray,
        gaba_field: np.ndarray,
        activity_field: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute reuptake across spatial field.

        Args:
            glutamate_field: Spatial glutamate concentration
            gaba_field: Spatial GABA concentration
            activity_field: Optional spatial activity pattern

        Returns:
            (glu_cleared_field, gaba_cleared_field)
        """
        cfg = self.config

        if activity_field is None:
            activity_field = np.full_like(glutamate_field, 0.5)

        activity_mod = 1.0 + 0.3 * (activity_field - 0.5)
        state_mod = self._get_state_efficiency()

        # Vectorized Michaelis-Menten
        glu_reuptake = (
            cfg.eaat2_vmax * activity_mod * state_mod *
            glutamate_field / (cfg.eaat2_km + glutamate_field + 1e-10)
        )
        glu_reuptake = np.maximum(glu_reuptake, cfg.eaat2_baseline * glutamate_field)

        gaba_reuptake = (
            cfg.gat3_vmax * activity_mod * state_mod *
            gaba_field / (cfg.gat3_km + gaba_field + 1e-10)
        )
        gaba_reuptake = np.maximum(gaba_reuptake, cfg.gat3_baseline * gaba_field)

        # Update mean calcium
        mean_glu = float(np.mean(glutamate_field))
        mean_activity = float(np.mean(activity_field))
        self._update_calcium(mean_glu, mean_activity)

        self._step_count += 1

        return glu_reuptake.astype(np.float32), gaba_reuptake.astype(np.float32)

    def compute_gliotransmission(self) -> dict[str, float]:
        """
        Compute gliotransmitter release based on calcium state.

        Astrocytes release:
        - Glutamate: Excites nearby neurons
        - D-serine: NMDA receptor co-agonist (potentiates plasticity)
        - ATP: Converted to adenosine (inhibitory, promotes sleep)

        Returns:
            Dict with glutamate, dserine, atp release amounts
        """
        cfg = self.config
        ca = self.state.calcium

        # Check release threshold and refractory
        if ca < cfg.gliotx_threshold or self.state.release_refractory:
            return {"glutamate": 0.0, "dserine": 0.0, "atp": 0.0}

        # Calcium-dependent release (Hill-like)
        release_prob = (ca - cfg.gliotx_threshold) / (1.0 - cfg.gliotx_threshold)
        release_prob = min(release_prob, 1.0)

        # Scale by glycogen (energy required for release)
        energy_scale = min(self.state.glycogen, 1.0)

        gliotx = {
            "glutamate": cfg.gliotx_glutamate * release_prob * energy_scale,
            "dserine": cfg.gliotx_dserine * release_prob * energy_scale,
            "atp": cfg.gliotx_atp * release_prob * energy_scale,
        }

        # Consume glycogen
        self.state.glycogen -= 0.01 * release_prob
        self.state.glycogen = max(0, self.state.glycogen)

        # Brief refractory after release
        if release_prob > 0.5:
            self.state.release_refractory = True
            self.state.last_release_time = self._step_count

        return gliotx

    def compute_metabolic_support(self, activity_level: float) -> float:
        """
        Compute lactate production for neuronal energy support.

        Astrocyte-neuron lactate shuttle (ANLS):
        - Astrocytes take up glucose
        - Convert to lactate
        - Export to neurons for oxidative metabolism

        Args:
            activity_level: Current neural activity

        Returns:
            Lactate available for neurons
        """
        cfg = self.config

        # Activity-dependent lactate production
        base_lactate = cfg.lactate_production * activity_level

        # Enhanced production during high activity (use glycogen)
        if activity_level > 0.7 and self.state.glycogen > 0.2:
            glycogen_contribution = 0.1 * (activity_level - 0.7)
            self.state.glycogen -= glycogen_contribution * 0.5
            base_lactate += glycogen_contribution

        # Replenish glycogen during low activity
        if activity_level < 0.3:
            self.state.glycogen += 0.01
            self.state.glycogen = min(self.state.glycogen, cfg.glycogen_buffer)

        return base_lactate

    def _update_calcium(self, glutamate: float, activity: float):
        """Update intracellular calcium based on inputs."""
        cfg = self.config
        ca = self.state.calcium

        # Glutamate triggers Ca2+ rise (mGluR activation)
        glu_drive = cfg.ca_rise_rate * glutamate * (1.0 - ca)

        # Activity also drives calcium
        activity_drive = cfg.ca_rise_rate * 0.5 * activity * (1.0 - ca)

        # Decay toward baseline
        decay = cfg.ca_decay_rate * (ca - 0.1)

        # Update
        self.state.calcium += glu_drive + activity_drive - decay
        self.state.calcium = np.clip(self.state.calcium, 0.0, 1.0)

        # Update activation state
        if self.state.calcium > cfg.reactive_threshold:
            self.state.activation_state = AstrocyteState.REACTIVE
        elif self.state.calcium > cfg.ca_threshold:
            self.state.activation_state = AstrocyteState.ACTIVATED
        else:
            self.state.activation_state = AstrocyteState.QUIESCENT

        # Clear refractory after ~100 steps
        if self.state.release_refractory:
            if self._step_count - self.state.last_release_time > 100:
                self.state.release_refractory = False

    def _get_state_efficiency(self) -> float:
        """Get transporter efficiency based on activation state."""
        state = self.state.activation_state

        if state == AstrocyteState.QUIESCENT:
            return 1.0  # Normal function
        elif state == AstrocyteState.ACTIVATED:
            return 1.2  # Enhanced reuptake
        else:  # REACTIVE
            return 0.6  # Impaired (pathological)

    def get_neuroprotection_score(self) -> float:
        """
        Compute neuroprotection score (0-1).

        Higher = better protection against excitotoxicity.
        Based on:
        - EAAT-2 efficiency
        - Glycogen reserves
        - Absence of reactive state
        """
        # Base on transporter efficiency
        efficiency = self._get_state_efficiency()

        # Glycogen reserves
        energy_score = self.state.glycogen / self.config.glycogen_buffer

        # Penalize reactive state
        state_penalty = 0.0 if self.state.activation_state != AstrocyteState.REACTIVE else 0.3

        score = 0.4 * efficiency + 0.3 * energy_score + 0.3 * (1.0 - state_penalty)
        return float(np.clip(score, 0.0, 1.0))

    def get_current_state(self) -> AstrocyteLayerState:
        """Get current astrocyte state."""
        return self.state

    def reset(self):
        """Reset to initial state."""
        self.state = AstrocyteLayerState()
        self._step_count = 0
        self._ca_history.clear()
        self._glu_history.clear()
        self._gaba_history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "step_count": self._step_count,
            "calcium": self.state.calcium,
            "activation_state": self.state.activation_state.value,
            "glycogen": self.state.glycogen,
            "glutamate_buffered": self.state.glutamate_buffered,
            "gaba_buffered": self.state.gaba_buffered,
            "total_glu_cleared": self.state.total_glu_cleared,
            "total_gaba_cleared": self.state.total_gaba_cleared,
            "excitotoxicity_events": self.state.excitotoxicity_events,
            "neuroprotection_score": self.get_neuroprotection_score(),
            "mean_calcium": float(np.mean(self._ca_history)) if self._ca_history else 0.0,
        }

    def validate_function(self) -> dict[str, Any]:
        """
        Validate astrocyte function against biological criteria.

        Returns:
            Dict with pass/fail for each criterion
        """
        stats = self.get_stats()

        # Criteria
        results = {
            # Should clear significant glutamate
            "glutamate_clearance": stats["total_glu_cleared"] > 0.1 * self._step_count if self._step_count > 100 else True,

            # Should maintain glycogen above critical
            "energy_reserves": self.state.glycogen > 0.1,

            # Should not be constantly reactive
            "not_pathological": self.state.activation_state != AstrocyteState.REACTIVE or self._step_count < 100,

            # Neuroprotection should be reasonable
            "neuroprotection": self.get_neuroprotection_score() > 0.5,

            # No excessive excitotoxicity
            "excitotoxicity_control": self.state.excitotoxicity_events < 0.01 * self._step_count if self._step_count > 100 else True,
        }

        results["all_pass"] = all(results.values())
        return results


def compute_tripartite_synapse(
    presynaptic: float,
    postsynaptic: float,
    astrocyte: AstrocyteLayer,
    glutamate: float,
    gaba: float
) -> dict[str, float]:
    """
    Compute tripartite synapse dynamics.

    The tripartite synapse includes:
    1. Presynaptic terminal (releases glutamate/GABA)
    2. Postsynaptic terminal (receives signal)
    3. Astrocyte (modulates both)

    Args:
        presynaptic: Presynaptic activity (0-1)
        postsynaptic: Postsynaptic activity (0-1)
        astrocyte: AstrocyteLayer instance
        glutamate: Current glutamate level
        gaba: Current GABA level

    Returns:
        Dict with updated neurotransmitter levels and modulation
    """
    # Compute astrocyte reuptake
    activity = 0.5 * (presynaptic + postsynaptic)
    glu_cleared, gaba_cleared = astrocyte.compute_reuptake(glutamate, gaba, activity)

    # Compute gliotransmission
    gliotx = astrocyte.compute_gliotransmission()

    # Net glutamate change
    # Pre releases, astrocyte clears, gliotx adds back
    net_glu = glutamate - glu_cleared + gliotx["glutamate"] * 0.5
    net_glu = np.clip(net_glu, 0.0, 1.0)

    # Net GABA change
    net_gaba = gaba - gaba_cleared
    net_gaba = np.clip(net_gaba, 0.0, 1.0)

    # D-serine potentiates NMDA (enhances plasticity)
    nmda_potentiation = 1.0 + gliotx["dserine"] * 2.0

    # ATP -> adenosine (inhibitory, sleep-promoting)
    adenosine = gliotx["atp"] * 0.8  # Conversion

    return {
        "glutamate": float(net_glu),
        "gaba": float(net_gaba),
        "glu_cleared": float(glu_cleared),
        "gaba_cleared": float(gaba_cleared),
        "nmda_potentiation": float(nmda_potentiation),
        "adenosine": float(adenosine),
        "astrocyte_calcium": astrocyte.state.calcium,
    }


# Backward compatibility alias
AstrocyteNetwork = AstrocyteLayer
