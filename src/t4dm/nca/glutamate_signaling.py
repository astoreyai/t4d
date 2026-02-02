"""
Synaptic vs Extrasynaptic Glutamate Signaling for T4DM.

Sprint 4 (P2): Separates glutamate pools with distinct receptor populations
and differential effects on plasticity and cell survival.

Biological Basis:
- Synaptic glutamate: Fast, localized, activates synaptic NMDA (NR2A)
  -> Pro-survival, CREB activation, LTP
- Extrasynaptic glutamate: Slow, diffuse, activates extrasynaptic NMDA (NR2B)
  -> Excitotoxic, CREB shutoff, LTD, cell death pathways

Key Mechanisms:
1. Synaptic release and fast clearance by perisynaptic EAAT-2
2. Spillover to extrasynaptic space when release > uptake capacity
3. Differential receptor activation (NR2A vs NR2B)
4. Opposing effects on plasticity (LTP vs LTD)
5. Excitotoxicity from sustained extrasynaptic glutamate

References:
- Hardingham & Bading (2010): Synaptic vs extrasynaptic NMDA receptor signalling
- Parsons & Raymond (2014): Extrasynaptic NMDA receptor involvement in CNS disorders
- Papouin & Bhattacharyya et al. (2012): Synaptic and extrasynaptic NMDA receptors
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GlutamatePool(Enum):
    """Glutamate compartment types."""
    SYNAPTIC = "synaptic"           # In synaptic cleft
    EXTRASYNAPTIC = "extrasynaptic"  # Outside cleft, volume transmission
    GLIAL = "glial"                  # Astrocyte-released


class NMDASubtype(Enum):
    """NMDA receptor subtypes with distinct locations and effects."""
    NR2A = "nr2a"  # Synaptic, pro-survival, fast kinetics
    NR2B = "nr2b"  # Extrasynaptic, excitotoxic, slow kinetics


class PlasticityDirection(Enum):
    """Direction of synaptic plasticity."""
    LTP = "ltp"        # Long-term potentiation
    LTD = "ltd"        # Long-term depression
    NEUTRAL = "neutral"  # No change


@dataclass
class GlutamateConfig:
    """Configuration for glutamate signaling dynamics.

    Parameters from literature:
    - Synaptic cleft glutamate: ~1mM peak, clears in ~1ms
    - Extrasynaptic glutamate: ~1-10µM, persists for seconds
    - NR2A activation EC50: ~1.5µM
    - NR2B activation EC50: ~0.4µM (higher affinity)
    """

    # Synaptic release parameters
    release_probability: float = 0.3      # Probability of vesicle release
    quantal_size: float = 0.1             # Glutamate per vesicle (normalized)
    release_sites: int = 10               # Number of release sites

    # Synaptic clearance (perisynaptic EAAT-2)
    synaptic_clearance_rate: float = 0.9   # Very fast clearance
    synaptic_clearance_tau: float = 0.001  # ~1ms time constant

    # Spillover dynamics
    spillover_threshold: float = 0.3       # Synaptic level for spillover
    spillover_fraction: float = 0.4        # Fraction escaping to extrasynaptic

    # Extrasynaptic dynamics
    extrasynaptic_clearance_rate: float = 0.03  # Much slower clearance
    extrasynaptic_diffusion: float = 0.05       # Spatial spread rate
    extrasynaptic_decay: float = 0.02           # Baseline decay

    # NMDA receptor parameters
    nr2a_ec50: float = 0.4                 # NR2A activation threshold
    nr2a_hill: float = 1.5                 # NR2A Hill coefficient
    nr2b_ec50: float = 0.15                # NR2B higher affinity
    nr2b_hill: float = 1.2                 # NR2B Hill coefficient

    # NMDA receptor kinetics (Hestrin 1990)
    # Time constants for NMDA current decay
    tau_nmda_nr2a: float = 0.050           # NR2A decay ~50ms (faster kinetics)
    tau_nmda_nr2b: float = 0.150           # NR2B decay ~150ms (slower kinetics)

    # AMPA receptor parameters (fast excitatory transmission)
    # Added per CompBio B15 validation - Hestrin 1990
    ampa_ec50: float = 0.5                 # AMPA activation threshold
    ampa_hill: float = 1.8                 # AMPA Hill coefficient
    tau_ampa: float = 0.005                # AMPA decay ~5ms (fast kinetics)
    ampa_conductance: float = 1.0          # Relative AMPA conductance

    # Mg2+ block (voltage-dependent, Jahr & Stevens 1990)
    mg_block_model: str = "bhatt1998"      # "simple" or "bhatt1998"
    mg_block_ic50: float = 0.5             # Mg2+ block parameter (simple model)
    mg_ext: float = 1.0                    # External Mg2+ concentration (mM)
    resting_potential: float = -0.7        # Normalized membrane potential

    # Plasticity parameters
    ltp_threshold: float = 0.15            # NR2A activation for LTP (lowered for realism)
    ltd_threshold: float = 0.2             # NR2B activation for LTD
    plasticity_rate: float = 0.01          # Plasticity update rate

    # Excitotoxicity
    excitotoxicity_threshold: float = 0.7  # Extrasynaptic level causing damage
    excitotoxicity_rate: float = 0.1       # Damage accumulation rate
    neuroprotection_factor: float = 0.5    # Synaptic NMDA protection

    # Time constants
    tau_synaptic: float = 0.002            # Synaptic glutamate decay (~2ms)
    tau_extrasynaptic: float = 2.0         # Extrasynaptic decay (~2s)


@dataclass
class GlutamateState:
    """Current state of glutamate signaling system."""

    # Glutamate pools
    synaptic_glu: float = 0.0              # Synaptic cleft [0, 1]
    extrasynaptic_glu: float = 0.05        # Extrasynaptic space [0, 1]
    glial_glu: float = 0.0                 # Astrocyte-released

    # Receptor activation
    nr2a_activation: float = 0.0           # Synaptic NMDA activation
    nr2b_activation: float = 0.0           # Extrasynaptic NMDA activation
    ampa_activation: float = 0.0           # AMPA receptor activation (fast)

    # Downstream signaling
    creb_activity: float = 0.5             # CREB transcription factor
    bdnf_level: float = 0.5                # BDNF expression
    cell_health: float = 1.0               # Cell viability [0, 1]

    # Plasticity state
    synaptic_weight: float = 1.0           # Current synaptic strength
    plasticity_direction: PlasticityDirection = PlasticityDirection.NEUTRAL

    # Cumulative measures
    total_ltp: float = 0.0
    total_ltd: float = 0.0
    excitotoxicity_damage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize state."""
        return {
            "synaptic_glu": self.synaptic_glu,
            "extrasynaptic_glu": self.extrasynaptic_glu,
            "nr2a_activation": self.nr2a_activation,
            "nr2b_activation": self.nr2b_activation,
            "ampa_activation": self.ampa_activation,
            "creb_activity": self.creb_activity,
            "cell_health": self.cell_health,
            "synaptic_weight": self.synaptic_weight,
            "plasticity_direction": self.plasticity_direction.value,
        }


class GlutamateSignaling:
    """
    Synaptic vs Extrasynaptic Glutamate Signaling System.

    Models the differential effects of glutamate location:

    Synaptic Glutamate:
    - Released from presynaptic terminal into cleft
    - Rapidly cleared by perisynaptic EAAT-2 (~1ms)
    - Activates synaptic NMDA (NR2A subunit)
    - Promotes LTP, CREB activation, BDNF expression
    - Neuroprotective

    Extrasynaptic Glutamate:
    - Spillover from cleft or volume transmission
    - Slowly cleared (~seconds)
    - Activates extrasynaptic NMDA (NR2B subunit)
    - Promotes LTD, CREB shutoff
    - Excitotoxic at sustained high levels

    Usage:
        glu = GlutamateSignaling()

        # Presynaptic release
        glu.release(presynaptic_activity=0.8)

        # Step dynamics
        glu.step(dt=0.01)

        # Get plasticity signal
        plasticity = glu.get_plasticity_signal()

        # Check cell health
        if glu.state.cell_health < 0.5:
            logger.warning("Excitotoxicity detected!")
    """

    def __init__(self, config: GlutamateConfig | None = None):
        """
        Initialize glutamate signaling system.

        Args:
            config: Configuration parameters
        """
        self.config = config or GlutamateConfig()
        self.state = GlutamateState()

        # Callbacks
        self._plasticity_callbacks: list[Callable[[float], None]] = []
        self._excitotoxicity_callbacks: list[Callable[[float], None]] = []

        # History
        self._synaptic_history: list[float] = []
        self._extrasynaptic_history: list[float] = []
        self._max_history = 1000

        # Simulation time
        self._simulation_time = 0.0

        logger.info("GlutamateSignaling initialized")

    # =========================================================================
    # Core Dynamics
    # =========================================================================

    def step(self, dt: float = 0.01) -> dict[str, float]:
        """
        Advance glutamate dynamics by one timestep.

        Args:
            dt: Timestep in seconds

        Returns:
            Dict with current state metrics
        """
        self._simulation_time += dt

        # 1. Compute receptor activations FIRST (based on current levels)
        self._update_receptor_activation()

        # 2. Compute spillover to extrasynaptic (before synaptic clears)
        self._compute_spillover(dt)

        # 3. Update synaptic glutamate (fast decay)
        self._update_synaptic_glu(dt)

        # 4. Update extrasynaptic glutamate (slow decay)
        self._update_extrasynaptic_glu(dt)

        # 5. Update downstream signaling
        self._update_creb_signaling(dt)

        # 6. Compute plasticity
        self._update_plasticity(dt)

        # 7. Check excitotoxicity
        self._update_excitotoxicity(dt)

        # Track history
        self._synaptic_history.append(self.state.synaptic_glu)
        self._extrasynaptic_history.append(self.state.extrasynaptic_glu)
        if len(self._synaptic_history) > self._max_history:
            self._synaptic_history = self._synaptic_history[-self._max_history:]
            self._extrasynaptic_history = self._extrasynaptic_history[-self._max_history:]

        return self.state.to_dict()

    def _update_synaptic_glu(self, dt: float) -> None:
        """Update synaptic glutamate with fast clearance."""
        # Fast exponential decay (perisynaptic EAAT-2)
        decay = self.config.synaptic_clearance_rate * self.state.synaptic_glu
        self.state.synaptic_glu -= decay * dt

        # Time constant smoothing
        alpha = dt / self.config.tau_synaptic
        target = 0.0  # Synaptic clears to baseline
        self.state.synaptic_glu += alpha * (target - self.state.synaptic_glu)

        # Clamp
        self.state.synaptic_glu = float(np.clip(self.state.synaptic_glu, 0.0, 1.0))

    def _compute_spillover(self, dt: float) -> None:
        """Compute glutamate spillover from synaptic to extrasynaptic."""
        if self.state.synaptic_glu > self.config.spillover_threshold:
            excess = self.state.synaptic_glu - self.config.spillover_threshold
            spillover = self.config.spillover_fraction * excess * dt

            self.state.synaptic_glu -= spillover
            self.state.extrasynaptic_glu += spillover

    def _update_extrasynaptic_glu(self, dt: float) -> None:
        """Update extrasynaptic glutamate with slow clearance."""
        # Slow decay (astrocyte EAAT-2 on non-perisynaptic membrane)
        decay = self.config.extrasynaptic_clearance_rate * self.state.extrasynaptic_glu
        self.state.extrasynaptic_glu -= decay * dt

        # Diffusion spreading (reduces local concentration)
        diffusion = self.config.extrasynaptic_diffusion * self.state.extrasynaptic_glu * dt
        self.state.extrasynaptic_glu -= diffusion

        # Add glial contribution if present
        self.state.extrasynaptic_glu += self.state.glial_glu * 0.1 * dt
        self.state.glial_glu *= (1 - 0.2 * dt)  # Glial release decays

        # Clamp
        self.state.extrasynaptic_glu = float(np.clip(
            self.state.extrasynaptic_glu, 0.0, 1.0
        ))

    def _update_receptor_activation(self) -> None:
        """Compute NMDA and AMPA receptor activations using Hill kinetics."""
        # NR2A (synaptic NMDA) - responds to synaptic glutamate
        glu_syn = self.state.synaptic_glu
        ec50_a = self.config.nr2a_ec50
        n_a = self.config.nr2a_hill

        if glu_syn > 0:
            nr2a = (glu_syn ** n_a) / (ec50_a ** n_a + glu_syn ** n_a)
        else:
            nr2a = 0.0

        # NR2B (extrasynaptic NMDA) - responds to extrasynaptic glutamate
        glu_extra = self.state.extrasynaptic_glu
        ec50_b = self.config.nr2b_ec50
        n_b = self.config.nr2b_hill

        if glu_extra > 0:
            nr2b = (glu_extra ** n_b) / (ec50_b ** n_b + glu_extra ** n_b)
        else:
            nr2b = 0.0

        # AMPA receptor activation (fast, no Mg2+ block)
        # Per CompBio B15: AMPA provides fast depolarization
        ec50_ampa = self.config.ampa_ec50
        n_ampa = self.config.ampa_hill

        if glu_syn > 0:
            ampa = (glu_syn ** n_ampa) / (ec50_ampa ** n_ampa + glu_syn ** n_ampa)
        else:
            ampa = 0.0

        # Apply Mg2+ block to NMDA only (voltage-dependent)
        # AMPA has no Mg2+ block
        if self.config.mg_block_model == "bhatt1998":
            # Bhatt et al. (1998): block = 1 / (1 + (Mg/3.57)*exp(-0.062*Vm))
            v_m = self.config.resting_potential * 100  # Convert to mV
            mg_block = 1.0 / (1.0 + (self.config.mg_ext / 3.57) * np.exp(-0.062 * v_m))
        else:
            # Simple model: At rest, NMDA is ~80% blocked
            mg_block = 0.2 + 0.8 * (1 - self.config.mg_block_ic50)

        self.state.nr2a_activation = float(nr2a * mg_block)
        self.state.nr2b_activation = float(nr2b * mg_block)
        self.state.ampa_activation = float(ampa * self.config.ampa_conductance)

    def _update_creb_signaling(self, dt: float) -> None:
        """
        Update CREB activity based on receptor balance.

        Synaptic NMDA (NR2A) -> CREB activation -> survival, plasticity
        Extrasynaptic NMDA (NR2B) -> CREB shutoff -> cell death pathways
        """
        # NR2A promotes CREB
        creb_activation = self.state.nr2a_activation * 0.5

        # NR2B suppresses CREB
        creb_suppression = self.state.nr2b_activation * 0.8

        # Net CREB change
        creb_target = 0.5 + creb_activation - creb_suppression
        creb_target = np.clip(creb_target, 0.0, 1.0)

        # Smooth update
        alpha = dt * 0.1  # Slow CREB dynamics
        self.state.creb_activity += alpha * (creb_target - self.state.creb_activity)
        self.state.creb_activity = float(np.clip(self.state.creb_activity, 0.0, 1.0))

        # BDNF follows CREB
        bdnf_target = self.state.creb_activity
        self.state.bdnf_level += alpha * (bdnf_target - self.state.bdnf_level)
        self.state.bdnf_level = float(np.clip(self.state.bdnf_level, 0.0, 1.0))

    def _update_plasticity(self, dt: float) -> None:
        """
        Update synaptic plasticity based on receptor activation patterns.

        NR2A dominant -> LTP
        NR2B dominant -> LTD
        """
        nr2a = self.state.nr2a_activation
        nr2b = self.state.nr2b_activation

        # Determine plasticity direction
        if nr2a > self.config.ltp_threshold and nr2a > nr2b:
            self.state.plasticity_direction = PlasticityDirection.LTP
            delta_w = self.config.plasticity_rate * (nr2a - self.config.ltp_threshold)
            self.state.synaptic_weight += delta_w * dt
            self.state.total_ltp += delta_w * dt

        elif nr2b > self.config.ltd_threshold and nr2b > nr2a:
            self.state.plasticity_direction = PlasticityDirection.LTD
            delta_w = -self.config.plasticity_rate * (nr2b - self.config.ltd_threshold)
            self.state.synaptic_weight += delta_w * dt
            self.state.total_ltd += abs(delta_w * dt)

        else:
            self.state.plasticity_direction = PlasticityDirection.NEUTRAL

        # Clamp weight
        self.state.synaptic_weight = float(np.clip(
            self.state.synaptic_weight, 0.1, 2.0
        ))

        # Fire plasticity callbacks
        plasticity_signal = self.get_plasticity_signal()
        for callback in self._plasticity_callbacks:
            callback(plasticity_signal)

    def _update_excitotoxicity(self, dt: float) -> None:
        """
        Update excitotoxicity based on sustained extrasynaptic glutamate.

        High NR2B activation + low NR2A protection = cell damage
        """
        extra_glu = self.state.extrasynaptic_glu
        nr2b = self.state.nr2b_activation

        # Neuroprotection from synaptic NMDA
        protection = self.config.neuroprotection_factor * self.state.nr2a_activation

        # Damage when extrasynaptic exceeds threshold
        if extra_glu > self.config.excitotoxicity_threshold:
            excess = extra_glu - self.config.excitotoxicity_threshold
            damage_rate = self.config.excitotoxicity_rate * excess * nr2b
            damage_rate *= (1 - protection)  # Reduce by protection

            self.state.excitotoxicity_damage += damage_rate * dt
            self.state.cell_health -= damage_rate * dt

        # Slow recovery when not toxic
        else:
            recovery = 0.01 * (1.0 - self.state.cell_health) * dt
            self.state.cell_health += recovery

        # Clamp
        self.state.cell_health = float(np.clip(self.state.cell_health, 0.0, 1.0))

        # Fire excitotoxicity callbacks
        if self.state.cell_health < 0.8:
            for callback in self._excitotoxicity_callbacks:
                callback(self.state.excitotoxicity_damage)

    # =========================================================================
    # Release and Input
    # =========================================================================

    def release(
        self,
        presynaptic_activity: float,
        stochastic: bool = True
    ) -> float:
        """
        Release glutamate from presynaptic terminal.

        Args:
            presynaptic_activity: Presynaptic firing rate [0, 1]
            stochastic: Whether to use stochastic release

        Returns:
            Amount of glutamate released
        """
        if stochastic:
            # Probabilistic release from multiple sites
            n_vesicles = 0
            for _ in range(self.config.release_sites):
                if np.random.random() < self.config.release_probability * presynaptic_activity:
                    n_vesicles += 1

            release = n_vesicles * self.config.quantal_size
        else:
            # Deterministic release
            expected_vesicles = (
                self.config.release_sites *
                self.config.release_probability *
                presynaptic_activity
            )
            release = expected_vesicles * self.config.quantal_size

        self.state.synaptic_glu += release
        self.state.synaptic_glu = float(np.clip(self.state.synaptic_glu, 0.0, 1.0))

        return release

    def inject_extrasynaptic(self, amount: float) -> None:
        """
        Inject glutamate directly into extrasynaptic space.

        Models volume transmission or pathological glutamate release.

        Args:
            amount: Glutamate amount to inject [0, 1]
        """
        self.state.extrasynaptic_glu += amount
        self.state.extrasynaptic_glu = float(np.clip(
            self.state.extrasynaptic_glu, 0.0, 1.0
        ))

    def glial_release(self, amount: float) -> None:
        """
        Release glutamate from astrocytes (gliotransmission).

        Glial glutamate primarily affects extrasynaptic space.

        Args:
            amount: Glutamate amount to release [0, 1]
        """
        self.state.glial_glu += amount
        self.state.glial_glu = float(np.clip(self.state.glial_glu, 0.0, 1.0))

    def set_membrane_potential(self, potential: float) -> None:
        """
        Set membrane potential for Mg2+ block calculation.

        Args:
            potential: Membrane potential [-1, 0] (normalized)
        """
        self.config.resting_potential = float(np.clip(potential, -1.0, 0.0))

    # =========================================================================
    # Output Methods
    # =========================================================================

    def get_plasticity_signal(self) -> float:
        """
        Get net plasticity signal.

        Returns:
            Positive = LTP, Negative = LTD, ~0 = no change
        """
        if self.state.plasticity_direction == PlasticityDirection.LTP:
            return self.state.nr2a_activation - self.config.ltp_threshold
        elif self.state.plasticity_direction == PlasticityDirection.LTD:
            return -(self.state.nr2b_activation - self.config.ltd_threshold)
        else:
            return 0.0

    def get_total_glutamate(self) -> float:
        """Get total glutamate across all pools."""
        return (
            self.state.synaptic_glu +
            self.state.extrasynaptic_glu +
            self.state.glial_glu
        )

    def get_receptor_balance(self) -> float:
        """
        Get NR2A/NR2B balance.

        Returns:
            Positive = NR2A dominant (neuroprotective)
            Negative = NR2B dominant (excitotoxic)
        """
        return self.state.nr2a_activation - self.state.nr2b_activation

    def is_excitotoxic(self) -> bool:
        """Check if currently in excitotoxic state."""
        return (
            self.state.extrasynaptic_glu > self.config.excitotoxicity_threshold and
            self.state.nr2b_activation > self.state.nr2a_activation
        )

    # =========================================================================
    # Callbacks
    # =========================================================================

    def register_plasticity_callback(self, callback: Callable[[float], None]) -> None:
        """Register callback for plasticity changes."""
        self._plasticity_callbacks.append(callback)

    def register_excitotoxicity_callback(self, callback: Callable[[float], None]) -> None:
        """Register callback for excitotoxicity events."""
        self._excitotoxicity_callbacks.append(callback)

    # =========================================================================
    # Statistics and State
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "synaptic_glu": self.state.synaptic_glu,
            "extrasynaptic_glu": self.state.extrasynaptic_glu,
            "nr2a_activation": self.state.nr2a_activation,
            "nr2b_activation": self.state.nr2b_activation,
            "ampa_activation": self.state.ampa_activation,
            "receptor_balance": self.get_receptor_balance(),
            "creb_activity": self.state.creb_activity,
            "bdnf_level": self.state.bdnf_level,
            "cell_health": self.state.cell_health,
            "synaptic_weight": self.state.synaptic_weight,
            "plasticity_direction": self.state.plasticity_direction.value,
            "total_ltp": self.state.total_ltp,
            "total_ltd": self.state.total_ltd,
            "excitotoxicity_damage": self.state.excitotoxicity_damage,
            "is_excitotoxic": self.is_excitotoxic(),
            "simulation_time": self._simulation_time,
        }

        if self._synaptic_history:
            stats["avg_synaptic"] = float(np.mean(self._synaptic_history))
            stats["avg_extrasynaptic"] = float(np.mean(self._extrasynaptic_history))

        return stats

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = GlutamateState()
        self._synaptic_history.clear()
        self._extrasynaptic_history.clear()
        self._simulation_time = 0.0
        logger.info("GlutamateSignaling reset")

    def save_state(self) -> dict[str, Any]:
        """Save state for persistence."""
        return {
            "state": self.state.to_dict(),
            "simulation_time": self._simulation_time,
        }

    def load_state(self, saved: dict[str, Any]) -> None:
        """Load state from persistence."""
        if "simulation_time" in saved:
            self._simulation_time = saved["simulation_time"]
        if "state" in saved:
            s = saved["state"]
            self.state.synaptic_glu = s.get("synaptic_glu", 0.0)
            self.state.extrasynaptic_glu = s.get("extrasynaptic_glu", 0.05)
            self.state.nr2a_activation = s.get("nr2a_activation", 0.0)
            self.state.nr2b_activation = s.get("nr2b_activation", 0.0)
            self.state.ampa_activation = s.get("ampa_activation", 0.0)
            self.state.cell_health = s.get("cell_health", 1.0)
            self.state.synaptic_weight = s.get("synaptic_weight", 1.0)


def create_glutamate_signaling(
    release_probability: float = 0.3,
    spillover_threshold: float = 0.4,
    excitotoxicity_threshold: float = 0.7,
) -> GlutamateSignaling:
    """
    Factory function to create glutamate signaling system.

    Args:
        release_probability: Vesicle release probability
        spillover_threshold: Synaptic level triggering spillover
        excitotoxicity_threshold: Extrasynaptic level causing damage

    Returns:
        Configured GlutamateSignaling
    """
    config = GlutamateConfig(
        release_probability=release_probability,
        spillover_threshold=spillover_threshold,
        excitotoxicity_threshold=excitotoxicity_threshold,
    )
    return GlutamateSignaling(config)


__all__ = [
    "GlutamateSignaling",
    "GlutamateConfig",
    "GlutamateState",
    "GlutamatePool",
    "NMDASubtype",
    "PlasticityDirection",
    "create_glutamate_signaling",
]
