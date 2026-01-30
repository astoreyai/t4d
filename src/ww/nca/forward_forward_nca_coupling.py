"""
Forward-Forward Algorithm - NCA Neural Field Coupling for World Weaver.

Implements H10 (cross-region consistency) by aligning Forward-Forward
local learning with the global NCA energy landscape:

1. FF Goodness ↔ Energy Landscape:
   - FF goodness = negative energy (high goodness = low energy basin)
   - FF threshold θ = energy barrier between attractors
   - Positive phase: descend energy; negative phase: ascend energy

2. Neuromodulator → FF Learning:
   - DA: Modulates learning rate (surprise-driven)
   - ACh: Gates between encoding (training) and retrieval modes
   - NE: Adjusts goodness threshold (arousal → selectivity)
   - 5-HT: Temporal credit assignment window

3. FF → NCA Feedback:
   - Layer goodness → Local field energy contribution
   - Learning events → Field state perturbations
   - Convergence status → Stability signals

Biological Basis:
- Local learning in cortical columns (Hinton 2022)
- Layer-wise objectives as proxy for metabolic efficiency
- Positive/negative phases as wake/sleep analogs

References:
- Hinton, G. E. (2022). The Forward-Forward Algorithm
- Friston, K. (2010). The free-energy principle
- Whittington & Bogacz (2017). Approximation of the error backpropagation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ww.nca.neural_field import NeurotransmitterState

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class FFPhase(Enum):
    """Forward-Forward learning phase."""

    POSITIVE = auto()  # Real data: maximize goodness
    NEGATIVE = auto()  # Fake data: minimize goodness
    INFERENCE = auto()  # No learning, just forward pass


class EnergyAlignment(Enum):
    """How FF goodness aligns with energy landscape."""

    BASIN = auto()     # In attractor basin (high goodness, low energy)
    BARRIER = auto()   # At energy barrier (threshold goodness)
    SADDLE = auto()    # Saddle point (unstable)
    TRANSITION = auto()  # Transitioning between attractors


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FFNCACouplingConfig:
    """
    Configuration for FF-NCA coupling.

    Biological parameters based on:
    - Hinton (2022): Forward-Forward algorithm
    - Schultz (1998): DA and reward prediction error
    - Hasselmo (2006): ACh and encoding/retrieval
    """

    # NT → FF modulation
    da_learning_rate_gain: float = 0.5    # DA → learning rate multiplier
    ach_phase_threshold: float = 0.6      # ACh level for positive phase bias
    ne_threshold_gain: float = 0.4        # NE → goodness threshold
    serotonin_credit_window: float = 0.3  # 5-HT → temporal credit decay

    # FF → Energy alignment
    goodness_energy_scale: float = 1.0    # Goodness to energy conversion
    layer_energy_weights: list[float] = field(default_factory=lambda: [1.0, 0.8, 0.6])

    # FF → NCA feedback
    goodness_to_stability: float = 0.3    # Goodness → field stability
    learning_to_perturbation: float = 0.2  # Learning events → field perturbation

    # Thresholds
    goodness_threshold_base: float = 0.5  # Base threshold for positive/negative
    convergence_tolerance: float = 0.01   # When to consider converged

    # Coupling behavior
    enable_energy_alignment: bool = True
    enable_neuromod_gating: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert 0.0 <= self.da_learning_rate_gain <= 1.0
        assert 0.0 <= self.ach_phase_threshold <= 1.0
        assert 0.0 <= self.ne_threshold_gain <= 1.0
        assert self.goodness_energy_scale > 0


@dataclass
class FFNCACouplingState:
    """Runtime state of FF-NCA coupling."""

    # Current phase
    phase: FFPhase = FFPhase.INFERENCE

    # NT-derived modulation
    learning_rate_multiplier: float = 1.0
    goodness_threshold: float = 0.5
    credit_decay: float = 0.9

    # Energy alignment
    alignment: EnergyAlignment = EnergyAlignment.BASIN
    total_energy: float = 0.0
    layer_energies: list[float] = field(default_factory=list)

    # FF state tracking
    layer_goodness: list[float] = field(default_factory=list)
    converged: bool = False
    learning_magnitude: float = 0.0

    # History
    goodness_history: list[float] = field(default_factory=list)
    energy_history: list[float] = field(default_factory=list)

    def update_history(self, max_len: int = 100):
        """Maintain bounded history."""
        if len(self.goodness_history) > max_len:
            self.goodness_history = self.goodness_history[-max_len:]
        if len(self.energy_history) > max_len:
            self.energy_history = self.energy_history[-max_len:]


# =============================================================================
# Core Coupling Class
# =============================================================================


class FFNCACoupling:
    """
    Bidirectional coupling between Forward-Forward and NCA neural field.

    Implements H10 cross-region consistency by aligning local FF learning
    with global NCA energy dynamics.

    Example:
        >>> coupling = FFNCACoupling()
        >>> nt_state = NeurotransmitterState(dopamine=0.8, acetylcholine=0.7)
        >>> params = coupling.modulate_ff_params(nt_state)
        >>> # Use params in FF learning
    """

    def __init__(self, config: FFNCACouplingConfig | None = None):
        """Initialize coupling."""
        self.config = config or FFNCACouplingConfig()
        self.state = FFNCACouplingState()

        logger.info(
            f"FFNCACoupling initialized: energy_alignment={self.config.enable_energy_alignment}, "
            f"neuromod_gating={self.config.enable_neuromod_gating}"
        )

    # -------------------------------------------------------------------------
    # NT State → FF Modulation
    # -------------------------------------------------------------------------

    def determine_phase(self, nt_state: NeurotransmitterState) -> FFPhase:
        """
        Determine FF phase from ACh level.

        High ACh: Bias toward positive phase (encoding)
        Low ACh: Bias toward negative phase (refinement)

        Based on Hasselmo's encoding/retrieval dynamics.
        """
        if not self.config.enable_neuromod_gating:
            return FFPhase.INFERENCE

        ach = nt_state.acetylcholine

        if ach > self.config.ach_phase_threshold:
            phase = FFPhase.POSITIVE
        elif ach < self.config.ach_phase_threshold - 0.2:
            phase = FFPhase.NEGATIVE
        else:
            phase = FFPhase.INFERENCE

        self.state.phase = phase
        return phase

    def compute_learning_rate_multiplier(
        self,
        nt_state: NeurotransmitterState
    ) -> float:
        """
        Compute learning rate multiplier from DA level.

        Higher DA (surprise/salience) → higher learning rate
        Implements reward prediction error modulation.

        Based on Schultz (1998) dopamine and learning.
        """
        if not self.config.enable_neuromod_gating:
            return 1.0

        da = nt_state.dopamine

        # DA above baseline increases learning, below decreases
        # Baseline is 0.5
        multiplier = 1.0 + (da - 0.5) * 2 * self.config.da_learning_rate_gain
        multiplier = np.clip(multiplier, 0.1, 3.0)

        self.state.learning_rate_multiplier = multiplier
        return multiplier

    def compute_goodness_threshold(
        self,
        nt_state: NeurotransmitterState
    ) -> float:
        """
        Compute goodness threshold from NE level.

        Higher NE (arousal) → higher threshold → more selective
        Only high-confidence patterns pass as positive.

        Based on Aston-Jones & Cohen (2005).
        """
        if not self.config.enable_neuromod_gating:
            return self.config.goodness_threshold_base

        ne = nt_state.norepinephrine

        threshold = self.config.goodness_threshold_base + \
                   (ne - 0.5) * self.config.ne_threshold_gain
        threshold = np.clip(threshold, 0.2, 0.9)

        self.state.goodness_threshold = threshold
        return threshold

    def compute_credit_decay(
        self,
        nt_state: NeurotransmitterState
    ) -> float:
        """
        Compute temporal credit decay from 5-HT level.

        Higher 5-HT → slower decay → longer temporal credit window
        Enables learning from delayed outcomes.

        Based on Doya (2002) serotonin and temporal discounting.
        """
        if not self.config.enable_neuromod_gating:
            return 0.9

        serotonin = nt_state.serotonin

        # Higher 5-HT = more patient = slower decay
        decay = 0.8 + serotonin * 0.15 * self.config.serotonin_credit_window
        decay = np.clip(decay, 0.7, 0.99)

        self.state.credit_decay = decay
        return decay

    def modulate_ff_params(
        self,
        nt_state: NeurotransmitterState
    ) -> dict:
        """
        Compute all FF modulation parameters from NT state.

        Returns dict with:
        - learning_rate_multiplier: Scale factor for learning rate
        - goodness_threshold: Threshold for positive classification
        - credit_decay: Temporal credit decay factor
        - phase: Suggested learning phase
        """
        phase = self.determine_phase(nt_state)
        lr_mult = self.compute_learning_rate_multiplier(nt_state)
        threshold = self.compute_goodness_threshold(nt_state)
        decay = self.compute_credit_decay(nt_state)

        return {
            "learning_rate_multiplier": lr_mult,
            "goodness_threshold": threshold,
            "credit_decay": decay,
            "phase": phase,
            "da_level": nt_state.dopamine,
            "ach_level": nt_state.acetylcholine,
            "ne_level": nt_state.norepinephrine,
            "serotonin_level": nt_state.serotonin,
        }

    # -------------------------------------------------------------------------
    # C2: NCA → FF Feedback
    # -------------------------------------------------------------------------

    def nca_to_ff_feedback(
        self,
        nca_energy: float,
        current_goodness_threshold: float | None = None
    ) -> float:
        """
        C2: NCA energy basin state modulates FF positive-phase threshold.

        When NCA is in a deep energy basin (near attractor), lower the FF
        goodness threshold to make pattern acceptance easier. This creates
        a feedback loop where stable NCA states facilitate FF learning.

        Biological basis:
        - Stable neural states (low energy) → high plasticity
        - Unstable states (high energy) → conservative learning
        - Energy landscape guides local learning objectives

        Args:
            nca_energy: Current NCA energy (lower = more stable)
            current_goodness_threshold: Override default threshold

        Returns:
            Modulated goodness threshold for FF learning
        """
        if not self.config.enable_energy_alignment:
            threshold = current_goodness_threshold or self.config.goodness_threshold_base
            return threshold

        threshold = current_goodness_threshold or self.state.goodness_threshold

        # Deep energy basin (< 0) → lower threshold (easier acceptance)
        # High energy (> 0) → higher threshold (stricter acceptance)
        # Scale factor: basin depth of -2 → -0.15 threshold adjustment
        energy_modulation = -np.tanh(nca_energy / 2.0) * 0.15

        modulated_threshold = threshold + energy_modulation
        modulated_threshold = np.clip(modulated_threshold, 0.2, 0.9)

        return float(modulated_threshold)

    # -------------------------------------------------------------------------
    # FF State → Energy Landscape Alignment
    # -------------------------------------------------------------------------

    def goodness_to_energy(self, goodness: float) -> float:
        """
        Convert FF goodness to energy landscape value.

        High goodness = low energy (in attractor basin)
        Low goodness = high energy (at barrier or saddle)

        Energy = -scale * log(goodness + eps)
        """
        if not self.config.enable_energy_alignment:
            return 0.0

        # ATOM-P4-7: Use np.maximum instead of addition to prevent log of negative
        # Logistic transform for bounded energy
        energy = -self.config.goodness_energy_scale * np.log(np.maximum(goodness, 1e-8))
        return float(np.clip(energy, -10, 10))

    def compute_total_energy(
        self,
        layer_goodness: list[float],
        weights: list[float] | None = None
    ) -> float:
        """
        Compute total energy from layer-wise goodness values.

        Uses weighted sum across layers with deeper layers
        contributing less (hierarchical energy landscape).
        """
        if not self.config.enable_energy_alignment:
            return 0.0

        if weights is None:
            weights = self.config.layer_energy_weights

        # Pad or truncate weights to match layers
        if len(weights) < len(layer_goodness):
            weights = weights + [0.5] * (len(layer_goodness) - len(weights))
        weights = weights[:len(layer_goodness)]

        layer_energies = []
        total = 0.0
        for g, w in zip(layer_goodness, weights):
            e = self.goodness_to_energy(g)
            layer_energies.append(e * w)
            total += e * w

        self.state.layer_energies = layer_energies
        self.state.layer_goodness = list(layer_goodness)
        self.state.total_energy = total
        self.state.energy_history.append(total)
        self.state.update_history()

        return total

    def determine_alignment(
        self,
        goodness: float,
        threshold: float | None = None
    ) -> EnergyAlignment:
        """
        Determine energy landscape alignment from goodness.

        BASIN: High goodness (> threshold + margin)
        BARRIER: Near threshold
        SADDLE: Low but unstable
        TRANSITION: Moving between states
        """
        if threshold is None:
            threshold = self.state.goodness_threshold

        margin = 0.15

        if goodness > threshold + margin:
            alignment = EnergyAlignment.BASIN
        elif goodness > threshold - margin:
            alignment = EnergyAlignment.BARRIER
        elif len(self.state.goodness_history) > 1:
            # Check if transitioning
            prev = self.state.goodness_history[-1] if self.state.goodness_history else goodness
            if abs(goodness - prev) > 0.1:
                alignment = EnergyAlignment.TRANSITION
            else:
                alignment = EnergyAlignment.SADDLE
        else:
            alignment = EnergyAlignment.SADDLE

        self.state.alignment = alignment
        self.state.goodness_history.append(goodness)
        self.state.update_history()

        return alignment

    # -------------------------------------------------------------------------
    # FF State → NCA Feedback
    # -------------------------------------------------------------------------

    def compute_stability_signal(
        self,
        layer_goodness: list[float]
    ) -> float:
        """
        Compute field stability signal from FF goodness.

        High goodness across layers → stable field
        Mixed goodness → transitional state
        """
        if not layer_goodness:
            return 0.0

        # Mean goodness as stability proxy
        mean_goodness = np.mean(layer_goodness)

        # Variance as instability proxy
        if len(layer_goodness) > 1:
            variance = np.var(layer_goodness)
            stability = mean_goodness * (1.0 - np.tanh(variance * 5))
        else:
            stability = mean_goodness

        return float(stability * self.config.goodness_to_stability)

    def compute_learning_perturbation(
        self,
        weight_updates: np.ndarray | None = None,
        learning_magnitude: float = 0.0
    ) -> np.ndarray:
        """
        Compute field perturbation from FF learning events.

        Large weight updates perturb the field, enabling
        learning-driven state transitions.

        Returns 6-NT perturbation array.
        """
        if weight_updates is not None:
            magnitude = float(np.mean(np.abs(weight_updates)))
        else:
            magnitude = learning_magnitude

        self.state.learning_magnitude = magnitude

        # Learning events boost excitatory NTs
        perturbation = np.array([
            magnitude * 0.4,   # DA: learning signal
            magnitude * 0.1,   # 5-HT: slight increase
            magnitude * 0.3,   # ACh: attention during learning
            magnitude * 0.2,   # NE: arousal
            -magnitude * 0.15, # GABA: reduced inhibition
            magnitude * 0.25,  # Glu: excitation
        ]) * self.config.learning_to_perturbation

        return perturbation

    def compute_nca_feedback(
        self,
        layer_goodness: list[float],
        weight_updates: np.ndarray | None = None,
        learning_magnitude: float = 0.0
    ) -> dict:
        """
        Compute all NCA feedback from FF state.

        Returns dict with:
        - stability_signal: Field stability from goodness
        - total_energy: Energy landscape value
        - alignment: Current energy alignment
        - field_perturbation: NT perturbation from learning
        """
        # Energy computation
        energy = self.compute_total_energy(layer_goodness)

        # Alignment determination
        mean_goodness = np.mean(layer_goodness) if layer_goodness else 0.0
        alignment = self.determine_alignment(mean_goodness)

        # Stability signal
        stability = self.compute_stability_signal(layer_goodness)

        # Learning perturbation
        perturbation = self.compute_learning_perturbation(
            weight_updates, learning_magnitude
        )

        return {
            "stability_signal": stability,
            "total_energy": energy,
            "layer_energies": self.state.layer_energies,
            "alignment": alignment,
            "field_perturbation": perturbation,
            "mean_goodness": mean_goodness,
            "converged": self.state.converged,
        }

    # -------------------------------------------------------------------------
    # Convergence Detection
    # -------------------------------------------------------------------------

    def check_convergence(
        self,
        layer_goodness: list[float],
        threshold: float | None = None
    ) -> bool:
        """
        Check if FF has converged (stable goodness across layers).

        Convergence = all layers above threshold with low variance.
        """
        if not layer_goodness:
            return False

        if threshold is None:
            threshold = self.state.goodness_threshold

        # All layers must exceed threshold
        all_above = all(g > threshold for g in layer_goodness)

        # Variance must be low
        if len(layer_goodness) > 1:
            variance = np.var(layer_goodness)
            low_variance = variance < self.config.convergence_tolerance
        else:
            low_variance = True

        converged = all_above and low_variance
        self.state.converged = converged

        return converged

    # -------------------------------------------------------------------------
    # Full Coupling Step
    # -------------------------------------------------------------------------

    def step(
        self,
        nt_state: NeurotransmitterState,
        layer_goodness: list[float] | None = None,
        weight_updates: np.ndarray | None = None
    ) -> tuple[dict, dict]:
        """
        Execute full bidirectional coupling step.

        Args:
            nt_state: Current neurotransmitter state
            layer_goodness: Goodness values per layer (optional)
            weight_updates: Recent weight updates (optional)

        Returns:
            Tuple of (ff_modulation, nca_feedback)
        """
        # Forward: NT → FF
        ff_modulation = self.modulate_ff_params(nt_state)

        # Check convergence if data available
        if layer_goodness:
            self.check_convergence(layer_goodness, ff_modulation["goodness_threshold"])

        # Backward: FF → NCA
        if layer_goodness:
            nca_feedback = self.compute_nca_feedback(
                layer_goodness, weight_updates
            )
        else:
            nca_feedback = {
                "stability_signal": 0.0,
                "total_energy": 0.0,
                "layer_energies": [],
                "alignment": EnergyAlignment.BASIN,
                "field_perturbation": np.zeros(6),
                "mean_goodness": 0.0,
                "converged": False,
            }

        return ff_modulation, nca_feedback

    def get_state(self) -> FFNCACouplingState:
        """Get current coupling state."""
        return self.state

    def reset(self):
        """Reset coupling state."""
        self.state = FFNCACouplingState()
        logger.debug("FFNCACoupling state reset")


# =============================================================================
# Factory Functions
# =============================================================================


def create_ff_nca_coupling(
    enable_energy: bool = True,
    enable_neuromod: bool = True,
    **kwargs
) -> FFNCACoupling:
    """
    Factory function for creating FF-NCA coupling.

    Args:
        enable_energy: Enable energy landscape alignment
        enable_neuromod: Enable neuromodulator gating
        **kwargs: Additional config parameters

    Returns:
        Configured FFNCACoupling instance
    """
    config = FFNCACouplingConfig(
        enable_energy_alignment=enable_energy,
        enable_neuromod_gating=enable_neuromod,
        **kwargs
    )

    return FFNCACoupling(config)
