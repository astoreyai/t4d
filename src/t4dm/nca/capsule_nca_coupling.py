"""
Capsule-NCA Neural Field Coupling for World Weaver.

Implements H10 (cross-region consistency) by integrating capsule networks
with the NCA neural field dynamics. Enables bidirectional information flow:

1. NT State → Capsule Modulation:
   - DA: Modulates routing temperature (sharpness)
   - NE: Affects squashing threshold (arousal-gated)
   - ACh: Gates encoding vs retrieval capsule mode
   - 5-HT: Stabilizes routing convergence (patience)

2. Capsule State → NCA Feedback:
   - Routing agreement → Field stability signal
   - Pose transformations → Attractor geometry hints
   - Capsule activations → Local field perturbations

Biological Basis:
- Cortical microcolumns as capsule analogs (Mountcastle, 1997)
- Neuromodulatory gating of columnar processing (Hasselmo, 2006)
- Binding problem solution through synchronized capsules

References:
- Sabour et al. (2017). Dynamic Routing Between Capsules
- Hinton et al. (2018). Matrix capsules with EM routing
- Hasselmo (2006). The role of acetylcholine in learning and memory
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from t4dm.nca.neural_field import NeurotransmitterState

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class CapsuleMode(Enum):
    """Capsule operating mode, gated by ACh levels."""

    ENCODING = auto()   # High ACh: sharp routing, store new patterns
    RETRIEVAL = auto()  # Low ACh: soft routing, pattern completion
    NEUTRAL = auto()    # Mid ACh: balanced operation


class CouplingStrength(Enum):
    """Strength of NT-capsule coupling."""

    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    FULL = 1.0


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CapsuleNCACouplingConfig:
    """
    Configuration for capsule-NCA coupling.

    Biological parameters based on:
    - Hasselmo (2006): ACh encoding/retrieval gating
    - Aston-Jones & Cohen (2005): NE arousal modulation
    - Schultz (1998): DA reward/salience signaling
    """

    # NT → Capsule modulation strengths
    da_routing_gain: float = 0.5       # DA → routing temperature
    ne_threshold_gain: float = 0.4     # NE → squashing threshold
    ach_mode_threshold: float = 0.6    # ACh level for encoding mode
    serotonin_stability_gain: float = 0.3  # 5-HT → routing stability

    # Capsule → NCA feedback strengths
    agreement_to_stability: float = 0.4  # Routing agreement → field stability
    pose_to_attractor: float = 0.3       # Pose → attractor geometry
    activation_to_field: float = 0.2     # Capsule activation → field perturbation

    # Biological constraints
    min_routing_temperature: float = 0.1   # Minimum routing sharpness
    max_routing_temperature: float = 2.0   # Maximum routing sharpness
    mode_hysteresis: float = 0.1           # Hysteresis for mode switching

    # Coupling behavior
    coupling_strength: CouplingStrength = CouplingStrength.MODERATE
    bidirectional: bool = True  # Enable both directions

    def __post_init__(self):
        """Validate biological plausibility of parameters."""
        assert 0.0 <= self.da_routing_gain <= 1.0, "DA gain must be in [0, 1]"
        assert 0.0 <= self.ne_threshold_gain <= 1.0, "NE gain must be in [0, 1]"
        assert 0.0 <= self.ach_mode_threshold <= 1.0, "ACh threshold must be in [0, 1]"
        assert 0.1 <= self.max_routing_temperature <= 5.0, "Temperature range invalid"


@dataclass
class CapsuleNCACouplingState:
    """
    Runtime state of the capsule-NCA coupling.

    Tracks bidirectional information flow for debugging and analysis.
    """

    # Current mode
    mode: CapsuleMode = CapsuleMode.NEUTRAL

    # NT-derived modulation values
    routing_temperature: float = 1.0
    squashing_threshold: float = 0.5
    stability_factor: float = 1.0

    # Capsule-derived feedback values
    mean_agreement: float = 0.0
    pose_coherence: float = 0.0
    activation_magnitude: float = 0.0

    # History for hysteresis
    mode_history: list = field(default_factory=list)
    agreement_history: list = field(default_factory=list)

    def update_history(self, max_len: int = 100):
        """Maintain bounded history."""
        if len(self.mode_history) > max_len:
            self.mode_history = self.mode_history[-max_len:]
        if len(self.agreement_history) > max_len:
            self.agreement_history = self.agreement_history[-max_len:]


# =============================================================================
# Core Coupling Class
# =============================================================================


class CapsuleNCACoupling:
    """
    Bidirectional coupling between capsule networks and NCA neural field.

    Implements H10 cross-region consistency by ensuring capsule routing
    respects neural field dynamics and vice versa.

    Example:
        >>> coupling = CapsuleNCACoupling()
        >>> nt_state = NeurotransmitterState(dopamine=0.7, acetylcholine=0.8)
        >>> modulated_params = coupling.modulate_capsule_params(nt_state)
        >>> # Use modulated_params in capsule routing
    """

    def __init__(self, config: CapsuleNCACouplingConfig | None = None):
        """Initialize coupling with configuration."""
        self.config = config or CapsuleNCACouplingConfig()
        self.state = CapsuleNCACouplingState()
        self._previous_ach = 0.5  # For hysteresis

        logger.info(
            f"CapsuleNCACoupling initialized: strength={self.config.coupling_strength.name}, "
            f"bidirectional={self.config.bidirectional}"
        )

    # -------------------------------------------------------------------------
    # NT State → Capsule Modulation (Forward Direction)
    # -------------------------------------------------------------------------

    def determine_mode(self, nt_state: NeurotransmitterState) -> CapsuleMode:
        """
        Determine capsule operating mode from ACh level.

        High ACh (>0.6): ENCODING - sharp routing, store patterns
        Low ACh (<0.4): RETRIEVAL - soft routing, pattern completion
        Mid ACh: NEUTRAL - balanced

        Based on Hasselmo (2006) encoding/retrieval dynamics.
        """
        ach = nt_state.acetylcholine
        threshold = self.config.ach_mode_threshold
        hysteresis = self.config.mode_hysteresis

        # Apply hysteresis to prevent mode flickering
        if self.state.mode == CapsuleMode.ENCODING:
            if ach < threshold - hysteresis:
                new_mode = CapsuleMode.RETRIEVAL
            elif ach < threshold:
                new_mode = CapsuleMode.NEUTRAL
            else:
                new_mode = CapsuleMode.ENCODING
        elif self.state.mode == CapsuleMode.RETRIEVAL:
            if ach > threshold + hysteresis:
                new_mode = CapsuleMode.ENCODING
            elif ach > threshold:
                new_mode = CapsuleMode.NEUTRAL
            else:
                new_mode = CapsuleMode.RETRIEVAL
        else:  # NEUTRAL
            if ach > threshold + hysteresis:
                new_mode = CapsuleMode.ENCODING
            elif ach < threshold - hysteresis:
                new_mode = CapsuleMode.RETRIEVAL
            else:
                new_mode = CapsuleMode.NEUTRAL

        self.state.mode = new_mode
        self.state.mode_history.append(new_mode)
        self._previous_ach = ach

        return new_mode

    def compute_routing_temperature(self, nt_state: NeurotransmitterState) -> float:
        """
        Compute routing temperature from DA and mode.

        Higher DA → sharper routing (lower temperature)
        ENCODING mode → sharper routing
        RETRIEVAL mode → softer routing (pattern completion)

        Temperature controls softmax sharpness in routing coefficients.
        """
        da = nt_state.dopamine
        mode = self.determine_mode(nt_state)

        # Base temperature inversely proportional to DA
        # High DA = high salience = sharp routing
        base_temp = 1.0 - (da - 0.5) * self.config.da_routing_gain

        # Mode modulation
        if mode == CapsuleMode.ENCODING:
            mode_factor = 0.7  # Sharper for encoding
        elif mode == CapsuleMode.RETRIEVAL:
            mode_factor = 1.5  # Softer for retrieval/completion
        else:
            mode_factor = 1.0

        temperature = base_temp * mode_factor

        # Clamp to valid range
        temperature = np.clip(
            temperature,
            self.config.min_routing_temperature,
            self.config.max_routing_temperature
        )

        self.state.routing_temperature = temperature
        return temperature

    def compute_squashing_threshold(self, nt_state: NeurotransmitterState) -> float:
        """
        Compute squashing threshold from NE level.

        Higher NE (arousal) → higher threshold → fewer active capsules
        Implements gain modulation for signal-to-noise control.

        Based on Aston-Jones & Cohen (2005) NE and attention.
        """
        ne = nt_state.norepinephrine

        # Higher NE raises threshold (more selective)
        threshold = 0.5 + (ne - 0.5) * self.config.ne_threshold_gain
        threshold = np.clip(threshold, 0.1, 0.9)

        self.state.squashing_threshold = threshold
        return threshold

    def compute_stability_factor(self, nt_state: NeurotransmitterState) -> float:
        """
        Compute routing stability factor from 5-HT level.

        Higher 5-HT → more stable routing (patience in convergence)
        Lower 5-HT → faster but potentially unstable routing

        Based on Doya (2002) serotonin and temporal discounting.
        """
        serotonin = nt_state.serotonin

        # Higher 5-HT = more patient = more routing iterations allowed
        stability = 0.5 + (serotonin - 0.5) * self.config.serotonin_stability_gain
        stability = np.clip(stability, 0.3, 1.0)

        self.state.stability_factor = stability
        return stability

    def modulate_capsule_params(
        self,
        nt_state: NeurotransmitterState
    ) -> dict:
        """
        Compute all capsule modulation parameters from NT state.

        Returns dict with:
        - routing_temperature: For softmax in routing
        - squashing_threshold: For activation gating
        - stability_factor: For convergence patience
        - mode: Current encoding/retrieval mode
        - routing_iterations: Adjusted iteration count
        """
        mode = self.determine_mode(nt_state)
        temperature = self.compute_routing_temperature(nt_state)
        threshold = self.compute_squashing_threshold(nt_state)
        stability = self.compute_stability_factor(nt_state)

        # Adjust routing iterations based on stability
        base_iterations = 3
        iterations = max(1, int(base_iterations * stability + 0.5))

        return {
            "routing_temperature": temperature,
            "squashing_threshold": threshold,
            "stability_factor": stability,
            "mode": mode,
            "routing_iterations": iterations,
            "ach_level": nt_state.acetylcholine,
            "da_level": nt_state.dopamine,
            "ne_level": nt_state.norepinephrine,
            "serotonin_level": nt_state.serotonin,
        }

    # -------------------------------------------------------------------------
    # Capsule State → NCA Feedback (Backward Direction)
    # -------------------------------------------------------------------------

    def compute_agreement_feedback(
        self,
        routing_coefficients: np.ndarray,
        capsule_activations: np.ndarray
    ) -> float:
        """
        Compute field stability signal from routing agreement.

        High routing agreement → stable field state
        Low agreement → transitional/uncertain state

        Returns stability signal in [0, 1].
        """
        if not self.config.bidirectional:
            return 0.0

        # Compute entropy of routing coefficients (lower = more agreement)
        # Normalize to get proper probabilities
        probs = routing_coefficients / (routing_coefficients.sum(axis=-1, keepdims=True) + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)

        # Convert to agreement score (inverse of entropy)
        max_entropy = np.log(routing_coefficients.shape[-1])
        agreement = 1.0 - (entropy.mean() / max_entropy)

        # Weight by activation magnitude
        activation_weight = np.tanh(np.mean(np.abs(capsule_activations)))
        weighted_agreement = agreement * (0.5 + 0.5 * activation_weight)

        self.state.mean_agreement = weighted_agreement
        self.state.agreement_history.append(weighted_agreement)
        self.state.update_history()

        return weighted_agreement * self.config.agreement_to_stability

    def compute_pose_feedback(
        self,
        pose_matrices: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Compute attractor geometry hints from pose transformations.

        Analyzes pose matrix structure to provide hints about
        attractor geometry in the neural field.

        Returns:
        - coherence: Pose coherence score [0, 1]
        - geometry_hint: Array of attractor geometry suggestions
        """
        if not self.config.bidirectional:
            return 0.0, np.zeros(6)

        # Compute pose coherence via singular value analysis
        # Coherent poses have similar transformation structure
        if pose_matrices.ndim == 3:
            # Stack of pose matrices
            coherence_scores = []
            for pm in pose_matrices:
                u, s, vh = np.linalg.svd(pm)
                # Coherence from condition number (low = coherent)
                condition = s[0] / (s[-1] + 1e-8)
                coherence_scores.append(1.0 / (1.0 + np.log1p(condition)))
            coherence = np.mean(coherence_scores)
        else:
            u, s, vh = np.linalg.svd(pose_matrices)
            condition = s[0] / (s[-1] + 1e-8)
            coherence = 1.0 / (1.0 + np.log1p(condition))

        # Geometry hint: principal directions suggest attractor geometry
        # Map to 6-NT space (simplified)
        geometry_hint = np.zeros(6)
        if pose_matrices.ndim >= 2:
            flat = pose_matrices.flatten()[:6] if len(pose_matrices.flatten()) >= 6 else np.zeros(6)
            geometry_hint = np.tanh(flat[:6])

        self.state.pose_coherence = coherence

        return coherence * self.config.pose_to_attractor, geometry_hint

    def compute_activation_feedback(
        self,
        capsule_activations: np.ndarray
    ) -> np.ndarray:
        """
        Compute local field perturbation from capsule activations.

        Strong capsule activations perturb the neural field locally,
        enabling capsule-driven state transitions.

        Returns perturbation array for neural field.
        """
        if not self.config.bidirectional:
            return np.zeros(6)

        # Compute activation statistics
        magnitude = np.mean(np.abs(capsule_activations))
        self.state.activation_magnitude = magnitude

        # Map to 6-NT perturbation
        # Strong activations boost excitatory NTs (Glu, DA)
        # and slightly suppress inhibitory (GABA)
        perturbation = np.array([
            magnitude * 0.3,   # DA: salience
            0.0,               # 5-HT: stable
            magnitude * 0.2,   # ACh: attention
            magnitude * 0.1,   # NE: arousal
            -magnitude * 0.1,  # GABA: reduced inhibition
            magnitude * 0.2,   # Glu: excitation
        ]) * self.config.activation_to_field * self.config.coupling_strength.value

        return perturbation

    def compute_nca_feedback(
        self,
        routing_coefficients: np.ndarray,
        capsule_activations: np.ndarray,
        pose_matrices: np.ndarray | None = None
    ) -> dict:
        """
        Compute all NCA feedback from capsule state.

        Returns dict with:
        - stability_signal: Field stability from agreement
        - pose_coherence: Coherence score
        - geometry_hint: Attractor geometry suggestions
        - field_perturbation: NT perturbation array
        """
        stability = self.compute_agreement_feedback(
            routing_coefficients, capsule_activations
        )

        if pose_matrices is not None:
            coherence, geometry = self.compute_pose_feedback(pose_matrices)
        else:
            coherence, geometry = 0.0, np.zeros(6)

        perturbation = self.compute_activation_feedback(capsule_activations)

        return {
            "stability_signal": stability,
            "pose_coherence": coherence,
            "geometry_hint": geometry,
            "field_perturbation": perturbation,
            "mean_agreement": self.state.mean_agreement,
            "activation_magnitude": self.state.activation_magnitude,
        }

    # -------------------------------------------------------------------------
    # Full Coupling Step
    # -------------------------------------------------------------------------

    def step(
        self,
        nt_state: NeurotransmitterState,
        routing_coefficients: np.ndarray | None = None,
        capsule_activations: np.ndarray | None = None,
        pose_matrices: np.ndarray | None = None
    ) -> tuple[dict, dict]:
        """
        Execute full bidirectional coupling step.

        Args:
            nt_state: Current neurotransmitter state
            routing_coefficients: Current routing coefficients (optional)
            capsule_activations: Current capsule activations (optional)
            pose_matrices: Current pose matrices (optional)

        Returns:
            Tuple of (capsule_modulation, nca_feedback)
        """
        # Forward: NT → Capsule
        capsule_modulation = self.modulate_capsule_params(nt_state)

        # Backward: Capsule → NCA (if data provided)
        if routing_coefficients is not None and capsule_activations is not None:
            nca_feedback = self.compute_nca_feedback(
                routing_coefficients, capsule_activations, pose_matrices
            )
        else:
            nca_feedback = {
                "stability_signal": 0.0,
                "pose_coherence": 0.0,
                "geometry_hint": np.zeros(6),
                "field_perturbation": np.zeros(6),
                "mean_agreement": 0.0,
                "activation_magnitude": 0.0,
            }

        return capsule_modulation, nca_feedback

    def get_state(self) -> CapsuleNCACouplingState:
        """Get current coupling state for inspection."""
        return self.state

    def reset(self):
        """Reset coupling state."""
        self.state = CapsuleNCACouplingState()
        self._previous_ach = 0.5
        logger.debug("CapsuleNCACoupling state reset")

    # -------------------------------------------------------------------------
    # Phase 6: Integrated Forward with NT Modulation
    # -------------------------------------------------------------------------

    def forward_with_nt_modulation(
        self,
        capsule_layer,
        embedding: np.ndarray,
        nt_state: "NeurotransmitterState",
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        NT-modulated forward pass through capsule layer with routing.

        This is THE KEY Phase 6 integration: combines NT modulation with
        routing-based pose learning. Uses forward_with_routing() from
        CapsuleLayer with NT-determined parameters.

        Args:
            capsule_layer: CapsuleLayer instance
            embedding: Input embedding [input_dim]
            nt_state: Current neurotransmitter state

        Returns:
            activations: NT-modulated capsule activations
            poses: Poses refined through NT-modulated routing
            combined_stats: Dictionary with routing stats + NT modulation info
        """
        # Get NT modulation parameters
        capsule_params = self.modulate_capsule_params(nt_state)

        # Determine learning mode from ACh
        # High ACh (encoding) = learn poses
        # Low ACh (retrieval) = don't learn
        learn_poses = capsule_params["mode"] == CapsuleMode.ENCODING

        # Compute adjusted learning rate from DA (reward signal)
        # Higher DA = stronger learning
        base_lr = getattr(capsule_layer.config, 'learning_rate', 0.01)
        da_modulated_lr = base_lr * (0.5 + capsule_params["da_level"])

        # Forward with NT-modulated routing
        activations, poses, routing_stats = capsule_layer.forward_with_routing(
            embedding,
            routing_iterations=capsule_params["routing_iterations"],
            learn_poses=learn_poses,
            learning_rate=da_modulated_lr if learn_poses else None,
        )

        # Compute NCA feedback from capsule state
        if hasattr(capsule_layer.state, 'routing_coefficients') and \
           capsule_layer.state.routing_coefficients is not None:
            nca_feedback = self.compute_nca_feedback(
                routing_coefficients=capsule_layer.state.routing_coefficients,
                capsule_activations=activations,
                pose_matrices=poses,
            )
        else:
            nca_feedback = {
                "stability_signal": 0.0,
                "pose_coherence": 0.0,
                "mean_agreement": routing_stats.get('mean_agreement', 0.0),
            }

        # Combine stats
        combined_stats = {
            **routing_stats,
            "nt_mode": capsule_params["mode"].name,
            "nt_routing_temperature": capsule_params["routing_temperature"],
            "nt_squashing_threshold": capsule_params["squashing_threshold"],
            "nt_routing_iterations": capsule_params["routing_iterations"],
            "nt_learn_poses": learn_poses,
            "nt_learning_rate": da_modulated_lr if learn_poses else 0.0,
            "nca_stability_signal": nca_feedback.get("stability_signal", 0.0),
            "nca_pose_coherence": nca_feedback.get("pose_coherence", 0.0),
        }

        logger.debug(
            f"Phase 6: NT-modulated capsule forward - "
            f"mode={capsule_params['mode'].name}, "
            f"learn={learn_poses}, "
            f"agreement={routing_stats.get('mean_agreement', 0):.3f}"
        )

        return activations, poses, combined_stats


# =============================================================================
# Factory Functions
# =============================================================================


def create_capsule_nca_coupling(
    strength: str = "moderate",
    bidirectional: bool = True,
    **kwargs
) -> CapsuleNCACoupling:
    """
    Factory function for creating capsule-NCA coupling.

    Args:
        strength: Coupling strength ("weak", "moderate", "strong", "full")
        bidirectional: Enable both NT→capsule and capsule→NCA
        **kwargs: Additional config parameters

    Returns:
        Configured CapsuleNCACoupling instance
    """
    strength_map = {
        "weak": CouplingStrength.WEAK,
        "moderate": CouplingStrength.MODERATE,
        "strong": CouplingStrength.STRONG,
        "full": CouplingStrength.FULL,
    }

    config = CapsuleNCACouplingConfig(
        coupling_strength=strength_map.get(strength, CouplingStrength.MODERATE),
        bidirectional=bidirectional,
        **kwargs
    )

    return CapsuleNCACoupling(config)
