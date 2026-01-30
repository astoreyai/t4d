"""
Unified Learning Signals for World Weaver.

Phase 6B: Deep Integration of All Learning Systems

This module unifies all learning signals into a coherent update mechanism:
1. Three-factor learning (eligibility * neuromod * DA)
2. Forward-Forward goodness modulation
3. Capsule routing agreement modulation
4. Neurogenesis structural updates

Theoretical Foundation:
=======================

The brain doesn't have separate learning systems operating independently.
Instead, multiple signals gate and modulate each other:

1. Three-Factor Base (Gerstner 2018):
   - Eligibility trace: Which synapses were active?
   - Neuromodulator gate: Should we learn now?
   - Dopamine surprise: How surprising was this?

2. Forward-Forward Goodness (Hinton 2022):
   - Local learning signal based on activity energy
   - No backpropagation required
   - High goodness = "positive" data representation

3. Capsule Agreement (Sabour 2017):
   - Routing-by-agreement for part-whole binding
   - High agreement = consistent representation
   - Low agreement = structural mismatch

4. Neurogenesis Signals (Kempermann 2015):
   - Structural plasticity: birth/death of neurons
   - Activity-dependent survival
   - Novelty-driven growth

Integration Formula:
    weight_delta = three_factor * (ff_mod + capsule_mod)
    structure_delta = neurogenesis_signal

This creates a system where:
- All signals must align for strong weight updates
- Structural changes are gated by novelty and activity
- Learning is truly emergent, not hand-programmed

References:
- Gerstner et al. (2018): Eligibility traces and three-factor learning
- Hinton, G. (2022): The Forward-Forward Algorithm
- Sabour et al. (2017): Dynamic Routing Between Capsules
- Kempermann, G. (2015): Activity Dependency in Adult Neurogenesis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from ww.encoding.neurogenesis import NeurogenesisManager
    from ww.nca.capsules import CapsuleLayer
    from ww.nca.forward_forward import ForwardForwardLayer
    from ww.nca.pose_learner import PoseDimensionDiscovery

from ww.learning.dopamine import DopamineSystem, RewardPredictionError
from ww.learning.eligibility import EligibilityTrace
from ww.learning.neuromodulators import NeuromodulatorOrchestra, NeuromodulatorState
from ww.learning.three_factor import ThreeFactorLearningRule, ThreeFactorSignal

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class UpdateType(Enum):
    """Type of learning update."""

    WEIGHT = auto()     # Synaptic weight changes
    STRUCTURE = auto()  # Structural changes (neurogenesis)
    BOTH = auto()       # Combined weight and structure


class SignalSource(Enum):
    """Source of learning signal."""

    THREE_FACTOR = "three_factor"
    FF_GOODNESS = "ff_goodness"
    CAPSULE_AGREEMENT = "capsule_agreement"
    NEUROGENESIS = "neurogenesis"
    COMBINED = "combined"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LearningContext:
    """
    Complete context for computing unified learning signals.

    Contains all inputs needed by the UnifiedLearningSignal to compute
    appropriate weight and structure updates.

    Attributes:
        memory_id: ID of memory being updated
        eligibility: Eligibility trace value [0, 1]
        neuromod_gate: Neuromodulator gating signal [0, inf)
        dopamine_surprise: Dopamine prediction error magnitude [0, 1]
        rpe_signed: Signed reward prediction error [-1, 1]
        ff_goodness: Forward-Forward goodness value (if available)
        ff_threshold: FF threshold for positive/negative classification
        capsule_agreement: Mean capsule routing agreement [0, 1]
        capsule_activations: Capsule activation magnitudes (if available)
        novelty_score: Novelty score for neurogenesis [0, inf)
        activity_level: Current activity level [0, 1]
        importance: Memory importance for protection [0, inf)
        embedding: Memory embedding vector (if available)
        base_lr: Base learning rate
        timestamp: When context was created
    """

    memory_id: UUID

    # Three-factor inputs
    eligibility: float = 0.0
    neuromod_gate: float = 1.0
    dopamine_surprise: float = 0.0
    rpe_signed: float = 0.0

    # Forward-Forward inputs
    ff_goodness: float | None = None
    ff_threshold: float = 2.0
    ff_is_positive: bool = True
    ff_probability: float = 0.5

    # Capsule inputs
    capsule_agreement: float | None = None
    capsule_activations: np.ndarray | None = None

    # Neurogenesis inputs
    novelty_score: float = 0.0
    activity_level: float = 0.0

    # General inputs
    importance: float = 0.0
    embedding: np.ndarray | None = None
    base_lr: float = 0.01

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Serialize to dictionary for logging."""
        return {
            "memory_id": str(self.memory_id),
            "eligibility": self.eligibility,
            "neuromod_gate": self.neuromod_gate,
            "dopamine_surprise": self.dopamine_surprise,
            "rpe_signed": self.rpe_signed,
            "ff_goodness": self.ff_goodness,
            "ff_is_positive": self.ff_is_positive,
            "capsule_agreement": self.capsule_agreement,
            "novelty_score": self.novelty_score,
            "activity_level": self.activity_level,
            "importance": self.importance,
            "base_lr": self.base_lr,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StructuralUpdate:
    """
    Structural update signal for neurogenesis.

    Represents a suggested structural change to the network:
    birth of new neurons or pruning of inactive ones.

    Attributes:
        should_add_neuron: Whether to add a neuron
        should_prune: Whether to consider pruning
        target_layer_idx: Layer index for structural change
        novelty_score: Novelty that triggered this signal
        activity_level: Current activity level
        maturity_delta: Suggested maturity increase
        birth_probability: Probability of neuron birth [0, 1]
        prune_candidates: Number of candidates for pruning
    """

    should_add_neuron: bool = False
    should_prune: bool = False
    target_layer_idx: int = 0
    novelty_score: float = 0.0
    activity_level: float = 0.0
    maturity_delta: float = 0.1
    birth_probability: float = 0.0
    prune_candidates: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "should_add_neuron": self.should_add_neuron,
            "should_prune": self.should_prune,
            "target_layer_idx": self.target_layer_idx,
            "novelty_score": self.novelty_score,
            "activity_level": self.activity_level,
            "maturity_delta": self.maturity_delta,
            "birth_probability": self.birth_probability,
            "prune_candidates": self.prune_candidates,
        }


@dataclass
class LearningUpdate:
    """
    Combined learning update from unified signal computation.

    Contains both weight changes and structural changes, along with
    metadata about how the update was computed.

    Attributes:
        weight_delta: Weight update magnitude (effective learning rate)
        structure_delta: Structural update signal
        effective_lr: Final effective learning rate
        three_factor_contrib: Contribution from three-factor rule
        ff_contrib: Contribution from FF goodness
        capsule_contrib: Contribution from capsule agreement
        update_type: Type of update (weight/structure/both)
        signal_sources: Which signals contributed
        should_update: Whether to apply the update
        confidence: Confidence in this update [0, 1]
        timestamp: When update was computed
    """

    weight_delta: float = 0.0
    structure_delta: StructuralUpdate = field(default_factory=StructuralUpdate)

    # Breakdown of contributions
    effective_lr: float = 0.0
    three_factor_contrib: float = 0.0
    ff_contrib: float = 0.0
    capsule_contrib: float = 0.0

    # Metadata
    update_type: UpdateType = UpdateType.WEIGHT
    signal_sources: list[SignalSource] = field(default_factory=list)
    should_update: bool = False
    confidence: float = 0.0

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "weight_delta": self.weight_delta,
            "structure_delta": self.structure_delta.to_dict(),
            "effective_lr": self.effective_lr,
            "three_factor_contrib": self.three_factor_contrib,
            "ff_contrib": self.ff_contrib,
            "capsule_contrib": self.capsule_contrib,
            "update_type": self.update_type.name,
            "signal_sources": [s.value for s in self.signal_sources],
            "should_update": self.should_update,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class UnifiedSignalConfig:
    """
    Configuration for unified learning signal computation.

    Attributes:
        ff_weight: Weight for FF goodness contribution [0, 1]
        capsule_weight: Weight for capsule agreement contribution [0, 1]
        neurogenesis_threshold: Novelty threshold for neurogenesis
        min_eligibility: Minimum eligibility for weight updates
        min_update_threshold: Minimum signal for update
        max_effective_lr: Maximum effective learning rate
        importance_protection: How much importance protects from updates
        enable_ff_modulation: Enable FF goodness modulation
        enable_capsule_modulation: Enable capsule agreement modulation
        enable_neurogenesis: Enable structural updates
        history_size: Maximum history entries to retain
    """

    # Signal weights (should sum to <= 1.0 for modulation)
    ff_weight: float = 0.3
    capsule_weight: float = 0.3

    # Thresholds
    neurogenesis_threshold: float = 0.5
    min_eligibility: float = 0.01
    min_update_threshold: float = 0.001
    max_effective_lr: float = 3.0

    # Protection
    importance_protection: float = 1.0

    # Feature flags
    enable_ff_modulation: bool = True
    enable_capsule_modulation: bool = True
    enable_neurogenesis: bool = True

    # History
    history_size: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert 0 <= self.ff_weight <= 1, "ff_weight must be in [0, 1]"
        assert 0 <= self.capsule_weight <= 1, "capsule_weight must be in [0, 1]"
        assert self.min_eligibility >= 0, "min_eligibility must be non-negative"
        assert self.max_effective_lr > 0, "max_effective_lr must be positive"


# =============================================================================
# Unified Learning Signal
# =============================================================================


class UnifiedLearningSignal:
    """
    Combine all learning signals into coherent update.

    This is THE central integration point for World Weaver's learning systems.
    It takes signals from:
    1. Three-factor rule (eligibility, neuromod, dopamine)
    2. Forward-Forward encoder (goodness gradients)
    3. Capsule layer (routing agreement)
    4. Neurogenesis manager (structural signals)

    And produces unified weight and structure updates.

    Core Formula:
        weight_delta = three_factor * (1 + ff_mod * ff_weight + capsule_mod * capsule_weight)
        structure_delta = neurogenesis_signal

    Where:
        three_factor = eligibility * neuromod_gate * dopamine_surprise
        ff_mod = goodness_gradient (normalized)
        capsule_mod = agreement_gradient (normalized)
        neurogenesis_signal = based on novelty and activity

    Example:
        >>> unified = UnifiedLearningSignal()
        >>> context = LearningContext(
        ...     memory_id=uuid4(),
        ...     eligibility=0.8,
        ...     neuromod_gate=1.2,
        ...     dopamine_surprise=0.5,
        ...     ff_goodness=3.0,
        ...     capsule_agreement=0.7,
        ... )
        >>> update = unified.compute_update(context)
        >>> print(f"Effective LR: {update.effective_lr:.4f}")
    """

    def __init__(
        self,
        config: UnifiedSignalConfig | None = None,
        three_factor: ThreeFactorLearningRule | None = None,
        ff_encoder: ForwardForwardLayer | None = None,
        capsule_layer: CapsuleLayer | None = None,
        neurogenesis: NeurogenesisManager | None = None,
        pose_learner: PoseDimensionDiscovery | None = None,
    ):
        """
        Initialize unified learning signal.

        Args:
            config: Configuration for signal computation
            three_factor: Three-factor learning rule (created if None)
            ff_encoder: Forward-Forward layer for goodness (optional)
            capsule_layer: Capsule layer for agreement (optional)
            neurogenesis: Neurogenesis manager for structure (optional)
            pose_learner: Pose learner for emergent poses (optional)
        """
        self.config = config or UnifiedSignalConfig()

        # Core three-factor rule (always present)
        self._three_factor = three_factor or ThreeFactorLearningRule(
            min_eligibility_threshold=self.config.min_eligibility,
            max_effective_lr=self.config.max_effective_lr,
        )

        # Optional component references
        self._ff_encoder = ff_encoder
        self._capsule_layer = capsule_layer
        self._neurogenesis = neurogenesis
        self._pose_learner = pose_learner

        # History tracking (bounded)
        self._update_history: list[LearningUpdate] = []
        self._context_history: list[LearningContext] = []

        # Statistics
        self._total_updates = 0
        self._weight_updates = 0
        self._structure_updates = 0
        self._skipped_updates = 0

        logger.info(
            f"Phase 6B: UnifiedLearningSignal initialized "
            f"(ff={ff_encoder is not None}, "
            f"capsule={capsule_layer is not None}, "
            f"neurogenesis={neurogenesis is not None})"
        )

    # -------------------------------------------------------------------------
    # Component Setters
    # -------------------------------------------------------------------------

    def set_ff_encoder(self, ff_encoder: ForwardForwardLayer) -> None:
        """Set Forward-Forward encoder for goodness modulation."""
        self._ff_encoder = ff_encoder
        logger.debug("Phase 6B: FF encoder connected to UnifiedLearningSignal")

    def set_capsule_layer(self, capsule_layer: CapsuleLayer) -> None:
        """Set capsule layer for agreement modulation."""
        self._capsule_layer = capsule_layer
        logger.debug("Phase 6B: Capsule layer connected to UnifiedLearningSignal")

    def set_neurogenesis(self, neurogenesis: NeurogenesisManager) -> None:
        """Set neurogenesis manager for structural updates."""
        self._neurogenesis = neurogenesis
        logger.debug("Phase 6B: Neurogenesis connected to UnifiedLearningSignal")

    def set_pose_learner(self, pose_learner: PoseDimensionDiscovery) -> None:
        """Set pose learner for emergent pose dimensions."""
        self._pose_learner = pose_learner
        logger.debug("Phase 6B: Pose learner connected to UnifiedLearningSignal")

    # -------------------------------------------------------------------------
    # Signal Extraction
    # -------------------------------------------------------------------------

    def get_ff_goodness_gradient(self) -> float:
        """
        Get goodness gradient from Forward-Forward encoder.

        The gradient indicates how much to adjust weights to improve
        goodness for positive samples or decrease for negative.

        Returns:
            Normalized goodness gradient [-1, 1]
        """
        if self._ff_encoder is None or not self.config.enable_ff_modulation:
            return 0.0

        state = self._ff_encoder.state

        # Get recent goodness history
        if state.positive_goodness_history and state.negative_goodness_history:
            # Compute separation: how well is positive vs negative distinguished?
            pos_mean = np.mean(state.positive_goodness_history[-100:])
            neg_mean = np.mean(state.negative_goodness_history[-100:])

            # Separation normalized by threshold
            separation = (pos_mean - neg_mean) / (self._ff_encoder.config.threshold_theta + 1e-8)

            # High separation = good learning, low gradient needed
            # Low separation = needs more learning
            gradient = 1.0 - min(1.0, max(0.0, separation))

            return float(gradient)

        # Default: use current goodness relative to threshold
        current_goodness = state.normalized_goodness
        threshold = self._ff_encoder.config.threshold_theta / self._ff_encoder.config.hidden_dim

        # Distance from threshold, normalized
        distance = current_goodness - threshold
        gradient = np.tanh(distance)  # Squash to [-1, 1]

        return float(gradient)

    def get_capsule_agreement_gradient(self) -> float:
        """
        Get agreement gradient from capsule layer.

        The gradient indicates how much routing agreement changed,
        which signals whether representations are stable.

        Returns:
            Normalized agreement gradient [0, 1]
        """
        if self._capsule_layer is None or not self.config.enable_capsule_modulation:
            return 0.0

        state = self._capsule_layer.state

        # Get mean agreement
        if state.mean_agreement > 0:
            # Agreement in [0, 1], use directly as modulation
            # High agreement = stable representation, less update needed
            # Low agreement = unstable, more update needed
            gradient = 1.0 - state.mean_agreement
            return float(gradient)

        return 0.0

    def get_structural_update(self, context: LearningContext) -> StructuralUpdate:
        """
        Get structural update signal from neurogenesis manager.

        Determines whether to add or remove neurons based on
        novelty scores and activity levels.

        Args:
            context: Learning context with novelty and activity info

        Returns:
            StructuralUpdate with birth/prune recommendations
        """
        update = StructuralUpdate(
            novelty_score=context.novelty_score,
            activity_level=context.activity_level,
        )

        if self._neurogenesis is None or not self.config.enable_neurogenesis:
            return update

        # Check novelty threshold for birth
        if context.novelty_score > self.config.neurogenesis_threshold:
            # High novelty: consider adding neuron
            # Probability scales with novelty magnitude
            birth_prob = min(1.0, context.novelty_score / (self.config.neurogenesis_threshold * 2))
            birth_prob *= self._neurogenesis.config.birth_rate

            update.should_add_neuron = True
            update.birth_probability = birth_prob

        # Check activity for pruning
        if context.activity_level < self._neurogenesis.config.survival_threshold:
            update.should_prune = True

        # Maturity increase based on activity
        update.maturity_delta = self._neurogenesis.config.integration_rate * context.activity_level

        return update

    # -------------------------------------------------------------------------
    # Core Computation
    # -------------------------------------------------------------------------

    def compute_update(self, context: LearningContext) -> LearningUpdate:
        """
        Compute unified learning update from context.

        This is THE main entry point. It combines all available signals
        into a coherent update that can be applied to the system.

        Args:
            context: LearningContext with all inputs

        Returns:
            LearningUpdate with weight and structure deltas
        """
        update = LearningUpdate()
        update.signal_sources = []

        # === Step 1: Three-Factor Base ===
        # This is always the foundation of learning
        three_factor = (
            context.eligibility *
            context.neuromod_gate *
            context.dopamine_surprise
        )

        # Handle NaN/Inf
        if not np.isfinite(three_factor):
            logger.warning(f"Non-finite three_factor: {three_factor}, using 0.0")
            three_factor = 0.0

        update.three_factor_contrib = three_factor
        update.signal_sources.append(SignalSource.THREE_FACTOR)

        # === Step 2: FF Goodness Modulation ===
        ff_mod = 0.0
        if self.config.enable_ff_modulation:
            if context.ff_goodness is not None:
                # Use provided goodness
                goodness_distance = context.ff_goodness - context.ff_threshold
                # Convert to modulation factor
                # Positive distance (good goodness) = positive modulation
                ff_mod = np.tanh(goodness_distance / context.ff_threshold)
            else:
                # Get from encoder
                ff_mod = self.get_ff_goodness_gradient()

            if abs(ff_mod) > 0.001:
                update.signal_sources.append(SignalSource.FF_GOODNESS)

        update.ff_contrib = ff_mod * self.config.ff_weight

        # === Step 3: Capsule Agreement Modulation ===
        capsule_mod = 0.0
        if self.config.enable_capsule_modulation:
            if context.capsule_agreement is not None:
                # Use provided agreement
                # Low agreement = more learning needed
                capsule_mod = 1.0 - context.capsule_agreement
            else:
                # Get from layer
                capsule_mod = self.get_capsule_agreement_gradient()

            if abs(capsule_mod) > 0.001:
                update.signal_sources.append(SignalSource.CAPSULE_AGREEMENT)

        update.capsule_contrib = capsule_mod * self.config.capsule_weight

        # === Step 4: Combined Weight Delta ===
        # Formula: three_factor * (1 + ff_mod + capsule_mod)
        # The (1 + ...) ensures three_factor is the base, modulated by others
        modulation_sum = update.ff_contrib + update.capsule_contrib
        weight_delta = three_factor * (1.0 + modulation_sum)

        # Apply bounds
        weight_delta = np.clip(weight_delta, 0.0, self.config.max_effective_lr)

        # Apply importance protection
        if context.importance > 0:
            protection_factor = 1.0 / (1.0 + context.importance * self.config.importance_protection)
            weight_delta *= protection_factor

        update.weight_delta = float(weight_delta)
        update.effective_lr = context.base_lr * update.weight_delta

        # === Step 5: Neurogenesis Signal ===
        update.structure_delta = self.get_structural_update(context)

        if update.structure_delta.should_add_neuron or update.structure_delta.should_prune:
            update.signal_sources.append(SignalSource.NEUROGENESIS)

        # === Step 6: Determine Update Type ===
        has_weight_update = update.weight_delta > self.config.min_update_threshold
        has_structure_update = (
            update.structure_delta.should_add_neuron or
            update.structure_delta.should_prune
        )

        if has_weight_update and has_structure_update:
            update.update_type = UpdateType.BOTH
        elif has_weight_update:
            update.update_type = UpdateType.WEIGHT
        elif has_structure_update:
            update.update_type = UpdateType.STRUCTURE

        # Should we apply this update?
        update.should_update = has_weight_update or has_structure_update

        # Confidence based on how many signals aligned
        if update.should_update:
            # More signals = higher confidence
            num_signals = len(update.signal_sources)
            update.confidence = min(1.0, num_signals / 3.0)

            # Boost confidence if signals agree in direction
            if update.ff_contrib > 0 and update.capsule_contrib > 0:
                update.confidence = min(1.0, update.confidence * 1.2)

        # === Step 7: Record and Return ===
        self._record_update(context, update)

        logger.debug(
            f"Phase 6B: Unified update computed "
            f"(weight={update.weight_delta:.4f}, "
            f"eff_lr={update.effective_lr:.6f}, "
            f"sources={[s.value for s in update.signal_sources]})"
        )

        return update

    def compute_from_three_factor_signal(
        self,
        signal: ThreeFactorSignal,
        ff_goodness: float | None = None,
        capsule_agreement: float | None = None,
        novelty_score: float = 0.0,
        activity_level: float = 0.0,
        base_lr: float = 0.01,
    ) -> LearningUpdate:
        """
        Convenience method to compute update from ThreeFactorSignal.

        Args:
            signal: Pre-computed three-factor signal
            ff_goodness: Optional FF goodness value
            capsule_agreement: Optional capsule agreement
            novelty_score: Novelty for neurogenesis
            activity_level: Current activity level
            base_lr: Base learning rate

        Returns:
            Unified learning update
        """
        context = LearningContext(
            memory_id=signal.memory_id,
            eligibility=signal.eligibility,
            neuromod_gate=signal.neuromod_gate,
            dopamine_surprise=signal.dopamine_surprise,
            rpe_signed=signal.rpe_raw,
            ff_goodness=ff_goodness,
            capsule_agreement=capsule_agreement,
            novelty_score=novelty_score,
            activity_level=activity_level,
            base_lr=base_lr,
        )

        return self.compute_update(context)

    def batch_compute(
        self,
        contexts: list[LearningContext],
    ) -> list[LearningUpdate]:
        """
        Compute updates for multiple contexts.

        Args:
            contexts: List of learning contexts

        Returns:
            List of learning updates
        """
        return [self.compute_update(ctx) for ctx in contexts]

    # -------------------------------------------------------------------------
    # History and Statistics
    # -------------------------------------------------------------------------

    def _record_update(self, context: LearningContext, update: LearningUpdate) -> None:
        """Record update to history."""
        self._context_history.append(context)
        self._update_history.append(update)

        self._total_updates += 1
        if update.should_update:
            if update.update_type in (UpdateType.WEIGHT, UpdateType.BOTH):
                self._weight_updates += 1
            if update.update_type in (UpdateType.STRUCTURE, UpdateType.BOTH):
                self._structure_updates += 1
        else:
            self._skipped_updates += 1

        # Trim history
        if len(self._update_history) > self.config.history_size:
            self._update_history = self._update_history[-self.config.history_size:]
            self._context_history = self._context_history[-self.config.history_size:]

    def get_recent_updates(self, n: int = 100) -> list[LearningUpdate]:
        """Get n most recent updates."""
        return self._update_history[-n:]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive statistics about unified signals.

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_updates": self._total_updates,
            "weight_updates": self._weight_updates,
            "structure_updates": self._structure_updates,
            "skipped_updates": self._skipped_updates,
            "update_rate": self._weight_updates / max(1, self._total_updates),
            "components": {
                "ff_encoder": self._ff_encoder is not None,
                "capsule_layer": self._capsule_layer is not None,
                "neurogenesis": self._neurogenesis is not None,
                "pose_learner": self._pose_learner is not None,
            },
            "config": {
                "ff_weight": self.config.ff_weight,
                "capsule_weight": self.config.capsule_weight,
                "neurogenesis_threshold": self.config.neurogenesis_threshold,
            },
        }

        # Recent update statistics
        if self._update_history:
            recent = self._update_history[-100:]
            stats["recent"] = {
                "mean_weight_delta": float(np.mean([u.weight_delta for u in recent])),
                "mean_effective_lr": float(np.mean([u.effective_lr for u in recent])),
                "mean_three_factor": float(np.mean([u.three_factor_contrib for u in recent])),
                "mean_ff_contrib": float(np.mean([u.ff_contrib for u in recent])),
                "mean_capsule_contrib": float(np.mean([u.capsule_contrib for u in recent])),
                "mean_confidence": float(np.mean([u.confidence for u in recent])),
            }

        return stats

    def clear_history(self) -> None:
        """Clear update history to free memory."""
        self._update_history.clear()
        self._context_history.clear()
        logger.info("Phase 6B: Unified signal history cleared")

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self._total_updates = 0
        self._weight_updates = 0
        self._structure_updates = 0
        self._skipped_updates = 0
        logger.info("Phase 6B: Unified signal statistics reset")


# =============================================================================
# Factory Functions
# =============================================================================


def create_unified_signal(
    ff_weight: float = 0.3,
    capsule_weight: float = 0.3,
    enable_neurogenesis: bool = True,
    max_effective_lr: float = 3.0,
) -> UnifiedLearningSignal:
    """
    Create a unified learning signal with common defaults.

    Args:
        ff_weight: Weight for FF goodness contribution
        capsule_weight: Weight for capsule agreement contribution
        enable_neurogenesis: Enable structural updates
        max_effective_lr: Maximum effective learning rate

    Returns:
        Configured UnifiedLearningSignal
    """
    config = UnifiedSignalConfig(
        ff_weight=ff_weight,
        capsule_weight=capsule_weight,
        enable_neurogenesis=enable_neurogenesis,
        max_effective_lr=max_effective_lr,
    )
    return UnifiedLearningSignal(config=config)


def create_fully_integrated_signal(
    three_factor: ThreeFactorLearningRule | None = None,
    ff_encoder: ForwardForwardLayer | None = None,
    capsule_layer: CapsuleLayer | None = None,
    neurogenesis: NeurogenesisManager | None = None,
    pose_learner: PoseDimensionDiscovery | None = None,
    config: UnifiedSignalConfig | None = None,
) -> UnifiedLearningSignal:
    """
    Create a fully integrated unified learning signal with all components.

    This is the recommended factory for production use when all
    components are available.

    Args:
        three_factor: Three-factor learning rule
        ff_encoder: Forward-Forward layer
        capsule_layer: Capsule layer
        neurogenesis: Neurogenesis manager
        pose_learner: Pose dimension learner
        config: Optional configuration

    Returns:
        Fully configured UnifiedLearningSignal
    """
    return UnifiedLearningSignal(
        config=config,
        three_factor=three_factor,
        ff_encoder=ff_encoder,
        capsule_layer=capsule_layer,
        neurogenesis=neurogenesis,
        pose_learner=pose_learner,
    )


__all__ = [
    # Core classes
    "UnifiedLearningSignal",
    "LearningContext",
    "LearningUpdate",
    "StructuralUpdate",
    "UnifiedSignalConfig",
    # Enums
    "UpdateType",
    "SignalSource",
    # Factory functions
    "create_unified_signal",
    "create_fully_integrated_signal",
]
