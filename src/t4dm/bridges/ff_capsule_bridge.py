"""
Phase 6A: FF-Capsule Bridge for T4DM.

Unifies Forward-Forward goodness with capsule routing agreement into a
single coherent signal for memory encoding and retrieval confidence.

Theoretical Foundation:
=======================

This bridge addresses a key insight from Hinton's work: both FF goodness
and capsule routing agreement measure pattern coherence, but in different ways:

1. FF Goodness (Hinton 2022):
   - Measures how well activations "fit" learned patterns
   - Local: each layer computes sum of squared activations
   - High goodness = familiar/positive pattern

2. Capsule Routing Agreement (Sabour et al. 2017, Hinton et al. 2018):
   - Measures consistency of part-whole predictions
   - Compositional: lower capsules vote on higher capsule poses
   - High agreement = coherent compositional structure

Combined Signal:
   confidence = alpha * ff_goodness + (1 - alpha) * routing_agreement

This combination provides:
- Familiarity (FF): "Have we seen this pattern before?"
- Structure (Capsules): "Do the parts form a coherent whole?"

Biological Mapping:
- FF goodness ~ metabolic efficiency / pattern recognition ease
- Routing agreement ~ binding through synchrony
- Combined confidence ~ integrated recognition signal

Integration with T4DM:
- Used by retrieval for confidence-weighted ranking
- Feeds back to learning systems via joint outcome signal
- Enables end-to-end representation learning

References:
- Hinton, G. (2022). The Forward-Forward Algorithm
- Sabour, Frosst, Hinton (2017). Dynamic Routing Between Capsules
- Hinton, Sabour, Frosst (2018). Matrix Capsules with EM Routing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols for Type Safety
# =============================================================================


class FFEncoderProtocol(Protocol):
    """Protocol for Forward-Forward encoder interface."""

    def encode(self, embedding: np.ndarray, training: bool = False) -> np.ndarray:
        """Encode embedding through FF layers."""
        ...

    def get_goodness(self, embedding: np.ndarray) -> float:
        """Get goodness score for embedding."""
        ...

    def learn_from_outcome(
        self,
        embedding: np.ndarray,
        outcome_score: float,
        three_factor_signal: Any | None = None,
        effective_lr: float | None = None,
    ) -> dict:
        """Update FF encoder from retrieval outcome."""
        ...

    @property
    def state(self) -> Any:
        """Access encoder state."""
        ...


class CapsuleLayerProtocol(Protocol):
    """Protocol for capsule layer interface."""

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass returning activations and poses."""
        ...

    def forward_with_routing(
        self,
        x: np.ndarray,
        routing_iterations: int | None = None,
        learn_poses: bool = True,
        learning_rate: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Forward pass with routing, returning activations, poses, and stats."""
        ...

    def learn_pose_from_routing(
        self,
        lower_poses: np.ndarray,
        predictions: np.ndarray,
        consensus_poses: np.ndarray,
        agreement_scores: np.ndarray,
        learning_rate: float | None = None,
    ) -> dict:
        """Update pose weights from routing agreement."""
        ...

    @property
    def state(self) -> Any:
        """Access layer state."""
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FFCapsuleBridgeConfig:
    """
    Configuration for FF-Capsule Bridge.

    Attributes:
        ff_weight: Weight for FF goodness in combined confidence (0-1)
        capsule_weight: Weight for routing agreement (auto-computed as 1 - ff_weight)
        goodness_threshold: FF goodness threshold for normalization
        agreement_threshold: Minimum routing agreement for contribution
        normalize_goodness: Whether to normalize goodness to [0, 1]
        normalize_agreement: Whether to normalize agreement to [0, 1]
        joint_learning: Enable joint learning from outcomes
        ff_learning_rate: Learning rate for FF encoder
        capsule_learning_rate: Learning rate for capsule pose updates
        track_history: Track statistics history
        max_history: Maximum history size
    """
    ff_weight: float = 0.6
    goodness_threshold: float = 2.0
    agreement_threshold: float = 0.3
    normalize_goodness: bool = True
    normalize_agreement: bool = True
    joint_learning: bool = True
    ff_learning_rate: float = 0.03
    capsule_learning_rate: float = 0.01
    track_history: bool = True
    max_history: int = 1000

    @property
    def capsule_weight(self) -> float:
        """Auto-computed capsule weight."""
        return 1.0 - self.ff_weight

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert 0.0 <= self.ff_weight <= 1.0, "ff_weight must be in [0, 1]"
        assert self.goodness_threshold > 0, "goodness_threshold must be positive"
        assert 0.0 <= self.agreement_threshold <= 1.0, "agreement_threshold must be in [0, 1]"


@dataclass
class FFCapsuleBridgeState:
    """
    State tracking for FF-Capsule Bridge.

    Tracks both FF and capsule statistics plus combined metrics.
    """
    # Forward pass counts
    total_forwards: int = 0

    # FF statistics
    last_ff_goodness: float = 0.0
    mean_ff_goodness: float = 0.0
    ff_above_threshold_count: int = 0

    # Capsule statistics
    last_routing_agreement: float = 0.0
    mean_routing_agreement: float = 0.0
    last_pose_change: float = 0.0

    # Combined statistics
    last_confidence: float = 0.0
    mean_confidence: float = 0.0

    # Learning statistics
    total_learn_calls: int = 0
    total_positive_outcomes: int = 0
    total_negative_outcomes: int = 0
    mean_outcome_score: float = 0.5

    # History (bounded)
    goodness_history: list[float] = field(default_factory=list)
    agreement_history: list[float] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)
    outcome_history: list[float] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "total_forwards": self.total_forwards,
            "last_ff_goodness": self.last_ff_goodness,
            "mean_ff_goodness": self.mean_ff_goodness,
            "last_routing_agreement": self.last_routing_agreement,
            "mean_routing_agreement": self.mean_routing_agreement,
            "last_confidence": self.last_confidence,
            "mean_confidence": self.mean_confidence,
            "total_learn_calls": self.total_learn_calls,
            "mean_outcome_score": self.mean_outcome_score,
        }


@dataclass
class CapsuleState:
    """
    Capsule state from forward pass with routing.

    Captures both activations and routing information.
    """
    activations: np.ndarray
    poses: np.ndarray
    agreement: float
    routing_stats: dict = field(default_factory=dict)


# =============================================================================
# FF-Capsule Bridge
# =============================================================================


class FFCapsuleBridge:
    """
    Unifies FF goodness with capsule routing agreement.

    Phase 6A: This bridge is THE key integration point between Hinton's
    Forward-Forward algorithm and capsule networks, creating a unified
    signal for pattern recognition confidence.

    The combined confidence measure:
        confidence = ff_weight * normalized_goodness + capsule_weight * routing_agreement

    Provides richer information than either alone:
    - FF goodness: How well does this match learned patterns? (familiarity)
    - Routing agreement: How coherent is the part-whole structure? (composition)

    Joint learning enables both systems to improve together:
    - Positive outcomes: Increase goodness AND strengthen routing
    - Negative outcomes: Decrease goodness AND weaken conflicting routes

    Example:
        ```python
        from t4dm.encoding.ff_encoder import FFEncoder
        from t4dm.nca.capsules import CapsuleLayer, CapsuleConfig
        from t4dm.bridges.ff_capsule_bridge import FFCapsuleBridge

        # Create components
        ff_encoder = FFEncoder()
        capsule_layer = CapsuleLayer(CapsuleConfig(input_dim=1024))

        # Create bridge
        bridge = FFCapsuleBridge(
            ff_encoder=ff_encoder,
            capsule_layer=capsule_layer
        )

        # Forward pass
        embedding = np.random.randn(1024).astype(np.float32)
        ff_output, capsule_state, confidence = bridge.forward(embedding)

        logger.info(f"Confidence: {confidence:.3f}")
        logger.info(f"FF goodness: {bridge.state.last_ff_goodness:.3f}")
        logger.info(f"Routing agreement: {bridge.state.last_routing_agreement:.3f}")

        # After retrieval outcome
        bridge.learn(outcome=0.8)  # Helpful retrieval
        ```
    """

    def __init__(
        self,
        ff_encoder: FFEncoderProtocol | None = None,
        capsule_layer: CapsuleLayerProtocol | None = None,
        config: FFCapsuleBridgeConfig | None = None,
    ):
        """
        Initialize FF-Capsule Bridge.

        Args:
            ff_encoder: FFEncoder instance for goodness computation
            capsule_layer: CapsuleLayer instance for routing
            config: Bridge configuration
        """
        self.ff_encoder = ff_encoder
        self.capsule_layer = capsule_layer
        self.config = config or FFCapsuleBridgeConfig()
        self.state = FFCapsuleBridgeState()

        # Track last inputs for learning
        self._last_embedding: np.ndarray | None = None
        self._last_ff_output: np.ndarray | None = None
        self._last_capsule_state: CapsuleState | None = None

        logger.info(
            f"Phase 6A: FFCapsuleBridge initialized "
            f"(ff_weight={self.config.ff_weight:.2f}, "
            f"capsule_weight={self.config.capsule_weight:.2f})"
        )

    def set_ff_encoder(self, ff_encoder: FFEncoderProtocol) -> None:
        """
        Set or update the FF encoder.

        Args:
            ff_encoder: FFEncoder instance
        """
        self.ff_encoder = ff_encoder
        logger.info("Phase 6A: FF encoder updated")

    def set_capsule_layer(self, capsule_layer: CapsuleLayerProtocol) -> None:
        """
        Set or update the capsule layer.

        Args:
            capsule_layer: CapsuleLayer instance
        """
        self.capsule_layer = capsule_layer
        logger.info("Phase 6A: Capsule layer updated")

    def _compute_ff_goodness(self, embedding: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute FF encoding and goodness.

        Args:
            embedding: Input embedding

        Returns:
            Tuple of (ff_output, goodness)
        """
        if self.ff_encoder is not None:
            ff_output = self.ff_encoder.encode(embedding, training=False)
            goodness = self.ff_encoder.get_goodness(embedding)
        else:
            # Fallback: use embedding directly, compute proxy goodness
            ff_output = embedding
            goodness = float(np.sum(embedding ** 2) / len(embedding))

        return ff_output, goodness

    def _compute_capsule_routing(
        self,
        ff_output: np.ndarray,
        learn_poses: bool = False,
    ) -> CapsuleState:
        """
        Compute capsule routing and agreement.

        Args:
            ff_output: Output from FF encoder (input to capsules)
            learn_poses: Whether to update pose weights during routing

        Returns:
            CapsuleState with activations, poses, and agreement
        """
        if self.capsule_layer is not None:
            activations, poses, routing_stats = self.capsule_layer.forward_with_routing(
                ff_output,
                learn_poses=learn_poses,
                learning_rate=self.config.capsule_learning_rate if learn_poses else None,
            )
            agreement = routing_stats.get('mean_agreement', 0.0)
        else:
            # Fallback: mock capsule response
            n_capsules = 32
            activations = np.abs(ff_output[:n_capsules]) if len(ff_output) >= n_capsules else np.zeros(n_capsules)
            activations = activations / (np.linalg.norm(activations) + 1e-8)
            poses = np.eye(4, dtype=np.float32)[np.newaxis, :, :].repeat(n_capsules, axis=0)
            agreement = float(np.mean(activations))  # Proxy: mean activation
            routing_stats = {'mean_agreement': agreement, 'fallback': True}

        return CapsuleState(
            activations=activations,
            poses=poses,
            agreement=agreement,
            routing_stats=routing_stats,
        )

    def _normalize_goodness(self, goodness: float) -> float:
        """
        Normalize goodness to [0, 1] range.

        Uses sigmoid around threshold for smooth normalization.

        Args:
            goodness: Raw goodness value

        Returns:
            Normalized goodness in [0, 1]
        """
        if not self.config.normalize_goodness:
            return goodness

        # Sigmoid centered on threshold
        distance = goodness - self.config.goodness_threshold
        normalized = 1.0 / (1.0 + np.exp(-distance))

        return float(normalized)

    def _normalize_agreement(self, agreement: float) -> float:
        """
        Normalize routing agreement.

        Agreement is already in [0, 1] from capsule routing,
        but we can apply threshold-based adjustment.

        Args:
            agreement: Raw agreement value

        Returns:
            Normalized agreement in [0, 1]
        """
        if not self.config.normalize_agreement:
            return agreement

        # Apply threshold: below threshold contributes less
        if agreement < self.config.agreement_threshold:
            # Scale down low-agreement signals
            normalized = agreement * (agreement / self.config.agreement_threshold)
        else:
            normalized = agreement

        return float(np.clip(normalized, 0.0, 1.0))

    def _compute_combined_confidence(
        self,
        goodness: float,
        agreement: float,
    ) -> float:
        """
        Compute combined confidence from FF goodness and routing agreement.

        Args:
            goodness: FF goodness (raw or normalized)
            agreement: Routing agreement (raw or normalized)

        Returns:
            Combined confidence in [0, 1]
        """
        # Normalize if configured
        norm_goodness = self._normalize_goodness(goodness)
        norm_agreement = self._normalize_agreement(agreement)

        # Weighted combination
        confidence = (
            self.config.ff_weight * norm_goodness +
            self.config.capsule_weight * norm_agreement
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def _update_state(
        self,
        goodness: float,
        agreement: float,
        confidence: float,
        pose_change: float = 0.0,
    ) -> None:
        """
        Update bridge state with new values.

        Args:
            goodness: FF goodness value
            agreement: Routing agreement value
            confidence: Combined confidence
            pose_change: Pose change from routing (if available)
        """
        # Update counts
        self.state.total_forwards += 1

        # Update last values
        self.state.last_ff_goodness = goodness
        self.state.last_routing_agreement = agreement
        self.state.last_confidence = confidence
        self.state.last_pose_change = pose_change

        # Update running means (EMA)
        alpha = 0.01
        self.state.mean_ff_goodness = (
            alpha * goodness + (1 - alpha) * self.state.mean_ff_goodness
        )
        self.state.mean_routing_agreement = (
            alpha * agreement + (1 - alpha) * self.state.mean_routing_agreement
        )
        self.state.mean_confidence = (
            alpha * confidence + (1 - alpha) * self.state.mean_confidence
        )

        # Update threshold count
        if goodness > self.config.goodness_threshold:
            self.state.ff_above_threshold_count += 1

        # Update timestamp
        self.state.timestamp = datetime.now()

        # Track history (bounded)
        if self.config.track_history:
            self.state.goodness_history.append(goodness)
            self.state.agreement_history.append(agreement)
            self.state.confidence_history.append(confidence)

            # Trim to max history
            max_hist = self.config.max_history
            if len(self.state.goodness_history) > max_hist:
                self.state.goodness_history = self.state.goodness_history[-max_hist:]
            if len(self.state.agreement_history) > max_hist:
                self.state.agreement_history = self.state.agreement_history[-max_hist:]
            if len(self.state.confidence_history) > max_hist:
                self.state.confidence_history = self.state.confidence_history[-max_hist:]

    def forward(
        self,
        embedding: np.ndarray,
        learn_poses: bool = False,
    ) -> tuple[np.ndarray, CapsuleState, float]:
        """
        Forward pass through both FF encoder and capsule routing.

        This is THE main method for using the bridge. It:
        1. Encodes embedding through FF layers (computes goodness)
        2. Routes FF output through capsules (computes agreement)
        3. Combines both into unified confidence score

        Args:
            embedding: Input embedding vector
            learn_poses: Whether to update capsule pose weights during routing

        Returns:
            Tuple of (ff_output, capsule_state, combined_confidence)
        """
        embedding = np.atleast_1d(embedding).astype(np.float32)

        # Store for learning
        self._last_embedding = embedding.copy()

        # Step 1: FF encoding
        ff_output, ff_goodness = self._compute_ff_goodness(embedding)
        self._last_ff_output = ff_output.copy()

        # Step 2: Capsule routing
        capsule_state = self._compute_capsule_routing(ff_output, learn_poses=learn_poses)
        self._last_capsule_state = capsule_state

        # Step 3: Combined confidence
        confidence = self._compute_combined_confidence(ff_goodness, capsule_state.agreement)

        # Update state
        pose_change = capsule_state.routing_stats.get('pose_change', 0.0)
        self._update_state(ff_goodness, capsule_state.agreement, confidence, pose_change)

        logger.debug(
            f"Phase 6A: Forward - goodness={ff_goodness:.3f}, "
            f"agreement={capsule_state.agreement:.3f}, confidence={confidence:.3f}"
        )

        return ff_output, capsule_state, confidence

    def compute_confidence(self, embedding: np.ndarray) -> float:
        """
        Compute confidence without returning intermediate outputs.

        Convenience method when only confidence is needed.

        Args:
            embedding: Input embedding vector

        Returns:
            Combined confidence score
        """
        _, _, confidence = self.forward(embedding, learn_poses=False)
        return confidence

    def learn(
        self,
        outcome: float,
        embedding: np.ndarray | None = None,
        effective_lr: float | None = None,
    ) -> dict:
        """
        Joint learning from outcome signal.

        This is THE key learning method. Both FF and capsule systems
        learn from the same outcome signal, enabling end-to-end
        representation learning.

        Positive outcomes (outcome > 0.5):
        - FF: Increase goodness for this pattern
        - Capsules: Reinforce routing that led to this

        Negative outcomes (outcome < 0.5):
        - FF: Decrease goodness for this pattern
        - Capsules: Weaken conflicting routes

        Args:
            outcome: Retrieval/task outcome in [0, 1]
            embedding: Optional embedding (uses last if None)
            effective_lr: Optional override learning rate

        Returns:
            Dictionary of learning statistics from both systems
        """
        if not self.config.joint_learning:
            return {"status": "joint_learning_disabled"}

        # Use provided or last embedding
        if embedding is not None:
            embedding = np.atleast_1d(embedding).astype(np.float32)
        elif self._last_embedding is not None:
            embedding = self._last_embedding
        else:
            return {"status": "no_embedding", "error": "No embedding available for learning"}

        stats = {
            "outcome": outcome,
            "ff_stats": {},
            "capsule_stats": {},
        }

        # FF learning
        if self.ff_encoder is not None:
            ff_lr = effective_lr or self.config.ff_learning_rate
            ff_stats = self.ff_encoder.learn_from_outcome(
                embedding=embedding,
                outcome_score=outcome,
                effective_lr=ff_lr,
            )
            stats["ff_stats"] = ff_stats

        # Capsule learning (already happens during forward_with_routing if enabled)
        # Here we do additional outcome-based learning
        if self.capsule_layer is not None and self._last_capsule_state is not None:
            capsule_lr = effective_lr or self.config.capsule_learning_rate

            # Modulate capsule learning by outcome
            # High outcome = reinforce, low outcome = weaken
            outcome_modulated_lr = capsule_lr * (2.0 * outcome - 1.0)  # Range: [-lr, +lr]

            # If we have access to learn_positive/learn_negative methods
            try:
                if outcome > 0.5:
                    capsule_stats = self.capsule_layer.learn_positive(
                        self._last_ff_output,
                        self._last_capsule_state.activations,
                        poses=self._last_capsule_state.poses,
                        learning_rate=abs(outcome_modulated_lr),
                    )
                else:
                    capsule_stats = self.capsule_layer.learn_negative(
                        self._last_ff_output,
                        self._last_capsule_state.activations,
                        poses=self._last_capsule_state.poses,
                        learning_rate=abs(outcome_modulated_lr),
                    )
                stats["capsule_stats"] = capsule_stats
            except AttributeError:
                # Capsule layer might not have these methods
                stats["capsule_stats"] = {"status": "learn_methods_not_available"}

        # Update learning state
        self.state.total_learn_calls += 1
        if outcome > 0.5:
            self.state.total_positive_outcomes += 1
        else:
            self.state.total_negative_outcomes += 1

        # Update mean outcome (EMA)
        alpha = 0.01
        self.state.mean_outcome_score = (
            alpha * outcome + (1 - alpha) * self.state.mean_outcome_score
        )

        # Track outcome history
        if self.config.track_history:
            self.state.outcome_history.append(outcome)
            if len(self.state.outcome_history) > self.config.max_history:
                self.state.outcome_history = self.state.outcome_history[-self.config.max_history:]

        logger.debug(
            f"Phase 6A: Learn - outcome={outcome:.3f}, "
            f"total_learns={self.state.total_learn_calls}"
        )

        return stats

    def get_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive bridge statistics.

        Returns:
            Dictionary with FF, capsule, and combined statistics
        """
        stats = {
            "total_forwards": self.state.total_forwards,
            "ff": {
                "last_goodness": self.state.last_ff_goodness,
                "mean_goodness": self.state.mean_ff_goodness,
                "above_threshold_ratio": (
                    self.state.ff_above_threshold_count / max(self.state.total_forwards, 1)
                ),
            },
            "capsule": {
                "last_agreement": self.state.last_routing_agreement,
                "mean_agreement": self.state.mean_routing_agreement,
                "last_pose_change": self.state.last_pose_change,
            },
            "combined": {
                "last_confidence": self.state.last_confidence,
                "mean_confidence": self.state.mean_confidence,
            },
            "learning": {
                "total_learn_calls": self.state.total_learn_calls,
                "positive_outcomes": self.state.total_positive_outcomes,
                "negative_outcomes": self.state.total_negative_outcomes,
                "mean_outcome_score": self.state.mean_outcome_score,
            },
            "config": {
                "ff_weight": self.config.ff_weight,
                "capsule_weight": self.config.capsule_weight,
                "goodness_threshold": self.config.goodness_threshold,
                "agreement_threshold": self.config.agreement_threshold,
                "joint_learning": self.config.joint_learning,
            },
            "has_ff_encoder": self.ff_encoder is not None,
            "has_capsule_layer": self.capsule_layer is not None,
        }

        return stats

    def reset_statistics(self) -> None:
        """Reset all statistics while preserving configuration."""
        self.state = FFCapsuleBridgeState()
        self._last_embedding = None
        self._last_ff_output = None
        self._last_capsule_state = None
        logger.info("Phase 6A: FFCapsuleBridge statistics reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FFCapsuleBridge(ff_weight={self.config.ff_weight:.2f}, "
            f"capsule_weight={self.config.capsule_weight:.2f}, "
            f"has_encoder={self.ff_encoder is not None}, "
            f"has_capsule={self.capsule_layer is not None})"
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_ff_capsule_bridge(
    ff_encoder: FFEncoderProtocol | None = None,
    capsule_layer: CapsuleLayerProtocol | None = None,
    ff_weight: float = 0.6,
    goodness_threshold: float = 2.0,
    joint_learning: bool = True,
) -> FFCapsuleBridge:
    """
    Factory function for FF-Capsule Bridge.

    Args:
        ff_encoder: Optional FFEncoder instance
        capsule_layer: Optional CapsuleLayer instance
        ff_weight: Weight for FF goodness (capsule weight is 1 - ff_weight)
        goodness_threshold: FF goodness threshold
        joint_learning: Enable joint learning from outcomes

    Returns:
        Configured FFCapsuleBridge
    """
    config = FFCapsuleBridgeConfig(
        ff_weight=ff_weight,
        goodness_threshold=goodness_threshold,
        joint_learning=joint_learning,
    )
    return FFCapsuleBridge(
        ff_encoder=ff_encoder,
        capsule_layer=capsule_layer,
        config=config,
    )


__all__ = [
    "FFCapsuleBridge",
    "FFCapsuleBridgeConfig",
    "FFCapsuleBridgeState",
    "CapsuleState",
    "create_ff_capsule_bridge",
]
