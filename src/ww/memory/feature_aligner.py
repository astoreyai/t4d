"""
FeatureAligner - Joint Gating-Retrieval Feature Alignment.

Phase 3 of HSA-inspired improvements. Projects gate features to
retrieval space for consistency loss computation, enabling joint
optimization of gating and retrieval systems.

Joint Loss Function:
    L_joint = L_gate + λ_r*L_retrieval + λ_c*L_consistency + λ_d*L_diversity

Where:
- L_gate: BCE on utility prediction (existing)
- L_retrieval: Ranking loss on retrieval scores (existing)
- L_consistency (NEW): Alignment between gate predictions and retrieval scores
- L_diversity (NEW): Entropy of gate decisions (prevents collapse)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result of feature alignment computation."""

    projected_features: np.ndarray  # Gate features projected to retrieval space
    consistency_loss: float  # Alignment loss between gate and retrieval
    diversity_loss: float  # Entropy of gate decisions
    joint_loss: float  # Combined loss
    gate_prediction: float  # Gate's utility prediction
    retrieval_score: float  # Retrieval system's score


@dataclass
class JointLossWeights:
    """Weights for joint loss components."""

    gate: float = 1.0  # L_gate weight
    retrieval: float = 0.5  # L_retrieval weight
    consistency: float = 0.3  # L_consistency weight
    diversity: float = 0.1  # L_diversity weight


class FeatureAligner:
    """
    Aligns gate and retrieval feature spaces for joint optimization.

    Projects high-dimensional gate features (247-dim after P0a) to the
    4-dimensional retrieval scoring space (semantic, recency, outcome, importance).

    This enables:
    1. Gate predictions to be consistent with retrieval scores
    2. Retrieval to benefit from gate's utility predictions
    3. End-to-end optimization of the memory pipeline

    Integration:
    - Gate features from LearnedMemoryGate.build_feature_vector()
    - Retrieval scores from recall() component scoring
    - Training signal from learn_from_outcome()
    """

    # Gate feature dimensions (after P0a content projection)
    GATE_FEATURE_DIM = 247
    # Retrieval score dimensions
    RETRIEVAL_DIM = 4  # semantic, recency, outcome, importance

    def __init__(
        self,
        gate_dim: int = 247,
        retrieval_dim: int = 4,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        loss_weights: JointLossWeights | None = None,
    ):
        """
        Initialize feature aligner.

        Args:
            gate_dim: Dimension of gate feature vectors
            retrieval_dim: Dimension of retrieval score vectors
            hidden_dim: Hidden layer dimension
            learning_rate: SGD learning rate
            loss_weights: Weights for joint loss components
        """
        self.gate_dim = gate_dim
        self.retrieval_dim = retrieval_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.loss_weights = loss_weights or JointLossWeights()

        # Projection network: gate_dim → hidden → retrieval_dim
        # Xavier initialization
        self.W1 = np.random.randn(hidden_dim, gate_dim).astype(np.float32) * np.sqrt(2.0 / (gate_dim + hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(retrieval_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / (hidden_dim + retrieval_dim))
        self.b2 = np.zeros(retrieval_dim, dtype=np.float32)

        # Training statistics
        self.n_updates = 0
        self.running_consistency_loss = 0.5
        self.running_diversity_loss = 0.5

        # Decision history for diversity tracking
        self._recent_decisions: list[float] = []
        self._max_history = 100

    def project(self, gate_features: np.ndarray) -> np.ndarray:
        """
        Project gate features to retrieval space.

        Args:
            gate_features: Gate feature vector [gate_dim]

        Returns:
            Projected features [retrieval_dim]
        """
        features = np.array(gate_features, dtype=np.float32)

        # Handle dimension mismatch
        if len(features) > self.gate_dim:
            features = features[:self.gate_dim]
        elif len(features) < self.gate_dim:
            features = np.pad(features, (0, self.gate_dim - len(features)))

        # Forward pass
        hidden = np.tanh(self.W1 @ features + self.b1)
        output = self._sigmoid(self.W2 @ hidden + self.b2)

        return output

    def compute_consistency_loss(
        self,
        gate_features: np.ndarray,
        retrieval_scores: dict[str, float],
    ) -> float:
        """
        Compute consistency loss between gate prediction and retrieval scores.

        Consistency loss encourages gate predictions to align with
        retrieval quality scores.

        Args:
            gate_features: Gate feature vector
            retrieval_scores: Dict with keys: semantic, recency, outcome, importance

        Returns:
            Consistency loss (lower = more aligned)
        """
        # Project gate features
        projected = self.project(gate_features)

        # Extract retrieval scores in order
        retrieval = np.array([
            retrieval_scores.get("semantic", 0.5),
            retrieval_scores.get("recency", 0.5),
            retrieval_scores.get("outcome", 0.5),
            retrieval_scores.get("importance", 0.5),
        ], dtype=np.float32)

        # L2 distance between projected and retrieval
        consistency_loss = float(np.mean((projected - retrieval) ** 2))

        return consistency_loss

    def compute_diversity_loss(self, gate_decision: float) -> float:
        """
        Compute diversity loss from gate decision history.

        Diversity loss prevents gate collapse (always STORE or always DISCARD).
        Uses entropy of recent decisions.

        Args:
            gate_decision: Gate's store probability [0, 1]

        Returns:
            Diversity loss (lower = more diverse)
        """
        # Track decision
        self._recent_decisions.append(gate_decision)
        if len(self._recent_decisions) > self._max_history:
            self._recent_decisions.pop(0)

        if len(self._recent_decisions) < 10:
            return 0.5  # Not enough history

        # Compute binary entropy of decisions
        decisions = np.array(self._recent_decisions)
        mean_decision = np.mean(decisions)

        # Entropy: -p*log(p) - (1-p)*log(1-p)
        # Maximum entropy at p=0.5, minimum at p=0 or p=1
        eps = 1e-8
        entropy = -(mean_decision * np.log(mean_decision + eps) +
                   (1 - mean_decision) * np.log(1 - mean_decision + eps))

        # Normalize to [0, 1] where 0 = max entropy (diverse)
        max_entropy = np.log(2)  # ~0.693
        diversity_loss = 1.0 - (entropy / max_entropy)

        return float(diversity_loss)

    def compute_joint_loss(
        self,
        gate_loss: float,
        retrieval_loss: float,
        consistency_loss: float,
        diversity_loss: float,
    ) -> float:
        """
        Compute weighted joint loss.

        Args:
            gate_loss: BCE loss from gate utility prediction
            retrieval_loss: Ranking loss from retrieval
            consistency_loss: Gate-retrieval alignment loss
            diversity_loss: Gate decision entropy loss

        Returns:
            Weighted joint loss
        """
        joint = (
            self.loss_weights.gate * gate_loss +
            self.loss_weights.retrieval * retrieval_loss +
            self.loss_weights.consistency * consistency_loss +
            self.loss_weights.diversity * diversity_loss
        )

        return float(joint)

    def align(
        self,
        gate_features: np.ndarray,
        gate_prediction: float,
        retrieval_scores: dict[str, float],
        gate_loss: float = 0.0,
        retrieval_loss: float = 0.0,
    ) -> AlignmentResult:
        """
        Compute full alignment result.

        Args:
            gate_features: Gate feature vector
            gate_prediction: Gate's utility prediction [0, 1]
            retrieval_scores: Component retrieval scores
            gate_loss: Optional gate BCE loss
            retrieval_loss: Optional retrieval ranking loss

        Returns:
            Complete alignment result with all losses
        """
        projected = self.project(gate_features)

        consistency_loss = self.compute_consistency_loss(gate_features, retrieval_scores)
        diversity_loss = self.compute_diversity_loss(gate_prediction)

        joint_loss = self.compute_joint_loss(
            gate_loss=gate_loss,
            retrieval_loss=retrieval_loss,
            consistency_loss=consistency_loss,
            diversity_loss=diversity_loss,
        )

        # Combined retrieval score for comparison
        combined_retrieval = float(np.mean(list(retrieval_scores.values())))

        return AlignmentResult(
            projected_features=projected,
            consistency_loss=consistency_loss,
            diversity_loss=diversity_loss,
            joint_loss=joint_loss,
            gate_prediction=gate_prediction,
            retrieval_score=combined_retrieval,
        )

    def update(
        self,
        gate_features: np.ndarray,
        retrieval_scores: dict[str, float],
        target_alignment: np.ndarray | None = None,
    ) -> float:
        """
        Update projection weights from alignment feedback.

        Args:
            gate_features: Gate feature vector
            retrieval_scores: Component retrieval scores
            target_alignment: Optional target projection (defaults to retrieval scores)

        Returns:
            Updated consistency loss
        """
        features = np.array(gate_features, dtype=np.float32)
        if len(features) > self.gate_dim:
            features = features[:self.gate_dim]
        elif len(features) < self.gate_dim:
            features = np.pad(features, (0, self.gate_dim - len(features)))

        # Target: retrieval scores
        if target_alignment is None:
            target = np.array([
                retrieval_scores.get("semantic", 0.5),
                retrieval_scores.get("recency", 0.5),
                retrieval_scores.get("outcome", 0.5),
                retrieval_scores.get("importance", 0.5),
            ], dtype=np.float32)
        else:
            target = np.array(target_alignment, dtype=np.float32)

        # Forward pass
        hidden = np.tanh(self.W1 @ features + self.b1)
        output = self._sigmoid(self.W2 @ hidden + self.b2)

        # Compute gradients (MSE loss)
        output_error = output - target
        d_output = output_error * output * (1 - output)  # Sigmoid derivative

        grad_W2 = np.outer(d_output, hidden)
        grad_b2 = d_output

        d_hidden = self.W2.T @ d_output
        d_hidden *= (1 - hidden ** 2)  # Tanh derivative

        grad_W1 = np.outer(d_hidden, features)
        grad_b1 = d_hidden

        # Update weights
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

        # Update statistics
        self.n_updates += 1
        loss = float(np.mean(output_error ** 2))
        self.running_consistency_loss = 0.9 * self.running_consistency_loss + 0.1 * loss

        return loss

    def get_statistics(self) -> dict:
        """Get aligner statistics."""
        return {
            "n_updates": self.n_updates,
            "running_consistency_loss": self.running_consistency_loss,
            "running_diversity_loss": self.running_diversity_loss,
            "recent_decision_mean": np.mean(self._recent_decisions) if self._recent_decisions else 0.5,
            "loss_weights": {
                "gate": self.loss_weights.gate,
                "retrieval": self.loss_weights.retrieval,
                "consistency": self.loss_weights.consistency,
                "diversity": self.loss_weights.diversity,
            },
        }

    def get_neuromod_learning_rate(
        self,
        base_lr: float,
        dopamine_rpe: float = 0.0,
        norepinephrine_gain: float = 1.0,
        ach_mode: str = "retrieval",
    ) -> float:
        """
        Compute neuromodulator-adjusted learning rate.

        Per HSA paper: Learning rate should be modulated by surprise
        (dopamine), arousal (NE), and encoding mode (ACh).

        Args:
            base_lr: Base learning rate
            dopamine_rpe: Reward prediction error [-1, 1]
            norepinephrine_gain: Arousal level [0.5, 2.0]
            ach_mode: "encoding" or "retrieval"

        Returns:
            Modulated learning rate
        """
        lr = base_lr

        # Surprise boost: larger |RPE| = faster learning
        lr *= 1.0 + abs(dopamine_rpe)

        # Arousal boost: higher NE = faster learning
        lr *= norepinephrine_gain

        # Encoding mode: faster learning (consolidation phase)
        if ach_mode == "encoding":
            lr *= 1.3

        # Clip to reasonable range
        lr = np.clip(lr, base_lr * 0.3, base_lr * 3.0)

        return float(lr)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# Module exports
__all__ = [
    "AlignmentResult",
    "FeatureAligner",
    "JointLossWeights",
]
