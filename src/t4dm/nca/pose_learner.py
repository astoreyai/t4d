"""
Emergent Pose Learning for Capsule Networks.

Implements Phase 4B: Pose dimensions emerge from routing agreement patterns
rather than being hand-set. This addresses Hinton critiques H8 (part-whole
representation) and H9 (frozen pose learning).

Theoretical Foundation:
=======================

1. The Problem with Hand-Set Poses (H8-H9):
   - Pre-defining pose dimensions (temporal, causal, etc.) assumes we know
     the right representational structure a priori
   - This violates the core insight that representations should be learned
   - "Instead of programming intelligence, we should be growing it" - Hinton

2. Emergent Pose Dimensions:
   - Pose structure should emerge from data through learning
   - High routing agreement indicates the transformation is capturing
     meaningful structure - reinforce it
   - Low agreement indicates transformation is not capturing structure -
     adjust toward consensus

3. Hebbian Learning for Poses:
   - "Neurons that fire together wire together"
   - When lower capsule poses and higher consensus agree, strengthen
     the transformation that produced that agreement
   - Agreement-modulated learning rate: high agreement = small updates,
     low agreement = larger updates toward consensus

4. Complementary Learning (Hinton-inspired):
   - Fast learning: Transform matrices adapt quickly to routing feedback
   - Slow learning: Dimension interpretations emerge gradually
   - This mirrors hippocampal (fast) and cortical (slow) learning

References:
- Hinton, G. E. (2022). The Forward-Forward Algorithm
- Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic Routing Between Capsules
- Hinton, G. E., Sabour, S., & Frosst, N. (2018). Matrix capsules with EM routing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PoseLearnerConfig:
    """
    Configuration for emergent pose learning.

    Parameters:
    - n_dimensions: Number of pose dimensions to learn
    - pose_dim: Size of pose transformation matrix (n_dimensions x n_dimensions)
    - base_lr: Base learning rate for pose updates
    - agreement_threshold: Agreement level above which to reduce learning rate
    - momentum: Momentum for transform matrix updates
    - regularization: L2 regularization strength for transforms
    - orthogonality_weight: Encourages orthogonal transforms (preserve distances)
    - history_length: Number of routing events to track for statistics
    """

    # Dimension configuration
    n_dimensions: int = 4  # Number of learned pose dimensions
    pose_dim: int = 4  # Pose matrix size (usually same as n_dimensions)
    hidden_dim: int = 16  # Hidden dimension for internal representations

    # Learning parameters
    base_lr: float = 0.01  # Base learning rate
    agreement_threshold: float = 0.7  # High-agreement threshold
    momentum: float = 0.9  # Momentum for updates
    weight_decay: float = 0.0001  # L2 regularization

    # Regularization
    orthogonality_weight: float = 0.1  # Encourage orthogonal transforms
    sparsity_weight: float = 0.01  # Encourage sparse dimension usage
    diversity_weight: float = 0.01  # Encourage diverse transforms

    # Numerical stability
    eps: float = 1e-8
    max_gradient_norm: float = 1.0  # Gradient clipping

    # History tracking
    history_length: int = 1000
    convergence_window: int = 50  # Window for convergence check
    convergence_threshold: float = 0.001  # Threshold for convergence

    # Dimension naming (start generic, can be updated by learning)
    initial_dimension_names: list[str] = field(
        default_factory=lambda: ["dim_0", "dim_1", "dim_2", "dim_3"]
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.n_dimensions > 0, "n_dimensions must be positive"
        assert self.pose_dim > 0, "pose_dim must be positive"
        assert 0 < self.base_lr <= 1, "base_lr must be in (0, 1]"
        assert 0 <= self.agreement_threshold <= 1, "agreement_threshold must be in [0, 1]"
        assert 0 <= self.momentum < 1, "momentum must be in [0, 1)"


# =============================================================================
# State Tracking
# =============================================================================


@dataclass
class PoseLearnerState:
    """
    State tracking for pose learning.

    Tracks learning progress, dimension emergence, and convergence metrics.
    """

    # Learning statistics
    total_updates: int = 0
    total_routing_events: int = 0

    # Agreement tracking
    mean_agreement: float = 0.0
    agreement_history: list[float] = field(default_factory=list)
    best_agreement: float = 0.0

    # Transform statistics
    mean_transform_norm: float = 0.0
    transform_update_norm: float = 0.0
    total_transform_change: float = 0.0

    # Convergence tracking
    is_converged: bool = False
    convergence_step: int = -1
    loss_history: list[float] = field(default_factory=list)

    # Dimension emergence tracking
    dimension_usage: list[float] = field(default_factory=list)  # Per-dimension usage
    dimension_correlations: np.ndarray | None = None  # Inter-dimension correlations


# =============================================================================
# Emergent Pose Dimension Discovery
# =============================================================================


class PoseDimensionDiscovery:
    """
    Learn pose dimensions from routing patterns (Hinton H8-H9).

    Instead of hand-setting pose dimensions (temporal, causal, etc.), this class
    discovers what dimensions are useful from routing-by-agreement. High agreement
    between lower capsule predictions and consensus poses indicates the transformation
    matrices are capturing meaningful structure.

    Core Mechanism:
    1. Lower capsules make predictions: u_j|i = W_ij @ pose_i
    2. Routing aggregates predictions into consensus: s_j = sum_i(c_ij * u_j|i)
    3. Agreement measures prediction-consensus match: a_ij = exp(-||u_j|i - s_j||)
    4. Learning: adjust W_ij based on agreement-weighted error

    Hebbian Interpretation:
    - High agreement = transformation is working = small updates (already good)
    - Low agreement = transformation not capturing structure = larger updates

    Attributes:
        config: PoseLearnerConfig instance
        state: PoseLearnerState tracking
        dimension_names: Current names for learned dimensions
        transform_matrices: Learned pose transformation matrices
        momentum_buffers: Momentum accumulators for each transform

    Example:
        >>> config = PoseLearnerConfig(n_dimensions=4)
        >>> learner = PoseDimensionDiscovery(config)
        >>> stats = learner.learn_from_routing(
        ...     lower_poses, upper_poses, routing_weights, agreement=0.6
        ... )
        >>> print(f"Mean agreement: {stats['mean_agreement']:.3f}")
    """

    def __init__(
        self,
        config: PoseLearnerConfig | None = None,
        n_dimensions: int = 4,
        hidden_dim: int = 16,
        random_seed: int | None = None,
    ):
        """
        Initialize emergent pose learner.

        Args:
            config: Full configuration object (overrides other params if provided)
            n_dimensions: Number of pose dimensions (if no config)
            hidden_dim: Hidden dimension for transforms (if no config)
            random_seed: Random seed for reproducibility
        """
        if config is not None:
            self.config = config
        else:
            self.config = PoseLearnerConfig(
                n_dimensions=n_dimensions,
                pose_dim=n_dimensions,
                hidden_dim=hidden_dim,
            )

        self.state = PoseLearnerState()
        self._rng = np.random.default_rng(random_seed)

        # Initialize dimension names (start generic)
        self.dimension_names = list(self.config.initial_dimension_names)
        while len(self.dimension_names) < self.config.n_dimensions:
            self.dimension_names.append(f"dim_{len(self.dimension_names)}")

        # Initialize transform matrices near identity
        self._init_transforms()

        # Momentum buffers for each transform
        self._momentum_buffers: list[np.ndarray] = [
            np.zeros_like(m) for m in self.transform_matrices
        ]

        # History buffers
        self._agreement_history: list[float] = []
        self._loss_history: list[float] = []

        # Dimension usage tracking
        self._dimension_activations: list[np.ndarray] = []

    def _init_transforms(self) -> None:
        """Initialize transformation matrices near identity."""
        n = self.config.n_dimensions
        pose_dim = self.config.pose_dim

        # Start each transform near identity with small random perturbations
        # This allows gradual divergence into distinct transformations
        scale = 0.1

        self.transform_matrices: list[np.ndarray] = []
        for i in range(n):
            # Start at identity
            matrix = np.eye(pose_dim, dtype=np.float32)
            # Add small random perturbations
            noise = self._rng.normal(0, scale, (pose_dim, pose_dim))
            matrix += noise.astype(np.float32)
            self.transform_matrices.append(matrix)

    # -------------------------------------------------------------------------
    # Core Learning Method
    # -------------------------------------------------------------------------

    def learn_from_routing(
        self,
        lower_poses: np.ndarray,
        upper_poses: np.ndarray,
        routing_weights: np.ndarray,
        agreement: float,
        learning_rate: float | None = None,
    ) -> dict:
        """
        Update pose transformations based on routing consensus.

        This is THE KEY method: poses emerge from routing agreement rather than
        being hand-set. When predictions match consensus well, we reinforce those
        transforms. When agreement is low, we adjust toward consensus.

        Args:
            lower_poses: Poses from lower capsules [num_lower, pose_dim, pose_dim]
                         OR [pose_dim, pose_dim] for single capsule
            upper_poses: Consensus poses from routing [num_upper, pose_dim, pose_dim]
                         OR [pose_dim, pose_dim] for single capsule
            routing_weights: Routing coefficients [num_lower, num_upper]
                            OR [num_lower] for single upper capsule
            agreement: Mean agreement score from routing (0 to 1)
            learning_rate: Optional override learning rate

        Returns:
            Dictionary with learning statistics:
            - mean_agreement: Current agreement level
            - effective_lr: Learning rate after agreement modulation
            - transform_change: Total change in transform matrices
            - consensus_distance: Distance from predictions to consensus
            - loss: Total loss (consensus + regularization)
        """
        # Ensure proper shape
        if lower_poses.ndim == 2:
            lower_poses = lower_poses.reshape(1, *lower_poses.shape)
        if upper_poses.ndim == 2:
            upper_poses = upper_poses.reshape(1, *upper_poses.shape)
        if routing_weights.ndim == 1:
            routing_weights = routing_weights.reshape(-1, 1)

        num_lower = lower_poses.shape[0]
        num_upper = upper_poses.shape[0]

        # Agreement-modulated learning rate (Hinton insight)
        # High agreement = already good, small updates
        # Low agreement = need larger updates toward consensus
        base_lr = learning_rate or self.config.base_lr

        if agreement > self.config.agreement_threshold:
            # High agreement: proportionally reduce learning rate
            effective_lr = base_lr * (1.0 - agreement)
        else:
            # Low agreement: use base learning rate
            effective_lr = base_lr

        # Compute consensus pose from routing weights
        # This is what the routing algorithm "decided" the pose should be
        consensus = self._compute_consensus_pose(lower_poses, routing_weights)

        # Compute predictions using current transforms
        predictions = self._compute_predictions(lower_poses)

        # Compute loss and gradients
        total_loss = 0.0
        total_transform_change = 0.0
        consensus_distances = []

        for t_idx, transform in enumerate(self.transform_matrices):
            # Prediction from this transform: W_t @ lower_pose
            if t_idx < len(predictions):
                pred = predictions[t_idx]  # [num_lower, pose_dim, pose_dim]
            else:
                continue

            # Error: difference from consensus (broadcast to all lower capsules)
            # Each lower capsule's prediction should match consensus
            error = consensus - pred.mean(axis=0)  # [pose_dim, pose_dim]
            consensus_distance = float(np.linalg.norm(error))
            consensus_distances.append(consensus_distance)

            # Loss: consensus matching + regularization
            consensus_loss = 0.5 * consensus_distance ** 2
            reg_loss = self._compute_regularization_loss(transform)
            loss = consensus_loss + reg_loss
            total_loss += loss

            # Hebbian update: strengthen transforms that predict consensus
            # Gradient: dL/dW = (1 - agreement) * lower_pose^T @ error
            # This is a Hebbian outer product: input correlates with error
            mean_lower = lower_poses.mean(axis=0)  # [pose_dim, pose_dim]

            # Compute gradient (approximation via outer product)
            gradient = np.einsum("ij,kl->ik", mean_lower, error)
            gradient = gradient[:self.config.pose_dim, :self.config.pose_dim]

            # Clip gradient for stability
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > self.config.max_gradient_norm:
                gradient = gradient * (self.config.max_gradient_norm / grad_norm)

            # Momentum update
            self._momentum_buffers[t_idx] = (
                self.config.momentum * self._momentum_buffers[t_idx] +
                (1 - self.config.momentum) * gradient
            )

            # Apply update with weight decay
            delta = effective_lr * self._momentum_buffers[t_idx]
            delta -= self.config.weight_decay * transform  # L2 regularization

            self.transform_matrices[t_idx] = transform + delta
            total_transform_change += float(np.linalg.norm(delta))

        # Update state
        self.state.total_updates += 1
        self.state.total_routing_events += 1
        self.state.mean_agreement = agreement
        self.state.transform_update_norm = total_transform_change
        self.state.total_transform_change += total_transform_change

        # Track history
        self._agreement_history.append(agreement)
        self._loss_history.append(total_loss)

        # Trim history to max length
        if len(self._agreement_history) > self.config.history_length:
            self._agreement_history = self._agreement_history[-self.config.history_length:]
            self._loss_history = self._loss_history[-self.config.history_length:]

        # Update state histories
        self.state.agreement_history = self._agreement_history.copy()
        self.state.loss_history = self._loss_history.copy()

        # Check convergence
        self._check_convergence()

        # Track best agreement
        if agreement > self.state.best_agreement:
            self.state.best_agreement = agreement

        return {
            "mean_agreement": agreement,
            "effective_lr": effective_lr,
            "transform_change": total_transform_change,
            "consensus_distance": float(np.mean(consensus_distances)) if consensus_distances else 0.0,
            "loss": total_loss,
            "is_converged": self.state.is_converged,
            "total_updates": self.state.total_updates,
        }

    def _compute_consensus_pose(
        self,
        lower_poses: np.ndarray,
        routing_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute consensus pose from routing weights.

        The consensus is a weighted average of lower poses, where weights
        come from routing-by-agreement.

        Args:
            lower_poses: [num_lower, pose_dim, pose_dim]
            routing_weights: [num_lower, num_upper] or [num_lower]

        Returns:
            Consensus pose [pose_dim, pose_dim]
        """
        if routing_weights.ndim == 1:
            routing_weights = routing_weights.reshape(-1, 1)

        # Normalize routing weights
        weights = routing_weights.sum(axis=1)  # [num_lower]
        weights = weights / (weights.sum() + self.config.eps)

        # Weighted average of lower poses
        consensus = np.einsum("n,nij->ij", weights, lower_poses)

        return consensus.astype(np.float32)

    def _compute_predictions(
        self,
        lower_poses: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Compute predictions from each transform matrix.

        Args:
            lower_poses: [num_lower, pose_dim, pose_dim]

        Returns:
            List of predictions, one per transform [num_lower, pose_dim, pose_dim]
        """
        predictions = []
        for transform in self.transform_matrices:
            # Apply transform to each lower pose
            pred = np.einsum("ij,njk->nik", transform, lower_poses)
            predictions.append(pred)
        return predictions

    def _compute_regularization_loss(self, transform: np.ndarray) -> float:
        """
        Compute regularization loss for a transform matrix.

        Includes:
        - Frobenius norm (weight decay)
        - Orthogonality encouragement
        """
        # Frobenius norm
        frob_loss = self.config.weight_decay * np.linalg.norm(transform, "fro") ** 2

        # Orthogonality: ||W^T W - I||
        wtw = transform.T @ transform
        identity = np.eye(transform.shape[0], dtype=np.float32)
        orth_loss = self.config.orthogonality_weight * np.linalg.norm(wtw - identity, "fro") ** 2

        return frob_loss + orth_loss

    def _check_convergence(self) -> None:
        """Check if learning has converged."""
        if len(self._loss_history) < self.config.convergence_window:
            return

        # Check variance of recent losses
        recent = self._loss_history[-self.config.convergence_window:]
        variance = np.var(recent)

        if variance < self.config.convergence_threshold:
            if not self.state.is_converged:
                self.state.is_converged = True
                self.state.convergence_step = self.state.total_updates
                logger.info(
                    f"Pose learning converged at step {self.state.convergence_step} "
                    f"with variance {variance:.6f}"
                )

    # -------------------------------------------------------------------------
    # Dimension Analysis
    # -------------------------------------------------------------------------

    def get_dimension_usage(self) -> np.ndarray:
        """
        Analyze which dimensions are most active/useful.

        Returns:
            Array of per-dimension usage scores [n_dimensions]
        """
        usage = np.zeros(self.config.n_dimensions, dtype=np.float32)

        for i, transform in enumerate(self.transform_matrices):
            # Measure "importance" by variance of transform
            # High variance = dimension is being used distinctively
            usage[i] = float(np.var(transform))

        return usage

    def get_dimension_correlations(self) -> np.ndarray:
        """
        Compute correlations between learned dimensions.

        Returns:
            Correlation matrix [n_dimensions, n_dimensions]
        """
        n = self.config.n_dimensions
        correlations = np.eye(n, dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                # Flatten transforms and compute correlation
                flat_i = self.transform_matrices[i].flatten()
                flat_j = self.transform_matrices[j].flatten()

                # Pearson correlation
                corr = np.corrcoef(flat_i, flat_j)[0, 1]
                correlations[i, j] = corr
                correlations[j, i] = corr

        self.state.dimension_correlations = correlations
        return correlations

    def get_transform_statistics(self) -> dict:
        """
        Get statistics about learned transforms.

        Returns:
            Dictionary with transform statistics
        """
        norms = [np.linalg.norm(t, "fro") for t in self.transform_matrices]
        determinants = [np.linalg.det(t) for t in self.transform_matrices]

        # Orthogonality errors
        orth_errors = []
        for t in self.transform_matrices:
            wtw = t.T @ t
            identity = np.eye(t.shape[0])
            orth_errors.append(np.linalg.norm(wtw - identity, "fro"))

        return {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "norms": norms,
            "determinants": determinants,
            "mean_determinant": float(np.mean(determinants)),
            "orthogonality_errors": orth_errors,
            "mean_orthogonality_error": float(np.mean(orth_errors)),
        }

    # -------------------------------------------------------------------------
    # Transform Application
    # -------------------------------------------------------------------------

    def apply_transform(
        self,
        pose: np.ndarray,
        dimension_idx: int = 0,
    ) -> np.ndarray:
        """
        Apply a learned transform to a pose.

        Args:
            pose: Input pose [pose_dim, pose_dim] or [batch, pose_dim, pose_dim]
            dimension_idx: Which dimension's transform to apply

        Returns:
            Transformed pose (same shape as input)
        """
        if dimension_idx >= len(self.transform_matrices):
            raise ValueError(
                f"dimension_idx {dimension_idx} >= n_dimensions {len(self.transform_matrices)}"
            )

        transform = self.transform_matrices[dimension_idx]

        if pose.ndim == 2:
            return (transform @ pose).astype(np.float32)
        else:
            return np.einsum("ij,njk->nik", transform, pose).astype(np.float32)

    def apply_all_transforms(
        self,
        pose: np.ndarray,
    ) -> np.ndarray:
        """
        Apply all learned transforms to a pose.

        Args:
            pose: Input pose [pose_dim, pose_dim]

        Returns:
            Transformed poses [n_dimensions, pose_dim, pose_dim]
        """
        results = []
        for transform in self.transform_matrices:
            results.append(transform @ pose)
        return np.array(results, dtype=np.float32)

    def compose_transforms(
        self,
        indices: list[int],
    ) -> np.ndarray:
        """
        Compose multiple transforms in sequence.

        Args:
            indices: List of dimension indices to compose

        Returns:
            Composed transformation matrix [pose_dim, pose_dim]
        """
        result = np.eye(self.config.pose_dim, dtype=np.float32)
        for idx in indices:
            if idx < len(self.transform_matrices):
                result = result @ self.transform_matrices[idx]
        return result

    # -------------------------------------------------------------------------
    # Integration with Capsule Layer
    # -------------------------------------------------------------------------

    def integrate_with_capsule_layer(
        self,
        capsule_layer,
        replace_transforms: bool = True,
    ) -> None:
        """
        Integrate learned transforms with a CapsuleLayer.

        This replaces the capsule layer's hand-set transforms with
        our emergent learned ones.

        Args:
            capsule_layer: CapsuleLayer instance to integrate with
            replace_transforms: If True, replace layer's transform matrices
        """
        if not hasattr(capsule_layer, "_transform_matrices"):
            logger.warning(
                "CapsuleLayer does not have _transform_matrices. "
                "Run routing first to initialize."
            )
            return

        if replace_transforms:
            # Get shape information from capsule layer
            layer_transforms = capsule_layer._transform_matrices
            num_lower, num_output = layer_transforms.shape[:2]

            # Create new transforms using our learned patterns
            new_transforms = np.zeros_like(layer_transforms)

            for i in range(num_lower):
                for j in range(num_output):
                    # Use our learned transforms as basis
                    # Cycle through our transforms
                    t_idx = (i * num_output + j) % len(self.transform_matrices)
                    new_transforms[i, j] = self.transform_matrices[t_idx]

            capsule_layer._transform_matrices = new_transforms
            logger.info(
                f"Integrated learned transforms into CapsuleLayer "
                f"({num_lower}x{num_output} transforms)"
            )

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """Get state dictionary for persistence."""
        return {
            "config": {
                "n_dimensions": self.config.n_dimensions,
                "pose_dim": self.config.pose_dim,
                "hidden_dim": self.config.hidden_dim,
                "base_lr": self.config.base_lr,
                "agreement_threshold": self.config.agreement_threshold,
                "momentum": self.config.momentum,
                "weight_decay": self.config.weight_decay,
            },
            "dimension_names": self.dimension_names,
            "transform_matrices": [t.tolist() for t in self.transform_matrices],
            "momentum_buffers": [m.tolist() for m in self._momentum_buffers],
            "state": {
                "total_updates": self.state.total_updates,
                "total_routing_events": self.state.total_routing_events,
                "mean_agreement": self.state.mean_agreement,
                "best_agreement": self.state.best_agreement,
                "is_converged": self.state.is_converged,
                "convergence_step": self.state.convergence_step,
            },
            "agreement_history": self._agreement_history,
            "loss_history": self._loss_history,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from dictionary."""
        # Restore transform matrices
        self.transform_matrices = [
            np.array(t, dtype=np.float32) for t in state_dict["transform_matrices"]
        ]

        # Restore momentum buffers
        self._momentum_buffers = [
            np.array(m, dtype=np.float32) for m in state_dict["momentum_buffers"]
        ]

        # Restore dimension names
        self.dimension_names = state_dict["dimension_names"]

        # Restore state
        state_data = state_dict["state"]
        self.state.total_updates = state_data["total_updates"]
        self.state.total_routing_events = state_data["total_routing_events"]
        self.state.mean_agreement = state_data["mean_agreement"]
        self.state.best_agreement = state_data["best_agreement"]
        self.state.is_converged = state_data["is_converged"]
        self.state.convergence_step = state_data["convergence_step"]

        # Restore histories
        self._agreement_history = state_dict.get("agreement_history", [])
        self._loss_history = state_dict.get("loss_history", [])

        logger.info(
            f"Loaded pose learner state: {self.state.total_updates} updates, "
            f"best agreement {self.state.best_agreement:.3f}"
        )

    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PoseDimensionDiscovery("
            f"n_dims={self.config.n_dimensions}, "
            f"pose_dim={self.config.pose_dim}, "
            f"updates={self.state.total_updates}, "
            f"agreement={self.state.mean_agreement:.3f})"
        )


# =============================================================================
# Capsule Integration Mixin
# =============================================================================


class PoseLearningMixin:
    """
    Mixin class for adding emergent pose learning to CapsuleLayer.

    This mixin can be used to extend existing CapsuleLayer with
    emergent pose learning capabilities without modifying the original class.

    Usage:
        class LearnableCapsuleLayer(CapsuleLayer, PoseLearningMixin):
            pass
    """

    def init_pose_learner(
        self,
        config: PoseLearnerConfig | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the pose learner component."""
        self._pose_learner = PoseDimensionDiscovery(
            config=config,
            random_seed=random_seed,
        )

    def learn_poses_from_routing_result(
        self,
        lower_poses: np.ndarray,
        upper_poses: np.ndarray,
        routing_weights: np.ndarray,
        agreement: float,
    ) -> dict:
        """
        Learn poses from a routing result.

        Args:
            lower_poses: Lower capsule poses
            upper_poses: Higher capsule consensus poses
            routing_weights: Routing coefficients
            agreement: Mean agreement score

        Returns:
            Learning statistics
        """
        if not hasattr(self, "_pose_learner"):
            self.init_pose_learner()

        return self._pose_learner.learn_from_routing(
            lower_poses=lower_poses,
            upper_poses=upper_poses,
            routing_weights=routing_weights,
            agreement=agreement,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_pose_learner(
    n_dimensions: int = 4,
    pose_dim: int = 4,
    learning_rate: float = 0.01,
    random_seed: int | None = None,
) -> PoseDimensionDiscovery:
    """
    Create a pose learner with common defaults.

    Args:
        n_dimensions: Number of pose dimensions to learn
        pose_dim: Size of pose transformation matrix
        learning_rate: Base learning rate
        random_seed: Random seed for reproducibility

    Returns:
        Configured PoseDimensionDiscovery instance
    """
    config = PoseLearnerConfig(
        n_dimensions=n_dimensions,
        pose_dim=pose_dim,
        base_lr=learning_rate,
    )
    return PoseDimensionDiscovery(config, random_seed=random_seed)


def create_learnable_capsule_system(
    input_dim: int = 1024,
    num_capsules: int = 32,
    capsule_dim: int = 16,
    pose_dim: int = 4,
    pose_learner_lr: float = 0.01,
    random_seed: int | None = None,
) -> tuple:
    """
    Create a capsule layer with integrated pose learning.

    Args:
        input_dim: Input embedding dimension
        num_capsules: Number of capsules
        capsule_dim: Activation dimension per capsule
        pose_dim: Pose matrix dimension
        pose_learner_lr: Learning rate for pose learning
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (CapsuleLayer, PoseDimensionDiscovery)
    """
    from t4dm.nca.capsules import CapsuleConfig, CapsuleLayer

    capsule_config = CapsuleConfig(
        input_dim=input_dim,
        num_capsules=num_capsules,
        capsule_dim=capsule_dim,
        pose_dim=pose_dim,
    )
    capsule_layer = CapsuleLayer(capsule_config, random_seed=random_seed)

    pose_config = PoseLearnerConfig(
        n_dimensions=pose_dim,
        pose_dim=pose_dim,
        base_lr=pose_learner_lr,
    )
    pose_learner = PoseDimensionDiscovery(pose_config, random_seed=random_seed)

    return capsule_layer, pose_learner
