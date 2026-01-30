"""
Pose Matrix Implementation for Capsule Networks.

Implements semantic pose transformations for abstract concepts (Hinton 2017).
Unlike image capsules which use spatial poses, memory capsules use semantic
dimensions: temporal, causal, semantic role, and certainty.

NOTE ON HAND-SET POSES (H8-H9 Critique):
=========================================
The semantic dimension setters (set_temporal, set_causal, etc.) represent
"hand-set" pose dimensions - they assume we know a priori what dimensions
are useful. This contradicts Hinton's insight that representations should
emerge from learning, not be programmed.

For emergent pose learning that discovers dimensions from data, use:
    from ww.nca.pose_learner import PoseDimensionDiscovery

The hand-set methods remain available for:
1. Backward compatibility
2. Explicit symbolic knowledge injection when needed
3. Debugging and visualization

Theoretical Foundation:
=======================

1. Pose Matrix (Sabour et al. 2017, Hinton et al. 2018):
   - 4x4 transformation matrix encoding "how" an entity is configured
   - Rows encode different semantic dimensions
   - Composition via matrix multiplication (like 3D rotations)

2. Semantic Dimensions for Memory (HAND-SET - see note above):
   - TEMPORAL: Past/future offset, duration, recurrence
   - CAUSAL: Cause-of, effect-of, enables, prevents
   - SEMANTIC_ROLE: Agent, patient, instrument, location
   - CERTAINTY: Definite, possible, negated, hypothetical

3. Agreement Score:
   - Used in routing-by-agreement (Sabour et al. 2017)
   - High agreement = consistent part-whole relationship
   - Low agreement = incompatible configurations

Integration with World Weaver:
==============================
- Capsule poses encode memory configurations
- Pose agreement drives routing decisions
- Composition enables relational reasoning
- For emergent learning, see pose_learner.PoseDimensionDiscovery

References:
- Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic Routing Between Capsules
- Hinton, G. E., Sabour, S., & Frosst, N. (2018). Matrix capsules with EM routing
- Hinton, G. E. (2022). The Forward-Forward Algorithm
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Semantic Dimension Enumeration
# =============================================================================


class SemanticDimension(IntEnum):
    """
    Semantic dimensions for pose matrix rows.

    Unlike spatial poses in image capsules, memory capsules
    encode abstract semantic relationships.

    NOTE: These are HAND-SET dimensions. For emergent dimension
    discovery, use PoseDimensionDiscovery from pose_learner.py.
    """

    TEMPORAL = 0  # Time-related: past/future, duration, recurrence
    CAUSAL = 1  # Causality: cause-of, effect-of, enables, prevents
    SEMANTIC_ROLE = 2  # Thematic: agent, patient, instrument, location
    CERTAINTY = 3  # Epistemic: definite, possible, negated, hypothetical


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PoseConfig:
    """
    Configuration for pose matrices.

    Parameters:
    - pose_dim: Size of transformation matrix (default 4x4)
    - init_scale: Initialization scale for matrix elements
    - regularization: Frobenius norm regularization strength
    - warn_on_hand_set: If True, warn when using hand-set dimension methods
    """

    # Matrix dimensions
    pose_dim: int = 4  # 4x4 transformation matrix

    # Initialization
    init_scale: float = 0.1  # Scale for random initialization
    init_identity: bool = True  # Start near identity transform

    # Regularization
    regularization: float = 0.01  # Frobenius norm penalty
    orthogonality_weight: float = 0.1  # Encourage orthogonal transforms

    # Agreement computation
    agreement_temperature: float = 1.0  # Softness of agreement
    use_log_agreement: bool = False  # Log-space agreement scores

    # Numerical stability
    eps: float = 1e-6

    # Phase 4B: Hand-set warning control
    warn_on_hand_set: bool = False  # Set True to enable deprecation warnings

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.pose_dim > 0, "pose_dim must be positive"
        assert self.init_scale > 0, "init_scale must be positive"


# =============================================================================
# Pose State
# =============================================================================


@dataclass
class PoseState:
    """
    State tracking for pose matrix operations.

    Tracks agreement scores, composition history, and regularization metrics.
    """

    # Current agreement scores
    last_agreement: float = 0.0
    mean_agreement: float = 0.0
    min_agreement: float = 1.0
    max_agreement: float = 0.0

    # Composition tracking
    total_compositions: int = 0
    composition_depth: int = 0  # How many transforms composed

    # Regularization metrics
    frobenius_norm: float = 0.0
    orthogonality_error: float = 0.0
    determinant: float = 1.0

    # Update count
    total_updates: int = 0

    # Phase 4B: Hand-set tracking
    hand_set_calls: int = 0  # Number of times hand-set methods were called


# =============================================================================
# Pose Matrix Class
# =============================================================================


class PoseMatrix:
    """
    Semantic transformation matrix for capsule poses.

    Encodes "how" an entity is configured in semantic space.
    Used for routing-by-agreement in capsule networks.

    Attributes:
        config: PoseConfig instance
        matrix: 4x4 numpy array (pose_dim x pose_dim)
        state: PoseState tracking metrics

    Example:
        >>> config = PoseConfig(pose_dim=4)
        >>> pose = PoseMatrix(config)
        >>> # DEPRECATED: Hand-set methods (see pose_learner.py for emergent learning)
        >>> pose.set_temporal(offset=0.5, duration=0.3)
        >>> pose.set_causal(cause_strength=0.8)
        >>> agreement = pose.agreement(other_pose)

    For emergent pose learning (recommended):
        >>> from ww.nca.pose_learner import PoseDimensionDiscovery
        >>> learner = PoseDimensionDiscovery()
        >>> stats = learner.learn_from_routing(lower_poses, upper_poses, routing_weights, agreement)
    """

    def __init__(
        self,
        config: PoseConfig | None = None,
        matrix: np.ndarray | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize pose matrix.

        Args:
            config: Configuration object
            matrix: Optional initial matrix (if None, initialized randomly)
            random_seed: Random seed for reproducibility
        """
        self.config = config or PoseConfig()
        self.state = PoseState()
        self._rng = np.random.default_rng(random_seed)

        if matrix is not None:
            assert matrix.shape == (self.config.pose_dim, self.config.pose_dim), \
                f"Matrix shape must be ({self.config.pose_dim}, {self.config.pose_dim})"
            self.matrix = matrix.astype(np.float32)
        else:
            self.matrix = self._initialize_matrix()

        self._update_state_metrics()

    def _initialize_matrix(self) -> np.ndarray:
        """Initialize pose matrix near identity."""
        dim = self.config.pose_dim

        if self.config.init_identity:
            # Start near identity transformation
            matrix = np.eye(dim, dtype=np.float32)
            noise = self._rng.normal(0, self.config.init_scale, (dim, dim))
            matrix += noise.astype(np.float32)
        else:
            # Random initialization
            matrix = self._rng.normal(
                0, self.config.init_scale, (dim, dim)
            ).astype(np.float32)

        return matrix

    def _update_state_metrics(self) -> None:
        """Update state with current matrix metrics."""
        self.state.frobenius_norm = float(np.linalg.norm(self.matrix, 'fro'))
        self.state.determinant = float(np.linalg.det(self.matrix))

        # Orthogonality error: ||M^T M - I||
        mtm = self.matrix.T @ self.matrix
        identity = np.eye(self.config.pose_dim, dtype=np.float32)
        self.state.orthogonality_error = float(np.linalg.norm(mtm - identity, 'fro'))

    def _warn_hand_set(self, method_name: str) -> None:
        """Emit deprecation warning for hand-set methods if configured."""
        self.state.hand_set_calls += 1
        if self.config.warn_on_hand_set:
            warnings.warn(
                f"PoseMatrix.{method_name}() uses hand-set dimensions (H8-H9 critique). "
                "For emergent pose learning, use PoseDimensionDiscovery from pose_learner.py. "
                "Set config.warn_on_hand_set=False to suppress this warning.",
                DeprecationWarning,
                stacklevel=3,
            )

    # -------------------------------------------------------------------------
    # Semantic Dimension Setters (HAND-SET - See H8-H9 critique)
    # -------------------------------------------------------------------------

    def set_temporal(
        self,
        offset: float = 0.0,
        duration: float = 0.0,
        recurrence: float = 0.0,
        certainty: float = 1.0,
    ) -> None:
        """
        Set temporal dimension of pose.

        NOTE: This is a HAND-SET method (H8-H9 critique). For emergent pose
        learning that discovers dimensions from data, use PoseDimensionDiscovery.

        Args:
            offset: Past (-1) to future (+1) offset
            duration: Event duration (0 = point, 1 = extended)
            recurrence: Repetition pattern (0 = once, 1 = recurring)
            certainty: Temporal certainty (0 = vague, 1 = precise)
        """
        self._warn_hand_set("set_temporal")
        row = SemanticDimension.TEMPORAL
        self.matrix[row, 0] = np.clip(offset, -1, 1)
        self.matrix[row, 1] = np.clip(duration, 0, 1)
        self.matrix[row, 2] = np.clip(recurrence, 0, 1)
        self.matrix[row, 3] = np.clip(certainty, 0, 1)
        self._update_state_metrics()

    def set_causal(
        self,
        cause_strength: float = 0.0,
        effect_strength: float = 0.0,
        enables: float = 0.0,
        prevents: float = 0.0,
    ) -> None:
        """
        Set causal dimension of pose.

        NOTE: This is a HAND-SET method (H8-H9 critique). For emergent pose
        learning that discovers dimensions from data, use PoseDimensionDiscovery.

        Args:
            cause_strength: Strength as cause (0 = not cause, 1 = primary cause)
            effect_strength: Strength as effect (0 = not effect, 1 = direct effect)
            enables: Enabling relationship strength
            prevents: Prevention relationship strength
        """
        self._warn_hand_set("set_causal")
        row = SemanticDimension.CAUSAL
        self.matrix[row, 0] = np.clip(cause_strength, 0, 1)
        self.matrix[row, 1] = np.clip(effect_strength, 0, 1)
        self.matrix[row, 2] = np.clip(enables, 0, 1)
        self.matrix[row, 3] = np.clip(prevents, 0, 1)
        self._update_state_metrics()

    def set_semantic_role(
        self,
        agent: float = 0.0,
        patient: float = 0.0,
        instrument: float = 0.0,
        location: float = 0.0,
    ) -> None:
        """
        Set semantic role dimension of pose.

        NOTE: This is a HAND-SET method (H8-H9 critique). For emergent pose
        learning that discovers dimensions from data, use PoseDimensionDiscovery.

        Args:
            agent: Agent/actor role strength
            patient: Patient/recipient role strength
            instrument: Instrument/tool role strength
            location: Location/setting role strength
        """
        self._warn_hand_set("set_semantic_role")
        row = SemanticDimension.SEMANTIC_ROLE
        self.matrix[row, 0] = np.clip(agent, 0, 1)
        self.matrix[row, 1] = np.clip(patient, 0, 1)
        self.matrix[row, 2] = np.clip(instrument, 0, 1)
        self.matrix[row, 3] = np.clip(location, 0, 1)
        self._update_state_metrics()

    def set_certainty(
        self,
        definite: float = 1.0,
        possible: float = 0.0,
        negated: float = 0.0,
        hypothetical: float = 0.0,
    ) -> None:
        """
        Set epistemic certainty dimension of pose.

        NOTE: This is a HAND-SET method (H8-H9 critique). For emergent pose
        learning that discovers dimensions from data, use PoseDimensionDiscovery.

        Args:
            definite: Definiteness (1 = certain fact)
            possible: Possibility (1 = mere possibility)
            negated: Negation strength (1 = fully negated)
            hypothetical: Hypothetical nature (1 = counterfactual)
        """
        self._warn_hand_set("set_certainty")
        row = SemanticDimension.CERTAINTY
        self.matrix[row, 0] = np.clip(definite, 0, 1)
        self.matrix[row, 1] = np.clip(possible, 0, 1)
        self.matrix[row, 2] = np.clip(negated, 0, 1)
        self.matrix[row, 3] = np.clip(hypothetical, 0, 1)
        self._update_state_metrics()

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def compose(self, other: PoseMatrix) -> PoseMatrix:
        """
        Compose two pose transformations.

        Composition is via matrix multiplication, analogous to
        3D rotation composition. Order matters: self @ other.

        Args:
            other: Another PoseMatrix to compose with

        Returns:
            New PoseMatrix with composed transformation

        Example:
            >>> # "John is agent" + "agent caused X" = "John caused X"
            >>> result = john_as_agent.compose(agent_causes_x)
        """
        composed_matrix = self.matrix @ other.matrix

        result = PoseMatrix(
            config=self.config,
            matrix=composed_matrix,
        )

        # Update state
        result.state.composition_depth = max(
            self.state.composition_depth,
            other.state.composition_depth
        ) + 1
        result.state.total_compositions = (
            self.state.total_compositions +
            other.state.total_compositions + 1
        )

        return result

    def agreement(self, other: PoseMatrix) -> float:
        """
        Compute pose agreement score.

        Agreement measures how similar two pose configurations are.
        Used in routing-by-agreement (Sabour et al. 2017).

        Args:
            other: Another PoseMatrix to compare with

        Returns:
            Agreement score in [0, 1] (higher = more similar)

        Example:
            >>> agreement = pose1.agreement(pose2)
            >>> if agreement > 0.8:
            ...     # High agreement: consistent part-whole relationship
        """
        # Frobenius distance between matrices
        diff = self.matrix - other.matrix
        distance = np.linalg.norm(diff, 'fro')

        # Convert to similarity (0 = identical, higher = more different)
        # Normalize by expected distance for random matrices
        max_distance = 2 * self.config.pose_dim  # Rough upper bound

        if self.config.use_log_agreement:
            # Log-space for numerical stability
            agreement = np.exp(-distance / self.config.agreement_temperature)
        else:
            # Linear agreement
            agreement = 1.0 - min(distance / max_distance, 1.0)

        # Update state
        self.state.last_agreement = float(agreement)
        self.state.total_updates += 1

        # Running statistics
        n = self.state.total_updates
        self.state.mean_agreement = (
            (n - 1) * self.state.mean_agreement + agreement
        ) / n
        self.state.min_agreement = min(self.state.min_agreement, agreement)
        self.state.max_agreement = max(self.state.max_agreement, agreement)

        return float(agreement)

    def inverse(self) -> PoseMatrix:
        """
        Compute inverse transformation.

        Returns:
            New PoseMatrix with inverse transformation
        """
        try:
            inv_matrix = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:
            # Singular matrix - use pseudoinverse
            logger.warning("Singular pose matrix, using pseudoinverse")
            inv_matrix = np.linalg.pinv(self.matrix)

        return PoseMatrix(config=self.config, matrix=inv_matrix)

    def interpolate(
        self,
        other: PoseMatrix,
        t: float,
    ) -> PoseMatrix:
        """
        Interpolate between two poses.

        Args:
            other: Target pose
            t: Interpolation factor (0 = self, 1 = other)

        Returns:
            Interpolated PoseMatrix
        """
        t = np.clip(t, 0, 1)
        interp_matrix = (1 - t) * self.matrix + t * other.matrix

        return PoseMatrix(config=self.config, matrix=interp_matrix)

    # -------------------------------------------------------------------------
    # Regularization
    # -------------------------------------------------------------------------

    def regularization_loss(self) -> float:
        """
        Compute regularization loss for training.

        Includes:
        - Frobenius norm penalty (prevent explosion)
        - Orthogonality encouragement (preserve distances)

        Returns:
            Regularization loss value
        """
        frob_loss = self.config.regularization * self.state.frobenius_norm ** 2
        orth_loss = self.config.orthogonality_weight * self.state.orthogonality_error ** 2

        return frob_loss + orth_loss

    def normalize(self) -> None:
        """Normalize matrix to unit Frobenius norm."""
        norm = np.linalg.norm(self.matrix, 'fro')
        if norm > self.config.eps:
            self.matrix /= norm
        self._update_state_metrics()

    def orthogonalize(self) -> None:
        """Project matrix to nearest orthogonal matrix via SVD."""
        U, _, Vt = np.linalg.svd(self.matrix, full_matrices=False)
        self.matrix = (U @ Vt).astype(np.float32)
        self._update_state_metrics()

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def to_vector(self) -> np.ndarray:
        """Flatten pose matrix to vector."""
        return self.matrix.flatten()

    @classmethod
    def from_vector(
        cls,
        vector: np.ndarray,
        config: PoseConfig | None = None,
    ) -> PoseMatrix:
        """Create PoseMatrix from flattened vector."""
        config = config or PoseConfig()
        matrix = vector.reshape(config.pose_dim, config.pose_dim)
        return cls(config=config, matrix=matrix)

    def copy(self) -> PoseMatrix:
        """Create a copy of this pose matrix."""
        return PoseMatrix(
            config=self.config,
            matrix=self.matrix.copy(),
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PoseMatrix(dim={self.config.pose_dim}, "
            f"det={self.state.determinant:.3f}, "
            f"frob={self.state.frobenius_norm:.3f})"
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_identity_pose(config: PoseConfig | None = None) -> PoseMatrix:
    """Create identity pose transformation."""
    config = config or PoseConfig()
    matrix = np.eye(config.pose_dim, dtype=np.float32)
    return PoseMatrix(config=config, matrix=matrix)


def create_random_pose(
    config: PoseConfig | None = None,
    random_seed: int | None = None,
) -> PoseMatrix:
    """Create random pose transformation."""
    return PoseMatrix(config=config, random_seed=random_seed)


def batch_agreement(
    poses1: np.ndarray,
    poses2: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Compute agreement scores for batches of poses.

    Args:
        poses1: Array of shape [N, pose_dim, pose_dim]
        poses2: Array of shape [N, pose_dim, pose_dim]
        temperature: Agreement softness

    Returns:
        Agreement scores of shape [N]
    """
    # Frobenius distance per pair
    diff = poses1 - poses2
    distances = np.linalg.norm(diff.reshape(len(poses1), -1), axis=1)

    # Convert to agreement
    max_distance = 2 * poses1.shape[1]
    agreements = 1.0 - np.minimum(distances / max_distance, 1.0)

    return agreements.astype(np.float32)


def batch_compose(
    poses1: np.ndarray,
    poses2: np.ndarray,
) -> np.ndarray:
    """
    Compose batches of pose matrices.

    Args:
        poses1: Array of shape [N, pose_dim, pose_dim]
        poses2: Array of shape [N, pose_dim, pose_dim]

    Returns:
        Composed poses of shape [N, pose_dim, pose_dim]
    """
    return np.matmul(poses1, poses2).astype(np.float32)
