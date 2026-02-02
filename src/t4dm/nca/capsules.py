"""
Capsule Network Implementation for T4DM.

Implements Hinton's capsule networks (2017, 2018) with semantic pose matrices
for part-whole representation in memory systems. Addresses H8 (part-whole)
and H9 (pose estimation) requirements.

Theoretical Foundation:
=======================

1. Capsules (Sabour et al. 2017):
   - Group of neurons encoding both existence AND configuration
   - Existence = activation magnitude (probability entity exists)
   - Configuration = pose matrix (how entity is configured)

2. Routing-by-Agreement (Sabour et al. 2017):
   - Lower capsules "vote" on higher capsule poses
   - Agreement = consistency of votes
   - Routing coefficients based on agreement scores
   - Iterative refinement (typically 3 iterations)

3. Squashing Function:
   - Preserves direction, squashes magnitude to [0, 1]
   - v = ||s||² / (1 + ||s||²) * s/||s||
   - Non-linearity for existence probability

Integration with T4DM:
==============================
- Capsules encode memory components (entities, relations, contexts)
- Poses encode semantic configurations (temporal, causal, role, certainty)
- Routing enables part-whole composition of memories
- Integrates with Forward-Forward learning for local updates

References:
- Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic Routing Between Capsules
- Hinton, G. E., Sabour, S., & Frosst, N. (2018). Matrix capsules with EM routing
- Hinton, G. E. (2022). The Forward-Forward Algorithm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.special import softmax

from t4dm.nca.pose import (
    PoseConfig,
    PoseMatrix,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class SquashType(Enum):
    """Squashing function type for capsule activations."""

    HINTON = "hinton"  # Original: ||s||² / (1 + ||s||²) * s/||s||
    NORM = "norm"  # Simple normalization: s / ||s||
    SIGMOID = "sigmoid"  # Element-wise sigmoid
    SOFTMAX = "softmax"  # Softmax over capsule dimension


class RoutingType(Enum):
    """Routing algorithm type."""

    DYNAMIC = "dynamic"  # Original dynamic routing (Sabour 2017)
    EM = "em"  # EM routing (Hinton 2018)
    ATTENTION = "attention"  # Attention-based routing


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CapsuleConfig:
    """
    Configuration for capsule layers.

    Parameters based on Sabour et al. (2017) and Hinton et al. (2018):
    - num_capsules: Number of capsule units in layer
    - capsule_dim: Dimensionality of each capsule's activation vector
    - pose_dim: Size of pose transformation matrix
    - routing_iterations: Number of routing iterations (typically 3)

    Biological mapping:
    - Capsules ~ cortical microcolumns encoding entity/configuration
    - Routing ~ lateral connections for binding
    - Poses ~ reference frame transformations
    """

    # Layer dimensions
    input_dim: int = 1024  # Input embedding dimension
    num_capsules: int = 32  # Number of capsules in layer
    capsule_dim: int = 16  # Activation dimension per capsule
    pose_dim: int = 4  # Pose matrix dimension (4x4)

    # Routing parameters
    routing_iterations: int = 3  # Number of routing iterations
    routing_type: str = "dynamic"  # dynamic, em, attention
    routing_temperature: float = 1.0  # Softmax temperature

    # Squashing
    squash_type: str = "hinton"  # hinton, norm, sigmoid, softmax

    # Learning
    learning_rate: float = 0.01
    use_ff_learning: bool = True  # Integrate with Forward-Forward
    ff_threshold: float = 2.0  # Forward-Forward goodness threshold

    # Reconstruction (for unsupervised learning)
    use_reconstruction: bool = False
    reconstruction_weight: float = 0.0005

    # Regularization
    sparsity_target: float = 0.05  # Target activation sparsity
    sparsity_weight: float = 0.001
    pose_regularization: float = 0.01

    # Biological constraints
    max_activation: float = 1.0
    min_activation: float = 0.0

    # History tracking
    max_history: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.num_capsules > 0, "num_capsules must be positive"
        assert self.capsule_dim > 0, "capsule_dim must be positive"
        assert self.pose_dim > 0, "pose_dim must be positive"
        assert self.routing_iterations >= 1, "routing_iterations must be >= 1"
        assert self.routing_type in ("dynamic", "em", "attention"), \
            f"Unknown routing_type: {self.routing_type}"
        assert self.squash_type in ("hinton", "norm", "sigmoid", "softmax"), \
            f"Unknown squash_type: {self.squash_type}"


# =============================================================================
# Capsule State
# =============================================================================


@dataclass
class CapsuleState:
    """
    State tracking for capsule layer.

    Tracks activations, poses, routing coefficients, and learning metrics.
    """

    # Current activations
    activations: np.ndarray | None = None  # [num_capsules]
    poses: np.ndarray | None = None  # [num_capsules, pose_dim, pose_dim]

    # Routing state
    routing_logits: np.ndarray | None = None  # [input_caps, output_caps]
    routing_coefficients: np.ndarray | None = None  # After softmax
    agreement_scores: np.ndarray | None = None  # Per-capsule agreement

    # Metrics
    mean_activation: float = 0.0
    sparsity: float = 0.0
    mean_agreement: float = 0.0

    # Forward-Forward metrics (if integrated)
    goodness: float = 0.0
    is_positive: bool = True
    confidence: float = 0.0

    # Update tracking
    total_forwards: int = 0
    total_routes: int = 0


# =============================================================================
# Capsule Layer
# =============================================================================


class CapsuleLayer:
    """
    Primary capsule layer with routing-by-agreement.

    Transforms input to capsule activations and poses, then routes
    to higher-level capsules using agreement-based routing.

    Attributes:
        config: CapsuleConfig instance
        state: CapsuleState tracking
        W_caps: Weight matrix for capsule activations [input_dim, num_caps * caps_dim]
        W_pose: Weight matrix for pose estimation [input_dim, num_caps * pose_dim²]

    Example:
        >>> config = CapsuleConfig(input_dim=1024, num_capsules=32)
        >>> layer = CapsuleLayer(config)
        >>> activations, poses = layer.forward(embedding)
        >>> print(f"Active capsules: {(activations > 0.5).sum()}")
    """

    def __init__(
        self,
        config: CapsuleConfig | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize capsule layer.

        Args:
            config: Layer configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config or CapsuleConfig()
        self.state = CapsuleState()
        self._rng = np.random.default_rng(random_seed)

        # Initialize weights
        self._init_weights()

        # Pose configuration
        self.pose_config = PoseConfig(pose_dim=self.config.pose_dim)

        # Track last input for pose learning (Phase 6)
        self._last_input: np.ndarray = np.array([])
        self._last_predictions: np.ndarray | None = None

        # ATOM-P2-22: Track transform matrix drift
        self._transform_norms: list[float] = []

    def _init_weights(self) -> None:
        """Initialize layer weights."""
        cfg = self.config

        # Xavier initialization scale
        scale_caps = np.sqrt(2.0 / (cfg.input_dim + cfg.num_capsules * cfg.capsule_dim))
        scale_pose = np.sqrt(2.0 / (cfg.input_dim + cfg.num_capsules * cfg.pose_dim ** 2))

        # Capsule activation weights
        self.W_caps = self._rng.normal(
            0, scale_caps,
            (cfg.input_dim, cfg.num_capsules * cfg.capsule_dim)
        ).astype(np.float32)

        # Pose estimation weights
        self.W_pose = self._rng.normal(
            0, scale_pose,
            (cfg.input_dim, cfg.num_capsules * cfg.pose_dim ** 2)
        ).astype(np.float32)

        # Bias terms
        self.b_caps = np.zeros(cfg.num_capsules * cfg.capsule_dim, dtype=np.float32)
        self.b_pose = np.zeros(cfg.num_capsules * cfg.pose_dim ** 2, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through capsule layer.

        Args:
            x: Input embedding [input_dim] or [batch, input_dim]

        Returns:
            activations: Capsule existence probabilities [num_capsules]
            poses: Pose matrices [num_capsules, pose_dim, pose_dim]
        """
        # Handle batched input
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_size = x.shape[0]

        # Track input for pose learning (Phase 6)
        self._last_input = x[0] if batch_size == 1 else x

        # Compute raw capsule outputs
        caps_raw = x @ self.W_caps + self.b_caps  # [batch, num_caps * caps_dim]
        pose_raw = x @ self.W_pose + self.b_pose  # [batch, num_caps * pose_dim²]

        # Reshape capsule activations
        caps_reshaped = caps_raw.reshape(
            batch_size, self.config.num_capsules, self.config.capsule_dim
        )

        # Squash to get existence probabilities
        activations = self._squash(caps_reshaped)  # [batch, num_caps, caps_dim]

        # Compute activation magnitudes (existence probability)
        activation_magnitudes = np.linalg.norm(activations, axis=-1)  # [batch, num_caps]

        # Reshape poses
        poses = pose_raw.reshape(
            batch_size,
            self.config.num_capsules,
            self.config.pose_dim,
            self.config.pose_dim
        )

        # Store state (single sample case)
        if batch_size == 1:
            self.state.activations = activation_magnitudes[0]
            self.state.poses = poses[0]
            self._update_state_metrics()

        self.state.total_forwards += 1

        return activation_magnitudes.squeeze(), poses.squeeze()

    def forward_with_routing(
        self,
        x: np.ndarray,
        routing_iterations: int | None = None,
        learn_poses: bool = True,
        learning_rate: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Forward pass with self-routing for pose refinement.

        This is THE KEY Phase 6 method: poses emerge from routing agreement
        rather than direct linear transform. Combines:
        1. Initial forward pass to get activations/poses
        2. Self-routing to refine poses through agreement
        3. Optional pose weight learning from routing

        Phase E2: Uses ACh-modulated routing iterations if set.

        Args:
            x: Input embedding [input_dim] or [batch, input_dim]
            routing_iterations: Number of routing iterations (default: config value)
            learn_poses: If True, update pose weights from routing agreement
            learning_rate: Optional override learning rate

        Returns:
            activations: Refined capsule activations [num_capsules]
            poses: Refined poses from routing [num_capsules, pose_dim, pose_dim]
            routing_stats: Dictionary with routing statistics
        """
        # Step 1: Initial forward pass
        initial_activations, initial_poses = self.forward(x)

        # Phase E2: Use ACh-modulated iterations if not explicitly overridden
        if routing_iterations is None:
            routing_iterations = self._get_effective_routing_iterations()

        # Step 2: Self-routing - route from this layer's capsules to refined poses
        # This implements routing-by-agreement among the capsules themselves
        refined_activations, refined_poses, routing_coeff = self.route(
            lower_activations=initial_activations,
            lower_poses=initial_poses,
            num_output_capsules=self.config.num_capsules,  # Same number (self-routing)
            iterations=routing_iterations,
            learn_poses=learn_poses,
            learning_rate=learning_rate,
        )

        # Update state with refined values
        self.state.activations = refined_activations
        self.state.poses = refined_poses
        self._update_state_metrics()

        # Compile routing statistics
        routing_stats = {
            'mean_agreement': self.state.mean_agreement,
            'initial_activation_mean': float(np.mean(initial_activations)),
            'refined_activation_mean': float(np.mean(refined_activations)),
            'pose_change': float(np.linalg.norm(refined_poses - initial_poses)),
            'routing_iterations': routing_iterations or self.config.routing_iterations,
            'learned_poses': learn_poses,
        }

        return refined_activations, refined_poses, routing_stats

    def _squash(self, s: np.ndarray) -> np.ndarray:
        """
        Apply squashing function.

        Hinton squash: v = ||s||² / (1 + ||s||²) * s/||s||

        Args:
            s: Input vectors [batch, num_caps, caps_dim]

        Returns:
            Squashed vectors with magnitude in [0, 1]
        """
        if self.config.squash_type == "hinton":
            # Original Hinton squashing
            norm_sq = np.sum(s ** 2, axis=-1, keepdims=True)
            norm = np.sqrt(norm_sq + 1e-8)
            scale = norm_sq / (1 + norm_sq)
            return scale * s / norm

        elif self.config.squash_type == "norm":
            # Simple L2 normalization
            norm = np.linalg.norm(s, axis=-1, keepdims=True) + 1e-8
            return s / norm

        elif self.config.squash_type == "sigmoid":
            # Element-wise sigmoid
            return 1 / (1 + np.exp(-s))

        elif self.config.squash_type == "softmax":
            # Softmax over capsule dimension
            return softmax(s, axis=-1)

        else:
            raise ValueError(f"Unknown squash type: {self.config.squash_type}")

    def _update_state_metrics(self) -> None:
        """Update state metrics from current activations."""
        if self.state.activations is not None:
            self.state.mean_activation = float(np.mean(self.state.activations))
            self.state.sparsity = float(
                np.mean(self.state.activations < self.config.sparsity_target)
            )

            # Forward-Forward goodness
            if self.config.use_ff_learning:
                self.state.goodness = float(np.sum(self.state.activations ** 2))
                self.state.is_positive = self.state.goodness > self.config.ff_threshold
                self.state.confidence = abs(self.state.goodness - self.config.ff_threshold)

    # -------------------------------------------------------------------------
    # Routing-by-Agreement
    # -------------------------------------------------------------------------

    def route(
        self,
        lower_activations: np.ndarray,
        lower_poses: np.ndarray,
        num_output_capsules: int,
        iterations: int | None = None,
        learn_poses: bool = False,
        learning_rate: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Route from lower capsules to higher capsules.

        Implements Hinton's routing-by-agreement algorithm.
        Phase 6: Optionally updates pose weights based on routing agreement.

        Args:
            lower_activations: Lower capsule activations [num_lower]
            lower_poses: Lower capsule poses [num_lower, pose_dim, pose_dim]
            num_output_capsules: Number of higher-level capsules
            iterations: Routing iterations (default: config value)
            learn_poses: If True, update pose weights from routing agreement (Phase 6)
            learning_rate: Optional override learning rate for pose learning

        Returns:
            higher_activations: [num_output]
            higher_poses: [num_output, pose_dim, pose_dim]
            routing_coefficients: [num_lower, num_output]
        """
        iterations = iterations or self.config.routing_iterations
        num_lower = len(lower_activations)

        # Initialize transformation matrices (W_ij for each lower->higher pair)
        if not hasattr(self, '_transform_matrices') or \
           self._transform_matrices.shape[0] != num_lower:
            self._init_transform_matrices(num_lower, num_output_capsules)

        # Initialize routing logits
        routing_logits = np.zeros(
            (num_lower, num_output_capsules),
            dtype=np.float32
        )

        # Compute predictions: u_j|i = W_ij @ pose_i
        # Shape: [num_lower, num_output, pose_dim, pose_dim]
        predictions = np.einsum(
            'ijkl,ikm->ijlm',
            self._transform_matrices,
            lower_poses
        )

        # Phase E2: Use modulated routing parameters if available
        effective_temperature = self._get_effective_routing_temperature()

        # Routing iterations
        for r in range(iterations):
            # ATOM-P2-21: Clip routing logits to prevent overflow
            # Pre-clip warning if any logit exceeds threshold
            max_logit = np.max(np.abs(routing_logits))
            if max_logit > 4.0:
                logger.warning(
                    f"Routing logit exceeds threshold: max={max_logit:.2f} "
                    f"(layer {self.layer_idx if hasattr(self, 'layer_idx') else 'unknown'})"
                )
            routing_logits = np.clip(routing_logits, -5.0, 5.0)

            # Softmax over output capsules
            # Phase E2: Use effective temperature (modulated by 5-HT)
            routing_coeff = softmax(
                routing_logits / effective_temperature,
                axis=1
            )

            # Weight by lower capsule activation
            weighted_coeff = routing_coeff * lower_activations[:, np.newaxis]

            # Compute higher capsule poses (weighted sum of predictions)
            # s_j = sum_i (c_ij * u_j|i)
            higher_poses = np.einsum('ij,ijkl->jkl', weighted_coeff, predictions)

            # Squash to get activations
            pose_norms = np.linalg.norm(
                higher_poses.reshape(num_output_capsules, -1),
                axis=1
            )
            higher_activations = self._squash_scalar(pose_norms)

            # Update routing logits based on agreement (except last iteration)
            if r < iterations - 1:
                # Agreement: how well does each prediction match the consensus?
                for i in range(num_lower):
                    for j in range(num_output_capsules):
                        agreement = self._pose_agreement(
                            predictions[i, j],
                            higher_poses[j]
                        )
                        routing_logits[i, j] += agreement

        # Store state
        self.state.routing_logits = routing_logits
        self.state.routing_coefficients = routing_coeff
        self.state.agreement_scores = self._compute_agreement_scores(
            predictions, higher_poses
        )
        self.state.mean_agreement = float(np.mean(self.state.agreement_scores))
        self.state.total_routes += 1

        # Store predictions for pose learning (Phase 6)
        self._last_predictions = predictions

        # Phase 6: Learn pose weights from routing agreement
        if learn_poses:
            self.learn_pose_from_routing(
                lower_poses=lower_poses,
                predictions=predictions,
                consensus_poses=higher_poses,
                agreement_scores=self.state.agreement_scores,
                learning_rate=learning_rate,
            )

        return higher_activations, higher_poses, routing_coeff

    def _init_transform_matrices(
        self,
        num_lower: int,
        num_output: int
    ) -> None:
        """Initialize transformation matrices for routing."""
        pose_dim = self.config.pose_dim
        scale = np.sqrt(2.0 / (pose_dim * pose_dim))

        # W_ij: transform from capsule i to capsule j
        # Shape: [num_lower, num_output, pose_dim, pose_dim]
        self._transform_matrices = self._rng.normal(
            0, scale,
            (num_lower, num_output, pose_dim, pose_dim)
        ).astype(np.float32)

        # Initialize near identity
        for i in range(num_lower):
            for j in range(num_output):
                self._transform_matrices[i, j] += np.eye(pose_dim, dtype=np.float32)

    def _squash_scalar(self, norms: np.ndarray) -> np.ndarray:
        """Squash scalar values (pose norms) to [0, 1]."""
        return norms ** 2 / (1 + norms ** 2)

    def _pose_agreement(
        self,
        prediction: np.ndarray,
        consensus: np.ndarray
    ) -> float:
        """Compute agreement between prediction and consensus."""
        diff = prediction - consensus
        distance = np.linalg.norm(diff)
        return float(np.exp(-distance / self.config.routing_temperature))

    def _compute_agreement_scores(
        self,
        predictions: np.ndarray,
        higher_poses: np.ndarray
    ) -> np.ndarray:
        """Compute agreement scores for all capsule pairs."""
        num_lower, num_output = predictions.shape[:2]
        # ATOM-P3-39: Vectorize O(n^2) loop using broadcasting
        # Diff: [num_lower, num_output, pose_dim, pose_dim]
        diff = predictions - higher_poses[np.newaxis, :, :, :]
        distances = np.linalg.norm(diff.reshape(num_lower, num_output, -1), axis=2)
        agreements = np.exp(-distances / self.config.routing_temperature)

        return agreements.astype(np.float32)

    # -------------------------------------------------------------------------
    # Pose Learning from Routing Agreement (Phase 6)
    # -------------------------------------------------------------------------

    def learn_pose_from_routing(
        self,
        lower_poses: np.ndarray,
        predictions: np.ndarray,
        consensus_poses: np.ndarray,
        agreement_scores: np.ndarray,
        learning_rate: float | None = None,
    ) -> dict:
        """
        Update pose weights based on routing agreement.

        This is THE KEY Phase 6 addition: poses emerge from routing, not hand-setting.
        When predictions match consensus well, reinforce those transformation weights.
        When agreement is low, adjust weights toward consensus.

        Args:
            lower_poses: Lower capsule poses [num_lower, pose_dim, pose_dim]
            predictions: Predicted higher poses [num_lower, num_output, pose_dim, pose_dim]
            consensus_poses: Final higher capsule poses [num_output, pose_dim, pose_dim]
            agreement_scores: Agreement between predictions and consensus [num_lower, num_output]
            learning_rate: Optional override learning rate

        Returns:
            Dictionary of learning statistics
        """
        lr = learning_rate or self.config.learning_rate
        num_lower, num_output = agreement_scores.shape
        pose_dim = self.config.pose_dim

        # Track statistics
        total_weight_change = 0.0
        total_transform_change = 0.0
        mean_agreement = float(np.mean(agreement_scores))

        # Update transform matrices based on agreement
        if hasattr(self, '_transform_matrices'):
            for i in range(num_lower):
                for j in range(num_output):
                    agreement = agreement_scores[i, j]

                    # Compute error: how far is prediction from consensus?
                    error = consensus_poses[j] - predictions[i, j]

                    # Learning signal: agreement-weighted error correction
                    # High agreement = small updates (already good)
                    # Low agreement = larger updates toward consensus
                    update_strength = (1.0 - agreement) * lr

                    # Gradient: dError/dW_ij ≈ lower_pose_i (chain rule approximation)
                    # Update: move transform toward producing consensus
                    transform_update = update_strength * np.einsum(
                        'kl,mn->km', lower_poses[i], error
                    )[:pose_dim, :pose_dim]

                    self._transform_matrices[i, j] += transform_update

                    # ATOM-P2-22: Monitor transform matrix drift
                    norm = np.linalg.norm(self._transform_matrices[i, j], 'fro')
                    self._transform_norms.append(norm)
                    if len(self._transform_norms) > 100:
                        mean_n = np.mean(self._transform_norms[-100:])
                        std_n = np.std(self._transform_norms[-100:])
                        if std_n > 0 and abs(norm - mean_n) > 3 * std_n:
                            logger.warning(
                                f"Transform drift detected: norm={norm:.4f}, "
                                f"mean={mean_n:.4f}, std={std_n:.4f} "
                                f"(transform [{i},{j}])"
                            )

                    total_transform_change += np.linalg.norm(transform_update)

        # Update pose estimation weights (W_pose) to better predict poses
        # that lead to high routing agreement
        if self.state.activations is not None and len(self._last_input) > 0:
            x = self._last_input

            # High agreement = reinforce those pose weights
            # Low agreement = adjust pose weights
            mean_pose = consensus_poses.mean(axis=0)

            # Compute pose gradient direction
            # W_pose shape: [input_dim, num_capsules * pose_dim²]
            current_pose_output = (x @ self.W_pose + self.b_pose).reshape(
                self.config.num_capsules, pose_dim, pose_dim
            )

            # Error toward high-agreement poses (broadcast mean to all capsules)
            pose_error = np.zeros_like(current_pose_output)
            for i in range(self.config.num_capsules):
                pose_error[i] = mean_pose - current_pose_output[i]

            # Gradient: outer product of input and flattened error
            x_reshaped = x.reshape(-1, 1)
            pose_error_flat = pose_error.flatten().reshape(1, -1)
            pose_gradient = x_reshaped @ pose_error_flat

            # Scale by mean agreement (higher agreement = less update needed)
            update_scale = (1.0 - mean_agreement) * lr

            # Update W_pose (ensure shapes match)
            self.W_pose += update_scale * pose_gradient

            total_weight_change = float(np.linalg.norm(update_scale * pose_gradient))

        return {
            'mean_agreement': mean_agreement,
            'weight_change': total_weight_change,
            'transform_change': total_transform_change,
            'num_pairs': num_lower * num_output,
            'learning_rate': lr,
        }

    # -------------------------------------------------------------------------
    # Learning
    # -------------------------------------------------------------------------

    def learn_positive(
        self,
        x: np.ndarray,
        activations: np.ndarray,
        poses: np.ndarray | None = None,
        learning_rate: float | None = None,
    ) -> dict:
        """
        Learn from positive sample (increase goodness).

        Forward-Forward style learning for capsules.
        Phase 6: Also updates pose weights to reinforce good representations.

        Args:
            x: Input embedding
            activations: Current capsule activations
            poses: Current capsule poses [num_caps, pose_dim, pose_dim] (Phase 6)
            learning_rate: Optional override learning rate

        Returns:
            Dictionary of learning statistics
        """
        lr = learning_rate or self.config.learning_rate

        # Compute goodness
        goodness = float(np.sum(activations ** 2))
        p = 1 / (1 + np.exp(-(goodness - self.config.ff_threshold)))

        # Gradient: increase goodness for positive samples
        # dG/dW = (1 - p) * d(sum(a²))/dW
        gradient_scale = (1 - p) * lr

        # Update capsule weights (simplified gradient)
        x_reshaped = x.reshape(-1, 1)
        caps_grad = x_reshaped @ activations.reshape(1, -1).repeat(
            self.config.capsule_dim, axis=1
        )

        self.W_caps += gradient_scale * caps_grad[:, :self.W_caps.shape[1]]

        # Phase 6: Update pose weights for positive samples
        pose_update_norm = 0.0
        if poses is not None:
            # Reinforce pose configuration that led to positive outcome
            pose_flat = poses.reshape(-1)
            pose_grad = x_reshaped @ pose_flat.reshape(1, -1)
            pose_update = gradient_scale * pose_grad[:, :self.W_pose.shape[1]]
            self.W_pose += pose_update
            pose_update_norm = float(np.linalg.norm(pose_update))

        return {
            'goodness': goodness,
            'probability': p,
            'gradient_scale': gradient_scale,
            'phase': 'positive',
            'pose_update_norm': pose_update_norm,
        }

    def learn_negative(
        self,
        x: np.ndarray,
        activations: np.ndarray,
        poses: np.ndarray | None = None,
        learning_rate: float | None = None,
    ) -> dict:
        """
        Learn from negative sample (decrease goodness).

        Phase 6: Also updates pose weights to suppress bad representations.

        Args:
            x: Input embedding
            activations: Current capsule activations
            poses: Current capsule poses [num_caps, pose_dim, pose_dim] (Phase 6)
            learning_rate: Optional override learning rate

        Returns:
            Dictionary of learning statistics
        """
        lr = learning_rate or self.config.learning_rate

        # Compute goodness
        goodness = float(np.sum(activations ** 2))
        p = 1 / (1 + np.exp(-(goodness - self.config.ff_threshold)))

        # Gradient: decrease goodness for negative samples
        gradient_scale = -p * lr

        # Update capsule weights
        x_reshaped = x.reshape(-1, 1)
        caps_grad = x_reshaped @ activations.reshape(1, -1).repeat(
            self.config.capsule_dim, axis=1
        )

        self.W_caps += gradient_scale * caps_grad[:, :self.W_caps.shape[1]]

        # Phase 6: Update pose weights for negative samples
        pose_update_norm = 0.0
        if poses is not None:
            # Suppress pose configuration that led to negative outcome
            pose_flat = poses.reshape(-1)
            pose_grad = x_reshaped @ pose_flat.reshape(1, -1)
            pose_update = gradient_scale * pose_grad[:, :self.W_pose.shape[1]]
            self.W_pose += pose_update
            pose_update_norm = float(np.linalg.norm(pose_update))

        return {
            'goodness': goodness,
            'probability': p,
            'gradient_scale': gradient_scale,
            'phase': 'negative',
            'pose_update_norm': pose_update_norm,
        }

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_most_active_capsules(self, k: int = 5) -> list[int]:
        """Get indices of k most active capsules."""
        if self.state.activations is None:
            return []
        return list(np.argsort(self.state.activations)[-k:][::-1])

    def get_pose_matrix(self, capsule_idx: int) -> PoseMatrix | None:
        """Get PoseMatrix object for specific capsule."""
        if self.state.poses is None:
            return None

        return PoseMatrix(
            config=self.pose_config,
            matrix=self.state.poses[capsule_idx]
        )

    # -------------------------------------------------------------------------
    # Phase E2: Neuromodulator Integration
    # -------------------------------------------------------------------------

    def receive_ach(self, level: float) -> None:
        """
        Receive acetylcholine modulation.

        Phase E2: High ACh → more routing iterations → sharper routing.
        Biological basis: ACh enhances cortical attention and precision.

        Args:
            level: ACh concentration [0, 1]
        """
        level = float(np.clip(level, 0, 1))

        # Modulate routing iterations
        # High ACh = more iterations = sharper/more refined routing
        base_iterations = 3  # Default from config
        ach_boost = int(level * 3)  # Up to +3 iterations at max ACh
        self._ach_modulated_iterations = base_iterations + ach_boost

        logger.debug(
            f"ACh modulation: level={level:.2f}, "
            f"routing_iterations={self._ach_modulated_iterations}"
        )

    def receive_5ht(self, level: float) -> None:
        """
        Receive serotonin modulation.

        Phase E2: High 5-HT → slower convergence → more patience.
        Biological basis: 5-HT increases temporal discounting (patience).

        Args:
            level: 5-HT concentration [0, 1]
        """
        level = float(np.clip(level, 0, 1))

        # Modulate convergence patience
        # High 5-HT = slower convergence = more thorough routing
        base_temp = self.config.routing_temperature
        patience_factor = 1 + 0.5 * level  # Up to 1.5x at max 5-HT
        self._5ht_modulated_temperature = base_temp * patience_factor

        logger.debug(
            f"5-HT modulation: level={level:.2f}, "
            f"routing_temperature={self._5ht_modulated_temperature:.3f}"
        )

    def _get_effective_routing_iterations(self) -> int:
        """Get effective routing iterations accounting for ACh modulation."""
        if hasattr(self, '_ach_modulated_iterations'):
            return self._ach_modulated_iterations
        return self.config.routing_iterations

    def _get_effective_routing_temperature(self) -> float:
        """Get effective routing temperature accounting for 5-HT modulation."""
        if hasattr(self, '_5ht_modulated_temperature'):
            return self._5ht_modulated_temperature
        return self.config.routing_temperature

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CapsuleLayer(in={self.config.input_dim}, "
            f"caps={self.config.num_capsules}×{self.config.capsule_dim}, "
            f"pose={self.config.pose_dim}×{self.config.pose_dim})"
        )


# =============================================================================
# Capsule Network
# =============================================================================


class CapsuleNetwork:
    """
    Multi-layer capsule network with routing.

    Stacks multiple capsule layers with routing-by-agreement
    between adjacent layers.

    Example:
        >>> network = CapsuleNetwork(
        ...     layer_configs=[
        ...         CapsuleConfig(input_dim=1024, num_capsules=32),
        ...         CapsuleConfig(input_dim=512, num_capsules=16),
        ...         CapsuleConfig(input_dim=256, num_capsules=8),
        ...     ]
        ... )
        >>> activations, poses = network.forward(embedding)
    """

    def __init__(
        self,
        layer_configs: list[CapsuleConfig],
        random_seed: int | None = None,
    ):
        """
        Initialize capsule network.

        Args:
            layer_configs: List of configurations for each layer
            random_seed: Random seed for reproducibility
        """
        self._rng = np.random.default_rng(random_seed)
        self.layers: list[CapsuleLayer] = []

        for i, config in enumerate(layer_configs):
            seed = None if random_seed is None else random_seed + i
            self.layers.append(CapsuleLayer(config, random_seed=seed))

    def forward(
        self,
        x: np.ndarray,
        return_all_layers: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | list[tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass through all layers.

        Args:
            x: Input embedding
            return_all_layers: If True, return activations from all layers

        Returns:
            Final layer (activations, poses) or list of all layers
        """
        all_outputs = []

        # First layer: direct forward
        activations, poses = self.layers[0].forward(x)
        all_outputs.append((activations, poses))

        # Subsequent layers: route from previous
        for i in range(1, len(self.layers)):
            num_output = self.layers[i].config.num_capsules
            activations, poses, _ = self.layers[i-1].route(
                activations, poses, num_output
            )
            all_outputs.append((activations, poses))

        if return_all_layers:
            return all_outputs
        return all_outputs[-1]

    def train_step(
        self,
        positive: np.ndarray,
        negative: np.ndarray,
        learn_poses: bool = True,
    ) -> dict:
        """
        Training step with positive and negative samples.

        Phase 6: Also updates pose weights during training.

        Args:
            positive: Positive (real) sample
            negative: Negative (corrupted) sample
            learn_poses: If True, also update pose weights (Phase 6)

        Returns:
            Dictionary of training statistics
        """
        stats = {'positive': [], 'negative': []}

        # Positive phase
        for layer in self.layers:
            activations, poses = layer.forward(positive)
            layer_stats = layer.learn_positive(
                positive,
                activations,
                poses=poses if learn_poses else None,
            )
            stats['positive'].append(layer_stats)

        # Negative phase
        for layer in self.layers:
            activations, poses = layer.forward(negative)
            layer_stats = layer.learn_negative(
                negative,
                activations,
                poses=poses if learn_poses else None,
            )
            stats['negative'].append(layer_stats)

        return stats

    def __repr__(self) -> str:
        """String representation."""
        layer_str = " -> ".join(
            f"{l.config.num_capsules}" for l in self.layers
        )
        return f"CapsuleNetwork({layer_str})"


# =============================================================================
# Factory Functions
# =============================================================================


def create_capsule_layer(
    input_dim: int = 1024,
    num_capsules: int = 32,
    capsule_dim: int = 16,
    pose_dim: int = 4,
    routing_iterations: int = 3,
    random_seed: int | None = None,
) -> CapsuleLayer:
    """
    Create a capsule layer with common defaults.

    Args:
        input_dim: Input embedding dimension
        num_capsules: Number of capsules
        capsule_dim: Activation dimension per capsule
        pose_dim: Pose matrix dimension
        routing_iterations: Number of routing iterations
        random_seed: Random seed for reproducibility

    Returns:
        Configured CapsuleLayer
    """
    config = CapsuleConfig(
        input_dim=input_dim,
        num_capsules=num_capsules,
        capsule_dim=capsule_dim,
        pose_dim=pose_dim,
        routing_iterations=routing_iterations,
    )
    return CapsuleLayer(config, random_seed=random_seed)


def create_capsule_network(
    layer_dims: list[int],
    input_dim: int = 1024,
    capsule_dim: int = 16,
    pose_dim: int = 4,
    random_seed: int | None = None,
) -> CapsuleNetwork:
    """
    Create a multi-layer capsule network.

    Args:
        layer_dims: Number of capsules in each layer (e.g., [32, 16, 8])
        input_dim: Input embedding dimension
        capsule_dim: Activation dimension per capsule
        pose_dim: Pose matrix dimension
        random_seed: Random seed for reproducibility

    Returns:
        Configured CapsuleNetwork
    """
    configs = []
    current_input = input_dim

    for i, num_caps in enumerate(layer_dims):
        config = CapsuleConfig(
            input_dim=current_input,
            num_capsules=num_caps,
            capsule_dim=capsule_dim,
            pose_dim=pose_dim,
        )
        configs.append(config)
        current_input = num_caps * capsule_dim

    return CapsuleNetwork(configs, random_seed=random_seed)
