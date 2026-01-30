"""
Anti-Hebbian Learning for Sparse, Decorrelated Representations.

Biological Basis:
- Foldiak (1990): Lateral inhibition creates sparse codes
- Bell & Sejnowski (1995): Independent component analysis in neural networks
- Olshausen & Field (1996): Sparse coding in visual cortex

Key Mechanism:
- Feedforward weights: Hebbian (increase correlation)
- Lateral weights: Anti-Hebbian (decrease correlation)
- Result: Sparse, decorrelated representations ideal for memory

The Foldiak algorithm:
1. Feedforward: dW_ff = eta * y * x^T (Hebbian)
2. Lateral: dW_lat = -eta * (y * y^T - I) (Anti-Hebbian, drives toward identity)
3. Recurrent settling: y = sigma(W_ff * x - W_lat * y)

References:
- Foldiak (1990): Forming sparse representations
- Olshausen & Field (1996): Sparse code for natural images
- Bell & Sejnowski (1995): Information-maximization for ICA
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AntiHebbianConfig:
    """Configuration for anti-Hebbian learning.

    Attributes:
        input_dim: Input dimension
        output_dim: Output dimension (number of units)
        learning_rate_ff: Learning rate for feedforward weights
        learning_rate_lat: Learning rate for lateral weights
        sparsity_target: Target sparsity (fraction of active units)
        decorrelation_strength: How strongly to decorrelate
        n_iterations: Number of settling iterations
        weight_decay: L2 regularization
        lateral_self_inhibition: Self-inhibition strength
    """
    input_dim: int = 1024
    output_dim: int = 512
    learning_rate_ff: float = 0.01
    learning_rate_lat: float = 0.001
    sparsity_target: float = 0.05
    decorrelation_strength: float = 1.0
    n_iterations: int = 5
    weight_decay: float = 1e-4
    lateral_self_inhibition: float = 0.1


class FoldiakLayer:
    """
    Single layer implementing Foldiak's anti-Hebbian algorithm.

    Creates sparse, decorrelated representations through:
    1. Hebbian feedforward learning (W_ff)
    2. Anti-Hebbian lateral learning (W_lat)
    3. Recurrent competition via lateral inhibition

    Biological interpretation:
    - W_ff: Thalamocortical connections
    - W_lat: Horizontal intracortical connections
    - Settling: Recurrent dynamics reaching equilibrium
    """

    def __init__(self, config: AntiHebbianConfig | None = None):
        """
        Initialize Foldiak layer.

        Args:
            config: Layer configuration
        """
        self.config = config or AntiHebbianConfig()

        # Initialize weights
        # Feedforward: small random, normalized
        self.W_ff = np.random.randn(
            self.config.output_dim,
            self.config.input_dim
        ).astype(np.float32) * 0.01

        # Normalize rows
        row_norms = np.linalg.norm(self.W_ff, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-8)
        self.W_ff /= row_norms

        # Lateral: start near identity (no inhibition)
        self.W_lat = np.eye(
            self.config.output_dim,
            dtype=np.float32
        ) * self.config.lateral_self_inhibition

        # Thresholds for sparsity control
        self.thresholds = np.zeros(self.config.output_dim, dtype=np.float32)

        # Running statistics
        self._activity_ema = np.zeros(self.config.output_dim, dtype=np.float32)
        self._correlation_matrix = np.eye(self.config.output_dim, dtype=np.float32)

        logger.info(
            f"FoldiakLayer initialized: {self.config.input_dim} -> "
            f"{self.config.output_dim}, sparsity_target={self.config.sparsity_target}"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with recurrent settling.

        Args:
            x: Input array [input_dim] or [batch, input_dim]

        Returns:
            Output activations [output_dim] or [batch, output_dim]
        """
        x = np.atleast_2d(x)
        batch_size = x.shape[0]

        # Initialize output
        y = np.zeros((batch_size, self.config.output_dim), dtype=np.float32)

        # Feedforward drive
        ff_input = x @ self.W_ff.T  # [batch, output_dim]

        # Recurrent settling
        for _ in range(self.config.n_iterations):
            # Lateral inhibition
            lateral = y @ self.W_lat.T

            # Total input
            total = ff_input - lateral - self.thresholds

            # Activation (ReLU for sparsity)
            y = np.maximum(0, total)

        # Squeeze if single input
        if y.shape[0] == 1:
            y = y.squeeze(0)

        return y

    def _settle(self, x: np.ndarray) -> np.ndarray:
        """
        Settle to equilibrium state (internal method).

        Args:
            x: Input [input_dim]

        Returns:
            Settled output [output_dim]
        """
        return self.forward(x)

    def learn(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Update weights using Foldiak's Hebbian/anti-Hebbian rules.

        Args:
            x: Input [input_dim] or [batch, input_dim]
            y: Output [output_dim] or [batch, output_dim]

        Returns:
            Dict with learning statistics
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        batch_size = x.shape[0]

        # Hebbian feedforward update: dW_ff = eta * y^T * x
        dW_ff = (y.T @ x) / batch_size
        self.W_ff += self.config.learning_rate_ff * dW_ff

        # Anti-Hebbian lateral update: dW_lat = -eta * (y^T * y - I)
        correlation = (y.T @ y) / batch_size
        target = np.eye(self.config.output_dim) * (y.mean() ** 2 + 0.01)
        dW_lat = self.config.decorrelation_strength * (correlation - target)

        # Zero diagonal (no self-inhibition learning)
        np.fill_diagonal(dW_lat, 0)

        self.W_lat += self.config.learning_rate_lat * dW_lat

        # Weight decay
        self.W_ff *= (1 - self.config.weight_decay)

        # Normalize feedforward weights
        row_norms = np.linalg.norm(self.W_ff, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-8)
        self.W_ff /= row_norms

        # Update thresholds for sparsity
        mean_activity = y.mean(axis=0)
        self._activity_ema = 0.99 * self._activity_ema + 0.01 * mean_activity
        threshold_error = self._activity_ema - self.config.sparsity_target
        self.thresholds += 0.001 * threshold_error

        # Update correlation matrix
        self._correlation_matrix = 0.99 * self._correlation_matrix + 0.01 * correlation

        return {
            "dW_ff_norm": float(np.linalg.norm(dW_ff)),
            "dW_lat_norm": float(np.linalg.norm(dW_lat)),
            "mean_activity": float(mean_activity.mean()),
            "sparsity": float((y > 0).mean()),
        }

    def compute_correlation_matrix(self) -> np.ndarray:
        """Get running correlation matrix estimate."""
        return self._correlation_matrix.copy()

    def get_sparsity(self) -> float:
        """Get current sparsity level based on activity EMA."""
        return float((self._activity_ema > 0.01).mean())

    def get_stats(self) -> dict:
        """Get layer statistics."""
        off_diagonal = self._correlation_matrix.copy()
        np.fill_diagonal(off_diagonal, 0)

        return {
            "sparsity": self.get_sparsity(),
            "mean_activity": float(self._activity_ema.mean()),
            "mean_threshold": float(self.thresholds.mean()),
            "W_ff_norm": float(np.linalg.norm(self.W_ff)),
            "W_lat_norm": float(np.linalg.norm(self.W_lat)),
            "mean_correlation": float(np.abs(off_diagonal).mean()),
            "max_correlation": float(np.abs(off_diagonal).max()),
        }


class AntiHebbianNetwork:
    """
    Multi-layer network with anti-Hebbian learning.

    Creates a hierarchy of progressively sparser, more decorrelated
    representations suitable for memory storage.

    Architecture:
        Input -> Layer1 (sparse) -> Layer2 (sparser) -> ...

    Each layer:
    - Reduces dimensionality
    - Increases sparsity
    - Decreases correlation
    """

    def __init__(
        self,
        layer_dims: list[int],
        sparsity_schedule: list[float] | None = None,
        base_config: AntiHebbianConfig | None = None,
    ):
        """
        Initialize multi-layer anti-Hebbian network.

        Args:
            layer_dims: Dimensions for each layer [input, layer1, layer2, ...]
            sparsity_schedule: Target sparsity for each layer
            base_config: Base configuration (overridden per layer)
        """
        self.layer_dims = layer_dims
        n_layers = len(layer_dims) - 1

        # Default sparsity schedule (progressively sparser)
        if sparsity_schedule is None:
            sparsity_schedule = [
                0.1 * (0.5 ** i) for i in range(n_layers)
            ]

        # Create layers
        self.layers: list[FoldiakLayer] = []
        for i in range(n_layers):
            config = AntiHebbianConfig(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                sparsity_target=sparsity_schedule[i],
                learning_rate_ff=(base_config.learning_rate_ff if base_config else 0.01),
                learning_rate_lat=(base_config.learning_rate_lat if base_config else 0.001),
            )
            self.layers.append(FoldiakLayer(config))

        logger.info(
            f"AntiHebbianNetwork initialized: {layer_dims}, "
            f"sparsity={sparsity_schedule}"
        )

    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        """
        Forward pass through all layers.

        Args:
            x: Input [input_dim] or [batch, input_dim]

        Returns:
            List of activations for each layer
        """
        activations = [x]
        current = x

        for layer in self.layers:
            current = layer.forward(current)
            activations.append(current)

        return activations

    def learn(self, x: np.ndarray) -> dict:
        """
        Learn from input using all layers.

        Args:
            x: Input [input_dim] or [batch, input_dim]

        Returns:
            Dict with per-layer statistics
        """
        # Forward pass
        activations = self.forward(x)

        # Backward learning (layer by layer)
        stats = {}
        for i, layer in enumerate(self.layers):
            layer_stats = layer.learn(activations[i], activations[i + 1])
            stats[f"layer_{i}"] = layer_stats

        return stats

    def get_decorrelation_stats(self) -> dict:
        """Get decorrelation statistics for all layers."""
        return {
            f"layer_{i}": layer.get_stats()
            for i, layer in enumerate(self.layers)
        }

    def get_final_representation(self, x: np.ndarray) -> np.ndarray:
        """Get the final sparse representation."""
        activations = self.forward(x)
        return activations[-1]


class SparseCodingLayer:
    """
    Sparse coding layer using iterative shrinkage-thresholding (ISTA).

    Alternative to Foldiak for achieving sparsity through
    L1-regularized reconstruction.

    min_y ||x - D*y||_2^2 + lambda * ||y||_1

    Biological interpretation:
    - D: Dictionary of receptive field patterns
    - y: Sparse code (neural activities)
    - lambda: Metabolic cost of activity
    """

    def __init__(
        self,
        input_dim: int = 1024,
        n_atoms: int = 512,
        sparsity_lambda: float = 0.1,
        learning_rate: float = 0.01,
        n_iterations: int = 20,
    ):
        """
        Initialize sparse coding layer.

        Args:
            input_dim: Input dimension
            n_atoms: Number of dictionary atoms (sparse units)
            sparsity_lambda: Sparsity penalty
            learning_rate: Dictionary learning rate
            n_iterations: ISTA iterations
        """
        self.input_dim = input_dim
        self.n_atoms = n_atoms
        self.sparsity_lambda = sparsity_lambda
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Dictionary (column normalized)
        self.D = np.random.randn(input_dim, n_atoms).astype(np.float32)
        self.D /= np.linalg.norm(self.D, axis=0, keepdims=True)

        # Precompute for ISTA
        self._update_lipschitz()

        logger.info(
            f"SparseCodingLayer initialized: {input_dim} -> {n_atoms}, "
            f"lambda={sparsity_lambda}"
        )

    def _update_lipschitz(self):
        """Update Lipschitz constant for ISTA step size."""
        # L = largest eigenvalue of D^T * D
        self._L = np.linalg.norm(self.D.T @ self.D, ord=2)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input using ISTA.

        Args:
            x: Input [input_dim] or [batch, input_dim]

        Returns:
            Sparse code [n_atoms] or [batch, n_atoms]
        """
        x = np.atleast_2d(x)

        # Initialize sparse code
        y = np.zeros((x.shape[0], self.n_atoms), dtype=np.float32)

        # ISTA iterations
        step_size = 1.0 / self._L

        for _ in range(self.n_iterations):
            # Gradient step
            residual = x - y @ self.D.T  # [batch, input_dim]
            grad = residual @ self.D  # [batch, n_atoms]
            y = y + step_size * grad

            # Soft thresholding (proximal operator for L1)
            threshold = step_size * self.sparsity_lambda
            y = np.sign(y) * np.maximum(np.abs(y) - threshold, 0)

        if y.shape[0] == 1:
            y = y.squeeze(0)

        return y

    def decode(self, y: np.ndarray) -> np.ndarray:
        """
        Decode sparse code to reconstruction.

        Args:
            y: Sparse code [n_atoms] or [batch, n_atoms]

        Returns:
            Reconstruction [input_dim] or [batch, input_dim]
        """
        y = np.atleast_2d(y)
        recon = y @ self.D.T

        if recon.shape[0] == 1:
            recon = recon.squeeze(0)

        return recon

    def learn(self, x: np.ndarray) -> dict:
        """
        Learn dictionary from input.

        Args:
            x: Input [input_dim] or [batch, input_dim]

        Returns:
            Dict with learning statistics
        """
        x = np.atleast_2d(x)

        # Encode
        y = self.encode(x)

        # Compute residual
        recon = self.decode(y)
        residual = x - recon

        # Dictionary update: D += eta * residual * y^T
        dD = (residual.T @ y) / x.shape[0]
        self.D += self.learning_rate * dD

        # Normalize columns
        self.D /= np.linalg.norm(self.D, axis=0, keepdims=True) + 1e-8

        # Update Lipschitz constant
        self._update_lipschitz()

        return {
            "reconstruction_error": float(np.mean(residual ** 2)),
            "sparsity": float((np.abs(y) > 0.01).mean()),
            "mean_activation": float(np.abs(y).mean()),
            "dictionary_change": float(np.linalg.norm(dD)),
        }


__all__ = [
    "AntiHebbianConfig",
    "FoldiakLayer",
    "AntiHebbianNetwork",
    "SparseCodingLayer",
]
