"""
Sparse encoder with k-Winner-Take-All activation.

Biological inspiration: Hippocampal dentate gyrus pattern separation.

The sparse encoder expands input representations to a higher-dimensional
space with very sparse activation (~2% of neurons active), providing:
- Pattern separation (similar inputs → distinct codes)
- Memory capacity increase
- Noise robustness
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Security limits
MAX_INPUT_DIM = 16384
MAX_HIDDEN_DIM = 131072  # 128K max for memory safety


@dataclass
class SparseEncoderConfig:
    """Configuration for sparse encoder."""
    input_dim: int = 1024        # BGE-M3 embedding dimension
    hidden_dim: int = 8192       # Expanded sparse dimension (8x expansion)
    sparsity: float = 0.02       # Target sparsity (2%)
    use_kwta: bool = True        # Use k-WTA (exact k active)
    lateral_inhibition: float = 0.2  # Lateral inhibition strength


def kwta(
    x: torch.Tensor,
    k: int,
    dim: int = -1
) -> torch.Tensor:
    """
    k-Winner-Take-All activation.

    Keeps top-k activations per sample, zeros the rest.

    Args:
        x: Input tensor
        k: Number of winners to keep
        dim: Dimension to apply k-WTA along

    Returns:
        Sparse tensor with exactly k non-zero values per sample
    """
    # Get top-k values and indices
    topk_values, topk_indices = torch.topk(x, k, dim=dim)

    # Create sparse output
    sparse = torch.zeros_like(x)
    sparse.scatter_(dim, topk_indices, topk_values)

    # Straight-through estimator for gradients
    return x + (sparse - x).detach()


class SparseEncoder(nn.Module):
    """
    Sparse encoder with k-Winner-Take-All activation.

    Expands 1024-dim input to 8192-dim sparse code with ~2% active neurons.
    Inspired by hippocampal DG pattern separation.

    Features:
    - Exact sparsity via k-WTA
    - Lateral inhibition for competition
    - Pattern separation (decorrelation)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 8192,
        sparsity: float = 0.02,
        use_kwta: bool = True,
        lateral_inhibition: float = 0.2
    ):
        super().__init__()

        # Security validation
        if input_dim > MAX_INPUT_DIM:
            raise ValueError(f"input_dim ({input_dim}) exceeds MAX_INPUT_DIM ({MAX_INPUT_DIM})")
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim > MAX_HIDDEN_DIM:
            raise ValueError(f"hidden_dim ({hidden_dim}) exceeds MAX_HIDDEN_DIM ({MAX_HIDDEN_DIM})")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0 < sparsity < 1:
            raise ValueError(f"sparsity must be in (0, 1), got {sparsity}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        self.k = max(1, int(hidden_dim * sparsity))  # Number of active neurons (min 1)
        self.use_kwta = use_kwta
        self.lateral_inhibition = lateral_inhibition

        # Expansion layer
        self.W_expand = nn.Linear(input_dim, hidden_dim)

        # Optional lateral inhibition weights
        if lateral_inhibition > 0:
            # Learned inhibition pattern
            self.W_inhibit = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.zeros_(self.W_inhibit.weight)
        else:
            self.W_inhibit = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for sparse coding."""
        # Small random weights for expansion
        nn.init.normal_(self.W_expand.weight, std=0.01)
        nn.init.zeros_(self.W_expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse representation.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Sparse code (batch, hidden_dim) with ~sparsity fraction active
        """
        # Expansion
        h = self.W_expand(x)

        # Apply lateral inhibition if enabled
        if self.W_inhibit is not None and self.lateral_inhibition > 0:
            inhibition = self.W_inhibit(F.relu(h))
            h = h - self.lateral_inhibition * inhibition

        # Apply sparsity
        if self.use_kwta:
            # Exact k winners
            sparse = kwta(h, self.k, dim=-1)
        else:
            # Soft thresholding
            threshold = torch.quantile(
                h.abs(), 1.0 - self.sparsity, dim=-1, keepdim=True
            )
            sparse = torch.where(h.abs() > threshold, h, torch.zeros_like(h))

        return sparse

    def decode(self, sparse_code: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input from sparse code (pseudo-inverse).

        Useful for validation and visualization.

        Args:
            sparse_code: Sparse representation (batch, hidden_dim)

        Returns:
            Reconstructed input (batch, input_dim)
        """
        # Pseudo-inverse reconstruction
        return F.linear(sparse_code, self.W_expand.weight.t())

    def get_active_indices(self, sparse_code: torch.Tensor) -> list[list[int]]:
        """
        Get indices of active neurons for each sample.

        Args:
            sparse_code: Sparse representation

        Returns:
            List of active neuron indices for each sample
        """
        return [
            torch.nonzero(sample).flatten().tolist()
            for sample in sparse_code
        ]

    def compute_pattern_overlap(
        self,
        code1: torch.Tensor,
        code2: torch.Tensor
    ) -> float:
        """
        Compute overlap between two sparse codes.

        Args:
            code1, code2: Sparse codes to compare

        Returns:
            Jaccard similarity of active neuron sets
        """
        active1 = (code1 != 0).float()
        active2 = (code2 != 0).float()

        intersection = (active1 * active2).sum()
        union = ((active1 + active2) > 0).float().sum()

        return (intersection / (union + 1e-8)).item()

    @property
    def actual_sparsity(self) -> float:
        """Target sparsity (for validation)."""
        return self.k / self.hidden_dim


class AdaptiveSparseEncoder(SparseEncoder):
    """
    Sparse encoder with adaptive sparsity based on input statistics.

    Adjusts k dynamically based on input magnitude/variance.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 8192,
        target_sparsity: float = 0.02,
        min_sparsity: float = 0.01,
        max_sparsity: float = 0.05,
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity=target_sparsity,
            **kwargs
        )

        self.target_sparsity = target_sparsity
        self.min_k = max(1, int(hidden_dim * min_sparsity))
        self.max_k = int(hidden_dim * max_sparsity)
        self.adaptation_rate = adaptation_rate

        # Running statistics for adaptation
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with adaptive sparsity."""
        # Update running statistics
        if self.training:
            batch_mean = x.mean()
            batch_var = x.var()
            self.running_mean = (
                (1 - self.adaptation_rate) * self.running_mean +
                self.adaptation_rate * batch_mean
            )
            self.running_var = (
                (1 - self.adaptation_rate) * self.running_var +
                self.adaptation_rate * batch_var
            )

        # Adapt k based on input magnitude
        magnitude_factor = (x.abs().mean() / (self.running_mean.abs() + 1e-8)).item()
        adapted_k = int(self.k * magnitude_factor)
        adapted_k = max(self.min_k, min(self.max_k, adapted_k))

        # Expansion
        h = self.W_expand(x)

        # Apply lateral inhibition
        if self.W_inhibit is not None and self.lateral_inhibition > 0:
            inhibition = self.W_inhibit(F.relu(h))
            h = h - self.lateral_inhibition * inhibition

        # k-WTA with adapted k
        return kwta(h, adapted_k, dim=-1)
