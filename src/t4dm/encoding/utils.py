"""
Utility functions for bioinspired encoding.
"""


import numpy as np
import torch
import torch.nn.functional as F


def compute_sparsity(x: torch.Tensor) -> float:
    """
    Compute sparsity ratio of a tensor.

    Args:
        x: Input tensor

    Returns:
        Fraction of non-zero elements
    """
    return (x != 0).float().mean().item()


def validate_sparsity(
    x: torch.Tensor,
    min_sparsity: float = 0.01,
    max_sparsity: float = 0.05
) -> tuple[bool, float]:
    """
    Validate sparsity is within biological range.

    Args:
        x: Sparse tensor
        min_sparsity: Minimum acceptable sparsity (default: 1%)
        max_sparsity: Maximum acceptable sparsity (default: 5%)

    Returns:
        (is_valid, actual_sparsity)
    """
    actual = compute_sparsity(x)
    is_valid = min_sparsity <= actual <= max_sparsity
    return is_valid, actual


def cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        x: Input tensor of shape (N, D)

    Returns:
        Similarity matrix of shape (N, N)
    """
    x_norm = F.normalize(x, p=2, dim=1)
    return torch.mm(x_norm, x_norm.t())


def compute_pattern_orthogonality(patterns: torch.Tensor) -> float:
    """
    Compute average orthogonality of patterns.

    Args:
        patterns: Tensor of shape (N, D) containing N patterns

    Returns:
        Average off-diagonal correlation (lower = more orthogonal)
    """
    if patterns.shape[0] < 2:
        return 1.0

    sim_matrix = cosine_similarity_matrix(patterns)

    # Get off-diagonal elements
    n = sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=bool, device=sim_matrix.device)
    off_diagonal = sim_matrix[mask]

    return 1.0 - off_diagonal.abs().mean().item()


def straight_through_estimator(
    x: torch.Tensor,
    threshold: torch.Tensor
) -> torch.Tensor:
    """
    Straight-through estimator for non-differentiable operations.

    Forward: Apply threshold
    Backward: Pass gradients through unchanged

    Args:
        x: Input tensor
        threshold: Threshold tensor

    Returns:
        Thresholded tensor with straight-through gradients
    """
    # Forward: hard threshold
    hard = (x > threshold).float()

    # Backward: pass through gradients
    return x + (hard - x).detach()


def exponential_decay(
    initial: float,
    tau: float,
    dt: float = 1.0
) -> float:
    """
    Compute exponential decay factor.

    Args:
        initial: Initial value
        tau: Time constant
        dt: Time step

    Returns:
        Decayed value
    """
    return initial * np.exp(-dt / tau)


def normalize_to_range(
    x: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Normalize tensor to specified range.

    Args:
        x: Input tensor
        min_val: Minimum output value
        max_val: Maximum output value

    Returns:
        Normalized tensor
    """
    x_min = x.min()
    x_max = x.max()

    if x_max - x_min < 1e-8:
        return torch.full_like(x, (min_val + max_val) / 2)

    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm * (max_val - min_val) + min_val


def add_noise(
    x: torch.Tensor,
    noise_std: float = 0.1,
    noise_type: str = "gaussian"
) -> torch.Tensor:
    """
    Add noise to tensor.

    Args:
        x: Input tensor
        noise_std: Standard deviation of noise
        noise_type: Type of noise ('gaussian', 'uniform')

    Returns:
        Noisy tensor
    """
    if noise_type == "gaussian":
        noise = torch.randn_like(x) * noise_std
    elif noise_type == "uniform":
        noise = (torch.rand_like(x) - 0.5) * 2 * noise_std
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return x + noise
