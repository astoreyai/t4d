"""
τ(t) Temporal Gate — gates memory writes and plasticity.

τ(t) = σ(λ_ε·ε + λ_Δ·novelty + λ_r·reward)

Output ∈ (0,1) controls whether a memory is written and
how strongly plasticity is applied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TemporalGate(nn.Module):
    """Learnable temporal write gate for memory encoding."""

    def __init__(
        self,
        lambda_epsilon: float = 1.0,
        lambda_delta: float = 1.0,
        lambda_r: float = 1.0,
    ):
        super().__init__()
        self.lambdas = nn.Parameter(
            torch.tensor([lambda_epsilon, lambda_delta, lambda_r])
        )

    def forward(
        self,
        prediction_error: Tensor,
        novelty: Tensor,
        reward: Tensor,
    ) -> Tensor:
        """
        Compute gate value τ(t).

        Args:
            prediction_error: Scalar or batch of prediction errors.
            novelty: Scalar or batch of novelty scores.
            reward: Scalar or batch of reward signals.

        Returns:
            Gate value(s) in (0, 1).
        """
        signals = torch.stack(
            [prediction_error, novelty, reward], dim=-1
        )
        return torch.sigmoid((signals * self.lambdas).sum(dim=-1))
