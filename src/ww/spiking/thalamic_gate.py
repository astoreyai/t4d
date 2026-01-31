"""
Thalamic Gate — Stage 1 of the spiking cortical block.

ACh-modulated input masking that filters incoming activations
before LIF integration. Higher ACh = stronger gating (encoding mode).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ThalamicGate(nn.Module):
    """ACh-modulated multiplicative input gate."""

    def __init__(self, input_dim: int, context_dim: int | None = None):
        super().__init__()
        self.gate_proj = nn.Linear(context_dim or input_dim, input_dim)

    def forward(
        self, x: Tensor, context: Tensor | None = None, ach_level: float = 0.5
    ) -> Tensor:
        """
        Apply thalamic gating.

        Args:
            x: Input tensor (..., input_dim).
            context: Optional context tensor for gate computation.
            ach_level: Acetylcholine level [0, 1] modulating gate strength.

        Returns:
            Gated input tensor.
        """
        gate_input = context if context is not None else x
        gate = torch.sigmoid(self.gate_proj(gate_input))
        gate = gate * (0.5 + ach_level)
        return gate * x
