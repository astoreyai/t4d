"""
Cortical Stack — N stacked cortical blocks with shared context.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from t4dm.spiking.cortical_block import CorticalBlock


class CorticalStack(nn.Module):
    """Stack of N cortical blocks with per-block state tracking."""

    def __init__(
        self,
        dim: int,
        num_blocks: int = 6,
        context_dim: int | None = None,
        num_heads: int = 8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CorticalBlock(dim, context_dim, num_heads)
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        neuromod_state: Any = None,
        states: list[dict] | None = None,
    ) -> tuple[Tensor, list[dict], list[dict[str, Any]]]:
        """
        Forward through all blocks.

        Args:
            x: (batch, seq, dim) input.
            context: Shared top-down context.
            neuromod_state: Object with .ach attribute (or None for default 0.5).
            states: Per-block recurrent states.

        Returns:
            (output, new_states, all_metrics)
        """
        states = states or [{}] * len(self.blocks)
        all_metrics: list[dict[str, Any]] = []

        ach = getattr(neuromod_state, "ach", 0.5) if neuromod_state else 0.5

        for i, block in enumerate(self.blocks):
            x, states[i], metrics = block(x, context, ach, states[i])
            all_metrics.append(metrics)

        return x, states, all_metrics
