"""
Apical Modulation — Stage 4 of the spiking cortical block.

Implements prediction error computation and Forward-Forward goodness
via apical dendrite-style multiplicative modulation (Ca2+ gate).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ApicalModulation(nn.Module):
    """Prediction error + Ca2+ gate + FF goodness."""

    def __init__(self, dim: int, context_dim: int | None = None):
        super().__init__()
        self.basal_proj = nn.Linear(dim, dim)
        self.apical_proj = nn.Linear(context_dim or dim, dim)
        self.calcium_gate = nn.Linear(dim, dim)

    def forward(
        self, basal_input: Tensor, apical_input: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute modulated output, prediction error, and goodness.

        Args:
            basal_input: Bottom-up feedforward input (..., dim).
            apical_input: Top-down predictive input (..., context_dim).

        Returns:
            (output, prediction_error, goodness)
            - output: Gated activation.
            - prediction_error: Squared difference norm per sample.
            - goodness: FF goodness G(h) = sum(h_i^2) per sample.
        """
        h_basal = self.basal_proj(basal_input)

        if apical_input is not None:
            h_apical = self.apical_proj(apical_input)
        else:
            h_apical = torch.zeros_like(h_basal)

        gate = torch.sigmoid(self.calcium_gate(h_apical))
        output = h_basal * gate

        prediction_error = (h_basal - h_apical).pow(2).sum(dim=-1)
        goodness = h_basal.pow(2).sum(dim=-1)

        return output, prediction_error, goodness
