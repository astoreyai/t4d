"""
Oscillator Bias — converts θ/γ/δ phase states to LIF bias currents.

Neural oscillations modulate spiking dynamics:
- θ (theta, 4-8 Hz): Gates episodic encoding
- γ (gamma, 30-100 Hz): Binding and attention
- δ (delta, 0.5-4 Hz): Deep sleep consolidation
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class OscillatorState:
    """Phase state of neural oscillators."""

    theta_phase: float = 0.0  # [0, 2π]
    gamma_phase: float = 0.0
    delta_phase: float = 0.0


class OscillatorBias(nn.Module):
    """Project oscillator phases to LIF bias currents."""

    def __init__(
        self,
        dim: int,
        theta_strength: float = 0.3,
        gamma_strength: float = 0.2,
        delta_strength: float = 0.1,
    ):
        super().__init__()
        self.theta_strength = theta_strength
        self.gamma_strength = gamma_strength
        self.delta_strength = delta_strength

        self.theta_proj = nn.Linear(1, dim, bias=False)
        self.gamma_proj = nn.Linear(1, dim, bias=False)
        self.delta_proj = nn.Linear(1, dim, bias=False)

    def forward(self, osc_state: OscillatorState) -> Tensor:
        """
        Compute bias currents from oscillator phases.

        Args:
            osc_state: Current oscillator phase state.

        Returns:
            Bias current tensor of shape (dim,).
        """
        device = self.theta_proj.weight.device

        theta_val = torch.tensor(
            [math.sin(osc_state.theta_phase)], device=device
        )
        gamma_val = torch.tensor(
            [math.sin(osc_state.gamma_phase)], device=device
        )
        delta_val = torch.tensor(
            [math.sin(osc_state.delta_phase)], device=device
        )

        bias = (
            self.theta_proj(theta_val) * self.theta_strength
            + self.gamma_proj(gamma_val) * self.gamma_strength
            + self.delta_proj(delta_val) * self.delta_strength
        )
        return bias.squeeze(0)
