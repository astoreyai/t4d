"""
Leaky Integrate-and-Fire neuron with surrogate gradient.

u(t+1) = α·u(t) + I(t)
spike  = Heaviside(u - v_thresh)
u(t+1) -= β·v_thresh·spike   (soft reset)

Backward pass uses ATan surrogate for the Heaviside step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _SurrogateSpike(torch.autograd.Function):
    """ATan surrogate gradient for Heaviside spike function."""

    @staticmethod
    def forward(ctx, membrane: Tensor, threshold: float, alpha: float = 2.0):
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.alpha = alpha
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (membrane,) = ctx.saved_tensors
        v = membrane - ctx.threshold
        # ATan surrogate: d/dx (1/π) arctan(α·x) + 0.5 = α / (π(1 + α²x²))
        a = ctx.alpha
        surrogate = a / (torch.pi * (1.0 + (a * v) ** 2))
        return grad_output * surrogate, None, None


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron layer with soft reset and STE."""

    def __init__(
        self,
        size: int,
        alpha: float = 0.9,
        v_thresh: float = 1.0,
        beta: float = 1.0,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.v_thresh = v_thresh
        self.beta = beta
        self.surrogate_alpha = surrogate_alpha

    def forward(
        self, input_current: Tensor, membrane_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Single timestep LIF update.

        Args:
            input_current: (batch, size) or (size,) input current.
            membrane_state: Previous membrane potential; zeros if None.

        Returns:
            (spikes, new_membrane_state)
        """
        if membrane_state is None:
            membrane_state = torch.zeros_like(input_current)

        # Leak + integrate
        u = self.alpha * membrane_state + input_current

        # Spike via surrogate
        spikes = _SurrogateSpike.apply(u, self.v_thresh, self.surrogate_alpha)

        # Soft reset: subtract threshold from membrane where spiked
        u = u - self.beta * self.v_thresh * spikes

        return spikes, u
