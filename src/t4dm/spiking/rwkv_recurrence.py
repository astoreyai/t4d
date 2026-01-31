"""
RWKV Recurrence — Stage 5 of the spiking cortical block.

O(N) linear recurrence with time-mixing and channel-mixing.
Constant memory per token via recurrent state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RWKVRecurrence(nn.Module):
    """RWKV-style time-mixing + channel-mixing with O(1) state per token."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Time mixing parameters
        self.time_decay = nn.Parameter(torch.zeros(dim))
        self.time_mix_k = nn.Parameter(torch.ones(dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(dim) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(dim) * 0.5)

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        # Channel mixing
        self.cm_mix_k = nn.Parameter(torch.ones(dim) * 0.5)
        self.cm_mix_r = nn.Parameter(torch.ones(dim) * 0.5)
        self.cm_key = nn.Linear(dim, dim * 4, bias=False)
        self.cm_value = nn.Linear(dim * 4, dim, bias=False)
        self.cm_receptance = nn.Linear(dim, dim, bias=False)

    def _time_mixing(
        self, x: Tensor, state: dict | None
    ) -> tuple[Tensor, dict]:
        """WKV kernel with token shift."""
        B, T, D = x.shape
        state = state or {}

        last_x = state.get("last_x", torch.zeros(B, 1, D, device=x.device))
        # Shift: interpolate between previous and current token
        shifted = torch.cat([last_x, x[:, :-1, :]], dim=1)

        xk = x * self.time_mix_k + shifted * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + shifted * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + shifted * (1 - self.time_mix_r)

        k = self.key(xk)      # (B, T, D)
        v = self.value(xv)     # (B, T, D)
        r = torch.sigmoid(self.receptance(xr))  # (B, T, D)

        # Simplified WKV: exponential decay accumulation
        w = -torch.exp(self.time_decay)  # Negative for decay
        wkv_num = state.get("wkv_num", torch.zeros(B, D, device=x.device))
        wkv_den = state.get("wkv_den", torch.zeros(B, D, device=x.device))

        outputs = []
        for t in range(T):
            kt, vt = k[:, t], v[:, t]
            ew = torch.exp(w)
            wkv_num = ew * wkv_num + torch.exp(kt) * vt
            wkv_den = ew * wkv_den + torch.exp(kt)
            outputs.append(wkv_num / (wkv_den + 1e-6))

        wkv = torch.stack(outputs, dim=1)  # (B, T, D)
        out = self.output(r * wkv)

        new_state = {
            "last_x": x[:, -1:, :].detach(),
            "wkv_num": wkv_num.detach(),
            "wkv_den": wkv_den.detach(),
        }
        return out, new_state

    def _channel_mixing(
        self, x: Tensor, state: dict | None
    ) -> tuple[Tensor, dict]:
        """Gated FFN with squared ReLU."""
        B, T, D = x.shape
        state = state or {}

        last_x = state.get("cm_last_x", torch.zeros(B, 1, D, device=x.device))
        shifted = torch.cat([last_x, x[:, :-1, :]], dim=1)

        xk = x * self.cm_mix_k + shifted * (1 - self.cm_mix_k)
        xr = x * self.cm_mix_r + shifted * (1 - self.cm_mix_r)

        k = torch.relu(self.cm_key(xk)).square()  # Squared ReLU
        v = self.cm_value(k)
        r = torch.sigmoid(self.cm_receptance(xr))

        new_state = {"cm_last_x": x[:, -1:, :].detach()}
        return r * v, new_state

    def forward(
        self, x: Tensor, state: dict | None = None
    ) -> tuple[Tensor, dict]:
        """
        RWKV forward pass.

        Args:
            x: (batch, seq, dim) input.
            state: Recurrent state dict from previous call.

        Returns:
            (output, new_state) where state has constant size.
        """
        state = state or {}
        tm_out, tm_state = self._time_mixing(x, state.get("tm"))
        x = x + tm_out
        cm_out, cm_state = self._channel_mixing(x, state.get("cm"))
        x = x + cm_out
        return x, {"tm": tm_state, "cm": cm_state}
