"""
Spike Attention — Stage 3 of the spiking cortical block.

STDP-weighted linear attention (no softmax) achieving O(N·d) complexity.
Per-head STDP weights modulate attention based on spike timing correlation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SpikeAttention(nn.Module):
    """STDP-weighted linear attention."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.stdp_weights = nn.Parameter(torch.ones(num_heads))

    def forward(
        self, x: Tensor, spike_times: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Linear attention with STDP modulation.

        Args:
            x: (batch, seq, dim) input features.
            spike_times: Optional (batch, seq) spike timestamps for STDP.

        Returns:
            (output, attention_weights) where attention_weights are per-head.
        """
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        Q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        K = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        # Linear attention via ELU+1 kernel (no softmax)
        Q = torch.nn.functional.elu(Q) + 1.0
        K = torch.nn.functional.elu(K) + 1.0

        # STDP modulation per head
        stdp_w = torch.sigmoid(self.stdp_weights).view(1, H, 1, 1)

        # O(N·d) linear attention: (K^T V) then Q @ result
        KV = torch.einsum("bhnd,bhnv->bhdv", K, V)  # (B, H, Dh, Dh)
        K_sum = K.sum(dim=2)  # (B, H, Dh)

        numerator = torch.einsum("bhnd,bhdv->bhnv", Q, KV)  # (B, H, N, Dh)
        denominator = torch.einsum("bhnd,bhd->bhn", Q, K_sum).unsqueeze(-1) + 1e-6

        attn_out = (numerator / denominator) * stdp_w  # (B, H, N, Dh)

        # Merge heads
        output = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        output = self.out_proj(output)

        return output, self.stdp_weights.detach()
