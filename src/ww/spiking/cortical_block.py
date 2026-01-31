"""
Cortical Block — 6-stage spiking cortical block with RMP-SNN residual.

Stages:
  1. Thalamic Gate (ACh-modulated input masking)
  2. LIF Integration (spike generation)
  3. Spike Attention (STDP-weighted linear attention)
  4. Apical Modulation (prediction error + FF goodness)
  5. RWKV Recurrence (O(N) linear recurrence)
  6. LIF Output + residual from stage 2
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from ww.spiking.apical_modulation import ApicalModulation
from ww.spiking.lif import LIFNeuron
from ww.spiking.rwkv_recurrence import RWKVRecurrence
from ww.spiking.spike_attention import SpikeAttention
from ww.spiking.thalamic_gate import ThalamicGate


class CorticalBlock(nn.Module):
    """6-stage spiking cortical block."""

    def __init__(
        self,
        dim: int,
        context_dim: int | None = None,
        num_heads: int = 8,
        alpha: float = 0.9,
        v_thresh: float = 1.0,
    ):
        super().__init__()
        self.thalamic = ThalamicGate(dim, context_dim)
        self.lif_integration = LIFNeuron(dim, alpha=alpha, v_thresh=v_thresh)
        self.spike_attention = SpikeAttention(dim, num_heads)
        self.apical = ApicalModulation(dim, context_dim)
        self.rwkv = RWKVRecurrence(dim)
        self.lif_output = LIFNeuron(dim, alpha=alpha, v_thresh=v_thresh)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        ach: float = 0.5,
        state: dict | None = None,
    ) -> tuple[Tensor, dict, dict[str, Any]]:
        """
        Forward pass through all 6 stages.

        Args:
            x: (batch, seq, dim) input.
            context: Optional top-down context.
            ach: Acetylcholine level for thalamic gate.
            state: Recurrent state dict with keys 'u2', 'rwkv', 'u6'.

        Returns:
            (output, new_state, metrics)
        """
        state = state or {}
        B, T, D = x.shape

        # Stage 1: Thalamic gate
        gated = self.thalamic(x, context, ach)

        # Stage 2: LIF integration (per-timestep)
        spikes_2_list, u2_list = [], []
        u2 = state.get("u2")
        for t in range(T):
            s, u2 = self.lif_integration(gated[:, t, :], u2)
            spikes_2_list.append(s)
            u2_list.append(u2)
        spikes_2 = __import__("torch").stack(spikes_2_list, dim=1)

        # Stage 3: Spike attention
        attn_out, attn_w = self.spike_attention(spikes_2)

        # Stage 4: Apical modulation
        modulated, pe, goodness = self.apical(attn_out, context)

        # Stage 5: RWKV recurrence
        recurrent, rwkv_state = self.rwkv(modulated, state.get("rwkv"))

        # Stage 6: Output LIF + RMP-SNN residual from stage 2
        out_spikes_list = []
        u6 = state.get("u6")
        for t in range(T):
            s, u6 = self.lif_output(recurrent[:, t, :], u6)
            out_spikes_list.append(s)
        out_spikes = __import__("torch").stack(out_spikes_list, dim=1)

        output = self.norm(out_spikes + spikes_2)  # Residual

        new_state = {"u2": u2, "rwkv": rwkv_state, "u6": u6}
        metrics = {"pe": pe, "goodness": goodness, "attn": attn_w}
        return output, new_state, metrics
