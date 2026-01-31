"""P3-04: Memory projection layers (Qwen hidden ↔ memory dim)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MemoryProjection(nn.Module):
    """Bidirectional projection between Qwen hidden dim and memory dim.

    encoder: qwen_dim → mem_dim  (for T4DX writes)
    decoder: mem_dim → qwen_dim  (for T4DX reads / re-injection)
    """

    def __init__(
        self,
        qwen_dim: int = 2048,
        mem_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(qwen_dim, mem_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mem_dim, mem_dim),
            nn.LayerNorm(mem_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(mem_dim, mem_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mem_dim, qwen_dim),
            nn.LayerNorm(qwen_dim),
        )

    def encode(self, hidden: Tensor) -> Tensor:
        """Project Qwen hidden states to memory space. [B,S,qwen_dim] → [B,S,mem_dim]."""
        return self.encoder(hidden)

    def decode(self, memory: Tensor) -> Tensor:
        """Project memory vectors back to Qwen hidden space. [B,S,mem_dim] → [B,S,qwen_dim]."""
        return self.decoder(memory)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Encode then decode (for reconstruction loss). Returns (encoded, decoded)."""
        encoded = self.encode(hidden)
        decoded = self.decode(encoded)
        return encoded, decoded
