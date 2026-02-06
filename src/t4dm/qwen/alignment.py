"""
Learned Embedding Alignment (W2-03).

Learnable alignment layer between Qwen hidden states and BGE-M3 embedding space.
Addresses the gap between frozen LLM features and memory retrieval needs.

Evidence Base: Graves (2014) "Neural Turing Machines"

Key Insight:
    The frozen Qwen hidden states may not directly align with the retrieval
    embedding space (BGE-M3). This learnable layer projects Qwen features
    into a compatible space while learning per-dimension importance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class AlignmentConfig:
    """Configuration for embedding alignment.

    Attributes:
        qwen_dim: Qwen hidden state dimension (default 2048).
        bge_dim: BGE-M3 embedding dimension (default 1024).
        hidden_dim: Intermediate projection dimension (default 1536).
        dropout: Dropout rate (default 0.1).
    """

    qwen_dim: int = 2048
    bge_dim: int = 1024
    hidden_dim: int = 1536
    dropout: float = 0.1


class EmbeddingAlignment(nn.Module):
    """Learnable alignment between Qwen hidden states and BGE-M3 space.

    Projects Qwen layer 18 hidden states (2048-dim) to BGE-M3 compatible
    embeddings (1024-dim) with learnable per-dimension importance weights.

    Architecture:
        1. Mean pooling over sequence
        2. Linear projection with LayerNorm + GELU + Dropout
        3. Output projection to BGE dimension
        4. Per-dimension importance scaling
        5. L2 normalization

    Example:
        >>> alignment = EmbeddingAlignment(qwen_dim=2048, bge_dim=1024)
        >>> qwen_hidden = torch.randn(1, 512, 2048)  # [batch, seq, hidden]
        >>> aligned = alignment(qwen_hidden)  # [batch, 1024]
    """

    def __init__(
        self,
        qwen_dim: int = 2048,
        bge_dim: int = 1024,
        hidden_dim: int = 1536,
        dropout: float = 0.1,
    ):
        """Initialize embedding alignment.

        Args:
            qwen_dim: Qwen hidden state dimension.
            bge_dim: Target BGE-M3 embedding dimension.
            hidden_dim: Intermediate projection dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        self.qwen_dim = qwen_dim
        self.bge_dim = bge_dim

        # Projection network
        self.projection = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bge_dim),
        )

        # Learnable importance weights per dimension
        self.dimension_importance = nn.Parameter(torch.ones(bge_dim))

    def forward(self, qwen_hidden: torch.Tensor) -> torch.Tensor:
        """Align Qwen hidden states to BGE-M3 space.

        Args:
            qwen_hidden: [batch, seq, qwen_dim] from Qwen layer 18.

        Returns:
            aligned: [batch, bge_dim] normalized aligned embedding.
        """
        # Pool over sequence dimension (mean pooling)
        pooled = qwen_hidden.mean(dim=1)  # [batch, qwen_dim]

        # Project to BGE space
        projected = self.projection(pooled)  # [batch, bge_dim]

        # Apply learned dimension importance
        aligned = projected * self.dimension_importance

        # L2 normalize
        aligned = F.normalize(aligned, p=2, dim=-1)

        return aligned


class AlignmentTrainer:
    """Train alignment layer to match Qwen→retrieval with BGE→retrieval.

    Uses MSE loss between aligned Qwen embeddings and BGE embeddings
    for the same text, encouraging the alignment to produce compatible
    representations.

    Example:
        >>> trainer = AlignmentTrainer(alignment, bge_model)
        >>> loss = trainer.train_step(qwen_hidden, texts)
    """

    def __init__(
        self,
        alignment: EmbeddingAlignment,
        bge_model: Any,
        lr: float = 1e-4,
    ):
        """Initialize alignment trainer.

        Args:
            alignment: EmbeddingAlignment module to train.
            bge_model: BGE-M3 model with encode() method.
            lr: Learning rate.
        """
        self.alignment = alignment
        self.bge_model = bge_model
        self.optimizer = torch.optim.AdamW(alignment.parameters(), lr=lr)

    def train_step(
        self,
        qwen_hidden: torch.Tensor,
        texts: list[str],
    ) -> float:
        """Perform single training step.

        Args:
            qwen_hidden: [batch, seq, qwen_dim] Qwen hidden states.
            texts: List of texts corresponding to each batch item.

        Returns:
            Loss value.
        """
        self.optimizer.zero_grad()

        # Get aligned embedding from Qwen
        aligned = self.alignment(qwen_hidden)

        # Get BGE embeddings for texts
        bge_embeddings = []
        for text in texts:
            emb = self.bge_model.encode(text)
            bge_embeddings.append(emb)

        bge_tensor = torch.tensor(
            np.stack(bge_embeddings),
            dtype=torch.float32,
            device=aligned.device,
        )

        # Normalize BGE embeddings
        bge_tensor = F.normalize(bge_tensor, p=2, dim=-1)

        # MSE loss between aligned and BGE
        loss = F.mse_loss(aligned, bge_tensor)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        qwen_hidden: torch.Tensor,
        texts: list[str],
    ) -> dict:
        """Evaluate alignment quality.

        Args:
            qwen_hidden: [batch, seq, qwen_dim] Qwen hidden states.
            texts: List of texts.

        Returns:
            Dict with metrics (mse, cosine_similarity).
        """
        self.alignment.eval()

        with torch.no_grad():
            aligned = self.alignment(qwen_hidden)

            # Get BGE embeddings
            bge_embeddings = []
            for text in texts:
                emb = self.bge_model.encode(text)
                bge_embeddings.append(emb)

            bge_tensor = torch.tensor(
                np.stack(bge_embeddings),
                dtype=torch.float32,
                device=aligned.device,
            )
            bge_tensor = F.normalize(bge_tensor, p=2, dim=-1)

            # Compute metrics
            mse = F.mse_loss(aligned, bge_tensor).item()
            cos_sim = F.cosine_similarity(aligned, bge_tensor, dim=-1).mean().item()

        self.alignment.train()

        return {
            "mse": mse,
            "cosine_similarity": cos_sim,
        }
