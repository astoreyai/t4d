"""P3-11: QLoRA rank search — grid search over r values."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class RankSearchResult:
    """Result of a single rank configuration."""

    rank: int
    trainable_params: int
    vram_mb: float
    perplexity: float | None = None
    recall_at_10: float | None = None


@dataclass
class RankSearchConfig:
    """Configuration for rank search."""

    ranks: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    eval_steps: int = 100
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


def estimate_lora_params(
    hidden_dim: int,
    num_layers: int,
    rank: int,
    num_targets: int = 2,
) -> int:
    """Estimate trainable LoRA parameters without loading a model.

    Each LoRA adapter adds 2 * hidden_dim * rank params per target module per layer.
    """
    return num_layers * num_targets * 2 * hidden_dim * rank


def estimate_vram_mb(
    base_vram_mb: float,
    lora_params: int,
    bytes_per_param: float = 2.0,  # bf16
) -> float:
    """Estimate total VRAM for a given rank."""
    lora_mb = lora_params * bytes_per_param / (1024 * 1024)
    return base_vram_mb + lora_mb


def run_rank_search(
    model_loader: Any,
    cfg: RankSearchConfig | None = None,
    hidden_dim: int = 2048,
    num_layers: int = 36,
) -> list[RankSearchResult]:
    """Run rank search (estimation-only mode — no actual training).

    For actual training-based search, use with a model_loader callable
    that returns (model, tokenizer) for each rank.
    """
    cfg = cfg or RankSearchConfig()
    results = []

    base_vram = 2000.0  # ~2GB for 4-bit Qwen 3B

    for rank in cfg.ranks:
        params = estimate_lora_params(
            hidden_dim, num_layers, rank, len(cfg.target_modules),
        )
        vram = estimate_vram_mb(base_vram, params)
        results.append(RankSearchResult(
            rank=rank,
            trainable_params=params,
            vram_mb=vram,
        ))
        logger.info(
            "Rank %d: %d params, %.1f MB VRAM (estimated)",
            rank, params, vram,
        )

    return results
