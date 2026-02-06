"""Qwen 2.5-3B + QLoRA + Spiking Cortical Adapter integration."""

from t4dm.qwen.alignment import (
    AlignmentConfig,
    AlignmentTrainer,
    EmbeddingAlignment,
)

__all__ = [
    # W2-03: Learned Embedding Alignment (Graves)
    "AlignmentConfig",
    "AlignmentTrainer",
    "EmbeddingAlignment",
]
