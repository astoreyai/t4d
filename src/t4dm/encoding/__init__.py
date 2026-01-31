"""
Bioinspired encoding components for World Weaver.

This module provides neural encoding inspired by biological systems:
- Dendritic neurons: Two-compartment model with context gating
- Sparse encoder: k-WTA sparse coding (2% sparsity target)
- Attractor network: Hopfield-style pattern completion
- FFEncoder: Learnable Forward-Forward encoder (Phase 5 - THE LEARNING GAP FIX)
"""

from t4dm.encoding.attractor import AttractorNetwork
from t4dm.encoding.dendritic import DendriticNeuron, DendriticProcessor
from t4dm.encoding.ff_encoder import (
    FFEncoder,
    FFEncoderConfig,
    FFEncoderState,
    create_ff_encoder,
    get_ff_encoder,
    reset_ff_encoder,
)
from t4dm.encoding.sparse import SparseEncoder, kwta

__all__ = [
    # Attractor
    "AttractorNetwork",
    # Dendritic
    "DendriticNeuron",
    "DendriticProcessor",
    # FF Encoder (Phase 5)
    "FFEncoder",
    "FFEncoderConfig",
    "FFEncoderState",
    "create_ff_encoder",
    "get_ff_encoder",
    "reset_ff_encoder",
    # Sparse
    "SparseEncoder",
    "kwta",
]
