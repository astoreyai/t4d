"""
Temporal Attention with Theta-Gamma Positional Encoding.

Biological Basis:
- Theta oscillations (4-8 Hz): Provide temporal organization for sequences
- Gamma oscillations (30-100 Hz): Bind items within theta cycles (7±2 items)
- Phase precession: Items shift phase relative to theta as learning progresses

Mathematical Formulation:

Theta-Phase Positional Modulation:
    PE_theta(pos, theta_cycle, gamma_slot) =
        PE_base(pos) + sin(gamma_slot / gamma_slots_per_theta) * W_theta

Relative Positional Attention:
    A[i,j] = (q_i @ k_j + q_i @ r_{i-j}) / sqrt(d_k)

References:
- Jensen & Lisman (2005): Hippocampal sequence encoding by theta phase
- Lisman & Jensen (2013): Theta-gamma neural code
- Vaswani et al. (2017): Attention is all you need
- Shaw et al. (2018): Self-attention with relative position representations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemporalAttentionConfig:
    """Configuration for temporal attention.

    Attributes:
        embed_dim: Embedding dimension
        max_sequence_length: Maximum sequence length
        positional_type: Type of positional encoding (sinusoidal, learnable)
        use_theta_modulation: Use theta-gamma phase modulation
        gamma_slots_per_theta: Number of gamma slots per theta cycle (7±2)
        theta_position_weight: Weight for theta modulation
        use_relative_positions: Use relative position embeddings
        max_relative_distance: Maximum relative distance to encode
        temporal_decay_rate: Decay rate for temporal similarity
        num_heads: Number of attention heads
        head_dim: Dimension per head
    """
    embed_dim: int = 1024
    max_sequence_length: int = 512
    positional_type: str = "learnable"
    use_theta_modulation: bool = True
    gamma_slots_per_theta: int = 7
    theta_position_weight: float = 0.3
    use_relative_positions: bool = True
    max_relative_distance: int = 128
    temporal_decay_rate: float = 0.1
    num_heads: int = 8
    head_dim: int = 64


class PositionalEncoding:
    """
    Positional encoding with optional theta-gamma modulation.

    Implements both sinusoidal and learnable positional encodings,
    with optional modulation by theta-gamma oscillatory phase.

    Biological interpretation:
    - Base encoding: Temporal context representation
    - Theta modulation: Sequence boundary marking
    - Gamma slots: Item position within working memory
    """

    def __init__(self, config: TemporalAttentionConfig):
        """
        Initialize positional encoding.

        Args:
            config: Temporal attention configuration
        """
        self.config = config

        # Initialize encodings
        if config.positional_type == "sinusoidal":
            self.encodings = self._init_sinusoidal()
        else:
            self.encodings = np.random.randn(
                config.max_sequence_length,
                config.embed_dim
            ).astype(np.float32) * 0.02

        # Theta modulation weights
        if config.use_theta_modulation:
            self.W_theta = np.random.randn(
                config.embed_dim
            ).astype(np.float32) * 0.01

        logger.debug(
            f"PositionalEncoding: max_len={config.max_sequence_length}, "
            f"type={config.positional_type}"
        )

    def _init_sinusoidal(self) -> np.ndarray:
        """Initialize sinusoidal positional encodings."""
        pe = np.zeros(
            (self.config.max_sequence_length, self.config.embed_dim),
            dtype=np.float32
        )

        position = np.arange(self.config.max_sequence_length)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.config.embed_dim, 2) *
            (-np.log(10000.0) / self.config.embed_dim)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def encode_positions(
        self,
        length: int,
        theta_phase: float | None = None,
        gamma_phases: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Get positional encodings for sequence.

        Args:
            length: Sequence length
            theta_phase: Current theta phase [0, 2π]
            gamma_phases: Gamma phases for each position [length]

        Returns:
            Positional encodings [length, embed_dim]
        """
        if length > self.config.max_sequence_length:
            raise ValueError(
                f"Length {length} exceeds max {self.config.max_sequence_length}"
            )

        # Base encodings
        pe = self.encodings[:length].copy()

        # Apply theta-gamma modulation
        if self.config.use_theta_modulation and theta_phase is not None:
            for i in range(length):
                gamma_slot = i % self.config.gamma_slots_per_theta
                modulation = self.get_theta_modulated_encoding(
                    position=i,
                    theta_cycle=int(i // self.config.gamma_slots_per_theta),
                    gamma_slot=gamma_slot,
                )
                pe[i] = pe[i] + self.config.theta_position_weight * modulation

        return pe

    def get_theta_modulated_encoding(
        self,
        position: int,
        theta_cycle: int,
        gamma_slot: int,
    ) -> np.ndarray:
        """
        Get theta-modulated positional encoding.

        PE_theta = sin(gamma_slot / gamma_slots) * W_theta

        Args:
            position: Absolute position
            theta_cycle: Which theta cycle (0, 1, 2, ...)
            gamma_slot: Slot within theta cycle (0 to gamma_slots-1)

        Returns:
            Modulation vector [embed_dim]
        """
        # Gamma phase within theta cycle
        gamma_phase = (
            2 * np.pi * gamma_slot / self.config.gamma_slots_per_theta
        )

        # Theta phase modulation
        modulation = np.sin(gamma_phase) * self.W_theta

        # Add theta cycle encoding
        cycle_encoding = np.sin(
            2 * np.pi * theta_cycle / 10.0 +
            np.arange(self.config.embed_dim) * 0.01
        ).astype(np.float32)

        return modulation + 0.1 * cycle_encoding


class RelativePositionEmbedding:
    """
    Relative position embeddings for sequence attention.

    Instead of absolute positions, encodes relative distances
    between positions. This allows better generalization.

    Biological interpretation:
    - Hippocampal time cells encode relative temporal distances
    - Compressed representation of temporal lag
    """

    def __init__(
        self,
        max_relative_distance: int,
        embed_dim: int,
    ):
        """
        Initialize relative position embedding.

        Args:
            max_relative_distance: Maximum relative distance to encode
            embed_dim: Embedding dimension
        """
        self.max_relative_distance = max_relative_distance
        self.embed_dim = embed_dim

        # 2 * max_dist + 1 possible relative positions
        n_positions = 2 * max_relative_distance + 1

        # Learnable embeddings
        self.embeddings = np.random.randn(
            n_positions, embed_dim
        ).astype(np.float32) * 0.02

        logger.debug(
            f"RelativePositionEmbedding: max_dist={max_relative_distance}"
        )

    def get_relative_embeddings(self, length: int) -> np.ndarray:
        """
        Get relative position embeddings for all position pairs.

        Args:
            length: Sequence length

        Returns:
            Relative embeddings [length, length, embed_dim]
        """
        # Compute relative positions
        positions = np.arange(length)
        relative_positions = positions[:, np.newaxis] - positions[np.newaxis, :]

        # Clip to valid range
        relative_positions = np.clip(
            relative_positions,
            -self.max_relative_distance,
            self.max_relative_distance,
        )

        # Shift to positive indices
        indices = relative_positions + self.max_relative_distance

        # Gather embeddings
        embeddings = self.embeddings[indices]

        return embeddings

    def get_relative_bias(self, length: int) -> np.ndarray:
        """
        Get relative position bias for attention.

        Returns scalar bias for each position pair.

        Args:
            length: Sequence length

        Returns:
            Bias matrix [length, length]
        """
        embeddings = self.get_relative_embeddings(length)

        # Project to scalar bias
        bias = embeddings.mean(axis=-1)

        return bias


class TemporalAttention:
    """
    Temporal attention with theta-gamma encoding.

    Combines:
    1. Multi-head self-attention
    2. Theta-gamma positional encoding
    3. Relative position embeddings
    4. Temporal decay for recency bias

    Biological interpretation:
    - Theta phase organizes retrieval order
    - Gamma slots bind items within working memory
    - Relative encoding supports sequence learning
    """

    def __init__(self, config: TemporalAttentionConfig | None = None):
        """
        Initialize temporal attention.

        Args:
            config: Temporal attention configuration
        """
        self.config = config or TemporalAttentionConfig()

        # Positional encoding
        self.positional = PositionalEncoding(self.config)

        # Relative position embedding
        if self.config.use_relative_positions:
            self.relative = RelativePositionEmbedding(
                max_relative_distance=self.config.max_relative_distance,
                embed_dim=self.config.head_dim,
            )
        else:
            self.relative = None

        # Attention projections
        self.W_q = np.random.randn(
            self.config.num_heads,
            self.config.head_dim,
            self.config.embed_dim,
        ).astype(np.float32) * 0.02

        self.W_k = np.random.randn(
            self.config.num_heads,
            self.config.head_dim,
            self.config.embed_dim,
        ).astype(np.float32) * 0.02

        self.W_v = np.random.randn(
            self.config.num_heads,
            self.config.head_dim,
            self.config.embed_dim,
        ).astype(np.float32) * 0.02

        # Output projection
        total_dim = self.config.num_heads * self.config.head_dim
        self.W_o = np.random.randn(
            self.config.embed_dim,
            total_dim,
        ).astype(np.float32) * 0.02

        logger.info(
            f"TemporalAttention: {self.config.num_heads} heads, "
            f"gamma_slots={self.config.gamma_slots_per_theta}"
        )

    def attend_sequence(
        self,
        query_seq: np.ndarray,
        memory_seq: np.ndarray,
        positions: np.ndarray | None = None,
        causal: bool = False,
    ) -> np.ndarray:
        """
        Attend over a sequence with temporal encoding.

        Args:
            query_seq: Query sequence [query_len, embed_dim]
            memory_seq: Memory sequence [memory_len, embed_dim]
            positions: Optional position indices
            causal: Apply causal mask

        Returns:
            Attended output [query_len, embed_dim]
        """
        query_seq = np.atleast_2d(query_seq)
        memory_seq = np.atleast_2d(memory_seq)

        query_len = query_seq.shape[0]
        memory_len = memory_seq.shape[0]

        # Add positional encodings
        query_pe = self.positional.encode_positions(query_len)
        memory_pe = self.positional.encode_positions(memory_len)

        query_seq = query_seq + query_pe
        memory_seq = memory_seq + memory_pe

        # Multi-head attention
        head_outputs = []

        for h in range(self.config.num_heads):
            # Project
            Q = query_seq @ self.W_q[h].T   # [query_len, head_dim]
            K = memory_seq @ self.W_k[h].T  # [memory_len, head_dim]
            V = memory_seq @ self.W_v[h].T  # [memory_len, head_dim]

            # Content attention
            scale = np.sqrt(self.config.head_dim)
            attention = (Q @ K.T) / scale  # [query_len, memory_len]

            # Add relative position bias
            if self.relative is not None and query_len == memory_len:
                rel_bias = self.relative.get_relative_bias(query_len)
                attention = attention + rel_bias

            # Temporal decay (recency bias)
            if positions is not None:
                decay = self._compute_temporal_decay(positions, positions)
                attention = attention + np.log(decay + 1e-8)

            # Causal mask
            if causal:
                mask = np.triu(
                    np.ones((query_len, memory_len), dtype=np.float32) * -1e9,
                    k=1
                )
                attention = attention + mask

            # Softmax
            attention = self._softmax(attention)

            # Output
            output = attention @ V  # [query_len, head_dim]
            head_outputs.append(output)

        # Concatenate heads
        concat = np.concatenate(head_outputs, axis=-1)

        # Project to output
        output = concat @ self.W_o.T

        return output

    def encode_temporal_context(
        self,
        items: np.ndarray,
        timestamps: np.ndarray,
        theta_cycles: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Encode items with temporal context.

        Args:
            items: Item embeddings [n_items, embed_dim]
            timestamps: Timestamps for each item [n_items]
            theta_cycles: Optional theta cycle for each item [n_items]

        Returns:
            Temporally-encoded items [n_items, embed_dim]
        """
        items = np.atleast_2d(items)
        n_items = items.shape[0]

        # Compute positions from timestamps
        sorted_indices = np.argsort(timestamps)
        positions = np.zeros(n_items, dtype=np.int32)
        positions[sorted_indices] = np.arange(n_items)

        # Add positional encoding
        for i in range(n_items):
            pos = positions[i]
            if theta_cycles is not None:
                cycle = int(theta_cycles[i])
                gamma_slot = pos % self.config.gamma_slots_per_theta
                pe = self.positional.get_theta_modulated_encoding(
                    position=pos,
                    theta_cycle=cycle,
                    gamma_slot=gamma_slot,
                )
                items[i] = items[i] + self.config.theta_position_weight * pe
            else:
                pe = self.positional.encodings[pos]
                items[i] = items[i] + pe

        return items

    def compute_temporal_similarity(
        self,
        pos1: int,
        pos2: int,
        time1: float,
        time2: float,
    ) -> float:
        """
        Compute temporal similarity between two positions.

        Combines positional and temporal factors.

        Args:
            pos1: First position
            pos2: Second position
            time1: First timestamp
            time2: Second timestamp

        Returns:
            Similarity score
        """
        # Positional similarity (via encoding)
        pe1 = self.positional.encodings[min(pos1, self.config.max_sequence_length - 1)]
        pe2 = self.positional.encodings[min(pos2, self.config.max_sequence_length - 1)]
        pos_sim = np.dot(pe1, pe2) / (
            np.linalg.norm(pe1) * np.linalg.norm(pe2) + 1e-8
        )

        # Temporal decay
        time_diff = abs(time2 - time1)
        time_decay = np.exp(-self.config.temporal_decay_rate * time_diff)

        return float(0.5 * pos_sim + 0.5 * time_decay)

    def _compute_temporal_decay(
        self,
        query_times: np.ndarray,
        memory_times: np.ndarray,
    ) -> np.ndarray:
        """Compute temporal decay matrix."""
        time_diff = np.abs(
            query_times[:, np.newaxis] - memory_times[np.newaxis, :]
        )
        decay = np.exp(-self.config.temporal_decay_rate * time_diff)
        return decay.astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)

    def get_stats(self) -> dict:
        """Get attention statistics."""
        return {
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "max_sequence_length": self.config.max_sequence_length,
            "gamma_slots_per_theta": self.config.gamma_slots_per_theta,
            "use_relative_positions": self.config.use_relative_positions,
            "W_q_norm": float(np.linalg.norm(self.W_q)),
            "W_k_norm": float(np.linalg.norm(self.W_k)),
            "W_v_norm": float(np.linalg.norm(self.W_v)),
        }


__all__ = [
    "TemporalAttentionConfig",
    "PositionalEncoding",
    "RelativePositionEmbedding",
    "TemporalAttention",
]
