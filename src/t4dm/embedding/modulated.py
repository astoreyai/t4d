"""
State-Dependent Embedding Modulation for World Weaver.

Implements context-aware embedding transformation based on neuromodulator state.
The brain doesn't produce static representations - embeddings should be modulated
by current cognitive state (encoding vs retrieval, arousal level, etc.).

Hinton-inspired: Representations are not fixed vectors but dynamic patterns
that depend on the current computational context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from t4dm.embedding.adapter import EmbeddingAdapter

logger = logging.getLogger(__name__)


class CognitiveMode(Enum):
    """Current cognitive processing mode."""
    ENCODING = "encoding"      # Storing new information
    RETRIEVAL = "retrieval"    # Recalling stored information
    CONSOLIDATION = "consolidation"  # During sleep/rest
    EXPLORATION = "exploration"      # High novelty seeking
    EXPLOITATION = "exploitation"    # Using known patterns


@dataclass
class NeuromodulatorState:
    """
    Current neuromodulator levels affecting embedding modulation.

    Maps to biological neuromodulators:
    - acetylcholine: Encoding/retrieval mode switching
    - dopamine: Salience and reward prediction
    - norepinephrine: Arousal and exploration
    - serotonin: Long-term credit assignment
    """
    acetylcholine: float = 0.5  # 0 = retrieval, 1 = encoding
    dopamine: float = 0.5       # Reward/salience signal
    norepinephrine: float = 0.5  # Arousal/exploration
    serotonin: float = 0.5      # Temporal credit assignment

    @property
    def mode(self) -> CognitiveMode:
        """Infer cognitive mode from neuromodulator levels."""
        if self.acetylcholine > 0.7:
            return CognitiveMode.ENCODING
        if self.acetylcholine < 0.3:
            return CognitiveMode.RETRIEVAL
        if self.norepinephrine > 0.7:
            return CognitiveMode.EXPLORATION
        if self.dopamine > 0.7:
            return CognitiveMode.EXPLOITATION
        return CognitiveMode.CONSOLIDATION

    @classmethod
    def for_encoding(cls) -> NeuromodulatorState:
        """Create state optimized for encoding new memories."""
        return cls(acetylcholine=0.9, dopamine=0.6, norepinephrine=0.4, serotonin=0.5)

    @classmethod
    def for_retrieval(cls) -> NeuromodulatorState:
        """Create state optimized for memory retrieval."""
        return cls(acetylcholine=0.2, dopamine=0.5, norepinephrine=0.3, serotonin=0.5)

    @classmethod
    def for_exploration(cls) -> NeuromodulatorState:
        """Create state for novel exploration."""
        return cls(acetylcholine=0.5, dopamine=0.7, norepinephrine=0.9, serotonin=0.4)


@dataclass
class ModulationConfig:
    """Configuration for embedding modulation."""

    # Dimension gating based on ACh levels
    ach_gate_strength: float = 0.3  # How much ACh affects dimension gating

    # Salience amplification based on DA
    da_amplification: float = 0.2  # How much DA amplifies salient dimensions

    # Exploration noise based on NE
    ne_noise_scale: float = 0.05  # Noise added during exploration

    # Temporal weighting based on 5-HT
    serotonin_decay_weight: float = 0.1  # Weight for temporal bias

    # Sparsity control
    sparsity_ratio: float = 0.1  # Target fraction of active dimensions

    # Cache modulated embeddings
    cache_modulated: bool = False  # Whether to cache (usually False for state-dependent)


class ModulatedEmbeddingAdapter(EmbeddingAdapter):
    """
    Embedding adapter with state-dependent modulation.

    Wraps a base adapter and applies modulation based on current
    neuromodulator state. This creates context-dependent representations
    where the same text produces different vectors depending on cognitive state.

    Example:
        base_adapter = MockEmbeddingAdapter(dimension=1024)
        modulated = ModulatedEmbeddingAdapter(
            adapter=base_adapter,
            config=ModulationConfig()
        )

        # Set encoding mode
        modulated.set_state(NeuromodulatorState.for_encoding())
        encoding_emb = await modulated.embed_query("test")

        # Set retrieval mode
        modulated.set_state(NeuromodulatorState.for_retrieval())
        retrieval_emb = await modulated.embed_query("test")

        # Embeddings will differ based on state
    """

    def __init__(
        self,
        adapter: EmbeddingAdapter,
        config: ModulationConfig | None = None,
    ):
        """
        Initialize modulated adapter.

        Args:
            adapter: Base embedding adapter to wrap
            config: Modulation configuration
        """
        super().__init__(dimension=adapter.dimension)
        self._adapter = adapter
        self._backend = adapter.backend
        self._config = config or ModulationConfig()
        self._state = NeuromodulatorState()

        # Pre-compute dimension masks for different modes
        self._encoding_mask = self._create_mode_mask(CognitiveMode.ENCODING)
        self._retrieval_mask = self._create_mode_mask(CognitiveMode.RETRIEVAL)

        # Salience projection (which dimensions are "important")
        # In a full implementation, this would be learned
        np.random.seed(42)
        self._salience_weights = np.abs(np.random.randn(self._dimension)).astype(np.float32)
        self._salience_weights /= self._salience_weights.sum()

    def _create_mode_mask(self, mode: CognitiveMode) -> np.ndarray:
        """
        Create dimension mask for a cognitive mode.

        Different modes emphasize different dimensions.
        This is a simplified model - in reality, this would be learned.
        """
        np.random.seed(hash(mode.value) % (2**32))
        mask = np.random.rand(self._dimension).astype(np.float32)

        # Different modes have different sparsity
        if mode == CognitiveMode.ENCODING:
            # Encoding uses more dimensions for discrimination
            threshold = np.percentile(mask, 30)
        elif mode == CognitiveMode.RETRIEVAL:
            # Retrieval focuses on most salient dimensions
            threshold = np.percentile(mask, 50)
        else:
            threshold = np.percentile(mask, 40)

        mask = (mask > threshold).astype(np.float32)
        return mask

    @property
    def state(self) -> NeuromodulatorState:
        """Get current neuromodulator state."""
        return self._state

    def set_state(self, state: NeuromodulatorState) -> None:
        """Set neuromodulator state for modulation."""
        self._state = state
        logger.debug(f"Modulation state set to mode: {state.mode.value}")

    def _apply_modulation(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply state-dependent modulation to embedding.

        Modulation steps:
        1. Gate dimensions based on ACh (encoding vs retrieval)
        2. Amplify salient dimensions based on DA
        3. Add exploration noise based on NE
        4. Apply temporal bias based on 5-HT
        5. Re-normalize
        """
        modulated = embedding.copy()

        # 1. ACh-based dimension gating
        if self._state.mode == CognitiveMode.ENCODING:
            gate = self._encoding_mask
        elif self._state.mode == CognitiveMode.RETRIEVAL:
            gate = self._retrieval_mask
        else:
            gate = np.ones(self._dimension, dtype=np.float32)

        # Blend gate based on ACh strength
        gate_blend = (1 - self._config.ach_gate_strength) + \
                     self._config.ach_gate_strength * gate
        modulated *= gate_blend

        # 2. DA-based salience amplification
        if self._state.dopamine > 0.5:
            da_boost = 1.0 + self._config.da_amplification * (self._state.dopamine - 0.5) * 2
            modulated *= (1 + self._salience_weights * (da_boost - 1))

        # 3. NE-based exploration noise
        if self._state.norepinephrine > 0.6:
            noise_scale = self._config.ne_noise_scale * (self._state.norepinephrine - 0.6) * 2.5
            noise = np.random.randn(self._dimension).astype(np.float32) * noise_scale
            modulated += noise

        # 4. Sparsification (soft)
        if self._config.sparsity_ratio < 1.0:
            k = int(self._dimension * self._config.sparsity_ratio)
            threshold = np.partition(np.abs(modulated), -k)[-k]
            modulated *= (np.abs(modulated) >= threshold * 0.5).astype(np.float32)

        # 5. Re-normalize
        norm = np.linalg.norm(modulated)
        if norm > 1e-8:
            modulated /= norm

        return modulated

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed query with state-dependent modulation.

        Args:
            query: Query text to embed

        Returns:
            Modulated embedding vector
        """
        import time

        start = time.perf_counter()

        # Get base embedding
        base_embedding = await self._adapter.embed_query(query)
        base_np = np.array(base_embedding, dtype=np.float32)

        # Apply modulation
        modulated = self._apply_modulation(base_np)

        latency = (time.perf_counter() - start) * 1000
        self._record_query(latency, cache_hit=False)

        return modulated.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts with state-dependent modulation.

        Args:
            texts: List of texts to embed

        Returns:
            List of modulated embedding vectors
        """
        import time

        if not texts:
            return []

        start = time.perf_counter()

        # Get base embeddings
        base_embeddings = await self._adapter.embed(texts)

        # Apply modulation to each
        modulated = []
        for emb in base_embeddings:
            emb_np = np.array(emb, dtype=np.float32)
            mod_emb = self._apply_modulation(emb_np)
            modulated.append(mod_emb.tolist())

        latency = (time.perf_counter() - start) * 1000
        self._record_documents(len(texts), latency)

        return modulated

    async def embed_query_unmodulated(self, query: str) -> list[float]:
        """
        Get base embedding without modulation.

        Useful for caching the base embedding separately.
        """
        return await self._adapter.embed_query(query)

    def get_modulation_stats(self) -> dict:
        """Get statistics about modulation."""
        return {
            "current_mode": self._state.mode.value,
            "acetylcholine": self._state.acetylcholine,
            "dopamine": self._state.dopamine,
            "norepinephrine": self._state.norepinephrine,
            "serotonin": self._state.serotonin,
            "config": {
                "ach_gate_strength": self._config.ach_gate_strength,
                "da_amplification": self._config.da_amplification,
                "ne_noise_scale": self._config.ne_noise_scale,
                "sparsity_ratio": self._config.sparsity_ratio,
            },
            "stats": self._stats.to_dict(),
        }


def create_modulated_adapter(
    base_adapter: EmbeddingAdapter,
    initial_state: NeuromodulatorState | None = None,
    config: ModulationConfig | None = None,
) -> ModulatedEmbeddingAdapter:
    """
    Factory function to create modulated embedding adapter.

    Args:
        base_adapter: Base adapter to wrap
        initial_state: Initial neuromodulator state
        config: Modulation configuration

    Returns:
        Configured modulated adapter
    """
    adapter = ModulatedEmbeddingAdapter(base_adapter, config)
    if initial_state:
        adapter.set_state(initial_state)
    return adapter


__all__ = [
    "CognitiveMode",
    "ModulatedEmbeddingAdapter",
    "ModulationConfig",
    "NeuromodulatorState",
    "create_modulated_adapter",
]
