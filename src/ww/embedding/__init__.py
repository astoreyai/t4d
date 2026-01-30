"""Embedding providers for World Weaver."""

from ww.embedding.adapter import (
    BGEM3Adapter,
    CachedEmbeddingAdapter,
    EmbeddingAdapter,
    EmbeddingBackend,
    EmbeddingProvider,
    EmbeddingStats,
    MockEmbeddingAdapter,
    clear_adapters,
    cosine_similarity,
    create_adapter,
    euclidean_distance,
    get_adapter,
    get_mock_adapter,
    normalize_embedding,
    register_adapter,
)
from ww.embedding.bge_m3 import BGEM3Embedding
from ww.embedding.contrastive_trainer import (
    AdapterMode,
    AdapterStats,
    ContrastiveAdapter,
    ContrastiveConfig,
    create_contrastive_adapter,
)
from ww.embedding.ensemble import (
    AdapterWeight,
    EnsembleEmbeddingAdapter,
    EnsembleStrategy,
    create_ensemble_adapter,
)
from ww.embedding.lora_adapter import (
    AdaptedBGEM3Provider,
    LoRAConfig,
    LoRAEmbeddingAdapter,
    LoRAState,
    RetrievalOutcome,
    create_adapted_provider,
    create_lora_adapter,
)
from ww.embedding.modulated import (
    CognitiveMode,
    ModulatedEmbeddingAdapter,
    ModulationConfig,
    NeuromodulatorState,
    create_modulated_adapter,
)
from ww.embedding.semantic_mock import (
    CONCEPT_CLUSTERS,
    SemanticConfig,
    SemanticMockAdapter,
    create_semantic_mock,
)

__all__ = [
    # Core
    "BGEM3Embedding",
    "EmbeddingAdapter",
    "EmbeddingBackend",
    "EmbeddingStats",
    "EmbeddingProvider",
    "BGEM3Adapter",
    "MockEmbeddingAdapter",
    "CachedEmbeddingAdapter",
    "create_adapter",
    "get_mock_adapter",
    "get_adapter",
    "register_adapter",
    "clear_adapters",
    "cosine_similarity",
    "euclidean_distance",
    "normalize_embedding",
    # Modulated embeddings
    "CognitiveMode",
    "NeuromodulatorState",
    "ModulationConfig",
    "ModulatedEmbeddingAdapter",
    "create_modulated_adapter",
    # Ensemble embeddings
    "EnsembleStrategy",
    "AdapterWeight",
    "EnsembleEmbeddingAdapter",
    "create_ensemble_adapter",
    # Semantic mock
    "SemanticConfig",
    "SemanticMockAdapter",
    "create_semantic_mock",
    "CONCEPT_CLUSTERS",
    # LoRA adapter
    "LoRAConfig",
    "LoRAEmbeddingAdapter",
    "LoRAState",
    "RetrievalOutcome",
    "AdaptedBGEM3Provider",
    "create_lora_adapter",
    "create_adapted_provider",
    # Contrastive adapter
    "ContrastiveAdapter",
    "ContrastiveConfig",
    "AdapterStats",
    "AdapterMode",
    "create_contrastive_adapter",
]
