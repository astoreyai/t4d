"""
Semantic Mock Embedding Adapter for World Weaver.

Provides a mock embedding adapter that produces semantically meaningful
embeddings for testing. Unlike the basic MockEmbeddingAdapter which uses
hash-based random embeddings, this adapter creates embeddings that have
semantic relationships - similar texts produce similar embeddings.

Hinton-inspired: Even mock representations should capture semantic structure.
The brain doesn't process random patterns; it processes meaningful ones.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from t4dm.embedding.adapter import EmbeddingAdapter, EmbeddingBackend

logger = logging.getLogger(__name__)


# Semantic concept clusters for testing
CONCEPT_CLUSTERS = {
    "programming": ["code", "function", "class", "variable", "method", "api", "debug", "test"],
    "memory": ["remember", "forget", "recall", "store", "retrieve", "encode", "consolidate"],
    "learning": ["learn", "train", "model", "gradient", "update", "adapt", "optimize"],
    "emotion": ["happy", "sad", "angry", "fear", "joy", "love", "hate", "emotion"],
    "time": ["past", "present", "future", "yesterday", "today", "tomorrow", "time", "date"],
    "science": ["hypothesis", "experiment", "data", "analysis", "research", "study", "theory"],
    "neural": ["neuron", "synapse", "brain", "cortex", "hippocampus", "activation", "plasticity"],
    "action": ["do", "make", "create", "build", "destroy", "run", "walk", "move"],
    "question": ["what", "why", "how", "when", "where", "who", "which"],
    "positive": ["good", "great", "excellent", "wonderful", "best", "success", "win"],
    "negative": ["bad", "poor", "terrible", "worst", "fail", "loss", "error"],
}


@dataclass
class SemanticConfig:
    """Configuration for semantic mock embeddings."""

    dimension: int = 128
    concept_weight: float = 0.6  # Weight for concept-based components
    noise_scale: float = 0.1    # Random noise for uniqueness
    positional_weight: float = 0.2  # Weight for word position info
    length_weight: float = 0.1  # Weight for text length info


class SemanticMockAdapter(EmbeddingAdapter):
    """
    Mock embedding adapter with semantic awareness.

    Produces embeddings where:
    - Semantically similar texts have high cosine similarity
    - Different texts have different embeddings (not random)
    - Embedding structure is deterministic and reproducible

    This enables meaningful testing of similarity-based retrieval,
    clustering, and other semantic operations.

    Example:
        adapter = SemanticMockAdapter(dimension=128)

        # Similar texts get similar embeddings
        emb1 = await adapter.embed_query("I love programming")
        emb2 = await adapter.embed_query("I enjoy coding")
        sim = cosine_similarity(emb1, emb2)  # High similarity

        # Different topics get different embeddings
        emb3 = await adapter.embed_query("The weather is nice")
        sim2 = cosine_similarity(emb1, emb3)  # Lower similarity
    """

    def __init__(
        self,
        config: SemanticConfig | None = None,
        dimension: int = 128,
        seed: int = 42,
    ):
        """
        Initialize semantic mock adapter.

        Args:
            config: Semantic configuration (overrides dimension if provided)
            dimension: Embedding dimension (if config not provided)
            seed: Random seed for reproducibility
        """
        self._config = config or SemanticConfig(dimension=dimension)
        super().__init__(dimension=self._config.dimension)
        self._backend = EmbeddingBackend.MOCK
        self._seed = seed

        # Pre-compute concept vectors
        self._concept_vectors = self._init_concept_vectors()

        # Cache for embeddings
        self._cache: dict[str, np.ndarray] = {}

    def _init_concept_vectors(self) -> dict[str, np.ndarray]:
        """Initialize deterministic concept vectors."""
        np.random.seed(self._seed)
        concept_vectors = {}

        # Create orthogonal-ish base vectors for each concept cluster
        n_concepts = len(CONCEPT_CLUSTERS)
        dim = self._config.dimension

        # Use fraction of dimensions for each concept
        dims_per_concept = max(4, dim // (n_concepts + 2))

        for i, (concept, keywords) in enumerate(CONCEPT_CLUSTERS.items()):
            # Create base vector for this concept
            base = np.zeros(dim, dtype=np.float32)

            # Activate specific dimensions for this concept
            start_idx = (i * dims_per_concept) % (dim - dims_per_concept)
            end_idx = start_idx + dims_per_concept

            # Use sinusoidal pattern for smooth concept encoding
            angles = np.linspace(0, 2 * np.pi, dims_per_concept)
            base[start_idx:end_idx] = np.sin(angles + i * np.pi / 4)

            # Normalize
            base = base / (np.linalg.norm(base) + 1e-8)
            concept_vectors[concept] = base

            # Also create vectors for each keyword (slight variations)
            for j, keyword in enumerate(keywords):
                keyword_vec = base.copy()
                # Add small keyword-specific variation
                np.random.seed(hash(keyword) % (2**32))
                noise = np.random.randn(dim).astype(np.float32) * 0.1
                keyword_vec = keyword_vec + noise
                keyword_vec = keyword_vec / (np.linalg.norm(keyword_vec) + 1e-8)
                concept_vectors[keyword] = keyword_vec

        return concept_vectors

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Lowercase and split on non-word characters
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _get_concept_embedding(self, text: str) -> np.ndarray:
        """Get concept-based embedding component."""
        tokens = self._tokenize(text)
        dim = self._config.dimension

        # Accumulate concept vectors for matching tokens
        concept_sum = np.zeros(dim, dtype=np.float32)
        matches = 0

        for token in tokens:
            if token in self._concept_vectors:
                concept_sum += self._concept_vectors[token]
                matches += 1

        if matches > 0:
            concept_sum = concept_sum / matches

        return concept_sum

    def _get_positional_embedding(self, text: str) -> np.ndarray:
        """Get positional embedding component based on word positions."""
        tokens = self._tokenize(text)
        dim = self._config.dimension

        if not tokens:
            return np.zeros(dim, dtype=np.float32)

        # Create positional encoding
        positional = np.zeros(dim, dtype=np.float32)

        for i, token in enumerate(tokens[:dim]):
            # Each token contributes based on position
            position_factor = (i + 1) / len(tokens)
            # Hash token to get consistent contribution
            np.random.seed(hash(token) % (2**32))
            token_component = np.random.randn(dim).astype(np.float32) * position_factor
            positional += token_component * 0.1

        positional = positional / (np.linalg.norm(positional) + 1e-8)
        return positional

    def _get_length_embedding(self, text: str) -> np.ndarray:
        """Get length-based embedding component."""
        dim = self._config.dimension
        length = len(text)

        # Encode length in first few dimensions
        length_vec = np.zeros(dim, dtype=np.float32)
        length_vec[0] = np.log1p(length) / 10.0  # Log-scaled length
        length_vec[1] = len(text.split()) / 100.0  # Word count
        length_vec[2] = len(set(self._tokenize(text))) / 50.0  # Unique tokens

        return length_vec

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute semantic embedding for text."""
        # Check cache
        if text in self._cache:
            return self._cache[text]

        dim = self._config.dimension

        # Combine embedding components
        concept = self._get_concept_embedding(text) * self._config.concept_weight
        positional = self._get_positional_embedding(text) * self._config.positional_weight
        length = self._get_length_embedding(text) * self._config.length_weight

        # Add noise for uniqueness
        np.random.seed(hash(text) % (2**32))
        noise = np.random.randn(dim).astype(np.float32) * self._config.noise_scale

        # Combine
        embedding = concept + positional + length + noise

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        # Cache result
        self._cache[text] = embedding

        return embedding

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed query with semantic awareness.

        Args:
            query: Query text

        Returns:
            Semantically meaningful embedding
        """
        self._record_query(0.1, cache_hit=query in self._cache)
        embedding = self._compute_embedding(query)
        return embedding.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts with semantic awareness.

        Args:
            texts: List of texts

        Returns:
            List of semantically meaningful embeddings
        """
        if not texts:
            return []

        self._record_documents(len(texts), 0.1 * len(texts))
        return [self._compute_embedding(text).tolist() for text in texts]

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()

    def get_similar_concepts(self, text: str, top_k: int = 5) -> list[str]:
        """
        Get most similar concept clusters for a text.

        Args:
            text: Input text
            top_k: Number of concepts to return

        Returns:
            List of concept names sorted by similarity
        """
        tokens = set(self._tokenize(text))

        # Score each concept by overlap
        scores = {}
        for concept, keywords in CONCEPT_CLUSTERS.items():
            overlap = len(tokens & set(keywords))
            if overlap > 0:
                scores[concept] = overlap

        sorted_concepts = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        return sorted_concepts[:top_k]


# Factory function
def create_semantic_mock(
    dimension: int = 128,
    seed: int = 42,
) -> SemanticMockAdapter:
    """
    Create semantic mock adapter.

    Args:
        dimension: Embedding dimension
        seed: Random seed

    Returns:
        Configured semantic mock adapter
    """
    return SemanticMockAdapter(dimension=dimension, seed=seed)


__all__ = [
    "CONCEPT_CLUSTERS",
    "SemanticConfig",
    "SemanticMockAdapter",
    "create_semantic_mock",
]
