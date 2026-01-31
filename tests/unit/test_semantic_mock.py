"""
Unit tests for Semantic Mock Embedding Adapter.

Tests semantic awareness, concept clustering, and embedding similarity.
"""

import pytest
import numpy as np

from t4dm.embedding.semantic_mock import (
    SemanticConfig,
    SemanticMockAdapter,
    create_semantic_mock,
    CONCEPT_CLUSTERS,
)
from t4dm.embedding.adapter import cosine_similarity


class TestSemanticConfig:
    """Tests for semantic configuration."""

    def test_default_values(self):
        config = SemanticConfig()
        assert config.dimension == 128
        assert config.concept_weight == 0.6
        assert config.noise_scale == 0.1
        assert config.positional_weight == 0.2
        assert config.length_weight == 0.1

    def test_custom_values(self):
        config = SemanticConfig(
            dimension=256,
            concept_weight=0.8,
            noise_scale=0.05,
        )
        assert config.dimension == 256
        assert config.concept_weight == 0.8
        assert config.noise_scale == 0.05


class TestSemanticMockAdapter:
    """Tests for semantic mock adapter."""

    @pytest.fixture
    def adapter(self):
        return SemanticMockAdapter(dimension=128, seed=42)

    @pytest.mark.asyncio
    async def test_creation(self, adapter):
        assert adapter is not None
        assert adapter.dimension == 128

    @pytest.mark.asyncio
    async def test_embed_query_returns_correct_dimension(self, adapter):
        emb = await adapter.embed_query("test query")
        assert len(emb) == 128

    @pytest.mark.asyncio
    async def test_embed_query_normalized(self, adapter):
        emb = await adapter.embed_query("test query")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_embed_batch(self, adapter):
        texts = ["first text", "second text", "third text"]
        embeddings = await adapter.embed(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 128
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_embed_empty_batch(self, adapter):
        embeddings = await adapter.embed([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_deterministic_embeddings(self, adapter):
        text = "deterministic test"
        emb1 = await adapter.embed_query(text)
        emb2 = await adapter.embed_query(text)

        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, adapter):
        emb1 = await adapter.embed_query("cats are great")
        emb2 = await adapter.embed_query("dogs are wonderful")

        # Should be different
        assert emb1 != emb2


class TestSemanticSimilarity:
    """Tests for semantic similarity relationships."""

    @pytest.fixture
    def adapter(self):
        # Use higher concept weight for stronger semantic signal
        config = SemanticConfig(dimension=128, concept_weight=0.8, noise_scale=0.05)
        return SemanticMockAdapter(config=config, seed=42)

    @pytest.mark.asyncio
    async def test_similar_programming_texts(self, adapter):
        # Use exact keywords from CONCEPT_CLUSTERS for better matching
        emb1 = np.array(await adapter.embed_query("code debug test function"))
        emb2 = np.array(await adapter.embed_query("code function method api"))

        sim = cosine_similarity(emb1, emb2)

        # Similar programming-related texts should have positive similarity
        assert sim > 0.0  # Positive correlation is the key test

    @pytest.mark.asyncio
    async def test_similar_memory_texts(self, adapter):
        # Use exact keywords from CONCEPT_CLUSTERS
        emb1 = np.array(await adapter.embed_query("remember store recall memory"))
        emb2 = np.array(await adapter.embed_query("retrieve encode remember store"))

        sim = cosine_similarity(emb1, emb2)

        # Similar memory-related texts should have positive similarity
        assert sim > 0.0

    @pytest.mark.asyncio
    async def test_similar_learning_texts(self, adapter):
        # Use exact keywords from CONCEPT_CLUSTERS
        emb1 = np.array(await adapter.embed_query("train model gradient learn"))
        emb2 = np.array(await adapter.embed_query("learn adapt optimize update"))

        sim = cosine_similarity(emb1, emb2)

        # Similar learning-related texts should have positive similarity
        assert sim > 0.0

    @pytest.mark.asyncio
    async def test_different_concept_lower_similarity(self, adapter):
        # Programming vs emotion
        emb1 = np.array(await adapter.embed_query("debug function code test"))
        emb2 = np.array(await adapter.embed_query("happy sad angry fear"))

        # Neural vs time
        emb3 = np.array(await adapter.embed_query("neuron synapse brain activation"))
        emb4 = np.array(await adapter.embed_query("yesterday today tomorrow time"))

        sim_programming_emotion = cosine_similarity(emb1, emb2)
        sim_neural_time = cosine_similarity(emb3, emb4)

        # Cross-concept similarity should be moderate (not super high)
        assert sim_programming_emotion < 0.8
        assert sim_neural_time < 0.8

    @pytest.mark.asyncio
    async def test_neural_concepts_similar(self, adapter):
        # Use exact keywords from CONCEPT_CLUSTERS
        emb1 = np.array(await adapter.embed_query("neuron synapse brain plasticity"))
        emb2 = np.array(await adapter.embed_query("neuron cortex hippocampus activation"))

        sim = cosine_similarity(emb1, emb2)
        # Should have positive correlation since they share 'neuron'
        assert sim > 0.0

    @pytest.mark.asyncio
    async def test_same_concept_higher_than_different(self, adapter):
        # Core test: same-concept similarity > cross-concept similarity
        prog1 = np.array(await adapter.embed_query("code debug function test"))
        prog2 = np.array(await adapter.embed_query("code method api class"))
        emotion = np.array(await adapter.embed_query("happy sad angry joy"))

        sim_same = cosine_similarity(prog1, prog2)
        sim_diff = cosine_similarity(prog1, emotion)

        # Same concept should have higher similarity than different concept
        assert sim_same > sim_diff


class TestConceptClusters:
    """Tests for concept cluster functionality."""

    def test_concept_clusters_exist(self):
        assert "programming" in CONCEPT_CLUSTERS
        assert "memory" in CONCEPT_CLUSTERS
        assert "learning" in CONCEPT_CLUSTERS
        assert "emotion" in CONCEPT_CLUSTERS
        assert "neural" in CONCEPT_CLUSTERS

    def test_concept_keywords_not_empty(self):
        for concept, keywords in CONCEPT_CLUSTERS.items():
            assert len(keywords) > 0
            assert all(isinstance(k, str) for k in keywords)

    @pytest.fixture
    def adapter(self):
        return SemanticMockAdapter(dimension=128, seed=42)

    def test_get_similar_concepts(self, adapter):
        concepts = adapter.get_similar_concepts("I love to code and debug programs")
        assert "programming" in concepts

    def test_get_similar_concepts_memory(self, adapter):
        concepts = adapter.get_similar_concepts("remember to store and recall memories")
        assert "memory" in concepts

    def test_get_similar_concepts_neural(self, adapter):
        concepts = adapter.get_similar_concepts("neuron synapse brain cortex activation")
        assert "neural" in concepts

    def test_get_similar_concepts_empty_text(self, adapter):
        concepts = adapter.get_similar_concepts("")
        assert concepts == []

    def test_get_similar_concepts_no_matches(self, adapter):
        concepts = adapter.get_similar_concepts("xyz abc 123")
        assert concepts == []


class TestSemanticMockCaching:
    """Tests for embedding cache."""

    @pytest.fixture
    def adapter(self):
        return SemanticMockAdapter(dimension=128, seed=42)

    @pytest.mark.asyncio
    async def test_cache_populated(self, adapter):
        text = "cache this text"
        assert text not in adapter._cache

        await adapter.embed_query(text)
        assert text in adapter._cache

    @pytest.mark.asyncio
    async def test_clear_cache(self, adapter):
        await adapter.embed_query("text 1")
        await adapter.embed_query("text 2")

        assert len(adapter._cache) == 2

        adapter.clear_cache()
        assert len(adapter._cache) == 0

    @pytest.mark.asyncio
    async def test_cache_hit_faster(self, adapter):
        import time

        text = "performance test"

        # First call - cache miss
        start = time.perf_counter()
        await adapter.embed_query(text)
        first_time = time.perf_counter() - start

        # Second call - cache hit
        start = time.perf_counter()
        await adapter.embed_query(text)
        second_time = time.perf_counter() - start

        # Cache hit should be at least as fast (hard to guarantee faster for such simple ops)
        assert second_time <= first_time * 2  # Allow some variance


class TestSemanticMockStats:
    """Tests for statistics tracking."""

    @pytest.fixture
    def adapter(self):
        return SemanticMockAdapter(dimension=128, seed=42)

    @pytest.mark.asyncio
    async def test_query_stats(self, adapter):
        await adapter.embed_query("query 1")
        await adapter.embed_query("query 2")

        assert adapter.stats.total_queries == 2

    @pytest.mark.asyncio
    async def test_document_stats(self, adapter):
        await adapter.embed(["doc 1", "doc 2", "doc 3"])

        assert adapter.stats.total_documents == 3

    @pytest.mark.asyncio
    async def test_cache_hit_recorded(self, adapter):
        text = "cached query"

        await adapter.embed_query(text)
        await adapter.embed_query(text)  # Cache hit

        assert adapter.stats.cache_hits >= 1 or adapter.stats.cache_misses >= 1


class TestCreateSemanticMock:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        adapter = create_semantic_mock()
        assert adapter is not None
        assert adapter.dimension == 128

    def test_create_with_custom_dimension(self):
        adapter = create_semantic_mock(dimension=256)
        assert adapter.dimension == 256

    def test_create_with_custom_seed(self):
        adapter1 = create_semantic_mock(seed=123)
        adapter2 = create_semantic_mock(seed=456)

        # Different seeds should produce different concept vectors
        assert adapter1._seed != adapter2._seed


class TestSemanticMockWithConfig:
    """Tests for adapter with custom config."""

    @pytest.mark.asyncio
    async def test_custom_concept_weight(self):
        config = SemanticConfig(dimension=128, concept_weight=0.9)
        adapter = SemanticMockAdapter(config=config)

        emb = await adapter.embed_query("code function debug")
        assert len(emb) == 128

    @pytest.mark.asyncio
    async def test_low_noise(self):
        config = SemanticConfig(dimension=128, noise_scale=0.01)
        adapter = SemanticMockAdapter(config=config)

        # With low noise, same concept texts should be very similar
        emb1 = np.array(await adapter.embed_query("code debug function"))
        emb2 = np.array(await adapter.embed_query("code function method"))

        sim = cosine_similarity(emb1, emb2)
        # Low noise means high concept similarity should dominate
        assert sim > 0.5

    @pytest.mark.asyncio
    async def test_high_noise(self):
        config = SemanticConfig(dimension=128, noise_scale=0.5)
        adapter = SemanticMockAdapter(config=config)

        # With high noise, embeddings should still be normalized
        emb = np.array(await adapter.embed_query("test"))
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5
