"""
Unit tests for World Weaver pattern separation module.

Tests DentateGyrus pattern separator and PatternCompletion.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from t4dm.memory.pattern_separation import (
    DentateGyrus,
    PatternCompletion,
    SeparationResult,
    create_dentate_gyrus,
)


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.call_count = 0

    async def embed_query(self, text: str) -> np.ndarray:
        """Generate deterministic embedding based on text hash."""
        self.call_count += 1
        np.random.seed(hash(text) % (2**32))
        emb = np.random.randn(self.dim).astype(np.float32)
        return emb / np.linalg.norm(emb)


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self, similar_items: list = None):
        self.similar_items = similar_items or []
        self.search_calls = []

    async def search(
        self,
        collection: str,
        vector: list = None,
        query_vector: np.ndarray = None,
        limit: int = 10,
        score_threshold: float = None
    ) -> list[dict]:
        """Return preset similar items."""
        # Support both 'vector' and 'query_vector' parameter names
        vec = vector if vector is not None else query_vector
        self.search_calls.append({
            "collection": collection,
            "query_vector": vec,
            "limit": limit,
            "score_threshold": score_threshold
        })
        return self.similar_items


class TestSeparationResult:
    """Tests for SeparationResult dataclass."""

    def test_creation(self):
        original = np.random.randn(128)
        separated = np.random.randn(128)
        result = SeparationResult(
            original_embedding=original,
            separated_embedding=separated,
            similar_count=5,
            max_similarity=0.85,
            separation_magnitude=0.1
        )
        assert result.similar_count == 5
        assert result.max_similarity == 0.85

    def test_was_separated_true(self):
        result = SeparationResult(
            original_embedding=np.zeros(128),
            separated_embedding=np.zeros(128),
            similar_count=3,
            max_similarity=0.8,
            separation_magnitude=0.05
        )
        assert result.was_separated is True

    def test_was_separated_false(self):
        result = SeparationResult(
            original_embedding=np.zeros(128),
            separated_embedding=np.zeros(128),
            similar_count=0,
            max_similarity=0.0,
            separation_magnitude=0.0
        )
        assert result.was_separated is False


class TestDentateGyrus:
    """Tests for DentateGyrus pattern separator."""

    @pytest.fixture
    def embedding_provider(self):
        return MockEmbeddingProvider(dim=128)

    @pytest.fixture
    def empty_store(self):
        return MockVectorStore(similar_items=[])

    @pytest.fixture
    def store_with_similar(self):
        # Create similar items with vectors
        items = []
        for i in range(3):
            vec = np.random.randn(128).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            items.append({
                "id": f"item_{i}",
                "vector": vec,
                "score": 0.85 - i * 0.03  # 0.85, 0.82, 0.79
            })
        return MockVectorStore(similar_items=items)

    def test_creation_default(self, embedding_provider, empty_store):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=empty_store,
            collection_name="episodes"
        )
        # MEMORY-HIGH-005 FIX: Default lowered from 0.75 to 0.55 for better separation
        assert dg.similarity_threshold == 0.55
        assert dg.max_separation == 0.3
        assert dg.min_separation == 0.05

    def test_creation_custom(self, embedding_provider, empty_store):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=empty_store,
            collection_name="custom_collection",
            similarity_threshold=0.8,
            max_separation=0.5,
            min_separation=0.1
        )
        assert dg.similarity_threshold == 0.8
        assert dg.max_separation == 0.5
        assert dg.collection_name == "custom_collection"

    @pytest.mark.asyncio
    async def test_encode_no_similar(self, embedding_provider, empty_store):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=empty_store,
            collection_name="episodes"
        )

        result = await dg.encode("test content")

        assert result is not None
        assert result.shape == (128,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
        assert len(empty_store.search_calls) == 1

    @pytest.mark.asyncio
    async def test_encode_with_similar_applies_separation(
        self, embedding_provider, store_with_similar
    ):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=store_with_similar,
            collection_name="episodes",
            use_sparse_coding=False  # Disable for easier testing
        )

        result = await dg.encode("test content with similar items")

        assert result is not None
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

        # Check history recorded
        history = dg.get_separation_history()
        assert len(history) == 1
        assert history[0].similar_count == 3
        assert history[0].was_separated

    @pytest.mark.asyncio
    async def test_encode_without_separation(
        self, embedding_provider, store_with_similar
    ):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=store_with_similar,
            collection_name="episodes"
        )

        # Encode without separation
        result = await dg.encode("test content", apply_separation=False)

        assert result is not None
        # Should not have searched store
        assert len(store_with_similar.search_calls) == 0

    @pytest.mark.asyncio
    async def test_encode_separation_produces_different_embedding(
        self, embedding_provider
    ):
        # Create similar items that are actually similar to what we'll encode
        # Use the same seed-based approach as MockEmbeddingProvider
        np.random.seed(hash("test content") % (2**32))
        base_vec = np.random.randn(128).astype(np.float32)
        base_vec = base_vec / np.linalg.norm(base_vec)

        # Create similar vectors (small perturbations of base)
        similar_items = []
        for i in range(3):
            vec = base_vec + np.random.randn(128).astype(np.float32) * 0.1
            vec = vec / np.linalg.norm(vec)
            similar_items.append({
                "id": f"item_{i}",
                "vector": vec,
                "score": 0.95 - i * 0.03  # High similarity
            })

        store = MockVectorStore(similar_items=similar_items)

        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=store,
            collection_name="episodes",
            use_sparse_coding=False
        )

        # Get separated embedding
        separated = await dg.encode("test content")

        # Get unseparated embedding
        unseparated = await dg.encode("test content", apply_separation=False)

        # They should be different (separation was applied)
        diff = np.linalg.norm(separated - unseparated)
        assert diff > 0.01  # Non-trivial difference

    @pytest.mark.asyncio
    async def test_encode_handles_search_failure(self, embedding_provider):
        failing_store = MockVectorStore()
        failing_store.search = AsyncMock(side_effect=Exception("Search failed"))

        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=failing_store,
            collection_name="episodes"
        )

        # Should not raise, returns base embedding
        result = await dg.encode("test content")
        assert result is not None
        assert result.shape == (128,)

    def test_orthogonalize_empty_similar(self, embedding_provider, empty_store):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=empty_store,
            collection_name="episodes"
        )

        target = np.random.randn(128).astype(np.float32)
        result = dg._orthogonalize(target, [])

        assert np.allclose(result, target)

    def test_orthogonalize_with_similar_reduces_similarity(
        self, embedding_provider, empty_store
    ):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=empty_store,
            collection_name="episodes",
            max_separation=0.5
        )

        # Create target that's similar to one direction
        similar_vec = np.array([1.0] + [0.0] * 127, dtype=np.float32)
        target = np.array([0.9, 0.1] + [0.0] * 126, dtype=np.float32)
        target = target / np.linalg.norm(target)

        similar_items = [{"vector": similar_vec, "score": 0.9}]

        result = dg._orthogonalize(target, similar_items)

        # Result should have reduced component in similar direction
        original_sim = np.dot(target, similar_vec)
        result_sim = np.dot(result / np.linalg.norm(result), similar_vec)

        assert result_sim < original_sim

    def test_sparsify_hard(self, embedding_provider, empty_store):
        """Test hard thresholding returns exactly k non-zeros."""
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=empty_store,
            collection_name="episodes",
            sparsity_ratio=0.1
        )

        # Dense embedding
        dense = np.random.randn(128).astype(np.float32)

        # Use hard thresholding
        sparse = dg._sparsify(dense, use_soft_threshold=False)

        # Should have exactly 10% non-zero with hard thresholding
        non_zero = np.count_nonzero(sparse)
        expected = int(128 * 0.1)
        assert non_zero == expected

    def test_sparsify_soft(self, embedding_provider, empty_store):
        """Test soft thresholding preserves gradient information."""
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=empty_store,
            collection_name="episodes",
            sparsity_ratio=0.1
        )

        # Dense embedding
        np.random.seed(42)  # Reproducible
        dense = np.random.randn(128).astype(np.float32)

        # Default uses soft thresholding
        sparse = dg._sparsify(dense, use_soft_threshold=True)

        # Soft thresholding may have slightly more non-zeros due to shrinkage
        # (values near threshold are shrunk toward zero rather than hard-zeroed)
        non_zero = np.count_nonzero(sparse)
        expected = int(128 * 0.1)
        # Allow some flexibility for soft thresholding behavior
        assert expected - 2 <= non_zero <= expected + 5

    def test_compute_separation(self, embedding_provider, empty_store):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=empty_store,
            collection_name="episodes"
        )

        # Identical embeddings
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert dg.compute_separation(a, b) < 1e-6

        # Orthogonal embeddings
        c = np.array([0.0, 1.0, 0.0])
        assert abs(dg.compute_separation(a, c) - 1.0) < 1e-6

        # Opposite embeddings
        d = np.array([-1.0, 0.0, 0.0])
        assert abs(dg.compute_separation(a, d) - 2.0) < 1e-6

    @pytest.mark.asyncio
    async def test_get_separation_history(
        self, embedding_provider, store_with_similar
    ):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=store_with_similar,
            collection_name="episodes"
        )

        await dg.encode("content 1")
        await dg.encode("content 2")
        await dg.encode("content 3")

        history = dg.get_separation_history()
        assert len(history) == 3

        # Test limit
        limited = dg.get_separation_history(limit=2)
        assert len(limited) == 2

    @pytest.mark.asyncio
    async def test_get_separation_history_only_separated(
        self, embedding_provider
    ):
        # Mix of similar and empty stores
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=MockVectorStore([]),  # No similar items
            collection_name="episodes"
        )

        await dg.encode("content 1")

        # Switch to store with similar items
        dg.vector_store = MockVectorStore([
            {"vector": np.random.randn(128), "score": 0.9}
        ])

        await dg.encode("content 2")

        # Only one had separation
        separated = dg.get_separation_history(only_separated=True)
        assert len(separated) == 1

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, embedding_provider, empty_store):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=empty_store,
            collection_name="episodes"
        )

        stats = dg.get_stats()
        assert stats["total_encodings"] == 0
        assert stats["separation_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats(self, embedding_provider, store_with_similar):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=store_with_similar,
            collection_name="episodes"
        )

        await dg.encode("content 1")
        await dg.encode("content 2")

        stats = dg.get_stats()
        assert stats["total_encodings"] == 2
        assert stats["separations_applied"] == 2
        assert stats["separation_rate"] == 1.0
        assert stats["avg_similar_count"] == 3.0

    @pytest.mark.asyncio
    async def test_clear_history(self, embedding_provider, store_with_similar):
        dg = DentateGyrus(
            embedding_provider=embedding_provider,
            vector_store=store_with_similar,
            collection_name="episodes"
        )

        await dg.encode("content")
        assert len(dg.get_separation_history()) == 1

        dg.clear_history()
        assert len(dg.get_separation_history()) == 0


class TestPatternCompletion:
    """Tests for PatternCompletion attractor network."""

    @pytest.fixture
    def completer(self):
        return PatternCompletion(
            embedding_dim=128,
            num_attractors=10,
            convergence_threshold=0.01,
            max_iterations=10
        )

    def test_creation(self):
        pc = PatternCompletion()
        assert pc.embedding_dim == 1024
        assert pc.num_attractors == 100
        assert pc.max_iterations == 10

    def test_creation_custom(self):
        pc = PatternCompletion(
            embedding_dim=256,
            num_attractors=50,
            convergence_threshold=0.001,
            max_iterations=20
        )
        assert pc.embedding_dim == 256
        assert pc.num_attractors == 50

    def test_add_attractor(self, completer):
        pattern = np.random.randn(128)
        completer.add_attractor(pattern)

        assert completer.get_attractor_count() == 1

    def test_add_attractor_normalizes(self, completer):
        pattern = np.random.randn(128) * 10  # Not unit norm

        completer.add_attractor(pattern)

        # Get the stored attractor
        nearest, _ = completer.find_nearest_attractor(pattern)
        assert abs(np.linalg.norm(nearest) - 1.0) < 1e-6

    def test_add_attractor_respects_limit(self, completer):
        # Add more than num_attractors
        for i in range(15):
            completer.add_attractor(np.random.randn(128))

        assert completer.get_attractor_count() == 10

    def test_complete_empty_returns_input(self, completer):
        partial = np.random.randn(128)

        completed, iterations = completer.complete(partial)

        assert np.allclose(completed, partial)
        assert iterations == 0

    def test_complete_converges_to_attractor(self, completer):
        # Add single attractor
        attractor = np.random.randn(128)
        attractor = attractor / np.linalg.norm(attractor)
        completer.add_attractor(attractor)

        # Create noisy version
        noise = np.random.randn(128) * 0.1
        noisy = attractor + noise
        noisy = noisy / np.linalg.norm(noisy)

        completed, iterations = completer.complete(noisy)

        # Should converge toward attractor
        similarity = np.dot(completed, attractor)
        assert similarity > 0.9

    def test_complete_respects_max_iterations(self, completer):
        # Add many random attractors (no clear convergence)
        for _ in range(10):
            completer.add_attractor(np.random.randn(128))

        partial = np.random.randn(128)

        _, iterations = completer.complete(partial)

        assert iterations <= completer.max_iterations

    def test_complete_with_mask(self, completer):
        # Add attractor
        attractor = np.zeros(128)
        attractor[0] = 1.0  # Unit vector in first dimension
        completer.add_attractor(attractor)

        # Partial pattern with known first dimension
        partial = np.zeros(128)
        partial[0] = 0.5
        partial[1] = 0.5

        # Mask: first dimension is known
        mask = np.zeros(128, dtype=bool)
        mask[0] = True

        completed, _ = completer.complete(partial, mask=mask)

        # First dimension should be preserved (approximately)
        # Note: normalization may slightly change it
        assert abs(completed[0]) > 0.1

    def test_find_nearest_attractor_empty(self, completer):
        pattern = np.random.randn(128)

        nearest, sim = completer.find_nearest_attractor(pattern)

        assert nearest is None
        assert sim == 0.0

    def test_find_nearest_attractor(self, completer):
        # Add several attractors
        attractors = [
            np.array([1.0] + [0.0] * 127),
            np.array([0.0, 1.0] + [0.0] * 126),
            np.array([0.0, 0.0, 1.0] + [0.0] * 125)
        ]

        for a in attractors:
            completer.add_attractor(a)

        # Query close to first attractor
        query = np.array([0.9, 0.1] + [0.0] * 126)

        nearest, sim = completer.find_nearest_attractor(query)

        # Should find first attractor
        assert np.dot(nearest, attractors[0]) > 0.99
        assert sim > 0.9

    def test_clear(self, completer):
        completer.add_attractor(np.random.randn(128))
        assert completer.get_attractor_count() == 1

        completer.clear()
        assert completer.get_attractor_count() == 0


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_dentate_gyrus(self):
        embedding = MockEmbeddingProvider()
        store = MockVectorStore()

        dg = create_dentate_gyrus(
            embedding_provider=embedding,
            vector_store=store,
            collection_name="test_collection",
            similarity_threshold=0.9
        )

        assert isinstance(dg, DentateGyrus)
        assert dg.collection_name == "test_collection"
        assert dg.similarity_threshold == 0.9


class TestIntegration:
    """Integration tests for pattern separation."""

    @pytest.mark.asyncio
    async def test_separation_then_completion(self):
        """Test that separated patterns can be completed."""
        embedding = MockEmbeddingProvider(dim=128)
        store = MockVectorStore([])

        dg = DentateGyrus(
            embedding_provider=embedding,
            vector_store=store,
            collection_name="episodes",
            use_sparse_coding=False
        )

        completer = PatternCompletion(
            embedding_dim=128,
            num_attractors=10
        )

        # Encode and add as attractor
        emb = await dg.encode("test pattern")
        completer.add_attractor(emb)

        # Create noisy version
        noisy = emb + np.random.randn(128) * 0.1
        noisy = noisy / np.linalg.norm(noisy)

        # Complete
        completed, _ = completer.complete(noisy)

        # Should recover original
        similarity = np.dot(completed, emb)
        assert similarity > 0.9

    @pytest.mark.asyncio
    async def test_multiple_similar_encodings_diverge(self):
        """Test that similar inputs produce divergent encodings."""
        embedding = MockEmbeddingProvider(dim=128)

        # Store with similar items
        np.random.seed(42)
        similar_vec = np.random.randn(128).astype(np.float32)
        similar_vec = similar_vec / np.linalg.norm(similar_vec)
        store = MockVectorStore([{"vector": similar_vec, "score": 0.9}])

        dg = DentateGyrus(
            embedding_provider=embedding,
            vector_store=store,
            collection_name="episodes",
            use_sparse_coding=False
        )

        # Encode same content multiple times (each should diverge)
        embs = []
        for i in range(3):
            # Reset seed for same base embedding
            embedding = MockEmbeddingProvider(dim=128)
            dg.embedding = embedding
            emb = await dg.encode(f"similar content {i}")
            embs.append(emb)

        # Due to random perturbation, they should differ
        # (though the difference may be small)
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                diff = np.linalg.norm(embs[i] - embs[j])
                # At least some difference
                assert diff > 0
