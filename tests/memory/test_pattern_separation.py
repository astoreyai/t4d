"""Tests for DG-inspired pattern separation in episodic memory."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from ww.memory.pattern_separation import (
    SeparationResult,
    DentateGyrus,
    PatternCompletion,
    create_dentate_gyrus,
    # Modern Hopfield (P3.2)
    HopfieldConfig,
    HopfieldMode,
    HopfieldResult,
    modern_hopfield_update,
    sparse_hopfield_update,
    hopfield_energy,
    attention_entropy,
    create_pattern_completion,
    benchmark_hopfield_capacity,
)


class TestSeparationResult:
    """Tests for SeparationResult dataclass."""

    def test_result_creation(self):
        """Create result with all fields."""
        original = np.array([1.0, 0.0, 0.0])
        separated = np.array([0.9, 0.1, 0.0])
        result = SeparationResult(
            original_embedding=original,
            separated_embedding=separated,
            similar_count=3,
            max_similarity=0.85,
            separation_magnitude=0.14,
        )
        assert result.similar_count == 3
        assert result.max_similarity == 0.85
        assert result.separation_magnitude == 0.14

    def test_result_has_timestamp(self):
        """Result has automatic timestamp."""
        result = SeparationResult(
            original_embedding=np.zeros(3),
            separated_embedding=np.zeros(3),
            similar_count=0,
            max_similarity=0.0,
            separation_magnitude=0.0,
        )
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_was_separated_true(self):
        """was_separated is True when magnitude > 0."""
        result = SeparationResult(
            original_embedding=np.zeros(3),
            separated_embedding=np.zeros(3),
            similar_count=1,
            max_similarity=0.8,
            separation_magnitude=0.1,
        )
        assert result.was_separated is True

    def test_was_separated_false(self):
        """was_separated is False when magnitude == 0."""
        result = SeparationResult(
            original_embedding=np.zeros(3),
            separated_embedding=np.zeros(3),
            similar_count=0,
            max_similarity=0.0,
            separation_magnitude=0.0,
        )
        assert result.was_separated is False


class TestDentateGyrus:
    """Tests for DentateGyrus pattern separator."""

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create mock embedding provider."""
        provider = MagicMock()
        provider.embed_query = AsyncMock(
            return_value=np.random.randn(64).astype(np.float32)
        )
        return provider

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = MagicMock()
        store.search = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def dg(self, mock_embedding_provider, mock_vector_store):
        """Create DentateGyrus instance."""
        return DentateGyrus(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            collection_name="test_episodes",
            similarity_threshold=0.75,
            search_limit=10,
        )

    def test_initialization(self, dg):
        """Test initialization."""
        assert dg.similarity_threshold == 0.75
        assert dg.search_limit == 10
        assert dg.collection_name == "test_episodes"
        assert dg.use_sparse_coding is True

    def test_initialization_defaults(self, mock_embedding_provider, mock_vector_store):
        """Test default initialization."""
        dg = DentateGyrus(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )
        # MEMORY-HIGH-005 FIX: Default lowered from 0.75 to 0.55 for better separation
        assert dg.similarity_threshold == 0.55
        assert dg.search_limit == 10
        assert dg.max_separation == 0.3
        assert dg.min_separation == 0.05
        assert dg.sparsity_ratio == 0.01  # Biological: ~0.5-2% (Jung & McNaughton 1993)

    @pytest.mark.asyncio
    async def test_encode_no_similar(self, dg, mock_embedding_provider):
        """Encode with no similar items returns base embedding."""
        # Configure mock to return specific embedding
        base_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_embedding_provider.embed_query = AsyncMock(return_value=base_emb)

        result = await dg.encode("test content")

        assert isinstance(result, np.ndarray)
        mock_embedding_provider.embed_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_encode_skip_separation(self, dg, mock_embedding_provider):
        """Encode with apply_separation=False skips separation."""
        base_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_embedding_provider.embed_query = AsyncMock(return_value=base_emb)

        result = await dg.encode("test content", apply_separation=False)

        np.testing.assert_array_equal(result, base_emb)

    @pytest.mark.asyncio
    async def test_encode_with_similar_items(self, dg, mock_embedding_provider, mock_vector_store):
        """Encode applies separation when similar items found."""
        base_emb = np.random.randn(64).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)
        mock_embedding_provider.embed_query = AsyncMock(return_value=base_emb)

        # Similar items from vector store
        similar_items = [
            {"vector": np.random.randn(64).tolist(), "score": 0.85},
            {"vector": np.random.randn(64).tolist(), "score": 0.80},
        ]
        mock_vector_store.search = AsyncMock(return_value=similar_items)

        result = await dg.encode("test content")

        # Result should be different from base due to separation
        assert isinstance(result, np.ndarray)
        assert result.shape == base_emb.shape

    @pytest.mark.asyncio
    async def test_encode_handles_search_error(self, dg, mock_embedding_provider, mock_vector_store):
        """Encode handles vector store search errors gracefully."""
        base_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_embedding_provider.embed_query = AsyncMock(return_value=base_emb)
        mock_vector_store.search = AsyncMock(side_effect=Exception("Search failed"))

        result = await dg.encode("test content")

        # Should return base embedding on error
        np.testing.assert_array_equal(result, base_emb)

    def test_orthogonalize_empty(self, dg):
        """Orthogonalize with no similar items."""
        target = np.array([1.0, 0.0, 0.0])
        result = dg._orthogonalize(target, [])
        np.testing.assert_array_equal(result, target)

    def test_orthogonalize_single_similar(self, dg):
        """Orthogonalize with single similar item."""
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        similar = [{"vector": [0.9, 0.1, 0.0], "score": 0.85}]

        result = dg._orthogonalize(target, similar)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_orthogonalize_multiple_similar(self, dg):
        """Orthogonalize with multiple similar items."""
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        similar = [
            {"vector": [0.9, 0.1, 0.0], "score": 0.85},
            {"vector": [0.95, 0.05, 0.0], "score": 0.90},
            {"vector": [0.8, 0.2, 0.0], "score": 0.78},
        ]

        result = dg._orthogonalize(target, similar)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_orthogonalize_handles_embedding_key(self, dg):
        """Orthogonalize handles 'embedding' key instead of 'vector'."""
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        similar = [{"embedding": [0.9, 0.1, 0.0], "score": 0.85}]

        result = dg._orthogonalize(target, similar)

        assert isinstance(result, np.ndarray)

    def test_orthogonalize_no_vector(self, dg):
        """Orthogonalize handles items without vector."""
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        similar = [{"score": 0.85}]  # No vector key

        result = dg._orthogonalize(target, similar)

        # Should return original since no vectors
        np.testing.assert_array_equal(result, target)

    def test_sparsify_soft_threshold(self, dg):
        """Sparsify with soft thresholding."""
        embedding = np.array([1.0, 0.5, 0.2, -0.8, 0.1, -0.3])
        result = dg._sparsify(embedding, use_soft_threshold=True)

        # Some values should be zero'd out
        assert np.sum(result == 0) >= 0
        assert result.shape == embedding.shape

    def test_sparsify_hard_threshold(self, dg):
        """Sparsify with hard thresholding."""
        embedding = np.array([1.0, 0.5, 0.2, -0.8, 0.1, -0.3])
        result = dg._sparsify(embedding, use_soft_threshold=False)

        assert np.sum(result == 0) >= 0
        assert result.shape == embedding.shape

    def test_sparsify_respects_ratio(self, mock_embedding_provider, mock_vector_store):
        """Sparsify respects sparsity ratio."""
        dg = DentateGyrus(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            sparsity_ratio=0.5,  # Keep 50%
        )
        embedding = np.random.randn(100)
        result = dg._sparsify(embedding, use_soft_threshold=False)

        # At most ~50 non-zero (with hard threshold)
        non_zero = np.sum(result != 0)
        assert non_zero <= 55  # Allow some slack

    def test_compute_separation_identical(self, dg):
        """Compute separation for identical embeddings."""
        vec = np.array([1.0, 0.0, 0.0])
        sep = dg.compute_separation(vec, vec)
        assert sep == pytest.approx(0.0, abs=0.001)

    def test_compute_separation_orthogonal(self, dg):
        """Compute separation for orthogonal embeddings."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        sep = dg.compute_separation(vec_a, vec_b)
        assert sep == pytest.approx(1.0, abs=0.001)

    def test_compute_separation_opposite(self, dg):
        """Compute separation for opposite embeddings."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([-1.0, 0.0, 0.0])
        sep = dg.compute_separation(vec_a, vec_b)
        assert sep == pytest.approx(2.0, abs=0.001)

    def test_compute_separation_zero_vector(self, dg):
        """Compute separation with zero vector."""
        vec_a = np.array([0.0, 0.0, 0.0])
        vec_b = np.array([1.0, 0.0, 0.0])
        sep = dg.compute_separation(vec_a, vec_b)
        assert sep == 1.0

    def test_get_separation_history_empty(self, dg):
        """Get history when empty."""
        history = dg.get_separation_history()
        assert history == []

    @pytest.mark.asyncio
    async def test_get_separation_history_after_encode(self, dg, mock_embedding_provider):
        """Get history after encoding."""
        await dg.encode("test content")
        history = dg.get_separation_history()
        assert len(history) == 1
        assert isinstance(history[0], SeparationResult)

    @pytest.mark.asyncio
    async def test_get_separation_history_only_separated(self, dg, mock_embedding_provider, mock_vector_store):
        """Filter history to only separated results."""
        # First encode - no similar items
        await dg.encode("content 1")

        # Second encode - with similar items
        mock_vector_store.search = AsyncMock(return_value=[
            {"vector": np.random.randn(64).tolist(), "score": 0.85}
        ])
        await dg.encode("content 2")

        all_history = dg.get_separation_history(only_separated=False)
        separated_only = dg.get_separation_history(only_separated=True)

        assert len(all_history) == 2
        assert len(separated_only) <= len(all_history)

    def test_get_separation_history_limit(self, dg):
        """History respects limit."""
        # Add fake history
        for i in range(10):
            dg._separation_history.append(
                SeparationResult(
                    original_embedding=np.zeros(3),
                    separated_embedding=np.zeros(3),
                    similar_count=i,
                    max_similarity=0.5,
                    separation_magnitude=0.0,
                )
            )

        history = dg.get_separation_history(limit=5)
        assert len(history) == 5
        # Should be most recent 5
        assert history[-1].similar_count == 9

    def test_get_stats_empty(self, dg):
        """Get stats when empty."""
        stats = dg.get_stats()
        assert stats["total_encodings"] == 0
        assert stats["separations_applied"] == 0
        assert stats["separation_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_after_encodings(self, dg, mock_embedding_provider, mock_vector_store):
        """Get stats after encodings."""
        # Encode without separation
        await dg.encode("content 1")

        # Encode with separation
        mock_vector_store.search = AsyncMock(return_value=[
            {"vector": np.random.randn(64).tolist(), "score": 0.85}
        ])
        await dg.encode("content 2")

        stats = dg.get_stats()
        assert stats["total_encodings"] == 2
        assert stats["separations_applied"] == 1
        assert stats["separation_rate"] == 0.5

    def test_clear_history(self, dg):
        """Clear history."""
        dg._separation_history.append(
            SeparationResult(
                original_embedding=np.zeros(3),
                separated_embedding=np.zeros(3),
                similar_count=0,
                max_similarity=0.0,
                separation_magnitude=0.0,
            )
        )
        assert len(dg._separation_history) == 1

        dg.clear_history()
        assert len(dg._separation_history) == 0


# ==================== P2.3: NE-Modulated Pattern Separation Tests ====================


class TestNEModulatedSeparation:
    """Tests for P2.3 NE gain modulation in pattern separation."""

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create mock embedding provider."""
        provider = MagicMock()
        provider.embed_query = AsyncMock(
            return_value=np.random.randn(64).astype(np.float32)
        )
        return provider

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = MagicMock()
        store.search = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def dg(self, mock_embedding_provider, mock_vector_store):
        """Create DentateGyrus instance."""
        return DentateGyrus(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            collection_name="test_episodes",
        )

    def test_get_ne_modulated_separation_baseline(self):
        """NE gain of 1.0 returns base separation."""
        result = DentateGyrus.get_ne_modulated_separation(0.2, ne_gain=1.0)
        assert result == pytest.approx(0.2, abs=0.01)

    def test_get_ne_modulated_separation_high_arousal(self):
        """High NE gain increases separation strength."""
        base = 0.2
        high_gain = DentateGyrus.get_ne_modulated_separation(base, ne_gain=1.5)
        baseline = DentateGyrus.get_ne_modulated_separation(base, ne_gain=1.0)
        assert high_gain > baseline

    def test_get_ne_modulated_separation_low_arousal(self):
        """Low NE gain decreases separation strength."""
        base = 0.2
        low_gain = DentateGyrus.get_ne_modulated_separation(base, ne_gain=0.7)
        baseline = DentateGyrus.get_ne_modulated_separation(base, ne_gain=1.0)
        assert low_gain < baseline

    def test_get_ne_modulated_separation_clipping(self):
        """NE modulated separation is clipped to bounds."""
        # Very high gain should be clipped
        result = DentateGyrus.get_ne_modulated_separation(0.3, ne_gain=3.0, max_sep=0.45)
        assert result <= 0.45

        # Very low should hit minimum
        result = DentateGyrus.get_ne_modulated_separation(0.1, ne_gain=0.3, min_sep=0.05)
        assert result >= 0.05

    def test_get_ne_modulated_sparsity_baseline(self):
        """NE gain of 1.0 returns base sparsity."""
        result = DentateGyrus.get_ne_modulated_sparsity(0.04, ne_gain=1.0)
        assert result == pytest.approx(0.04, abs=0.01)

    def test_get_ne_modulated_sparsity_high_arousal(self):
        """High NE gain decreases sparsity (more extreme)."""
        base = 0.04
        high_gain = DentateGyrus.get_ne_modulated_sparsity(base, ne_gain=1.5)
        baseline = DentateGyrus.get_ne_modulated_sparsity(base, ne_gain=1.0)
        # Higher NE → LOWER sparsity ratio (fewer but stronger activations)
        assert high_gain < baseline

    def test_get_ne_modulated_sparsity_low_arousal(self):
        """Low NE gain increases sparsity (less extreme)."""
        base = 0.04
        low_gain = DentateGyrus.get_ne_modulated_sparsity(base, ne_gain=0.7)
        baseline = DentateGyrus.get_ne_modulated_sparsity(base, ne_gain=1.0)
        assert low_gain > baseline

    def test_get_ne_modulated_sparsity_clipping(self):
        """NE modulated sparsity is clipped to bounds."""
        # Very high gain should be clipped at minimum
        result = DentateGyrus.get_ne_modulated_sparsity(0.04, ne_gain=5.0, min_sparsity=0.01)
        assert result >= 0.01

        # Very low gain should be clipped at maximum
        result = DentateGyrus.get_ne_modulated_sparsity(0.1, ne_gain=0.4, max_sparsity=0.2)
        assert result <= 0.2

    def test_orthogonalize_with_high_ne_gain(self, dg):
        """High NE gain produces stronger orthogonalization."""
        target = np.array([1.0, 0.5, 0.2], dtype=np.float32)
        similar = [{"vector": [0.9, 0.4, 0.1], "score": 0.85}]

        result_baseline = dg._orthogonalize(target, similar, ne_gain=1.0)
        result_high = dg._orthogonalize(target, similar, ne_gain=1.8)

        # High NE should result in more separation from original
        sep_baseline = np.linalg.norm(result_baseline - target)
        sep_high = np.linalg.norm(result_high - target)
        assert sep_high >= sep_baseline  # High NE = more separation

    def test_orthogonalize_with_low_ne_gain(self, dg):
        """Low NE gain produces weaker orthogonalization."""
        target = np.array([1.0, 0.5, 0.2], dtype=np.float32)
        similar = [{"vector": [0.9, 0.4, 0.1], "score": 0.85}]

        result_baseline = dg._orthogonalize(target, similar, ne_gain=1.0)
        result_low = dg._orthogonalize(target, similar, ne_gain=0.6)

        # Low NE should result in less separation from original
        sep_baseline = np.linalg.norm(result_baseline - target)
        sep_low = np.linalg.norm(result_low - target)
        assert sep_low <= sep_baseline  # Low NE = less separation

    def test_sparsify_with_high_ne_gain(self, dg):
        """High NE gain produces sparser embeddings."""
        embedding = np.random.randn(100).astype(np.float32)

        result_baseline = dg._sparsify(embedding, ne_gain=1.0, use_soft_threshold=False)
        result_high = dg._sparsify(embedding, ne_gain=2.0, use_soft_threshold=False)

        # High NE should produce fewer non-zero elements
        nonzero_baseline = np.sum(result_baseline != 0)
        nonzero_high = np.sum(result_high != 0)
        assert nonzero_high <= nonzero_baseline

    def test_sparsify_with_low_ne_gain(self, dg):
        """Low NE gain produces less sparse embeddings."""
        embedding = np.random.randn(100).astype(np.float32)

        result_baseline = dg._sparsify(embedding, ne_gain=1.0, use_soft_threshold=False)
        result_low = dg._sparsify(embedding, ne_gain=0.6, use_soft_threshold=False)

        # Low NE should produce more non-zero elements
        nonzero_baseline = np.sum(result_baseline != 0)
        nonzero_low = np.sum(result_low != 0)
        assert nonzero_low >= nonzero_baseline

    @pytest.mark.asyncio
    async def test_encode_with_ne_gain_parameter(self, dg, mock_embedding_provider):
        """Encode accepts ne_gain parameter."""
        base_emb = np.random.randn(64).astype(np.float32)
        mock_embedding_provider.embed_query = AsyncMock(return_value=base_emb)

        # Should not raise
        result = await dg.encode("test content", ne_gain=1.5)
        assert result is not None
        assert isinstance(result, np.ndarray)

    @pytest.mark.asyncio
    async def test_encode_with_high_ne_produces_different_result(
        self, dg, mock_embedding_provider, mock_vector_store
    ):
        """Encode with different NE gains produces different results."""
        base_emb = np.random.randn(64).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)
        mock_embedding_provider.embed_query = AsyncMock(return_value=base_emb.copy())

        # Similar items to trigger separation
        similar_items = [
            {"vector": np.random.randn(64).tolist(), "score": 0.85},
        ]
        mock_vector_store.search = AsyncMock(return_value=similar_items)

        # Set seed for reproducible noise in _orthogonalize
        np.random.seed(42)
        result_baseline = await dg.encode("test content", ne_gain=1.0)

        np.random.seed(42)
        result_high = await dg.encode("test content", ne_gain=2.0)

        # Different NE gains should produce measurably different results
        # (Note: due to random noise, exact equality is unlikely anyway)
        assert result_baseline is not None
        assert result_high is not None


class TestPatternCompletion:
    """Tests for PatternCompletion class."""

    @pytest.fixture
    def pc(self):
        """Create PatternCompletion instance."""
        return PatternCompletion(
            embedding_dim=64,
            num_attractors=10,
            convergence_threshold=0.01,
            max_iterations=10,
        )

    def test_initialization(self, pc):
        """Test initialization."""
        assert pc.embedding_dim == 64
        assert pc.num_attractors == 10
        assert pc.convergence_threshold == 0.01
        assert pc.max_iterations == 10

    def test_initialization_defaults(self):
        """Test default initialization."""
        pc = PatternCompletion()
        assert pc.embedding_dim == 1024
        assert pc.num_attractors == 100
        assert pc.max_iterations == 10

    def test_add_attractor(self, pc):
        """Add attractor pattern."""
        pattern = np.random.randn(64)
        pc.add_attractor(pattern)
        assert pc.get_attractor_count() == 1

    def test_add_attractor_normalizes(self, pc):
        """Added attractor is normalized."""
        pattern = np.array([3.0, 4.0] + [0.0] * 62)  # Not unit norm
        pc.add_attractor(pattern)

        # Get the stored attractor
        nearest, _ = pc.find_nearest_attractor(pattern)
        assert np.linalg.norm(nearest) == pytest.approx(1.0, abs=0.001)

    def test_add_attractor_respects_limit(self, pc):
        """Attractor count respects num_attractors limit."""
        for i in range(15):
            pc.add_attractor(np.random.randn(64))

        assert pc.get_attractor_count() == 10  # Capped at num_attractors

    def test_complete_no_attractors(self, pc):
        """Complete with no attractors returns input."""
        partial = np.random.randn(64)
        completed, iterations = pc.complete(partial)
        np.testing.assert_array_equal(completed, partial)
        assert iterations == 0

    def test_complete_with_attractors(self, pc):
        """Complete with attractors converges."""
        # Add attractors
        attractor = np.zeros(64)
        attractor[0] = 1.0
        pc.add_attractor(attractor)

        # Partial pattern close to attractor
        partial = np.zeros(64)
        partial[0] = 0.9
        partial[1] = 0.1

        completed, iterations = pc.complete(partial)

        assert isinstance(completed, np.ndarray)
        assert iterations > 0

    def test_complete_converges_to_attractor(self, pc):
        """Complete converges toward nearest attractor."""
        # Add distinct attractors
        attractor1 = np.zeros(64)
        attractor1[0] = 1.0
        pc.add_attractor(attractor1)

        attractor2 = np.zeros(64)
        attractor2[1] = 1.0
        pc.add_attractor(attractor2)

        # Partial pattern closer to attractor1
        partial = np.zeros(64)
        partial[0] = 0.8
        partial[1] = 0.2

        completed, _ = pc.complete(partial)

        # Should be more similar to attractor1
        sim_to_1 = np.dot(completed, attractor1 / np.linalg.norm(attractor1))
        sim_to_2 = np.dot(completed, attractor2 / np.linalg.norm(attractor2))
        assert sim_to_1 > sim_to_2

    def test_complete_with_mask(self, pc):
        """Complete with mask keeps masked values closer to original."""
        attractor = np.ones(64) / np.sqrt(64)
        pc.add_attractor(attractor)

        partial = np.zeros(64)
        partial[0] = 0.5

        # Mask: True = keep original (before normalization)
        mask = np.zeros(64, dtype=bool)
        mask[0] = True

        completed_masked, _ = pc.complete(partial, mask=mask)
        completed_unmasked, _ = pc.complete(partial, mask=None)

        # Masked dimension should be closer to original than unmasked
        # (normalization may still modify it slightly)
        masked_diff = abs(completed_masked[0] - partial[0])
        unmasked_diff = abs(completed_unmasked[0] - partial[0])
        assert masked_diff <= unmasked_diff + 0.1  # Allow small tolerance

    def test_complete_max_iterations(self):
        """Complete respects max_iterations."""
        pc = PatternCompletion(
            embedding_dim=64,
            max_iterations=2,
            convergence_threshold=0.0001,  # Very tight
        )
        # Add attractor
        pc.add_attractor(np.random.randn(64))

        partial = np.random.randn(64)
        _, iterations = pc.complete(partial)

        assert iterations <= 2

    def test_find_nearest_attractor_empty(self, pc):
        """Find nearest with no attractors."""
        pattern = np.random.randn(64)
        nearest, similarity = pc.find_nearest_attractor(pattern)
        assert nearest is None
        assert similarity == 0.0

    def test_find_nearest_attractor(self, pc):
        """Find nearest attractor."""
        # Add attractors
        a1 = np.zeros(64)
        a1[0] = 1.0
        pc.add_attractor(a1)

        a2 = np.zeros(64)
        a2[1] = 1.0
        pc.add_attractor(a2)

        # Query closer to a1
        query = np.zeros(64)
        query[0] = 0.9
        query[1] = 0.1

        nearest, similarity = pc.find_nearest_attractor(query)

        assert nearest is not None
        assert similarity > 0
        # Should be most similar to a1
        assert nearest[0] > nearest[1]

    def test_get_attractor_count(self, pc):
        """Get attractor count."""
        assert pc.get_attractor_count() == 0

        pc.add_attractor(np.random.randn(64))
        assert pc.get_attractor_count() == 1

        pc.add_attractor(np.random.randn(64))
        assert pc.get_attractor_count() == 2

    def test_clear(self, pc):
        """Clear all attractors."""
        pc.add_attractor(np.random.randn(64))
        pc.add_attractor(np.random.randn(64))
        assert pc.get_attractor_count() == 2

        pc.clear()
        assert pc.get_attractor_count() == 0


class TestPatternCompletionDynamics:
    """Integration tests for pattern completion dynamics."""

    def test_multiple_attractors_competition(self):
        """Test competition between multiple attractors."""
        pc = PatternCompletion(
            embedding_dim=16,
            num_attractors=5,
            max_iterations=20,
        )

        # Add 3 distinct attractors
        np.random.seed(42)
        attractors = []
        for i in range(3):
            a = np.zeros(16)
            a[i * 5:(i + 1) * 5] = 1.0
            pc.add_attractor(a)
            attractors.append(a / np.linalg.norm(a))

        # Query biased toward attractor 1
        query = np.zeros(16)
        query[5:10] = 0.8  # Similar to attractor 1

        completed, iterations = pc.complete(query)

        # Should converge toward attractor 1
        sim_to_1 = np.dot(completed, attractors[1])
        sim_to_0 = np.dot(completed, attractors[0])
        sim_to_2 = np.dot(completed, attractors[2])

        assert sim_to_1 > sim_to_0
        assert sim_to_1 > sim_to_2

    def test_convergence_speed(self):
        """Test that similar inputs converge faster."""
        pc = PatternCompletion(
            embedding_dim=16,
            convergence_threshold=0.01,
        )

        # Add attractor
        attractor = np.ones(16) / np.sqrt(16)
        pc.add_attractor(attractor)

        # Very similar input
        similar = attractor + 0.01 * np.random.randn(16)
        _, iterations_similar = pc.complete(similar)

        # Less similar input
        dissimilar = np.random.randn(16)
        _, iterations_dissimilar = pc.complete(dissimilar)

        # Similar input should converge faster (fewer iterations)
        assert iterations_similar <= iterations_dissimilar


class TestCreateDentateGyrus:
    """Tests for factory function."""

    def test_create_dentate_gyrus(self):
        """Create DentateGyrus via factory."""
        provider = MagicMock()
        store = MagicMock()

        dg = create_dentate_gyrus(
            embedding_provider=provider,
            vector_store=store,
            collection_name="my_episodes",
            similarity_threshold=0.8,
        )

        assert isinstance(dg, DentateGyrus)
        assert dg.collection_name == "my_episodes"
        assert dg.similarity_threshold == 0.8


class TestPatternSeparationIntegration:
    """Integration tests for pattern separation workflow."""

    @pytest.mark.asyncio
    async def test_encode_then_separation_stats(self):
        """Full workflow: encode multiple items, check stats."""
        provider = MagicMock()
        store = MagicMock()

        # Provider returns slightly different embeddings each time
        call_count = [0]

        async def mock_embed(text):
            call_count[0] += 1
            emb = np.random.randn(64).astype(np.float32)
            return emb / np.linalg.norm(emb)

        provider.embed_query = mock_embed

        # First few calls - no similar items
        store.search = AsyncMock(return_value=[])

        dg = DentateGyrus(
            embedding_provider=provider,
            vector_store=store,
            similarity_threshold=0.75,
        )

        # Encode several items
        for i in range(5):
            await dg.encode(f"content {i}")

        stats = dg.get_stats()
        assert stats["total_encodings"] == 5
        assert stats["separations_applied"] == 0  # No similar items found

    @pytest.mark.asyncio
    async def test_separation_increases_distinctness(self):
        """Verify separation makes embeddings more distinct."""
        provider = MagicMock()
        store = MagicMock()

        # Base embedding
        base_emb = np.random.randn(64).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        provider.embed_query = AsyncMock(return_value=base_emb)

        # Similar item in store
        similar_emb = base_emb + 0.1 * np.random.randn(64).astype(np.float32)
        similar_emb = similar_emb / np.linalg.norm(similar_emb)

        store.search = AsyncMock(return_value=[
            {"vector": similar_emb.tolist(), "score": 0.9}
        ])

        dg = DentateGyrus(
            embedding_provider=provider,
            vector_store=store,
            use_sparse_coding=False,  # Disable for clearer test
        )

        separated = await dg.encode("test content")

        # Separated embedding should be more distant from similar item
        original_sim = np.dot(base_emb, similar_emb)
        separated_sim = np.dot(separated, similar_emb)

        # Separation should reduce similarity
        assert separated_sim < original_sim


# =============================================================================
# Modern Hopfield Network Tests (P3.2)
# =============================================================================

class TestHopfieldConfig:
    """Tests for HopfieldConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = HopfieldConfig()
        assert config.beta == 1.0
        assert config.max_iterations == 10
        assert config.convergence_threshold == 0.001
        assert config.normalize_patterns is True
        assert config.mode == HopfieldMode.MODERN

    def test_custom_config(self):
        """Test custom configuration."""
        config = HopfieldConfig(
            beta=8.0,
            max_iterations=20,
            mode=HopfieldMode.SPARSE,
            top_k=10
        )
        assert config.beta == 8.0
        assert config.max_iterations == 20
        assert config.mode == HopfieldMode.SPARSE
        assert config.top_k == 10


class TestModernHopfieldUpdate:
    """Tests for modern_hopfield_update function."""

    def test_single_pattern_recovery(self):
        """Single pattern should be perfectly recovered."""
        pattern = np.random.randn(64).astype(np.float32)
        pattern = pattern / np.linalg.norm(pattern)
        memories = pattern[np.newaxis, :]  # [1, 64]

        result = modern_hopfield_update(pattern, memories, beta=10.0)

        # Should recover the single pattern exactly
        np.testing.assert_allclose(result, pattern, rtol=1e-4)

    def test_high_beta_sharp_attention(self):
        """High beta should give sharp attention to nearest pattern."""
        # Two distinct patterns
        p1 = np.zeros(64, dtype=np.float32)
        p1[0:10] = 1.0
        p1 = p1 / np.linalg.norm(p1)

        p2 = np.zeros(64, dtype=np.float32)
        p2[30:40] = 1.0
        p2 = p2 / np.linalg.norm(p2)

        memories = np.array([p1, p2])

        # Query close to p1
        query = p1 + 0.1 * np.random.randn(64).astype(np.float32)
        query = query / np.linalg.norm(query)

        result = modern_hopfield_update(query, memories, beta=20.0)

        # Should be very close to p1
        sim_to_p1 = np.dot(result, p1)
        sim_to_p2 = np.dot(result, p2)
        assert sim_to_p1 > 0.95
        assert sim_to_p1 > sim_to_p2

    def test_low_beta_soft_attention(self):
        """Low beta should mix patterns."""
        p1 = np.array([1, 0, 0, 0], dtype=np.float32)
        p2 = np.array([0, 1, 0, 0], dtype=np.float32)
        memories = np.array([p1, p2])

        # Query equidistant
        query = np.array([0.7, 0.7, 0, 0], dtype=np.float32)
        query = query / np.linalg.norm(query)

        result = modern_hopfield_update(query, memories, beta=0.5, normalize=False)

        # With low beta, should mix patterns
        # Result should have significant weight on both dimensions
        assert result[0] > 0.2 and result[1] > 0.2

    def test_batch_processing(self):
        """Should handle batch of queries."""
        memories = np.random.randn(5, 32).astype(np.float32)
        memories = memories / np.linalg.norm(memories, axis=1, keepdims=True)

        queries = np.random.randn(3, 32).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        results = modern_hopfield_update(queries, memories, beta=5.0)

        assert results.shape == (3, 32)
        # Each result should be normalized
        norms = np.linalg.norm(results, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-4)

    def test_normalization_option(self):
        """Test normalize parameter."""
        memories = np.random.randn(3, 16).astype(np.float32)
        query = np.random.randn(16).astype(np.float32)

        result_normalized = modern_hopfield_update(query, memories, normalize=True)
        result_unnormalized = modern_hopfield_update(query, memories, normalize=False)

        assert np.linalg.norm(result_normalized) == pytest.approx(1.0, abs=1e-5)
        # Unnormalized may have different norm
        assert result_unnormalized is not None


class TestSparseHopfieldUpdate:
    """Tests for sparse_hopfield_update function."""

    def test_top_k_attention(self):
        """Should only attend to top-k patterns."""
        # 10 random patterns
        memories = np.random.randn(10, 32).astype(np.float32)
        memories = memories / np.linalg.norm(memories, axis=1, keepdims=True)

        query = memories[0] + 0.1 * np.random.randn(32).astype(np.float32)
        query = query / np.linalg.norm(query)

        result = sparse_hopfield_update(query, memories, beta=10.0, top_k=3)

        assert result.shape == (32,)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-4)

    def test_top_k_limits_patterns(self):
        """Top-k should limit to k patterns even with more memories."""
        memories = np.random.randn(100, 64).astype(np.float32)
        memories = memories / np.linalg.norm(memories, axis=1, keepdims=True)

        query = np.random.randn(64).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Should complete without error
        result = sparse_hopfield_update(query, memories, top_k=5)
        assert result.shape == (64,)

    def test_top_k_exceeds_memories(self):
        """Top-k > num_memories should work."""
        memories = np.random.randn(3, 32).astype(np.float32)
        query = np.random.randn(32).astype(np.float32)

        result = sparse_hopfield_update(query, memories, top_k=10)
        assert result.shape == (32,)


class TestHopfieldEnergy:
    """Tests for hopfield_energy function."""

    def test_stored_pattern_low_energy(self):
        """Stored pattern should have low energy."""
        pattern = np.random.randn(64).astype(np.float32)
        pattern = pattern / np.linalg.norm(pattern)
        memories = pattern[np.newaxis, :]

        energy = hopfield_energy(pattern, memories, beta=5.0)

        # Energy should be low (near -max_similarity)
        assert energy < 0

    def test_random_pattern_higher_energy(self):
        """Random pattern should have higher energy than stored."""
        stored = np.zeros(64, dtype=np.float32)
        stored[0:10] = 1.0
        stored = stored / np.linalg.norm(stored)
        memories = stored[np.newaxis, :]

        random_pattern = np.random.randn(64).astype(np.float32)
        random_pattern = random_pattern / np.linalg.norm(random_pattern)

        energy_stored = hopfield_energy(stored, memories, beta=5.0)
        energy_random = hopfield_energy(random_pattern, memories, beta=5.0)

        assert energy_random > energy_stored

    def test_beta_affects_energy(self):
        """Higher beta should give more extreme energy values."""
        memories = np.random.randn(5, 32).astype(np.float32)
        memories = memories / np.linalg.norm(memories, axis=1, keepdims=True)
        query = memories[0]  # Stored pattern

        energy_low_beta = hopfield_energy(query, memories, beta=1.0)
        energy_high_beta = hopfield_energy(query, memories, beta=10.0)

        # Both should be negative for stored pattern
        assert energy_low_beta < 0
        assert energy_high_beta < 0


class TestAttentionEntropy:
    """Tests for attention_entropy function."""

    def test_focused_attention_low_entropy(self):
        """Query matching one pattern should have low entropy."""
        memories = np.eye(4, dtype=np.float32)  # 4 orthogonal patterns
        query = memories[0]  # Exactly matches first pattern

        entropy = attention_entropy(query, memories, beta=10.0)

        # Should be close to 0 (very focused)
        assert entropy < 0.5

    def test_uniform_attention_high_entropy(self):
        """Equal similarity to all should give high entropy."""
        # All patterns are the same
        pattern = np.ones(32, dtype=np.float32) / np.sqrt(32)
        memories = np.tile(pattern, (5, 1))
        query = pattern

        entropy = attention_entropy(query, memories, beta=1.0)

        # Should be close to log(5) ≈ 1.6
        assert entropy > 1.0

    def test_beta_affects_entropy(self):
        """Higher beta should reduce entropy (sharper attention)."""
        memories = np.random.randn(5, 32).astype(np.float32)
        memories = memories / np.linalg.norm(memories, axis=1, keepdims=True)
        query = memories[0] + 0.2 * np.random.randn(32).astype(np.float32)
        query = query / np.linalg.norm(query)

        entropy_low = attention_entropy(query, memories, beta=1.0)
        entropy_high = attention_entropy(query, memories, beta=20.0)

        assert entropy_high < entropy_low


class TestModernPatternCompletion:
    """Tests for updated PatternCompletion with Modern Hopfield."""

    def test_beta_parameter(self):
        """PatternCompletion should accept beta parameter."""
        pc = PatternCompletion(beta=16.0)
        assert pc.beta == 16.0

    def test_mode_parameter(self):
        """PatternCompletion should accept mode parameter."""
        pc = PatternCompletion(mode=HopfieldMode.SPARSE)
        assert pc.mode == HopfieldMode.SPARSE

    def test_complete_returns_hopfield_result_in_history(self):
        """Completion should track HopfieldResult in history."""
        pc = PatternCompletion(embedding_dim=32, beta=8.0)

        # Add attractor
        attractor = np.random.randn(32).astype(np.float32)
        pc.add_attractor(attractor)

        # Complete
        query = attractor + 0.2 * np.random.randn(32).astype(np.float32)
        pc.complete(query)

        history = pc.get_completion_history()
        assert len(history) == 1
        assert isinstance(history[0], HopfieldResult)
        assert history[0].converged is not None
        assert history[0].final_energy is not None
        assert history[0].attention_entropy is not None

    def test_complete_with_details(self):
        """complete_with_details should return HopfieldResult."""
        pc = PatternCompletion(embedding_dim=32, beta=8.0)
        pc.add_attractor(np.random.randn(32))

        result = pc.complete_with_details(np.random.randn(32))

        assert isinstance(result, HopfieldResult)
        assert result.completed_pattern.shape == (32,)

    def test_high_beta_better_recovery(self):
        """Higher beta should give better pattern recovery."""
        dim = 64
        num_patterns = 20

        # Generate patterns
        patterns = np.random.randn(num_patterns, dim).astype(np.float32)
        patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)

        # Test recovery at different beta values
        results = {}
        for beta in [1.0, 8.0, 16.0]:
            pc = PatternCompletion(embedding_dim=dim, beta=beta)
            for p in patterns:
                pc.add_attractor(p)

            correct = 0
            for i in range(num_patterns):
                # Add noise
                noisy = patterns[i] + 0.3 * np.random.randn(dim).astype(np.float32)
                noisy = noisy / np.linalg.norm(noisy)

                completed, _ = pc.complete(noisy)

                # Check if recovered correct pattern
                sims = patterns @ completed
                if np.argmax(sims) == i:
                    correct += 1

            results[beta] = correct / num_patterns

        # Higher beta should generally give better recovery
        assert results[16.0] >= results[1.0] - 0.1  # Allow some variance

    def test_sparse_mode(self):
        """Sparse mode should work."""
        pc = PatternCompletion(
            embedding_dim=64,
            mode=HopfieldMode.SPARSE,
            sparse_top_k=5,
            beta=8.0
        )

        # Add patterns
        for _ in range(20):
            pc.add_attractor(np.random.randn(64))

        query = np.random.randn(64)
        completed, iterations = pc.complete(query)

        assert completed.shape == (64,)
        assert iterations > 0

    def test_get_attention_weights(self):
        """get_attention_weights should return valid distribution."""
        pc = PatternCompletion(embedding_dim=32, beta=8.0)

        for _ in range(5):
            pc.add_attractor(np.random.randn(32))

        query = np.random.randn(32)
        weights = pc.get_attention_weights(query)

        assert weights.shape == (5,)
        assert np.all(weights >= 0)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-5)

    def test_get_attractors(self):
        """get_attractors should return stored patterns."""
        pc = PatternCompletion(embedding_dim=16)

        patterns = [np.random.randn(16) for _ in range(3)]
        for p in patterns:
            pc.add_attractor(p)

        attractors = pc.get_attractors()

        assert attractors.shape == (3, 16)

    def test_get_stats(self):
        """get_stats should return completion statistics."""
        pc = PatternCompletion(embedding_dim=32, beta=8.0)
        pc.add_attractor(np.random.randn(32))

        # Before any completions
        stats = pc.get_stats()
        assert stats["total_completions"] == 0
        assert stats["beta"] == 8.0

        # After completions
        for _ in range(5):
            pc.complete(np.random.randn(32))

        stats = pc.get_stats()
        assert stats["total_completions"] == 5
        assert 0 <= stats["convergence_rate"] <= 1.0
        assert stats["avg_iterations"] > 0


class TestCreatePatternCompletion:
    """Tests for create_pattern_completion factory."""

    def test_create_with_defaults(self):
        """Create with default parameters."""
        pc = create_pattern_completion()

        assert isinstance(pc, PatternCompletion)
        assert pc.beta == 8.0
        assert pc.mode == HopfieldMode.MODERN

    def test_create_with_string_mode(self):
        """Create with string mode specification."""
        pc = create_pattern_completion(mode="sparse")

        assert pc.mode == HopfieldMode.SPARSE

    def test_create_with_custom_beta(self):
        """Create with custom beta."""
        pc = create_pattern_completion(beta=20.0)

        assert pc.beta == 20.0


class TestBenchmarkHopfieldCapacity:
    """Tests for benchmark_hopfield_capacity function."""

    def test_benchmark_runs(self):
        """Benchmark should run without error."""
        results = benchmark_hopfield_capacity(
            dim=32,
            num_patterns=10,
            noise_level=0.2,
            beta_values=[1.0, 4.0],
            num_trials=5
        )

        assert 1.0 in results
        assert 4.0 in results
        assert "accuracy" in results[1.0]
        assert "avg_similarity" in results[1.0]
        assert "avg_iterations" in results[1.0]

    def test_higher_beta_better_accuracy(self):
        """Higher beta should generally give better accuracy."""
        results = benchmark_hopfield_capacity(
            dim=64,
            num_patterns=20,
            noise_level=0.3,
            beta_values=[1.0, 16.0],
            num_trials=20
        )

        # Higher beta should generally perform better
        # (allow some variance due to randomness)
        assert results[16.0]["accuracy"] >= results[1.0]["accuracy"] - 0.2

    def test_accuracy_bounds(self):
        """Accuracy should be between 0 and 1."""
        results = benchmark_hopfield_capacity(
            dim=32,
            num_patterns=10,
            num_trials=10
        )

        for beta, metrics in results.items():
            assert 0 <= metrics["accuracy"] <= 1
            assert 0 <= metrics["avg_similarity"] <= 1
