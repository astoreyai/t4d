"""
Unit tests for T4DM Lite - minimal in-memory vector store.

Tests the simple store/search/delete interface without any dependencies
on the full T4DM stack.
"""

import numpy as np
import pytest

from t4dm.lite import (
    Memory,
    _cosine_similarity,
    _hash_embedding,
    delete,
    search,
    store,
)


class TestHashEmbedding:
    """Tests for the mock hash-based embedding function."""

    def test_returns_correct_dimension(self):
        """Embedding should return vector of specified dimension."""
        embedding = _hash_embedding("hello", dim=384)
        assert embedding.shape == (384,)

    def test_different_dimensions(self):
        """Should support various dimensions."""
        for dim in [64, 128, 256, 512, 1024]:
            embedding = _hash_embedding("test", dim=dim)
            assert embedding.shape == (dim,)

    def test_deterministic(self):
        """Same input should produce same output."""
        e1 = _hash_embedding("hello world")
        e2 = _hash_embedding("hello world")
        np.testing.assert_array_equal(e1, e2)

    def test_normalized(self):
        """Embeddings should be unit normalized."""
        embedding = _hash_embedding("test string")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_case_insensitive(self):
        """Embeddings should be case-insensitive."""
        e1 = _hash_embedding("Hello World")
        e2 = _hash_embedding("hello world")
        np.testing.assert_array_equal(e1, e2)

    def test_different_texts_differ(self):
        """Different texts should produce different embeddings."""
        e1 = _hash_embedding("hello")
        e2 = _hash_embedding("goodbye")
        assert not np.allclose(e1, e2)


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(v1, v2)) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert abs(_cosine_similarity(v1, v2) + 1.0) < 1e-6

    def test_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 2.0])
        assert _cosine_similarity(v1, v2) == 0.0


class TestMemoryStore:
    """Tests for Memory.store() method."""

    def test_store_returns_id(self):
        """Store should return a non-empty ID."""
        mem = Memory()
        mid = mem.store("test content")
        assert mid is not None
        assert len(mid) > 0

    def test_store_with_custom_id(self):
        """Store should accept custom ID."""
        mem = Memory()
        mid = mem.store("test content", id="custom-id-123")
        assert mid == "custom-id-123"

    def test_store_increments_count(self):
        """Store should increment memory count."""
        mem = Memory()
        assert mem.count() == 0
        mem.store("first")
        assert mem.count() == 1
        mem.store("second")
        assert mem.count() == 2

    def test_store_empty_raises(self):
        """Store should reject empty content."""
        mem = Memory()
        with pytest.raises(ValueError, match="empty"):
            mem.store("")

    def test_store_whitespace_only_raises(self):
        """Store should reject whitespace-only content."""
        mem = Memory()
        with pytest.raises(ValueError, match="empty"):
            mem.store("   ")

    def test_store_with_metadata(self):
        """Store should accept metadata."""
        mem = Memory()
        mid = mem.store("test", metadata={"key": "value"})
        result = mem.get(mid)
        assert result["metadata"]["key"] == "value"


class TestMemorySearch:
    """Tests for Memory.search() method."""

    def test_search_empty_returns_empty(self):
        """Search on empty memory should return empty list."""
        mem = Memory()
        results = mem.search("anything")
        assert results == []

    def test_search_returns_results(self):
        """Search should return matching memories."""
        mem = Memory()
        mem.store("Python is a programming language")
        mem.store("JavaScript is also a language")
        mem.store("The weather is nice today")

        results = mem.search("programming language")
        # Hash-based embedding returns all results, not semantic ranking
        assert len(results) == 3
        # All stored items should be present
        contents = {r["content"] for r in results}
        assert "Python is a programming language" in contents

    def test_search_respects_k(self):
        """Search should return at most k results."""
        mem = Memory()
        for i in range(10):
            mem.store(f"Memory number {i}")

        results = mem.search("memory", k=3)
        assert len(results) == 3

    def test_search_returns_correct_fields(self):
        """Search results should have expected fields."""
        mem = Memory()
        mem.store("test content", metadata={"tag": "test"})

        results = mem.search("test")
        assert len(results) == 1

        result = results[0]
        assert "id" in result
        assert "content" in result
        assert "score" in result
        assert "timestamp" in result
        assert "metadata" in result
        assert result["content"] == "test content"
        assert result["metadata"]["tag"] == "test"

    def test_search_scores_ordered(self):
        """Search results should be ordered by score descending."""
        mem = Memory()
        mem.store("apple fruit")
        mem.store("banana fruit")
        mem.store("car vehicle")

        results = mem.search("apple", k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestMemoryDelete:
    """Tests for Memory.delete() method."""

    def test_delete_existing(self):
        """Delete should return True for existing memory."""
        mem = Memory()
        mid = mem.store("test")
        assert mem.delete(mid) is True
        assert mem.count() == 0

    def test_delete_nonexistent(self):
        """Delete should return False for nonexistent memory."""
        mem = Memory()
        assert mem.delete("nonexistent-id") is False

    def test_delete_removes_from_search(self):
        """Deleted memories should not appear in search."""
        mem = Memory()
        mid = mem.store("unique searchable content")
        mem.delete(mid)

        results = mem.search("unique searchable")
        assert len(results) == 0


class TestMemoryGet:
    """Tests for Memory.get() method."""

    def test_get_existing(self):
        """Get should return memory data for existing ID."""
        mem = Memory()
        mid = mem.store("test content")
        result = mem.get(mid)

        assert result is not None
        assert result["id"] == mid
        assert result["content"] == "test content"
        assert "timestamp" in result

    def test_get_nonexistent(self):
        """Get should return None for nonexistent ID."""
        mem = Memory()
        assert mem.get("nonexistent") is None


class TestMemoryClear:
    """Tests for Memory.clear() method."""

    def test_clear_removes_all(self):
        """Clear should remove all memories."""
        mem = Memory()
        mem.store("one")
        mem.store("two")
        mem.store("three")

        mem.clear()
        assert mem.count() == 0

    def test_clear_empty_is_safe(self):
        """Clear on empty memory should not raise."""
        mem = Memory()
        mem.clear()  # Should not raise
        assert mem.count() == 0


class TestCustomEmbedding:
    """Tests for custom embedding function support."""

    def test_custom_embed_fn(self):
        """Memory should use custom embedding function."""
        call_count = [0]

        def custom_embed(text: str) -> np.ndarray:
            call_count[0] += 1
            return np.ones(64)

        mem = Memory(embedding_dim=64, embed_fn=custom_embed)
        mem.store("test")
        mem.search("test")

        # Should have been called twice (store + search)
        assert call_count[0] == 2


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_store_and_search(self):
        """Module-level store and search should work."""
        # Note: these use a global default instance
        mid = store("module level test content unique12345")
        results = search("module level unique12345")

        assert len(results) > 0
        found = any("unique12345" in r["content"] for r in results)
        assert found

        # Clean up
        delete(mid)

    def test_delete(self):
        """Module-level delete should work."""
        mid = store("to be deleted module test")
        assert delete(mid) is True
        assert delete(mid) is False  # Already deleted
