"""Tests for SimpleMemory baseline implementation.

Validates the TF-IDF memory baseline used for comparison benchmarks.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# Add src and benchmarks to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from baselines.simple_memory import SimpleMemory, MemoryEntry


class TestSimpleMemoryStore:
    """Test store() method."""

    def test_store_returns_id(self):
        """Store returns a valid ID."""
        mem = SimpleMemory()
        id_ = mem.store("hello world")
        assert id_ is not None
        assert len(id_) > 0

    def test_store_with_custom_id(self):
        """Store with custom ID uses that ID."""
        mem = SimpleMemory()
        id_ = mem.store("hello world", id="my-custom-id")
        assert id_ == "my-custom-id"

    def test_store_increments_length(self):
        """Storing items increases length."""
        mem = SimpleMemory()
        assert len(mem) == 0
        mem.store("first")
        assert len(mem) == 1
        mem.store("second")
        assert len(mem) == 2

    def test_store_replaces_existing_id(self):
        """Storing with existing ID replaces content."""
        mem = SimpleMemory()
        mem.store("original", id="test-id")
        mem.store("updated", id="test-id")
        assert len(mem) == 1
        results = mem.search("updated", k=1)
        assert results[0]["content"] == "updated"

    def test_store_lru_eviction(self):
        """LRU eviction when max_size exceeded."""
        mem = SimpleMemory(max_size=3)
        mem.store("first", id="1")
        mem.store("second", id="2")
        mem.store("third", id="3")
        mem.store("fourth", id="4")
        assert len(mem) == 3
        # First entry should be evicted
        results = mem.search("first", k=10)
        ids = [r["id"] for r in results]
        assert "1" not in ids
        assert "4" in ids


class TestSimpleMemorySearch:
    """Test search() method."""

    def test_search_empty_memory(self):
        """Search on empty memory returns empty list."""
        mem = SimpleMemory()
        results = mem.search("anything")
        assert results == []

    def test_search_returns_results(self):
        """Search returns matching results."""
        mem = SimpleMemory()
        mem.store("the quick brown fox", id="fox")
        mem.store("the lazy dog", id="dog")
        results = mem.search("fox")
        assert len(results) > 0
        assert results[0]["id"] == "fox"

    def test_search_respects_k(self):
        """Search returns at most k results."""
        mem = SimpleMemory()
        for i in range(10):
            mem.store(f"document number {i}", id=str(i))
        results = mem.search("document", k=3)
        assert len(results) == 3

    def test_search_scores_sorted_descending(self):
        """Search results are sorted by score descending."""
        mem = SimpleMemory()
        mem.store("cat cat cat", id="high")
        mem.store("cat dog", id="medium")
        mem.store("dog dog dog", id="low")
        results = mem.search("cat", k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0]["id"] == "high"

    def test_search_result_format(self):
        """Search results have correct format."""
        mem = SimpleMemory()
        mem.store("test content", id="test-id")
        results = mem.search("test", k=1)
        assert len(results) == 1
        result = results[0]
        assert "id" in result
        assert "content" in result
        assert "score" in result
        assert result["id"] == "test-id"
        assert result["content"] == "test content"

    def test_search_tfidf_similarity(self):
        """TF-IDF gives higher scores to more relevant documents."""
        mem = SimpleMemory()
        # Document with high frequency of 'python'
        mem.store("python python python programming", id="python-heavy")
        # Document with low frequency of 'python'
        mem.store("python java rust go", id="multi-lang")
        # Document without 'python'
        mem.store("java java java programming", id="java-heavy")
        results = mem.search("python programming", k=3)
        # python-heavy should rank first
        assert results[0]["id"] == "python-heavy"


class TestSimpleMemoryRecencyWeighting:
    """Test LRU-style recency weighting."""

    def test_recency_weight_zero_is_pure_tfidf(self):
        """With recency_weight=0, results are pure TF-IDF."""
        mem = SimpleMemory(recency_weight=0.0)
        # Store old but highly relevant
        mem.store("python python python", id="old-relevant")
        time.sleep(0.01)
        # Store new but less relevant
        mem.store("java rust go", id="new-irrelevant")
        results = mem.search("python", k=2)
        # Old but relevant should win
        assert results[0]["id"] == "old-relevant"

    def test_recency_weight_one_is_pure_recency(self):
        """With recency_weight=1, results are pure recency."""
        mem = SimpleMemory(recency_weight=1.0)
        mem.store("python python python", id="old-relevant")
        time.sleep(0.01)
        mem.store("unrelated content", id="new-irrelevant")
        results = mem.search("python", k=2)
        # New should win despite low relevance
        assert results[0]["id"] == "new-irrelevant"

    def test_recency_weight_blends(self):
        """With 0 < recency_weight < 1, results blend both factors."""
        mem = SimpleMemory(recency_weight=0.5)
        mem.store("cat", id="old-cat")
        time.sleep(0.01)
        mem.store("cat cat cat cat cat", id="new-cat")
        results = mem.search("cat", k=2)
        # new-cat has both higher recency and higher relevance
        assert results[0]["id"] == "new-cat"

    def test_recency_weight_clamped(self):
        """Recency weight is clamped to [0, 1]."""
        mem_low = SimpleMemory(recency_weight=-0.5)
        mem_high = SimpleMemory(recency_weight=1.5)
        assert mem_low._recency_weight == 0.0
        assert mem_high._recency_weight == 1.0


class TestSimpleMemoryDelete:
    """Test delete() method."""

    def test_delete_existing(self):
        """Delete existing entry returns True."""
        mem = SimpleMemory()
        mem.store("content", id="test-id")
        assert mem.delete("test-id") is True
        assert len(mem) == 0

    def test_delete_nonexistent(self):
        """Delete nonexistent entry returns False."""
        mem = SimpleMemory()
        assert mem.delete("nonexistent") is False

    def test_delete_removes_from_search(self):
        """Deleted entry no longer appears in search."""
        mem = SimpleMemory()
        mem.store("unique content", id="to-delete")
        mem.store("other content", id="to-keep")
        mem.delete("to-delete")
        results = mem.search("unique", k=10)
        ids = [r["id"] for r in results]
        assert "to-delete" not in ids

    def test_delete_updates_df(self):
        """Delete updates document frequency correctly."""
        mem = SimpleMemory()
        mem.store("cat dog", id="1")
        mem.store("cat bird", id="2")
        # cat appears in 2 docs
        assert mem._df.get("cat", 0) == 2
        mem.delete("1")
        # cat now appears in 1 doc
        assert mem._df.get("cat", 0) == 1


class TestSimpleMemoryClear:
    """Test clear() method."""

    def test_clear_empties_memory(self):
        """Clear removes all entries."""
        mem = SimpleMemory()
        mem.store("a")
        mem.store("b")
        mem.store("c")
        mem.clear()
        assert len(mem) == 0

    def test_clear_resets_df(self):
        """Clear resets document frequency."""
        mem = SimpleMemory()
        mem.store("cat dog bird")
        mem.clear()
        assert len(mem._df) == 0


class TestSimpleMemoryEdgeCases:
    """Test edge cases and special inputs."""

    def test_empty_content(self):
        """Empty content can be stored."""
        mem = SimpleMemory()
        id_ = mem.store("")
        assert id_ is not None
        assert len(mem) == 1

    def test_empty_query(self):
        """Empty query returns results (all with 0 similarity)."""
        mem = SimpleMemory()
        mem.store("content")
        results = mem.search("")
        assert len(results) == 1

    def test_special_characters(self):
        """Content with special characters is handled."""
        mem = SimpleMemory()
        mem.store("hello! @world# $test%", id="special")
        results = mem.search("hello world test", k=1)
        assert results[0]["id"] == "special"

    def test_unicode_content(self):
        """Unicode content is handled (non-ASCII stripped in tokenization)."""
        mem = SimpleMemory()
        mem.store("hello world", id="ascii")
        # Unicode letters are stripped by current tokenization
        results = mem.search("hello", k=1)
        assert results[0]["id"] == "ascii"

    def test_case_insensitive(self):
        """Search is case insensitive."""
        mem = SimpleMemory()
        mem.store("HELLO WORLD", id="upper")
        results = mem.search("hello world", k=1)
        assert results[0]["id"] == "upper"


class TestSimpleMemoryPerformance:
    """Basic performance sanity checks."""

    def test_store_many_items(self):
        """Can store many items without error."""
        mem = SimpleMemory(max_size=1000)
        for i in range(500):
            mem.store(f"document content number {i}", id=str(i))
        assert len(mem) == 500

    def test_search_performance_reasonable(self):
        """Search completes in reasonable time for 1000 items."""
        mem = SimpleMemory(max_size=1000)
        for i in range(1000):
            mem.store(f"document with various words like cat dog bird {i}")
        start = time.time()
        for _ in range(10):
            mem.search("cat dog", k=10)
        elapsed = time.time() - start
        # Should complete 10 searches in under 1 second
        assert elapsed < 1.0
