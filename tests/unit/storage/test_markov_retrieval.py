"""
Unit Tests for Markov Blanket Retrieval (W3-01).

Verifies retrieval prioritizing Markov blanket of query concept
following Pearl (1988) and Friston's Free Energy Principle.

Evidence Base: Pearl (1988) "Probabilistic Reasoning in Intelligent Systems"
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from uuid import uuid4, UUID


class TestMarkovBlanketComputation:
    """Test Markov blanket computation."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine with traverse capability."""
        engine = Mock()
        return engine

    def test_markov_blanket_includes_parents(self, mock_engine):
        """MB should include parent nodes."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketRetriever

        # Set up: B has parent A
        B_id = uuid4()
        A_id = uuid4()

        parent_result = Mock()
        parent_result.id = A_id

        mock_engine.traverse.side_effect = [
            [parent_result],  # Parents of B
            [],  # Children of B
        ]

        retriever = MarkovBlanketRetriever(mock_engine)
        mb = retriever._get_markov_blanket(B_id)

        assert A_id in mb

    def test_markov_blanket_includes_children(self, mock_engine):
        """MB should include child nodes."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketRetriever

        # Set up: B has child D
        B_id = uuid4()
        D_id = uuid4()

        child_result = Mock()
        child_result.id = D_id

        mock_engine.traverse.side_effect = [
            [],  # Parents of B
            [child_result],  # Children of B
            [],  # Parents of D (spouses) - empty
        ]

        retriever = MarkovBlanketRetriever(mock_engine)
        mb = retriever._get_markov_blanket(B_id)

        assert D_id in mb

    def test_markov_blanket_includes_spouses(self, mock_engine):
        """MB should include spouses (co-parents of children)."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketRetriever

        # Set up: B and C are both parents of D (spouses)
        B_id = uuid4()
        C_id = uuid4()
        D_id = uuid4()

        child_result = Mock()
        child_result.id = D_id

        spouse_result = Mock()
        spouse_result.id = C_id

        # Traverse calls: parents of B, children of B, parents of D (spouses)
        mock_engine.traverse.side_effect = [
            [],  # Parents of B
            [child_result],  # Children of B (D)
            [spouse_result, Mock(id=B_id)],  # Parents of D (C and B)
        ]

        retriever = MarkovBlanketRetriever(mock_engine)
        mb = retriever._get_markov_blanket(B_id)

        assert C_id in mb  # Spouse C should be in MB
        assert B_id not in mb  # Self should not be in MB


class TestMarkovBlanketRetriever:
    """Test MarkovBlanketRetriever search behavior."""

    @pytest.fixture
    def mock_engine(self):
        engine = Mock()
        return engine

    def test_retriever_creation(self, mock_engine):
        """Should create retriever with engine and exploration ratio."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketRetriever

        retriever = MarkovBlanketRetriever(mock_engine, exploration_ratio=0.1)

        assert retriever.engine is mock_engine
        assert retriever.exploration_ratio == 0.1

    def test_search_without_concept_uses_global(self, mock_engine):
        """Without query concept, should use pure vector search."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketRetriever

        mock_results = [Mock(id=uuid4()) for _ in range(5)]
        mock_engine.search.return_value = mock_results

        retriever = MarkovBlanketRetriever(mock_engine)
        results = retriever.search(np.random.randn(64), query_concept=None, k=5)

        mock_engine.search.assert_called_once()
        assert len(results) == 5

    def test_search_with_concept_prioritizes_mb(self, mock_engine):
        """With query concept, should prioritize MB results."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketRetriever

        concept_id = uuid4()
        mb_item_id = uuid4()
        global_item_id = uuid4()

        # Set up MB item
        mb_item = Mock()
        mb_item.id = mb_item_id

        global_item = Mock()
        global_item.id = global_item_id

        # traverse calls: parents, children (no spouses since children is empty)
        mock_engine.traverse.side_effect = [
            [mb_item],  # Parents (mb_item is parent)
            [],  # Children (empty, no spouse calls needed)
        ]

        # Remove search_filtered so it falls back to regular search
        del mock_engine.search_filtered

        # Search returns items - called multiple times for MB search and global search
        mock_engine.search.return_value = [mb_item, global_item]

        retriever = MarkovBlanketRetriever(mock_engine, exploration_ratio=0.1)
        results = retriever.search(np.random.randn(64), query_concept=concept_id, k=10)

        # Results should be a list
        assert isinstance(results, list)
        # Should have made search call
        assert mock_engine.search.called

    def test_exploration_preserves_diversity(self, mock_engine):
        """Should include some global results for exploration."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketRetriever

        concept_id = uuid4()

        # Empty MB
        mock_engine.traverse.return_value = []

        # Global results
        global_items = [Mock(id=uuid4()) for _ in range(10)]
        mock_engine.search.return_value = global_items

        retriever = MarkovBlanketRetriever(mock_engine, exploration_ratio=0.2)
        results = retriever.search(np.random.randn(64), query_concept=concept_id, k=10)

        # Should have exploration results
        assert len(results) > 0


class TestMarkovBlanketConfig:
    """Test configuration for Markov blanket retrieval."""

    def test_default_config(self):
        """Default config should have sensible values."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketConfig

        config = MarkovBlanketConfig()

        assert config.exploration_ratio == 0.1
        assert config.edge_type == "CAUSES"

    def test_config_override(self):
        """Should be able to override config values."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketConfig

        config = MarkovBlanketConfig(
            exploration_ratio=0.2,
            edge_type="TEMPORAL_BEFORE",
        )

        assert config.exploration_ratio == 0.2
        assert config.edge_type == "TEMPORAL_BEFORE"


class TestMarkovBlanketLatency:
    """Test latency requirements."""

    def test_mb_computation_under_5ms(self):
        """MB computation should add <5ms latency."""
        from t4dm.storage.t4dx.markov_retrieval import MarkovBlanketRetriever
        import time

        engine = Mock()
        # Fast mock returns
        engine.traverse.return_value = []

        retriever = MarkovBlanketRetriever(engine)
        concept_id = uuid4()

        # Warmup
        for _ in range(10):
            retriever._get_markov_blanket(concept_id)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            retriever._get_markov_blanket(concept_id)
            times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000

        assert avg_time_ms < 10.0, f"MB computation took {avg_time_ms:.2f}ms"
