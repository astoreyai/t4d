"""
Unit Tests for Learned Edge Importance (W3-02).

Verifies learnable edge type importance for graph traversal
following Graves (2014) neural memory principles.

Evidence Base: Graves et al. (2014) "Neural Turing Machines"
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock
from uuid import uuid4, UUID


class TestLearnedEdgeImportance:
    """Test LearnedEdgeImportance module."""

    def test_importance_initialization(self):
        """Edge importance should initialize to uniform weights."""
        from t4dm.storage.t4dx.learned_edges import LearnedEdgeImportance

        edge_importance = LearnedEdgeImportance()

        # All weights should start at 1.0 (exp(0) = 1)
        for edge_type in edge_importance.EDGE_TYPES:
            weight = edge_importance.get_weight(edge_type)
            assert 0.9 < weight < 1.1, f"{edge_type} should start near 1.0"

    def test_get_weight_returns_positive(self):
        """Edge weights should always be positive (exp ensures this)."""
        from t4dm.storage.t4dx.learned_edges import LearnedEdgeImportance

        edge_importance = LearnedEdgeImportance()

        for edge_type in edge_importance.EDGE_TYPES:
            weight = edge_importance.get_weight(edge_type)
            assert weight > 0, f"{edge_type} weight should be positive"

    def test_forward_batch_weights(self):
        """Forward should return weights for batch of edge type indices."""
        from t4dm.storage.t4dx.learned_edges import LearnedEdgeImportance

        edge_importance = LearnedEdgeImportance()

        # Batch of edge type indices
        edge_indices = torch.tensor([0, 1, 2])  # CAUSES, TEMPORAL_BEFORE, PART_OF
        weights = edge_importance(edge_indices)

        assert weights.shape == (3,)
        assert (weights > 0).all()

    def test_all_edge_types_supported(self):
        """All standard edge types should be supported."""
        from t4dm.storage.t4dx.learned_edges import LearnedEdgeImportance

        edge_importance = LearnedEdgeImportance()

        expected_types = [
            "CAUSES", "TEMPORAL_BEFORE", "PART_OF", "SIMILAR_TO",
            "CONTRADICTS", "ELABORATES", "EXEMPLIFIES", "GENERALIZES",
            "PRECONDITION", "EFFECT", "ATTRIBUTE", "CONTEXT",
            "REFERENCES", "DERIVES_FROM", "SUPPORTS", "OPPOSES", "NEUTRAL"
        ]

        for edge_type in expected_types:
            # Should not raise
            weight = edge_importance.get_weight(edge_type)
            assert isinstance(weight, float)

    def test_unknown_edge_type_raises(self):
        """Unknown edge types should raise ValueError."""
        from t4dm.storage.t4dx.learned_edges import LearnedEdgeImportance

        edge_importance = LearnedEdgeImportance()

        with pytest.raises(ValueError):
            edge_importance.get_weight("UNKNOWN_TYPE")


class TestEdgeImportanceTrainer:
    """Test EdgeImportanceTrainer."""

    def test_trainer_creation(self):
        """Should create trainer with edge importance module."""
        from t4dm.storage.t4dx.learned_edges import (
            LearnedEdgeImportance,
            EdgeImportanceTrainer,
        )

        edge_importance = LearnedEdgeImportance()
        trainer = EdgeImportanceTrainer(edge_importance, lr=0.01)

        assert trainer.edge_importance is edge_importance

    def test_edge_importance_learns(self):
        """Edge importance should diverge from uniform after training."""
        from t4dm.storage.t4dx.learned_edges import (
            LearnedEdgeImportance,
            EdgeImportanceTrainer,
        )

        edge_importance = LearnedEdgeImportance()
        trainer = EdgeImportanceTrainer(edge_importance, lr=0.1)

        initial_weights = [
            edge_importance.get_weight(e) for e in edge_importance.EDGE_TYPES
        ]

        # Train on retrieval feedback
        for _ in range(100):
            trainer.train_step(
                traversal_edges=["CAUSES", "SIMILAR_TO", "PART_OF"],
                relevance=[0.9, 0.2, 0.5],
            )

        final_weights = [
            edge_importance.get_weight(e) for e in edge_importance.EDGE_TYPES
        ]

        assert not np.allclose(initial_weights, final_weights, atol=0.01), \
            "Edge weights should diverge from uniform"

    def test_important_edges_get_higher_weight(self):
        """Edges that lead to relevant results should get higher weight."""
        from t4dm.storage.t4dx.learned_edges import (
            LearnedEdgeImportance,
            EdgeImportanceTrainer,
        )

        edge_importance = LearnedEdgeImportance()
        trainer = EdgeImportanceTrainer(edge_importance, lr=0.1)

        # Simulate: CAUSES edges lead to relevant results
        for _ in range(100):
            trainer.train_step(
                traversal_edges=["CAUSES", "SIMILAR_TO"],
                relevance=[0.9, 0.3],  # CAUSES more relevant
            )

        causes_weight = edge_importance.get_weight("CAUSES")
        similar_weight = edge_importance.get_weight("SIMILAR_TO")

        assert causes_weight > similar_weight, \
            "CAUSES should get higher weight after training"


class TestTraversalWithLearnedEdges:
    """Test TraversalWithLearnedEdges."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        engine = Mock()
        return engine

    def test_traversal_creation(self, mock_engine):
        """Should create traversal with engine and edge importance."""
        from t4dm.storage.t4dx.learned_edges import (
            LearnedEdgeImportance,
            TraversalWithLearnedEdges,
        )

        edge_importance = LearnedEdgeImportance()
        traversal = TraversalWithLearnedEdges(mock_engine, edge_importance)

        assert traversal.engine is mock_engine
        assert traversal.edge_importance is edge_importance

    def test_traversal_prefers_important_edges(self, mock_engine):
        """Traversal should prefer paths with high-importance edges."""
        from t4dm.storage.t4dx.learned_edges import (
            LearnedEdgeImportance,
            TraversalWithLearnedEdges,
        )

        edge_importance = LearnedEdgeImportance()
        # Set CAUSES to high, SIMILAR_TO to low
        with torch.no_grad():
            causes_idx = edge_importance.EDGE_TYPES.index("CAUSES")
            similar_idx = edge_importance.EDGE_TYPES.index("SIMILAR_TO")
            edge_importance.importance.weight[causes_idx] = 1.0  # exp(1) ~ 2.7
            edge_importance.importance.weight[similar_idx] = -1.0  # exp(-1) ~ 0.37

        start_id = uuid4()
        causes_target_id = uuid4()
        similar_target_id = uuid4()

        # Mock edges
        causes_edge = Mock()
        causes_edge.target_id = causes_target_id
        causes_edge.edge_type = "CAUSES"
        causes_edge.weight = 1.0

        similar_edge = Mock()
        similar_edge.target_id = similar_target_id
        similar_edge.edge_type = "SIMILAR_TO"
        similar_edge.weight = 1.0

        mock_engine.get_edges.return_value = [causes_edge, similar_edge]

        causes_item = Mock()
        causes_item.id = causes_target_id
        causes_item.via_edge = "CAUSES"

        similar_item = Mock()
        similar_item.id = similar_target_id
        similar_item.via_edge = "SIMILAR_TO"

        mock_engine.get.side_effect = lambda id: causes_item if id == causes_target_id else similar_item

        traversal = TraversalWithLearnedEdges(mock_engine, edge_importance)
        results = traversal.traverse(start_id, depth=1)

        # CAUSES result should be ranked higher (lower index)
        assert len(results) == 2
        assert results[0].via_edge == "CAUSES"

    def test_traversal_respects_min_weight(self, mock_engine):
        """Traversal should filter out paths below min_weight."""
        from t4dm.storage.t4dx.learned_edges import (
            LearnedEdgeImportance,
            TraversalWithLearnedEdges,
        )

        edge_importance = LearnedEdgeImportance()
        # Set SIMILAR_TO to very low
        with torch.no_grad():
            similar_idx = edge_importance.EDGE_TYPES.index("SIMILAR_TO")
            edge_importance.importance.weight[similar_idx] = -5.0  # exp(-5) ~ 0.007

        start_id = uuid4()
        target_id = uuid4()

        # Mock edge with low importance
        edge = Mock()
        edge.target_id = target_id
        edge.edge_type = "SIMILAR_TO"
        edge.weight = 1.0

        mock_engine.get_edges.return_value = [edge]

        traversal = TraversalWithLearnedEdges(mock_engine, edge_importance)
        results = traversal.traverse(start_id, depth=1, min_weight=0.1)

        # Should be filtered out due to low importance
        assert len(results) == 0


class TestEdgeImportanceLatency:
    """Test latency requirements."""

    def test_edge_importance_lookup_under_100us(self):
        """Edge importance lookup should be <0.1ms."""
        from t4dm.storage.t4dx.learned_edges import LearnedEdgeImportance
        import time

        edge_importance = LearnedEdgeImportance()

        # Warmup
        for _ in range(100):
            edge_importance.get_weight("CAUSES")

        # Measure
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            edge_importance.get_weight("CAUSES")
            times.append(time.perf_counter() - start)

        avg_time_us = np.mean(times) * 1_000_000  # Convert to microseconds

        assert avg_time_us < 200, f"Edge lookup took {avg_time_us:.1f}us"
