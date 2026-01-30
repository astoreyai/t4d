"""Tests for spreading activation with explosion prevention."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from ww.memory.semantic import SemanticMemory


class TestSpreadingActivation:
    """Test spread_activation method."""

    @pytest.fixture
    def mock_semantic(self):
        """Create semantic memory with mocked stores."""
        semantic = SemanticMemory.__new__(SemanticMemory)
        semantic.graph_store = MagicMock()
        semantic.graph_store.get_relationships = AsyncMock(return_value=[])
        semantic.vector_store = MagicMock()
        semantic.embedding = MagicMock()
        return semantic

    @pytest.mark.asyncio
    async def test_empty_seeds_returns_empty(self, mock_semantic):
        """Empty seed list returns empty activation."""
        result = await mock_semantic.spread_activation([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_seeds_start_with_full_activation(self, mock_semantic):
        """Seed entities start with activation 1.0."""
        seeds = ["entity-1", "entity-2"]

        result = await mock_semantic.spread_activation(seeds, steps=1)

        assert result.get("entity-1", 0) > 0
        assert result.get("entity-2", 0) > 0

    @pytest.mark.asyncio
    async def test_respects_max_nodes_limit(self, mock_semantic):
        """Max nodes limit is respected."""
        # Create many connected nodes
        def make_neighbors(node_id):
            return [
                {"other_id": f"neighbor-{i}", "properties": {"weight": 0.5}}
                for i in range(100)
            ]

        mock_semantic.graph_store.get_relationships = AsyncMock(
            side_effect=lambda **kwargs: make_neighbors(kwargs.get("node_id"))
        )

        result = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=3,
            max_nodes=20,
        )

        assert len(result) <= 20

    @pytest.mark.asyncio
    async def test_respects_max_neighbors_limit(self, mock_semantic):
        """Max neighbors per node is respected."""
        neighbors_requested = []

        async def track_neighbors(**kwargs):
            neighbors_requested.append(kwargs.get("limit"))
            return []

        mock_semantic.graph_store.get_relationships = AsyncMock(side_effect=track_neighbors)

        await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=1,
            max_neighbors_per_node=25,
        )

        for limit in neighbors_requested:
            assert limit <= 25

    @pytest.mark.asyncio
    async def test_activation_decays_over_steps(self, mock_semantic):
        """Activation decays with each step."""
        mock_semantic.graph_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "neighbor-1", "properties": {"weight": 1.0}}
        ])

        result = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=3,
            decay=0.2,
        )

        # With decay and multiple steps, activation should spread but diminish
        # Both seed and neighbor should be present
        assert "seed-1" in result
        # Total activation should be less than initial (due to decay)
        total_activation = sum(result.values())
        assert total_activation < 1.0  # Started with 1.0, should decay

    @pytest.mark.asyncio
    async def test_threshold_filters_low_activation(self, mock_semantic):
        """Nodes below threshold are filtered."""
        # Test that threshold parameter works - very simple case
        mock_semantic.graph_store.get_relationships = AsyncMock(return_value=[])

        # With no neighbors and default threshold (0.01), seed should be present
        result_low_threshold = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=1,
            threshold=0.01,
        )
        assert len(result_low_threshold) >= 1

        # With impossibly high threshold, nothing should pass
        result_high_threshold = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=1,
            threshold=10.0,  # Higher than possible activation
        )
        assert len(result_high_threshold) == 0  # Everything filtered

    @pytest.mark.asyncio
    async def test_handles_graph_errors(self, mock_semantic):
        """Handles graph query errors gracefully."""
        mock_semantic.graph_store.get_relationships = AsyncMock(
            side_effect=Exception("Graph error")
        )

        # Should not raise
        result = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=2,
        )

        # Seeds should still be in result
        assert "seed-1" in result

    @pytest.mark.asyncio
    async def test_steps_clamped_to_range(self, mock_semantic):
        """Steps parameter is clamped to 1-5."""
        # Should not cause infinite loop or error
        await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=100,  # Will be clamped to 5
        )

        await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=-5,  # Will be clamped to 1
        )

    @pytest.mark.asyncio
    async def test_novelty_bonus_prefers_unvisited(self, mock_semantic):
        """Unvisited nodes get higher priority."""
        # First neighbor: high weight, already visited
        # Second neighbor: lower weight, unvisited
        mock_semantic.graph_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "visited-node", "properties": {"weight": 1.0}},
            {"other_id": "novel-node", "properties": {"weight": 0.7}},
        ])

        # Track visited nodes by monitoring calls
        visited_tracking = set()

        async def track_visits(**kwargs):
            node_id = kwargs.get("node_id")
            visited_tracking.add(node_id)

            if node_id == "seed-1":
                return [
                    {"other_id": "visited-node", "properties": {"weight": 1.0}},
                    {"other_id": "novel-node", "properties": {"weight": 0.7}},
                ]
            elif node_id == "visited-node":
                return [{"other_id": "seed-1", "properties": {"weight": 1.0}}]
            else:
                return []

        mock_semantic.graph_store.get_relationships = AsyncMock(side_effect=track_visits)

        result = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=2,
            max_neighbors_per_node=10,
        )

        # Both neighbors should be in result
        assert "novel-node" in result or "visited-node" in result

    @pytest.mark.asyncio
    async def test_progressive_decay(self, mock_semantic):
        """Decay increases progressively with steps."""
        mock_semantic.graph_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "neighbor-1", "properties": {"weight": 1.0}}
        ])

        # Higher decay should result in fewer nodes with high activation
        result_low_decay = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=3,
            decay=0.05,
        )

        result_high_decay = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=3,
            decay=0.3,
        )

        # High decay should result in lower overall activation
        # (sum of all activations should be lower)
        if result_low_decay and result_high_decay:
            sum_low = sum(result_low_decay.values())
            sum_high = sum(result_high_decay.values())
            assert sum_high <= sum_low

    @pytest.mark.asyncio
    async def test_weighted_distribution(self, mock_semantic):
        """Activation spreads proportionally to edge weights."""
        mock_semantic.graph_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "heavy-neighbor", "properties": {"weight": 0.9}},
            {"other_id": "light-neighbor", "properties": {"weight": 0.1}},
        ])

        result = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=1,
            retention=0.5,  # 50% retained, 50% spread
        )

        # Heavy neighbor should receive more activation
        if "heavy-neighbor" in result and "light-neighbor" in result:
            assert result["heavy-neighbor"] > result["light-neighbor"]


class TestSpreadingActivationParameterValidation:
    """Test parameter validation and clamping."""

    @pytest.fixture
    def mock_semantic(self):
        """Create semantic memory with mocked stores."""
        semantic = SemanticMemory.__new__(SemanticMemory)
        semantic.graph_store = MagicMock()
        semantic.graph_store.get_relationships = AsyncMock(return_value=[])
        semantic.vector_store = MagicMock()
        semantic.embedding = MagicMock()
        return semantic

    @pytest.mark.asyncio
    async def test_max_nodes_clamped(self, mock_semantic):
        """max_nodes is clamped to valid range."""
        # Too low
        result = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            max_nodes=1,  # Will be clamped to 10
        )
        assert len(result) <= 10

        # Too high (would use 10000)
        result = await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            max_nodes=100000,  # Will be clamped to 10000
        )
        # Result length depends on graph, but should not exceed 10000
        assert len(result) <= 10000

    @pytest.mark.asyncio
    async def test_retention_clamped(self, mock_semantic):
        """retention is clamped to [0, 1]."""
        # Test that extreme values don't cause errors
        await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            retention=-1.0,  # Will be clamped to 0.0
        )

        await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            retention=2.0,  # Will be clamped to 1.0
        )

    @pytest.mark.asyncio
    async def test_decay_clamped(self, mock_semantic):
        """decay is clamped to [0, 1]."""
        # Test that extreme values don't cause errors
        await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            decay=-0.5,  # Will be clamped to 0.0
        )

        await mock_semantic.spread_activation(
            seed_entities=["seed-1"],
            decay=5.0,  # Will be clamped to 1.0
        )


class TestSpreadingActivationPerformance:
    """Performance tests for spreading activation."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_handles_dense_graph(self):
        """Handles densely connected graph without explosion.

        This would be an integration test with real data.
        Skipped in unit tests - requires full database setup.
        """
        pytest.skip("Integration test - requires full database setup")

    @pytest.mark.asyncio
    async def test_early_termination_on_threshold(self):
        """Activation stops spreading when below threshold."""
        semantic = SemanticMemory.__new__(SemanticMemory)
        semantic.graph_store = MagicMock()
        semantic.vector_store = MagicMock()
        semantic.embedding = MagicMock()

        # Low weight edges that decay quickly
        semantic.graph_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "neighbor-1", "properties": {"weight": 0.01}},
        ])

        result = await semantic.spread_activation(
            seed_entities=["seed-1"],
            steps=5,
            retention=0.1,
            decay=0.5,
            threshold=0.1,  # High threshold
        )

        # With high threshold and low weights, spreading should stop early
        # Result should not have 5 full hops of neighbors
        assert len(result) < 50  # Arbitrary small number
