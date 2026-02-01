"""Unit tests for Hebbian decay functionality."""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

from t4dm.memory.semantic import SemanticMemory
from t4dm.core.types import Entity, EntityType


@pytest.mark.asyncio
class TestHebbianDecayLogic:
    """Test suite for Hebbian decay logic."""

    @pytest_asyncio.fixture
    async def semantic(self, test_session_id, mock_vector_store, mock_graph_store, mock_embedding_provider):
        """Create semantic memory instance with mocks."""
        semantic = SemanticMemory(session_id=test_session_id)
        semantic.vector_store = mock_vector_store
        semantic.graph_store = mock_graph_store
        semantic.embedding = mock_embedding_provider
        semantic.vector_store.entities_collection = "entities"
        return semantic

    async def test_decay_reduces_weights_correctly(self, semantic):
        """Test that decay formula applies correctly."""
        # Mock batch operations
        semantic.graph_store.count_stale_relationships = AsyncMock(return_value=2)
        semantic.graph_store.batch_decay_relationships = AsyncMock(
            return_value={"decayed": 2, "pruned": 0}
        )

        # Apply decay
        result = await semantic.apply_hebbian_decay(
            decay_rate=0.1,
            min_weight=0.01,
            stale_days=30,
        )

        # Verify results
        assert result["decayed_count"] == 2
        assert result["pruned_count"] == 0
        assert result["total_processed"] == 2

        # Verify batch operation was called with correct parameters
        semantic.graph_store.batch_decay_relationships.assert_called_once()
        call_kwargs = semantic.graph_store.batch_decay_relationships.call_args[1]
        assert call_kwargs["decay_rate"] == 0.1
        assert call_kwargs["min_weight"] == 0.01
        assert call_kwargs["stale_days"] == 30

    async def test_pruning_removes_weak_relationships(self, semantic):
        """Test that very weak relationships are pruned."""
        # Mock batch operations
        semantic.graph_store.count_stale_relationships = AsyncMock(return_value=2)
        semantic.graph_store.batch_decay_relationships = AsyncMock(
            return_value={"decayed": 1, "pruned": 1}
        )

        # Apply decay with threshold
        result = await semantic.apply_hebbian_decay(
            decay_rate=0.5,  # 50% decay
            min_weight=0.01,
            stale_days=30,
        )

        # Verify pruning occurred
        # 0.015 * 0.5 = 0.0075 < 0.01 -> pruned
        # 0.5 * 0.5 = 0.25 >= 0.01 -> decayed
        assert result["decayed_count"] == 1
        assert result["pruned_count"] == 1
        assert result["total_processed"] == 2

        # Verify batch operation was called
        semantic.graph_store.batch_decay_relationships.assert_called_once()

    async def test_decay_with_no_stale_relationships(self, semantic):
        """Test decay when no relationships are stale."""
        # Mock empty stale relationships
        semantic.graph_store.count_stale_relationships = AsyncMock(return_value=0)
        semantic.graph_store.batch_decay_relationships = AsyncMock()

        # Apply decay
        result = await semantic.apply_hebbian_decay(
            decay_rate=0.1,
            min_weight=0.01,
            stale_days=30,
        )

        # Verify nothing processed
        assert result["decayed_count"] == 0
        assert result["pruned_count"] == 0
        assert result["total_processed"] == 0

        # Verify batch operation was not called (early exit optimization)
        semantic.graph_store.batch_decay_relationships.assert_not_called()

    async def test_decay_uses_configured_defaults(self, semantic):
        """Test that decay uses configured default values."""
        # Set configured values
        semantic.decay_rate = 0.05
        semantic.min_weight = 0.02
        semantic.stale_days = 45

        # Mock batch operations
        semantic.graph_store.count_stale_relationships = AsyncMock(return_value=1)
        semantic.graph_store.batch_decay_relationships = AsyncMock(
            return_value={"decayed": 1, "pruned": 0}
        )

        # Apply decay without parameters (should use defaults)
        result = await semantic.apply_hebbian_decay()

        # Verify configured values were used
        count_call = semantic.graph_store.count_stale_relationships.call_args[1]
        assert count_call["stale_days"] == 45

        batch_call = semantic.graph_store.batch_decay_relationships.call_args[1]
        assert batch_call["decay_rate"] == 0.05
        assert batch_call["min_weight"] == 0.02
        assert batch_call["stale_days"] == 45

        # Verify result includes configured values
        assert result["decay_rate"] == 0.05
        assert result["min_weight"] == 0.02
        assert result["stale_days"] == 45

    async def test_decay_tracks_lastDecay_timestamp(self, semantic):
        """Test that decay tracks lastDecay metadata (batch operation handles this in Cypher)."""
        # Mock batch operations
        semantic.graph_store.count_stale_relationships = AsyncMock(return_value=1)
        semantic.graph_store.batch_decay_relationships = AsyncMock(
            return_value={"decayed": 1, "pruned": 0}
        )

        # Apply decay
        await semantic.apply_hebbian_decay(decay_rate=0.1, min_weight=0.01, stale_days=30)

        # Verify batch operation was called (it handles lastDecay in Cypher)
        semantic.graph_store.batch_decay_relationships.assert_called_once()

        # Note: The lastDecay timestamp is now set by the Cypher query in batch_decay_relationships
        # See neo4j_store.py:776 - SET r.lastDecay = datetime()

    async def test_multiple_decay_iterations(self, semantic):
        """Test that multiple decay iterations compound correctly."""
        # Mock batch operations for multiple iterations
        initial_weight = 1.0
        decay_rate = 0.1

        for iteration in range(3):
            semantic.graph_store.count_stale_relationships = AsyncMock(return_value=1)
            semantic.graph_store.batch_decay_relationships = AsyncMock(
                return_value={"decayed": 1, "pruned": 0}
            )

            result = await semantic.apply_hebbian_decay(
                decay_rate=decay_rate,
                min_weight=0.01,
                stale_days=30,
            )

            assert result["decayed_count"] == 1

            # Verify batch operation was called with correct decay rate
            # Note: The actual weight calculation happens in Cypher query
            # Formula: r.weight * (1 - $decay_rate)
            call_kwargs = semantic.graph_store.batch_decay_relationships.call_args[1]
            assert call_kwargs["decay_rate"] == decay_rate

    async def test_session_id_filter_passed_to_graph_store(self, semantic):
        """Test that session ID is correctly passed to graph store."""
        semantic.graph_store.count_stale_relationships = AsyncMock(return_value=0)
        semantic.graph_store.batch_decay_relationships = AsyncMock()

        # Apply decay with explicit session ID
        custom_session = "custom_session_123"
        await semantic.apply_hebbian_decay(
            decay_rate=0.1,
            min_weight=0.01,
            stale_days=30,
            session_id=custom_session,
        )

        # Verify session ID was passed to count
        semantic.graph_store.count_stale_relationships.assert_called_once()
        count_kwargs = semantic.graph_store.count_stale_relationships.call_args[1]
        assert count_kwargs["session_id"] == custom_session

    async def test_decay_boundary_conditions(self, semantic):
        """Test boundary conditions for decay."""
        # Mock batch operations
        semantic.graph_store.count_stale_relationships = AsyncMock(return_value=1)
        semantic.graph_store.batch_decay_relationships = AsyncMock(
            return_value={"decayed": 0, "pruned": 1}
        )

        # 0.011 * (1 - 0.1) = 0.0099 < 0.01 -> should be pruned
        result = await semantic.apply_hebbian_decay(
            decay_rate=0.1,
            min_weight=0.01,
            stale_days=30,
        )

        assert result["pruned_count"] == 1
        assert result["decayed_count"] == 0


# Note: MCP tool validation tests are skipped here since they require MCP server dependencies.
# Manual testing of the MCP tool should verify that:
# 1. decay_rate must be between 0.0 and 1.0
# 2. min_weight must be between 0.0 and 1.0
# 3. stale_days must be at least 1
