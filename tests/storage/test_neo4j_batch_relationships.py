"""Tests for P4.2 batch_create_relationships operation."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ww.storage.neo4j_store import Neo4jStore
from ww.storage.resilience import CircuitBreaker, CircuitBreakerConfig


class TestBatchCreateRelationships:
    """Test suite for batch_create_relationships (P4.2 N+1 fix)."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        tx = MagicMock()

        # Mock transaction context manager
        tx.__aenter__ = AsyncMock(return_value=tx)
        tx.__aexit__ = AsyncMock(return_value=None)
        tx.run = AsyncMock(return_value=MagicMock(single=AsyncMock(return_value={"created": 5})))
        tx.commit = AsyncMock()

        # Mock session context manager
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.begin_transaction = AsyncMock(return_value=tx)

        driver.session = MagicMock(return_value=session)
        return driver

    @pytest.fixture
    def store(self, mock_driver):
        """Create Neo4j store with mocked driver."""
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver
        store._initialized = True
        store.database = "neo4j"
        store.timeout = 30.0
        store._get_driver = AsyncMock(return_value=mock_driver)
        # Add circuit breaker (required by _with_timeout)
        store._circuit_breaker = CircuitBreaker("test_neo4j")
        return store

    @pytest.mark.asyncio
    async def test_empty_relationships_returns_zero(self, store):
        """Test that empty input returns 0."""
        result = await store.batch_create_relationships([])
        assert result == 0

    @pytest.mark.asyncio
    async def test_batch_create_single_relationship(self, store):
        """Test creating a single relationship."""
        relationships = [
            ("node1", "node2", "RELATES_TO", {"weight": 1.0})
        ]

        result = await store.batch_create_relationships(relationships)

        # Should return count from UNWIND query
        assert result == 5  # Mocked return value

    @pytest.mark.asyncio
    async def test_batch_create_multiple_same_type(self, store):
        """Test creating multiple relationships of same type."""
        relationships = [
            ("node1", "node2", "SOURCE_OF", {"weight": 1.0}),
            ("node2", "node3", "SOURCE_OF", {"weight": 0.8}),
            ("node3", "node4", "SOURCE_OF", {"weight": 0.5}),
        ]

        result = await store.batch_create_relationships(relationships)
        assert result >= 0

    @pytest.mark.asyncio
    async def test_batch_create_multiple_types_groups_correctly(self, store, mock_driver):
        """Test that different relationship types are grouped."""
        relationships = [
            ("node1", "node2", "SOURCE_OF", {}),
            ("node2", "node3", "RELATES_TO", {}),
            ("node3", "node4", "SOURCE_OF", {}),  # Same as first type
        ]

        await store.batch_create_relationships(relationships)

        # Get the transaction from the mock chain
        session = mock_driver.session.return_value
        tx = await session.begin_transaction()

        # Should have 2 tx.run calls (one for each relationship type)
        assert tx.run.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_size_limit_exceeded(self, store):
        """Test that exceeding batch size raises ValueError."""
        # Create 1001 relationships (exceeds 1000 limit)
        relationships = [
            (f"node{i}", f"node{i+1}", "RELATES_TO", {})
            for i in range(1001)
        ]

        with pytest.raises(ValueError, match="exceeds maximum"):
            await store.batch_create_relationships(relationships)

    @pytest.mark.asyncio
    async def test_invalid_relationship_type_rejected(self, store):
        """Test that invalid relationship types are rejected."""
        relationships = [
            ("node1", "node2", "INVALID TYPE WITH SPACES", {})
        ]

        with pytest.raises(ValueError):
            await store.batch_create_relationships(relationships)

    @pytest.mark.asyncio
    async def test_batch_create_with_properties(self, store, mock_driver):
        """Test that properties are passed to UNWIND query."""
        relationships = [
            ("ep1", "entity1", "SOURCE_OF", {
                "weight": 0.95,
                "coAccessCount": 1,
                "lastCoAccess": "2025-12-31T12:00:00",
            }),
        ]

        await store.batch_create_relationships(relationships)

        # Verify query was called with relationship data
        session = mock_driver.session.return_value
        tx = await session.begin_transaction()
        tx.run.assert_called()

        # Check the call args contain rels parameter
        call_args = tx.run.call_args
        assert "rels" in call_args.kwargs or (len(call_args.args) > 1 and "rels" in call_args.args[1])

    @pytest.mark.asyncio
    async def test_batch_create_none_properties_handled(self, store):
        """Test that None properties are converted to empty dict."""
        relationships = [
            ("node1", "node2", "RELATES_TO", None)
        ]

        # Should not raise
        result = await store.batch_create_relationships(relationships)
        assert result >= 0


class TestBatchCreateRelationshipsPerformance:
    """Performance characteristics tests for batch operations."""

    @pytest.fixture
    def mock_driver(self):
        """Create mock driver with call counting."""
        driver = MagicMock()
        session = MagicMock()
        tx = MagicMock()

        tx.__aenter__ = AsyncMock(return_value=tx)
        tx.__aexit__ = AsyncMock(return_value=None)
        tx.run = AsyncMock(return_value=MagicMock(single=AsyncMock(return_value={"created": 10})))
        tx.commit = AsyncMock()

        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.begin_transaction = AsyncMock(return_value=tx)

        driver.session = MagicMock(return_value=session)
        return driver

    @pytest.fixture
    def store_with_counter(self, mock_driver):
        """Create store that counts driver calls."""
        store = Neo4jStore.__new__(Neo4jStore)
        store._driver = mock_driver
        store._initialized = True
        store.database = "neo4j"
        store.timeout = 30.0
        store._get_driver = AsyncMock(return_value=mock_driver)
        store._circuit_breaker = CircuitBreaker("test_neo4j")
        store._call_count = 0
        return store

    @pytest.mark.asyncio
    async def test_batch_reduces_queries_vs_individual(self, store_with_counter, mock_driver):
        """Test that batch uses fewer queries than individual calls would.

        P4.2: 100 relationships should use 1 query, not 100.
        """
        relationships = [
            (f"ep{i}", f"entity{i}", "SOURCE_OF", {"weight": 0.9})
            for i in range(100)
        ]

        await store_with_counter.batch_create_relationships(relationships)

        # Get transaction and check call count
        session = mock_driver.session.return_value
        tx = await session.begin_transaction()

        # With batch UNWIND, should be 1 call (all same type)
        # vs 100 individual create_relationship calls
        assert tx.run.call_count == 1
