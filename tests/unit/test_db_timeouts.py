"""
Unit tests for database timeout handling in Neo4j and Qdrant stores.

Tests cover:
1. Normal operations complete within timeout
2. Slow operations raise DatabaseTimeoutError
3. DatabaseTimeoutError contains correct operation name and timeout value
4. Timeout is configurable via constructor parameter
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from typing import Any

from t4dm.storage.neo4j_store import (
    Neo4jStore,
    DatabaseTimeoutError as Neo4jDatabaseTimeoutError,
    DEFAULT_DB_TIMEOUT as NEO4J_DEFAULT_TIMEOUT,
)
from t4dm.storage.qdrant_store import (
    QdrantStore,
    DatabaseTimeoutError as QdrantDatabaseTimeoutError,
    DEFAULT_DB_TIMEOUT as QDRANT_DEFAULT_TIMEOUT,
)


# ============================================================================
# Neo4j Timeout Tests
# ============================================================================


class TestNeo4jDatabaseTimeoutError:
    """Test DatabaseTimeoutError exception class."""

    def test_initialization(self):
        """Test DatabaseTimeoutError is initialized with correct attributes."""
        error = Neo4jDatabaseTimeoutError(operation="create_node", timeout=30)

        assert error.operation == "create_node"
        assert error.timeout == 30
        assert str(error) == "Database operation 'create_node' timed out after 30s"

    def test_error_message_formatting(self):
        """Test error message contains operation name and timeout."""
        error = Neo4jDatabaseTimeoutError(operation="search_entities", timeout=15.5)

        assert "search_entities" in str(error)
        assert "15.5" in str(error)
        assert "timed out" in str(error)

    def test_error_inheritance(self):
        """Test DatabaseTimeoutError is an Exception."""
        error = Neo4jDatabaseTimeoutError(operation="test", timeout=10)
        assert isinstance(error, Exception)


class TestNeo4jWithTimeoutHelper:
    """Test Neo4jStore._with_timeout helper method."""

    @pytest.mark.asyncio
    async def test_with_timeout_fast_operation_completes(self):
        """Test that operations completing within timeout are returned successfully."""
        store = Neo4jStore(timeout=5.0)

        async def fast_operation():
            await asyncio.sleep(0.1)
            return "success_value"

        result = await store._with_timeout(fast_operation(), "fast_op")

        assert result == "success_value"

    @pytest.mark.asyncio
    async def test_with_timeout_slow_operation_raises_error(self):
        """Test that operations exceeding timeout raise DatabaseTimeoutError."""
        store = Neo4jStore(timeout=0.1)

        async def slow_operation():
            await asyncio.sleep(1.0)
            return "should_not_get_here"

        with pytest.raises(Neo4jDatabaseTimeoutError) as exc_info:
            await store._with_timeout(slow_operation(), "slow_op")

        assert exc_info.value.operation == "slow_op"
        assert exc_info.value.timeout == 0.1

    @pytest.mark.asyncio
    async def test_with_timeout_preserves_return_value(self):
        """Test that _with_timeout preserves the return value of the coroutine."""
        store = Neo4jStore(timeout=5.0)

        test_data = {"key": "value", "number": 42}

        async def operation_with_result():
            await asyncio.sleep(0.05)
            return test_data

        result = await store._with_timeout(operation_with_result(), "test_op")

        assert result == test_data
        assert result is test_data

    @pytest.mark.asyncio
    async def test_with_timeout_preserves_exception_from_operation(self):
        """Test that exceptions from the coroutine are preserved."""
        store = Neo4jStore(timeout=5.0)

        class CustomException(Exception):
            pass

        async def failing_operation():
            raise CustomException("operation failed")

        with pytest.raises(CustomException):
            await store._with_timeout(failing_operation(), "failing_op")

    @pytest.mark.asyncio
    async def test_with_timeout_error_message_includes_operation_name(self):
        """Test that timeout error message includes the operation name."""
        store = Neo4jStore(timeout=0.1)

        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises(Neo4jDatabaseTimeoutError) as exc_info:
            await store._with_timeout(slow_op(), "create_relationship")

        assert "create_relationship" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_with_timeout_respects_configured_timeout(self):
        """Test that _with_timeout respects the timeout value passed to constructor."""
        custom_timeout = 0.15
        store = Neo4jStore(timeout=custom_timeout)

        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises(Neo4jDatabaseTimeoutError) as exc_info:
            await store._with_timeout(slow_op(), "test")

        assert exc_info.value.timeout == custom_timeout


class TestNeo4jTimeoutConfigurability:
    """Test that Neo4jStore timeout is configurable."""

    def test_default_timeout_value(self):
        """Test that default timeout is set to DEFAULT_DB_TIMEOUT."""
        store = Neo4jStore()
        assert store.timeout == NEO4J_DEFAULT_TIMEOUT

    def test_custom_timeout_in_constructor(self):
        """Test that custom timeout can be passed to constructor."""
        custom_timeout = 45.0
        store = Neo4jStore(timeout=custom_timeout)
        assert store.timeout == custom_timeout

    def test_timeout_can_be_very_small(self):
        """Test that very small timeouts can be configured."""
        store = Neo4jStore(timeout=0.01)
        assert store.timeout == 0.01

    def test_timeout_can_be_very_large(self):
        """Test that very large timeouts can be configured."""
        store = Neo4jStore(timeout=3600.0)
        assert store.timeout == 3600.0

    @pytest.mark.asyncio
    async def test_timeout_affects_operation_behavior(self):
        """Test that configured timeout actually affects when operations fail."""
        short_timeout_store = Neo4jStore(timeout=0.05)
        long_timeout_store = Neo4jStore(timeout=5.0)

        async def variable_delay_op():
            await asyncio.sleep(0.2)

        # Short timeout should fail
        with pytest.raises(Neo4jDatabaseTimeoutError):
            await short_timeout_store._with_timeout(variable_delay_op(), "test")

        # Long timeout should succeed
        result = await long_timeout_store._with_timeout(variable_delay_op(), "test")
        assert result is None


# ============================================================================
# Qdrant Timeout Tests
# ============================================================================


class TestQdrantDatabaseTimeoutError:
    """Test DatabaseTimeoutError exception class for Qdrant."""

    def test_initialization(self):
        """Test DatabaseTimeoutError is initialized with correct attributes."""
        error = QdrantDatabaseTimeoutError(operation="search", timeout=30)

        assert error.operation == "search"
        assert error.timeout == 30
        assert str(error) == "Database operation 'search' timed out after 30s"

    def test_error_message_formatting(self):
        """Test error message contains operation name and timeout."""
        error = QdrantDatabaseTimeoutError(operation="batch_add", timeout=20.5)

        assert "batch_add" in str(error)
        assert "20.5" in str(error)
        assert "timed out" in str(error)

    def test_error_inheritance(self):
        """Test DatabaseTimeoutError is an Exception."""
        error = QdrantDatabaseTimeoutError(operation="test", timeout=10)
        assert isinstance(error, Exception)


class TestQdrantWithTimeoutHelper:
    """Test QdrantStore._with_timeout helper method."""

    @pytest.mark.asyncio
    async def test_with_timeout_fast_operation_completes(self):
        """Test that operations completing within timeout are returned successfully."""
        store = QdrantStore(timeout=5.0)

        async def fast_operation():
            await asyncio.sleep(0.1)
            return "vector_results"

        result = await store._with_timeout(fast_operation(), "search_op")

        assert result == "vector_results"

    @pytest.mark.asyncio
    async def test_with_timeout_slow_operation_raises_error(self):
        """Test that operations exceeding timeout raise DatabaseTimeoutError."""
        store = QdrantStore(timeout=0.1)

        async def slow_operation():
            await asyncio.sleep(1.0)
            return "should_not_get_here"

        with pytest.raises(QdrantDatabaseTimeoutError) as exc_info:
            await store._with_timeout(slow_operation(), "slow_search")

        assert exc_info.value.operation == "slow_search"
        assert exc_info.value.timeout == 0.1

    @pytest.mark.asyncio
    async def test_with_timeout_preserves_return_value(self):
        """Test that _with_timeout preserves the return value of the coroutine."""
        store = QdrantStore(timeout=5.0)

        test_results = [("id-1", 0.95, {"data": "payload"}), ("id-2", 0.87, {})]

        async def search_operation():
            await asyncio.sleep(0.05)
            return test_results

        result = await store._with_timeout(search_operation(), "test_search")

        assert result == test_results
        assert result is test_results

    @pytest.mark.asyncio
    async def test_with_timeout_preserves_exception_from_operation(self):
        """Test that exceptions from the coroutine are preserved."""
        store = QdrantStore(timeout=5.0)

        class VectorError(Exception):
            pass

        async def failing_operation():
            raise VectorError("vector processing failed")

        with pytest.raises(VectorError):
            await store._with_timeout(failing_operation(), "failing_search")

    @pytest.mark.asyncio
    async def test_with_timeout_error_message_includes_operation_name(self):
        """Test that timeout error message includes the operation name."""
        store = QdrantStore(timeout=0.1)

        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises(QdrantDatabaseTimeoutError) as exc_info:
            await store._with_timeout(slow_op(), "add(episodic_collection)")

        assert "add(episodic_collection)" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_with_timeout_respects_configured_timeout(self):
        """Test that _with_timeout respects the timeout value passed to constructor."""
        custom_timeout = 0.2
        store = QdrantStore(timeout=custom_timeout)

        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises(QdrantDatabaseTimeoutError) as exc_info:
            await store._with_timeout(slow_op(), "test")

        assert exc_info.value.timeout == custom_timeout


class TestQdrantTimeoutConfigurability:
    """Test that QdrantStore timeout is configurable."""

    def test_default_timeout_value(self):
        """Test that default timeout is set to DEFAULT_DB_TIMEOUT."""
        store = QdrantStore()
        assert store.timeout == QDRANT_DEFAULT_TIMEOUT

    def test_custom_timeout_in_constructor(self):
        """Test that custom timeout can be passed to constructor."""
        custom_timeout = 60.0
        store = QdrantStore(timeout=custom_timeout)
        assert store.timeout == custom_timeout

    def test_timeout_can_be_very_small(self):
        """Test that very small timeouts can be configured."""
        store = QdrantStore(timeout=0.01)
        assert store.timeout == 0.01

    def test_timeout_can_be_very_large(self):
        """Test that very large timeouts can be configured."""
        store = QdrantStore(timeout=7200.0)
        assert store.timeout == 7200.0

    @pytest.mark.asyncio
    async def test_timeout_affects_operation_behavior(self):
        """Test that configured timeout actually affects when operations fail."""
        short_timeout_store = QdrantStore(timeout=0.05)
        long_timeout_store = QdrantStore(timeout=5.0)

        async def variable_delay_op():
            await asyncio.sleep(0.15)

        # Short timeout should fail
        with pytest.raises(QdrantDatabaseTimeoutError):
            await short_timeout_store._with_timeout(variable_delay_op(), "search")

        # Long timeout should succeed
        result = await long_timeout_store._with_timeout(variable_delay_op(), "search")
        assert result is None


# ============================================================================
# Cross-Store Timeout Comparison Tests
# ============================================================================


class TestTimeoutConsistencyAcrossStores:
    """Test that timeout behavior is consistent between Neo4j and Qdrant stores."""

    def test_both_stores_have_same_default_timeout(self):
        """Test that both stores have the same default timeout value."""
        neo4j_store = Neo4jStore()
        qdrant_store = QdrantStore()

        assert neo4j_store.timeout == NEO4J_DEFAULT_TIMEOUT
        assert qdrant_store.timeout == QDRANT_DEFAULT_TIMEOUT
        assert NEO4J_DEFAULT_TIMEOUT == QDRANT_DEFAULT_TIMEOUT == 30

    def test_both_stores_accept_custom_timeout(self):
        """Test that both stores accept custom timeout values."""
        custom_timeout = 45.0

        neo4j_store = Neo4jStore(timeout=custom_timeout)
        qdrant_store = QdrantStore(timeout=custom_timeout)

        assert neo4j_store.timeout == custom_timeout
        assert qdrant_store.timeout == custom_timeout

    @pytest.mark.asyncio
    async def test_both_stores_timeout_similarly(self):
        """Test that both stores timeout with similar behavior."""
        timeout = 0.1

        neo4j_store = Neo4jStore(timeout=timeout)
        qdrant_store = QdrantStore(timeout=timeout)

        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises(Neo4jDatabaseTimeoutError) as neo4j_exc:
            await neo4j_store._with_timeout(slow_op(), "test")

        with pytest.raises(QdrantDatabaseTimeoutError) as qdrant_exc:
            await qdrant_store._with_timeout(slow_op(), "test")

        # Both should have the same timeout value in error
        assert neo4j_exc.value.timeout == timeout
        assert qdrant_exc.value.timeout == timeout

    @pytest.mark.asyncio
    async def test_both_stores_preserve_return_values(self):
        """Test that both stores preserve return values correctly."""
        neo4j_store = Neo4jStore(timeout=5.0)
        qdrant_store = QdrantStore(timeout=5.0)

        neo4j_result = {"id": "node-1", "data": "test"}
        qdrant_result = [("id-1", 0.95, {"payload": "data"})]

        async def neo4j_op():
            await asyncio.sleep(0.05)
            return neo4j_result

        async def qdrant_op():
            await asyncio.sleep(0.05)
            return qdrant_result

        neo4j_value = await neo4j_store._with_timeout(neo4j_op(), "test")
        qdrant_value = await qdrant_store._with_timeout(qdrant_op(), "test")

        assert neo4j_value == neo4j_result
        assert qdrant_value == qdrant_result


# ============================================================================
# Edge Cases and Special Scenarios
# ============================================================================


class TestTimeoutEdgeCases:
    """Test edge cases and special timeout scenarios."""

    @pytest.mark.asyncio
    async def test_timeout_at_exact_boundary(self):
        """Test operation that completes at exactly the timeout boundary."""
        store = Neo4jStore(timeout=0.2)

        async def operation_at_boundary():
            # Sleep slightly less than timeout to ensure completion
            await asyncio.sleep(0.19)
            return "completed"

        result = await store._with_timeout(operation_at_boundary(), "boundary_test")
        assert result == "completed"

    @pytest.mark.asyncio
    async def test_timeout_with_very_small_timeout(self):
        """Test behavior with very small timeout (should timeout quickly)."""
        store = QdrantStore(timeout=0.001)

        async def any_operation():
            await asyncio.sleep(0.1)
            return "result"

        with pytest.raises(QdrantDatabaseTimeoutError):
            await store._with_timeout(any_operation(), "zero_timeout_test")

    @pytest.mark.asyncio
    async def test_multiple_timeouts_in_sequence(self):
        """Test that multiple timeout-protected operations work correctly."""
        store = Neo4jStore(timeout=0.2)

        async def operation_1():
            await asyncio.sleep(0.05)
            return "result_1"

        async def operation_2():
            await asyncio.sleep(0.08)
            return "result_2"

        async def operation_3():
            await asyncio.sleep(0.1)
            return "result_3"

        result_1 = await store._with_timeout(operation_1(), "op_1")
        result_2 = await store._with_timeout(operation_2(), "op_2")
        result_3 = await store._with_timeout(operation_3(), "op_3")

        assert result_1 == "result_1"
        assert result_2 == "result_2"
        assert result_3 == "result_3"

    @pytest.mark.asyncio
    async def test_timeout_with_exception_in_operation(self):
        """Test that timeout doesn't mask exceptions from operations."""
        store = QdrantStore(timeout=5.0)

        class DatabaseError(Exception):
            pass

        async def failing_operation():
            await asyncio.sleep(0.01)
            raise DatabaseError("Connection failed")

        with pytest.raises(DatabaseError) as exc_info:
            await store._with_timeout(failing_operation(), "failing_op")

        assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_different_operations_with_different_durations(self):
        """Test that timeout is applied consistently across operations."""
        store = Neo4jStore(timeout=0.15)

        durations = [0.05, 0.08, 0.12]

        for duration in durations:
            async def operation():
                await asyncio.sleep(duration)
                return duration

            result = await store._with_timeout(operation(), f"op_{duration}")
            assert result == duration

        # Last one should timeout
        async def timeout_operation():
            await asyncio.sleep(0.20)

        with pytest.raises(Neo4jDatabaseTimeoutError):
            await store._with_timeout(timeout_operation(), "timeout_op")


class TestTimeoutErrorDetails:
    """Test that timeout errors contain accurate diagnostic information."""

    @pytest.mark.asyncio
    async def test_error_contains_exact_timeout_value(self):
        """Test that error contains the exact timeout value used."""
        timeout_values = [0.05, 0.1, 0.2, 0.5]

        for timeout in timeout_values:
            store = Neo4jStore(timeout=timeout)

            async def slow_op():
                await asyncio.sleep(1.0)

            with pytest.raises(Neo4jDatabaseTimeoutError) as exc_info:
                await store._with_timeout(slow_op(), "test")

            assert exc_info.value.timeout == timeout

    @pytest.mark.asyncio
    async def test_error_operation_name_matches_provided_name(self):
        """Test that error operation name matches what was provided."""
        store = QdrantStore(timeout=0.1)

        operation_names = [
            "search(episodes)",
            "add(entities)",
            "batch_delete",
            "update_payload(procedures)",
        ]

        async def slow_op():
            await asyncio.sleep(1.0)

        for op_name in operation_names:
            with pytest.raises(QdrantDatabaseTimeoutError) as exc_info:
                await store._with_timeout(slow_op(), op_name)

            assert exc_info.value.operation == op_name

    @pytest.mark.asyncio
    async def test_error_string_representation(self):
        """Test that error string representation is informative."""
        store = Neo4jStore(timeout=0.15)

        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises(Neo4jDatabaseTimeoutError) as exc_info:
            await store._with_timeout(slow_op(), "critical_operation")

        error_str = str(exc_info.value)
        assert "critical_operation" in error_str
        assert "0.15" in error_str
        assert "timed out" in error_str.lower()


class TestTimeoutWithMockedOperations:
    """Test timeout behavior with mocked async operations."""

    @pytest.mark.asyncio
    async def test_mocked_slow_neo4j_operation(self):
        """Test timeout with mocked slow Neo4j operation."""
        store = Neo4jStore(timeout=0.1)

        async def mocked_slow_operation():
            await asyncio.sleep(1.0)
            return {"result": "never_reached"}

        with pytest.raises(Neo4jDatabaseTimeoutError) as exc_info:
            await store._with_timeout(mocked_slow_operation(), "mocked_operation")

        assert exc_info.value.operation == "mocked_operation"
        assert exc_info.value.timeout == 0.1

    @pytest.mark.asyncio
    async def test_mocked_fast_qdrant_operation(self):
        """Test that fast mocked operations complete successfully."""
        store = QdrantStore(timeout=5.0)

        async def mocked_fast_operation():
            await asyncio.sleep(0.05)
            return [("id-1", 0.95, {})]

        result = await store._with_timeout(mocked_fast_operation(), "mocked_search")
        assert result == [("id-1", 0.95, {})]


class TestTimeoutRecovery:
    """Test that store remains usable after timeout errors."""

    @pytest.mark.asyncio
    async def test_store_usable_after_timeout_error(self):
        """Test that store can be used for subsequent operations after timeout."""
        store = Neo4jStore(timeout=0.1)

        # First operation times out
        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises(Neo4jDatabaseTimeoutError):
            await store._with_timeout(slow_op(), "first_op")

        # Second operation succeeds
        async def fast_op():
            await asyncio.sleep(0.05)
            return "success"

        result = await store._with_timeout(fast_op(), "second_op")
        assert result == "success"

    @pytest.mark.asyncio
    async def test_qdrant_store_usable_after_timeout(self):
        """Test that Qdrant store is usable after timeout error."""
        store = QdrantStore(timeout=0.1)

        # First operation times out
        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises(QdrantDatabaseTimeoutError):
            await store._with_timeout(slow_op(), "slow_search")

        # Second operation succeeds
        async def fast_op():
            await asyncio.sleep(0.05)
            return [("id", 0.9, {})]

        result = await store._with_timeout(fast_op(), "fast_search")
        assert len(result) == 1
