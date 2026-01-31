"""Tests for storage hooks module."""

import pytest
import time

from t4dm.hooks.storage import (
    StorageHook,
    ConnectionHook,
    QueryHook,
    ErrorHook,
    RetryHook,
    QueryTimingHook,
    ConnectionPoolMonitorHook,
    ExponentialBackoffRetryHook,
    CircuitBreakerHook,
    QueryCacheHook,
)
from t4dm.hooks.base import HookContext, HookPhase, HookPriority


class TestStorageHook:
    """Tests for StorageHook base class."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = ConnectionHook(
            name="test",
            storage_type="qdrant",
        )
        assert hook.name == "test"
        assert hook.storage_type == "qdrant"

    def test_should_execute_with_matching_type(self):
        """Test should_execute with matching storage type."""
        hook = ConnectionHook(name="test", storage_type="qdrant")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="connect",
            input_data={"storage_type": "qdrant"},
        )
        assert hook.should_execute(ctx) is True

    def test_should_execute_with_non_matching_type(self):
        """Test should_execute with non-matching storage type."""
        hook = ConnectionHook(name="test", storage_type="qdrant")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="connect",
            input_data={"storage_type": "neo4j"},
        )
        assert hook.should_execute(ctx) is False

    def test_should_execute_without_filter(self):
        """Test should_execute without storage type filter."""
        hook = ConnectionHook(name="test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="connect",
            input_data={"storage_type": "any"},
        )
        assert hook.should_execute(ctx) is True


class TestConnectionHook:
    """Tests for ConnectionHook class."""

    @pytest.mark.asyncio
    async def test_execute_connect(self):
        """Test connect event execution."""
        hook = ConnectionHook(name="conn_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="connection",
            input_data={
                "storage_type": "qdrant",
                "event": "connect",
                "connection_string": "localhost:6333",
                "pool_size": 5,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_disconnect(self):
        """Test disconnect event execution."""
        hook = ConnectionHook(name="conn_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="connection",
            input_data={
                "storage_type": "qdrant",
                "event": "disconnect",
                "reason": "shutdown",
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestQueryHook:
    """Tests for QueryHook class."""

    @pytest.mark.asyncio
    async def test_execute_pre(self):
        """Test pre-query execution."""
        hook = QueryHook(name="query_test")
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="query",
            input_data={
                "storage_type": "neo4j",
                "query": "MATCH (n) RETURN n",
                "query_type": "read",
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_execute_post(self):
        """Test post-query execution."""
        hook = QueryHook(name="query_test")
        ctx = HookContext(
            phase=HookPhase.POST,
            operation="query",
            input_data={"storage_type": "neo4j", "query_type": "read"},
            output_data={"row_count": 10, "duration_ms": 15.5},
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestErrorHook:
    """Tests for ErrorHook class."""

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test error hook execution."""
        hook = ErrorHook(name="error_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="error",
            input_data={
                "storage_type": "qdrant",
                "error_type": "connection",
                "error": ConnectionError("timeout"),
                "attempt": 2,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_on_error(self):
        """Test on_error callback."""
        hook = ErrorHook(name="error_test")
        ctx = HookContext(phase=HookPhase.ON, operation="test")
        ctx.set_error(ValueError("test error"))
        await hook.on_error(ctx)


class TestRetryHook:
    """Tests for RetryHook class."""

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test retry hook execution."""
        hook = RetryHook(name="retry_test")
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="retry",
            input_data={
                "storage_type": "neo4j",
                "operation": "query",
                "attempt": 2,
                "max_attempts": 3,
                "backoff_ms": 100,
            },
        )
        result = await hook.execute(ctx)
        assert result is ctx


class TestQueryTimingHook:
    """Tests for QueryTimingHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = QueryTimingHook(slow_query_threshold_ms=500)
        assert hook.name == "query_timing"
        assert hook.slow_query_threshold == 500
        assert hook.query_times == []

    @pytest.mark.asyncio
    async def test_timing_measurement(self):
        """Test query timing measurement."""
        hook = QueryTimingHook()

        # Pre-phase
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="query",
            input_data={"storage_type": "qdrant"},
        )
        await hook.execute(ctx)
        assert "query_start" in ctx.metadata

        # Simulate some time passing
        time.sleep(0.01)

        # Post-phase
        ctx.phase = HookPhase.POST
        ctx.output_data = {}
        await hook.execute(ctx)

        assert len(hook.query_times) == 1
        assert hook.query_times[0] >= 10  # At least 10ms

    def test_get_stats(self):
        """Test getting timing statistics."""
        hook = QueryTimingHook()
        hook.query_times = [10, 20, 30, 40, 50]

        stats = hook.get_stats()
        assert "timing" in stats
        assert stats["timing"]["count"] == 5
        assert stats["timing"]["avg_ms"] == 30
        assert stats["timing"]["min_ms"] == 10
        assert stats["timing"]["max_ms"] == 50


class TestConnectionPoolMonitorHook:
    """Tests for ConnectionPoolMonitorHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = ConnectionPoolMonitorHook()
        assert hook.name == "connection_pool_monitor"
        assert hook.active_connections == 0

    @pytest.mark.asyncio
    async def test_connect_event(self):
        """Test connection count on connect."""
        hook = ConnectionPoolMonitorHook()
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="connection",
            input_data={"event": "connect"},
        )
        await hook.execute(ctx)
        assert hook.active_connections == 1
        assert ctx.metadata["active_connections"] == 1

    @pytest.mark.asyncio
    async def test_disconnect_event(self):
        """Test connection count on disconnect."""
        hook = ConnectionPoolMonitorHook()
        hook.active_connections = 5

        ctx = HookContext(
            phase=HookPhase.ON,
            operation="connection",
            input_data={"event": "disconnect"},
        )
        await hook.execute(ctx)
        assert hook.active_connections == 4


class TestExponentialBackoffRetryHook:
    """Tests for ExponentialBackoffRetryHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = ExponentialBackoffRetryHook(
            base_delay_ms=200,
            max_delay_ms=10000,
            multiplier=3.0,
        )
        assert hook.name == "exponential_backoff"
        assert hook.base_delay == 200
        assert hook.max_delay == 10000
        assert hook.multiplier == 3.0

    @pytest.mark.asyncio
    async def test_backoff_calculation(self):
        """Test backoff delay calculation."""
        hook = ExponentialBackoffRetryHook(
            base_delay_ms=100,
            max_delay_ms=1000,
            multiplier=2.0,
        )

        # Attempt 1: 100ms
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="retry",
            input_data={"attempt": 1},
        )
        await hook.execute(ctx)
        assert ctx.input_data["backoff_ms"] == 100

        # Attempt 2: 200ms
        ctx.input_data["attempt"] = 2
        await hook.execute(ctx)
        assert ctx.input_data["backoff_ms"] == 200

        # Attempt 5: should cap at max (1000ms)
        ctx.input_data["attempt"] = 5
        await hook.execute(ctx)
        assert ctx.input_data["backoff_ms"] == 1000


class TestCircuitBreakerHook:
    """Tests for CircuitBreakerHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = CircuitBreakerHook(failure_threshold=3, reset_timeout_seconds=30)
        assert hook.name == "circuit_breaker"
        assert hook.failure_threshold == 3
        assert hook.reset_timeout == 30
        assert hook.failures == {}
        assert hook.open_circuits == set()

    @pytest.mark.asyncio
    async def test_circuit_opens_on_threshold(self):
        """Test circuit opens when threshold is reached."""
        hook = CircuitBreakerHook(failure_threshold=3)

        for i in range(3):
            ctx = HookContext(
                phase=HookPhase.ON,
                operation="error",
                input_data={"storage_type": "qdrant"},
            )
            await hook.execute(ctx)

        assert hook.is_open("qdrant") is True

    @pytest.mark.asyncio
    async def test_circuit_state_in_metadata(self):
        """Test circuit state is added to metadata."""
        hook = CircuitBreakerHook(failure_threshold=1)
        ctx = HookContext(
            phase=HookPhase.ON,
            operation="error",
            input_data={"storage_type": "neo4j"},
        )
        await hook.execute(ctx)
        assert ctx.metadata["circuit_open"] is True


class TestQueryCacheHook:
    """Tests for QueryCacheHook example implementation."""

    def test_initialization(self):
        """Test hook initialization."""
        hook = QueryCacheHook(cache_size=500, ttl_seconds=600)
        assert hook.name == "query_cache"
        assert hook.cache_size == 500
        assert hook.ttl == 600
        assert hook.hits == 0
        assert hook.misses == 0

    @pytest.mark.asyncio
    async def test_cache_miss_read_query(self):
        """Test cache miss for read query."""
        hook = QueryCacheHook()
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="query",
            input_data={
                "query_type": "read",
                "query": "SELECT *",
                "parameters": {},
            },
        )
        await hook.execute(ctx)
        assert ctx.metadata["cache_hit"] is False
        assert hook.misses == 1

    @pytest.mark.asyncio
    async def test_skip_non_read_queries(self):
        """Test caching is skipped for non-read queries."""
        hook = QueryCacheHook()
        ctx = HookContext(
            phase=HookPhase.PRE,
            operation="query",
            input_data={"query_type": "write"},
        )
        await hook.execute(ctx)
        # Should not set cache_hit metadata
        assert "cache_hit" not in ctx.metadata
