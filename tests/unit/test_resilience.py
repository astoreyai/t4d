"""
Unit tests for World Weaver storage resilience module.

Tests CircuitBreaker, InMemoryFallback, GracefulDegradation, and PendingOperation.
"""

import pytest
import asyncio
from datetime import datetime

from ww.storage.resilience import (
    CircuitState,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    InMemoryFallback,
    GracefulDegradation,
    PendingOperation,
    get_circuit_breaker,
    get_graceful_degradation,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self):
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.reset_timeout == 60.0
        assert config.success_threshold == 2

    def test_custom_values(self):
        config = CircuitBreakerConfig(
            failure_threshold=10,
            reset_timeout=120.0,
            success_threshold=5
        )
        assert config.failure_threshold == 10
        assert config.reset_timeout == 120.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            reset_timeout=1.0,
            success_threshold=2
        )
        return CircuitBreaker("test_unique", config)

    def test_creation(self, breaker):
        assert breaker.name == "test_unique"
        assert breaker.state == CircuitState.CLOSED

    def test_initial_state_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_open is False

    @pytest.mark.asyncio
    async def test_success_stays_closed(self, breaker):
        async def succeeding():
            return "ok"

        result = await breaker.execute(succeeding)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_opens_after_threshold(self, breaker):
        async def failing():
            raise Exception("Fail")

        # Record failures up to threshold
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.execute(failing)

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True

    @pytest.mark.asyncio
    async def test_open_circuit_raises_error(self, breaker):
        async def failing():
            raise Exception("Fail")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.execute(failing)

        # Next call should raise CircuitBreakerError
        async def succeeding():
            return "ok"

        with pytest.raises(CircuitBreakerError):
            await breaker.execute(succeeding)

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, breaker):
        async def failing():
            raise Exception("Fail")

        # Force open
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.execute(failing)

        assert breaker.state == CircuitState.OPEN

        # Wait for reset timeout
        await asyncio.sleep(1.1)

        # Check state should transition to half-open
        await breaker._check_state()
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_closes(self, breaker):
        async def failing():
            raise Exception("Fail")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.execute(failing)

        # Wait for timeout
        await asyncio.sleep(1.1)
        await breaker._check_state()
        assert breaker.state == CircuitState.HALF_OPEN

        async def succeeding():
            return "ok"

        # Record successes (need success_threshold=2)
        await breaker.execute(succeeding)
        await breaker.execute(succeeding)

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, breaker):
        async def failing():
            raise Exception("Fail")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.execute(failing)

        # Wait for timeout
        await asyncio.sleep(1.1)
        await breaker._check_state()
        assert breaker.state == CircuitState.HALF_OPEN

        # Record failure - should reopen
        with pytest.raises(Exception):
            await breaker.execute(failing)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, breaker):
        async def failing():
            raise Exception("Fail")

        # Open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.execute(failing)

        assert breaker.state == CircuitState.OPEN

        # Reset
        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED

    def test_failure_count_property(self, breaker):
        assert breaker.failure_count == 0

    def test_get_stats(self, breaker):
        stats = breaker.get_stats()
        assert "name" in stats
        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats


class TestInMemoryFallback:
    """Tests for InMemoryFallback cache."""

    @pytest.fixture
    def fallback(self):
        return InMemoryFallback(
            max_entries=100,
            max_pending_ops=50
        )

    def test_creation(self, fallback):
        stats = fallback.get_stats()
        assert stats["total_entries"] == 0
        assert stats["pending_operations"] == 0

    def test_put_and_get(self, fallback):
        fallback.put("coll", "key1", {"data": "value"})
        result = fallback.get("coll", "key1")
        assert result == {"data": "value"}

    def test_get_missing_returns_none(self, fallback):
        result = fallback.get("coll", "nonexistent")
        assert result is None

    def test_put_with_queue_for_write(self, fallback):
        fallback.put("coll", "key1", "data", queue_for_write=True)
        pending = fallback.get_pending_operations()
        assert len(pending) == 1

    def test_put_without_queue_for_write(self, fallback):
        fallback.put("coll", "key1", "data", queue_for_write=False)
        pending = fallback.get_pending_operations()
        assert len(pending) == 0

    def test_delete(self, fallback):
        fallback.put("coll", "key1", "data", queue_for_write=False)
        assert fallback.get("coll", "key1") is not None

        fallback.delete("coll", "key1", queue_for_delete=False)
        assert fallback.get("coll", "key1") is None

    def test_delete_queues_operation(self, fallback):
        fallback.put("coll", "key1", "data", queue_for_write=False)
        fallback.delete("coll", "key1", queue_for_delete=True)

        pending = fallback.get_pending_operations()
        assert len(pending) == 1
        assert pending[0].operation_type == "delete"

    def test_clear_pending_operations(self, fallback):
        fallback.put("coll", "k1", "v1")
        fallback.put("coll", "k2", "v2")

        assert len(fallback.get_pending_operations()) == 2

        count = fallback.clear_pending_operations()
        assert count == 2
        assert len(fallback.get_pending_operations()) == 0

    def test_max_entries_eviction(self):
        fallback = InMemoryFallback(max_entries=5, max_pending_ops=10)

        for i in range(10):
            fallback.put("coll", f"key{i}", f"value{i}", queue_for_write=False)

        stats = fallback.get_stats()
        assert stats["total_entries"] <= 5

    def test_get_pending_operations(self, fallback):
        fallback.put("coll", "k1", "v1", queue_for_write=True)
        fallback.put("coll", "k2", "v2", queue_for_write=True)

        pending = fallback.get_pending_operations()
        assert len(pending) == 2

    def test_mark_operation_complete(self, fallback):
        fallback.put("coll", "key1", "data", queue_for_write=True)
        assert len(fallback.get_pending_operations()) == 1

        pending = fallback.get_pending_operations()
        fallback.mark_operation_complete(pending[0])

        assert len(fallback.get_pending_operations()) == 0

    def test_get_stats(self, fallback):
        fallback.put("coll", "k1", "v1", queue_for_write=False)
        fallback.get("coll", "k1")  # Cache hit
        fallback.get("coll", "missing")  # Cache miss

        stats = fallback.get_stats()
        assert "total_entries" in stats
        assert "collections" in stats
        assert "pending_operations" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1

    def test_clear(self, fallback):
        fallback.put("coll", "k1", "v1")
        fallback.put("coll", "k2", "v2")
        fallback.clear()

        stats = fallback.get_stats()
        assert stats["total_entries"] == 0
        assert stats["pending_operations"] == 0


class TestPendingOperation:
    """Tests for PendingOperation dataclass."""

    def test_creation(self):
        op = PendingOperation(
            operation_type="write",
            collection="episodes",
            key="ep_123",
            data={"content": "test"},
            timestamp=datetime.now()
        )
        assert op.operation_type == "write"
        assert op.retry_count == 0
        assert op.max_retries == 3

    def test_retry_count_default(self):
        op = PendingOperation(
            operation_type="write",
            collection="test",
            key="key",
            data=None,
        )
        assert op.retry_count == 0

    def test_max_retries_default(self):
        op = PendingOperation(
            operation_type="write",
            collection="test",
            key="key",
            data=None,
        )
        assert op.max_retries == 3

    def test_custom_retry_values(self):
        op = PendingOperation(
            operation_type="write",
            collection="test",
            key="key",
            data=None,
            retry_count=2,
            max_retries=5
        )
        assert op.retry_count == 2
        assert op.max_retries == 5


class TestGracefulDegradation:
    """Tests for GracefulDegradation orchestrator."""

    @pytest.fixture
    def degradation(self):
        return GracefulDegradation(
            name="test_gd_unique",
            circuit_config=CircuitBreakerConfig(
                failure_threshold=2,
                reset_timeout=1.0
            ),
            fallback_max_entries=100,
            fallback_max_pending=100
        )

    def test_creation(self, degradation):
        assert degradation.name == "test_gd_unique"
        assert degradation.is_degraded is False

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, degradation):
        async def primary():
            return "success"

        result, from_fallback = await degradation.execute_with_fallback(
            primary_func=primary,
            fallback_key="test_key",
            collection="test_coll"
        )

        assert result == "success"
        assert from_fallback is False

    @pytest.mark.asyncio
    async def test_execute_with_fallback_uses_cache_on_failure(self, degradation):
        # Pre-populate cache
        degradation.fallback.put("test_coll", "test_key", "cached_value", queue_for_write=False)

        async def failing_primary():
            raise Exception("Primary failed")

        result, from_fallback = await degradation.execute_with_fallback(
            primary_func=failing_primary,
            fallback_key="test_key",
            collection="test_coll"
        )

        assert result == "cached_value"
        assert from_fallback is True

    @pytest.mark.asyncio
    async def test_execute_with_fallback_raises_when_no_cache(self, degradation):
        async def failing_primary():
            raise Exception("Primary failed")

        with pytest.raises(Exception, match="Primary failed"):
            await degradation.execute_with_fallback(
                primary_func=failing_primary,
                fallback_key="nonexistent_key",
                collection="test_coll"
            )

    @pytest.mark.asyncio
    async def test_execute_caches_successful_results(self, degradation):
        async def primary():
            return {"data": "value"}

        await degradation.execute_with_fallback(
            primary_func=primary,
            fallback_key="new_key",
            collection="coll",
            cache_result=True
        )

        # Should be cached
        cached = degradation.fallback.get("coll", "new_key")
        assert cached == {"data": "value"}

    @pytest.mark.asyncio
    async def test_write_with_queue_success(self, degradation):
        async def write_func():
            return True

        success = await degradation.write_with_queue(
            write_func=write_func,
            collection="coll",
            key="key",
            data={"value": 1}
        )

        assert success is True
        # Should not be queued on success
        assert len(degradation.fallback.get_pending_operations()) == 0

    @pytest.mark.asyncio
    async def test_write_with_queue_queues_on_failure(self, degradation):
        async def failing_write():
            raise Exception("Write failed")

        success = await degradation.write_with_queue(
            write_func=failing_write,
            collection="coll",
            key="key",
            data={"value": 1}
        )

        assert success is False
        # Should be queued for retry
        assert len(degradation.fallback.get_pending_operations()) == 1

    @pytest.mark.asyncio
    async def test_drain_pending_operations_success(self, degradation):
        # Queue some operations
        async def failing_write():
            raise Exception("Fail")

        await degradation.write_with_queue(failing_write, "coll", "k1", "d1")
        await degradation.write_with_queue(failing_write, "coll", "k2", "d2")

        assert len(degradation.fallback.get_pending_operations()) == 2

        # Reset circuit breaker to allow draining (simulates recovery)
        await degradation.circuit_breaker.reset()

        # Now replay successfully
        replayed_ops = []

        async def replay_func(op):
            replayed_ops.append(op)
            return True

        success, failed = await degradation.drain_pending_operations(replay_func)

        assert success == 2
        assert failed == 0
        assert len(replayed_ops) == 2

    @pytest.mark.asyncio
    async def test_drain_pending_operations_partial_failure(self, degradation):
        # Queue operations
        async def failing_write():
            raise Exception("Fail")

        await degradation.write_with_queue(failing_write, "coll", "k1", "d1")
        await degradation.write_with_queue(failing_write, "coll", "k2", "d2")

        # Reset circuit breaker to allow draining (simulates recovery)
        await degradation.circuit_breaker.reset()

        # Replay with partial failure
        call_count = 0

        async def partial_replay(op):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True
            raise Exception("Replay failed")  # Need to raise to trigger failure handling

        success, failed = await degradation.drain_pending_operations(partial_replay)

        assert success == 1
        # failed == 0 because retry_count < max_retries
        # The operation is just re-queued, not marked as permanently failed

    def test_is_degraded_initially_false(self, degradation):
        assert degradation.is_degraded is False

    @pytest.mark.asyncio
    async def test_becomes_degraded_after_failures(self, degradation):
        async def failing():
            raise Exception("Fail")

        # Exhaust circuit breaker
        for _ in range(3):
            try:
                await degradation.execute_with_fallback(failing, "key", "coll")
            except:
                pass

        assert degradation.is_degraded is True

    def test_get_stats(self, degradation):
        stats = degradation.get_stats()
        assert "name" in stats
        assert "is_degraded" in stats
        assert "circuit_breaker" in stats
        assert "fallback" in stats


class TestGlobalFactoryFunctions:
    """Tests for global factory functions."""

    def test_get_circuit_breaker_creates_new(self):
        cb1 = get_circuit_breaker("unique_name_cb_1")
        assert cb1 is not None
        assert cb1.name == "unique_name_cb_1"

    def test_get_circuit_breaker_returns_same(self):
        cb1 = get_circuit_breaker("shared_name_cb")
        cb2 = get_circuit_breaker("shared_name_cb")
        assert cb1 is cb2

    def test_get_graceful_degradation_creates_new(self):
        gd1 = get_graceful_degradation("unique_gd_factory_1")
        assert gd1 is not None
        assert gd1.name == "unique_gd_factory_1"

    def test_get_graceful_degradation_returns_same(self):
        gd1 = get_graceful_degradation("shared_gd_factory")
        gd2 = get_graceful_degradation("shared_gd_factory")
        assert gd1 is gd2


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    @pytest.fixture
    def quick_breaker(self):
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout=0.1,
            success_threshold=1
        )
        return CircuitBreaker("quick_test_unique", config)

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, quick_breaker):
        async def failing():
            raise Exception("Fail")

        async def succeeding():
            return "ok"

        # Start closed
        assert quick_breaker.state == CircuitState.CLOSED

        # Fail to open
        for _ in range(2):
            with pytest.raises(Exception):
                await quick_breaker.execute(failing)
        assert quick_breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Check state to transition to half-open
        await quick_breaker._check_state()
        assert quick_breaker.state == CircuitState.HALF_OPEN

        # Success to close
        await quick_breaker.execute(succeeding)
        assert quick_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self, quick_breaker):
        async def failing():
            raise Exception("Fail")

        # Open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await quick_breaker.execute(failing)

        # Wait and transition to half-open
        await asyncio.sleep(0.15)
        await quick_breaker._check_state()
        assert quick_breaker.state == CircuitState.HALF_OPEN

        # Fail to reopen
        with pytest.raises(Exception):
            await quick_breaker.execute(failing)
        assert quick_breaker.state == CircuitState.OPEN
