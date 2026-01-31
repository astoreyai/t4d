"""
Tests for Emergency Management (Phase 9).

Tests panic mode, graceful shutdown, and circuit breakers.
"""

import asyncio
import time

import pytest

from t4dm.core.emergency import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    EmergencyManager,
    PanicEvent,
    PanicLevel,
    TrackedRequest,
    get_emergency_manager,
    reset_emergency_manager,
)


class TestPanicLevel:
    """Tests for PanicLevel enum."""

    def test_ordering(self):
        """Test panic levels are properly ordered."""
        assert PanicLevel.NONE < PanicLevel.DEGRADED
        assert PanicLevel.DEGRADED < PanicLevel.LIMITED
        assert PanicLevel.LIMITED < PanicLevel.CRITICAL
        assert PanicLevel.CRITICAL < PanicLevel.TOTAL

    def test_values(self):
        """Test panic level values."""
        assert PanicLevel.NONE == 0
        assert PanicLevel.DEGRADED == 1
        assert PanicLevel.LIMITED == 2
        assert PanicLevel.CRITICAL == 3
        assert PanicLevel.TOTAL == 4


class TestPanicEvent:
    """Tests for PanicEvent."""

    def test_creation(self):
        """Test event creation."""
        event = PanicEvent(
            level=PanicLevel.DEGRADED,
            reason="Test reason",
            source="test",
        )
        assert event.level == PanicLevel.DEGRADED
        assert event.reason == "Test reason"
        assert event.source == "test"
        assert event.timestamp is not None

    def test_to_dict(self):
        """Test serialization."""
        event = PanicEvent(
            level=PanicLevel.CRITICAL,
            reason="Database failure",
            source="db_monitor",
            metadata={"db": "postgres"},
        )
        data = event.to_dict()
        assert data["level"] == "CRITICAL"
        assert data["reason"] == "Database failure"
        assert data["source"] == "db_monitor"
        assert "timestamp" in data


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_seconds == 30.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=10.0,
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 2
        assert config.timeout_seconds == 10.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker."""
        return CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=0.1,  # Short for testing
            ),
        )

    def test_initial_state(self, breaker):
        """Test initial state is closed."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request() is True

    def test_opens_on_failures(self, breaker):
        """Test circuit opens after threshold failures."""
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.allow_request() is False

    def test_half_open_after_timeout(self, breaker):
        """Test circuit goes half-open after timeout."""
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        assert breaker.state == CircuitState.HALF_OPEN

    def test_closes_on_success_in_half_open(self, breaker):
        """Test circuit closes after successes in half-open."""
        # Open the circuit
        for _ in range(3):
            breaker.record_failure()

        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Record successes
        breaker.record_success()
        breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self, breaker):
        """Test circuit reopens on failure in half-open."""
        # Open the circuit
        for _ in range(3):
            breaker.record_failure()

        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Any failure reopens
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

    def test_reset(self, breaker):
        """Test manual reset."""
        for _ in range(3):
            breaker.record_failure()

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request() is True

    def test_success_resets_failure_count(self, breaker):
        """Test success resets failure count in closed state."""
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        breaker.record_failure()

        # Should still be closed (5 failures needed, but success reset count)
        assert breaker.state == CircuitState.CLOSED

    def test_get_stats(self, breaker):
        """Test statistics."""
        breaker.record_failure()
        breaker.record_failure()

        stats = breaker.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 2


class TestEmergencyManager:
    """Tests for EmergencyManager."""

    @pytest.fixture
    def em(self):
        """Create fresh emergency manager."""
        reset_emergency_manager()
        return EmergencyManager()

    def test_initial_state(self, em):
        """Test initial state is normal."""
        assert em.panic_level == PanicLevel.NONE
        assert em.is_shutting_down is False
        assert em.in_flight_count == 0

    def test_panic_escalation(self, em):
        """Test panic level only escalates."""
        em.panic(PanicLevel.DEGRADED, reason="First issue")
        assert em.panic_level == PanicLevel.DEGRADED

        em.panic(PanicLevel.LIMITED, reason="Second issue")
        assert em.panic_level == PanicLevel.LIMITED

        # Should not de-escalate via panic()
        em.panic(PanicLevel.DEGRADED, reason="Trying to de-escalate")
        assert em.panic_level == PanicLevel.LIMITED

    def test_recovery(self, em):
        """Test recovery to lower level."""
        em.panic(PanicLevel.CRITICAL, reason="Crisis")
        em.recover(PanicLevel.DEGRADED)

        assert em.panic_level == PanicLevel.DEGRADED

        em.recover(PanicLevel.NONE)
        assert em.panic_level == PanicLevel.NONE

    def test_is_safe_to_proceed(self, em):
        """Test safety check."""
        assert em.is_safe_to_proceed() is True

        em.panic(PanicLevel.DEGRADED, reason="Minor issue")
        assert em.is_safe_to_proceed() is True  # Below LIMITED

        em.panic(PanicLevel.LIMITED, reason="Major issue")
        assert em.is_safe_to_proceed() is True  # At limit

        em.panic(PanicLevel.CRITICAL, reason="Critical issue")
        assert em.is_safe_to_proceed() is False  # Above LIMITED

    def test_is_write_allowed(self, em):
        """Test write permission."""
        assert em.is_write_allowed() is True

        em.panic(PanicLevel.DEGRADED, reason="Minor")
        assert em.is_write_allowed() is True

        em.panic(PanicLevel.LIMITED, reason="Read-only mode")
        assert em.is_write_allowed() is False

    def test_is_read_allowed(self, em):
        """Test read permission."""
        assert em.is_read_allowed() is True

        em.panic(PanicLevel.LIMITED, reason="Read-only")
        assert em.is_read_allowed() is True

        em.panic(PanicLevel.CRITICAL, reason="No access")
        assert em.is_read_allowed() is False

    def test_circuit_breaker_creation(self, em):
        """Test circuit breaker creation."""
        cb1 = em.get_circuit_breaker("database")
        cb2 = em.get_circuit_breaker("database")

        assert cb1 is cb2  # Same instance

        cb3 = em.get_circuit_breaker("api")
        assert cb3 is not cb1  # Different instance

    def test_request_tracking(self, em):
        """Test in-flight request tracking."""
        assert em.in_flight_count == 0

        em.track_request_start()
        assert em.in_flight_count == 1

        em.track_request_start()
        assert em.in_flight_count == 2

        em.track_request_end()
        assert em.in_flight_count == 1

        em.track_request_end()
        assert em.in_flight_count == 0

        # Shouldn't go negative
        em.track_request_end()
        assert em.in_flight_count == 0

    def test_shutdown_handler_registration(self, em):
        """Test shutdown handler registration."""
        called = []

        def handler():
            called.append(True)

        em.register_shutdown_handler(handler)
        assert len(em._shutdown_handlers) == 1

    def test_panic_events(self, em):
        """Test panic event tracking."""
        em.panic(PanicLevel.DEGRADED, reason="Issue 1")
        em.panic(PanicLevel.LIMITED, reason="Issue 2")

        events = em.get_panic_events()
        assert len(events) == 2
        assert events[0]["reason"] == "Issue 1"
        assert events[1]["reason"] == "Issue 2"

    def test_get_stats(self, em):
        """Test statistics."""
        em.panic(PanicLevel.DEGRADED, reason="Test")
        em.get_circuit_breaker("test_cb")

        stats = em.get_stats()
        assert stats["panic_level"] == "DEGRADED"
        assert stats["is_shutting_down"] is False
        assert "test_cb" in stats["circuit_breakers"]


class TestTrackedRequest:
    """Tests for TrackedRequest context manager."""

    @pytest.fixture
    def em(self):
        """Create fresh emergency manager."""
        reset_emergency_manager()
        return EmergencyManager()

    def test_sync_context_manager(self, em):
        """Test synchronous context manager."""
        assert em.in_flight_count == 0

        with TrackedRequest(em):
            assert em.in_flight_count == 1

        assert em.in_flight_count == 0

    def test_sync_context_manager_exception(self, em):
        """Test tracking works even with exceptions."""
        assert em.in_flight_count == 0

        with pytest.raises(ValueError):
            with TrackedRequest(em):
                assert em.in_flight_count == 1
                raise ValueError("Test error")

        assert em.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_async_context_manager(self, em):
        """Test asynchronous context manager."""
        assert em.in_flight_count == 0

        async with TrackedRequest(em):
            assert em.in_flight_count == 1

        assert em.in_flight_count == 0


class TestGracefulShutdown:
    """Tests for graceful shutdown."""

    @pytest.fixture
    def em(self):
        """Create fresh emergency manager."""
        reset_emergency_manager()
        return EmergencyManager()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_calls_handlers(self, em):
        """Test shutdown calls handlers."""
        called = []

        async def async_handler():
            called.append("async")

        def sync_handler():
            called.append("sync")

        em.register_shutdown_handler(async_handler)
        em.register_shutdown_handler(sync_handler)

        await em.graceful_shutdown(timeout=1.0)

        assert "async" in called
        assert "sync" in called
        assert em.is_shutting_down is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_waits_for_requests(self, em):
        """Test shutdown waits for in-flight requests."""
        em.track_request_start()

        async def complete_request():
            await asyncio.sleep(0.1)
            em.track_request_end()

        # Start completing request in background
        asyncio.create_task(complete_request())

        await em.graceful_shutdown(timeout=1.0)

        assert em.in_flight_count == 0


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_emergency_manager_singleton(self):
        """Test singleton creation."""
        reset_emergency_manager()
        em1 = get_emergency_manager()
        em2 = get_emergency_manager()
        assert em1 is em2

    def test_reset_emergency_manager(self):
        """Test singleton reset."""
        em1 = get_emergency_manager()
        reset_emergency_manager()
        em2 = get_emergency_manager()
        assert em1 is not em2
