"""
Emergency Management for World Weaver.

Phase 9: Panic mode, graceful shutdown, and circuit breakers.

Provides:
- Panic mode: Immediately halt all operations
- Graceful shutdown: Drain connections and complete in-flight requests
- Circuit breakers: Prevent cascade failures
- Health degradation: Gradual service degradation

Usage:
    from t4dm.core.emergency import get_emergency_manager, PanicLevel

    em = get_emergency_manager()

    # Enter panic mode
    em.panic(PanicLevel.CRITICAL, reason="Database unreachable")

    # Check if safe to proceed
    if em.is_safe_to_proceed():
        do_operation()

    # Graceful shutdown
    await em.graceful_shutdown(timeout=30.0)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class PanicLevel(IntEnum):
    """Panic severity levels."""

    NONE = 0        # Normal operation
    DEGRADED = 1    # Some features disabled, core functional
    LIMITED = 2     # Read-only mode, no writes
    CRITICAL = 3    # Block all requests except health
    TOTAL = 4       # Complete shutdown, no responses


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls, circuit tripped
    HALF_OPEN = "half_open"  # Testing if safe to resume


@dataclass
class PanicEvent:
    """Record of a panic event."""

    level: PanicLevel
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": self.level.name,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = 5      # Failures before opening
    success_threshold: int = 3      # Successes in half-open before closing
    timeout_seconds: float = 30.0   # Time before half-open attempt
    half_open_max_calls: int = 3    # Max calls in half-open state


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascade failures.

    Tracks failures and opens the circuit to prevent repeated calls
    to a failing service. Periodically allows test calls (half-open)
    to check if the service has recovered.

    Example:
        breaker = CircuitBreaker("database")

        if breaker.allow_request():
            try:
                result = db.query(...)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        else:
            raise ServiceUnavailableError()
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._lock = threading.RLock()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0

        logger.debug(f"CircuitBreaker '{name}' initialized")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_timeout()
            return self._state

    def _check_timeout(self) -> None:
        """Check if timeout has expired for half-open transition."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.config.timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"Circuit '{self.name}' transitioning to half-open")

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed.

        Returns:
            True if request should proceed
        """
        with self._lock:
            self._check_timeout()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            # OPEN state
            return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(f"Circuit '{self.name}' closed (recovered)")
            else:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning(f"Circuit '{self.name}' reopened (failure in half-open)")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit '{self.name}' opened after "
                        f"{self._failure_count} failures"
                    )

    def reset(self) -> None:
        """Reset the circuit to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            logger.info(f"Circuit '{self.name}' manually reset")

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure": self._last_failure_time,
            }


class EmergencyManager:
    """
    Central emergency management for World Weaver.

    Coordinates panic modes, graceful shutdown, and circuit breakers
    across the system.

    Example:
        em = EmergencyManager()

        # Register shutdown handler
        em.register_shutdown_handler(cleanup_database)

        # Enter degraded mode
        em.panic(PanicLevel.DEGRADED, reason="High error rate")

        # Check before operations
        if em.is_safe_to_proceed():
            do_work()

        # Graceful shutdown
        await em.graceful_shutdown(timeout=30.0)
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._panic_level = PanicLevel.NONE
        self._panic_events: list[PanicEvent] = []
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._shutdown_handlers: list[Callable[[], Awaitable[None] | None]] = []
        self._is_shutting_down = False
        self._shutdown_event = asyncio.Event() if asyncio.get_event_loop().is_running() else None
        self._in_flight_count = 0
        self._in_flight_lock = threading.Lock()

        # Register signal handlers
        self._register_signals()

        logger.info("EmergencyManager initialized")

    def _register_signals(self) -> None:
        """Register system signal handlers."""
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except (OSError, ValueError):
            # May fail in non-main thread or when no signals available
            logger.debug("Could not register signal handlers")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, initiating shutdown")

        # Schedule async shutdown
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.graceful_shutdown())
        except RuntimeError:
            # No running loop, do sync cleanup
            self._is_shutting_down = True

    @property
    def panic_level(self) -> PanicLevel:
        """Get current panic level."""
        with self._lock:
            return self._panic_level

    def panic(
        self,
        level: PanicLevel,
        reason: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Enter panic mode.

        Args:
            level: Severity level
            reason: Why panic was triggered
            source: Component that triggered panic
            metadata: Additional context
        """
        with self._lock:
            old_level = self._panic_level

            # Only escalate, never de-escalate via panic()
            if level > self._panic_level:
                self._panic_level = level

                event = PanicEvent(
                    level=level,
                    reason=reason,
                    source=source,
                    metadata=metadata or {},
                )
                self._panic_events.append(event)

                logger.error(
                    f"PANIC: Level {old_level.name} -> {level.name}: {reason}"
                )

                # Log to separate panic log if available
                self._log_panic(event)

    def _log_panic(self, event: PanicEvent) -> None:
        """Log panic event to dedicated log."""
        # Could integrate with external alerting here
        pass

    def recover(self, level: PanicLevel = PanicLevel.NONE) -> None:
        """
        Attempt recovery to a lower panic level.

        Args:
            level: Target panic level (must be lower than current)
        """
        with self._lock:
            if level < self._panic_level:
                old_level = self._panic_level
                self._panic_level = level
                logger.info(f"Recovery: {old_level.name} -> {level.name}")

    def is_safe_to_proceed(self, min_level: PanicLevel = PanicLevel.LIMITED) -> bool:
        """
        Check if it's safe to proceed with an operation.

        Args:
            min_level: Minimum acceptable panic level

        Returns:
            True if safe to proceed
        """
        with self._lock:
            if self._is_shutting_down:
                return False
            return self._panic_level <= min_level

    def is_write_allowed(self) -> bool:
        """Check if write operations are allowed."""
        with self._lock:
            if self._is_shutting_down:
                return False
            return self._panic_level < PanicLevel.LIMITED

    def is_read_allowed(self) -> bool:
        """Check if read operations are allowed."""
        with self._lock:
            if self._is_shutting_down:
                return False
            return self._panic_level < PanicLevel.CRITICAL

    def get_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker.

        Args:
            name: Unique name for the breaker
            config: Optional configuration

        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(name, config)
            return self._circuit_breakers[name]

    def register_shutdown_handler(
        self,
        handler: Callable[[], Awaitable[None] | None],
    ) -> None:
        """
        Register a handler to be called during graceful shutdown.

        Handlers are called in reverse registration order (LIFO).

        Args:
            handler: Async or sync cleanup function
        """
        with self._lock:
            self._shutdown_handlers.append(handler)
            logger.debug(f"Registered shutdown handler: {handler.__name__}")

    def track_request_start(self) -> None:
        """Track the start of an in-flight request."""
        with self._in_flight_lock:
            self._in_flight_count += 1

    def track_request_end(self) -> None:
        """Track the end of an in-flight request."""
        with self._in_flight_lock:
            self._in_flight_count = max(0, self._in_flight_count - 1)

    @property
    def in_flight_count(self) -> int:
        """Get count of in-flight requests."""
        with self._in_flight_lock:
            return self._in_flight_count

    @property
    def is_shutting_down(self) -> bool:
        """Check if system is shutting down."""
        return self._is_shutting_down

    async def graceful_shutdown(self, timeout: float = 30.0) -> None:
        """
        Perform graceful shutdown.

        1. Stop accepting new requests
        2. Wait for in-flight requests to complete
        3. Call registered shutdown handlers
        4. Final cleanup

        Args:
            timeout: Maximum seconds to wait for in-flight requests
        """
        if self._is_shutting_down:
            logger.debug("Shutdown already in progress")
            return

        self._is_shutting_down = True
        logger.warning(f"Graceful shutdown initiated (timeout={timeout}s)")

        # Set panic level to block new requests
        self.panic(PanicLevel.CRITICAL, reason="Graceful shutdown", source="emergency")

        # Wait for in-flight requests to drain
        start_time = time.time()
        while self.in_flight_count > 0:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(
                    f"Shutdown timeout: {self.in_flight_count} requests still in-flight"
                )
                break

            remaining = timeout - elapsed
            logger.info(
                f"Waiting for {self.in_flight_count} in-flight requests "
                f"({remaining:.1f}s remaining)"
            )
            await asyncio.sleep(0.5)

        # Call shutdown handlers in reverse order
        logger.info(f"Calling {len(self._shutdown_handlers)} shutdown handlers")
        for handler in reversed(self._shutdown_handlers):
            try:
                result = handler()
                if asyncio.iscoroutine(result):
                    await result
                logger.debug(f"Shutdown handler completed: {handler.__name__}")
            except Exception as e:
                logger.error(f"Shutdown handler error: {handler.__name__}: {e}")

        logger.warning("Graceful shutdown complete")

    def get_panic_events(self, limit: int = 100) -> list[dict]:
        """Get recent panic events."""
        with self._lock:
            return [e.to_dict() for e in self._panic_events[-limit:]]

    def get_stats(self) -> dict:
        """Get emergency manager statistics."""
        with self._lock:
            circuit_stats = {
                name: cb.get_stats()
                for name, cb in self._circuit_breakers.items()
            }

            return {
                "panic_level": self._panic_level.name,
                "is_shutting_down": self._is_shutting_down,
                "in_flight_count": self.in_flight_count,
                "panic_events": len(self._panic_events),
                "shutdown_handlers": len(self._shutdown_handlers),
                "circuit_breakers": circuit_stats,
            }


# ============================================================================
# Context Manager for Request Tracking
# ============================================================================


class TrackedRequest:
    """Context manager for tracking in-flight requests."""

    def __init__(self, em: EmergencyManager):
        self._em = em

    def __enter__(self) -> TrackedRequest:
        self._em.track_request_start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._em.track_request_end()

    async def __aenter__(self) -> TrackedRequest:
        self._em.track_request_start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._em.track_request_end()


# ============================================================================
# Singleton
# ============================================================================

_emergency_manager: EmergencyManager | None = None
_lock = threading.Lock()


def get_emergency_manager() -> EmergencyManager:
    """Get or create the singleton emergency manager."""
    global _emergency_manager
    if _emergency_manager is None:
        with _lock:
            if _emergency_manager is None:
                _emergency_manager = EmergencyManager()
    return _emergency_manager


def reset_emergency_manager() -> None:
    """Reset singleton (for testing)."""
    global _emergency_manager
    with _lock:
        _emergency_manager = None


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "EmergencyManager",
    "PanicEvent",
    "PanicLevel",
    "TrackedRequest",
    "get_emergency_manager",
    "reset_emergency_manager",
]
