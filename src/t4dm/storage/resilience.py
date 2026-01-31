"""
Resilience patterns for World Weaver storage backends.

Provides circuit breaker, retry, bulkhead, and graceful degradation patterns
for storage operations.

Graceful Degradation Strategy:
1. Primary backend available: Normal operation
2. Primary failing: Circuit breaker opens, fallback to in-memory cache
3. Recovery: Queue operations for replay when primary recovers
4. Full recovery: Drain queue, close circuit breaker
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation, requests flow through
    OPEN = "open"           # Failing, requests are rejected immediately
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, failures: int, reset_time: float):
        self.name = name
        self.failures = failures
        self.reset_time = reset_time
        super().__init__(
            f"Circuit breaker '{name}' is OPEN after {failures} failures. "
            f"Will retry in {reset_time:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 2        # Successes to close from half-open
    reset_timeout: float = 60.0       # Seconds before trying half-open
    excluded_exceptions: tuple = ()    # Exceptions that don't count as failures


@dataclass
class CircuitBreakerState:
    """Runtime state for a circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_transition_time: float = field(default_factory=time.time)

    def record_failure(self) -> None:
        """Record a failure."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = time.time()

    def record_success(self) -> None:
        """Record a success."""
        self.success_count += 1
        self.failure_count = 0

    def reset(self) -> None:
        """Reset counters."""
        self.failure_count = 0
        self.success_count = 0


class CircuitBreaker:
    """
    Circuit breaker implementation for storage operations.

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Failing fast, rejecting all requests
    - HALF_OPEN: Testing recovery with limited requests

    Example:
        cb = CircuitBreaker("qdrant", config=CircuitBreakerConfig(failure_threshold=5))

        @cb.protect
        async def query_qdrant():
            return await client.search(...)
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit (e.g., "qdrant", "neo4j")
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state.state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state.state == CircuitState.OPEN

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._state.failure_count

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state.state
        self._state.state = new_state
        self._state.last_transition_time = time.time()

        if old_state != new_state:
            logger.warning(
                f"Circuit breaker '{self.name}' transitioned: "
                f"{old_state.value} -> {new_state.value}"
            )

    async def _check_state(self) -> None:
        """Check and update circuit state based on timeouts."""
        async with self._lock:
            if self._state.state == CircuitState.OPEN:
                # Check if reset timeout has elapsed
                elapsed = time.time() - self._state.last_failure_time
                if elapsed >= self.config.reset_timeout:
                    await self._transition_to(CircuitState.HALF_OPEN)
                    self._state.reset()

    async def _handle_success(self) -> None:
        """Handle successful operation."""
        async with self._lock:
            self._state.record_success()

            if self._state.state == CircuitState.HALF_OPEN:
                # Check if enough successes to close
                if self._state.success_count >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    self._state.reset()
                    logger.info(f"Circuit breaker '{self.name}' recovered")

    async def _handle_failure(self, exc: Exception) -> None:
        """Handle failed operation."""
        # Check if this exception should be excluded
        if isinstance(exc, self.config.excluded_exceptions):
            return

        async with self._lock:
            self._state.record_failure()

            if self._state.state == CircuitState.HALF_OPEN:
                # Single failure in half-open reopens circuit
                await self._transition_to(CircuitState.OPEN)
                logger.warning(
                    f"Circuit breaker '{self.name}' reopened after recovery failure"
                )

            elif self._state.state == CircuitState.CLOSED:
                # Check if we've hit the threshold
                if self._state.failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
                    logger.error(
                        f"Circuit breaker '{self.name}' OPENED after "
                        f"{self._state.failure_count} failures"
                    )

    async def execute(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from function
        """
        await self._check_state()

        if self._state.state == CircuitState.OPEN:
            time_remaining = (
                self.config.reset_timeout -
                (time.time() - self._state.last_failure_time)
            )
            raise CircuitBreakerError(
                self.name,
                self._state.failure_count,
                max(0, time_remaining),
            )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._handle_success()
            return result

        except Exception as e:
            await self._handle_failure(e)
            raise

    def protect(
        self,
        func: Callable[..., Any],
    ) -> Callable[..., Any]:
        """
        Decorator to protect a function with circuit breaker.

        Example:
            @circuit_breaker.protect
            async def risky_operation():
                ...
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.state.value,
            "failure_count": self._state.failure_count,
            "success_count": self._state.success_count,
            "last_failure": self._state.last_failure_time,
            "last_transition": self._state.last_transition_time,
        }

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
            self._state.reset()
            logger.info(f"Circuit breaker '{self.name}' manually reset")


# Global circuit breakers for storage backends
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.

    Args:
        name: Circuit breaker identifier
        config: Optional configuration (only used on creation)

    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_storage_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all storage circuit breakers."""
    return _circuit_breakers.copy()


async def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    for cb in _circuit_breakers.values():
        await cb.reset()


# ============================================================================
# Graceful Degradation
# ============================================================================

@dataclass
class PendingOperation:
    """A pending operation to retry when service recovers."""

    operation_type: str  # "write", "update", "delete"
    collection: str
    key: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3


class InMemoryFallback:
    """
    In-memory fallback storage for graceful degradation.

    Provides temporary storage when primary backends are unavailable.
    Data is held in memory and queued for write-through when
    the primary recovers.

    Limitations:
    - No persistence across restarts
    - Limited by available memory
    - No advanced query capabilities (only key-value)
    """

    def __init__(
        self,
        max_entries: int = 10000,
        max_pending_ops: int = 1000
    ):
        """
        Initialize fallback storage.

        Args:
            max_entries: Maximum entries to cache
            max_pending_ops: Maximum pending operations to queue
        """
        self.max_entries = max_entries
        self.max_pending_ops = max_pending_ops

        # In-memory cache: collection -> {key -> value}
        self._cache: dict[str, dict[str, Any]] = {}

        # Pending write operations for replay
        self._pending_ops: deque[PendingOperation] = deque(maxlen=max_pending_ops)

        # Statistics
        self._hits = 0
        self._misses = 0
        self._fallback_writes = 0

    def get(self, collection: str, key: str) -> Any | None:
        """
        Get value from fallback cache.

        Args:
            collection: Collection name
            key: Item key

        Returns:
            Cached value or None
        """
        if collection in self._cache and key in self._cache[collection]:
            self._hits += 1
            return self._cache[collection][key]

        self._misses += 1
        return None

    def put(
        self,
        collection: str,
        key: str,
        value: Any,
        queue_for_write: bool = True
    ) -> bool:
        """
        Store value in fallback cache.

        Args:
            collection: Collection name
            key: Item key
            value: Value to store
            queue_for_write: Whether to queue for write-through

        Returns:
            True if stored successfully
        """
        # Initialize collection if needed
        if collection not in self._cache:
            self._cache[collection] = {}

        # Check capacity
        total_entries = sum(len(c) for c in self._cache.values())
        if total_entries >= self.max_entries:
            # Evict oldest entries (LRU-like)
            self._evict_oldest()

        self._cache[collection][key] = value
        self._fallback_writes += 1

        if queue_for_write:
            self._pending_ops.append(PendingOperation(
                operation_type="write",
                collection=collection,
                key=key,
                data=value
            ))

        return True

    def delete(self, collection: str, key: str, queue_for_delete: bool = True) -> bool:
        """
        Delete from fallback cache.

        Args:
            collection: Collection name
            key: Item key
            queue_for_delete: Whether to queue for delete on primary

        Returns:
            True if deleted
        """
        deleted = False
        if collection in self._cache and key in self._cache[collection]:
            del self._cache[collection][key]
            deleted = True

        if queue_for_delete:
            self._pending_ops.append(PendingOperation(
                operation_type="delete",
                collection=collection,
                key=key,
                data=None
            ))

        return deleted

    def _evict_oldest(self) -> None:
        """Evict oldest entries when at capacity."""
        # Simple strategy: remove 10% of entries from largest collection
        if not self._cache:
            return

        largest = max(self._cache.keys(), key=lambda c: len(self._cache[c]))
        collection = self._cache[largest]

        # Remove first 10% of entries
        n_remove = max(1, len(collection) // 10)
        keys_to_remove = list(collection.keys())[:n_remove]

        for key in keys_to_remove:
            del collection[key]

        logger.debug(f"Evicted {n_remove} entries from fallback cache ({largest})")

    def get_pending_operations(self) -> list[PendingOperation]:
        """Get all pending operations for replay."""
        return list(self._pending_ops)

    def clear_pending_operations(self) -> int:
        """Clear pending operations queue."""
        count = len(self._pending_ops)
        self._pending_ops.clear()
        return count

    def mark_operation_complete(self, op: PendingOperation) -> None:
        """Mark a pending operation as complete."""
        try:
            self._pending_ops.remove(op)
        except ValueError:
            pass  # Already removed

    def get_stats(self) -> dict:
        """Get fallback statistics."""
        total_entries = sum(len(c) for c in self._cache.values())
        return {
            "total_entries": total_entries,
            "collections": len(self._cache),
            "pending_operations": len(self._pending_ops),
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "fallback_writes": self._fallback_writes,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear all fallback data."""
        self._cache.clear()
        self._pending_ops.clear()
        self._hits = 0
        self._misses = 0
        self._fallback_writes = 0


class GracefulDegradation:
    """
    Orchestrates graceful degradation for storage operations.

    Combines circuit breaker, fallback cache, and pending operation
    queue to provide resilient storage access.

    Usage:
        gd = GracefulDegradation("qdrant")

        # Attempt operation with fallback
        result = await gd.execute_with_fallback(
            primary_func=lambda: qdrant.search(...),
            fallback_func=lambda: fallback.get("searches", query_hash),
            write_through=True
        )

        # Drain pending operations on recovery
        await gd.drain_pending_operations(replay_func)
    """

    def __init__(
        self,
        name: str,
        circuit_config: CircuitBreakerConfig | None = None,
        fallback_max_entries: int = 10000,
        fallback_max_pending: int = 1000
    ):
        """
        Initialize graceful degradation.

        Args:
            name: Identifier for this degradation handler
            circuit_config: Circuit breaker configuration
            fallback_max_entries: Max entries in fallback cache
            fallback_max_pending: Max pending operations
        """
        self.name = name
        self.circuit_breaker = get_circuit_breaker(name, circuit_config)
        self.fallback = InMemoryFallback(
            max_entries=fallback_max_entries,
            max_pending_ops=fallback_max_pending
        )

        self._degraded_since: datetime | None = None
        self._recovery_attempts = 0

    @property
    def is_degraded(self) -> bool:
        """Check if currently in degraded mode."""
        return self.circuit_breaker.is_open

    @property
    def degradation_duration(self) -> float | None:
        """Get duration of current degradation in seconds."""
        if self._degraded_since is None:
            return None
        return (datetime.now() - self._degraded_since).total_seconds()

    async def execute_with_fallback(
        self,
        primary_func: Callable[..., Any],
        fallback_key: str | None = None,
        collection: str = "default",
        cache_result: bool = True
    ) -> tuple[Any, bool]:
        """
        Execute operation with fallback on failure.

        Args:
            primary_func: Primary async function to execute
            fallback_key: Key for fallback cache lookup
            collection: Collection for fallback cache
            cache_result: Whether to cache successful result

        Returns:
            Tuple of (result, used_fallback)
        """
        try:
            # Try primary via circuit breaker
            result = await self.circuit_breaker.execute(primary_func)

            # Cache successful result if requested
            if cache_result and fallback_key:
                self.fallback.put(collection, fallback_key, result, queue_for_write=False)

            # Clear degradation state on success
            if self._degraded_since is not None:
                logger.info(
                    f"GracefulDegradation '{self.name}' recovered after "
                    f"{self.degradation_duration:.1f}s"
                )
                self._degraded_since = None
                self._recovery_attempts = 0

            return result, False

        except CircuitBreakerError:
            # Circuit is open - use fallback
            if self._degraded_since is None:
                self._degraded_since = datetime.now()
                logger.warning(f"GracefulDegradation '{self.name}' entering degraded mode")

            if fallback_key:
                fallback_result = self.fallback.get(collection, fallback_key)
                if fallback_result is not None:
                    return fallback_result, True

            # No fallback available
            raise

        except Exception as e:
            # Other exception - may still have fallback
            if fallback_key:
                fallback_result = self.fallback.get(collection, fallback_key)
                if fallback_result is not None:
                    logger.debug(f"Using fallback for '{fallback_key}' due to: {e}")
                    return fallback_result, True
            raise

    async def write_with_queue(
        self,
        write_func: Callable[..., Any],
        collection: str,
        key: str,
        data: Any
    ) -> bool:
        """
        Write with queuing for retry on failure.

        Args:
            write_func: Primary write function
            collection: Collection name
            key: Item key
            data: Data to write

        Returns:
            True if written to primary, False if queued
        """
        try:
            await self.circuit_breaker.execute(write_func)

            # Also update fallback cache
            self.fallback.put(collection, key, data, queue_for_write=False)

            return True

        except (CircuitBreakerError, Exception) as e:
            logger.warning(f"Write failed for '{key}', queuing for retry: {e}")

            # Store in fallback and queue for retry
            self.fallback.put(collection, key, data, queue_for_write=True)

            return False

    async def drain_pending_operations(
        self,
        replay_func: Callable[[PendingOperation], Any],
        max_batch: int = 100
    ) -> tuple[int, int]:
        """
        Drain pending operations when primary recovers.

        Args:
            replay_func: Async function to replay each operation
            max_batch: Maximum operations per drain call

        Returns:
            Tuple of (successful, failed) counts
        """
        if self.circuit_breaker.is_open:
            return 0, 0

        pending = self.fallback.get_pending_operations()[:max_batch]
        successful = 0
        failed = 0

        for op in pending:
            try:
                await replay_func(op)
                self.fallback.mark_operation_complete(op)
                successful += 1

            except Exception as e:
                op.retry_count += 1
                if op.retry_count >= op.max_retries:
                    logger.error(
                        f"Permanently failed pending operation: {op.operation_type} "
                        f"{op.collection}/{op.key} after {op.retry_count} retries"
                    )
                    self.fallback.mark_operation_complete(op)
                    failed += 1
                else:
                    logger.warning(
                        f"Retry {op.retry_count} failed for {op.collection}/{op.key}: {e}"
                    )

        if successful > 0:
            logger.info(f"Drained {successful} pending operations for '{self.name}'")

        return successful, failed

    def get_stats(self) -> dict:
        """Get degradation statistics."""
        return {
            "name": self.name,
            "is_degraded": self.is_degraded,
            "degradation_duration_s": self.degradation_duration,
            "recovery_attempts": self._recovery_attempts,
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "fallback": self.fallback.get_stats(),
        }


# Global graceful degradation instances
_graceful_degradations: dict[str, GracefulDegradation] = {}


def get_graceful_degradation(
    name: str,
    circuit_config: CircuitBreakerConfig | None = None
) -> GracefulDegradation:
    """
    Get or create a graceful degradation handler.

    Args:
        name: Handler identifier
        circuit_config: Optional circuit breaker configuration

    Returns:
        GracefulDegradation instance
    """
    if name not in _graceful_degradations:
        _graceful_degradations[name] = GracefulDegradation(name, circuit_config)
    return _graceful_degradations[name]


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerState",
    "CircuitState",
    "GracefulDegradation",
    "InMemoryFallback",
    "PendingOperation",
    "get_circuit_breaker",
    "get_graceful_degradation",
    "get_storage_circuit_breakers",
    "reset_all_circuit_breakers",
]
