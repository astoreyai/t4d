"""
Storage module lifecycle hooks for T4DM.

Provides hooks for:
- Connection lifecycle (connect, disconnect)
- Query instrumentation (pre/post query)
- Error handling and recovery
- Retry logic
"""

import logging
import time
from typing import Any

from t4dm.hooks.base import Hook, HookContext, HookPriority

logger = logging.getLogger(__name__)


class StorageHook(Hook):
    """Base class for storage-related hooks."""

    def __init__(
        self,
        name: str,
        priority: HookPriority = HookPriority.NORMAL,
        enabled: bool = True,
        storage_type: str | None = None,
    ):
        """
        Initialize storage hook.

        Args:
            name: Hook identifier
            priority: Execution priority
            enabled: Whether hook is active
            storage_type: Filter by storage type (t4dx, etc.)
        """
        super().__init__(name, priority, enabled)
        self.storage_type = storage_type

    def should_execute(self, context: HookContext) -> bool:
        """Check if hook should execute based on storage type filter."""
        if not super().should_execute(context):
            return False

        if self.storage_type:
            store_type = context.input_data.get("storage_type")
            return store_type == self.storage_type

        return True


class ConnectionHook(StorageHook):
    """
    Hook executed during connection lifecycle.

    Phases:
    - ON: Connection established
    - ON: Connection closed
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute connection hook.

        Context data (connect):
        - input_data["storage_type"]: t4dx
        - input_data["connection_string"]: Connection URI
        - input_data["pool_size"]: Connection pool size
        - input_data["event"]: "connect" or "disconnect"

        Context data (disconnect):
        - input_data["storage_type"]: t4dx
        - input_data["event"]: "disconnect"
        - input_data["reason"]: Disconnect reason

        Returns:
            Modified context
        """
        storage_type = context.input_data.get("storage_type", "unknown")
        event = context.input_data.get("event", "unknown")

        logger.info(f"[{self.name}] Connection event: {storage_type}.{event}")
        return context


class QueryHook(StorageHook):
    """
    Hook executed around database queries.

    Phases:
    - PRE: Before query execution (optimization, logging)
    - POST: After query execution (timing, result processing)
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute query hook.

        Context data (PRE):
        - input_data["storage_type"]: t4dx
        - input_data["query"]: Query string/object
        - input_data["parameters"]: Query parameters
        - input_data["query_type"]: read/write/delete

        Context data (POST):
        - output_data["result"]: Query result
        - output_data["row_count"]: Number of rows/documents
        - output_data["duration_ms"]: Execution time

        Returns:
            Modified context
        """
        storage_type = context.input_data.get("storage_type", "unknown")
        query_type = context.input_data.get("query_type", "unknown")
        phase = context.phase.value

        if phase == "pre":
            query = context.input_data.get("query", "")
            logger.debug(
                f"[{self.name}] Pre-query {storage_type}.{query_type}: "
                f"{str(query)[:100]}..."
            )
        else:
            duration = context.output_data.get("duration_ms", 0) if context.output_data else 0
            rows = context.output_data.get("row_count", 0) if context.output_data else 0
            logger.debug(
                f"[{self.name}] Post-query {storage_type}: "
                f"{rows} rows in {duration:.2f}ms"
            )

        return context


class ErrorHook(StorageHook):
    """
    Hook executed when storage errors occur.

    Use for:
    - Error classification
    - Retry decision making
    - Fallback logic
    - Alert generation
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute error handling hook.

        Context data:
        - input_data["storage_type"]: t4dx
        - input_data["error_type"]: connection/timeout/query/unknown
        - input_data["error"]: Exception object
        - input_data["operation"]: Failed operation name
        - input_data["attempt"]: Retry attempt number

        Returns:
            Modified context with error handling decisions
        """
        storage_type = context.input_data.get("storage_type", "unknown")
        error_type = context.input_data.get("error_type", "unknown")
        attempt = context.input_data.get("attempt", 1)

        logger.error(
            f"[{self.name}] Storage error: {storage_type}.{error_type} "
            f"(attempt {attempt})"
        )

        return context

    async def on_error(self, context: HookContext) -> None:
        """Handle errors from other hooks."""
        logger.error(f"[{self.name}] Hook execution error: {context.error}")


class RetryHook(StorageHook):
    """
    Hook executed when retry is attempted.

    Use for:
    - Backoff calculation
    - Retry logging
    - Circuit breaker logic
    """

    async def execute(self, context: HookContext) -> HookContext:
        """
        Execute retry hook.

        Context data:
        - input_data["storage_type"]: t4dx
        - input_data["operation"]: Operation being retried
        - input_data["attempt"]: Current attempt number
        - input_data["max_attempts"]: Maximum retry attempts
        - input_data["backoff_ms"]: Backoff delay in milliseconds

        Returns:
            Modified context with retry decisions
        """
        storage_type = context.input_data.get("storage_type", "unknown")
        operation = context.input_data.get("operation", "unknown")
        attempt = context.input_data.get("attempt", 1)
        max_attempts = context.input_data.get("max_attempts", 3)

        logger.warning(
            f"[{self.name}] Retry {storage_type}.{operation}: "
            f"attempt {attempt}/{max_attempts}"
        )

        return context


# Example implementations

class QueryTimingHook(QueryHook):
    """Example: Measure and log query execution times."""

    def __init__(self, slow_query_threshold_ms: float = 1000):
        super().__init__(
            name="query_timing",
            priority=HookPriority.HIGH,
        )
        self.slow_query_threshold = slow_query_threshold_ms
        self.query_times: list[float] = []

    async def execute(self, context: HookContext) -> HookContext:
        if context.phase.value == "pre":
            # Record start time
            context.metadata["query_start"] = time.time()

        elif context.phase.value == "post":
            # Calculate duration
            start = context.metadata.get("query_start", time.time())
            duration_ms = (time.time() - start) * 1000
            self.query_times.append(duration_ms)

            if context.output_data:
                context.output_data["duration_ms"] = duration_ms

            # Log slow queries
            if duration_ms > self.slow_query_threshold:
                query = context.input_data.get("query", "")
                logger.warning(
                    f"SLOW QUERY ({duration_ms:.2f}ms): {str(query)[:200]}..."
                )

        return context

    def get_stats(self) -> dict:
        """Get query timing statistics."""
        stats = super().get_stats()
        if self.query_times:
            stats["timing"] = {
                "count": len(self.query_times),
                "avg_ms": sum(self.query_times) / len(self.query_times),
                "min_ms": min(self.query_times),
                "max_ms": max(self.query_times),
                "p95_ms": sorted(self.query_times)[int(len(self.query_times) * 0.95)],
            }
        return stats


class ConnectionPoolMonitorHook(ConnectionHook):
    """Example: Monitor connection pool health."""

    def __init__(self):
        super().__init__(
            name="connection_pool_monitor",
            priority=HookPriority.NORMAL,
        )
        self.active_connections = 0

    async def execute(self, context: HookContext) -> HookContext:
        event = context.input_data.get("event", "")

        if event == "connect":
            self.active_connections += 1
            logger.info(f"Active connections: {self.active_connections}")
        elif event == "disconnect":
            self.active_connections = max(0, self.active_connections - 1)
            logger.info(f"Active connections: {self.active_connections}")

        context.metadata["active_connections"] = self.active_connections
        return context


class ExponentialBackoffRetryHook(RetryHook):
    """Example: Implement exponential backoff for retries."""

    def __init__(
        self,
        base_delay_ms: float = 100,
        max_delay_ms: float = 30000,
        multiplier: float = 2.0,
    ):
        super().__init__(
            name="exponential_backoff",
            priority=HookPriority.CRITICAL,
        )
        self.base_delay = base_delay_ms
        self.max_delay = max_delay_ms
        self.multiplier = multiplier

    async def execute(self, context: HookContext) -> HookContext:
        import asyncio

        attempt = context.input_data.get("attempt", 1)

        # Calculate backoff delay
        delay_ms = min(
            self.base_delay * (self.multiplier ** (attempt - 1)),
            self.max_delay,
        )

        logger.info(
            f"Exponential backoff: attempt {attempt}, "
            f"waiting {delay_ms:.0f}ms"
        )

        # Update context
        if context.input_data:
            context.input_data["backoff_ms"] = delay_ms

        # Apply delay
        await asyncio.sleep(delay_ms / 1000)

        return context


class CircuitBreakerHook(ErrorHook):
    """Example: Implement circuit breaker pattern."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: int = 60,
    ):
        super().__init__(
            name="circuit_breaker",
            priority=HookPriority.CRITICAL,
        )
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout_seconds
        self.failures: dict[str, list[float]] = {}
        self.open_circuits: set[str] = set()

    async def execute(self, context: HookContext) -> HookContext:
        storage_type = context.input_data.get("storage_type", "unknown")
        now = time.time()

        # Initialize failure tracking
        if storage_type not in self.failures:
            self.failures[storage_type] = []

        # Clean old failures
        cutoff = now - self.reset_timeout
        self.failures[storage_type] = [
            t for t in self.failures[storage_type]
            if t > cutoff
        ]

        # Record new failure
        self.failures[storage_type].append(now)

        # Check threshold
        if len(self.failures[storage_type]) >= self.failure_threshold:
            if storage_type not in self.open_circuits:
                self.open_circuits.add(storage_type)
                logger.error(
                    f"CIRCUIT BREAKER OPEN: {storage_type} "
                    f"({self.failure_threshold} failures)"
                )
        else:
            # Close circuit if within threshold
            if storage_type in self.open_circuits:
                self.open_circuits.remove(storage_type)
                logger.info(f"CIRCUIT BREAKER CLOSED: {storage_type}")

        context.metadata["circuit_open"] = storage_type in self.open_circuits
        return context

    def is_open(self, storage_type: str) -> bool:
        """Check if circuit is open for storage type."""
        return storage_type in self.open_circuits


class QueryCacheHook(QueryHook):
    """Example: Cache query results for read operations."""

    def __init__(self, cache_size: int = 1000, ttl_seconds: int = 300):
        super().__init__(
            name="query_cache",
            priority=HookPriority.HIGH,
        )
        self.cache: dict[str, tuple[float, Any]] = {}
        self.cache_size = cache_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _make_cache_key(self, context: HookContext) -> str:
        """Generate cache key from query context."""
        query = str(context.input_data.get("query", ""))
        params = str(context.input_data.get("parameters", {}))
        return f"{query}:{params}"

    async def execute(self, context: HookContext) -> HookContext:
        query_type = context.input_data.get("query_type", "")

        # Only cache read queries
        if query_type != "read":
            return context

        cache_key = self._make_cache_key(context)
        now = time.time()

        if context.phase.value == "pre":
            # Check cache
            if cache_key in self.cache:
                timestamp, result = self.cache[cache_key]
                if now - timestamp < self.ttl:
                    # Cache hit
                    self.hits += 1
                    context.metadata["cache_hit"] = True
                    context.metadata["cached_result"] = result
                    logger.debug(f"Query cache HIT: {cache_key[:50]}...")
                    return context

            # Cache miss
            self.misses += 1
            context.metadata["cache_hit"] = False

        elif context.phase.value == "post" and not context.metadata.get("cache_hit"):
            # Store result in cache
            if context.output_data:
                result = context.output_data.get("result")

                # Evict old entries if needed
                if len(self.cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
                    del self.cache[oldest_key]

                self.cache[cache_key] = (now, result)
                logger.debug(f"Query cache STORE: {cache_key[:50]}...")

        return context
