"""
Metrics Collection for T4DM.

Provides counters, timers, and gauges for monitoring memory operations.
Thread-safe and async-compatible.
"""

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Optional

# Thread-safe singleton for metrics
_metrics_instance: Optional["MetricsCollector"] = None
_metrics_lock = threading.Lock()


@dataclass
class OperationMetrics:
    """Metrics for a single operation type."""
    count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    last_called: datetime | None = None

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        if self.count == 0:
            return 0.0
        return self.total_duration_ms / self.count

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.count == 0:
            return 1.0
        return (self.count - self.error_count) / self.count

    def record(self, duration_ms: float, error: bool = False) -> None:
        """Record an operation."""
        self.count += 1
        self.total_duration_ms += duration_ms
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        self.last_called = datetime.utcnow()
        if error:
            self.error_count += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "error_count": self.error_count,
            "success_rate": round(self.success_rate, 4),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "min_duration_ms": round(self.min_duration_ms, 2) if self.count > 0 else 0,
            "max_duration_ms": round(self.max_duration_ms, 2),
            "last_called": self.last_called.isoformat() if self.last_called else None,
        }


class MetricsCollector:
    """
    Collects and aggregates metrics for T4DM operations.

    Thread-safe singleton that tracks:
    - Operation counts and timing
    - Error rates
    - Database operation stats
    - Memory type breakdowns
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._lock = threading.Lock()
        self._operations: dict[str, OperationMetrics] = defaultdict(OperationMetrics)
        self._gauges: dict[str, float] = {}
        self._start_time = datetime.utcnow()
        # MEM-001 FIX: Limit maximum tracked operations and gauges
        self._max_operations = 10000
        self._max_gauges = 1000

    def record_operation(
        self,
        operation: str,
        duration_ms: float,
        error: bool = False,
        **tags,
    ) -> None:
        """
        Record an operation execution.

        Args:
            operation: Operation name (e.g., "episodic.create")
            duration_ms: Duration in milliseconds
            error: Whether operation failed
            **tags: Additional tags for categorization
        """
        # Build key with tags
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            key = f"{operation}[{tag_str}]"
        else:
            key = operation

        with self._lock:
            # MEM-001 FIX: Check capacity before adding new operation
            if key not in self._operations and len(self._operations) >= self._max_operations:
                self._cleanup_old_operations()
            self._operations[key].record(duration_ms, error)

    def _cleanup_old_operations(self) -> None:
        """Remove least-used operations to enforce size limit."""
        # Keep operations with highest counts
        sorted_ops = sorted(
            self._operations.items(),
            key=lambda x: x[1].count,
            reverse=True
        )
        keep_keys = set(k for k, _ in sorted_ops[:self._max_operations // 2])
        self._operations = defaultdict(
            OperationMetrics,
            {k: v for k, v in self._operations.items() if k in keep_keys}
        )

    def set_gauge(self, name: str, value: float) -> None:
        """
        Set a gauge value.

        Args:
            name: Gauge name
            value: Current value
        """
        with self._lock:
            # MEM-001 FIX: Check capacity before adding new gauge
            if name not in self._gauges and len(self._gauges) >= self._max_gauges:
                # Just skip new gauges if at capacity (gauges typically don't grow that much)
                return
            self._gauges[name] = value

    def increment_counter(self, name: str, value: int = 1) -> None:
        """
        Increment a simple counter.

        Args:
            name: Counter name
            value: Increment amount
        """
        with self._lock:
            self._operations[name].count += value

    def get_metrics(self) -> dict[str, Any]:
        """
        Get all collected metrics.

        Returns:
            Dictionary of all metrics
        """
        with self._lock:
            return {
                "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
                "operations": {
                    name: metrics.to_dict()
                    for name, metrics in self._operations.items()
                },
                "gauges": dict(self._gauges),
            }

    def get_operation_metrics(self, operation: str) -> dict | None:
        """
        Get metrics for a specific operation.

        Args:
            operation: Operation name

        Returns:
            Metrics dict or None
        """
        with self._lock:
            if operation in self._operations:
                return self._operations[operation].to_dict()
            return None

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Summary of key metrics
        """
        with self._lock:
            total_ops = sum(m.count for m in self._operations.values())
            total_errors = sum(m.error_count for m in self._operations.values())
            total_duration = sum(m.total_duration_ms for m in self._operations.values())

            # Find slowest operations
            slowest = sorted(
                [(k, v.avg_duration_ms) for k, v in self._operations.items() if v.count > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            # Find most erroring operations
            most_errors = sorted(
                [(k, v.error_count, v.success_rate) for k, v in self._operations.items() if v.error_count > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            return {
                "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
                "total_operations": total_ops,
                "total_errors": total_errors,
                "overall_success_rate": round((total_ops - total_errors) / max(total_ops, 1), 4),
                "total_duration_ms": round(total_duration, 2),
                "avg_duration_ms": round(total_duration / max(total_ops, 1), 2),
                "slowest_operations": [
                    {"operation": k, "avg_ms": round(v, 2)} for k, v in slowest
                ],
                "error_prone_operations": [
                    {"operation": k, "errors": e, "success_rate": round(s, 4)}
                    for k, e, s in most_errors
                ],
                "gauges": dict(self._gauges),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._operations.clear()
            self._gauges.clear()
            self._start_time = datetime.utcnow()


def get_metrics() -> MetricsCollector:
    """
    Get the singleton metrics collector.

    Returns:
        MetricsCollector instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = MetricsCollector()
    return _metrics_instance


def timed_operation(
    operation: str,
    **tags,
) -> Callable:
    """
    Decorator for timing async operations.

    Args:
        operation: Operation name
        **tags: Additional categorization tags

    Example:
        @timed_operation("episodic.create", memory_type="episode")
        async def create(self, content: str) -> Episode:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            error = False
            try:
                return await func(*args, **kwargs)
            except Exception:
                error = True
                raise
            finally:
                duration_ms = (time.time() - start) * 1000
                get_metrics().record_operation(operation, duration_ms, error, **tags)
        return wrapper
    return decorator


def count_operation(operation: str) -> Callable:
    """
    Decorator for counting operation calls.

    Args:
        operation: Operation name

    Example:
        @count_operation("api.requests")
        async def handle_request(self):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            get_metrics().increment_counter(operation)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class Timer:
    """
    Context manager for timing code blocks.

    Example:
        with Timer("db.query") as t:
            result = await db.query(...)
        # Automatically records duration
    """

    def __init__(self, operation: str, **tags):
        """
        Initialize timer.

        Args:
            operation: Operation name
            **tags: Additional tags
        """
        self.operation = operation
        self.tags = tags
        self.start_time = 0.0
        self.duration_ms = 0.0

    def __enter__(self) -> "Timer":
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record."""
        self.duration_ms = (time.time() - self.start_time) * 1000
        error = exc_type is not None
        get_metrics().record_operation(self.operation, self.duration_ms, error, **self.tags)


class AsyncTimer:
    """
    Async context manager for timing code blocks.

    Example:
        async with AsyncTimer("db.query") as t:
            result = await db.query(...)
    """

    def __init__(self, operation: str, **tags):
        """Initialize timer."""
        self.operation = operation
        self.tags = tags
        self.start_time = 0.0
        self.duration_ms = 0.0

    async def __aenter__(self) -> "AsyncTimer":
        """Start timing."""
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record."""
        self.duration_ms = (time.time() - self.start_time) * 1000
        error = exc_type is not None
        get_metrics().record_operation(self.operation, self.duration_ms, error, **self.tags)
