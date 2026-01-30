"""
Prometheus Metrics Integration for World Weaver.

Provides Prometheus-compatible metric collection and export for:
- Memory operations (retrieval, encoding, consolidation)
- Embedding operations (generation, modulation, ensemble)
- Temporal dynamics (phase transitions, session management)
- Storage health (circuit breakers, latency)
- Neuromodulator states (ACh, DA, NE, 5-HT levels)

Compatible with prometheus_client library when available,
falls back to lightweight internal implementation otherwise.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not available, using internal metrics")


# ============================================================================
# Internal Metric Types (fallback when prometheus_client unavailable)
# ============================================================================


@dataclass
class InternalCounter:
    """Lightweight counter metric."""
    name: str
    description: str
    labels: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment counter."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self.labels[label_key] = self.labels.get(label_key, 0) + value

    def get(self, **labels) -> float:
        """Get counter value."""
        label_key = tuple(sorted(labels.items()))
        return self.labels.get(label_key, 0)


@dataclass
class InternalGauge:
    """Lightweight gauge metric."""
    name: str
    description: str
    labels: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float, **labels) -> None:
        """Set gauge value."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self.labels[label_key] = value

    def get(self, **labels) -> float:
        """Get gauge value."""
        label_key = tuple(sorted(labels.items()))
        return self.labels.get(label_key, 0.0)

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment gauge."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self.labels[label_key] = self.labels.get(label_key, 0) + value

    def dec(self, value: float = 1.0, **labels) -> None:
        """Decrement gauge."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self.labels[label_key] = self.labels.get(label_key, 0) - value


@dataclass
class InternalHistogram:
    """Lightweight histogram metric."""
    name: str
    description: str
    buckets: tuple = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    labels: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, **labels) -> None:
        """Record observation."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            if label_key not in self.labels:
                self.labels[label_key] = []
            self.labels[label_key].append(value)

    def get_percentile(self, percentile: float, **labels) -> float:
        """Get percentile value."""
        label_key = tuple(sorted(labels.items()))
        values = self.labels.get(label_key, [])
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]


# ============================================================================
# World Weaver Metrics Registry
# ============================================================================


class WWMetrics:
    """
    World Weaver Prometheus Metrics.

    Provides standardized metrics for all system components.

    Usage:
        metrics = WWMetrics()

        # Record retrieval
        metrics.memory_retrieval_total.inc(memory_type="episodic")
        metrics.memory_retrieval_latency.observe(0.05, memory_type="episodic")

        # Set neuromodulator state
        metrics.neuromodulator_level.set(0.8, modulator="acetylcholine")

        # Export for Prometheus scraping
        output = metrics.export()
    """

    def __init__(self, prefix: str = "ww"):
        """Initialize metrics with optional prefix."""
        self._prefix = prefix
        self._use_prometheus = PROMETHEUS_AVAILABLE

        if self._use_prometheus:
            self._registry = CollectorRegistry()
            self._init_prometheus_metrics()
        else:
            self._init_internal_metrics()

        logger.info(f"WWMetrics initialized (prometheus={self._use_prometheus})")

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus client metrics."""
        # Memory operations
        self.memory_retrieval_total = Counter(
            f"{self._prefix}_memory_retrieval_total",
            "Total memory retrievals",
            ["memory_type", "session_id"],
            registry=self._registry,
        )
        self.memory_retrieval_latency = Histogram(
            f"{self._prefix}_memory_retrieval_latency_seconds",
            "Memory retrieval latency",
            ["memory_type"],
            registry=self._registry,
        )
        self.memory_encoding_total = Counter(
            f"{self._prefix}_memory_encoding_total",
            "Total memory encodings",
            ["memory_type"],
            registry=self._registry,
        )

        # Embedding operations
        self.embedding_generation_total = Counter(
            f"{self._prefix}_embedding_generation_total",
            "Total embedding generations",
            ["adapter_type"],
            registry=self._registry,
        )
        self.embedding_generation_latency = Histogram(
            f"{self._prefix}_embedding_generation_latency_seconds",
            "Embedding generation latency",
            ["adapter_type"],
            registry=self._registry,
        )
        self.embedding_cache_hits = Counter(
            f"{self._prefix}_embedding_cache_hits_total",
            "Embedding cache hits",
            registry=self._registry,
        )
        self.embedding_cache_misses = Counter(
            f"{self._prefix}_embedding_cache_misses_total",
            "Embedding cache misses",
            registry=self._registry,
        )

        # Ensemble operations
        self.ensemble_adapters_healthy = Gauge(
            f"{self._prefix}_ensemble_adapters_healthy",
            "Number of healthy ensemble adapters",
            registry=self._registry,
        )
        self.ensemble_adapter_failures = Counter(
            f"{self._prefix}_ensemble_adapter_failures_total",
            "Ensemble adapter failures",
            ["adapter_index"],
            registry=self._registry,
        )

        # Modulation
        self.modulation_operations_total = Counter(
            f"{self._prefix}_modulation_operations_total",
            "Total embedding modulation operations",
            ["cognitive_mode"],
            registry=self._registry,
        )

        # Temporal dynamics
        self.temporal_phase = Gauge(
            f"{self._prefix}_temporal_phase",
            "Current temporal phase (0=active, 1=idle, 2=consolidating, 3=sleeping)",
            registry=self._registry,
        )
        self.active_sessions = Gauge(
            f"{self._prefix}_active_sessions",
            "Number of active sessions",
            registry=self._registry,
        )
        self.session_duration = Histogram(
            f"{self._prefix}_session_duration_seconds",
            "Session durations",
            registry=self._registry,
        )

        # Neuromodulators
        self.neuromodulator_level = Gauge(
            f"{self._prefix}_neuromodulator_level",
            "Current neuromodulator level",
            ["modulator"],
            registry=self._registry,
        )

        # Plasticity
        self.reconsolidation_updates_total = Counter(
            f"{self._prefix}_reconsolidation_updates_total",
            "Total reconsolidation updates",
            registry=self._registry,
        )
        self.eligibility_traces_active = Gauge(
            f"{self._prefix}_eligibility_traces_active",
            "Active eligibility traces",
            registry=self._registry,
        )

        # Storage
        self.storage_operations_total = Counter(
            f"{self._prefix}_storage_operations_total",
            "Total storage operations",
            ["backend", "operation"],
            registry=self._registry,
        )
        self.storage_latency = Histogram(
            f"{self._prefix}_storage_latency_seconds",
            "Storage operation latency",
            ["backend", "operation"],
            registry=self._registry,
        )
        self.circuit_breaker_state = Gauge(
            f"{self._prefix}_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half_open, 2=open)",
            ["backend"],
            registry=self._registry,
        )

        # Consolidation
        self.consolidation_cycles_total = Counter(
            f"{self._prefix}_consolidation_cycles_total",
            "Total consolidation cycles",
            ["phase"],
            registry=self._registry,
        )
        self.memories_consolidated = Counter(
            f"{self._prefix}_memories_consolidated_total",
            "Memories consolidated",
            ["phase"],
            registry=self._registry,
        )

    def _init_internal_metrics(self) -> None:
        """Initialize internal fallback metrics."""
        # Memory operations
        self.memory_retrieval_total = InternalCounter(
            f"{self._prefix}_memory_retrieval_total",
            "Total memory retrievals",
        )
        self.memory_retrieval_latency = InternalHistogram(
            f"{self._prefix}_memory_retrieval_latency_seconds",
            "Memory retrieval latency",
        )
        self.memory_encoding_total = InternalCounter(
            f"{self._prefix}_memory_encoding_total",
            "Total memory encodings",
        )

        # Embedding operations
        self.embedding_generation_total = InternalCounter(
            f"{self._prefix}_embedding_generation_total",
            "Total embedding generations",
        )
        self.embedding_generation_latency = InternalHistogram(
            f"{self._prefix}_embedding_generation_latency_seconds",
            "Embedding generation latency",
        )
        self.embedding_cache_hits = InternalCounter(
            f"{self._prefix}_embedding_cache_hits_total",
            "Embedding cache hits",
        )
        self.embedding_cache_misses = InternalCounter(
            f"{self._prefix}_embedding_cache_misses_total",
            "Embedding cache misses",
        )

        # Ensemble operations
        self.ensemble_adapters_healthy = InternalGauge(
            f"{self._prefix}_ensemble_adapters_healthy",
            "Number of healthy ensemble adapters",
        )
        self.ensemble_adapter_failures = InternalCounter(
            f"{self._prefix}_ensemble_adapter_failures_total",
            "Ensemble adapter failures",
        )

        # Modulation
        self.modulation_operations_total = InternalCounter(
            f"{self._prefix}_modulation_operations_total",
            "Total embedding modulation operations",
        )

        # Temporal dynamics
        self.temporal_phase = InternalGauge(
            f"{self._prefix}_temporal_phase",
            "Current temporal phase",
        )
        self.active_sessions = InternalGauge(
            f"{self._prefix}_active_sessions",
            "Number of active sessions",
        )
        self.session_duration = InternalHistogram(
            f"{self._prefix}_session_duration_seconds",
            "Session durations",
        )

        # Neuromodulators
        self.neuromodulator_level = InternalGauge(
            f"{self._prefix}_neuromodulator_level",
            "Current neuromodulator level",
        )

        # Plasticity
        self.reconsolidation_updates_total = InternalCounter(
            f"{self._prefix}_reconsolidation_updates_total",
            "Total reconsolidation updates",
        )
        self.eligibility_traces_active = InternalGauge(
            f"{self._prefix}_eligibility_traces_active",
            "Active eligibility traces",
        )

        # Storage
        self.storage_operations_total = InternalCounter(
            f"{self._prefix}_storage_operations_total",
            "Total storage operations",
        )
        self.storage_latency = InternalHistogram(
            f"{self._prefix}_storage_latency_seconds",
            "Storage operation latency",
        )
        self.circuit_breaker_state = InternalGauge(
            f"{self._prefix}_circuit_breaker_state",
            "Circuit breaker state",
        )

        # Consolidation
        self.consolidation_cycles_total = InternalCounter(
            f"{self._prefix}_consolidation_cycles_total",
            "Total consolidation cycles",
        )
        self.memories_consolidated = InternalCounter(
            f"{self._prefix}_memories_consolidated_total",
            "Memories consolidated",
        )

    def export(self) -> str:
        """Export metrics in Prometheus format."""
        if self._use_prometheus:
            return generate_latest(self._registry).decode("utf-8")
        return self._export_internal()

    def _export_internal(self) -> str:
        """Export internal metrics in Prometheus text format."""
        lines = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, (InternalCounter, InternalGauge)):
                for labels, value in attr.labels.items():
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels) if labels else ""
                    metric_name = attr.name
                    if label_str:
                        lines.append(f"{metric_name}{{{label_str}}} {value}")
                    else:
                        lines.append(f"{metric_name} {value}")
            elif isinstance(attr, InternalHistogram):
                for labels, values in attr.labels.items():
                    if values:
                        label_str = ",".join(f'{k}="{v}"' for k, v in labels) if labels else ""
                        count = len(values)
                        total = sum(values)
                        metric_name = attr.name
                        if label_str:
                            lines.append(f"{metric_name}_count{{{label_str}}} {count}")
                            lines.append(f"{metric_name}_sum{{{label_str}}} {total}")
                        else:
                            lines.append(f"{metric_name}_count {count}")
                            lines.append(f"{metric_name}_sum {total}")
        return "\n".join(lines)

    def get_content_type(self) -> str:
        """Get content type for metrics endpoint."""
        if self._use_prometheus:
            return CONTENT_TYPE_LATEST
        return "text/plain; charset=utf-8"


# ============================================================================
# Singleton Access
# ============================================================================


_metrics_instance: WWMetrics | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> WWMetrics:
    """Get or create singleton metrics instance."""
    global _metrics_instance
    with _metrics_lock:
        if _metrics_instance is None:
            _metrics_instance = WWMetrics()
        return _metrics_instance


def reset_metrics() -> None:
    """Reset metrics instance (for testing)."""
    global _metrics_instance
    with _metrics_lock:
        _metrics_instance = None


# ============================================================================
# Decorators
# ============================================================================


def track_latency(metric_name: str, **static_labels):
    """
    Decorator to track function latency.

    Usage:
        @track_latency("memory_retrieval", memory_type="episodic")
        async def retrieve_episode(query: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics()
            histogram = getattr(metrics, f"{metric_name}_latency", None)
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                if histogram:
                    _observe_histogram(histogram, duration, static_labels)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = get_metrics()
            histogram = getattr(metrics, f"{metric_name}_latency", None)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                if histogram:
                    _observe_histogram(histogram, duration, static_labels)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _observe_histogram(histogram: Any, value: float, labels: dict) -> None:
    """Observe histogram value, handling both prometheus and internal types."""
    if isinstance(histogram, InternalHistogram):
        histogram.observe(value, **labels)
    elif PROMETHEUS_AVAILABLE and hasattr(histogram, "labels"):
        # Prometheus client: use labels() method
        if labels:
            histogram.labels(**labels).observe(value)
        else:
            histogram.observe(value)
    else:
        try:
            histogram.observe(value, **labels)
        except TypeError:
            histogram.observe(value)


def _increment_counter(counter: Any, labels: dict) -> None:
    """Increment counter, handling both prometheus and internal types."""
    if isinstance(counter, InternalCounter):
        counter.inc(**labels)
    elif PROMETHEUS_AVAILABLE and hasattr(counter, "labels"):
        # Prometheus client: use labels() method
        if labels:
            counter.labels(**labels).inc()
        else:
            counter.inc()
    else:
        try:
            counter.inc(**labels)
        except TypeError:
            counter.inc()


def count_calls(metric_name: str, **static_labels):
    """
    Decorator to count function calls.

    Usage:
        @count_calls("memory_encoding", memory_type="episodic")
        async def encode_episode(content: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics()
            counter = getattr(metrics, f"{metric_name}_total", None)
            if counter:
                _increment_counter(counter, static_labels)
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = get_metrics()
            counter = getattr(metrics, f"{metric_name}_total", None)
            if counter:
                _increment_counter(counter, static_labels)
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


import asyncio  # Move to top if not there

# ============================================================================
# FastAPI Router for /metrics endpoint (Phase 10.2)
# ============================================================================

try:
    from fastapi import APIRouter, Response

    prometheus_router = APIRouter()

    @prometheus_router.get(
        "/metrics",
        summary="Prometheus metrics",
        description="Expose metrics in Prometheus text format for scraping",
        response_class=Response,
        tags=["Observability"],
    )
    async def metrics_endpoint():
        """
        Prometheus metrics endpoint.

        Returns metrics in Prometheus text exposition format.
        Compatible with Prometheus scraping and Grafana dashboards.

        Metrics include:
        - Memory operations (retrieval, encoding, consolidation)
        - Embedding operations (generation, cache hits/misses)
        - Neuromodulator states (ACh, DA, NE, 5-HT levels)
        - Storage health (circuit breakers, latency)
        - Temporal dynamics (phase, active sessions)
        """
        metrics = get_metrics()
        content = metrics.export()
        content_type = metrics.get_content_type()
        return Response(content=content, media_type=content_type)

    ROUTER_AVAILABLE = True
except ImportError:
    prometheus_router = None
    ROUTER_AVAILABLE = False
    logger.debug("FastAPI not available, prometheus router disabled")


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "ROUTER_AVAILABLE",
    "InternalCounter",
    "InternalGauge",
    "InternalHistogram",
    "WWMetrics",
    "count_calls",
    "get_metrics",
    "prometheus_router",
    "reset_metrics",
    "track_latency",
]
