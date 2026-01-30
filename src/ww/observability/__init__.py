"""
Observability module for World Weaver.

Provides structured logging, metrics, health checks, distributed tracing, and performance monitoring.
"""

from ww.observability.health import (
    ComponentHealth,
    HealthChecker,
    HealthStatus,
    SystemHealth,
    get_health_checker,
)
from ww.observability.logging import (
    OperationLogger,
    clear_context,
    configure_logging,
    get_logger,
    log_operation,
    set_context,
)
from ww.observability.metrics import (
    AsyncTimer,
    MetricsCollector,
    Timer,
    count_operation,
    get_metrics,
    timed_operation,
)
from ww.observability.prometheus import (
    PROMETHEUS_AVAILABLE,
    InternalCounter,
    InternalGauge,
    InternalHistogram,
    WWMetrics,
    count_calls,
    reset_metrics,
    track_latency,
)
from ww.observability.prometheus import (
    get_metrics as get_prometheus_metrics,
)
from ww.observability.tracing import (
    add_span_attribute,
    add_span_event,
    get_trace_context,
    get_tracer,
    init_tracing,
    shutdown_tracing,
    trace_span,
    traced,
    traced_sync,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "log_operation",
    "OperationLogger",
    "set_context",
    "clear_context",
    # Metrics
    "MetricsCollector",
    "get_metrics",
    "timed_operation",
    "count_operation",
    "Timer",
    "AsyncTimer",
    # Health
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "get_health_checker",
    # Tracing
    "init_tracing",
    "get_tracer",
    "shutdown_tracing",
    "traced",
    "traced_sync",
    "trace_span",
    "add_span_attribute",
    "add_span_event",
    "get_trace_context",
    # Prometheus
    "PROMETHEUS_AVAILABLE",
    "WWMetrics",
    "get_prometheus_metrics",
    "reset_metrics",
    "track_latency",
    "count_calls",
    "InternalCounter",
    "InternalGauge",
    "InternalHistogram",
]
