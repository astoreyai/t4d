"""
Observability module for T4DM.

Provides structured logging, metrics, health checks, distributed tracing, and performance monitoring.
"""

from t4dm.observability.health import (
    ComponentHealth,
    HealthChecker,
    HealthStatus,
    SystemHealth,
    get_health_checker,
)
from t4dm.observability.logging import (
    OperationLogger,
    clear_context,
    configure_logging,
    get_logger,
    log_operation,
    set_context,
)
from t4dm.observability.metrics import (
    AsyncTimer,
    MetricsCollector,
    Timer,
    count_operation,
    get_metrics,
    timed_operation,
)
from t4dm.observability.prometheus import (
    PROMETHEUS_AVAILABLE,
    InternalCounter,
    InternalGauge,
    InternalHistogram,
    WWMetrics,
    count_calls,
    reset_metrics,
    track_latency,
)
from t4dm.observability.prometheus import (
    get_metrics as get_prometheus_metrics,
)
from t4dm.observability.tracing import (
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
from t4dm.observability.integration_metrics import (
    ConsciousnessMetrics,  # Backward compat alias
    IITMetricsComputer,
    IntegrationMetrics,
)
from t4dm.observability.decision_trace import (
    DecisionTrace,
    DecisionTracer,
    disable_decision_tracing,
    enable_decision_tracing,
    get_decision_tracer,
    is_decision_tracing_enabled,
    reset_decision_tracer,
    traced_decision,
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
    # IIT-inspired Integration Metrics (renamed from ConsciousnessMetrics)
    "IntegrationMetrics",
    "ConsciousnessMetrics",  # Backward compat alias
    "IITMetricsComputer",
    # Decision Tracing for Bio-Inspired Components
    "DecisionTrace",
    "DecisionTracer",
    "traced_decision",
    "get_decision_tracer",
    "enable_decision_tracing",
    "disable_decision_tracing",
    "is_decision_tracing_enabled",
    "reset_decision_tracer",
]
