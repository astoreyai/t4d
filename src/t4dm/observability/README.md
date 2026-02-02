# Observability Module

**6 files | ~2,200 lines | Centrality: 6**

The observability module provides integrated monitoring, tracing, logging, and health checks for T4DM with OpenTelemetry, Prometheus, and structured JSON logging.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OBSERVABILITY STACK                              │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        TRACING (OpenTelemetry)                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐│ │
│  │  │ @traced      │  │ trace_span   │  │ OTLP Export (gRPC/TLS)     ││ │
│  │  │ @traced_sync │  │ context mgr  │  │ Batch: 512 spans / 5s      ││ │
│  │  └──────────────┘  └──────────────┘  └────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        METRICS (Prometheus)                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐│ │
│  │  │ Counters     │  │ Gauges       │  │ Histograms                 ││ │
│  │  │ retrieval_   │  │ neuromod_    │  │ latency_seconds            ││ │
│  │  │ total        │  │ level        │  │ (0.005...10s buckets)      ││ │
│  │  └──────────────┘  └──────────────┘  └────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        LOGGING (Structured JSON)                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐│ │
│  │  │ LogContext   │  │ AuditLogger  │  │ LOG-001: Injection         ││ │
│  │  │ session_id   │  │ P3-SEC-L5    │  │ Prevention (sanitize)      ││ │
│  │  │ operation_id │  │ auth events  │  │                            ││ │
│  │  └──────────────┘  └──────────────┘  └────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        HEALTH (Async Checks)                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐│ │
│  │  │ check_qdrant │  │ check_neo4j  │  │ Kubernetes Probes          ││ │
│  │  │ check_embed  │  │ check_metrics│  │ liveness / readiness       ││ │
│  │  └──────────────┘  └──────────────┘  └────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `tracing.py` | ~400 | OpenTelemetry distributed tracing |
| `prometheus.py` | ~600 | Prometheus metrics with dual-mode support |
| `metrics.py` | ~370 | Internal metrics collection |
| `logging.py` | ~530 | Structured JSON logging + audit trail |
| `health.py` | ~300 | Async health checks for components |
| `__init__.py` | ~100 | 33 public exports |

## Components

### Tracing (OpenTelemetry)

W3C trace context propagation with secure TLS:

```python
from t4dm.observability import init_tracing, traced, trace_span, get_tracer

# Initialize tracing
tracer = init_tracing(
    service_name="t4dm",
    otlp_endpoint="https://otel-collector:4317"
)

# Decorator (async)
@traced("create_episode", kind=SpanKind.SERVER)
async def create_episode(content: str) -> Episode:
    add_span_attribute("content_length", len(content))
    return await memory.create(content)

# Decorator (sync)
@traced_sync("compute_similarity")
def compute_similarity(a: list, b: list) -> float:
    return cosine_similarity(a, b)

# Context manager
async with trace_span("batch_operation", batch_size=100) as span:
    for item in batch:
        await process(item)
    add_span_event("batch_complete", {"count": len(batch)})

# Get trace context for propagation
context = get_trace_context()
# {'traceparent': '00-abc123-def456-01', 'tracestate': '...'}
```

**Configuration**:
```bash
T4DM_OTEL_ENABLED=true
T4DM_OTEL_ENDPOINT=https://otel-collector:4317
T4DM_OTEL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
T4DM_OTEL_HEADERS='{"Authorization": "Bearer token"}'
T4DM_OTEL_SERVICE_NAME=t4dm
T4DM_OTEL_BATCH_DELAY_MS=5000
T4DM_OTEL_MAX_EXPORT_BATCH_SIZE=512
```

### Metrics (Prometheus)

Dual-mode: uses `prometheus_client` when available, falls back to internal:

```python
from t4dm.observability import (
    get_prometheus_metrics, track_latency, count_calls,
    PROMETHEUS_AVAILABLE
)

metrics = get_prometheus_metrics()

# Decorators
@track_latency("memory_retrieval", memory_type="episode")
async def retrieve(query: str):
    return await memory.search(query)

@count_calls("encoding_operations", adapter="bge_m3")
async def encode(text: str):
    return await embedder.embed(text)

# Manual metrics
metrics.memory_retrieval_total.inc(memory_type="episode", session_id="abc")
metrics.embedding_generation_latency_seconds.observe(0.05, adapter_type="bge_m3")
metrics.neuromodulator_level.set(0.7, modulator="dopamine")
metrics.circuit_breaker_state.set(0, backend="qdrant")  # 0=closed, 1=half_open, 2=open

# Export for Prometheus scraping
text = metrics.export()
content_type = metrics.get_content_type()
```

**Available Metrics**:

| Metric | Type | Labels |
|--------|------|--------|
| `ww_memory_retrieval_total` | Counter | memory_type, session_id |
| `ww_memory_retrieval_latency_seconds` | Histogram | memory_type |
| `ww_embedding_generation_total` | Counter | adapter_type |
| `ww_embedding_cache_hits_total` | Counter | - |
| `ww_neuromodulator_level` | Gauge | modulator |
| `ww_storage_operations_total` | Counter | backend, operation |
| `ww_circuit_breaker_state` | Gauge | backend |
| `ww_consolidation_cycles_total` | Counter | phase |

### Internal Metrics

Thread-safe application-level metrics:

```python
from t4dm.observability import (
    get_metrics, timed_operation, Timer, AsyncTimer
)

collector = get_metrics()

# Decorator
@timed_operation("episodic.create", memory_type="episode")
async def create_episode(content: str):
    return await memory.create(content)

# Context manager (sync)
with Timer("vector_search"):
    results = search(query)

# Context manager (async)
async with AsyncTimer("graph_query"):
    nodes = await neo4j.query(cypher)

# Manual recording
collector.record_operation("custom_op", duration_ms=45.2, error=False)
collector.set_gauge("active_sessions", 5)
collector.increment_counter("requests_total", 1)

# Get summary
summary = collector.get_summary()
# {'top_5_slowest': [...], 'top_5_error_prone': [...]}
```

**Memory Safety (MEM-001)**:
- Max 10,000 operations tracked
- Max 1,000 gauges
- Automatic cleanup keeps highest-count operations

### Logging (Structured JSON)

JSON-structured logging with security hardening:

```python
from t4dm.observability import (
    configure_logging, get_logger, OperationLogger,
    log_operation, set_context, clear_context
)

# Configure
configure_logging(
    level="INFO",
    json_output=True,
    log_file="/var/log/t4dm.log"
)

# Get logger
logger = get_logger("t4dm.memory")
logger.info("Memory initialized", extra={"session_id": "abc"})

# Set context for all logs in this scope
set_context(session_id="abc", operation_id="create-123")
logger.info("Creating episode")  # Automatically includes session_id
clear_context()

# Operation logging (context manager)
async with OperationLogger("create_episode", session_id="abc") as log:
    episode = await memory.create(content)
    log.set_result(episode_id=str(episode.id))
# Logs start and end with timing, outcome, and result

# Decorator
@log_operation("recall_episodes", session_id="abc")
async def recall(query: str):
    return await memory.search(query)
```

**Log Format**:
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "t4dm.memory",
  "message": "Episode created",
  "session_id": "abc123",
  "operation_id": "create-456",
  "duration_ms": 45.2,
  "extra": {"episode_id": "ep-789"}
}
```

**Security (LOG-001)**: Log injection prevention via message sanitization.

### Audit Logging (P3-SEC-L5)

Security-focused event logging:

```python
from t4dm.observability.logging import get_audit_logger, AuditEventType

audit = get_audit_logger()

# Session events
audit.log_session_created(
    session_id="sess-abc",
    ip="192.168.1.100",
    user_agent="Mozilla/5.0..."
)
audit.log_session_deleted(session_id="sess-abc", reason="timeout")

# Auth events
audit.log_auth_success(session_id="sess-abc", method="api_key")
audit.log_auth_failure(ip="192.168.1.200", reason="invalid_api_key")

# Data events
audit.log_bulk_delete(
    session_id="sess-abc",
    collection="episodes",
    count=50,
    ip="192.168.1.100"
)

# Access control
audit.log_permission_denied(
    session_id="sess-abc",
    resource="admin_panel",
    action="access"
)
audit.log_rate_limit_exceeded(
    ip="192.168.1.100",
    limit=100,
    window_seconds=60
)

# Admin actions
audit.log_admin_action(action="config_change", key="max_sessions")
```

**Audit Event Types**:
- `session.created`, `session.deleted`
- `auth.success`, `auth.failure`
- `bulk.delete`
- `rate_limit.exceeded`
- `permission.denied`
- `config.change`
- `admin.action`

### Health Checks

Async component health monitoring:

```python
from t4dm.observability import (
    get_health_checker, HealthStatus, ComponentHealth, SystemHealth
)

checker = get_health_checker()

# Check individual components
qdrant_health = await checker.check_qdrant()
# ComponentHealth(name='qdrant', status=HEALTHY, latency_ms=5.2)

neo4j_health = await checker.check_neo4j()
embedding_health = await checker.check_embedding()
metrics_health = await checker.check_metrics()

# Check all (parallel)
system_health = await checker.check_all()
# SystemHealth(
#     status=HEALTHY,
#     components=[...],
#     version="1.0.0",
#     uptime_seconds=3600.0
# )

# Kubernetes probes
is_alive = await checker.check_liveness()   # Always True
is_ready = await checker.check_readiness()  # True if not UNHEALTHY

# Convert to dict for API response
health_dict = system_health.to_dict()
```

**Status Aggregation**:
- `HEALTHY`: All components healthy
- `DEGRADED`: Some components degraded (system functional)
- `UNHEALTHY`: Critical component unhealthy

## Configuration

### Development

```python
configure_logging(level="DEBUG", json_output=False)
init_tracing(console_export=True)  # Spans to console
```

### Production

```python
configure_logging(level="INFO", json_output=True, log_file="/var/log/t4dm.log")

# Environment variables
T4DM_OTEL_ENABLED=true
T4DM_OTEL_ENDPOINT=https://otel-collector:4317
T4DM_OTEL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
```

## Public API

```python
# Logging (6 symbols)
configure_logging, get_logger, log_operation
OperationLogger, set_context, clear_context

# Metrics (6 symbols)
MetricsCollector, get_metrics, timed_operation
count_operation, Timer, AsyncTimer

# Health (5 symbols)
HealthChecker, HealthStatus, ComponentHealth
SystemHealth, get_health_checker

# Tracing (9 symbols)
init_tracing, get_tracer, shutdown_tracing
traced, traced_sync, trace_span
add_span_attribute, add_span_event, get_trace_context

# Prometheus (8 symbols)
PROMETHEUS_AVAILABLE, WWMetrics, get_prometheus_metrics
reset_metrics, track_latency, count_calls
InternalCounter, InternalGauge, InternalHistogram
```

## Complete Example

```python
from t4dm.observability import (
    configure_logging, init_tracing, get_health_checker,
    get_metrics, get_prometheus_metrics, traced, OperationLogger
)

# Setup
configure_logging(level="INFO", json_output=True)
init_tracing()

# Decorated function with full observability
@traced("create_episode", kind=SpanKind.SERVER)
@timed_operation("episodic.create")
async def create_episode(content: str, session_id: str):
    async with OperationLogger("create_episode", session_id=session_id) as log:
        episode = await episodic.create(content)
        log.set_result(episode_id=str(episode.id))
        return episode

# Health check endpoint
async def health_endpoint():
    checker = get_health_checker()
    health = await checker.check_all()
    return {
        "status": health.status.value,
        "components": [c.to_dict() for c in health.components],
        "ready": await checker.check_readiness()
    }

# Metrics endpoint
async def metrics_endpoint():
    metrics = get_prometheus_metrics()
    return Response(
        content=metrics.export(),
        media_type=metrics.get_content_type()
    )
```

## Testing

```bash
# Run observability tests
pytest tests/observability/ -v

# With coverage
pytest tests/observability/ --cov=t4dm.observability

# Security tests (log injection)
pytest tests/observability/test_logging.py -v -k injection
```

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `record_operation()` | O(1) | Lock + dict insert |
| `get_summary()` | O(n) | Sorts operations (n ≤ 10k) |
| `trace_span()` | O(1) | Span creation |
| `check_all()` | O(4) | 4 parallel checks, 5s timeout |
| `log_message()` | O(1) | JSON serialization |

## Security

- **LOG-001**: Message sanitization prevents log injection
- **P3-SEC-L5**: Audit trail for compliance/forensics
- **TLS**: Custom certificates for OTLP export
- **Thread Safety**: All singletons use locks
