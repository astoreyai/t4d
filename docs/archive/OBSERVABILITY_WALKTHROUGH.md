# World Weaver Observability Walkthrough

**Version**: 0.1.0 | **Last Updated**: 2025-12-09

A comprehensive guide to monitoring, logging, tracing, and health checking in the World Weaver memory system.

---

## Table of Contents

1. [Overview](#overview)
2. [Structured Logging](#structured-logging)
3. [Metrics Collection](#metrics-collection)
4. [Health Checks](#health-checks)
5. [Distributed Tracing](#distributed-tracing)
6. [Prometheus Integration](#prometheus-integration)
7. [WebSocket Events](#websocket-events)
8. [Integration Patterns](#integration-patterns)

---

## Overview

The observability module provides five pillars of system insight:

| Pillar | Module | Purpose |
|--------|--------|---------|
| **Logging** | `logging.py` | Structured JSON logs with context |
| **Metrics** | `metrics.py` | Counters, gauges, timing |
| **Health** | `health.py` | Component health checks |
| **Tracing** | `tracing.py` | OpenTelemetry distributed tracing |
| **Prometheus** | `prometheus.py` | Prometheus-compatible metrics |

### Module Location

```
ww/observability/
├── __init__.py        # 33 public exports
├── logging.py         # Structured logging
├── metrics.py         # Metrics collection
├── health.py          # Health checks
├── tracing.py         # OpenTelemetry
└── prometheus.py      # Prometheus integration
```

---

## Structured Logging

**File**: `observability/logging.py`

### Configuration

```python
from ww.observability import configure_logging, get_logger

# Configure at startup
configure_logging(
    level="INFO",           # DEBUG, INFO, WARNING, ERROR
    format="json",          # json or text
    output="stdout"         # stdout, stderr, or file path
)

# Get logger for module
logger = get_logger(__name__)
logger.info("Memory system started", extra={"session_id": "abc123"})
```

### Context Management

```python
from ww.observability import set_context, clear_context

# Set request context (thread-local)
set_context(
    request_id="req-001",
    session_id="session-abc",
    user_id="user-123"
)

# All subsequent logs include context
logger.info("Processing request")  # Includes request_id, session_id

# Clear when done
clear_context()
```

### Operation Logging

```python
from ww.observability import OperationLogger, log_operation

# Decorator style
@log_operation("create_episode")
async def create_episode(content: str):
    # Automatically logs start, success/failure, duration
    ...

# Context manager style
with OperationLogger("recall_episodes") as op:
    results = await recall(query)
    op.add_metadata(result_count=len(results))
```

### Log Output (JSON)

```json
{
  "timestamp": "2025-12-09T01:30:00Z",
  "level": "INFO",
  "message": "Episode created",
  "request_id": "req-001",
  "session_id": "session-abc",
  "operation": "create_episode",
  "duration_ms": 45.2,
  "episode_id": "ep-123"
}
```

---

## Metrics Collection

**File**: `observability/metrics.py`

### MetricsCollector

Thread-safe singleton for collecting metrics:

```python
from ww.observability import MetricsCollector, get_metrics

# Get singleton instance
metrics = get_metrics()

# Record operation
metrics.record_operation(
    name="create_episode",
    duration_ms=45.2,
    success=True,
    metadata={"memory_type": "episodic"}
)

# Set gauge value
metrics.set_gauge("active_sessions", 5)
metrics.set_gauge("buffer_size", 128)

# Get statistics
stats = metrics.get_stats("create_episode")
# Returns: {"count": 100, "success_rate": 0.98, "avg_ms": 42.1, "min_ms": 10, "max_ms": 150}
```

### Decorators

```python
from ww.observability import timed_operation, count_operation

@timed_operation("embedding_generation")
async def generate_embedding(text: str):
    # Automatically tracks timing
    return await embedder.embed(text)

@count_operation("cache_hit")
def check_cache(key: str):
    # Counts invocations
    return cache.get(key)
```

### Timer Context Managers

```python
from ww.observability import Timer, AsyncTimer

# Sync timer
with Timer("qdrant_search") as t:
    results = vector_store.search(query)
print(f"Search took {t.duration_ms}ms")

# Async timer
async with AsyncTimer("neo4j_query") as t:
    nodes = await graph_store.query(cypher)
```

### Limits

- Max operations tracked: 10,000
- Max gauges: 1,000
- Automatic cleanup of old entries

---

## Health Checks

**File**: `observability/health.py`

### HealthChecker

```python
from ww.observability import HealthChecker, get_health_checker, HealthStatus

# Get singleton
checker = get_health_checker()

# Check all components
system_health = await checker.check_all()

# Check specific component
qdrant_health = await checker.check_component("qdrant")
```

### Health Status

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"       # Fully operational
    DEGRADED = "degraded"     # Partial functionality
    UNHEALTHY = "unhealthy"   # Not operational
    UNKNOWN = "unknown"       # Cannot determine
```

### Component Health

```python
@dataclass
class ComponentHealth:
    name: str                    # Component name
    status: HealthStatus         # Current status
    message: str                 # Status description
    latency_ms: Optional[float]  # Check latency
    metadata: Dict[str, Any]     # Additional info
```

### Built-in Checks

| Component | Check | Timeout |
|-----------|-------|---------|
| `qdrant` | Vector store ping | 5s |
| `neo4j` | Graph store connectivity | 5s |
| `embedding` | Embedding service | 5s |
| `memory` | Memory service init | 5s |

### Custom Health Check

```python
from ww.observability import HealthChecker, ComponentHealth, HealthStatus

checker = get_health_checker()

@checker.register("custom_service")
async def check_custom() -> ComponentHealth:
    try:
        await custom_service.ping()
        return ComponentHealth(
            name="custom_service",
            status=HealthStatus.HEALTHY,
            message="Service responding"
        )
    except Exception as e:
        return ComponentHealth(
            name="custom_service",
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )
```

### System Health Response

```python
@dataclass
class SystemHealth:
    status: HealthStatus              # Overall status
    components: List[ComponentHealth] # Individual checks
    timestamp: datetime               # Check time
    uptime_seconds: float             # System uptime
```

---

## Distributed Tracing

**File**: `observability/tracing.py`

### OpenTelemetry Integration

```python
from ww.observability import init_tracing, shutdown_tracing

# Initialize at startup
init_tracing(
    service_name="world-weaver",
    otlp_endpoint="http://localhost:4317",  # Optional OTLP collector
    console_export=True                      # Also print to console
)

# Shutdown on exit
shutdown_tracing()
```

### Traced Operations

```python
from ww.observability import traced, traced_sync, add_span_attribute

# Async decorator
@traced("create_episode")
async def create_episode(content: str):
    add_span_attribute("content_length", len(content))
    ...

# Sync decorator
@traced_sync("compute_embedding")
def compute_embedding(text: str):
    ...
```

### Manual Spans

```python
from ww.observability import trace_span, add_span_event

async with trace_span("complex_operation") as span:
    # Phase 1
    add_span_event("phase_1_complete", {"items": 10})

    # Phase 2
    result = await phase_2()
    add_span_attribute("result_count", len(result))
```

### Trace Context Propagation

```python
from ww.observability import get_trace_context

# Get current trace context for propagation
context = get_trace_context()
# Pass to downstream service
headers = {"traceparent": context["traceparent"]}
```

---

## Prometheus Integration

**File**: `observability/prometheus.py`

### Availability Check

```python
from ww.observability import PROMETHEUS_AVAILABLE

if PROMETHEUS_AVAILABLE:
    # Use native prometheus_client
    from prometheus_client import Counter
else:
    # Use internal fallback
    from ww.observability import InternalCounter as Counter
```

### WWMetrics

Predefined metrics for World Weaver:

```python
from ww.observability import WWMetrics

metrics = WWMetrics()

# Memory operations
metrics.memory_ops.labels(operation="create", memory_type="episodic").inc()

# Embedding latency
metrics.embedding_latency.observe(0.045)

# Neuromodulator states
metrics.neuromod_dopamine.set(0.72)
metrics.neuromod_acetylcholine.set(0.45)

# Export for scraping
text_output = metrics.export()
```

### Tracked Metrics

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `ww_memory_ops_total` | Counter | operation, type | Memory operations |
| `ww_embedding_latency_seconds` | Histogram | - | Embedding generation |
| `ww_storage_latency_seconds` | Histogram | backend | Storage operations |
| `ww_circuit_breaker_state` | Gauge | backend | Circuit breaker status |
| `ww_neuromod_dopamine` | Gauge | - | Dopamine level |
| `ww_neuromod_norepinephrine` | Gauge | - | Norepinephrine level |
| `ww_neuromod_acetylcholine` | Gauge | - | Acetylcholine mode |
| `ww_neuromod_serotonin` | Gauge | - | Serotonin level |

### Decorators

```python
from ww.observability import track_latency, count_calls

@track_latency("qdrant_search")
async def search_vectors(query):
    ...

@count_calls("cache_operations", labels={"type": "hit"})
def cache_hit(key):
    ...
```

---

## WebSocket Events

**File**: `api/websocket.py`

### Channels

| Channel | Endpoint | Purpose |
|---------|----------|---------|
| Events | `/ws/events` | All system events |
| Memory | `/ws/memory` | Memory operations |
| Learning | `/ws/learning` | Learning updates |
| Health | `/ws/health` | Health metrics |

### Event Types

```python
class EventType(Enum):
    # System
    START = "system.start"
    SHUTDOWN = "system.shutdown"
    CHECKPOINT_CREATED = "system.checkpoint"
    WAL_ROTATED = "system.wal_rotated"

    # Memory
    MEMORY_ADDED = "memory.added"
    MEMORY_PROMOTED = "memory.promoted"
    MEMORY_REMOVED = "memory.removed"
    MEMORY_CONSOLIDATED = "memory.consolidated"

    # Learning
    GATE_UPDATED = "learning.gate_updated"
    SCORER_UPDATED = "learning.scorer_updated"
    TRACE_CREATED = "learning.trace_created"
    TRACE_DECAYED = "learning.trace_decayed"

    # Neuromodulators
    DOPAMINE_RPE = "neuromod.dopamine_rpe"
    SEROTONIN_MOOD = "neuromod.serotonin_mood"
    NOREPINEPHRINE_AROUSAL = "neuromod.ne_arousal"

    # Health
    HEALTH_UPDATE = "health.update"
    HEALTH_WARNING = "health.warning"
    HEALTH_ERROR = "health.error"
```

### Subscribing (Client)

```javascript
// JavaScript client
const ws = new WebSocket('ws://localhost:8765/ws/health');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Health update:', data);
};
```

### Broadcasting (Server)

```python
from ww.api.websocket import broadcast_event

# Broadcast health update
await broadcast_event(
    channel="health",
    event_type="health.update",
    data={
        "status": "healthy",
        "components": {...}
    }
)
```

---

## Integration Patterns

### API Server Integration

```python
# api/server.py
from fastapi import FastAPI
from ww.observability import (
    configure_logging,
    init_tracing,
    get_health_checker
)

app = FastAPI()

@app.on_event("startup")
async def startup():
    configure_logging(level="INFO", format="json")
    init_tracing(service_name="world-weaver")

@app.get("/health")
async def health():
    checker = get_health_checker()
    return await checker.check_all()
```

### MCP Tool Integration

```python
# mcp/tools/episodic.py
from ww.observability import traced, get_metrics

@traced("mcp.create_episode")
async def create_episode(content: str):
    metrics = get_metrics()

    with Timer("episode_creation") as t:
        episode = await memory.create(content)

    metrics.record_operation("create_episode", t.duration_ms)
    return episode
```

### Prometheus Endpoint

```python
# api/routes/metrics.py
from fastapi import APIRouter
from ww.observability import WWMetrics

router = APIRouter()

@router.get("/metrics")
async def prometheus_metrics():
    metrics = WWMetrics()
    return Response(
        content=metrics.export(),
        media_type="text/plain"
    )
```

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `logging.py` | ~500 | Structured logging, context |
| `metrics.py` | ~600 | Thread-safe metrics |
| `health.py` | ~400 | Component health checks |
| `tracing.py` | ~500 | OpenTelemetry tracing |
| `prometheus.py` | ~700 | Prometheus integration |
| `__init__.py` | 95 | Public API (33 exports) |

---

## See Also

- [System Walkthrough](SYSTEM_WALKTHROUGH.md) - Overall architecture
- [API Walkthrough](API_WALKTHROUGH.md) - REST/MCP endpoints
- [Persistence Architecture](PERSISTENCE_ARCHITECTURE.md) - WAL/checkpoint monitoring
