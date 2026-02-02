# Observability
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/observability/`

## What
Structured logging, metrics collection, health checks, distributed tracing, and Prometheus integration for the T4DM memory system.

## How
- **Logging** (`logging.py`): Structured logging with `OperationLogger`, context-aware fields via `set_context`/`clear_context`, and `log_operation` decorator.
- **Metrics** (`metrics.py`): `MetricsCollector` with `Timer`/`AsyncTimer` for latency tracking, `timed_operation` and `count_operation` decorators.
- **Health** (`health.py`): `HealthChecker` aggregates per-component health into `SystemHealth` with status enum (healthy/degraded/unhealthy).
- **Tracing** (`tracing.py`): OpenTelemetry-based distributed tracing with `traced`/`traced_sync` decorators, span attributes, and context propagation.
- **Prometheus** (`prometheus.py`): Optional Prometheus export with `WWMetrics` singleton, internal counter/gauge/histogram types, `track_latency` and `count_calls` decorators.

## Why
Production memory systems need observability to diagnose latency issues, track memory operation throughput, detect backend failures (Neo4j/Qdrant), and correlate operations across distributed traces.

## Key Files
| File | Purpose |
|------|---------|
| `logging.py` | Structured logging with context propagation |
| `metrics.py` | Internal metrics collection and timing |
| `health.py` | Component and system health checks |
| `tracing.py` | OpenTelemetry distributed tracing |
| `prometheus.py` | Prometheus metrics export (optional) |

## Data Flow
```
Memory Operation --> traced decorator --> span created
                 --> timed_operation --> latency recorded
                 --> OperationLogger --> structured log emitted
                 --> Prometheus counter/histogram updated
```

## Integration Points
- **All modules**: Import `get_logger`, `traced`, `timed_operation` for instrumentation
- **API layer**: Health endpoint exposes `HealthChecker` results
- **Persistence**: Tracks WAL/checkpoint latencies
- **Storage**: Circuit breaker state feeds health checks
