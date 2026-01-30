# Monitoring Guide

Observability and monitoring for World Weaver.

## Overview

```mermaid
graph TB
    subgraph WW["World Weaver"]
        API[API Server]
        OTEL[OpenTelemetry]
    end

    subgraph Collectors["Collectors"]
        PROM[Prometheus]
        JAEGER[Jaeger]
        LOKI[Loki]
    end

    subgraph Visualization["Visualization"]
        GRAFANA[Grafana]
    end

    API --> OTEL
    OTEL --> PROM
    OTEL --> JAEGER
    API --> LOKI
    PROM --> GRAFANA
    JAEGER --> GRAFANA
    LOKI --> GRAFANA
```

## Metrics

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `ww_episodes_total` | Counter | Total episodes created |
| `ww_recall_duration_seconds` | Histogram | Recall operation latency |
| `ww_embedding_cache_hits` | Counter | Embedding cache hits |
| `ww_embedding_cache_misses` | Counter | Embedding cache misses |
| `ww_neo4j_pool_active` | Gauge | Active Neo4j connections |
| `ww_qdrant_queue_depth` | Gauge | Qdrant operation queue |

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'world-weaver'
    static_configs:
      - targets: ['ww-api:8765']
    metrics_path: '/metrics'
```

### Key Dashboards

#### Memory Operations

```
# Recall latency percentiles
histogram_quantile(0.95, rate(ww_recall_duration_seconds_bucket[5m]))

# Operations per second
rate(ww_episodes_total[5m])

# Cache hit rate
rate(ww_embedding_cache_hits[5m]) /
(rate(ww_embedding_cache_hits[5m]) + rate(ww_embedding_cache_misses[5m]))
```

#### Storage Health

```
# Neo4j pool utilization
ww_neo4j_pool_active / ww_neo4j_pool_size

# Qdrant queue depth
ww_qdrant_queue_depth
```

## Tracing

### OpenTelemetry Setup

```python
from ww.observability import WWObserver

observer = WWObserver(
    service_name="world-weaver",
    jaeger_endpoint="http://jaeger:14268/api/traces"
)
```

### Trace Structure

```mermaid
gantt
    title Recall Operation Trace
    dateFormat  X
    axisFormat %L

    section API
    HTTP Request :a1, 0, 5
    Parse Request :a2, 5, 2
    Serialize Response :a3, 95, 5

    section Memory
    Generate Embedding :b1, 7, 30
    Pattern Completion :b2, 37, 10

    section Storage
    Qdrant Search :c1, 47, 20
    Neo4j Query :c2, 67, 15

    section Learning
    Hebbian Update :d1, 82, 10
```

### Trace Attributes

| Attribute | Description |
|-----------|-------------|
| `ww.session_id` | Session identifier |
| `ww.operation` | Operation type |
| `ww.memory_type` | Memory subsystem |
| `ww.result_count` | Number of results |

## Logging

### Structured Logging

```python
import structlog

logger = structlog.get_logger()
logger.info(
    "episode_created",
    episode_id=str(episode.id),
    session_id=session_id,
    content_length=len(content)
)
```

### Log Levels

| Level | Use Case |
|-------|----------|
| ERROR | Failures, exceptions |
| WARNING | Degraded operation |
| INFO | Normal operations |
| DEBUG | Detailed tracing |

### Loki Configuration

```yaml
# promtail.yml
server:
  http_listen_port: 9080

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: ww
    static_configs:
      - targets:
          - localhost
        labels:
          job: world-weaver
          __path__: /var/log/ww/*.log
```

## Alerting

### Critical Alerts

```yaml
# prometheus-rules.yml
groups:
  - name: world-weaver
    rules:
      - alert: WWHighLatency
        expr: histogram_quantile(0.95, rate(ww_recall_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High recall latency"

      - alert: WWErrorRate
        expr: rate(ww_errors_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate"

      - alert: WWNeo4jPoolExhausted
        expr: ww_neo4j_pool_active / ww_neo4j_pool_size > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Neo4j pool near exhaustion"
```

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Recall p95 latency | > 200ms | > 500ms |
| Error rate | > 1% | > 5% |
| Cache hit rate | < 70% | < 50% |
| Pool utilization | > 80% | > 95% |

## Health Checks

### Endpoints

```bash
# Overall health
curl http://localhost:8765/api/v1/health

# Detailed status
curl http://localhost:8765/api/v1/stats
```

### Response Format

```json
{
  "status": "healthy",
  "timestamp": "2026-01-03T12:00:00Z",
  "version": "0.4.0",
  "components": {
    "neo4j": "connected",
    "qdrant": "connected",
    "embedding": "ready"
  }
}
```

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /api/v1/health
    port: 8765
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /api/v1/health
    port: 8765
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

## Grafana Dashboards

### Import Dashboards

```bash
# Download dashboard JSON
curl -O https://raw.githubusercontent.com/astoreyai/ww/master/dashboards/ww-overview.json

# Import via API
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @ww-overview.json
```

### Dashboard Panels

1. **Overview**
   - Request rate
   - Error rate
   - Latency percentiles

2. **Memory Operations**
   - Store/Recall counts
   - Memory type distribution
   - Cache performance

3. **Storage**
   - Neo4j metrics
   - Qdrant metrics
   - Connection pool status

4. **Learning**
   - Dopamine signals
   - Hebbian updates
   - Consolidation events

## Detailed Observability Architecture

### Complete Telemetry Pipeline

```mermaid
graph TB
    subgraph Application["World Weaver Application"]
        API[API Server]
        MEM[Memory Operations]
        LEARN[Learning System]
        STORE[Storage Layer]
    end

    subgraph Instrumentation["Instrumentation Layer"]
        OTEL_SDK[OpenTelemetry SDK]
        METRICS[Metrics Exporter]
        TRACES[Trace Exporter]
        LOGS[Log Processor]
    end

    subgraph Collection["Collection Layer"]
        OTEL_COL[OTEL Collector]
        PROM[Prometheus]
        JAEGER[Jaeger]
        LOKI[Loki]
    end

    subgraph Storage["Telemetry Storage"]
        PROM_DB[(Prometheus TSDB)]
        JAEGER_DB[(Jaeger Storage)]
        LOKI_DB[(Loki Storage)]
    end

    subgraph Visualization["Visualization & Alerting"]
        GRAFANA[Grafana]
        ALERT[Alertmanager]
        PAGER[PagerDuty/Slack]
    end

    API --> OTEL_SDK
    MEM --> OTEL_SDK
    LEARN --> OTEL_SDK
    STORE --> OTEL_SDK

    OTEL_SDK --> METRICS
    OTEL_SDK --> TRACES
    OTEL_SDK --> LOGS

    METRICS --> OTEL_COL
    TRACES --> OTEL_COL
    LOGS --> OTEL_COL

    OTEL_COL --> PROM
    OTEL_COL --> JAEGER
    OTEL_COL --> LOKI

    PROM --> PROM_DB
    JAEGER --> JAEGER_DB
    LOKI --> LOKI_DB

    PROM_DB --> GRAFANA
    JAEGER_DB --> GRAFANA
    LOKI_DB --> GRAFANA

    PROM --> ALERT
    ALERT --> PAGER

    style OTEL_SDK fill:#e8f5e9
    style GRAFANA fill:#e3f2fd
    style ALERT fill:#ffebee
```

### Trace Span Hierarchy

```mermaid
graph TB
    subgraph RootSpan["Root: HTTP Request"]
        HTTP[ww.http.request<br/>method, path, status]
    end

    subgraph Level1["Level 1: Operations"]
        STORE_OP[ww.store<br/>content_length, importance]
        RECALL_OP[ww.recall<br/>query, limit]
        CONSOL_OP[ww.consolidate<br/>phase]
    end

    subgraph Level2["Level 2: Subsystems"]
        EMBED[ww.embedding<br/>cache_hit, latency]
        GATE[ww.gate<br/>decision, evidence]
        PATTERN[ww.pattern_completion<br/>iterations]
        NCA[ww.nca<br/>nt_levels]
    end

    subgraph Level3["Level 3: Storage"]
        NEO4J_SPAN[ww.neo4j<br/>query, rows]
        QDRANT_SPAN[ww.qdrant<br/>collection, matches]
        SAGA_SPAN[ww.saga<br/>state, duration]
    end

    subgraph Level4["Level 4: Learning"]
        DA_SPAN[ww.dopamine<br/>rpe, signal]
        HEBB_SPAN[ww.hebbian<br/>updates, strength]
        STDP_SPAN[ww.stdp<br/>potentiation, depression]
    end

    HTTP --> STORE_OP
    HTTP --> RECALL_OP
    HTTP --> CONSOL_OP

    STORE_OP --> EMBED
    STORE_OP --> GATE
    RECALL_OP --> EMBED
    RECALL_OP --> PATTERN
    RECALL_OP --> NCA

    EMBED --> QDRANT_SPAN
    GATE --> NEO4J_SPAN
    PATTERN --> QDRANT_SPAN

    STORE_OP --> SAGA_SPAN
    SAGA_SPAN --> NEO4J_SPAN
    SAGA_SPAN --> QDRANT_SPAN

    STORE_OP --> DA_SPAN
    RECALL_OP --> HEBB_SPAN
    DA_SPAN --> STDP_SPAN

    style HTTP fill:#e8f5e9
    style SAGA_SPAN fill:#fff3e0
```

### Detailed Store Trace

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Embed
    participant Gate
    participant Saga
    participant Neo4j
    participant Qdrant
    participant Learn

    Note over Client,Learn: Trace: ww.store (trace_id: abc123)

    rect rgb(232, 245, 233)
        Note over API: Span: ww.http.request
        Client->>API: POST /api/v1/store
        API->>API: Parse & validate
    end

    rect rgb(227, 242, 253)
        Note over Embed: Span: ww.embedding
        API->>Embed: generate_embedding()
        Note right of Embed: cache_hit: false<br/>latency_ms: 12
        Embed-->>API: embedding[1024]
    end

    rect rgb(255, 243, 224)
        Note over Gate: Span: ww.gate
        API->>Gate: evaluate()
        Note right of Gate: decision: ACCEPT<br/>evidence: 0.85
        Gate-->>API: accept
    end

    rect rgb(232, 245, 233)
        Note over Saga: Span: ww.saga
        API->>Saga: begin()

        rect rgb(227, 242, 253)
            Note over Qdrant: Span: ww.qdrant
            Saga->>Qdrant: upsert()
            Note right of Qdrant: collection: episodes<br/>point_id: xyz789
            Qdrant-->>Saga: ok
        end

        rect rgb(255, 243, 224)
            Note over Neo4j: Span: ww.neo4j
            Saga->>Neo4j: create_node()
            Note right of Neo4j: labels: Episode<br/>properties: 5
            Neo4j-->>Saga: ok
        end

        Saga-->>API: committed
    end

    rect rgb(255, 235, 238)
        Note over Learn: Span: ww.dopamine
        API->>Learn: signal()
        Note right of Learn: rpe: 0.3<br/>update: true
    end

    API-->>Client: 201 Created
```

### Metric Collection Points

```mermaid
graph TB
    subgraph API["API Layer Metrics"]
        M1[ww_http_requests_total<br/>method, path, status]
        M2[ww_http_request_duration<br/>method, path]
        M3[ww_active_connections<br/>gauge]
    end

    subgraph Memory["Memory Metrics"]
        M4[ww_episodes_total<br/>session, type]
        M5[ww_recall_duration<br/>session, type]
        M6[ww_buffer_size<br/>gauge]
        M7[ww_gate_decisions<br/>decision]
    end

    subgraph Storage["Storage Metrics"]
        M8[ww_neo4j_queries<br/>type]
        M9[ww_neo4j_latency<br/>type]
        M10[ww_qdrant_operations<br/>type]
        M11[ww_saga_transactions<br/>outcome]
    end

    subgraph Cache["Cache Metrics"]
        M12[ww_embedding_cache_hits]
        M13[ww_embedding_cache_misses]
        M14[ww_cache_size<br/>gauge]
    end

    subgraph Learning["Learning Metrics"]
        M15[ww_dopamine_signals<br/>type]
        M16[ww_hebbian_updates]
        M17[ww_consolidation_events<br/>phase]
    end

    subgraph NCA["NCA Metrics"]
        M18[ww_nca_nt_levels<br/>neurotransmitter]
        M19[ww_nca_theta_phase]
        M20[ww_nca_spatial_cells_active]
    end

    style API fill:#e8f5e9
    style Memory fill:#e3f2fd
    style Storage fill:#fff3e0
    style Learning fill:#ffebee
```

### Log Correlation

```mermaid
sequenceDiagram
    participant App as Application
    participant Log as Structlog
    participant OTEL as OpenTelemetry
    participant Loki as Loki

    App->>OTEL: get_current_span()
    OTEL-->>App: span_context

    App->>Log: logger.info("event", **context)

    rect rgb(232, 245, 233)
        Note over Log: Add trace context
        Log->>Log: trace_id = span.trace_id
        Log->>Log: span_id = span.span_id
    end

    Log->>Loki: {<br/>  "event": "episode_created",<br/>  "trace_id": "abc123",<br/>  "span_id": "def456",<br/>  "episode_id": "ep_789",<br/>  "session_id": "sess_001"<br/>}

    Note over Loki: Queryable by trace_id<br/>Links to Jaeger
```

### Alert Flow

```mermaid
sequenceDiagram
    participant Prom as Prometheus
    participant Rules as Alert Rules
    participant AM as Alertmanager
    participant Route as Routing
    participant Slack
    participant PD as PagerDuty

    Prom->>Rules: evaluate every 15s

    rect rgb(255, 235, 238)
        Note over Rules: Rule: WWHighLatency
        Rules->>Rules: p95 > 500ms for 5m
        Rules->>AM: FIRING
    end

    AM->>AM: Deduplicate
    AM->>AM: Group by severity

    AM->>Route: route(alert)

    alt severity == critical
        Route->>PD: page on-call
        Route->>Slack: #ww-alerts-critical
    else severity == warning
        Route->>Slack: #ww-alerts
    end

    Note over AM: After 5m resolved
    AM->>Route: RESOLVED
    Route->>Slack: Recovery notification
```

### Dashboard Layout

```mermaid
graph TB
    subgraph Row1["Row 1: Overview"]
        P1[Request Rate<br/>Graph]
        P2[Error Rate<br/>Graph]
        P3[P95 Latency<br/>Graph]
        P4[Active Sessions<br/>Stat]
    end

    subgraph Row2["Row 2: Memory Operations"]
        P5[Store/Recall<br/>Stacked Graph]
        P6[Memory Types<br/>Pie Chart]
        P7[Buffer Size<br/>Gauge]
        P8[Gate Decisions<br/>Bar Chart]
    end

    subgraph Row3["Row 3: Storage Health"]
        P9[Neo4j Queries<br/>Graph]
        P10[Qdrant Ops<br/>Graph]
        P11[Connection Pool<br/>Gauge]
        P12[Saga Success<br/>Percentage]
    end

    subgraph Row4["Row 4: Cache & Learning"]
        P13[Cache Hit Rate<br/>Graph]
        P14[Dopamine RPE<br/>Graph]
        P15[Hebbian Updates<br/>Counter]
        P16[Consolidation<br/>Events]
    end

    subgraph Row5["Row 5: NCA Dynamics"]
        P17[NT Levels<br/>Multi-line]
        P18[Theta Phase<br/>Graph]
        P19[Spatial Cells<br/>Heatmap]
        P20[Coupling<br/>Strength]
    end

    style Row1 fill:#e8f5e9
    style Row2 fill:#e3f2fd
    style Row3 fill:#fff3e0
    style Row4 fill:#ffebee
    style Row5 fill:#f3e5f5
```
