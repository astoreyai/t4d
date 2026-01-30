# OpenTelemetry Distributed Tracing

World Weaver now includes OpenTelemetry distributed tracing for request flow visibility across MCP gateway → memory services → storage layers.

## Features

- **W3C Trace Context Propagation**: Automatic trace context propagation through async call chains
- **Decorator-Based Instrumentation**: Simple `@traced` decorators for functions
- **Span Attributes & Events**: Enrich spans with custom metadata
- **OTLP Export**: Export to Jaeger, Tempo, or any OTLP-compatible backend
- **Console Debug Mode**: Local debugging with console span export

## Configuration

Tracing is configured via environment variables in `.env` or configuration settings:

```bash
# Enable tracing
WW_OTEL_ENABLED=true

# OTLP endpoint (Jaeger, Tempo, etc.)
WW_OTEL_ENDPOINT=http://localhost:4317

# Service name
WW_OTEL_SERVICE_NAME=world-weaver

# Console export for debugging
WW_OTEL_CONSOLE=true
```

## Architecture

### Trace Flow

```
MCP Gateway (SERVER span)
  ├─ Episodic Memory (INTERNAL span)
  │   ├─ Qdrant Search (CLIENT span)
  │   └─ Neo4j Query (CLIENT span)
  ├─ Semantic Memory (INTERNAL span)
  │   ├─ Qdrant Search (CLIENT span)
  │   └─ Neo4j Query (CLIENT span)
  └─ Procedural Memory (INTERNAL span)
      ├─ Qdrant Search (CLIENT span)
      └─ Neo4j Query (CLIENT span)
```

### Span Kinds

- **SERVER**: MCP tool entry points (traced RPC calls)
- **CLIENT**: External service calls (Qdrant, Neo4j)
- **INTERNAL**: Internal service methods

## Usage

### Initialization

Tracing is automatically initialized at server startup if `WW_OTEL_ENABLED=true`:

```python
from ww.observability.tracing import init_tracing, shutdown_tracing

# Initialize
init_tracing(
    service_name="world-weaver",
    otlp_endpoint="http://localhost:4317",
    console_export=True  # For debugging
)

# Shutdown (called automatically on exit)
shutdown_tracing()
```

### Decorating Functions

```python
from opentelemetry.trace import SpanKind
from ww.observability.tracing import traced

@traced("my_service.operation", kind=SpanKind.INTERNAL)
async def my_operation(param: str):
    # Function automatically traced
    return result
```

### Manual Span Creation

```python
from ww.observability.tracing import trace_span

async def process_batch():
    async with trace_span("batch_processing", batch_size=100):
        # Code inside span
        pass
```

### Adding Span Metadata

```python
from ww.observability.tracing import add_span_attribute, add_span_event

async def search_vectors():
    add_span_attribute("collection", "episodes")
    add_span_attribute("limit", 10)

    # Perform search

    add_span_event("cache_miss", {"key": "episode_123"})
```

## Instrumentation Points

### MCP Gateway

All 17 MCP tools are instrumented:

- **Episodic**: `create_episode`, `recall_episodes`, `query_at_time`, `mark_important`
- **Semantic**: `create_entity`, `create_relation`, `semantic_recall`, `spread_activation`, `supersede_fact`
- **Procedural**: `create_skill`, `recall_skill`, `execute_skill`, `deprecate_skill`
- **System**: `consolidate_now`, `get_provenance`, `get_session_id`, `memory_stats`

### Memory Services

Core memory operations:

- `EpisodicMemory.create()`, `EpisodicMemory.recall()`
- `SemanticMemory.create_entity()`, `SemanticMemory.recall()`
- `ProceduralMemory.create_skill()`, `ProceduralMemory.recall_skill()`

### Storage Layer

Vector and graph operations:

- `QdrantStore.search()`, `QdrantStore.add()`, `QdrantStore.delete()`
- `Neo4jStore.query()`, `Neo4jStore.create_node()`

## Visualization

### Jaeger Setup

```bash
# Start Jaeger all-in-one
docker run -d \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Configure World Weaver
export WW_OTEL_ENABLED=true
export WW_OTEL_ENDPOINT=http://localhost:4317

# Run server
ww-memory

# View traces at http://localhost:16686
```

### Grafana Tempo Setup

```bash
# Start Tempo
docker run -d \
  -p 3200:3200 \
  -p 4317:4317 \
  grafana/tempo:latest \
  -config.file=/etc/tempo.yaml

# Configure World Weaver
export WW_OTEL_ENABLED=true
export WW_OTEL_ENDPOINT=http://localhost:4317
```

## Example Trace

A `recall_episodes` request generates this trace:

```
mcp.recall_episodes (235ms)
  ├─ episodic.recall (220ms)
  │   ├─ qdrant.search (180ms)
  │   │   collection=ww_episodes
  │   │   limit=10
  │   └─ scoring_computation (35ms)
  └─ response_formatting (5ms)
```

## Debugging

### Console Export

Enable console output for local debugging:

```bash
export WW_OTEL_ENABLED=true
export WW_OTEL_CONSOLE=true
```

Spans will be printed to console with full details.

### Trace Context

Get current trace context for manual propagation:

```python
from ww.observability.tracing import get_trace_context

ctx = get_trace_context()
# {'traceparent': '00-...', 'tracestate': '...'}
```

## Testing

Run tracing tests:

```bash
pytest tests/test_tracing.py -v
```

Coverage includes:
- Initialization and configuration
- Decorator functionality
- Context managers
- Span attributes and events
- Error recording
- Nested spans

## Performance

- **Overhead**: <1ms per span with OTLP export
- **Sampling**: Not implemented (100% sampling by default)
- **Batching**: Spans are batched before export (default: 512 spans)

## Best Practices

1. **Use appropriate span kinds**: SERVER for entry points, CLIENT for external calls, INTERNAL for internal logic
2. **Add meaningful attributes**: Include IDs, counts, filters for debugging
3. **Record errors**: Exceptions are automatically recorded with stack traces
4. **Minimize nested spans**: Keep trace trees shallow for readability
5. **Use semantic naming**: `service.operation` format (e.g., `episodic.create`)

## Troubleshooting

### Traces not appearing

1. Check `WW_OTEL_ENABLED=true` is set
2. Verify OTLP endpoint is reachable
3. Enable console export: `WW_OTEL_CONSOLE=true`
4. Check server logs for initialization messages

### Performance impact

Tracing adds minimal overhead (<1ms/span). If concerned:

1. Reduce instrumentation depth (remove INTERNAL spans)
2. Implement sampling (future enhancement)
3. Use async batch export (default)

### Missing context propagation

Ensure all async functions use `await` and spans are created in the correct async context.

## Future Enhancements

- **Sampling**: Probabilistic sampling for high-volume deployments
- **Metrics**: Span metrics (duration histograms, counts)
- **Exemplars**: Link traces to metrics
- **Baggage**: Cross-service metadata propagation
- **Custom exporters**: Additional backends (Zipkin, X-Ray)
