# Neo4j Connection Pooling

## Overview

The Neo4j driver has been configured with connection pooling for optimal performance under load. Connection pooling reuses database connections across multiple operations, reducing overhead and improving throughput.

## Configuration

Connection pool settings are managed through environment variables or the Settings class:

```python
from t4dm.core.config import Settings

settings = Settings(
    neo4j_pool_size=50,              # Max connections in pool (default: 50)
    neo4j_connection_timeout=30.0,    # Connection timeout in seconds (default: 30.0)
    neo4j_connection_lifetime=3600.0, # Max connection lifetime in seconds (default: 3600.0)
)
```

### Environment Variables

You can also configure via environment variables:

```bash
export T4DM_NEO4J_POOL_SIZE=100
export T4DM_NEO4J_CONNECTION_TIMEOUT=60.0
export T4DM_NEO4J_CONNECTION_LIFETIME=7200.0
```

## Driver Configuration

The driver is initialized with the following parameters:

- `max_connection_pool_size`: Maximum number of connections in the pool
- `connection_acquisition_timeout`: Time to wait for an available connection
- `max_connection_lifetime`: Maximum time a connection can be reused before being closed
- `connection_timeout`: Time to wait when establishing a new connection
- `keep_alive`: Enables TCP keep-alive on connections

## Health Checks

Monitor the connection pool health:

```python
from t4dm.storage.t4dx_graph_adapter import T4DXGraphAdapter

store = T4DXGraphAdapter()
await store.initialize()

# Check database connectivity and pool health
health = await store.health_check()
print(health)
# {
#     'status': 'healthy',
#     'database': 'neo4j',
#     'uri': 'bolt://localhost:7687',
#     'pool_metrics': {
#         'connections_in_use': 5,
#         'pool_size': 10,
#         'max_pool_size': 50
#     }
# }
```

## Pool Metrics

Get detailed connection pool metrics:

```python
metrics = await store.get_pool_metrics()
print(metrics)
# {
#     'driver_initialized': True,
#     'uri': 'bolt://localhost:7687',
#     'database': 'neo4j',
#     'connections_in_use': 15,
#     'current_pool_size': 20,
#     'max_pool_size': 50,
#     'configured_max_pool_size': 50,
#     'connection_timeout': 30.0,
#     'connection_lifetime': 3600.0
# }
```

## Best Practices

### Pool Size

- **Default (50)**: Good for most applications
- **High concurrency (100+)**: If you have many concurrent operations
- **Low concurrency (10-20)**: For single-user or low-traffic scenarios

### Connection Timeout

- **Default (30s)**: Suitable for most network conditions
- **Fast networks (10-15s)**: Local or high-speed connections
- **Slow networks (60s+)**: Remote or unreliable connections

### Connection Lifetime

- **Default (3600s / 1 hour)**: Balances connection reuse and resource management
- **Long-lived (7200s+)**: When connection overhead is high
- **Short-lived (1800s)**: To ensure fresh connections more frequently

## Troubleshooting

### Connection Pool Exhaustion

If you see timeouts acquiring connections:

```python
# Increase pool size
settings = Settings(neo4j_pool_size=100)

# Or increase acquisition timeout
settings = Settings(neo4j_connection_timeout=60.0)
```

### Stale Connections

If you encounter connection errors:

```python
# Reduce connection lifetime to force fresh connections
settings = Settings(neo4j_connection_lifetime=1800.0)  # 30 minutes
```

### Monitoring

Regularly check pool metrics in production:

```python
import asyncio

async def monitor_pool():
    store = T4DXGraphAdapter()
    while True:
        metrics = await store.get_pool_metrics()

        if metrics.get('connections_in_use', 0) > 0.8 * metrics.get('max_pool_size', 50):
            print("WARNING: Pool utilization > 80%")

        await asyncio.sleep(60)  # Check every minute
```

## Performance Impact

Connection pooling provides significant performance improvements:

- **Without pooling**: ~10-50ms overhead per connection
- **With pooling**: ~1-5ms overhead per operation
- **Throughput**: 2-5x improvement for high-concurrency workloads

## Example: High-Load Scenario

```python
import asyncio
from t4dm.storage.t4dx_graph_adapter import T4DXGraphAdapter

async def high_load_example():
    # Configure for high load
    store = T4DXGraphAdapter()
    await store.initialize()

    # Simulate concurrent operations
    tasks = []
    for i in range(100):
        task = store.create_node(
            label="Episode",
            properties={
                "id": f"ep-{i}",
                "content": f"Event {i}",
                "sessionId": "test",
            }
        )
        tasks.append(task)

    # Execute concurrently - pool handles connection reuse
    results = await asyncio.gather(*tasks)

    # Check pool utilization
    metrics = await store.get_pool_metrics()
    print(f"Used {metrics.get('connections_in_use', 0)} connections")

    await store.close()

# Run
asyncio.run(high_load_example())
```

## Integration Tests

Run the connection pool tests:

```bash
pytest tests/unit/test_neo4j_connection_pool.py -v
```

## References

- [Neo4j Python Driver Documentation](https://neo4j.com/docs/api/python-driver/current/)
- [Connection Pooling Best Practices](https://neo4j.com/docs/operations-manual/current/performance/connection-pooling/)
