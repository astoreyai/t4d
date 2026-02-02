# Performance Tuning Guide

Optimize World Weaver for your workload.

## Quick Wins

### 1. Enable Embedding Cache

The embedding cache reduces redundant model calls:

```yaml
# t4dm.yaml
embedding_cache_size: 10000  # Increase from default 1000
embedding_cache_ttl: 3600    # 1 hour TTL
```

### 2. Use Batch Operations

```python
from ww import memory

# Instead of individual stores
for content in contents:
    await memory.store(content)  # Slow

# Use batch (when available)
await memory.store_batch(contents)  # Fast
```

### 3. Enable Connection Pooling

```yaml
# t4dm.yaml
neo4j_pool_size: 50     # Default is 50
qdrant_pool_size: 10    # Default is 10
```

## Memory Operations

### Store Performance

| Factor | Impact | Optimization |
|--------|--------|--------------|
| Embedding generation | ~100ms | Cache, batch |
| Vector upsert | ~10ms | Batch operations |
| Graph creation | ~5ms | Batch relationships |
| Hook execution | Variable | Minimize hooks |

### Recall Performance

| Factor | Impact | Optimization |
|--------|--------|--------------|
| Query embedding | ~100ms | Cache queries |
| Vector search | ~10-50ms | Use hierarchical |
| Score fusion | ~5ms | Reduce components |
| Reranking | ~10ms | Limit candidates |

## Hierarchical Search

Enable cluster-based search for large collections:

```python
from ww.memory.episodic import EpisodicMemory

memory = EpisodicMemory(
    hierarchical_search=True,
    cluster_selection_k=3  # Top 3 clusters
)
```

Performance impact:
- **Without**: O(n) scan
- **With**: O(K + k·n/K) where K = clusters

## Embedding Optimization

### GPU Acceleration

```yaml
# t4dm.yaml
embedding_device: cuda      # Use GPU
embedding_batch_size: 64    # Larger batches
embedding_fp16: true        # Half precision
```

### CPU Fallback

```yaml
# t4dm.yaml
embedding_device: cpu
embedding_batch_size: 8     # Smaller batches
embedding_threads: 4        # Parallel threads
```

## Storage Optimization

### Qdrant Settings

```yaml
# t4dm.yaml
qdrant_timeout: 30          # Connection timeout
qdrant_grpc: true           # Use gRPC (faster)
qdrant_prefer_grpc: true
```

### Neo4j Settings

```yaml
# t4dm.yaml
neo4j_pool_size: 100        # More connections
neo4j_max_transaction_retry: 3
neo4j_acquisition_timeout: 30
```

## Consolidation Tuning

### HDBSCAN Parameters

```yaml
# t4dm.yaml
consolidation_min_cluster_size: 5
consolidation_min_samples: 3
consolidation_max_samples: 2000  # Cap for O(n²) mitigation
```

### Scheduling

```yaml
# t4dm.yaml
consolidation_interval: 3600    # Every hour
consolidation_batch_size: 500   # Episodes per batch
```

## Memory Usage

### Embedding Cache

```python
# Monitor cache hit rate
from ww.embedding import get_embedding_service

service = get_embedding_service()
stats = service.cache_stats()
print(f"Hit rate: {stats['hits'] / stats['total']:.2%}")
```

### Buffer Sizes

| Buffer | Default | Recommended Range |
|--------|---------|-------------------|
| Working Memory | 4 | 4-7 (Cowan's limit) |
| CA1 Buffer | 50 | 25-100 |
| Episode Buffer | 100 | 50-200 |
| Error History | 500 | 100-1000 |

## Monitoring

### Key Metrics

```python
from ww.observability import get_metrics

metrics = get_metrics()

# Query latency percentiles
print(f"p50: {metrics.recall_latency_p50}ms")
print(f"p95: {metrics.recall_latency_p95}ms")
print(f"p99: {metrics.recall_latency_p99}ms")

# Cache performance
print(f"Cache hit rate: {metrics.cache_hit_rate:.2%}")

# Storage health
print(f"Neo4j pool utilization: {metrics.neo4j_pool_util:.2%}")
print(f"Qdrant queue depth: {metrics.qdrant_queue_depth}")
```

### Alerts

Set up alerts for:

| Metric | Warning | Critical |
|--------|---------|----------|
| Recall p95 | >200ms | >500ms |
| Cache hit rate | <70% | <50% |
| Pool utilization | >80% | >95% |
| Error rate | >1% | >5% |

## Benchmarking

Run the built-in benchmarks:

```bash
# Performance benchmarks
pytest tests/performance/ -v

# Specific benchmarks
pytest tests/performance/test_benchmarks.py -v -k "recall"
```

### Custom Benchmarks

```python
import time
from ww import memory

async def benchmark_recall(n_queries: int = 100):
    times = []
    for i in range(n_queries):
        start = time.perf_counter()
        await memory.recall(f"query {i}")
        times.append(time.perf_counter() - start)

    print(f"Mean: {sum(times)/len(times)*1000:.2f}ms")
    print(f"p95: {sorted(times)[int(n_queries*0.95)]*1000:.2f}ms")
```

## Scaling Guidelines

### Small (< 10K memories)

- Default settings work well
- Single Qdrant/Neo4j instance
- In-memory embedding cache

### Medium (10K - 100K memories)

- Enable hierarchical search
- Increase cache sizes
- Consider GPU for embeddings
- Monitor consolidation time

### Large (100K - 1M memories)

- Distributed Qdrant cluster
- Neo4j cluster or causal cluster
- Dedicated embedding service
- Background consolidation
- Consider sharding by session

### Very Large (> 1M memories)

- Multiple Qdrant shards
- Neo4j Enterprise cluster
- Embedding service with load balancing
- Async consolidation workers
- Consider hierarchical memory tiers
