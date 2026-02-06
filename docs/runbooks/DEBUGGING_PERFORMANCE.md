# T4DM Performance Debugging Runbook

**Location**: `/mnt/projects/t4d/t4dm/docs/runbooks/DEBUGGING_PERFORMANCE.md`

Profile, monitor, and optimize T4DM memory operations, inference speed, and resource usage.

---

## Symptoms

### High Latency (Store Operations > 5 seconds)
- POST `/episodes/store` takes 5+ seconds
- Consolidation runs during store (blocking)
- MemTable flush causes tail latencies

**Likelihood**: Synchronous flush, compaction during write, or large embeddings

### Slow Search (Query > 2 seconds)
- `/episodes/search` exceeds 2 second SLA
- Linearly degrades with data size
- HNSW index not being used

**Likelihood**: Full table scan, stale HNSW, or no kappa filtering

### High Memory Usage (> 16 GB)
- Python heap grows unbounded
- Embedding cache not evicting
- Old consolidation snapshots not released

**Likelihood**: Embedding cache misconfigured, garbage collection issues

### CPU Saturation
- Service uses 100% CPU on single core
- Consolidation blocks inference
- No parallelism between operations

**Likelihood**: Synchronous sleep replay, no threading for compaction

### Poor Inference Throughput (< 10 tokens/sec)
- Qwen model inference slow despite GPU
- Spiking blocks add significant latency
- Batching not reducing latency

**Likelihood**: Spiking block not vectorized, synchronous memory lookups

### Volatile Latency (Tail p99 >> Median)
- Median latency 100ms, p99 > 5 seconds
- Consolidation pauses
- Garbage collection stalls

**Likelihood**: Concurrent consolidation/flush, GC pause during operations

---

## Diagnostic Commands

### System-Level Monitoring

```bash
# Real-time resource usage
top -p $(pgrep -f t4dm-service)

# Memory breakdown
ps aux | grep t4dm-service | grep -v grep | awk '{print $6}'

# Per-core CPU usage
mpstat -P ALL 1 5

# Disk I/O activity
iotop -p $(pgrep -f t4dm-service)

# Network I/O (if distributed)
nethogs

# Cache hit rates
perf stat -p $(pgrep -f t4dm-service) -e cache-references,cache-misses
```

### T4DM Performance Endpoints

```bash
# Get latency metrics
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/observability/latency?window_seconds=60"

# Get operation counts
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/observability/ops-count?session_id=$SESSION_ID"

# Get batch statistics
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/observability/batch-stats?session_id=$SESSION_ID"

# Get embedding cache hit rate
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/observability/cache-stats?session_id=$SESSION_ID"

# Get consolidation timing
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/consolidation/timing?session_id=$SESSION_ID"

# Get query execution plan
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/observability/query-profile" \
  -d "{\"query\": \"...\", \"profile\": true, \"session_id\": \"$SESSION_ID\"}"

# Get Prometheus metrics
curl -H "X-API-Key: $T4DM_API_KEY" \
  http://localhost:8000/metrics
```

### Inference Profiling

```bash
# Profile a single forward pass
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/qwen/trace" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"Hello world\",
    \"profile\": true,
    \"session_id\": \"$SESSION_ID\"
  }"

# Get layer-wise inference time
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/qwen/layer-timing?session_id=$SESSION_ID"

# Get spiking block latency
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/spiking/layer-latency?session_id=$SESSION_ID"

# Profile memory retrieval
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/observability/memory-retrieval-profile" \
  -d "{\"k\": 10, \"session_id\": \"$SESSION_ID\", \"profile\": true}"
```

### Consolidation Profiling

```bash
# Time consolidation phases
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/consolidation/profile-nrem" \
  -d "{\"session_id\": \"$SESSION_ID\"}"

curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/consolidation/profile-rem" \
  -d "{\"session_id\": \"$SESSION_ID\"}"

# Check compaction metrics
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/consolidation/compaction-metrics?session_id=$SESSION_ID"
```

---

## Code-Based Profiling

### Profile Memory Store Operations

```python
import time
from t4dm.sdk.client import T4DMClient
from statistics import mean, stdev

client = T4DMClient()

# Warm up
client.episodes.store(content="warmup", session_id="perf_test")

# Benchmark store latency
latencies = []
for i in range(100):
    start = time.time()
    client.episodes.store(
        content=f"Memory {i}",
        session_id="perf_test",
    )
    latencies.append(time.time() - start)

print("Store Operation Latencies:")
print(f"  Mean: {mean(latencies)*1000:.1f} ms")
print(f"  Median: {sorted(latencies)[50]*1000:.1f} ms")
print(f"  p99: {sorted(latencies)[99]*1000:.1f} ms")
print(f"  Max: {max(latencies)*1000:.1f} ms")
print(f"  Std: {stdev(latencies)*1000:.1f} ms")

# Breakdown by phase
print("\nStore phase timing (sample):")
result = client.episodes.store(
    content="Profile",
    session_id="perf_test",
    profile=True,
)
for phase, duration_ms in result.timing.items():
    print(f"  {phase}: {duration_ms:.1f} ms")
```

### Profile Search Operations

```python
import time
from t4dm.sdk.client import T4DMClient

client = T4DMClient()

# Store test data
for i in range(1000):
    client.episodes.store(
        content=f"Memory {i}",
        session_id="search_test",
    )

# Benchmark search latency
latencies = []
for i in range(100):
    start = time.time()
    results = client.episodes.search(
        query=f"query {i}",
        session_id="search_test",
        k=10,
    )
    latencies.append(time.time() - start)

import statistics
print("Search Operation Latencies:")
print(f"  Mean: {statistics.mean(latencies)*1000:.1f} ms")
print(f"  Median: {sorted(latencies)[50]*1000:.1f} ms")
print(f"  p99: {sorted(latencies)[99]*1000:.1f} ms")

# Profile query components
result, query_plan = client.episodes.search(
    query="test",
    session_id="search_test",
    k=10,
    profile=True,
)

print("\nQuery execution plan:")
print(f"  Segments to search: {query_plan['num_segments']}")
print(f"  Items scanned: {query_plan['items_scanned']}")
print(f"  HNSW candidates: {query_plan['hnsw_candidates']}")
print(f"  Reranking cost: {query_plan['rerank_time_ms']:.1f} ms")
```

### Profile Qwen Inference

```python
import time
import torch
from t4dm.qwen.loader import load_qwen_model

model = load_qwen_model("Qwen/Qwen2.5-3B-Instruct")

# Warm up
input_ids = torch.tensor([[101, 2054, 2003, 102]])
with torch.no_grad():
    _ = model.forward(input_ids)

# Benchmark inference
latencies = []
for i in range(10):
    input_ids = torch.tensor([[101, 2054, 2003, 102]] * (i+1))  # Increase batch

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.forward(input_ids)
    latencies.append(time.perf_counter() - start)

    batch_size = input_ids.shape[0]
    tokens = input_ids.shape[1]
    throughput = (batch_size * tokens) / latencies[-1]
    print(f"Batch {batch_size}x{tokens}: {latencies[-1]*1000:.1f} ms ({throughput:.0f} tokens/sec)")

# Profile layer-wise timing
from torch.utils.benchmark import Timer

timer = Timer(
    "model.forward(input_ids)",
    globals={"model": model, "input_ids": input_ids},
    num_threads=1,
)

# Layer profiling
print("\nLayer-wise timing:")
for layer_idx in range(4):  # Sample 4 layers
    # Hook to measure individual layer time
    layer = model.model.layers[layer_idx]

    def layer_forward_hook(module, input, output):
        return output

    hook = layer.register_forward_hook(layer_forward_hook)

    with torch.no_grad():
        _ = model.forward(input_ids)

    hook.remove()
```

### Profile Spiking Blocks

```python
import time
import torch
from t4dm.spiking.cortical_stack import CorticalStack

stack = CorticalStack(num_blocks=6, hidden_size=256)

# Warm up
x = torch.randn(1, 256)
hidden = stack.init_hidden_state(1)
for _ in range(5):
    x, hidden = stack(x, hidden)

# Benchmark spiking latency
latencies = []
for i in range(100):
    x = torch.randn(1, 256)
    hidden = stack.init_hidden_state(1)

    start = time.perf_counter()
    x, hidden = stack(x, hidden)
    latencies.append(time.perf_counter() - start)

import statistics
print("Spiking Block Latencies:")
print(f"  Mean: {statistics.mean(latencies)*1000:.2f} ms")
print(f"  Median: {sorted(latencies)[50]*1000:.2f} ms")
print(f"  p99: {sorted(latencies)[99]*1000:.2f} ms")

# Per-block breakdown
print("\nPer-block timing:")
for block_idx, block in enumerate(stack.blocks):
    start = time.perf_counter()
    for _ in range(100):
        _, hidden = block(x, hidden)
    elapsed = (time.perf_counter() - start) / 100
    print(f"  Block {block_idx}: {elapsed*1000:.2f} ms")
```

### Monitor Memory Growth

```python
import tracemalloc
from t4dm.sdk.client import T4DMClient

tracemalloc.start()

client = T4DMClient()

# Baseline
snapshot1 = tracemalloc.take_snapshot()

# Store 1000 memories
for i in range(1000):
    client.episodes.store(
        content=f"Memory {i}",
        session_id="memory_test",
    )

snapshot2 = tracemalloc.take_snapshot()

# Analyze growth
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
print("Memory growth (top 10):")
for stat in top_stats[:10]:
    print(f"  {stat}")

# Check cache behavior
from t4dm.embedding.cache import EmbeddingCache

cache = EmbeddingCache(max_size=1000)
print(f"\nEmbedding cache stats:")
print(f"  Size: {cache.size()}")
print(f"  Hit rate: {cache.hit_rate():.1%}")
print(f"  Evictions: {cache.eviction_count}")
```

### Profile Consolidation

```python
import time
from t4dm.consolidation.service import get_consolidation_service

consol = get_consolidation_service(session_id="perf_test")

# Profile NREM consolidation
start = time.perf_counter()
result = consol.consolidate_nrem()
nrem_time = time.perf_counter() - start

print(f"NREM consolidation: {nrem_time:.2f} seconds")
print(f"  Items processed: {result['items_processed']}")
print(f"  Throughput: {result['items_processed'] / nrem_time:.0f} items/sec")

# Profile REM consolidation
start = time.perf_counter()
result = consol.consolidate_rem()
rem_time = time.perf_counter() - start

print(f"REM consolidation: {rem_time:.2f} seconds")
print(f"  Items processed: {result['items_processed']}")
print(f"  Prototypes created: {result['prototypes_created']}")
print(f"  Throughput: {result['items_processed'] / rem_time:.0f} items/sec")
```

---

## Common Performance Issues & Solutions

### Issue: Store Operations Take 5+ Seconds

**Symptoms**:
- Consistent 5+ second latency for store
- Correlates with consolidation logs
- Tail latency much higher than median

**Diagnosis**:
```python
import time
from t4dm.sdk.client import T4DMClient

client = T4DMClient()

# Measure store phases
result = client.episodes.store(
    content="Test",
    session_id="perf_test",
    profile=True,
)

for phase, ms in result.timing.items():
    print(f"{phase}: {ms:.1f} ms")
    if ms > 1000:
        print(f"  ^^^ SLOW PHASE")

# Check if consolidation is blocking
print(f"\nConsolidation blocking: {result.consolidation_time_ms > 0}")
```

**Solutions**:

1. **Reduce flush threshold** (flush more gradually):
   ```bash
   export T4DM_FLUSH_THRESHOLD=500  # was 1000
   systemctl restart t4dm-service
   ```

2. **Disable synchronous consolidation**:
   ```bash
   export T4DM_CONSOLIDATION_BLOCKING=false
   # Schedule consolidation separately
   ```

3. **Batch store operations**:
   ```python
   # Instead of storing one-by-one
   memories = [{"content": f"Mem {i}"} for i in range(100)]
   client.episodes.batch_store(memories, session_id="test")
   ```

4. **Use async/background flush**:
   ```python
   # Return immediately, flush in background
   result = client.episodes.store(
       content="Test",
       session_id="test",
       flush_async=True,
   )
   ```

### Issue: Search Queries Slow (> 2 seconds)

**Symptoms**:
- Search latency grows with data size
- HNSW index doesn't help
- Sequential scan happening

**Diagnosis**:
```bash
# Check if HNSW is being used
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/observability/query-profile" \
  -d "{
    \"query\": \"test\",
    \"profile\": true,
    \"session_id\": \"$SESSION_ID\"
  }"

# Should show HNSW candidates, not full scan
```

**Solutions**:

1. **Rebuild HNSW index**:
   ```bash
   curl -H "X-API-Key: $T4DM_API_KEY" \
     -X POST "http://localhost:8000/storage/rebuild-hnsw" \
     -d "{\"session_id\": \"$SESSION_ID\"}"
   ```

2. **Increase HNSW M parameter** (better connectivity):
   ```bash
   export T4DM_HNSW_M=24  # was 16
   # Rebuild index
   ```

3. **Reduce search k** (search fewer neighbors):
   ```python
   results = client.episodes.search(
       query="test",
       session_id=session_id,
       k=5,  # was 10
   )
   ```

4. **Add kappa filtering** (prune stale memories):
   ```python
   results = client.episodes.search(
       query="test",
       session_id=session_id,
       k=10,
       kappa_min=0.1,  # Only search consolidated memories
   )
   ```

5. **Use time range filtering**:
   ```python
   from datetime import datetime, timedelta

   now = datetime.now()
   week_ago = now - timedelta(days=7)

   results = client.episodes.search(
       query="test",
       session_id=session_id,
       k=10,
       time_min=week_ago,  # Only recent memories
   )
   ```

### Issue: High Memory Usage (> 16 GB)

**Symptoms**:
- Heap grows unbounded during long sessions
- Embedding cache not evicting
- Memory pressure increases

**Diagnosis**:
```python
import tracemalloc
from t4dm.embedding.cache import EmbeddingCache

# Check cache
cache = EmbeddingCache()
print(f"Cache size: {cache.size()}")
print(f"Cache max: {cache.max_size}")
print(f"Utilization: {cache.size() / cache.max_size:.1%}")

# Check for memory leaks
tracemalloc.start()
# Do some operations
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:5]:
    print(stat)
```

**Solutions**:

1. **Reduce embedding cache size**:
   ```bash
   export T4DM_EMBEDDING_CACHE_SIZE=500  # was 1000
   ```

2. **Enable aggressive eviction policy**:
   ```bash
   export T4DM_CACHE_EVICTION_POLICY=lru_aggressive
   ```

3. **Archive old memories**:
   ```bash
   curl -H "X-API-Key: $T4DM_API_KEY" \
     -X POST "http://localhost:8000/storage/archive" \
     -d "{
       \"session_id\": \"$SESSION_ID\",
       \"days_old\": 30,
       \"compress\": true
     }"
   ```

4. **Increase checkpoint frequency**:
   ```bash
   export T4DM_CHECKPOINT_INTERVAL_SECONDS=300  # was 600
   ```

5. **Force garbage collection**:
   ```bash
   curl -H "X-API-Key: $T4DM_API_KEY" \
     -X POST "http://localhost:8000/storage/gc"
   ```

### Issue: CPU Saturation During Consolidation

**Symptoms**:
- CPU goes to 100% during sleep
- Inference blocked during consolidation
- No concurrent execution

**Diagnosis**:
```bash
# Monitor CPU during consolidation
watch -n 1 'ps aux | grep t4dm-service | awk "{print \$3}"'

# Check thread count
ps -p $(pgrep -f t4dm-service) -L | wc -l
```

**Solutions**:

1. **Enable background consolidation**:
   ```bash
   export T4DM_CONSOLIDATION_BACKGROUND=true
   ```

2. **Reduce consolidation batch size**:
   ```bash
   export T4DM_CONSOLIDATION_BATCH_SIZE=100  # was 1000
   ```

3. **Increase consolidation interval**:
   ```bash
   export T4DM_CONSOLIDATION_INTERVAL_SECONDS=60  # was 10
   ```

4. **Use thread pool for compaction**:
   ```bash
   export T4DM_COMPACTOR_THREADS=2  # Parallel compaction
   ```

### Issue: Poor Inference Throughput (< 10 tokens/sec)

**Symptoms**:
- Qwen inference slow despite GPU available
- Spiking blocks add latency
- Batching doesn't help much

**Diagnosis**:
```python
from t4dm.qwen.unified_model import UnifiedModel
import torch

model = UnifiedModel()

# Profile per-layer
input_ids = torch.tensor([[101, 2054, 2003, 102]])

# Find slow layer
for i, layer in enumerate(model.model.layers):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        _ = layer(input_ids)
    end.record()

    torch.cuda.synchronize()
    print(f"Layer {i}: {start.elapsed_time(end):.1f} ms")
```

**Solutions**:

1. **Reduce batch size** (if using batching):
   ```python
   # Smaller batches may have better cache behavior
   batch_size = 4  # was 32
   ```

2. **Enable quantization**:
   ```bash
   export T4DM_QWEN_QUANTIZATION=int8
   ```

3. **Reduce spiking block complexity**:
   ```bash
   export T4DM_SPIKING_NUM_BLOCKS=3  # was 6
   ```

4. **Cache model outputs**:
   ```bash
   export T4DM_INFERENCE_CACHE_SIZE=10000
   ```

5. **Use GPU batching**:
   ```bash
   export T4DM_GPU_BATCH_SIZE=32
   ```

---

## Performance Tuning Checklist

### For Low-Latency Stores:
- [ ] Set `T4DM_FLUSH_THRESHOLD=500` (more frequent flush)
- [ ] Enable `T4DM_CONSOLIDATION_BLOCKING=false`
- [ ] Use `flush_async=True` in store calls
- [ ] Monitor p99 latency: `GET /observability/latency`

### For Fast Search:
- [ ] Rebuild HNSW: `POST /storage/rebuild-hnsw`
- [ ] Increase `T4DM_HNSW_M=24`
- [ ] Use `kappa_min` filtering for recent memories
- [ ] Monitor query plan: `POST /observability/query-profile`

### For Lower Memory:
- [ ] Reduce `T4DM_EMBEDDING_CACHE_SIZE=500`
- [ ] Enable aggressive eviction: `T4DM_CACHE_EVICTION_POLICY=lru_aggressive`
- [ ] Archive old memories: `POST /storage/archive?days_old=30`
- [ ] Increase checkpoint frequency: `T4DM_CHECKPOINT_INTERVAL_SECONDS=300`

### For Better Throughput:
- [ ] Enable background consolidation: `T4DM_CONSOLIDATION_BACKGROUND=true`
- [ ] Increase `T4DM_CONSOLIDATION_BATCH_SIZE=1000`
- [ ] Use thread pool: `T4DM_COMPACTOR_THREADS=2`
- [ ] Monitor ops/sec: `GET /observability/ops-count`

### For Faster Inference:
- [ ] Enable quantization: `T4DM_QWEN_QUANTIZATION=int8`
- [ ] Reduce spiking blocks: `T4DM_SPIKING_NUM_BLOCKS=3`
- [ ] Cache outputs: `T4DM_INFERENCE_CACHE_SIZE=10000`
- [ ] Profile layers: `GET /qwen/layer-timing`
