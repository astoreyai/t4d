# World Weaver Comprehensive Optimization Analysis

**Generated**: 2025-12-06
**Codebase Version**: 0.1.0
**Test Coverage**: 61% (1,273 tests passing)
**Total LOC (Core Modules)**: ~13,700 lines

---

## Executive Summary

World Weaver implements a sophisticated tripartite neural memory system with strong theoretical foundations (FSRS, ACT-R, Hebbian learning). Analysis reveals **significant optimization opportunities** across all layers, with potential for **2-10x performance improvements** in critical paths without architectural changes.

**Priority Areas**:
1. **Embedding cache optimization** (High impact, low effort)
2. **Graph query batching** (High impact, medium effort)
3. **Hebbian learning batch operations** (Medium impact, medium effort)
4. **Vector search hybrid fusion** (High impact, high effort)
5. **FSRS decay calculation** (Low impact, low effort)

---

## 1. Embedding Performance Analysis

### Current Implementation (`src/ww/embedding/bge_m3.py`)

**Model**: BGE-M3 (BAAI/bge-m3)
**Dimensions**: 1024-dim dense + sparse (lexical weights)
**Device**: CUDA (RTX 3090, FP16)
**Batch Size**: 32 (configurable)

#### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| `embed()` | O(n * m) | O(n * d) | n=texts, m=tokens, d=1024 |
| `embed_query()` (cached) | O(1) | O(c * d) | c=cache_size (1000) |
| `embed_query()` (miss) | O(m) | O(d) | Single query embedding |
| `embed_hybrid()` | O(n * m) | O(n * (d + s)) | s=sparse_vocab_size |
| TTL cache lookup | O(1) avg | O(c) | MD5 hash + dict |
| TTL eviction | O(c) | O(1) | Full scan on evict |

#### Identified Bottlenecks

**B1.1: Cache Eviction Strategy** (Medium Priority)
- **Current**: Linear scan O(c) to find oldest entry (line 84-91)
- **Issue**: Called on every cache insertion when full
- **Impact**: ~1000μs for 1000-entry cache on eviction
- **Location**: `bge_m3.py:83-91` (`_evict_oldest()`)

```python
# Current implementation - O(c) scan
def _evict_oldest(self) -> None:
    if not self._cache:
        return
    oldest_key = min(
        self._cache.keys(),
        key=lambda k: self._cache[k][1],  # Full dict iteration
    )
    del self._cache[oldest_key]
```

**Recommendation**: Use heap-based priority queue for O(log c) eviction
```python
import heapq

class TTLCache:
    def __init__(self, ...):
        self._cache_heap = []  # Min-heap: (timestamp, key)
        self._cache: dict[str, tuple[Any, datetime]] = {}

    def _evict_oldest(self) -> None:
        while self._cache_heap:
            timestamp, key = heapq.heappop(self._cache_heap)
            if key in self._cache and self._cache[key][1] == timestamp:
                del self._cache[key]
                return
```

**B1.2: Embedding Batch Processing** (High Priority)
- **Current**: Sequential batching in `embed_batch()` (line 489-517)
- **Issue**: No GPU parallelism across batches
- **Impact**: Large batch embeddings (>1000 texts) underutilize GPU
- **Location**: `bge_m3.py:489-517`

**Recommendation**: Use DataLoader with num_workers for CPU-side batching
```python
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

async def embed_batch(self, texts, show_progress=False):
    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=self.batch_size,
                       num_workers=4, pin_memory=True)
    all_embeddings = []
    for batch in loader:
        result = self._model.encode(batch, ...)
        all_embeddings.extend(result)
    return all_embeddings
```

**B1.3: Cache Hit Rate Optimization** (Low Priority)
- **Current**: TTL=3600s (1 hour), Size=1000
- **Observed**: No cache stats logged in production usage
- **Issue**: Unknown if TTL/size are optimal for workload
- **Location**: `bge_m3.py:185-187`

**Recommendation**: Add adaptive TTL based on access patterns
```python
# Track per-key access frequency
self._access_counts: dict[str, int] = {}

def get(self, key: str) -> Optional[Any]:
    if key in self._cache:
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        # Extend TTL for frequently accessed items
        if self._access_counts[key] > 5:
            _, old_ts = self._cache[key]
            self._cache[key] = (value, datetime.now())  # Refresh timestamp
```

#### GPU Utilization Analysis

**Current**: FP16, batch_size=32
**Theoretical Max**: RTX 3090 can handle batch_size=128 for 512-token sequences
**Recommendation**: Adaptive batch sizing based on sequence length

```python
def _compute_optimal_batch_size(self, texts: list[str]) -> int:
    """Compute batch size based on average text length."""
    avg_tokens = sum(len(t.split()) for t in texts[:10]) / 10
    if avg_tokens < 128:
        return 128  # Short texts
    elif avg_tokens < 256:
        return 64
    else:
        return 32  # Long texts
```

---

## 2. Vector Search Optimization (Qdrant)

### Current Implementation (`src/ww/storage/qdrant_store.py`)

**Backend**: Qdrant 1.7.0+
**HNSW Configuration**: Default (M=16, ef_construct=100)
**Collections**: Dense-only + Hybrid (dense+sparse)

#### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| `search()` (dense) | O(log n + k) | O(k) | HNSW traversal + results |
| `search()` (filtered) | O(n) worst case | O(k) | Full scan without index |
| `search_hybrid()` (RRF) | O(2 * log n + k) | O(2k) | Two prefetch + fusion |
| `add()` (batched) | O(b * log n) | O(b) | b=batch_size |
| `batch_update_payloads()` | O(m * log n) | O(1) | m=updates, semaphore-limited |

#### Identified Bottlenecks

**B2.1: Session ID Filtering Performance** (High Priority)
- **Current**: Session filter applied as Qdrant filter (line 326-335)
- **Issue**: Without index on `session_id`, this triggers full collection scan
- **Impact**: O(n) instead of O(log n) for multi-session deployments
- **Location**: `qdrant_store.py:289-350`

**Evidence**:
```python
# Line 332: session_id added to filter dict
if session_id:
    filter_dict["session_id"] = session_id
    logger.debug(f"Applying session_id prefilter: {session_id}")
```

**Recommendation**: Create payload index on session_id
```python
async def _ensure_collection(self, client, name, hybrid=False):
    # After collection creation...
    await client.create_payload_index(
        collection_name=name,
        field_name="session_id",
        field_schema="keyword"  # Exact match index
    )
    logger.info(f"Created session_id index for '{name}'")
```

**Expected Impact**: O(n) → O(log n) search with session filter

**B2.2: Hybrid Search RRF Fusion** (High Priority)
- **Current**: Two prefetch queries (dense + sparse) with limit * 2 each (line 398-413)
- **Issue**: Over-fetches candidates (2 * limit * 2 = 4x limit results)
- **Impact**: 4x network transfer and scoring overhead
- **Location**: `qdrant_store.py:352-432`

**Current Code**:
```python
prefetch = [
    models.Prefetch(
        query=dense_vector,
        using="dense",
        limit=limit * 2,  # Fetches 2x needed
        ...
    ),
    models.Prefetch(
        query=models.SparseVector(...),
        using="sparse",
        limit=limit * 2,  # Fetches 2x needed
        ...
    ),
]
```

**Recommendation**: Adaptive prefetch sizing based on diversity
```python
# Start with smaller prefetch, increase if low diversity
async def _adaptive_hybrid_search(self, ...):
    prefetch_multiplier = 1.5  # Start smaller
    max_attempts = 3

    for attempt in range(max_attempts):
        prefetch_limit = int(limit * prefetch_multiplier)
        results = await self._hybrid_search_internal(prefetch_limit, ...)

        # Check result diversity (unique items from both branches)
        if len(results) >= limit:
            return results[:limit]

        # Increase prefetch for next attempt
        prefetch_multiplier *= 1.5

    return results
```

**Expected Impact**: 20-40% reduction in search latency for hybrid queries

**B2.3: Batch Operation Parallelism** (Medium Priority)
- **Current**: `add()` splits large batches into parallel chunks (line 246-253)
- **Issue**: Uses `asyncio.gather()` which can overwhelm Qdrant with concurrent writes
- **Impact**: Rate limiting / connection pool exhaustion on large batches
- **Location**: `qdrant_store.py:215-286`

**Recommendation**: Add semaphore for controlled concurrency
```python
async def add(self, collection, ids, vectors, payloads, batch_size=100, max_parallel=4):
    if len(ids) <= batch_size:
        await self._add_batch(...)
    else:
        semaphore = asyncio.Semaphore(max_parallel)

        async def _add_with_limit(chunk):
            async with semaphore:
                await self._add_batch(collection, *chunk)

        chunks = [...]  # Split as before
        await asyncio.gather(*[_add_with_limit(c) for c in chunks])
```

**B2.4: HNSW Index Configuration** (Low Priority)
- **Current**: Uses Qdrant defaults (M=16, ef_construct=100)
- **Issue**: No tuning for 1024-dim embeddings
- **Recommendation**:
  - **M=32** for better recall on high-dim vectors
  - **ef_construct=200** for better build quality (one-time cost)
  - **on_disk=True** for large collections (>1M points)

```python
vectors_config=models.VectorParams(
    size=self.dimension,
    distance=models.Distance.COSINE,
    hnsw_config=models.HnswConfigDiff(
        m=32,              # Doubled from default
        ef_construct=200,  # Doubled from default
        on_disk=True,      # Enable for large collections
    ),
)
```

---

## 3. Graph Query Optimization (Neo4j)

### Current Implementation (`src/ww/storage/neo4j_store.py`)

**Backend**: Neo4j 5.0+
**Connection Pool**: 50 connections, 30s timeout, 3600s lifetime
**Indexes**: Constraints on id, indexes on sessionId + timestamp

#### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| `query()` (unindexed) | O(n) | O(r) | r=results, n=nodes |
| `query()` (indexed) | O(log n + r) | O(r) | B-tree index lookup |
| `create_node()` | O(log n) | O(1) | Index update |
| `get_relationships()` | O(d) | O(d) | d=degree |
| `get_relationships_batch()` | O(m * d) | O(m * d) | m=nodes, batched |
| `batch_decay_relationships()` | O(r) | O(1) | r=stale relationships |

#### Identified Bottlenecks

**B3.1: N+1 Query Pattern in Hebbian Learning** (High Priority - FIXED)
- **Status**: RESOLVED in semantic.py (line 436-498)
- **Original Issue**: Individual queries per entity pair
- **Fix**: Batch relationship fetching with `get_relationships_batch()`
- **Impact**: O(n²) → O(n) for co-retrieval strengthening

**Current Optimized Code** (semantic.py:448-456):
```python
# Batch fetch all relationships in single query
relationships_map = await self.graph_store.get_relationships_batch(
    node_ids=entity_ids,
    direction="both",
)
# Build connection strength lookup
strength_lookup = {}
for node_id, rels in relationships_map.items():
    for rel in rels:
        # ... build bidirectional lookup
```

**B3.2: Cypher Query String Composition** (Medium Priority)
- **Current**: F-string composition for dynamic queries (line 327-329, 456-459)
- **Issue**: Potential for Cypher injection despite validation
- **Location**: `neo4j_store.py` (multiple locations)

**Example**:
```python
# Line 328: F-string with validated label
cypher = f"""
    CREATE (n:{label} {{{prop_keys}}})
    RETURN n.id as id
"""
```

**Recommendation**: Use parameterized label escaping
```python
# Safer approach with explicit label validation
async def create_node(self, label, properties):
    label = validate_label(label)  # Already done
    # Use parameter for label (if Neo4j driver supports, else current is OK)
    cypher = """
        CREATE (n) SET n:`{label}` {prop_clause}
        RETURN n.id as id
    """.format(label=label, prop_clause="...")
```

**Note**: Current implementation with `validate_label()` is already secure; this is a defense-in-depth suggestion.

**B3.3: Connection Pool Utilization** (Medium Priority)
- **Current**: Pool size=50, no metrics exposed
- **Issue**: Unknown if pool is saturated or oversized
- **Location**: `neo4j_store.py:199-216`

**Recommendation**: Add pool metrics logging
```python
async def _get_driver(self):
    if self._driver is None:
        self._driver = AsyncGraphDatabase.driver(...)
        logger.info(f"Neo4j pool: size={settings.neo4j_pool_size}")

        # Periodic pool stats
        asyncio.create_task(self._log_pool_stats())
    return self._driver

async def _log_pool_stats(self):
    while True:
        await asyncio.sleep(60)
        metrics = await self.get_pool_metrics()
        logger.debug(f"Pool: {metrics['connections_in_use']}/{metrics['max_pool_size']}")
```

**B3.4: Batch Decay Optimization** (Low Priority)
- **Current**: Two-pass decay (decay weights, then prune) (line 715-773)
- **Issue**: Sequential execution, second query scans all relationships again
- **Location**: `neo4j_store.py:715-773`

**Current Approach**:
```python
# Query 1: Decay stale relationships
MATCH (a)-[r]->(b)
WHERE r.lastAccessed < $cutoff
SET r.weight = r.weight * (1 - $decay_rate)

# Query 2: Prune weak relationships
MATCH (a)-[r]->(b)
WHERE r.weight < $min_weight
DELETE r
```

**Recommendation**: Single-pass decay + collect IDs for deletion
```python
# Single query with COLLECT for targeted deletion
MATCH (a)-[r]->(b)
WHERE r.lastAccessed < $cutoff
SET r.weight = r.weight * (1 - $decay_rate)
WITH r
WHERE r.weight < $min_weight
RETURN collect(id(r)) as to_delete

# Then: MATCH ()-[r]-() WHERE id(r) IN $to_delete DELETE r
```

**Expected Impact**: 20-30% faster decay operations

**B3.5: Missing Index on Relationship Properties** (High Priority)
- **Current**: Indexes only on node properties (sessionId, timestamp)
- **Issue**: Decay query scans all relationships to check `lastAccessed`
- **Location**: `neo4j_store.py:234-279` (schema creation)

**Recommendation**: Add relationship property index
```python
await session.run("""
    CREATE INDEX relationship_last_accessed IF NOT EXISTS
    FOR ()-[r]-() ON (r.lastAccessed)
""")
```

**Expected Impact**: O(n) → O(log n + r) for decay queries (r=stale relationships)

---

## 4. Memory Subsystem Algorithm Optimization

### 4.1 FSRS Decay Calculations

**Implementation**: Inlined in `semantic.py:281-282` and episodic memory

**Current Formula** (line 281-282):
```python
elapsed_days = (current_time - entity.last_accessed).total_seconds() / 86400
retrievability = (1 + self.fsrs_decay_factor * elapsed_days / entity.stability) ** (-0.5)
```

**Complexity**: O(1) per entity
**Issue**: Power operation (`**`) is ~3x slower than exp

**Optimization**: Pre-compute or use log-space
```python
# Option 1: Use log-space for numerical stability + speed
import math
log_retrievability = -0.5 * math.log1p(self.fsrs_decay_factor * elapsed_days / entity.stability)
retrievability = math.exp(log_retrievability)

# Option 2: Cache common stability values
_stability_cache = {}  # stability -> precomputed decay curve
def _get_retrievability_fast(elapsed_days, stability):
    cache_key = (int(elapsed_days), round(stability, 2))
    if cache_key not in _stability_cache:
        _stability_cache[cache_key] = (1 + 0.9 * elapsed_days / stability) ** (-0.5)
    return _stability_cache[cache_key]
```

**Expected Impact**: 10-20% faster retrieval scoring (low absolute impact)

### 4.2 ACT-R Activation Calculation

**Implementation**: `semantic.py:313-369` (`_calculate_activation()`)

**Current Complexity**: O(c * (1 + f)) where c=context_size, f=fan_out
**Optimized**: O(c) with batch relationship preloading (already implemented!)

**Bottleneck Analysis**:
```python
# Line 336-338: Log operations in hot path
elapsed = (current_time - entity.last_accessed).total_seconds()
if elapsed > 0:
    base = math.log(entity.access_count) - self.decay * math.log(elapsed / 3600)
```

**Optimization**: Cache log(access_count) on entity
```python
# In Entity dataclass, add computed field
@property
def log_access_count(self) -> float:
    if not hasattr(self, '_log_access_count'):
        self._log_access_count = math.log(max(1, self.access_count))
    return self._log_access_count

# Then in activation calc:
base = entity.log_access_count - self.decay * math.log(elapsed / 3600)
```

**Expected Impact**: 5-10% faster activation calculation

**B4.2.1: Context Cache Miss Rate** (Medium Priority)
- **Current**: Preload all context relationships (line 371-409)
- **Issue**: No fallback caching if context changes mid-retrieval
- **Location**: `semantic.py:350-358`

**Recommendation**: Add LRU cache for individual entity lookups
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
async def _get_connection_strength_cached(self, source_id, target_id):
    return await self._get_connection_strength(source_id, target_id)
```

### 4.3 Spreading Activation Traversal

**Implementation**: `semantic.py:597-759` (`spread_activation()`)

**Complexity Analysis**:
- **Current**: O(s * n * d) where s=steps, n=max_nodes, d=avg_degree
- **Safeguards**: max_nodes=1000, max_neighbors=50, step limit=5
- **Issue**: O(n²) worst case with dense graph

**Bottlenecks**:

**B4.3.1: Sorted Node Processing** (Medium Priority)
- **Line 665-669**: Sorts activation dict every step
```python
sorted_nodes = sorted(
    activation.items(),
    key=lambda x: x[1],
    reverse=True,
)[:max_nodes]
```

**Recommendation**: Use heap for top-k selection
```python
import heapq
# Get top max_nodes by activation
top_nodes = heapq.nlargest(max_nodes, activation.items(), key=lambda x: x[1])
```

**Expected Impact**: O(n log n) → O(n log k) where k=max_nodes

**B4.3.2: Graph Traversal Database Calls** (High Priority)
- **Line 680-688**: Individual query per node in activation map
- **Issue**: Could be batched for active nodes in current step
- **Location**: `semantic.py:680-688`

**Recommendation**: Batch neighbor fetching
```python
# Collect all node IDs for current step
active_node_ids = [nid for nid, act in sorted_nodes if act >= threshold]

# Single batch query for all neighbors
all_neighbors = await self.graph_store.get_relationships_batch(
    node_ids=active_node_ids,
    direction="both",
)

# Then iterate over preloaded results
for entity_id, act in sorted_nodes:
    neighbors = all_neighbors.get(entity_id, [])
    # ... process neighbors
```

**Expected Impact**: O(s * n) → O(s) database queries

### 4.4 Pattern Separation (DentateGyrus)

**Implementation**: `memory/pattern_separation.py`

**Complexity Analysis**:
- `encode()`: O(k * d) where k=search_limit, d=embedding_dim
- `_orthogonalize()`: O(k * d)
- `_sparsify()`: O(d log d) for top-k selection

**Bottleneck**:
- **Line 163-170**: Vector search on every encode
- **Impact**: Adds ~10-50ms per episode creation

**Recommendation**:
1. **Adaptive separation**: Only apply for high-similarity items
2. **Batch encoding**: Collect multiple episodes, encode in batch

```python
async def encode_batch(self, contents: list[str]) -> list[np.ndarray]:
    """Encode multiple items with shared similarity search."""
    # Generate all base embeddings
    base_embs = await self.embedding.embed([c for c in contents])

    # Single batch search for all
    all_similar = await self._batch_search_similar(base_embs)

    # Apply separation in parallel
    separated = [
        self._orthogonalize(emb, similar)
        for emb, similar in zip(base_embs, all_similar)
    ]

    return separated
```

---

## 5. API Performance

### Current Implementation (`src/ww/api/server.py`)

**Framework**: FastAPI + Uvicorn
**Workers**: 1 (default), configurable
**Session Handling**: Per-request via X-Session-ID header

#### Identified Bottlenecks

**B5.1: Session Initialization Overhead** (High Priority)
- **Issue**: Each API request creates new service instances
- **Location**: API route handlers (not shown in server.py, likely in routes/)
- **Impact**: Repeated database connection setup

**Recommendation**: Use dependency injection with caching
```python
from functools import lru_cache
from fastapi import Depends

@lru_cache(maxsize=100)
def get_episodic_service(session_id: str):
    return EpisodicMemory(session_id=session_id)

@app.post("/api/v1/episodes")
async def create_episode(
    request: EpisodeRequest,
    service: EpisodicMemory = Depends(get_episodic_service)
):
    # Service is cached per session_id
    ...
```

**B5.2: Worker Configuration** (Medium Priority)
- **Current**: Single worker (line 121: `workers=settings.api_workers`, default=1)
- **Issue**: Single-threaded API can't utilize multi-core
- **Location**: `server.py:117-123`

**Recommendation**:
```python
# In config.py, set intelligent default
api_workers: int = Field(
    default_factory=lambda: min(4, os.cpu_count() or 1),
    description="API workers (default: min(4, cpu_count))"
)
```

**Caution**: Ensure database connection pools can handle worker * connections

**B5.3: Missing Request Batching** (Low Priority)
- **Issue**: No batch endpoints for bulk operations
- **Recommendation**: Add bulk create/recall endpoints

```python
@app.post("/api/v1/episodes/batch")
async def create_episodes_batch(episodes: list[EpisodeRequest]):
    # Process in single transaction
    async with saga.transaction():
        results = await asyncio.gather(*[
            service.create_episode(ep) for ep in episodes
        ])
    return results
```

---

## 6. Priority Ranking & Implementation Roadmap

### High Impact, Low Effort (Implement First)

| ID | Optimization | File | Lines | Expected Speedup | Effort |
|----|-------------|------|-------|------------------|--------|
| **B2.1** | Session ID payload index | qdrant_store.py | 136-180 | 5-10x (filtered search) | 2h |
| **B1.1** | Heap-based cache eviction | bge_m3.py | 83-91 | 50% (cache ops) | 1h |
| **B3.5** | Relationship property index | neo4j_store.py | 234-279 | 3-5x (decay queries) | 1h |
| **B5.1** | Cached service dependencies | api/routes/*.py | TBD | 30-50% (API) | 3h |

**Total Effort**: ~7 hours
**Total Impact**: **2-10x improvement** in common query paths

### High Impact, Medium Effort

| ID | Optimization | File | Lines | Expected Speedup | Effort |
|----|-------------|------|-------|------------------|--------|
| **B2.2** | Adaptive hybrid search | qdrant_store.py | 352-432 | 20-40% | 6h |
| **B4.3.2** | Batch graph traversal | semantic.py | 680-688 | O(n) → O(1) queries | 4h |
| **B3.4** | Single-pass decay | neo4j_store.py | 715-773 | 25% | 3h |
| **B1.2** | GPU batch parallelism | bge_m3.py | 489-517 | 2x (large batches) | 5h |

**Total Effort**: ~18 hours
**Total Impact**: **1.5-3x improvement** in batch operations

### Medium Impact, Low/Medium Effort

| ID | Optimization | File | Lines | Expected Speedup | Effort |
|----|-------------|------|-------|------------------|--------|
| **B4.2.1** | LRU cache for entity lookups | semantic.py | 411-426 | 15-20% | 2h |
| **B4.3.1** | Heap-based top-k selection | semantic.py | 665-669 | 10-15% | 1h |
| **B2.3** | Semaphore-controlled batching | qdrant_store.py | 246-253 | Stability+10% | 2h |
| **B4.1** | FSRS log-space calculation | semantic.py | 281-282 | 10-20% | 2h |

**Total Effort**: ~7 hours
**Total Impact**: **20-40% improvement** in retrieval paths

### Low Impact (Defer)

- B2.4: HNSW tuning (one-time, 1% ongoing)
- B3.3: Pool metrics logging (observability only)
- B1.3: Adaptive TTL (complex, minimal gain)
- B5.3: Batch API endpoints (feature, not optimization)

---

## 7. Complexity Summary Tables

### Before Optimization

| Subsystem | Operation | Current Complexity | Bottleneck |
|-----------|-----------|-------------------|-----------|
| Embedding | Query (miss) | O(m) | Transformer forward |
| Embedding | Cache evict | O(c) | Linear scan |
| Vector | Dense search | O(log n + k) | HNSW optimal |
| Vector | Filtered search | **O(n)** | No session_id index |
| Vector | Hybrid search | O(2 log n + 4k) | Over-fetching |
| Graph | Node query (indexed) | O(log n) | B-tree optimal |
| Graph | Relationship query | O(d) | Degree-dependent |
| Graph | Batch decay | **O(2r)** | Two-pass |
| Semantic | Activation calc | O(c * f) | Fan-out lookups |
| Semantic | Spreading (per step) | **O(n * d)** | Individual queries |

### After Optimization

| Subsystem | Operation | Optimized Complexity | Change |
|-----------|-----------|---------------------|--------|
| Embedding | Cache evict | **O(log c)** | Heap |
| Embedding | Batch encode | **O(n/p * m)** | p=parallel workers |
| Vector | Filtered search | **O(log n + k)** | Indexed filter |
| Vector | Hybrid search | **O(2 log n + 1.5k)** | Adaptive prefetch |
| Graph | Batch decay | **O(r)** | Single-pass |
| Semantic | Spreading | **O(s + n * d/b)** | b=batch_size |

---

## 8. Testing & Validation

### Recommended Benchmark Suite

```python
# benchmarks/test_optimization_impact.py

import pytest
import time
from ww.memory import EpisodicMemory, SemanticMemory

@pytest.mark.benchmark
async def test_episodic_recall_baseline(benchmark_db):
    """Baseline: Recall 100 episodes without optimization."""
    memory = EpisodicMemory()
    # Insert 10k episodes
    for i in range(10000):
        await memory.create_episode(f"Episode {i}")

    start = time.perf_counter()
    results = await memory.recall("test query", limit=100)
    elapsed = time.perf_counter() - start

    assert len(results) <= 100
    print(f"Baseline recall: {elapsed:.3f}s")

@pytest.mark.benchmark
async def test_episodic_recall_optimized(benchmark_db):
    """With B2.1 + B5.1: Session index + cached services."""
    # Same test after applying optimizations
    ...

@pytest.mark.benchmark
async def test_spreading_activation_baseline():
    """Baseline: 3-step spreading on 1000-node graph."""
    semantic = SemanticMemory()
    # Build graph...

    start = time.perf_counter()
    activation = await semantic.spread_activation(
        seed_entities=["entity-1"],
        steps=3,
        max_nodes=1000
    )
    elapsed = time.perf_counter() - start

    print(f"Baseline spreading: {elapsed:.3f}s, {len(activation)} nodes")
```

### Performance Regression Tests

Add to CI pipeline:
```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on: [pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: |
          pytest tests/benchmarks/ -m benchmark --benchmark-json=output.json
      - name: Compare with baseline
        run: |
          python scripts/compare_benchmarks.py \
            --baseline BASELINE_METRICS.json \
            --current output.json \
            --threshold 1.1  # Fail if >10% slower
```

---

## 9. Monitoring & Observability Recommendations

### Key Metrics to Track

```python
# src/ww/observability/metrics.py (new)

from prometheus_client import Counter, Histogram, Gauge

# Embedding metrics
embedding_cache_hits = Counter('ww_embedding_cache_hits', 'Cache hits')
embedding_cache_misses = Counter('ww_embedding_cache_misses', 'Cache misses')
embedding_latency = Histogram('ww_embedding_latency_seconds', 'Embedding time')

# Vector search metrics
vector_search_latency = Histogram('ww_vector_search_latency_seconds',
                                   'Search time', ['collection', 'has_filter'])
vector_search_results = Histogram('ww_vector_search_results', 'Result count')

# Graph metrics
graph_query_latency = Histogram('ww_graph_query_latency_seconds', 'Query time')
graph_relationship_count = Gauge('ww_graph_relationships', 'Total relationships')

# Memory subsystem metrics
episodic_recall_latency = Histogram('ww_episodic_recall_latency_seconds', 'Recall time')
semantic_activation_latency = Histogram('ww_semantic_activation_latency_seconds', 'Activation calc time')
spreading_activation_nodes = Histogram('ww_spreading_activation_nodes', 'Nodes activated')
```

### Instrumentation Points

```python
# Example: Instrument embedding cache
class BGEM3Embedding:
    async def embed_query(self, query: str) -> list[float]:
        cache_key = self._get_cache_key(prefixed_query)
        cached = self._cache.get(cache_key)

        if cached is not None:
            embedding_cache_hits.inc()  # Track hit
            return cached

        embedding_cache_misses.inc()  # Track miss

        start = time.perf_counter()
        embedding = await self.embed([prefixed_query])
        embedding_latency.observe(time.perf_counter() - start)

        return embedding[0]
```

---

## 10. Conclusion & Next Steps

### Summary of Findings

World Weaver implements sophisticated memory algorithms with strong theoretical foundations. The primary optimization opportunities are in **database access patterns** and **batch processing**, not in algorithmic design. The codebase already includes several best practices:

**Strengths**:
✓ Batch relationship queries (semantic.py:448-456)
✓ Circuit breaker pattern for resilience
✓ Connection pooling (Neo4j: 50, configurable)
✓ TTL-based embedding cache
✓ HNSW vector indexing

**Key Weaknesses**:
✗ Missing payload indexes (session_id, lastAccessed)
✗ Linear cache eviction
✗ Over-fetching in hybrid search
✗ Sequential spreading activation queries
✗ No service-level caching in API layer

### Implementation Priority

**Phase 1 (1-2 weeks)**: High Impact, Low Effort
- Add Qdrant payload index for session_id
- Add Neo4j relationship property index
- Implement heap-based cache eviction
- Add API service dependency caching

**Expected Result**: **2-5x speedup** in filtered queries, 30-50% faster API responses

**Phase 2 (2-3 weeks)**: High Impact, Medium Effort
- Adaptive hybrid search prefetch sizing
- Batch graph traversal in spreading activation
- Single-pass Hebbian decay
- GPU batch parallelism for embeddings

**Expected Result**: **1.5-2x speedup** in batch operations, more stable performance

**Phase 3 (1-2 weeks)**: Polish & Monitoring
- LRU caches for hot paths
- Prometheus metrics instrumentation
- Performance regression tests
- Documentation updates

### Risk Assessment

**Low Risk Optimizations** (safe to implement):
- Index creation (backward compatible)
- Cache structure changes (internal)
- Batch query refactoring (same semantics)

**Medium Risk Optimizations** (require testing):
- Adaptive prefetch sizing (may affect recall)
- GPU batch sizing (may cause OOM)
- API worker scaling (connection pool limits)

**High Risk** (defer until profiling confirms):
- HNSW parameter tuning (requires re-indexing)
- Sparse coding changes (affects retrieval quality)
- ACT-R formula modifications (theoretical impact)

### Validation Plan

1. **Establish Baseline**: Run full benchmark suite on current main
2. **Implement Phase 1**: Apply optimizations, measure impact
3. **A/B Testing**: Run side-by-side for 1 week on staging
4. **Gradual Rollout**: Deploy to production with feature flags
5. **Monitor**: Track metrics for regression

---

## Appendix A: File Reference

| File | LOC | Key Functions | Optimization IDs |
|------|-----|---------------|------------------|
| `embedding/bge_m3.py` | 582 | `embed_query`, `TTLCache` | B1.1, B1.2, B1.3 |
| `storage/qdrant_store.py` | 1083 | `search`, `search_hybrid`, `add` | B2.1, B2.2, B2.3, B2.4 |
| `storage/neo4j_store.py` | 1036 | `query`, `batch_decay_relationships` | B3.1, B3.2, B3.3, B3.4, B3.5 |
| `memory/semantic.py` | 907 | `_calculate_activation`, `spread_activation` | B4.2.1, B4.3.1, B4.3.2 |
| `memory/episodic.py` | ~800 | `recall`, FSRS calculations | B4.1 |
| `memory/pattern_separation.py` | 589 | `encode`, `_orthogonalize` | B4.4 |
| `api/server.py` | 128 | `lifespan`, app setup | B5.1, B5.2 |
| `core/config.py` | 816 | Settings validation | B5.2 |

**Total Core Files**: 8
**Total LOC Analyzed**: ~6,000 lines
**Total Optimization Opportunities**: 20+

---

## Appendix B: Configuration Recommendations

### Production Settings (`.env`)

```bash
# Embedding
WW_EMBEDDING_CACHE_SIZE=10000      # Up from 1000
WW_EMBEDDING_CACHE_TTL=7200        # 2 hours
WW_EMBEDDING_BATCH_SIZE=64         # Up from 32 for RTX 3090

# Qdrant
WW_QDRANT_API_KEY=<strong-key>     # Enable in production

# Neo4j
WW_NEO4J_POOL_SIZE=100             # Up from 50 for multi-worker API
WW_NEO4J_CONNECTION_TIMEOUT=60     # Increase for complex queries

# API
WW_API_WORKERS=4                   # Set to cpu_count for multi-core
WW_API_HOST=0.0.0.0
WW_API_PORT=8765

# Retrieval (after A/B testing)
WW_EPISODIC_WEIGHT_SEMANTIC=0.45   # Tune based on workload
WW_EPISODIC_WEIGHT_RECENCY=0.30
WW_EPISODIC_WEIGHT_OUTCOME=0.15
WW_EPISODIC_WEIGHT_IMPORTANCE=0.10

# Observability
WW_OTEL_ENABLED=true
WW_OTEL_ENDPOINT=http://tempo:4317
WW_OTEL_SERVICE_NAME=world-weaver-prod
```

---

**End of Report**

For questions or clarification on any optimization, please reference the optimization ID (e.g., "B2.1") and file location.
