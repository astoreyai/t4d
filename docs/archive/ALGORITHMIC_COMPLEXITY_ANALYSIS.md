# T4DM Algorithmic Complexity Analysis

**Date**: 2026-01-03
**Analyst**: T4DM Algorithm Design Agent
**Codebase Version**: Latest (commit ea7094d)

---

## Executive Summary

This document provides comprehensive algorithmic complexity analysis of T4DM's core components, including memory operations, embedding computations, consolidation algorithms, and prediction systems. Key findings:

1. **Critical Path Operations**: Most operations are O(n) or O(log n), with HDBSCAN clustering as the main O(n log n) operation
2. **Identified Bottlenecks**: Two O(n^2) patterns found in pairwise similarity computations (now mitigated)
3. **Parallelization**: Extensive async/await patterns with semaphore-limited concurrency
4. **Space Efficiency**: Buffer sizes are bounded with explicit limits throughout

---

## Table of Contents

1. [Critical Path Analysis](#1-critical-path-analysis)
2. [Data Structures](#2-data-structures)
3. [Bottleneck Identification](#3-bottleneck-identification)
4. [Parallelization Opportunities](#4-parallelization-opportunities)
5. [Space Complexity](#5-space-complexity)
6. [Recommendations](#6-recommendations)

---

## 1. Critical Path Analysis

### 1.1 Memory Store Operations

#### Episode Creation (`EpisodicMemory.create`)
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py:678-921`

```
Time Complexity:
- Embedding generation: O(L) where L = input token length
- Pattern separation: O(d) where d = embedding dimension (1024)
- Gate decision: O(d) - MLP forward pass
- Vector store upsert: O(log n) amortized (HNSW index)
- Graph node creation: O(1) via index
- Temporal linking: O(1) - single relationship create

Total: O(L + d + log n) ~= O(L) dominated by embedding

Space Complexity:
- Embedding: O(d) = O(1024)
- Payload: O(1) fixed fields
- Saga state: O(1)
```

#### Episode Recall (`EpisodicMemory.recall`)
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py:1054-1433`

```
Time Complexity:
- Query embedding: O(L)
- Pattern completion: O(a * i) where a = attractors, i = iterations (max 5)
- Query-memory separation: O(d)
- Cluster selection (hierarchical): O(K) where K = clusters
- Vector search:
  - Flat: O(n * d) brute force or O(log n * d) with HNSW
  - Hierarchical: O(K + k * n/K) where k = selected clusters
- Score computation: O(k) per result
- Learned reranking: O(k * d) for k results
- Buffer probe: O(b * d) where b = buffer size

Best case (hierarchical): O(L + K + k * n/K + k * d)
Worst case (flat search): O(L + n * d)

Space Complexity:
- Results buffer: O(k * d) for embeddings
- Score components: O(k * 4)
```

### 1.2 Embedding Computations

#### BGE-M3 Embedding (`BGEM3Embedding.embed`)
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/embedding/bge_m3.py:287-332`

```
Time Complexity:
- Per batch: O(B * L * d) where B = batch size, L = sequence length
- With caching: O(1) for cache hits (MD5 hash lookup)
- Cache eviction: O(log n) using heap (P2-OPT-B1.1 fix)

Space Complexity:
- Model parameters: ~350M parameters (fixed)
- Cache: O(C) where C = max_cache_size (default 1000)
- Heap for LRU: O(C)
```

#### TTLCache Eviction
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/embedding/bge_m3.py:22-157`

```
OPTIMIZED (P2-OPT-B1.1):
- set(): O(log n) - heap push
- _evict_oldest(): O(log n) - heap pop (was O(n) before fix)
- get(): O(1) - dict lookup
- evict_expired(): O(n) - full scan (batch operation)
```

### 1.3 Consolidation Algorithms

#### HDBSCAN Clustering (`ConsolidationService._cluster_episodes`)
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/service.py:1033-1158`

```
Time Complexity:
- HDBSCAN: O(n log n) average case, O(n^2) worst case
- Stratified sampling (if n > max_samples): O(n) for sampling
- Cluster assignment for non-sampled: O(m * k * d) where m = unsampled, k = clusters

With sampling (n > 2000):
  Total: O(s log s + m * k) where s = sampled size (2000 default)

Without sampling (n <= 2000):
  Total: O(n log n)

Space Complexity:
- Embeddings array: O(n * d)
- Distance matrix (if computed): O(n^2) - HDBSCAN avoids this with tree structure
- Cluster labels: O(n)
```

#### Duplicate Detection (`ConsolidationService._find_duplicates`)
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/service.py:952-1031`

```
OPTIMIZED (was O(n^2), now O(n * k)):
- Uses Qdrant ANN search instead of pairwise comparison
- Per episode: O(k) for k nearest neighbors
- Total: O(n * k) where k = candidates per episode (10)
- Seen pairs tracking: O(p) where p = found pairs

Space Complexity:
- Episode map: O(n)
- Seen pairs set: O(p)
```

### 1.4 Prediction Forward Passes

#### Latent Predictor (`LatentPredictor.predict`)
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/prediction/latent_predictor.py:204-258`

```
Time Complexity:
- Per layer: O(h * d) where h = hidden dim, d = input/output dim
- Total: O(L * h * d) where L = num_hidden_layers (2)
- With residual: O(d) additional
- Normalization: O(d)

Total: O(h * d) = O(512 * 1024) = O(524,288) = O(1) constant

Space Complexity:
- Weights: O(L * h * d) ~= 2MB per predictor
- Activations: O(h + d)
```

#### Hierarchical Predictor (`HierarchicalPredictor.predict`)
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/prediction/hierarchical_predictor.py:149-183`

```
Time Complexity:
- Context encoding: O(c * d) where c = context size
- Fast predictor: O(h * d)
- Medium predictor: O(h * d)
- Slow predictor: O(h * d)
- Combined: O(1)

Total: O(c * d + 3 * h * d) = O(c * d)

Space Complexity:
- Episode buffer: O(B * d) where B = max buffer size (100)
- Pending predictions: O(300) - 100 per horizon
- Error history: O(500)
```

---

## 2. Data Structures

### 2.1 Buffer Implementations

#### Working Memory Buffer
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/working_memory.py:112-639`

```
Structure: list[WorkingMemoryItem[T]]
Capacity: Fixed (default 4, Cowan's limit)

Operations:
- load(): O(1) amortized with O(n) decay application
- retrieve(): O(n) linear scan by ID
- update_priority(): O(n) linear scan
- get_by_priority(): O(n log n) sort
- get_most_attended(): O(n) via np.argmax
- _evict_lowest(): O(n) via np.argmin

Space: O(capacity * item_size) = O(4) constant
```

#### Buffer Manager (CA1-like)
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/buffer_manager.py:74-728`

```
Structure: dict[UUID, BufferedItem]
Capacity: Bounded (default 50)

Operations:
- add(): O(1) dict insert, O(n) eviction if full
- probe(): O(n * d) cosine similarity computation
- tick(): O(n) evaluation + O(k log k) sort for promotion
- accumulate_evidence(): O(1) dict lookup + update

Space: O(B * (d + f)) where B = buffer size, d = embedding dim, f = features dim
```

#### Prediction Queue
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/prediction/hierarchical_predictor.py:121-129`

```
Structure: dict[str, list[tuple[step, target_step, Prediction]]]
Capacity: Bounded (100 per horizon, 300 total)

Operations:
- append: O(1)
- resolve: O(p) where p = pending at target step
- trim: O(1) slice operation

Space: O(300 * prediction_size)
```

### 2.2 Graph Structures

#### Causal Graph (Neo4j)
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/storage/t4dx_graph_adapter.py`

```
Structure: Property graph with Episode, Entity, Procedure nodes

Operations:
- create_node(): O(1) with index
- create_relationship(): O(1)
- batch_create_relationships(): O(n) with UNWIND optimization
- get_relationships(): O(k) where k = relationship count
- find_path(): O(d^k) BFS with depth limit k (max 10)

Indexes:
- episode_id: UNIQUE constraint
- entity_id: UNIQUE constraint
- procedure_id: UNIQUE constraint
- session_id: Index for filtering

Space: O(N + R) where N = nodes, R = relationships
```

#### Semantic Cluster Graph
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/cluster_index.py:78-661`

```
Structure:
- dict[str, ClusterMeta] for metadata
- np.ndarray[n_clusters, d] centroid matrix

Operations:
- register_cluster(): O(K) matrix rebuild
- select_clusters(): O(K * d) batch matmul
- update_centroid(): O(K) matrix rebuild
- find_nearest_cluster(): O(K * d) batch similarity

Space: O(K * (d + m)) where K = clusters, m = avg members per cluster
```

### 2.3 Index Structures

#### Vector Index (Qdrant HNSW)
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/storage/t4dx_vector_adapter.py`

```
Structure: HNSW (Hierarchical Navigable Small World)

Operations:
- insert: O(log n * M) where M = connectivity
- search: O(log n * ef) where ef = search expansion factor
- filtered search: O(log n) with payload index, O(n) without

Payload Indexes (P2-OPT-B2.1):
- session_id: KEYWORD index for O(log n) filtering

Space: O(n * (d + M * log n))
```

#### Temporal Index (Qdrant timestamp)
```
Operations:
- Range query: O(log n + k) with index
- Without index: O(n) scan

Note: Timestamp filtering uses payload range filter
```

#### Learned Sparse Index
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/learned_sparse_index.py:68-467`

```
Structure:
- W_shared: [hidden x embed] = [256 x 1024]
- W_cluster: [max_clusters x hidden] = [500 x 256]
- W_feature: [embed x hidden] = [1024 x 256]
- W_sparsity: [1 x hidden] = [1 x 256]

Operations:
- forward(): O(K * h + d * h) = O((K + d) * h)
- update(): O(K * h + d * h) - same as forward + gradients
- get_feature_weighted_score(): O(h * d)

Space: O(K * h + d * h) ~= 390KB per index
```

---

## 3. Bottleneck Identification

### 3.1 O(n^2) or Worse Operations

| Operation | File:Line | Complexity | Mitigation |
|-----------|-----------|------------|------------|
| ~~Pairwise duplicate detection~~ | service.py:952 | ~~O(n^2)~~ | **FIXED**: Now O(n*k) via ANN |
| HDBSCAN worst case | service.py:1114 | O(n^2) | Sampling cap at 2000 |
| Path finding | t4dx_graph_adapter.py:1218 | O(d^k) | Depth limit of 10 |
| Full buffer probe | buffer_manager.py:226 | O(n*d) | Buffer capped at 50 |

### 3.2 Memory-Intensive Operations

| Operation | File:Line | Memory Usage | Mitigation |
|-----------|-----------|--------------|------------|
| Episode clustering | service.py:1102 | O(n*d) | Sampling reduces to O(s*d) |
| Centroid matrix | cluster_index.py:203 | O(K*d) | Cluster pruning |
| Embedding cache | bge_m3.py:208 | O(C*d) | TTL eviction + size limit |
| Prediction buffer | hierarchical_predictor.py:121 | O(B*d) | Max 100 per predictor |
| Eviction history | working_memory.py:156 | O(H) | Bounded at 10000 |

### 3.3 I/O Bound Operations

| Operation | File:Line | Bottleneck | Mitigation |
|-----------|-----------|------------|------------|
| Qdrant search | t4dx_vector_adapter.py:444 | Network RTT | Batch operations |
| Neo4j queries | t4dx_graph_adapter.py:460 | Connection pool | Pool size configurable |
| Embedding model | bge_m3.py:258 | GPU memory | FP16 mode, caching |
| Hybrid collection | t4dx_vector_adapter.py:551 | Double storage | Optional feature |

---

## 4. Parallelization Opportunities

### 4.1 Embarrassingly Parallel Operations

| Operation | File:Line | Current Status | Recommendation |
|-----------|-----------|----------------|----------------|
| Batch embedding | bge_m3.py:484 | Sequential batches | Already batched, GPU parallelism |
| Batch vector upsert | t4dx_vector_adapter.py:298 | Parallel with semaphore | Good (max_concurrency=10) |
| Batch payload update | t4dx_vector_adapter.py:979 | Parallel with semaphore | Good (max_concurrency=10) |
| Cluster assignment | service.py:896 | Sequential | Could parallelize centroid comparison |

### 4.2 Batch Processing Opportunities

| Operation | File:Line | Current | Opportunity |
|-----------|-----------|---------|-------------|
| Relationship creation | t4dx_graph_adapter.py:642 | **OPTIMIZED** | batch_create_relationships with UNWIND |
| Provenance linking | service.py:721 | **OPTIMIZED** | Batch collection before create |
| Payload updates | service.py:579 | Per-duplicate | Could batch after loop |

### 4.3 Async/Concurrent Patterns in Use

```python
# Pattern 1: Semaphore-limited concurrency
# File: t4dx_vector_adapter.py:341-353
semaphore = asyncio.Semaphore(max_concurrency)
async def add_with_limit(chunk_ids, chunk_vecs, chunk_payloads):
    async with semaphore:
        await self._add_batch(...)

# Pattern 2: asyncio.gather for parallel operations
# File: t4dx_vector_adapter.py:356-363
tasks = [add_with_limit(chunk_ids, chunk_vecs, chunk_payloads)
         for chunk_ids, chunk_vecs, chunk_payloads in chunks]
await asyncio.gather(*tasks)

# Pattern 3: Circuit breaker for resilience
# File: t4dx_vector_adapter.py:96-106
async def _with_timeout(self, coro, operation: str):
    async def _execute():
        async with asyncio.timeout(self.timeout):
            return await coro
    return await self._circuit_breaker.execute(_execute)

# Pattern 4: Lock-protected singleton initialization
# File: t4dx_vector_adapter.py:127-146
async with self._get_init_lock():
    if self._client is None:
        self._client = AsyncQdrantClient(...)
```

---

## 5. Space Complexity

### 5.1 Memory Footprint of Major Components

| Component | Size Formula | Typical Size | Notes |
|-----------|--------------|--------------|-------|
| BGE-M3 Model | Fixed | ~1.3GB | FP16 on GPU |
| Embedding Cache | O(C * d * 4B) | ~4MB | 1000 entries x 1024 x 4 bytes |
| Cluster Index | O(K * (d + m)) | ~2MB | 500 clusters, 1024 dim, 10 avg members |
| Working Memory | O(4 * item) | ~16KB | 4 items with embeddings |
| Buffer Manager | O(50 * (d + f)) | ~250KB | 50 items with embeddings + features |
| Prediction Buffer | O(100 * d) | ~400KB | 100 episodes x 1024 x 4 bytes |
| Learned Fusion | O(h*d + h*4) | ~135KB | 32*1024 + 32*4 matrices |
| Learned Reranker | O(h*d + h*20) | ~140KB | Similar to fusion |
| Sparse Index | O(K*h + d*h) | ~390KB | 500*256 + 1024*256 |

**Total In-Memory Footprint**: ~1.5GB (dominated by BGE-M3 model)

### 5.2 Buffer Size Management

| Buffer | Max Size | Eviction Policy | File:Line |
|--------|----------|-----------------|-----------|
| Embedding cache | 1000 | TTL + LRU heap | bge_m3.py:34 |
| Working memory | 4 | Priority-based | working_memory.py:127 |
| CA1 buffer | 50 | Lowest evidence | buffer_manager.py:139 |
| Episode buffer (predictor) | 100 | FIFO tail | hierarchical_predictor.py:142 |
| Error history | 500 | FIFO tail | hierarchical_predictor.py:129 |
| Eviction history | 10000 | FIFO tail | working_memory.py:157 |
| Pending predictions | 100 per horizon | FIFO | hierarchical_predictor.py:167 |
| STDP spike history | 100 per entity | Age-based | stdp.py:176 |
| Cooldown dict | 100 | Age-based | buffer_manager.py:153 |

### 5.3 State Persistence Overhead

| State | Serialization | Storage Location | Size Estimate |
|-------|---------------|------------------|---------------|
| Learned Fusion | JSON/dict | Checkpoint file | ~4KB |
| Learned Reranker | JSON/dict | Checkpoint file | ~4KB |
| Cluster Index | JSON/dict | Checkpoint file | ~2MB |
| Sparse Index | JSON/dict | Checkpoint file | ~1.5MB |
| STDP weights | JSON/dict | Checkpoint file | Variable (per synapse) |
| Predictor weights | JSON/dict | Checkpoint file | ~2MB per predictor |

**WAL (Write-Ahead Log)**:
- File: `/mnt/projects/t4d/t4dm/src/t4dm/persistence/wal.py`
- Entry size: O(operation_size)
- Rotation: Configurable threshold

---

## 6. Recommendations

### 6.1 High Priority Optimizations

1. **Cluster Member Filtering in Qdrant**
   - Current: Post-filter after vector search
   - Recommended: Use Qdrant's native ID filter for hierarchical search
   - Impact: Reduces O(k * n/K) to O(k * log(n/K))
   - Location: `episodic.py:1221-1234`

2. **Batch Cosine Similarity in Buffer Probe**
   - Current: Per-item cosine computation
   - Recommended: Vectorized numpy batch operation
   - Impact: Better cache utilization, SIMD acceleration
   - Location: `buffer_manager.py:256-271`

### 6.2 Medium Priority Optimizations

3. **Lazy Centroid Matrix Rebuild**
   - Current: Rebuild on every cluster add/remove
   - Recommended: Batch updates, rebuild on first query after changes
   - Impact: Reduces O(K) to O(1) for sequential cluster operations
   - Location: `cluster_index.py:194`

4. **Streaming Entity Extraction**
   - Current: Load all episodes, then batch extract
   - Recommended: Stream with async generator
   - Impact: Reduces peak memory from O(n) to O(batch_size)
   - Location: `service.py:1369-1370`

### 6.3 Monitoring Recommendations

| Metric | Threshold | Action |
|--------|-----------|--------|
| Cluster selection time | > 10ms | Reduce cluster count or increase K |
| Buffer probe time | > 50ms | Reduce buffer size |
| Embedding cache hit rate | < 70% | Increase cache size or TTL |
| HDBSCAN sampling trigger | Frequent | Tune consolidation frequency |
| Neo4j pool active sessions | > 80% | Increase pool size |

### 6.4 Algorithm Variants to Consider

1. **Product Quantization for Vector Search**
   - Would reduce memory by 8-16x
   - Slight recall tradeoff
   - Qdrant supports this natively

2. **Approximate Clustering (Mini-batch K-means)**
   - O(n) instead of O(n log n)
   - Faster consolidation cycles
   - Less biologically plausible

3. **Locality-Sensitive Hashing for Buffer Probe**
   - O(1) expected for similarity search
   - Would enable larger buffer sizes
   - Implementation overhead

---

## Appendix: Key File References

| Component | Primary File |
|-----------|--------------|
| Episodic Memory | `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` |
| Working Memory | `/mnt/projects/t4d/t4dm/src/t4dm/memory/working_memory.py` |
| Buffer Manager | `/mnt/projects/t4d/t4dm/src/t4dm/memory/buffer_manager.py` |
| Cluster Index | `/mnt/projects/t4d/t4dm/src/t4dm/memory/cluster_index.py` |
| Sparse Index | `/mnt/projects/t4d/t4dm/src/t4dm/memory/learned_sparse_index.py` |
| Consolidation | `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/service.py` |
| Embeddings | `/mnt/projects/t4d/t4dm/src/t4dm/embedding/bge_m3.py` |
| Prediction | `/mnt/projects/t4d/t4dm/src/t4dm/prediction/hierarchical_predictor.py` |
| Vector Store | `/mnt/projects/t4d/t4dm/src/t4dm/storage/t4dx_vector_adapter.py` |
| Graph Store | `/mnt/projects/t4d/t4dm/src/t4dm/storage/t4dx_graph_adapter.py` |
| STDP Learning | `/mnt/projects/t4d/t4dm/src/t4dm/learning/stdp.py` |

---

*Generated by T4DM Algorithm Design Agent*
