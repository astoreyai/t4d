# Storage Module

**Path**: `t4dm/storage/` | **Files**: 6 | **Lines**: ~3,500

Storage abstraction layer with Neo4j (graph), Qdrant (vector), resilience patterns, and saga transactions.

---

## Quick Start

```python
from ww.storage import Neo4jStore, QdrantStore, get_circuit_breaker

# Vector store
qdrant = QdrantStore(url="http://localhost:6333")
await qdrant.initialize()
await qdrant.add("episodes", ids, vectors, payloads)
results = await qdrant.search("episodes", query_vector, limit=10)

# Graph store
neo4j = Neo4jStore(uri="bolt://localhost:7687", user="neo4j", password="...")
await neo4j.initialize()
node_id = await neo4j.create_node("Episode", {"content": "..."})
await neo4j.create_relationship(source_id, target_id, "RELATES_TO")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │  MemorySaga  │
                    │  (Pre-built) │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   QdrantStore   │ │   Neo4jStore    │ │   MemorySaga    │
│   (Vectors)     │ │   (Graph)       │ │  (Coordination) │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ BGE-M3 (1024d)  │ │ Cypher queries  │ │ Compensation    │
│ Hybrid search   │ │ Pool metrics    │ │ Rollback        │
│ Session index   │ │ Injection prev  │ │ Atomicity       │
└────────┬────────┘ └────────┬────────┘ └─────────────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
         ┌─────────────────────────────────┐
         │      Circuit Breaker Layer      │
         ├─────────────────────────────────┤
         │ CLOSED ─→ OPEN ─→ HALF_OPEN     │
         │ (5 failures → 60s timeout)      │
         └─────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────────────────┐
         │   Graceful Degradation Layer    │
         ├─────────────────────────────────┤
         │ InMemoryFallback (10K cache)    │
         │ Pending ops queue (LRU)         │
         │ Drain on recovery               │
         └─────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────────────────┐
         │     ColdStorageManager          │
         │   (Filesystem/S3/PostgreSQL)    │
         └─────────────────────────────────┘
```

---

## File Structure

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| `neo4j_store.py` | 1,270 | Graph database | `Neo4jStore` |
| `qdrant_store.py` | 1,150 | Vector database | `QdrantStore` |
| `saga.py` | 515 | Cross-store transactions | `Saga`, `MemorySaga` |
| `resilience.py` | 1,123 | Fault tolerance | `CircuitBreaker`, `GracefulDegradation` |
| `archive.py` | 450 | Cold storage | `ColdStorageManager`, `FilesystemArchive` |

---

## Neo4j Store

### Security (P3-SEC-L2)

Cypher injection prevention via whitelist + regex:

```python
# Allowed labels
ALLOWED_NODE_LABELS = {"Episode", "Entity", "Procedure"}

# Allowed relationship types (17 types)
ALLOWED_RELATIONSHIP_TYPES = {
    "USES", "PRODUCES", "REQUIRES", "CAUSES", "PART_OF",
    "SIMILAR_TO", "IMPLEMENTS", "IMPROVES_ON", "CONSOLIDATED_INTO",
    "SOURCE_OF", "RELATES_TO", "HAS_CONTEXT", "DERIVED_FROM",
    "SUPERSEDES", "DEPENDS_ON", "TEMPORAL_BEFORE", "TEMPORAL_AFTER"
}
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `create_node(label, props)` | Create Episode/Entity/Procedure |
| `get_node(id, label)` | Retrieve by ID |
| `update_node(id, label, updates)` | Partial update |
| `delete_node(id, label)` | DETACH DELETE |
| `batch_create_nodes(label, props_list)` | Bulk insert |
| `create_relationship(src, tgt, type, props)` | Create edge |
| `get_relationships(node_id)` | Outgoing edges |
| `strengthen_relationship(src, tgt, type, decay)` | Increment weight |
| `batch_decay_relationships(node_id, factor, min)` | Apply decay |
| `find_path(src, tgt, max_depth, types)` | BFS path finding |
| `query(cypher, params)` | Raw Cypher |

### Connection Pool (P2-OPT-B3.3)

```python
stats = await neo4j.get_pool_stats()
# {acquisitions, failures, avg_time, max_time, active_sessions}
```

---

## Qdrant Store

### Collections

- `episodes` - Episodic memory embeddings
- `entities` - Semantic entity embeddings
- `procedures` - Skill embeddings

### Key Methods

| Method | Purpose |
|--------|---------|
| `add(collection, ids, vectors, payloads)` | Bulk insert |
| `search(collection, vector, limit, filter)` | k-NN search |
| `search_hybrid(coll, dense, sparse, limit)` | Dense + sparse |
| `get(collection, ids)` | Retrieve by IDs |
| `get_with_vectors(collection, ids)` | Include vectors |
| `batch_update_vectors(coll, ids, vectors)` | Update embeddings |
| `delete(collection, ids)` | Remove points |
| `update_payload(coll, ids, updates)` | Metadata update |
| `scroll(collection, limit, offset)` | Pagination |

### Session Isolation (P2-OPT-B2.1)

```python
# Indexed payload field for fast session filtering
await qdrant.search(
    "episodes",
    vector=query_vec,
    session_id="my-session",  # Uses payload index
    limit=10,
)
```

### Parallel Batching (QDRANT-001/002)

```python
await qdrant.add(
    "episodes",
    ids=ids,
    vectors=vectors,
    payloads=payloads,
    batch_size=100,
    max_concurrency=10,  # Semaphore limit
)
# Rollback on failure: only deletes successfully uploaded
```

---

## Saga Pattern

Cross-store atomicity via compensation:

```python
from ww.storage import Saga, MemorySaga

# Generic saga
saga = Saga("create-memory", timeout=60.0)
saga.add_step(
    name="create-vector",
    action=lambda: qdrant.add(...),
    compensate=lambda: qdrant.delete(...),
)
saga.add_step(
    name="create-node",
    action=lambda: neo4j.create_node(...),
    compensate=lambda: neo4j.delete_node(...),
)
result = await saga.execute()

# Pre-built memory sagas
memory_saga = MemorySaga(qdrant, neo4j)
result = await memory_saga.create_episode(
    episode_id, vector, payload, graph_props
)
result = await memory_saga.delete_memory(memory_id, "Episode")
```

### Saga States

```
PENDING → RUNNING → COMMITTED
                  ↓
              COMPENSATING → COMPENSATED
                           ↓
                         FAILED
```

### Compensation Order

LIFO (last step first) for proper rollback.

---

## Circuit Breaker

Fail fast on repeated failures:

```python
from ww.storage import CircuitBreaker, CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,      # Failures before opening
    success_threshold=2,      # Successes to close
    reset_timeout=60.0,       # Seconds before half-open
)

breaker = get_circuit_breaker("qdrant", config)

# Protected execution
result = await breaker.execute(qdrant.search, ...)

# Or as decorator
@breaker.protect
async def my_function():
    ...
```

### States

| State | Behavior |
|-------|----------|
| CLOSED | Normal operation |
| OPEN | Immediate fail (fast fail) |
| HALF_OPEN | Test with single request |

---

## Graceful Degradation

Fallback when primary unavailable:

```python
from ww.storage import GracefulDegradation

degradation = GracefulDegradation(breaker, fallback)

# Try primary, fall back to cache
result = await degradation.execute_with_fallback(
    primary_func=lambda: qdrant.search(...),
    fallback_key="query-hash",
    collection="episodes",
)

# Write with queueing on failure
await degradation.write_with_queue(
    write_func=lambda: qdrant.add(...),
    collection="episodes",
    key=episode_id,
    data=episode_data,
)

# Replay queued ops on recovery
await degradation.drain_pending_operations(replay_func)
```

### InMemoryFallback

- Max 10,000 entries (LRU eviction)
- Key-value only (no complex queries)
- Pending ops queue for replay

---

## Cold Storage (Archive)

Move old memories to cheaper storage:

```python
from ww.storage import ColdStorageManager, ArchiveConfig

config = ArchiveConfig(
    backend="filesystem",  # or "s3", "postgres"
    base_path="/var/lib/t4dm/archive",
    compression=True,  # gzip
    archive_retention_days=365*5,  # 5 years
)

manager = ColdStorageManager(config)

# Archive old memory
await manager.store(memory_id, data, metadata)

# Retrieve
data = await manager.retrieve(memory_id)
```

### Directory Structure

```
archive/
├── episodes/
│   └── ab/  # First 2 chars of ID
│       └── abc123.json.gz
├── entities/
│   └── ...
└── metadata/
    └── ab/
        └── abc123.meta.json
```

---

## Performance Optimizations

| Optimization | Location | Speedup |
|--------------|----------|---------|
| Session ID indexing | Qdrant | O(log n) vs O(n) |
| Hybrid prefetch | Qdrant | 1.5x configurable |
| Connection pooling | Neo4j | Metrics tracking |
| Parallel batching | Qdrant | 10 concurrent |
| LRU cache | Neo4j relations | 1000 entries |

---

## Error Handling

### Sanitized Errors

Removes sensitive data from exceptions:
- Connection URIs with credentials
- Hostnames and IP addresses
- Internal file paths
- API keys and tokens

```python
# Raises sanitized
raise DatabaseConnectionError("create_node")
# Message: "Database connection failed: create_node"
# NOT: "bolt://user:pass@host:7687 connection refused"
```

### Timeout Errors

```python
raise DatabaseTimeoutError("search", timeout=30)
```

---

## Configuration

```python
# Neo4j
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "..."
neo4j_pool_size = 50
neo4j_timeout = 30

# Qdrant
qdrant_url = "http://localhost:6333"
qdrant_api_key = None
qdrant_timeout = 30
qdrant_hybrid_prefetch_multiplier = 1.5

# Circuit Breaker
failure_threshold = 5
reset_timeout = 60
```

---

## Dependencies

**External**:
- `neo4j` - Neo4j Python driver
- `qdrant-client` - Qdrant Python client
- `asyncio` - Async operations

**Internal**:
- `ww.core.config` - Settings
- `ww.observability` - Tracing
