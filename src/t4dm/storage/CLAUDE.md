# Storage
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/storage/`

## What
Dual-store backend providing Neo4j (graph relationships) and Qdrant (vector similarity) with saga-pattern cross-store transactions and circuit breaker resilience.

## How
- **Neo4j Store** (`neo4j_store.py`): Graph store for entities, relationships, and episode metadata. Cypher queries for traversal and semantic graph operations.
- **Qdrant Store** (`qdrant_store.py`): Vector store for embedding similarity search. Stores episode/entity/skill vectors with metadata payloads.
- **Saga** (`saga.py`): Compensation-based atomicity across Neo4j and Qdrant. Each saga step has a forward action and compensating rollback. `CompensationError` signals manual reconciliation needed.
- **Resilience** (`resilience.py`): `CircuitBreaker` per store with states (closed/open/half-open). Prevents cascading failures when a backend is down. Configurable thresholds via `CircuitBreakerConfig`.
- **Archive** (`archive.py`): Cold storage for aged-out memories.

## Why
Graph + vector is the natural dual for memory: Neo4j captures relational structure (entity-to-entity, episode sequences), Qdrant enables fast semantic similarity. The saga pattern provides eventual consistency without distributed transactions.

## Key Files
| File | Purpose |
|------|---------|
| `neo4j_store.py` | Graph storage for relationships and metadata |
| `qdrant_store.py` | Vector storage for embedding search |
| `saga.py` | Cross-store transaction coordination |
| `resilience.py` | Circuit breaker for backend fault tolerance |
| `archive.py` | Cold storage for aged memories |

## Data Flow
```
Memory Store Request --> Saga --> Step 1: Qdrant.upsert(vector)
                              --> Step 2: Neo4j.create(node + edges)
                              --> On failure: compensate (rollback step 1)
Memory Recall --> Qdrant.search(vector) --> Neo4j.enrich(relationships)
```

## Integration Points
- **Persistence**: WAL logs all storage mutations for crash recovery
- **Bridges**: `ww.bridges` wraps storage with higher-level memory semantics
- **Observability**: Circuit breaker state feeds health checks
