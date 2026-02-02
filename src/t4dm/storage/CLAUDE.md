# Storage
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/storage/`

## What
Storage backend providing T4DX embedded spatiotemporal engine with LSM-based compaction, HNSW vector indexing, and CSR graph structure. Memory consolidation maps directly to LSM compaction phases.

## How
- **T4DX Engine** (`t4dx/`): Embedded storage engine combining vector similarity search, graph traversal, and temporal indexing in a single process. LSM segments provide the physical storage layer.
- **Resilience** (`resilience.py`): `CircuitBreaker` with states (closed/open/half-open). Prevents cascading failures. Configurable thresholds via `CircuitBreakerConfig`.
- **Archive** (`archive.py`): Cold storage for aged-out memories.

## Why
Co-locating vectors, edges, metadata, and temporal indices in a single embedded engine eliminates network hops and enables LSM compaction to serve as biological memory consolidation (NREM merge, REM clustering, PRUNE garbage collection).

## Key Files
| File | Purpose |
|------|---------|
| `t4dx/` | Embedded spatiotemporal storage engine |
| `resilience.py` | Circuit breaker for fault tolerance |
| `archive.py` | Cold storage for aged memories |

## Data Flow
```
Memory Store Request --> T4DX Engine --> MemTable (working memory)
                                    --> WAL (durability)
                                    --> Flush to segments (consolidation)
Memory Recall --> T4DX HNSW search --> CSR graph enrichment --> results
```

## Integration Points
- **Persistence**: WAL logs all storage mutations for crash recovery
- **Bridges**: `t4dm.bridges` wraps storage with higher-level memory semantics
- **Observability**: Circuit breaker state feeds health checks
- **Consolidation**: LSM compaction phases = sleep-phase consolidation
