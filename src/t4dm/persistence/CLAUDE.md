# Persistence
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/persistence/`

## What
Crash-safe durability layer providing Write-Ahead Logging (WAL), periodic checkpointing, cold/warm start recovery, and graceful shutdown handling.

## How
- **WAL** (`wal.py`): Append-only log of `WALEntry` records keyed by `WALOperation` type. Detects corruption via checksums. All memory mutations are logged before execution.
- **Checkpoint** (`checkpoint.py`): `CheckpointManager` periodically snapshots system state to disk via `CheckpointConfig`-controlled intervals. Snapshots include buffer contents, learning weights, and consolidation state.
- **Recovery** (`recovery.py`): `RecoveryManager` implements warm start (load checkpoint + replay WAL) and cold start (initialize defaults). Returns `RecoveryResult` with mode and stats.
- **Shutdown** (`shutdown.py`): `ShutdownManager` handles SIGTERM/SIGINT -- drains in-flight ops (30s timeout), forces a checkpoint, flushes WAL, closes connections.
- **Manager** (`manager.py`): `PersistenceManager` is the unified facade coordinating WAL, checkpoints, recovery, and shutdown. Accessed via `get_persistence()` singleton.

## Why
The dual-store architecture (Neo4j + Qdrant) has no built-in distributed transaction support. WAL ensures operations can be replayed after crashes, and checkpoints minimize replay time on restart.

## Key Files
| File | Purpose |
|------|---------|
| `wal.py` | Write-ahead log with corruption detection |
| `checkpoint.py` | Periodic state snapshots |
| `recovery.py` | Cold/warm start protocols |
| `shutdown.py` | Graceful shutdown with signal handling |
| `manager.py` | Unified persistence facade |

## Data Flow
```
STARTUP --> RecoveryManager --> checkpoint exists? --> warm/cold start --> serve
RUNTIME --> WAL.append(op) --> execute op --> periodic checkpoint
SHUTDOWN --> drain ops --> force checkpoint --> flush WAL --> close stores
```

## Integration Points
- **Storage**: WAL records Neo4j/Qdrant mutations for replay
- **Temporal**: Checkpoint captures temporal dynamics state
- **Observability**: Persistence metrics tracked via visualization module
