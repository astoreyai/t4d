# Persistence Module

**Path**: `ww/persistence/` | **Files**: 6 | **Lines**: ~4,500

Crash-safe durability with Write-Ahead Logging, checkpointing, recovery, and graceful shutdown.

---

## Quick Start

```python
from ww.persistence import (
    PersistenceManager, PersistenceConfig,
    WALOperation, get_persistence,
)

# Create manager
config = PersistenceConfig(data_directory=Path("/var/lib/ww/data"))
persistence = PersistenceManager(config)

# Register components for checkpointing
persistence.register_component("buffer", buffer_manager)
persistence.register_component("scorer", scorer)

# Register WAL replay handlers
persistence.register_replay_handler(
    WALOperation.BUFFER_ADD,
    replay_buffer_add,
)

# Start (performs recovery)
result = await persistence.start()
print(f"Mode: {result.mode}, WAL replayed: {result.wal_entries_replayed}")

# Log operations (before applying to state)
lsn = await persistence.log(WALOperation.BUFFER_ADD, {"id": "mem-1", ...})

# Graceful shutdown
await persistence.shutdown()
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Code                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PersistenceManager (Orchestrator)               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │     WAL     │  │  Checkpoint  │  │    Recovery       │  │
│  │             │  │   Manager    │  │    Manager        │  │
│  │ • append()  │  │ • create()   │  │ • cold/warm start │  │
│  │ • fsync()   │  │ • restore()  │  │ • replay handlers │  │
│  │ • segment   │  │ • compress   │  │ • validators      │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
│                          │                                   │
│              ┌───────────┴──────────┐                       │
│              │   ShutdownManager    │                       │
│              │ • Signal handling    │                       │
│              │ • Drain + checkpoint │                       │
│              │ • Cleanup callbacks  │                       │
│              └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────┐
              │   Durable Storage (Disk)     │
              ├──────────────────────────────┤
              │ data_directory/              │
              │  ├── wal/                    │
              │  │   ├── segment_00000001.wal│
              │  │   └── segment_00000002.wal│
              │  └── checkpoints/            │
              │      └── checkpoint_NNNN.bin │
              └──────────────────────────────┘
```

---

## File Structure

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| `manager.py` | 800+ | Unified orchestrator | `PersistenceManager`, `PersistenceConfig` |
| `wal.py` | 900+ | Write-Ahead Log | `WriteAheadLog`, `WALEntry`, `WALOperation` |
| `checkpoint.py` | 700+ | State snapshots | `CheckpointManager`, `Checkpoint` |
| `recovery.py` | 600+ | Cold/warm start | `RecoveryManager`, `RecoveryResult` |
| `shutdown.py` | 500+ | Graceful shutdown | `ShutdownManager`, `ShutdownPhase` |

---

## Write-Ahead Log (WAL)

### Guarantee

Operations logged **before** applied to in-memory state. On crash, replay recovers state.

### WAL Operations

```python
class WALOperation(IntEnum):
    # Buffer operations
    BUFFER_ADD = 1
    BUFFER_REMOVE = 2
    BUFFER_PROMOTE = 3

    # Gate learning
    GATE_WEIGHT_UPDATE = 10
    GATE_BIAS_UPDATE = 11

    # Neuromodulator state
    DOPAMINE_EXPECTATION = 20
    DOPAMINE_RPE = 21
    SEROTONIN_MOOD = 22

    # Eligibility traces
    TRACE_CREATE = 30
    TRACE_UPDATE = 31

    # Consolidation
    CONSOLIDATION_START = 40
    CONSOLIDATION_COMMIT = 41

    # Checkpointing
    CHECKPOINT_START = 50
    CHECKPOINT_COMPLETE = 51
```

### Binary Format

```
[4 bytes] Magic number (0xWW4C)
[8 bytes] LSN (Log Sequence Number)
[8 bytes] Timestamp (nanoseconds)
[2 bytes] Operation type
[4 bytes] Payload length
[N bytes] Payload (msgpack/JSON)
[4 bytes] CRC32 checksum
```

### Usage

```python
from ww.persistence import WriteAheadLog, WALConfig

wal = WriteAheadLog(WALConfig(
    directory=Path("/var/lib/ww/wal"),
    segment_max_size=64*1024*1024,  # 64MB
    sync_mode="fsync",
))

await wal.open()
lsn = await wal.append(WALOperation.BUFFER_ADD, {"id": "mem-1"})
await wal.sync()  # Force disk sync
await wal.close()
```

### Segment Management

- Files: `segment_NNNNNNNN.wal` (8-digit)
- Max 64MB per segment
- Auto-rotation on size limit
- Truncation after checkpoint

---

## Checkpointing

### Checkpoint Contents

```python
@dataclass
class Checkpoint:
    lsn: int              # WAL LSN at checkpoint
    timestamp: float      # Unix timestamp
    version: int = 1      # Schema version

    # Component states
    gate_state: dict | None
    scorer_state: dict | None
    buffer_state: dict | None
    traces_state: dict | None
    neuromod_state: dict | None
    custom_states: dict   # Extensible

    checksum: str         # SHA256
```

### Checkpointable Protocol

Components must implement:

```python
class CheckpointableComponent(Protocol):
    def get_checkpoint_state(self) -> dict:
        """Capture state for checkpoint."""
        ...

    def restore_from_checkpoint(self, state: dict) -> None:
        """Restore state from checkpoint."""
        ...
```

### Automatic Checkpointing

```python
config = CheckpointConfig(
    directory=Path("/var/lib/ww/checkpoints"),
    interval_seconds=300.0,      # Every 5 minutes
    operation_threshold=1000,    # Or every 1000 ops
    max_checkpoints=5,           # Keep last 5
    compression=True,            # gzip
)

manager = CheckpointManager(config)
manager.register_component("buffer", buffer)
await manager.start()  # Background loop
```

### Atomic Write Pattern

1. Write to `checkpoint_NNNN.tmp`
2. fsync
3. Rename to `checkpoint_NNNN.bin.gz`
4. fsync directory

---

## Recovery

### Recovery Modes

| Mode | Description |
|------|-------------|
| COLD_START | No checkpoint, initialize fresh |
| WARM_START | Checkpoint found, restore + replay |
| FORCED_COLD | User-requested fresh start |

### Recovery Flow

```
Startup
   │
   ├─ Checkpoint exists?
   │  ├─ YES → WARM START
   │  │   ├─→ Restore checkpoint state
   │  │   ├─→ Replay WAL after checkpoint LSN
   │  │   └─→ Validate consistency
   │  │
   │  └─ NO → COLD START
   │      ├─→ Run cold start initializers
   │      ├─→ Load from storage (if configured)
   │      └─→ Rebuild indexes
   │
   └─→ Return RecoveryResult
```

### Replay Handlers

```python
async def replay_buffer_add(entry: WALEntry) -> None:
    memory = reconstruct_memory(entry.payload)
    buffer.add(memory)

persistence.register_replay_handler(
    WALOperation.BUFFER_ADD,
    replay_buffer_add,
)
```

### Recovery Result

```python
@dataclass
class RecoveryResult:
    mode: RecoveryMode
    success: bool
    checkpoint_lsn: int
    wal_entries_replayed: int
    components_restored: dict[str, bool]
    errors: list[str]
    duration_seconds: float
```

---

## Graceful Shutdown

### Signal Handling (MCP-CRITICAL fixes)

- **Minimal signal handler**: Only sets atomic flag
- **No logging in handler**: Prevents deadlock
- **No async in handler**: Prevents crash
- **Second signal = force exit**

### Shutdown Phases

```
SIGTERM/SIGINT
      │
      ▼
[DRAINING] Stop new requests, wait for in-flight (30s)
      │
      ▼
[CHECKPOINTING] Create final checkpoint (60s)
      │
      ▼
Flush & close WAL
      │
      ▼
[CLEANING] Run cleanup callbacks in reverse priority (30s)
      │
      ▼
[CLOSED] Exit(0)
```

### Usage

```python
from ww.persistence import ShutdownManager, OperationContext

shutdown = ShutdownManager(config)
shutdown.set_checkpoint_function(create_checkpoint)
shutdown.register_cleanup(close_connections, priority=100)
shutdown.install_handlers()

# Track in-flight operations
async with OperationContext(shutdown):
    await process_request()

# Or with decorator
@track_operation(shutdown)
async def handle_request():
    ...
```

---

## PersistenceManager

Unified interface combining all components:

```python
from ww.persistence import PersistenceManager, PersistenceConfig

config = PersistenceConfig(
    data_directory=Path("/var/lib/ww/data"),

    # WAL
    wal_sync_mode="fsync",
    wal_segment_max_size=64*1024*1024,

    # Checkpoint
    checkpoint_interval_seconds=300,
    checkpoint_compression=True,

    # Shutdown
    drain_timeout_seconds=30,
)

persistence = PersistenceManager(config)

# Register components
persistence.register_component("buffer", buffer)
persistence.register_replay_handler(WALOperation.BUFFER_ADD, handler)
persistence.register_cleanup(cleanup_fn, priority=50)

# Lifecycle
result = await persistence.start()  # Recovery
lsn = await persistence.log(op, payload)  # WAL append
await persistence.create_checkpoint()  # Manual checkpoint
await persistence.shutdown()  # Graceful stop
```

### Status Query

```python
status = persistence.get_status()
# {
#   "current_lsn": 12345,
#   "checkpoint_lsn": 12000,
#   "is_started": True,
#   "should_shutdown": False,
# }
```

---

## Configuration Best Practices

### Development

```python
config = PersistenceConfig(
    data_directory=Path("/tmp/ww_dev"),
    checkpoint_interval_seconds=60,
    checkpoint_max_count=3,
)
```

### Production

```python
config = PersistenceConfig(
    data_directory=Path("/var/lib/ww/data"),
    wal_sync_mode="fsync",
    wal_segment_max_size=256*1024*1024,
    checkpoint_interval_seconds=3600,
    checkpoint_compression=True,
    drain_timeout_seconds=120,
)
```

### Performance (durability trade-off)

```python
config = PersistenceConfig(
    wal_sync_mode="fdatasync",  # Slightly faster
    checkpoint_operation_threshold=5000,
)
```

---

## Data Flow

### Normal Operation

```
Application
    │
    ├─→ persistence.log(BUFFER_ADD, {...})
    │   │
    │   ├─→ WAL.append() [BEFORE state change]
    │   │   ├─→ Serialize
    │   │   ├─→ Write to segment
    │   │   ├─→ fsync
    │   │   └─→ Return LSN
    │   │
    │   └─→ CheckpointManager.record_operation()
    │
    ├─→ buffer.add(memory) [AFTER WAL]
    │
    └─→ If crash: replay restores state
```

### Checkpoint Trigger

```
Every 1 second:
   ├─→ Check time elapsed
   ├─→ Check operation count
   │
   ├─ If threshold exceeded:
   │  └─→ create_checkpoint()
   │      ├─→ Collect component states
   │      ├─→ Serialize + compress
   │      ├─→ Atomic write
   │      ├─→ Clean up old checkpoints
   │      └─→ WAL.mark_checkpoint()
```

---

## Dependencies

**Internal**:
- Uses `msgpack` for WAL payload serialization
- Uses `pickle` + `gzip` for checkpoints

**External**:
- `asyncio` - Async operations
- Standard library: `hashlib`, `struct`, `signal`
