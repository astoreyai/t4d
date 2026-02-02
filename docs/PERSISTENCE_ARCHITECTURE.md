# World Weaver Persistence Architecture

## Overview

The persistence layer ensures World Weaver survives crashes and restarts without data loss. It implements the Write-Ahead Logging (WAL) pattern used by production databases.

## Components

### 1. Write-Ahead Log (WAL)
**Location**: `src/t4dm/persistence/wal.py`

Logs all state-changing operations BEFORE they're applied:

```python
# Log before state change
lsn = await wal.append(WALOperation.BUFFER_ADD, {
    "memory_id": memory.id,
    "content": memory.content,
})

# Then apply to in-memory state
buffer.add(memory)
```

**Features**:
- Binary format with CRC32 checksums
- Segment rotation (64MB default)
- Recovery from corruption
- LSN (Log Sequence Number) tracking

### 2. Checkpoint Manager
**Location**: `src/t4dm/persistence/checkpoint.py`

Periodic snapshots of all in-memory state:

```python
checkpoint = Checkpoint(
    lsn=current_lsn,
    gate_state=gate.state_dict(),
    scorer_state=scorer.state_dict(),
    buffer_state=buffer.serialize(),
    neuromod_state=neuromod.state_dict(),
)
```

**Features**:
- Atomic writes (temp file + rename)
- Gzip compression
- SHA-256 verification
- Automatic cleanup of old checkpoints

### 3. Recovery Manager
**Location**: `src/t4dm/persistence/recovery.py`

Handles cold and warm starts:

```
COLD START (no checkpoint):
  └─► Initialize defaults
  └─► System learns from scratch

WARM START (checkpoint exists):
  └─► Load checkpoint
  └─► Replay WAL entries
  └─► Resume from pre-crash state
```

### 4. Shutdown Manager
**Location**: `src/t4dm/persistence/shutdown.py`

Graceful shutdown with data preservation:

```
SIGTERM received
  └─► Stop accepting requests
  └─► Drain in-flight operations (30s timeout)
  └─► Create final checkpoint
  └─► Flush WAL
  └─► Close connections
  └─► Exit(0)
```

**Signal Safety**:
- No async in signal handlers
- No logging in signal handlers
- Thread-safe shutdown flag

### 5. Persistence Manager
**Location**: `src/t4dm/persistence/manager.py`

Unified interface that coordinates all components:

```python
from ww.persistence import PersistenceManager, PersistenceConfig

persistence = PersistenceManager(PersistenceConfig(
    data_directory=Path("/var/lib/ww"),
))

# Register components
persistence.register_component("buffer", buffer_manager)
persistence.register_component("gate", learned_gate)

# Start (performs recovery)
result = await persistence.start()

# Log operations
await persistence.log(WALOperation.BUFFER_ADD, {...})

# Shutdown gracefully
await persistence.shutdown()
```

## Data Layout

```
/var/lib/world-weaver/
├── wal/
│   ├── segment_00000001.wal
│   ├── segment_00000002.wal
│   └── segment_00000003.wal
├── checkpoints/
│   ├── checkpoint_0000000000012345.bin.gz
│   ├── checkpoint_0000000000012500.bin.gz
│   └── checkpoint_0000000000012700.bin.gz
└── recovery.log
```

## API Endpoints

### REST API
```
GET  /api/v1/persistence/status     - System status
GET  /api/v1/persistence/checkpoints - List checkpoints
POST /api/v1/persistence/checkpoint  - Create checkpoint
GET  /api/v1/persistence/wal        - WAL status
POST /api/v1/persistence/wal/truncate - Truncate old WAL
GET  /api/v1/persistence/health     - Health check
```

### WebSocket Channels
```
ws://localhost:8080/ws/events    - All events
ws://localhost:8080/ws/memory    - Memory events
ws://localhost:8080/ws/learning  - Learning events
ws://localhost:8080/ws/health    - Health metrics
```

## Event Types

| Event | Channel | Description |
|-------|---------|-------------|
| `system.checkpoint` | events | Checkpoint created |
| `memory.added` | memory | Memory added to buffer |
| `memory.promoted` | memory | Memory promoted to storage |
| `learning.gate_updated` | learning | Gate weights updated |
| `neuromod.dopamine_rpe` | learning | Dopamine RPE signal |
| `health.update` | health | Health metrics update |
| `health.warning` | health | Health warning |

## Recovery Guarantees

| Scenario | Data Loss |
|----------|-----------|
| Clean shutdown | None |
| SIGKILL | Since last fsync (~ms) |
| Power loss | Since last fsync (~ms) |
| Disk corruption | Roll back to last valid checkpoint |

## Configuration

```python
PersistenceConfig(
    data_directory=Path("/var/lib/ww"),

    # WAL settings
    wal_sync_mode="fsync",  # "fsync", "fdatasync", "none"
    wal_segment_max_size=64 * 1024 * 1024,  # 64MB
    wal_max_segments=100,

    # Checkpoint settings
    checkpoint_interval_seconds=300.0,  # 5 minutes
    checkpoint_operation_threshold=1000,
    checkpoint_max_count=5,
    checkpoint_compression=True,

    # Shutdown settings
    drain_timeout_seconds=30.0,
    checkpoint_timeout_seconds=60.0,
)
```

## Deployment

### Systemd
```ini
[Service]
ExecStart=/opt/t4dm/venv/bin/python -m ww.mcp.persistent_server
TimeoutStopSec=60
Restart=always
```

### Docker
```yaml
services:
  world-weaver:
    restart: always
    stop_grace_period: 60s
    volumes:
      - ww-data:/var/lib/world-weaver
```

## Testing

```bash
# Run persistence tests
pytest tests/persistence/ -v

# Test WAL recovery
pytest tests/persistence/test_wal.py::TestWriteAheadLog::test_recovery_after_close

# Test checkpoint roundtrip
pytest tests/persistence/test_checkpoint.py::TestCheckpoint::test_serialize_deserialize_roundtrip
```

## Monitoring

### Key Metrics
- `ww_wal_lsn` - Current WAL LSN
- `ww_checkpoint_lsn` - Last checkpoint LSN
- `ww_checkpoint_age_seconds` - Time since last checkpoint
- `ww_operations_since_checkpoint` - Uncommitted operations

### Alerts
- Checkpoint age > 10 minutes: Warning
- Operations since checkpoint > 10000: Warning
- Cold start detected: Warning
