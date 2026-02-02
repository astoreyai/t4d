# Persistence Layer

T4DM implements durable persistence with write-ahead logging, checkpointing, and recovery mechanisms.

## Overview

```mermaid
graph TB
    subgraph Application["Application Layer"]
        API[Memory API]
        CACHE[Write Cache]
    end

    subgraph Persistence["Persistence Layer"]
        WAL[Write-Ahead Log]
        CKPT[Checkpoint Manager]
        RECOVERY[Recovery Engine]
    end

    subgraph Storage["Durable Storage"]
        WAL_FILE[(WAL Files)]
        CKPT_FILE[(Checkpoint Files)]
        NEO4J[(Neo4j)]
        QDRANT[(Qdrant)]
    end

    API --> CACHE
    CACHE --> WAL
    WAL --> WAL_FILE
    WAL --> NEO4J
    WAL --> QDRANT

    CKPT --> CKPT_FILE
    CKPT --> NEO4J
    CKPT --> QDRANT

    RECOVERY --> WAL_FILE
    RECOVERY --> CKPT_FILE

    style WAL fill:#e8f5e9
    style CKPT fill:#e3f2fd
    style RECOVERY fill:#fff3e0
```

## Write-Ahead Log (WAL)

All mutations are logged before being applied to ensure durability.

### WAL Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant WAL as WAL Writer
    participant Disk as WAL File
    participant Neo4j
    participant Qdrant

    Client->>API: store_memory(episode)

    rect rgb(232, 245, 233)
        Note over WAL: Step 1: Log First
        API->>WAL: append(operation)
        WAL->>Disk: write + fsync
        Disk-->>WAL: durable
        WAL-->>API: log_sequence_number
    end

    rect rgb(227, 242, 253)
        Note over WAL: Step 2: Apply
        API->>Neo4j: create_node()
        API->>Qdrant: upsert_vector()
        Neo4j-->>API: success
        Qdrant-->>API: success
    end

    rect rgb(232, 245, 233)
        Note over WAL: Step 3: Confirm
        API->>WAL: mark_applied(lsn)
    end

    API-->>Client: success
```

### WAL Entry Structure

```mermaid
graph LR
    subgraph WALEntry["WAL Entry"]
        LSN[LSN: 12345]
        TS[Timestamp]
        OP[Operation Type]
        DATA[Serialized Data]
        CRC[CRC32 Checksum]
    end

    subgraph Operations["Operation Types"]
        STORE[STORE_EPISODE]
        UPDATE[UPDATE_EPISODE]
        DELETE[DELETE_EPISODE]
        ENTITY[CREATE_ENTITY]
        REL[CREATE_RELATIONSHIP]
    end

    OP --> STORE
    OP --> UPDATE
    OP --> DELETE
    OP --> ENTITY
    OP --> REL

    style LSN fill:#e8f5e9
    style CRC fill:#ffebee
```

### WAL Segment Management

```mermaid
graph TB
    subgraph Active["Active Segment"]
        SEG_N[wal_000042.log<br/>Current writes]
    end

    subgraph Archived["Archived Segments"]
        SEG_1[wal_000040.log]
        SEG_2[wal_000041.log]
    end

    subgraph Cleanup["Cleanup"]
        CKPT_LSN[Checkpoint LSN: 40000]
        PRUNE[Prune old segments]
    end

    SEG_N -->|segment full<br/>16MB| SEG_2
    SEG_2 -->|rotated| SEG_1

    CKPT_LSN --> PRUNE
    PRUNE -->|delete if LSN < checkpoint| SEG_1

    style SEG_N fill:#e8f5e9
    style SEG_1 fill:#ffebee
```

### WAL Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segment_size` | 16 MB | Max segment file size |
| `sync_mode` | `fsync` | Durability mode |
| `buffer_size` | 4 KB | Write buffer size |
| `retention` | 7 days | Archive retention |
| `compression` | `lz4` | Segment compression |

## Checkpointing

Periodic snapshots reduce recovery time by establishing known-good states.

### Checkpoint Lifecycle

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE --> PREPARING: trigger_checkpoint()
    IDLE --> IDLE: wait interval

    PREPARING --> WRITING: acquire_lock
    PREPARING --> IDLE: lock_failed

    WRITING --> SYNCING: write_complete
    WRITING --> FAILED: write_error

    SYNCING --> VERIFYING: sync_complete
    SYNCING --> FAILED: sync_error

    VERIFYING --> COMPLETE: checksum_valid
    VERIFYING --> FAILED: checksum_invalid

    COMPLETE --> CLEANING: success
    CLEANING --> IDLE: old_checkpoints_removed

    FAILED --> IDLE: retry_later

    note right of WRITING
        Write Neo4j snapshot
        Write Qdrant snapshot
        Write metadata
    end note

    note right of VERIFYING
        Verify checksums
        Validate consistency
    end note
```

### Checkpoint Process

```mermaid
sequenceDiagram
    participant Timer as Checkpoint Timer
    participant CKPT as Checkpoint Manager
    participant Neo4j
    participant Qdrant
    participant Disk as Checkpoint Files

    Note over Timer: Every 15 minutes
    Timer->>CKPT: trigger_checkpoint()

    rect rgb(232, 245, 233)
        Note over CKPT: Phase 1: Prepare
        CKPT->>CKPT: pause_writes()
        CKPT->>CKPT: get_current_lsn()
    end

    rect rgb(227, 242, 253)
        Note over CKPT: Phase 2: Snapshot
        par Parallel snapshots
            CKPT->>Neo4j: create_snapshot()
            CKPT->>Qdrant: create_snapshot()
        end
        Neo4j-->>CKPT: snapshot_path
        Qdrant-->>CKPT: snapshot_path
    end

    rect rgb(255, 243, 224)
        Note over CKPT: Phase 3: Write Metadata
        CKPT->>Disk: write_manifest(lsn, paths, checksums)
        Disk-->>CKPT: success
    end

    rect rgb(232, 245, 233)
        Note over CKPT: Phase 4: Cleanup
        CKPT->>CKPT: resume_writes()
        CKPT->>Disk: prune_old_checkpoints()
        CKPT->>Disk: prune_old_wal_segments()
    end

    CKPT-->>Timer: checkpoint_complete
```

### Checkpoint Structure

```mermaid
graph TB
    subgraph Checkpoint["Checkpoint Directory"]
        MANIFEST[manifest.json]
        NEO4J_SNAP[neo4j_snapshot/]
        QDRANT_SNAP[qdrant_snapshots/]
    end

    subgraph Manifest["Manifest Contents"]
        M_LSN[checkpoint_lsn: 50000]
        M_TS[timestamp: 2026-01-03T12:00:00Z]
        M_NEO[neo4j_checksum: abc123...]
        M_QD[qdrant_checksum: def456...]
        M_VER[version: 0.4.0]
    end

    MANIFEST --> M_LSN
    MANIFEST --> M_TS
    MANIFEST --> M_NEO
    MANIFEST --> M_QD
    MANIFEST --> M_VER

    style MANIFEST fill:#e8f5e9
```

### Checkpoint Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interval` | 15 min | Checkpoint frequency |
| `retention` | 3 | Checkpoints to keep |
| `timeout` | 5 min | Max checkpoint duration |
| `parallel` | true | Parallel backend snapshots |

## Recovery

System recovery from WAL and checkpoints after crash or restart.

### Recovery Flow

```mermaid
graph TB
    subgraph Startup["System Startup"]
        INIT[Initialize]
        DETECT[Detect Recovery Need]
    end

    subgraph Recovery["Recovery Process"]
        FIND_CKPT[Find Latest Checkpoint]
        RESTORE[Restore Checkpoint]
        FIND_WAL[Find WAL After Checkpoint]
        REPLAY[Replay WAL Entries]
        VERIFY[Verify Consistency]
    end

    subgraph Complete["Ready"]
        READY[System Ready]
    end

    INIT --> DETECT
    DETECT -->|clean shutdown| READY
    DETECT -->|unclean shutdown| FIND_CKPT

    FIND_CKPT --> RESTORE
    RESTORE --> FIND_WAL
    FIND_WAL --> REPLAY
    REPLAY --> VERIFY
    VERIFY -->|consistent| READY
    VERIFY -->|inconsistent| FIND_CKPT

    style READY fill:#e8f5e9
    style REPLAY fill:#fff3e0
```

### Recovery Sequence

```mermaid
sequenceDiagram
    participant Main as Main Process
    participant Recovery as Recovery Engine
    participant Disk as Storage
    participant Neo4j
    participant Qdrant

    Main->>Recovery: start_recovery()

    rect rgb(227, 242, 253)
        Note over Recovery: Phase 1: Load Checkpoint
        Recovery->>Disk: read_latest_checkpoint()
        Disk-->>Recovery: checkpoint_20260103_1200
        Recovery->>Recovery: validate_checksums()

        par Restore backends
            Recovery->>Neo4j: restore_snapshot()
            Recovery->>Qdrant: restore_snapshot()
        end

        Neo4j-->>Recovery: restored to LSN 50000
        Qdrant-->>Recovery: restored to LSN 50000
    end

    rect rgb(255, 243, 224)
        Note over Recovery: Phase 2: Replay WAL
        Recovery->>Disk: find_wal_segments(after: 50000)
        Disk-->>Recovery: [wal_000042.log, wal_000043.log]

        loop For each WAL entry
            Recovery->>Recovery: read_entry()
            Recovery->>Recovery: validate_crc()
            Recovery->>Neo4j: apply_operation()
            Recovery->>Qdrant: apply_operation()
        end
    end

    rect rgb(232, 245, 233)
        Note over Recovery: Phase 3: Verify
        Recovery->>Neo4j: count_nodes()
        Recovery->>Qdrant: count_vectors()
        Recovery->>Recovery: compare_counts()
    end

    Recovery-->>Main: recovery_complete(lsn: 52340)
```

### Recovery Modes

```mermaid
graph TB
    subgraph Modes["Recovery Modes"]
        FAST[Fast Recovery<br/>Checkpoint + WAL]
        FULL[Full Recovery<br/>Complete rebuild]
        POINT[Point-in-Time<br/>Recover to LSN]
    end

    subgraph Triggers["Triggered By"]
        CRASH[Crash/Restart]
        CORRUPT[Corruption Detected]
        USER[User Request]
    end

    CRASH --> FAST
    CORRUPT --> FULL
    USER --> POINT

    style FAST fill:#e8f5e9
    style FULL fill:#ffebee
    style POINT fill:#fff3e0
```

### Recovery Statistics

| Metric | Description |
|--------|-------------|
| `recovery_duration_seconds` | Total recovery time |
| `checkpoint_restore_seconds` | Checkpoint load time |
| `wal_entries_replayed` | WAL entries processed |
| `wal_replay_rate` | Entries per second |
| `recovery_lsn_gap` | LSN distance recovered |

## Graceful Shutdown

Orderly shutdown ensures no data loss.

### Shutdown Sequence

```mermaid
sequenceDiagram
    participant Signal as SIGTERM
    participant Main as Main Process
    participant API as API Server
    participant WAL as WAL Writer
    participant CKPT as Checkpoint Manager
    participant Backends as Neo4j/Qdrant

    Signal->>Main: shutdown_signal()

    rect rgb(255, 243, 224)
        Note over Main: Phase 1: Stop Accepting
        Main->>API: stop_accepting_requests()
        API-->>Main: no new requests
    end

    rect rgb(227, 242, 253)
        Note over Main: Phase 2: Drain
        Main->>API: drain_pending_requests()
        API-->>Main: all requests complete
    end

    rect rgb(232, 245, 233)
        Note over Main: Phase 3: Flush
        Main->>WAL: flush_buffer()
        WAL-->>Main: all entries synced
    end

    rect rgb(227, 242, 253)
        Note over Main: Phase 4: Checkpoint
        Main->>CKPT: create_shutdown_checkpoint()
        CKPT->>Backends: snapshot()
        Backends-->>CKPT: success
        CKPT-->>Main: checkpoint_complete
    end

    rect rgb(232, 245, 233)
        Note over Main: Phase 5: Close
        Main->>WAL: close()
        Main->>Backends: disconnect()
    end

    Main->>Main: write_clean_shutdown_marker()
    Main->>Signal: exit(0)
```

### Shutdown States

```mermaid
stateDiagram-v2
    [*] --> RUNNING

    RUNNING --> DRAINING: shutdown_signal
    DRAINING --> FLUSHING: requests_drained
    FLUSHING --> CHECKPOINTING: wal_flushed
    CHECKPOINTING --> CLOSING: checkpoint_done
    CLOSING --> STOPPED: connections_closed

    STOPPED --> [*]

    state DRAINING {
        [*] --> WaitingRequests
        WaitingRequests: Max 30s timeout
    }

    state CHECKPOINTING {
        [*] --> CreatingSnapshot
        CreatingSnapshot: Final checkpoint
    }
```

## Data Integrity

Multiple layers ensure data integrity.

### Integrity Checks

```mermaid
graph TB
    subgraph Write["Write Path"]
        W_CRC[CRC32 on WAL entries]
        W_FSYNC[fsync after write]
        W_VERIFY[Read-after-write verify]
    end

    subgraph Checkpoint["Checkpoint"]
        C_SHA[SHA256 on snapshots]
        C_MANIFEST[Manifest checksums]
        C_ATOMIC[Atomic rename]
    end

    subgraph Recovery["Recovery"]
        R_VALIDATE[Validate all CRCs]
        R_COMPARE[Cross-backend comparison]
        R_REPAIR[Auto-repair if possible]
    end

    subgraph Runtime["Runtime"]
        RT_AUDIT[Periodic audit]
        RT_ALERT[Integrity alerts]
    end

    Write --> Checkpoint
    Checkpoint --> Recovery
    Recovery --> Runtime

    style W_CRC fill:#e8f5e9
    style C_SHA fill:#e8f5e9
    style R_VALIDATE fill:#e8f5e9
```

### Integrity Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `verify_on_read` | true | Verify CRC on WAL read |
| `verify_on_recovery` | true | Full verification on recovery |
| `audit_interval` | 1 hour | Background integrity check |
| `repair_mode` | `conservative` | Auto-repair strategy |
