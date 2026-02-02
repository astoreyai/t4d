# T4DM System Lifecycle

**Version**: 1.0.0
**Status**: Production Ready
**Last Updated**: 2025-12-08

## Overview

T4DM is designed as an **always-on memory system** that survives crashes and restarts without data loss. This document explains the complete system lifecycle including startup, operation, and shutdown.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          SUPERVISOR                                  │
│            (systemd, Docker, Kubernetes, PM2, etc.)                 │
│                     restart=always                                   │
│                     healthcheck enabled                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      WORLD WEAVER SERVICE                            │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Persistence Layer                            │ │
│  │                                                                 │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐ │ │
│  │  │    WAL      │  │  Checkpoint  │  │  Recovery Manager     │ │ │
│  │  │  (append)   │  │  (periodic)  │  │  (cold/warm start)    │ │ │
│  │  └──────┬──────┘  └──────┬───────┘  └───────────┬───────────┘ │ │
│  │         │                │                      │             │ │
│  │         ▼                ▼                      ▼             │ │
│  │  ┌─────────────────────────────────────────────────────────┐  │ │
│  │  │              DURABLE STORAGE (disk)                      │  │ │
│  │  │   /var/lib/t4dm/wal/       /var/lib/t4dm/checkpoints/        │  │ │
│  │  └─────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Memory Components                            │ │
│  │                                                                 │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │ │
│  │  │  Buffer  │  │   Gate   │  │  Scorer  │  │  Traces  │       │ │
│  │  │ Manager  │  │ (neural) │  │ (neural) │  │          │       │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │ │
│  │                                                                 │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │ │
│  │  │ Dopamine │  │Serotonin │  │Norepinep.│  │Acetylch. │       │ │
│  │  │   (DA)   │  │  (5-HT)  │  │   (NE)   │  │  (ACh)   │       │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    External Storage                             │ │
│  │                                                                 │ │
│  │  ┌──────────────────────┐  ┌──────────────────────┐            │ │
│  │  │        Qdrant        │  │        Neo4j         │            │ │
│  │  │   (vector store)     │  │    (graph store)     │            │ │
│  │  │   - Episodic         │  │   - Semantic         │            │ │
│  │  │   - Working memory   │  │   - Procedural       │            │ │
│  │  └──────────────────────┘  └──────────────────────┘            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Startup Modes

### Cold Start (First Run / Data Loss)

**Triggered when**: No checkpoint file exists

```
COLD START SEQUENCE
═══════════════════

1. Detect no checkpoint
   └─► Initialize fresh

2. Create default state
   ├─► Gate: Xavier-initialized weights
   ├─► Scorer: Xavier-initialized weights
   ├─► Buffer: Empty
   ├─► Traces: Empty
   └─► Neuromodulators: Default baselines

3. Verify storage connections
   ├─► Qdrant: Ping and verify collections
   └─► Neo4j: Ping and verify constraints

4. Initialize WAL
   └─► Create segment_00000001.wal

5. Start serving
   └─► System ready (learning from scratch)
```

**What's Lost on Cold Start**:
- All learned gate weights (how to decide what to remember)
- All learned scorer weights (how to rank relevance)
- All eligibility traces (temporal credit assignment)
- All neuromodulator state (mood, arousal, expectations)
- Buffer contents (pending memories not yet consolidated)

**What's Preserved** (in Qdrant/Neo4j):
- Consolidated episodic memories
- Semantic knowledge graph
- Procedural memories

### Warm Start (Normal Restart)

**Triggered when**: Valid checkpoint file exists

```
WARM START SEQUENCE
═══════════════════

1. Find latest checkpoint
   └─► checkpoint_0000000000012345.bin.gz

2. Load checkpoint
   ├─► Verify checksum (SHA-256)
   ├─► Decompress (gzip)
   └─► Deserialize state

3. Restore components
   ├─► Gate: Load weights from checkpoint
   ├─► Scorer: Load weights from checkpoint
   ├─► Buffer: Load pending memories
   ├─► Traces: Load eligibility traces
   └─► Neuromodulators: Load DA/5-HT/NE/ACh state

4. Replay WAL
   ├─► Find checkpoint LSN (e.g., 12345)
   ├─► Read WAL entries > 12345
   └─► Replay each operation:
       ├─► BUFFER_ADD → buffer.add(memory)
       ├─► GATE_WEIGHT_UPDATE → gate.update(weights)
       └─► etc.

5. Verify consistency
   ├─► Check buffer count matches expected
   └─► Verify storage connectivity

6. Start serving
   └─► System ready (learning preserved!)
```

**Recovery Guarantees**:
- Zero data loss for operations that were logged to WAL
- At most lose operations between last fsync and crash
- Checkpoint + WAL replay = exact pre-crash state

## Shutdown Modes

### Graceful Shutdown (SIGTERM/SIGINT)

```
GRACEFUL SHUTDOWN SEQUENCE
══════════════════════════

Signal received (SIGTERM/SIGINT)
           │
           ▼
┌─────────────────────────────────┐
│  Phase 1: STOP ACCEPTING        │
│                                 │
│  • Set shutdown flag            │
│  • Reject new requests          │
│  • Return 503 for incoming      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Phase 2: DRAIN (30s timeout)   │
│                                 │
│  • Wait for in-flight ops       │
│  • Track operation count → 0    │
│  • Timeout: force proceed       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Phase 3: CHECKPOINT            │
│                                 │
│  • Create final checkpoint      │
│  • Serialize all state          │
│  • Atomic write to disk         │
│  • fsync                        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Phase 4: WAL CLOSE             │
│                                 │
│  • Log SYSTEM_SHUTDOWN          │
│  • Flush WAL                    │
│  • fsync                        │
│  • Close file                   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Phase 5: CLEANUP               │
│                                 │
│  • Run cleanup callbacks        │
│  • Close Qdrant connection      │
│  • Close Neo4j connection       │
│  • Release resources            │
└────────────┬────────────────────┘
             │
             ▼
         EXIT(0)
```

### Crash Recovery

```
CRASH SCENARIO
══════════════

System crashes (SIGKILL, power loss, kernel panic)
           │
           ▼
┌─────────────────────────────────┐
│  WAL has partial/complete ops   │
│  Last checkpoint at LSN 10000   │
│  WAL has entries to LSN 10150   │
│  (150 ops since checkpoint)     │
└────────────┬────────────────────┘
             │
     On next startup...
             │
             ▼
┌─────────────────────────────────┐
│  WARM START with WAL Replay     │
│                                 │
│  1. Load checkpoint (LSN 10000) │
│  2. Replay WAL 10001-10150      │
│  3. State = pre-crash state!    │
└─────────────────────────────────┘
```

**Worst Case Data Loss**:
- Operations between last `fsync` and crash
- Default: fsync after EVERY write = ~0 loss
- "fdatasync" mode: similar, slightly faster
- "none" mode: up to OS buffer contents (~30s of ops)

## Operation Logging

### What Gets Logged to WAL

| Operation | WAL Entry | Payload |
|-----------|-----------|---------|
| Add memory to buffer | `BUFFER_ADD` | memory_id, content, embedding, metadata |
| Remove from buffer | `BUFFER_REMOVE` | memory_id |
| Promote to storage | `BUFFER_PROMOTE` | memory_id, target_store |
| Gate weight update | `GATE_WEIGHT_UPDATE` | layer, weights (delta or full) |
| Scorer weight update | `SCORER_WEIGHT_UPDATE` | layer, weights |
| Dopamine expectation | `DOPAMINE_EXPECTATION` | context_id, expected_value |
| Serotonin mood change | `SEROTONIN_MOOD` | mood_state, timestamp |
| Create eligibility trace | `TRACE_CREATE` | trace_id, strength, context |
| Update trace | `TRACE_UPDATE` | trace_id, new_strength |
| Decay all traces | `TRACE_DECAY` | decay_factor |

### Operation Flow

```python
# CORRECT: Log BEFORE state change
async def add_memory(memory):
    # 1. Log to WAL (durable)
    lsn = await persistence.log(WALOperation.BUFFER_ADD, {
        "memory_id": memory.id,
        "content": memory.content,
    })

    # 2. Apply to in-memory state
    buffer.add(memory)

    return lsn

# WRONG: State change before log (data loss on crash!)
async def add_memory_wrong(memory):
    buffer.add(memory)  # ← If crash here, memory lost!
    await persistence.log(...)
```

## Checkpoint Contents

A checkpoint contains:

```python
Checkpoint(
    # Metadata
    lsn=12345,              # WAL position at checkpoint time
    timestamp=1733680000.0, # Unix timestamp
    version=1,              # Checkpoint format version

    # Neural network states
    gate_state={
        "W1": [[...], [...], ...],  # Input→Hidden weights
        "b1": [...],                 # Hidden biases
        "W2": [[...], [...], ...],  # Hidden→Output weights
        "b2": [...],                 # Output biases
    },
    scorer_state={...},  # Similar structure

    # Buffer contents
    buffer_state={
        "items": [
            {"id": "mem_001", "content": "...", "embedding": [...], ...},
            {"id": "mem_002", ...},
        ],
        "size": 42,
        "max_size": 100,
    },

    # Cluster index
    cluster_state={
        "centroids": [[...], [...], ...],
        "assignments": {...},
    },

    # Eligibility traces
    traces_state={
        "trace_001": {"strength": 0.8, "context": "...", "age": 5},
        "trace_002": {...},
    },

    # Neuromodulator state
    neuromod_state={
        "dopamine": {
            "expectations": {"ctx_1": 0.5, "ctx_2": 0.7},
            "baseline": 0.3,
        },
        "serotonin": {
            "mood": 0.6,
            "recent_rewards": [0.8, 0.3, 0.9],
        },
        "norepinephrine": {
            "arousal": 0.4,
            "arousal_history": [0.3, 0.4, 0.5],
        },
        "acetylcholine": {
            "encoding_strength": 0.7,
        },
    },
)
```

## Deployment Configuration

### Systemd Service

```ini
# /etc/systemd/system/t4dm.service
[Unit]
Description=T4DM Memory System
After=network.target qdrant.service neo4j.service
Wants=qdrant.service neo4j.service

[Service]
Type=notify
User=ww
Group=ww
WorkingDirectory=/opt/t4dm
ExecStart=/opt/t4dm/venv/bin/python -m ww.server
ExecReload=/bin/kill -HUP $MAINPID

# Graceful shutdown
TimeoutStopSec=60
KillMode=mixed
KillSignal=SIGTERM
SendSIGKILL=yes

# Auto-restart
Restart=always
RestartSec=5

# Resource limits
MemoryMax=8G
CPUQuota=400%

# Environment
Environment=T4DM_DATA_DIR=/var/lib/t4dm
Environment=T4DM_LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  t4dm:
    image: t4dm:latest
    restart: always
    stop_grace_period: 60s
    volumes:
      - ww-data:/var/lib/t4dm
    environment:
      - T4DM_DATA_DIR=/var/lib/t4dm
      - T4DM_CHECKPOINT_INTERVAL=300
      - T4DM_WAL_SYNC_MODE=fsync
    depends_on:
      qdrant:
        condition: service_healthy
      neo4j:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    deploy:
      resources:
        limits:
          memory: 8G

  qdrant:
    image: qdrant/qdrant:latest
    restart: always
    volumes:
      - qdrant-data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  neo4j:
    image: neo4j:5
    restart: always
    volumes:
      - neo4j-data:/data
    environment:
      - NEO4J_AUTH=neo4j/password
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  ww-data:
  qdrant-data:
  neo4j-data:
```

### Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: t4dm
spec:
  replicas: 1  # Single instance (stateful)
  strategy:
    type: Recreate  # No rolling update for stateful
  template:
    spec:
      terminationGracePeriodSeconds: 60
      containers:
      - name: t4dm
        image: t4dm:latest
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: data
          mountPath: /var/lib/t4dm
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 5"]  # Allow drain
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: ww-data-pvc
```

## Monitoring

### Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness check (is process alive?) |
| `GET /ready` | Readiness check (can accept requests?) |
| `GET /status` | Detailed system status |

### Metrics to Monitor

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `ww_wal_lsn` | - | Current WAL LSN (monotonic) |
| `ww_checkpoint_lsn` | LSN drift > 10000 | Last checkpoint LSN |
| `ww_checkpoint_age_seconds` | > 600 | Time since last checkpoint |
| `ww_buffer_size` | > 80% capacity | Buffer utilization |
| `ww_in_flight_ops` | > 100 sustained | Active operations |
| `ww_recovery_mode` | cold_start | Startup mode (warm=normal) |

### Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
- name: t4dm
  rules:
  - alert: WWCheckpointStale
    expr: time() - ww_checkpoint_timestamp > 600
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "WW checkpoint is stale"

  - alert: WWBufferFull
    expr: ww_buffer_size / ww_buffer_capacity > 0.9
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "WW buffer nearly full"

  - alert: WWColdStart
    expr: ww_recovery_mode == 1
    labels:
      severity: warning
    annotations:
      summary: "WW performed cold start (data may be lost)"
```

## Disaster Recovery

### Backup Strategy

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR=/backup/t4dm/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# 1. Create fresh checkpoint (forces fsync)
curl -X POST http://localhost:8080/api/v1/checkpoint

# 2. Copy checkpoint files
cp -a /var/lib/t4dm/checkpoints/* $BACKUP_DIR/

# 3. Backup Qdrant
qdrant-backup --output $BACKUP_DIR/qdrant/

# 4. Backup Neo4j
neo4j-admin dump --to=$BACKUP_DIR/neo4j.dump

# 5. Compress
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR
```

### Recovery Procedure

```bash
# Disaster recovery
#!/bin/bash
BACKUP_FILE=$1

# 1. Stop service
systemctl stop t4dm

# 2. Extract backup
tar -xzf $BACKUP_FILE -C /tmp/restore

# 3. Restore checkpoint
cp /tmp/restore/checkpoints/* /var/lib/t4dm/checkpoints/

# 4. Restore Qdrant
qdrant-restore --input /tmp/restore/qdrant/

# 5. Restore Neo4j
neo4j-admin load --from=/tmp/restore/neo4j.dump --force

# 6. Clear WAL (will replay from checkpoint)
rm -rf /var/lib/t4dm/wal/*

# 7. Start service (warm start from restored checkpoint)
systemctl start t4dm
```

## FAQ

### Q: What if power goes down suddenly?

**A**: The system uses WAL with fsync. On next startup:
1. Load last checkpoint
2. Replay WAL entries after checkpoint
3. State restored to last fsynced operation

Maximum loss: Operations between last fsync and crash (typically milliseconds with default settings).

### Q: Can I run multiple instances?

**A**: Not currently. T4DM is designed as a **single-instance stateful service**. The in-memory learning state (gate/scorer weights, eligibility traces) cannot be easily distributed.

Future: Possible leader-follower setup where followers handle reads and replicate WAL from leader.

### Q: How much disk space does WAL use?

**A**:
- Each segment: 64MB max
- Default max segments: 100
- Max WAL size: ~6.4GB
- After checkpoint, old segments are truncated

Checkpoint files: ~1-10MB each (compressed), keep last 5.

### Q: What if checkpoint file is corrupted?

**A**: The system keeps multiple checkpoints (default: 5). If latest fails validation (SHA-256 mismatch), it tries older checkpoints. If all fail, performs cold start.

### Q: How do I force a fresh start?

**A**:
```bash
# Option 1: Delete persistence data
rm -rf /var/lib/t4dm/*
systemctl start t4dm

# Option 2: API call
curl -X POST http://localhost:8080/api/v1/reset?confirm=true
```

---

**Document Version**: 1.0.0
**Author**: T4DM Team
**Review Date**: 2025-12-08
