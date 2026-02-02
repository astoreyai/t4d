# Storage Resilience - Circuit Breaker and T4DX WAL Durability

## T4DX Storage Resilience Architecture

This diagram illustrates the resilience mechanisms protecting T4DM's embedded T4DX storage engine.

```mermaid
graph TB
    subgraph "Application Layer"
        APP[Memory Operations<br/>Store, Retrieve, Update]
    end

    subgraph "Resilience Layer"
        CB_T4DX[T4DX Circuit Breaker<br/>State: CLOSED/HALF_OPEN/OPEN]
        FALLBACK[Fallback Cache<br/>In-memory degradation]
    end

    subgraph "T4DX Embedded Storage Engine"
        WAL[Write-Ahead Log<br/>JSON-lines + fsync]
        MT[MemTable<br/>Sorted in-memory buffer]
        SEG[Immutable Segments<br/>LSM-tree on disk]
        COMPACT[Compaction Engine<br/>NREM/REM/PRUNE]
    end

    subgraph "Health Monitoring"
        HEALTH[Health Checker<br/>WAL + segment integrity]
        METRICS[Metrics Collector<br/>Write/read latency, compaction stats]
        ALERTS[Alert System<br/>Circuit state changes]
    end

    %% Application to resilience
    APP --> CB_T4DX

    %% Circuit breaker to T4DX
    CB_T4DX -->|CLOSED| WAL
    WAL --> MT
    MT -->|flush| SEG
    SEG --> COMPACT

    %% Fallback path
    CB_T4DX -.->|OPEN| FALLBACK

    %% Health monitoring
    HEALTH --> CB_T4DX
    CB_T4DX --> METRICS
    METRICS --> ALERTS

    %% Recovery paths
    HEALTH -.->|Recovery detected| CB_T4DX
    FALLBACK -.->|Drain queue on recovery| WAL

    style APP fill:#e1f5ff
    style CB_T4DX fill:#fff4e1
    style FALLBACK fill:#ffebee
    style WAL fill:#ffcdd2
    style MT fill:#f3e5f5
    style SEG fill:#f3e5f5
    style COMPACT fill:#e8f5e9
    style HEALTH fill:#e0f2f1
    style METRICS fill:#e0f2f1
    style ALERTS fill:#e0f2f1
```

## Circuit Breaker Pattern

### State Machine

```mermaid
stateDiagram-v2
    [*] --> CLOSED: Initialize
    CLOSED --> OPEN: Failures >= threshold (5)
    OPEN --> HALF_OPEN: Reset timeout (60s)
    HALF_OPEN --> CLOSED: Successes >= threshold (2)
    HALF_OPEN --> OPEN: Any failure
    CLOSED --> CLOSED: Success (reset counter)
    CLOSED --> CLOSED: Failure (increment counter)
    HALF_OPEN --> HALF_OPEN: Success (increment counter)
```

### Circuit Breaker Configuration

```python
@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker wrapping T4DX."""
    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 2        # Successes to close from half-open
    reset_timeout: float = 60.0       # Seconds before trying half-open
    excluded_exceptions: tuple = ()   # Exceptions that don't count
```

### State Descriptions

| State | Behavior | Transitions |
|-------|----------|-------------|
| **CLOSED** | Normal operation, requests flow to T4DX | -> OPEN after 5 failures |
| **OPEN** | Fail fast, redirect to fallback cache | -> HALF_OPEN after 60s timeout |
| **HALF_OPEN** | Testing recovery, limited requests to T4DX | -> CLOSED after 2 successes, -> OPEN on any failure |

## T4DX WAL Durability Model

### Write Path

```mermaid
sequenceDiagram
    participant App as Application
    participant CB as Circuit Breaker
    participant WAL as T4DX WAL
    participant MT as MemTable
    participant Seg as Segment

    App->>CB: insert(memory_item)
    CB->>WAL: append(INSERT, item)
    WAL->>WAL: serialize JSON-line
    WAL->>WAL: fsync()
    WAL-->>CB: LSN assigned

    CB->>MT: apply(item)
    MT->>MT: sorted insert

    alt MemTable full (64MB or 10k items)
        MT->>Seg: flush to new segment
        Seg->>Seg: write sorted run + HNSW + CSR
        MT->>MT: clear
        WAL->>WAL: truncate up to flush LSN
    end

    CB-->>App: Success (LSN)
```

### Compaction = Consolidation

```mermaid
graph TB
    subgraph "LSM Compaction (= Biological Sleep)"
        FLUSH[MemTable Flush<br/>= Working Memory -> Episodic]
        NREM[NREM Compaction<br/>= Merge segments + kappa updates + STDP]
        REM[REM Compaction<br/>= Cluster + prototype creation]
        PRUNE[PRUNE Compaction<br/>= GC tombstoned + low-kappa]
    end

    FLUSH -->|segments accumulate| NREM
    NREM -->|kappa rises| REM
    REM -->|prototypes stable| PRUNE

    style FLUSH fill:#e1f5fe
    style NREM fill:#e8eaf6
    style REM fill:#f3e5f5
    style PRUNE fill:#ffebee
```

### Recovery Model

| Failure Mode | T4DX Recovery |
|-------------|---------------|
| Process crash | WAL replay from last flush LSN restores MemTable |
| Disk full | Circuit breaker opens, fallback cache queues writes |
| Corrupt segment | Skip segment, rebuild from WAL + remaining segments |
| Corrupt WAL entry | CRC check fails, truncate WAL at corruption point |
| Power loss | fsync guarantees WAL durability up to last sync |

## Graceful Degradation Strategy

### Degradation Levels

```mermaid
graph TB
    A[Normal Operation] -->|T4DX disk error| B[Level 1: Read-Only Mode]
    A -->|T4DX WAL fails| C[Level 2: In-Memory Only]

    B --> E[Serve from existing segments + MemTable]
    C --> F[Fallback to LRU cache]

    E --> H[Log to replay queue]
    F --> H

    H --> I{T4DX recovers?}
    I -->|Yes| J[Drain queue, restore service]
    I -->|No| K[Continue degraded mode]

    J --> A

    style A fill:#e8f5e9
    style B fill:#fff9c4
    style C fill:#ffccbc
    style F fill:#ffebee
    style J fill:#c8e6c9
```

### Degradation Level Details

| Level | T4DX Status | Capabilities | Limitations |
|-------|-------------|--------------|-------------|
| **0: Normal** | Healthy | Full read/write/search/compact | None |
| **1: Read-Only** | Disk errors on write | Read from segments + MemTable | No new writes, no compaction |
| **2: In-Memory** | WAL unavailable | LRU cache (1000 items) | No persistence, limited capacity |

## Health Monitoring

### Health Check Loop

```python
class T4DXHealthChecker:
    """Periodic health checks for T4DX storage engine."""

    def __init__(self, t4dx_engine, check_interval: float = 30.0):
        self.engine = t4dx_engine
        self.check_interval = check_interval

    async def check_wal(self) -> bool:
        """Verify WAL is writable and fsync works."""
        try:
            test_lsn = await self.engine.wal.append_test_entry()
            return test_lsn is not None
        except Exception:
            return False

    async def check_segments(self) -> bool:
        """Verify segment files are readable."""
        try:
            for seg in self.engine.segments:
                if not seg.verify_checksum():
                    return False
            return True
        except Exception:
            return False
```

## Configuration

```python
# Circuit breaker config (wraps T4DX)
CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=2,
    reset_timeout=60.0,
)

# T4DX WAL config
WAL_CONFIG = {
    "format": "json-lines",
    "fsync": True,
    "max_segment_size": "64MB",
    "truncate_on_flush": True,
}

# Fallback cache config
FALLBACK_CONFIG = {
    "max_size": 1000,
    "max_queue_size": 10000,
    "drain_batch_size": 100,
}

# Health check config
HEALTH_CONFIG = {
    "check_interval_seconds": 30.0,
    "timeout_seconds": 5.0,
    "alert_threshold": 3,
}
```

## Performance Impact

| Resilience Feature | Latency Overhead | Memory Overhead | CPU Overhead |
|--------------------|------------------|-----------------|--------------|
| Circuit Breaker | <100us | ~100 bytes | Negligible |
| T4DX WAL fsync | ~1-5ms per write | ~64MB (WAL buffer) | ~2% |
| Fallback Cache | <50us | ~1MB (1000 items) | ~5% |
| Health Checks | 0 (async) | ~10KB | ~1% |
| **Total** | **<5ms** | **~66MB** | **~8%** |

## Resilience Pattern Summary

| Pattern | Purpose | Failure Mode | Recovery |
|---------|---------|--------------|----------|
| **Circuit Breaker** | Fast failure detection for T4DX | 5 consecutive failures | 60s timeout -> 2 successes |
| **WAL + fsync** | Durability guarantee | Process crash | WAL replay from last flush LSN |
| **LSM Compaction** | Segment management + consolidation | Corrupt segment | Rebuild from WAL |
| **Fallback Cache** | Graceful degradation | T4DX unavailable | Queue replay on recovery |
| **Health Checks** | Proactive monitoring | Slow degradation | Alert before failure |
