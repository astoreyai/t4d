# Storage Resilience Patterns

Complete storage architecture with resilience, saga, and cold storage patterns.

## Full Storage Architecture

```mermaid
flowchart TB
    subgraph App["Application Layer"]
        MEM[Memory Operations]
        CONS[Consolidation]
        API[REST API]
    end

    subgraph Saga["Saga Pattern"]
        SAGA_C[Saga Coordinator]
        STEP1[Step 1: Vector]
        STEP2[Step 2: Graph]
        COMP1[Compensate 1]
        COMP2[Compensate 2]
    end

    subgraph Circuit["Circuit Breaker Layer"]
        CB_Q[CB: Qdrant]
        CB_N[CB: Neo4j]
        STATE[States: CLOSED→OPEN→HALF_OPEN]
    end

    subgraph Fallback["Graceful Degradation"]
        CACHE[In-Memory Cache<br/>10K entries, LRU]
        QUEUE[Pending Ops Queue]
        DRAIN[Drain on Recovery]
    end

    subgraph Storage["Storage Backends"]
        QDRANT[(Qdrant<br/>Vector Store)]
        NEO4J[(Neo4j<br/>Graph Store)]
    end

    subgraph Cold["Cold Storage"]
        ARCHIVE[Archive Manager]
        FS[Filesystem]
        S3[S3 Bucket]
        PG[PostgreSQL]
    end

    %% App to Saga
    MEM --> SAGA_C
    CONS --> SAGA_C
    API --> SAGA_C

    %% Saga steps
    SAGA_C --> STEP1
    STEP1 --> STEP2
    STEP1 -.->|"fail"| COMP1
    STEP2 -.->|"fail"| COMP2
    COMP2 --> COMP1

    %% Circuit breaker
    STEP1 --> CB_Q
    STEP2 --> CB_N

    %% Fallback
    CB_Q -->|"OPEN"| CACHE
    CB_N -->|"OPEN"| CACHE
    CACHE --> QUEUE
    QUEUE --> DRAIN

    %% Storage
    CB_Q -->|"CLOSED"| QDRANT
    CB_N -->|"CLOSED"| NEO4J

    %% Cold storage
    QDRANT --> ARCHIVE
    NEO4J --> ARCHIVE
    ARCHIVE --> FS
    ARCHIVE --> S3
    ARCHIVE --> PG

    %% Styling
    classDef saga fill:#e8f5e9,stroke:#2e7d32
    classDef circuit fill:#fff3e0,stroke:#ef6c00
    classDef fallback fill:#fce4ec,stroke:#c2185b
    classDef storage fill:#e3f2fd,stroke:#1565c0
    classDef cold fill:#f3e5f5,stroke:#7b1fa2

    class SAGA_C,STEP1,STEP2,COMP1,COMP2 saga
    class CB_Q,CB_N,STATE circuit
    class CACHE,QUEUE,DRAIN fallback
    class QDRANT,NEO4J storage
    class ARCHIVE,FS,S3,PG cold
```

## Circuit Breaker States

```mermaid
stateDiagram-v2
    [*] --> CLOSED: Initial state

    CLOSED --> CLOSED: Success
    CLOSED --> OPEN: Failures ≥ threshold (5)

    OPEN --> HALF_OPEN: Timeout expires (60s)

    HALF_OPEN --> CLOSED: Successes ≥ threshold (2)
    HALF_OPEN --> OPEN: Any failure

    note right of CLOSED: Normal operation
    note right of OPEN: Fast fail, no requests
    note right of HALF_OPEN: Test single request
```

## Saga Transaction Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant Saga as Saga Coordinator
    participant Q as Qdrant
    participant N as Neo4j

    App->>Saga: create_episode(data)

    rect rgb(200, 255, 200)
        Note over Saga,N: Forward Phase
        Saga->>Q: Step 1: add_vector()
        Q-->>Saga: Success (vector_id)
        Saga->>N: Step 2: create_node()
        N-->>Saga: Success (node_id)
    end

    Saga-->>App: SagaResult(COMMITTED)

    alt Step 2 Fails
        rect rgb(255, 200, 200)
            Note over Saga,Q: Compensation Phase (LIFO)
            N--xSaga: Error
            Saga->>Q: Compensate 1: delete_vector()
            Q-->>Saga: Deleted
        end
        Saga-->>App: SagaResult(COMPENSATED)
    end
```

## Graceful Degradation Flow

```mermaid
flowchart TB
    subgraph Request["Incoming Request"]
        REQ[Read/Write Request]
    end

    subgraph Primary["Primary Path"]
        TRY[Try Primary]
        CB[Circuit Breaker Check]
        EXEC[Execute on Backend]
    end

    subgraph Fallback["Fallback Path"]
        CACHE_R[Read from Cache]
        CACHE_W[Write to Cache]
        QUEUE_W[Queue for Retry]
    end

    subgraph Recovery["Recovery"]
        DETECT[Detect Recovery]
        DRAIN[Drain Queue]
        REPLAY[Replay Operations]
    end

    REQ --> CB

    CB -->|"CLOSED"| TRY
    TRY -->|"Success"| EXEC
    TRY -->|"Fail"| CB_UPDATE[Update CB State]
    CB_UPDATE --> CACHE_R

    CB -->|"OPEN"| CACHE_R

    CACHE_R -->|"Read"| RETURN[Return Cached]
    CACHE_W -->|"Write"| QUEUE_W

    DETECT --> DRAIN
    DRAIN --> REPLAY
    REPLAY --> EXEC

    %% Styling
    classDef primary fill:#e8f5e9,stroke:#2e7d32
    classDef fallback fill:#fff3e0,stroke:#ef6c00
    classDef recovery fill:#e3f2fd,stroke:#1565c0

    class TRY,CB,EXEC primary
    class CACHE_R,CACHE_W,QUEUE_W fallback
    class DETECT,DRAIN,REPLAY recovery
```

## Cold Storage Lifecycle

```mermaid
flowchart LR
    subgraph Active["Active Storage"]
        HOT[Hot Data<br/>< 90 days]
        WARM[Warm Data<br/>90-365 days]
    end

    subgraph Archive["Archive Process"]
        SELECT[Select Candidates]
        COMPRESS[Gzip Compress]
        WRITE[Write to Archive]
        META[Store Metadata]
    end

    subgraph Cold["Cold Storage"]
        FS[(Filesystem)]
        S3[(S3)]
        PG[(PostgreSQL)]
    end

    subgraph Retrieval["Retrieval"]
        REQ[Retrieval Request]
        LOOKUP[Metadata Lookup]
        DECOMP[Decompress]
        RESTORE[Restore to Active]
    end

    HOT -->|"Age > 90d"| WARM
    WARM -->|"Age > 365d"| SELECT

    SELECT --> COMPRESS
    COMPRESS --> WRITE
    WRITE --> FS
    WRITE --> S3
    WRITE --> PG
    WRITE --> META

    REQ --> LOOKUP
    LOOKUP --> FS
    LOOKUP --> S3
    LOOKUP --> PG
    FS --> DECOMP
    S3 --> DECOMP
    PG --> DECOMP
    DECOMP --> RESTORE
    RESTORE --> HOT
```

## Storage Security (P3-SEC-L2)

```mermaid
flowchart LR
    subgraph Input["User Input"]
        LABEL[Node Label]
        REL[Relationship Type]
        PROP[Property Name]
    end

    subgraph Validation["Validation Layer"]
        WHITELIST[Whitelist Check]
        REGEX[Regex Validation]
        SANITIZE[Sanitize Values]
    end

    subgraph Allowed["Allowed Values"]
        LABELS[Episode, Entity, Procedure]
        RELS[17 Relationship Types]
        PROPS[Alphanumeric + underscore]
    end

    subgraph Output["Safe Query"]
        CYPHER[Parameterized Cypher]
    end

    LABEL --> WHITELIST
    REL --> WHITELIST
    PROP --> REGEX

    WHITELIST --> LABELS
    WHITELIST --> RELS
    REGEX --> PROPS

    LABELS --> SANITIZE
    RELS --> SANITIZE
    PROPS --> SANITIZE

    SANITIZE --> CYPHER
```

## Performance Optimizations

```mermaid
flowchart TB
    subgraph Qdrant["Qdrant Optimizations"]
        SESSION_IDX[Session ID Index<br/>P2-OPT-B2.1]
        HYBRID[Hybrid Prefetch<br/>P2-OPT-B2.2]
        PARALLEL[Parallel Batching<br/>QDRANT-001]
    end

    subgraph Neo4j["Neo4j Optimizations"]
        POOL[Connection Pool<br/>P2-OPT-B3.3]
        BATCH[Batch Operations]
        CACHE[LRU Cache<br/>1000 entries]
    end

    subgraph Saga["Saga Optimizations"]
        TIMEOUT[Global Timeout: 60s]
        LIFO[LIFO Compensation]
        ASYNC[Async Execution]
    end

    SESSION_IDX -->|"O(log n)"| SEARCH[Faster Session Filter]
    HYBRID -->|"1.5x"| RECALL[Better Recall]
    PARALLEL -->|"10 concurrent"| UPLOAD[Faster Upload]

    POOL -->|"Metrics"| HEALTH[Pool Health]
    BATCH -->|"Atomic"| MULTI[Multi-Node Ops]
    CACHE -->|"TTL 5min"| RELATIONS[Relation Lookup]

    TIMEOUT --> BOUND[Bounded Execution]
    LIFO --> CORRECT[Correct Rollback Order]
    ASYNC --> PERF[Better Performance]
```

## Configuration Reference

| Component | Parameter | Default | Purpose |
|-----------|-----------|---------|---------|
| Circuit Breaker | failure_threshold | 5 | Failures before OPEN |
| Circuit Breaker | success_threshold | 2 | Successes to close |
| Circuit Breaker | reset_timeout | 60s | Time before HALF_OPEN |
| Fallback Cache | max_entries | 10,000 | LRU eviction limit |
| Saga | timeout | 60s | Global saga timeout |
| Qdrant | batch_size | 100 | Upload batch size |
| Qdrant | max_concurrency | 10 | Parallel uploads |
| Neo4j | pool_size | 50 | Connection pool |
| Archive | retention_days | 1825 | 5-year retention |
| Archive | compression | gzip | Compression method |
