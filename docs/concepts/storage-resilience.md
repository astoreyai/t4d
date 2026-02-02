# Storage Resilience

T4DM implements multiple resilience patterns to ensure data durability and graceful degradation.

## Overview

```mermaid
graph TB
    subgraph Resilience["Resilience Patterns"]
        CB[Circuit Breaker]
        SAGA[Saga Coordinator]
        GD[Graceful Degradation]
        FC[Fallback Cache]
    end

    subgraph Backends["Storage Backends"]
        NEO4J[(Neo4j)]
        QDRANT[(Qdrant)]
    end

    subgraph Recovery["Recovery"]
        HEALTH[Health Monitor]
        RETRY[Retry Logic]
        DRAIN[Queue Drain]
    end

    CB --> NEO4J
    CB --> QDRANT
    SAGA --> CB
    GD --> FC
    HEALTH --> CB
    RETRY --> SAGA
    DRAIN --> FC

    style CB fill:#ffebee
    style SAGA fill:#e8f5e9
    style GD fill:#fff3e0
    style FC fill:#e3f2fd
```

## Circuit Breaker Pattern

Prevents cascade failures by temporarily blocking requests to failing backends.

### State Machine

```mermaid
stateDiagram-v2
    [*] --> CLOSED: Initialize

    CLOSED --> OPEN: failure_count >= 5
    CLOSED --> CLOSED: success (reset count)

    OPEN --> HALF_OPEN: timeout (60s)
    OPEN --> OPEN: reject requests

    HALF_OPEN --> CLOSED: 2 successes
    HALF_OPEN --> OPEN: any failure

    note right of CLOSED
        Normal operation
        Tracking failures
    end note

    note right of OPEN
        Rejecting all requests
        Waiting for timeout
    end note

    note right of HALF_OPEN
        Testing recovery
        Limited requests
    end note
```

### Implementation

```mermaid
sequenceDiagram
    participant Client
    participant CB as CircuitBreaker
    participant Backend as Neo4j/Qdrant

    rect rgb(232, 245, 233)
        Note over CB: CLOSED State
        Client->>CB: request()
        CB->>Backend: forward request
        Backend-->>CB: success
        CB-->>Client: response
    end

    rect rgb(255, 235, 238)
        Note over CB: Failures accumulate
        Client->>CB: request()
        CB->>Backend: forward request
        Backend-->>CB: error
        CB->>CB: increment failure_count
        CB-->>Client: error
        Note over CB: failure_count >= 5
        CB->>CB: transition to OPEN
    end

    rect rgb(255, 243, 224)
        Note over CB: OPEN State
        Client->>CB: request()
        CB-->>Client: CircuitOpenError
        Note over CB: After 60s timeout
        CB->>CB: transition to HALF_OPEN
    end

    rect rgb(227, 242, 253)
        Note over CB: HALF_OPEN State
        Client->>CB: request()
        CB->>Backend: probe request
        Backend-->>CB: success
        CB->>CB: success_count++
        Note over CB: 2 successes
        CB->>CB: transition to CLOSED
    end
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `failure_threshold` | 5 | Failures before opening |
| `reset_timeout` | 60s | Time before half-open |
| `success_threshold` | 2 | Successes to close |
| `half_open_max` | 3 | Max concurrent in half-open |

## Saga Pattern

Coordinates distributed transactions across Neo4j and Qdrant with automatic rollback.

### Transaction Flow

```mermaid
sequenceDiagram
    participant API
    participant Saga as SagaCoordinator
    participant Neo4j
    participant Qdrant

    API->>Saga: store_memory(episode)

    rect rgb(232, 245, 233)
        Note over Saga: Phase 1: Prepare
        Saga->>Neo4j: begin_transaction()
        Neo4j-->>Saga: tx_id_1
        Saga->>Qdrant: begin_transaction()
        Qdrant-->>Saga: tx_id_2
    end

    rect rgb(227, 242, 253)
        Note over Saga: Phase 2: Execute
        Saga->>Neo4j: create_node(episode)
        Neo4j-->>Saga: node_id
        Saga->>Qdrant: upsert_vector(embedding)
        Qdrant-->>Saga: point_id
    end

    rect rgb(232, 245, 233)
        Note over Saga: Phase 3: Commit
        Saga->>Neo4j: commit(tx_id_1)
        Neo4j-->>Saga: committed
        Saga->>Qdrant: commit(tx_id_2)
        Qdrant-->>Saga: committed
    end

    Saga-->>API: success
```

### Rollback Flow

```mermaid
sequenceDiagram
    participant API
    participant Saga as SagaCoordinator
    participant Neo4j
    participant Qdrant

    API->>Saga: store_memory(episode)

    Saga->>Neo4j: begin_transaction()
    Neo4j-->>Saga: tx_id_1
    Saga->>Qdrant: begin_transaction()
    Qdrant-->>Saga: tx_id_2

    Saga->>Neo4j: create_node(episode)
    Neo4j-->>Saga: node_id

    rect rgb(255, 235, 238)
        Note over Saga: Qdrant fails
        Saga->>Qdrant: upsert_vector(embedding)
        Qdrant-->>Saga: ERROR
    end

    rect rgb(255, 243, 224)
        Note over Saga: Compensating Actions
        Saga->>Neo4j: rollback(tx_id_1)
        Neo4j-->>Saga: rolled_back
        Saga->>Qdrant: rollback(tx_id_2)
        Qdrant-->>Saga: rolled_back
    end

    Saga-->>API: SagaRollbackError
```

### Saga States

```mermaid
stateDiagram-v2
    [*] --> PENDING: create_saga()

    PENDING --> PREPARING: start()
    PREPARING --> PREPARED: all backends ready
    PREPARING --> ABORTING: prepare failed

    PREPARED --> EXECUTING: execute()
    EXECUTING --> EXECUTED: all operations done
    EXECUTING --> COMPENSATING: operation failed

    EXECUTED --> COMMITTING: commit()
    COMMITTING --> COMMITTED: all committed
    COMMITTING --> COMPENSATING: commit failed

    COMPENSATING --> COMPENSATED: rollback complete
    COMPENSATING --> FAILED: rollback failed

    ABORTING --> ABORTED: cleanup done

    COMMITTED --> [*]
    COMPENSATED --> [*]
    ABORTED --> [*]
    FAILED --> [*]
```

## Graceful Degradation

System continues operating with reduced functionality when backends fail.

### Degradation Levels

```mermaid
graph TB
    subgraph Level0["Level 0: Normal"]
        L0_DESC[Full functionality]
        L0_NEO4J[Neo4j: UP]
        L0_QDRANT[Qdrant: UP]
    end

    subgraph Level1["Level 1: No Graph"]
        L1_DESC[Vector search only]
        L1_NEO4J[Neo4j: DOWN]
        L1_QDRANT[Qdrant: UP]
    end

    subgraph Level2["Level 2: No Vector"]
        L2_DESC[Graph queries only]
        L2_NEO4J[Neo4j: UP]
        L2_QDRANT[Qdrant: DOWN]
    end

    subgraph Level3["Level 3: Cache Only"]
        L3_DESC[In-memory cache]
        L3_NEO4J[Neo4j: DOWN]
        L3_QDRANT[Qdrant: DOWN]
    end

    Level0 -->|Neo4j fails| Level1
    Level0 -->|Qdrant fails| Level2
    Level1 -->|Qdrant fails| Level3
    Level2 -->|Neo4j fails| Level3

    Level1 -->|Neo4j recovers| Level0
    Level2 -->|Qdrant recovers| Level0
    Level3 -->|Any recovers| Level1
    Level3 -->|Any recovers| Level2

    style Level0 fill:#e8f5e9
    style Level1 fill:#fff3e0
    style Level2 fill:#fff3e0
    style Level3 fill:#ffebee
```

### Feature Availability

| Feature | Level 0 | Level 1 | Level 2 | Level 3 |
|---------|---------|---------|---------|---------|
| Store memory | Full | Pending queue | Pending queue | Cache only |
| Vector search | Full | Full | Disabled | Disabled |
| Graph traversal | Full | Disabled | Full | Disabled |
| Relationship queries | Full | Disabled | Full | Disabled |
| Entity extraction | Full | Delayed | Delayed | Disabled |
| Consolidation | Full | Partial | Partial | Disabled |

### Degradation State Machine

```mermaid
stateDiagram-v2
    [*] --> NORMAL

    NORMAL --> DEGRADED_GRAPH: neo4j_circuit_open
    NORMAL --> DEGRADED_VECTOR: qdrant_circuit_open

    DEGRADED_GRAPH --> NORMAL: neo4j_recovered
    DEGRADED_GRAPH --> EMERGENCY: qdrant_circuit_open

    DEGRADED_VECTOR --> NORMAL: qdrant_recovered
    DEGRADED_VECTOR --> EMERGENCY: neo4j_circuit_open

    EMERGENCY --> DEGRADED_GRAPH: qdrant_recovered
    EMERGENCY --> DEGRADED_VECTOR: neo4j_recovered

    state NORMAL {
        [*] --> FullOperation
        FullOperation: All backends available
        FullOperation: Full feature set
    }

    state DEGRADED_GRAPH {
        [*] --> VectorOnly
        VectorOnly: Neo4j unavailable
        VectorOnly: Queue graph operations
    }

    state DEGRADED_VECTOR {
        [*] --> GraphOnly
        GraphOnly: Qdrant unavailable
        GraphOnly: Queue vector operations
    }

    state EMERGENCY {
        [*] --> CacheOnly
        CacheOnly: Both backends down
        CacheOnly: In-memory operation
    }
```

## Fallback Cache

In-memory cache provides resilience during backend outages.

### Cache Architecture

```mermaid
graph TB
    subgraph Operations["Incoming Operations"]
        STORE[Store Request]
        RECALL[Recall Request]
    end

    subgraph Cache["Fallback Cache"]
        LRU[LRU Cache<br/>1000 items]
        PENDING[Pending Queue<br/>10000 items]
        TTL[TTL Manager<br/>1 hour expiry]
    end

    subgraph Backends["Backends"]
        NEO4J[(Neo4j)]
        QDRANT[(Qdrant)]
    end

    STORE --> LRU
    STORE --> PENDING
    RECALL --> LRU

    PENDING -->|drain on recovery| NEO4J
    PENDING -->|drain on recovery| QDRANT

    style LRU fill:#e3f2fd
    style PENDING fill:#fff3e0
```

### Queue Drain Process

```mermaid
sequenceDiagram
    participant Monitor as HealthMonitor
    participant Cache as FallbackCache
    participant Queue as PendingQueue
    participant Neo4j
    participant Qdrant

    Note over Monitor: Backend recovered
    Monitor->>Cache: on_backend_recovered(neo4j)

    Cache->>Queue: get_pending_count()
    Queue-->>Cache: 150 items

    loop Drain Queue (batch of 10)
        Cache->>Queue: dequeue_batch(10)
        Queue-->>Cache: items[0:10]

        par Parallel writes
            Cache->>Neo4j: batch_write(items)
            Cache->>Qdrant: batch_write(items)
        end

        Neo4j-->>Cache: success
        Qdrant-->>Cache: success

        Cache->>Queue: ack_batch(10)
    end

    Cache->>Monitor: drain_complete()
```

## Health Monitoring

Continuous monitoring of backend health with automatic recovery.

### Health Check Flow

```mermaid
graph TB
    subgraph Monitor["Health Monitor"]
        TIMER[30s Timer]
        CHECK[Health Check]
        METRICS[Metrics Collector]
    end

    subgraph Backends["Backend Checks"]
        NEO4J_H[Neo4j Health<br/>MATCH (n) RETURN 1]
        QDRANT_H[Qdrant Health<br/>GET /health]
    end

    subgraph Actions["Actions"]
        ALERT[Alert System]
        CB_UPDATE[Circuit Breaker Update]
        DEGRADE[Degradation Manager]
    end

    TIMER --> CHECK
    CHECK --> NEO4J_H
    CHECK --> QDRANT_H

    NEO4J_H --> METRICS
    QDRANT_H --> METRICS

    METRICS -->|unhealthy| ALERT
    METRICS -->|state change| CB_UPDATE
    METRICS -->|level change| DEGRADE

    style ALERT fill:#ffebee
    style CB_UPDATE fill:#fff3e0
    style DEGRADE fill:#e3f2fd
```

### Metrics Collected

| Metric | Type | Description |
|--------|------|-------------|
| `ww_backend_up` | Gauge | Backend availability (0/1) |
| `ww_circuit_state` | Gauge | Circuit breaker state |
| `ww_degradation_level` | Gauge | Current degradation level |
| `ww_pending_queue_size` | Gauge | Pending operations count |
| `ww_cache_hit_rate` | Gauge | Fallback cache hit rate |
| `ww_recovery_duration` | Histogram | Time to recover |

## Performance Impact

| Pattern | Overhead | When Active |
|---------|----------|-------------|
| Circuit Breaker | < 100μs | Always |
| Saga Coordinator | 1-5ms | Store operations |
| Fallback Cache | < 50μs | Backend failures |
| Health Monitor | Negligible | Background |
| **Total Normal** | < 5ms | - |
| **Total Degraded** | < 10ms | - |
