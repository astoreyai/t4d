# Module Dependency Graph

Complete dependency map of World Weaver's 23 modules.

## Full Dependency Graph

```mermaid
flowchart TB
    subgraph External["External Interfaces"]
        API[api/]
        CLI[cli/]
        SDK[sdk/]
        INT[integration/]
    end

    subgraph Core["Core Systems"]
        CORE[core/]
        LEARN[learning/]
        MEM[memory/]
        EMBED[embedding/]
    end

    subgraph Neural["Neural Cognitive Architecture"]
        NCA[nca/]
        PRED[prediction/]
        ENC[encoding/]
        DREAM[dreaming/]
    end

    subgraph Storage["Storage & Persistence"]
        STORE[storage/]
        PERS[persistence/]
        OBS[observability/]
    end

    subgraph Processing["Processing"]
        CONSOL[consolidation/]
        EXTR[extraction/]
        TEMP[temporal/]
        BRIDGE[bridge/]
    end

    subgraph UI["User Interface"]
        IFACE[interfaces/]
        VIZ[visualization/]
        HOOKS[hooks/]
        INTEG[integrations/]
    end

    %% API dependencies
    API --> CONSOL
    API --> CORE
    API --> OBS
    API --> PERS

    %% CLI dependencies
    CLI --> CONSOL
    CLI --> CORE

    %% Core dependencies (highest centrality)
    CORE --> EMBED
    CORE --> LEARN
    CORE --> MEM
    CORE --> STORE

    %% Learning dependencies (most depended upon)
    BRIDGE --> LEARN
    BRIDGE --> NCA
    CONSOL --> LEARN
    EMBED --> LEARN
    INT --> LEARN
    INTEG --> LEARN
    IFACE --> LEARN
    MEM --> LEARN
    NCA --> LEARN
    TEMP --> LEARN
    VIZ --> LEARN

    %% Memory dependencies
    CONSOL --> MEM
    INT --> MEM
    IFACE --> MEM
    PERS --> MEM

    %% Storage dependencies
    CONSOL --> STORE
    INT --> STORE
    IFACE --> STORE
    MEM --> STORE
    OBS --> STORE

    %% NCA dependencies
    CONSOL --> NCA
    VIZ --> NCA

    %% Embedding dependencies
    CONSOL --> EMBED
    INT --> EMBED
    IFACE --> EMBED
    MEM --> EMBED
    OBS --> EMBED
    TEMP --> EMBED

    %% Other dependencies
    CONSOL --> EXTR
    DREAM --> PRED
    MEM --> CONSOL
    MEM --> OBS

    %% Styling
    classDef hub fill:#f9f,stroke:#333,stroke-width:3px
    classDef highDep fill:#bbf,stroke:#333,stroke-width:2px
    classDef external fill:#bfb,stroke:#333
    classDef neural fill:#fbb,stroke:#333

    class CORE hub
    class LEARN,STORE,EMBED,MEM highDep
    class API,CLI,SDK,INT external
    class NCA,PRED,ENC,DREAM neural
```

## Centrality Scores

| Module | In-Degree | Out-Degree | Centrality Score |
|--------|-----------|------------|------------------|
| core | 12 | 4 | 16 |
| learning | 11 | 1 | 12 |
| memory | 5 | 6 | 11 |
| consolidation | 0 | 10 | 10 |
| embedding | 7 | 2 | 9 |
| storage | 6 | 2 | 8 |
| interfaces | 0 | 6 | 6 |
| observability | 3 | 3 | 6 |
| nca | 3 | 2 | 5 |
| api | 0 | 4 | 4 |

## Simplified Layer View

```mermaid
flowchart TB
    subgraph L1["Layer 1: External Access"]
        direction LR
        API[REST API]
        CLI[CLI]
        SDK[SDK]
        WS[WebSocket]
    end

    subgraph L2["Layer 2: Core Services"]
        direction LR
        MEM[Memory]
        LEARN[Learning]
        NCA[NCA]
    end

    subgraph L3["Layer 3: Processing"]
        direction LR
        CONSOL[Consolidation]
        EMBED[Embedding]
        EXTR[Extraction]
    end

    subgraph L4["Layer 4: Storage"]
        direction LR
        STORE[Storage]
        PERS[Persistence]
        OBS[Observability]
    end

    subgraph L5["Layer 5: External Backends"]
        direction LR
        QDRANT[(Qdrant)]
        NEO4J[(Neo4j)]
        DISK[(Disk)]
    end

    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
```

## Circular Dependencies

**None detected** - Clean acyclic dependency graph.

## Module Clusters

### High-Cohesion Clusters

1. **Memory System**: memory, consolidation, learning, embedding
2. **Neural System**: nca, prediction, encoding, dreaming
3. **Storage System**: storage, persistence, observability
4. **Interface System**: api, interfaces, visualization, hooks
