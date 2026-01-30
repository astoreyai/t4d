# World Weaver - System Architecture

## High-Level System Overview

This diagram shows the complete World Weaver system architecture, from client interfaces through memory subsystems to storage backends.

```mermaid
graph TB
    subgraph "Client Layer"
        MCP[MCP Gateway<br/>stdio JSON-RPC]
        SDK[Python SDK<br/>Sync/Async]
        API[REST API<br/>FastAPI]
    end

    subgraph "Memory Orchestration"
        WM[Working Memory<br/>7±2 items, attentional blink]
        DG[Dentate Gyrus<br/>Pattern Separation]
        LG[Learned Memory Gate<br/>Thompson Sampling]
    end

    subgraph "Tripartite Memory System"
        EM[Episodic Memory<br/>Autobiographical events<br/>Bi-temporal versioning]
        SM[Semantic Memory<br/>Knowledge graph<br/>Hebbian learning]
        PM[Procedural Memory<br/>Skills & workflows<br/>Build-Retrieve-Update]
    end

    subgraph "Neural Mechanisms"
        NMO[Neuromodulator Orchestra<br/>DA, NE, 5-HT, ACh, GABA]
        HP[Homeostatic Plasticity<br/>Synaptic scaling]
        RC[Reconsolidation Engine<br/>Memory updating]
    end

    subgraph "Consolidation Layer"
        NREM[NREM Sleep Phase<br/>SWR replay, E→S transfer]
        REM[REM Sleep Phase<br/>Creative abstraction]
        PRUNE[Synaptic Pruning<br/>Downscaling weak connections]
    end

    subgraph "Storage Backends"
        NEO[(Neo4j Graph DB<br/>Episodes & Knowledge)]
        QDR[(Qdrant Vector DB<br/>Embeddings)]
        CB[Circuit Breakers<br/>Resilience patterns]
    end

    subgraph "Embedding Layer"
        BGE[BGE-M3 Adapter<br/>1024-dim embeddings]
        CACHE[Embedding Cache<br/>LRU eviction]
    end

    MCP --> WM
    SDK --> WM
    API --> WM

    WM --> DG
    DG --> LG
    LG --> EM
    LG --> SM
    LG --> PM

    EM --> NMO
    SM --> NMO
    PM --> NMO

    NMO --> HP
    NMO --> RC

    EM --> NREM
    SM --> NREM
    NREM --> REM
    REM --> PRUNE
    PRUNE --> SM

    EM --> BGE
    SM --> BGE
    PM --> BGE
    BGE --> CACHE

    EM --> CB
    SM --> CB
    PM --> CB
    CB --> NEO
    CB --> QDR

    style MCP fill:#e1f5ff
    style SDK fill:#e1f5ff
    style API fill:#e1f5ff
    style WM fill:#fff4e1
    style DG fill:#fff4e1
    style LG fill:#fff4e1
    style EM fill:#e8f5e9
    style SM fill:#e8f5e9
    style PM fill:#e8f5e9
    style NMO fill:#f3e5f5
    style HP fill:#f3e5f5
    style RC fill:#f3e5f5
    style NREM fill:#ffe0b2
    style REM fill:#ffe0b2
    style PRUNE fill:#ffe0b2
    style NEO fill:#ffebee
    style QDR fill:#ffebee
    style CB fill:#ffebee
    style BGE fill:#e0f2f1
    style CACHE fill:#e0f2f1
```

## Component Descriptions

### Client Layer
- **MCP Gateway**: stdio-based JSON-RPC 2.0 server for Claude Code integration
- **Python SDK**: Synchronous and asynchronous clients with context managers
- **REST API**: FastAPI-based HTTP interface with OpenAPI documentation

### Memory Orchestration
- **Working Memory**: Capacity-limited (7±2 items) with attentional blink modeling
- **Dentate Gyrus**: Sparse pattern separation to distinguish similar inputs
- **Learned Memory Gate**: Online Bayesian logistic regression deciding what to store

### Tripartite Memory System
- **Episodic Memory**: Autobiographical events with T_ref/T_sys bi-temporal versioning
- **Semantic Memory**: Entity-relationship graph with Hebbian strengthening on co-retrieval
- **Procedural Memory**: Skill storage using Memp framework (Build-Retrieve-Update)

### Neural Mechanisms
- **Neuromodulator Orchestra**: Dopamine (RPE), Norepinephrine (arousal), Serotonin (credit assignment), Acetylcholine (encoding/retrieval mode), GABA (inhibition)
- **Homeostatic Plasticity**: Synaptic scaling to maintain network stability
- **Reconsolidation Engine**: Dopamine-modulated memory updating during retrieval

### Consolidation Layer
- **NREM Sleep Phase**: Sharp-wave ripple (SWR) replay transfers episodes to semantic memory
- **REM Sleep Phase**: Creative abstraction finds patterns across memory clusters
- **Synaptic Pruning**: Downscaling weak connections during sleep consolidation

### Storage Backends
- **Neo4j**: Graph database for episodes, entities, and relationships
- **Qdrant**: Vector database for semantic similarity search
- **Circuit Breakers**: Resilience patterns with graceful degradation

### Embedding Layer
- **BGE-M3 Adapter**: Local 1024-dimensional embedding generation
- **Embedding Cache**: LRU cache to minimize redundant computations

## Key Data Flows

1. **Memory Storage**: Client → Working Memory → Pattern Separation → Learned Gate → Tripartite Memory → Storage Backends
2. **Memory Retrieval**: Client → Query → Vector/Graph Search → Neuromodulator Weighting → Results
3. **Consolidation**: NREM (replay) → REM (abstraction) → Prune → Updated Semantic Memory
4. **Learning**: Retrieval + Outcome → Neuromodulators → Gate/Scorer Update → Improved Future Performance

## Key Metrics

| Component | Metric | Target |
|-----------|--------|--------|
| Learned Gate | Decision latency | <5ms |
| Working Memory | Capacity | 7±2 items |
| Pattern Separation | Sparsity | 2-5% active |
| NREM Consolidation | SWR compression | 10-20x |
| Circuit Breaker | Failure threshold | 5 failures |
| Embedding Cache | Hit rate | >80% |

## Technology Stack

- **Languages**: Python 3.11+
- **Databases**: Neo4j Community 5.x, Qdrant 1.7+
- **ML**: PyTorch 2.x, sentence-transformers, BGE-M3
- **API**: FastAPI, Pydantic 2.x
- **MCP**: FastMCP 0.4+, Anthropic MCP SDK
- **Testing**: pytest, 1259 tests, 79% coverage
