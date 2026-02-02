# T4DM - System Architecture

## High-Level System Overview

This diagram shows the complete T4DM system architecture: Qwen 2.5-3B backbone with spiking cortical adapter, backed by T4DX embedded storage engine.

```mermaid
graph TB
    subgraph "Client Layer"
        MCP[MCP Gateway<br/>stdio JSON-RPC]
        SDK[Python SDK<br/>Sync/Async]
        API[REST API<br/>FastAPI]
    end

    subgraph "LLM + Spiking Adapter"
        QWEN_LO[Qwen 2.5-3B<br/>Layers 0-17 (frozen + QLoRA)]
        SPIKE[Spiking Cortical Stack<br/>6 blocks: LIF + thalamic gate<br/>+ spike attention + apical<br/>+ RWKV recurrence]
        QWEN_HI[Qwen 2.5-3B<br/>Layers 18-35 (frozen + QLoRA)]
        LM_HEAD[LM Head<br/>Token generation]
        NEUROMOD[Neuromodulator Bus<br/>DA, NE, ACh, 5-HT]
    end

    subgraph "Memory Orchestration"
        TAU[Temporal Gate tau-t<br/>Write gating]
        KAPPA[Kappa Gradient<br/>Consolidation level 0..1]
        MI[MemoryItem<br/>Unified memory record]
    end

    subgraph "Consolidation Layer"
        NREM[NREM Compaction<br/>Merge segments + kappa updates + STDP]
        REM[REM Compaction<br/>Cluster + prototype creation]
        PRUNE[PRUNE Compaction<br/>GC tombstoned + low-kappa]
    end

    subgraph "T4DX Embedded Storage Engine"
        WAL[Write-Ahead Log<br/>JSON-lines + fsync]
        MT[MemTable<br/>In-memory sorted buffer]
        SEG[Segments<br/>Immutable sorted runs]
        HNSW[HNSW Index<br/>4D vector search]
        CSR[CSR Graph<br/>Edge storage]
        KIDX[Kappa Index<br/>Secondary index on kappa]
        PROV[Provenance<br/>Forward/backward trace]
        BIT[Bitemporal<br/>What-did-we-know-when]
    end

    subgraph "Embedding Layer"
        BGE[BGE-M3 Adapter<br/>1024-dim embeddings]
        CACHE[Embedding Cache<br/>LRU eviction]
    end

    subgraph "Resilience"
        CB[Circuit Breaker<br/>Wraps T4DX operations]
    end

    MCP --> TAU
    SDK --> TAU
    API --> TAU

    TAU --> MI
    MI --> KAPPA

    QWEN_LO --> SPIKE
    SPIKE --> QWEN_HI
    QWEN_HI --> LM_HEAD
    NEUROMOD -.-> SPIKE

    MI --> WAL
    WAL --> MT
    MT -->|flush| SEG
    SEG --> HNSW
    SEG --> CSR
    SEG --> KIDX
    SEG --> PROV
    SEG --> BIT

    NREM -->|merge segments| SEG
    REM -->|create prototypes| SEG
    PRUNE -->|tombstone GC| SEG

    MI --> BGE
    BGE --> CACHE

    CB --> WAL

    style MCP fill:#e1f5ff
    style SDK fill:#e1f5ff
    style API fill:#e1f5ff
    style QWEN_LO fill:#e8eaf6
    style SPIKE fill:#fff3e0
    style QWEN_HI fill:#e8eaf6
    style LM_HEAD fill:#e8eaf6
    style NEUROMOD fill:#f3e5f5
    style TAU fill:#fff4e1
    style KAPPA fill:#fff4e1
    style MI fill:#e8f5e9
    style NREM fill:#ffe0b2
    style REM fill:#ffe0b2
    style PRUNE fill:#ffe0b2
    style WAL fill:#ffebee
    style MT fill:#ffebee
    style SEG fill:#ffebee
    style HNSW fill:#ffebee
    style CSR fill:#ffebee
    style CB fill:#ffebee
    style BGE fill:#e0f2f1
    style CACHE fill:#e0f2f1
```

## Component Descriptions

### Client Layer
- **MCP Gateway**: stdio-based JSON-RPC 2.0 server for Claude Code integration
- **Python SDK**: Synchronous and asynchronous clients with context managers
- **REST API**: FastAPI-based HTTP interface with OpenAPI documentation

### LLM + Spiking Adapter
- **Qwen 2.5-3B (frozen + QLoRA)**: 4-bit NF4 quantized backbone, LoRA(r=16) on q_proj + v_proj (~15M trainable params)
- **Spiking Cortical Stack**: 6 blocks with LIF neurons, thalamic gate, spike attention, apical modulation, RWKV recurrence (~50-80M trainable params)
- **LM Head**: Standard autoregressive token generation
- **Neuromodulator Bus**: DA (reward), NE (arousal), ACh (encoding/retrieval), 5-HT (credit assignment) injected per spiking layer

### Memory Orchestration
- **Temporal Gate tau(t)**: `sigma(lambda_e * e + lambda_D * novelty + lambda_r * reward)` -- gates memory writes and plasticity
- **Kappa Gradient**: Continuous consolidation level [0,1] replacing discrete memory stores (0.0=raw episodic, 1.0=stable knowledge)
- **MemoryItem**: Unified memory record with embedding, metadata, kappa, timestamps, edges

### Consolidation Layer
- **NREM Compaction**: LSM segment merge = biological NREM replay. Updates kappa, applies STDP weight changes
- **REM Compaction**: Cluster memories, create prototype abstractions
- **PRUNE Compaction**: Garbage-collect tombstoned items and low-kappa memories

### T4DX Embedded Storage Engine
- **Write-Ahead Log**: JSON-lines format with fsync for durability
- **MemTable**: In-memory sorted buffer (working memory analog)
- **Segments**: Immutable sorted runs on disk (LSM-tree style)
- **HNSW Index**: 4D vector similarity search (space + time)
- **CSR Graph**: Compressed sparse row edge storage for relationships
- **Kappa Index**: Secondary index for kappa-range queries
- **Provenance**: Forward/backward trace for EU AI Act compliance
- **Bitemporal**: "What did we know when" temporal queries

### Embedding Layer
- **BGE-M3 Adapter**: Local 1024-dimensional embedding generation
- **Embedding Cache**: LRU cache to minimize redundant computations

### Resilience
- **Circuit Breaker**: Wraps T4DX operations with CLOSED/HALF_OPEN/OPEN states

## Key Data Flows

1. **Memory Storage**: Client -> tau(t) gate -> MemoryItem -> T4DX WAL -> MemTable -> auto-flush -> Segment
2. **Memory Retrieval**: Client -> Query -> HNSW vector search + CSR graph traversal -> kappa-weighted results
3. **Consolidation**: NREM compaction (segment merge + STDP) -> REM compaction (cluster + prototype) -> PRUNE (GC)
4. **Inference**: Input -> Qwen(0-17) -> Spiking Stack (6 blocks) -> Qwen(18-35) -> LM Head -> tokens

## Key Metrics

| Component | Metric | Target |
|-----------|--------|--------|
| tau(t) Gate | Decision latency | <1ms |
| MemTable | Flush threshold | 64MB or 10k items |
| HNSW | Search latency (k=10) | <5ms |
| Spiking Stack | Forward pass | <50ms |
| Circuit Breaker | Failure threshold | 5 failures |
| Embedding Cache | Hit rate | >80% |

## Technology Stack

- **Languages**: Python 3.11+
- **LLM**: Qwen 2.5-3B (4-bit NF4 via bitsandbytes)
- **Spiking**: Custom PyTorch LIF neurons with surrogate gradients
- **Storage**: T4DX embedded engine (LSM-tree + HNSW + CSR graph)
- **ML**: PyTorch 2.x, sentence-transformers, BGE-M3
- **API**: FastAPI, Pydantic 2.x
- **MCP**: FastMCP 0.4+, Anthropic MCP SDK
- **Testing**: pytest, 8905 tests

## VRAM Budget

| Component | VRAM |
|-----------|------|
| Qwen 2.5-3B (4-bit) | ~2GB |
| QLoRA adapters | ~0.5GB |
| Spiking cortical stack | ~1GB |
| KV cache + activations | ~4-6GB |
| **Inference total** | **~10GB** |
| Training overhead | +6GB |
| **Training total** | **~16GB** |
| **Hardware limit** | **24GB** |
