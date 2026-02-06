# T4DM - System Architecture

**Version**: 2.0.0
**Status**: Production Ready
**Last Updated**: 2026-02-05

---

## Vision

T4DM is a biologically-inspired memory system combining a frozen Qwen2.5-3B language backbone with trainable spiking cortical blocks, backed by T4DX — a custom embedded spatiotemporal storage engine. Time is treated as a first-class dimension.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │  REST API   │ │  MCP Server │ │ Python SDK  │ │ Adapters  │ │
│  │  (FastAPI)  │ │  (FastMCP)  │ │ (sync/async)│ │(LC/LI/AG) │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      QWEN BACKBONE (4-bit)                      │
│  ┌─────────────────────────────┐ ┌─────────────────────────────┐│
│  │  Layers 0-17 (frozen)       │ │  Layers 18-35 (frozen)      ││
│  │  + QLoRA adapters (~8M)     │ │  + QLoRA adapters (~8M)     ││
│  └─────────────────────────────┘ └─────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                SPIKING CORTICAL STACK (×6 blocks)               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ① Thalamic Gate → ② LIF Integration → ③ Spike Attention   ││
│  │ → ④ Apical Modulation → ⑤ RWKV Recurrence → ⑥ Residual    ││
│  │                                                             ││
│  │ Neuromodulator Bus: DA | NE | ACh | 5-HT (per-block inject)││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                       MEMORY LAYER                              │
│  ┌───────────────────┐  ┌───────────────────────────────────┐  │
│  │   EPISODIC        │  │         SEMANTIC                  │  │
│  │   (κ < 0.3)       │  │         (κ > 0.6)                 │  │
│  │  • Raw events     │  │  • Consolidated concepts          │  │
│  │  • FSRS decay     │  │  • Hebbian weights                │  │
│  └───────────────────┘  └───────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    PROCEDURAL                               ││
│  │   • Skill patterns • Execution traces • Build-Retrieve-Update│
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                    LEARNING LAYER                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  STDP (Spike-Timing Dependent Plasticity)                 │  │
│  │  Hebbian Learning (co-activation strengthening)           │  │
│  │  Three-Factor Rule (pre × post × neuromodulator)          │  │
│  │  Eligibility Traces (temporal credit assignment)          │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                  T4DX STORAGE ENGINE                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        ││
│  │  │   WAL   │  │MemTable│  │Segments │  │ HNSW    │        ││
│  │  │(persist)│→ │(memory) │→ │ (LSM)   │→ │(vectors)│        ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        ││
│  │                                                             ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        ││
│  │  │CSR Graph│  │κ-Index  │  │Bitemporal│ │Provenance│       ││
│  │  │ (edges) │  │(consol.)│  │ (audit) │  │ (trace) │        ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                   OBSERVABILITY LAYER                           │
│     Prometheus  │  OpenTelemetry  │  22 Visualization Modules  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### κ (Kappa) Gradient

Continuous consolidation level [0,1] replacing discrete memory stores:

| κ Range | State | Description |
|---------|-------|-------------|
| 0.0-0.1 | Raw | Just encoded, volatile |
| 0.1-0.3 | Replayed | NREM-strengthened |
| 0.3-0.6 | Transitional | Being abstracted |
| 0.6-0.9 | Semantic | Consolidated concept |
| 0.9-1.0 | Stable | Permanent knowledge |

### τ(t) Temporal Gate

Controls memory write operations:

```
τ(t) = σ(λ_ε·ε + λ_Δ·novelty + λ_r·reward)
```

- ε: Prediction error (high = surprising, should store)
- novelty: Semantic distance from existing memories
- reward: Outcome-based reinforcement

### LSM Compaction = Memory Consolidation

| Compaction Phase | Biological Analog | κ Effect |
|------------------|-------------------|----------|
| FLUSH | Working → Episodic | κ = 0.0 |
| NREM | Replay + STDP | κ → 0.15-0.4 |
| REM | Prototype creation | κ → 0.6-0.85 |
| PRUNE | Forgetting curve | κ < 0.1 deleted |

### Nine Storage Primitives

All memory operations map to these T4DX primitives:

1. **INSERT** - Store new item
2. **GET** - Retrieve by ID
3. **SEARCH** - Vector similarity (HNSW)
4. **UPDATE_FIELDS** - Modify metadata
5. **UPDATE_EDGE_WEIGHT** - Hebbian learning
6. **TRAVERSE** - Graph navigation (CSR)
7. **SCAN** - Range queries
8. **DELETE** - Remove item
9. **BATCH_SCALE_WEIGHTS** - Bulk weight decay

---

## Data Flow

### Store Memory

```
Input → τ(t) gate → Encode (Time2Vec + embedding)
      → T4DX.INSERT → WAL.append → MemTable.insert
      → [flush] → Segment creation → HNSW index update
```

### Retrieve Memory

```
Query → Embed → T4DX.SEARCH (HNSW k-NN)
      → T4DX.TRAVERSE (graph enrichment)
      → Score (ACT-R activation + neuromod state)
      → Rerank → Return top-k
```

### Consolidate

```
Scheduler → Select candidates (by κ, age, importance)
          → NREM phase: Replay sequences, STDP weight updates
          → REM phase: HDBSCAN clustering, prototype creation
          → Compaction: Merge segments, update κ values
          → PRUNE: Remove low-κ items
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| LLM Backbone | Qwen2.5-3B (4-bit NF4) |
| Spiking Blocks | PyTorch + STE surrogate gradient |
| Storage | T4DX (embedded, Rust-inspired LSM) |
| API | FastAPI + Pydantic |
| MCP | FastMCP (stdio JSON-RPC) |
| Embeddings | Sentence-Transformers |
| Observability | Prometheus + OpenTelemetry |

---

## Resource Requirements

### Minimum (Development)
- CPU: 4 cores
- RAM: 8GB
- Disk: 20GB

### Recommended (Production)
- CPU: 8+ cores
- RAM: 24GB (Qwen + T4DX)
- GPU: 16GB+ VRAM (training)
- Disk: 100GB SSD

### VRAM Budget (24GB target)
- Qwen 4-bit: ~4GB
- Spiking blocks: ~2GB
- QLoRA gradients: ~4GB
- Activations: ~6GB
- Headroom: ~8GB

---

## Diagrams

See [diagrams/DIAGRAM_SUMMARY.md](diagrams/DIAGRAM_SUMMARY.md) for complete inventory:

- `t4dm_full_system.d2` - Complete system architecture
- `t4dx_storage_engine.d2` - Storage internals
- `t4dm_spiking_block.d2` - 6-stage cortical block
- `kappa_gradient_consolidation.mermaid` - κ progression

---

## References

- Tulving, E. (1972). Episodic and semantic memory
- Anderson, J. R. (1993). ACT-R: A theory of higher level cognition
- Schultz, W. (1997). Dopamine reward prediction error
- Hinton, G. (2022). Forward-Forward algorithm
- Anthropic (2024). Mechanistic interpretability

---

See also:
- [SYSTEM_ARCHITECTURE_MASTER.md](SYSTEM_ARCHITECTURE_MASTER.md) - Detailed module map
- [MEMORY_ARCHITECTURE.md](MEMORY_ARCHITECTURE.md) - Tripartite memory design
- [BRAIN_REGION_MAPPING.md](BRAIN_REGION_MAPPING.md) - Neuroscience foundations
