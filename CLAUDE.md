# T4DM - Temporal 4D Memory

**Path**: `/mnt/projects/t4d/t4dm/`
**Version**: 2.0.0
**Status**: Active Development
**Plan**: [docs/plans/FULL_SYSTEM_PLAN.md](docs/plans/FULL_SYSTEM_PLAN.md)

---

## Project Overview

T4DM is a biologically-inspired memory system combining a frozen Qwen2.5-3B (4-bit) language backbone with trainable spiking cortical blocks as a memory adapter, backed by T4DX — a custom embedded spatiotemporal storage engine. Time is first-class.

**Architecture**: Qwen(0-17, QLoRA) → Spiking Cortical Stack (6 blocks) → Qwen(18-35, QLoRA) → LM head, with T4DX read/write gated by τ(t).

**Key Features**:
- Frozen Qwen2.5-3B + QLoRA adapters (~15M trainable params)
- 6 spiking cortical blocks: LIF neurons + thalamic gate + spike attention + apical modulation + RWKV recurrence (~50-80M trainable params)
- T4DX embedded LSM-style storage engine (replaces Neo4j + Qdrant + Saga)
- κ-gradient memory consolidation: LSM compaction = biological sleep phases
- Neuromodulator bus (DA, NE, ACh, 5-HT) driving spiking block dynamics
- 9,600+ tests

**Hardware Target**: i9-24core, 24GB VRAM, 128GB RAM, 4TB disk
**VRAM Budget**: ~10GB inference / ~16GB training out of 24GB

---

## T4D Stack Position

```
┌─────────┐
│  T4DV   │  Visualization (3D/4D rendering)
├─────────┤
│  T4DM   │  Memory + LLM + Spiking (THIS PROJECT)
├─────────┤
│  T4DX   │  Embedded in T4DM (custom storage engine)
└─────────┘
```

T4DX is no longer a separate service — it's an embedded storage engine inside T4DM.

---

## Core Concepts

### κ (Kappa) Gradient
Continuous consolidation level [0,1] replacing discrete memory stores:
- κ=0.0: Raw episodic (just encoded)
- κ~0.15: Replayed (NREM strengthened)
- κ~0.4: Transitional (being abstracted)
- κ~0.85: Semantic concept (REM prototype)
- κ=1.0: Stable knowledge (fully consolidated)

### LSM Compaction = Memory Consolidation
- MemTable flush = working memory → episodic store
- NREM compaction = merge segments + κ updates + STDP
- REM compaction = cluster + prototype creation
- PRUNE compaction = GC tombstoned + low-κ items

### τ(t) Temporal Gate
`τ(t) = σ(λ_ε·ε + λ_Δ·novelty + λ_r·reward)` — gates memory writes and plasticity.

### Nine Storage Primitives
INSERT, GET, SEARCH, UPDATE_FIELDS, UPDATE_EDGE_WEIGHT, TRAVERSE, SCAN, DELETE, BATCH_SCALE_WEIGHTS — every codebase access pattern maps to one of these.

---

## Architecture

```
src/t4dm/
├── core/                      # Core types + protocols
│   ├── types.py               # Episode, Entity, Procedure, Relationship
│   ├── protocols.py           # VectorStore, GraphStore protocols
│   ├── memory_item.py         # Unified MemoryItem (κ-based)
│   ├── query_policies.py      # κ-based query routing
│   └── temporal_gate.py       # τ(t) write gate
├── spiking/                   # Spiking cortical blocks
│   ├── lif.py                 # LIF neuron with STE surrogate gradient
│   ├── thalamic_gate.py       # Stage ① input gating
│   ├── spike_attention.py     # Stage ③ STDP-weighted attention
│   ├── apical_modulation.py   # Stage ④ prediction error + FF goodness
│   ├── rwkv_recurrence.py     # Stage ⑤ linear recurrence
│   ├── cortical_block.py      # Composed 6-stage block
│   ├── cortical_stack.py      # ×6 blocks with shared context
│   ├── neuromod_bus.py        # NT injection per layer
│   └── oscillator_bias.py     # θ/γ/δ phase bias
├── qwen/                      # Qwen 3B + QLoRA
│   ├── loader.py              # 4-bit NF4 model loading
│   ├── qlora.py               # LoRA(r=16) on q_proj + v_proj
│   ├── extractor.py           # Hidden state hooks (layer 18)
│   ├── projections.py         # 2048↔1024 memory projection
│   ├── unified_model.py       # Full composed model
│   ├── training.py            # Phase 1: surrogate gradient + QLoRA
│   ├── local_learning.py      # Phase 2: three-factor local learning
│   ├── inference.py           # Token generation pipeline
│   └── visibility.py          # Glass-box activation hooks
├── storage/
│   ├── t4dx/                  # Embedded storage engine
│   │   ├── index.py           # LSM segments core
│   │   ├── hnsw.py            # HNSW vector index
│   │   ├── bloom.py           # Bloom filter for ID checks
│   │   ├── csr_graph.py       # CSR graph structure
│   │   ├── memory_adapter.py  # MemoryItem ↔ T4DX adapter
│   │   ├── persistence.py     # WAL + segment serialization
│   │   ├── vector_adapter.py  # VectorStore protocol impl
│   │   ├── graph_adapter.py   # GraphStore protocol impl
│   │   ├── kappa_index.py     # κ secondary index
│   │   ├── bitemporal.py      # "What did we know when" queries
│   │   └── provenance.py      # Forward/backward trace
├── adapters/                  # Framework adapters
│   ├── langchain.py           # LangChain BaseMemory
│   ├── llamaindex.py          # LlamaIndex VectorStore
│   ├── autogen.py             # AutoGen Memory protocol
│   └── crewai.py              # CrewAI Memory
├── sdk/
│   └── simple.py              # Simple 3-line API
├── consolidation/             # Sleep-phase consolidation
├── learning/                  # Hebbian, STDP, neuromodulators
├── nca/                       # Neural Circuit Architecture (brain region simulations)
│   └── cerebellum.py          # Cerebellar timing model
├── memory/                    # Episodic, semantic, procedural stores
├── bridges/                   # Higher-level memory semantics
├── persistence/               # WAL, checkpoint, recovery
├── api/                       # FastAPI REST endpoints
│   └── routes/
│       ├── compat.py          # Mem0-compatible REST shim
│       └── ws_viz.py          # WebSocket visualization streaming
├── visualization/             # Visualization modules
│   ├── kappa_gradient.py      # κ distribution + consolidation flow
│   ├── t4dx_metrics.py        # LSM compaction stats
│   ├── spiking_dynamics.py    # Spike rasters, membrane potentials
│   ├── qwen_metrics.py        # QLoRA weight norms, projections
│   ├── neuromod_layers.py     # NT injection per block
│   ├── oscillator_injection.py # Theta/gamma/delta phase viz
│   └── stream.py              # Real-time streaming
├── observability/             # Metrics, tracing
└── schemas/
    └── openai_tools.json      # OpenAI function tool schemas
```

---

## Plan Summary

### Core Implementation (58 atoms) - ALL COMPLETED

| Phase | Atoms | Description | Status |
|-------|-------|-------------|--------|
| P1: Foundation | 12 | MemoryItem, τ gate, LIF, spiking blocks, neuromod bus | COMPLETED |
| P2: T4DX Storage | 15 | Custom engine, adapters, migration, saga removal | COMPLETED |
| P3: Qwen Integration | 11 | Qwen loader, QLoRA, unified model, training, inference | COMPLETED |
| P4: Consolidation | 8 | Sleep phases with spiking replay, κ updates | COMPLETED |
| P5: Diagrams | 5 | Architecture diagrams (D2) | COMPLETED |
| P6: Validation | 5 | Bio-plausibility, benchmarks, E2E, glass-box | COMPLETED |
| P7: Persistence | 2 | Checkpoint v3, recovery v2 for T4DX + spiking state | COMPLETED |

### Optimization Phases (70 atoms) - ALL COMPLETED

| Phase | Atoms | Description | Status |
|-------|-------|-------------|--------|
| A: Code Cleanup | 12 | Dead code, bugs, naming, WAL unification | COMPLETED (Sprint 1) |
| B: T4DX Storage Completion | 7 | HNSW, CSR, kappa index, bitemporal, provenance | COMPLETED (Sprint 2) |
| C: Diagram Overhaul | 22 | Architecture, biological fixes, Neo4j removal, new diagrams | COMPLETED (Sprints 1-2) |
| D: Visualization Suite | 7 | Kappa, T4DX, spiking, Qwen, neuromod, oscillator viz | COMPLETED (Sprint 4) |
| E: Plugin & Framework Adapters | 8 | LangChain, LlamaIndex, AutoGen, CrewAI, Mem0, MCP | COMPLETED (Sprint 3) |
| F: Documentation & Taxonomy | 8 | Taxonomy, competitive analysis, integration guide | COMPLETED (Sprint 3) |
| G: Production Hardening | 6 | Benchmarks, concurrency, cerebellum | COMPLETED (Sprint 4) |

See [FULL_SYSTEM_PLAN.md](docs/plans/FULL_SYSTEM_PLAN.md) for complete atom specifications.

---

## Existing Diagrams

Located in `docs/diagrams/`. Key files:

| Diagram | Format | Status |
|---------|--------|--------|
| `t4dm_spiking_block.d2` | D2 | Current (spiking block stages) |
| `t4dm_snn_vaswani.d2` | D2 | Current (Vaswani-style SNN layout) |
| `t4dm_snn_transformer_architecture.mermaid` | Mermaid | Current (SNN overview) |
| `01_system_architecture.*` | SVG/PNG | Current (T4DX + Qwen + Spiking) |
| `05_storage_architecture.*` | Mermaid/SVG/PNG | Current (T4DX LSM engine) |
| `14_storage_subsystem.*` | Mermaid/SVG/PNG | Current (T4DX subsystem) |
| `23_class_storage.*` | Mermaid/SVG/PNG | Current (T4DX classes) |

---

## Quick Commands

```bash
# Development
make dev              # Start development server
make test             # Run all tests
make test-fast        # Run fast unit tests only
make coverage         # Generate coverage report

# Testing
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/biology/        # Biological plausibility tests
pytest -x --tb=short        # Stop on first failure
```

---

## Integration Points

| System | Integration |
|--------|-------------|
| **T4DV** | T4DV visualizes T4DM memory structures and spiking activations |
| **Kymera** | Kymera uses T4DM for agent memory via API |

---

## Key Decisions

1. **Custom storage over PostgreSQL/existing DB**: Co-locates vectors, edges, metadata, temporal indices. LSM compaction = biological consolidation. Zero network hops.
2. **QLoRA + spiking adapter (novel)**: No published work combines QLoRA fine-tuning with spiking neural network adapters on a frozen LLM.
3. **κ gradient over discrete stores**: Continuous consolidation level eliminates cross-store transactions for type transitions.
4. **Overlay pattern for learning writes**: Hebbian Δw, STDP, κ mutations are O(1) dict inserts into MemTable, consumed during compaction.

---
