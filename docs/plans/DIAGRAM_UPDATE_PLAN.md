# Diagram Update Plan for T4DM v2.0 Architecture

**Date**: 2026-01-30
**Scope**: Update all architectural diagrams to reflect the v2.0 plan (Qwen+QLoRA, spiking cortical blocks, T4DX embedded engine, κ-gradient consolidation)

---

## Audit Summary

### Current (no changes needed)
| Diagram | Format | What it shows |
|---------|--------|---------------|
| `t4dm_spiking_block.d2` | D2 | 6-stage spiking cortical block with neuromod injection + oscillatory scaffold. Already shows T4DX unified memory. |
| `t4dm_snn_vaswani.d2` | D2 | Full system Vaswani-style: BGE-M3 → spiking core → T4DX → consolidation. Already v2.0 target architecture. |
| `t4dm_snn_transformer_architecture.mermaid` | Mermaid | Complete 9-layer system flowchart with T4DX, κ policies, adenosine sleep. Already v2.0. |
| `persistence_layer.mmd` | Mermaid | WAL + checkpoint + recovery + shutdown. Architecture is valid, applies to T4DX. |
| `wal_flow.mmd` | Mermaid | WAL append sequence. Valid for T4DX WAL. |
| `recovery_flow.mmd` | Mermaid | Cold/warm start protocol. Valid. |
| `checkpoint_lifecycle.mmd` | Mermaid | Checkpoint state machine. Valid. |
| `shutdown_flow.mmd` | Mermaid | Graceful shutdown sequence. Valid. |
| `07_class_bioinspired.mmd` | Mermaid | Sparse encoder, dendritic neuron, Hopfield, eligibility traces. All still valid. |
| `22_class_learning.mmd` | Mermaid | LearnedMemoryGate, Bayesian LR, STDP, Hebbian. Valid. |
| `34_state_memory_gate.mmd` | Mermaid | Memory gate FSM (Idle→Eval→Pending→Update). Valid. |
| `neural_pathways.md` | Markdown | Complete neuromodulator system (DA/NE/ACh/5-HT/GABA). Valid. |
| `neuromodulator_pathways.md` | Markdown | Detailed pathway map with signal flow. Valid. |
| `credit_assignment_flow.mermaid` | Mermaid | Temporal credit assignment via eligibility traces. Valid. |
| `adenosine_homeostasis.mermaid` | Mermaid | Sleep-wake cycle adenosine model. Valid. |
| `consolidation_flow.md` | Markdown | NREM/REM/PRUNE algorithm documentation. Valid (needs minor T4DX op mapping). |
| `embedding_pipeline.md` | Markdown | BGE-M3 adapters and caching. Valid. |
| `learning_signals.md` | Markdown | Signal definitions. Valid. |

### Must Update (6 diagrams)

| # | Diagram | Issue | Action |
|---|---------|-------|--------|
| 1 | `05_storage_architecture.mmd` | Shows Neo4j+Qdrant+Saga dual-store | **Rewrite**: T4DX engine internals (WAL→MemTable→Segments→Compactor) |
| 2 | `14_storage_subsystem.mmd` | Shows Saga coordinator, circuit breakers for external stores | **Rewrite**: T4DX 9-operation API, segment lifecycle |
| 3 | `23_class_storage.mmd` | UML with Neo4jStore, QdrantStore, SagaCoordinator | **Rewrite**: T4DXEngine, MemTable, Segment, Compactor, VectorAdapter, GraphAdapter |
| 4 | `01_system_architecture` (mmd in system_architecture.md) | Storage section lists Qdrant/Neo4j/PostgreSQL | **Update**: Replace storage section with T4DX, add spiking blocks + Qwen |
| 5 | `21_class_memory.mmd` | References Neo4j knowledge_graph, Qdrant vector_store | **Update**: Replace storage refs with T4DX engine calls |
| 6 | `31_state_circuit_breaker.mmd` | 3-state FSM for external store circuit breakers | **Deprecate** or repurpose for T4DX segment health |

### Must Create (5 new diagrams from plan Phase 6)

| # | Diagram | Format | Content |
|---|---------|--------|---------|
| 7 | `t4dm_full_system.d2` | D2 | Qwen(frozen,4-bit)+QLoRA → Spiking Stack → T4DX + all lateral modules |
| 8 | `t4dx_storage_engine.d2` | D2 | WAL→MemTable(overlays)→Segments(HNSW+CSR+κ-array)→Compactor(NREM/REM/PRUNE)→Global Index |
| 9 | `t4dm_data_flow.d2` | D2 | Token→Qwen+QLoRA(0-17)→spiking↔T4DX→Qwen+QLoRA(18-35)→output, τ(t) gating |
| 10 | `t4dm_training_pipeline.d2` | D2 | Phase 1(QLoRA+STE)→Phase 2(three-factor)→Phase 3(online) |
| 11 | `t4dm_memory_lifecycle.d2` | D2 | INSERT(τ)→MemTable(κ=0)→Flush→L0→NREM(κ+=0.05)→L1→REM(κ+=0.2)→L2→Prune/Semantic(κ→1.0) |

### Should Update (3 documentation files)

| # | File | Issue | Action |
|---|------|-------|--------|
| 12 | `system_architecture.md` | Storage section, tech stack list Neo4j/Qdrant | Update storage section + tech stack |
| 13 | `memory_subsystems.md` | References Neo4j graph storage | Update storage refs, add κ-gradient explanation |
| 14 | `DIAGRAM_SUMMARY.md` | References "T4DM", old file list | Rewrite with new diagram inventory |

### Should Deprecate (4 items)

| # | File | Reason |
|---|------|--------|
| 15 | `storage_resilience.md` | References Neo4j+Qdrant Saga pattern |
| 16 | `storage_patterns.md` | References old query patterns |
| 17 | `module_dependencies.md` | Missing spiking/qwen/t4dx modules |
| 18 | `generate_diagrams.py` | Matplotlib-based PNG generator, replaced by D2+Mermaid |

---

## Execution Order

### Batch 1: New D2 diagrams (create first — these define the target architecture)
1. `t4dx_storage_engine.d2` — T4DX internals (most referenced by other updates)
2. `t4dm_full_system.d2` — Complete system overview
3. `t4dm_data_flow.d2` — End-to-end data flow
4. `t4dm_training_pipeline.d2` — Training phases
5. `t4dm_memory_lifecycle.d2` — κ progression through compaction

### Batch 2: Rewrite outdated Mermaid diagrams
6. `05_storage_architecture.mmd` — T4DX replaces dual-store
7. `14_storage_subsystem.mmd` — T4DX subsystem detail
8. `23_class_storage.mmd` — New UML for T4DX classes

### Batch 3: Update existing diagrams
9. `system_architecture.md` — Update storage + add spiking/Qwen
10. `21_class_memory.mmd` — Replace store references
11. `31_state_circuit_breaker.mmd` — Deprecate or repurpose

### Batch 4: Documentation updates
12. `memory_subsystems.md` — Add κ-gradient, remove Neo4j refs
13. `consolidation_flow.md` — Map consolidation ops to T4DX compactor
14. `DIAGRAM_SUMMARY.md` — Full rewrite

### Batch 5: Deprecation
15. Add `DEPRECATED` header to: `storage_resilience.md`, `storage_patterns.md`, `module_dependencies.md`
16. Keep `generate_diagrams.py` for historical reference, add deprecation comment

---

## Rendering

D2 diagrams: `d2 input.d2 output.svg` and `d2 input.d2 output.png`
Mermaid diagrams: `mmdc -i input.mmd -o output.svg` and `mmdc -i input.mmd -o output.png`

All rendered files should be committed alongside sources for GitHub preview.

---

## Diagram Content Specifications

### t4dx_storage_engine.d2 (New)
```
Top level: T4DX Engine box
  Write Path: Item → WAL.append → MemTable.insert
  Read Path: Query → MemTable → Segments (newest→oldest)

  MemTable box:
    items[], vectors[], edges{}, id_idx{}, kappa_sorted
    field_overlays{}, edge_deltas{}
    → flush trigger (10K items or timer)

  Segments box (×N, stacked):
    Per segment: hnsw.bin, vectors.npy, kappa.npy, edges.npz, items.npy, vardata.bin, bloom.bin
    Summary stats: time_min/max, κ_min/max, count

  Global Structures box:
    id_map{}, cross_edges CSR, manifest[], tombstones{}

  WAL box:
    Append-only, 64MB segments, HMAC-SHA256, fsync

  Compactor box (= Consolidation):
    NREM: merge L0→L1, apply κ+Δw
    REM: cluster L1→L2, create prototypes
    PRUNE: rewrite dropping tombstones + low-κ

  Arrows: WAL→MemTable, MemTable→Segments (flush), Segments→Compactor, Compactor→Segments (new)
```

### t4dm_full_system.d2 (New)
```
Left: Input (tokens)
Center top: Qwen 3B (frozen, 4-bit NF4) — layers 0-17 with QLoRA adapters
Center: Spiking Cortical Stack (6 blocks, trainable)
  Each block: Thalamic→LIF→SpikeAttn→Apical→RWKV→Residual
Center bottom: Qwen 3B layers 18-35 with QLoRA adapters → LM head
Right: Output (tokens)

Left rail: Neuromodulator Bus (DA→STDP LR, NE→gain, ACh→gate, 5-HT→patience)
Right rail: Oscillatory Scaffold (θ 4-8Hz, γ 30-100Hz, δ 0.5-4Hz)

Bottom: T4DX Engine (MemTable → Segments)
  ↔ bidirectional with spiking stack (SEARCH for context, INSERT gated by τ(t))

Below T4DX: Compactor (NREM/REM/PRUNE) triggered by Sleep Scheduler (adenosine pressure)

Sidebar: Memory Projections (2048↔1024) connecting Qwen hidden states to T4DX vectors
```

### t4dm_data_flow.d2 (New)
```
Horizontal flow:
Token → Tokenizer → Qwen Embedding → Qwen Layers 0-17 (QLoRA) →
  → mem_encoder(2048→1024) → T4DX SEARCH → mem_decoder(1024→2048) →
  → Spiking Cortical Stack (6 blocks) →
  → Qwen Layers 18-35 (QLoRA) → LM Head → Output Token

Vertical branches from spiking stack:
  ↓ τ(t) gate → T4DX INSERT (if τ > threshold)
  ↓ STDP → T4DX UPDATE_EDGE_WEIGHT
  ↓ Learning signals → T4DX UPDATE_FIELDS (κ, importance)

Bottom: T4DX segments with compaction arrows
```

### t4dm_training_pipeline.d2 (New)
```
Three columns:

Phase 1 (Supervised):
  Data: instruction tuning corpus
  Trainable: QLoRA(15M) + spiking(50-80M) + projections
  Loss: CE + FF goodness
  Optimizer: AdamW, mixed precision
  Gradient: backprop through QLoRA + STE through LIF

Phase 2 (Local Learning):
  Data: online experience
  Trainable: spiking params only (QLoRA optionally frozen)
  Rule: three-factor (pre×post×neuromod)
  STDP: eligibility traces + DA/NE modulation
  No backprop through Qwen

Phase 3 (Online):
  Data: live conversations
  Learning: three-factor + consolidation (sleep cycles)
  κ progression: 0→0.15→0.4→0.85→1.0
  Consolidation: NREM replay, REM clustering, PRUNE
```

### t4dm_memory_lifecycle.d2 (New)
```
Horizontal flow with κ axis:

κ=0.0: INSERT (τ gate) → MemTable (labile, mutable)
  ↓ flush (10K items)
κ=0.0: Segment L0 (immutable)
  ↓ NREM compact (replay, STDP, κ+=0.05 per replay)
κ=0.15: Segment L1
  ↓ REM compact (HDBSCAN cluster, create prototypes, κ+=0.2)
κ=0.85: Segment L2 (semantic concepts, item_type→semantic)
  ↓ repeated consolidation
κ=1.0: Stable knowledge

Branch at any point: κ < θ_forget → PRUNE (tombstone + GC)
Branch: access → UPDATE_FIELDS(access_count++) → FSRS decay adjustment
```
