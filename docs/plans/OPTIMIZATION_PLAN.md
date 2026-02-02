# T4DM Optimization & Production Readiness Plan

**Generated**: 2026-02-02
**Scope**: Full codebase analysis → atomic execution plan
**Agents**: 7 parallel opus analyses (architecture, diagrams, competitive, neurobiology, storage, visualization, integration)

---

## Executive Summary

T4DM is a 155,724-line Python codebase with 334 source files, 9,600+ passing tests, and 81% coverage. It implements a biologically-inspired memory system that is architecturally unique in the AI memory market — no competitor combines spiking neural adapters, continuous κ-gradient consolidation, custom LSM storage, and neuromodulator dynamics.

**Current State**: All 58 plan atoms (P1-P7) implemented. All 70 optimization atoms (A-G) implemented. Neo4j/Qdrant/Saga removed. T4DX embedded engine operational. 9,600+ tests passing.

**Key Findings**:
- 48 of ~57 diagrams are outdated (Neo4j/Qdrant refs, biological errors, parameter mismatches)
- T4DX storage engine missing 5 planned components (HNSW, CSR graph, kappa index, bitemporal, provenance)
- 7 visualization modules needed for new architecture (spiking, kappa, T4DX metrics, Qwen adapter)
- 0 framework adapters exist (LangChain, LlamaIndex, AutoGen, CrewAI)
- GraphStore protocol still references Cypher (Neo4j legacy)
- Dual WAL systems (T4DX simple + legacy binary) not unified
- Cerebellum is the largest neuroscience gap
- NCA module name is misleading (it's brain region simulation, not cellular automata)

---

## Completion Status

All 70 optimization atoms across phases A-G have been COMPLETED in 4 sprints:

### Sprint 1 (commit 46f78c6) - 21 atoms
- **A-01**: Remove legacy storage file references
- **A-02**: Fix GraphStore.query() Cypher parameter
- **A-03**: Fix archive.py datetime.timedelta bug
- **A-04**: Unify dual WAL systems
- **A-05**: Update package docstring "World Weaver" → "T4DM"
- **A-06**: Fix VectorAdapter stubs (scroll, count)
- **A-07**: Add threading locks to T4DXEngine
- **A-08**: Fix visualization.py Neo4j comment
- **A-09**: Update observability/CLAUDE.md references
- **A-10**: Update src/t4dm/CLAUDE.md references
- **A-11**: Update storage/CLAUDE.md for T4DX
- **A-12**: Clean remaining "World Weaver" / "ww" references
- **C-01**: Create 01_system_architecture
- **C-02**: Create 05_storage_architecture
- **C-03**: Create 14_storage_subsystem
- **C-04**: Create 23_class_storage
- **C-05**: Render all D2 diagrams to PNG/SVG
- **C-06**: Fix GP→Thalamus GABA correction
- **C-07**: Fix hippocampal_circuit biological errors
- **C-08**: Fix ff_nca_coupling per-layer goodness
- **C-09**: Fix spindle_ripple_coupling causality

### Sprint 2 (commit 635a885) - 29 atoms
- **B-01**: Implement HNSW vector index
- **B-02**: Implement CSR graph structure
- **B-03**: Implement kappa_index.py
- **B-04**: Implement bitemporal.py
- **B-05**: Implement provenance.py
- **B-06**: Add WAL group commit
- **B-07**: Add bloom filters to segments
- **C-10**: Fix hpc_trisynaptic (BLA, mossy cells)
- **C-11**: Fix swr_replay parameters
- **C-12**: Fix vta_circuit burst frequency
- **C-13**: Rewrite system_architecture.md (T4DX)
- **C-14**: Rewrite storage_resilience.md (no Saga)
- **C-15**: Update 41_seq_store_memory.mmd
- **C-16**: Update persistence diagrams
- **C-17**: Update README.md (remove Neo4j/Qdrant)
- **C-18**: Create kappa_gradient_consolidation diagram
- **C-19**: Create tau_temporal_gate diagram
- **C-20**: Create sdk_client_architecture diagram
- **C-21**: Create plugin_integration_map diagram
- **C-22**: Update DIAGRAM_SUMMARY.md

### Sprint 3 (commit 7c06887) - 16 atoms
- **E-01**: Simple 3-line API (sdk/simple.py)
- **E-02**: LangChain BaseMemory adapter
- **E-03**: LlamaIndex VectorStore adapter
- **E-04**: AutoGen Memory protocol adapter
- **E-05**: CrewAI Memory adapter
- **E-06**: Mem0-compatible REST API shim
- **E-07**: OpenAI function tool schemas
- **E-08**: Rename MCP tools (ww→t4dm)
- **F-01**: Update BRAIN_REGION_MAPPING.md
- **F-02**: Add Frankland & Bontempi 2005 citation
- **F-03**: Create NEUROSCIENCE_TAXONOMY.md
- **F-04**: Create COMPETITIVE_ANALYSIS.md
- **F-05**: Create INTEGRATION_GUIDE.md
- **F-06**: Document biological fidelity levels
- **F-07**: Rename NCA module references
- **F-08**: Update FULL_SYSTEM_PLAN.md

### Sprint 4 (commit 97c0360) - 13 atoms
- **D-01**: Kappa gradient visualizer
- **D-02**: T4DX storage metrics
- **D-03**: Spiking neuron dashboard
- **D-04**: Qwen adapter metrics
- **D-05**: Neuromod bus per-layer viz
- **D-06**: Oscillator bias viz
- **D-07**: Update persistence_state.py
- **G-01**: Add LongMemEval benchmark runner
- **G-02**: Add concurrent access tests for T4DX
- **G-03**: Add performance benchmarks
- **G-04**: Add checkpoint_v3 and recovery_v2 tests
- **G-05**: Implement WebSocket streaming
- **G-06**: Add cerebellar module

---

## Competitive Position

| System | Architecture | T4DM Advantage |
|--------|-------------|----------------|
| **Mem0** ($24M) | LLM extraction + vector DB | Trainable params, consolidation, temporal |
| **Letta/MemGPT** | Tiered + LLM self-edit | Biological consolidation > sleep-time compute |
| **Zep** | Temporal knowledge graph | Bitemporal + spiking + neuromod |
| **Supermemory** | RAG + knowledge graph | Custom engine, STDP learning |
| **LangMem** | Framework memory primitives | Full system vs primitives |

**Unique differentiators (no competitor has)**:
1. Spiking neural adapter on frozen LLM (QLoRA + cortical blocks)
2. LSM compaction = biological consolidation
3. Continuous κ-gradient (not discrete memory types)
4. Neuromodulator dynamics (DA, NE, ACh, 5-HT)
5. Bitemporal queries ("what did we know when")
6. 50-80M trainable memory parameters
7. Custom embedded storage engine (zero network hops)

---

## Neurobiology Taxonomy

### Level 1: Brain Regions → Modules

| Brain Region | Module | Files | Accuracy |
|-------------|--------|-------|----------|
| Hippocampus (DG/CA3/CA1) | nca/hippocampus.py, encoding/sparse.py, encoding/attractor.py | 1,268 + 276 + 486 lines | Excellent |
| Neocortex (PFC) | nca/wm_gating.py, nca/theta_gamma_integration.py | 736 + 320 lines | Good |
| Thalamus | spiking/thalamic_gate.py, nca/sleep_spindles.py | 39 + 608 lines | Good |
| Basal Ganglia | nca/striatal_msn.py, nca/substantia_nigra.py | 939 + 429 lines | Excellent |
| Amygdala | nca/amygdala.py | 574 lines | Good |
| VTA (DA) | nca/vta.py, learning/dopamine.py | 868 + 970 lines | Excellent |
| Locus Coeruleus (NE) | nca/locus_coeruleus.py, learning/norepinephrine.py | 1,102 + 464 lines | Excellent |
| Raphe (5-HT) | nca/raphe.py, learning/serotonin.py | 968 + 554 lines | Good |
| Nucleus Basalis (ACh) | nca/nucleus_basalis.py, learning/acetylcholine.py | 554 + 501 lines | Good |
| Astrocytes | nca/astrocyte.py | 518 lines | Impressive |
| Glymphatic | nca/glymphatic.py | 740 lines | Novel |
| **Cerebellum** | **NOT IMPLEMENTED** | -- | **Critical gap** |

### Level 2: Neural Mechanisms → Algorithms

| Mechanism | Algorithm | Accuracy |
|-----------|-----------|----------|
| LTP/LTD | Bounded Hebbian + BCM sliding threshold | Simplified but stable |
| STDP | Exponential window (τ+=17ms, τ-=34ms) + DA modulation | Accurate (Bi & Poo 1998) |
| SWR replay | 10x compressed, PE-weighted, forward/reverse | Good (Foster & Wilson 2006) |
| Theta-gamma | PAC with WM capacity estimation | Good (Lisman & Jensen 2013) |
| Pattern separation | 8x expansion + k-WTA (4% sparsity) | Accurate (Treves & Rolls 1994) |
| Pattern completion | Modern Hopfield attractor | Good (Ramsauer 2020) |
| Three-factor | Eligibility × neuromod × DA surprise | Accurate (Frémaux & Gerstner 2016) |
| Memory consolidation | κ gradient [0,1] via LSM compaction | Novel and defensible |

### Level 3: Key Equations

| Model | Formula | Reference |
|-------|---------|-----------|
| LIF neuron | `u(t+1) = α·u(t) + I(t)`, ATan surrogate | Zenke & Ganguli 2018 |
| FSRS decay | `R(t,S) = (1 + 0.9·t/S)^(-0.5)` | Wixted & Ebbesen 1991 |
| τ(t) gate | `τ(t) = σ(λ_ε·ε + λ_Δ·novelty + λ_r·reward)` | Novel |
| ACT-R activation | `A_i = ln(Σ t_j^(-d)) + Σ W_j·S_ji + ε` | Anderson 2007 |

### Terminology Corrections Needed

1. **NCA module** → Rename to `circuits/` or `brain_regions/` (not cellular automata)
2. **"Neuromodulator bus"** → Document as metaphorical (neuroscience: "ascending modulatory system")
3. **"Spiking cortical block"** → Document as functional abstraction, not cortical column model
4. **κ gradient** → Add Frankland & Bontempi (2005) citation for systems consolidation basis
5. **BRAIN_REGION_MAPPING.md** → Outdated: marks amygdala, sleep spindles, LC phasic as "missing" when implemented

---

## Atomic Execution Plan

### Phase A: Code Cleanup & Dead Code Removal (12 atoms)

| ID | Task | Files | Effort |
|----|------|-------|--------|
| A-01 | Remove legacy storage files still referenced in CLAUDE.md docs | storage/CLAUDE.md, src/t4dm/CLAUDE.md | S |
| A-02 | Fix GraphStore.query() — remove Cypher parameter, replace with generic query dict | core/protocols.py, storage/t4dx/graph_adapter.py | M |
| A-03 | Fix archive.py bug: `datetime.timedelta` → `timedelta` (line 396) | storage/archive.py | S |
| A-04 | Remove dual WAL: unify T4DX WAL with persistence/wal.py binary format | storage/t4dx/wal.py, persistence/wal.py | L |
| A-05 | Update package docstring "World Weaver" → "T4DM" in `__init__.py` | src/t4dm/__init__.py | S |
| A-06 | Fix VectorAdapter stubs: `scroll()` returns empty, `count()` returns 0 | storage/t4dx/vector_adapter.py | M |
| A-07 | Add threading locks to T4DXEngine for concurrent access safety | storage/t4dx/engine.py | M |
| A-08 | Fix visualization.py line 2082 Neo4j comment | api/routes/visualization.py | S |
| A-09 | Update observability/CLAUDE.md Neo4j/Qdrant reference | observability/CLAUDE.md | S |
| A-10 | Update src/t4dm/CLAUDE.md: remove Neo4j/Qdrant/Saga references, fix "ww" naming | src/t4dm/CLAUDE.md | M |
| A-11 | Update storage/CLAUDE.md: rewrite for T4DX architecture | storage/CLAUDE.md | M |
| A-12 | Clean remaining "World Weaver" / "ww" references in Python docstrings | grep -r "World Weaver\|ww\." src/ | M |

### Phase B: T4DX Storage Engine Completion (7 atoms)

| ID | Task | Files | Effort |
|----|------|-------|--------|
| B-01 | Implement HNSW vector index in SegmentReader (replace brute-force) | storage/t4dx/segment.py, new: hnsw.py | L |
| B-02 | Implement CSR graph structure for edge traversal (replace flat lists) | storage/t4dx/segment.py, memtable.py | L |
| B-03 | Implement kappa_index.py — secondary index for κ-range queries | new: storage/t4dx/kappa_index.py | M |
| B-04 | Implement bitemporal.py — "what did we know when" query planner | new: storage/t4dx/bitemporal.py | L |
| B-05 | Implement provenance.py — forward/backward lineage trace | new: storage/t4dx/provenance.py | M |
| B-06 | Add WAL group commit (batch multiple ops between fsyncs) | storage/t4dx/wal.py | M |
| B-07 | Add bloom filters to segments for faster negative GET | storage/t4dx/segment.py | M |

### Phase C: Diagram Overhaul (22 atoms)

#### C1: Architecture diagrams (critical — replace deleted/outdated)

| ID | Task | Format | Priority |
|----|------|--------|----------|
| C-01 | Create 01_system_architecture — Qwen+Spiking+T4DX+Neuromod top-level | Mermaid | CRITICAL |
| C-02 | Create 05_storage_architecture — T4DX LSM engine | Mermaid | CRITICAL |
| C-03 | Create 14_storage_subsystem — T4DX subsystem decomposition | Mermaid | HIGH |
| C-04 | Create 23_class_storage — T4DX class diagram | Mermaid | HIGH |
| C-05 | Render all 9 D2 diagrams to PNG/SVG | D2 → PNG/SVG | HIGH |

#### C2: Biological corrections (from DIAGRAM_RECONCILIATION_PLAN.md)

| ID | Task | File | Issue |
|----|------|------|-------|
| C-06 | Fix GP→Thalamus: Glu → GABA | 57_connectome_regions.mermaid | CRITICAL bio error |
| C-07 | Fix hippocampal_circuit: grid→place reversed, missing BLA, wrong ACh | hippocampal_circuit.mermaid | CRITICAL bio error |
| C-08 | Fix ff_nca_coupling: single global goodness → per-layer | ff_nca_coupling.mermaid | CRITICAL bio error |
| C-09 | Fix spindle_ripple_coupling: wrong causality direction | spindle_ripple_coupling.mermaid | CRITICAL bio error |
| C-10 | Fix hpc_trisynaptic: add BLA, mossy cells, EC feedback | hpc_trisynaptic.mermaid | HIGH |
| C-11 | Fix swr_replay: 20x→10x, add ACh suppression, 90% reverse | swr_replay.mermaid | HIGH |
| C-12 | Fix vta_circuit: burst 15-25→30 Hz, add PHASIC_PAUSE | vta_circuit.mermaid | HIGH |

#### C3: Architecture updates (Neo4j/Qdrant removal)

| ID | Task | File |
|----|------|------|
| C-13 | Rewrite system_architecture.md — remove Neo4j+Qdrant | docs/diagrams/system_architecture.md |
| C-14 | Rewrite storage_resilience.md — remove Saga | docs/diagrams/storage_resilience.md |
| C-15 | Update 41_seq_store_memory.mmd — T4DX flow | docs/diagrams/41_seq_store_memory.mmd |
| C-16 | Update persistence diagrams (5 files) — T4DX WAL/checkpoint/recovery | persistence_layer.mmd + 4 more |
| C-17 | Update README.md — remove Neo4j/Qdrant/Saga refs | docs/diagrams/README.md |

#### C4: New diagrams

| ID | Task | Format |
|----|------|--------|
| C-18 | Create kappa_gradient_consolidation — κ 0.0→1.0 through LSM compaction | Mermaid |
| C-19 | Create tau_temporal_gate — τ(t) write gating inputs/outputs | Mermaid |
| C-20 | Create sdk_client_architecture — 3-tier SDK (REST→Agent→WWAgent) | Mermaid |
| C-21 | Create plugin_integration_map — hooks, adapters, MCP, framework ports | Mermaid |
| C-22 | Update DIAGRAM_SUMMARY.md — reflect current state | Markdown |

### Phase D: Visualization Suite Expansion (7 atoms)

| ID | Task | New File | Data Source |
|----|------|----------|-------------|
| D-01 | Kappa gradient visualizer — κ distribution, consolidation flow | visualization/kappa_gradient.py | storage/t4dx (ItemRecord.kappa) |
| D-02 | T4DX storage metrics — LSM compaction stats, MemTable flush rates, segment counts | visualization/t4dx_metrics.py | storage/t4dx/engine.py |
| D-03 | Spiking neuron dashboard — LIF membrane potentials, spike rasters, gate states | visualization/spiking_dynamics.py | spiking/*.py |
| D-04 | Qwen adapter metrics — hidden state norms, QLoRA weight norms, projection stats | visualization/qwen_metrics.py | qwen/*.py |
| D-05 | Neuromod bus per-layer — DA/NE/ACh/5-HT injection per cortical block layer | visualization/neuromod_layers.py | spiking/neuromod_bus.py |
| D-06 | Oscillator bias viz — theta/gamma/delta phase injection into spiking blocks | visualization/oscillator_injection.py | spiking/oscillator_bias.py |
| D-07 | Update persistence_state.py — add T4DX WAL/segment metrics | visualization/persistence_state.py | storage/t4dx/wal.py |

### Phase E: Plugin & Framework Adapters (8 atoms)

| ID | Task | New File | Effort |
|----|------|----------|--------|
| E-01 | Simple 3-line API: `T4DM(url).add(text)` / `.search(query)` / `.get_all()` | sdk/simple.py | S |
| E-02 | LangChain BaseMemory adapter | adapters/langchain.py | M |
| E-03 | LlamaIndex VectorStore adapter | adapters/llamaindex.py | M |
| E-04 | AutoGen Memory protocol adapter | adapters/autogen.py | M |
| E-05 | CrewAI Memory adapter | adapters/crewai.py | S |
| E-06 | Mem0-compatible REST API shim (`/v1/memories/` CRUD) | api/routes/compat.py | M |
| E-07 | OpenAI function tool schemas (JSON) | schemas/openai_tools.json | S |
| E-08 | Rename MCP tools: `ww_store`→`t4dm_store`, etc. | mcp/server.py, mcp/tools.py | S |

### Phase F: Documentation & Taxonomy (8 atoms)

| ID | Task | File |
|----|------|------|
| F-01 | Update BRAIN_REGION_MAPPING.md — mark implemented modules correctly | docs/BRAIN_REGION_MAPPING.md |
| F-02 | Add Frankland & Bontempi 2005 citation to κ gradient docs | docs/MEMORY_ARCHITECTURE.md |
| F-03 | Create NEUROSCIENCE_TAXONOMY.md — formal 4-level mapping | new: docs/NEUROSCIENCE_TAXONOMY.md |
| F-04 | Create COMPETITIVE_ANALYSIS.md — positioning doc | new: docs/COMPETITIVE_ANALYSIS.md |
| F-05 | Create INTEGRATION_GUIDE.md — how to plug T4DM into agent frameworks | new: docs/guides/INTEGRATION_GUIDE.md |
| F-06 | Document biological fidelity levels per module (circuit/mechanism/functional/engineering) | docs/NEUROSCIENCE_TAXONOMY.md |
| F-07 | Rename NCA module references in docs → "brain circuits" / "neural circuits" | All CLAUDE.md + docs referencing NCA |
| F-08 | Update FULL_SYSTEM_PLAN.md — mark completed atoms, add new phases | docs/plans/FULL_SYSTEM_PLAN.md |

### Phase G: Production Hardening (6 atoms)

| ID | Task | Scope |
|----|------|-------|
| G-01 | Add LongMemEval benchmark runner (competitive credibility) | new: benchmarks/ |
| G-02 | Add concurrent access tests for T4DX engine | tests/unit/storage/ |
| G-03 | Add performance benchmarks for search at 10K/100K/1M items | tests/performance/ |
| G-04 | Add checkpoint_v3 and recovery_v2 unit tests | tests/unit/persistence/ |
| G-05 | Implement WebSocket streaming for real-time visualization | api/websocket.py → visualization modules |
| G-06 | Add cerebellar module (timing, error-driven learning, predictive models) | new: nca/cerebellum.py |

---

## Execution Priority

### Sprint 1: Foundation (A + C critical)
**Goal**: Clean codebase, fix critical diagram errors
- A-01 through A-12 (code cleanup)
- C-01 through C-05 (critical architecture diagrams)
- C-06 through C-09 (critical biological corrections)

### Sprint 2: Storage & Diagrams (B + C remaining)
**Goal**: Complete T4DX, update all diagrams
- B-01 through B-07 (T4DX completion)
- C-10 through C-22 (remaining diagrams)

### Sprint 3: Integration (E + F)
**Goal**: Framework adapters, documentation
- E-01 through E-08 (plugin adapters)
- F-01 through F-08 (documentation)

### Sprint 4: Visualization & Hardening (D + G)
**Goal**: New viz modules, benchmarks, production readiness
- D-01 through D-07 (visualization expansion)
- G-01 through G-06 (production hardening)

---

## Totals

| Phase | Atoms | Description | Status |
|-------|-------|-------------|--------|
| A: Code Cleanup | 12 | Dead code, bugs, naming, WAL unification | COMPLETED (Sprint 1) |
| B: T4DX Storage | 7 | HNSW, CSR, kappa index, bitemporal, provenance | COMPLETED (Sprint 2) |
| C: Diagrams | 22 | Architecture, biological fixes, Neo4j removal, new | COMPLETED (Sprints 1-2) |
| D: Visualization | 7 | Kappa, T4DX, spiking, Qwen, neuromod, oscillator | COMPLETED (Sprint 4) |
| E: Adapters | 8 | LangChain, LlamaIndex, AutoGen, CrewAI, Mem0, MCP | COMPLETED (Sprint 3) |
| F: Documentation | 8 | Taxonomy, competitive analysis, integration guide | COMPLETED (Sprint 3) |
| G: Production | 6 | Benchmarks, concurrency, cerebellum | COMPLETED (Sprint 4) |
| **Total** | **70** | | **ALL COMPLETED** |
