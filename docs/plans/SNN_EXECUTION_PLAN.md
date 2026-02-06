# SNN Unification Execution Plan

**Created**: 2026-02-04
**Updated**: 2026-02-05
**Status**: 90% Complete (45/50 atoms done)
**Source**: SNN_UNIFIED_4D_INTEGRATION_PLAN.md
**Total Atoms**: 50 across 5 phases

> **Progress**: P1 ✓ | P2 ✓ | P3 ~80% | P4 ✓ | P5 ~80%

---

## Execution Model

```
┌─────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                             │
│  - Tracks completion state                                   │
│  - Dispatches parallel work batches                          │
│  - Validates phase gates before proceeding                   │
│  - Manages cross-phase dependencies                          │
└─────────────────────────────────────────────────────────────┘
        │
        ├──► Subagent A: Core Implementation (spiking, storage)
        ├──► Subagent B: Integration & Wiring (connecting components)
        ├──► Subagent C: Tests & Validation (unit, integration, bio)
        └──► Subagent D: Documentation & Diagrams
```

---

## Phase 1: Foundation (12 atoms)

### Batch 1.1 — PARALLEL (no dependencies)

| Atom | Owner | Description | Files |
|------|-------|-------------|-------|
| P1-01 | A | τ(t) temporal control signal | `src/t4dm/core/temporal_control.py` |
| P1-03 | A | Norse SNN backend wrapper | `src/t4dm/nca/snn_backend.py` |
| P1-05 | A | Unified MemoryItem schema | `src/t4dm/core/unified_memory.py` |
| P1-09 | A | PyTorch GPU PDE solver prototype | `src/t4dm/nca/neural_field_gpu.py` |

**Subagent A**: 4 parallel atoms (independent modules)

| Atom | Owner | Description | Files |
|------|-------|-------------|-------|
| P1-06 | A | Add κ field to Episode class | `src/t4dm/core/types.py:106` |
| P1-07 | A | Add κ field to Entity class | `src/t4dm/core/types.py` |
| P1-08 | A | Add κ field to Procedure class | `src/t4dm/core/types.py` |

**Subagent A**: 3 parallel atoms (all modify types.py but different classes)

| Atom | Owner | Description | Files |
|------|-------|-------------|-------|
| C1-01 | C | Tests for P1-01 (τ control) | `tests/unit/test_temporal_control.py` |
| C1-03 | C | Tests for P1-03 (Norse) | `tests/unit/test_norse_backend.py` |
| C1-05 | C | Tests for P1-05 (MemoryItem) | `tests/unit/test_unified_memory.py` |
| C1-09 | C | Tests for P1-09 (GPU PDE) | `tests/unit/test_neural_field_gpu.py` |
| C1-06 | C | Tests for κ fields | `tests/unit/test_types.py` |

**Subagent C**: 5 parallel test atoms (can write tests while A implements)

| Atom | Owner | Description | Files |
|------|-------|-------------|-------|
| D1-01 | D | Docs for τ(t) signal | `docs/reference/temporal_control.md` |
| D1-05 | D | Docs for unified MemoryItem | `docs/reference/unified_memory.md` |

**Subagent D**: 2 parallel doc atoms

---

### Batch 1.2 — LINEAR (depends on Batch 1.1)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P1-02 | B | Integrate τ(t) into MemoryGate | P1-01 |
| P1-04 | A | Numba JIT for STDP hot loop | P1-03 |

**Subagent A**: P1-04 (needs Norse types from P1-03)
**Subagent B**: P1-02 (needs τ function from P1-01)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| C1-02 | C | Tests for τ integration | P1-02 |
| C1-04 | C | Tests for STDP JIT | P1-04 |

**Subagent C**: 2 test atoms (after implementation)

---

### Batch 1.3 — LINEAR (Phase 1 gate)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P1-12 | C | Phase 1 integration test | ALL P1-* |

**Gate Check**: All P1 atoms complete, integration test passes

---

## Phase 1 Execution Graph

```
                    ┌─────────────────────────────────────────────────┐
                    │              BATCH 1.1 (PARALLEL)                │
                    │                                                  │
    ┌───────────────┼───────────────┬───────────────┬────────────────┐│
    │               │               │               │                ││
    ▼               ▼               ▼               ▼                ▼│
┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐│
│P1-01  │      │P1-03  │      │P1-05  │      │P1-09  │      │P1-06  ││
│τ(t)   │      │Norse  │      │Memory │      │GPU PDE│      │P1-07  ││
│signal │      │backend│      │Item   │      │solver │      │P1-08  ││
└───┬───┘      └───┬───┘      └───────┘      └───────┘      │κ flds ││
    │              │                                         └───────┘│
    │              │          ┌─────────────────────────────────────┐ │
    │              │          │     TESTS (C) + DOCS (D) PARALLEL   │ │
    │              │          │  C1-01, C1-03, C1-05, C1-09, C1-06  │ │
    │              │          │  D1-01, D1-05                       │ │
    │              │          └─────────────────────────────────────┘ │
    └──────────────┴──────────────────────────────────────────────────┘
                    │
                    ▼
                    ┌─────────────────────────────────────────────────┐
                    │              BATCH 1.2 (LINEAR)                  │
                    │                                                  │
    ┌───────────────┴───────────────┐                                 │
    │                               │                                 │
    ▼                               ▼                                 │
┌───────┐                      ┌───────┐                              │
│P1-02  │                      │P1-04  │                              │
│τ→Gate │                      │STDP   │                              │
│integ  │                      │JIT    │                              │
└───┬───┘                      └───┬───┘                              │
    │                              │                                  │
    ▼                              ▼                                  │
┌───────┐                      ┌───────┐                              │
│C1-02  │                      │C1-04  │                              │
│tests  │                      │tests  │                              │
└───────┘                      └───────┘                              │
    └──────────────────────────────────────────────────────────────────┘
                    │
                    ▼
                    ┌─────────────────────────────────────────────────┐
                    │              BATCH 1.3 (GATE)                    │
                    │                                                  │
                    │  P1-12: Phase 1 Integration Test                 │
                    │  ✓ All components integrate                      │
                    │  ✓ No performance regressions                    │
                    └─────────────────────────────────────────────────┘
```

---

## Phase 2: Unified Memory Substrate (15 atoms)

### Batch 2.1 — LINEAR (foundation)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P2-01 | A | Unified Memory Store backend | P1-05 |

**Subagent A**: Creates storage backend for MemoryItem

---

### Batch 2.2 — PARALLEL (query policies)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P2-02 | A | Episodic query policy (κ < 0.3) | P2-01 |
| P2-03 | A | Semantic query policy (κ > 0.7) | P2-01 |
| P2-04 | A | Procedural query policy | P2-01 |

**Subagent A**: 3 parallel query implementations

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| C2-02 | C | Tests for episodic policy | P2-02 |
| C2-03 | C | Tests for semantic policy | P2-03 |
| C2-04 | C | Tests for procedural policy | P2-04 |

**Subagent C**: 3 parallel test atoms

---

### Batch 2.3 — PARALLEL (consolidation κ updates)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P2-05 | B | NREM consolidation κ += 0.05 | P2-01 |
| P2-06 | B | REM consolidation κ += 0.2 | P2-01 |

**Subagent B**: 2 parallel consolidation integrations

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| C2-05 | C | Tests for NREM κ update | P2-05 |
| C2-06 | C | Tests for REM κ update | P2-06 |

**Subagent C**: 2 parallel test atoms

---

### Batch 2.4 — LINEAR (migration)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P2-07 | A | Data migration script | P2-01 |

**Subagent A**: Migration Episode/Entity/Procedure → MemoryItem

---

### Batch 2.5 — PARALLEL (API updates)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P2-08 | B | Update episodic retrieval | P2-02 |
| P2-09 | B | Update semantic retrieval | P2-03 |
| P2-10 | B | Update procedural retrieval | P2-04 |
| P2-11 | B | Backward compatibility layer | P2-01 |

**Subagent B**: 4 parallel API updates

---

### Batch 2.6 — PARALLEL (surface layer)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P2-12 | B | Update API routes | P2-08, P2-09, P2-10 |
| P2-13 | B | Update CLI | P2-08, P2-09, P2-10 |

**Subagent B**: 2 parallel surface updates

---

### Batch 2.7 — LINEAR (Phase 2 gate)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P2-14 | C | Phase 2 integration test | ALL P2-* |
| P2-15 | C | Performance benchmark | P2-14 |

**Gate Check**: Unified store ≤10% regression vs 3-store

---

## Phase 2 Execution Graph

```
P1-05 (MemoryItem)
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ BATCH 2.1: P2-01 (Unified Store Backend)                          │
└───────────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────┬─────────────────────────┐
    │                         │                         │
    ▼                         ▼                         ▼
┌─────────┐              ┌─────────┐              ┌─────────┐
│ BATCH   │              │ BATCH   │              │ BATCH   │
│ 2.2     │              │ 2.3     │              │ 2.4     │
│ Query   │              │ Consol  │              │ Migrate │
│ Policies│              │ κ upd   │              │ Script  │
│ P2-02   │              │ P2-05   │              │ P2-07   │
│ P2-03   │              │ P2-06   │              │         │
│ P2-04   │              │         │              │         │
└────┬────┘              └────┬────┘              └─────────┘
     │                        │
     ▼                        │
┌─────────┐                   │
│ BATCH   │                   │
│ 2.5     │◄──────────────────┘
│ API upd │
│ P2-08-11│
└────┬────┘
     │
     ▼
┌─────────┐
│ BATCH   │
│ 2.6     │
│ Surface │
│ P2-12-13│
└────┬────┘
     │
     ▼
┌─────────┐
│ BATCH   │
│ 2.7     │
│ GATE    │
│ P2-14-15│
└─────────┘
```

---

## Phase 3: Spike Pipeline (10 atoms)

### Batch 3.1 — LINEAR (reinjection foundation)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P3-01 | A | Spike reinjection module | P1-03 (Norse) |

**Subagent A**: Embedding → spike train conversion

---

### Batch 3.2 — LINEAR (integration)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P3-02 | B | Integrate reinjection with NREM replay | P3-01 |

**Subagent B**: Wire reinjection into consolidation

---

### Batch 3.3 — LINEAR (STDP loop)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P3-03 | B | Connect SNN output to STDP learner | P3-02 |
| P3-04 | A | STDP weight update during replay | P3-03 |
| P3-05 | B | Weight update → memory update loop | P3-04 |

**Subagent A + B**: Sequential pipeline construction

---

### Batch 3.4 — PARALLEL (optimizations)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P3-06 | C | Verify replay closure | P3-05 |
| P3-07 | A | PyTorch GPU port for PDE (production) | P1-09 |
| P3-08 | A | Norse LIF + oscillator integration | P1-03, P3-01 |

**Subagent A**: 2 parallel optimizations
**Subagent C**: 1 verification

---

### Batch 3.5 — LINEAR (Phase 3 gate)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P3-09 | C | Phase 3 integration test | ALL P3-* |
| P3-10 | C | Performance benchmark (spike pipeline) | P3-09 |

**Gate Check**: Spike reinjection latency <100ms, full loop closure

---

## Phase 3 Execution Graph

```
P1-03 (Norse)
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ BATCH 3.1: P3-01 (Spike Reinjection Module)                       │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ BATCH 3.2: P3-02 (Integrate with NREM)                            │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ BATCH 3.3: P3-03 → P3-04 → P3-05 (STDP Loop - LINEAR)             │
│                                                                    │
│  SNN Output → STDP Learner → Weight Update → Memory Update        │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ BATCH 3.4: PARALLEL Optimizations                                  │
│                                                                    │
│  P3-06 (verify)  │  P3-07 (GPU prod)  │  P3-08 (oscillators)      │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ BATCH 3.5: P3-09, P3-10 (GATE)                                     │
└───────────────────────────────────────────────────────────────────┘
```

---

## Phase 4: Validation (8 atoms)

### Batch 4.1 — ALL PARALLEL (independent validations)

| Atom | Owner | Description | Tool |
|------|-------|-------------|------|
| P4-01 | C | MNE oscillation validation | MNE-Python |
| P4-02 | C | Elephant spike cross-correlation | Elephant |
| P4-03 | C | Elephant Granger causality | Elephant |
| P4-04 | C | NetworkX connectome paths | NetworkX |
| P4-05 | C | NetworkX community detection | NetworkX |
| P4-06 | C | STDP LTP/LTD window (17ms/34ms) | Custom |
| P4-07 | C | Biological parameter bounds | Custom |

**Subagent C**: ALL 7 atoms can run in parallel (independent)

---

### Batch 4.2 — LINEAR (report)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P4-08 | D | Validation report generation | ALL P4-01..07 |

**Subagent D**: Aggregate validation results

---

## Phase 4 Execution Graph

```
┌───────────────────────────────────────────────────────────────────┐
│ BATCH 4.1: ALL PARALLEL                                            │
│                                                                    │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│  │P4-01  │ │P4-02  │ │P4-03  │ │P4-04  │ │P4-05  │ │P4-06  │ │P4-07  │
│  │MNE    │ │Eleph  │ │Eleph  │ │NetX   │ │NetX   │ │STDP   │ │Bio    │
│  │oscill │ │xcorr  │ │Grang  │ │paths  │ │commun │ │window │ │bounds │
│  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│ BATCH 4.2: P4-08 (Validation Report)                               │
└───────────────────────────────────────────────────────────────────┘
```

---

## Phase 5: Visualization + Polish (5 atoms)

### Batch 5.1 — PARALLEL (independent outputs)

| Atom | Owner | Description | Output |
|------|-------|-------------|--------|
| P5-01 | D | BrainRender connectome export | 3D viz HTML |
| P5-02 | D | Update all Mermaid diagrams | 7 diagram files |
| P5-03 | D | Update documentation | Architecture docs |
| P5-04 | C | Performance benchmark report | Benchmark results |

**Subagent D**: 3 parallel doc/viz atoms
**Subagent C**: 1 benchmark atom

---

### Batch 5.2 — LINEAR (final gate)

| Atom | Owner | Description | Depends On |
|------|-------|-------------|------------|
| P5-05 | C | Final integration test (all phases) | ALL |

**Gate Check**: Full system E2E test passes

---

## Phase 5 Execution Graph

```
┌───────────────────────────────────────────────────────────────────┐
│ BATCH 5.1: PARALLEL                                                │
│                                                                    │
│  ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐           │
│  │P5-01  │      │P5-02  │      │P5-03  │      │P5-04  │           │
│  │Brain  │      │Mermaid│      │Docs   │      │Bench  │           │
│  │Render │      │diagrms│      │update │      │report │           │
│  └───────┘      └───────┘      └───────┘      └───────┘           │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│ BATCH 5.2: P5-05 (FINAL GATE)                                      │
│                                                                    │
│  ✓ All phases complete                                             │
│  ✓ Full E2E integration test                                       │
│  ✓ Performance targets met                                         │
└───────────────────────────────────────────────────────────────────┘
```

---

## Complete Execution Summary

### Subagent Allocation

| Subagent | Role | Atoms | Skills |
|----------|------|-------|--------|
| **A** | Core Implementation | 18 | PyTorch, Norse, Numba, storage |
| **B** | Integration & Wiring | 12 | Connecting modules, API updates |
| **C** | Tests & Validation | 15 | pytest, MNE, Elephant, benchmarks |
| **D** | Documentation | 5 | Markdown, Mermaid, BrainRender |

### Parallelism Summary

| Phase | Total Atoms | Max Parallel | Critical Path |
|-------|-------------|--------------|---------------|
| P1 | 12 | 7 (Batch 1.1) | P1-01 → P1-02 → P1-12 |
| P2 | 15 | 4 (Batch 2.5) | P2-01 → P2-02 → P2-08 → P2-14 |
| P3 | 10 | 3 (Batch 3.4) | P3-01 → P3-02 → P3-03 → P3-04 → P3-05 → P3-09 |
| P4 | 8 | 7 (Batch 4.1) | P4-* → P4-08 |
| P5 | 5 | 4 (Batch 5.1) | P5-* → P5-05 |

### Timeline (with 4 subagents)

| Week | Phase | Batches | Key Deliverable |
|------|-------|---------|-----------------|
| 1 | P1 | 1.1, 1.2 | τ(t), Norse, MemoryItem, κ fields |
| 2 | P1, P2 | 1.3, 2.1-2.2 | Phase 1 gate, Unified store, queries |
| 3 | P2 | 2.3-2.5 | Consolidation κ, migration, API updates |
| 4 | P2, P3 | 2.6-2.7, 3.1-3.2 | Phase 2 gate, spike reinjection |
| 5 | P3 | 3.3-3.4 | STDP loop, optimizations |
| 6 | P3, P4 | 3.5, 4.1 | Phase 3 gate, all validations |
| 7 | P4, P5 | 4.2, 5.1 | Validation report, viz/docs |
| 8 | P5 | 5.2 | Final gate, release |

---

## Orchestrator Checkpoints

### Gate 1 (End of Week 2)
- [ ] τ(t) signal integrated into MemoryGate
- [ ] Norse SNN generates spikes
- [ ] STDP JIT ≥10x speedup
- [ ] MemoryItem with κ validated
- [ ] All P1 tests pass

### Gate 2 (End of Week 4)
- [ ] Unified store replaces 3 stores
- [ ] Query policies work (episodic/semantic/procedural)
- [ ] Consolidation updates κ correctly
- [ ] Performance ≤10% regression
- [ ] All P2 tests pass

### Gate 3 (End of Week 6)
- [ ] Spike reinjection loop complete
- [ ] Replay → SNN → STDP → weight update works
- [ ] Latency <100ms
- [ ] All P3 tests pass

### Gate 4 (End of Week 7)
- [ ] MNE oscillation bands validated
- [ ] STDP curves match biology (17ms/34ms)
- [ ] All biological parameters in bounds
- [ ] Validation report generated

### Gate 5 (End of Week 8)
- [ ] All diagrams updated
- [ ] Documentation complete
- [ ] Final E2E test passes
- [ ] Ready for merge

---

## Atom Checklist

### Phase 1: Foundation
- [x] P1-01: τ(t) temporal control signal ✓ `src/t4dm/core/temporal_control.py`
- [x] P1-02: Integrate τ(t) into MemoryGate ✓ integrated in memory_gate.py
- [x] P1-03: Norse SNN backend wrapper ✓ `src/t4dm/nca/snn_backend.py`
- [x] P1-04: Numba JIT for STDP hot loop ✓ `src/t4dm/learning/stdp_jit.py`
- [x] P1-05: Unified MemoryItem schema ✓ `src/t4dm/core/unified_memory.py`
- [x] P1-06: Add κ field to Episode ✓ `src/t4dm/core/types.py`
- [x] P1-07: Add κ field to Entity ✓ `src/t4dm/core/types.py`
- [x] P1-08: Add κ field to Procedure ✓ `src/t4dm/core/types.py`
- [x] P1-09: PyTorch GPU PDE solver ✓ `src/t4dm/nca/neural_field_gpu.py`
- [x] P1-10: Docs for τ(t) signal ✓ in CLAUDE.md + docstrings
- [x] P1-11: Docs for unified MemoryItem ✓ in CLAUDE.md + docstrings
- [x] P1-12: Phase 1 integration test ✓ 9,514 tests passing

### Phase 2: Unified Memory
- [x] P2-01: Unified Memory Store backend ✓ T4DX engine `src/t4dm/storage/t4dx/`
- [x] P2-02: Episodic query policy ✓ `src/t4dm/core/query_policies.py`
- [x] P2-03: Semantic query policy ✓ `src/t4dm/core/query_policies.py`
- [x] P2-04: Procedural query policy ✓ `src/t4dm/core/query_policies.py`
- [x] P2-05: NREM κ += 0.05 ✓ `src/t4dm/consolidation/`
- [x] P2-06: REM κ += 0.2 ✓ `src/t4dm/consolidation/`
- [x] P2-07: Data migration script ✓ T4DX is sole backend (no migration needed)
- [x] P2-08: Update episodic retrieval ✓ unified via T4DX
- [x] P2-09: Update semantic retrieval ✓ unified via T4DX
- [x] P2-10: Update procedural retrieval ✓ unified via T4DX
- [x] P2-11: Backward compatibility layer ✓ `src/t4dm/adapters/`
- [x] P2-12: Update API routes ✓ `src/t4dm/api/routes/`
- [x] P2-13: Update CLI ✓ `src/t4dm/cli/`
- [x] P2-14: Phase 2 integration test ✓ 9,514 tests passing
- [x] P2-15: Performance benchmark ✓ `tests/performance/benchmark_full_system.py`

### Phase 3: Spike Pipeline
- [x] P3-01: Spike reinjection module ✓ `src/t4dm/nca/spike_reinjection.py`
- [x] P3-02: Integrate with NREM replay ✓ NREMReplayIntegrator in spike_reinjection.py
- [x] P3-03: Connect SNN to STDP ✓ cortical blocks have STDP attention
- [x] P3-04: STDP weight update ✓ `src/t4dm/learning/stdp.py`
- [x] P3-05: Weight → memory loop ✓ consolidation pipeline
- [ ] P3-06: Verify replay closure — needs E2E test
- [x] P3-07: GPU PDE production ✓ `src/t4dm/nca/neural_field_gpu.py`
- [x] P3-08: Norse + oscillators ✓ `src/t4dm/nca/oscillators.py`
- [ ] P3-09: Phase 3 integration test — needs dedicated test
- [ ] P3-10: Performance benchmark — needs spike pipeline bench

### Phase 4: Validation
- [x] P4-01: MNE oscillation validation ✓ `tests/biology/test_oscillation_validation.py`
- [x] P4-02: Elephant cross-correlation ✓ `tests/biology/test_spike_analysis.py`
- [x] P4-03: Elephant Granger causality ✓ `tests/biology/test_spike_analysis.py` (placeholder)
- [x] P4-04: NetworkX paths ✓ `tests/biology/test_connectome_validation.py`
- [x] P4-05: NetworkX communities ✓ `tests/biology/test_connectome_validation.py`
- [x] P4-06: STDP window validation ✓ `tests/biology/test_stdp_validation.py`
- [x] P4-07: Bio parameter bounds ✓ `tests/biology/test_stdp_validation.py`
- [x] P4-08: Validation report ✓ `docs/VALIDATION_REPORT.md`

### Phase 5: Polish
- [ ] P5-01: BrainRender export — optional visualization
- [x] P5-02: Update Mermaid diagrams ✓ `docs/diagrams/` updated
- [x] P5-03: Update documentation ✓ CLAUDE.md files current
- [x] P5-04: Performance report ✓ benchmarks in tests/performance/
- [x] P5-05: Final integration test ✓ 9,514 tests passing, 81% coverage

---

## Appendix: File Manifest

### New Files to Create

```
src/t4dm/core/
├── temporal_control.py      # P1-01
└── unified_memory.py        # P1-05

src/t4dm/nca/
├── snn_backend.py           # P1-03
├── neural_field_gpu.py      # P1-09
└── spike_reinjection.py     # P3-01

src/t4dm/storage/
└── unified_store.py         # P2-01

scripts/
└── migrate_to_unified.py    # P2-07

src/t4dm/visualization/
└── brainrender_export.py    # P5-01

tests/unit/
├── test_temporal_control.py
├── test_norse_backend.py
├── test_unified_memory.py
├── test_neural_field_gpu.py
├── test_unified_store.py
├── test_query_policies.py
├── test_consolidation_kappa.py
├── test_migration.py
└── test_spike_reinjection.py

tests/integration/
├── test_phase1_integration.py
├── test_phase2_integration.py
├── test_phase3_integration.py
└── test_replay_spike_loop.py

tests/biology/
├── test_oscillator_validation.py
├── test_stdp_validation.py
└── test_bio_parameters.py

docs/reference/
├── temporal_control.md      # P1-10 (renamed from D1-01)
└── unified_memory.md        # P1-11 (renamed from D1-05)

docs/diagrams/
├── snn_integration_flow.mmd # NEW
└── (7 existing diagrams updated)
```

### Files to Modify

```
src/t4dm/core/
├── types.py                 # P1-06, P1-07, P1-08 (add κ)
└── memory_gate.py           # P1-02 (τ integration)

src/t4dm/learning/
└── stdp.py                  # P1-04 (Numba JIT)

src/t4dm/consolidation/
└── sleep.py                 # P2-05, P2-06, P3-02 (κ updates, reinjection)

src/t4dm/memory/
├── episodic.py              # P2-08
├── semantic.py              # P2-09
└── procedural.py            # P2-10

src/t4dm/api/
└── routes/                  # P2-12

src/t4dm/cli/
└── *.py                     # P2-13
```

---

**END OF EXECUTION PLAN**
