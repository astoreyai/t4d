# Phase 7 Wiring Diagram

## Current State: Disconnected Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CURRENT ARCHITECTURE                            │
│                         (Components in Isolation)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │   BRIDGES       │     │   TEMPORAL      │     │   DREAMING      │   │
│  │                 │     │                 │     │                 │   │
│  │ • FF Encoding   │     │ • Dynamics     │     │ • Consolidation │   │
│  │ • Capsule       │     │ • Session      │     │ • Quality       │   │
│  │ • Dopamine      │     │ • Integration  │     │ • Trajectory    │   │
│  │ • Memory-NCA    │     │                 │     │                 │   │
│  └────────╳────────┘     └────────╳────────┘     └────────╳────────┘   │
│           ╳                       ╳                       ╳            │
│           ╳  NOT CONNECTED        ╳  NOT CONNECTED        ╳            │
│           ╳                       ╳                       ╳            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    CORE MEMORY SYSTEM                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │ Episodic │  │ Semantic │  │Procedural│  │  Unified │       │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│           │                                                            │
│           ▼                                                            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    CONSOLIDATION SERVICE                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │    │
│  │  │   Sleep  │  │   STDP   │  │ Glymphatic│                     │    │
│  │  │  (NREM/  │  │Integration│  │  Bridge  │                     │    │
│  │  │   REM)   │  │          │  │          │                     │    │
│  │  └──────────┘  └──────────┘  └──────────┘                     │    │
│  └────────────────────────────────────────────────────────────────┘    │
│           │                                                            │
│           ▼                                                            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    LEARNING SYSTEM                             │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │    │
│  │  │Three-    │  │Eligibility│  │Neuromod  │                     │    │
│  │  │ Factor   │  │  Traces  │  │Orchestra │                     │    │
│  │  └──────────┘  └──────────┘  └──────────┘                     │    │
│  └────────────────────────────────────────────────────────────────┘    │
│           ╳                                                            │
│           ╳  COUPLING NOT UPDATED                                      │
│           ╳                                                            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    NCA DYNAMICS                                │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │  Neural  │  │ Coupling │  │  Energy  │  │Attractors│       │    │
│  │  │  Field   │  │  (STATIC)│  │Landscape │  │          │       │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Legend:  ╳ = Not Connected / Broken Wire
```

---

## Target State: Fully Integrated System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TARGET ARCHITECTURE                             │
│                         (All Components Wired)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        BRIDGE CONTAINER                          │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │   │
│  │  │ FFEncoding │  │  Capsule   │  │  Dopamine  │  │ Memory-NCA │ │   │
│  │  │   Bridge   │  │   Bridge   │  │   Bridge   │  │   Bridge   │ │   │
│  │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘ │   │
│  └─────────┼───────────────┼───────────────┼───────────────┼───────┘   │
│            │               │               │               │           │
│            ▼               ▼               ▼               ▼           │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    CORE MEMORY SYSTEM                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │ Episodic │◄─│ Semantic │  │Procedural│  │  Unified │◄──────┼────┼── SessionManager
│  │  │  + FF    │  │+ Capsule │  │          │  │ + Bridge │       │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │    │
│  └───────┼─────────────┼─────────────┼─────────────┼─────────────┘    │
│          │             │             │             │                   │
│          └─────────────┴──────┬──────┴─────────────┘                   │
│                               ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    CONSOLIDATION SERVICE                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │   Sleep  │──│   STDP   │  │ Glymphatic│  │ DREAMING │       │    │
│  │  │  (NREM)  │  │Integration│  │  Bridge  │  │(REM NOW  │       │    │
│  │  │    ↓     │  │          │  │          │  │ WIRED!)  │       │    │
│  │  │   REM ───┼──┼──────────┼──┼──────────┼──┼──►       │       │    │
│  │  └────┬─────┘  └────┬─────┘  └──────────┘  └────┬─────┘       │    │
│  └───────┼─────────────┼───────────────────────────┼─────────────┘    │
│          │             │                           │                   │
│          ▼             ▼                           ▼                   │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    LEARNING SYSTEM                             │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │Three-    │──│Eligibility│──│Neuromod  │  │Plasticity│       │    │
│  │  │ Factor   │  │  Traces  │  │Orchestra │  │Coordinator       │    │
│  │  └────┬─────┘  └──────────┘  └────┬─────┘  └──────────┘       │    │
│  └───────┼───────────────────────────┼───────────────────────────┘    │
│          │                           │                                 │
│          ▼                           ▼                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    NCA DYNAMICS                                │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │  Neural  │  │ Coupling │  │  Energy  │  │Attractors│       │    │
│  │  │  Field   │  │(LEARNING)│◄─│Landscape │  │          │       │    │
│  │  │          │  │  ↑       │  │          │  │          │       │    │
│  │  └──────────┘  └──┼───────┘  └──────────┘  └──────────┘       │    │
│  │                   │                                            │    │
│  │                   └──────── RPE from VTA ──────────────────────┘    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    TEMPORAL MODULE (NOW WIRED)                   │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │   │
│  │  │  Session   │◄─│  Temporal  │◄─│ Plasticity │                 │   │
│  │  │  Manager   │  │  Dynamics  │  │Coordinator │                 │   │
│  │  └────────────┘  └────────────┘  └────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Legend:  ─► = Data Flow    ◄─ = Modulation
```

---

## Sprint Execution Map

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                           PARALLEL EXECUTION MAP                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  SPRINT 7.1 (BRIDGES) ─────────────────────────────────────────────────  ║
║  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      ║
║  │ 7.1.1       │  │ 7.1.2       │  │ 7.1.3       │  │ 7.1.4       │      ║
║  │ FF→Episodic │  │ Caps→Seman  │  │ DA→Consol   │  │ NCA→Unified │      ║
║  │ PARALLEL    │  │ PARALLEL    │  │ PARALLEL    │  │ PARALLEL    │      ║
║  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      ║
║        │               │               │               │                  ║
║        └───────────────┴───────────────┴───────────────┘                  ║
║                                │                                          ║
║                                ▼                                          ║
║  SPRINT 7.2 (TEMPORAL) ────────────────────────────────────────────────  ║
║  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                       ║
║  │ 7.2.1       │  │ 7.2.2       │  │ 7.2.3       │                       ║
║  │ Session→Mem │→─│ Plasticity  │→─│ Dynamics→   │                       ║
║  │ SEQUENTIAL  │  │ →Learning   │  │ Consol      │                       ║
║  └─────────────┘  └─────────────┘  └─────────────┘                       ║
║                                │                                          ║
║                                ▼                                          ║
║  SPRINT 7.3 (DREAMING) ────────────────────────────────────────────────  ║
║  ┌─────────────┐  ┌─────────────┐                                        ║
║  │ 7.3.1       │  │ 7.3.2       │                                        ║
║  │ Dream→REM   │→─│ Quality→    │                                        ║
║  │ SEQUENTIAL  │  │ Selection   │                                        ║
║  └─────────────┘  └─────────────┘                                        ║
║                                │                                          ║
║                                ▼                                          ║
║  SPRINT 7.4 (ENERGY) ──────────────────────────────────────────────────  ║
║  ┌─────────────┐  ┌─────────────┐                                        ║
║  │ 7.4.1       │  │ 7.4.2       │                                        ║
║  │ Coupling→   │→─│ RPE→        │                                        ║
║  │ ThreeFactor │  │ Coupling    │                                        ║
║  └─────────────┘  └─────────────┘                                        ║
║                                │                                          ║
║                                ▼                                          ║
║  SPRINT 7.5-7.6 (POLISH) ──────────────────────────────────────────────  ║
║  ┌─────────────┐  ┌─────────────┐                                        ║
║  │ 7.5.*       │  │ 7.6.*       │                                        ║
║  │ Bio Params  │  │ Test        │                                        ║
║  │ PARALLEL    │  │ Coverage    │                                        ║
║  └─────────────┘  └─────────────┘                                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Connection Priority Matrix

| Priority | Connection | Source → Target | Sprint |
|----------|------------|-----------------|--------|
| **P0** | FF → Encoding | `episodic.encode()` → `ff_bridge.process()` | 7.1.1 |
| **P0** | Capsule → Retrieval | `semantic.recall()` → `capsule_bridge.compute_boosts()` | 7.1.2 |
| **P0** | DA → Consolidation | `consolidation.run()` → `dopamine_bridge.compute_rpe()` | 7.1.3 |
| **P1** | NCA → Memory | `unified.search()` → `nca_bridge.modulate_retrieval()` | 7.1.4 |
| **P1** | Session → Memory | Memory ops → `session_manager.track()` | 7.2.1 |
| **P1** | Dream → REM | `sleep.rem_phase()` → `dream_consolidation.process()` | 7.3.1 |
| **P2** | Coupling → Learning | `three_factor.compute()` → `coupling.update_from_energy()` | 7.4.1 |
| **P2** | RPE → Coupling | `vta.compute_rpe()` → `coupling.update_from_energy()` | 7.4.2 |
| **P3** | Bio params | Parameter corrections | 7.5.* |
| **P3** | Test coverage | 27 new test files | 7.6.* |

---

## Files to Modify

### Sprint 7.1 (Bridges)
```
MODIFY: src/t4dm/memory/episodic.py      → Add FF bridge call in encode()
MODIFY: src/t4dm/memory/semantic.py      → Add Capsule bridge call in recall()
MODIFY: src/t4dm/consolidation/service.py → Add Dopamine bridge
MODIFY: src/t4dm/memory/unified.py       → Add NCA bridge
CREATE: src/t4dm/core/bridge_container.py → Factory for all bridges
```

### Sprint 7.2 (Temporal)
```
MODIFY: src/t4dm/memory/episodic.py      → Add session tracking
MODIFY: src/t4dm/memory/unified.py       → Add session tracking
MODIFY: src/t4dm/learning/three_factor.py → Add plasticity coordinator
MODIFY: src/t4dm/consolidation/sleep.py  → Add temporal dynamics
```

### Sprint 7.3 (Dreaming)
```
MODIFY: src/t4dm/consolidation/sleep.py  → Add dream consolidation call
MODIFY: src/t4dm/consolidation/sleep.py  → Add dream quality evaluation
```

### Sprint 7.4 (Energy)
```
MODIFY: src/t4dm/learning/three_factor.py → Add coupling update call
MODIFY: src/t4dm/nca/vta.py             → Route RPE to coupling
MODIFY: src/t4dm/nca/coupling.py        → Accept learning signals
```

---

**Status**: PLANNING COMPLETE - Ready for parallel execution
