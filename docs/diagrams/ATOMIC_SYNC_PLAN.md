# Atomic Diagram Sync Plan — Code → Diagram Alignment

**Date**: 2026-01-29
**Source**: CODE_DIAGRAM_DIFF.md (57 issues)
**Constraint**: Diagram-only changes. Zero Python code modified.
**Principle**: Diagrams must reflect code reality. Aspirational features get `:::planned` styling.

---

## Phase 1: Fix Stale Annotations (3 atoms)

These diagrams have warnings about missing code that HAS SINCE BEEN IMPLEMENTED.

### ATOM-S1: Remove SubiculumLayer warning
- **File**: `hpc_trisynaptic.mermaid:37`
- **Old**: `SUBICULUM["Subiculum<br/>⚠ Diagram correct; code needs SubiculumLayer class"]:::code_needed`
- **New**: `SUBICULUM["Subiculum<br/>(1024-dim, gain modulation)"]`
- **Ref**: `hippocampus.py:456-512` — class exists

### ATOM-S2: Remove ACh gating warning
- **File**: `hippocampal_circuit.mermaid:78`
- **Old**: `ACH_NOTE["⚠ ACh gating logic exists in code (_apply_ach_gating)<br/>but not yet called in CA1.process() — needs code wiring"]`
- **New**: `ACH_NOTE["ACh gating: _apply_ach_gating() called in process()<br/>High ACh suppresses CA3→CA1 (max 80%)"]`
- **Ref**: `hippocampus.py:741, 1127-1148`

### ATOM-S3: Remove EC split warning
- **File**: `hpc_trisynaptic.mermaid:4-5`
- **Old**: `EC_II["Entorhinal Cortex<br/>Layer II<br/>⚠ Diagram correct; code uses single EC input — needs split"]:::code_needed`
- **New**: `EC_II["Entorhinal Cortex<br/>Layer II<br/>(ec_layer2_input param)"]`
- **Same for EC_III** on line 5
- **Ref**: `hippocampus.py:692-720`

---

## Phase 2: Fix Parameter Mismatches (8 atoms)

### ATOM-P1: SWR compression factor
- **File**: `swr_replay.mermaid:16`
- **Old**: `COMPRESS["Time Compression<br/>(20x faster)"]`
- **New**: `COMPRESS["Time Compression<br/>(10x faster)"]`
- **Ref**: `swr_coupling.py:109`

### ATOM-P2: VTA burst rate range
- **File**: `vta_circuit.mermaid:19`
- **Old**: `PHASIC["Phasic<br/>(15-25 Hz burst)"]`
- **New**: `PHASIC["Phasic Burst<br/>(15-30 Hz)"]`
- **Ref**: `vta.py:62` — `burst_peak_rate=30.0`

### ATOM-P3: Add PHASIC_PAUSE to VTA
- **File**: `vta_circuit.mermaid`
- **Add**: `PAUSE["Phasic Pause<br/>(0 Hz, 200-500ms)"]` inside Firing subgraph
- **Add**: `RPE -->|"-δ→pause"| PAUSE` (replace TONIC target)
- **Ref**: `vta.py:46-50` — 3 firing modes

### ATOM-P4: DA baseline 0.3
- **File**: `33_state_neuromod.mmd:8`
- **Old**: `DA: 0.5, NE: 0.5`
- **New**: `DA: 0.3, NE: 0.5`
- **Ref**: `vta.py:59`

### ATOM-P5: ACh baseline 0.3
- **File**: `33_state_neuromod.mmd:9`
- **Old**: `ACh: 0.5, 5-HT: 0.5`
- **New**: `ACh: 0.3, 5-HT: 0.4`
- **Ref**: `nucleus_basalis.py:59`, `raphe.py:76`

### ATOM-P6: LC arousal states → 5
- **File**: `13_neuromodulation_subsystem.mmd:19`
- **Old**: `NE_AROU[Arousal Detector<br/>4 states]`
- **New**: `NE_AROU[Arousal Detector<br/>5 modes: QUIESCENT, TONIC_LOW,<br/>TONIC_OPTIMAL, TONIC_HIGH, PHASIC]`
- **Ref**: `locus_coeruleus.py:45-51`

### ATOM-P7: Consolidation trigger interval
- **File**: `08_consolidation_pipeline.mmd:4`
- **Old**: `TimeBasedTrigger[Time-Based<br/>Every 15 min]`
- **New**: `TimeBasedTrigger[Time-Based<br/>Every 1.5h default]`
- **Ref**: `service.py:160`

### ATOM-P8: DA burst threshold text
- **File**: `33_state_neuromod.mmd:25`
- **Old**: `DA > 1.5`
- **New**: `RPE > 0 → burst`
- **Ref**: VTA fires phasic on positive RPE, not absolute DA level

---

## Phase 3: Fix Logic Mismatches (2 atoms)

### ATOM-L1: Add reverse replay to SWR diagram
- **File**: `swr_replay.mermaid`
- **Add** to ReplayContent subgraph: `DIRECTION["Replay Direction<br/>90% reverse, 10% forward<br/>(Foster & Wilson 2006)"]`
- **Add edge**: `SEQUENCE -->|"direction"| DIRECTION`
- **Ref**: `sleep.py:83-87`

### ATOM-L2: Homeostatic target annotation
- **File**: `08_consolidation_pipeline.mmd:77`
- **Change** PruneWeak note to include: `"Weight target: 10.0 (downscale factor 0.9)<br/>Conceptual biological equiv: 3% activity"`
- **Ref**: `sleep.py:499-500`

---

## Phase 4: Annotate Aspirational Features (17 atoms)

Add `:::planned` styling to diagram features that don't exist in code.

### ATOM-A1: BLA in hippocampal_circuit
- **File**: `hippocampal_circuit.mermaid:68-70`
- **Change**: `BLA["Basolateral Amygdala"]` → `BLA["Basolateral Amygdala (PLANNED)"]:::planned`
- **Change edges to dashed**: `BLA -.->|"emotional salience"| CA1` etc.

### ATOM-A2: PFC in hippocampal_circuit
- **File**: `hippocampal_circuit.mermaid:5,8`
- **Change**: Add `:::planned` to PFC node
- **Change edge to dashed**: `PFC -.->|"contextual/goal modulation (PLANNED)"| EC`

### ATOM-A3: Mossy cells/hilar interneurons in hpc_trisynaptic
- **File**: `hpc_trisynaptic.mermaid:56-62`
- **Add** `:::planned` to MOSSY and HILAR_IN nodes
- **Change edges to dashed**

### ATOM-A4: EC V→II/III feedback + neocortex output
- **File**: `hpc_trisynaptic.mermaid:78-82`
- **Add** `:::planned` to EC_V and CORTEX nodes
- **Change edges to dashed**

### ATOM-A5: Grid/place cells in hippocampal_circuit
- **File**: `hippocampal_circuit.mermaid:32-37`
- **Add** `:::planned` to Grid subgraph nodes
- **Add note**: "spatial_cells.py exists separately; not wired to hippocampus.py"

### ATOM-A6: Reconsolidation pathway
- **File**: `hippocampal_circuit.mermaid:57-58`
- **Change**: `CA1 -->|...` → `CA1 -.->|...` (dashed) and add `:::planned`

### ATOM-A7: STN hyperdirect in striatal diagram
- **File**: `56_striatal_msn_d1d2.mermaid:8-16`
- **Add** `:::planned` to STN, GPe, GPi nodes
- **Change edges to dashed**

### ATOM-A8: VTA missing inputs
- **File**: `vta_circuit.mermaid:4-8`
- **Add** `:::planned` to LH, LDT, RMTg nodes
- Keep RAPHE_IN solid (implemented)

### ATOM-A9: Velocity plasticity
- **File**: `54_transmission_delays.mermaid:50-53`
- **Add** `:::planned` to Plasticity and UpdateVel nodes

### ATOM-A10: Capsule reconstruction decoder
- **File**: `capsule_routing.mermaid:72-83`
- **Add** `:::planned` to Decoder subgraph nodes

### ATOM-A11: Abstract NeuromodulatorSystem base
- **File**: `24_class_neuromod.mmd:16-25`
- **Add note**: "Abstract base class not implemented; each system is standalone"

### ATOM-A12: NBM→Hippocampus wiring
- **File**: `13_neuromodulation_subsystem.mmd:52`
- **Change edge to dashed**: `NBM -.->|"ACh: encoding mode (PLANNED)"| HPC`

### ATOM-A13: NE encoding gain unwired
- **File**: `hippocampal_circuit.mermaid:84-85`
- **Add annotation**: "⚠ _apply_ne_encoding_gain() exists but not called"
- **Add** `:::partial` styling to LC edges

### ATOM-A14: Medial Septum
- **File**: `hippocampal_circuit.mermaid:77`
- **Add** `:::partial` to MS node
- **Add note**: "Theta uses external oscillator; no dedicated MS class"

### ATOM-A15–A17: Remaining B-items
- B16/B17: Sequence diagrams reference "Orchestra" — add note "Maps to NeuromodulatorOrchestra in learning/neuromodulators.py"

---

## Phase 5: Add Missing Code Features to Diagrams (12 atoms)

### ATOM-C1: Add SNc to nca_module_map
- **File**: `nca_module_map.mermaid`
- **Add**: `SNc["SubstantiaNigraCir<br/>Motor DA"]` in Neuromod subgraph
- **Add edge**: `SNc -->|"DA modulates"| STR`

### ATOM-C2: Add DopamineIntegration to nca_module_map
- **File**: `nca_module_map.mermaid`
- **Add**: `DAI["DopamineIntegration<br/>VTA+Field+HPC"]` with edges to VTA, NF, CA1

### ATOM-C3: Add DA delay buffer to eligibility_trace.mermaid
- **File**: `eligibility_trace.mermaid:23`
- **Change**: `DA_TIMING["Arrives 0.5-2s<br/>after event"]` → `DA_TIMING["DelayBuffer<br/>1000ms biological delay<br/>(dopamine_integration.py)"]`

### ATOM-C4: Add neuromod crosstalk to 13_neuromodulation_subsystem
- **File**: `13_neuromodulation_subsystem.mmd`
- **Add edges**: `DA -.->|"inhibits"| SER`, `NE -.->|"excites"| ACH`, `ACH -.->|"gates"| DA`
- **Add note**: "Crosstalk: neuromod_crosstalk.py"

### ATOM-C5: Add reverse replay direction to swr_replay
- Covered by ATOM-L1

### ATOM-C6: Add multi-night note to consolidation
- **File**: `43_seq_consolidation.mmd`
- **Add note after line 103**: "Night 1: light, Night 2-3: deep, Night 4+: all (service.py:311-369)"

### ATOM-C7: Add adenosine NT suppression
- **File**: `sleep_subsystems.mermaid`
- **Add**: `AD_ACC -->|"suppresses DA/NE/ACh<br/>potentiates GABA"| NT_MOD["NT Modulation"]`

### ATOM-C8: Add neurogenesis note to FF diagram
- **File**: `ff_nca_coupling.mermaid`
- **Add note**: "Layers support neurogenesis: add_neuron()/remove_neuron() (forward_forward.py:550-636)"

### ATOM-C9: Add emergent pose learning to capsule diagram
- **File**: `capsule_routing.mermaid`
- **Add note**: "Emergent pose dimension discovery via routing agreement (pose_learner.py:260-409)"

### ATOM-C10: Add stability analysis to nca_module_map
- **File**: `nca_module_map.mermaid`
- **Add**: `STAB["StabilityAnalyzer<br/>Hessian + Lyapunov"]` connected to EN (Energy)

### ATOM-C11: Add persistence to system architecture
- **File**: `01_system_architecture.mmd` (if exists) or `nca_module_map.mermaid`
- **Add note**: "Persistence: WAL + checkpointing + graceful shutdown (persistence/manager.py)"

### ATOM-C12: Add security middleware note to API diagram
- **File**: `15_api_subsystem.mmd` (if exists)
- **Add note**: "Security: XSS/CSP/HSTS headers, 5MB size limit, CORS validation"

---

## Execution Summary

| Phase | Atoms | Files Touched | Description |
|-------|-------|---------------|-------------|
| 1: Stale annotations | 3 | 2 | Remove outdated warnings |
| 2: Parameters | 8 | 5 | Fix numbers to match code |
| 3: Logic | 2 | 2 | Fix behavioral descriptions |
| 4: Aspirational | 17 | 10 | Add :::planned to unimplemented |
| 5: Missing features | 12 | 8 | Add code features to diagrams |
| **Total** | **42** | **~18 unique** | |

### Dependencies
- Phase 1: All independent
- Phase 2: All independent
- Phase 3: All independent
- Phase 4: All independent (can parallelize with 1-3)
- Phase 5: After Phase 2 (some files overlap)

### classDef Definitions Needed
```mermaid
classDef planned fill:#e0e0e0,stroke:#999,stroke-dasharray: 5 5
classDef partial fill:#fff3cd,stroke:#ffc107,stroke-dasharray: 5 5
classDef code_needed fill:#fff3cd,stroke:#ffc107,stroke-width:2px,stroke-dasharray: 5 5
```
