# T4DM Architecture Diagrams

**Last Updated**: 2026-02-02
**Total Diagrams**: 59+ (source files + HTML visualizations)
**Formats**: Mermaid source (.mmd, .mermaid), D2 source (.d2) -> Rendered (PNG, SVG)

---

## Overview

This directory contains the complete visual documentation for T4DM's biologically-inspired memory architecture. The system uses a Qwen 2.5-3B backbone with spiking cortical adapter, backed by T4DX embedded storage engine.

### Diagram Categories

| Prefix | Category | Count | Description |
|--------|----------|-------|-------------|
| **01-10** | System Architecture | 9 | High-level system views, data flow, storage |
| **11-20** | Subsystem Decomposition | 5 | Detailed subsystem breakdowns |
| **21-30** | Class Diagrams | 5 | Object-oriented design (Memory, Learning, Storage, Neuromod, Consolidation) |
| **31-40** | State Diagrams | 4 | State machines for stateful components |
| **41-49** | Sequence Diagrams | 4 | Temporal interaction flows |
| **50-59** | Biology/NCA Specific | 10 | Neural cellular automata and biological circuits |
| **Named** | Feature Diagrams | 23+ | Specific subsystems and features |

---

## Map of Maps

### Hierarchy and Cross-References

```
01_system_architecture (TOP LEVEL)
├── 02_bioinspired_components
│   ├── 50-59_biology/* (NCA details)
│   ├── ff_nca_coupling
│   ├── capsule_routing
│   └── hippocampal_circuit
├── 03_data_flow
│   ├── 41_seq_store_memory (T4DX: tau gate -> MemoryItem -> WAL -> MemTable -> Segment)
│   ├── 42_seq_retrieve_memory
│   └── 43_seq_consolidation
├── 05_storage_architecture
│   ├── 14_storage_subsystem
│   ├── 23_class_storage
│   ├── persistence_layer (T4DX WAL + segments + compaction)
│   ├── checkpoint_lifecycle (checkpoint_v3: T4DX + spiking state)
│   ├── recovery_flow (T4DX WAL replay + segment scan)
│   ├── shutdown_flow (T4DX MemTable flush + WAL truncate)
│   └── wal_flow (T4DX JSON-lines WAL with fsync)
├── 06_memory_systems
│   ├── 11_memory_subsystem
│   ├── 21_class_memory
│   ├── 34_state_memory_gate
│   ├── memory_lifecycle
│   └── eligibility_trace
├── 07_class_bioinspired -> Forward-Forward, Capsules
├── 08_consolidation_pipeline
│   ├── 25_class_consolidation
│   ├── 32_state_consolidation
│   ├── 43_seq_consolidation
│   ├── 44_seq_sleep_replay
│   ├── consolidation_stages
│   ├── sleep_cycle
│   ├── sleep_subsystems
│   ├── swr_replay
│   └── spindle_ripple_coupling
└── 10_observability

Spiking Architecture Tree:
t4dm_spiking_block.d2 (6-stage spiking block)
t4dm_snn_vaswani.d2 (Vaswani-style SNN layout)
t4dm_snn_transformer_architecture.mermaid (SNN overview)

Learning Subsystem Tree:
12_learning_subsystem
├── 22_class_learning
├── three_factor_rule
├── credit_assignment_flow
└── eligibility_trace

Neuromodulation Subsystem Tree:
13_neuromodulation_subsystem
├── 24_class_neuromod
├── 33_state_neuromod
├── neuromod_orchestra
├── neuromodulator_pathways
├── adenosine_homeostasis
└── vta_circuit (dopamine, 20-40 Hz burst, phasic pause)

NCA/Biology Tree:
50_vae_generator
51_glymphatic_system
52_spatial_cells
53_theta_gamma_integration
54_transmission_delays
55_astrocyte_tripartite
56_striatal_msn_d1d2
57_connectome_regions
58_causal_discovery
59_self_supervised_credit
hpc_trisynaptic (trisynaptic loop + BLA + mossy cells)
```

---

## System Architecture (01-10)

High-level system views showing the overall design.

| Diagram | Description | Links To |
|---------|-------------|----------|
| ![01](01_system_architecture.png) | **System Architecture** - Top-level: Qwen + Spiking + T4DX | 02, 03, 05, 06, 10 |
| ![02](02_bioinspired_components.png) | **Bio-inspired Components** - Hinton architectures (FF, Capsules, NCA) | 07, 50-59 |
| ![03](03_data_flow.png) | **Data Flow** - How information moves through the system | 41, 42, 43 |
| ![05](05_storage_architecture.png) | **Storage Architecture** - T4DX embedded engine (LSM + HNSW + CSR) | 14, 23 |
| ![06](06_memory_systems.png) | **Memory Systems** - Kappa-gradient memory (episodic -> semantic) | 11, 21 |
| ![07](07_class_bioinspired.png) | **Class: Bio-inspired** - Forward-Forward and Capsule Network classes | 02 |
| ![08](08_consolidation_pipeline.png) | **Consolidation Pipeline** - NREM/REM/PRUNE compaction | 25, 32, 43, 44 |
| ![10](10_observability.png) | **Observability** - Telemetry, tracing, and monitoring | 01 |

### Markdown Architecture Docs
| File | Description |
|------|-------------|
| `system_architecture.md` | Full system architecture with Qwen + Spiking + T4DX |
| `storage_resilience.md` | Circuit breaker + T4DX WAL durability model |

---

## Subsystem Decomposition (11-20)

| Diagram | Description | Links To |
|---------|-------------|----------|
| ![11](11_memory_subsystem.png) | **Memory Subsystem** - MemoryItem with kappa gradient | 21, 34, 41, 42 |
| ![12](12_learning_subsystem.png) | **Learning Subsystem** - Hebbian, STDP, Forward-Forward | 22, three_factor_rule |
| ![13](13_neuromodulation_subsystem.png) | **Neuromodulation Subsystem** - Dopamine, Serotonin, Acetylcholine, Adenosine | 24, 33 |
| ![14](14_storage_subsystem.png) | **Storage Subsystem** - T4DX persistence layer | 23, 05 |
| ![15](15_api_subsystem.png) | **API Subsystem** - REST endpoints and SDK | 01 |

---

## Class Diagrams (21-30)

| Diagram | Description | Key Classes |
|---------|-------------|-------------|
| ![21](21_class_memory.png) | **Class: Memory** | `MemoryItem`, `MemoryStore`, kappa-based routing |
| ![22](22_class_learning.png) | **Class: Learning** | `HebbianLearner`, `STDPLearner`, `ForwardForwardLearner` |
| ![23](23_class_storage.png) | **Class: Storage** | `T4DXEngine`, `MemTable`, `Segment`, `WAL` |
| ![24](24_class_neuromod.png) | **Class: Neuromodulation** | `DopamineModulator`, `SerotoninModulator`, `AdenosineMonitor` |
| ![25](25_class_consolidation.png) | **Class: Consolidation** | `ConsolidationService`, `HDBSCANClusterer`, `FESConsolidator`, `LabilityTracker` |

---

## State Diagrams (31-40)

| Diagram | Description | States |
|---------|-------------|--------|
| ![31](31_state_circuit_breaker.png) | **State: Circuit Breaker** | Closed -> Open -> Half-Open (wraps T4DX) |
| ![32](32_state_consolidation.png) | **State: Consolidation** | Idle -> Clustering -> Merging -> Complete |
| ![33](33_state_neuromod.png) | **State: Neuromodulation** | Baseline -> Elevated -> Depleted |
| ![34](34_state_memory_gate.png) | **State: Memory Gate** | Open -> Filtering -> Closed |

---

## Sequence Diagrams (41-49)

| Diagram | Description | Participants |
|---------|-------------|--------------|
| ![41](41_seq_store_memory.png) | **Sequence: Store Memory** | API -> tau(t) gate -> MemoryItem -> T4DX WAL -> MemTable -> Segment |
| ![42](42_seq_retrieve_memory.png) | **Sequence: Retrieve Memory** | Query -> HNSW search -> CSR traversal -> kappa-weighted results |
| ![43](43_seq_consolidation.png) | **Sequence: Consolidation** | Scheduler -> NREM/REM/PRUNE compaction |
| ![44](44_seq_sleep_replay.png) | **Sequence: Sleep Replay** | Sleep -> Hippocampus -> SWR -> Neocortex -> Consolidation -> Glymphatic |

---

## Biology & NCA Diagrams (50-59)

| Diagram | Description | Bio Inspiration |
|---------|-------------|-----------------|
| ![50](50_vae_generator.png) | **VAE Generator** | Generative model for pattern completion |
| ![51](51_glymphatic_system.png) | **Glymphatic System** | Metabolic waste clearance during sleep |
| ![52](52_spatial_cells.png) | **Spatial Cells** | Place cells, Grid cells, Border cells |
| ![53](53_theta_gamma_integration.png) | **Theta-Gamma Integration** | Nested oscillations for sequence encoding |
| ![54](54_transmission_delays.png) | **Transmission Delays** | Axonal conduction delays for temporal credit |
| ![55](55_astrocyte_tripartite.png) | **Astrocyte Tripartite Synapse** | Glial modulation of synaptic plasticity |
| ![56](56_striatal_msn_d1d2.png) | **Striatal MSN D1/D2** | Direct/Indirect pathway competition |
| ![57](57_connectome_regions.png) | **Connectome Regions** | Brain region connectivity map |
| ![58](58_causal_discovery.png) | **Causal Discovery** | Temporal causality inference |
| ![59](59_self_supervised_credit.png) | **Self-Supervised Credit** | Credit assignment without labels |

---

## Feature Diagrams (Named)

### Spiking Architecture (D2)
| Diagram | Description |
|---------|-------------|
| `t4dm_spiking_block.d2` | 6-stage spiking cortical block (LIF + thalamic + spike attn + apical + RWKV) |
| `t4dm_snn_vaswani.d2` | Vaswani-style SNN layout showing Qwen + spiking adapter |
| `t4dm_snn_transformer_architecture.mermaid` | SNN transformer overview |

### Memory Features
| Diagram | Description |
|---------|-------------|
| ![memory_lifecycle](memory_lifecycle.png) | Memory lifecycle: Store -> Encode -> Consolidate -> Retrieve -> Decay |
| ![eligibility_trace](eligibility_trace.png) | Eligibility traces for delayed reward attribution |

### Learning Features
| Diagram | Description |
|---------|-------------|
| ![three_factor_rule](three_factor_rule.png) | Three-factor Hebbian rule: Pre x Post x Neuromodulator |
| ![credit_assignment_flow](credit_assignment_flow.png) | Temporal credit assignment flow |

### Consolidation Features
| Diagram | Description |
|---------|-------------|
| ![consolidation_stages](consolidation_stages.png) | Synaptic -> Systems consolidation stages |
| ![swr_replay](swr_replay.png) | Sharp-Wave Ripple replay (90% reverse rest, forward active) |
| ![spindle_ripple_coupling](spindle_ripple_coupling.png) | Sleep spindle <-> hippocampal ripple coordination |
| ![sleep_cycle](sleep_cycle.png) | Sleep cycle state machine (Wake -> NREM -> REM) |
| ![sleep_subsystems](sleep_subsystems.png) | Sleep subsystems: Spindles, SWR, Glymphatic |

### Neuromodulation Features
| Diagram | Description |
|---------|-------------|
| ![neuromod_orchestra](neuromod_orchestra.png) | Orchestrated neuromodulator dynamics |
| ![neuromodulator_pathways](neuromodulator_pathways.png) | Dopamine, Serotonin, ACh, Adenosine pathways |
| ![adenosine_homeostasis](adenosine_homeostasis.png) | Adenosine accumulation and sleep pressure |
| ![vta_circuit](vta_circuit.png) | VTA dopamine circuit (20-40 Hz burst, phasic pause, NAc feedback) |

### NCA/Hinton Architecture Features
| Diagram | Description |
|---------|-------------|
| ![nca_module_map](nca_module_map.png) | NCA module connectivity map |
| ![ff_nca_coupling](ff_nca_coupling.png) | Forward-Forward <-> NCA integration |
| ![capsule_routing](capsule_routing.png) | Capsule network routing-by-agreement |
| ![hippocampal_circuit](hippocampal_circuit.png) | Hippocampal CA1/CA3/DG circuit |
| ![hpc_trisynaptic](hpc_trisynaptic.png) | Trisynaptic loop: EC -> DG -> CA3 -> CA1 (+ BLA, mossy cells, EC V/VI feedback) |

### Persistence Features (T4DX)
| Diagram | Description |
|---------|-------------|
| ![persistence_layer](persistence_layer.png) | T4DX persistence: WAL + LSM segments + compaction engine |
| ![checkpoint_lifecycle](checkpoint_lifecycle.png) | Checkpoint_v3: T4DX memtable + spiking state + QLoRA |
| ![recovery_flow](recovery_flow.png) | T4DX WAL replay + segment scan recovery |
| ![shutdown_flow](shutdown_flow.png) | T4DX MemTable flush + WAL truncate shutdown |
| ![wal_flow](wal_flow.png) | T4DX JSON-lines WAL with fsync |

---

## Interactive Visualizations

| File | Description | Technology |
|------|-------------|------------|
| `architecture_3d.html` | 3D force-directed graph of system architecture | Three.js, D3.js |
| `interactive_network.html` | Interactive network map of components | vis.js |

Open these files in a web browser for exploration.

---

## Rendering

All diagrams are available in multiple formats:

1. **Mermaid source**: `.mmd` or `.mermaid` files (editable Mermaid syntax)
2. **D2 source**: `.d2` files (editable D2 syntax, for spiking diagrams)
3. **PNG**: Rasterized images at 2400x1800 resolution
4. **SVG**: Vector graphics (scalable, recommended for documentation)

### Regenerating Diagrams

```bash
cd /mnt/projects/t4d/t4dm/docs/diagrams
python generate_diagrams.py  # Renders all to /tmp/t4dm-diagrams/
```

Or manually:

```bash
# Mermaid
mmdc -i <source>.mmd -o <output>.png -w 2400 -H 1800

# D2
d2 <source>.d2 <output>.svg
```

---

## Diagram Dependencies

Key dependency relationships:

```
Store/Retrieve Flow:
  41_seq_store_memory -> tau(t) -> MemoryItem -> T4DX WAL -> MemTable -> Segment
  42_seq_retrieve_memory -> HNSW search -> CSR traversal -> kappa-weighted results

Consolidation Flow:
  44_seq_sleep_replay -> 43_seq_consolidation -> 25_class_consolidation -> 08_consolidation_pipeline
  44_seq_sleep_replay -> sleep_cycle, swr_replay, spindle_ripple_coupling
  Compaction (NREM/REM/PRUNE) -> T4DX segments

Learning Flow:
  22_class_learning -> 12_learning_subsystem -> three_factor_rule, eligibility_trace

Neuromodulation Flow:
  24_class_neuromod -> 13_neuromodulation_subsystem -> neuromod_orchestra
  adenosine_homeostasis -> sleep_cycle -> 44_seq_sleep_replay

Spiking Architecture:
  t4dm_spiking_block.d2 -> t4dm_snn_vaswani.d2 -> system_architecture.md

NCA/Biology:
  02_bioinspired_components -> 07_class_bioinspired -> 50-59_biology/*
  hippocampal_circuit, hpc_trisynaptic -> 52_spatial_cells, swr_replay
```

---

## Coverage Analysis

### Complete Coverage
- System Architecture: Qwen + Spiking + T4DX embedded engine
- Class Diagrams for all core subsystems (Memory, Learning, Storage, Neuromod, Consolidation)
- State Diagrams for all stateful components (Circuit Breaker, Consolidation, Neuromod, Memory Gate)
- Sequence Diagrams for critical flows (Store via T4DX, Retrieve, Consolidate, Sleep Replay)
- Biology/NCA coverage (10+ diagrams for neural mechanisms)
- Spiking architecture (D2 diagrams for cortical blocks)
- Persistence diagrams updated for T4DX WAL + checkpoint_v3

### Optional Extensions (Future Work)
- 26_class_nca: NCA subsystem class diagram
- 27_class_spiking: Spiking cortical stack class diagram
- 35_state_learning: Learning state machine
- 45_seq_learning: Learning interaction sequence (STDP flow)
- 46_seq_inference: Qwen + Spiking inference pipeline sequence

---

## References

For implementation details, see:
- **Architecture**: `/mnt/projects/t4d/t4dm/docs/ARCHITECTURE.md`
- **Full Plan**: `/mnt/projects/t4d/t4dm/docs/plans/FULL_SYSTEM_PLAN.md`
- **Memory**: `/mnt/projects/t4d/t4dm/docs/MEMORY_ARCHITECTURE.md`
- **Learning**: `/mnt/projects/t4d/t4dm/docs/LEARNING_ARCHITECTURE.md`
- **Biology**: `/mnt/projects/t4d/t4dm/docs/BRAIN_REGION_MAPPING.md`
- **Math**: `/mnt/projects/t4d/t4dm/docs/MATHEMATICAL_FOUNDATIONS.md`

For code implementation:
- **Memory**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/`
- **Spiking**: `/mnt/projects/t4d/t4dm/src/t4dm/spiking/`
- **Qwen**: `/mnt/projects/t4d/t4dm/src/t4dm/qwen/`
- **T4DX Storage**: `/mnt/projects/t4d/t4dm/src/t4dm/storage/t4dx/`
- **Learning**: `/mnt/projects/t4d/t4dm/src/t4dm/learning/`
- **Consolidation**: `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/`
- **NCA**: `/mnt/projects/t4d/t4dm/src/t4dm/nca/`

---

**Maintained by**: T4DM Architecture Team
**Diagram Standards**: Mermaid.js (UML-compliant), D2 (spiking architecture)
**Rendering**: mermaid-cli (mmdc) v10+, d2 v0.6+
