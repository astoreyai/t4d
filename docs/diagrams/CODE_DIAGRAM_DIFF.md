# Code vs Diagram Consolidated Diff Report

**Date**: 2026-01-29
**Method**: 6 parallel agents analyzed every code file against every diagram
**Scope**: All `.py` in `src/t4dm/` vs all `.mmd`/`.mermaid` in `docs/diagrams/`

---

## Consolidated Issue Registry

### Category A: Diagram Claims Something Code Doesn't Do (Fix Diagram)

| ID | Diagram File | Diagram Claim | Code Reality | Fix |
|----|-------------|---------------|--------------|-----|
| A1 | `hpc_trisynaptic.mermaid:37` | "⚠ code needs SubiculumLayer class" | `SubiculumLayer` EXISTS at `hippocampus.py:456-512` | Remove warning annotation |
| A2 | `hippocampal_circuit.mermaid:78` | "ACh gating not called in CA1.process()" | `_apply_ach_gating()` IS called at `hippocampus.py:741` | Remove warning annotation |
| A3 | `hpc_trisynaptic.mermaid:4-5` | "code uses single EC input — needs split" | Code supports `ec_layer2_input` + `ec_layer3_input` params | Remove warning annotation |
| A4 | `swr_replay.mermaid:16` | "20x compression" | Code uses `compression_factor=10.0` (`swr_coupling.py:109`, `sleep.py:542`) | Change diagram to "10x" |
| A5 | `vta_circuit.mermaid:19` | "Phasic (15-25 Hz burst)" | Code: `burst_peak_rate=30.0` Hz (`vta.py:62`) | Change diagram to "15-30 Hz burst" |
| A6 | `33_state_neuromod.mmd:8` | "DA: 0.5" baseline | VTA code: `tonic_da_level=0.3` (`vta.py:59`) | Change diagram to "DA: 0.3" |
| A7 | `33_state_neuromod.mmd:9` | "ACh: 0.5" baseline | NBM code: `baseline_ach=0.3` (`nucleus_basalis.py:59`) | Change diagram to "ACh: 0.3" |
| A8 | `33_state_neuromod.mmd:9` | "5-HT: 0.5" baseline | Raphe code: `setpoint=0.4` (`raphe.py:76`) | Change diagram to "5-HT: 0.4" |
| A9 | `13_neuromodulation_subsystem.mmd:19` | "Arousal Detector 4 states" | LC code: `LCFiringMode` has 5 states (`locus_coeruleus.py:45-51`) | Change to "5 states" |
| A10 | `33_state_neuromod.mmd:25` | "DA > 1.5" for burst | Code uses RPE sign (positive → burst), no absolute threshold | Change to "RPE > 0 → burst" |
| A11 | `swr_replay.mermaid:19` | Shows only forward replay | Code: 90% reverse replay (`sleep.py:83-87`, Foster & Wilson 2006) | Add "90% reverse, 10% forward" |
| A12 | `08_consolidation_pipeline.mmd:4` | "Every 15 min" time trigger | Code default: 1.5h (`service.py:160`) | Change to "Every 1.5h" |
| A13 | `vta_circuit.mermaid:19-20` | Only TONIC and PHASIC modes | Code has 3: TONIC, PHASIC_BURST, PHASIC_PAUSE (`vta.py:46-50`) | Add PHASIC_PAUSE |

### Category B: Diagram Shows Something Code Doesn't Have (Aspirational — Annotate)

| ID | Diagram File | Diagram Shows | Code Status | Fix |
|----|-------------|---------------|-------------|-----|
| B1 | `hippocampal_circuit.mermaid:68-70` | BLA emotional salience/valence to CA1/EC | Not implemented | Add `:::planned` |
| B2 | `hippocampal_circuit.mermaid:5,8` | PFC contextual/goal modulation to EC | Not implemented | Add `:::planned` |
| B3 | `hpc_trisynaptic.mermaid:57-60` | Mossy cells + hilar interneurons DG feedback | Not implemented | Add `:::planned` |
| B4 | `hpc_trisynaptic.mermaid:81-82` | EC V→II/III feedback loops | Not implemented | Add `:::planned` |
| B5 | `hpc_trisynaptic.mermaid:38-39` | EC Layer V → Neocortex output | Not implemented | Add `:::planned` |
| B6 | `hippocampal_circuit.mermaid:32-36` | Grid cells (MEC input) and place cells (CA3/CA1 output) | Not implemented in hippocampus module (spatial_cells.py exists separately) | Add `:::planned` or cross-ref |
| B7 | `hippocampal_circuit.mermaid:57-58` | Reconsolidation: CA1→Lability→CA3 | Not implemented | Add `:::planned` |
| B8 | `hippocampal_circuit.mermaid:77,86` | Medial Septum as theta pacemaker | Theta uses external oscillator, no MS class | Add `:::partial` |
| B9 | `56_striatal_msn_d1d2.mermaid:8-16` | STN hyperdirect pathway (Cortex→STN→GPi, GPe↔STN) | Code only has D1/D2 pathways | Add `:::planned` |
| B10 | `vta_circuit.mermaid:4-8` | LH, LDT, RMTg inputs to VTA | Only Raphe input implemented | Add `:::planned` |
| B11 | `54_transmission_delays.mermaid:50-53` | Velocity plasticity (activity-dependent myelination) | Not implemented | Add `:::planned` |
| B12 | `58_causal_discovery.mermaid` | Entire causal discovery system | No code exists at all | Already marked `:::partial` |
| B13 | `capsule_routing.mermaid:72-83` | Reconstruction decoder | Not in capsule code | Add `:::planned` |
| B14 | `24_class_neuromod.mmd:16-25` | Abstract `NeuromodulatorSystem` base class | No abstract base exists | Add note "inheritance not implemented" |
| B15 | `13_neuromodulation_subsystem.mmd:52` | NBM→Hippocampus direct wiring | NBM has no `connect_to_hippocampus()` method | Add `:::planned` |
| B16 | `41_seq_store_memory.mmd` | "Orchestra" component providing NeuroState | No Orchestra in store path | Rename to match code component |
| B17 | `42_seq_retrieve_memory.mmd` | "Orchestra" in retrieve path | Same | Rename to match code component |

### Category C: Code Has Something Diagram Doesn't Show (Add to Diagram)

| ID | Code File | Code Feature | Relevant Diagram | Fix |
|----|-----------|-------------|-----------------|-----|
| C1 | `substantia_nigra.py:104` | `SubstantiaNigraCircuit` — motor DA separate from VTA | `vta_circuit.mermaid` | Add SNc node or separate diagram |
| C2 | `dopamine_integration.py:184-679` | `DopamineIntegration` — integrates VTA+field+hippocampus | No diagram | Add to `nca_module_map.mermaid` |
| C3 | `dopamine_integration.py:48-115` | `DelayBuffer` — 1000ms biological DA arrival delay | No diagram | Add to eligibility trace or VTA diagram |
| C4 | `neuromod_crosstalk.py:85-131` | DA↔5-HT, NE↔ACh, ACh↔DA crosstalk | No diagram | Add to `13_neuromodulation_subsystem.mmd` |
| C5 | `sleep.py:480-601` | VAE generative replay (Hinton wake-sleep) | `50_vae_generator.mermaid` shows VAE but not in sleep flow | Add edge from VAE to sleep |
| C6 | `sleep.py:926-1015` | VTA/dopamine RPE during replay | Sleep diagrams | Add DA prioritization to consolidation |
| C7 | `sleep.py:311-369` | Multi-night progressive consolidation | No diagram | Add note to `43_seq_consolidation.mmd` |
| C8 | `hippocampus.py:1150-1159` | `_apply_ne_encoding_gain()` exists but never called | `hippocampal_circuit.mermaid` shows NE→DG/CA3 | Mark as `:::partial` (code exists, unwired) |
| C9 | `spatial_cells.py:105-423` | Place cells (100), grid cells (3×32), gridness scores | No dedicated diagram | Add to `hippocampal_circuit.mermaid` spatial subgraph |
| C10 | `cross_modal_binding.py:261-680` | Episodic↔Semantic↔Procedural gamma binding | No diagram | Add to `nca_module_map.mermaid` |
| C11 | `amygdala.py` | Emotional valence, fear learning/extinction | No dedicated diagram | Referenced in HPC diagrams but no detail |
| C12 | `astrocyte.py` | Tripartite synapse, EAAT-2, GAT-3, calcium waves | No dedicated diagram | Add to `nca_module_map.mermaid` |
| C13 | `stability.py:1-810` | Hessian, eigenvalue, Lyapunov stability analysis | No diagram | Add to `nca_module_map.mermaid` |
| C14 | `persistence/manager.py` | WAL, checkpointing, recovery, graceful shutdown | Not in main architecture diagram | Add to `01_system_architecture.mmd` |
| C15 | `api/server.py:174-338` | Security middleware stack (XSS, CSP, HSTS, size limit) | Not in any diagram | Add to `15_api_subsystem.mmd` |
| C16 | `pose_learner.py:260-409` | Emergent pose dimension discovery | Not in capsule diagram | Add to `capsule_routing.mermaid` |
| C17 | `forward_forward.py:550-636` | Neurogenesis — add/remove neurons dynamically | Not in FF diagram | Add to `ff_nca_coupling.mermaid` |
| C18 | `coupling.py:603-770` | Jacobian stability, attractor analysis, fixed points | Not in NCA diagram | Add to `nca_module_map.mermaid` |
| C19 | `adenosine.py:396-418` | Adenosine suppresses DA/NE/ACh, potentiates GABA | Not in sleep diagrams | Add to `sleep_subsystems.mermaid` |

### Category D: Parameter Mismatches (Fix Diagram to Match Code)

| ID | Parameter | Code Value | Diagram Value | File |
|----|-----------|-----------|---------------|------|
| D1 | SWR compression | 10x | 20x | `swr_coupling.py:109` vs `swr_replay.mermaid:16` |
| D2 | VTA burst rate | 30 Hz | 15-25 Hz | `vta.py:62` vs `vta_circuit.mermaid:19` |
| D3 | DA baseline | 0.3 | 0.5 | `vta.py:59` vs `33_state_neuromod.mmd:8` |
| D4 | ACh baseline | 0.3 | 0.5 | `nucleus_basalis.py:59` vs `33_state_neuromod.mmd:9` |
| D5 | 5-HT baseline | 0.4 | 0.5 | `raphe.py:76` vs `33_state_neuromod.mmd:9` |
| D6 | LC firing modes | 5 states | 4 states | `locus_coeruleus.py:45-51` vs `13_neuromodulation_subsystem.mmd:19` |
| D7 | Consolidation interval | 1.5h | 15min | `service.py:160` vs `08_consolidation_pipeline.mmd:4` |
| D8 | Homeostatic target | weight=10.0 | activity=3% | `sleep.py:500` vs `08_consolidation_pipeline.mmd:91` |

---

## Statistics

| Category | Count |
|----------|-------|
| A: Diagram wrong (fix diagram) | 13 |
| B: Aspirational (annotate diagram) | 17 |
| C: Undocumented code (add to diagram) | 19 |
| D: Parameter mismatch (fix diagram) | 8 |
| **Total issues** | **57** |

---
