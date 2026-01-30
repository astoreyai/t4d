# T4DM Diagram Reconciliation Plan — Atomic Surgical Edits

**Date**: 2026-01-29
**Scope**: All 52 `.mmd` / `.mermaid` files in `docs/diagrams/`
**Constraint**: Diagram-only changes. No Python code modified.
**Auditors**: Hinton AI Architect, Neuroscience Pathway Specialist

---

## Consolidated Issue Registry

48 issues from two independent audits, deduplicated and cross-referenced.

### CRITICAL (7)

| ID | Issue | File(s) | Citation |
|----|-------|---------|----------|
| C1 | GP→Thalamus labeled Glu, must be GABA | `57_connectome_regions.mermaid:69` | Albin 1989, DeLong 1990 |
| C2 | Amygdala→HPC emotional tagging pathway missing | `hippocampal_circuit.mermaid`, `hpc_trisynaptic.mermaid` | Phelps 2004, Richter-Levin 2000 |
| C3 | FF shows single global goodness; must be per-layer | `ff_nca_coupling.mermaid` | Hinton 2022 §2.1 |
| C4 | FF missing layer-local learning structure | `ff_nca_coupling.mermaid` | Hinton 2022 §3 |
| C5 | Spindle→SWR causality arrow wrong (SO drives both) | `spindle_ripple_coupling.mermaid` | Clemens 2011, Staresina 2015 |
| C6 | ACh gating dead code — diagram implies it works | `hippocampal_circuit.mermaid` | Hasselmo 1999 |
| C7 | Grid→Place cell direction reversed | `hippocampal_circuit.mermaid` | Moser 2008 |

### MAJOR (25)

| ID | Issue | File(s) | Citation |
|----|-------|---------|----------|
| M1 | SWR frequency 80-120Hz (must be 150-250Hz) | `sleep_subsystems.mermaid:9` | Buzsáki 2015 |
| M2 | DA baseline 1.0 vs 0.5 elsewhere | `33_state_neuromod.mmd:8` vs `neuromod_orchestra.mermaid` | Schultz 1998 |
| M3 | Salience = DA×NE×ACh multiplicative (zero kills all) | `02_bioinspired_components.mmd:77`, `06_memory_systems.mmd:24` | Mather GANE 2011, Dayan 2012 |
| M4 | GABA classified as neuromodulator (fast NT, not modulator) | `13_neuromodulation_subsystem.mmd:36`, `24_class_neuromod.mmd:98`, `33_state_neuromod.mmd:10` | Dayan 2012, Marder 2012 |
| M5 | VAE generator disconnected from all systems | `50_vae_generator.mermaid` | van de Ven 2020 |
| M6 | ACh suppression of SWR during REM not shown | `swr_replay.mermaid` | Hasselmo 1999, Vandecasteele 2014 |
| M7 | Spindle timing non-monotonic (T3=+100ms < T2=+200ms) | `spindle_ripple_coupling.mermaid:38-43` | Staresina 2015 |
| M8 | LC→HPC NE pathway missing from HPC diagrams | `hippocampal_circuit.mermaid`, `hpc_trisynaptic.mermaid` | Takeuchi 2016, Mather GANE 2016 |
| M9 | STN/hyperdirect pathway missing from basal ganglia | `56_striatal_msn_d1d2.mermaid` | Frank 2006, Nambu 2002 |
| M10 | Capsule routing missing reconstruction decoder | `capsule_routing.mermaid` | Sabour 2017 |
| M11 | Medial septum theta pacemaker missing | `53_theta_gamma_integration.mermaid`, HPC diagrams | Buzsáki 2002 |
| M12 | EC Layer V→II/III feedback loop missing | `hpc_trisynaptic.mermaid` | Witter 2007, van Strien 2009 |
| M13 | Glutamate absent from main circuit diagrams | `13_neuromodulation_subsystem.mmd`, HPC diagrams | — |
| M14 | NAc→VTA GABAergic feedback missing | `vta_circuit.mermaid` | Watabe-Uchida 2012 |
| M15 | Retrieval→neuromod feedback loop missing | `42_seq_retrieve_memory.mmd` | Lisman & Grace 2005 |
| M16 | Cortical→HPC top-down feedback missing | `hippocampal_circuit.mermaid`, `hpc_trisynaptic.mermaid` | Preston & Eichenbaum 2013 |
| M17 | Capsule-NCA coupling direction vague | `nca_module_map.mermaid:93` | Hinton 2022/2023 |
| M18 | DA burst threshold fixed 0.5 (should be RPE=0) | `13_neuromodulation_subsystem.mmd:15` | Schultz 1997 |
| M19 | Cerebellum entirely missing (procedural timing) | All diagrams | Doya 1999 |
| M20 | Causal discovery localist (memory IDs not features) | `58_causal_discovery.mermaid` | Schölkopf 2021 |
| M21 | DG sparsity inconsistent (2% vs 4% across diagrams) | `hippocampal_circuit.mermaid`, `hpc_trisynaptic.mermaid`, `02_bioinspired_components.mmd` | Jung & McNaughton 1993 |
| M22 | Sleep cycle SWR label says "0.5-2 Hz" (ripple is 150-250Hz) | `sleep_cycle.mermaid` | Buzsáki 2015 |
| M23 | Distributed representation absent from memory diagrams | `03_data_flow.mmd`, `06_memory_systems.mmd` | Hinton distributed repr. principle |
| M24 | No diagram connecting ACh source (NBM) to encoding mode | `13_neuromodulation_subsystem.mmd` vs HPC diagrams | Hasselmo 2006 |
| M25 | Consolidation cycle labeled "10-60s" without "compressed" note | `43_seq_consolidation.mmd:103` | — |

### MINOR (16)

| ID | Issue | File(s) | Citation |
|----|-------|---------|----------|
| m1 | Working memory "7±2" outdated (should be 4±1) | `11_memory_subsystem.mmd:7`, `53_theta_gamma_integration.mermaid:6` | Cowan 2001 |
| m2 | Eligibility tau=20s lacks biological citation | `12_learning_subsystem.mmd:14-16` | He 2015, Gerstner 2018 |
| m3 | Homeostatic 3% target unjustified | `08_consolidation_pipeline.mmd:77`, `43_seq_consolidation.mmd:82` | Turrigiano 2008 |
| m4 | Reconsolidation missing from HPC circuit diagrams | `hippocampal_circuit.mermaid`, `hpc_trisynaptic.mermaid` | Nader 2000 |
| m5 | TRN→Thalamus missing from SWR replay diagram | `swr_replay.mermaid` | Steriade 1993 |
| m6 | NBM→HPC absent from neuromod subsystem diagram | `13_neuromodulation_subsystem.mmd` | — |
| m7 | Mossy cell feedback loop missing from HPC | `hpc_trisynaptic.mermaid` | Scharfman 2007 |
| m8 | Consolidation→WM feedback missing | `08_consolidation_pipeline.mmd` | — |
| m9 | Learning subsystem RETR→GATE annotation vague | `12_learning_subsystem.mmd` | — |
| m10 | Transmission delays not cross-ref'd to spindle timing | `54_transmission_delays.mermaid` | — |
| m11 | FF negative data generation mechanism unspecified | `ff_nca_coupling.mermaid` | Hinton 2022 |
| m12 | NAc→VTA feedback missing (minor path) | `vta_circuit.mermaid` | Watabe-Uchida 2012 |
| m13 | Subiculum missing from code (diagram correct) | `hpc_trisynaptic.mermaid` | — |
| m14 | EC Layer II/III distinction not in code | HPC diagrams | — |
| m15 | DG sparsity: pick 4% consistently (code uses 4%) | Multiple | Jung & McNaughton 1993 |
| m16 | Consolidation timing note needs "compressed simulation" | `43_seq_consolidation.mmd` | — |

---

## Atomic Edit Plan

### Constraints

1. **Diagram-only** — zero Python changes
2. **One edit per atom** — each atom touches exactly one file (or one logical unit)
3. **No regressions** — edits preserve existing correct content
4. **Mermaid-valid** — every edit must produce valid mermaid syntax
5. **Execution order** — independent atoms can parallelize; dependencies noted with `AFTER:`

---

### Phase 1: Critical Fixes (7 atoms)

#### ATOM-C1: Fix GP→Thalamus NT label
- **File**: `57_connectome_regions.mermaid`
- **Edit**: Replace `GP -->|Glu| Thalamus` → `GP -->|"GABA (inhibitory)"| Thalamus`
- **Add**: Note: "Basal ganglia output is tonic GABAergic inhibition; D1 striatal activity disinhibits thalamus (Albin 1989)"

#### ATOM-C2: Add Amygdala→HPC pathway
- **File**: `hippocampal_circuit.mermaid`
- **Edit**: Add nodes `BLA["Basolateral Amygdala"]` and edges `BLA -->|"emotional salience"| CA1`, `BLA -->|"valence tag"| EC`
- **File**: `hpc_trisynaptic.mermaid`
- **Edit**: Add `BLA -->|"emotional gating"| CA1` and `BLA -.->|"arousal boost"| DG`
- **Note**: "BLA projects to CA1 and EC, gating emotional memory formation (Phelps 2004; McGaugh 2004)"

#### ATOM-C3+C4: Redesign FF diagram with per-layer goodness
- **File**: `ff_nca_coupling.mermaid`
- **Edit**: Replace single `FF_GOOD["Goodness Function G = sum(h^2)"]` with:
  - `L1_G["Layer 1: G₁ = Σ(h₁²), θ₁"]`
  - `L2_G["Layer 2: G₂ = Σ(h₂²), θ₂"]`
  - `LN_G["Layer N: Gₙ = Σ(hₙ²), θₙ"]`
  - Each layer has independent pos/neg evaluation
  - Add note: "Each layer independently classifies input as real (G>θ) or negative (G<θ). No backward pass. No global loss. (Hinton 2022)"
  - Show NCA coupling per-layer: energy basin depth modulates per-layer θ
- **Also fix**: Add negative data source annotation: "Negative samples via label corruption or VAE generation (see 50_vae_generator)"

#### ATOM-C5: Remove spindle→SWR causality
- **File**: `spindle_ripple_coupling.mermaid`
- **Edit**: Remove edge `SIGMA --> CA3` (or equivalent spindle→SWR arrow)
- **Add**: `SO_UP["SO Up-State"] --> SIGMA` and `SO_UP --> SWR` (both driven by slow oscillation)
- **Note**: "SO up-state independently drives spindles and SWRs; spindles do not trigger SWRs (Clemens 2011; Staresina 2015)"

#### ATOM-C6: Annotate ACh gating as implemented-but-unwired
- **File**: `hippocampal_circuit.mermaid`
- **Edit**: Where ACh encoding/retrieval mode is shown, add annotation: `:::partial` style
- **Note**: "ACh gating logic exists in code (_apply_ach_gating) but is not called in CA1.process() — needs code wiring"

#### ATOM-C7: Fix grid→place cell direction
- **File**: `hippocampal_circuit.mermaid`
- **Edit**: Replace `GRID --> PLACE --> DG` with `MEC["Medial EC (Grid Cells)"] --> DG` and `CA1 --> PLACE["Place Cells (emerge in CA1/CA3)"]`
- **Note**: "Grid cells are INPUT to hippocampus via MEC; place cells are OUTPUT/emergent in CA3/CA1 (Moser 2008)"

---

### Phase 2: Major Fixes — Frequencies & Parameters (7 atoms)

#### ATOM-M1: Fix SWR frequency in sleep_subsystems
- **File**: `sleep_subsystems.mermaid`
- **Edit**: Line 9: `SWR_DET["Ripple Detection (80-120 Hz)"]` → `SWR_DET["Ripple Detection (150-250 Hz)"]`

#### ATOM-M2: Fix DA baseline consistency
- **File**: `33_state_neuromod.mmd`
- **Edit**: Line 8: `DA: 1.0` → `DA: 0.5`
- **Note**: "Tonic DA baseline normalized to 0.5, consistent with other NT baselines and neuromod_orchestra.mermaid"

#### ATOM-M3: Fix salience formula
- **File**: `02_bioinspired_components.mmd`
- **Edit**: Replace `Salience = DA * NE * ACh` → `Salience = w₁·DA + w₂·NE + w₃·DA·NE + w₄·ACh_gate`
- **Note**: "Additive with interaction term; single NT=0 does not nullify (Mather GANE 2011; Dayan 2012)"
- **File**: `06_memory_systems.mmd`
- **Edit**: Same formula update if salience appears

#### ATOM-M7: Fix spindle timing to absolute monotonic
- **File**: `spindle_ripple_coupling.mermaid`
- **Edit**: Replace timing nodes:
  - `T0["t=0ms: SO Up State"]`
  - `T1["t=200ms: Spindle onset"]`
  - `T2["t=400ms: Spindle peak"]`
  - `T3["t=500ms: SWR nested in spindle trough"]`
- **Note**: "Absolute timestamps per Staresina 2015; monotonically increasing"
- **AFTER**: ATOM-C5

#### ATOM-M18: Fix DA burst threshold to RPE
- **File**: `13_neuromodulation_subsystem.mmd`
- **Edit**: Replace `DA_BURST["Burst/Dip Signals reward > 0.5"]` → `DA_BURST["Burst/Dip: RPE = R - V; burst if RPE>0, dip if RPE<0"]`
- **Note**: "Dopamine encodes RPE, not absolute reward (Schultz 1997)"

#### ATOM-M21: Standardize DG sparsity to 4%
- **File**: `hpc_trisynaptic.mermaid` — change "~2%" → "4%"
- **File**: `02_bioinspired_components.mmd` — change "Top 2% Active" → "Top 4% Active"
- **Note**: "4% matches code (hippocampus.py:70) and Jung & McNaughton 1993"

#### ATOM-M22: Fix sleep cycle SWR label
- **File**: `sleep_cycle.mermaid`
- **Edit**: Replace "SWR Events (0.5-2 Hz)" → "SWR Events (ripples 150-250 Hz, occurrence rate 0.5-2/s)"

---

### Phase 3: Major Fixes — Missing Pathways (10 atoms)

#### ATOM-M4: Reclassify GABA
- **File**: `13_neuromodulation_subsystem.mmd`
- **Edit**: Remove GABA from neuromodulator orchestra subgraph. Add separate subgraph: `subgraph E_I_BALANCE["Excitatory/Inhibitory Balance"]` with GABA and Glutamate as fast neurotransmitters
- **File**: `24_class_neuromod.mmd`
- **Edit**: Remove GABA from neuromodulator class hierarchy; add note "GABA/Glu: fast ionotropic transmission, not neuromodulation"
- **File**: `33_state_neuromod.mmd`
- **Edit**: Remove GABA: 0.5 from state; add note about E/I balance as separate system

#### ATOM-M5: Connect VAE to sleep + FF
- **File**: `50_vae_generator.mermaid`
- **Edit**: Add output edges:
  - `VAE_OUT -->|"synthetic replay"| SLEEP_CONSOL["Sleep Consolidation"]`
  - `VAE_OUT -->|"negative samples"| FF_NEG["FF Negative Phase"]`
  - `VAE_OUT -->|"generative compression"| MEM_LIFECYCLE["Memory Lifecycle"]`
- **Note**: "VAE serves CLS generative replay (van de Ven 2020) and FF negative data (Hinton 2022)"

#### ATOM-M6: Add ACh SWR suppression during REM
- **File**: `swr_replay.mermaid`
- **Edit**: Add node `ACH_REM["ACh Peak (REM)"]` with edge `ACH_REM -->|"suppresses"| SWR_GEN["SWR Generation"]`
- **Note**: "High ACh during REM suppresses hippocampal SWRs; replay is NREM-specific (Hasselmo 1999; Vandecasteele 2014)"

#### ATOM-M8: Add LC→HPC NE pathway
- **File**: `hippocampal_circuit.mermaid`
- **Edit**: Add `LC["Locus Coeruleus"] -->|"NE: novelty tagging"| DG` and `LC -->|"NE: arousal gain"| CA3`
- **File**: `hpc_trisynaptic.mermaid`
- **Edit**: Add `LC -->|"NE"| DG_GC` and `LC -->|"NE"| CA3_PYR`
- **Note**: "NE from LC gates novelty-dependent encoding (Takeuchi 2016; Mather GANE 2016)"

#### ATOM-M9: Add STN hyperdirect pathway
- **File**: `56_striatal_msn_d1d2.mermaid`
- **Edit**: Add nodes `STN["Subthalamic Nucleus"]`, edges:
  - `CORTEX -->|"Glu (hyperdirect)"| STN`
  - `STN -->|"Glu"| GPi`
  - `GPe -->|"GABA"| STN`
- **Note**: "Hyperdirect pathway provides rapid inhibition for stop signals (Frank 2006; Nambu 2002)"

#### ATOM-M10: Add capsule reconstruction decoder
- **File**: `capsule_routing.mermaid`
- **Edit**: Add `OUTPUT_CAPS -->|"reconstruction"| DECODER["Reconstruction Decoder"]` → `DECODER -->|"reconstruction loss"| INPUT_REPR`
- **Note**: "Decoder regularizes capsule representations (Sabour 2017). Distinguish from GLOM (Hinton 2021)"

#### ATOM-M11: Add medial septum theta source
- **File**: `53_theta_gamma_integration.mermaid`
- **Edit**: Add `MS["Medial Septum / Diagonal Band"]` with edges:
  - `MS -->|"ACh + GABA"| HPC_THETA["Hippocampal Theta"]`
- **Note**: "MS/DBB is the theta pacemaker (Buzsáki 2002)"
- **File**: `hippocampal_circuit.mermaid`
- **Edit**: Add MS as theta source

#### ATOM-M12: Add EC V→II/III feedback
- **File**: `hpc_trisynaptic.mermaid`
- **Edit**: Add edge `EC_V -->|"feedback"| EC_II` and `EC_V -->|"feedback"| EC_III`
- **Note**: "Completes hippocampal-entorhinal loop for multi-pass consolidation (Witter 2007; van Strien 2009)"

#### ATOM-M15: Add retrieval→neuromod feedback
- **File**: `42_seq_retrieve_memory.mmd`
- **Edit**: After retrieval scoring, add:
  - `RetrievalResult ->> Orchestra: RPE from retrieval outcome`
  - `Orchestra ->> VTA: update DA (surprising retrieval)`
  - `Orchestra ->> LC: update NE (retrieval failure → uncertainty)`
- **Note**: "Retrieval outcomes modulate neuromodulators (Lisman & Grace 2005)"

#### ATOM-M16: Add cortical→HPC top-down feedback
- **File**: `hippocampal_circuit.mermaid`
- **Edit**: Add `PFC["Prefrontal Cortex"] -->|"contextual/goal modulation"| EC`
- **File**: `hpc_trisynaptic.mermaid`
- **Edit**: Add `PFC -->|"top-down"| EC_III`
- **Note**: "PFC→EC provides goal-directed modulation (Preston & Eichenbaum 2013)"

---

### Phase 4: Major Fixes — Architecture (8 atoms)

#### ATOM-M13: Add glutamate to main circuits
- **File**: `hippocampal_circuit.mermaid`
- **Edit**: Label perforant path, mossy fibers, Schaffer collaterals as "Glu"
- **File**: `13_neuromodulation_subsystem.mmd`
- **Edit**: Reference glutamate in E/I balance subgraph (from ATOM-M4)

#### ATOM-M14: Add NAc→VTA GABA feedback
- **File**: `vta_circuit.mermaid`
- **Edit**: Add `NAc -->|"GABA (feedback)"| VTA_GABA["VTA GABA Neurons"]`
- **Note**: "Mesolimbic feedback loop regulates DA neuron activity (Watabe-Uchida 2012)"

#### ATOM-M17: Clarify capsule-NCA coupling
- **File**: `nca_module_map.mermaid`
- **Edit**: Replace vague `CAP -.->|Capsule-NCA Coupling| FF` with:
  - `CAP -->|"agreement scores → FF layer goodness"| FF`
  - `FF -->|"per-layer energy → capsule routing precision"| CAP`
- **Note**: "Capsule agreement maps to FF goodness; FF energy modulates routing (Hinton 2022/2023)"

#### ATOM-M19: Add cerebellum note
- **File**: `01_system_architecture.mmd`
- **Edit**: Add `CEREBELLUM["Cerebellum (PLANNED)"]:::planned` with dashed edge to procedural memory
- **Note**: "Cerebellum implements error-based motor learning complementary to striatal RL (Doya 1999). Not yet implemented."
- **File**: `nca_module_map.mermaid`
- **Edit**: Add planned cerebellum node

#### ATOM-M20: Annotate causal discovery as localist
- **File**: `58_causal_discovery.mermaid`
- **Edit**: Add note: "Current: causal edges between memory IDs (symbolic). Target: latent causal models in embedding space (Schölkopf 2021). Migration planned."
- **Style**: Mark as `:::partial`

#### ATOM-M23: Add distributed representation note to memory diagrams
- **File**: `06_memory_systems.mmd`
- **Edit**: Add subgraph note: "Memories are distributed patterns across sparse encoder units. Retrieval = pattern completion via attractor dynamics (Hopfield), not database lookup. UUIDs are handles, not the representation."
- **File**: `03_data_flow.mmd`
- **Edit**: Add annotation at embedding step: "Distributed representation: memory = attractor state in high-dim energy landscape"

#### ATOM-M24: Connect NBM→HPC in neuromod diagram
- **File**: `13_neuromodulation_subsystem.mmd`
- **Edit**: Add `NBM -->|"ACh: encoding mode"| HPC` edge in cholinergic subgraph
- **Note**: "NBM ACh projection switches hippocampus between encoding (high ACh) and retrieval (low ACh) modes (Hasselmo 2006)"

#### ATOM-M25: Add "compressed simulation" note
- **File**: `43_seq_consolidation.mmd`
- **Edit**: At line 103, change "Cycle typically 10-60 seconds" → "Cycle: 10-60s (compressed simulation; biological equivalent: 90-min ultradian cycle)"

---

### Phase 5: Minor Fixes (16 atoms)

#### ATOM-m1: Update WM capacity
- **File**: `11_memory_subsystem.mmd`
- **Edit**: `7+/-2 items` → `4±1 items (Cowan 2001)`
- **File**: `53_theta_gamma_integration.mermaid`
- **Edit**: `7 slots` → `4 slots`

#### ATOM-m2: Annotate eligibility trace timescales
- **File**: `12_learning_subsystem.mmd`
- **Edit**: Add note at traces: "Fast τ≈1-5s (Ca²⁺ transients), Slow τ≈minutes (CaMKII/protein synthesis). Standard τ=20s is software interpolation (He 2015; Gerstner 2018)"

#### ATOM-m3: Annotate homeostatic target
- **File**: `08_consolidation_pipeline.mmd`
- **Edit**: Add note: "3% target activity is a system parameter; biological homeostatic scaling targets firing rate set-points (Turrigiano 2008; Tononi SHY 2006)"
- **File**: `43_seq_consolidation.mmd`
- **Edit**: Same note

#### ATOM-m4: Add reconsolidation to HPC diagrams
- **File**: `hippocampal_circuit.mermaid`
- **Edit**: Add `CA1_OUT -->|"retrieval destabilizes"| LABILITY["Lability Window"]` → `LABILITY -->|"reconsolidation"| CA3`
- **Note**: "Retrieval makes traces labile (Nader 2000; Lee 2009)"

#### ATOM-m5: Add TRN to SWR replay
- **File**: `swr_replay.mermaid`
- **Edit**: Add `TRN["Thalamic Reticular Nucleus"] -->|"GABA"| THAL["Thalamus"]` with note "TRN inhibition generates spindles via rebound bursting (Steriade 1993)"

#### ATOM-m6: Add NBM→HPC in neuromod subsystem
- Covered by ATOM-M24

#### ATOM-m7: Add mossy cell feedback
- **File**: `hpc_trisynaptic.mermaid`
- **Edit**: Add `MOSSY_CELLS -->|"excitatory feedback"| GC` and `HILAR_IN["Hilar Interneurons"] -->|"inhibitory feedback"| GC`
- **Note**: "Mossy cell excitation + interneuron inhibition modulates DG pattern separation (Scharfman 2007)"

#### ATOM-m8: Add consolidation→WM feedback
- **File**: `08_consolidation_pipeline.mmd`
- **Edit**: Add edge from final output back to `WM["Working Memory Context"]` with label "updated context"

#### ATOM-m9: Annotate learning feedback
- **File**: `12_learning_subsystem.mmd`
- **Edit**: Replace vague "feedback" label on RETR→GATE with "retrieval outcome (helpful/not) → Bayesian LR gate update"

#### ATOM-m10: Cross-reference delays to spindle timing
- **File**: `54_transmission_delays.mermaid`
- **Edit**: Add note: "See spindle_ripple_coupling.mermaid for how these delays affect SO→spindle→SWR timing"

#### ATOM-m11: Annotate FF negative data
- Covered by ATOM-C3+C4

#### ATOM-m12: Add NAc→VTA minor path
- Covered by ATOM-M14

#### ATOM-m13+m14: Annotate subiculum and EC layers as code-side fixes
- **File**: `hpc_trisynaptic.mermaid`
- **Edit**: Add `:::code_needed` style to Subiculum node with note "Diagram correct; code needs SubiculumLayer class"
- **Edit**: Add `:::code_needed` style to EC_II, EC_III with note "Diagram correct; code uses single EC input — needs split"

#### ATOM-m15: Standardize DG sparsity
- Covered by ATOM-M21

#### ATOM-m16: Compressed simulation note
- Covered by ATOM-M25

---

## Execution Matrix

```
Phase 1 (CRITICAL):  C1, C2, C3+C4, C5, C6, C7          — 6 atoms, all independent
Phase 2 (PARAMS):    M1, M2, M3, M7*, M18, M21, M22      — 7 atoms, M7 AFTER C5
Phase 3 (PATHWAYS):  M4, M5, M6, M8, M9, M10, M11, M12, M15, M16 — 10 atoms, all independent
Phase 4 (ARCH):      M13*, M14, M17, M19, M20, M23, M24, M25     — 8 atoms, M13 AFTER M4
Phase 5 (MINOR):     m1, m2, m3, m4, m5, m7, m8, m9, m10, m13+m14 — 10 atoms, all independent
                                                           TOTAL: 41 atoms
```

### Parallelization

- **Within each phase**: All atoms are independent unless marked `AFTER:`
- **Across phases**: Phase 2+ can start after Phase 1 completes
- **Maximum parallelism**: 6 (Phase 1), 7 (Phase 2), 10 (Phase 3), 8 (Phase 4), 10 (Phase 5)

### Files Touched Per Phase

| Phase | Files Modified | Files Created |
|-------|---------------|---------------|
| 1 | 5 diagram files | 0 |
| 2 | 8 diagram files | 0 |
| 3 | 10 diagram files | 0 |
| 4 | 7 diagram files | 0 |
| 5 | 9 diagram files | 0 |
| **Total** | **~30 unique files** | **0** |

---

## Verification Checklist

After each phase:
1. `npx -p @mermaid-js/mermaid-cli mmdc -i <file> -o /dev/null` — syntax validation
2. Visual inspection of changed edges/nodes
3. Cross-reference: if edge added in file A, verify consistency in files B,C that show same pathway
4. No orphan nodes (every node has at least one edge)
