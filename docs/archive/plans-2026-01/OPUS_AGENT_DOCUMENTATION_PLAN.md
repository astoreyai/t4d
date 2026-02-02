# T4DM Comprehensive Documentation Analysis Plan

**Version**: 2.0
**Created**: 2026-01-04
**Agents**: ww-hinton (Neural Architecture) + ww-compbio (Biological Validation)
**Scope**: Complete codebase analysis and documentation overhaul

---

## Executive Summary

This plan orchestrates two specialized Opus agents to comprehensively analyze and document the T4DM codebase:

1. **ww-hinton**: Neural architecture analysis, learning algorithms, representation theory, Hinton-inspired design validation
2. **ww-compbio**: Biological validation, neuroscience accuracy, parameter auditing, literature verification

**Current State**:
- Hinton Score: 7.9/10 (target: 9.5/10)
- Biology Score: 94/100 (target: 98/100)
- Documentation: 255 files, ~60% coverage
- Diagrams: 8 mermaid, 75+ static

**Target State**:
- Hinton Score: 9.5/10
- Biology Score: 98/100
- Documentation: 100% coverage
- Diagrams: 30+ mermaid (cross-referenced)

---

## Phase 1: Module Inventory & Classification

### 1.1 NCA Region (29 modules) - PRIMARY TARGET

| Module | Hinton Tasks | CompBio Tasks | Priority |
|--------|--------------|---------------|----------|
| **forward_forward.py** | H1: FF algorithm fidelity | B1: Wake/sleep analogs | P0 |
| **forward_forward_nca_coupling.py** | H1: Goodness-energy alignment | B1: NT modulation | P0 |
| **capsules.py** | H2: Dynamic routing | B2: Microcolumn analogs | P0 |
| **capsule_nca_coupling.py** | H2: Pose transformations | B2: ACh gating | P0 |
| **pose.py** | H2: Part-whole hierarchies | - | P1 |
| **energy.py** | H3: Energy landscapes | B3: Hopfield networks | P0 |
| **attractors.py** | H3: Attractor dynamics | B3: Memory basins | P1 |
| **stability.py** | H3: Bifurcation analysis | B3: Homeostasis | P1 |
| **vta.py** | - | B4: DA circuit (Schultz) | P0 |
| **dopamine_integration.py** | H4: TD error | B4: D1/D2 receptors | P0 |
| **raphe.py** | - | B5: 5-HT (patience) | P0 |
| **locus_coeruleus.py** | - | B6: NE (arousal) | P0 |
| **striatal_msn.py** | H4: Action selection | B7: D1/D2 MSN | P0 |
| **striatal_coupling.py** | H4: Direct/indirect | B7: TAN pause | P1 |
| **hippocampus.py** | H5: Memory indexing | B8: DG→CA3→CA1 | P0 |
| **spatial_cells.py** | H5: Place/grid cells | B8: O'Keefe/Moser | P1 |
| **oscillators.py** | - | B9: Theta/gamma bands | P0 |
| **sleep_spindles.py** | - | B9: 11-16 Hz spindles | P1 |
| **theta_gamma_integration.py** | - | B9: PAC coupling | P1 |
| **swr_coupling.py** | H5: Replay | B10: 150-250 Hz ripples | P0 |
| **glymphatic.py** | - | B11: AQP4 clearance | P0 |
| **glymphatic_consolidation_bridge.py** | H6: Sleep pruning | B11: Xie et al. | P0 |
| **adenosine.py** | - | B12: Sleep pressure | P1 |
| **neural_field.py** | H3: 6D NT state | B13: Field dynamics | P0 |
| **coupling.py** | H3: Cross-region K matrix | B13: Coupling constants | P1 |
| **delays.py** | - | B14: Axonal delays | P2 |
| **connectome.py** | H6: Graph structure | B14: Anatomical | P1 |
| **astrocyte.py** | - | B15: Tripartite synapse | P1 |
| **glutamate_signaling.py** | - | B15: AMPA/NMDA | P1 |

### 1.2 Learning Region (23 modules)

| Module | Hinton Tasks | CompBio Tasks | Priority |
|--------|--------------|---------------|----------|
| **eligibility.py** | H7: Trace dynamics | B16: STDP windows | P0 |
| **credit_flow.py** | H7: TD(λ) | B16: Eligibility decay | P0 |
| **three_factor.py** | H7: Pre×Post×DA | B17: Neuromodulation | P0 |
| **stdp.py** | - | B17: Timing curves | P0 |
| **dopamine.py** | H4: RPE learning | B4: Phasic/tonic | P0 |
| **serotonin.py** | H8: Patience discount | B5: 5-HT1A/2A | P0 |
| **norepinephrine.py** | H8: Gain modulation | B6: LC modes | P0 |
| **acetylcholine.py** | H8: Enc/ret gating | B18: Hasselmo | P0 |
| **neuromodulators.py** | H8: NT orchestra | B18: Integration | P1 |
| **homeostatic.py** | H9: Synaptic scaling | B19: Turrigiano | P1 |
| **plasticity.py** | H9: Meta-plasticity | B19: BCM theory | P1 |
| **fsrs.py** | H10: Spaced repetition | - | P1 |
| **scorer.py** | H10: Memory strength | - | P1 |
| **collector.py** | - | - | P2 |
| **events.py** | - | - | P2 |
| **hooks.py** | - | - | P2 |
| **persistence.py** | - | - | P2 |
| **causal_discovery.py** | H11: Causal graphs | - | P1 |
| **neuro_symbolic.py** | H11: Symbol grounding | - | P1 |
| **self_supervised.py** | H1: Contrastive | B20: Self-organization | P1 |
| **cold_start.py** | - | - | P2 |
| **inhibition.py** | - | B21: GABAergic | P1 |
| **reconsolidation.py** | - | B22: Memory updating | P1 |

### 1.3 Memory Region (12 modules)

| Module | Hinton Tasks | CompBio Tasks | Priority |
|--------|--------------|---------------|----------|
| **episodic.py** | H12: Episode encoding | B23: HPC-dependent | P0 |
| **fast_episodic.py** | H12: Fast binding | B23: One-shot | P0 |
| **semantic.py** | H12: Concept extraction | B24: Neocortical | P0 |
| **procedural.py** | H12: Skill learning | B25: Striatal | P1 |
| **working_memory.py** | H13: Capacity limits | B26: PFC-like | P0 |
| **buffer_manager.py** | H13: Buffer dynamics | - | P1 |
| **cluster_index.py** | H14: Hierarchical | - | P1 |
| **learned_sparse_index.py** | H14: Learned hashing | - | P1 |
| **pattern_separation.py** | H5: Orthogonalization | B27: DG function | P0 |
| **forgetting.py** | H15: Adaptive decay | B28: Interference | P1 |
| **unified.py** | - | - | P2 |
| **feature_aligner.py** | H14: Alignment | - | P2 |

### 1.4 Consolidation Region (5 modules)

| Module | Hinton Tasks | CompBio Tasks | Priority |
|--------|--------------|---------------|----------|
| **sleep.py** | H6: Sleep consolidation | B29: Stickgold | P0 |
| **service.py** | H6: Consolidation orchestration | - | P1 |
| **fes_consolidator.py** | H6: FES integration | - | P1 |
| **stdp_integration.py** | H7: STDP consolidation | B17: Timing | P1 |
| **parallel.py** | - | - | P2 |

### 1.5 Visualization Region (18 modules)

| Module | Hinton Tasks | CompBio Tasks | Priority |
|--------|--------------|---------------|----------|
| **telemetry_hub.py** | H16: Multi-scale | B30: Timescales | P0 |
| **ff_visualizer.py** | H1: Goodness viz | - | P1 |
| **capsule_visualizer.py** | H2: Routing viz | - | P1 |
| **glymphatic_visualizer.py** | - | B11: Clearance viz | P1 |
| **da_telemetry.py** | - | B4: DA dynamics | P0 |
| **stability_monitor.py** | H3: Bifurcation | B3: Stability | P0 |
| **nt_state_dashboard.py** | H3: 6D state | B13: NT viz | P0 |
| **pac_telemetry.py** | - | B9: PAC MI | P0 |
| **swr_telemetry.py** | - | B10: Replay | P0 |
| **energy_landscape.py** | H3: Energy surface | - | P1 |
| **coupling_dynamics.py** | H3: K matrix | - | P1 |
| *Others* | - | - | P2 |

---

## Phase 2: Agent Task Allocation

### 2.1 Hinton Agent (ww-hinton) - 16 Task Batches

#### Batch H1: Forward-Forward Algorithm
**Files**: forward_forward.py, forward_forward_nca_coupling.py, self_supervised.py
**Analysis Focus**:
- Goodness function G(h) = Σh² implementation
- Positive/negative phase separation
- Threshold θ learning dynamics
- Goodness ↔ NCA energy alignment
- Sleep-phase negative generation

**Documentation Output**:
- Update `docs/concepts/forward-forward.md`
- Create `docs/reference/ff-api.md`
- Update `docs/diagrams/ff_nca_coupling.mermaid`

**Validation Criteria**:
- Hinton 2022 paper alignment score
- Layer-local learning verification
- Energy landscape correspondence

---

#### Batch H2: Capsule Networks
**Files**: capsules.py, capsule_nca_coupling.py, pose.py
**Analysis Focus**:
- Dynamic routing implementation
- Pose vector transformations
- Part-whole hierarchies
- EM routing vs dynamic routing
- ACh-modulated routing temperature

**Documentation Output**:
- Update `docs/concepts/capsules.md`
- Create `docs/reference/capsule-api.md`
- Update `docs/diagrams/capsule_routing.mermaid`

**Validation Criteria**:
- Sabour et al. 2017 alignment
- Hinton 2018 matrix capsules
- Routing convergence properties

---

#### Batch H3: Energy Landscape & Stability
**Files**: energy.py, attractors.py, stability.py, neural_field.py, coupling.py
**Analysis Focus**:
- 6D NT state space geometry
- Attractor basin identification
- Bifurcation detection
- Cross-region coupling matrix K
- Lyapunov stability

**Documentation Output**:
- Update `docs/concepts/nca.md` (energy section)
- Create `docs/science/energy-dynamics.md`
- Update `docs/diagrams/nca_module_map.mermaid`

**Validation Criteria**:
- Hopfield network correspondence
- Free energy principle alignment
- Stability margin validation

---

#### Batch H4: Credit Assignment
**Files**: eligibility.py, credit_flow.py, dopamine.py, dopamine_integration.py, striatal_msn.py, striatal_coupling.py
**Analysis Focus**:
- TD(λ) implementation
- Eligibility trace dynamics
- RPE computation
- DA → learning rate modulation
- Direct/indirect pathway balance

**Documentation Output**:
- Update `docs/science/learning-theory.md` (credit section)
- Create `docs/reference/credit-api.md`
- Update `docs/diagrams/credit_assignment_flow.mermaid`

**Validation Criteria**:
- TD learning correctness
- λ decay verification
- RPE biological ranges

---

#### Batch H5: Memory Indexing & Retrieval
**Files**: hippocampus.py, spatial_cells.py, swr_coupling.py, pattern_separation.py, cluster_index.py, learned_sparse_index.py
**Analysis Focus**:
- DG → CA3 → CA1 flow
- Pattern separation/completion
- SWR-triggered replay
- Hierarchical indexing
- Learned sparse representations

**Documentation Output**:
- Update `docs/concepts/memory-types.md`
- Create `docs/reference/retrieval-api.md`
- Update `docs/diagrams/hippocampal_circuit.mermaid`

**Validation Criteria**:
- HPC circuit fidelity
- Replay sequence accuracy
- Index efficiency metrics

---

#### Batch H6: Sleep & Consolidation
**Files**: sleep.py, glymphatic_consolidation_bridge.py, service.py, fes_consolidator.py, connectome.py
**Analysis Focus**:
- Sleep stage transitions
- NREM/REM consolidation roles
- Replay-clearance coordination
- Memory pruning decisions
- Graph-based consolidation

**Documentation Output**:
- Update `docs/concepts/glymphatic.md`
- Create `docs/science/consolidation-theory.md`
- Update `docs/diagrams/sleep_cycle.mermaid`

**Validation Criteria**:
- Sleep architecture accuracy
- Consolidation timing
- Pruning effectiveness

---

#### Batch H7: Three-Factor Learning
**Files**: three_factor.py, stdp.py, stdp_integration.py
**Analysis Focus**:
- Pre × Post × Modulator rule
- STDP timing windows
- Neuromodulator gating
- Integration with eligibility

**Documentation Output**:
- Update `docs/science/learning-theory.md` (three-factor section)
- Create `docs/reference/stdp-api.md`

**Validation Criteria**:
- Three-factor rule correctness
- STDP window shapes
- Modulator influence

---

#### Batch H8: Neuromodulator Orchestra
**Files**: serotonin.py, norepinephrine.py, acetylcholine.py, neuromodulators.py
**Analysis Focus**:
- 5-HT patience/discount
- NE arousal/gain
- ACh encoding/retrieval
- Orchestra coordination

**Documentation Output**:
- Update `docs/concepts/brain-mapping.md`
- Create `docs/reference/neuromod-api.md`
- Update `docs/diagrams/neuromodulator_pathways.mermaid`

**Validation Criteria**:
- Receptor specificity
- Timescale separation
- Mode switching dynamics

---

#### Batch H9: Homeostatic Plasticity
**Files**: homeostatic.py, plasticity.py
**Analysis Focus**:
- Synaptic scaling rules
- Meta-plasticity (BCM theory)
- Stability-plasticity tradeoff

**Documentation Output**:
- Update `docs/science/learning-theory.md` (homeostatic section)

**Validation Criteria**:
- Turrigiano alignment
- BCM sliding threshold
- Firing rate homeostasis

---

#### Batch H10: Spaced Repetition
**Files**: fsrs.py, scorer.py
**Analysis Focus**:
- FSRS algorithm
- Memory strength scoring
- Optimal scheduling

**Documentation Output**:
- Create `docs/concepts/spaced-repetition.md`
- Create `docs/reference/fsrs-api.md`

**Validation Criteria**:
- FSRS paper alignment
- Retention curve accuracy

---

#### Batch H11: Neuro-Symbolic Integration
**Files**: causal_discovery.py, neuro_symbolic.py
**Analysis Focus**:
- Causal graph extraction
- Symbol grounding
- Neural-symbolic bridge

**Documentation Output**:
- Create `docs/concepts/neuro-symbolic.md`

**Validation Criteria**:
- Causal inference correctness
- Symbol-subsymbolic binding

---

#### Batch H12: Memory Types
**Files**: episodic.py, fast_episodic.py, semantic.py, procedural.py
**Analysis Focus**:
- Episode encoding/retrieval
- Semantic extraction
- Procedural skill learning
- Memory type interactions

**Documentation Output**:
- Update `docs/concepts/memory-types.md`
- Create `docs/reference/memory-api.md`
- Update `docs/diagrams/memory_lifecycle.mermaid`

**Validation Criteria**:
- Tulving's taxonomy
- HPC vs striatal dependence
- Consolidation trajectories

---

#### Batch H13: Working Memory
**Files**: working_memory.py, buffer_manager.py
**Analysis Focus**:
- Capacity limits (7±2)
- Buffer dynamics
- Theta-gamma binding

**Documentation Output**:
- Create `docs/concepts/working-memory.md`

**Validation Criteria**:
- Miller's law
- PFC-like maintenance
- Gamma-nested slots

---

#### Batch H14: Indexing & Alignment
**Files**: cluster_index.py, learned_sparse_index.py, feature_aligner.py
**Analysis Focus**:
- Hierarchical clustering
- Learned sparse hashing
- Feature alignment

**Documentation Output**:
- Create `docs/reference/indexing-api.md`

**Validation Criteria**:
- Retrieval efficiency
- Alignment accuracy

---

#### Batch H15: Forgetting & Interference
**Files**: forgetting.py
**Analysis Focus**:
- Adaptive decay
- Interference resolution
- Forgetting curves

**Documentation Output**:
- Create `docs/science/forgetting-theory.md`

**Validation Criteria**:
- Ebbinghaus alignment
- Interference patterns

---

#### Batch H16: Telemetry Integration
**Files**: telemetry_hub.py, ff_visualizer.py, capsule_visualizer.py, stability_monitor.py, energy_landscape.py, coupling_dynamics.py
**Analysis Focus**:
- Multi-scale integration
- Cross-module correlation
- Real-time monitoring

**Documentation Output**:
- Update `docs/reference/telemetry-api.md`

**Validation Criteria**:
- Timescale coverage
- Alert accuracy

---

### 2.2 CompBio Agent (ww-compbio) - 30 Task Batches

#### Batch B1: VTA Dopamine Circuit
**Files**: vta.py, dopamine_integration.py, dopamine.py
**Validation Focus**:
- Tonic/phasic firing (Grace & Bunney 1984)
- RPE computation (Schultz 1998)
- D1/D2 receptor affinities
- DA temporal dynamics (50-500ms)

**Literature Verification**:
- Schultz, W. (1998). Predictive reward signal of dopamine neurons
- Grace, A.A., & Bunney, B.S. (1984). The control of firing pattern

**Parameter Audit**:
| Parameter | Current | Literature | Source |
|-----------|---------|------------|--------|
| tonic_rate | 4 Hz | 3-5 Hz | Grace 1991 |
| phasic_burst | 20 Hz | 15-30 Hz | Schultz 1997 |
| D1_Kd | 1 μM | 0.5-2 μM | Seeman 2006 |
| D2_Kd | 10 nM | 5-20 nM | Seeman 2006 |

---

#### Batch B2: Capsule-Microcolumn Analogy
**Files**: capsules.py, capsule_nca_coupling.py
**Validation Focus**:
- Cortical microcolumn structure (Mountcastle 1997)
- Columnar processing (300-600 μm)
- Neuromodulatory gating

**Literature Verification**:
- Mountcastle, V.B. (1997). The columnar organization of the neocortex

---

#### Batch B3: Energy & Attractor Dynamics
**Files**: energy.py, attractors.py, stability.py
**Validation Focus**:
- Hopfield attractor properties
- Free energy principle (Friston 2010)
- Stability margins

**Literature Verification**:
- Hopfield, J.J. (1982). Neural networks and physical systems
- Friston, K. (2010). The free-energy principle

---

#### Batch B4: Dopamine Learning
**Files**: dopamine.py (learning module)
**Validation Focus**:
- DA burst/pause asymmetry
- Phasic responses (100-200 ms)
- Reward magnitude coding

---

#### Batch B5: Raphe Serotonin
**Files**: raphe.py, serotonin.py
**Validation Focus**:
- 5-HT receptor subtypes (1A, 2A, etc.)
- Patience/temporal discounting (Daw 2002)
- Tonic 5-HT modulation

**Literature Verification**:
- Jacobs, B.L., & Azmitia, E.C. (1992). Structure and function of serotonin
- Daw, N.D., Kakade, S., & Dayan, P. (2002). Opponent interactions

**Parameter Audit**:
| Parameter | Current | Literature | Source |
|-----------|---------|------------|--------|
| tonic_rate | 1-3 Hz | 0.5-3 Hz | Jacobs 1992 |
| 5HT1A_affinity | high | Ki ~1 nM | Hoyer 2002 |
| 5HT2A_affinity | moderate | Ki ~10 nM | Hoyer 2002 |

---

#### Batch B6: Locus Coeruleus NE
**Files**: locus_coeruleus.py, norepinephrine.py
**Validation Focus**:
- Tonic/phasic modes (Aston-Jones 2005)
- Arousal modulation
- Surprise/novelty detection
- Gain modulation

**Literature Verification**:
- Aston-Jones, G., & Cohen, J.D. (2005). An integrative theory of LC-NE function

**Parameter Audit**:
| Parameter | Current | Literature | Source |
|-----------|---------|------------|--------|
| tonic_rate | 0.5-5 Hz | 0.1-5 Hz | Aston-Jones 2005 |
| phasic_burst | 10-20 Hz | 8-20 Hz | Berridge 2003 |
| mode_switch | 0.3 | 0.2-0.5 | Aston-Jones 2005 |

---

#### Batch B7: Striatal MSN
**Files**: striatal_msn.py, striatal_coupling.py
**Validation Focus**:
- D1/D2 MSN populations
- Direct/indirect pathways
- TAN pause mechanism
- Action selection

**Literature Verification**:
- Surmeier, D.J. et al. (2007). D1 and D2 dopamine-receptor modulation
- Graybiel, A.M. (1995). Building action repertoires

**Parameter Audit**:
| Parameter | Current | Literature | Source |
|-----------|---------|------------|--------|
| D1_MSN_ratio | 50% | 45-55% | Gerfen 1990 |
| TAN_pause | 200 ms | 150-300 ms | Apicella 2007 |
| DA_D1_gain | 2x | 1.5-3x | Surmeier 2007 |

---

#### Batch B8: Hippocampal Circuit
**Files**: hippocampus.py, spatial_cells.py
**Validation Focus**:
- DG → CA3 → CA1 pathway
- Pattern separation (DG)
- Pattern completion (CA3)
- Place/grid cells (O'Keefe, Moser)

**Literature Verification**:
- O'Keefe, J., & Nadel, L. (1978). The Hippocampus as a Cognitive Map
- Moser, E.I. et al. (2008). Place cells, grid cells, and memory

---

#### Batch B9: Oscillators
**Files**: oscillators.py, sleep_spindles.py, theta_gamma_integration.py
**Validation Focus**:
- Theta band (4-8 Hz)
- Gamma band (30-100 Hz)
- Sleep spindles (11-16 Hz)
- Phase-amplitude coupling

**Literature Verification**:
- Buzsáki, G. (2006). Rhythms of the Brain
- Lisman, J.E., & Jensen, O. (2013). Theta-gamma neural code

**Parameter Audit**:
| Parameter | Current | Literature | Source |
|-----------|---------|------------|--------|
| theta_freq | 6 Hz | 4-8 Hz | Buzsáki 2006 |
| gamma_freq | 40 Hz | 30-80 Hz | Buzsáki 2006 |
| spindle_freq | 12 Hz | 11-16 Hz | Steriade 1993 |
| PAC_MI | 0.4 | 0.3-0.7 | Canolty 2006 |

---

#### Batch B10: SWR Replay
**Files**: swr_coupling.py
**Validation Focus**:
- Ripple frequency (150-250 Hz)
- Replay compression (5-20x)
- Forward/reverse replay
- HPC-cortical coordination

**Literature Verification**:
- Buzsáki, G. (1989). Two-stage model of memory trace formation
- Foster, D.J., & Wilson, M.A. (2006). Reverse replay

---

#### Batch B11: Glymphatic System
**Files**: glymphatic.py, glymphatic_consolidation_bridge.py
**Validation Focus**:
- AQP4 channel polarization
- CSF-ISF exchange
- Sleep-dependent clearance (60% increase)
- Amyloid-beta clearance

**Literature Verification**:
- Xie, L. et al. (2013). Sleep drives metabolite clearance
- Nedergaard, M. (2013). Garbage truck of the brain

**Parameter Audit**:
| Parameter | Current | Literature | Source |
|-----------|---------|------------|--------|
| sleep_clearance_boost | 1.6x | 1.6x | Xie 2013 |
| AQP4_polarization | 0.7 | 0.6-0.8 | Iliff 2012 |
| CSF_flow_rate | Variable | Sleep-gated | Fultz 2019 |

---

#### Batch B12: Adenosine Sleep Pressure
**Files**: adenosine.py
**Validation Focus**:
- Sleep homeostasis (Borbély 1982)
- Adenosine accumulation
- A1/A2A receptor dynamics

**Literature Verification**:
- Borbély, A.A. (1982). A two process model of sleep regulation

---

#### Batch B13: Neural Field Dynamics
**Files**: neural_field.py, coupling.py
**Validation Focus**:
- 6D NT state space
- Coupling constant K matrix
- Field stability

---

#### Batch B14: Connectome & Delays
**Files**: connectome.py, delays.py
**Validation Focus**:
- Anatomical connectivity
- Axonal conduction delays (1-10 ms/mm)

---

#### Batch B15: Glial Cells
**Files**: astrocyte.py, glutamate_signaling.py
**Validation Focus**:
- Tripartite synapse
- AMPA/NMDA dynamics
- Glutamate reuptake

**Literature Verification**:
- Araque, A. et al. (1999). Tripartite synapses

---

#### Batch B16-B30: Additional Validation Tasks
(Similar structure for remaining modules)

---

## Phase 3: Diagram Generation Plan

### 3.1 New Mermaid Diagrams Required (22)

| # | Diagram | Agent | Priority |
|---|---------|-------|----------|
| 1 | `ff_layer_stack.mermaid` | Hinton | P0 |
| 2 | `ff_goodness_energy.mermaid` | Hinton | P0 |
| 3 | `capsule_hierarchy.mermaid` | Hinton | P0 |
| 4 | `capsule_routing_algorithm.mermaid` | Hinton | P0 |
| 5 | `energy_attractors.mermaid` | Hinton | P1 |
| 6 | `nt_6d_state.mermaid` | CompBio | P0 |
| 7 | `vta_circuit.mermaid` | CompBio | P0 |
| 8 | `raphe_circuit.mermaid` | CompBio | P0 |
| 9 | `lc_circuit.mermaid` | CompBio | P0 |
| 10 | `striatal_pathways.mermaid` | CompBio | P0 |
| 11 | `hpc_trisynaptic.mermaid` | CompBio | P0 |
| 12 | `oscillator_bands.mermaid` | CompBio | P0 |
| 13 | `pac_coupling.mermaid` | CompBio | P1 |
| 14 | `swr_replay.mermaid` | CompBio | P0 |
| 15 | `glymphatic_flow.mermaid` | CompBio | P0 |
| 16 | `sleep_stages.mermaid` | CompBio | P0 |
| 17 | `eligibility_trace.mermaid` | Hinton | P0 |
| 18 | `three_factor_rule.mermaid` | Hinton | P0 |
| 19 | `memory_consolidation.mermaid` | Both | P0 |
| 20 | `system_integration.mermaid` | Both | P0 |
| 21 | `telemetry_timescales.mermaid` | Both | P1 |
| 22 | `cross_region_coupling.mermaid` | Both | P0 |

### 3.2 Existing Diagrams to Update (8)

| Diagram | Updates Needed |
|---------|----------------|
| `ff_nca_coupling.mermaid` | Add Phase 4 connections |
| `capsule_routing.mermaid` | Add ACh gating |
| `hippocampal_circuit.mermaid` | Add spatial cells |
| `neuromodulator_pathways.mermaid` | Add receptor details |
| `sleep_cycle.mermaid` | Add glymphatic overlay |
| `memory_lifecycle.mermaid` | Add forgetting paths |
| `credit_assignment_flow.mermaid` | Add eligibility |
| `nca_module_map.mermaid` | Add Phase 4 modules |

---

## Phase 4: Documentation Structure

### 4.1 New Documents Required

| Document | Location | Agent | Priority |
|----------|----------|-------|----------|
| `ff-api.md` | reference/ | Hinton | P0 |
| `capsule-api.md` | reference/ | Hinton | P0 |
| `credit-api.md` | reference/ | Hinton | P0 |
| `retrieval-api.md` | reference/ | Hinton | P1 |
| `neuromod-api.md` | reference/ | CompBio | P0 |
| `stdp-api.md` | reference/ | CompBio | P0 |
| `memory-api.md` | reference/ | Hinton | P1 |
| `telemetry-api.md` | reference/ | Both | P1 |
| `indexing-api.md` | reference/ | Hinton | P2 |
| `fsrs-api.md` | reference/ | Hinton | P2 |
| `energy-dynamics.md` | science/ | Hinton | P0 |
| `consolidation-theory.md` | science/ | Both | P0 |
| `forgetting-theory.md` | science/ | Hinton | P1 |
| `spaced-repetition.md` | concepts/ | Hinton | P1 |
| `working-memory.md` | concepts/ | Hinton | P1 |
| `neuro-symbolic.md` | concepts/ | Hinton | P2 |
| `biological-parameters.md` | science/ | CompBio | P0 |

### 4.2 Documents to Update

| Document | Updates | Agent |
|----------|---------|-------|
| `concepts/forward-forward.md` | H1 findings | Hinton |
| `concepts/capsules.md` | H2 findings | Hinton |
| `concepts/nca.md` | H3 energy section | Hinton |
| `concepts/memory-types.md` | H5, H12 findings | Hinton |
| `concepts/glymphatic.md` | B11 validation | CompBio |
| `concepts/brain-mapping.md` | All B-tasks | CompBio |
| `science/learning-theory.md` | H4, H7, H9 | Hinton |
| `science/biology-audit.md` | B1-B30 results | CompBio |

---

## Phase 5: Execution Schedule

### Sprint 1: Foundation (Days 1-3)

**Day 1 - Parallel Launch**:
- ww-hinton: H1 (FF) + H2 (Capsules)
- ww-compbio: B1 (VTA) + B5 (Raphe) + B6 (LC)

**Day 2 - Core Systems**:
- ww-hinton: H3 (Energy) + H4 (Credit)
- ww-compbio: B7 (Striatum) + B8 (HPC) + B9 (Oscillators)

**Day 3 - Integration**:
- ww-hinton: H5 (Memory) + H6 (Sleep)
- ww-compbio: B10 (SWR) + B11 (Glymphatic) + B12 (Adenosine)

### Sprint 2: Learning & Plasticity (Days 4-5)

**Day 4**:
- ww-hinton: H7 (Three-Factor) + H8 (Neuromod)
- ww-compbio: B16-B18 (STDP, learning validation)

**Day 5**:
- ww-hinton: H9 (Homeostatic) + H10 (FSRS)
- ww-compbio: B19-B22 (Plasticity validation)

### Sprint 3: Memory & Advanced (Days 6-7)

**Day 6**:
- ww-hinton: H11 (Neuro-Symbolic) + H12 (Memory Types)
- ww-compbio: B23-B25 (Memory system validation)

**Day 7**:
- ww-hinton: H13-H16 (WM, Indexing, Telemetry)
- ww-compbio: B26-B30 (Remaining validation)

### Sprint 4: Integration & Polish (Days 8-10)

**Day 8**:
- Cross-validate agent findings
- Resolve parameter conflicts
- Generate integrated diagrams

**Day 9**:
- Update all documentation
- Generate remaining diagrams
- Run full test suite

**Day 10**:
- Final review
- Quality metrics verification
- Knowledge base update

---

## Phase 6: Quality Metrics

### 6.1 Target Scores

| Metric | Current | Sprint 1 | Sprint 2 | Sprint 3 | Final |
|--------|---------|----------|----------|----------|-------|
| Hinton Score | 7.9/10 | 8.5/10 | 9.0/10 | 9.3/10 | 9.5/10 |
| Biology Score | 94/100 | 95/100 | 96/100 | 97/100 | 98/100 |
| Doc Coverage | 60% | 75% | 85% | 95% | 100% |
| Diagram Count | 8 | 15 | 22 | 28 | 30 |
| Test Coverage | 79% | 82% | 85% | 88% | 90% |

### 6.2 Hinton Score Breakdown

| Component | Current | Target | Gap |
|-----------|---------|--------|-----|
| FF Algorithm | 0.8 | 1.0 | TD(λ) credit |
| Capsule Networks | 0.7 | 0.9 | EM routing |
| Energy Landscape | 0.9 | 1.0 | Attractor validation |
| Credit Assignment | 0.7 | 0.95 | TD(λ) implementation |
| Memory Indexing | 0.8 | 0.95 | Learned sparse |
| Sleep Consolidation | 0.8 | 0.95 | Negative generation |
| Representation | 0.8 | 0.95 | Distributed codes |
| TOTAL | 7.9/10 | 9.5/10 | +1.6 |

### 6.3 Biology Score Breakdown

| System | Current | Target | Gap |
|--------|---------|--------|-----|
| VTA/DA | 92/100 | 98/100 | Receptor kinetics |
| Raphe/5-HT | 89/100 | 96/100 | Receptor subtypes |
| LC/NE | 91/100 | 97/100 | Mode dynamics |
| Striatum | 90/100 | 96/100 | TAN timing |
| HPC | 93/100 | 98/100 | Spatial cells |
| Oscillators | 95/100 | 99/100 | PAC validation |
| Glymphatic | 94/100 | 99/100 | Clearance rates |
| Sleep | 93/100 | 98/100 | Stage transitions |
| TOTAL | 94/100 | 98/100 | +4 |

---

## Phase 7: Agent Prompts

### 7.1 ww-hinton Agent Template

```
TASK: [BATCH_ID] - [BATCH_NAME]

FILES TO ANALYZE:
- [file1.py]
- [file2.py]
- ...

ANALYSIS FOCUS:
1. [Focus point 1]
2. [Focus point 2]
3. ...

HINTON CRITERIA:
- Does implementation follow Hinton's principles?
- Is layer-local learning implemented correctly?
- Are representations distributed appropriately?
- Is the energy landscape well-formed?

DELIVERABLES:
1. Code analysis summary (findings, issues, improvements)
2. Documentation updates (specific file paths)
3. Diagram specifications (mermaid syntax)
4. Test recommendations

VALIDATION:
- Score current implementation (0-1)
- Identify specific gaps
- Propose improvements with priorities

OUTPUT FORMAT:
## Analysis Summary
[Summary]

## Code Issues
| File | Line | Issue | Severity | Fix |
|------|------|-------|----------|-----|

## Documentation Updates
[Specific updates with file paths]

## Diagrams
[Mermaid code blocks]

## Score
| Criterion | Score | Notes |
|-----------|-------|-------|
```

### 7.2 ww-compbio Agent Template

```
TASK: [BATCH_ID] - [BATCH_NAME]

FILES TO VALIDATE:
- [file1.py]
- [file2.py]
- ...

BIOLOGICAL SYSTEMS:
1. [System 1]
2. [System 2]
3. ...

LITERATURE VERIFICATION:
- [Author Year]: [Finding to verify]
- ...

PARAMETER AUDIT:
| Parameter | Code Value | Literature Range | Source |
|-----------|------------|------------------|--------|

VALIDATION CRITERIA:
- Are biological ranges respected?
- Are timing dynamics accurate?
- Are receptor affinities correct?
- Is anatomical organization preserved?

DELIVERABLES:
1. Parameter audit report
2. Literature verification results
3. Documentation updates
4. Diagram specifications
5. Test recommendations

OUTPUT FORMAT:
## Biological Validation Summary
[Summary]

## Parameter Audit
| Parameter | Current | Literature | Status | Fix |
|-----------|---------|------------|--------|-----|

## Literature Verification
| Citation | Claim | Verified | Notes |
|----------|-------|----------|-------|

## Documentation Updates
[Specific updates]

## Score
| System | Score | Notes |
|--------|-------|-------|
```

---

## Appendix A: Module-to-Document Mapping

| Module | Primary Doc | Secondary Docs |
|--------|-------------|----------------|
| forward_forward.py | concepts/forward-forward.md | reference/ff-api.md |
| capsules.py | concepts/capsules.md | reference/capsule-api.md |
| energy.py | concepts/nca.md | science/energy-dynamics.md |
| vta.py | concepts/brain-mapping.md | science/biology-audit.md |
| hippocampus.py | concepts/memory-types.md | reference/retrieval-api.md |
| ... | ... | ... |

---

## Appendix B: Diagram-to-Module Mapping

| Diagram | Modules Covered | Type |
|---------|-----------------|------|
| ff_nca_coupling.mermaid | forward_forward*.py | flowchart |
| capsule_routing.mermaid | capsules.py, pose.py | sequence |
| vta_circuit.mermaid | vta.py, dopamine*.py | flowchart |
| ... | ... | ... |

---

## Appendix C: Test Coverage Requirements

| Module | Current | Target | Tests Needed |
|--------|---------|--------|--------------|
| forward_forward.py | 85% | 95% | +12 |
| capsules.py | 82% | 95% | +15 |
| vta.py | 88% | 95% | +8 |
| hippocampus.py | 80% | 95% | +18 |
| ... | ... | ... | ... |

---

**Plan Status**: Ready for Execution
**Estimated Duration**: 10 days
**Agent Hours**: ~80 (40 each)
**Output**: 100% documented, 98/100 biology, 9.5/10 Hinton
