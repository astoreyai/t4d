# World Weaver Comprehensive Documentation Analysis Plan V3

**Version**: 3.0
**Created**: 2026-01-04
**Status**: Active - Sprint 2 Ready
**Agents**: ww-hinton (Neural Architecture) + ww-compbio (Biological Validation)

---

## Executive Summary

This plan orchestrates two specialized Opus agents to comprehensively analyze and document the World Weaver codebase. V3 reflects completed Sprint 1 work and plans remaining analysis.

### Current Progress

| Metric | Start | After Sprint 1 | Target |
|--------|-------|----------------|--------|
| Hinton Score | 7.9/10 | 8.4/10 | 9.5/10 |
| Biology Score | 94/100 | 95/100 | 98/100 |
| Doc Coverage | 60% | 70% | 100% |
| Mermaid Diagrams | 8 | 8 | 30 |
| API Docs Created | 1 | 4 | 17 |

### Sprint 1 Completed Tasks

**ww-hinton (H1-H6)**:
- [x] H1: Forward-Forward (0.85/1.0) - ff-api.md created
- [x] H2: Capsules (0.82/1.0) - capsule-api.md created
- [x] H3: Energy Landscape (0.90/1.0)
- [x] H4: Credit Assignment (0.78/1.0)
- [x] H5: Memory Indexing (0.88/1.0)
- [x] H6: Sleep Consolidation (0.83/1.0)

**ww-compbio (B1-B12)**:
- [x] B1-B4: VTA/DA (88/100)
- [x] B5: Raphe (92/100)
- [x] B6: LC (90/100)
- [x] B7: Striatum (85/100)
- [x] B8: HPC (94/100)
- [x] B9: Oscillators (96/100)
- [x] B10: SWR (93/100)
- [x] B11: Glymphatic (89/100) - **FIXED**: clearance 0.9→0.7
- [x] B12: Adenosine (91/100)

**Critical Fixes Applied**:
- sleep.py Protocol methods added
- glymphatic.py clearance_nrem_deep: 0.9 → 0.7 (Xie et al. 2013)

---

## Module Inventory (195 Python Files)

### Category 1: NCA Region (29 modules) - 70% ANALYZED

| Module | Status | Hinton | CompBio | Priority |
|--------|--------|--------|---------|----------|
| forward_forward.py | ✓ Done | H1: 0.85 | - | P0 |
| forward_forward_nca_coupling.py | ✓ Done | H1: 0.85 | B1 | P0 |
| capsules.py | ✓ Done | H2: 0.82 | B2 | P0 |
| capsule_nca_coupling.py | ✓ Done | H2: 0.82 | - | P0 |
| pose.py | ✓ Done | H2: 0.82 | - | P1 |
| energy.py | ✓ Done | H3: 0.90 | B3 | P0 |
| attractors.py | ✓ Done | H3: 0.90 | - | P1 |
| stability.py | ✓ Done | H3: 0.90 | - | P1 |
| neural_field.py | ✓ Done | H3: 0.90 | B13 | P0 |
| coupling.py | Pending | H3-ext | B13 | P1 |
| vta.py | ✓ Done | - | B1: 88 | P0 |
| dopamine_integration.py | ✓ Done | H4: 0.78 | B4 | P0 |
| raphe.py | ✓ Done | - | B5: 92 | P0 |
| locus_coeruleus.py | ✓ Done | - | B6: 90 | P0 |
| striatal_msn.py | ✓ Done | H4: 0.78 | B7: 85 | P0 |
| striatal_coupling.py | Pending | H4-ext | B7 | P1 |
| hippocampus.py | ✓ Done | H5: 0.88 | B8: 94 | P0 |
| spatial_cells.py | ✓ Done | H5: 0.88 | B8 | P1 |
| oscillators.py | ✓ Done | - | B9: 96 | P0 |
| sleep_spindles.py | Pending | - | B9-ext | P1 |
| theta_gamma_integration.py | Pending | - | B9-ext | P1 |
| swr_coupling.py | ✓ Done | H5: 0.88 | B10: 93 | P0 |
| glymphatic.py | ✓ FIXED | - | B11: 89 | P0 |
| glymphatic_consolidation_bridge.py | ✓ Done | H6: 0.83 | B11 | P0 |
| adenosine.py | ✓ Done | - | B12: 91 | P1 |
| delays.py | Pending | - | B14 | P2 |
| connectome.py | Pending | H6-ext | B14 | P1 |
| astrocyte.py | Pending | - | B15 | P1 |
| glutamate_signaling.py | Pending | - | B15 | P1 |

### Category 2: Learning Region (23 modules) - 20% ANALYZED

| Module | Status | Hinton | CompBio | Priority |
|--------|--------|--------|---------|----------|
| eligibility.py | Pending | H7 | B16 | P0 |
| credit_flow.py | Pending | H7 | B16 | P0 |
| three_factor.py | Pending | H7 | B17 | P0 |
| stdp.py | Pending | - | B17 | P0 |
| dopamine.py | Pending | H4-ext | B4 | P0 |
| serotonin.py | Pending | H8 | B5 | P0 |
| norepinephrine.py | Pending | H8 | B6 | P0 |
| acetylcholine.py | Pending | H8 | B18 | P0 |
| neuromodulators.py | Pending | H8 | B18 | P1 |
| homeostatic.py | Pending | H9 | B19 | P1 |
| plasticity.py | Pending | H9 | B19 | P1 |
| fsrs.py | Pending | H10 | - | P1 |
| scorer.py | Pending | H10 | - | P1 |
| causal_discovery.py | Pending | H11 | - | P1 |
| neuro_symbolic.py | Pending | H11 | - | P1 |
| self_supervised.py | Pending | H1-ext | B20 | P1 |
| inhibition.py | Pending | - | B21 | P1 |
| reconsolidation.py | Pending | - | B22 | P1 |
| collector.py | Pending | - | - | P2 |
| events.py | Pending | - | - | P2 |
| hooks.py | Pending | - | - | P2 |
| persistence.py | Pending | - | - | P2 |
| cold_start.py | Pending | - | - | P2 |

### Category 3: Memory Region (12 modules) - 30% ANALYZED

| Module | Status | Hinton | CompBio | Priority |
|--------|--------|--------|---------|----------|
| episodic.py | Pending | H12 | B23 | P0 |
| fast_episodic.py | Pending | H12 | B23 | P0 |
| semantic.py | Pending | H12 | B24 | P0 |
| procedural.py | Pending | H12 | B25 | P1 |
| working_memory.py | Pending | H13 | B26 | P0 |
| buffer_manager.py | Pending | H13 | - | P1 |
| cluster_index.py | Pending | H14 | - | P1 |
| learned_sparse_index.py | Pending | H14 | - | P1 |
| pattern_separation.py | ✓ Done | H5: 0.88 | B27 | P0 |
| forgetting.py | Pending | H15 | B28 | P1 |
| unified.py | Pending | - | - | P2 |
| feature_aligner.py | Pending | H14 | - | P2 |

### Category 4: Consolidation (5 modules) - 40% ANALYZED

| Module | Status | Hinton | CompBio | Priority |
|--------|--------|--------|---------|----------|
| sleep.py | ✓ FIXED | H6: 0.83 | B29 | P0 |
| service.py | Pending | H6-ext | - | P1 |
| fes_consolidator.py | Pending | H6-ext | - | P1 |
| stdp_integration.py | Pending | H7 | B17 | P1 |
| parallel.py | Pending | - | - | P2 |

### Category 5: Visualization (18 modules) - 30% ANALYZED

| Module | Status | Hinton | CompBio | Priority |
|--------|--------|--------|---------|----------|
| telemetry_hub.py | ✓ Updated | H16 | B30 | P0 |
| ff_visualizer.py | ✓ Done | H1 | - | P1 |
| capsule_visualizer.py | ✓ Done | H2 | - | P1 |
| glymphatic_visualizer.py | ✓ Done | - | B11 | P1 |
| da_telemetry.py | Pending | - | B4 | P0 |
| stability_monitor.py | Pending | H3 | B3 | P0 |
| nt_state_dashboard.py | Pending | H3 | B13 | P0 |
| pac_telemetry.py | Pending | - | B9 | P0 |
| swr_telemetry.py | Pending | - | B10 | P0 |
| energy_landscape.py | Pending | H3 | - | P1 |
| coupling_dynamics.py | Pending | H3 | - | P1 |
| activation_heatmap.py | Pending | - | - | P2 |
| embedding_projections.py | Pending | - | - | P2 |
| neuromodulator_state.py | Pending | H8 | - | P1 |
| consolidation_replay.py | Pending | H6 | B10 | P1 |
| plasticity_traces.py | Pending | H7 | B17 | P1 |
| pattern_separation.py | Pending | H5 | B27 | P1 |
| validation.py | Pending | - | - | P2 |
| persistence_state.py | Pending | - | - | P2 |

### Category 6: Other Regions (108 modules)

| Region | Count | Status | Priority |
|--------|-------|--------|----------|
| API | 12 | Pending | P2 |
| Bridge | 2 | Pending | P1 |
| CLI | 2 | Pending | P2 |
| Core | 14 | Pending | P1 |
| Dreaming | 4 | Pending | P1 |
| Embedding | 10 | Pending | P1 |
| Encoding | 4 | Pending | P1 |
| Extraction | 2 | Pending | P2 |
| Hooks | 6 | Pending | P2 |
| Integration | 4 | Pending | P2 |
| Integrations/Kymera | 10 | Pending | P2 |
| Interfaces | 8 | Pending | P2 |
| Observability | 6 | Pending | P2 |
| Persistence | 6 | Pending | P1 |
| Prediction | 6 | Pending | P1 |
| SDK | 3 | Pending | P2 |
| Storage | 6 | Pending | P1 |
| Temporal | 4 | Pending | P1 |

---

## Sprint 2 Plan: Learning & Plasticity

### Hinton Agent Tasks (H7-H11)

#### H7: Three-Factor Learning & Credit Assignment
**Files**: eligibility.py, credit_flow.py, three_factor.py, stdp_integration.py
**Focus**:
- TD(λ) implementation correctness
- Eligibility trace dynamics (γλ decay)
- Pre × Post × Modulator rule
- Integration with DA/5-HT/NE/ACh

**Validation Criteria**:
- Eligibility decay matches λ=0.9-0.95
- Three-factor gating correct
- DA modulation follows RPE

**Output**: Update science/learning-theory.md, create reference/credit-api.md

---

#### H8: Neuromodulator Orchestra
**Files**: serotonin.py, norepinephrine.py, acetylcholine.py, neuromodulators.py
**Focus**:
- 5-HT → patience/discount (γ modulation)
- NE → arousal/gain (inverted-U)
- ACh → encoding/retrieval gating
- Orchestra coordination

**Validation Criteria**:
- 5-HT γ range: 0.85-0.97
- NE gain: inverted-U validated
- ACh threshold: enc > ret

**Output**: Create reference/neuromod-api.md, update diagrams

---

#### H9: Homeostatic Plasticity
**Files**: homeostatic.py, plasticity.py
**Focus**:
- Synaptic scaling (Turrigiano)
- BCM sliding threshold
- Stability-plasticity tradeoff

**Validation Criteria**:
- Scaling maintains E/I balance
- BCM θ adapts correctly

**Output**: Update science/learning-theory.md (homeostatic section)

---

#### H10: Spaced Repetition
**Files**: fsrs.py, scorer.py
**Focus**:
- FSRS algorithm fidelity
- Memory strength scoring
- Retrievability curves

**Validation Criteria**:
- R(t,S) = (1 + 0.9t/S)^(-0.5)
- Retention target: 0.9

**Output**: Create concepts/spaced-repetition.md, reference/fsrs-api.md

---

#### H11: Neuro-Symbolic Integration
**Files**: causal_discovery.py, neuro_symbolic.py
**Focus**:
- Causal graph extraction
- Symbol grounding
- Neural↔symbolic bridge

**Output**: Create concepts/neuro-symbolic.md

---

### CompBio Agent Tasks (B13-B22)

#### B13: Neural Field Dynamics
**Files**: neural_field.py, coupling.py
**Focus**:
- 6D NT state space geometry
- Coupling matrix K biological constraints
- Field stability criteria

**Parameter Audit**:
| Parameter | Current | Literature | Source |
|-----------|---------|------------|--------|
| alpha_da | 10.0 | 8-12 Hz | Grace 1991 |
| alpha_5ht | 2.0 | 1-3 Hz | Jacobs 1992 |
| diffusion_da | 0.1 | 0.05-0.15 | Rice 2000 |

---

#### B14: Connectome & Delays
**Files**: connectome.py, delays.py
**Focus**:
- Anatomical connectivity patterns
- Axonal conduction delays (1-10 ms/mm)
- Region-specific modulation

---

#### B15: Glial Cells
**Files**: astrocyte.py, glutamate_signaling.py
**Focus**:
- Tripartite synapse (Araque 1999)
- AMPA/NMDA receptor dynamics
- Glutamate reuptake timing

---

#### B16: STDP Windows
**Files**: eligibility.py, credit_flow.py, stdp.py
**Focus**:
- STDP timing windows (20-50 ms)
- Asymmetric LTP/LTD
- Eligibility decay rates

**Literature Verification**:
- Bi & Poo (1998): Timing-dependent plasticity
- Markram et al. (1997): Regulation of synaptic efficacy

---

#### B17: Three-Factor Modulation
**Files**: three_factor.py, stdp_integration.py
**Focus**:
- DA gating of STDP
- 5-HT modulation of plasticity
- Timing of modulator arrival

---

#### B18: Acetylcholine
**Files**: acetylcholine.py (learning)
**Focus**:
- Hasselmo encoding/retrieval model
- Muscarinic M1/M4 effects
- Nicotinic attention modulation

**Literature Verification**:
- Hasselmo, M.E. (2006): The role of acetylcholine in learning and memory

---

#### B19: Homeostatic Mechanisms
**Files**: homeostatic.py, plasticity.py
**Focus**:
- Turrigiano synaptic scaling
- TNF-α signaling
- Sleep-dependent downscaling

---

#### B20-B22: Additional Validation
**Files**: self_supervised.py, inhibition.py, reconsolidation.py
**Focus**:
- Self-organization principles
- GABAergic inhibition
- Memory reconsolidation

---

## Sprint 3 Plan: Memory & Advanced Systems

### Hinton Agent (H12-H16)

| Batch | Files | Focus |
|-------|-------|-------|
| H12 | episodic.py, semantic.py, procedural.py | Memory type encoding |
| H13 | working_memory.py, buffer_manager.py | Capacity limits (7±2) |
| H14 | cluster_index.py, learned_sparse_index.py | Indexing efficiency |
| H15 | forgetting.py | Adaptive decay |
| H16 | telemetry_hub.py, *_visualizer.py | Multi-scale integration |

### CompBio Agent (B23-B30)

| Batch | Files | Focus |
|-------|-------|-------|
| B23 | episodic.py, fast_episodic.py | HPC-dependent encoding |
| B24 | semantic.py | Neocortical semantics |
| B25 | procedural.py | Striatal skills |
| B26 | working_memory.py | PFC maintenance |
| B27 | pattern_separation.py | DG function |
| B28 | forgetting.py | Interference |
| B29 | sleep.py | Stickgold consolidation |
| B30 | telemetry_hub.py | Timescale validation |

---

## Diagram Generation Plan

### Existing Diagrams (8) - Update Required

| Diagram | Updates Needed | Status |
|---------|----------------|--------|
| ff_nca_coupling.mermaid | Phase 4 glymphatic | ✓ Updated |
| capsule_routing.mermaid | ACh gating details | Pending |
| hippocampal_circuit.mermaid | Spatial cell connections | Pending |
| neuromodulator_pathways.mermaid | Receptor subtype details | Pending |
| sleep_cycle.mermaid | Glymphatic clearance rates | ✓ Updated |
| memory_lifecycle.mermaid | Forgetting paths | Pending |
| credit_assignment_flow.mermaid | Eligibility traces | Pending |
| nca_module_map.mermaid | Phase 4 modules | ✓ Updated |

### New Diagrams Required (22)

| # | Name | Agent | Priority | Status |
|---|------|-------|----------|--------|
| 1 | ff_layer_stack.mermaid | Hinton | P0 | Pending |
| 2 | ff_goodness_energy.mermaid | Hinton | P0 | Pending |
| 3 | capsule_hierarchy.mermaid | Hinton | P0 | Pending |
| 4 | capsule_routing_detail.mermaid | Hinton | P0 | Pending |
| 5 | energy_attractors.mermaid | Hinton | P1 | Pending |
| 6 | nt_6d_state.mermaid | CompBio | P0 | Pending |
| 7 | vta_circuit.mermaid | CompBio | P0 | Pending |
| 8 | raphe_circuit.mermaid | CompBio | P0 | Pending |
| 9 | lc_circuit.mermaid | CompBio | P0 | Pending |
| 10 | striatal_pathways.mermaid | CompBio | P0 | Pending |
| 11 | hpc_trisynaptic.mermaid | CompBio | P0 | Pending |
| 12 | oscillator_bands.mermaid | CompBio | P0 | Pending |
| 13 | pac_coupling.mermaid | CompBio | P1 | Pending |
| 14 | swr_replay.mermaid | CompBio | P0 | Pending |
| 15 | glymphatic_flow.mermaid | CompBio | P0 | Pending |
| 16 | sleep_stages.mermaid | CompBio | P0 | Pending |
| 17 | eligibility_trace.mermaid | Hinton | P0 | Pending |
| 18 | three_factor_rule.mermaid | Hinton | P0 | Pending |
| 19 | memory_consolidation.mermaid | Both | P0 | Pending |
| 20 | system_integration.mermaid | Both | P0 | Pending |
| 21 | telemetry_timescales.mermaid | Both | P1 | Pending |
| 22 | cross_region_coupling.mermaid | Both | P0 | Pending |

---

## Documentation Status

### API Reference Docs

| Document | Status | Agent |
|----------|--------|-------|
| ff-api.md | ✓ Created | Hinton |
| capsule-api.md | ✓ Created | Hinton |
| nca-api.md | ✓ Updated | Both |
| glymphatic-api.md | ✓ Created | CompBio |
| credit-api.md | Pending | Hinton |
| neuromod-api.md | Pending | CompBio |
| stdp-api.md | Pending | CompBio |
| memory-api.md | Pending | Hinton |
| retrieval-api.md | Pending | Hinton |
| telemetry-api.md | Pending | Both |
| indexing-api.md | Pending | Hinton |
| fsrs-api.md | Pending | Hinton |

### Concept Docs

| Document | Status | Updates Needed |
|----------|--------|----------------|
| forward-forward.md | Exists | Add adaptive θ |
| capsules.md | Exists | Add EM routing |
| nca.md | Exists | Add energy section |
| glymphatic.md | Exists | Update clearance rates |
| memory-types.md | Exists | Add H12 findings |
| brain-mapping.md | Exists | Add B1-B30 |
| spaced-repetition.md | Pending | Create new |
| working-memory.md | Pending | Create new |
| neuro-symbolic.md | Pending | Create new |

### Science Docs

| Document | Status | Updates Needed |
|----------|--------|----------------|
| learning-theory.md | Exists | Add H4,H7,H9 |
| biology-audit.md | Exists | Add B1-B30 |
| energy-dynamics.md | Pending | Create new |
| consolidation-theory.md | Pending | Create new |
| forgetting-theory.md | Pending | Create new |
| biological-parameters.md | Pending | Create new |

---

## Execution Schedule

### Sprint 2: Learning & Plasticity (Days 1-3)

**Day 1 - Parallel Launch**:
```
ww-hinton: H7 (Credit) + H8 (Neuromod)
ww-compbio: B13 (Neural Field) + B14 (Connectome) + B15 (Glial)
```

**Day 2 - Core Learning**:
```
ww-hinton: H9 (Homeostatic) + H10 (FSRS)
ww-compbio: B16 (STDP) + B17 (Three-Factor) + B18 (ACh)
```

**Day 3 - Integration**:
```
ww-hinton: H11 (Neuro-Symbolic)
ww-compbio: B19 (Homeostatic) + B20-B22 (Self-org, Inhibition, Reconsolidation)
```

### Sprint 3: Memory & Advanced (Days 4-6)

**Day 4**:
```
ww-hinton: H12 (Memory Types) + H13 (Working Memory)
ww-compbio: B23 (Episodic) + B24 (Semantic) + B25 (Procedural)
```

**Day 5**:
```
ww-hinton: H14 (Indexing) + H15 (Forgetting)
ww-compbio: B26 (WM) + B27 (Pattern Sep) + B28 (Interference)
```

**Day 6**:
```
ww-hinton: H16 (Telemetry)
ww-compbio: B29 (Consolidation) + B30 (Timescales)
```

### Sprint 4: Integration & Polish (Days 7-10)

**Day 7-8**: Generate all 22 new diagrams
**Day 9**: Create remaining API docs (8)
**Day 10**: Final validation, knowledge base update

---

## Quality Metrics & Targets

### Score Progression

| Metric | Sprint 1 | Sprint 2 | Sprint 3 | Final |
|--------|----------|----------|----------|-------|
| Hinton Score | 8.4/10 | 9.0/10 | 9.3/10 | 9.5/10 |
| Biology Score | 95/100 | 97/100 | 98/100 | 98/100 |
| Doc Coverage | 70% | 85% | 95% | 100% |
| Diagrams | 8 | 18 | 26 | 30 |
| API Docs | 4 | 8 | 12 | 17 |
| Tests | 79% | 85% | 88% | 90% |

### Gap Analysis from Sprint 1

**Hinton Gaps Identified**:
| Component | Current | Target | Fix Required |
|-----------|---------|--------|--------------|
| FF Adaptive θ | 0.85 | 0.95 | Add adaptive threshold |
| Capsule EM routing | 0.82 | 0.92 | Implement EM |
| Credit TD(λ) | 0.78 | 0.95 | Fix λ decay |
| SWR→CA3 trigger | 0.88 | 0.95 | Add trigger mechanism |
| Replay-clearance lock | 0.83 | 0.93 | Add coordination |

**CompBio Gaps Identified**:
| System | Current | Target | Fix Required |
|--------|---------|--------|--------------|
| VTA D1/D2 Kd | 88 | 95 | Add receptor affinities |
| Striatum d1_d2_ratio | 85 | 93 | Add ratio parameter |
| Glymphatic clearance | 89→95 | 98 | ✓ FIXED (0.7) |
| SWR reverse_prob | 93 | 97 | Add reverse_probability |

---

## Agent Prompt Templates

### Hinton Agent Prompt (Sprint 2)

```markdown
TASK: Sprint 2 Hinton Analysis (H7-H11)

BATCHES:
- H7: Three-Factor Learning (eligibility.py, credit_flow.py, three_factor.py)
- H8: Neuromodulator Orchestra (serotonin.py, norepinephrine.py, acetylcholine.py)
- H9: Homeostatic Plasticity (homeostatic.py, plasticity.py)
- H10: Spaced Repetition (fsrs.py, scorer.py)
- H11: Neuro-Symbolic (causal_discovery.py, neuro_symbolic.py)

ANALYSIS FOCUS:
1. TD(λ) implementation with λ decay
2. Three-factor Pre×Post×Modulator rule
3. 5-HT patience, NE gain, ACh gating
4. BCM sliding threshold
5. FSRS retrievability curves

DELIVERABLES:
1. Code analysis per batch with scores (0-1)
2. Gap identification with severity (P0/P1/P2)
3. Documentation updates (specific paths)
4. Mermaid diagram specs
5. Test recommendations

OUTPUT FORMAT: [Standard H-batch format]
```

### CompBio Agent Prompt (Sprint 2)

```markdown
TASK: Sprint 2 CompBio Validation (B13-B22)

BATCHES:
- B13: Neural Field (neural_field.py, coupling.py)
- B14: Connectome (connectome.py, delays.py)
- B15: Glial (astrocyte.py, glutamate_signaling.py)
- B16: STDP (eligibility.py, stdp.py)
- B17: Three-Factor (three_factor.py, stdp_integration.py)
- B18: ACh (acetylcholine.py)
- B19: Homeostatic (homeostatic.py, plasticity.py)
- B20-B22: Self-org, Inhibition, Reconsolidation

VALIDATION FOCUS:
1. Parameter ranges vs literature
2. Timing dynamics (ms precision)
3. Receptor affinities
4. Anatomical organization

DELIVERABLES:
1. Parameter audit table per batch
2. Literature verification
3. Score per system (0-100)
4. Fixes required with priority
5. Documentation updates

OUTPUT FORMAT: [Standard B-batch format]
```

---

## Approval Request

Ready to execute Sprint 2 with parallel agent deployment:

1. **Launch ww-hinton** for H7-H11 (Learning & Plasticity)
2. **Launch ww-compbio** for B13-B22 (Validation)
3. Apply fixes as identified
4. Generate diagrams in Sprint 4
5. Complete all documentation

**Estimated Duration**: 10 days
**Agent Hours**: ~120 hours total (60 per agent)

---

## Appendix: File Counts by Region

| Region | Python Files | Test Files | Doc Files |
|--------|--------------|------------|-----------|
| NCA | 29 | 24 | 8 |
| Learning | 23 | 18 | 4 |
| Memory | 12 | 10 | 3 |
| Consolidation | 5 | 4 | 2 |
| Visualization | 18 | 6 | 3 |
| API | 12 | 5 | 2 |
| Core | 14 | 12 | 2 |
| Embedding | 10 | 8 | 1 |
| Encoding | 4 | 4 | 1 |
| Other | 68 | 30 | 10 |
| **TOTAL** | **195** | **121** | **36** |

---

*Plan Version 3.0 - Ready for Sprint 2 Execution*
