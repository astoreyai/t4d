# World Weaver Documentation Analysis Plan

**Created**: 2026-01-04
**Objective**: Comprehensive analysis and documentation of all WW regions using specialized Opus agents
**Agents**: ww-hinton (neural architecture), ww-compbio (biological validation)

---

## Executive Summary

This plan orchestrates a systematic documentation overhaul using two specialized agents:
1. **ww-hinton**: Geoffrey Hinton-inspired analysis of neural architectures, learning algorithms, and representation theory
2. **ww-compbio**: Computational biology validation of biological parameters, pathways, and neuroscience accuracy

The output will be updated documentation, new diagrams, and a unified knowledge base.

---

## Phase 1: Module Inventory & Classification

### 1.1 NCA Module Regions (30 files)

| Region | Files | Primary Function | Agent Assignment |
|--------|-------|------------------|------------------|
| **Neuromodulators** | vta.py, raphe.py, locus_coeruleus.py | DA/5-HT/NE dynamics | compbio |
| **Striatal System** | striatal_msn.py, striatal_coupling.py | GO/NO-GO pathways | compbio |
| **Hippocampal** | hippocampus.py, spatial_cells.py | DG/CA3/CA1 + place/grid cells | compbio |
| **Oscillators** | oscillators.py, sleep_spindles.py, theta_gamma_integration.py | Neural rhythms | compbio |
| **Sleep/Wake** | adenosine.py, glymphatic.py, swr_coupling.py | Sleep pressure + SWR | compbio |
| **Glia** | astrocyte.py, glutamate_signaling.py | Tripartite synapse | compbio |
| **Neural Field** | neural_field.py, coupling.py, energy.py, attractors.py | PDE dynamics | hinton |
| **Hinton Architectures** | forward_forward.py, capsules.py, pose.py | FF + Capsule nets | hinton |
| **Integration** | *_nca_coupling.py, glymphatic_consolidation_bridge.py | Cross-region bridges | hinton + compbio |
| **Infrastructure** | connectome.py, delays.py, stability.py | System plumbing | hinton |

### 1.2 Learning Module Regions (24 files)

| Region | Files | Primary Function | Agent Assignment |
|--------|-------|------------------|------------------|
| **Neuromodulators** | dopamine.py, serotonin.py, norepinephrine.py, acetylcholine.py | NT-based learning | compbio |
| **Plasticity** | stdp.py, plasticity.py, three_factor.py | Synaptic plasticity | compbio |
| **Credit Assignment** | eligibility.py, credit_flow.py, causal_discovery.py | TD/eligibility traces | hinton |
| **Memory Scoring** | scorer.py, self_supervised.py, fsrs.py | Retrieval ranking | hinton |
| **Homeostasis** | homeostatic.py, inhibition.py | Stability mechanisms | compbio |
| **Infrastructure** | cold_start.py, persistence.py, collector.py, hooks.py | System plumbing | hinton |

### 1.3 Memory Module Regions (13 files)

| Region | Files | Primary Function | Agent Assignment |
|--------|-------|------------------|------------------|
| **Tripartite Memory** | episodic.py, semantic.py, procedural.py | Core memory types | hinton |
| **Fast Systems** | fast_episodic.py, working_memory.py, buffer_manager.py | Short-term storage | hinton |
| **Indexing** | cluster_index.py, learned_sparse_index.py, pattern_separation.py | Retrieval optimization | hinton |
| **Maintenance** | forgetting.py, feature_aligner.py, unified.py | Memory hygiene | hinton + compbio |

---

## Phase 2: Agent Task Distribution

### 2.1 Hinton Agent Tasks (ww-hinton)

**Focus**: Neural architecture review, learning algorithms, representation theory

#### Task H1: Forward-Forward Analysis
```yaml
files:
  - src/t4dm/nca/forward_forward.py
  - src/t4dm/nca/forward_forward_nca_coupling.py
output:
  - docs/concepts/forward-forward.md (update)
  - Goodness function diagram
  - Energy landscape visualization
questions:
  - Is the goodness threshold adaptive?
  - How does FF integrate with backprop-trained components?
  - Is the negative sample generation biologically plausible?
```

#### Task H2: Capsule Network Analysis
```yaml
files:
  - src/t4dm/nca/capsules.py
  - src/t4dm/nca/pose.py
  - src/t4dm/nca/capsule_nca_coupling.py
output:
  - docs/concepts/capsules.md (update)
  - Routing by agreement diagram
  - Pose transformation visualization
questions:
  - Is dynamic routing differentiable end-to-end?
  - How do capsules represent part-whole hierarchies?
  - What's the relationship to cortical columns?
```

#### Task H3: Energy Landscape & Attractors
```yaml
files:
  - src/t4dm/nca/energy.py
  - src/t4dm/nca/attractors.py
  - src/t4dm/nca/neural_field.py
output:
  - docs/concepts/nca.md (update energy section)
  - Attractor basin visualization
  - State transition diagram
questions:
  - Are attractors learned or pre-defined?
  - What determines attractor stability?
  - How does noise injection affect convergence?
```

#### Task H4: Credit Assignment & Eligibility
```yaml
files:
  - src/t4dm/learning/eligibility.py
  - src/t4dm/learning/credit_flow.py
  - src/t4dm/learning/causal_discovery.py
output:
  - docs/science/learning-theory.md (update)
  - Credit flow diagram
  - TD-λ visualization
questions:
  - Is credit assignment graph-aware?
  - How does causal discovery integrate with replay?
  - What's the maximum credit horizon?
```

#### Task H5: Memory Architecture Review
```yaml
files:
  - src/t4dm/memory/episodic.py
  - src/t4dm/memory/fast_episodic.py
  - src/t4dm/memory/buffer_manager.py
  - src/t4dm/memory/unified.py
output:
  - docs/concepts/memory-types.md (update)
  - Memory flow diagram
  - Buffer → LTM promotion visualization
questions:
  - Is the learned gate truly end-to-end?
  - How does buffer evidence accumulate?
  - What prevents catastrophic forgetting?
```

#### Task H6: Retrieval & Indexing
```yaml
files:
  - src/t4dm/memory/cluster_index.py
  - src/t4dm/memory/learned_sparse_index.py
  - src/t4dm/memory/pattern_separation.py
output:
  - docs/reference/retrieval.md (new)
  - Hierarchical retrieval diagram
  - Sparse addressing visualization
questions:
  - Is retrieval learned or heuristic?
  - How does sparsity affect recall quality?
  - What's the O(n) complexity for k-NN?
```

### 2.2 CompBio Agent Tasks (ww-compbio)

**Focus**: Biological plausibility, neuroscience accuracy, parameter validation

#### Task B1: VTA Dopamine Circuit
```yaml
files:
  - src/t4dm/nca/vta.py
  - src/t4dm/learning/dopamine.py
  - src/t4dm/nca/dopamine_integration.py
output:
  - docs/science/biology-audit.md (update VTA section)
  - VTA circuit diagram (update)
  - RPE computation validation
validation:
  - Tonic rate: 1-8 Hz (Schultz 1998)
  - Burst peak: 15-30 Hz (Grace & Bunney 1984)
  - TD-γ: 0.9-0.99
```

#### Task B2: Raphe Serotonin System
```yaml
files:
  - src/t4dm/nca/raphe.py
  - src/t4dm/learning/serotonin.py
output:
  - docs/science/biology-audit.md (update Raphe section)
  - Patience model diagram
  - Temporal discounting validation
validation:
  - Baseline rate: 1-3 Hz (Jacobs & Azmitia 1992)
  - Autoreceptor Hill coefficient: 1.5-2.5
  - Reuptake tau: 0.3-1.0s
```

#### Task B3: Locus Coeruleus NE System
```yaml
files:
  - src/t4dm/nca/locus_coeruleus.py
  - src/t4dm/learning/norepinephrine.py
output:
  - docs/science/biology-audit.md (update LC section)
  - Surprise model diagram
  - Arousal-gain curve validation
validation:
  - Tonic optimal: 2-5 Hz (Aston-Jones 2005)
  - Phasic peak: 10-20 Hz
  - Yerkes-Dodson midpoint: ~0.6
```

#### Task B4: Hippocampal System
```yaml
files:
  - src/t4dm/nca/hippocampus.py
  - src/t4dm/nca/spatial_cells.py
  - src/t4dm/memory/pattern_separation.py
output:
  - docs/science/biology-audit.md (update Hippocampus section)
  - DG/CA3/CA1 circuit diagram
  - Pattern separation/completion validation
validation:
  - DG sparsity: 0.5-5% (Jung & McNaughton 1993)
  - CA3 beta: 5-20 (Hopfield temperature)
  - Gridness score: >0.3 (Sargolini 2006)
```

#### Task B5: Striatal GO/NO-GO
```yaml
files:
  - src/t4dm/nca/striatal_msn.py
  - src/t4dm/nca/striatal_coupling.py
output:
  - docs/science/biology-audit.md (update Striatal section)
  - D1/D2 pathway diagram
  - Action selection validation
validation:
  - D2 affinity > D1 affinity
  - Lateral inhibition: 0.2-0.5
  - MSN tau: 20-100ms
```

#### Task B6: Neural Oscillations
```yaml
files:
  - src/t4dm/nca/oscillators.py
  - src/t4dm/nca/sleep_spindles.py
  - src/t4dm/nca/theta_gamma_integration.py
output:
  - docs/science/biology-audit.md (update Oscillations section)
  - Frequency band diagram
  - PAC validation
validation:
  - Theta: 4-8 Hz
  - Gamma: 30-100 Hz
  - Spindles: 11-16 Hz
  - Ripples: 150-250 Hz
```

#### Task B7: Sleep/Wake System
```yaml
files:
  - src/t4dm/nca/adenosine.py
  - src/t4dm/nca/glymphatic.py
  - src/t4dm/nca/swr_coupling.py
  - src/t4dm/nca/glymphatic_consolidation_bridge.py
output:
  - docs/science/biology-audit.md (update Sleep section)
  - Two-process model diagram
  - SWR gating validation
validation:
  - Adenosine accumulation: ~16h to saturation
  - Sleep onset threshold: 0.5-0.9
  - SWR compression: 5-20x
```

#### Task B8: Glial Systems
```yaml
files:
  - src/t4dm/nca/astrocyte.py
  - src/t4dm/nca/glutamate_signaling.py
output:
  - docs/science/biology-audit.md (update Glia section)
  - Tripartite synapse diagram
  - Glutamate clearance validation
validation:
  - EAAT2 Km: 10-50 µM
  - Ca2+ wave timescale: seconds
  - NR2B > NR2A affinity
```

---

## Phase 3: Documentation Deliverables

### 3.1 Documents to Update

| Document | Current State | Updates Needed | Agent |
|----------|---------------|----------------|-------|
| `docs/concepts/nca.md` | Good, dated | Add Phase 4 content (capsules, FF-NCA) | hinton |
| `docs/concepts/forward-forward.md` | Good | Add FF-NCA coupling section | hinton |
| `docs/concepts/capsules.md` | New | Expand with biological analogs | hinton + compbio |
| `docs/concepts/bioinspired.md` | Good | Add glymphatic, spindles | compbio |
| `docs/science/biology-audit.md` | Outdated (82/100) | Update to 95/100 | compbio |
| `docs/science/learning-theory.md` | Good | Update Hinton score to 9.5 | hinton |
| `docs/DIAGRAMS.md` | Dated | Add new architecture diagrams | hinton |
| `docs/BIOINSPIRED_DIAGRAMS.md` | Dated | Add biological pathway diagrams | compbio |

### 3.2 New Documents to Create

| Document | Purpose | Agent |
|----------|---------|-------|
| `docs/concepts/glymphatic.md` | Sleep-gated waste clearance | compbio |
| `docs/concepts/cross-region-integration.md` | H10 coupling mechanisms | hinton |
| `docs/reference/nca-api.md` | NCA module API reference | hinton |
| `docs/science/biological-parameters.md` | Unified parameter reference | compbio |
| `docs/diagrams/neural_pathways_v2.md` | Updated pathway diagrams | compbio |

### 3.3 Diagrams to Generate

| Diagram | Type | Description | Agent |
|---------|------|-------------|-------|
| `nca_module_map.mermaid` | Architecture | Full NCA module dependency graph | hinton |
| `neuromodulator_pathways.mermaid` | Biology | DA/5-HT/NE/ACh interaction diagram | compbio |
| `hippocampal_circuit.mermaid` | Biology | DG→CA3→CA1 with NT modulation | compbio |
| `sleep_cycle.mermaid` | Biology | Adenosine→Sleep stages→SWR→Replay | compbio |
| `ff_nca_coupling.mermaid` | Architecture | Goodness↔Energy landscape | hinton |
| `capsule_routing.mermaid` | Architecture | Dynamic routing with NT modulation | hinton |
| `memory_lifecycle.mermaid` | Architecture | Buffer→Evidence→Promote/Discard | hinton |
| `credit_assignment_flow.mermaid` | Architecture | TD-λ with eligibility traces | hinton |

---

## Phase 4: Execution Schedule

### Sprint A: Architecture Analysis (Hinton Agent)

```
Day 1: H1 + H2 (Forward-Forward + Capsules)
Day 2: H3 + H4 (Energy/Attractors + Credit Assignment)
Day 3: H5 + H6 (Memory Architecture + Retrieval)
Day 4: Documentation synthesis + diagram generation
```

### Sprint B: Biology Validation (CompBio Agent)

```
Day 1: B1 + B2 (VTA Dopamine + Raphe Serotonin)
Day 2: B3 + B4 (Locus Coeruleus + Hippocampus)
Day 3: B5 + B6 (Striatum + Oscillations)
Day 4: B7 + B8 (Sleep/Wake + Glia)
Day 5: Parameter reconciliation + audit update
```

### Sprint C: Integration & Finalization

```
Day 1: Cross-validate Hinton + CompBio findings
Day 2: Update all documentation
Day 3: Generate final diagrams
Day 4: Create unified knowledge base entry
```

---

## Phase 5: Quality Gates

### 5.1 Documentation Completeness Checklist

- [ ] Every NCA module has corresponding docs section
- [ ] Every learning algorithm has theory explanation
- [ ] Every biological parameter has literature citation
- [ ] Every diagram has text explanation
- [ ] All cross-references are valid

### 5.2 Biology Validation Score Target

| Category | Current | Target | Gap |
|----------|---------|--------|-----|
| NCA Dynamics | 88 | 95 | +7 |
| Learning Systems | 80 | 92 | +12 |
| Memory Consolidation | 78 | 93 | +15 |
| Prediction Systems | 85 | 95 | +10 |
| **Overall** | **82** | **95** | **+13** |

### 5.3 Hinton Plausibility Score Target

| Criterion | Current | Target |
|-----------|---------|--------|
| Representation Learning | 9.0 | 9.5 |
| Local Learning (FF) | 9.0 | 9.5 |
| Part-Whole (Capsules) | 9.0 | 9.5 |
| Credit Assignment | 8.5 | 9.5 |
| **Overall** | **9.0** | **9.5** |

---

## Agent Prompts

### Hinton Agent Prompt Template

```
You are analyzing World Weaver's neural architecture from Geoffrey Hinton's perspective.

Focus areas:
1. Representation learning - Are representations learned, not hand-crafted?
2. Local learning rules - Can learning happen without global backprop?
3. Biological plausibility - Does it map to known neural mechanisms?
4. Efficiency - Is compute/memory usage reasonable?

For each module, provide:
- Architecture summary (1 paragraph)
- Hinton-score evaluation (1-10 with justification)
- Key strengths and gaps
- Suggested improvements
- Diagram specification (mermaid format)
```

### CompBio Agent Prompt Template

```
You are validating World Weaver's biological accuracy from a computational neuroscience perspective.

Focus areas:
1. Parameter ranges - Are values within literature-reported ranges?
2. Circuit topology - Do connections match known anatomy?
3. Temporal dynamics - Are timescales biologically realistic?
4. Functional behavior - Does the system produce expected outputs?

For each module, provide:
- Biological mapping (which brain region/circuit)
- Parameter validation table (value, range, source)
- Accuracy score (1-100 with breakdown)
- Literature references (at least 3 per module)
- Diagram specification (mermaid format)
```

---

## Appendix: File Inventory

### NCA Module Files (30)

```
src/t4dm/nca/
├── __init__.py
├── adenosine.py              # Sleep pressure (compbio)
├── astrocyte.py              # Tripartite synapse (compbio)
├── attractors.py             # Energy basins (hinton)
├── capsule_nca_coupling.py   # Capsule↔NCA bridge (hinton)
├── capsules.py               # Part-whole (hinton)
├── connectome.py             # Structural connectivity (hinton)
├── coupling.py               # NT coupling matrix (hinton)
├── delays.py                 # Axonal delays (compbio)
├── dopamine_integration.py   # DA integration (compbio)
├── energy.py                 # Energy landscape (hinton)
├── forward_forward.py        # FF algorithm (hinton)
├── forward_forward_nca_coupling.py  # FF↔NCA bridge (hinton)
├── glymphatic.py             # Waste clearance (compbio)
├── glymphatic_consolidation_bridge.py  # Sleep↔consolidation (compbio)
├── glutamate_signaling.py    # NMDA/AMPA (compbio)
├── hippocampus.py            # DG/CA3/CA1 (compbio)
├── locus_coeruleus.py        # NE system (compbio)
├── neural_field.py           # PDE dynamics (hinton)
├── oscillators.py            # Frequency bands (compbio)
├── pose.py                   # Pose estimation (hinton)
├── raphe.py                  # 5-HT system (compbio)
├── sleep_spindles.py         # Thalamocortical (compbio)
├── spatial_cells.py          # Place/grid cells (compbio)
├── stability.py              # System stability (hinton)
├── striatal_coupling.py      # Striatum↔NCA (compbio)
├── striatal_msn.py           # D1/D2 MSNs (compbio)
├── swr_coupling.py           # Sharp-wave ripples (compbio)
├── theta_gamma_integration.py # PAC (compbio)
└── vta.py                    # VTA dopamine (compbio)
```

### Learning Module Files (24)

```
src/t4dm/learning/
├── __init__.py
├── acetylcholine.py          # ACh modulation (compbio)
├── causal_discovery.py       # Causal inference (hinton)
├── cold_start.py             # Bootstrap (hinton)
├── collector.py              # Experience collection (hinton)
├── credit_flow.py            # Credit assignment (hinton)
├── dopamine.py               # DA learning (compbio)
├── eligibility.py            # Eligibility traces (hinton)
├── events.py                 # Learning events (hinton)
├── fsrs.py                   # Spaced repetition (hinton)
├── homeostatic.py            # Homeostasis (compbio)
├── hooks.py                  # Learning hooks (hinton)
├── inhibition.py             # GABA inhibition (compbio)
├── neuro_symbolic.py         # Hybrid reasoning (hinton)
├── neuromodulators.py        # Orchestra (compbio)
├── norepinephrine.py         # NE learning (compbio)
├── persistence.py            # Weight persistence (hinton)
├── plasticity.py             # Synaptic tags (compbio)
├── reconsolidation.py        # Memory update (compbio)
├── scorer.py                 # Retrieval scoring (hinton)
├── self_supervised.py        # SSL credit (hinton)
├── serotonin.py              # 5-HT learning (compbio)
├── stdp.py                   # STDP rules (compbio)
└── three_factor.py           # 3-factor rule (compbio)
```

### Memory Module Files (13)

```
src/t4dm/memory/
├── __init__.py
├── buffer_manager.py         # CA1-like buffer (hinton)
├── cluster_index.py          # Hierarchical index (hinton)
├── episodic.py               # Episodic memory (hinton)
├── fast_episodic.py          # Fast encoding (hinton)
├── feature_aligner.py        # Gate↔retrieval (hinton)
├── forgetting.py             # Active forgetting (compbio)
├── learned_sparse_index.py   # Sparse addressing (hinton)
├── pattern_separation.py     # DG-like separation (compbio)
├── procedural.py             # Skill memory (hinton)
├── semantic.py               # Entity memory (hinton)
├── unified.py                # Unified interface (hinton)
└── working_memory.py         # WM slots (hinton)
```
