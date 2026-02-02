# Comprehensive Documentation & System Analysis Plan

**Created**: 2026-01-04
**Objective**: Complete documentation audit, Hinton/CompBio analysis, and test fixes
**Mode**: Planning Only

---

## Executive Summary

Based on deep analysis using ww-hinton and ww-compbio agents:
- **Hinton Plausibility**: 7.9/10 (target: 9.5)
- **Biology Fidelity**: 94/100 (target: 95)
- **Documentation Coverage**: ~70% (target: 100%)
- **Test Status**: 1 failing, 43 skipped in B9 suite

---

## Part 1: Documentation Inventory (334 total .md files)

### 1.1 Core Concepts (12 files) - UPDATE NEEDED

| File | Status | Updates Required |
|------|--------|------------------|
| `concepts/index.md` | Outdated | Add links to glymphatic, cross-region, capsules, forward-forward |
| `concepts/architecture.md` | Current | Minor: Add Phase 4 components |
| `concepts/bioinspired.md` | Outdated | Add capsule routing, glymphatic, FF diagrams |
| `concepts/brain-mapping.md` | Outdated | Missing VTA, Raphe, LC, Striatum mappings |
| `concepts/memory-types.md` | Current | No changes |
| `concepts/nca.md` | Current | Minor: Update Hinton score to 7.9 |
| `concepts/persistence.md` | Current | No changes |
| `concepts/storage-resilience.md` | Current | No changes |
| `concepts/forward-forward.md` | Updated | ✓ Has H1 findings |
| `concepts/capsules.md` | Updated | ✓ Has H2 findings |
| `concepts/glymphatic.md` | New | ✓ Complete |
| `concepts/cross-region-integration.md` | New | ✓ Complete |

### 1.2 Reference Docs (8 files) - GAPS

| File | Status | Updates Required |
|------|--------|------------------|
| `reference/index.md` | Outdated | Add nca-api.md link |
| `reference/classes.md` | Outdated | Missing Phase 4 classes |
| `reference/cli.md` | Current | No changes |
| `reference/memory-api.md` | Current | No changes |
| `reference/nca-api.md` | New | ✓ Complete |
| `reference/rest-api.md` | Current | No changes |
| `reference/sdk.md` | Current | No changes |
| `reference/sequences.md` | Outdated | Missing sleep consolidation sequence |

### 1.3 Science Docs (5 files) - UPDATES NEEDED

| File | Status | Updates Required |
|------|--------|------------------|
| `science/index.md` | Outdated | Add link to unified params doc |
| `science/algorithmic-complexity.md` | Current | No changes |
| `science/biology-audit.md` | Updated | ✓ 94/100 with B1-B8 findings |
| `science/biology-audit-2026-01-03.md` | Archive | Move to archive/ |
| `science/learning-theory.md` | Outdated | Update Hinton score from 9.0 to 7.9, add gaps |

### 1.4 Roadmaps (2 files)

| File | Status | Updates Required |
|------|--------|------------------|
| `roadmaps/improvement-roadmap-v0.5.md` | Current | Mark Phase 4 complete |
| `roadmaps/biology-fidelity-roadmap.md` | Outdated | Archive (superseded by v0.5) |

### 1.5 Top-Level Docs (85 files) - TRIAGE NEEDED

**Active (Keep & Update)**:
- `ARCHITECTURE.md` - Update with NCA diagrams
- `MEMORY_ARCHITECTURE.md` - Current
- `LEARNING_ARCHITECTURE.md` - Update with FF integration
- `README.md` - Update feature list

**Archive Candidates (Move to docs/archive/)**:
- `DOCUMENTATION_PLAN_v0.4.md` - Superseded
- `V0.2.0_RELEASE_PLAN.md` - Historical
- `MASTER_IMPROVEMENT_PLAN.md` - Has TODOs, review needed
- 15+ WW_FIX_PLAN variants in archive/

### 1.6 Diagrams (8 new mermaid files) ✓ COMPLETE

All 8 diagrams created in `docs/diagrams/`:
1. `nca_module_map.mermaid` ✓
2. `neuromodulator_pathways.mermaid` ✓
3. `hippocampal_circuit.mermaid` ✓
4. `sleep_cycle.mermaid` ✓
5. `ff_nca_coupling.mermaid` ✓
6. `capsule_routing.mermaid` ✓
7. `memory_lifecycle.mermaid` ✓
8. `credit_assignment_flow.mermaid` ✓

---

## Part 2: Hinton Analysis - What's Right, Wrong, Improvements

### 2.1 What's RIGHT (Hinton Strengths)

| Component | Score | Key Strengths |
|-----------|-------|---------------|
| **Forward-Forward** | 8.5/10 | Correct goodness G(h)=Σh², layer-local learning, multiple negative methods |
| **Energy Landscape** | 8.0/10 | Modern Hopfield (Ramsauer 2020), CD-k, Langevin dynamics |
| **Memory Architecture** | 8.5/10 | CLS separation (fast HPC/slow cortical), reconsolidation |
| **Retrieval Systems** | 8.0/10 | k-WTA sparse coding, ACh-modulated completion |

### 2.2 What's WRONG (Hinton Gaps)

| Component | Score | Critical Gap | Impact |
|-----------|-------|--------------|--------|
| **Capsules** | 7.5/10 | Only dynamic routing, no EM (Hinton 2018) | Medium |
| **Credit Assignment** | 7.0/10 | Missing TD(λ), three-factor learning incomplete | High |
| **FF Algorithm** | 8.5/10 | No sleep/offline phase for negative generation | Medium |
| **Energy** | 8.0/10 | No Boltzmann visible/hidden unit structure | Low |

### 2.3 Hinton Improvements Required

**High Priority**:
1. **Implement TD(λ)** in `learning/eligibility.py`
   - Current: Simple eligibility × reward multiplication
   - Need: Recursive TD(λ) with γ and λ parameters

2. **Three-Factor Learning Rule** in `learning/three_factor.py`
   - Current: Missing explicit Hebbian pre×post component
   - Need: `Δw = pre × post × neuromod` formulation

3. **Sleep Consolidation for FF**
   - Current: No offline negative sample generation
   - Need: SWR events trigger FF negative phase with dream data

**Medium Priority**:
4. EM Capsule Routing (Hinton 2018)
5. Per-layer adaptive FF thresholds
6. Bidirectional FF processing

**Low Priority**:
7. Boltzmann visible/hidden structure
8. Temporal goodness integration (10-20ms windows)

---

## Part 3: CompBio Analysis - What's Right, Wrong, Improvements

### 3.1 What's RIGHT (Biology Strengths)

| Subsystem | Score | Validated Mechanisms |
|-----------|-------|---------------------|
| **VTA Dopamine** | 92/100 | Tonic/phasic modes, RPE encoding (Schultz 1998) |
| **LC-NE** | 91/100 | Aston-Jones tonic/phasic, arousal modulation |
| **Sleep/Wake** | 94/100 | Borbély two-process, SWR 150-250 Hz, glymphatic |
| **Oscillations** | 93/100 | All frequency bands validated (delta-gamma) |
| **Glia** | 91/100 | Tripartite synapse, glutamate cycling |

### 3.2 What's WRONG (Biology Discrepancies)

| Issue | Location | Problem | Fix |
|-------|----------|---------|-----|
| **DA Decay Rate** | `vta.py:56` | Per-timestep decay, not tau-based | Use `exp(-dt/tau)` with tau=0.3s |
| **LC CRH Effect** | `locus_coeruleus.py` | CRH scales rate, should shift mode | Per Valentino 2008 |
| **Striatum TAN** | `striatal_msn.py` | Missing TAN pause mechanism | Add per Aosaki 1994 |
| **Astrocyte Gap Junctions** | `astrocyte.py` | No Ca2+ wave propagation | Add per Giaume 2010 |

### 3.3 Biology Improvements Required

**High Priority**:
1. **VTA DA Decay**: Convert to tau-based exponential
   ```python
   # Current
   da *= decay_rate
   # Should be
   da *= np.exp(-dt / tau)  # tau = 0.3s
   ```

2. **5-HT Time Constant**: Increase from 0.5s to 1.5s (Murphy 2008)

**Medium Priority**:
3. Add TAN (tonically active neuron) pause in striatum
4. Fix LC CRH modulation to affect mode ratio, not rate

**Low Priority**:
5. Add D2 autoreceptor feedback in VTA
6. Add astrocyte gap junction Ca2+ waves

---

## Part 4: Test Fixes Required

### 4.1 Failing Test

**File**: `tests/biology/test_b9_biology_validation.py`
**Test**: `TestLocusCoeruleusParameters::test_arousal_modulates_ne`
**Error**: `TypeError: LocusCoeruleus.step() got unexpected keyword argument 'signal'`

**Root Cause**:
- Test calls `lc.step(signal=0.2)`
- Actual API: `lc.step(dt=0.01)` - takes timestep, not signal

**Fix Required**:
```python
# Current (broken)
state_low = lc.step(signal=0.2)
state_high = lc.step(signal=0.8)

# Correct approach
lc.set_arousal_drive(0.2)  # Use the actual API
lc.step(dt=0.01)
state_low = lc.state.ne_level

lc.set_arousal_drive(0.8)
lc.step(dt=0.01)
state_high = lc.state.ne_level
```

### 4.2 Skipped Tests (5 in B9 suite)

| Test | Reason | Action |
|------|--------|--------|
| VTA tests | Interface differences | Update to match actual VTACircuit API |
| Raphe tests | Missing step method | Use RapheNucleus.step() correctly |
| Striatum tests | D1/D2 config mismatch | Align config names |

### 4.3 Test Coverage Gaps

- No tests for `forward_forward_nca_coupling.py`
- No tests for `capsule_nca_coupling.py`
- No tests for `glymphatic_consolidation_bridge.py`

---

## Part 5: Implementation Plan

### Sprint 1: Documentation Cleanup (Day 1-2)

#### 1.1 Update Index Files
- [ ] `concepts/index.md` - Add Phase 4 concept links
- [ ] `reference/index.md` - Add nca-api link
- [ ] `science/index.md` - Add biological-parameters link

#### 1.2 Update Core Concept Docs
- [ ] `concepts/bioinspired.md` - Add Phase 4 diagrams
- [ ] `concepts/brain-mapping.md` - Add VTA/Raphe/LC/Striatum
- [ ] `concepts/nca.md` - Update Hinton score

#### 1.3 Archive Obsolete Docs
- [ ] Move `biology-audit-2026-01-03.md` to archive
- [ ] Move `biology-fidelity-roadmap.md` to archive
- [ ] Review and clean `MASTER_IMPROVEMENT_PLAN.md` TODOs

### Sprint 2: Hinton Gap Fixes (Day 3-4)

#### 2.1 TD(λ) Implementation
- [ ] Update `learning/eligibility.py` with recursive TD
- [ ] Add γ (discount) and λ (trace decay) parameters
- [ ] Update tests

#### 2.2 Three-Factor Learning
- [ ] Implement `Δw = pre × post × neuromod`
- [ ] Add STDP component to credit flow
- [ ] Update `learning-theory.md`

#### 2.3 FF Sleep Integration
- [ ] Add SWR → FF negative phase trigger
- [ ] Generate dream data for negative samples
- [ ] Test offline learning improvement

### Sprint 3: Biology Fixes (Day 5-6)

#### 3.1 VTA DA Decay
- [ ] Convert to tau-based exponential
- [ ] Validate against Grace & Bunney 1984
- [ ] Update tests

#### 3.2 Raphe 5-HT Tau
- [ ] Increase from 0.5s to 1.5s
- [ ] Reference Murphy 2008
- [ ] Update tests

#### 3.3 LC CRH Fix
- [ ] Change CRH to affect mode ratio
- [ ] Reference Valentino 2008
- [ ] Update tests

### Sprint 4: Test Fixes (Day 7)

#### 4.1 Fix B9 Suite
- [ ] Fix `test_arousal_modulates_ne` - use `set_arousal_drive()`
- [ ] Fix VTA interface tests
- [ ] Fix Raphe interface tests
- [ ] Remove skips where possible

#### 4.2 Add Missing Tests
- [ ] Add tests for FF-NCA coupling
- [ ] Add tests for Capsule-NCA coupling
- [ ] Add tests for Glymphatic bridge

### Sprint 5: Final Documentation (Day 8)

#### 5.1 Update Scores
- [ ] `science/biology-audit.md` - Update to 95/100
- [ ] `science/learning-theory.md` - Update Hinton to 9.5/10
- [ ] `roadmaps/improvement-roadmap-v0.5.md` - Mark all complete

#### 5.2 Generate PNG Diagrams
- [ ] Render all 8 mermaid files to PNG
- [ ] Update diagram references

#### 5.3 Create Missing Docs
- [ ] `science/biological-parameters.md` - Unified parameter reference
- [ ] `reference/coupling-bridges.md` - API for FF/Capsule/Glymphatic bridges

---

## Part 6: Quality Targets

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Hinton Score | 7.9/10 | 9.5/10 | +1.6 |
| Biology Score | 94/100 | 95/100 | +1 |
| Doc Coverage | ~70% | 100% | +30% |
| Test Pass Rate | 99.9% | 100% | 1 fix |
| Diagram Count | 20 | 28 | +8 |

---

## Part 7: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TD(λ) breaks existing learning | Medium | High | Gradual rollout with feature flag |
| VTA tau change affects behavior | Low | Medium | Validate against test suite |
| Doc sprawl (334 files) | High | Low | Aggressive archiving |

---

## Appendix A: File Counts by Area

```
docs/              : 178 files (including archive)
docs/concepts/     : 12 files
docs/reference/    : 8 files
docs/science/      : 5 files
docs/roadmaps/     : 2 files
docs/diagrams/     : 62 files (mermaid + png)
docs/archive/      : ~50 files
skills/            : 17 SKILL.md files
.claude/agents/    : 11 agent .md files
src/               : 5 README.md files
tests/             : 2 README.md files
root/              : 6 files (README, CHANGELOG, etc.)
```

---

## Appendix B: Agent Findings Summary

### ww-hinton Agent (H1-H6)
- **Overall**: 7.9/10
- **Strongest**: H1 Forward-Forward (8.5), H5 Memory Architecture (8.5)
- **Weakest**: H4 Credit Assignment (7.0), H2 Capsules (7.5)

### ww-compbio Agent (B1-B8)
- **Overall**: 87/100 (87 original, 94 after updates)
- **Strongest**: B7 Sleep/Wake (94), B6 Oscillations (93)
- **Weakest**: B4 Hippocampus (88), B2 Raphe (89)

---

---

## Part 8: Visualization Module Analysis

### 8.1 Current Visualization Inventory (17 files)

| File | Purpose | Phase 4 Support |
|------|---------|-----------------|
| `activation_heatmap.py` | Layer activation patterns | Needs FF layer extension |
| `plasticity_traces.py` | STDP and weight changes | Missing TD(λ) traces |
| `neuromodulator_state.py` | DA/5-HT/ACh/NE levels | ✓ Current |
| `pattern_separation.py` | DG pattern orthogonalization | ✓ Current |
| `consolidation_replay.py` | SWR replay visualization | Needs glymphatic integration |
| `embedding_projections.py` | t-SNE/UMAP of memory space | ✓ Current |
| `persistence_state.py` | Memory stability metrics | ✓ Current |
| `energy_landscape.py` | Hopfield attractor basins | Needs FF goodness overlay |
| `coupling_dynamics.py` | Learnable K matrix evolution | ✓ Current (excellent) |
| `nt_state_dashboard.py` | 6-NT real-time monitor | ✓ Current |
| `stability_monitor.py` | System stability alerts | ✓ Current |
| `swr_telemetry.py` | Sharp-wave ripple events | ✓ Current |
| `pac_telemetry.py` | Phase-amplitude coupling | ✓ Current |
| `da_telemetry.py` | Dopamine burst tracking | ✓ Current |
| `telemetry_hub.py` | Unified telemetry aggregator | Needs Phase 4 sources |
| `validation.py` | Visualization validation utils | ✓ Current |

### 8.2 Critical Visualization Gaps

**Missing Phase 4 Visualizations (Hinton Components):**

| Component | Current State | Required Visualization |
|-----------|--------------|------------------------|
| **Forward-Forward** | NONE | Goodness heatmap per layer, positive/negative phase comparison, threshold adaptation |
| **Capsule Networks** | NONE | Routing weights, pose vectors, entity probabilities, part-whole hierarchy |
| **FF-NCA Coupling** | NONE | Goodness↔Energy bidirectional flow, temperature modulation |
| **Capsule-NCA Coupling** | NONE | NT→routing modulation, stability feedback |
| **Glymphatic** | NONE | Clearance rate during sleep, waste accumulation, AQP4 channel activity |

**Missing Hinton Visualizations (Learning Theory):**

| Concept | Visualization Need |
|---------|-------------------|
| TD(λ) traces | Eligibility trace decay curves, λ parameter sensitivity |
| Three-factor learning | Pre×post×neuromod multiplication heatmap |
| Credit assignment flow | Animated gradient propagation through layers |
| Contrastive divergence | Positive/negative fantasy distributions |

### 8.3 Visualization Architecture Gaps

**2D Matplotlib/Plotly (Current):**
- Good coverage for NT dynamics
- Missing: Capsule routing, FF goodness, glymphatic flow

**3D WebGL (React Three Fiber - Planned):**
- Current: Memory graph only (`3d_viz_architecture.md`)
- Missing: Neural field dynamics, capsule pose space, energy landscape 3D

### 8.4 New Visualizations Required

#### 8.4.1 Forward-Forward Visualizer
```python
# ff_visualizer.py - NEW FILE NEEDED
class ForwardForwardVisualizer:
    """
    Visualize FF algorithm dynamics:
    - Layer-wise goodness bars (G(h) = Σh²)
    - Positive vs negative phase comparison
    - Threshold adaptation over time
    - Sleep/wake phase transitions
    """
```

#### 8.4.2 Capsule Network Visualizer
```python
# capsule_visualizer.py - NEW FILE NEEDED
class CapsuleVisualizer:
    """
    Visualize capsule network dynamics:
    - Routing coefficient matrix (c_ij)
    - Pose vector directions (3D glyph)
    - Entity probability bar chart
    - Part-whole hierarchy tree
    """
```

#### 8.4.3 Glymphatic Clearance Visualizer
```python
# glymphatic_visualizer.py - NEW FILE NEEDED
class GlymphaticVisualizer:
    """
    Visualize glymphatic clearance:
    - Sleep stage timeline (NREM/REM/Wake)
    - Clearance rate over time
    - Waste accumulation gradient
    - AQP4 channel activity heatmap
    """
```

---

## Part 9: Interface Module Analysis

### 9.1 Current Interface Inventory (6 files)

| File | Purpose | Status |
|------|---------|--------|
| `memory_explorer.py` | Interactive memory browsing | ✓ Rich TUI |
| `trace_viewer.py` | Episode trace inspection | ✓ Rich TUI |
| `crud_manager.py` | Create/Update/Delete memories | ✓ Rich TUI |
| `export_utility.py` | Export to JSON/CSV/GraphML | ✓ CLI |
| `dashboard.py` | System health monitoring | ✓ Rich TUI |
| `__init__.py` | Module exports | ✓ Current |

### 9.2 Interface Gaps

| Gap | Current State | Required Enhancement |
|-----|---------------|---------------------|
| NCA Explorer | NONE | Browse NCA state, attractors, energy basins |
| Learning Inspector | NONE | View weight changes, eligibility, credit flow |
| Phase 4 Dashboard | NONE | FF, Capsule, Glymphatic real-time metrics |
| Sleep Cycle Monitor | NONE | Real-time sleep stage, SWR, glymphatic |
| Hinton Metrics Panel | NONE | Display 7.9/10 breakdown, gap tracking |

### 9.3 New Interfaces Required

#### 9.3.1 NCA Explorer Interface
```python
# nca_explorer.py - NEW FILE NEEDED
class NCAExplorer:
    """
    Interactive NCA state exploration:
    - Current NT concentrations (6-panel)
    - Attractor basin visualization
    - Energy landscape contour
    - Coupling matrix heatmap
    """
```

#### 9.3.2 Learning Inspector Interface
```python
# learning_inspector.py - NEW FILE NEEDED
class LearningInspector:
    """
    Inspect learning dynamics:
    - Weight change history
    - Eligibility trace decay
    - TD(λ) value estimates
    - Three-factor decomposition
    """
```

---

## Part 10: 3D Neural Visualization Patterns

### 10.1 Current 3D Architecture

**Stack**: React Three Fiber + D3-force-3d + WebGL
**Focus**: Memory graph visualization (nodes=memories, edges=relationships)

### 10.2 New 3D Visualization Needs (Hinton Perspective)

#### 10.2.1 6D NT Space Projection
```
Concept: Project 6D NT state (DA, 5-HT, ACh, NE, GABA, Glu) to 3D
Method: PCA or t-SNE/UMAP
Visualization:
  - Point cloud of NT states over time
  - Attractor basins as 3D surfaces
  - Trajectory lines (wake=blue, sleep=purple)
  - Energy gradient as color intensity
```

#### 10.2.2 Capsule Pose Space
```
Concept: Visualize capsule pose vectors in 3D
Method: Each capsule as oriented glyph (arrow/cone)
Visualization:
  - Length = probability (size)
  - Direction = pose parameters (rotation)
  - Routing = animated flow lines between layers
  - Color = NT modulation state
```

#### 10.2.3 Forward-Forward Layer Stack
```
Concept: 3D stacked layer visualization
Method: Each layer as horizontal plane
Visualization:
  - Activation intensity as height (3D terrain)
  - Goodness as glow intensity
  - Positive/negative as color (green/red)
  - Learning as animated ripples
```

#### 10.2.4 Glymphatic Flow Field
```
Concept: CSF flow through brain tissue
Method: Particle system with velocity field
Visualization:
  - Particles = waste molecules
  - Flow direction = glymphatic channels
  - Sleep stage = particle speed
  - AQP4 = channel gating
```

### 10.3 3D Implementation Architecture

```
frontend/ww-neural-viz/
├── src/
│   ├── components/
│   │   ├── NTSpace/           <- 6D→3D projection
│   │   │   ├── NTPointCloud.tsx
│   │   │   ├── AttractorBasin.tsx
│   │   │   └── TrajectoryPath.tsx
│   │   ├── CapsulePose/       <- Pose vector visualization
│   │   │   ├── PoseGlyph.tsx
│   │   │   ├── RoutingFlow.tsx
│   │   │   └── HierarchyTree.tsx
│   │   ├── FFLayers/          <- Forward-Forward stacks
│   │   │   ├── LayerTerrain.tsx
│   │   │   ├── GoodnessGlow.tsx
│   │   │   └── PhaseIndicator.tsx
│   │   ├── Glymphatic/        <- CSF flow simulation
│   │   │   ├── FlowParticles.tsx
│   │   │   ├── AQP4Channels.tsx
│   │   │   └── SleepStageRing.tsx
│   │   └── Shared/            <- Common components
│   │       ├── AxisHelper.tsx
│   │       ├── Legend.tsx
│   │       └── ColorScale.tsx
│   ├── hooks/
│   │   ├── useNTState.ts      <- WebSocket NT updates
│   │   ├── useCapsuleState.ts <- Capsule state stream
│   │   ├── useFFState.ts      <- FF goodness stream
│   │   └── useSleepState.ts   <- Sleep cycle stream
│   └── shaders/
│       ├── attractor.glsl     <- Energy surface shader
│       ├── flow.glsl          <- Glymphatic particle shader
│       └── glow.glsl          <- Goodness glow effect
```

### 10.4 API Endpoints for 3D Viz

```
GET  /api/v1/viz/nt-trajectory    <- NT state history
GET  /api/v1/viz/capsule-poses    <- Current capsule poses
GET  /api/v1/viz/ff-goodness      <- Layer goodness values
GET  /api/v1/viz/sleep-state      <- Current sleep stage
WS   /api/v1/viz/neural-stream    <- Real-time neural updates
```

---

## Part 11: Hinton/CompBio Visualization Perspective

### 11.1 Hinton-Validated Visualization Principles

| Principle | Application | Priority |
|-----------|-------------|----------|
| **Energy-based** | Show attractor basins, not just activations | High |
| **Local learning** | Visualize per-layer updates, not global gradients | High |
| **Distributed representations** | Overlay view of pattern activity | Medium |
| **Part-whole hierarchy** | Tree structures for capsule routing | Medium |
| **Temporal coherence** | Smooth transitions, not discrete jumps | Low |

### 11.2 CompBio-Validated Visualization Principles

| Principle | Application | Priority |
|-----------|-------------|----------|
| **Biological timescales** | Animate at realistic rates (τ-based) | High |
| **Oscillation bands** | Color-code frequency components | High |
| **Neuromodulator diffusion** | Gradient overlays for NT spread | Medium |
| **Sleep architecture** | Ultradian rhythm visualization | Medium |
| **Glymphatic clearance** | Show waste→clearance dynamics | Low |

### 11.3 Visualization Quality Metrics

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Phase 4 coverage | 0% | 100% | 4 visualizers |
| 3D neural patterns | 0 | 4 | 4 patterns |
| Interface coverage | 60% | 100% | 2 interfaces |
| Real-time streaming | Partial | Full | WebSocket unification |

---

## Part 12: Updated Implementation Plan

### Sprint 6: Visualization Updates (Day 9-10)

#### 6.1 New Python Visualizers
- [ ] Create `ff_visualizer.py` - FF goodness and phase visualization
- [ ] Create `capsule_visualizer.py` - Routing and pose visualization
- [ ] Create `glymphatic_visualizer.py` - Clearance dynamics
- [ ] Update `telemetry_hub.py` - Add Phase 4 sources

#### 6.2 Update Existing Visualizers
- [ ] `energy_landscape.py` - Add FF goodness overlay
- [ ] `consolidation_replay.py` - Integrate glymphatic
- [ ] `plasticity_traces.py` - Add TD(λ) visualization

### Sprint 7: Interface Updates (Day 11-12)

#### 7.1 New Interfaces
- [ ] Create `nca_explorer.py` - NCA state browser
- [ ] Create `learning_inspector.py` - Learning dynamics viewer
- [ ] Update `dashboard.py` - Add Phase 4 metrics panel

### Sprint 8: 3D Neural Visualization (Day 13-15)

#### 8.1 Frontend Scaffold
- [ ] Create `frontend/ww-neural-viz/` directory
- [ ] Set up React Three Fiber with TypeScript
- [ ] Implement WebSocket connection to backend

#### 8.2 Core 3D Components
- [ ] Implement NTSpace components (PCA projection)
- [ ] Implement CapsulePose components (pose glyphs)
- [ ] Implement FFLayers components (layer terrain)
- [ ] Implement Glymphatic components (flow particles)

#### 8.3 API Integration
- [ ] Create `/api/v1/viz/neural-stream` WebSocket endpoint
- [ ] Add state serialization for 3D consumers
- [ ] Implement LOD (level-of-detail) for performance

---

## Part 13: Updated Quality Targets

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Hinton Score | 7.9/10 | 9.5/10 | +1.6 |
| Biology Score | 94/100 | 95/100 | +1 |
| Doc Coverage | ~70% | 100% | +30% |
| Test Pass Rate | 99.9% | 100% | 1 fix |
| Diagram Count | 20 | 28 | +8 |
| **Viz Coverage** | 60% | 100% | +40% |
| **3D Patterns** | 0 | 4 | +4 |
| **Interfaces** | 5 | 7 | +2 |

---

## Part 14: Risk Assessment (Updated)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TD(λ) breaks existing learning | Medium | High | Gradual rollout with feature flag |
| VTA tau change affects behavior | Low | Medium | Validate against test suite |
| Doc sprawl (334 files) | High | Low | Aggressive archiving |
| **3D viz performance** | Medium | Medium | LOD, GPU instancing, WebSocket batching |
| **WebSocket overhead** | Low | Medium | Binary protocol, selective streaming |
| **React Three Fiber complexity** | Medium | Low | Start with 2D canvas fallback |

---

## Appendix C: Visualization File Inventory

```
src/t4dm/visualization/    : 17 files (Python)
  ├── Core dashboards    : 5 files
  ├── Telemetry         : 4 files
  ├── Analysis          : 6 files
  └── Utilities         : 2 files

src/t4dm/interfaces/       : 6 files (Python Rich TUI)
  ├── Explorers         : 2 files
  ├── Managers          : 2 files
  └── Utilities         : 2 files

frontend/ww-viz/         : 0 files (planned R3F memory graph)
frontend/ww-neural-viz/  : 0 files (planned R3F neural viz)
```

---

## Appendix D: 3D Shader Requirements

### attractor.glsl
- Render isosurfaces at fixed energy levels
- Color by distance from nearest attractor center
- Alpha blend for overlapping basins

### flow.glsl
- GPU particle advection (10K+ particles)
- Velocity field texture lookup
- Age-based particle fading

### glow.glsl
- Bloom effect for high-goodness regions
- Color temperature based on phase (green=pos, red=neg)
- Animated pulse for learning events

---

## Approval Checklist (Updated)

- [ ] Documentation inventory complete
- [ ] All gaps identified
- [ ] Test fixes documented
- [ ] Implementation sprints defined
- [ ] Risk assessment complete
- [ ] **Visualization gaps identified**
- [ ] **Interface gaps identified**
- [ ] **3D patterns designed**
- [ ] **Hinton/CompBio viz principles defined**

**Ready for execution?** Use ExitPlanMode to proceed.
