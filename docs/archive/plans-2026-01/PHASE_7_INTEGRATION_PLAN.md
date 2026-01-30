# World Weaver Phase 7: Complete System Integration

**Generated**: 2026-01-04 | **Status**: PLANNING | **Agents**: ww-hinton, ww-compbio, Explore

---

## Executive Summary

Three Opus agents analyzed the codebase and identified a critical pattern: **"Beautiful infrastructure without learning."** The system has sophisticated components that exist in isolation. The primary need is **wiring**, not new features.

### Key Metrics
- **6983 tests passing** (78% coverage)
- **30+ NCA modules** implemented
- **3 bridges** built but never instantiated
- **Temporal module** completely disconnected
- **Dreaming** not wired to sleep consolidation
- **Biological accuracy**: 85/100

---

## Part 1: Critical Integration Gaps

### 1.1 Bridges Never Instantiated

| Bridge | File | Status | Should Connect To |
|--------|------|--------|-------------------|
| `PredictiveCodingDopamineBridge` | `bridges/dopamine_bridge.py` | TESTED, NOT USED | `consolidation/service.py` |
| `CapsuleRetrievalBridge` | `bridges/capsule_bridge.py` | TESTED, NOT USED | `memory/semantic.py` |
| `FFEncodingBridge` | `bridges/ff_encoding_bridge.py` | TESTED, NOT USED | `memory/episodic.py` |
| `MemoryNCABridge` | `bridge/memory_nca.py` | TESTED, NOT USED | `memory/unified.py` |

### 1.2 Temporal Module Completely Orphaned

**Files** (all unused in production):
- `temporal/dynamics.py` - TemporalDynamics class
- `temporal/integration.py` - PlasticityCoordinator
- `temporal/session.py` - SessionManager

**Zero imports** from any source file outside the module.

### 1.3 Dreaming Not Connected to Sleep

- `dreaming/consolidation.py` → `DreamConsolidation` class exists
- `consolidation/sleep.py` → REM phase has NO call to dreaming
- Missing: `dream_consolidator.process()` during REM

### 1.4 Energy-Based Learning Not Learning

- `nca/coupling.py` → `update_from_energy()` method exists
- `learning/three_factor.py` → Never calls coupling update
- The coupling matrix K is static, not adaptive

---

## Part 2: Biological Accuracy Corrections

### 2.1 Parameter Adjustments Needed

| File | Parameter | Current | Target | Citation |
|------|-----------|---------|--------|----------|
| `nca/hippocampus.py` | `dg_expansion` | 4x | 5x | Amaral 1990 |
| `nca/hippocampus.py` | `dg_sparsity` | 1% | 2% | Chawla 2005 |
| `nca/vta.py` | `da_decay_rate` | 0.1 | 0.15 | Cragg & Rice 2004 |
| `nca/raphe.py` | `desensitization_rate` | 0.01 | 0.005 | Blier 1987 |

### 2.2 Missing Literature Citations

| Module | Missing Citation | For Parameter |
|--------|-----------------|---------------|
| vta.py | Grace & Bunney 1984 | tonic_rate, burst_peak_rate |
| vta.py | Bayer & Glimcher 2005 | rpe_to_da_gain |
| hippocampus.py | Amaral et al. 1990 | DG:CA3 ratio |
| sleep.py | Tononi & Cirelli 2006 | prune_threshold |
| oscillators.py | Hasselmo 2006 | theta_ach_sensitivity |

---

## Part 3: Implementation Sprints

### Sprint 7.1: Bridge Wiring (Priority: CRITICAL)

**Goal**: Wire all bridges into production code paths

#### Task 7.1.1: Wire FFEncodingBridge to Episodic Memory
```
Source: memory/episodic.py :: encode()
Target: bridges/ff_encoding_bridge.py :: process()
Action:
  1. Create bridge instance in memory service init
  2. Call bridge.process(embedding) before encoding
  3. Use guidance.encoding_multiplier to modulate storage
```

#### Task 7.1.2: Wire CapsuleRetrievalBridge to Semantic Memory
```
Source: memory/semantic.py :: recall()
Target: bridges/capsule_bridge.py :: compute_boosts()
Action:
  1. Create bridge instance in semantic memory init
  2. Call bridge.compute_boosts(query, candidates)
  3. Add boosts to retrieval scores
```

#### Task 7.1.3: Wire DopamineBridge to Consolidation
```
Source: consolidation/service.py :: consolidate()
Target: bridges/dopamine_bridge.py :: process()
Action:
  1. Create bridge in consolidation service
  2. Call bridge.compute_rpe(prediction, outcome)
  3. Use RPE to weight memory priority
```

#### Task 7.1.4: Wire MemoryNCABridge to Unified Memory
```
Source: memory/unified.py :: search(), encode()
Target: bridge/memory_nca.py :: augment_encoding(), modulate_retrieval()
Action:
  1. Initialize bridge with NCA instance
  2. Call augment_encoding() before storage
  3. Call modulate_retrieval() during search
```

### Sprint 7.2: Temporal Integration (Priority: HIGH)

**Goal**: Connect temporal module to memory lifecycle

#### Task 7.2.1: Wire SessionManager to Memory Operations
```
Source: memory/episodic.py, memory/unified.py
Target: temporal/session.py :: SessionManager
Action:
  1. Create session on first memory operation
  2. Track operations within session
  3. Close session on timeout/explicit end
```

#### Task 7.2.2: Wire PlasticityCoordinator to Learning
```
Source: learning/three_factor.py :: compute()
Target: temporal/integration.py :: PlasticityCoordinator
Action:
  1. Get current plasticity state from coordinator
  2. Modulate learning rate based on temporal phase
  3. Update coordinator state after learning
```

#### Task 7.2.3: Wire TemporalDynamics to Consolidation
```
Source: consolidation/sleep.py :: run_cycle()
Target: temporal/dynamics.py :: TemporalDynamics
Action:
  1. Track temporal context during replay
  2. Weight memories by temporal distance
  3. Update dynamics after consolidation
```

### Sprint 7.3: Dreaming-Sleep Integration (Priority: HIGH)

**Goal**: Wire dreaming to REM phase of sleep consolidation

#### Task 7.3.1: Create DreamConsolidation in Sleep Service
```
Source: consolidation/sleep.py
Target: dreaming/consolidation.py :: DreamConsolidation
Action:
  1. Initialize DreamConsolidation with prediction components
  2. Call during REM phase
  3. Use dream outputs to boost memory priorities
```

#### Task 7.3.2: Wire Dream Quality to Memory Selection
```
Source: dreaming/quality.py :: DreamQualityEvaluator
Target: consolidation/sleep.py :: select_for_replay()
Action:
  1. Evaluate dream quality after generation
  2. High-quality dreams → higher replay priority
  3. Track dream quality metrics
```

### Sprint 7.4: Energy-Based Learning (Priority: MEDIUM)

**Goal**: Make coupling matrix adaptive

#### Task 7.4.1: Wire Coupling to Three-Factor Learning
```
Source: learning/three_factor.py :: compute()
Target: nca/coupling.py :: update_from_energy()
Action:
  1. After computing learning signal, call coupling.update_from_energy()
  2. Pass current NT state as data_state
  3. Use eligibility traces to modulate update
```

#### Task 7.4.2: Add RPE-Based Coupling Updates
```
Source: nca/vta.py :: compute_rpe()
Target: nca/coupling.py :: update_from_energy()
Action:
  1. Route VTA RPE to coupling update
  2. Positive RPE → strengthen current state couplings
  3. Negative RPE → weaken current state couplings
```

### Sprint 7.5: Biological Parameter Corrections (Priority: MEDIUM)

#### Task 7.5.1: Hippocampus Parameter Updates
- Update `dg_expansion_factor` from 4 to 5
- Update `dg_sparsity` from 0.01 to 0.02
- Add docstring citations

#### Task 7.5.2: VTA Parameter Updates
- Validate `da_decay_rate` against DAT kinetics
- Add Grace & Bunney 1984 citation
- Add Bayer & Glimcher 2005 citation

#### Task 7.5.3: Sleep Parameter Updates
- Add Tononi & Cirelli 2006 citation for synaptic homeostasis
- Validate prune_threshold against literature

### Sprint 7.6: Test Coverage (Priority: MEDIUM)

**Goal**: Add tests for 27 untested source files

#### High Priority (Core Functionality):
- `prediction/context_encoder.py`
- `prediction/hierarchical_predictor.py`
- `prediction/latent_predictor.py`
- `prediction/prediction_integration.py`
- `memory/unified.py`

#### Medium Priority (Integration):
- `nca/theta_gamma_integration.py`
- `integrations/kymera/action_router.py`
- `learning/causal_discovery.py`

#### Lower Priority (Visualization):
- `visualization/capsule_visualizer.py`
- `visualization/ff_visualizer.py`
- `visualization/glymphatic_visualizer.py`

---

## Part 4: Architecture Fixes

### 4.1 Module Consolidation

**Issue**: `integration/` vs `integrations/` confusion

**Fix**:
1. Merge `integration/ccapi_*` into `integrations/ccapi/`
2. Update all imports
3. Remove duplicate `integration/` folder

### 4.2 Dependency Injection for Bridges

**Current**: Bridges instantiated ad-hoc or not at all

**Fix**: Create `core/container.py` with bridge factory:
```python
class BridgeContainer:
    def __init__(self, nca: NeuralCognitiveArchitecture):
        self.ff_bridge = create_ff_encoding_bridge(nca.ff_layer)
        self.capsule_bridge = create_capsule_bridge(nca.capsule_layer)
        self.dopamine_bridge = create_pc_dopamine_bridge(nca.pc_layer, nca.da_system)
        self.memory_nca_bridge = MemoryNCABridge(nca)
```

### 4.3 Feature Flags for Experimental Subsystems

Add to `core/config.py`:
```python
class FeatureFlags:
    enable_ff_encoding: bool = True
    enable_capsule_retrieval: bool = True
    enable_dreaming: bool = True
    enable_temporal_tracking: bool = True
    enable_energy_learning: bool = False  # Experimental
```

---

## Part 5: Execution Order

### Phase 7A: Critical Wiring (Parallel Execution OK)
1. [7.1.1] FFEncodingBridge → Episodic
2. [7.1.2] CapsuleRetrievalBridge → Semantic
3. [7.1.3] DopamineBridge → Consolidation
4. [7.1.4] MemoryNCABridge → Unified

### Phase 7B: Temporal & Dreaming (Sequential)
1. [7.2.1] SessionManager → Memory
2. [7.2.2] PlasticityCoordinator → Learning
3. [7.3.1] DreamConsolidation → Sleep REM
4. [7.3.2] DreamQuality → Selection

### Phase 7C: Deep Learning (Sequential)
1. [7.4.1] Coupling → Three-Factor
2. [7.4.2] RPE → Coupling

### Phase 7D: Polish (Parallel OK)
1. [7.5.*] Biological parameter corrections
2. [7.6.*] Test coverage expansion

---

## Part 6: Success Criteria

### Functional
- [ ] All bridges instantiated and called in production paths
- [ ] Temporal module tracking memory operations
- [ ] Dreaming active during REM consolidation
- [ ] Coupling matrix updating from learning signals

### Quality
- [ ] All 6983+ tests still passing
- [ ] Coverage ≥ 80%
- [ ] No new test failures introduced
- [ ] Biological accuracy ≥ 90/100

### Documentation
- [ ] All parameter citations added
- [ ] Integration points documented
- [ ] Feature flags documented

---

## Part 7: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Bridge wiring breaks existing tests | Medium | High | Add integration tests first |
| Temporal module adds overhead | Low | Medium | Feature flag to disable |
| Dreaming slows consolidation | Medium | Medium | Async dreaming, parallel |
| Coupling updates destabilize | High | High | Small learning rate, bounds |

---

## Appendix A: File Change Matrix

| Sprint | Files Modified | Files Created | Tests Added |
|--------|---------------|---------------|-------------|
| 7.1 | 4 memory files, 4 bridge files | 0 | 8 integration tests |
| 7.2 | 3 temporal files, 2 memory files | 0 | 4 integration tests |
| 7.3 | 1 sleep file, 2 dreaming files | 0 | 3 integration tests |
| 7.4 | 2 learning files, 1 nca file | 0 | 2 integration tests |
| 7.5 | 5 nca files | 0 | 5 validation tests |
| 7.6 | 0 | 0 | 27 unit tests |

---

## Appendix B: Agent Analysis Sources

### ww-hinton Agent Findings
- "Architecture without learning" pattern identified
- 9 integration points NOT CONNECTED
- Generative model stub (but VAE now exists)
- Priority: Wire the plumbing

### ww-compbio Agent Findings
- Biological accuracy: 85/100
- 4 parameter adjustments needed
- 9 missing citations
- DG sparsity and expansion ratio corrections

### Explore Agent Findings
- 3 bridges exported but never instantiated
- Temporal module completely orphaned
- Dreaming not connected to sleep
- 27 source files without tests

---

**Plan Status**: READY FOR IMPLEMENTATION
**Recommended Approach**: Execute Sprints 7.1-7.4 in parallel where possible
**Estimated Test Impact**: +50 new tests, 0 regressions expected
