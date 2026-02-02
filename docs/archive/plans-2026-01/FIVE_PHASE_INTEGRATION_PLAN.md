# World Weaver: 5-Phase Integration Plan

**Created**: 2026-01-07
**Based On**: Comprehensive Codebase Analysis (CompBio, Hinton, Architecture)
**Execution Model**: Parallel Agent Architecture

---

## Plan Overview

### Current State
| Metric | Score | Target |
|--------|-------|--------|
| CompBio | 5.5/10 | 8.0/10 |
| Hinton | 7.2/10 | 8.5/10 |
| Architecture | 8.2/10 | 9.0/10 |
| Integration | 25% | 85% |

### Phase Summary
| Phase | Focus | Duration | Parallel Agents |
|-------|-------|----------|-----------------|
| 1 | Critical Wiring | 2 days | 4 agents |
| 2 | Bridge Activation | 2 days | 3 agents |
| 3 | Neuromodulator Integration | 2 days | 3 agents |
| 4 | Advanced Features | 3 days | 3 agents |
| 5 | Validation & Reanalysis | 1 day | 4 agents |

**Total Duration**: ~10 days
**Total Agent Invocations**: 17 parallel tasks + 3 reanalysis

---

## Phase 1: Critical Wiring

**Objective**: Wire the most critical disconnected subsystems
**Duration**: 2 days
**Priority**: CRITICAL

### Parallel Agent Tasks

#### Agent 1A: Sleep → Reconsolidation Wiring
```yaml
agent_type: ww-compbio
description: Wire sleep replay to actually update embeddings
files:
  - src/t4dm/consolidation/sleep.py (lines 1450-1500)
  - src/t4dm/learning/reconsolidation.py
tasks:
  - Add ReconsolidationEngine.batch_reconsolidate() call in _replay_episode()
  - Pass replayed episode IDs to reconsolidation
  - Add lability window check before reconsolidation
  - Update embedding vectors in Qdrant after consolidation
tests:
  - test_sleep_actually_updates_embeddings()
  - test_lability_window_prevents_early_recon()
  - test_batch_reconsolidation_during_nrem()
```

#### Agent 1B: VTA → STDP Modulation
```yaml
agent_type: ww-compbio
description: Connect VTA dopamine to STDP learning rates
files:
  - src/t4dm/nca/vta.py
  - src/t4dm/learning/stdp.py
  - NEW: src/t4dm/integration/stdp_vta_bridge.py
tasks:
  - Create STDPVTABridge class
  - Add dopamine_level parameter to STDPLearner.compute_update()
  - Modulate A+ by DA: A+ *= (1 + 0.5 * (DA - 0.5))
  - Auto-read DA from VTA singleton
tests:
  - test_high_da_increases_ltp()
  - test_low_da_increases_ltd()
  - test_stdp_vta_bridge_integration()
```

#### Agent 1C: VAE Training Loop
```yaml
agent_type: ww-compbio
description: Fix VAE wake-sleep training trigger
files:
  - src/t4dm/consolidation/sleep.py (lines 556-600)
  - src/t4dm/hooks/session_lifecycle.py
tasks:
  - Add periodic train_vae_from_wake() calls
  - Trigger VAE training in pre-consolidation hook
  - Add wake sample collection during active retrieval
  - Configure minimum samples before training (default: 100)
tests:
  - test_vae_trains_from_wake_samples()
  - test_vae_training_triggers_on_schedule()
  - test_synthetic_memories_generated_during_sleep()
```

#### Agent 1D: VTA → Sleep Replay RPE
```yaml
agent_type: ww-compbio
description: Generate RPE from replay sequences during consolidation
files:
  - src/t4dm/consolidation/sleep.py
  - src/t4dm/nca/vta.py
tasks:
  - Pass replay sequences to VTA.process_sequence()
  - Generate TD errors from temporal predictions
  - Feed RPE back to credit assignment
  - Log RPE distribution during sleep
tests:
  - test_replay_generates_rpe()
  - test_rpe_affects_replay_priority()
  - test_vta_active_during_consolidation()
```

### Phase 1 Cleanup & Documentation

#### Agent 1E: Testing & Cleanup (Sequential after 1A-1D)
```yaml
agent_type: test-runner
description: Run integration tests and cleanup
tasks:
  - Run all sleep consolidation tests
  - Run all STDP tests
  - Run all VTA tests
  - Fix any regressions
  - Remove dead code
  - Update docstrings
```

#### Agent 1F: Documentation Update (Sequential after 1E)
```yaml
agent_type: boris
description: Update documentation
files:
  - docs/ARCHITECTURE.md
  - docs/BIOLOGICAL_INTEGRATION.md
  - README.md
tasks:
  - Document new wiring connections
  - Update integration diagrams
  - Add usage examples for new bridges
```

---

## Phase 2: Bridge Activation

**Objective**: Activate existing but dormant bridge infrastructure
**Duration**: 2 days
**Priority**: HIGH

### Parallel Agent Tasks

#### Agent 2A: FFCapsuleBridge Wiring
```yaml
agent_type: ww-hinton
description: Wire Forward-Forward + Capsule bridge into episodic memory
files:
  - src/t4dm/memory/episodic.py
  - src/t4dm/bridges/ff_capsule_bridge.py
  - src/t4dm/core/bridge_container.py
tasks:
  - Instantiate FFCapsuleBridge in EpisodicMemory.__init__()
  - Call bridge.forward() in store() method
  - Use goodness score for novelty detection
  - Store capsule activations in episode metadata
tests:
  - test_ff_bridge_processes_embeddings()
  - test_capsule_activations_stored()
  - test_novelty_detection_from_goodness()
```

#### Agent 2B: BridgeContainer Activation
```yaml
agent_type: boris
description: Activate lazy-initialized bridge container
files:
  - src/t4dm/core/bridge_container.py
  - src/t4dm/memory/episodic.py
  - src/t4dm/api/deps.py
tasks:
  - Trigger bridge container init in session creation
  - Pass required components (FF layer, capsule layer, VTA)
  - Add bridge container to API dependencies
  - Ensure cleanup on session end
tests:
  - test_bridge_container_initializes_on_session()
  - test_bridges_available_after_init()
  - test_bridge_cleanup_on_session_end()
```

#### Agent 2C: Capsule Pose Storage
```yaml
agent_type: ww-hinton
description: Store capsule pose matrices in vector database
files:
  - src/t4dm/storage/qdrant_store.py
  - src/t4dm/memory/episodic.py
  - src/t4dm/nca/capsules.py
tasks:
  - Add pose_matrices field to Qdrant payload schema
  - Store capsule activations alongside embeddings
  - Add pose retrieval method
  - Use routing agreement for confidence scoring
tests:
  - test_pose_matrices_stored()
  - test_pose_matrices_retrieved()
  - test_routing_agreement_as_confidence()
```

### Phase 2 Cleanup & Documentation

#### Agent 2D: Testing & Cleanup
```yaml
agent_type: test-runner
description: Run bridge integration tests
tasks:
  - Run all FF encoder tests
  - Run all capsule tests
  - Run bridge integration tests
  - Verify 80%+ coverage maintained
```

#### Agent 2E: Documentation Update
```yaml
agent_type: boris
description: Document bridge architecture
files:
  - docs/BRIDGE_ARCHITECTURE.md (NEW)
  - docs/API_REFERENCE.md
tasks:
  - Create bridge architecture documentation
  - Document bridge lifecycle
  - Add configuration examples
```

---

## Phase 3: Neuromodulator Integration

**Objective**: Connect neuromodulator system to all learning pathways
**Duration**: 2 days
**Priority**: HIGH

### Parallel Agent Tasks

#### Agent 3A: NT → Credit Flow
```yaml
agent_type: ww-compbio
description: Wire neuromodulators to credit assignment
files:
  - src/t4dm/learning/credit_flow.py
  - src/t4dm/nca/neural_field.py
tasks:
  - Add get_neuromodulator_context() reading NeuralFieldSolver
  - Modulate learning rate: lr *= (1 + DA - 0.5*5HT)
  - Add ACh attention modulation
  - Add NE exploration modulation
tests:
  - test_da_increases_learning_rate()
  - test_5ht_decreases_learning_rate()
  - test_ach_modulates_attention()
  - test_ne_modulates_exploration()
```

#### Agent 3B: NT → Three-Factor Learning
```yaml
agent_type: ww-compbio
description: Enhance three-factor learning with full NT context
files:
  - src/t4dm/learning/three_factor.py
  - src/t4dm/learning/eligibility.py
tasks:
  - Add full NT state to eligibility trace computation
  - Implement protein synthesis gate (lability window)
  - Add GABA inhibition to learning
  - Connect to NeuralFieldSolver state
tests:
  - test_eligibility_modulated_by_nt()
  - test_protein_synthesis_gate()
  - test_gaba_inhibits_learning()
```

#### Agent 3C: NT → Neurogenesis
```yaml
agent_type: ww-compbio
description: Wire neuromodulators to neurogenesis
files:
  - src/t4dm/encoding/neurogenesis.py
  - src/t4dm/nca/vta.py
tasks:
  - Add DA modulation to birth rate: birth_rate *= (1 + DA)
  - Add ACh modulation to maturation speed
  - Add stress hormone suppression (high NE inhibits birth)
  - Connect to VTA singleton
tests:
  - test_da_enhances_neurogenesis()
  - test_high_ne_suppresses_birth()
  - test_ach_speeds_maturation()
```

### Phase 3 Cleanup & Documentation

#### Agent 3D: Testing & Cleanup
```yaml
agent_type: test-runner
description: Run neuromodulator integration tests
tasks:
  - Run all credit flow tests
  - Run all three-factor tests
  - Run all neurogenesis tests
  - Integration test: NT → all learning pathways
```

#### Agent 3E: Documentation Update
```yaml
agent_type: boris
description: Document neuromodulator integration
files:
  - docs/NEUROMODULATOR_INTEGRATION.md (NEW)
  - docs/LEARNING_SYSTEMS.md
tasks:
  - Create neuromodulator documentation
  - Document NT → learning pathway mappings
  - Add biological references
```

---

## Phase 4: Advanced Features

**Objective**: Implement missing Hinton features and complete capsule integration
**Duration**: 3 days
**Priority**: MEDIUM

### Parallel Agent Tasks

#### Agent 4A: Boltzmann Machine Implementation
```yaml
agent_type: ww-hinton
description: Implement RBM/DBN for Hinton-aligned generative modeling
files:
  - NEW: src/t4dm/nca/rbm.py
  - NEW: src/t4dm/nca/dbn.py
  - src/t4dm/consolidation/sleep.py
tasks:
  - Create RestrictedBoltzmannMachine class
  - Implement contrastive divergence training
  - Create DeepBeliefNetwork with layerwise pretraining
  - Option to use DBN instead of VAE for dream generation
tests:
  - test_rbm_learns_distribution()
  - test_dbn_layerwise_pretraining()
  - test_gibbs_sampling_generates_memories()
  - test_dbn_dream_generation()
```

#### Agent 4B: Hopfield Energy Retrieval
```yaml
agent_type: ww-hinton
description: Route retrievals through energy minimization
files:
  - src/t4dm/nca/energy.py
  - src/t4dm/memory/episodic.py
  - src/t4dm/encoding/attractor.py
tasks:
  - Add HopfieldRetrieval mode to episodic memory
  - Use attractor dynamics for retrieval
  - Connect arousal (NE) to temperature parameter
  - Benchmark vs pure vector similarity
tests:
  - test_hopfield_retrieval_converges()
  - test_arousal_affects_retrieval_sharpness()
  - test_attractor_basins_form()
```

#### Agent 4C: Capsule Hierarchy
```yaml
agent_type: ww-hinton
description: Build hierarchical memory structure with capsules
files:
  - src/t4dm/nca/capsules.py
  - src/t4dm/memory/episodic.py
  - NEW: src/t4dm/memory/hierarchical.py
tasks:
  - Create MemoryCapsuleLayer (entity, relation, context)
  - Build episode → concept → schema hierarchy
  - Use routing agreement for relationship strength
  - Connect to semantic memory extraction
tests:
  - test_episode_to_concept_routing()
  - test_schema_formation()
  - test_hierarchical_retrieval()
```

### Phase 4 Cleanup & Documentation

#### Agent 4D: Testing & Cleanup
```yaml
agent_type: test-runner
description: Run advanced feature tests
tasks:
  - Run all RBM/DBN tests
  - Run all Hopfield tests
  - Run all capsule hierarchy tests
  - Performance benchmarks
```

#### Agent 4E: Documentation Update
```yaml
agent_type: boris
description: Document advanced features
files:
  - docs/BOLTZMANN_MACHINES.md (NEW)
  - docs/HOPFIELD_RETRIEVAL.md (NEW)
  - docs/CAPSULE_HIERARCHY.md (NEW)
tasks:
  - Create feature documentation
  - Add theoretical background
  - Add usage examples
```

---

## Phase 5: Validation & Reanalysis

**Objective**: Validate all integrations and re-score with analysis agents
**Duration**: 1 day
**Priority**: CRITICAL

### Parallel Validation Tasks

#### Agent 5A: Integration Test Suite
```yaml
agent_type: test-runner
description: Run comprehensive integration tests
tasks:
  - Create tests/integration/test_biology_integration.py
  - Test: Sleep → Reconsolidation → Embedding Update
  - Test: VTA → STDP → Weight Change
  - Test: NT → All Learning Pathways
  - Test: FF → Capsule → Storage
  - Test: Hopfield → Retrieval
  - Verify all subsystems communicate
```

#### Agent 5B: Coverage Verification
```yaml
agent_type: test-runner
description: Ensure coverage targets maintained
tasks:
  - Run pytest --cov with full suite
  - Target: 80%+ overall coverage
  - Target: 90%+ for integration code
  - Identify and fix any coverage gaps
```

#### Agent 5C: Performance Benchmarks
```yaml
agent_type: boris
description: Benchmark performance impact
tasks:
  - Benchmark retrieval latency (pre vs post)
  - Benchmark sleep consolidation time
  - Benchmark memory usage
  - Document any regressions
```

### Reanalysis (Sequential after 5A-5C)

#### Agent 5D: CompBio Reanalysis
```yaml
agent_type: ww-compbio
description: Re-evaluate biological accuracy
tasks:
  - Re-score all 8 subsystems
  - Verify integration matrix improvements
  - Identify remaining gaps
  - Target: 8.0/10 overall
```

#### Agent 5E: Hinton Reanalysis
```yaml
agent_type: ww-hinton
description: Re-evaluate Hinton principle adherence
tasks:
  - Re-score all 10 principles
  - Verify RBM implementation quality
  - Assess FF/Capsule integration
  - Target: 8.5/10 overall
```

#### Agent 5F: Architecture Reanalysis
```yaml
agent_type: boris
description: Re-evaluate architecture quality
tasks:
  - Re-score all 10 dimensions
  - Verify integration debt reduced
  - Assess new module quality
  - Target: 9.0/10 overall
```

### Final Documentation

#### Agent 5G: Final Documentation Update
```yaml
agent_type: boris
description: Final documentation pass
files:
  - COMPREHENSIVE_CODEBASE_ANALYSIS.md (UPDATE)
  - CHANGELOG.md
  - README.md
tasks:
  - Update all scores with reanalysis results
  - Document completed integrations
  - Update architecture diagrams
  - Create release notes
```

---

## Execution Order

```
Phase 1 (Day 1-2):
├── [PARALLEL] Agent 1A: Sleep→Recon
├── [PARALLEL] Agent 1B: VTA→STDP
├── [PARALLEL] Agent 1C: VAE Training
├── [PARALLEL] Agent 1D: VTA→Sleep RPE
├── [SEQUENTIAL] Agent 1E: Testing & Cleanup
└── [SEQUENTIAL] Agent 1F: Documentation

Phase 2 (Day 3-4):
├── [PARALLEL] Agent 2A: FFCapsuleBridge
├── [PARALLEL] Agent 2B: BridgeContainer
├── [PARALLEL] Agent 2C: Capsule Storage
├── [SEQUENTIAL] Agent 2D: Testing & Cleanup
└── [SEQUENTIAL] Agent 2E: Documentation

Phase 3 (Day 5-6):
├── [PARALLEL] Agent 3A: NT→CreditFlow
├── [PARALLEL] Agent 3B: NT→ThreeFactor
├── [PARALLEL] Agent 3C: NT→Neurogenesis
├── [SEQUENTIAL] Agent 3D: Testing & Cleanup
└── [SEQUENTIAL] Agent 3E: Documentation

Phase 4 (Day 7-9):
├── [PARALLEL] Agent 4A: RBM/DBN
├── [PARALLEL] Agent 4B: Hopfield Retrieval
├── [PARALLEL] Agent 4C: Capsule Hierarchy
├── [SEQUENTIAL] Agent 4D: Testing & Cleanup
└── [SEQUENTIAL] Agent 4E: Documentation

Phase 5 (Day 10):
├── [PARALLEL] Agent 5A: Integration Tests
├── [PARALLEL] Agent 5B: Coverage Check
├── [PARALLEL] Agent 5C: Benchmarks
├── [SEQUENTIAL] Agent 5D: CompBio Reanalysis
├── [SEQUENTIAL] Agent 5E: Hinton Reanalysis
├── [SEQUENTIAL] Agent 5F: Architecture Reanalysis
└── [SEQUENTIAL] Agent 5G: Final Documentation
```

---

## Success Criteria

### Per-Phase Gates

| Phase | Gate Criteria |
|-------|---------------|
| 1 | All 4 critical wirings complete, tests pass |
| 2 | All bridges active, capsule storage working |
| 3 | NT modulates all learning pathways |
| 4 | RBM/DBN functional, Hopfield retrieval working |
| 5 | All tests pass, scores meet targets |

### Final Targets

| Metric | Current | Target | Validation |
|--------|---------|--------|------------|
| CompBio | 5.5/10 | 8.0/10 | Agent 5D reanalysis |
| Hinton | 7.2/10 | 8.5/10 | Agent 5E reanalysis |
| Architecture | 8.2/10 | 9.0/10 | Agent 5F reanalysis |
| Integration | 25% | 85% | Integration matrix |
| Coverage | 81% | 85% | pytest --cov |

---

## Agent Type Mapping

| Agent Type | Use For |
|------------|---------|
| `ww-compbio` | Biological accuracy, neural wiring |
| `ww-hinton` | Hinton principles, FF/Capsule/RBM |
| `boris` | Architecture, documentation |
| `test-runner` | Testing, cleanup, coverage |

---

## Risk Mitigation

1. **Integration Conflicts**: Run tests after each parallel batch
2. **Performance Regression**: Benchmark at each phase
3. **Coverage Drop**: Verify 80%+ maintained
4. **Scope Creep**: Strict adherence to task definitions

---

*Plan generated from CompBio, Hinton, Architecture analysis*
*Execution model: Parallel agents within phases, sequential phases*
