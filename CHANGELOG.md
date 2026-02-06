# Changelog

All notable changes to T4DM are documented here.

## [2.1.0] - 2026-02-06

### Remediation Plan (Phases 1-4)

**Target**: Improve usability, validation, and documentation based on adversarial persona analysis

#### Added

- **SimpleBaseline** (`src/t4dm/lite.py`)
  - Minimal 3-method API for quick prototyping: `store()`, `search()`, `delete()`
  - In-memory only, zero dependencies beyond numpy
  - Hash-based mock embedding for testing

- **Decision Tracing** (`src/t4dm/observability/decision_trace.py`)
  - `@traced_decision` decorator for bio-inspired component logging
  - `DecisionTracer` with ring buffer and JSON output
  - Filtering by component, decision type, and confidence

- **Persistence Checksums** (`src/t4dm/persistence/integrity.py`)
  - SHA-256 checksums for data integrity verification
  - `ChecksumMixin` for segment writers
  - `IntegrityError` exception with detailed mismatch info

- **Benchmark Test Suite** (`tests/benchmarks/`)
  - `test_bioplausibility.py`: 16 tests for neuroscience compliance
  - `test_longmemeval.py`: 17 tests for long-term memory performance
  - `test_dmr.py`: 18 tests for dense memory retrieval
  - Pytest markers: `@pytest.mark.benchmark`, `bioplausibility`, `memory`, `retrieval`

- **Core Layer Tests**
  - `tests/unit/core/test_types.py`: 75 tests, 100% coverage
  - `tests/unit/core/test_memory_gate.py`: 62 tests, 92% coverage
  - `tests/unit/core/test_config.py`: 42 tests, 65% coverage
  - `tests/unit/core/test_protocols.py`: 38 tests, 72% coverage

- **Debugging Runbooks** (`docs/runbooks/`)
  - `DEBUGGING_MEMORY.md`: Memory operation troubleshooting
  - `DEBUGGING_STORAGE.md`: T4DX storage engine issues
  - `DEBUGGING_SPIKING.md`: Spiking block dynamics
  - `DEBUGGING_PERFORMANCE.md`: Performance optimization

- **Validation Documentation**
  - `docs/BENCHMARK_RESULTS.md`: 51/51 tests passing (100%)
  - `docs/COMPARISON.md`: SimpleBaseline vs Full T4DM comparison
  - `docs/guides/ABLATION_STUDY.md`: Component contribution methodology
  - Updated `docs/VALIDATION_REPORT.md` with Phase 3 results
  - Updated `docs/LIMITATIONS.md` with Phase 1-3 completion

- **CI/CD Integration**
  - Added benchmark-tests job to `.github/workflows/test.yml`
  - Added Makefile targets: `make benchmark`, `benchmark-bio`, `benchmark-dmr`, `benchmark-longmem`

#### Changed

- **Terminology**: Renamed `ConsciousnessMetrics` to `IntegrationMetrics`
  - Backward compatibility aliases maintained
  - Metrics measure computational integration, not consciousness claims

- **Documentation**: Added terminology clarification to `dreaming/` module
  - "Dreaming" refers to DreamerV3-style generative replay, not subjective experience

---

## [0.5.0] - 2026-01-04

### Phase 4: Capsule Networks & Integration (Sprints 11-12)

**Target**: Hinton 9.0→9.5 | Biology 94→95

#### Added

- **Capsule Networks** (`ww.nca.capsules`)
  - `CapsuleNetwork`: Part-whole hierarchical representations (Hinton 2017)
  - Dynamic routing by agreement with configurable iterations
  - Pose transformation matrices for spatial reasoning
  - Length-based probability via squashing nonlinearity
  - NT-modulated routing: DA→temperature, NE→threshold, ACh→mode

- **Glymphatic System** (`ww.nca.glymphatic`)
  - `GlymphaticSystem`: Sleep-dependent waste clearance analog
  - Sleep stage-specific clearance rates (0.1 wake → 1.0 NREM deep)
  - ISF (interstitial fluid) flow simulation
  - Waste accumulation and clearance tracking
  - Integration with adenosine sleep pressure

- **Capsule-NCA Coupling** (`ww.nca.capsule_nca_coupling`)
  - `CapsuleNCACoupling`: Bidirectional capsule ↔ NCA field coupling
  - NT state → routing modulation (DA, NE, ACh, 5-HT)
  - Capsule state → NCA feedback (agreement → stability)
  - Pose transformations → attractor geometry hints
  - Routing temperature adjustment (0.5-2.0 range)

- **Forward-Forward NCA Coupling** (`ww.nca.forward_forward_nca_coupling`)
  - `FFNCACoupling`: FF goodness ↔ NCA energy landscape alignment
  - Goodness = negative energy (high goodness = low energy basin)
  - FF threshold θ = energy barrier between attractors
  - DA → learning rate modulation (surprise-driven)
  - ACh → phase gating (encoding vs retrieval)
  - NE → threshold adjustment (arousal → selectivity)

- **Glymphatic-Consolidation Bridge** (`ww.nca.glymphatic_consolidation_bridge`)
  - `GlymphaticConsolidationBridge`: Sleep-gated clearance ↔ memory consolidation
  - Spindle-triggered micro-clearance windows (11-16 Hz)
  - Delta up-state bulk clearance (0.5-4 Hz)
  - Active replay memory protection during consolidation
  - Stale/weak memory tagging for clearance
  - Sleep stage transitions (WAKE → NREM_LIGHT → NREM_DEEP → REM)

- **H10 Cross-Region Integration Tests** (`tests/integration/test_h10_cross_region_consistency.py`)
  - 46 tests for cross-system consistency:
    - TestCapsuleNCACoupling: 12 tests
    - TestFFNCACoupling: 12 tests
    - TestGlymphaticConsolidationBridge: 12 tests
    - TestCrossSystemIntegration: 5 tests
    - TestBiologyValidation: 5 tests

- **B9 Biology Validation Suite** (`tests/biology/test_b9_biology_validation.py`)
  - 50+ parameter validation tests:
    - VTA dopamine parameters (Schultz 1998, Grace & Bunney 1984)
    - Raphe serotonin parameters (Jacobs & Azmitia 1992)
    - Locus coeruleus NE parameters (Aston-Jones 2005)
    - Hippocampal DG/CA3/CA1 parameters (Jung & McNaughton 1993)
    - Striatal D1/D2 receptor affinities
    - Oscillation frequencies (theta, gamma, ripple)
    - Cross-module timing consistency

#### Changed
- Hinton Plausibility Score: 9.0 → 9.5
- Biology Fidelity Score: 94 → 95
- Test count: 6710 → ~6900

#### Documentation
- Updated `docs/roadmaps/improvement-roadmap-v0.5.md` (Phase 4 complete)
- Updated PARAMETERS.md with new coupling parameters

---

## [0.5.0-alpha] - 2026-01-03

### Phase 3: Forward-Forward Integration (Sprint 10)

**Target**: Hinton 8.5→9.0 | Biology 92→94

#### Added

- **Forward-Forward Algorithm** (`ww.nca.forward_forward`)
  - `ForwardForwardLayer`: Single FF layer with local learning (Hinton 2022)
  - `ForwardForwardNetwork`: Multi-layer FF network without backpropagation
  - Goodness function: G(h) = Σ h_i² (sum of squared activations)
  - Positive phase: maximize goodness for real data
  - Negative phase: minimize goodness for corrupted data
  - Threshold classification (G > θ → positive)
  - Hebbian-like learning rule (pre/post correlation)
  - Multiple negative sample generation: noise, shuffle, adversarial, hybrid, wrong_label
  - Neuromodulator integration: DA→learning rate, ACh→phase, NE→threshold
  - Factory functions: `create_ff_layer()`, `create_ff_network()`

- **Grid Cell Hexagonal Validation** (`ww.nca.spatial_cells`)
  - `validate_hexagonal_pattern()`: Validate grid cell firing patterns
  - `compute_gridness_score()`: Sargolini et al. (2006) gridness metric
  - 6-fold rotational symmetry detection (Moser et al. 2008)
  - 2D spatial autocorrelation via FFT
  - Module-specific gridness computation

- **Phase 3 Test Suite**: 59 new tests
  - `tests/nca/test_forward_forward.py` - 39 unit tests
  - `tests/nca/test_ff_biology_benchmarks.py` - 20 biology benchmarks

#### Changed
- Hinton Plausibility Score: 8.5 → 9.0
- Biology Fidelity Score: 92 → 94
- Test count: 6651 → 6710

#### Documentation
- New `docs/concepts/forward-forward.md` with algorithm explanation and examples
- Updated `docs/science/learning-theory.md` (score 8.5→9.0, FF marked as implemented)
- Updated `docs/roadmaps/improvement-roadmap-v0.5.md` (Phase 3 complete)

---

### Phase 2: SWR and Neuromodulator Dynamics (Sprints 8-9)

**Target**: Hinton 8.0→8.5 | Biology 91→93

#### Added

- **SWR Timing Refinement** (`ww.nca.swr_coupling`)
  - Validated 150-250 Hz ripple frequency range (Buzsaki 2015, Carr et al. 2011)
  - Constants: `RIPPLE_FREQ_MIN=150`, `RIPPLE_FREQ_MAX=250`, `RIPPLE_FREQ_OPTIMAL=180`
  - Frequency validation with `validate_ripple_frequency()`
  - Configuration validation in `SWRConfig.__post_init__`

- **Wake-Sleep State Separation** (`ww.nca.swr_coupling.WakeSleepMode`)
  - 5 states: ACTIVE_WAKE, QUIET_WAKE, NREM_LIGHT, NREM_DEEP, REM
  - ACh/NE-based state inference
  - State-specific SWR probabilities (0% active wake, 90% deep NREM)
  - State-dependent ripple frequency modulation
  - Separate wake/sleep SWR counting

- **Serotonin Patience Model** (`ww.nca.raphe.PatienceModel`)
  - Temporal discounting based on Doya (2002), Miyazaki et al. (2014)
  - Serotonin-dependent discount rate (gamma): low 5-HT → impatient
  - Temporal horizon estimation (3-50 steps based on 5-HT)
  - Wait/don't-wait decision API with `evaluate_wait_decision()`
  - Integration with RapheNucleus: `get_discount_rate()`, `get_temporal_horizon()`

- **Surprise-Driven NE** (`ww.nca.locus_coeruleus.SurpriseModel`)
  - Uncertainty signaling based on Dayan & Yu (2006), Nassar et al. (2012)
  - Prediction error tracking with `observe_prediction_outcome()`
  - Expected vs unexpected uncertainty distinction
  - Change point detection (Bayesian update)
  - Adaptive learning rate modulation (high surprise → high LR)
  - Integration with LocusCoeruleus: `get_surprise_level()`, `should_update_model()`

- **Phase 2 Test Suite**: 65 new tests in `tests/phase2/`
  - `test_swr_phase2.py` - 17 tests
  - `test_patience_model.py` - 20 tests
  - `test_surprise_model.py` - 28 tests

#### Changed
- `RapheNucleus` now integrates `PatienceModel` with comprehensive patience API
- `LocusCoeruleus` now integrates `SurpriseModel` with uncertainty signaling API
- `SWRNeuralFieldCoupling` includes Phase 2 wake/sleep gating
- `create_swr_coupling()` factory supports `enable_state_gating` parameter

---

### Phase 1: Foundation Enhancements (Sprints 5-7)

**Target**: Hinton 7.4→8.0 | Biology 87→91

#### Added

- **Delta Oscillations** (`ww.nca.oscillators.DeltaOscillator`)
  - 0.5-4 Hz oscillator for slow-wave sleep (NREM stage 3-4)
  - Adenosine-sensitive frequency modulation (sensitivity: 0.6)
  - Up-state/down-state dynamics with ~500ms transitions
  - Consolidation gating (1.0 during up-states)
  - Synaptic downscaling signal during down-states
  - Sleep threshold at depth >= 0.3

- **Sleep Spindles** (`ww.nca.sleep_spindles`)
  - `SleepSpindleGenerator`: 11-16 Hz thalamocortical bursts
  - 4-phase spindle lifecycle: SILENT → RISING → PLATEAU → FALLING
  - GABA/ACh modulation for realistic dynamics
  - `SpindleDeltaCoupler`: Couples spindles to delta up-state onsets
  - Refractory period (200ms) prevents spindle overlap
  - Spindle density tracking (10-15/min healthy range)

- **Contrastive Adapter** (`ww.embedding.contrastive_trainer`)
  - `ContrastiveAdapter`: Learnable projection on frozen BGE-M3 embeddings
  - InfoNCE loss with temperature scaling (default τ=0.07)
  - Hard negative mining from candidate pools
  - Temporal contrastive loss for sequence coherence
  - Adam optimizer with momentum (β1=0.9, β2=0.999)
  - Learned temperature parameter
  - Weight save/load for persistence
  - Health monitoring and statistics

- **Sleep State Enum** (`ww.nca.oscillators.SleepState`)
  - WAKE, NREM1, NREM2, NREM3, REM states
  - Integrated with FrequencyBandGenerator

#### Changed
- `FrequencyBandGenerator` now includes delta oscillator
- `OscillatorState` extended with delta-related fields
- Hinton Plausibility Score: 7.4 → 8.0
- Biology Fidelity Score: 87 → 91
- 88 new tests (delta: 8, spindles: 18, contrastive: 24, integration: 38)

#### Documentation
- Updated `docs/concepts/nca.md` with Sleep Oscillations section
- Updated `docs/concepts/bioinspired.md` with delta/spindle/contrastive sections
- Updated `docs/science/learning-theory.md` (score 7.8→8.0, marked gaps addressed)
- Updated `docs/reference/sequences.md` with delta-spindle coordination diagram
- New roadmap: `docs/roadmaps/improvement-roadmap-v0.5.md`

## [0.4.0] - 2026-01-03

### Added
- **P4-1: Hierarchical Multi-Timescale Prediction** (`ww.prediction.hierarchical_predictor`)
  - 3 prediction horizons: fast (1-step), medium (5-step), slow (15-step)
  - Different learning rates per timescale (0.01, 0.001, 0.0001)
  - Automatic error resolution when target episodes arrive
  - Inspired by DreamerV3 multi-horizon architecture

- **P4-2: Causal Discovery Integration** (`ww.learning.causal_discovery`)
  - `CausalGraph` with directed edges and strength tracking
  - `CausalAttributor` for outcome attribution
  - Counterfactual learning support (`observe_counterfactual`)
  - Based on Richens & Everitt ICLR 2024 causal learning insights

- **P4-3: Place/Grid Cell Spatial Prediction** (`ww.nca.spatial_cells`)
  - 100 place cells with Gaussian receptive fields
  - 3 grid modules at different spatial scales (0.3, 0.5, 0.8)
  - Hexagonal grid cell responses (Nobel Prize 2014 - O'Keefe, Moser)
  - Path integration and neighbor finding in embedding space

- **P4-4: Theta-Gamma Coupling** (`ww.nca.theta_gamma_integration`)
  - Plasticity gating by theta phase (encoding boost 2x, retrieval suppression 0.3x)
  - Working memory slots via gamma cycles (7±2 capacity from Miller's Law)
  - PAC-reward meta-learning for phase-amplitude coupling
  - Alpha-based inhibition signal

- **P4 Test Suite**: 31 new tests in `tests/p4/`

### Changed
- Package version updated to 0.4.0
- Test suite expanded to 6,540 tests (80% coverage)
- Strategic analysis updated to reflect P1-P4 complete
- `ww.nca.__init__` exports new spatial and theta-gamma modules
- `ww.learning.__init__` exports causal discovery classes
- `ww.prediction.__init__` exports hierarchical predictor classes

## [0.3.0] - 2026-01-02

### Added
- **P1: Prediction Foundation**
  - Prioritized replay with prediction error
  - Self-supervised credit estimation
  - GABA as lateral inhibition in striatal MSN

- **P2: JEPA-Style Latent Prediction** (`ww.prediction`)
  - `ContextEncoder` with attention-weighted aggregation
  - `LatentPredictor` 2-layer MLP
  - `PredictionTracker` for error computation
  - Trained predictor beats random/mean baselines

- **P3: Dreaming System** (`ww.dreaming`)
  - 15-step dream trajectory generation
  - 4-metric quality evaluation (coherence, smoothness, novelty, informativeness)
  - REM-phase consolidation integration

### Changed
- Test suite expanded to 6,510 tests
- Coverage at 80%

## [0.2.0] - 2026-01-02

### Added
- **Simplified Memory API**: New `from ww import memory` interface
  - `memory.store()` - Store episodic content
  - `memory.recall()` - Recall across all memory types
  - `memory.store_entity()` - Store semantic entities
  - `memory.store_skill()` - Store procedural skills
  - `memory.session()` - Context manager for session isolation

- **CLI Tool**: New `ww` command for terminal-based memory management
  - `ww store` - Store content from command line
  - `ww recall` - Search memories
  - `ww status` - Show system status
  - `t4dm serve` - Start REST API server
  - `ww config` - Manage configuration
  - Subcommands: `ww episodic`, `ww semantic`, `ww procedural`

- **YAML Configuration**: New config file support
  - Searches `t4dm.yaml`, `~/.t4dm/config.yaml`, `/etc/t4dm/config.yaml`
  - Environment variable overrides (WW_* prefix)
  - `load_settings_from_yaml()` function
  - `reset_settings()` for cache clearing

### Changed
- Package version updated to 0.2.0
- Removed MCP Gateway dependency (mcp, fastmcp, anthropic packages)
- Updated pyproject.toml with new entry points (`ww`, `ww-api`)
- Dependencies: Added pydantic-settings, pyyaml, typer, rich
- Observability packages moved to optional `[observability]` extra
- Test suite expanded to 6,400+ tests

### Removed
- MCP Gateway server (`ww.mcp` module)
- MCP-specific tools and resources
- Direct dependency on anthropic SDK

### Fixed
- Episode creation with proper session_id field
- Entity creation with correct EntityType enum values
- Procedure creation with proper Domain enum values
- Config tests isolated from environment pollution

## [3.0.0-dev] - 2025-12-07

### Added
- **Phase 6: Learning Systems**
  - Dopamine-based reward prediction (TD learning, surprise signals)
  - Serotonin-based long-term credit assignment (eligibility traces)
  - ACT-R activation with spreading activation
  - FSRS spaced repetition integration
  - Neuromodulator factory for custom learning systems

- **Phase 7: Extensibility & Integration**
  - Memory hook system (pre/post/on hooks with priorities)
  - Built-in hooks: CachingRecallHook, ValidationHook, AuditTrailHook
  - HebbianUpdateHook for co-access strengthening
  - MCP compatibility layer with graceful fallback
  - Kymera AI orchestrator integration
  - OpenTelemetry-compatible tracing (WWObserver)
  - Claude Code ccapi integration

### Changed
- Test suite expanded from 255 to 4043 tests
- Coverage increased to 79%
- Project structure reorganized with new modules

### Fixed
- Entity route enum value consistency (uppercase values)
- Memory hook identity checks for custom stores
- Serotonin system numerical tolerance in tests

## [2.0.1] - 2025-11-27

### Fixed
- Session isolation: Added `sessionId` to Neo4j graph properties for episodic memory
- Session isolation: Added missing payload fields (`ingested_at`, `access_count`, `last_accessed`)
- Batch queries: Fixed Cypher parameter issue (`$id` -> `id` for UNWIND variable)
- Batch queries: Fixed return structure consistency (`other_id`, nested `properties`)
- Integration tests: Changed deterministic UUIDs to random to avoid constraint errors

### Changed
- HDBSCAN is now an optional dependency (`pip install world-weaver[consolidation]`)
- Tests requiring HDBSCAN are automatically skipped when not installed
- Archived 22 analysis/checklist markdown files to `docs/archive/`
- Updated README with current implementation status (255 tests passing)

## [2.0.0] - 2025-11-27

### Added
- **Phase 1: Critical Bug Fixes**
  - Cypher injection prevention with label validation
  - CompensationError for saga failure handling
  - Session-scoped event loop for async tests
  - Neo4j indexes for session isolation

- **Phase 2: Performance**
  - HDBSCAN clustering (3x faster than O(n²))
  - Batch relationship queries (N+1 elimination)
  - LRU embedding cache with MD5 hashing
  - Neo4j connection pooling (50 connections)
  - Parallel Hebbian updates

- **Phase 3: Security**
  - Rate limiting (100 req/60s sliding window)
  - Input sanitization (null bytes, control chars)
  - Authentication context with role-based access
  - Secure config loading with secret masking
  - Request ID tracking for audit trails

- **Phase 4: Test Coverage**
  - 222+ test methods across all modules
  - Property-based tests with Hypothesis
  - Integration and performance benchmarks
  - CI coverage gate at 70%

- **Phase 5: API Cleanup**
  - Standardized naming (create_skill, recall_skill)
  - TypedDict for all responses
  - Consistent error codes
  - Pagination with offset/limit/has_more
  - OpenAPI schema generation

### Changed
- `build_skill` renamed to `create_skill` (deprecated alias kept)
- `how_to` renamed to `recall_skill` (deprecated alias kept)
- `retrieve` renamed to `recall_skill` in procedural.py

### Deprecated
- `build_skill()` - use `create_skill()`
- `how_to()` - use `recall_skill()`
- `procedural.build()` - use `create_skill()`
- `procedural.retrieve()` - use `recall_skill()`

### Fixed
- Missing asyncio import in semantic.py
- Race condition in async lock initialization
- Session isolation in store singletons

## [1.0.0] - 2025-11-01

### Added
- Initial release
- Tripartite memory system (episodic, semantic, procedural)
- MCP gateway with 17 tools
- FSRS, ACT-R, Hebbian algorithms
