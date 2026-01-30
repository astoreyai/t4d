# World Weaver: 9-Phase Improvement Plan

**Created**: 2026-01-07 | **Status**: PLANNING
**Target**: Address all findings from CompBio, Hinton, and Architecture analyses
**Duration**: 12-14 weeks with parallel execution

---

## Executive Summary

This plan addresses findings from three specialized agent analyses:

| Agent | Current Score | Target Score | Key Gap |
|-------|---------------|--------------|---------|
| **CompBio** | 92/100 | 98/100 | 4 minor biological issues |
| **Hinton** | 6.5/10 | 8.0/10 | Frozen embeddings, hand-set poses |
| **Architecture** | 7.7/10 | 9.0/10 | God objects, no caching |

### Parallel Execution Strategy

```
Week 1-2:   [Phase 1: Foundation]     CompBio-A | Arch-A | Hinton-A  (parallel)
Week 3-4:   [Phase 2: Core Learning]  CompBio-B | Arch-B | Hinton-B  (parallel)
Week 5-6:   [Phase 3: Performance]    Architecture-focused
Week 7-8:   [Phase 4: Neural Growth]  CompBio-C | Hinton-C           (parallel)
Week 9-10:  [Phase 5: Generative]     Hinton-focused
Week 11:    [Phase 6: Integration]    All agents coordinate
Week 12:    [Phase 7: Quality]        Architecture + Documentation
Week 13:    [Phase 8: Validation]     Full testing sprint
Week 14:    [Phase 9: Reanalysis]     Agent re-evaluation
```

---

## Phase 1: Foundation (Week 1-2)

**Objective**: Establish core infrastructure for all subsequent work

### 1A. Architecture: Episodic Decomposition
**Owner**: Architecture Agent | **Effort**: 40h | **Parallel**: Yes

Split `memory/episodic.py` (3,616 lines) into focused modules:

```
src/ww/memory/
├── episodic.py              # Facade (400 lines) - backward compatible
├── episodic_storage.py      # CRUD + saga (800 lines)
├── episodic_retrieval.py    # Search + scoring (1,200 lines)
├── episodic_learning.py     # Reconsolidation + three-factor (600 lines)
├── episodic_fusion.py       # Learned fusion (400 lines)
└── episodic_saga.py         # Transaction coordination (400 lines)
```

**Files to Create**:
- `src/ww/memory/episodic_storage.py`
- `src/ww/memory/episodic_retrieval.py`
- `src/ww/memory/episodic_learning.py`
- `src/ww/memory/episodic_fusion.py`
- `src/ww/memory/episodic_saga.py`

**Files to Modify**:
- `src/ww/memory/episodic.py` → Facade only

**Testing**:
- All 7,970 existing tests must pass
- No new tests needed (facade preserves API)

**Documentation**:
- Update `docs/architecture.md` with new module structure
- Add `docs/reference/episodic-modules.md`

### 1B. CompBio: VTA Exponential Decay + TAN Pause
**Owner**: CompBio Agent | **Effort**: 24h | **Parallel**: Yes

**Fix 1: VTA Exponential Decay**
```python
# src/ww/nca/vta.py - Replace linear with exponential
# Current (linear):
da_level -= decay_rate * dt

# Fixed (exponential, Grace & Bunney 1984):
tau_decay = 0.2  # 200ms time constant
da_level = da_target + (da_level - da_target) * np.exp(-dt / tau_decay)
```

**Fix 2: TAN Pause Mechanism**
```python
# src/ww/nca/striatal_msn.py - Add cholinergic interneuron pause
class TANPauseMechanism:
    """Tonically Active Neuron pause during unexpected rewards (Aosaki 1994)."""
    pause_duration: float = 0.2  # 200ms

    def process_reward_signal(self, reward: float, expected: float) -> float:
        surprise = abs(reward - expected)
        if surprise > self.threshold:
            self._initiate_pause()
        return self._get_ach_level()
```

**Files to Modify**:
- `src/ww/nca/vta.py` - Exponential decay
- `src/ww/nca/striatal_msn.py` - TAN pause
- `src/ww/nca/dopamine_integration.py` - Wire TAN to DA system

**Testing**:
- `tests/nca/test_vta_decay.py` - Verify exponential curve
- `tests/nca/test_tan_pause.py` - Verify 200ms pause timing
- Integration: DA + TAN coordination

**Documentation**:
- Update `docs/science/biology-audit.md` with fixes
- Add citations: Grace & Bunney (1984), Aosaki et al. (1994)

### 1C. Hinton: Retrieval Feedback Loop
**Owner**: Hinton Agent | **Effort**: 32h | **Parallel**: Yes

Establish the missing feedback loop from retrieval outcomes to embedding adaptation:

```python
# src/ww/learning/retrieval_feedback.py (NEW)
class RetrievalFeedbackCollector:
    """Collect implicit feedback from retrieval outcomes."""

    def record_retrieval(self, query_id: str, results: list[str],
                         clicked: list[str], dwell_times: dict[str, float]):
        """Record which results were useful."""
        for result_id in results:
            relevance = self._compute_relevance(result_id, clicked, dwell_times)
            self._feedback_buffer.append(RetrievalFeedback(
                query_id=query_id,
                result_id=result_id,
                relevance=relevance,
                timestamp=datetime.now()
            ))

    def get_training_batch(self, batch_size: int) -> list[RetrievalFeedback]:
        """Get batch for adapter training."""
        return self._feedback_buffer.sample(batch_size)
```

**Files to Create**:
- `src/ww/learning/retrieval_feedback.py`
- `src/ww/learning/feedback_signals.py`

**Files to Modify**:
- `src/ww/memory/episodic_retrieval.py` - Hook feedback collection

**Testing**:
- `tests/learning/test_retrieval_feedback.py` - 15+ tests
- Integration: Feedback → Learning signal

**Documentation**:
- Create `docs/concepts/learning-loop.md`

---

## Phase 2: Core Learning (Week 3-4)

**Objective**: Implement core learning mechanisms

### 2A. CompBio: STDP + Astrocyte Enhancements
**Owner**: CompBio Agent | **Effort**: 20h | **Parallel**: Yes

**Fix 3: Multiplicative STDP**
```python
# src/ww/learning/stdp.py - Add weight dependence
def compute_weight_update(self, pre_time: float, post_time: float,
                          current_weight: float) -> float:
    dt = post_time - pre_time
    if dt > 0:  # LTP
        # Multiplicative: larger weights → smaller updates
        return self.a_plus * (self.w_max - current_weight)**self.mu * np.exp(-dt/self.tau_plus)
    else:  # LTD
        return -self.a_minus * current_weight**self.mu * np.exp(dt/self.tau_minus)
```

**Fix 4: Astrocyte Gap Junctions**
```python
# src/ww/nca/astrocyte.py - Add Ca2+ wave propagation
class AstrocyteNetwork:
    """Gap junction-mediated Ca2+ waves (Scemes & Bhattacharji 2003)."""
    propagation_speed: float = 15.0  # μm/s
    coupling_strength: float = 0.3

    def propagate_calcium_wave(self, source_idx: int, amplitude: float):
        """Propagate Ca2+ signal through astrocyte network."""
        for neighbor_idx in self._get_neighbors(source_idx):
            distance = self._get_distance(source_idx, neighbor_idx)
            delay = distance / self.propagation_speed
            attenuated = amplitude * np.exp(-distance / self.space_constant)
            self._schedule_activation(neighbor_idx, delay, attenuated)
```

**Files to Modify**:
- `src/ww/learning/stdp.py` - Multiplicative term
- `src/ww/nca/astrocyte.py` - Gap junctions

**Testing**:
- `tests/learning/test_stdp_multiplicative.py`
- `tests/nca/test_astrocyte_network.py`

### 2B. Architecture: Router Refactoring
**Owner**: Architecture Agent | **Effort**: 16h | **Parallel**: Yes

Split `create_ww_router()` (429 lines) into route groups:

```python
# src/ww/api/routes/config.py - Refactored structure
def create_config_routes() -> APIRouter:
    """Config-specific routes."""
    router = APIRouter(prefix="/config", tags=["config"])
    # ~100 lines
    return router

def create_preset_routes() -> APIRouter:
    """Preset management routes."""
    router = APIRouter(prefix="/presets", tags=["presets"])
    # ~80 lines
    return router

def create_ww_router() -> APIRouter:
    """Main router - delegates to sub-routers."""
    router = APIRouter()
    router.include_router(create_config_routes())
    router.include_router(create_preset_routes())
    # ~50 lines total
    return router
```

**Files to Modify**:
- `src/ww/api/routes/config.py` - Split into groups

**Testing**:
- All API tests must pass unchanged

### 2C. Hinton: Embedding Adaptation
**Owner**: Hinton Agent | **Effort**: 48h | **Parallel**: Yes

Implement LoRA adapter training from retrieval feedback:

```python
# src/ww/encoding/online_adapter.py (NEW)
class OnlineEmbeddingAdapter:
    """Train LoRA adapters from retrieval feedback."""

    def __init__(self, base_dim: int = 1024, adapter_rank: int = 32):
        self.lora_A = np.random.randn(base_dim, adapter_rank) * 0.01
        self.lora_B = np.zeros((adapter_rank, base_dim))
        self.scale = 0.1

    def adapt(self, embedding: np.ndarray) -> np.ndarray:
        """Apply learned adaptation."""
        delta = embedding @ self.lora_A @ self.lora_B
        return embedding + self.scale * delta

    def train_step(self, query_emb: np.ndarray, positive_embs: list[np.ndarray],
                   negative_embs: list[np.ndarray], lr: float = 0.001):
        """Contrastive training from feedback."""
        # Pull positives closer, push negatives away
        adapted_query = self.adapt(query_emb)

        pos_loss = sum(1 - cosine_sim(adapted_query, p) for p in positive_embs)
        neg_loss = sum(max(0, cosine_sim(adapted_query, n) - margin) for n in negative_embs)

        # Gradient update to LoRA matrices
        self._update_lora(pos_loss + neg_loss, lr)
```

**Files to Create**:
- `src/ww/encoding/online_adapter.py`
- `src/ww/encoding/adapter_training.py`

**Files to Modify**:
- `src/ww/encoding/ff_encoder.py` - Integrate adapter
- `src/ww/memory/episodic_retrieval.py` - Use adapted embeddings

**Testing**:
- `tests/encoding/test_online_adapter.py` - 20+ tests
- Validation: MRR improvement over baseline

**Documentation**:
- Update `docs/LEARNING_FLOW.md`

---

## Phase 3: Performance Infrastructure (Week 5-6)

**Objective**: Add caching and rate limiting for production readiness

### 3A. Redis Caching Layer
**Owner**: Architecture Agent | **Effort**: 32h

```python
# src/ww/core/cache.py (NEW)
class RedisCache:
    """Multi-tier caching for embeddings and search results."""

    EMBEDDING_TTL = 3600      # 1 hour - expensive to compute
    SEARCH_TTL = 300          # 5 minutes - query-dependent
    GRAPH_TTL = 600           # 10 minutes - semi-static

    async def get_embedding(self, text_hash: str) -> np.ndarray | None:
        """Get cached embedding."""
        data = await self.redis.get(f"emb:{text_hash}")
        return np.frombuffer(data, dtype=np.float32) if data else None

    async def cache_embedding(self, text_hash: str, embedding: np.ndarray):
        """Cache embedding with TTL."""
        await self.redis.setex(
            f"emb:{text_hash}",
            self.EMBEDDING_TTL,
            embedding.tobytes()
        )
```

**Files to Create**:
- `src/ww/core/cache.py`
- `src/ww/core/cache_config.py`

**Files to Modify**:
- `src/ww/embedding/adapter.py` - Add cache checks
- `src/ww/storage/qdrant_store.py` - Cache search results

### 3B. API Rate Limiting
**Owner**: Architecture Agent | **Effort**: 16h

```python
# src/ww/api/middleware/rate_limit.py (NEW)
class TokenBucketRateLimiter:
    """Rate limiting with token bucket algorithm."""

    def __init__(self, rate: int = 100, burst: int = 200):
        self.rate = rate      # requests per minute
        self.burst = burst    # max burst size
        self.buckets: dict[str, TokenBucket] = {}

    async def check_rate_limit(self, client_id: str) -> bool:
        bucket = self.buckets.setdefault(client_id, TokenBucket(self.rate, self.burst))
        return bucket.consume(1)
```

**Files to Create**:
- `src/ww/api/middleware/rate_limit.py`

**Files to Modify**:
- `src/ww/api/server.py` - Add middleware

**Testing**:
- `tests/api/test_rate_limiting.py`
- Load test: Verify 429 responses under load

---

## Phase 4: Neural Growth (Week 7-8)

**Objective**: Implement adaptive neural capacity

### 4A. CompBio: Activity-Dependent Neurogenesis
**Owner**: CompBio Agent | **Effort**: 40h | **Parallel**: Yes

Address "frozen embeddings" with biologically-inspired neuron birth/death:

```python
# src/ww/encoding/neurogenesis.py (NEW)
class NeurogenesisManager:
    """Activity-dependent neuron birth/death (Kempermann 2015)."""

    birth_rate: float = 0.001     # ~700 neurons/day equivalent
    survival_threshold: float = 0.1
    maturation_epochs: int = 10

    def maybe_add_neuron(self, novelty_score: float, layer: ForwardForwardLayer):
        """Birth new neuron if high novelty detected."""
        if novelty_score > self.novelty_threshold and random() < self.birth_rate:
            new_weights = self._initialize_immature_weights(layer)
            layer.add_neuron(new_weights, maturity=0.0)

    def prune_inactive(self, layer: ForwardForwardLayer, activity_history: np.ndarray):
        """Remove neurons with low cumulative activity."""
        mean_activity = activity_history.mean(axis=0)
        to_remove = np.where(mean_activity < self.survival_threshold)[0]
        for idx in reversed(to_remove):
            layer.remove_neuron(idx)
```

**Files to Create**:
- `src/ww/encoding/neurogenesis.py`

**Files to Modify**:
- `src/ww/encoding/ff_encoder.py` - Integrate neurogenesis
- `src/ww/nca/forward_forward.py` - Add/remove neuron methods

**Testing**:
- `tests/encoding/test_neurogenesis.py` - 15+ tests
- Validation: Neuron count stabilizes over time

### 4B. Hinton: Emergent Pose Learning
**Owner**: Hinton Agent | **Effort**: 56h | **Parallel**: Yes

Make capsule poses emerge from routing agreement:

```python
# src/ww/nca/pose_learner.py (NEW)
class PoseDimensionDiscovery:
    """Learn pose dimensions from routing patterns."""

    def __init__(self, n_dimensions: int = 4, hidden_dim: int = 16):
        # Start with semantic hints but allow emergence
        self.dimension_names = ["dim_0", "dim_1", "dim_2", "dim_3"]
        self.transform_matrices = [np.eye(4) for _ in range(n_dimensions)]

    def learn_from_routing(self, lower_poses: np.ndarray, upper_poses: np.ndarray,
                           routing_weights: np.ndarray, agreement: float):
        """Update pose transformations based on routing consensus."""
        if agreement > self.agreement_threshold:
            # High agreement: small updates (already good)
            lr = self.base_lr * (1.0 - agreement)
        else:
            # Low agreement: larger updates toward consensus
            lr = self.base_lr

        consensus = self._compute_consensus_pose(lower_poses, routing_weights)
        prediction = lower_poses @ self.transform_matrices[0]

        # Hebbian: strengthen transforms that predict consensus
        delta = lr * np.outer(lower_poses.mean(axis=0), consensus - prediction.mean(axis=0))
        self.transform_matrices[0] += delta
```

**Files to Create**:
- `src/ww/nca/pose_learner.py`

**Files to Modify**:
- `src/ww/nca/capsules.py` - Use learned poses
- `src/ww/nca/pose.py` - Remove hard-coded setters

**Testing**:
- `tests/nca/test_pose_learner.py` - 20+ tests
- Validation: Pose dimensions correlate with content structure

---

## Phase 5: Generative Replay (Week 9-10)

**Objective**: Implement VAE-based memory replay during consolidation

### 5A. VAE Training Integration
**Owner**: Hinton Agent | **Effort**: 48h

```python
# src/ww/learning/vae_training.py (NEW)
class VAEReplayTrainer:
    """Train VAE from wake experiences for generative replay."""

    def __init__(self, vae: VAEGenerator, memory: EpisodicMemory):
        self.vae = vae
        self.memory = memory
        self.training_buffer: list[np.ndarray] = []

    async def collect_wake_samples(self, n_samples: int = 100):
        """Collect recent embeddings for VAE training."""
        recent = await self.memory.get_recent(hours=24, limit=n_samples)
        self.training_buffer.extend([ep.embedding for ep in recent])

    def train_vae(self, epochs: int = 10, batch_size: int = 32):
        """Train VAE on collected samples."""
        for epoch in range(epochs):
            for batch in self._get_batches(batch_size):
                loss = self.vae.train_step(batch)
            logger.info(f"VAE epoch {epoch}: loss={loss:.4f}")
```

### 5B. Sleep Consolidation Wiring
**Owner**: Hinton Agent | **Effort**: 32h

```python
# src/ww/consolidation/sleep.py - Modify to use VAE replay
async def _run_nrem_phase(self):
    """NREM: Replay via VAE generation."""
    # Generate synthetic memories
    n_replay = self.config.replay_samples_per_cycle
    synthetic = self.vae_trainer.vae.generate(n_replay, temperature=0.8)

    # Interleave with real memories (prevent catastrophic forgetting)
    real = await self.memory.get_recent(hours=6, limit=n_replay // 2)
    combined = self._interleave(synthetic, [r.embedding for r in real])

    # Replay through FF encoder
    for emb in combined:
        self.ff_encoder.replay_consolidation(emb, is_synthetic=True)
```

**Files to Modify**:
- `src/ww/consolidation/sleep.py` - VAE integration
- `src/ww/learning/generative_replay.py` - Use VAE not stored patterns

**Testing**:
- `tests/consolidation/test_vae_replay.py` - 15+ tests
- Validation: Memory retention improves with VAE vs stored replay

---

## Phase 6: Deep Integration (Week 11)

**Objective**: Unify all learning systems

### 6A. FF-Capsule Bridge
**Owner**: Hinton Agent | **Effort**: 40h

```python
# src/ww/bridges/ff_capsule_bridge.py (NEW)
class FFCapsuleBridge:
    """Unify FF goodness with capsule routing."""

    def forward(self, embedding: np.ndarray) -> tuple[np.ndarray, CapsuleState]:
        # FF encoding
        ff_output = self.ff_encoder.encode(embedding)
        ff_goodness = self.ff_encoder.compute_goodness(ff_output)

        # Capsule routing
        capsule_state = self.capsule_layer.forward_with_routing(ff_output)
        routing_agreement = capsule_state.agreement

        # Combined confidence
        confidence = 0.6 * ff_goodness + 0.4 * routing_agreement

        return ff_output, capsule_state, confidence

    def learn(self, outcome: float):
        """Joint learning from outcome."""
        # FF learns from goodness
        self.ff_encoder.learn_from_outcome(outcome)

        # Capsules learn pose alignment
        self.capsule_layer.learn_pose_from_routing(self.last_state, outcome)
```

### 6B. Learning Signals Unification
**Owner**: All Agents | **Effort**: 24h

```python
# src/ww/learning/unified_signals.py (NEW)
class UnifiedLearningSignal:
    """Combine all learning signals into coherent update."""

    def compute_update(self, context: LearningContext) -> LearningUpdate:
        # Three-factor base
        three_factor = (context.eligibility *
                       context.neuromod_gate *
                       context.dopamine_surprise)

        # FF goodness modulation
        ff_mod = self.ff_encoder.get_goodness_gradient()

        # Capsule agreement modulation
        capsule_mod = self.capsule_layer.get_agreement_gradient()

        # Neurogenesis signal (birth/death)
        neuro_signal = self.neurogenesis.get_structural_update()

        return LearningUpdate(
            weight_delta=three_factor * (ff_mod + capsule_mod),
            structure_delta=neuro_signal
        )
```

---

## Phase 7: Quality & Observability (Week 12)

**Objective**: Production hardening

### 7A. Print → Logger Conversion
**Owner**: Architecture Agent | **Effort**: 16h

Convert all 232 `print()` statements to structured logging:

```python
# Before:
print(f"Processing {len(items)} items")

# After:
logger.info("Processing items", extra={"count": len(items)})
```

**Files to Modify**: 47 files with print statements

### 7B. Bridge Test Coverage
**Owner**: Architecture Agent | **Effort**: 24h

Increase bridge module coverage from ~60% to 85%+:

**Files to Create**:
- `tests/bridges/test_ff_capsule_bridge.py`
- `tests/bridges/test_ff_retrieval_scorer.py`
- `tests/bridges/test_unified_signals.py`
- 15+ additional test files

### 7C. Documentation Updates
**Owner**: All Agents | **Effort**: 16h

Update all affected documentation:

| File | Updates |
|------|---------|
| `docs/architecture.md` | New module structure |
| `docs/concepts/learning-loop.md` | Feedback system |
| `docs/concepts/neurogenesis.md` | Neural growth (NEW) |
| `docs/concepts/emergent-poses.md` | Pose learning (NEW) |
| `docs/science/biology-audit.md` | Fixed parameters |
| `docs/reference/api.md` | Rate limiting |
| `README.md` | Updated test count, features |

---

## Phase 8: Integration Validation (Week 13)

**Objective**: Comprehensive testing and benchmarking

### 8A. Full Test Suite
- All 8,000+ tests must pass
- New tests: 200+ added
- Coverage target: 85%+

### 8B. Performance Benchmarks

| Metric | Before | Target |
|--------|--------|--------|
| Cached query latency | 50ms | 10ms |
| Embedding generation | 100ms | 100ms (cached: 5ms) |
| Graph traversal (100 nodes) | 2000ms | 200ms |
| Memory consolidation cycle | 5min | 3min |

### 8C. Learning Validation

| Metric | Baseline | Target |
|--------|----------|--------|
| Retrieval MRR | 0.65 | 0.75 (+15%) |
| Routing convergence | 5 iterations | 3 iterations |
| Memory retention (7 day) | 70% | 85% |
| FF goodness separation | 0.3 | 0.6 |

### 8D. Regression Testing
- No degradation in existing functionality
- All API contracts preserved
- Backward compatibility verified

---

## Phase 9: Agent Reanalysis (Week 14)

**Objective**: Verify improvements with same agents

### 9A. CompBio Reanalysis
```
Expected Score: 92/100 → 98/100

Improvements:
- VTA exponential decay: +1 point
- TAN pause mechanism: +2 points
- Multiplicative STDP: +1 point
- Astrocyte gap junctions: +1 point
- Neurogenesis integration: +1 point
```

### 9B. Hinton Reanalysis
```
Expected Score: 6.5/10 → 8.0/10

Improvements:
- Retrieval feedback loop: +0.5 points
- Embedding adaptation: +0.5 points
- Emergent poses: +0.5 points
- VAE generative replay: +0.5 points
- FF-Capsule integration: +0.5 points
```

### 9C. Architecture Reanalysis
```
Expected Score: 7.7/10 → 9.0/10

Improvements:
- Episodic decomposition: +0.5 points
- Router refactoring: +0.3 points
- Redis caching: +0.3 points
- Rate limiting: +0.2 points
- Print → logger: +0.2 points
- Test coverage 85%: +0.3 points
```

### 9D. Final Report
Generate comprehensive comparison:
- Before/after scores by component
- Performance benchmark results
- Test coverage delta
- Remaining gaps for future work

---

## Resource Requirements

### Parallel Execution Map

```
Week    | CompBio        | Hinton           | Architecture
--------|----------------|------------------|------------------
1-2     | VTA+TAN        | Feedback Loop    | Episodic Split
3-4     | STDP+Astrocyte | Embedding Adapt  | Router Refactor
5-6     | (testing)      | (testing)        | Caching+RateLimit
7-8     | Neurogenesis   | Emergent Poses   | (support)
9-10    | (support)      | VAE Replay       | (support)
11      | Integration    | FF-Capsule       | Integration
12      | Documentation  | Documentation    | Quality Sprint
13      | Testing        | Testing          | Testing
14      | Reanalysis     | Reanalysis       | Reanalysis
```

### Effort Summary

| Agent | Total Hours | Weeks |
|-------|-------------|-------|
| CompBio | 84h | 2-3 |
| Hinton | 224h | 5-6 |
| Architecture | 144h | 3-4 |
| **Total** | **452h** | **~12 weeks parallel** |

### Risk Mitigation

1. **Test Breakage**: Facade pattern ensures API stability
2. **Performance Regression**: Benchmarks at each phase gate
3. **Learning Instability**: Feature flags for new learning systems
4. **Integration Conflicts**: Dedicated integration week (Phase 6)

---

## Success Criteria

### Quantitative
- [ ] CompBio score: 98/100 (+6)
- [ ] Hinton score: 8.0/10 (+1.5)
- [ ] Architecture score: 9.0/10 (+1.3)
- [ ] Test coverage: 85%+
- [ ] All 8,000+ tests passing
- [ ] Retrieval MRR: +15%
- [ ] Cached latency: <10ms

### Qualitative
- [ ] System actually learns from retrieval outcomes
- [ ] Poses emerge from data, not hand-set
- [ ] Consolidation uses generative replay
- [ ] No god objects >500 lines
- [ ] All print() converted to logger

---

## Appendix: File Index

### New Files to Create (27)
```
src/ww/memory/episodic_storage.py
src/ww/memory/episodic_retrieval.py
src/ww/memory/episodic_learning.py
src/ww/memory/episodic_fusion.py
src/ww/memory/episodic_saga.py
src/ww/learning/retrieval_feedback.py
src/ww/learning/feedback_signals.py
src/ww/learning/vae_training.py
src/ww/learning/unified_signals.py
src/ww/encoding/online_adapter.py
src/ww/encoding/adapter_training.py
src/ww/encoding/neurogenesis.py
src/ww/nca/pose_learner.py
src/ww/bridges/ff_capsule_bridge.py
src/ww/core/cache.py
src/ww/core/cache_config.py
src/ww/api/middleware/rate_limit.py
tests/learning/test_retrieval_feedback.py
tests/learning/test_vae_training.py
tests/encoding/test_online_adapter.py
tests/encoding/test_neurogenesis.py
tests/nca/test_pose_learner.py
tests/nca/test_vta_decay.py
tests/nca/test_tan_pause.py
tests/bridges/test_ff_capsule_bridge.py
tests/api/test_rate_limiting.py
docs/concepts/neurogenesis.md
docs/concepts/emergent-poses.md
docs/concepts/learning-loop.md
```

### Files to Modify (23)
```
src/ww/memory/episodic.py
src/ww/nca/vta.py
src/ww/nca/striatal_msn.py
src/ww/nca/dopamine_integration.py
src/ww/nca/astrocyte.py
src/ww/nca/capsules.py
src/ww/nca/pose.py
src/ww/nca/forward_forward.py
src/ww/learning/stdp.py
src/ww/learning/generative_replay.py
src/ww/encoding/ff_encoder.py
src/ww/encoding/adapter.py
src/ww/consolidation/sleep.py
src/ww/storage/qdrant_store.py
src/ww/api/routes/config.py
src/ww/api/server.py
docs/architecture.md
docs/science/biology-audit.md
docs/reference/api.md
docs/LEARNING_FLOW.md
README.md
+ 47 files for print→logger conversion
```

---

**Plan Status**: READY FOR REVIEW
**Next Step**: User approval, then begin Phase 1
