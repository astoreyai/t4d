# T4DM: Master Implementation Plan

**Version**: 0.5.0 → 1.0.0
**Created**: 2026-01-05
**Target Completion**: Q1 2026
**Current Status**: 6,785 tests passing, 80% coverage, Hinton 7.9/10, Bio 7.5/10

---

## Executive Summary

T4DM is a biologically-inspired neural memory system implementing Hinton's Forward-Forward algorithm, capsule networks, and tripartite memory architecture. This plan outlines the path from current alpha state to production-ready v1.0.

### Current Scores
| Metric | Score | Target |
|--------|-------|--------|
| Hinton Fidelity | 7.9/10 | 9.5/10 |
| Biological Plausibility | 7.5/10 | 9.0/10 |
| Production Readiness | 70% | 95% |
| Test Coverage | 80% | 90% |

### Critical Gaps
1. **TD(λ) temporal credit assignment** - blocks advanced learning
2. **Claude Code hooks** - blocks primary use case
3. **CA1 novelty detection** - blocks biological accuracy
4. **Sleep-FF integration** - blocks consolidation quality

---

## Phase 1: Hinton Algorithm Completion (Week 1-2)

### H1: TD(λ) Temporal Credit Assignment
**Priority**: CRITICAL | **Impact**: +0.5 Hinton score
**Current**: Simple eligibility × reward multiplication
**Required**: Full TD(λ) with discount factor γ and trace decay λ

**Implementation**:
```python
# File: src/t4dm/learning/temporal_difference.py (NEW)

@dataclass
class TDConfig:
    gamma: float = 0.99      # Discount factor (bio: 0.95-0.99)
    lambda_: float = 0.9     # Trace decay (bio: 0.7-0.95)
    alpha: float = 0.01      # Learning rate

class TemporalDifferenceEngine:
    """TD(λ) implementation for temporal credit assignment."""

    def compute_td_error(
        self,
        reward: float,
        value_current: float,
        value_next: float,
        terminal: bool = False
    ) -> float:
        """δ = r + γV(s') - V(s)"""
        if terminal:
            return reward - value_current
        return reward + self.config.gamma * value_next - value_current

    def update_eligibility_traces(
        self,
        traces: Dict[str, EligibilityTrace],
        gradients: Dict[str, np.ndarray]
    ) -> Dict[str, EligibilityTrace]:
        """e(t) = γλe(t-1) + ∇V(s)"""
        for key, trace in traces.items():
            trace.value = (
                self.config.gamma * self.config.lambda_ * trace.value
                + gradients.get(key, 0)
            )
        return traces

    def apply_credit(
        self,
        td_error: float,
        traces: Dict[str, EligibilityTrace]
    ) -> Dict[str, float]:
        """Δw = α × δ × e"""
        return {
            key: self.config.alpha * td_error * trace.value
            for key, trace in traces.items()
        }
```

**Integration Points**:
- `src/t4dm/learning/dopamine.py`: Replace simple RPE with TD(λ)
- `src/t4dm/learning/eligibility.py`: Add trace decay
- `src/t4dm/learning/three_factor.py`: Use TD error as third factor

**Tests Required**:
- `tests/learning/test_td_lambda.py`: 20+ tests
- Verify γ=0 reduces to immediate reward
- Verify λ=0 reduces to TD(0)
- Verify λ=1 reduces to Monte Carlo

---

### H2: Sleep-Phase FF Negative Generation
**Priority**: HIGH | **Impact**: +0.2 Hinton score
**Current**: No offline negative generation during consolidation
**Required**: Generate negative examples during sleep replay

**Implementation**:
```python
# File: src/t4dm/consolidation/ff_sleep_integration.py (NEW)

class FFSleepIntegration:
    """Integrate Forward-Forward with sleep consolidation."""

    async def generate_sleep_negatives(
        self,
        replayed_episodes: List[Episode],
        corruption_rate: float = 0.3
    ) -> List[Tuple[Episode, np.ndarray]]:
        """Generate negative examples during SWR replay."""
        negatives = []
        for episode in replayed_episodes:
            # During sleep, we can use stronger corruption
            corrupted = self.ff_layer.generate_negative(
                episode.embedding,
                method="hybrid",  # Mix of noise + shuffle
                corruption_strength=corruption_rate
            )
            negatives.append((episode, corrupted))
        return negatives

    async def consolidation_learning_step(
        self,
        positives: List[Episode],
        negatives: List[Tuple[Episode, np.ndarray]]
    ) -> LearningMetrics:
        """Run FF learning during consolidation."""
        metrics = LearningMetrics()

        # Positive phase: strengthen real memories
        for episode in positives:
            goodness = self.ff_layer.forward(episode.embedding, training=True)
            self.ff_layer.positive_phase_update(episode.embedding, goodness)
            metrics.positive_goodness.append(goodness)

        # Negative phase: learn to reject corrupted
        for episode, corrupted in negatives:
            goodness = self.ff_layer.forward(corrupted, training=True)
            self.ff_layer.negative_phase_update(corrupted, goodness)
            metrics.negative_goodness.append(goodness)

        return metrics
```

**Integration Points**:
- `src/t4dm/consolidation/sleep.py`: Add FF phase to consolidation loop
- `src/t4dm/nca/forward_forward.py`: Add batch training methods
- `src/t4dm/nca/swr_coupling.py`: Coordinate with SWR events

---

### H3: EM Capsule Routing (Optional Enhancement)
**Priority**: MEDIUM | **Impact**: +0.3 Hinton score
**Current**: Dynamic routing by agreement only
**Required**: Expectation-Maximization routing (Hinton 2018)

**Implementation**:
```python
# File: src/t4dm/nca/em_routing.py (NEW)

class EMRoutingLayer:
    """EM routing for capsule networks (Hinton 2018)."""

    def route(
        self,
        votes: np.ndarray,  # [batch, input_caps, output_caps, pose_dim]
        num_iterations: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Route using EM algorithm."""
        # E-step: compute assignment probabilities
        # M-step: update pose parameters

        activations = np.ones(self.num_output_caps) / self.num_output_caps
        poses = np.zeros((self.num_output_caps, self.pose_dim))

        for _ in range(num_iterations):
            # E-step: p(c|x) ∝ a_c × N(v; μ_c, σ²)
            assignments = self._e_step(votes, poses, activations)

            # M-step: update μ, σ², a
            poses, variances, activations = self._m_step(votes, assignments)

        return poses, activations
```

---

## Phase 2: Biological Fidelity Improvements (Week 2-3)

### B1: CA1 Novelty Detection Layer
**Priority**: CRITICAL | **Impact**: +1.0 Bio score
**Current**: Missing entirely
**Required**: CA1 comparator detecting DG/CA3 mismatch

**Implementation**:
```python
# File: src/t4dm/nca/hippocampus.py (MODIFY)

class CA1Layer:
    """CA1 novelty detection via expectation mismatch."""

    def __init__(self, config: HippocampalConfig):
        self.ec_projection = np.random.randn(config.ec_dim, config.ca1_dim) * 0.1
        self.ca3_projection = np.random.randn(config.ca3_dim, config.ca1_dim) * 0.1
        self.mismatch_threshold = 0.3

    def compute_novelty(
        self,
        ec_input: np.ndarray,
        ca3_output: np.ndarray
    ) -> NoveltySignal:
        """Compute novelty as mismatch between EC expectation and CA3 completion."""
        # Project both to CA1 space
        ec_projected = ec_input @ self.ec_projection
        ca3_projected = ca3_output @ self.ca3_projection

        # Normalize
        ec_norm = ec_projected / (np.linalg.norm(ec_projected) + 1e-8)
        ca3_norm = ca3_projected / (np.linalg.norm(ca3_projected) + 1e-8)

        # Mismatch = 1 - cosine similarity
        similarity = np.dot(ec_norm, ca3_norm)
        mismatch = 1.0 - max(0.0, similarity)

        return NoveltySignal(
            novelty_score=mismatch,
            is_novel=mismatch > self.mismatch_threshold,
            ec_representation=ec_projected,
            ca3_representation=ca3_projected
        )

    def modulate_encoding(
        self,
        novelty: NoveltySignal,
        dopamine_level: float
    ) -> float:
        """Novel items + high DA = strong encoding signal."""
        return novelty.novelty_score * dopamine_level
```

**Integration Points**:
- `src/t4dm/nca/hippocampus.py`: Add CA1Layer to HippocampalCircuit
- `src/t4dm/memory/episodic.py`: Use novelty for encoding decisions
- `src/t4dm/learning/dopamine.py`: Connect to DA release

---

### B2: STDP Implementation
**Priority**: HIGH | **Impact**: +0.5 Bio score
**Current**: Hebbian approximation (co-activation)
**Required**: Spike-timing dependent plasticity

**Implementation**:
```python
# File: src/t4dm/learning/stdp.py (NEW - expand existing stub)

@dataclass
class STDPConfig:
    tau_plus: float = 20.0    # LTP time constant (ms)
    tau_minus: float = 20.0   # LTD time constant (ms)
    a_plus: float = 0.005     # LTP amplitude
    a_minus: float = 0.00525  # LTD amplitude (slightly stronger)
    w_max: float = 1.0        # Maximum weight
    w_min: float = 0.0        # Minimum weight

class STDPRule:
    """Spike-timing dependent plasticity."""

    def compute_weight_change(
        self,
        pre_spike_times: np.ndarray,
        post_spike_times: np.ndarray,
        current_weight: float
    ) -> float:
        """
        Δw based on spike timing:
        - Pre before post (Δt > 0): LTP
        - Post before pre (Δt < 0): LTD
        """
        delta_w = 0.0

        for t_pre in pre_spike_times:
            for t_post in post_spike_times:
                delta_t = t_post - t_pre  # ms

                if delta_t > 0:  # Pre before post → LTP
                    delta_w += self.config.a_plus * np.exp(-delta_t / self.config.tau_plus)
                else:  # Post before pre → LTD
                    delta_w -= self.config.a_minus * np.exp(delta_t / self.config.tau_minus)

        # Apply weight bounds
        new_weight = np.clip(
            current_weight + delta_w,
            self.config.w_min,
            self.config.w_max
        )
        return new_weight
```

---

### B3: Grid Cell Spatial Coding
**Priority**: MEDIUM | **Impact**: +0.3 Bio score
**Current**: Implicit in embeddings
**Required**: Hexagonal grid cell implementation for EC

**Implementation**:
```python
# File: src/t4dm/nca/grid_cells.py (NEW)

class GridCellModule:
    """Entorhinal cortex grid cell simulation."""

    def __init__(
        self,
        num_cells: int = 100,
        grid_scales: List[float] = [0.5, 0.7, 1.0, 1.4, 2.0],
        orientations: List[float] = None
    ):
        self.num_cells = num_cells
        self.grid_scales = grid_scales
        self.orientations = orientations or [0, 15, 30, 45, 60]
        self._initialize_grid_fields()

    def _initialize_grid_fields(self):
        """Initialize hexagonal grid fields."""
        self.fields = []
        for scale in self.grid_scales:
            for orientation in self.orientations:
                field = HexagonalGridField(
                    scale=scale,
                    orientation=np.radians(orientation)
                )
                self.fields.append(field)

    def encode_position(
        self,
        x: float,
        y: float
    ) -> np.ndarray:
        """Encode 2D position as grid cell population code."""
        activations = np.zeros(len(self.fields))
        for i, field in enumerate(self.fields):
            activations[i] = field.compute_activation(x, y)
        return activations

    def encode_semantic_position(
        self,
        embedding: np.ndarray
    ) -> np.ndarray:
        """Map high-dim embedding to grid cell code via UMAP projection."""
        # Project to 2D conceptual space
        x, y = self._project_to_2d(embedding)
        return self.encode_position(x, y)
```

---

## Phase 3: Integration & Production Readiness (Week 3-4)

### I1: Claude Code SessionStart/SessionEnd Hooks
**Priority**: CRITICAL | **Impact**: Primary use case enablement

**Implementation**:
```python
# File: src/t4dm/hooks/claude_code.py (NEW)

class SessionStartHook:
    """Load context on Claude Code session start."""

    async def execute(self, context: HookContext) -> HookContext:
        session_id = context.session_id
        project_dir = context.metadata.get("project_dir", ".")

        # 1. Load recent episodic context
        episodes = await self.memory.episodic.recall(
            query=f"project:{project_dir}",
            limit=20,
            recency_weight=0.7
        )

        # 2. Spread activation to semantic entities
        entity_ids = self._extract_entity_ids(episodes)
        activated = await self.memory.semantic.spread_activation(
            source_ids=entity_ids,
            decay=0.7,
            max_depth=3
        )

        # 3. Surface relevant procedures
        skills = await self.memory.procedural.recall(
            query=f"cwd:{project_dir}",
            limit=10
        )

        # 4. Format context for injection
        context.memory_context = self._format_context(
            episodes=episodes,
            entities=activated,
            skills=skills
        )

        return context


class SessionEndHook:
    """Persist and consolidate on Claude Code session end."""

    async def execute(self, context: HookContext) -> HookContext:
        # 1. Extract entities from session
        entities = await self.extractor.extract_entities(
            context.conversation_history
        )

        # 2. Create session episode
        episode = await self.memory.episodic.create(
            content=context.session_summary,
            metadata={
                "project": context.project_dir,
                "duration": context.duration,
                "files_modified": context.files_modified
            }
        )

        # 3. Record unresolved tasks
        if context.pending_tasks:
            for task in context.pending_tasks:
                await self.memory.procedural.create_or_update(
                    name=task.name,
                    status="pending",
                    context=task.context
                )

        # 4. Trigger lightweight consolidation
        await self.consolidation.run_lightweight(
            session_id=context.session_id,
            episodes=[episode],
            entities=entities
        )

        return context
```

**Deployment**:
```json
// ~/.claude/settings.json
{
  "hooks": {
    "sessionStart": {
      "command": "python -m t4dm.hooks.claude_code session_start",
      "timeout": 5000
    },
    "sessionEnd": {
      "command": "python -m t4dm.hooks.claude_code session_end",
      "timeout": 10000
    }
  }
}
```

---

### I2: Batch Operation Endpoints
**Priority**: HIGH | **Impact**: 10x ingestion performance

**Implementation**:
```python
# File: src/t4dm/api/routes/batch.py (NEW)

router = APIRouter(prefix="/api/v1/batch", tags=["Batch Operations"])

@router.post("/episodes", response_model=BatchCreateResponse)
async def batch_create_episodes(
    episodes: List[EpisodeCreate],
    session_id: str = Header(alias="X-Session-ID")
) -> BatchCreateResponse:
    """Create multiple episodes in a single transaction."""
    async with memory.transaction() as tx:
        results = []
        for episode in episodes:
            result = await tx.episodic.create(episode)
            results.append(result)
        await tx.commit()

    return BatchCreateResponse(
        created=len(results),
        ids=[r.id for r in results],
        duration_ms=tx.duration_ms
    )

@router.post("/entities", response_model=BatchCreateResponse)
async def batch_create_entities(
    entities: List[EntityCreate],
    session_id: str = Header(alias="X-Session-ID")
) -> BatchCreateResponse:
    """Create multiple entities in a single transaction."""
    # Similar implementation
    ...
```

---

### I3: Session Persistence
**Priority**: HIGH | **Impact**: State recovery across restarts

**Implementation**:
```python
# File: src/t4dm/core/session_store.py (NEW)

class RedisSessionStore:
    """Persistent session storage with Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = aioredis.from_url(redis_url)
        self.ttl = 86400 * 7  # 7 days

    async def save_session(
        self,
        session_id: str,
        state: SessionState
    ) -> None:
        """Persist session state."""
        key = f"ww:session:{session_id}"
        await self.redis.setex(
            key,
            self.ttl,
            state.model_dump_json()
        )

    async def load_session(
        self,
        session_id: str
    ) -> Optional[SessionState]:
        """Load session state."""
        key = f"ww:session:{session_id}"
        data = await self.redis.get(key)
        if data:
            return SessionState.model_validate_json(data)
        return None
```

---

### I4: Learned Retrieval Scoring
**Priority**: HIGH | **Impact**: Better retrieval quality

**Implementation**:
```python
# File: src/t4dm/memory/learned_scorer.py (ENHANCE existing)

class LearnedRetrievalScorer:
    """Online-trainable retrieval scoring."""

    def __init__(self):
        # Feature weights (start with reasonable defaults)
        self.weights = {
            "similarity": 0.4,
            "recency": 0.2,
            "importance": 0.15,
            "access_count": 0.1,
            "context_match": 0.15
        }
        self.learning_rate = 0.01
        self.update_count = 0

    def score(
        self,
        candidate: MemoryItem,
        query_context: QueryContext
    ) -> float:
        """Compute weighted score."""
        features = self._extract_features(candidate, query_context)
        return sum(
            self.weights[k] * features[k]
            for k in self.weights
        )

    def learn_from_feedback(
        self,
        retrieved: List[MemoryItem],
        used: Set[str],
        context: QueryContext
    ) -> None:
        """Update weights based on implicit feedback."""
        for item in retrieved:
            features = self._extract_features(item, context)
            target = 1.0 if item.id in used else 0.0
            prediction = self.score(item, context)
            error = target - prediction

            # Gradient descent update
            for k in self.weights:
                self.weights[k] += self.learning_rate * error * features[k]

        self.update_count += 1
```

---

## Phase 4: Codebase Cleanup (Week 4)

### C1: Module Consolidation
**Problem**: Duplicate modules (`bridge` vs `bridges`, `integration` vs `integrations`)

**Actions**:
1. Merge `src/t4dm/bridge/` into `src/t4dm/bridges/`
2. Merge `src/t4dm/integration/` into `src/t4dm/integrations/`
3. Update all imports
4. Add deprecation warnings for old paths

```bash
# Migration script
git mv src/t4dm/bridge/memory_nca.py src/t4dm/bridges/memory_nca_bridge.py
git mv src/t4dm/integration/*.py src/t4dm/integrations/ccapi/
```

---

### C2: Archive Cleanup
**Problem**: 106 archive docs, many outdated

**Actions**:
1. Review `docs/archive/` - move truly obsolete to `docs/.archive/`
2. Consolidate similar docs (5 TEST_*.md → TEST_SUMMARY.md)
3. Update ROADMAP.md to reference current state only
4. Remove completed TODO sections from docs

---

### C3: Test Consolidation
**Problem**: Some test files duplicated or overlap

**Actions**:
1. Merge `tests/biology/` into `tests/nca/` (same components)
2. Rename confusing test files
3. Add missing `__init__.py` files
4. Update pytest.ini markers

---

### C4: Dead Code Removal
**Problem**: Unused imports, stub methods

**Actions**:
1. Run `ruff check --select F401` for unused imports
2. Remove `NotImplementedError` stubs without implementation
3. Remove deprecated compatibility shims
4. Clean up `# TODO: remove` comments

---

## Phase 5: Test Completion (Ongoing)

### T1: Missing Module Tests
| Module | Current | Target | Gap |
|--------|---------|--------|-----|
| `nca/capsules.py` | 12 tests | 25 tests | +13 |
| `nca/forward_forward.py` | 8 tests | 20 tests | +12 |
| `learning/td_lambda.py` | 0 tests | 20 tests | +20 (new) |
| `hooks/claude_code.py` | 0 tests | 15 tests | +15 (new) |
| `bridges/ff_encoding.py` | 5 tests | 15 tests | +10 |

### T2: Integration Tests
- Add Claude Code hook integration tests
- Add batch operation stress tests
- Add session persistence tests
- Add failover scenario tests

### T3: Performance Benchmarks
- Add 100K episode retrieval benchmark
- Add concurrent session benchmark
- Add consolidation throughput benchmark

---

## Parallel Execution Plan

### Agent Assignments

| Agent Type | Tasks | Week |
|------------|-------|------|
| `ww-hinton` | H1: TD(λ), H2: Sleep-FF, H3: EM Routing | 1-2 |
| `ww-compbio` | B1: CA1 Layer, B2: STDP, B3: Grid Cells | 2-3 |
| `kymera-go-backend` | I1: Claude Hooks, I2: Batch Ops, I3: Session | 3-4 |
| `ww-validator` | T1-T3: All test completion | 1-4 |
| `boris` | C1-C4: Codebase cleanup | 4 |

### Parallel Execution Commands

```bash
# Week 1-2: Hinton + Tests in parallel
claude --agent ww-hinton "Implement TD(λ) in src/t4dm/learning/temporal_difference.py"
claude --agent ww-validator "Add tests for TD(λ) implementation"

# Week 2-3: Biology + Integration in parallel
claude --agent ww-compbio "Implement CA1 novelty layer in hippocampus.py"
claude --agent kymera-go-backend "Implement Claude Code hooks"

# Week 4: Cleanup
claude --agent boris "Analyze and clean up codebase"
```

---

## Success Criteria

### v0.5.0 (End of Week 2)
- [ ] TD(λ) implemented with tests
- [ ] Sleep-FF integration working
- [ ] 6,900+ tests passing

### v0.6.0 (End of Week 3)
- [ ] CA1 novelty detection integrated
- [ ] Claude Code hooks deployed
- [ ] Batch operations available

### v0.7.0 (End of Week 4)
- [ ] STDP implementation complete
- [ ] Session persistence working
- [ ] Codebase cleaned up
- [ ] 90%+ test coverage

### v1.0.0 (Q1 2026)
- [ ] Hinton Fidelity: 9.5/10
- [ ] Biological Plausibility: 9.0/10
- [ ] Production Readiness: 95%
- [ ] Documentation complete
- [ ] 5+ production deployments

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| TD(λ) breaks existing learning | Medium | High | Feature flag, A/B test |
| CA1 integration causes regressions | Low | High | Comprehensive test suite |
| Claude hooks add latency | Medium | Medium | Async execution, caching |
| Module consolidation breaks imports | Low | High | Deprecation warnings first |

---

## Appendix: File Locations

### New Files to Create
- `src/t4dm/learning/temporal_difference.py`
- `src/t4dm/consolidation/ff_sleep_integration.py`
- `src/t4dm/nca/em_routing.py`
- `src/t4dm/nca/grid_cells.py`
- `src/t4dm/hooks/claude_code.py`
- `src/t4dm/api/routes/batch.py`
- `src/t4dm/core/session_store.py`

### Files to Modify
- `src/t4dm/nca/hippocampus.py` - Add CA1Layer
- `src/t4dm/learning/stdp.py` - Expand stub
- `src/t4dm/learning/dopamine.py` - Integrate TD(λ)
- `src/t4dm/consolidation/sleep.py` - Add FF phase
- `src/t4dm/memory/learned_scorer.py` - Enhance

### Files to Delete/Merge
- `src/t4dm/bridge/` → merge into `src/t4dm/bridges/`
- `src/t4dm/integration/` → merge into `src/t4dm/integrations/`
- Outdated archive docs

---

**Document Status**: APPROVED FOR IMPLEMENTATION
**Next Action**: Begin Phase 1 with TD(λ) implementation
**Owner**: Aaron (with agent assistance)
