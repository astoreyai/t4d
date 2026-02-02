# T4DM Phase 6+ Implementation Plan

## NEXUS: Neuro-symbolic Experience Xtraction with Unified Scoring

*Synthesized from: Hinton Architecture Review, CompBio Analysis, Literature Review, Gap Analysis*

---

## Current State Assessment

| Metric | Status |
|--------|--------|
| Test Coverage | 64% (1527 tests) |
| Learning Module | 87-100% coverage |
| E2E Pipeline | All 6 tests passing |
| Gap Coverage | 3.5/6 gaps addressed |

### What's Working
- Tripartite architecture (episodic/semantic/procedural)
- Hebbian co-retrieval strengthening
- FSRS power-law forgetting
- ListMLE ranking loss
- Prioritized experience replay
- HDBSCAN clustering for consolidation

### Critical Gaps (from Research)
1. **Fixed fusion weights (60/40)** - Should be learned per-query
2. **Wall-clock TD-λ decay** - Should be event-indexed
3. **No LTD** - Only strengthening, no weakening
4. **No pattern separation** - Similar memories interfere
5. **No sleep replay** - Manual consolidation only
6. **No working memory** - Everything persists immediately
7. **No reconsolidation** - Embeddings frozen after creation

---

## Phase 6: Critical Fixes (Hinton Priority)

### 6.1 Learned Fusion Weights
**File**: `src/t4dm/learning/neuro_symbolic.py:548-587`
**Issue**: Fixed 60/40 neural/symbolic split contradicts learning goals

```python
class LearnedFusion(nn.Module):
    def __init__(self, embed_dim: int = 1024, n_components: int = 4):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_components),
            nn.Softmax(dim=-1)
        )

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Returns learned weights for [neural, symbolic, recency, outcome]"""
        return self.weight_net(query_embedding)
```

### 6.2 Event-Indexed TD-λ Decay
**File**: `src/t4dm/learning/collector.py:470-510`
**Issue**: Traces decay by wall-clock time, not learning events

```python
def decay_traces(self, event_index: int) -> int:
    """Apply TD-λ decay per event transition, not time."""
    for memory_id, trace in self.traces.items():
        # Decay based on events since last retrieval
        events_elapsed = event_index - trace.last_event
        decay_factor = (gamma * lambda_) ** events_elapsed
        trace.value *= decay_factor
```

### 6.3 Memory Reconsolidation
**New File**: `src/t4dm/learning/reconsolidation.py`
**Issue**: Embeddings frozen after creation - should update on retrieval outcomes

```python
def reconsolidate(
    memory_embedding: np.ndarray,
    query_embedding: np.ndarray,
    outcome_score: float,
    learning_rate: float = 0.01
) -> np.ndarray:
    """Update embedding based on retrieval outcome."""
    advantage = outcome_score - 0.5  # Centered
    direction = query_embedding - memory_embedding
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    update = learning_rate * advantage * direction
    new_embedding = memory_embedding + update
    return new_embedding / np.linalg.norm(new_embedding)
```

---

## Phase 7: Biological Mechanisms (CompBio Priority)

### 7.1 Long-Term Depression (LTD)
**File**: `src/t4dm/memory/semantic.py`
**Issue**: Only LTP implemented - network weights can only grow

```python
async def apply_ltd(self, activated: list[Entity], ltd_rate: float = 0.05):
    """Weaken connections not recently co-activated."""
    activated_ids = {e.id for e in activated}

    for entity in activated:
        neighbors = await self.graph_store.get_relationships(str(entity.id))
        for rel in neighbors:
            if rel["other_id"] not in activated_ids:
                # LTD: competitive weakening
                current = rel["properties"]["weight"]
                new_weight = max(0.01, current - ltd_rate * current)
                await self.graph_store.update_relationship_weight(...)
```

### 7.2 Homeostatic Synaptic Scaling
**Issue**: Hub nodes can accumulate unbounded activation

```python
async def apply_homeostatic_scaling(self, target_total: float = 10.0):
    """Global scaling to maintain network stability."""
    for entity in await self.get_all_entities():
        rels = await self.graph_store.get_relationships(str(entity.id), "out")
        total = sum(r["properties"]["weight"] for r in rels)

        if total > target_total * 1.2:
            scale = target_total / total
            for rel in rels:
                new_weight = rel["properties"]["weight"] * scale
                await self.graph_store.update_relationship_weight(...)
```

### 7.3 Pattern Separation (Dentate Gyrus)
**New File**: `src/t4dm/memory/pattern_separation.py`
**Issue**: Similar inputs have similar embeddings, causing interference

```python
class DentateGyrus:
    """Orthogonalize similar inputs to reduce interference."""

    async def encode(self, content: str) -> np.ndarray:
        base_emb = await self.embedding.embed_query(content)

        # Find similar recent episodes
        similar = await self.vector_store.search(
            vector=base_emb, limit=10, score_threshold=0.75
        )

        if similar:
            # Random projection to separate
            separation = self._compute_separation(base_emb, similar)
            return self._orthogonalize(base_emb, separation)

        return base_emb
```

### 7.4 Sleep-Based Consolidation
**New File**: `src/t4dm/consolidation/sleep.py`
**Issue**: Consolidation is manual/scheduled, not replay-based

```python
class SleepConsolidation:
    """Simulate sleep-based memory consolidation with replay."""

    async def nrem_phase(self, session_id: str):
        """Slow-wave sleep: replay high-value experiences."""
        recent = await self.episodic.get_recent(hours=24)

        # Prioritize by value = outcome + importance + recency
        prioritized = sorted(
            recent,
            key=lambda r: 0.4*r.outcome + 0.3*r.importance + 0.3*r.recency,
            reverse=True
        )[:100]

        for episode in prioritized:
            await self._replay_to_semantic(episode)

    async def rem_phase(self, session_id: str):
        """REM phase: creative integration and abstraction."""
        clusters = await self.semantic.cluster_recent()
        for cluster in clusters:
            await self._create_abstract_concept(cluster)

    async def prune_phase(self):
        """Synaptic downscaling: remove weak connections."""
        await self.apply_homeostatic_scaling()
        await self.prune_weak_connections(threshold=0.05)
```

### 7.5 Working Memory Buffer
**New File**: `src/t4dm/memory/working_memory.py`
**Issue**: All memories persist immediately - no transient buffer

```python
class WorkingMemory:
    """Transient buffer for active context (3-4 items)."""

    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.buffer: list[WorkingMemoryItem] = []
        self.attention_weights: list[float] = []

    async def load(self, item: Any, priority: float = 0.5):
        """Load with priority-based eviction."""
        if len(self.buffer) >= self.capacity:
            # Evict lowest priority
            min_idx = np.argmin(self.attention_weights)
            evicted = self.buffer.pop(min_idx)
            # Consolidate evicted to episodic
            await self.episodic.create(content=evicted.content, ...)

        self.buffer.append(WorkingMemoryItem(content=item, priority=priority))
        self.attention_weights.append(priority)
```

---

## Phase 8: Unified Credit Assignment

### 8.1 Cross-Memory Credit Signal
**Issue**: Feedback siloed by memory type - no global learning signal

```python
class UnifiedCreditAssignment:
    """Propagate credit across memory types."""

    def assign_credit(self, outcome: Outcome, trajectory: list[Memory]):
        # Calculate mutual information between memories and outcome
        for memory in trajectory:
            mi = self.mutual_information(memory, outcome)
            self.update_memory_weight(memory, mi)

        # Cross-memory propagation
        for i, mem_i in enumerate(trajectory):
            for mem_j in trajectory[i+1:]:
                if mem_i.type != mem_j.type:
                    self.create_cross_link(mem_i, mem_j, outcome.value)
```

### 8.2 Counterfactual Credit Assignment
**Issue**: Can't determine "what would outcome be without this memory?"

```python
async def counterfactual_credit(
    self, memories: list[Memory], outcome: float
) -> dict[str, float]:
    """Estimate credit via counterfactual reasoning."""
    credits = {}

    for memory in memories:
        # Estimate outcome without this memory
        others = [m for m in memories if m.id != memory.id]
        counterfactual = await self._predict_outcome(others)

        # Credit = actual - counterfactual
        credits[memory.id] = outcome - counterfactual

    return credits
```

---

## Phase 9: 3D Visualization (React Three Fiber)

### Architecture
```
React Three Fiber (R3F)     <- Rendering
D3-force-3d                 <- Physics (Barnes-Hut)
Jotai                       <- State management
WebGL                       <- GPU acceleration
```

### Components
- **Nodes**: Episodic (blue), Semantic (green), Procedural (orange)
- **Edges**: Thickness = Hebbian weight
- **Effects**: Glow for activity, opacity for retrievability
- **Controls**: Rotate, zoom, pan, select, prune

### API Endpoints
```
GET  /api/v1/viz/graph       <- Full graph with 3D positions
GET  /api/v1/viz/activity    <- Recent activity metrics
POST /api/v1/viz/export      <- Export PNG/SVG/GLTF
WS   /api/v1/viz/stream      <- Real-time updates
```

---

## Phase 10: NEXUS Branding & Documentation

### Acronym
**NEXUS**: Neuro-symbolic Experience Xtraction with Unified Scoring

### Tagline
"Growing intelligence through experience, not just storing it"

### Documentation
1. Architecture overview with Mermaid diagrams
2. API reference with examples
3. Integration guides (LangChain, LlamaIndex migration)
4. Research paper: "First Production Tripartite Memory with Hebbian Learning"

---

## Implementation Sequence

| Phase | Focus | Duration | Priority |
|-------|-------|----------|----------|
| 6.1 | Learned Fusion Weights | 2 days | CRITICAL |
| 6.2 | Event-Indexed TD-λ | 1 day | CRITICAL |
| 6.3 | Reconsolidation | 2 days | CRITICAL |
| 7.1 | LTD Implementation | 1 day | HIGH |
| 7.2 | Homeostatic Scaling | 1 day | HIGH |
| 7.3 | Pattern Separation | 3 days | MEDIUM |
| 7.4 | Sleep Consolidation | 3 days | MEDIUM |
| 7.5 | Working Memory | 2 days | MEDIUM |
| 8.x | Unified Credit | 3 days | MEDIUM |
| 9.x | 3D Visualization | 2 weeks | LOW |
| 10.x | Branding/Docs | 3 days | LOW |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Test Coverage | 64% | 80% |
| Gaps Addressed | 3.5/6 | 5.5/6 |
| Learning Module | 87-100% | 95-100% |
| Retrieval Quality | Baseline | +15% MRR |
| Consolidation Automation | Manual | Scheduled |

---

*Generated: 2025-12-06*
*Based on: Hinton Review, CompBio Analysis, Literature Review (40 papers)*
