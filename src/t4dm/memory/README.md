# Memory Module

**Path**: `t4dm/memory/` | **Files**: 13 | **Lines**: ~10,870

Tripartite memory system implementing episodic, semantic, and procedural memory with biologically-inspired algorithms.

---

## Quick Start

```python
from ww.memory import EpisodicMemory, SemanticMemory, ProceduralMemory, UnifiedMemoryService

# Individual memory systems
episodic = EpisodicMemory(session_id="my-session")
await episodic.initialize()
episode_id = await episodic.create(content="User asked about Python")

# Unified search across all memory types
unified = UnifiedMemoryService(episodic, semantic, procedural)
results = await unified.search("Python programming", limit=10)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    UnifiedMemoryService                          │
│              (Parallel search + merge across types)              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Episodic    │   │   Semantic    │   │  Procedural   │
│    (3129 L)   │   │   (1115 L)    │   │    (973 L)    │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ ClusterIndex  │   │ ACT-R Activ.  │   │ SkillExecute  │
│ SparseIndex   │   │ Hebbian Learn │   │ SuccessTrack  │
│ PatternSep    │   │ GraphTraverse │   │ Deprecation   │
│ BufferManager │   │ LRU Cache     │   │ Version Ctrl  │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
              ┌─────────────────────────────┐
              │  Storage Layer (Qdrant+Neo4j) │
              └─────────────────────────────┘
```

---

## File Structure

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| `episodic.py` | 3,129 | Autobiographical events | `EpisodicMemory`, `LearnedFusionWeights`, `LearnedReranker` |
| `semantic.py` | 1,115 | Knowledge graph | `SemanticMemory` |
| `procedural.py` | 973 | Skills/procedures | `ProceduralMemory` |
| `pattern_separation.py` | 1,212 | DG orthogonalization | `DentateGyrus`, `PatternCompletion`, Modern Hopfield |
| `working_memory.py` | 639 | Capacity-limited buffer | `WorkingMemory[T]`, `AttentionalBlink` |
| `forgetting.py` | 556 | Active forgetting | `ActiveForgettingSystem`, `RetentionPolicy` |
| `cluster_index.py` | 661 | Hierarchical retrieval | `ClusterIndex`, `ClusterMeta` |
| `learned_sparse_index.py` | 474 | Query-dependent addressing | `LearnedSparseIndex` |
| `fast_episodic.py` | 458 | Rapid encoding (CA3) | `FastEpisodicStore`, `FESEntry` |
| `feature_aligner.py` | 410 | Gate-retrieval alignment | `FeatureAligner`, `JointLossWeights` |
| `buffer_manager.py` | 728 | Evidence accumulation | `BufferManager`, `PromotionDecision` |
| `unified.py` | 442 | Cross-memory search | `UnifiedMemoryService` |

---

## Memory Types

### 1. Episodic Memory

Autobiographical events with temporal and contextual grounding.

```python
# Create episode
episode_id = await episodic.create(
    content="User fixed authentication bug",
    project="my-project",
    outcome="success",
    valence=0.8,  # Emotional importance
)

# Recall with learned fusion weights
results = await episodic.recall(
    query="authentication issues",
    limit=10,
    similarity_threshold=0.5,
)
```

**Scoring Components** (learned weights):
- Semantic similarity (vector distance)
- Recency (temporal decay)
- Outcome history (success/failure)
- Importance (emotional valence)

**Learning**: `LearnedFusionWeights` (2-layer MLP) + `LearnedReranker` adapt online from retrieval outcomes.

### 2. Semantic Memory

Knowledge graph with Hebbian-weighted relationships.

```python
# Create entity
entity_id = await semantic.create_entity(
    name="FastAPI",
    entity_type="TECHNOLOGY",
    summary="Modern Python web framework",
)

# Create relationship
await semantic.relate(
    source_id=entity_id,
    target_id=other_id,
    relation_type="USES",
    weight=0.5,
)

# ACT-R spreading activation
results = await semantic.recall(
    query="web frameworks",
    spread_depth=2,  # Activation spreading hops
)
```

**Algorithms**:
- **Hebbian Learning**: Relationship weights increase with co-occurrence
- **ACT-R Activation**: Base-level + spreading activation with decay

### 3. Procedural Memory

Learned skills with execution tracking.

```python
# Create skill
skill_id = await procedural.create_skill(
    name="Deploy to Production",
    domain="DEVOPS",
    task="Deploy application to production server",
    steps=[
        {"action": "run_tests", "tool": "pytest"},
        {"action": "build_image", "tool": "docker"},
        {"action": "deploy", "tool": "kubectl"},
    ],
)

# Record execution
await procedural.execute(skill_id, success=True, duration=120)

# Natural language lookup
result = await procedural.how_to("deploy my app")
```

---

## Key Algorithms

### Pattern Separation (Dentate Gyrus)

Prevents interference between similar memories.

```python
from ww.memory import DentateGyrus

dg = DentateGyrus(target_sparsity=0.01)  # 1% active (biologically accurate)
orthogonalized = dg.orthogonalize(new_embedding, recent_embeddings)
```

**How it works**:
1. Find interfering memories (high similarity)
2. Compute average interference direction
3. Project new embedding away from interference
4. Normalize to unit sphere

### Modern Hopfield Networks

Exponential capacity pattern completion.

```python
from ww.memory.pattern_separation import modern_hopfield_update

# Continuous Hopfield (Ramsauer 2020)
completed = modern_hopfield_update(
    query=partial_pattern,
    memories=stored_patterns,
    beta=10.0,  # Sharpness (higher = more exact)
)
```

**Capacity**: O(d^(n-1)) vs classical O(d)

### Hierarchical Retrieval (ClusterIndex)

Two-stage O(log n) retrieval.

```python
from ww.memory import ClusterIndex

index = ClusterIndex()
clusters = index.select_clusters(query_embedding, k=5)
# Then search within selected clusters only
```

**Speedup**: ~67x for 100K episodes with K=500 clusters

### Learned Sparse Addressing

Query-dependent feature attention.

```
Query [1024-dim]
    │
    ▼ Shared MLP [hidden=256]
    ├─→ Cluster Head → softmax → cluster attention [K]
    ├─→ Feature Head → sigmoid → feature attention [d]
    └─→ Sparsity Gate → sigmoid → sparsity level [1]
```

---

## Working Memory

Capacity-limited buffer (~4 items, Cowan's number).

```python
from ww.memory import WorkingMemory

wm = WorkingMemory[str](capacity=4)
wm.load("important-item", priority=0.9)

# Rehearsal prevents decay
wm.rehearse("important-item")

# Consolidate to episodic
await wm.consolidate_to(episodic)
```

**States**: ACTIVE → DECAYING → EVICTED → CONSOLIDATED

---

## Active Forgetting

Bounded memory growth with multi-strategy forgetting.

```python
from ww.memory import ActiveForgettingSystem, RetentionPolicy

policy = RetentionPolicy(
    max_episodes=100_000,
    soft_limit_ratio=0.8,  # Start forgetting at 80%
    access_half_life_days=30,
)

forgetting = ActiveForgettingSystem(policy)
candidates = await forgetting.identify_candidates()
await forgetting.run_forgetting_cycle()
```

**Strategies**:
- DECAY: Time-based
- INTERFERENCE: Similar memory competition
- VALUE: Importance thresholds
- HYBRID: All combined

---

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Episode Create | O(log n) | Vector + graph insertion |
| Episode Recall (flat) | O(n) | Full k-NN search |
| Episode Recall (hierarchical) | O(K + k*n/K) | ~67x faster |
| Entity Relation | O(1) | LRU cache hit |
| Hopfield Update | O(d*p) | d=dim, p=patterns |
| Working Memory Evict | O(c log c) | c=capacity (~4) |

---

## Configuration

Via `ww.core.config`:

```python
# Episodic
episodic_vector_weight=0.4
episodic_recency_weight=0.3
episodic_outcome_weight=0.2
episodic_importance_weight=0.1

# Semantic (ACT-R)
actr_decay=0.5
actr_threshold=-3.0
actr_noise=0.25

# Hebbian
hebbian_learning_rate=0.1
hebbian_decay_rate=0.01
hebbian_stale_days=90
```

---

## Dependencies

**Internal**:
- `ww.storage` - Qdrant + Neo4j backends
- `ww.embedding` - BGE-M3 provider
- `ww.learning` - Neuromodulators, reconsolidation
- `ww.consolidation` - Sleep consolidation

**External**:
- numpy, PyTorch (learning components)
- asyncio (concurrent operations)

---

## Biological Inspirations

| Mechanism | Biological Basis | Implementation |
|-----------|-----------------|----------------|
| Pattern Separation | Dentate Gyrus | `DentateGyrus` class |
| Pattern Completion | CA3 autoassociation | Modern Hopfield |
| Working Memory | Prefrontal cortex | Capacity-limited buffer |
| Forgetting | Active forgetting | Multi-strategy system |
| ACT-R Activation | Cognitive architecture | Spreading activation |
| FSRS Decay | Spaced repetition | Retrievability calculation |
