# Memory Store and Recall Flow

**Version**: 0.1.0
**Last Updated**: 2025-12-09

This document traces the complete data flow for memory storage and retrieval operations.

---

## Table of Contents

1. [Memory Store Flow](#memory-store-flow)
2. [Memory Recall Flow](#memory-recall-flow)
3. [Files Touched](#files-touched)
4. [Key Data Structures](#key-data-structures)

---

## Memory Store Flow

### Overview

```
User Input → MCP/API → Embed → Separate → Neuromod → Gate → Saga → Store
```

### Step-by-Step Trace

#### 1. Entry Point

**File**: `mcp/tools/episodic.py:create_episode()`

```python
@mcp_app.tool()
@rate_limited
@with_request_id
@traced("mcp.create_episode")
async def create_episode(
    content: str,
    context: Optional[dict] = None,
    outcome: str = "neutral",
    valence: float = 0.5,
    auto_extract: bool = True,
) -> dict:
```

**Actions**:
- Rate limiting check
- Input validation (content, outcome, valence)
- Get services: episodic, semantic, procedural

---

#### 2. Embedding Generation

**File**: `embedding/bge_m3.py:embed_query()`

```python
embedding = await self.embedding.embed_query(content)
# Returns: 1024-dimensional dense vector
```

**If hybrid enabled**:
```python
dense_vecs, sparse_vecs = await self.embedding.embed_hybrid([content])
```

---

#### 3. Pattern Separation (DG-Style)

**File**: `memory/pattern_separation.py:DentateGyrus.encode()`

```python
if self._pattern_separation_enabled:
    embedding = await self.pattern_separator.encode(content)
```

**Purpose**: Orthogonalize similar embeddings to reduce interference

**Mechanism**:
- Compute similarity to recent embeddings
- If similarity > threshold (0.75): Apply sparse coding
- Push embedding away from similar items

---

#### 4. Neuromodulation Processing

**File**: `learning/neuromodulators.py:process_query()`

```python
neuromod_state = self.orchestra.process_query(
    query_embedding=np.array(embedding),
    is_question=False,  # Storage = encoding mode
    explicit_importance=valence if valence != 0.5 else None
)
```

**Returns**: `NeuromodulatorState`
```python
{
    dopamine_rpe: 0.0,        # Updated on outcome
    norepinephrine_gain: 1.2, # Arousal from novelty
    acetylcholine_mode: "encoding",
    serotonin_mood: 0.6,
    inhibition_sparsity: 0.0
}
```

---

#### 5. Learned Memory Gating

**File**: `core/learned_gate.py:LearnedMemoryGate.predict()`

```python
gate_decision = self.learned_gate.predict(
    content_embedding=np.array(embedding),
    context=gate_context,
    neuromod_state=neuromod_state,
    explore=True  # Thompson sampling
)
```

**Returns**: `GateDecision`
```python
{
    action: GateAction.STORE,  # or BUFFER or SKIP
    probability: 0.78,
    exploration_boost: 0.05,
    features: {...}
}
```

**Decision Logic**:
| Probability | Action |
|-------------|--------|
| p > 0.6 | STORE |
| 0.3 < p < 0.6 | BUFFER |
| p < 0.3 | SKIP |

---

#### 6. Episode Object Creation

**File**: `memory/episodic.py`

```python
episode = Episode(
    id=uuid4(),
    session_id=self.session_id,
    content=content,
    embedding=embedding,
    context=EpisodeContext(**context),
    outcome=Outcome(outcome),
    emotional_valence=valence,
    stability=self.default_stability,
    timestamp=datetime.now(),
)
```

---

#### 7. Saga Execution (Dual-Store)

**File**: `storage/saga.py:Saga.execute()`

```python
saga = Saga(f"create_episode_{episode.id}")

# Step 1: Vector store
saga.add_step(
    name="add_vector",
    action=lambda: self.vector_store.add(
        collection=self.vector_store.episodes_collection,
        ids=[str(episode.id)],
        vectors=[embedding],
        payloads=[self._to_payload(episode)],
    ),
    compensate=lambda: self.vector_store.delete(...)
)

# Step 2: Graph store
saga.add_step(
    name="create_node",
    action=lambda: self.graph_store.create_node(
        label="Episode",
        properties=self._to_graph_props(episode),
    ),
    compensate=lambda: self.graph_store.delete_node(...)
)

result = await saga.execute()
```

**Atomicity**: If Step 2 fails, Step 1's compensate() is called (rollback)

---

#### 8. Qdrant Storage

**File**: `storage/qdrant_store.py:add()`

```python
await client.upsert(
    collection_name=collection,
    points=[
        models.PointStruct(
            id=episode_id,
            vector=embedding,
            payload={
                "content": episode.content,
                "session_id": episode.session_id,
                "timestamp": episode.timestamp.isoformat(),
                "outcome": episode.outcome.value,
                "emotional_valence": episode.emotional_valence,
                ...
            }
        )
    ]
)
```

---

#### 9. Neo4j Storage

**File**: `storage/neo4j_store.py:create_node()`

```cypher
CREATE (e:Episode {
    episode_id: $id,
    sessionId: $session_id,
    content: $content,
    timestamp: $timestamp,
    outcome: $outcome,
    valence: $valence
})
```

---

#### 10. Post-Storage Processing

**Pattern Completion Attractor**:
```python
self.pattern_completion.add_attractor(np.array(embedding))
```

**Register Gate Label** (for later training):
```python
self.learned_gate.register_pending(
    episode.id,
    gate_decision.features,
    raw_content_embedding=embedding,
    neuromod_state=gate_decision.neuromod_state
)
```

**Entity Extraction** (if enabled):
```python
extractor = create_default_extractor()
extracted = await extractor.extract(content)
for entity in extracted:
    await semantic.create_entity(name=entity.name, ...)
    await semantic.create_relationship(episode.id, entity.id, ...)
```

---

## Memory Recall Flow

### Overview

```
Query → Embed → Neuromod → Complete → Search → Score → Rerank → Inhibit → Return
```

### Step-by-Step Trace

#### 1. Entry Point

**File**: `mcp/tools/episodic.py:recall_episodes()`

```python
@mcp_app.tool()
@rate_limited
@with_session_validation
@traced("mcp.recall_episodes")
async def recall_episodes(
    query: str,
    limit: int = 10,
    session_filter: Optional[str] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
) -> dict:
```

---

#### 2. Query Embedding

**File**: `embedding/bge_m3.py:embed_query()`

```python
query_emb = await self.embedding.embed_query(query)
query_emb_np = np.array(query_emb)
```

---

#### 3. Neuromodulation Processing

**File**: `learning/neuromodulators.py:process_query()`

```python
neuro_state = self.orchestra.process_query(
    query_embedding=query_emb_np,
    is_question=True,  # Retrieval mode
    explicit_importance=None
)
```

**Effects**:
- **NE**: Arousal from novelty → modulates search threshold
- **ACh**: Sets retrieval mode (prioritize recall)
- **Threshold**: High arousal = lower threshold = broader search

---

#### 4. Pattern Completion (CA3-Style)

**File**: `memory/pattern_separation.py:PatternCompletion.complete()`

```python
if use_pattern_completion:
    completed, iterations = self.pattern_completion.complete(query_arr)
    if iterations > 0:
        # Blend: 70% original + 30% completed
        query_emb = (0.7 * query_arr + 0.3 * completed).tolist()
```

**Purpose**: Complete partial/vague queries toward stored patterns

---

#### 5. Hierarchical Cluster Search

**File**: `memory/cluster_index.py` + `memory/learned_sparse_index.py`

```python
# Select top-k clusters by centroid similarity
selected_clusters = self.cluster_index.search(query_emb, k=3)

# Learned sparse addressing weights cluster attention
cluster_weights = self.sparse_index.compute_weights(query_emb)
```

**Benefit**: O(log n) instead of O(n) full scan

---

#### 6. Vector Search

**File**: `storage/qdrant_store.py:search()`

```python
results = await client.search(
    collection_name=self.episodes_collection,
    query_vector=query_emb,
    limit=limit,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_filter)
            )
        ]
    ) if session_filter else None,
    score_threshold=modulated_threshold,
)
```

---

#### 7. Multi-Component Scoring

**File**: `memory/episodic.py:_score_results()`

For each candidate:

```python
# Component scores
semantic = cosine_similarity(query_emb, episode.embedding)
recency = exp(-λ * hours_since_access)
outcome = {success: 0.8, neutral: 0.5, failure: 0.2}[episode.outcome]
importance = episode.emotional_valence

# Learned fusion weights (query-dependent)
weights = self.learned_fusion.compute_weights(query_emb)
# weights = [0.42, 0.28, 0.18, 0.12]  # Example

# Final score
score = weights @ [semantic, recency, outcome, importance]
```

---

#### 8. Learned Re-Ranking

**File**: `memory/episodic.py:LearnedReranker.rerank()`

```python
for result in scored_results:
    # 2nd-pass MLP scoring
    features = [*component_scores, *compressed_query]
    adjustment = mlp.forward(features)  # [-0.2, 0.2]

    # Residual blend
    result.score = 0.7 * initial_score + 0.3 * (initial_score + adjustment)

scored_results.sort(key=lambda x: x.score, reverse=True)
```

---

#### 9. Inhibitory Dynamics (GABA)

**File**: `learning/inhibition.py:InhibitoryNetwork.apply_inhibition()`

```python
inhibited = self.orchestra.process_retrieval(
    retrieved_ids=[r.item.id for r in results],
    scores={str(r.item.id): r.score for r in results},
    embeddings={...}
)
```

**Effect**: Winner-take-all sharpening, suppresses weak competitors

---

#### 10. Eligibility Traces

**File**: `learning/serotonin.py:add_eligibility()`

```python
for mem_id in retrieved_ids:
    self.serotonin.add_eligibility(
        mem_id,
        strength=scores[str(mem_id)]
    )
```

**Purpose**: Enable delayed credit assignment when outcome arrives

---

#### 11. Return Results

```python
return {
    "query": query,
    "count": len(results),
    "episodes": [
        {
            "id": str(r.item.id),
            "content": r.item.content[:200],
            "timestamp": r.item.timestamp.isoformat(),
            "outcome": r.item.outcome.value,
            "score": round(r.score, 4),
            "components": {
                "semantic": 0.92,
                "recency": 0.78,
                "outcome": 0.80,
                "importance": 0.75,
            }
        }
        for r in results
    ]
}
```

---

## Files Touched

### Store Operation

| Phase | File | Purpose |
|-------|------|---------|
| Entry | `mcp/tools/episodic.py` | MCP tool handler |
| Entry | `mcp/gateway.py` | Rate limiting, auth |
| Embed | `embedding/bge_m3.py` | Text → vector |
| Separate | `memory/pattern_separation.py` | DG orthogonalization |
| Neuromod | `learning/neuromodulators.py` | Orchestra coordinator |
| Neuromod | `learning/norepinephrine.py` | Arousal/novelty |
| Neuromod | `learning/acetylcholine.py` | Encoding mode |
| Gate | `core/learned_gate.py` | Storage decision |
| Memory | `memory/episodic.py` | EpisodicMemory.create() |
| Saga | `storage/saga.py` | Transaction orchestration |
| Vector | `storage/qdrant_store.py` | Vector storage |
| Graph | `storage/neo4j_store.py` | Knowledge graph |
| Attractor | `memory/pattern_separation.py` | PatternCompletion |
| Extract | `extraction/entity_extractor.py` | Entity extraction |

### Recall Operation

| Phase | File | Purpose |
|-------|------|---------|
| Entry | `mcp/tools/episodic.py` | MCP tool handler |
| Embed | `embedding/bge_m3.py` | Query → vector |
| Neuromod | `learning/neuromodulators.py` | Mode/arousal |
| Neuromod | `learning/norepinephrine.py` | Threshold modulation |
| Neuromod | `learning/acetylcholine.py` | Retrieval mode |
| Complete | `memory/pattern_separation.py` | Attractor completion |
| Cluster | `memory/cluster_index.py` | Hierarchical search |
| Sparse | `memory/learned_sparse_index.py` | Sparse addressing |
| Vector | `storage/qdrant_store.py` | Similarity search |
| Score | `memory/episodic.py` | Multi-component score |
| Fusion | `memory/episodic.py` | LearnedFusionWeights |
| Rerank | `memory/episodic.py` | LearnedReranker |
| Inhibit | `learning/inhibition.py` | GABA sharpening |
| Eligibility | `learning/serotonin.py` | Credit assignment |
| Reconsolidate | `learning/reconsolidation.py` | Embedding update |

---

## Key Data Structures

### Episode

```python
@dataclass
class Episode:
    id: UUID
    session_id: str
    content: str
    embedding: List[float]  # 1024-dim
    context: EpisodeContext
    outcome: Outcome  # success/failure/partial/neutral
    emotional_valence: float  # [0, 1]
    stability: float  # FSRS parameter
    timestamp: datetime
```

### ScoredResult

```python
@dataclass
class ScoredResult:
    item: Episode  # or Entity or Procedure
    score: float  # Combined score [0, 1]
    components: Dict[str, float]  # Individual component scores
```

### NeuromodulatorState

```python
@dataclass
class NeuromodulatorState:
    dopamine_rpe: float  # Reward prediction error
    norepinephrine_gain: float  # Arousal gain
    acetylcholine_mode: str  # encoding/balanced/retrieval
    serotonin_mood: float  # Long-term mood
    inhibition_sparsity: float  # GABA sparsity
```

### GateDecision

```python
@dataclass
class GateDecision:
    action: GateAction  # STORE/BUFFER/SKIP
    probability: float  # P(useful)
    exploration_boost: float  # Thompson sampling bonus
    features: np.ndarray  # Gate input features
    neuromod_state: Optional[NeuromodulatorState]
```
