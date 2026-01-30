# Biological Memory Mechanisms: Analysis for World Weaver Design

**Date**: 2025-12-06
**Author**: World Weaver Computational Biology Agent
**Purpose**: Evaluate WW architecture against neuroscience research

---

## Executive Summary

World Weaver implements a tripartite memory system (episodic, semantic, procedural) inspired by cognitive neuroscience, but several key biological mechanisms are missing or simplified. This analysis identifies gaps and recommends enhancements based on 2024-2025 research in computational neuroscience.

**Key Findings**:
1. Pattern separation/completion not explicitly modeled
2. Sleep-based consolidation missing temporal orchestration
3. Working memory capacity limits not implemented
4. Hebbian learning lacks homeostatic balance
5. Replay mechanisms absent from consolidation

---

## 1. Hippocampal Memory: Episodic Encoding

### Biological Mechanisms

#### Pattern Separation (Dentate Gyrus)

The dentate gyrus orthogonalizes overlapping input patterns at encoding to reduce interference between similar memories. Recent research shows:

- **DG-CA3 Pattern Separation**: The DG-CA3 pair demonstrates pattern separation during naturalistic stimuli, creating distinct memory traces for similar experiences.
- **Neurogenesis Role**: New neurons in the dentate gyrus facilitate pattern separation by reducing overlap between memory engrams. Systems consolidation can reshape engrams through "promiscuous activity" where hippocampal neurons become less discriminative over time.
- **Computational Requirement**: Similar inputs must be mapped to dissimilar representations to prevent catastrophic interference.

#### Pattern Completion (CA3-CA1)

CA3 and CA1 subfields perform pattern completion, reinstating full memories from partial cues:

- **Holistic Reinstatement**: Hippocampal pattern completion drives neocortical reinstatement of all event information, even 24h after encoding.
- **Additive, Not Compensatory**: Hippocampus and neocortex work together additively rather than the hippocampus becoming redundant after consolidation.
- **CA3-CA1 and CA1-SUB**: Both pairs show evidence of pattern completion during retrieval.

### Current WW Implementation

```python
# episodic.py lines 238-347
async def recall(self, query, limit=10, ...):
    # Vector similarity search
    query_emb = await self.embedding.embed_query(query)
    results = await self.vector_store.search(...)

    # Weighted scoring
    combined_score = (
        0.4 * semantic_score +
        0.25 * recency_score +
        0.2 * outcome_score +
        0.15 * importance_score
    )
```

**Strengths**:
- Semantic similarity via embeddings
- Recency weighting (exponential decay)
- Outcome-based prioritization
- FSRS stability tracking

**Missing**:
- **No Pattern Separation**: Similar episodes are not orthogonalized at encoding. Two similar conversations will have similar embeddings, risking interference.
- **No Pattern Completion**: Retrieval is purely vector similarity, not completion from partial cues. Real hippocampus can reconstruct entire episodes from fragments.
- **No DG/CA3/CA1 Separation**: Architecture treats hippocampus as monolithic vector store, missing computational specialization.

### Recommendations

1. **Add Pattern Separation Layer**:
   ```python
   class DentateGyrus:
       """Orthogonalize similar inputs to reduce interference."""

       async def encode(self, content: str, context: dict) -> np.ndarray:
           # Base embedding
           base_emb = await self.embedding.embed_query(content)

           # Find similar recent episodes
           similar = await self.vector_store.search(
               vector=base_emb,
               limit=10,
               score_threshold=0.75,
               filter={"timestamp": {"gte": one_week_ago}}
           )

           # Orthogonalize if similar episodes exist
           if similar:
               # Random projection to separate similar patterns
               separation_vector = self._compute_separation(base_emb, similar)
               orthogonal_emb = base_emb + self.separation_strength * separation_vector
               orthogonal_emb = orthogonal_emb / np.linalg.norm(orthogonal_emb)
               return orthogonal_emb

           return base_emb
   ```

2. **Add Pattern Completion Mechanism**:
   ```python
   class CA3Completion:
       """Reconstruct full episodes from partial cues via associative recall."""

       async def complete(self, partial_cue: str, context: dict) -> list[Episode]:
           # Multi-hop retrieval: find fragments, then complete
           cue_emb = await self.embedding.embed_query(partial_cue)

           # CA3 recurrent connections: spread activation
           activated = await self.semantic.spread_activation(
               seed_entities=await self._extract_entities(partial_cue),
               steps=2,
               retention=0.6,
           )

           # CA1 output: synthesize complete episode
           episodes = await self._reconstruct_from_activation(activated, context)
           return episodes
   ```

3. **Staged Encoding Pipeline**:
   ```
   Input → DG (separation) → CA3 (storage) → CA1 (consolidation prep) → Neocortex
   ```

---

## 2. Synaptic Plasticity: LTP/LTD and Hebbian Learning

### Biological Mechanisms

#### Long-Term Potentiation/Depression

- **LTP**: Persistent increase in synaptic strength when pre/post neurons co-activate (Hebbian). Generates spatial schemas resistant to generalization.
- **LTD**: Persistent decrease in synaptic strength. Enables dynamic updating and inclusion of detailed content.
- **Interplay**: LTP and LTD work together to create complex associative memories. LTP builds schemas, LTD refines them.

#### Spike-Timing Dependent Plasticity (STDP)

- **Temporal Window**: Pre before post (within ~20ms) → LTP. Post before pre → LTD.
- **Magnitude**: Smaller time differences produce stronger changes.
- **Asymmetry**: Critical for causal learning - "pre predicts post" strengthens forward connections.

#### Homeostatic Plasticity

- **Synaptic Scaling**: Global adjustment to prevent runaway potentiation or depression.
- **Distance-Dependent Scaling**: Distal synapses may have different scaling rules.
- **Integration Challenge**: Homeostatic mechanisms must preserve Hebbian-coded information while preventing instability.

### Current WW Implementation

```python
# semantic.py lines 193-204
def strengthen_connection(current_weight: float, learning_rate: float = 0.1) -> float:
    """Bounded Hebbian update approaching 1.0 asymptotically"""
    return current_weight + learning_rate * (1.0 - current_weight)

async def _strengthen_co_retrieval(self, results: list[ScoredResult]):
    # Batch strengthen relationships between co-retrieved entities
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1:]:
            await self.graph_store.strengthen_relationship(
                source_id=str(e1.id),
                target_id=str(e2.id),
                learning_rate=self.learning_rate,  # 0.1 default
            )
```

**Strengths**:
- Bounded growth (approaches 1.0 asymptotically)
- Co-retrieval strengthening (Hebbian principle)
- Decay mechanism (lines 761-842)

**Missing**:
- **No LTD**: Only strengthening, no weakening when entities are NOT co-retrieved.
- **No STDP**: No temporal ordering effects. Co-retrieval within 1s vs 1hr treated identically.
- **No Homeostatic Scaling**: Global normalization missing. Hub nodes can accumulate unbounded activation.
- **No Molecular Timescales**: Single learning rate; real synapses have early-phase (minutes) and late-phase (hours) LTP.

### Recommendations

1. **Add LTD for Non-Co-Retrieval**:
   ```python
   async def update_non_coactivated_weights(
       self,
       activated: list[Entity],
       all_neighbors: list[Entity],
       ltd_rate: float = 0.05,
   ):
       """Weaken connections not recently activated (competitive learning)."""
       activated_ids = {e.id for e in activated}

       for entity in activated:
           neighbors = await self.graph_store.get_relationships(str(entity.id))

           for rel in neighbors:
               other_id = rel["other_id"]
               if other_id not in activated_ids:
                   # LTD: decrease weight
                   current_weight = rel["properties"]["weight"]
                   new_weight = max(
                       self.min_weight,
                       current_weight - ltd_rate * current_weight
                   )
                   await self.graph_store.update_relationship_weight(
                       entity.id, other_id, new_weight
                   )
   ```

2. **Add STDP Temporal Kernel**:
   ```python
   def stdp_weight_change(
       self,
       delta_t_ms: float,  # post_time - pre_time
       current_weight: float,
   ) -> float:
       """STDP asymmetric temporal learning window."""
       if delta_t_ms > 0:  # pre before post → LTP
           A_plus = 0.1
           tau_plus = 20  # ms
           delta_w = A_plus * math.exp(-delta_t_ms / tau_plus)
       else:  # post before pre → LTD
           A_minus = -0.12
           tau_minus = 20
           delta_w = A_minus * math.exp(delta_t_ms / tau_minus)

       # Bound to [0, 1]
       new_weight = current_weight + delta_w
       return max(0.0, min(1.0, new_weight))
   ```

3. **Add Homeostatic Synaptic Scaling**:
   ```python
   async def apply_homeostatic_scaling(
       self,
       target_total_weight: float = 10.0,
       time_window_hours: int = 24,
   ):
       """Global scaling to maintain network stability."""
       for entity in await self.get_all_entities():
           rels = await self.graph_store.get_relationships(str(entity.id), "out")

           total_weight = sum(r["properties"]["weight"] for r in rels)

           if total_weight > target_total_weight * 1.2:
               # Scale down all weights proportionally
               scale_factor = target_total_weight / total_weight
               for rel in rels:
                   new_weight = rel["properties"]["weight"] * scale_factor
                   await self.graph_store.update_relationship_weight(
                       entity.id,
                       rel["other_id"],
                       new_weight,
                   )
   ```

---

## 3. Memory Consolidation: Systems Theory and Sleep

### Biological Mechanisms

#### Systems Consolidation Theory

- **Standard Theory**: Memories initially depend on hippocampus, then transfer to neocortex over days/weeks.
- **2024 Update**: Hippocampus and neocortex work *additively*, not in replacement. Hippocampus remains engaged even after consolidation.
- **Engram Reshaping**: Systems consolidation reshapes hippocampal engrams themselves through synaptic pruning and neurogenesis-driven plasticity.

#### Sleep-Dependent Consolidation

- **Slow-Wave Sleep (SWS)**: Optimal state for consolidation. Cortical slow oscillations coordinate hippocampal replay.
- **Hippocampal Replay**: Sharp-wave ripples (SPW-Rs) during sleep reactivate recent memory traces in compressed temporal sequences.
- **Thalamocortical Loops**: Interaction between slow oscillations and synaptic plasticity maps hippocampal traces to persistent cortical representations.
- **Awake Replay**: Brief rest periods after learning also show reactivation, correlating with subsequent memory performance.

#### Selective Consolidation

- **Not All Memories Consolidate**: High-reward, high-salience memories preferentially replayed and consolidated.
- **Simulation-Selection Model**: CA3 generates diverse activity patterns; CA1 selectively reinforces high-value patterns.
- **Reinforcement Learning**: Value signals modulate consolidation, prioritizing memories relevant to future decisions.

### Current WW Implementation

```python
# consolidation/service.py lines 253-404
async def _consolidate_deep(self, session_filter, hours=168):
    # Get episodes from past week
    episodes = await episodic.recall_by_timerange(start_time, end_time, ...)

    # Cluster by embedding similarity (HDBSCAN)
    clusters = await self._cluster_episodes(episodes, threshold=0.75)

    # Extract entities from clusters with min_occurrences threshold
    for cluster in clusters:
        if len(cluster) < self.min_occurrences:  # default: 3
            continue

        entity_info = self._extract_entity_from_cluster(cluster)
        entity = await semantic.create_entity(...)

        # Create provenance links
        for ep in cluster:
            await self.graph_store.create_relationship(
                source_id=str(ep.id),
                target_id=str(entity.id),
                rel_type=RelationType.SOURCE_OF.value,
            )
```

**Strengths**:
- Episodic → semantic transfer (basic consolidation)
- Frequency threshold (min_occurrences) mimics selective consolidation
- Provenance tracking (source episodes linked to entities)

**Missing**:
- **No Sleep-Like Orchestration**: Consolidation is manual/scheduled, not triggered by rest periods.
- **No Replay Mechanism**: Doesn't reactivate recent episodes in compressed sequences.
- **No Temporal Coordination**: No slow oscillation-like rhythms or temporal structure.
- **No Value-Based Selection**: Doesn't prioritize high-reward or important episodes beyond simple valence.
- **No Multi-Stage Process**: Single-pass transfer, not iterative refinement over multiple sleep cycles.

### Recommendations

1. **Add Replay-Based Consolidation**:
   ```python
   class SleepConsolidation:
       """Simulate sleep-based memory consolidation with replay."""

       async def slow_wave_sleep_cycle(
           self,
           duration_minutes: int = 90,
           replay_rate_hz: float = 0.1,  # 6 replays/minute
       ):
           """Simulate one SWS cycle with hippocampal replay."""
           # Get high-value recent episodes
           recent = await self.episodic.recall(
               query="*",
               limit=1000,
               time_start=datetime.now() - timedelta(hours=24),
           )

           # Prioritize by value = outcome + importance + recency
           prioritized = sorted(
               recent,
               key=lambda r: (
                   0.4 * r.components["outcome"] +
                   0.3 * r.components["importance"] +
                   0.3 * r.components["recency"]
               ),
               reverse=True,
           )[:100]

           # Replay in compressed sequences (temporal compression ~20x)
           for episode in prioritized:
               # Reactivate: strengthen semantic links
               entities = await self.extract_entities(episode.content)

               # Strengthen co-occurrence (Hebbian)
               await self.semantic._strengthen_co_retrieval(
                   [ScoredResult(item=e, score=1.0) for e in entities]
               )

               # Extract to semantic if pattern emerges
               if await self._is_recurring_pattern(episode, entities):
                   await self._extract_to_semantic(episode, entities)

               # Simulate inter-replay interval (~10s)
               await asyncio.sleep(1.0 / replay_rate_hz)
   ```

2. **Add Multi-Stage Consolidation**:
   ```python
   async def systems_consolidation_cascade(
       self,
       episode: Episode,
       age_hours: float,
   ):
       """Progressive transfer from hippocampus to neocortex."""
       # Stage 1 (0-24h): Hippocampus-dependent, high detail
       if age_hours < 24:
           return await self.episodic.recall(...)

       # Stage 2 (1-7d): Mixed hippocampal-neocortical
       elif age_hours < 168:
           epi_results = await self.episodic.recall(...)
           sem_results = await self.semantic.recall(...)
           # Blend with hippocampal weight = 0.7
           return self._blend_results(epi_results, sem_results, hip_weight=0.7)

       # Stage 3 (>7d): Primarily neocortical
       else:
           sem_results = await self.semantic.recall(...)
           epi_results = await self.episodic.recall(...)
           # Blend with hippocampal weight = 0.3
           return self._blend_results(epi_results, sem_results, hip_weight=0.3)
   ```

3. **Add Value-Based Selection**:
   ```python
   def compute_consolidation_priority(self, episode: Episode) -> float:
       """Compute priority for consolidation (mimics dopamine/value signals)."""
       # Reward signal
       outcome_value = {
           Outcome.SUCCESS: 1.0,
           Outcome.PARTIAL: 0.5,
           Outcome.NEUTRAL: 0.2,
           Outcome.FAILURE: 0.0,
       }[episode.outcome]

       # Novelty signal
       novelty = 1.0 - episode.stability / 10.0  # unstable = novel

       # Salience signal
       salience = episode.emotional_valence

       # Combined value
       priority = (
           0.5 * outcome_value +
           0.3 * novelty +
           0.2 * salience
       )

       return priority
   ```

---

## 4. Working Memory: Prefrontal Cortex and Capacity Limits

### Biological Mechanisms

#### Capacity Limits

- **Miller's Law**: 7±2 items (classic), updated to 3-4 chunks in modern research.
- **Flexibility-Capacity Trade-off**: The same neural mechanism underlying flexibility also limits capacity.
- **Sparse Representations**: Random recurrent networks maintain working memory via sparse coding, improving with sparsity.

#### Prefrontal Cortex Role

- **Attention Control**: Dorsolateral PFC actively maintains access to stimulus representations.
- **Prioritization**: Superior precentral sulcus (sPCS) controls which items receive priority in WM.
- **Distributed Storage**: Working memory depends on cell type-specific connectivity (parvalbumin interneuron gradients).
- **Dynamic Coding**: Optimal information loading uses inputs orthogonal to delay-period activity.

#### Neural Mechanisms

- **Persistent Activity**: Maintained through recurrent excitation (classical view).
- **Synaptic Theory**: Short-term facilitation at synapses without persistent spiking (alternative view).
- **Prefrontal-Basal Ganglia Loops**: Basal ganglia gate access to working memory via disinhibition.

### Current WW Implementation

**Working memory is essentially absent** in WW. All episodic memory goes directly to persistent storage (Qdrant/Neo4j). There is no:
- Transient buffer for current context
- Capacity limit
- Active maintenance mechanism
- Prioritization/gating

The closest analog is the in-context window of the LLM itself, which is external to WW.

### Recommendations

1. **Add Working Memory Buffer**:
   ```python
   class WorkingMemory:
       """Transient buffer for active context (3-4 items)."""

       def __init__(self, capacity: int = 4):
           self.capacity = capacity
           self.buffer: list[WorkingMemoryItem] = []
           self.attention_weights: list[float] = []

       async def load(self, item: Any, priority: float = 0.5):
           """Load item into WM with priority-based eviction."""
           wm_item = WorkingMemoryItem(
               content=item,
               priority=priority,
               loaded_at=datetime.now(),
           )

           # Evict if at capacity
           if len(self.buffer) >= self.capacity:
               # Evict lowest priority item
               min_idx = np.argmin(self.attention_weights)
               evicted = self.buffer.pop(min_idx)
               self.attention_weights.pop(min_idx)

               # Consolidate evicted to episodic
               await self.episodic.create(
                   content=evicted.content,
                   context={"source": "working_memory_eviction"},
                   outcome="neutral",
                   valence=evicted.priority,
               )

           self.buffer.append(wm_item)
           self.attention_weights.append(priority)

       def recall(self, cue: str) -> list[Any]:
           """Retrieve from WM (fast, no persistence)."""
           # Match cue to buffer items
           matches = []
           for item, weight in zip(self.buffer, self.attention_weights):
               similarity = self._compute_similarity(cue, item.content)
               if similarity > 0.5:
                   matches.append((item, similarity * weight))

           # Return top matches sorted by attention-weighted similarity
           matches.sort(key=lambda x: x[1], reverse=True)
           return [m[0] for m in matches]
   ```

2. **Add Prefrontal Control**:
   ```python
   class PrefrontalControl:
       """Gate and prioritize working memory contents."""

       async def update_priorities(
           self,
           wm: WorkingMemory,
           task_goals: list[str],
       ):
           """Update attention weights based on task relevance."""
           for i, item in enumerate(wm.buffer):
               # Compute goal relevance
               relevance = max(
                   self._goal_relevance(item, goal)
                   for goal in task_goals
               )

               # Recency bonus
               age_s = (datetime.now() - item.loaded_at).total_seconds()
               recency = math.exp(-age_s / 60.0)  # decay over 1 minute

               # Update attention weight
               wm.attention_weights[i] = 0.7 * relevance + 0.3 * recency

       async def gate_to_longterm(
           self,
           wm: WorkingMemory,
           consolidation_threshold: float = 0.7,
       ):
           """Transfer high-value WM items to episodic memory."""
           for item, weight in zip(wm.buffer, wm.attention_weights):
               if weight > consolidation_threshold:
                   await self.episodic.create(
                       content=item.content,
                       context={"source": "working_memory_gate"},
                       outcome="neutral",
                       valence=weight,
                   )
   ```

---

## 5. Procedural Memory: Basal Ganglia and Motor Learning

### Biological Mechanisms

- **Striatum**: Input stage, receives cortical/thalamic inputs, learns action-outcome associations.
- **Globus Pallidus**: Output regulation via direct (go) and indirect (no-go) pathways.
- **Cerebellum**: Fine motor timing and error correction (forward models).
- **Chunking**: Sequences become unitized through repeated practice, reducing cognitive load.
- **Habit Formation**: Transition from goal-directed (prefrontal) to habitual (striatal) control.

### Current WW Implementation

```python
# memory/procedural.py
class ProceduralMemory:
    async def build(self, trajectory: list[Action], outcome: Outcome):
        """BUILD: Extract procedure from successful trajectory."""
        if outcome.success_score < 0.7:
            return None

        steps = self.extract_steps(trajectory)
        script = self.abstract_script(steps)
        return Procedure(name=..., steps=steps, script=script, ...)

    async def retrieve(self, task_description: str):
        """RETRIEVE: Match task to procedures."""
        candidates = self.vector_search(query_vec, limit)
        return sorted(candidates, key=lambda p: p.success_rate, reverse=True)

    async def update(self, procedure_id: str, feedback: Feedback):
        """UPDATE: Learn from execution outcomes."""
        if feedback.success:
            proc.success_rate = (
                (proc.success_rate * proc.execution_count + 1) /
                (proc.execution_count + 1)
            )
```

**Strengths**:
- Build from trajectories (experience-driven)
- Success rate tracking (outcome-based learning)
- Retrieval by similarity (generalization)
- Update from feedback (reinforcement learning)

**Missing**:
- **No Chunking**: Multi-step procedures not hierarchically composed.
- **No Habit Formation**: No transition from controlled to automatic execution.
- **No Forward Models**: No predictive simulation of procedure outcomes.
- **No Temporal Credit Assignment**: Success/failure assigned to entire procedure, not specific steps.

### Recommendations

1. **Add Hierarchical Chunking**:
   ```python
   class ProcedureChunk:
       """Hierarchical procedure representation."""
       name: str
       atomic_steps: list[Step]
       sub_chunks: list[ProcedureChunk]
       execution_count: int
       avg_duration_ms: float

   async def chunk_procedure(
       self,
       procedure: Procedure,
       min_chunk_size: int = 3,
       max_chunk_size: int = 7,
   ) -> ProcedureChunk:
       """Chunk procedure into hierarchical sub-procedures."""
       # Find repeating patterns in steps
       patterns = self._find_repeating_subsequences(
           procedure.steps,
           min_length=min_chunk_size,
       )

       # Create chunks for common patterns
       chunks = []
       for pattern in patterns:
           if pattern.count >= 3:  # appeared 3+ times
               chunk = ProcedureChunk(
                   name=f"{procedure.name}_{pattern.id}",
                   atomic_steps=pattern.steps,
                   sub_chunks=[],
               )
               chunks.append(chunk)

       # Replace pattern occurrences with chunk references
       compressed_steps = self._replace_with_chunks(procedure.steps, chunks)

       return ProcedureChunk(
           name=procedure.name,
           atomic_steps=compressed_steps,
           sub_chunks=chunks,
       )
   ```

2. **Add Temporal Credit Assignment**:
   ```python
   async def update_with_temporal_credit(
       self,
       procedure: Procedure,
       feedback: Feedback,
       trajectory: list[ActionResult],
   ):
       """Assign credit/blame to specific steps based on when error occurred."""
       if feedback.success:
           # Uniform credit to all steps
           for step in procedure.steps:
               step.success_count += 1
       else:
           # Identify failure point
           failure_step = self._identify_failure_step(trajectory, feedback)

           # TD(λ) eligibility traces
           for i, step in enumerate(procedure.steps):
               distance_to_failure = abs(i - failure_step)
               credit = 0.9 ** distance_to_failure  # exponential decay

               step.failure_attribution += credit
   ```

---

## 6. Semantic Networks: Temporal Cortex and Concept Formation

### Biological Mechanisms

- **Anterior Temporal Lobe (ATL)**: Hub for semantic knowledge, especially for concrete object concepts.
- **Posterior Cortex**: Distributed sensorimotor features (visual, auditory, tactile).
- **Hub-and-Spoke Model**: ATL as hub integrates distributed spoke representations.
- **Conceptual Hierarchy**: Abstract concepts emerge from statistical regularities across experiences.

### Current WW Implementation

```python
# semantic.py lines 202-311
async def recall(self, query, context_entities, ...):
    # Vector similarity
    results = await self.vector_store.search(...)

    # ACT-R activation (base-level + spreading)
    for entity in results:
        activation = await self._calculate_activation(entity, context, ...)

        # Base-level: log(access_count) - decay * log(time_since_access)
        base = math.log(access_count) - self.decay * math.log(elapsed / 3600)

        # Spreading: W * Hebbian_weight * (S - ln(fan_out))
        spreading = sum(W * strength * (S - math.log(fan)) for src in context)

        total_activation = base + spreading + noise
```

**Strengths**:
- Hebbian knowledge graph (co-retrieval strengthening)
- ACT-R activation (recency/frequency + spreading)
- Fan effect (hub nodes spread less activation)
- Entity types (CONCEPT, PERSON, TOOL, etc.)

**Missing**:
- **No Conceptual Hierarchy**: Flat entity types, no superordinate/subordinate relations.
- **No Distributed Features**: Entities are unitary nodes, not feature compositions.
- **No Category Learning**: No abstraction of categories from instances.

### Recommendations

1. **Add Conceptual Hierarchy**:
   ```python
   class ConceptHierarchy:
       """Hierarchical semantic network with IS-A relations."""

       async def create_concept_hierarchy(
           self,
           instances: list[Entity],
           abstraction_level: int = 3,
       ):
           """Build hierarchy from instances to abstract categories."""
           # Level 0: Instances (e.g., "my_car", "neighbor_car")
           level0 = instances

           # Level 1: Basic categories (e.g., "car")
           level1 = await self._cluster_by_similarity(
               level0,
               threshold=0.7,
               label_fn=self._extract_common_features,
           )

           # Level 2: Superordinate (e.g., "vehicle")
           level2 = await self._cluster_by_similarity(
               level1,
               threshold=0.5,
               label_fn=self._extract_common_features,
           )

           # Create IS-A relationships
           for instance in level0:
               basic = self._find_parent(instance, level1)
               await self.semantic.create_relationship(
                   source_id=instance.id,
                   target_id=basic.id,
                   relation_type=RelationType.IS_A,
               )

           for basic in level1:
               super_cat = self._find_parent(basic, level2)
               await self.semantic.create_relationship(
                   source_id=basic.id,
                   target_id=super_cat.id,
                   relation_type=RelationType.IS_A,
               )
   ```

2. **Add Feature-Based Composition**:
   ```python
   class FeatureBasedConcept:
       """Concepts as compositions of distributed features."""

       def __init__(self, name: str):
           self.name = name
           self.visual_features: dict[str, float] = {}
           self.functional_features: dict[str, float] = {}
           self.contextual_features: dict[str, float] = {}

       async def compute_similarity(
           self,
           other: FeatureBasedConcept,
       ) -> float:
           """Feature-weighted similarity."""
           visual_sim = self._cosine_similarity(
               self.visual_features,
               other.visual_features,
           )
           functional_sim = self._cosine_similarity(
               self.functional_features,
               other.functional_features,
           )
           contextual_sim = self._cosine_similarity(
               self.contextual_features,
               other.contextual_features,
           )

           # Weight by modality importance (learned)
           return (
               0.4 * visual_sim +
               0.3 * functional_sim +
               0.3 * contextual_sim
           )
   ```

---

## Summary of Gaps and Priorities

### Critical Missing Mechanisms

| Mechanism | Biological Function | Impact on WW | Priority |
|-----------|-------------------|-------------|----------|
| **Pattern Separation** | Orthogonalize similar inputs (DG) | Prevents interference between similar episodes | HIGH |
| **Pattern Completion** | Reconstruct from partial cues (CA3) | Enables fragment-based recall | HIGH |
| **Sleep Replay** | Consolidate high-value memories | Automates consolidation timing | MEDIUM |
| **LTD** | Weaken unused connections | Prevents weight saturation | HIGH |
| **STDP** | Temporal causality learning | Improves predictive relationships | MEDIUM |
| **Working Memory** | Transient buffer (3-4 items) | Reduces persistence overhead | MEDIUM |
| **Homeostatic Scaling** | Prevent runaway potentiation | Network stability | HIGH |
| **Value-Based Selection** | Prioritize important memories | Improves consolidation quality | MEDIUM |
| **Hierarchical Chunking** | Compress procedures | Improves skill efficiency | LOW |
| **Conceptual Hierarchy** | Abstract categories | Better generalization | MEDIUM |

### Recommended Implementation Sequence

**Phase 1: Stability (Weeks 1-2)**
1. Add LTD for competitive learning
2. Implement homeostatic synaptic scaling
3. Add pattern separation layer

**Phase 2: Consolidation (Weeks 3-4)**
4. Add replay-based consolidation
5. Implement value-based selection
6. Add multi-stage consolidation

**Phase 3: Retrieval (Weeks 5-6)**
7. Add pattern completion mechanism
8. Implement working memory buffer
9. Add STDP temporal learning

**Phase 4: Abstraction (Weeks 7-8)**
10. Add conceptual hierarchy
11. Implement hierarchical chunking

---

## Key Papers to Reference

### Hippocampal Consolidation
1. Ding et al. (2024). "Cell type-specific connectome predicts distributed working memory." eLife.
2. Hawkins et al. (2024). "An Enduring Role for Hippocampal Pattern Completion." Journal of Neuroscience.
3. Paller lab (2025). "Awake reactivation of cortical memory traces predicts subsequent memory." Progress in Neurobiology.

### Synaptic Plasticity
4. Kemp & Manahan-Vaughan (2024). "Interplay of hippocampal LTP and LTD in enabling memory representations." Royal Society.
5. ArXiv (2024). "Computational models of learning and synaptic plasticity."

### Systems Consolidation
6. Schlichting & Preston (2024). "A generative model of memory construction and consolidation." Nature Human Behaviour.
7. Frontiers (2024). "Memory consolidation from a reinforcement learning perspective."
8. Journal of Theoretical Biology (2024). "Segregation-to-Integration Transformation (SIT) Model."

### Working Memory
9. Prioritizing Working Memory Resources (2024). Journal of Neuroscience (bioRxiv preprint).
10. PNAS (2023). "Optimal information loading into working memory explains dynamic coding in PFC."

---

## Conclusion

World Weaver's tripartite architecture provides a solid cognitive foundation, but lacks several key biological mechanisms:

1. **Pattern separation/completion**: Current vector similarity doesn't capture DG orthogonalization or CA3 reconstruction.
2. **Balanced plasticity**: Only LTP implemented; missing LTD, STDP, and homeostatic scaling.
3. **Sleep consolidation**: Manual/scheduled rather than replay-based and temporally structured.
4. **Working memory**: Absent entirely; all memories persist immediately.
5. **Hierarchical abstraction**: Flat entity types without conceptual hierarchies or feature composition.

Implementing these mechanisms would significantly improve:
- **Interference resistance** (pattern separation)
- **Partial-cue retrieval** (pattern completion)
- **Network stability** (LTD + homeostatic scaling)
- **Consolidation quality** (replay + value selection)
- **Efficiency** (working memory + chunking)

The recommended phased approach prioritizes stability and consolidation improvements first, followed by retrieval and abstraction enhancements.

---

## Sources

- [Pattern separation and completion in hippocampus](https://pmc.ncbi.nlm.nih.gov/articles/PMC3812781/)
- [Hippocampal pattern completion after 24h delay](https://www.jneurosci.org/content/44/18/e1740232024)
- [Memory consolidation from reinforcement learning perspective](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1538741/full)
- [Awake reactivation predicts memory consolidation](https://faculty.wcas.northwestern.edu/~paller/Progress%20in%20Neurobiology%202025.pdf)
- [Systems consolidation reshapes hippocampal engrams](https://bioengineer.org/systems-consolidation-reshapes-hippocampal-engrams/)
- [LTP and LTD interplay in memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC11343234/)
- [Computational models of synaptic plasticity](https://arxiv.org/pdf/2412.05501)
- [Cell type-specific connectome and working memory](https://www.cns.nyu.edu/wanglab/publications/pdf/Ding_2024.pdf)
- [Prioritizing working memory resources](https://www.biorxiv.org/content/10.1101/2024.05.11.593696v3.full.pdf)
- [Optimal information loading into working memory](https://www.pnas.org/doi/full/10.1073/pnas.2307991120)
- [Generative model of memory construction](https://pubmed.ncbi.nlm.nih.gov/38242925/)
- [Pattern separation during naturalistic stimuli](https://onlinelibrary.wiley.com/doi/am-pdf/10.1002/hbm.70150)
