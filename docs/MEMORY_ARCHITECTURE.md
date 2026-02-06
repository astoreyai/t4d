# T4DM - Tripartite Neural Memory Architecture

**Version**: 0.2.0 | **Status**: Specification Complete, T4DX Migration Done
**Target**: Archimedes (RTX 3090 24GB, i9, 128GB RAM)

---

## Cognitive Foundation

This memory system implements Tulving's (1972) tripartite model with neural pathway semantics:

| Memory Type | Function | Biological Analog |
|-------------|----------|-------------------|
| **Episodic** | "What happened when" - autobiographical events | Hippocampus |
| **Semantic** | "What I know" - abstracted facts and concepts | Neocortex |
| **Procedural** | "How to do things" - learned skills | Basal ganglia, cerebellum |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MCP MEMORY GATEWAY                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ CC Inst 1  │  │ CC Inst 2  │  │ CC Inst 3  │  │ CC Inst N  │           │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘           │
│        └────────────────┴────────────────┴────────────────┘                │
│                                  │                                          │
│                    ┌─────────────┴─────────────┐                           │
│                    │    Session Namespacing    │                           │
│                    └─────────────┬─────────────┘                           │
├─────────────────────────────────┼───────────────────────────────────────────┤
│                         WORKING MEMORY                                      │
│              (Current context, active goals, ephemeral)                    │
├─────────────────────────────────┼───────────────────────────────────────────┤
│  ┌─────────────────────────────┴─────────────────────────────┐            │
│  │                    EPISODIC MEMORY                        │            │
│  │  ┌─────────────────────────────────────────────────────┐  │            │
│  │  │  T4DX Storage (κ < 0.3)                             │  │            │
│  │  │  • Session-bound episodes                           │  │            │
│  │  │  • Bi-temporal versioning (T_ref, T_sys)            │  │            │
│  │  │  • Vector embeddings (HNSW index)                   │  │            │
│  │  │  • FSRS stability tracking                          │  │            │
│  │  └─────────────────────────────────────────────────────┘  │            │
│  └───────────────────────────────────────────────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐            │
│  │                    SEMANTIC MEMORY                        │            │
│  │  ┌─────────────────────────────────────────────────────┐  │            │
│  │  │  T4DX Storage (κ > 0.6)                             │  │            │
│  │  │  • Entity nodes with decay properties               │  │            │
│  │  │  • Hebbian-weighted relationships (CSR graph)       │  │            │
│  │  │  • ACT-R activation-based retrieval                 │  │            │
│  │  │  • Spreading activation                             │  │            │
│  │  └─────────────────────────────────────────────────────┘  │            │
│  └───────────────────────────────────────────────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐            │
│  │                   PROCEDURAL MEMORY                       │            │
│  │  ┌─────────────────────────────────────────────────────┐  │            │
│  │  │  Workflow DAGs (Memp Pattern)                       │  │            │
│  │  │  • Fine-grained step sequences                      │  │            │
│  │  │  • High-level script abstractions                   │  │            │
│  │  │  • Build-Retrieve-Update lifecycle                  │  │            │
│  │  │  • Skill consolidation                              │  │            │
│  │  └─────────────────────────────────────────────────────┘  │            │
│  └───────────────────────────────────────────────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│                       CONSOLIDATION ENGINE                                  │
│  • Episodic → Semantic transfer (hippocampal replay)                       │
│  • Skill consolidation from trajectories                                    │
│  • CRDT conflict resolution                                                 │
│  • FSRS decay scheduling                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Resources |
|-----------|------------|-----------|
| Storage Engine | T4DX (embedded LSM) | In-process, zero network hops |
| Embeddings | BGE-M3 (local) | ~4GB VRAM (fp16) |
| Reranker | BGE-reranker-large | ~2GB VRAM (fp16) |
| MCP Framework | TypeScript SDK + FastMCP | - |
| Decay Algorithm | FSRS (py-fsrs) | - |
| Conflict Resolution | OR-Set CRDT | - |

**Total Allocation**: ~20GB RAM system, ~6GB VRAM

> **Note**: T4DM 2.0 uses T4DX, a custom embedded LSM-style spatiotemporal database where vectors, edges, metadata, and temporal indices are co-located. κ (kappa) determines memory type: κ < 0.3 = episodic, κ > 0.6 = semantic.

---

## 1. Episodic Memory

### Purpose

Store autobiographical events - specific interactions, decisions, and outcomes bound to temporal-spatial context. Preserves particularity: "We discussed X on Tuesday" not "X exists."

### T4DX ItemRecord Schema

```cypher
CREATE (e:Episode {
  id: randomUUID(),
  sessionId: $session,           // Instance namespace
  content: $rawContent,          // Full interaction text
  embedding: $vector,            // 1024-dim BGE-M3
  timestamp: datetime(),         // Event time (T_ref)
  ingestedAt: datetime(),        // System time (T_sys)
  context: $spatialContext,      // Project, file, tool context
  outcome: $result,              // success/failure/partial
  emotionalValence: $valence,    // Importance signal [0,1]
  accessCount: 1,
  lastAccessed: datetime(),
  stability: 1.0                 // FSRS stability (days)
})
```

### Bi-Temporal Versioning

| Time | Purpose |
|------|---------|
| T_ref | When event occurred in real world |
| T_sys | When memory was ingested |

Enables: "What did we know on November 15?" and automatic fact supersession.

### Retrieval Algorithm

```python
def episodic_retrieval(query: str, current_time: datetime, limit: int = 10):
    """Combine semantic similarity, recency, outcome, and importance"""
    query_vec = bge_m3.encode(query)
    candidates = neo4j.vector_search('episode-index', query_vec, limit * 3)

    scored = []
    for ep in candidates:
        semantic_score = ep.similarity
        recency = math.exp(-0.1 * days_since(ep.timestamp, current_time))
        outcome_weight = 1.2 if ep.outcome == 'success' else 0.8
        importance = ep.emotionalValence

        score = (0.4 * semantic_score +
                 0.25 * recency +
                 0.2 * outcome_weight +
                 0.15 * importance)
        scored.append((ep, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)[:limit]
```

---

## 2. Semantic Memory

### Purpose

Store generalized knowledge abstracted from episodic experiences. Transform context-bound episodes into context-free knowledge through semanticization.

### T4DX Schema

```cypher
// Entity node with decay properties
CREATE (e:Entity {
  id: randomUUID(),
  name: $entityName,
  entityType: $type,             // CONCEPT, PERSON, PROJECT, TOOL
  summary: $shortDescription,
  details: $expandedContext,
  embedding: $vector,
  source: $derivationSource,     // episode_id or 'user_provided'
  stability: 1.0,                // FSRS stability
  accessCount: 1,
  lastAccessed: datetime(),
  createdAt: datetime(),
  validFrom: datetime(),         // Bi-temporal
  validTo: null
})

// Hebbian-weighted relationship
CREATE (a)-[:RELATED_TO {
  relationType: $type,           // USES, PRODUCES, REQUIRES, CAUSES
  weight: 0.1,                   // Hebbian strength [0, 1]
  coAccessCount: 1,
  lastCoAccess: datetime()
}]->(b)
```

### Hebbian Strengthening

```python
def strengthen_connection(current_weight: float, learning_rate: float = 0.1) -> float:
    """Bounded Hebbian update approaching 1.0 asymptotically"""
    return current_weight + learning_rate * (1.0 - current_weight)

def on_co_retrieval(entity_a: str, entity_b: str):
    """Fire together, wire together"""
    rel = get_or_create_relationship(entity_a, entity_b)
    rel.weight = strengthen_connection(rel.weight)
    rel.coAccessCount += 1
    rel.lastCoAccess = datetime.now()
```

### ACT-R Activation Retrieval

Total activation combines base-level (recency/frequency) with spreading activation:

```
Aᵢ = Bᵢ + Σⱼ(Wⱼ × Sⱼᵢ) + ε

Where: Bᵢ = ln(Σⱼ tⱼ^(-d))  with decay d = 0.5
```

```python
class ACTRRetrieval:
    def __init__(self, decay: float = 0.5, threshold: float = 0, noise_s: float = 0.5):
        self.d = decay
        self.tau = threshold
        self.s = noise_s

    def base_level(self, access_times: list[datetime], current_time: datetime) -> float:
        """Power-law decay based on access history"""
        total = sum((current_time - t).total_seconds()**(-self.d)
                    for t in access_times)
        return math.log(total) if total > 0 else float('-inf')

    def spreading(self, entity: Entity, context_entities: list[Entity], S: float = 1.6) -> float:
        """Spreading activation from context"""
        if not context_entities:
            return 0
        W = 1.0 / len(context_entities)
        return sum(W * (S - math.log(get_fan(src))) for src in context_entities)

    def total_activation(self, entity: Entity, context: list[Entity], current_time: datetime) -> float:
        B = self.base_level(entity.access_times, current_time)
        S = self.spreading(entity, context)
        noise = random.gauss(0, self.s)
        return B + S + noise
```

---

## 3. Procedural Memory

### Purpose

Store "how-to" knowledge - learned skills, workflows, and action sequences. Implements Memp framework (Zhejiang/Alibaba, 2025) for learnable, transferable procedural knowledge.

### Dual-Format Storage

| Format | Purpose |
|--------|---------|
| Fine-grained steps | Verbatim action sequences with full context |
| Script abstractions | Distilled procedures capturing essential patterns |

### T4DX Schema

```cypher
CREATE (p:Procedure {
  id: randomUUID(),
  name: $procedureName,
  domain: $taskDomain,           // trading, research, coding
  triggerPattern: $pattern,      // When to invoke
  steps: $stepArray,             // Fine-grained sequence
  script: $abstractScript,       // High-level abstraction
  embedding: $vector,
  successRate: 0.0,
  executionCount: 0,
  lastExecuted: null,
  version: 1,
  deprecated: false
})
```

### Build-Retrieve-Update Lifecycle

```python
class ProceduralMemory:
    def build(self, trajectory: list[Action], outcome: Outcome) -> Procedure:
        """BUILD: Distill successful trajectory into procedure"""
        if outcome.success_score < 0.7:
            return None  # Only learn from good outcomes

        steps = self.extract_steps(trajectory)
        script = self.abstract_script(steps)

        return Procedure(
            name=self.generate_name(trajectory),
            steps=steps,
            script=script,
            embedding=self.embed(script)
        )

    def retrieve(self, task_description: str, limit: int = 5) -> list[Procedure]:
        """RETRIEVE: Match task to stored procedures"""
        query_vec = bge_m3.encode(task_description)
        candidates = self.vector_search(query_vec, limit * 2)

        # Filter deprecated and rank by success rate
        active = [p for p in candidates if not p.deprecated]
        return sorted(active, key=lambda p: p.successRate, reverse=True)[:limit]

    def update(self, procedure_id: str, execution_feedback: Feedback):
        """UPDATE: Learn from execution outcomes"""
        proc = self.get(procedure_id)

        if execution_feedback.success:
            # Reinforce successful procedure
            proc.successRate = (
                (proc.successRate * proc.executionCount + 1) /
                (proc.executionCount + 1)
            )
        else:
            # Reflect and potentially revise
            revised = self.reflect_on_failure(proc, execution_feedback)
            if revised.divergence > self.REVISION_THRESHOLD:
                proc.version += 1
                proc.steps = revised.steps
                proc.script = revised.script

        proc.executionCount += 1
        proc.lastExecuted = datetime.now()

        # Deprecate consistently failing procedures
        if proc.executionCount > 10 and proc.successRate < 0.3:
            proc.deprecated = True
```

---

## 4. Consolidation Engine

### Episodic → Semantic Transfer

Mirrors hippocampal-neocortical consolidation during biological sleep:

```python
class ConsolidationEngine:
    def __init__(self, min_episode_count: int = 3, consolidation_interval: int = 3600):
        self.min_count = min_episode_count
        self.interval = consolidation_interval

    async def consolidate_episodes(self, episode_batch: list[Episode]):
        """Extract semantic knowledge from episodic memories"""

        # Identify recurring entities across episodes
        entity_mentions = self.extract_entities(episode_batch)
        recurring = [e for e in entity_mentions if e.count >= self.min_count]

        for entity in recurring:
            # Abstract stable facts from episodic contexts
            facts = self.extract_stable_facts(entity, episode_batch)

            # Create or update semantic entity
            semantic_entity = await self.semantic_store.upsert(
                name=entity.name,
                entity_type=entity.type,
                facts=facts,
                source_episodes=[ep.id for ep in entity.episodes]
            )

            # Extract relationships between co-occurring entities
            for related in self.find_co_occurring(entity, episode_batch):
                strength = self.compute_association_strength(entity, related)
                await self.semantic_store.create_relationship(
                    semantic_entity, related, weight=strength
                )

    def extract_stable_facts(self, entity: EntityMention, episodes: list[Episode]) -> list[Fact]:
        """Identify facts that appear consistently across episodes"""
        fact_candidates = defaultdict(list)

        for ep in entity.episodes:
            facts = self.llm_extract_facts(entity.name, ep.content)
            for fact in facts:
                fact_candidates[fact.normalized].append(fact)

        # Keep facts that appear in majority of episodes
        threshold = len(entity.episodes) * 0.5
        return [facts[0] for key, facts in fact_candidates.items()
                if len(facts) >= threshold]
```

### FSRS Decay

```python
def fsrs_retrievability(t: float, S: float) -> float:
    """
    FSRS retrievability formula
    t: elapsed time (days)
    S: stability (days until R drops to 90%)
    """
    return (1 + 0.9 * t / S) ** (-0.5)

def update_stability_on_retrieval(memory, success: bool, S: float, D: float = 0.5):
    """Update stability after retrieval attempt"""
    if success:
        # Successful retrieval increases stability
        memory.stability = S * (1 + math.exp(D) * (11 - memory.difficulty) *
                                 (S ** -0.2) * (math.exp(0.1 * (1 - memory.retrievability)) - 1))
    else:
        # Failed retrieval decreases stability
        memory.stability = D * (S ** 0.2) * (math.exp(0.1 * memory.difficulty) - 1)
```

---

## 5. Multi-Instance Concurrent Access

### Session Namespacing

```
┌─────────────────────────────────────────────────────┐
│               MCP Memory Gateway                    │
├─────────────────────────────────────────────────────┤
│  CC Instance 1 ────┐                                │
│  CC Instance 2 ────┼──→ Session-Namespaced Episodes │
│  CC Instance N ────┘    + Shared Semantic/Procedural│
└─────────────────────────────────────────────────────┘
```

- **Episodic**: Session-scoped (each instance has own stream)
- **Semantic**: Shared (all instances see same knowledge)
- **Procedural**: Shared (skills available to all)

### CRDT Conflict Resolution (OR-Set)

```python
@dataclass
class EntityCRDT:
    name: str
    observations: ORSet[str]      # Add/remove with tombstones
    last_write: VectorClock       # Causal ordering

def merge_entities(local: EntityCRDT, remote: EntityCRDT) -> EntityCRDT:
    """Mathematically guaranteed convergence without coordination"""
    return EntityCRDT(
        name=local.name,
        observations=or_set_merge(local.observations, remote.observations),
        last_write=vector_clock_merge(local.last_write, remote.last_write)
    )
```

---

## 6. MCP Tool Interface

| Natural Language | MCP Tool | Memory System |
|------------------|----------|---------------|
| "remember this" | `create_episode` + `extract_entities` | Episodic → triggers consolidation |
| "recall X" | `semantic_search` + `actr_activation` | Semantic + Episodic hybrid |
| "how do I X" | `retrieve_procedure` | Procedural |
| "learn this skill" | `build_procedure` | Procedural |
| "X is related to Y" | `create_relationship` | Semantic graph edges |
| "forget X" | `soft_delete` + `accelerate_decay` | All (soft delete preferred) |

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [x] T4DX embedded storage (no external databases needed)
- [ ] BGE-M3 embedding service (local GPU)
- [ ] Basic MCP server skeleton

### Phase 2: Episodic Memory (Week 2-3)
- [ ] Episode creation and storage
- [ ] Bi-temporal versioning
- [ ] Retrieval with decay weighting
- [ ] Session namespacing

### Phase 3: Semantic Memory (Week 3-4)
- [ ] Entity extraction pipeline
- [ ] Knowledge graph creation
- [ ] Hebbian relationship strengthening
- [ ] ACT-R activation retrieval

### Phase 4: Procedural Memory (Week 4-5)
- [ ] Procedure storage schema
- [ ] Build from trajectories
- [ ] Retrieve by task similarity
- [ ] Update from feedback

### Phase 5: Consolidation (Week 5-6)
- [ ] Background episodic→semantic transfer
- [ ] Skill consolidation
- [ ] FSRS decay scheduling

### Phase 6: Multi-Instance (Week 6-7)
- [ ] CRDT implementation
- [ ] Concurrent access testing
- [ ] Conflict resolution verification

---

## Integration with T4DM

The tripartite memory system replaces the simplified hot/warm/cold model:

| WW Original | Tripartite Equivalent |
|-------------|----------------------|
| Hot (in-session) | Working Memory + Current Episodic |
| Warm (vector store) | Semantic Memory (knowledge graph) |
| Cold (persistent) | Consolidated Episodic + Procedural |

### New WW Agents

| Agent | Role |
|-------|------|
| ww-episodic | Episodic memory operations |
| ww-semantic | Semantic memory with Hebbian learning |
| ww-procedural | Skill storage and retrieval |
| ww-consolidate | Background memory consolidation |

---

## 7. Kappa-Gradient Consolidation (Systems Consolidation)

The discrete tripartite memory stores (episodic, semantic, procedural) are unified by a continuous consolidation gradient, kappa (kappa), inspired by systems consolidation theory.

### Theoretical Basis

Frankland & Bontempi (2005) demonstrated that memories undergo systems consolidation: initially dependent on hippocampus, they gradually become neocortical over time. This process is not a discrete transfer but a gradual shift in dependence.

T4DM models this with a continuous kappa in [0, 1]:

| kappa Range | State | Biological Analog |
|------------|-------|-------------------|
| 0.0 | Raw episodic (just encoded) | Hippocampus-dependent |
| ~0.15 | Replayed (NREM strengthened) | Early consolidation |
| ~0.4 | Transitional (being abstracted) | Hippocampal-neocortical dialogue |
| ~0.85 | Semantic concept (REM prototype) | Neocortex-dependent |
| 1.0 | Stable knowledge (fully consolidated) | Neocortical only |

### Mapping to T4DX LSM Compaction

The T4DX storage engine's compaction operations directly implement consolidation:

- **MemTable flush** = working memory to episodic (kappa = 0.0)
- **NREM compaction** = merge segments + kappa += 0.05 per replay + STDP weight updates
- **REM compaction** = cluster items + create prototypes (kappa += 0.2, item_type to semantic)
- **PRUNE compaction** = garbage-collect tombstoned + low-kappa items

This eliminates the need for cross-store transactions (the former Saga pattern). An item's transition from episodic to semantic is a field update, not a delete-and-reinsert.

### Key Advantage Over Discrete Stores

In the old architecture (Neo4j + Qdrant + Saga), moving a memory from episodic to semantic required:
1. Read from episodic store
2. Delete from episodic store
3. Insert into semantic store
4. Compensate on failure (Saga)

With kappa-gradient on T4DX, the same operation is:
1. `UPDATE_FIELDS(id, {kappa: 0.85, item_type: semantic})`

---

## References

- Tulving, E. (1972). Episodic and semantic memory
- Squire, L.R. (1992). Memory systems of the brain
- Anderson, J.R. (2007). ACT-R cognitive architecture
- **Frankland, P.W. & Bontempi, B. (2005). The organization of recent and remote memories. Nature Reviews Neuroscience, 6(2), 119-130.** -- Theoretical basis for kappa-gradient systems consolidation
- McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. Psychological Review, 102(3), 419-457.
- Zep Graphiti - 94.8% accuracy on DMR benchmarks
- Memp Framework (Zhejiang/Alibaba, 2025)
- FSRS - 20-30% improvement over SM-2
- Nature Human Behaviour (2024) - Generative models of memory
