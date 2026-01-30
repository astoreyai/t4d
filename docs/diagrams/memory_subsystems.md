# Memory Subsystems - Tripartite Architecture

## Memory System Interactions

This diagram illustrates the interactions between Episodic, Semantic, Procedural, and Working Memory subsystems, showing data flow and consolidation pathways.

```mermaid
graph TB
    subgraph "Working Memory (Prefrontal)"
        WM_GATE[Attentional Gate<br/>7±2 item capacity]
        WM_BUFFER[Active Buffer<br/>Current context]
        WM_REHEARSAL[Maintenance Rehearsal<br/>Keep-alive mechanism]
    end

    subgraph "Episodic Memory (Hippocampus)"
        EP_ENCODE[Episode Encoding<br/>T_ref & T_sys timestamps]
        EP_STORE[(Episode Store<br/>Neo4j temporal graph)]
        EP_RETRIEVE[Episode Retrieval<br/>Semantic + Recency + Outcome]
        EP_INDEX[Vector Index<br/>BGE-M3 1024-dim]
    end

    subgraph "Semantic Memory (Neocortex)"
        SEM_ENTITY[Entity Nodes<br/>CONCEPT, PERSON, PLACE, etc.]
        SEM_GRAPH[(Knowledge Graph<br/>Neo4j relationships)]
        SEM_HEBBIAN[Hebbian Strengthening<br/>w → w + α(1-w)]
        SEM_ACTR[ACT-R Retrieval<br/>Base-level + Spreading]
    end

    subgraph "Procedural Memory (Basal Ganglia)"
        PROC_BUILD[Build from Trajectory<br/>Extract steps + abstract]
        PROC_STORE[(Skill Storage<br/>Fine + coarse grain)]
        PROC_RETRIEVE[Skill Retrieval<br/>Task similarity match]
        PROC_UPDATE[Success Rate Update<br/>Execution feedback]
    end

    subgraph "Consolidation (Sleep)"
        CONS_NREM[NREM: SWR Replay<br/>Episode → Semantic transfer]
        CONS_REM[REM: Abstraction<br/>Cluster → Concept]
        CONS_SKILL[Skill Consolidation<br/>Successful trajectories]
    end

    %% Working Memory flows
    WM_GATE --> WM_BUFFER
    WM_BUFFER --> WM_REHEARSAL
    WM_BUFFER --> EP_ENCODE
    WM_BUFFER --> SEM_ENTITY

    %% Episodic Memory flows
    EP_ENCODE --> EP_STORE
    EP_ENCODE --> EP_INDEX
    EP_RETRIEVE --> EP_STORE
    EP_RETRIEVE --> EP_INDEX
    EP_RETRIEVE --> WM_BUFFER

    %% Semantic Memory flows
    SEM_ENTITY --> SEM_GRAPH
    SEM_ACTR --> SEM_GRAPH
    SEM_ACTR --> WM_BUFFER
    SEM_HEBBIAN --> SEM_GRAPH

    %% Procedural Memory flows
    PROC_BUILD --> PROC_STORE
    PROC_RETRIEVE --> PROC_STORE
    PROC_RETRIEVE --> WM_BUFFER
    PROC_UPDATE --> PROC_STORE

    %% Consolidation flows
    EP_STORE -.->|Replay| CONS_NREM
    CONS_NREM -.->|Transfer| SEM_GRAPH
    SEM_GRAPH -.->|Cluster| CONS_REM
    CONS_REM -.->|Abstract| SEM_ENTITY
    EP_STORE -.->|Successful| CONS_SKILL
    CONS_SKILL -.->|Distill| PROC_STORE

    %% Cross-retrieval flows
    EP_RETRIEVE -.->|Co-retrieval| SEM_HEBBIAN
    SEM_ACTR -.->|Context| EP_RETRIEVE
    PROC_RETRIEVE -.->|Example| EP_RETRIEVE

    style WM_GATE fill:#fff4e1
    style WM_BUFFER fill:#fff4e1
    style WM_REHEARSAL fill:#fff4e1
    style EP_ENCODE fill:#e8f5e9
    style EP_STORE fill:#e8f5e9
    style EP_RETRIEVE fill:#e8f5e9
    style EP_INDEX fill:#e8f5e9
    style SEM_ENTITY fill:#e1f5ff
    style SEM_GRAPH fill:#e1f5ff
    style SEM_HEBBIAN fill:#e1f5ff
    style SEM_ACTR fill:#e1f5ff
    style PROC_BUILD fill:#f3e5f5
    style PROC_STORE fill:#f3e5f5
    style PROC_RETRIEVE fill:#f3e5f5
    style PROC_UPDATE fill:#f3e5f5
    style CONS_NREM fill:#ffe0b2
    style CONS_REM fill:#ffe0b2
    style CONS_SKILL fill:#ffe0b2
```

## Memory Type Specifications

### Working Memory (7±2 items)

**Biological Analog**: Prefrontal cortex
**Purpose**: Short-term active maintenance of task-relevant information

| Property | Value | Description |
|----------|-------|-------------|
| Capacity | 7±2 items | Miller's Law limit |
| Retention | ~18 seconds | Without rehearsal |
| Encoding | Direct input | From perception or retrieval |
| Retrieval | Instant | Active in buffer |
| Eviction | FIFO + Priority | Oldest/lowest priority out |

**Key Features**:
- Attentional blink: 200-500ms refractory period after item insertion
- Maintenance rehearsal: Periodic refresh prevents decay
- Integration with all memory systems: Episodes, semantic entities, and skills

### Episodic Memory (Autobiographical Events)

**Biological Analog**: Hippocampus
**Purpose**: Store specific events with temporal-spatial context

| Property | Value | Description |
|----------|-------|-------------|
| Schema | Episode nodes | Neo4j with bi-temporal versioning |
| Embedding | BGE-M3 1024-dim | Semantic similarity search |
| Temporal | T_ref, T_sys | Event time vs system time |
| Retrieval | Multi-factor | 0.4×semantic + 0.25×recency + 0.2×outcome + 0.15×importance |
| Decay | FSRS | Power-law forgetting curve |

**Key Features**:
- Bi-temporal versioning enables "what did we know on date X"
- Outcome weighting: Successful episodes weighted 1.2×, failures 0.8×
- Session namespacing: Each client instance has isolated episode stream
- Consolidation: High-value episodes replayed during NREM sleep

### Semantic Memory (Generalized Knowledge)

**Biological Analog**: Neocortex
**Purpose**: Abstract, context-free knowledge derived from episodic experiences

| Property | Value | Description |
|----------|-------|-------------|
| Schema | Entity-Relationship | Neo4j labeled property graph |
| Entity Types | CONCEPT, PERSON, PLACE, EVENT, OBJECT, SKILL | Typed nodes |
| Relationships | RELATED_TO, USES, PRODUCES, CAUSES | Hebbian-weighted edges |
| Retrieval | ACT-R | Base-level activation + spreading |
| Strengthening | Hebbian | w → w + 0.1(1-w) on co-retrieval |

**Key Features**:
- Hebbian learning: "Fire together, wire together" - co-retrieval strengthens edges
- ACT-R activation: Combines recency/frequency (base-level) with spreading from context
- Fan-out normalization: Prevents hub nodes from dominating retrieval
- Abstraction from episodes: Consolidation extracts stable facts

### Procedural Memory (Skills & Workflows)

**Biological Analog**: Basal ganglia, cerebellum
**Purpose**: Store "how-to" knowledge - learned skills and action sequences

| Property | Value | Description |
|----------|-------|-------------|
| Schema | Procedure DAGs | Fine-grained steps + coarse scripts |
| Storage | Dual-format | Verbatim trajectory + abstract pattern |
| Lifecycle | Build-Retrieve-Update | Memp framework |
| Metrics | Success rate | Track execution outcomes |
| Versioning | Incremental | Reflect on failures, version updates |

**Key Features**:
- Build: Extract procedures from successful trajectories (outcome ≥ 0.7)
- Retrieve: Task similarity matching + success rate ranking
- Update: Learn from feedback, revise on repeated failures
- Deprecation: Mark as deprecated if success_rate < 0.3 after 10 executions

## Cross-System Interactions

### 1. Episodic → Semantic (Consolidation)
- NREM sleep replays high-value episodes
- Recurring entities across ≥3 episodes create semantic nodes
- Stable facts extracted from episodic contexts
- Co-occurring entities create Hebbian relationships

### 2. Semantic → Episodic (Retrieval Context)
- Spreading activation from semantic entities
- Provides context for episode retrieval
- Strengthens relationships on co-retrieval

### 3. Procedural → Episodic (Skill Building)
- Successful episode trajectories distilled into skills
- Examples linked to procedures for context
- Execution feedback updates episode outcomes

### 4. Working → All (Active Processing)
- Working memory items trigger retrievals
- Retrieved memories enter working memory
- Maintenance rehearsal keeps items active
- Overflow evicts least important items

## Retrieval Scoring Functions

### Episodic Retrieval
```
score = 0.4 × semantic_similarity
      + 0.25 × recency_weight
      + 0.2 × outcome_bonus
      + 0.15 × importance

recency_weight = exp(-0.1 × days_elapsed)
outcome_bonus = 1.2 if success else 0.8
importance = emotional_valence ∈ [0, 1]
```

### Semantic Retrieval (ACT-R)
```
Activation(i) = BaseLevel(i) + Spreading(i) + Noise

BaseLevel(i) = ln(Σⱼ tⱼ^(-0.5))  # Power-law decay
Spreading(i) = Σₛ W(s) × (1.6 - ln(fan(s)))  # From context
Noise ~ Gaussian(0, 0.5)  # Stochastic variability
```

### Procedural Retrieval
```
score = 0.6 × semantic_similarity(task, script)
      + 0.4 × success_rate

Filter: NOT deprecated AND execution_count > 0
Rank: By score descending
```

## Consolidation Timing

| Phase | Trigger | Duration | Operations |
|-------|---------|----------|------------|
| NREM | Manual/scheduled | ~75% of cycle | SWR replay, E→S transfer |
| REM | After NREM | ~25% of cycle | Clustering, abstraction |
| Prune | End of cycle | Variable | Synaptic downscaling |

## Performance Characteristics

| Memory Type | Avg Retrieval Latency | Storage Size | Forgetting Curve |
|-------------|----------------------|--------------|------------------|
| Working | <1ms | 7±2 items | 18s decay |
| Episodic | 50-200ms | Unbounded | FSRS power-law |
| Semantic | 10-50ms | Unbounded | ACT-R activation |
| Procedural | 20-100ms | Unbounded | Usage-based |
