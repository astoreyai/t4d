# T4DM - Agents & Skills Catalog

**Version**: 0.1.0-alpha
**Last Updated**: 2025-11-27

---

## Agent Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATION TIER                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   ww-init       │  │  ww-conductor   │  │  ww-session     │         │
│  │   (Bootstrap)   │  │  (Orchestrator) │  │  (Context Mgr)  │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
├─────────────────────────────────────────────────────────────────────────┤
│                           MEMORY TIER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  ww-memory      │  │  ww-semantic    │  │  ww-graph       │         │
│  │  (Storage)      │  │  (Embeddings)   │  │  (Relationships)│         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
├─────────────────────────────────────────────────────────────────────────┤
│                          KNOWLEDGE TIER                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  ww-knowledge   │  │  ww-retriever   │  │  ww-synthesizer │         │
│  │  (Capture)      │  │  (Search)       │  │  (Integration)  │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
├─────────────────────────────────────────────────────────────────────────┤
│                           DOMAIN TIER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  ww-neuro       │  │  ww-compbio     │  │  ww-algorithm   │         │
│  │  (Neuroscience) │  │  (Comp Biology) │  │  (Algorithm)    │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
├─────────────────────────────────────────────────────────────────────────┤
│                          WORKFLOW TIER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  ww-planner     │  │  ww-finetune    │  │  ww-validator   │         │
│  │  (Task Plan)    │  │  (Model Tune)   │  │  (Verification) │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## TIER 1: ORCHESTRATION AGENTS

### 1.1 ww-init (Initializer Agent)

**Purpose**: Bootstrap new projects/sessions following Anthropic's initializer pattern

**Capabilities**:
- Generate comprehensive feature lists from high-level specs
- Create progress tracking files (`ww-progress.json`)
- Set up environment scripts (`init.sh`)
- Establish verification test suites
- Initialize git repository with proper structure

**Tools Available**:
- Read, Write, Edit, Bash, Glob, Grep
- TodoWrite (for feature decomposition)

**When to Use**:
- First run on any new project
- When resetting/restarting a project
- When scope significantly changes

**Inputs**:
```json
{
  "project_spec": "High-level description of what to build",
  "project_type": "assistant|knowledge_base|workflow|domain_specific",
  "constraints": ["provider_agnostic", "claude_code_compatible"]
}
```

**Outputs**:
- `ww-progress.json` - Progress tracking file
- `ww-features.json` - Feature requirements (all marked `passes: false`)
- `init.sh` - Environment bootstrap script
- Initial git commit

**Agent Definition**:
```yaml
name: ww-init
description: Bootstrap T4DM projects with proper structure, progress tracking, and feature lists following Anthropic's initializer pattern
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
  - TodoWrite
```

---

### 1.2 ww-conductor (Orchestration Agent)

**Purpose**: Route tasks to appropriate agents, manage multi-agent workflows

**Capabilities**:
- Analyze incoming requests and route to correct agent(s)
- Coordinate multi-agent task sequences
- Manage parallel agent execution
- Aggregate results from multiple agents
- Handle agent failures and retries

**Tools Available**:
- Task (spawn sub-agents)
- Read, Write (state files)
- TodoWrite (workflow tracking)

**When to Use**:
- Complex requests requiring multiple agents
- Ambiguous requests needing classification
- Multi-step workflows
- When coordination between agents needed

**Routing Logic**:
```
Request Analysis → Agent Selection → Execution Plan → Parallel/Sequential → Aggregation
```

**Decision Matrix**:
| Request Type | Primary Agent | Supporting Agents |
|-------------|---------------|-------------------|
| "Store this knowledge" | ww-knowledge | ww-semantic, ww-memory |
| "Find information about X" | ww-retriever | ww-semantic, ww-graph |
| "Analyze neural pathway" | ww-neuro | ww-graph, ww-knowledge |
| "Design algorithm for X" | ww-algorithm | ww-graph, ww-validator |
| "Fine-tune model for X" | ww-finetune | ww-knowledge, ww-validator |

**Agent Definition**:
```yaml
name: ww-conductor
description: Orchestrate multi-agent workflows, route requests to appropriate agents, coordinate parallel execution
tools:
  - Task
  - Read
  - Write
  - TodoWrite
  - AskUserQuestion
```

---

### 1.3 ww-session (Session Manager Agent)

**Purpose**: Manage context bridging across sessions (Anthropic's "getting bearings" pattern)

**Capabilities**:
- Load previous session state
- Summarize recent progress
- Identify current focus and blockers
- Restore working context
- Generate session handoff summaries

**Session Start Sequence**:
1. `pwd` - Verify working directory
2. Read `ww-progress.json` - Recent work summary
3. Read `ww-features.json` - Current feature status
4. `git log --oneline -10` - Recent commits
5. Run verification tests - Check for regressions
6. Identify next task - Based on priority

**Session End Sequence**:
1. Summarize completed work
2. Update progress file
3. Update feature status
4. Commit changes with descriptive message
5. Document blockers/next steps

**Agent Definition**:
```yaml
name: ww-session
description: Manage session lifecycle - context loading at start, state persistence at end, progress tracking throughout
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
```

---

## TIER 2: MEMORY AGENTS

### 2.1 ww-memory (Storage Agent)

**Purpose**: Unified interface to all storage backends (hot/warm/cold)

**Capabilities**:
- Store documents with metadata
- Retrieve by ID or query
- Manage TTL and caching
- Handle storage provider abstraction
- Coordinate between storage tiers

**Storage Tiers**:
```
HOT (ms)     → In-memory dict, LRU cache
WARM (sec)   → Vector store (ChromaDB/FAISS)
COLD (min)   → SQLite/PostgreSQL, Document store
```

**Operations**:
| Operation | Description |
|-----------|-------------|
| `store(doc, tier)` | Store document at specified tier |
| `retrieve(id)` | Get document by ID (checks all tiers) |
| `query(embedding, k)` | Semantic search across warm storage |
| `promote(id)` | Move document to hotter tier |
| `demote(id)` | Move document to colder tier |
| `expire(id)` | Remove from hot, keep in cold |

**Agent Definition**:
```yaml
name: ww-memory
description: Manage multi-tier storage - in-memory cache, vector stores, and persistent databases with automatic tiering
tools:
  - Read
  - Write
  - Bash
```

---

### 2.2 ww-semantic (Embedding Agent)

**Purpose**: Generate and manage embeddings, semantic similarity operations

**Capabilities**:
- Text-to-embedding conversion (provider-agnostic)
- Batch embedding generation
- Similarity computation
- Clustering and categorization
- Embedding caching and versioning

**Supported Providers**:
| Provider | Model | Dimensions | Use Case |
|----------|-------|------------|----------|
| Voyage AI | voyage-3 | 1024 | High quality, general |
| OpenAI | text-embedding-3-large | 3072 | High capacity |
| Cohere | embed-v3 | 1024 | Multilingual |
| Local | all-MiniLM-L6-v2 | 384 | Offline, fast |

**Operations**:
| Operation | Description |
|-----------|-------------|
| `embed(texts)` | Generate embeddings for text list |
| `embed_query(query)` | Query-optimized embedding |
| `similarity(a, b)` | Cosine similarity between embeddings |
| `cluster(embeddings, k)` | K-means clustering |
| `reduce(embeddings, dim)` | Dimensionality reduction (UMAP/PCA) |

**Agent Definition**:
```yaml
name: ww-semantic
description: Provider-agnostic embedding generation, similarity computation, and semantic clustering
tools:
  - Read
  - Write
  - Bash
```

---

### 2.3 ww-graph (Graph Agent)

**Purpose**: Manage knowledge graphs, entity relationships, graph algorithms

**Capabilities**:
- Entity extraction and linking
- Relationship detection and typing
- Graph construction and updates
- Path finding and traversal
- Community detection
- Graph querying (Cypher-like)

**Graph Schema**:
```
Nodes: Entity, Concept, Document, Agent, Task
Edges: RELATES_TO, DERIVED_FROM, DEPENDS_ON, CREATED_BY, PART_OF
```

**Operations**:
| Operation | Description |
|-----------|-------------|
| `add_node(entity, type, props)` | Add entity to graph |
| `add_edge(src, dst, type, props)` | Create relationship |
| `find_path(src, dst, max_hops)` | Shortest path between entities |
| `neighbors(entity, depth)` | Get connected entities |
| `subgraph(entities)` | Extract subgraph |
| `communities()` | Detect clusters |

**Agent Definition**:
```yaml
name: ww-graph
description: Knowledge graph management - entity extraction, relationship detection, graph traversal and querying
tools:
  - Read
  - Write
  - Bash
  - Grep
```

---

### 2.4 ww-episodic (Episodic Memory Agent)

**Purpose**: Store and retrieve autobiographical events with temporal-spatial context

**Cognitive Foundation**: Episodic memory (Tulving, 1972) preserves particularity - "what happened when" rather than abstracted facts.

**Capabilities**:
- Store episodes with bi-temporal versioning (T_ref, T_sys)
- Decay-weighted retrieval (FSRS algorithm)
- Session namespacing for multi-instance access
- Point-in-time historical queries
- Outcome and importance tracking

**Episode Schema**:
| Field | Type | Description |
|-------|------|-------------|
| content | string | Full interaction text |
| embedding | vector | 1024-dim BGE-M3 |
| timestamp | datetime | When event occurred (T_ref) |
| ingestedAt | datetime | When memory created (T_sys) |
| context | object | Project, file, tool, cwd |
| outcome | enum | success/failure/partial/neutral |
| valence | float | Importance signal [0,1] |
| stability | float | FSRS stability (days) |

**Retrieval Formula**:
```
score = 0.4*semantic + 0.25*recency + 0.2*outcome + 0.15*importance
```

**Agent Definition**:
```yaml
name: ww-episodic
description: Autobiographical memory with bi-temporal versioning and FSRS decay
tools:
  - Read
  - Write
  - Bash
  - Grep
  - Glob
```

---

### 2.5 ww-semantic-mem (Semantic Memory Agent)

**Purpose**: Hebbian-weighted knowledge graph with ACT-R activation-based retrieval

**Cognitive Foundation**: Semantic memory stores context-free knowledge abstracted from episodes - "what I know" rather than "what happened."

**Capabilities**:
- Store entities with decay properties
- Hebbian-weighted relationships (strengthened on co-retrieval)
- ACT-R activation-based retrieval
- Spreading activation through graph
- Bi-temporal fact versioning

**Entity Types**:
| Type | Description |
|------|-------------|
| CONCEPT | Abstract idea (e.g., "Hebbian learning") |
| PERSON | Individual (e.g., "Tulving") |
| PROJECT | Work item (e.g., "T4DM") |
| TOOL | Software/utility (e.g., "Neo4j") |
| TECHNIQUE | Method/approach (e.g., "FSRS decay") |
| FACT | Discrete knowledge (e.g., "BGE-M3 uses 1024 dims") |

**Hebbian Strengthening**:
```
w' = w + learning_rate * (1 - w)
```
Bounded update approaching 1.0 asymptotically.

**ACT-R Activation**:
```
A = B + sum(W * S) + noise
```
Base-level activation + spreading activation + noise.

**Agent Definition**:
```yaml
name: ww-semantic-mem
description: Hebbian-weighted knowledge graph with ACT-R activation retrieval
tools:
  - Read
  - Write
  - Bash
  - Grep
  - Glob
```

---

### 2.6 ww-procedural (Procedural Memory Agent)

**Purpose**: Store learned skills with Memp build-retrieve-update lifecycle

**Cognitive Foundation**: Procedural memory stores "how-to" knowledge that becomes automatic with practice.

**Capabilities**:
- Dual-format storage (fine-grained steps + abstract script)
- Build procedures from successful trajectories
- Retrieve matching procedures by task
- Update based on execution feedback
- Deprecate consistently failing procedures

**Procedure Schema**:
| Field | Type | Description |
|-------|------|-------------|
| name | string | Procedure identifier |
| domain | enum | coding/research/trading/devops/writing |
| triggerPattern | string | When to invoke |
| steps | array | Fine-grained action sequence |
| script | string | High-level abstraction |
| successRate | float | Execution success tracking |
| executionCount | int | Times executed |

**Memp Lifecycle**:
| Phase | Action |
|-------|--------|
| BUILD | Distill successful trajectory into procedure (score >= 0.7) |
| RETRIEVE | Match task to stored procedures by similarity + success rate |
| UPDATE | Reinforce on success, reflect and revise on failure |

**Agent Definition**:
```yaml
name: ww-procedural
description: Memp-based skill storage with build-retrieve-update lifecycle
tools:
  - Read
  - Write
  - Bash
  - Grep
  - Glob
```

---

### 2.7 ww-consolidate (Consolidation Engine)

**Purpose**: Transform episodic experiences into semantic knowledge, consolidate skills

**Cognitive Foundation**: Mimics biological sleep-phase consolidation where hippocampal (episodic) memories transfer to neocortical (semantic) storage.

**Capabilities**:
- Episodic→Semantic transfer (semanticization)
- Pattern extraction from successful trajectories
- Skill consolidation (merge similar procedures)
- Provenance chain maintenance
- Scheduled consolidation cycles

**Consolidation Types**:
| Type | Trigger | Threshold |
|------|---------|-----------|
| Episodic→Semantic | Similar episodes | 3+, >0.75 similarity |
| Pattern extraction | Successful patterns | 3+ successes, >0.8 rate |
| Skill merge | Similar procedures | 2+ procs, >0.85 similarity |

**Schedule**:
| Cycle | Frequency | Purpose |
|-------|-----------|---------|
| Light | 2-4 hours | Quick pattern check |
| Deep | Daily | Full transfer, weight decay |
| Skill | Weekly | Procedural optimization |

**Agent Definition**:
```yaml
name: ww-consolidate
description: Memory consolidation engine for episodic->semantic transfer and skill merging
tools:
  - Read
  - Write
  - Bash
  - Grep
  - Glob
```

---

## TIER 3: KNOWLEDGE AGENTS

### 3.1 ww-knowledge (Knowledge Capture Agent)

**Purpose**: Extract, structure, and store knowledge from various sources

**Capabilities**:
- Extract knowledge from conversations
- Parse documents into knowledge units
- Apply appropriate schemas/templates
- Generate metadata and tags
- Link to existing knowledge

**Knowledge Types**:
| Type | Description | Template |
|------|-------------|----------|
| Concept | Definition/explanation | concept.json |
| Procedure | Step-by-step process | procedure.json |
| Fact | Discrete piece of information | fact.json |
| Relationship | Connection between entities | relationship.json |
| Decision | Choice with rationale | decision.json |

**Extraction Pipeline**:
```
Source → Parse → Chunk → Classify → Schema → Embed → Store → Link
```

**Agent Definition**:
```yaml
name: ww-knowledge
description: Extract, structure, and store knowledge from conversations and documents with proper schemas and linking
tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Task (ww-semantic, ww-graph)
```

---

### 3.2 ww-retriever (Retrieval Agent)

**Purpose**: Find and retrieve relevant knowledge using multiple strategies

**Capabilities**:
- Semantic search (embedding similarity)
- Keyword search (BM25)
- Hybrid search (RRF fusion)
- Graph-augmented retrieval
- Re-ranking and filtering

**Retrieval Strategies**:
| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Semantic | Embedding similarity | Conceptual queries |
| Keyword | BM25 text match | Specific terms |
| Hybrid | RRF(semantic, keyword) | General queries |
| Graph | Follow relationships | Connected knowledge |
| Multi-hop | Chain retrievals | Complex questions |

**Operations**:
| Operation | Description |
|-----------|-------------|
| `search(query, strategy, k)` | Retrieve top-k results |
| `rerank(query, results)` | Re-score results |
| `expand(query)` | Query expansion |
| `filter(results, criteria)` | Apply metadata filters |

**Agent Definition**:
```yaml
name: ww-retriever
description: Multi-strategy knowledge retrieval - semantic search, keyword matching, hybrid fusion, and graph traversal
tools:
  - Read
  - Grep
  - Glob
  - Task (ww-semantic, ww-graph)
```

---

### 3.3 ww-synthesizer (Synthesis Agent)

**Purpose**: Integrate multiple knowledge sources into coherent outputs

**Capabilities**:
- Combine information from multiple retrievals
- Resolve conflicts between sources
- Generate summaries and overviews
- Create structured reports
- Answer complex questions

**Synthesis Modes**:
| Mode | Description |
|------|-------------|
| Aggregate | Combine all relevant information |
| Compare | Contrast multiple perspectives |
| Summarize | Condense to key points |
| Explain | Generate explanatory narrative |
| Report | Structured document output |

**Agent Definition**:
```yaml
name: ww-synthesizer
description: Integrate knowledge from multiple sources, resolve conflicts, generate summaries and structured reports
tools:
  - Read
  - Write
  - Task (ww-retriever)
```

---

## TIER 4: DOMAIN AGENTS

### 4.1 ww-neuro (Neuroscience Agent)

**Purpose**: Specialized assistant for neuroscience research and modeling

**Capabilities**:
- Neural pathway analysis
- Brain region mapping
- Cognitive process modeling
- Literature synthesis (neuroscience)
- Experimental design support

**Domain Knowledge**:
- Brain anatomy (regions, connections)
- Neurotransmitter systems
- Cognitive functions
- Research methodologies
- Common datasets (HCP, ABCD, etc.)

**Integrations**:
- Allen Brain Atlas API
- NeuroSynth
- PubMed (neuroscience subset)

**Agent Definition**:
```yaml
name: ww-neuro
description: Neuroscience research assistant - neural pathway analysis, brain mapping, cognitive modeling, literature synthesis
tools:
  - Read
  - Write
  - WebFetch
  - WebSearch
  - Task (ww-knowledge, ww-graph)
```

---

### 4.2 ww-compbio (Computational Biology Agent)

**Purpose**: Specialized assistant for computational biology and bioinformatics

**Capabilities**:
- Sequence analysis
- Protein structure prediction
- Gene expression analysis
- Pathway modeling
- Literature synthesis (biology)

**Domain Knowledge**:
- Molecular biology
- Genomics and proteomics
- Metabolic pathways
- Bioinformatics tools
- Common databases (UniProt, NCBI, etc.)

**Integrations**:
- UniProt API
- NCBI Entrez
- KEGG Pathways
- PDB

**Agent Definition**:
```yaml
name: ww-compbio
description: Computational biology assistant - sequence analysis, protein structure, pathway modeling, bioinformatics
tools:
  - Read
  - Write
  - Bash
  - WebFetch
  - WebSearch
  - Task (ww-knowledge, ww-graph)
```

---

### 4.3 ww-algorithm (Algorithm Design Agent)

**Purpose**: Design, analyze, and implement algorithms

**Capabilities**:
- Algorithm design from requirements
- Complexity analysis (time/space)
- Optimization strategies
- Graph algorithm specialization
- Formal correctness reasoning

**Algorithm Categories**:
| Category | Examples |
|----------|----------|
| Graph | DFS, BFS, Dijkstra, A*, PageRank |
| Optimization | Gradient descent, simulated annealing |
| Search | Binary search, hash tables, tries |
| ML | Backprop, attention, clustering |
| Numerical | Matrix ops, FFT, integration |

**Design Process**:
```
Requirements → Approach Selection → Pseudocode → Analysis → Implementation → Testing
```

**Agent Definition**:
```yaml
name: ww-algorithm
description: Algorithm design and analysis - complexity analysis, optimization, graph algorithms, formal reasoning
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Task (ww-validator)
```

---

## TIER 5: WORKFLOW AGENTS

### 5.1 ww-planner (Planning Agent)

**Purpose**: Decompose complex tasks into actionable plans

**Capabilities**:
- Task decomposition
- Dependency analysis
- Priority assignment
- Resource estimation
- Risk identification

**Planning Methodology**:
```
Goal → Decompose → Dependencies → Sequence → Assign → Track
```

**Output Format**:
```json
{
  "goal": "High-level objective",
  "tasks": [
    {
      "id": "T1",
      "description": "Task description",
      "agent": "ww-agent-name",
      "dependencies": [],
      "priority": 1,
      "estimated_effort": "small|medium|large"
    }
  ],
  "critical_path": ["T1", "T3", "T5"],
  "risks": ["Risk description"]
}
```

**Agent Definition**:
```yaml
name: ww-planner
description: Task decomposition and planning - break complex goals into actionable tasks with dependencies and priorities
tools:
  - Read
  - Write
  - TodoWrite
  - Task (ww-conductor)
```

---

### 5.2 ww-finetune (Fine-Tuning Orchestrator)

**Purpose**: Manage model fine-tuning workflows

**Capabilities**:
- Dataset preparation and validation
- Training configuration generation
- Progress monitoring
- Evaluation and comparison
- Model versioning

**Workflow Stages**:
```
Data Prep → Validation → Config → Training → Evaluation → Deployment
```

**Supported Frameworks**:
- Hugging Face Transformers
- OpenAI Fine-tuning API
- Anthropic (when available)
- Local (LoRA, QLoRA)

**Agent Definition**:
```yaml
name: ww-finetune
description: Model fine-tuning orchestration - dataset prep, training config, monitoring, evaluation, versioning
tools:
  - Read
  - Write
  - Bash
  - Task (ww-validator)
```

---

### 5.3 ww-validator (Validation Agent)

**Purpose**: Verify correctness, quality, and completeness

**Capabilities**:
- Test execution and reporting
- Code quality checks
- Output validation
- Regression detection
- Coverage analysis

**Validation Types**:
| Type | Description |
|------|-------------|
| Unit | Individual component tests |
| Integration | Cross-component tests |
| E2E | End-to-end workflows |
| Quality | Linting, type checking |
| Semantic | Output correctness |

**Agent Definition**:
```yaml
name: ww-validator
description: Validation and verification - test execution, quality checks, regression detection, coverage analysis
tools:
  - Read
  - Bash
  - Grep
  - Glob
```

---

## SKILLS CATALOG

Skills are user-facing workflows that orchestrate agents for specific tasks.

### Skill: ww-capture

**Trigger**: "Save this to knowledge base", "Remember this"

**Workflow**:
1. Extract key information from context
2. Classify knowledge type
3. Generate embeddings (ww-semantic)
4. Store in memory (ww-memory)
5. Update graph (ww-graph)
6. Confirm to user

---

### Skill: ww-recall

**Trigger**: "What do I know about X", "Find information on Y"

**Workflow**:
1. Parse query intent
2. Generate query embedding (ww-semantic)
3. Retrieve candidates (ww-retriever)
4. Expand via graph (ww-graph)
5. Synthesize response (ww-synthesizer)
6. Present to user

---

### Skill: ww-project-init

**Trigger**: "Start new project", "Initialize WW for X"

**Workflow**:
1. Gather project requirements
2. Bootstrap environment (ww-init)
3. Generate feature list
4. Create progress tracking
5. Initial commit
6. Report status

---

### Skill: ww-session-start

**Trigger**: Session start hook

**Workflow**:
1. Load session state (ww-session)
2. Summarize recent progress
3. Identify current focus
4. Check for blockers
5. Display status

---

### Skill: ww-session-end

**Trigger**: Session end hook

**Workflow**:
1. Summarize session work
2. Update progress file
3. Update feature status
4. Commit changes
5. Generate handoff notes

---

### Skill: ww-design-algorithm

**Trigger**: "Design algorithm for X", "How would I implement Y"

**Workflow**:
1. Gather requirements
2. Research existing approaches (ww-retriever)
3. Design algorithm (ww-algorithm)
4. Analyze complexity
5. Generate implementation
6. Validate (ww-validator)

---

### Skill: ww-research-neuro

**Trigger**: "Research neural X", "Analyze brain region Y"

**Workflow**:
1. Parse research question
2. Search literature (ww-neuro)
3. Query knowledge base (ww-retriever)
4. Build relationship map (ww-graph)
5. Synthesize findings (ww-synthesizer)
6. Present results

---

### Skill: ww-research-bio

**Trigger**: "Analyze protein X", "Research pathway Y"

**Workflow**:
1. Parse research question
2. Query databases (ww-compbio)
3. Search literature
4. Map relationships (ww-graph)
5. Synthesize findings (ww-synthesizer)
6. Present results

---

## AGENT COMMUNICATION PROTOCOL

### Message Format

```json
{
  "id": "msg-uuid",
  "timestamp": "2025-11-27T10:00:00Z",
  "from": "ww-conductor",
  "to": "ww-semantic",
  "type": "request|response|event",
  "action": "embed",
  "payload": {
    "texts": ["text to embed"]
  },
  "context": {
    "session_id": "session-uuid",
    "task_id": "task-uuid",
    "parent_msg_id": "parent-uuid"
  }
}
```

### Interaction Patterns

**1. Request-Response**
```
Conductor → Agent: Request
Agent → Conductor: Response
```

**2. Pipeline**
```
A → B → C → D (sequential processing)
```

**3. Fan-Out/Fan-In**
```
Conductor → [A, B, C] (parallel)
[A, B, C] → Conductor (aggregate)
```

**4. Event-Driven**
```
Agent → Event Bus → Subscribers
```

---

## DEPENDENCY GRAPH

```
                    ┌─────────────┐
                    │ ww-conductor│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  ┌───────────┐     ┌───────────┐     ┌───────────┐
  │ ww-session│     │ ww-planner│     │ ww-init   │
  └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
        │                 │                 │
        └────────┬────────┴────────┬────────┘
                 │                 │
                 ▼                 ▼
          ┌───────────┐     ┌───────────┐
          │ww-knowledge│     │ww-retriever│
          └─────┬─────┘     └─────┬─────┘
                │                 │
        ┌───────┴───────┬─────────┴───────┐
        │               │                 │
        ▼               ▼                 ▼
  ┌───────────┐  ┌───────────┐     ┌───────────┐
  │ ww-memory │  │ww-semantic│     │ ww-graph  │
  └───────────┘  └───────────┘     └───────────┘
```

---

## IMPLEMENTATION PRIORITY

### Phase 1: Core Infrastructure
1. ww-memory (storage foundation)
2. ww-semantic (embedding foundation)
3. ww-session (context management)

### Phase 2: Knowledge Layer
4. ww-knowledge (capture)
5. ww-retriever (search)
6. ww-graph (relationships)

### Phase 3: Orchestration
7. ww-init (bootstrap)
8. ww-conductor (routing)
9. ww-synthesizer (integration)

### Phase 4: Workflow
10. ww-planner (decomposition)
11. ww-validator (verification)

### Phase 5: Domain
12. ww-algorithm (design)
13. ww-neuro (neuroscience)
14. ww-compbio (biology)
15. ww-finetune (training)

---

## NEXT: Create Individual Agent Specifications

Each agent needs:
1. Detailed SKILL.md for Claude Code
2. Agent definition for Claude Agent SDK
3. Tool configurations
4. Test cases
5. Integration examples
