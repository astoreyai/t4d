# T4DM Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    API / Gateway Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  REST API       │  │  MCP Gateway    │  │  Python SDK         │  │
│  │  (FastAPI)      │  │  (17 tools)     │  │  (sync/async)       │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────────┘  │
└───────────┼────────────────────┼────────────────────┼───────────────┘
            └────────────────────┼────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────────┐
│                        Hook Layer (Pre/On/Post)                      │
│  CachingRecallHook │ ValidationHook │ AuditTrailHook │ HebbianHook  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────────┐
│                        Memory Layer                                  │
├─────────────────┬─────────────────┬──────────────────────────────────┤
│   Episodic      │    Semantic     │       Procedural                 │
│  (FSRS decay)   │ (ACT-R, Hebb)   │    (skill matching)              │
│  - Episodes     │ - Entities      │    - Procedures                  │
│  - Context      │ - Relationships │    - Success tracking            │
│  - Outcomes     │ - Spreading     │    - Version control             │
└────────┬────────┴────────┬────────┴────────┬─────────────────────────┘
         │                 │                 │
┌────────┴─────────────────┴─────────────────┴─────────────────────────┐
│                       Learning Layer                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────────────┐    │
│  │  Dopamine System        │  │  Serotonin System               │    │
│  │  - Reward prediction    │  │  - Eligibility traces           │    │
│  │  - Surprise signals     │  │  - Long-term credit             │    │
│  │  - TD learning          │  │  - Mood adaptation              │    │
│  └─────────────────────────┘  └─────────────────────────────────┘    │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────┴─────────────────────────────────────┐
│                         Storage Layer                                 │
├──────────────────────────┬───────────────────────────────────────────┤
│   Qdrant (vectors)       │       Neo4j (graph)                       │
│   - Episode embeddings   │       - Entity relationships              │
│   - Entity embeddings    │       - Procedure steps                   │
│   - Skill embeddings     │       - Temporal links                    │
└──────────────────────────┴───────────────────────────────────────────┘
                                 │
┌────────────────────────────────┴─────────────────────────────────────┐
│                      Observability Layer                              │
│  WWObserver │ Event Tracing │ Citation Extraction │ Stats Collection │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. MCP Gateway (`/mnt/projects/t4d/t4dm/src/t4dm/mcp/`)

**Purpose**: Expose T4DM memory system to Claude Code via MCP protocol.

#### Files
- **memory_gateway.py** (450+ lines): FastMCP server with 17 tools
- **validation.py**: Input sanitization (UUIDs, enums, ranges)
- **errors.py**: Standardized error codes (NOT_FOUND, INVALID_INPUT, RATE_LIMITED)
- **types.py**: TypedDict response structures
- **schema.py**: OpenAPI schema generation

#### Security Features
- **Rate Limiting**: 100 requests/60s per session (sliding window)
- **Input Validation**: UUID format, enum values, numeric ranges
- **Error Standardization**: Machine-readable error codes

#### 17 MCP Tools

**Episodic Memory**:
1. `remember_episode` - Store new autobiographical event
2. `recall_episodes` - Search episodes with FSRS decay scoring
3. `get_episode` - Retrieve single episode by ID
4. `query_at_time` - Historical queries (bi-temporal)
5. `mark_important` - Increase emotional valence

**Semantic Memory**:
6. `create_entity` - Create knowledge entity (concept, person, project, tool)
7. `create_relationship` - Link entities with Hebbian weights
8. `recall_entities` - Search with ACT-R activation
9. `get_entity` - Retrieve single entity
10. `supersede_entity` - Update with versioning
11. `spread_activation` - Propagate activation through graph

**Procedural Memory**:
12. `store_procedure` - Save skill/workflow
13. `retrieve_procedures` - Match skills to task
14. `record_execution` - Update success rate
15. `deprecate_procedure` - Mark obsolete

**Consolidation**:
16. `consolidate_memory` - Run consolidation (light/deep/skill/all)

**Metadata**:
17. `get_memory_stats` - System statistics

---

### 2. Memory Layer (`/mnt/projects/t4d/t4dm/src/t4dm/memory/`)

#### Episodic Memory (`episodic.py`)
**Responsibilities**:
- Store autobiographical events with temporal-spatial context
- Apply FSRS decay during retrieval
- Track access patterns for stability updates

**Key Methods**:
- `create(content, context, outcome, valence)` → Episode
- `recall(query, limit, time_filters)` → list[ScoredResult]
- `query_at_time(query, point_in_time)` → list[ScoredResult]
- `mark_important(episode_id, new_valence)` → Episode

**Data Flow**:
```
User input → Embed (BGE-M3) → Store (Qdrant + Neo4j)
Query → Embed → Vector search → FSRS scoring → Results
```

#### Semantic Memory (`semantic.py`)
**Responsibilities**:
- Manage abstracted knowledge entities
- Maintain Hebbian-weighted relationship graph
- Calculate ACT-R activation with spreading

**Key Methods**:
- `create_entity(name, type, summary)` → Entity
- `create_relationship(source, target, type, weight)` → Relationship
- `recall(query, context_entities, limit)` → list[ScoredResult]
- `supersede(entity_id, new_summary)` → Entity (versioning)
- `spread_activation(seeds, steps, decay)` → dict[str, float]

**Optimizations**:
- Batch relationship queries (lines 320-358)
- Parallel Hebbian strengthening (lines 431-445)
- Pre-loaded context cache for spreading activation

**Data Flow**:
```
Entity creation → Embed → Store
Recall → Embed query → Vector search → ACT-R activation →
  Spreading (if context) → Hebbian strengthening → Results
```

#### Procedural Memory (`procedural.py`)
**Responsibilities**:
- Store learned skills and workflows
- Track success rates and execution history
- Support version control and deprecation

**Key Methods**:
- `store(name, script, steps, domain)` → Procedure
- `retrieve(task, domain, limit)` → list[ScoredResult]
- `record_execution(procedure_id, success, duration)` → Procedure
- `deprecate(procedure_id, reason, consolidated_into)` → Procedure

**Data Flow**:
```
Procedure → Embed script → Store
Task → Embed → Vector search → Success rate scoring → Results
Execution → Update stats → Recalculate success rate
```

---

### 3. Storage Layer (`/mnt/projects/t4d/t4dm/src/t4dm/storage/`)

#### Qdrant (`t4dx_vector_adapter.py`)
**Purpose**: Vector similarity search for embeddings

**Collections**:
- `episodes`: 1024-dim BGE-M3 embeddings
- `entities`: 1024-dim BGE-M3 embeddings
- `procedures`: 1024-dim BGE-M3 embeddings

**Key Operations**:
- `add(collection, ids, vectors, payloads)` - Bulk insert
- `search(collection, vector, limit, filter)` - Cosine similarity
- `get(collection, ids)` - Fetch by ID
- `update_payload(collection, id, payload)` - Partial update

**Filtering**:
- Session isolation: `filter={"session_id": "..."}`
- Timestamp ranges: `filter={"timestamp": {"gte": "...", "lte": "..."}}`

**Connection**:
- Host: `localhost:6333`
- Async client with connection pooling

#### Neo4j (`t4dx_graph_adapter.py`)
**Purpose**: Graph queries for relationships and provenance

**Node Labels**:
- `Episode`: Episodic events
- `Entity`: Semantic knowledge
- `Procedure`: Skills/workflows

**Relationship Types**:
- `USES`: Entity uses another entity
- `REQUIRES`: Dependency relationship
- `PRODUCES`: Entity produces output
- `SOURCE_OF`: Episode sourced an entity (consolidation)
- `CONSOLIDATED_INTO`: Procedure merged into another

**Key Operations**:
- `create_node(label, properties)` - Add node
- `create_relationship(source, target, type, props)` - Add edge
- `get_relationships(node_id, direction)` - Fetch neighbors
- `get_relationships_batch(node_ids)` - Bulk fetch (optimization)
- `strengthen_relationship(source, target, learning_rate)` - Hebbian update

**Connection**:
- URI: `bolt://localhost:7687`
- Async driver with session pooling

#### Saga Pattern (`saga.py`)
**Purpose**: Coordinate multi-store transactions

**Not yet implemented** - Future work for ACID guarantees across Qdrant + Neo4j.

---

### 4. Consolidation Layer (`/mnt/projects/t4d/t4dm/src/t4dm/consolidation/`)

#### Consolidation Service (`service.py`)

**Types**:
1. **Light**: Quick deduplication (O(n²) pairwise comparison)
2. **Deep**: Episodic → Semantic transfer (HDBSCAN clustering)
3. **Skill**: Merge similar procedures (HDBSCAN + success rate ranking)
4. **All**: Run all consolidation types

**Deep Consolidation Algorithm**:
```
1. Recall last 500 episodes
2. Cluster by embedding similarity (HDBSCAN, min_cluster_size=3)
3. For each cluster:
   - Extract entity from context (project/tool/concept)
   - Check for similar existing entity
   - Create or supersede entity
   - Link episodes with SOURCE_OF relationships
4. Return metrics (episodes consolidated, entities created/updated)
```

**Skill Consolidation Algorithm**:
```
1. Retrieve all active procedures
2. Cluster by script similarity (HDBSCAN, min_cluster_size=2)
3. For each cluster:
   - Sort by success_rate descending
   - Merge steps into best procedure
   - Deprecate others with CONSOLIDATED_INTO
4. Return metrics (procedures merged, deprecated)
```

---

## Data Flow Examples

### Example 1: Episode Creation
```
User: "I just fixed the auth bug in user_service.py"
  ↓
MCP Tool: remember_episode(content="...", context={project: "myapp", file: "user_service.py"})
  ↓
Validation: Check content is non-empty
  ↓
Embedding: BGE-M3.embed_query(content) → [1024 floats]
  ↓
Storage:
  - Qdrant.add(collection="episodes", vector=embedding, payload={session_id, content, context, ...})
  - Neo4j.create_node(label="Episode", properties={id, sessionId, content[:500], timestamp, ...})
  ↓
Response: Episode{id, timestamp, content, context, outcome="success"}
```

### Example 2: Semantic Recall with Context
```
User: recall_entities(query="authentication methods", context_entities=["uuid-of-auth-entity"])
  ↓
Embedding: BGE-M3.embed_query("authentication methods") → query_vec
  ↓
Qdrant Search: Find top 30 entities by cosine similarity
  ↓
For each candidate:
  - Load context entity from vector store
  - Batch fetch relationships from Neo4j
  - Calculate ACT-R activation:
      base = ln(access_count) - decay * ln(elapsed_time)
      spreading = W * strength * (S - ln(fan)) for each context entity
      noise = gauss(0, 0.1)
      activation = base + spreading + noise
  - Calculate FSRS retrievability: (1 + 0.9 * days/stability)^(-0.5)
  - Combine: score = 0.4*semantic + 0.35*norm(activation) + 0.25*retrievability
  ↓
Sort by score, take top 10
  ↓
Hebbian Strengthening: For each pair in results:
  - Parallel update relationship weights: w' = w + 0.1 * (1 - w)
  ↓
Response: list[ScoredResult] with component breakdown
```

### Example 3: Deep Consolidation
```
Scheduler/User: consolidate_memory(type="deep")
  ↓
Episodic Memory: Recall last 500 episodes (wildcard query)
  ↓
Embedding Service: Batch embed all episode contents
  ↓
HDBSCAN: Cluster by cosine similarity (min_cluster_size=3)
  → clusters = [[ep1, ep2, ep3], [ep4, ep5, ep6, ep7], ...]
  → noise = [ep8, ep9]  (no cluster)
  ↓
For each cluster:
  - Extract entity info:
      * If common project → EntityType.PROJECT
      * Elif common tool → EntityType.TOOL
      * Else → EntityType.CONCEPT
  - Search for existing similar entity (similarity > 0.9)
  - If exists: supersede with new summary
  - Else: create new entity
  - Link each episode to entity with SOURCE_OF relationship
  ↓
Response: {consolidated_episodes: 15, new_entities_created: 3, entities_updated: 1, confidence: 0.83}
```

---

## Session Isolation

**Problem**: Multiple Claude Code instances must not interfere with each other.

**Solution**: Session-based namespacing

**Implementation**:
1. Each memory service initialized with `session_id`
2. Qdrant filtering: `filter={"session_id": session_id}`
3. Neo4j properties: `sessionId` on all nodes
4. Default session: `"default"` (for shared knowledge)

**Cross-Session Queries**:
- Explicitly pass `session_filter=None` to query all sessions
- Use case: Shared knowledge base, multi-user consolidation

---

## Performance Characteristics

### Latency (single operation)
| Operation | Typical Latency | Bottleneck |
|-----------|----------------|------------|
| remember_episode | 50-100ms | Embedding generation |
| recall_episodes | 100-200ms | Vector search + FSRS |
| recall_entities (no context) | 80-150ms | Vector search |
| recall_entities (with context) | 150-300ms | ACT-R + batch Neo4j |
| consolidate (deep) | 5-15s | HDBSCAN + entity creation |

### Throughput
- **Rate limit**: 100 req/min per session
- **Concurrent sessions**: Unlimited (isolated by session_id)
- **Embedding batch size**: 32 (BGE-M3 optimal)

### Scaling Bottlenecks
1. **Embedding generation**: CPU-bound (BGE-M3 on CPU)
   - Mitigation: GPU acceleration or remote API
2. **HDBSCAN**: O(n log n), memory-intensive for >10k items
   - Mitigation: Incremental clustering, distributed HDBSCAN
3. **Neo4j fan-out**: Spreading activation expensive for high-degree nodes
   - Mitigation: Batch queries, caching

---

## Error Handling

### Error Codes (standardized)
| Code | Description | HTTP Status | Example |
|------|-------------|-------------|---------|
| INVALID_INPUT | Validation failed | 400 | Invalid UUID format |
| NOT_FOUND | Resource missing | 404 | Episode ID not found |
| RATE_LIMITED | Too many requests | 429 | >100 req/min |
| INTERNAL_ERROR | Unexpected failure | 500 | Database connection failed |
| STORAGE_ERROR | Storage operation failed | 503 | Qdrant timeout |

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "episode_id must be a valid UUID",
    "details": {
      "field": "episode_id",
      "value": "not-a-uuid"
    }
  }
}
```

### Retry Strategy
- **Rate limiting**: Exponential backoff (2^n seconds)
- **Storage errors**: 3 retries with 1s delay
- **Validation errors**: No retry (client must fix input)

---

## Configuration

**File**: `/mnt/projects/t4d/t4dm/src/t4dm/core/config.py`

### Key Settings
```python
# Embeddings
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# FSRS
FSRS_DEFAULT_STABILITY = 1.0

# ACT-R
ACTR_DECAY = 0.5
ACTR_NOISE = 0.1
ACTR_THRESHOLD = 0.5

# Hebbian
HEBBIAN_LEARNING_RATE = 0.1
HEBBIAN_INITIAL_WEIGHT = 0.1

# Consolidation
CONSOLIDATION_MIN_SIMILARITY = 0.75
CONSOLIDATION_MIN_OCCURRENCES = 3
CONSOLIDATION_SKILL_SIMILARITY = 0.85

# Retrieval weights
RETRIEVAL_SEMANTIC_WEIGHT = 0.4
RETRIEVAL_RECENCY_WEIGHT = 0.25
RETRIEVAL_OUTCOME_WEIGHT = 0.2
RETRIEVAL_IMPORTANCE_WEIGHT = 0.15

# Storage
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "worldweaver"

# Rate limiting
RATE_LIMIT_MAX_REQUESTS = 100
RATE_LIMIT_WINDOW_SECONDS = 60
```

---

## Deployment

### Local Development
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start Neo4j
docker run -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/worldweaver \
  neo4j:latest

# Install dependencies
cd /mnt/projects/ww
pip install -e .

# Run MCP server
python -m t4dm.mcp.memory_gateway
```

### Production Considerations
1. **Embeddings**: Switch to GPU or OpenAI API
2. **Databases**: Use managed Qdrant + Neo4j (Aura)
3. **Monitoring**: Add Prometheus metrics for latency, error rates
4. **Backups**: Automated snapshots of Qdrant collections + Neo4j dump
5. **Scaling**: Horizontal scaling with session sharding

---

## Testing Strategy

### Unit Tests
- Memory services: Mock storage layer
- Algorithms: Fixed inputs, verify outputs
- Validation: Boundary cases, invalid inputs

### Integration Tests
- End-to-end flows: Create → Recall → Consolidate
- Storage: Real Qdrant + Neo4j (test containers)
- MCP: Mock FastMCP client

### Performance Tests
- Load testing: 1000 episodes, measure recall latency
- Consolidation: 10k episodes, verify HDBSCAN scaling
- Concurrent sessions: 10 sessions, check isolation

**Test Coverage**: Target 85%+ for core memory services

---

---

### 5. Learning Systems (`/mnt/projects/t4d/t4dm/src/t4dm/learning/`)

#### Dopamine System (`dopamine.py`)
**Purpose**: Reward prediction and surprise signals for reinforcement learning.

**Components**:
- `RewardSignal`: Dataclass for reward events with prediction errors
- `DopamineSystem`: TD learning with configurable parameters
  - Baseline reward tracking
  - Prediction error calculation
  - Surprise signal generation

**Key Methods**:
- `record_reward(value)` → RewardSignal (with prediction error)
- `get_predicted_reward()` → float
- `get_stats()` → dict

#### Serotonin System (`serotonin.py`)
**Purpose**: Long-term credit assignment across temporal gaps.

**Components**:
- `EligibilityTrace`: Decaying activation trace for memories
- `TemporalContext`: Session-scoped goal tracking
- `SerotoninSystem`: Credit distribution with eligibility traces

**Key Methods**:
- `start_context(session_id, goal)` → None
- `add_eligibility(memory_id, strength)` → None
- `receive_outcome(score, context_id)` → dict[memory_id, credit]
- `get_long_term_value(memory_id)` → float

---

### 6. Hook System (`/mnt/projects/t4d/t4dm/src/t4dm/hooks/`)

**Purpose**: Extensible pre/post/on hooks for memory operations.

#### Base Classes
- `Hook`: Abstract base with priority, enabled, should_execute
- `HookContext`: Operation context (phase, input/output data, metadata)
- `HookPhase`: Enum (PRE, ON, POST)
- `HookPriority`: Enum (CRITICAL, HIGH, NORMAL, LOW)

#### Memory Hooks (`memory.py`)
- `CreateHook`, `RecallHook`, `UpdateHook`, `AccessHook`, `DecayHook`
- Memory-type filtering support

#### Built-in Implementations
- `CachingRecallHook`: LRU cache for recall queries
- `ValidationHook`: Content length and required field validation
- `AuditTrailHook`: Operation logging for compliance
- `HebbianUpdateHook`: Co-access relationship strengthening

---

### 7. Observability (`/mnt/projects/t4d/t4dm/src/t4dm/observability/`)

**Purpose**: Tracing and monitoring for Claude Code integration.

#### WWObserver (`tracing.py`)
OpenTelemetry-compatible observer for Claude Code ccapi events.

**Supported Events**:
- `tool_use`: Track MCP tool invocations
- `agent_start`/`agent_end`: Session lifecycle
- `observation`: Capture system observations
- `error`: Error tracking with context

**Features**:
- Span processing with parent/child relationships
- Event routing by type
- Citation extraction from agent outputs
- Statistics collection

---

### 8. Integrations (`/mnt/projects/t4d/t4dm/src/t4dm/integrations/`)

#### Kymera Integration (`kymera/`)
Native integration with Kymera AI orchestrator.

**Components**:
- Memory service adapters
- Event forwarding
- Session synchronization

---

## Future Enhancements

1. **Saga Pattern**: ACID transactions across Qdrant + Neo4j
2. **Streaming Consolidation**: Incremental clustering for large datasets
3. **Multi-Modal Embeddings**: Support for images, code snippets
4. **Forgetting Curves**: Automatic pruning of low-importance memories
5. ~~**Distributed Tracing**: OpenTelemetry for request flow debugging~~ ✅ Implemented
6. **Conflict Resolution**: CRDTs for multi-user semantic knowledge
7. **Explanation**: Return provenance chains for retrieval results

---

## Architecture Principles

1. **Cognitive Fidelity**: Mirror human memory systems (episodic/semantic/procedural)
2. **Session Isolation**: Multi-tenant by design (session_id namespacing)
3. **Graceful Degradation**: Fallback to simpler algorithms on failure
4. **Observable**: Log all decisions (scoring, clustering, consolidation)
5. **Tunable**: Expose hyperparameters via config (no hardcoded magic numbers)
6. **Testable**: Clear boundaries between layers (memory/storage/MCP)

---

## References

1. **MCP Protocol**: https://modelcontextprotocol.io/
2. **Qdrant**: https://qdrant.tech/documentation/
3. **Neo4j**: https://neo4j.com/docs/
4. **FastMCP**: https://github.com/jlowin/fastmcp
5. **BGE-M3**: https://huggingface.co/BAAI/bge-m3
