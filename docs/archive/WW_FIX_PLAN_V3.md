# World Weaver Fix Plan v3.0

**Created**: 2025-11-27 | **Based On**: Gap Analysis | **Priority**: Production-Critical

> Post-hardening phases addressing architectural gaps, algorithm completion, and missing features.

## Status Dashboard

| Phase | Status | Priority | Tasks | Description |
|-------|--------|----------|-------|-------------|
| Phase 1-6 | COMPLETE | - | 48 | Initial hardening (bugs, perf, security, tests, API, docs) |
| Phase 7: Integration | COMPLETE | P0 | 4 | Saga integration, tracing, gateway refactor |
| Phase 8: Algorithms | COMPLETE | P1 | 3 | Hebbian decay, config params, LLM triggers |
| Phase 9: Features | COMPLETE | P2 | 3 | Batch ops, cross-memory search, auto-extraction |

### 🎉 ALL PHASES COMPLETE (2025-11-27)
World Weaver is now **PRODUCTION READY** with 58/58 tasks completed.

### Phase 9 Completion Summary (2025-11-27)
Feature gaps filled:
- P9-001: Batch MCP operations (4 batch functions, 16 tests)
  - `create_episodes_batch`, `create_entities_batch`, `create_skills_batch`, `recall_batch`
  - Max 100 items per batch, partial failure handling, parallel recall
- P9-002: Cross-memory search (unified.py service, 2 MCP tools, 9 tests)
  - `search_all_memories` - parallel search across all memory types
  - `get_related_memories` - graph-based relationship traversal
  - Session isolation, min_score filtering, asyncio.gather parallelism
- P9-003: Auto entity extraction (extraction module, 5 classes, 30 tests)
  - RegexEntityExtractor (10 patterns: email, URL, phone, date, money, etc.)
  - LLMEntityExtractor (PERSON, ORG, LOCATION, CONCEPT, TECHNOLOGY, EVENT)
  - CompositeEntityExtractor with deduplication
  - `auto_extract` flag in create_episode, background job in consolidation

### Phase 8 Completion Summary (2025-11-27)
Algorithm completion:
- P8-001: Hebbian decay (exponential decay for stale relationships, auto-pruning)
- P8-002: 44 algorithm params moved to config (FSRS, ACT-R, Hebbian, HDBSCAN, weights)
- P8-003: Semantic trigger matching (embedding-based, replaces substring TODO)

### Phase 7 Completion Summary (2025-11-27)
Integration hardening complete:
- P7-001: Saga integrated into all memory services (7 methods wrapped, 13 tests)
- P7-002: OpenTelemetry tracing (17 instrumentation points, Jaeger-compatible)
- P7-003: Gateway refactored (1,582→10 modules, 19 tools + 4 resources)
- P7-004: Operational scripts (backup.sh, restore.sh, health_check.sh, migrate.sh)

---

## Phase 7: Integration Hardening (P0)

**Goal**: Eliminate data corruption risks and enable production debugging.

### TASK-P7-001: Integrate Saga into Memory Services
**Files**: `src/t4dm/memory/episodic.py`, `semantic.py`, `procedural.py`
**Severity**: CRITICAL
**Description**: Wrap all create/update/delete operations in saga transactions

```python
# Current (UNSAFE):
async def create(self, content: str, ...) -> Episode:
    await self.vector_store.add(...)  # If this succeeds...
    await self.graph_store.create_node(...)  # ...but this fails = data inconsistency

# Fixed (SAFE):
async def create(self, content: str, ...) -> Episode:
    saga = Saga("create_episode")
    saga.add_step(
        name="add_vector",
        action=lambda: self.vector_store.add(...),
        compensation=lambda: self.vector_store.delete(episode_id)
    )
    saga.add_step(
        name="create_node",
        action=lambda: self.graph_store.create_node(...),
        compensation=lambda: self.graph_store.delete_node(episode_id)
    )
    return await saga.execute()
```

**Validation**: Test partial failure recovery

### TASK-P7-002: Add OpenTelemetry Tracing
**Files**: `src/t4dm/observability/tracing.py` (NEW), all memory services
**Severity**: HIGH
**Description**: Propagate request context through entire call stack

```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind

tracer = trace.get_tracer("world-weaver")

def traced(name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name, kind=SpanKind.INTERNAL) as span:
                span.set_attribute("session_id", kwargs.get("session_id", "unknown"))
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator
```

**Validation**: Traces visible in Jaeger/Zipkin

### TASK-P7-003: Refactor Gateway into Modules
**Files**: `src/t4dm/mcp/memory_gateway.py` → split into modules
**Severity**: HIGH
**Description**: Split 1,582-line file into focused modules

```
src/t4dm/mcp/
├── __init__.py
├── gateway.py           # Main MCP app, shared decorators (200 lines)
├── tools/
│   ├── __init__.py
│   ├── episodic.py      # create_episode, recall_episodes, etc. (300 lines)
│   ├── semantic.py      # create_entity, semantic_recall, etc. (300 lines)
│   ├── procedural.py    # create_skill, recall_skill, etc. (300 lines)
│   └── system.py        # consolidate_now, memory_stats, etc. (200 lines)
├── validation.py        # Already exists
├── types.py             # Already exists
└── errors.py            # Already exists
```

**Validation**: All 17 tools still work after refactor

### TASK-P7-004: Create Operational Scripts
**Files**: `scripts/backup.sh`, `scripts/restore.sh`, `scripts/health_check.sh`
**Severity**: MEDIUM
**Description**: Create missing operational scripts referenced in docs

```bash
# scripts/backup.sh
#!/bin/bash
set -e
BACKUP_DIR="${BACKUP_DIR:-/var/backups/ww}"
DATE=$(date +%Y%m%d_%H%M%S)

# Neo4j backup
docker exec ww-neo4j neo4j-admin database dump neo4j --to-path=/backups
docker cp ww-neo4j:/backups/neo4j.dump "$BACKUP_DIR/neo4j_$DATE.dump"

# Qdrant backup
curl -X POST "http://localhost:6333/collections/episodes/snapshots"
# ... copy snapshot
```

**Validation**: Scripts execute without error

---

## Phase 8: Algorithm Completion (P1)

**Goal**: Complete cognitive algorithm implementations.

### TASK-P8-001: Implement Hebbian Decay
**File**: `src/t4dm/memory/semantic.py`
**Description**: Add decay for unused relationships (currently only strengthening)

```python
async def _apply_hebbian_decay(self, decay_rate: float = 0.01):
    """Decay relationships not accessed recently."""
    # Find relationships not accessed in last N days
    stale_rels = await self.graph_store.find_stale_relationships(days=30)
    for rel in stale_rels:
        new_weight = rel.weight * (1 - decay_rate)
        if new_weight < 0.01:  # Prune very weak relationships
            await self.graph_store.delete_relationship(rel.id)
        else:
            await self.graph_store.update_relationship(rel.id, weight=new_weight)
```

### TASK-P8-002: Expose Algorithm Parameters to Config
**File**: `src/t4dm/core/config.py`
**Description**: Move hardcoded algorithm parameters to Settings

```python
class Settings(BaseSettings):
    # FSRS parameters
    fsrs_decay_factor: float = Field(default=0.9, ge=0.1, le=1.0)
    fsrs_initial_stability: float = Field(default=1.0, ge=0.1)

    # ACT-R parameters
    actr_strength_param: float = Field(default=1.6, ge=0.1, le=5.0)
    actr_decay_param: float = Field(default=0.5, ge=0.1, le=1.0)

    # Hebbian parameters
    hebbian_learning_rate: float = Field(default=0.1, ge=0.01, le=0.5)
    hebbian_decay_rate: float = Field(default=0.01, ge=0.001, le=0.1)

    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = Field(default=3, ge=2)
```

### TASK-P8-003: Add LLM-based Trigger Matching
**File**: `src/t4dm/memory/procedural.py`
**Description**: Replace TODO with semantic trigger matching

```python
async def _match_trigger_semantic(self, query: str, procedures: list[Procedure]) -> list[tuple[Procedure, float]]:
    """Use embedding similarity for trigger matching instead of substring."""
    query_embedding = await self.embedding_provider.embed_query(query)

    scored = []
    for proc in procedures:
        # Embed trigger pattern
        trigger_embedding = await self.embedding_provider.embed_query(proc.trigger_pattern)
        similarity = cosine_similarity(query_embedding, trigger_embedding)
        if similarity > 0.7:  # Threshold
            scored.append((proc, similarity))

    return sorted(scored, key=lambda x: x[1], reverse=True)
```

---

## Phase 9: Feature Gaps (P2)

**Goal**: Add missing features for production usability.

### TASK-P9-001: Batch MCP Operations
**File**: `src/t4dm/mcp/tools/episodic.py`
**Description**: Add batch create/recall operations

```python
@mcp_app.tool()
async def create_episodes_batch(
    episodes: list[dict],
    session_id: str = None,
) -> dict:
    """Create multiple episodes in one request."""
    created = []
    for ep_data in episodes[:100]:  # Limit batch size
        episode = await create_episode(
            content=ep_data["content"],
            metadata=ep_data.get("metadata"),
            session_id=session_id,
        )
        created.append(episode)
    return {"episodes": created, "count": len(created)}
```

### TASK-P9-002: Cross-Memory Search
**File**: `src/t4dm/mcp/tools/system.py`
**Description**: Search across all memory types in one query

```python
@mcp_app.tool()
async def search_all_memories(
    query: str,
    limit: int = 10,
    session_id: str = None,
) -> dict:
    """Search episodes, entities, and skills simultaneously."""
    episodes, entities, skills = await asyncio.gather(
        episodic.recall(query, k=limit, session_id=session_id),
        semantic.recall(query, k=limit, session_id=session_id),
        procedural.recall_skill(query, k=limit, session_id=session_id),
    )
    return {
        "episodes": [e.to_dict() for e in episodes],
        "entities": [e.to_dict() for e in entities],
        "skills": [s.to_dict() for s in skills],
        "query": query,
    }
```

### TASK-P9-003: Auto Entity Extraction
**File**: `src/t4dm/consolidation/service.py`
**Description**: Automatically extract entities from new episodes

```python
async def extract_entities_realtime(self, episode: Episode) -> list[Entity]:
    """Extract entities from episode content using NER or LLM."""
    # Use spaCy or LLM for entity extraction
    entities = await self._extract_named_entities(episode.content)

    created = []
    for name, entity_type in entities:
        entity = await self.semantic.create_entity(
            name=name,
            entity_type=entity_type,
            source=f"episode:{episode.id}",
            session_id=episode.session_id,
        )
        created.append(entity)

    return created
```

---

## Dependency Graph

```
Phase 7 (Integration) ──┬── P7-001: Saga integration
                        ├── P7-002: OpenTelemetry
                        ├── P7-003: Gateway refactor
                        └── P7-004: Ops scripts
                              │
Phase 8 (Algorithms) ───┬── P8-001: Hebbian decay
                        ├── P8-002: Config params (depends on P7-001)
                        └── P8-003: LLM triggers
                              │
Phase 9 (Features) ─────┬── P9-001: Batch ops (depends on P7-003)
                        ├── P9-002: Cross-memory search
                        └── P9-003: Auto extraction (depends on P8-003)
```

---

## Estimated Effort

| Phase | Tasks | Days | Cumulative |
|-------|-------|------|------------|
| Phase 7 | 4 | 8 | 8 days |
| Phase 8 | 3 | 6 | 14 days |
| Phase 9 | 3 | 7 | 21 days |

**Total to production-ready**: 3 weeks

---

## Quick Reference

| Task | File(s) | Severity | Est. |
|------|---------|----------|------|
| P7-001 | memory/*.py | CRITICAL | 3d |
| P7-002 | observability/tracing.py | HIGH | 2d |
| P7-003 | mcp/tools/*.py | HIGH | 2d |
| P7-004 | scripts/*.sh | MEDIUM | 1d |
| P8-001 | memory/semantic.py | HIGH | 2d |
| P8-002 | core/config.py | MEDIUM | 2d |
| P8-003 | memory/procedural.py | MEDIUM | 2d |
| P9-001 | mcp/tools/episodic.py | MEDIUM | 1d |
| P9-002 | mcp/tools/system.py | MEDIUM | 3d |
| P9-003 | consolidation/service.py | MEDIUM | 3d |

---

## Final Statistics

| Metric | Count |
|--------|-------|
| Total Tasks | 58 |
| Phases Completed | 9 |
| Tests Added | 100+ |
| Files Created/Modified | 50+ |
| MCP Tools | 23 |
| Config Parameters | 50+ |

**Status**: ALL PHASES COMPLETE - World Weaver is PRODUCTION READY
