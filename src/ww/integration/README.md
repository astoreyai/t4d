# Integration Module (ccapi)

**4 files | ~1,700 lines | Centrality: 5**

The integration module provides adapters connecting external agent frameworks (llm_agents/ccapi) to World Weaver's tripartite memory system and learning feedback loops.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       CCAPI INTEGRATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    WWMemory (Protocol Adapter)                      ││
│  │  ccapi Memory interface → WW Episodic Memory                        ││
│  │  add() | search() | get_context() | save/load                       ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    WWObserver (Event Adapter)                       ││
│  │  ccapi events → WW Learning outcomes + credit assignment            ││
│  │  agent.end | tool.end | memory.retrieve → OutcomeType               ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    FastAPI Routes                                   ││
│  │  REST endpoints for remote agent integration                        ││
│  │  /store | /search | /context | /outcome | /learning/*               ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `ccapi_memory.py` | ~445 | Memory protocol adapter |
| `ccapi_observer.py` | ~570 | Observer protocol for outcome feedback |
| `ccapi_routes.py` | ~590 | FastAPI router with endpoints |
| `__init__.py` | ~70 | Public API exports |

## Memory Protocol Adapter

### WWMemory

Implements llm_agents Memory interface:

```python
from ww.integration import WWMemory, create_ww_memory

memory = await create_ww_memory(
    session_id="agent-session",
    max_messages=1000
)

# Add message
memory.add(Message(role="user", content="Hello"))
memory.add_many([msg1, msg2, msg3])

# Search (sync with async fallback)
results = memory.search("recent conversations", limit=10)

# Async search (preferred)
results = await memory.search_async("query", limit=10)

# Get messages
recent = memory.get_messages(limit=20)
by_role = memory.get_messages_by_role("assistant")

# Context for LLM injection
context = memory.get_context(
    max_messages=10,
    max_tokens=4000,
    include_system=True
)

# Persistence
memory.save("memory.json")
memory.load("memory.json")
memory.clear()
```

**Integration**:
- Stores as `Episode` objects
- Session-isolated
- Emits retrieval events to learning system
- Lazy-loads: BGEM3Embedder, EpisodicMemory, QdrantStore

## Observer Protocol Adapter

### WWObserver

Converts ccapi events to WW learning outcomes:

```python
from ww.integration import WWObserver, create_ww_observer

observer = await create_ww_observer(
    session_id="agent-session",
    enable_learning=True,
    buffer_size=100
)

# Event handling
observer.on_event(Event(
    type="agent.end",
    name="task_complete",
    data={"status": "success", "result": "..."}
))

# Span tracking
observer.on_span_start(span)
observer.on_span_end(span)

# Batch processing
observer.flush()  # Process pending tasks

# Cleanup
observer.close()

# Telemetry
retrievals = observer.get_recent_retrievals()
stats = observer.get_session_stats()
```

### Event Routing

| Event Type | Handler | Outcome |
|------------|---------|---------|
| `agent.end` | `_handle_agent_end()` | SUCCESS/PARTIAL/FAILURE |
| `agent.error` | `_handle_agent_error()` | FAILURE |
| `tool.end` | `_handle_tool_end()` | Procedural signal |
| `tool.error` | `_handle_tool_error()` | Tool failure |
| `memory.retrieve` | `_handle_memory_retrieve()` | Citation tracking |

### Learning Signal Generation

```python
# Event → Outcome mapping
status="success" → OutcomeType.SUCCESS, score=1.0
status="partial" → OutcomeType.PARTIAL, score=0.5
status="error"   → OutcomeType.FAILURE, score=0.0

# User feedback modifiers
feedback="positive" → score += 0.2
feedback="negative" → score -= 0.3

# Memory citations extracted from tool results
citations = extract_memory_ids(event.data.get("tool_results"))

# Record to collector
collector.record_outcome(
    outcome_type=OutcomeType.SUCCESS,
    success_score=0.8,
    context="agent:task",
    explicit_citations=[uuid1, uuid2]
)
```

## FastAPI Routes

### Router Factory

```python
from ww.integration import create_ww_router

router = create_ww_router(
    session_id="default",
    enable_learning=True
)

app.include_router(router, prefix="/ww")
```

### Memory Endpoints

#### POST /store

Store memory across types:

```python
# Request
{
    "content": "Learned about REST APIs",
    "memory_type": "episodic",  # episodic|semantic|procedural
    "outcome": "success",
    "emotional_valence": 0.8,
    "metadata": {"source": "tutorial"}
}

# Response
{
    "success": true,
    "memory_id": "uuid",
    "memory_type": "episodic",
    "message": "Episode stored successfully"
}
```

#### POST /search

Unified search across memory types:

```python
# Request
{
    "query": "REST API authentication",
    "memory_types": ["episodic", "semantic"],
    "limit": 10,
    "min_score": 0.5,
    "session_id": "agent-session"
}

# Response
{
    "query": "REST API authentication",
    "total_count": 8,
    "results": [...],
    "by_type": {
        "episodic": 5,
        "semantic": 3
    }
}
```

#### GET /context

Session context for LLM injection:

```python
# Response
{
    "session_id": "agent-session",
    "episodes": [...],
    "entities": [...],
    "skills": [...],
    "toon_json": "..."  # Optional compact format
}
```

### Learning Endpoints

#### POST /outcome

Record task outcome for credit assignment:

```python
# Request
{
    "outcome_type": "success",
    "success_score": 0.9,
    "context": "Completed authentication task",
    "memory_citations": ["uuid1", "uuid2"]
}

# Response
{
    "success": true,
    "outcome_id": "uuid",
    "matched_retrievals": 2
}
```

#### GET /learning/stats

```python
# Response
{
    "total_experiences": 1000,
    "total_retrievals": 500,
    "total_outcomes": 200,
    "avg_success_score": 0.78,
    "replay_buffer_size": 150
}
```

#### POST /learning/flush

Flush pending learning events.

## Data Models

### Request Models

```python
class MemoryStoreRequest(BaseModel):
    content: str
    memory_type: str = "episodic"
    metadata: dict = {}
    outcome: str | None = None
    emotional_valence: float = 0.5
    entity_name: str | None = None
    entity_type: str | None = None
    skill_name: str | None = None
    domain: str | None = None

class MemorySearchRequest(BaseModel):
    query: str
    memory_types: list[str] = ["episodic", "semantic", "procedural"]
    limit: int = 10
    min_score: float = 0.0
    session_id: str | None = None

class OutcomeRequest(BaseModel):
    outcome_type: str  # success|partial|failure|neutral
    success_score: float  # 0.0-1.0
    context: str | None = None
    memory_citations: list[str] = []
```

### Response Models

```python
class MemoryStoreResponse(BaseModel):
    success: bool
    memory_id: str
    memory_type: str
    message: str

class MemorySearchResponse(BaseModel):
    query: str
    total_count: int
    results: list[dict]
    by_type: dict[str, int]

class LearningStatsResponse(BaseModel):
    total_experiences: int
    total_retrievals: int
    total_outcomes: int
    avg_success_score: float
    replay_buffer_size: int
```

## Error Handling

### Async/Sync Context

```python
# ASYNC-001 Fix: Detect running loop
try:
    loop = asyncio.get_running_loop()
    # Fall back to simple search
    return self._simple_search(query, limit)
except RuntimeError:
    # Safe to create new loop
    loop = asyncio.new_event_loop()
```

### Graceful Degradation

- Async store fails → logs warning, continues buffering
- Unified init fails → HTTPException 503
- Learning disabled → continues without credit assignment
- Failed retrievals → fallback to simple text search

## Session Isolation

```python
# All memory operations scoped by session_id
episode = Episode(
    content=content,
    session_id=session_id,
    ...
)

# Recall filters by session
results = await episodic.recall(
    query=query,
    session_filter=session_id
)

# "default" session acts as global (no filter)
```

## Integration Points

### With WW Memory

```python
from ww.memory import EpisodicMemory, SemanticMemory, ProceduralMemory
from ww.memory.unified import UnifiedMemoryService

# Store episode
await episodic.store(episode)

# Unified search
await unified.search(query, k=10, memory_types=["episodic", "semantic"])
```

### With Learning

```python
from ww.learning.collector import get_collector
from ww.learning.events import MemoryType, OutcomeType
from ww.learning.hooks import emit_retrieval_event

# Emit retrieval signal
emit_retrieval_event(
    query=query,
    memory_type=MemoryType.EPISODIC,
    results=[...],
    session_id=session_id
)

# Record outcome
collector.record_outcome(
    outcome_type=OutcomeType.SUCCESS,
    success_score=0.8,
    explicit_citations=[uuid1]
)
```

### With Embeddings

```python
from ww.embedding.bge_m3 import BGEM3Embedder

embedder = await BGEM3Embedder.create()
```

## Type Stubs

Avoids hard dependency on llm_agents:

```python
# Defined internally
class Message:
    role: str
    content: str
    name: str | None
    tool_call_id: str | None
    metadata: dict

class Event:
    type: str
    name: str
    data: dict
    severity: str
    timestamp: datetime

class Span:
    name: str
    trace_id: str
    span_id: str
    status: str
    duration_ms: float
```

## Testing

```bash
# Run integration tests
pytest tests/unit/test_ccapi_integration.py -v

# With coverage
pytest tests/unit/test_ccapi_integration.py --cov=ww.integration
```

**Test Coverage**: 695+ lines, comprehensive unit tests

## Installation

```bash
# Integration included in core
pip install -e "."

# With FastAPI
pip install -e ".[api]"
```

## Public API

```python
# Memory adapter
WWMemory, create_ww_memory, Message

# Observer adapter
WWObserver, create_ww_observer, Event, Span, EventType

# Routes
create_ww_router

# Models
MemoryStoreRequest, MemoryStoreResponse
MemorySearchRequest, MemorySearchResponse
OutcomeRequest, OutcomeResponse
LearningStatsResponse
```

## Design Patterns

| Pattern | Usage |
|---------|-------|
| Adapter | WWMemory/WWObserver wrap WW APIs |
| Type Stubs | Avoid hard dependency on ccapi |
| Lazy Loading | Components loaded on demand |
| Graceful Degradation | Sync fallback when async unavailable |
| Session Isolation | Multi-tenant support |
