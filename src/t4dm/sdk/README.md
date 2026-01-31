# SDK Module

**3 files | ~850 lines | Centrality: 0**

The SDK module provides Python client libraries for World Weaver's REST API with both synchronous and asynchronous implementations.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SDK ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    AsyncWorldWeaverClient                            ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  ││
│  │  │  Episodic    │  │   Semantic   │  │      Procedural          │  ││
│  │  │  7 methods   │  │  7 methods   │  │      7 methods           │  ││
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    WorldWeaverClient (Sync)                          ││
│  │  Simplified sync wrapper for common operations                       ││
│  └─────────────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         Pydantic Models                              ││
│  │  Episode | Entity | Skill | RecallResult | ActivationResult         ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `client.py` | ~700 | Async and sync HTTP clients |
| `models.py` | ~120 | Pydantic response models |
| `__init__.py` | ~25 | Public API exports |

## Installation

```bash
pip install -e ".[sdk]"
```

**Dependencies**: `httpx>=0.25.0`, `pydantic>=2.0`

## Quick Start

### Async Client (Recommended)

```python
from ww.sdk import AsyncWorldWeaverClient

async with AsyncWorldWeaverClient(
    base_url="http://localhost:8765",
    session_id="my-session"
) as client:
    # Store episode
    episode = await client.create_episode(
        content="Learned about decorators",
        project="learning",
        outcome="success",
        emotional_valence=0.8
    )

    # Search episodes
    results = await client.recall_episodes(
        query="decorators",
        limit=10
    )
    for ep, score in zip(results.episodes, results.scores):
        print(f"{ep.content[:50]}... (score: {score:.3f})")
```

### Sync Client

```python
from ww.sdk import WorldWeaverClient

with WorldWeaverClient() as client:
    status = client.health()
    episode = client.create_episode(
        content="Python async patterns",
        outcome="success"
    )
```

## API Reference

### Initialization

```python
client = AsyncWorldWeaverClient(
    base_url="http://localhost:8765",  # API endpoint
    session_id="my-session",           # Memory isolation
    timeout=30.0                       # Request timeout
)

# Context manager (recommended)
async with AsyncWorldWeaverClient() as client:
    pass

# Manual connection
await client.connect()
await client.close()
```

### System Operations

```python
# Health check
status = await client.health()  # HealthStatus

# Memory statistics
stats = await client.stats()  # MemoryStats

# Trigger consolidation
result = await client.consolidate(deep=False)
```

### Episodic Memory (7 methods)

```python
# Create episode
episode = await client.create_episode(
    content="...",
    project="project-name",
    file="path/to/file.py",
    tool="pytest",
    outcome="success",  # success|failure|neutral
    emotional_valence=0.8,  # 0-1
    timestamp=datetime.now()
)

# Get by ID
episode = await client.get_episode(episode_id)

# List with pagination
episodes, total = await client.list_episodes(
    page=1,
    page_size=20,
    project="filter-project",
    outcome="success"
)

# Semantic search
results = await client.recall_episodes(
    query="search term",
    limit=10,
    min_similarity=0.5,
    project="filter-project"
)

# Delete
await client.delete_episode(episode_id)

# Mark important
episode = await client.mark_important(episode_id, importance=1.0)
```

### Semantic Memory (7 methods)

```python
# Create entity
entity = await client.create_entity(
    name="Python",
    entity_type="language",
    summary="A programming language",
    details="Detailed description...",
    source="episode_id or 'user_provided'"
)

# Get by ID
entity = await client.get_entity(entity_id)

# List with filter
entities = await client.list_entities(
    entity_type="language",
    limit=50
)

# Create relationship
relation = await client.create_relation(
    source_id=python_id,
    target_id=decorators_id,
    relation_type="has_feature",
    weight=0.9
)

# Semantic search
entities = await client.recall_entities(
    query="programming languages",
    limit=10,
    entity_types=["language", "tool"]
)

# Spreading activation
result = await client.spread_activation(
    entity_id=start_id,
    depth=2,
    threshold=0.1
)

# Update entity (bi-temporal)
entity = await client.supersede_entity(
    entity_id=entity_id,
    name="Python 3",
    entity_type="language",
    summary="Updated summary"
)
```

### Procedural Memory (7 methods)

```python
# Create skill
skill = await client.create_skill(
    name="run_tests",
    domain="development",
    task="Execute test suite",
    steps=[
        {"order": 1, "action": "Activate venv", "tool": "bash"},
        {"order": 2, "action": "Run pytest", "tool": "bash"}
    ],
    trigger_pattern="run.*tests?"
)

# Get by ID
skill = await client.get_skill(skill_id)

# List with filter
skills = await client.list_skills(
    domain="development",
    include_deprecated=False,
    limit=50
)

# Semantic search
skills = await client.recall_skills(
    query="testing",
    domain="development",
    limit=5
)

# Record execution
skill = await client.record_execution(
    skill_id=skill_id,
    success=True,
    duration_ms=1250,
    notes="All tests passed"
)

# Deprecate skill
skill = await client.deprecate_skill(
    skill_id=skill_id,
    replacement_id=new_skill_id
)

# Get step-by-step instructions
skill, steps, confidence = await client.how_to(
    query="run test suite",
    domain="development"
)
```

## Data Models

### Episode

```python
class Episode(BaseModel):
    id: UUID
    session_id: str
    content: str
    timestamp: datetime
    outcome: str
    emotional_valence: float
    context: EpisodeContext
    access_count: int
    stability: float
    retrievability: float | None
```

### Entity

```python
class Entity(BaseModel):
    id: UUID
    name: str
    entity_type: str
    summary: str
    details: str | None
    source: str | None
    stability: float
    access_count: int
    created_at: datetime
```

### Skill

```python
class Skill(BaseModel):
    id: UUID
    name: str
    domain: str
    trigger_pattern: str | None
    steps: list[Step]
    script: str | None
    success_rate: float
    execution_count: int
    version: int
    deprecated: bool
```

### Result Types

```python
class RecallResult(BaseModel):
    query: str
    episodes: list[Episode]
    scores: list[float]

class ActivationResult(BaseModel):
    entities: list[Entity]
    activations: list[float]
    paths: list[list[str]]
```

## Error Handling

```python
from ww.sdk.client import (
    WorldWeaverError,
    ConnectionError,
    NotFoundError,
    RateLimitError
)

try:
    async with AsyncWorldWeaverClient() as client:
        result = await client.recall_episodes("query")
except NotFoundError as e:
    print(f"Not found: {e.status_code}")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except WorldWeaverError as e:
    print(f"API error: {e.status_code}")
```

## Configuration

### Environment Variables

```bash
WORLD_WEAVER_API_URL=http://localhost:8765
WORLD_WEAVER_SESSION_ID=default_session
WORLD_WEAVER_TIMEOUT=30.0
```

### Client Configuration

```python
client = AsyncWorldWeaverClient(
    base_url="http://localhost:8765",
    session_id="my-session",
    timeout=30.0
)
```

## Usage Examples

### Session Isolation

```python
# User 1 session
async with AsyncWorldWeaverClient(session_id="user_1") as client1:
    ep1 = await client1.create_episode(content="User 1 data")

# User 2 session
async with AsyncWorldWeaverClient(session_id="user_2") as client2:
    ep2 = await client2.create_episode(content="User 2 data")

# Memories are isolated by session_id
```

### Entity Graph

```python
async with AsyncWorldWeaverClient() as client:
    # Create entities
    python = await client.create_entity(
        name="Python",
        entity_type="language",
        summary="Programming language"
    )
    decorators = await client.create_entity(
        name="Decorators",
        entity_type="feature",
        summary="Function transformation"
    )

    # Link entities
    await client.create_relation(
        source_id=python.id,
        target_id=decorators.id,
        relation_type="has_feature",
        weight=0.9
    )

    # Explore network
    result = await client.spread_activation(
        entity_id=python.id,
        depth=2
    )
```

### Skill Execution Tracking

```python
async with AsyncWorldWeaverClient() as client:
    # Get skill
    skill, steps, confidence = await client.how_to("run tests")

    if skill:
        # Execute and record
        success = run_tests()
        await client.record_execution(
            skill_id=skill.id,
            success=success,
            duration_ms=1500
        )
```

## Testing

```bash
# Run SDK tests
pytest tests/unit/test_sdk.py -v

# With coverage
pytest tests/unit/test_sdk.py --cov=ww.sdk
```

## Public API

```python
# Clients
AsyncWorldWeaverClient
WorldWeaverClient

# Models
Episode, Entity, Skill
RecallResult, ActivationResult
HealthStatus, MemoryStats
EpisodeContext, Relationship, Step

# Exceptions
WorldWeaverError
ConnectionError
NotFoundError
RateLimitError
```

## Design Patterns

| Pattern | Usage |
|---------|-------|
| Context Manager | Auto connect/disconnect |
| Pydantic Models | Type-safe responses |
| Error Hierarchy | Specific exception types |
| Session Headers | X-Session-ID for isolation |
