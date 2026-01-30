# World Weaver Python SDK

## Installation

```bash
pip install world-weaver[api]
```

Or install from source:
```bash
git clone https://github.com/astoreyai/world-weaver
cd world-weaver
pip install -e ".[api]"
```

## Quick Start

### Synchronous Client

```python
from ww.sdk import WorldWeaverClient

# Using context manager (recommended)
with WorldWeaverClient(base_url="http://localhost:8765", session_id="my-session") as ww:
    # Create an episode
    episode = ww.create_episode(
        content="Learned about Python async/await patterns",
        project="learning",
        outcome="success",
        emotional_valence=0.8,
    )
    print(f"Created episode: {episode.id}")

    # Search memories
    results = ww.recall_episodes("async patterns")
    for ep in results.episodes:
        print(f"  - {ep.content[:50]}...")
```

### Asynchronous Client

```python
import asyncio
from ww.sdk import AsyncWorldWeaverClient

async def main():
    async with AsyncWorldWeaverClient(session_id="my-session") as ww:
        # Create entity
        entity = await ww.create_entity(
            name="AsyncIO",
            entity_type="CONCEPT",
            summary="Python's asynchronous I/O framework",
        )

        # Spread activation from entity
        result = await ww.spread_activation(
            entity_id=entity.id,
            depth=2,
            threshold=0.1,
        )
        for e, activation in zip(result.entities, result.activations):
            print(f"{e.name}: {activation:.2f}")

asyncio.run(main())
```

## API Reference

### WorldWeaverClient / AsyncWorldWeaverClient

#### Constructor

```python
client = WorldWeaverClient(
    base_url="http://localhost:8765",  # API server URL
    session_id="default",               # Session for memory isolation
    timeout=30.0,                        # Request timeout in seconds
)
```

### System Methods

#### health() -> HealthStatus
Check API health status.

```python
status = ww.health()
print(f"Status: {status.status}, Version: {status.version}")
```

#### stats() -> MemoryStats
Get memory statistics.

```python
stats = await ww.stats()
print(f"Episodes: {stats.episodic['total_episodes']}")
```

#### consolidate(deep: bool = False) -> dict
Trigger memory consolidation.

```python
result = await ww.consolidate(deep=True)
```

### Episodic Memory

#### create_episode(...) -> Episode
Create a new episodic memory.

```python
episode = await ww.create_episode(
    content="What I learned or did",
    project="project-name",
    file="path/to/file.py",
    tool="python",
    outcome="success",  # success, failure, partial, neutral
    emotional_valence=0.7,  # 0.0 to 1.0 (importance)
    timestamp=datetime.now(),
)
```

#### get_episode(episode_id: UUID) -> Episode
Get episode by ID.

#### list_episodes(page, page_size, project, outcome) -> (list[Episode], int)
List episodes with pagination.

```python
episodes, total = await ww.list_episodes(page=1, page_size=20, project="ww")
```

#### recall_episodes(query, limit, min_similarity, project, outcome) -> RecallResult
Semantic search for episodes.

```python
results = await ww.recall_episodes(
    query="memory algorithms",
    limit=10,
    min_similarity=0.5,
)
for episode, score in zip(results.episodes, results.scores):
    print(f"{score:.2f}: {episode.content[:50]}...")
```

#### delete_episode(episode_id: UUID) -> None
Delete an episode.

#### mark_important(episode_id: UUID, importance: float) -> Episode
Mark episode as important (reduces decay).

### Semantic Memory

#### create_entity(...) -> Entity
Create a knowledge entity.

```python
entity = await ww.create_entity(
    name="FSRS",
    entity_type="TECHNIQUE",  # CONCEPT, PERSON, PROJECT, TOOL, TECHNIQUE, FACT
    summary="Free Spaced Repetition Scheduler",
    details="Extended description...",
)
```

#### get_entity(entity_id: UUID) -> Entity
Get entity by ID.

#### list_entities(entity_type, limit) -> list[Entity]
List entities with filtering.

#### create_relation(...) -> Relationship
Create relationship between entities.

```python
relation = await ww.create_relation(
    source_id=entity1.id,
    target_id=entity2.id,
    relation_type="IMPLEMENTS",  # USES, PRODUCES, REQUIRES, CAUSES, etc.
    weight=0.5,
)
```

#### recall_entities(query, limit, entity_types) -> list[Entity]
Semantic search for entities.

#### spread_activation(entity_id, depth, threshold) -> ActivationResult
Perform spreading activation from an entity.

```python
result = await ww.spread_activation(
    entity_id=start_entity.id,
    depth=3,
    threshold=0.1,
)
```

#### supersede_entity(entity_id, name, entity_type, summary, details) -> Entity
Update entity with bi-temporal versioning.

### Procedural Memory

#### create_skill(...) -> Skill
Create a procedural skill.

```python
skill = await ww.create_skill(
    name="git-commit",
    domain="coding",  # coding, research, trading, devops, writing
    task="Create a git commit with proper message",
    steps=[
        {"order": 1, "action": "Stage changes", "tool": "git"},
        {"order": 2, "action": "Write commit message"},
        {"order": 3, "action": "Create commit", "tool": "git"},
    ],
    script="Stage, write message, commit",
)
```

#### get_skill(skill_id: UUID) -> Skill
Get skill by ID.

#### list_skills(domain, include_deprecated, limit) -> list[Skill]
List skills with filtering.

#### recall_skills(query, domain, limit) -> list[Skill]
Semantic search for skills.

#### record_execution(skill_id, success, duration_ms, notes) -> Skill
Record skill execution result.

```python
skill = await ww.record_execution(
    skill_id=skill.id,
    success=True,
    duration_ms=5000,
)
print(f"Success rate: {skill.success_rate:.1%}")
```

#### deprecate_skill(skill_id, replacement_id) -> Skill
Deprecate a skill with optional replacement.

#### how_to(query, domain) -> (Skill | None, list[str], float)
Natural language query for procedural knowledge.

```python
skill, steps, confidence = await ww.how_to("write pytest tests")
if skill:
    print(f"Found: {skill.name} (confidence: {confidence:.0%})")
    for step in steps:
        print(f"  {step}")
```

## Data Models

### Episode
```python
@dataclass
class Episode:
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
@dataclass
class Entity:
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
@dataclass
class Skill:
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
    created_at: datetime
```

## Error Handling

```python
from ww.sdk.client import (
    WorldWeaverError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
)

try:
    episode = await ww.get_episode(some_id)
except NotFoundError:
    print("Episode not found")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ConnectionError:
    print("Cannot connect to World Weaver API")
except WorldWeaverError as e:
    print(f"API error: {e.status_code} - {e.response}")
```

## Best Practices

1. **Use context managers** for proper connection cleanup
2. **Prefer async client** for better performance with many requests
3. **Set appropriate session_id** to isolate memory namespaces
4. **Use semantic search** rather than listing all records
5. **Record skill executions** to improve success rate tracking
6. **Mark important memories** to reduce decay
