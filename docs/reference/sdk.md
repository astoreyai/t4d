# Python SDK Reference

Python client library for World Weaver.

## Installation

```bash
pip install world-weaver
```

## Quick Start

```python
from ww.sdk import AsyncWorldWeaverClient, WorldWeaverClient

# Async client
async with AsyncWorldWeaverClient() as client:
    episode = await client.create_episode("Hello World")
    results = await client.recall_episodes("hello")

# Sync client
with WorldWeaverClient() as client:
    episode = client.create_episode("Hello World")
    results = client.recall_episodes("hello")
```

## Client Configuration

```python
client = AsyncWorldWeaverClient(
    base_url="http://localhost:8765",
    api_key="your-api-key",
    session_id="my-session",
    timeout=30.0
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | str | `"http://localhost:8765"` | API server URL |
| `api_key` | str | None | API key for authentication |
| `session_id` | str | `"default"` | Session identifier |
| `timeout` | float | 30.0 | Request timeout in seconds |

## Episodes

### Create Episode

```python
episode = await client.create_episode(
    content="Learned about Python SDK",
    project="sdk-project",
    file="sdk.py",
    tool="python",
    outcome="SUCCESS",
    emotional_valence=0.8
)
```

### Get Episode

```python
episode = await client.get_episode(episode_id)
```

### List Episodes

```python
episodes = await client.list_episodes(
    offset=0,
    limit=50,
    project="sdk-project"
)

for ep in episodes.items:
    print(f"{ep.id}: {ep.content[:50]}")
```

### Recall Episodes

```python
results = await client.recall_episodes(
    query="Python SDK",
    limit=10,
    min_similarity=0.7
)

for r in results:
    print(f"Score: {r.score:.2f} - {r.episode.content}")
```

### Update Episode

```python
updated = await client.update_episode(
    episode_id,
    content="Updated content",
    outcome="SUCCESS"
)
```

### Delete Episode

```python
await client.delete_episode(episode_id)
```

### Mark Important

```python
await client.mark_important(episode_id, importance=1.0)
```

## Entities

### Create Entity

```python
entity = await client.create_entity(
    name="Python SDK",
    entity_type="TOOL",
    summary="Client library for World Weaver"
)
```

### Recall Entities

```python
results = await client.recall_entities(
    query="client library",
    limit=10
)
```

### Spread Activation

```python
activation = await client.spread_activation(
    entity_id,
    depth=2,
    decay_factor=0.5
)

for entity_id, score in activation.activations.items():
    print(f"{entity_id}: {score:.3f}")
```

## Skills

### Create Skill

```python
skill = await client.create_skill(
    name="run_tests",
    domain="CODING",
    description="Run pytest test suite",
    script="pytest tests/ -v"
)
```

### Recall Skills

```python
results = await client.recall_skills(
    query="run tests",
    limit=5
)
```

### How-To Query

```python
skill, steps, confidence = await client.how_to("run the tests")

print(f"Skill: {skill.name} (confidence: {confidence:.2f})")
for step in steps:
    print(f"  {step.order}. {step.action}")
```

### Record Execution

```python
await client.record_execution(
    skill_id,
    success=True,
    duration_ms=1500,
    output="10 tests passed"
)
```

## System

### Health Check

```python
health = await client.health()
print(f"Status: {health.status}")
print(f"Version: {health.version}")
```

### Statistics

```python
stats = await client.stats()
print(f"Episodes: {stats.episodic.count}")
print(f"Entities: {stats.semantic.entity_count}")
print(f"Skills: {stats.procedural.skill_count}")
```

### Consolidate

```python
await client.consolidate(deep=True)
```

## Error Handling

```python
from ww.sdk import (
    WorldWeaverError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ValidationError
)

try:
    episode = await client.get_episode(episode_id)
except NotFoundError:
    print("Episode not found")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except ConnectionError:
    print("Failed to connect to server")
except ValidationError as e:
    print(f"Invalid request: {e.message}")
except WorldWeaverError as e:
    print(f"API error: {e}")
```

## Models

### Episode

```python
@dataclass
class Episode:
    id: UUID
    session_id: str
    content: str
    timestamp: datetime
    outcome: str | None
    emotional_valence: float
    context: EpisodeContext
    access_count: int
    stability: float
    retrievability: float
```

### Entity

```python
@dataclass
class Entity:
    id: UUID
    name: str
    entity_type: str
    summary: str
    details: dict
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
    trigger_pattern: str
    steps: list[Step]
    script: str | None
    success_rate: float
    execution_count: int
    last_executed: datetime | None
    version: int
    deprecated: bool
    created_at: datetime
```

## Async vs Sync

### Async Client (Recommended)

```python
import asyncio
from ww.sdk import AsyncWorldWeaverClient

async def main():
    async with AsyncWorldWeaverClient() as client:
        # Concurrent operations
        results = await asyncio.gather(
            client.recall_episodes("query 1"),
            client.recall_episodes("query 2"),
            client.recall_episodes("query 3")
        )

asyncio.run(main())
```

### Sync Client

```python
from ww.sdk import WorldWeaverClient

with WorldWeaverClient() as client:
    # Sequential operations
    r1 = client.recall_episodes("query 1")
    r2 = client.recall_episodes("query 2")
```

## Advanced Usage

### Custom Session

```python
# Different sessions for different contexts
async with AsyncWorldWeaverClient(session_id="project-a") as client_a:
    await client_a.create_episode("Project A data")

async with AsyncWorldWeaverClient(session_id="project-b") as client_b:
    await client_b.create_episode("Project B data")
```

### Batch Operations

```python
# Create multiple episodes
episodes = []
for content in contents:
    ep = await client.create_episode(content)
    episodes.append(ep)
```

### Pagination

```python
all_episodes = []
offset = 0
while True:
    result = await client.list_episodes(offset=offset, limit=100)
    all_episodes.extend(result.items)
    if not result.has_more:
        break
    offset += 100
```
