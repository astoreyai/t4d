# T4DM SDK Quickstart Guide

**Version**: 2.0.0
**Purpose**: Get started with T4DM Python SDK in 5 minutes

---

## Installation

```bash
# From PyPI
pip install t4dm

# With API extras
pip install t4dm[api]

# From source
git clone https://github.com/astoreyai/t4dm
cd t4dm
pip install -e ".[api]"
```

---

## 1. Basic Usage

### Store and Recall

```python
from t4dm.sdk import T4DMClient

# Connect to T4DM server
client = T4DMClient(
    base_url="http://localhost:8765",
    session_id="quickstart-demo"
)

# Store an episodic memory
episode = client.create_episode(
    content="Learned how to use T4DM for AI memory",
    context={
        "project": "my-app",
        "category": "learning"
    },
    outcome="success",
    emotional_valence=0.8  # Positive experience
)
print(f"Created: {episode.id}")

# Recall relevant memories
results = client.recall_episodes("T4DM memory", limit=5)
for ep in results.episodes:
    print(f"  [{ep.timestamp}] {ep.content}")

# Don't forget to close
client.close()
```

### Context Manager (Recommended)

```python
from t4dm.sdk import T4DMClient

# Automatic cleanup with context manager
with T4DMClient(session_id="my-session") as client:
    client.create_episode(content="Task completed")
    results = client.recall_episodes("completed")
```

---

## 2. Async Usage

```python
import asyncio
from t4dm.sdk import AsyncT4DMClient

async def main():
    async with AsyncT4DMClient(session_id="async-demo") as client:
        # Parallel memory operations
        tasks = [
            client.create_episode(content=f"Event {i}")
            for i in range(10)
        ]
        episodes = await asyncio.gather(*tasks)

        print(f"Created {len(episodes)} episodes")

asyncio.run(main())
```

---

## 3. Memory Types

### Episodic (Events & Experiences)

```python
# Store an experience
episode = client.create_episode(
    content="User clicked the Buy button",
    context={
        "user_id": "u123",
        "page": "checkout",
        "button": "buy"
    },
    outcome="success",
    emotional_valence=0.7
)

# Search by time range
from datetime import datetime, timedelta
yesterday = datetime.now() - timedelta(days=1)

results = client.recall_episodes(
    query="checkout",
    time_range={"start": yesterday.isoformat()}
)
```

### Semantic (Concepts & Knowledge)

```python
# Create a concept
entity = client.create_entity(
    name="T4DM",
    entity_type="CONCEPT",
    summary="Biologically-inspired AI memory system"
)

# Create relationships
client.create_relation(
    source_id=entity.id,
    target_name="AI",
    relation_type="IS_A"
)

# Spread activation to find related concepts
result = client.spread_activation(
    entity_id=entity.id,
    depth=2,
    threshold=0.1
)
for ent, score in zip(result.entities, result.activations):
    print(f"  {ent.name}: {score:.2f}")
```

### Procedural (Skills & Patterns)

```python
# Register a skill
skill = client.create_skill(
    name="send_email",
    description="Send an email to a recipient",
    input_schema={"to": "str", "subject": "str", "body": "str"},
    output_schema={"success": "bool", "message_id": "str"}
)

# Track execution
client.record_execution(
    skill_id=skill.id,
    success=True,
    duration_ms=150,
    context={"to": "user@example.com"}
)

# Get skill recommendations
recommendations = client.recommend_skills(
    context="need to notify the team"
)
```

---

## 4. κ (Kappa) Queries

Query by consolidation level:

```python
# Only get consolidated (stable) memories
stable = client.recall_episodes(
    query="important concepts",
    kappa_min=0.6
)

# Get recent (unconsolidated) memories
recent = client.recall_episodes(
    query="today's events",
    kappa_max=0.3
)

# Get κ distribution
from t4dm.sdk import VizClient
viz = VizClient(base_url="http://localhost:8765")
distribution = viz.kappa_distribution()
print(f"Episodic: {distribution['band_counts']['episodic']}")
print(f"Semantic: {distribution['band_counts']['semantic']}")
```

---

## 5. Consolidation

Trigger memory consolidation:

```python
# Light consolidation (fast, NREM-like)
result = client.consolidate(phase="nrem")
print(f"Replayed: {result['replayed']} memories")

# Deep consolidation (slower, REM-like)
result = client.consolidate(phase="rem")
print(f"Created: {result['prototypes']} semantic prototypes")

# Full sleep cycle
result = client.consolidate(phase="full_cycle")
```

---

## 6. Visualization Data

Access visualization endpoints:

```python
from t4dm.sdk import VizClient

viz = VizClient(base_url="http://localhost:8765")

# Get real-time metrics
metrics = viz.realtime_metrics()
print(f"Total memories: {metrics['kappa']['total_items']}")
print(f"Mean κ: {metrics['kappa']['mean_kappa']:.2f}")

# Get oscillator phases
phases = viz.oscillator_phase()
print(f"Theta: {phases['theta_phase']:.2f}")
print(f"Gamma: {phases['gamma_phase']:.2f}")

# Export all viz data
export = viz.export_all()
for module, data in export['modules'].items():
    print(f"  {module}: {len(data)} data points")
```

---

## 7. Error Handling

```python
from t4dm.sdk import T4DMClient
from t4dm.sdk.errors import (
    RateLimitError,
    NotFoundError,
    ValidationError
)

client = T4DMClient(session_id="error-handling")

try:
    result = client.recall_episodes("query")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except NotFoundError:
    print("Memory not found")
except ValidationError as e:
    print(f"Invalid request: {e.details}")
```

---

## 8. Agent Integration

For multi-agent systems:

```python
from t4dm.sdk import AgentClient

# Agent-specific memory space
agent = AgentClient(
    base_url="http://localhost:8765",
    agent_id="research-agent",
    session_id="project-alpha"
)

# Store agent-scoped memory
agent.remember("Found relevant paper on memory systems")

# Recall with agent context
memories = agent.recall("memory systems", include_other_agents=False)

# Share with other agents
agent.share_memory(memory_id, target_agents=["analysis-agent"])
```

---

## Next Steps

- [Full SDK Reference](../SDK.md)
- [REST API Documentation](../API.md)
- [Integration Guide](../integration/README.md)
- [Architecture Overview](../ARCHITECTURE.md)
