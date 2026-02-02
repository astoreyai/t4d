# Quick Start

Get started with T4DM in 5 minutes.

## Prerequisites

- Python 3.10+
- T4DM installed (`pip install t4dm`)

## Your First Memory Operations

### 1. Store a Memory

=== "Python API"

    ```python
    from ww import memory
    import asyncio

    async def main():
        # Store an episodic memory
        await memory.store(
            "Learned that T4DM uses tripartite memory architecture",
            importance=0.8,
            tags=["learning", "architecture"]
        )
        print("Memory stored!")

    asyncio.run(main())
    ```

=== "CLI"

    ```bash
    ww store "Learned that T4DM uses tripartite memory architecture" \
        --importance 0.8 \
        --tags "learning,architecture"
    ```

=== "REST API"

    ```bash
    curl -X POST http://localhost:8765/api/v1/episodes \
      -H "Content-Type: application/json" \
      -d '{
        "content": "Learned that T4DM uses tripartite memory architecture",
        "emotional_valence": 0.8
      }'
    ```

### 2. Recall Memories

=== "Python API"

    ```python
    from ww import memory
    import asyncio

    async def main():
        # Search for related memories
        results = await memory.recall("memory architecture", limit=5)

        for r in results:
            print(f"[{r.memory_type}] {r.content[:50]}... (score: {r.score:.2f})")

    asyncio.run(main())
    ```

=== "CLI"

    ```bash
    ww recall "memory architecture" --k 5
    ```

=== "REST API"

    ```bash
    curl -X POST http://localhost:8765/api/v1/episodes/recall \
      -H "Content-Type: application/json" \
      -d '{"query": "memory architecture", "limit": 5}'
    ```

### 3. Store Semantic Knowledge

=== "Python API"

    ```python
    from ww import memory
    import asyncio

    async def main():
        # Store an entity (semantic memory)
        await memory.store_entity(
            name="T4DM",
            description="A biologically-inspired memory system for AI",
            entity_type="concept"
        )

        # Store another entity and link them
        await memory.store_entity(
            name="Tripartite Memory",
            description="Memory architecture with episodic, semantic, and procedural subsystems",
            entity_type="concept"
        )
        print("Entities stored!")

    asyncio.run(main())
    ```

=== "CLI"

    ```bash
    ww semantic add "T4DM" \
        --desc "A biologically-inspired memory system for AI" \
        --type concept
    ```

### 4. Store Procedural Skills

=== "Python API"

    ```python
    from ww import memory
    import asyncio

    async def main():
        # Store a skill (procedural memory)
        await memory.store_skill(
            name="query_memories",
            script="results = await memory.recall(query, limit=k)",
            domain="coding",
            description="Query T4DM for relevant memories"
        )
        print("Skill stored!")

    asyncio.run(main())
    ```

=== "CLI"

    ```bash
    ww procedural add "query_memories" \
        --desc "Query T4DM for relevant memories"
    ```

## Session Isolation

Use session contexts for isolated memory spaces:

```python
from ww import memory
import asyncio

async def main():
    # Project A's memories
    async with memory.session("project-alpha") as m:
        await m.store("Alpha-specific configuration")
        results = await m.recall("configuration")
        print(f"Project Alpha has {len(results)} results")

    # Project B's memories (isolated)
    async with memory.session("project-beta") as m:
        await m.store("Beta-specific settings")
        results = await m.recall("configuration")
        print(f"Project Beta has {len(results)} results")

asyncio.run(main())
```

## Using the REST API Server

### Start the Server

```bash
# Start with default settings
t4dm serve

# Or with custom port
t4dm serve --port 8080

# Or with auto-reload for development
t4dm serve --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/episodes` | POST | Create episode |
| `/api/v1/episodes/recall` | POST | Search episodes |
| `/api/v1/entities` | POST | Create entity |
| `/api/v1/entities/recall` | POST | Search entities |
| `/api/v1/skills` | POST | Create skill |
| `/api/v1/skills/recall` | POST | Search skills |

### Interactive Documentation

Access Swagger UI at `http://localhost:8765/docs` for interactive API exploration.

## Next Steps

<div class="grid cards" markdown>

-   :material-cog: **[Configuration](configuration.md)**

    ---

    Configure storage backends, embedding models, and more

-   :material-brain: **[Architecture](../concepts/architecture.md)**

    ---

    Understand the tripartite memory architecture

-   :material-hook: **[Hooks](../guides/hooks.md)**

    ---

    Extend T4DM with custom hooks

-   :material-api: **[API Reference](../reference/rest-api.md)**

    ---

    Complete API documentation

</div>
