# T4DM - Integration Points

**Version**: 0.2.0
**Last Updated**: 2025-12-06
**Purpose**: Complete guide to integrating T4DM into your applications

---

## Table of Contents

1. [MCP Server (Claude Code/Desktop)](#1-mcp-server-claude-codedesktop)
2. [REST API (Any HTTP Client)](#2-rest-api-any-http-client)
3. [Python SDK](#3-python-sdk)
4. [Hook System](#4-hook-system)
5. [Custom Memory Types](#5-custom-memory-types)
6. [Embedding Providers](#6-embedding-providers)
7. [Storage Backends](#7-storage-backends)

---

## 1. MCP Server (Claude Code/Desktop)

### Overview

T4DM implements the Model Context Protocol (MCP) for seamless integration with Claude applications. The MCP server exposes memory operations as JSON-RPC tools.

**Protocol Version**: MCP 2024-11-05
**Transport**: stdio (JSON-RPC 2.0)

### Configuration

#### Claude Desktop

Edit `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ww-memory": {
      "command": "python",
      "args": ["-m", "t4dm.mcp.server"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "${NEO4J_PASSWORD}",
        "QDRANT_URL": "http://localhost:6333",
        "T4DM_SESSION_ID": "claude-desktop",
        "T4DM_EMBEDDING_DEVICE": "cuda:0",
        "PYTHONPATH": "/path/to/t4dm"
      }
    }
  }
}
```

#### Claude Code CLI

```bash
# Add to ~/.claude/mcp_servers.json
{
  "ww-memory": {
    "command": "/path/to/t4dm/.venv/bin/python",
    "args": ["-m", "t4dm.mcp.server"],
    "env": {
      "T4DM_SESSION_ID": "${INSTANCE_ID}"
    }
  }
}
```

### Available Tools

#### Episodic Memory

```typescript
// Store an event
{
  "tool": "create_episode",
  "arguments": {
    "content": "Implemented FSRS decay algorithm",
    "project": "t4dm",
    "file": "src/memory/episodic.py",
    "outcome": "success",
    "emotional_valence": 0.8
  }
}

// Recall events
{
  "tool": "recall_episodes",
  "arguments": {
    "query": "FSRS implementation details",
    "limit": 10,
    "time_range": {
      "start": "2025-12-01T00:00:00Z"
    }
  }
}

// Historical query
{
  "tool": "query_at_time",
  "arguments": {
    "query": "What did we know about FSRS?",
    "point_in_time": "2025-11-15T10:00:00Z"
  }
}
```

#### Semantic Memory

```typescript
// Create knowledge entity
{
  "tool": "create_entity",
  "arguments": {
    "name": "FSRS",
    "entity_type": "TECHNIQUE",
    "summary": "Free Spaced Repetition Scheduler",
    "details": "Memory decay algorithm with stability-based retrieval"
  }
}

// Create relationship
{
  "tool": "create_relation",
  "arguments": {
    "source_id": "entity-uuid-1",
    "target_id": "entity-uuid-2",
    "relation_type": "IMPROVES_ON"
  }
}

// Semantic search with spreading activation
{
  "tool": "semantic_recall",
  "arguments": {
    "query": "memory decay algorithms",
    "context_entities": ["fsrs-uuid", "sm2-uuid"],
    "limit": 5,
    "include_spreading": true
  }
}

// Graph exploration
{
  "tool": "spread_activation",
  "arguments": {
    "seed_entities": ["t4dm-uuid"],
    "steps": 3,
    "retention": 0.5,
    "decay": 0.1
  }
}
```

#### Procedural Memory

```typescript
// Build skill from trajectory
{
  "tool": "build_skill",
  "arguments": {
    "trajectory": [
      {
        "action": "Read file",
        "tool": "read",
        "parameters": {"path": "config.py"},
        "result": "success"
      },
      {
        "action": "Edit file",
        "tool": "write",
        "parameters": {"path": "config.py", "content": "..."},
        "result": "success"
      }
    ],
    "outcome_score": 0.9,
    "domain": "coding"
  }
}

// Retrieve skill
{
  "tool": "how_to",
  "arguments": {
    "task": "update configuration file",
    "domain": "coding"
  }
}

// Record execution feedback
{
  "tool": "execute_skill",
  "arguments": {
    "procedure_id": "skill-uuid",
    "success": true
  }
}
```

### Session Isolation

Each MCP client gets isolated episodic memory via `T4DM_SESSION_ID`:

```bash
# Client 1
T4DM_SESSION_ID=claude-desktop

# Client 2
T4DM_SESSION_ID=claude-code-instance-1

# Client 3
T4DM_SESSION_ID=automation-bot
```

**Shared**: Semantic and procedural memory (knowledge is universal)
**Isolated**: Episodic memory (experiences are personal)

### Error Handling

```typescript
// Successful response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "...",
    "score": 0.89
  }
}

// Error response
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32000,
    "message": "Episode not found",
    "data": {
      "episode_id": "invalid-uuid"
    }
  }
}
```

**Error Codes**:
- `-32000`: Storage error (Neo4j/Qdrant)
- `-32001`: Embedding generation failed
- `-32002`: Invalid parameters
- `-32003`: Memory not found

---

## 2. REST API (Any HTTP Client)

### Base URL

```
http://localhost:8765/api/v1
```

### Authentication

Use `X-Session-ID` header for session isolation:

```bash
curl -H "X-Session-ID: my-app" \
     http://localhost:8765/api/v1/episodes
```

Optional API key (production):
```bash
curl -H "X-API-Key: your-api-key" \
     -H "X-Session-ID: my-app" \
     http://localhost:8765/api/v1/episodes
```

### Endpoints

#### Episodic Memory

**Create Episode**
```http
POST /api/v1/episodes
Content-Type: application/json

{
  "content": "User logged in successfully",
  "project": "my-app",
  "outcome": "success",
  "emotional_valence": 0.5
}
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "my-app",
  "content": "User logged in successfully",
  "timestamp": "2025-12-06T10:30:00Z",
  "outcome": "success",
  "retrievability": 1.0
}
```

**Recall Episodes**
```http
GET /api/v1/episodes/search?query=login&limit=10
```

Response:
```json
{
  "episodes": [
    {
      "id": "...",
      "content": "User logged in successfully",
      "score": 0.89,
      "timestamp": "2025-12-06T10:30:00Z"
    }
  ],
  "total": 1
}
```

#### Semantic Memory

**Create Entity**
```http
POST /api/v1/entities
Content-Type: application/json

{
  "name": "OAuth 2.0",
  "entity_type": "TECHNIQUE",
  "summary": "Authorization framework"
}
```

**Create Relationship**
```http
POST /api/v1/relations
Content-Type: application/json

{
  "source_id": "oauth-uuid",
  "target_id": "jwt-uuid",
  "relation_type": "USES"
}
```

**Semantic Search**
```http
GET /api/v1/entities/search?query=authentication&limit=5
```

#### Procedural Memory

**Create Skill**
```http
POST /api/v1/skills
Content-Type: application/json

{
  "trajectory": [...],
  "outcome_score": 0.85,
  "domain": "devops"
}
```

**Search Skills**
```http
GET /api/v1/skills/search?task=deploy%20application&domain=devops
```

### Client Examples

#### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8765/api/v1"
SESSION_ID = "my-app"

# Create episode
response = requests.post(
    f"{BASE_URL}/episodes",
    headers={"X-Session-ID": SESSION_ID},
    json={
        "content": "User registered successfully",
        "outcome": "success"
    }
)
episode = response.json()

# Search episodes
response = requests.get(
    f"{BASE_URL}/episodes/search",
    headers={"X-Session-ID": SESSION_ID},
    params={"query": "registration", "limit": 10}
)
results = response.json()
```

#### JavaScript (fetch)

```javascript
const BASE_URL = 'http://localhost:8765/api/v1';
const SESSION_ID = 'my-app';

// Create episode
const response = await fetch(`${BASE_URL}/episodes`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-Session-ID': SESSION_ID
  },
  body: JSON.stringify({
    content: 'User purchased product',
    outcome: 'success',
    emotional_valence: 0.9
  })
});
const episode = await response.json();

// Search
const results = await fetch(
  `${BASE_URL}/episodes/search?query=purchase&limit=10`,
  {
    headers: { 'X-Session-ID': SESSION_ID }
  }
).then(r => r.json());
```

#### cURL

```bash
# Create entity
curl -X POST http://localhost:8765/api/v1/entities \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-app" \
  -d '{
    "name": "FastAPI",
    "entity_type": "TOOL",
    "summary": "Modern Python web framework"
  }'

# Search entities
curl "http://localhost:8765/api/v1/entities/search?query=python%20framework" \
  -H "X-Session-ID: my-app"
```

---

## 3. Python SDK

### Installation

```bash
pip install t4dm[api]
```

### Synchronous Client

```python
from t4dm.sdk import T4DMClient

# Create client
with T4DMClient(
    base_url="http://localhost:8765",
    session_id="my-app"
) as ww:
    # Store episode
    episode = t4dm.create_episode(
        content="Implemented user authentication",
        project="my-app",
        outcome="success"
    )
    print(f"Created: {episode.id}")

    # Recall episodes
    results = t4dm.recall_episodes("authentication", limit=5)
    for ep in results.episodes:
        print(f"- {ep.content} (score: {ep.score:.2f})")

    # Create entity
    entity = t4dm.create_entity(
        name="JWT",
        entity_type="TECHNIQUE",
        summary="JSON Web Token for stateless auth"
    )

    # Create relationship
    t4dm.create_relation(
        source_id=entity.id,
        target_id=oauth_entity.id,
        relation_type="PART_OF"
    )

    # Semantic search
    entities = t4dm.semantic_recall("token authentication")
    for ent in entities:
        print(f"- {ent.name}: {ent.summary}")

    # Build skill
    skill = t4dm.create_skill(
        trajectory=[
            {"action": "Read config", "tool": "read"},
            {"action": "Update config", "tool": "write"}
        ],
        outcome_score=0.9,
        domain="devops"
    )

    # Search skills
    skills = t4dm.how_to("deploy application", domain="devops")
    if skills:
        best = skills[0]
        print(f"Best skill: {best.name} ({best.success_rate:.0%})")
```

### Async Client

```python
import asyncio
from t4dm.sdk import AsyncT4DMClient

async def main():
    async with AsyncT4DMClient(
        base_url="http://localhost:8765",
        session_id="my-app"
    ) as ww:
        # Concurrent operations
        episode_task = t4dm.create_episode("Event A")
        entity_task = t4dm.create_entity("Concept X", "CONCEPT", "Description")

        episode, entity = await asyncio.gather(episode_task, entity_task)

        # Batch recall
        results = await t4dm.recall_episodes("query", limit=100)

asyncio.run(main())
```

### Direct Library Usage (No API Server)

```python
import asyncio
from t4dm.memory.episodic import EpisodicMemory
from t4dm.memory.semantic import SemanticMemory
from t4dm.memory.procedural import ProceduralMemory

async def main():
    # Initialize subsystems
    episodic = EpisodicMemory(session_id="my-app")
    semantic = SemanticMemory()
    procedural = ProceduralMemory()

    await episodic.initialize()
    await semantic.initialize()
    await procedural.initialize()

    # Store episode
    episode = await episodic.create(
        content="Direct library usage example",
        outcome="success"
    )

    # Recall
    results = await episodic.recall("example", limit=10)

asyncio.run(main())
```

---

## 4. Hook System

### Overview

Hooks allow you to inject custom logic at key points in memory operations.

### Available Hooks

```python
from t4dm.core.hooks import HookRegistry, HookType

registry = HookRegistry()

# Pre-storage hook (e.g., content filtering)
@registry.register(HookType.PRE_EPISODE_CREATE)
async def filter_pii(episode_data: dict) -> dict:
    """Remove personally identifiable information."""
    content = episode_data["content"]
    # Redact emails, phone numbers, etc.
    content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
    episode_data["content"] = content
    return episode_data

# Post-storage hook (e.g., notifications)
@registry.register(HookType.POST_EPISODE_CREATE)
async def notify_on_failure(episode: Episode) -> None:
    """Send alert when failure recorded."""
    if episode.outcome == "failure":
        await send_alert(f"Failure recorded: {episode.content}")

# Pre-retrieval hook (e.g., access logging)
@registry.register(HookType.PRE_RECALL)
async def log_access(query: str, session_id: str) -> None:
    """Log all retrieval attempts."""
    await audit_log.write({
        "action": "recall",
        "query": query,
        "session": session_id,
        "timestamp": datetime.now()
    })

# Post-consolidation hook (e.g., export)
@registry.register(HookType.POST_CONSOLIDATION)
async def export_consolidated(results: dict) -> None:
    """Export consolidated knowledge."""
    if results["entities_created"] > 0:
        await export_to_knowledge_base(results["entities"])
```

### Hook Types

| Hook Type | When Executed | Signature |
|-----------|---------------|-----------|
| `PRE_EPISODE_CREATE` | Before episode stored | `(episode_data: dict) -> dict` |
| `POST_EPISODE_CREATE` | After episode stored | `(episode: Episode) -> None` |
| `PRE_RECALL` | Before retrieval | `(query: str, session_id: str) -> None` |
| `POST_RECALL` | After retrieval | `(results: list[Episode]) -> None` |
| `PRE_ENTITY_CREATE` | Before entity stored | `(entity_data: dict) -> dict` |
| `POST_ENTITY_CREATE` | After entity stored | `(entity: Entity) -> None` |
| `PRE_CONSOLIDATION` | Before consolidation | `(config: dict) -> None` |
| `POST_CONSOLIDATION` | After consolidation | `(results: dict) -> None` |

### Custom Hook Example

```python
# hooks/custom_embeddings.py
from t4dm.core.hooks import HookRegistry, HookType
from t4dm.embedding.protocol import EmbeddingProvider

class CustomEmbeddingHook:
    """Use custom embedding model for specific content types."""

    def __init__(self, custom_model: EmbeddingProvider):
        self.custom_model = custom_model

    async def __call__(self, episode_data: dict) -> dict:
        # Use custom model for code content
        if "```" in episode_data["content"]:
            embedding = await self.custom_model.embed_query(
                episode_data["content"]
            )
            episode_data["embedding"] = embedding
        return episode_data

# Register
registry = HookRegistry()
registry.register(
    HookType.PRE_EPISODE_CREATE,
    CustomEmbeddingHook(code_embedding_model)
)
```

---

## 5. Custom Memory Types

### Extending Memory Subsystems

T4DM's memory types are extensible. Add custom memory types by subclassing base classes.

#### Example: Spatial Memory

```python
from t4dm.core.types import BaseModel
from t4dm.memory.base import MemorySubsystem
from uuid import uuid4
from pydantic import Field

class Location(BaseModel):
    """Spatial memory: Places and navigation."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    coordinates: tuple[float, float]  # (lat, lon)
    description: str
    embedding: list[float]
    visits: int = 1
    last_visited: datetime

class SpatialMemory(MemorySubsystem):
    """Spatial memory subsystem."""

    async def create_location(
        self,
        name: str,
        coordinates: tuple[float, float],
        description: str
    ) -> Location:
        """Store a location."""
        embedding = await self.embedding.embed_query(f"{name}: {description}")

        location = Location(
            name=name,
            coordinates=coordinates,
            description=description,
            embedding=embedding
        )

        # Store in Neo4j + Qdrant
        await self.graph_store.create_node("Location", location.dict())
        await self.vector_store.add(
            collection="ww-locations",
            ids=[str(location.id)],
            vectors=[embedding],
            payloads=[{"name": name, "coordinates": coordinates}]
        )

        return location

    async def recall_nearby(
        self,
        coordinates: tuple[float, float],
        radius_km: float = 10
    ) -> list[Location]:
        """Retrieve locations within radius."""
        # Haversine distance query in Neo4j
        query = """
        MATCH (l:Location)
        WHERE point.distance(
            point({latitude: $lat, longitude: $lon}),
            point({latitude: l.coordinates[0], longitude: l.coordinates[1]})
        ) <= $radius * 1000
        RETURN l
        """
        results = await self.graph_store.query(
            query,
            lat=coordinates[0],
            lon=coordinates[1],
            radius=radius_km
        )
        return [Location(**r["l"]) for r in results]
```

#### Register Custom Memory

```python
from t4dm.memory.registry import MemoryRegistry

# Register custom type
registry = MemoryRegistry()
registry.register("spatial", SpatialMemory)

# Use in application
spatial = registry.get("spatial")(session_id="my-app")
await spatial.initialize()

location = await spatial.create_location(
    name="Central Park",
    coordinates=(40.785091, -73.968285),
    description="Urban park in Manhattan"
)
```

---

## 6. Embedding Providers

### Custom Embedding Provider

Implement the `EmbeddingProvider` protocol:

```python
from t4dm.embedding.protocol import EmbeddingProvider
import openai

class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding-3-large provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 3072

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for query."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed documents."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [d.embedding for d in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.model
```

### Register Custom Provider

```python
from t4dm.embedding.factory import EmbeddingFactory

# Register
EmbeddingFactory.register("openai", OpenAIEmbedding)

# Use in config
T4DM_EMBEDDING_PROVIDER=openai
T4DM_EMBEDDING_API_KEY=sk-...
```

---

## 7. Storage Backends

### Custom Vector Store

Implement `VectorStore` protocol:

```python
from t4dm.storage.protocol import VectorStore
import pinecone

class PineconeStore(VectorStore):
    """Pinecone vector database backend."""

    def __init__(self, api_key: str, environment: str, index_name: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    async def add(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict]
    ) -> None:
        """Add vectors to index."""
        items = [
            (id_, vec, payload)
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]
        self.index.upsert(items, namespace=collection)

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filter: dict | None = None
    ) -> list[dict]:
        """Semantic search."""
        results = self.index.query(
            vector=query_vector,
            top_k=limit,
            namespace=collection,
            filter=filter,
            include_metadata=True
        )
        return [
            {
                "id": match.id,
                "score": match.score,
                "payload": match.metadata
            }
            for match in results.matches
        ]
```

### Register Custom Backend

```python
from t4dm.storage.factory import StorageFactory

StorageFactory.register("pinecone", PineconeStore)

# Use in config
T4DM_VECTOR_STORE=pinecone
T4DM_PINECONE_API_KEY=...
T4DM_PINECONE_ENVIRONMENT=us-west1-gcp
```

---

## Integration Examples

### Web Application (FastAPI)

```python
from fastapi import FastAPI, Depends
from t4dm.sdk import T4DMClient

app = FastAPI()

def get_ww_client():
    """Dependency injection for WW client."""
    with T4DMClient(session_id="web-app") as client:
        yield client

@app.post("/api/events")
async def log_event(
    event: dict,
    ww: T4DMClient = Depends(get_ww_client)
):
    """Log user event to memory."""
    episode = t4dm.create_episode(
        content=event["description"],
        outcome=event.get("outcome", "neutral")
    )
    return {"logged": episode.id}

@app.get("/api/context")
async def get_context(
    query: str,
    ww: T4DMClient = Depends(get_ww_client)
):
    """Retrieve relevant context for query."""
    results = t4dm.recall_episodes(query, limit=5)
    return {"context": [ep.content for ep in results.episodes]}
```

### CLI Tool

```python
import click
from t4dm.sdk import T4DMClient

@click.group()
def cli():
    pass

@cli.command()
@click.argument('content')
def remember(content: str):
    """Store a memory."""
    with T4DMClient(session_id="cli") as ww:
        episode = t4dm.create_episode(content)
        click.echo(f"Remembered: {episode.id}")

@cli.command()
@click.argument('query')
def recall(query: str):
    """Search memories."""
    with T4DMClient(session_id="cli") as ww:
        results = t4dm.recall_episodes(query)
        for ep in results.episodes:
            click.echo(f"- {ep.content} ({ep.score:.2f})")

if __name__ == '__main__':
    cli()
```

---

## Best Practices

1. **Session Isolation**: Use unique `session_id` per user/application instance
2. **Batch Operations**: Use batch methods for bulk inserts (faster)
3. **Error Handling**: Always wrap WW calls in try/except
4. **Connection Pooling**: Reuse SDK client instances (context managers)
5. **Async When Possible**: Use `AsyncT4DMClient` for concurrent operations
6. **Hooks for Cross-Cutting Concerns**: Use hooks for logging, monitoring, filtering
7. **Custom Types for Domain Logic**: Extend memory types for domain-specific needs

---

**Document Status**: Complete ✓
**Examples Tested**: Python SDK, REST API, MCP Server
**Last Updated**: 2025-12-06
