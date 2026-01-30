# API Reference

Complete API documentation for World Weaver.

## API Layers

World Weaver provides four access layers:

```mermaid
graph TB
    subgraph External["External Access"]
        REST[REST API<br/>HTTP/WS]
        CLI[CLI<br/>Terminal]
    end

    subgraph Internal["Internal Access"]
        SDK[Python SDK<br/>HTTP Client]
        MEM[Memory API<br/>Direct]
    end

    subgraph Core["Core Services"]
        EP[Episodic]
        SEM[Semantic]
        PROC[Procedural]
    end

    REST --> Core
    CLI --> Core
    SDK --> REST
    MEM --> Core
```

## Quick Reference

| Interface | Best For | Authentication | Async |
|-----------|----------|----------------|-------|
| [REST API](rest-api.md) | External services | API Key | Yes |
| [Python SDK](sdk.md) | Python apps | API Key | Yes/No |
| [CLI](cli.md) | Terminal users | None | No |
| [Memory API](memory-api.md) | Direct embedding | None | Yes |
| [NCA API](nca-api.md) | Neural dynamics | None | Yes |

## Endpoint Overview

```mermaid
graph LR
    subgraph Episodes
        E1[POST /episodes]
        E2[GET /episodes]
        E3[POST /episodes/recall]
    end

    subgraph Entities
        N1[POST /entities]
        N2[GET /entities]
        N3[POST /entities/recall]
    end

    subgraph Skills
        S1[POST /skills]
        S2[GET /skills]
        S3[POST /skills/recall]
    end

    subgraph System
        SY1[GET /health]
        SY2[GET /stats]
        SY3[POST /consolidate]
    end
```

## Data Flow

### Store Operation

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Hooks
    participant Memory
    participant Storage

    Client->>API: POST /episodes
    API->>Hooks: PRE hooks
    Hooks->>Memory: create()
    Memory->>Memory: Generate embedding
    Memory->>Memory: Gate decision
    Memory->>Storage: Qdrant upsert
    Memory->>Storage: Neo4j create
    Storage-->>Memory: Success
    Memory-->>Hooks: Episode
    Hooks->>Hooks: POST hooks
    Hooks-->>API: Episode
    API-->>Client: 201 Created
```

### Recall Operation

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Hooks
    participant Memory
    participant Storage

    Client->>API: POST /episodes/recall
    API->>Hooks: PRE hooks (cache check)
    Hooks->>Memory: recall()
    Memory->>Memory: Query embedding
    Memory->>Storage: Qdrant search
    Storage-->>Memory: Candidates
    Memory->>Memory: Score fusion
    Memory->>Memory: Reranking
    Memory-->>Hooks: Results
    Hooks->>Hooks: POST hooks (cache update)
    Hooks-->>API: Results
    API-->>Client: 200 OK
```

## Authentication

### API Key Authentication

```mermaid
sequenceDiagram
    participant Client
    participant Middleware
    participant API

    Client->>Middleware: Request + X-API-Key
    Middleware->>Middleware: Validate key
    alt Valid Key
        Middleware->>API: Forward request
        API-->>Client: Response
    else Invalid Key
        Middleware-->>Client: 401 Unauthorized
    end
```

### Session Isolation

```mermaid
graph TB
    subgraph Session_A["Session A"]
        A1[Episodes A]
        A2[Entities A]
    end

    subgraph Session_B["Session B"]
        B1[Episodes B]
        B2[Entities B]
    end

    subgraph Storage
        Q[(Qdrant)]
        N[(Neo4j)]
    end

    Session_A -->|session_id=A| Storage
    Session_B -->|session_id=B| Storage
```

## Common Patterns

### Pagination

```python
# SDK
results = await client.list_episodes(offset=0, limit=50)
while results.has_more:
    results = await client.list_episodes(
        offset=results.offset + 50,
        limit=50
    )
```

### Error Handling

```python
from ww.sdk import WorldWeaverClient, NotFoundError, RateLimitError

try:
    episode = await client.get_episode(episode_id)
except NotFoundError:
    print("Episode not found")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
```

### WebSocket Events

```mermaid
sequenceDiagram
    participant Client
    participant WS
    participant Memory

    Client->>WS: Connect /ws/events
    WS-->>Client: Connected

    Memory->>WS: memory_added
    WS-->>Client: Event: memory_added

    Memory->>WS: memory_promoted
    WS-->>Client: Event: memory_promoted

    Client->>WS: Disconnect
```
