# World Weaver Memory Gateway MCP Server

Version: 0.1.0 | Protocol: MCP 2024-11-05

## Overview

The Memory Gateway is an MCP server that provides Claude Code instances with access to the tripartite memory system (episodic, semantic, procedural). It acts as a unified interface for memory operations across all memory types.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code Instances                     │
│  ┌────────┐  ┌────────┐  ┌────────┐                        │
│  │ CC #1  │  │ CC #2  │  │ CC #N  │                        │
│  └───┬────┘  └───┬────┘  └───┬────┘                        │
│      │           │           │                              │
│      └───────────┴───────────┘                              │
│                  │                                          │
│                  ▼                                          │
│  ┌──────────────────────────────────┐                      │
│  │     MCP Memory Gateway           │                      │
│  │  ┌───────────────────────────┐   │                      │
│  │  │ Session Manager           │   │                      │
│  │  │ - Instance namespacing    │   │                      │
│  │  │ - Request routing         │   │                      │
│  │  └───────────────────────────┘   │                      │
│  └──────────────────────────────────┘                      │
│                  │                                          │
│    ┌─────────────┼─────────────┐                           │
│    ▼             ▼             ▼                           │
│ ┌──────┐    ┌──────┐    ┌──────────┐                      │
│ │Episod│    │Semant│    │Procedural│                      │
│ │  ic  │    │  ic  │    │          │                      │
│ └──┬───┘    └──┬───┘    └────┬─────┘                      │
│    │           │             │                              │
│    └───────────┴─────────────┘                              │
│                │                                            │
│    ┌───────────┴───────────┐                               │
│    ▼                       ▼                               │
│ ┌──────┐              ┌───────┐                            │
│ │Neo4j │              │Qdrant │                            │
│ │Graph │              │Vector │                            │
│ └──────┘              └───────┘                            │
└─────────────────────────────────────────────────────────────┘
```

## Server Configuration

```json
{
  "mcpServers": {
    "ww-memory": {
      "command": "python",
      "args": ["-m", "ww.mcp.memory_gateway"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "${NEO4J_PASSWORD}",
        "QDRANT_URL": "http://localhost:6333",
        "BGE_MODEL_PATH": "/models/bge-m3",
        "WW_SESSION_ID": "${INSTANCE_ID}"
      }
    }
  }
}
```

## MCP Tools

### Episodic Memory Tools

#### `create_episode`

Store a new autobiographical event.

```json
{
  "name": "create_episode",
  "description": "Store an autobiographical event with temporal-spatial context",
  "inputSchema": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "Full interaction text or event description"
      },
      "context": {
        "type": "object",
        "properties": {
          "project": {"type": "string"},
          "file": {"type": "string"},
          "tool": {"type": "string"},
          "cwd": {"type": "string"}
        }
      },
      "outcome": {
        "type": "string",
        "enum": ["success", "failure", "partial", "neutral"],
        "default": "neutral"
      },
      "valence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "description": "Importance signal (0-1)"
      }
    },
    "required": ["content"]
  }
}
```

#### `recall_episodes`

Retrieve episodes with decay-weighted scoring.

```json
{
  "name": "recall_episodes",
  "description": "Retrieve autobiographical events by semantic similarity with recency decay",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query"
      },
      "limit": {
        "type": "integer",
        "default": 10
      },
      "session_filter": {
        "type": "string",
        "description": "Filter to specific session ID (optional)"
      },
      "time_range": {
        "type": "object",
        "properties": {
          "start": {"type": "string", "format": "date-time"},
          "end": {"type": "string", "format": "date-time"}
        }
      }
    },
    "required": ["query"]
  }
}
```

#### `query_at_time`

Point-in-time historical query.

```json
{
  "name": "query_at_time",
  "description": "What did we know at a specific point in time?",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "point_in_time": {
        "type": "string",
        "format": "date-time"
      },
      "limit": {"type": "integer", "default": 10}
    },
    "required": ["query", "point_in_time"]
  }
}
```

#### `mark_important`

Increase valence for existing episode.

```json
{
  "name": "mark_important",
  "description": "Mark an episode as important (increase valence)",
  "inputSchema": {
    "type": "object",
    "properties": {
      "episode_id": {"type": "string"},
      "new_valence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1
      }
    },
    "required": ["episode_id"]
  }
}
```

### Semantic Memory Tools

#### `create_entity`

Store new knowledge entity.

```json
{
  "name": "create_entity",
  "description": "Create a semantic knowledge entity",
  "inputSchema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "entity_type": {
        "type": "string",
        "enum": ["CONCEPT", "PERSON", "PROJECT", "TOOL", "TECHNIQUE", "FACT"]
      },
      "summary": {"type": "string"},
      "details": {"type": "string"},
      "source": {"type": "string"}
    },
    "required": ["name", "entity_type", "summary"]
  }
}
```

#### `create_relation`

Create Hebbian-weighted relationship.

```json
{
  "name": "create_relation",
  "description": "Create a relationship between two entities",
  "inputSchema": {
    "type": "object",
    "properties": {
      "source_id": {"type": "string"},
      "target_id": {"type": "string"},
      "relation_type": {
        "type": "string",
        "enum": ["USES", "PRODUCES", "REQUIRES", "CAUSES", "PART_OF", "SIMILAR_TO", "IMPLEMENTS"]
      },
      "initial_weight": {
        "type": "number",
        "default": 0.1
      }
    },
    "required": ["source_id", "target_id", "relation_type"]
  }
}
```

#### `semantic_recall`

ACT-R activation-based retrieval.

```json
{
  "name": "semantic_recall",
  "description": "Retrieve semantic entities with ACT-R activation scoring",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "context_entities": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Entity IDs to spread activation from"
      },
      "limit": {"type": "integer", "default": 10},
      "include_spreading": {
        "type": "boolean",
        "default": true
      }
    },
    "required": ["query"]
  }
}
```

#### `spread_activation`

Graph-based association search.

```json
{
  "name": "spread_activation",
  "description": "Spread activation through knowledge graph from seed entities",
  "inputSchema": {
    "type": "object",
    "properties": {
      "seed_entities": {
        "type": "array",
        "items": {"type": "string"}
      },
      "steps": {"type": "integer", "default": 3},
      "retention": {"type": "number", "default": 0.5},
      "decay": {"type": "number", "default": 0.1},
      "threshold": {"type": "number", "default": 0.01}
    },
    "required": ["seed_entities"]
  }
}
```

#### `supersede_fact`

Update entity with versioning.

```json
{
  "name": "supersede_fact",
  "description": "Update an entity with bi-temporal versioning",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_id": {"type": "string"},
      "new_summary": {"type": "string"},
      "new_details": {"type": "string"}
    },
    "required": ["entity_id", "new_summary"]
  }
}
```

### Procedural Memory Tools

#### `build_skill`

Create procedure from trajectory.

```json
{
  "name": "build_skill",
  "description": "Build a reusable procedure from a successful trajectory",
  "inputSchema": {
    "type": "object",
    "properties": {
      "trajectory": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "action": {"type": "string"},
            "tool": {"type": "string"},
            "parameters": {"type": "object"},
            "result": {"type": "string"}
          }
        }
      },
      "outcome_score": {
        "type": "number",
        "minimum": 0,
        "maximum": 1
      },
      "domain": {
        "type": "string",
        "enum": ["coding", "research", "trading", "devops", "writing"]
      },
      "trigger_pattern": {"type": "string"}
    },
    "required": ["trajectory", "outcome_score", "domain"]
  }
}
```

#### `how_to`

Retrieve matching procedure.

```json
{
  "name": "how_to",
  "description": "Retrieve procedures matching a task description",
  "inputSchema": {
    "type": "object",
    "properties": {
      "task": {"type": "string"},
      "domain": {"type": "string"},
      "limit": {"type": "integer", "default": 5}
    },
    "required": ["task"]
  }
}
```

#### `execute_skill`

Run procedure with feedback.

```json
{
  "name": "execute_skill",
  "description": "Record execution of a procedure and update success rate",
  "inputSchema": {
    "type": "object",
    "properties": {
      "procedure_id": {"type": "string"},
      "success": {"type": "boolean"},
      "error": {"type": "string"},
      "failed_step": {"type": "integer"},
      "context": {"type": "string"}
    },
    "required": ["procedure_id", "success"]
  }
}
```

#### `deprecate_skill`

Mark procedure as outdated.

```json
{
  "name": "deprecate_skill",
  "description": "Mark a procedure as deprecated",
  "inputSchema": {
    "type": "object",
    "properties": {
      "procedure_id": {"type": "string"},
      "reason": {"type": "string"}
    },
    "required": ["procedure_id"]
  }
}
```

### Consolidation Tools

#### `consolidate_now`

Trigger immediate consolidation cycle.

```json
{
  "name": "consolidate_now",
  "description": "Trigger immediate memory consolidation",
  "inputSchema": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string",
        "enum": ["light", "deep", "skill", "all"],
        "default": "light"
      }
    }
  }
}
```

#### `get_provenance`

Trace knowledge back to source episodes.

```json
{
  "name": "get_provenance",
  "description": "Get the source episodes for a semantic entity",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_id": {"type": "string"}
    },
    "required": ["entity_id"]
  }
}
```

### Utility Tools

#### `get_session_id`

Get current session identifier.

```json
{
  "name": "get_session_id",
  "description": "Get the current Claude Code instance session ID",
  "inputSchema": {"type": "object", "properties": {}}
}
```

#### `memory_stats`

Get memory system statistics.

```json
{
  "name": "memory_stats",
  "description": "Get statistics about the memory system",
  "inputSchema": {
    "type": "object",
    "properties": {
      "include_detailed": {"type": "boolean", "default": false}
    }
  }
}
```

## MCP Resources

### `memory://episodes/{session_id}`

Access episodes for a specific session.

### `memory://entities/{entity_type}`

Browse entities by type.

### `memory://procedures/{domain}`

List procedures by domain.

### `memory://stats`

Memory system statistics and health.

## Response Formats

### Episode Response

```json
{
  "id": "uuid",
  "content": "Implemented Hebbian strengthening...",
  "timestamp": "2025-11-27T10:00:00Z",
  "context": {
    "project": "world-weaver",
    "file": "semantic.py"
  },
  "outcome": "success",
  "valence": 0.8,
  "score": 0.89,
  "retrievability": 0.95
}
```

### Entity Response

```json
{
  "id": "uuid",
  "name": "FSRS",
  "entity_type": "TECHNIQUE",
  "summary": "Free Spaced Repetition Scheduler",
  "activation": 0.75,
  "relationships": [
    {"target": "SM-2", "type": "IMPROVES_ON", "weight": 0.8}
  ]
}
```

### Procedure Response

```json
{
  "id": "uuid",
  "name": "Edit Python Function",
  "domain": "coding",
  "trigger_pattern": "modify existing function",
  "steps": [...],
  "script": "PROCEDURE: Edit Python Function...",
  "success_rate": 0.85,
  "execution_count": 12
}
```

## Error Handling

```json
{
  "error": {
    "code": "MEMORY_NOT_FOUND",
    "message": "Episode with ID xyz not found",
    "details": {...}
  }
}
```

Error Codes:
- `MEMORY_NOT_FOUND`: Requested memory item doesn't exist
- `EMBEDDING_FAILED`: Failed to generate embedding
- `CONSOLIDATION_FAILED`: Consolidation operation failed
- `INVALID_SESSION`: Session ID not recognized
- `STORAGE_ERROR`: Database operation failed

## Usage Examples

### Store Interaction as Episode

```python
# After successful code edit
result = await mcp.call_tool("create_episode", {
    "content": "Added Hebbian strengthening to semantic memory module",
    "context": {
        "project": "world-weaver",
        "file": "src/memory/semantic.py",
        "tool": "Edit"
    },
    "outcome": "success",
    "valence": 0.7
})
```

### Retrieve Relevant Knowledge

```python
# During task planning
result = await mcp.call_tool("semantic_recall", {
    "query": "memory decay algorithms",
    "context_entities": ["world-weaver", "cognitive-science"],
    "limit": 5
})

# Returns entities ranked by ACT-R activation
for entity in result["entities"]:
    print(f"{entity['name']}: {entity['summary']}")
```

### Find Procedure for Task

```python
# When starting similar task
result = await mcp.call_tool("how_to", {
    "task": "add a new parameter to an existing function",
    "domain": "coding"
})

if result["procedures"]:
    best = result["procedures"][0]
    print(f"Found procedure: {best['name']}")
    print(f"Success rate: {best['success_rate']}")
    for step in best["steps"]:
        print(f"  {step['order']}. {step['action']}")
```

## Implementation Notes

### Session Namespacing

Each Claude Code instance gets a unique session ID from the `WW_SESSION_ID` environment variable. Episodes are tagged with this ID, allowing:
- Instance-isolated episodic memory
- Cross-instance queries when explicitly requested
- Consolidation merges insights to shared semantic memory

### Embedding Pipeline

1. Text received via MCP tool call
2. BGE-M3 model generates 1024-dim embedding on local GPU
3. Embedding stored in Qdrant with metadata
4. Graph node created in Neo4j with reference

### Hebbian Updates

On every co-retrieval (multiple entities returned together):
1. Identify all pairs of retrieved entities
2. For each pair with existing relationship: `w' = w + 0.1 * (1 - w)`
3. Update `coAccessCount` and `lastCoAccess`

### FSRS Stability Updates

On every episode access:
1. Calculate current retrievability: `R = (1 + 0.9 * t/S)^(-0.5)`
2. If successful retrieval: `S' = S * (1 + 0.1 * (1 - R))`
3. Update `lastAccessed` and `accessCount`

## Deployment

### Prerequisites

- Neo4j 5.x with APOC plugin
- Qdrant 1.x
- Python 3.11+
- CUDA 11.8+ (for BGE-M3 on GPU)

### Docker Compose

```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    environment:
      - NEO4J_AUTH=neo4j/password

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  ww-memory:
    build: ./mcp
    depends_on:
      - neo4j
      - qdrant
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - QDRANT_URL=http://qdrant:6333
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

volumes:
  neo4j_data:
  qdrant_data:
```

### Claude Code Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "ww-memory": {
      "command": "docker",
      "args": ["exec", "-i", "ww-memory", "python", "-m", "ww.mcp.serve"],
      "env": {
        "WW_SESSION_ID": "${CLAUDE_INSTANCE_ID}"
      }
    }
  }
}
```
