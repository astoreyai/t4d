# World Weaver API Reference

**Version**: 1.0.0 | **Last Updated**: 2025-11-27

## Overview

World Weaver exposes 17 MCP tools for tripartite memory operations across episodic (autobiographical events), semantic (knowledge graph), and procedural (learned skills) memory systems.

**Base URL**: MCP stdio interface (no HTTP endpoint)

**Protocol**: Model Context Protocol (MCP)

**Rate Limiting**: 100 requests per minute per session

## Authentication

All requests require session context. The session ID is automatically derived from environment settings or can be specified per request.

**Session Configuration**:
```bash
export WW_SESSION_ID=your-session-id
```

**Rate Limits**:
- 100 requests per 60-second window
- Per-session tracking
- Returns `rate_limited` error with `retry_after` seconds

## Error Handling

All tools return consistent error responses:

```json
{
  "error": "error_code",
  "message": "Human-readable error description",
  "field": "parameter_name"  // Optional: which field caused the error
}
```

**Error Codes**:

| Code | HTTP Equiv | Description |
|------|------------|-------------|
| `validation_error` | 400 | Invalid input parameter |
| `not_found` | 404 | Resource does not exist |
| `rate_limited` | 429 | Rate limit exceeded |
| `timeout` | 504 | Operation timed out |
| `internal_error` | 500 | Server-side error |
| `unauthorized` | 401 | Authentication required |
| `forbidden` | 403 | Insufficient permissions |

---

## Episodic Memory

Episodic memory stores autobiographical events with temporal-spatial context, emotional valence, and outcome tracking. Uses ACT-R activation and FSRS retrievability.

### create_episode

Store an autobiographical event with temporal-spatial context.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `content` | string | Yes | - | Full interaction text or event description (max 100,000 chars) |
| `context` | object | No | `null` | Spatial context (project, file, tool, cwd) |
| `outcome` | string | No | `"neutral"` | Result: `success`, `failure`, `partial`, `neutral` |
| `valence` | number | No | `0.5` | Emotional importance (0.0-1.0) |

**Response:**
```json
{
  "id": "ep-abc123",
  "content": "User asked about Python async...",
  "timestamp": "2025-11-27T10:30:00Z",
  "session_id": "default",
  "outcome": "success",
  "valence": 0.8,
  "retrievability": 0.95
}
```

**Example:**
```json
{
  "content": "User requested implementation of binary search with error handling",
  "context": {
    "project": "algorithms",
    "file": "search.py",
    "tool": "write_file",
    "cwd": "/home/user/projects/algorithms"
  },
  "outcome": "success",
  "valence": 0.9
}
```

---

### recall_episodes

Retrieve autobiographical events by semantic similarity with recency decay.

**Scoring Formula**: `0.4*semantic + 0.25*recency + 0.2*outcome + 0.15*importance`

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language search query (max 10,000 chars) |
| `limit` | integer | No | `10` | Maximum number of results (min 1) |
| `offset` | integer | No | `0` | Pagination offset (min 0) |
| `session_filter` | string | No | `null` | Filter to specific session ID |
| `time_start` | string | No | `null` | ISO datetime for range start |
| `time_end` | string | No | `null` | ISO datetime for range end |

**Response:**
```json
{
  "query": "Python async implementations",
  "count": 10,
  "total": 47,
  "offset": 0,
  "limit": 10,
  "has_more": true,
  "episodes": [
    {
      "id": "ep-abc123",
      "content": "Implemented async event loop...",
      "timestamp": "2025-11-27T09:15:00Z",
      "outcome": "success",
      "score": 0.8523,
      "components": {
        "semantic": 0.92,
        "recency": 0.85,
        "outcome": 1.0,
        "importance": 0.8
      }
    }
  ]
}
```

**Example:**
```json
{
  "query": "debugging database connection issues",
  "limit": 5,
  "time_start": "2025-11-20T00:00:00Z",
  "time_end": "2025-11-27T23:59:59Z"
}
```

---

### query_at_time

What did we know at a specific point in time? Uses T_sys (ingestion time) for bi-temporal queries.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `point_in_time` | string | Yes | - | ISO datetime for historical query |
| `limit` | integer | No | `10` | Maximum results |

**Response:**
```json
{
  "query": "Python best practices",
  "point_in_time": "2025-11-15T12:00:00Z",
  "count": 8,
  "episodes": [
    {
      "id": "ep-xyz789",
      "content": "Discussed type hints and docstrings...",
      "timestamp": "2025-11-14T10:30:00Z",
      "score": 0.78
    }
  ]
}
```

---

### mark_important

Mark an episode as important (increase emotional valence for better retrievability).

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `episode_id` | string | Yes | - | Episode UUID |
| `new_valence` | number | No | `+0.2` | Specific valence (0.0-1.0), or increment by 0.2 if null |

**Response:**
```json
{
  "id": "ep-abc123",
  "valence": 0.9,
  "message": "Episode marked with valence 0.9"
}
```

---

## Semantic Memory

Semantic memory maintains a Hebbian-weighted knowledge graph with entities, relationships, ACT-R activation, and bi-temporal versioning.

### create_entity

Create a semantic knowledge entity in the graph.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `name` | string | Yes | - | Canonical entity name (max 500 chars) |
| `entity_type` | string | Yes | - | `CONCEPT`, `PERSON`, `PROJECT`, `TOOL`, `TECHNIQUE`, `FACT` |
| `summary` | string | Yes | - | Short description (max 2,000 chars) |
| `details` | string | No | `null` | Expanded context |
| `source` | string | No | `null` | Episode ID or `"user_provided"` |

**Response:**
```json
{
  "id": "ent-def456",
  "name": "Binary Search Algorithm",
  "entity_type": "TECHNIQUE",
  "summary": "Efficient O(log n) search on sorted arrays",
  "created_at": "2025-11-27T10:35:00Z"
}
```

**Example:**
```json
{
  "name": "FastAPI",
  "entity_type": "TOOL",
  "summary": "Modern Python web framework for building APIs",
  "details": "Supports async/await, automatic OpenAPI docs, type hints with Pydantic",
  "source": "user_provided"
}
```

---

### create_relation

Create a Hebbian-weighted relationship between entities. Weights strengthen with co-activation.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `source_id` | string | Yes | - | Source entity UUID |
| `target_id` | string | Yes | - | Target entity UUID |
| `relation_type` | string | Yes | - | `USES`, `PRODUCES`, `REQUIRES`, `CAUSES`, `PART_OF`, `SIMILAR_TO`, `IMPLEMENTS` |
| `initial_weight` | number | No | `0.1` | Starting weight (0.0-1.0) |

**Response:**
```json
{
  "source_id": "ent-abc123",
  "target_id": "ent-def456",
  "relation_type": "USES",
  "weight": 0.1
}
```

**Example:**
```json
{
  "source_id": "ent-fastapi-123",
  "target_id": "ent-pydantic-456",
  "relation_type": "REQUIRES",
  "initial_weight": 0.3
}
```

---

### semantic_recall

Retrieve semantic entities with ACT-R activation scoring and optional spreading activation.

**Scoring**: Vector similarity + ACT-R activation + FSRS retrievability + spreading activation

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language search query |
| `context_entities` | array[string] | No | `null` | Entity IDs to spread activation from |
| `limit` | integer | No | `10` | Maximum results |
| `offset` | integer | No | `0` | Pagination offset |
| `include_spreading` | boolean | No | `true` | Include spreading activation from context |

**Response:**
```json
{
  "query": "web frameworks",
  "count": 5,
  "total": 12,
  "offset": 0,
  "limit": 5,
  "has_more": true,
  "entities": [
    {
      "id": "ent-fastapi-123",
      "name": "FastAPI",
      "entity_type": "TOOL",
      "summary": "Modern Python web framework",
      "score": 0.8745,
      "components": {
        "similarity": 0.92,
        "activation": 0.85,
        "retrievability": 0.88,
        "spreading": 0.15
      }
    }
  ]
}
```

---

### spread_activation

Spread activation through knowledge graph from seed entities using ACT-R-inspired propagation.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `seed_entities` | array[string] | Yes | - | Entity IDs to start propagation |
| `steps` | integer | No | `3` | Number of propagation steps |
| `retention` | number | No | `0.5` | Activation retained at each node (0.0-1.0) |
| `decay` | number | No | `0.1` | Activation decay per step (0.0-1.0) |
| `threshold` | number | No | `0.01` | Minimum activation to continue (0.0-1.0) |

**Response:**
```json
{
  "seed_count": 2,
  "activated_count": 15,
  "activations": [
    {
      "entity_id": "ent-abc123",
      "activation": 0.85
    },
    {
      "entity_id": "ent-def456",
      "activation": 0.42
    }
  ]
}
```

---

### supersede_fact

Update an entity with bi-temporal versioning. Old version's `validTo` is set, new version created.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `entity_id` | string | Yes | - | Entity UUID to supersede |
| `new_summary` | string | Yes | - | Updated summary (max 2,000 chars) |
| `new_details` | string | No | `null` | Updated details |

**Response:**
```json
{
  "old_id": "ent-abc123",
  "new_id": "ent-abc124",
  "name": "FastAPI",
  "summary": "Modern Python web framework with enhanced performance",
  "valid_from": "2025-11-27T11:00:00Z"
}
```

---

## Procedural Memory

Procedural memory learns reusable skills from successful trajectories using Memp (Memory for Procedures).

### create_skill

Create a reusable procedure from a successful trajectory. Only learns from high-success outcomes (score >= 0.7).

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `trajectory` | array[object] | Yes | - | List of action dicts with `tool`, `parameters`, `result` |
| `outcome_score` | number | Yes | - | Success score (0.0-1.0), must be >= 0.7 to learn |
| `domain` | string | Yes | - | `coding`, `research`, `trading`, `devops`, `writing` |
| `trigger_pattern` | string | No | `null` | When to invoke this procedure |
| `name` | string | No | `null` | Optional procedure name |

**Response (success):**
```json
{
  "created": true,
  "id": "proc-ghi789",
  "name": "Deploy FastAPI to production",
  "domain": "devops",
  "trigger_pattern": "deploy web application",
  "step_count": 7,
  "script": "# Step 1: Run tests\npytest tests/\n..."
}
```

**Response (score too low):**
```json
{
  "created": false,
  "reason": "Outcome score 0.6 < 0.7 threshold"
}
```

**Example:**
```json
{
  "trajectory": [
    {
      "tool": "run_command",
      "parameters": {"command": "pytest tests/"},
      "result": "All tests passed"
    },
    {
      "tool": "run_command",
      "parameters": {"command": "docker build -t myapp ."},
      "result": "Image built successfully"
    },
    {
      "tool": "run_command",
      "parameters": {"command": "docker push myapp"},
      "result": "Pushed to registry"
    }
  ],
  "outcome_score": 0.95,
  "domain": "devops",
  "trigger_pattern": "containerize and deploy application"
}
```

---

### recall_skill

Recall procedures matching a task description.

**Scoring Formula**: `0.6*similarity + 0.3*success_rate + 0.1*experience`

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `task` | string | Yes | - | Task description to match |
| `domain` | string | No | `null` | Optional domain filter |
| `limit` | integer | No | `5` | Maximum results |
| `offset` | integer | No | `0` | Pagination offset |

**Response:**
```json
{
  "task": "deploy web application",
  "count": 3,
  "total": 8,
  "offset": 0,
  "limit": 3,
  "has_more": true,
  "procedures": [
    {
      "id": "proc-ghi789",
      "name": "Deploy FastAPI to production",
      "domain": "devops",
      "trigger_pattern": "deploy web application",
      "success_rate": 0.92,
      "execution_count": 12,
      "score": 0.8734,
      "steps": [
        {
          "order": 1,
          "action": "Run test suite",
          "tool": "run_command"
        },
        {
          "order": 2,
          "action": "Build Docker image",
          "tool": "run_command"
        }
      ]
    }
  ]
}
```

---

### execute_skill

Record execution of a procedure and update statistics. May auto-deprecate if consistently failing.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `procedure_id` | string | Yes | - | Procedure UUID |
| `success` | boolean | Yes | - | Whether execution succeeded |
| `error` | string | No | `null` | Error message if failed |
| `failed_step` | integer | No | `null` | Step number that failed |
| `context` | string | No | `null` | Execution context |

**Response:**
```json
{
  "id": "proc-ghi789",
  "name": "Deploy FastAPI to production",
  "success_rate": 0.923,
  "execution_count": 13,
  "deprecated": false
}
```

---

### deprecate_skill

Mark a procedure as deprecated to prevent future retrieval.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `procedure_id` | string | Yes | - | Procedure UUID |
| `reason` | string | No | `null` | Deprecation reason |

**Response:**
```json
{
  "id": "proc-ghi789",
  "deprecated": true,
  "reason": "Replaced by containerized deployment workflow"
}
```

---

## Consolidation & Metadata

### consolidate_now

Trigger immediate memory consolidation cycle.

**Consolidation Types**:
- `light`: Quick deduplication and cleanup
- `deep`: Full semantic extraction from episodes
- `skill`: Procedure optimization and merging
- `all`: Complete consolidation cycle

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `consolidation_type` | string | No | `"light"` | `light`, `deep`, `skill`, or `all` |
| `session_filter` | string | No | `null` | Limit to specific session |

**Response:**
```json
{
  "consolidation_type": "deep",
  "session_filter": null,
  "started_at": "2025-11-27T12:00:00Z",
  "completed_at": "2025-11-27T12:05:32Z",
  "duration_seconds": 332,
  "metrics": {
    "episodes_processed": 145,
    "entities_extracted": 23,
    "relationships_created": 47,
    "duplicates_merged": 8
  }
}
```

---

### get_provenance

Get the source episodes for a semantic entity. Traces knowledge back to autobiographical origins.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `entity_id` | string | Yes | - | Entity UUID |

**Response:**
```json
{
  "entity_id": "ent-abc123",
  "found": true,
  "name": "FastAPI",
  "source": "ep-xyz789",
  "created_at": "2025-11-27T10:35:00Z",
  "valid_from": "2025-11-27T10:35:00Z"
}
```

---

## Utility Tools

### get_session_id

Get the current Claude Code instance session ID.

**Parameters:** None

**Response:**
```json
{
  "session_id": "default"
}
```

---

### memory_stats

Get statistics about the memory system health and storage.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `include_detailed` | boolean | No | `false` | Include per-collection statistics |

**Response:**
```json
{
  "session_id": "default",
  "initialized": true,
  "storage": {
    "neo4j": "connected",
    "qdrant": "connected"
  },
  "collections": {
    "episodes": 145,
    "entities": 67,
    "procedures": 12
  }
}
```

---

## Deprecated Tools (Backward Compatibility)

The following tools are deprecated and will be removed in a future version. Use the recommended alternatives.

| Deprecated | Use Instead | Removal Version |
|------------|-------------|-----------------|
| `build_skill` | `create_skill` | 2.0.0 |
| `how_to` | `recall_skill` | 2.0.0 |

---

## MCP Resources

World Weaver provides read-only resources accessible via MCP resource URIs:

- `memory://stats` - System statistics and health
- `memory://episodes/{session_id}` - Episodes for a session
- `memory://entities/{entity_type}` - Entities by type
- `memory://procedures/{domain}` - Procedures by domain

**Example**:
```json
{
  "uri": "memory://episodes/default",
  "mimeType": "application/json"
}
```

---

## Rate Limiting Details

**Window**: 60 seconds (sliding)
**Limit**: 100 requests per session
**Reset**: Automatic (oldest request expires from window)

**Rate Limit Response**:
```json
{
  "error": "rate_limited",
  "message": "Rate limit exceeded for session 'default'",
  "retry_after": 15.3
}
```

**Headers** (if using HTTP transport):
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1701092415
```

---

## Best Practices

1. **Episodic Memory**: Use high valence (0.8-1.0) for critical events you want to retrieve later
2. **Semantic Memory**: Always provide source provenance when extracting from episodes
3. **Procedural Memory**: Only submit trajectories with outcome_score >= 0.7
4. **Pagination**: Use `offset` and `limit` for large result sets instead of fetching all at once
5. **Consolidation**: Run `consolidate_now` with type `"light"` daily, `"deep"` weekly
6. **Context Entities**: Use `context_entities` in `semantic_recall` for better relevance

---

## OpenAPI Schema

A full OpenAPI 3.0 specification is available at:
- **JSON**: `/mnt/projects/t4d/t4dm/openapi.json`
- **YAML**: `/mnt/projects/t4d/t4dm/openapi.yaml`

Generate latest schema:
```bash
cd /mnt/projects/ww
python -m ww.mcp.schema
```
