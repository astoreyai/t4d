# T4DM API Walkthrough

**Version**: 0.1.0 | **Last Updated**: 2025-12-09

A comprehensive guide to the T4DM REST API and MCP interface, covering all memory operations, persistence management, and real-time event streaming.

---

## Table of Contents

1. [REST API Overview](#rest-api-overview)
2. [Episodic Memory Endpoints](#episodic-memory-endpoints)
3. [Semantic Memory Endpoints](#semantic-memory-endpoints)
4. [Procedural Memory Endpoints](#procedural-memory-endpoints)
5. [Persistence Endpoints](#persistence-endpoints)
6. [WebSocket Channels](#websocket-channels)
7. [MCP Tools Reference](#mcp-tools-reference)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)

---

## REST API Overview

### Base URL Structure

```
http://localhost:8765/api/v1
```

**Interactive Documentation**:
- Swagger UI: `http://localhost:8765/docs`
- ReDoc: `http://localhost:8765/redoc`
- OpenAPI Schema: `http://localhost:8765/openapi.json`

### Authentication

T4DM uses session-based isolation via the `X-Session-ID` header. Each session maintains its own memory namespace.

```bash
curl -H "X-Session-ID: my-session" http://localhost:8765/api/v1/health
```

**Default Session**: If no session ID is provided, the system uses `"default"`.

**Session Validation**:
- Must be 1-128 characters
- Allowed characters: `a-z`, `A-Z`, `0-9`, `-`, `_`
- Invalid sessions return `400 Bad Request`

### Rate Limiting

**Limits**: 100 requests per minute per session (configurable)

**Rate Limit Headers**:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1638360000
```

**Rate Limit Exceeded Response**:
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 30

{
  "detail": "Rate limit exceeded. Try again in 30 seconds."
}
```

### Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Status Codes**:
- `200 OK` - Success
- `201 Created` - Resource created
- `204 No Content` - Success with no response body
- `400 Bad Request` - Invalid parameters or validation error
- `404 Not Found` - Resource does not exist
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server-side error
- `503 Service Unavailable` - Storage connection issues

**Error Sanitization**: Errors are sanitized to prevent information leakage:
- Database credentials and URIs are masked
- Internal file paths are hidden
- Stack traces are logged server-side but not returned to clients

---

## Episodic Memory Endpoints

Episodic memory stores time-sequenced experiences with FSRS (Free Spaced Repetition Scheduler) decay tracking.

### Create Episode

Store a new episodic memory with temporal context.

**Endpoint**: `POST /api/v1/episodes`

**Request Body**:
```json
{
  "content": "Implemented FSRS decay algorithm for memory retrieval",
  "project": "t4dm",
  "file": "src/memory/episodic.py",
  "tool": "python",
  "outcome": "success",
  "emotional_valence": 0.8,
  "timestamp": "2025-12-09T10:30:00Z"
}
```

**Parameters**:
- `content` (required): Episode content (max 50KB)
- `project` (optional): Project context (max 500 chars)
- `file` (optional): File path context (max 1000 chars)
- `tool` (optional): Tool used (max 200 chars)
- `outcome` (optional): One of `"success"`, `"failure"`, `"neutral"` (default: `"neutral"`)
- `emotional_valence` (optional): Importance score 0.0-1.0 (default: 0.5)
- `timestamp` (optional): Event time (defaults to now)

**Response**: `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "default",
  "content": "Implemented FSRS decay algorithm for memory retrieval",
  "timestamp": "2025-12-09T10:30:00Z",
  "outcome": "success",
  "emotional_valence": 0.8,
  "context": {
    "project": "t4dm",
    "file": "src/memory/episodic.py",
    "tool": "python"
  },
  "access_count": 1,
  "stability": 1.0,
  "retrievability": 1.0
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/episodes \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "content": "Learned about memory consolidation techniques",
    "project": "research",
    "outcome": "success",
    "emotional_valence": 0.9
  }'
```

---

### Get Episode

Retrieve a specific episode by ID. Updates access count and retrievability.

**Endpoint**: `GET /api/v1/episodes/{episode_id}`

**Response**: `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "default",
  "content": "Implemented FSRS decay algorithm...",
  "timestamp": "2025-12-09T10:30:00Z",
  "outcome": "success",
  "emotional_valence": 0.8,
  "context": {
    "project": "t4dm",
    "file": "src/memory/episodic.py"
  },
  "access_count": 3,
  "stability": 1.5,
  "retrievability": 0.95
}
```

**Example**:
```bash
curl http://localhost:8765/api/v1/episodes/550e8400-e29b-41d4-a716-446655440000 \
  -H "X-Session-ID: my-session"
```

---

### List Episodes

List episodes with pagination and filtering.

**Endpoint**: `GET /api/v1/episodes`

**Query Parameters**:
- `page` (default: 1): Page number (≥1)
- `page_size` (default: 20): Results per page (1-100)
- `project` (optional): Filter by project
- `outcome` (optional): Filter by outcome (`success`, `failure`, `neutral`)

**Response**: `200 OK`
```json
{
  "episodes": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "...",
      "timestamp": "2025-12-09T10:30:00Z",
      "outcome": "success",
      "emotional_valence": 0.8,
      "context": {"project": "t4dm"},
      "access_count": 1,
      "stability": 1.0,
      "retrievability": 1.0
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 20
}
```

**Example**:
```bash
curl "http://localhost:8765/api/v1/episodes?page=1&page_size=10&project=research&outcome=success" \
  -H "X-Session-ID: my-session"
```

---

### Recall Episodes (Semantic Search)

Search episodes using BGE-M3 embeddings for similarity matching.

**Endpoint**: `POST /api/v1/episodes/recall`

**Request Body**:
```json
{
  "query": "memory decay algorithms",
  "limit": 10,
  "min_similarity": 0.5,
  "project": "t4dm",
  "outcome": "success"
}
```

**Parameters**:
- `query` (required): Search query (max 10KB)
- `limit` (default: 10): Max results (1-100)
- `min_similarity` (default: 0.5): Minimum similarity threshold (0.0-1.0)
- `project` (optional): Filter by project
- `outcome` (optional): Filter by outcome

**Response**: `200 OK`
```json
{
  "query": "memory decay algorithms",
  "episodes": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "Implemented FSRS decay algorithm...",
      "timestamp": "2025-12-09T10:30:00Z",
      "outcome": "success",
      "emotional_valence": 0.8,
      "context": {"project": "t4dm"},
      "access_count": 2,
      "stability": 1.2,
      "retrievability": 0.98
    }
  ],
  "scores": [0.92, 0.87, 0.75]
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/episodes/recall \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "query": "FSRS implementation details",
    "limit": 5,
    "min_similarity": 0.7
  }'
```

---

### Mark Episode Important

Increase emotional valence and stability to reduce decay.

**Endpoint**: `POST /api/v1/episodes/{episode_id}/mark-important`

**Query Parameters**:
- `importance` (default: 1.0): Importance score (0.0-1.0)

**Response**: `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "default",
  "content": "...",
  "emotional_valence": 1.0,
  "stability": 10.0,
  "retrievability": 1.0
}
```

**Example**:
```bash
curl -X POST "http://localhost:8765/api/v1/episodes/550e8400-e29b-41d4-a716-446655440000/mark-important?importance=1.0" \
  -H "X-Session-ID: my-session"
```

---

### Delete Episode

Remove an episode from memory.

**Endpoint**: `DELETE /api/v1/episodes/{episode_id}`

**Response**: `204 No Content`

**Example**:
```bash
curl -X DELETE http://localhost:8765/api/v1/episodes/550e8400-e29b-41d4-a716-446655440000 \
  -H "X-Session-ID: my-session"
```

---

## Semantic Memory Endpoints

Semantic memory implements a knowledge graph with ACT-R activation spreading and Hebbian learning.

### Create Entity

Create a new knowledge entity (concept, person, place, etc.).

**Endpoint**: `POST /api/v1/entities`

**Request Body**:
```json
{
  "name": "FSRS Algorithm",
  "entity_type": "CONCEPT",
  "summary": "Free Spaced Repetition Scheduler for memory decay",
  "details": "Implements power-law forgetting curve with stability tracking and retrievability calculations.",
  "source": "episode-550e8400"
}
```

**Parameters**:
- `name` (required): Entity name
- `entity_type` (required): One of `"CONCEPT"`, `"PERSON"`, `"PLACE"`, `"EVENT"`, `"OBJECT"`, `"SKILL"`
- `summary` (required): Short description
- `details` (optional): Extended description
- `source` (optional): Source episode ID or `"user_provided"`

**Response**: `201 Created`
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440000",
  "name": "FSRS Algorithm",
  "entity_type": "CONCEPT",
  "summary": "Free Spaced Repetition Scheduler for memory decay",
  "details": "Implements power-law forgetting curve...",
  "source": "episode-550e8400",
  "stability": 1.0,
  "access_count": 0,
  "created_at": "2025-12-09T10:30:00Z",
  "valid_from": "2025-12-09T10:30:00Z",
  "valid_to": null
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/entities \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "name": "ACT-R",
    "entity_type": "CONCEPT",
    "summary": "Adaptive Control of Thought-Rational cognitive architecture"
  }'
```

---

### Get Entity

Retrieve a specific entity by ID.

**Endpoint**: `GET /api/v1/entities/{entity_id}`

**Response**: `200 OK`
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440000",
  "name": "FSRS Algorithm",
  "entity_type": "CONCEPT",
  "summary": "Free Spaced Repetition Scheduler for memory decay",
  "details": "...",
  "source": "user_provided",
  "stability": 1.5,
  "access_count": 12,
  "created_at": "2025-12-09T10:30:00Z",
  "valid_from": "2025-12-09T10:30:00Z",
  "valid_to": null
}
```

**Example**:
```bash
curl http://localhost:8765/api/v1/entities/660e8400-e29b-41d4-a716-446655440000 \
  -H "X-Session-ID: my-session"
```

---

### List Entities

List entities with optional type filtering.

**Endpoint**: `GET /api/v1/entities`

**Query Parameters**:
- `entity_type` (optional): Filter by type
- `limit` (default: 50): Max results (1-500)

**Response**: `200 OK`
```json
{
  "entities": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440000",
      "name": "FSRS Algorithm",
      "entity_type": "CONCEPT",
      "summary": "...",
      "stability": 1.5,
      "access_count": 12
    }
  ],
  "total": 45
}
```

**Example**:
```bash
curl "http://localhost:8765/api/v1/entities?entity_type=CONCEPT&limit=20" \
  -H "X-Session-ID: my-session"
```

---

### Create Relationship

Create a relationship between two entities. Supports Hebbian weight strengthening on co-access.

**Endpoint**: `POST /api/v1/entities/relations`

**Request Body**:
```json
{
  "source_id": "660e8400-e29b-41d4-a716-446655440000",
  "target_id": "770e8400-e29b-41d4-a716-446655440000",
  "relation_type": "IMPLEMENTS",
  "weight": 0.5
}
```

**Parameters**:
- `source_id` (required): Source entity UUID
- `target_id` (required): Target entity UUID
- `relation_type` (required): One of `"IMPLEMENTS"`, `"USES"`, `"RELATED_TO"`, `"PART_OF"`, `"INSTANCE_OF"`, `"DEPENDS_ON"`, `"CAUSED_BY"`
- `weight` (default: 0.1): Initial weight (0.0-1.0)

**Response**: `201 Created`
```json
{
  "source_id": "660e8400-e29b-41d4-a716-446655440000",
  "target_id": "770e8400-e29b-41d4-a716-446655440000",
  "relation_type": "IMPLEMENTS",
  "weight": 0.5,
  "co_access_count": 0
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/entities/relations \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "source_id": "660e8400-e29b-41d4-a716-446655440000",
    "target_id": "770e8400-e29b-41d4-a716-446655440000",
    "relation_type": "RELATED_TO",
    "weight": 0.3
  }'
```

---

### Recall Entities (Semantic Search)

Search entities using BGE-M3 embeddings.

**Endpoint**: `POST /api/v1/entities/recall`

**Request Body**:
```json
{
  "query": "memory algorithms",
  "limit": 10,
  "entity_types": ["CONCEPT", "TECHNIQUE"]
}
```

**Parameters**:
- `query` (required): Search query
- `limit` (default: 10): Max results (1-100)
- `entity_types` (optional): Filter by entity types

**Response**: `200 OK`
```json
{
  "entities": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440000",
      "name": "FSRS Algorithm",
      "entity_type": "CONCEPT",
      "summary": "...",
      "stability": 1.5,
      "access_count": 12
    }
  ],
  "total": 8
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/entities/recall \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "query": "cognitive architecture",
    "limit": 5,
    "entity_types": ["CONCEPT"]
  }'
```

---

### Spread Activation

Perform ACT-R spreading activation from a starting entity, traversing the knowledge graph.

**Endpoint**: `POST /api/v1/entities/spread-activation`

**Request Body**:
```json
{
  "entity_id": "660e8400-e29b-41d4-a716-446655440000",
  "depth": 2,
  "threshold": 0.1
}
```

**Parameters**:
- `entity_id` (required): Starting entity UUID
- `depth` (default: 2): Traversal depth (1-5)
- `threshold` (default: 0.1): Minimum activation level (0.0-1.0)

**Response**: `200 OK`
```json
{
  "entities": [
    {
      "id": "770e8400-e29b-41d4-a716-446655440000",
      "name": "Memory Decay",
      "entity_type": "CONCEPT",
      "summary": "...",
      "stability": 2.0
    }
  ],
  "activations": [1.0, 0.8, 0.65, 0.42],
  "paths": [
    ["FSRS"],
    ["FSRS", "IMPLEMENTS", "Memory"],
    ["FSRS", "RELATED_TO", "ACT-R", "USES", "Activation"]
  ]
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/entities/spread-activation \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "entity_id": "660e8400-e29b-41d4-a716-446655440000",
    "depth": 3,
    "threshold": 0.2
  }'
```

---

### Supersede Entity

Create a new version of an entity and mark the old version as invalid (bi-temporal versioning).

**Endpoint**: `POST /api/v1/entities/{entity_id}/supersede`

**Request Body**:
```json
{
  "name": "FSRS Algorithm v2",
  "entity_type": "CONCEPT",
  "summary": "Updated FSRS with improved stability calculations",
  "details": "..."
}
```

**Response**: `200 OK`
```json
{
  "id": "880e8400-e29b-41d4-a716-446655440000",
  "name": "FSRS Algorithm v2",
  "entity_type": "CONCEPT",
  "summary": "Updated FSRS with improved stability calculations",
  "stability": 1.0,
  "access_count": 0,
  "created_at": "2025-12-09T11:00:00Z",
  "valid_from": "2025-12-09T11:00:00Z",
  "valid_to": null
}
```

**Note**: The old entity's `valid_to` field is set to the current timestamp.

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/entities/660e8400-e29b-41d4-a716-446655440000/supersede \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "name": "FSRS v2.0",
    "entity_type": "CONCEPT",
    "summary": "Improved version with better accuracy"
  }'
```

---

## Procedural Memory Endpoints

Procedural memory stores skills and procedures with execution tracking.

### Create Skill

Store a new procedural skill with step-by-step instructions.

**Endpoint**: `POST /api/v1/skills`

**Request Body**:
```json
{
  "name": "pytest-test-creation",
  "domain": "coding",
  "task": "Create pytest unit tests for Python functions",
  "steps": [
    {
      "order": 1,
      "action": "Identify function under test",
      "tool": "read",
      "parameters": {},
      "expected_outcome": "Function signature and behavior understood"
    },
    {
      "order": 2,
      "action": "Create test file in tests/",
      "tool": "write",
      "parameters": {"pattern": "test_*.py"},
      "expected_outcome": "Test file created"
    },
    {
      "order": 3,
      "action": "Write test cases with assertions",
      "tool": null,
      "parameters": {},
      "expected_outcome": "Test cases cover edge cases"
    },
    {
      "order": 4,
      "action": "Run pytest to verify",
      "tool": "bash",
      "parameters": {"command": "pytest"},
      "expected_outcome": "All tests pass"
    }
  ],
  "script": "Given a function, create a test file with multiple test cases covering edge cases",
  "trigger_pattern": "write tests for"
}
```

**Parameters**:
- `name` (required): Skill name
- `domain` (required): One of `"coding"`, `"research"`, `"writing"`, `"analysis"`, `"communication"`, `"other"`
- `task` (required): Task description (used for embedding)
- `steps` (optional): List of procedure steps
  - `order`: Step order (≥1)
  - `action`: Action description
  - `tool`: Tool to use (optional)
  - `parameters`: Step parameters (default: `{}`)
  - `expected_outcome`: Expected result (optional)
- `script` (optional): High-level script description
- `trigger_pattern` (optional): Pattern for automatic invocation

**Response**: `201 Created`
```json
{
  "id": "990e8400-e29b-41d4-a716-446655440000",
  "name": "pytest-test-creation",
  "domain": "coding",
  "trigger_pattern": "write tests for",
  "steps": [...],
  "script": "Given a function, create a test file...",
  "success_rate": 0.0,
  "execution_count": 0,
  "last_executed": null,
  "version": 1,
  "deprecated": false,
  "created_at": "2025-12-09T10:30:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/skills \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "name": "git-commit-workflow",
    "domain": "coding",
    "task": "Create a git commit with proper message",
    "steps": [
      {"order": 1, "action": "Stage changes", "tool": "bash"},
      {"order": 2, "action": "Write commit message", "tool": "write"},
      {"order": 3, "action": "Commit changes", "tool": "bash"}
    ]
  }'
```

---

### Get Skill

Retrieve a specific skill by ID.

**Endpoint**: `GET /api/v1/skills/{skill_id}`

**Response**: `200 OK`
```json
{
  "id": "990e8400-e29b-41d4-a716-446655440000",
  "name": "pytest-test-creation",
  "domain": "coding",
  "trigger_pattern": "write tests for",
  "steps": [...],
  "script": "...",
  "success_rate": 0.85,
  "execution_count": 20,
  "last_executed": "2025-12-09T10:00:00Z",
  "version": 1,
  "deprecated": false,
  "created_at": "2025-12-08T15:30:00Z"
}
```

**Example**:
```bash
curl http://localhost:8765/api/v1/skills/990e8400-e29b-41d4-a716-446655440000 \
  -H "X-Session-ID: my-session"
```

---

### List Skills

List skills with filtering and deprecation control.

**Endpoint**: `GET /api/v1/skills`

**Query Parameters**:
- `domain` (optional): Filter by domain
- `include_deprecated` (default: false): Include deprecated skills
- `limit` (default: 50): Max results (1-200)

**Response**: `200 OK`
```json
{
  "skills": [
    {
      "id": "990e8400-e29b-41d4-a716-446655440000",
      "name": "pytest-test-creation",
      "domain": "coding",
      "success_rate": 0.85,
      "execution_count": 20,
      "deprecated": false
    }
  ],
  "total": 12
}
```

**Example**:
```bash
curl "http://localhost:8765/api/v1/skills?domain=coding&include_deprecated=false&limit=20" \
  -H "X-Session-ID: my-session"
```

---

### Recall Skills (Semantic Search)

Search for skills by task description using BGE-M3 embeddings.

**Endpoint**: `POST /api/v1/skills/recall`

**Request Body**:
```json
{
  "query": "how to write unit tests",
  "domain": "coding",
  "limit": 5
}
```

**Parameters**:
- `query` (required): Task description
- `domain` (optional): Filter by domain
- `limit` (default: 5): Max results (1-50)

**Response**: `200 OK`
```json
{
  "skills": [
    {
      "id": "990e8400-e29b-41d4-a716-446655440000",
      "name": "pytest-test-creation",
      "domain": "coding",
      "steps": [...],
      "success_rate": 0.85,
      "execution_count": 20
    }
  ],
  "total": 3
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/skills/recall \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "query": "create pytest tests",
    "domain": "coding",
    "limit": 3
  }'
```

---

### Execute Skill

Record a skill execution and update success metrics.

**Endpoint**: `POST /api/v1/skills/{skill_id}/execute`

**Request Body**:
```json
{
  "success": true,
  "duration_ms": 5000,
  "notes": "Successfully created 5 test cases"
}
```

**Parameters**:
- `success` (required): Whether execution succeeded
- `duration_ms` (optional): Execution duration in milliseconds
- `notes` (optional): Execution notes

**Response**: `200 OK`
```json
{
  "id": "990e8400-e29b-41d4-a716-446655440000",
  "name": "pytest-test-creation",
  "success_rate": 0.87,
  "execution_count": 21,
  "last_executed": "2025-12-09T10:30:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/skills/990e8400-e29b-41d4-a716-446655440000/execute \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session" \
  -d '{
    "success": true,
    "notes": "All tests passed"
  }'
```

---

### Deprecate Skill

Mark a skill as deprecated, optionally pointing to a replacement.

**Endpoint**: `POST /api/v1/skills/{skill_id}/deprecate`

**Query Parameters**:
- `replacement_id` (optional): UUID of replacement skill

**Response**: `200 OK`
```json
{
  "id": "990e8400-e29b-41d4-a716-446655440000",
  "name": "pytest-test-creation",
  "deprecated": true,
  "version": 1
}
```

**Example**:
```bash
curl -X POST "http://localhost:8765/api/v1/skills/990e8400-e29b-41d4-a716-446655440000/deprecate?replacement_id=aa0e8400-e29b-41d4-a716-446655440000" \
  -H "X-Session-ID: my-session"
```

---

### How-To Query

Natural language query for procedural knowledge with step-by-step instructions.

**Endpoint**: `GET /api/v1/skills/how-to/{query}`

**Query Parameters**:
- `domain` (optional): Filter by domain

**Response**: `200 OK`
```json
{
  "query": "write pytest tests",
  "skill": {
    "id": "990e8400-e29b-41d4-a716-446655440000",
    "name": "pytest-test-creation",
    "domain": "coding",
    "success_rate": 0.85
  },
  "steps": [
    "1. Identify function under test (using read)",
    "2. Create test file in tests/ (using write)",
    "3. Write test cases with assertions",
    "4. Run pytest to verify (using bash)"
  ],
  "confidence": 0.85
}
```

**No Match Response**:
```json
{
  "query": "unknown task",
  "skill": null,
  "steps": ["No matching skill found. Try creating one or rephrasing your query."],
  "confidence": 0.0
}
```

**Example**:
```bash
curl "http://localhost:8765/api/v1/skills/how-to/write%20pytest%20tests?domain=coding" \
  -H "X-Session-ID: my-session"
```

---

## Persistence Endpoints

Manage persistence layer including checkpoints, WAL, and recovery status.

### System Status

Get overall persistence system status including LSN and checkpoint info.

**Endpoint**: `GET /api/v1/persistence/status`

**Response**: `200 OK`
```json
{
  "started": true,
  "mode": "warm_start",
  "current_lsn": 15234,
  "checkpoint_lsn": 15000,
  "operations_since_checkpoint": 234,
  "uptime_seconds": 3600.5,
  "shutdown_requested": false
}
```

**Fields**:
- `started`: Whether persistence layer is running
- `mode`: Startup mode (`cold_start`, `warm_start`, `forced_cold`)
- `current_lsn`: Current Write-Ahead Log sequence number
- `checkpoint_lsn`: LSN of last checkpoint
- `operations_since_checkpoint`: Operations since last checkpoint
- `uptime_seconds`: Time since startup
- `shutdown_requested`: Whether shutdown is pending

**Example**:
```bash
curl http://localhost:8765/api/v1/persistence/status
```

---

### List Checkpoints

List available checkpoint files with metadata.

**Endpoint**: `GET /api/v1/persistence/checkpoints`

**Response**: `200 OK`
```json
[
  {
    "lsn": 15000,
    "timestamp": "2025-12-09T10:00:00Z",
    "size_bytes": 2048576,
    "components": ["buffer", "gate", "scorer", "neuromod"]
  },
  {
    "lsn": 14500,
    "timestamp": "2025-12-09T09:00:00Z",
    "size_bytes": 1998234,
    "components": ["buffer", "gate", "scorer", "neuromod"]
  }
]
```

**Example**:
```bash
curl http://localhost:8765/api/v1/persistence/checkpoints
```

---

### Create Checkpoint

Manually create a checkpoint (useful before deployments or maintenance).

**Endpoint**: `POST /api/v1/persistence/checkpoint`

**Request Body** (optional):
```json
{
  "force": false
}
```

**Parameters**:
- `force` (default: false): Force checkpoint even if recent one exists

**Response**: `200 OK`
```json
{
  "success": true,
  "lsn": 15234,
  "duration_seconds": 0.85,
  "message": "Checkpoint created at LSN 15234"
}
```

**Error Response**:
```json
{
  "success": false,
  "lsn": 0,
  "duration_seconds": 0.12,
  "message": "Checkpoint failed: disk full"
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/persistence/checkpoint \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

---

### WAL Status

Get Write-Ahead Log status including segment information.

**Endpoint**: `GET /api/v1/persistence/wal`

**Response**: `200 OK`
```json
{
  "current_lsn": 15234,
  "checkpoint_lsn": 15000,
  "segment_count": 5,
  "total_size_bytes": 10485760,
  "oldest_segment": 142,
  "current_segment": 147
}
```

**Example**:
```bash
curl http://localhost:8765/api/v1/persistence/wal
```

---

### Truncate WAL

Remove old WAL segments before last checkpoint to reclaim disk space.

**Endpoint**: `POST /api/v1/persistence/wal/truncate`

**Response**: `200 OK`
```json
{
  "success": true,
  "segments_removed": 3,
  "message": "Removed 3 WAL segments"
}
```

**Example**:
```bash
curl -X POST http://localhost:8765/api/v1/persistence/wal/truncate
```

---

### Recovery Info

Get information about the last recovery/startup.

**Endpoint**: `GET /api/v1/persistence/recovery`

**Response**: `200 OK`
```json
{
  "mode": "warm_start",
  "success": true,
  "checkpoint_lsn": 14500,
  "wal_entries_replayed": 234,
  "components_restored": {
    "buffer": true,
    "gate": true,
    "scorer": true,
    "neuromod": true
  },
  "errors": [],
  "duration_seconds": 2.5
}
```

**Example**:
```bash
curl http://localhost:8765/api/v1/persistence/recovery
```

---

### Health Check

Health check for persistence layer (used by load balancers and monitoring).

**Endpoint**: `GET /api/v1/persistence/health`

**Response**: `200 OK`
```json
{
  "status": "healthy",
  "current_lsn": 15234,
  "checkpoint_lsn": 15000,
  "warnings": []
}
```

**Degraded Response**:
```json
{
  "status": "degraded",
  "current_lsn": 25234,
  "checkpoint_lsn": 15000,
  "warnings": [
    "Checkpoint stale (650s old)",
    "Many uncommitted operations (10234)"
  ]
}
```

**Error Response**: `503 Service Unavailable`
```json
{
  "detail": "Persistence not started"
}
```

**Example**:
```bash
curl http://localhost:8765/api/v1/persistence/health
```

---

## WebSocket Channels

Real-time event streaming for system monitoring and debugging.

### Event Types

**System Events**:
- `system.start` - System startup
- `system.shutdown` - System shutdown
- `system.checkpoint` - Checkpoint created
- `system.wal_rotated` - WAL segment rotated

**Memory Events**:
- `memory.added` - Memory stored
- `memory.promoted` - Memory promoted (e.g., episode → entity)
- `memory.removed` - Memory deleted
- `memory.consolidated` - Memory consolidation completed

**Learning Events**:
- `learning.gate_updated` - Gate weights updated
- `learning.scorer_updated` - Scorer weights updated
- `learning.trace_created` - Eligibility trace created
- `learning.trace_decayed` - Eligibility trace decayed

**Neuromodulator Events**:
- `neuromod.dopamine_rpe` - Dopamine reward prediction error
- `neuromod.serotonin_mood` - Serotonin mood update
- `neuromod.norepinephrine_arousal` - Norepinephrine arousal update

**Health Events**:
- `health.update` - Health metrics update
- `health.warning` - Health warning
- `health.error` - Health error

### Channels

**All Events**: `ws://localhost:8765/ws/events`

Receives all event types.

**Memory Events**: `ws://localhost:8765/ws/memory`

Receives only memory-related events (`memory.*`).

**Learning Events**: `ws://localhost:8765/ws/learning`

Receives only learning and neuromodulator events (`learning.*`, `neuromod.*`).

**Health Events**: `ws://localhost:8765/ws/health`

Receives only health-related events (`health.*`).

### Event Format

All events follow this JSON format:

```json
{
  "type": "memory.added",
  "timestamp": 1638360000.123,
  "datetime": "2025-12-09T10:30:00.123Z",
  "data": {
    "memory_id": "550e8400-e29b-41d4-a716-446655440000",
    "content_preview": "Implemented FSRS decay algorithm..."
  }
}
```

### WebSocket Protocol

**Ping/Pong**:

Send `"ping"` to keep connection alive:

```json
{"type": "pong", "timestamp": 1638360000.123}
```

### Connection Examples

**JavaScript**:
```javascript
const ws = new WebSocket('ws://localhost:8765/ws/events');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`[${data.type}]`, data.data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

// Keep alive
setInterval(() => ws.send('ping'), 30000);
```

**Python**:
```python
import asyncio
import json
import websockets

async def listen():
    uri = "ws://localhost:8765/ws/memory"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            event = json.loads(message)
            print(f"[{event['type']}] {event['data']}")

asyncio.run(listen())
```

**curl** (using websocat):
```bash
websocat ws://localhost:8765/ws/events
```

### Example Events

**Memory Added**:
```json
{
  "type": "memory.added",
  "timestamp": 1638360000.123,
  "datetime": "2025-12-09T10:30:00.123Z",
  "data": {
    "memory_id": "550e8400-e29b-41d4-a716-446655440000",
    "content_preview": "Implemented FSRS decay algorithm for memory retrieval"
  }
}
```

**Checkpoint Created**:
```json
{
  "type": "system.checkpoint",
  "timestamp": 1638360000.456,
  "datetime": "2025-12-09T10:30:00.456Z",
  "data": {
    "lsn": 15234,
    "duration_seconds": 0.85
  }
}
```

**Dopamine RPE**:
```json
{
  "type": "neuromod.dopamine_rpe",
  "timestamp": 1638360000.789,
  "datetime": "2025-12-09T10:30:00.789Z",
  "data": {
    "context": "episode_recall",
    "rpe": 0.15,
    "expected": 0.7,
    "actual": 0.85
  }
}
```

**Health Warning**:
```json
{
  "type": "health.warning",
  "timestamp": 1638360001.234,
  "datetime": "2025-12-09T10:30:01.234Z",
  "data": {
    "warning": "Checkpoint stale",
    "details": {
      "age_seconds": 650
    }
  }
}
```

### Connection Status

Get WebSocket connection counts per channel:

**Endpoint**: `GET /ws/status`

**Response**:
```json
{
  "connections": {
    "events": 3,
    "memory": 1,
    "learning": 2,
    "health": 1
  },
  "channels": ["events", "memory", "learning", "health"]
}
```

---

## MCP Tools Reference

T4DM provides 26 MCP (Model Context Protocol) tools for integration with Claude Code and other MCP-compatible clients.

### Episodic Memory Tools

**create_episode**
```python
# Store a new episodic memory
{
  "content": "Implemented feature X",
  "outcome": "success",
  "emotional_valence": 0.8,
  "project": "my-project"
}
```

**recall_episodes**
```python
# Search episodes by content
{
  "query": "feature implementation",
  "limit": 10
}
```

**query_at_time**
```python
# Query episodes at specific timestamp
{
  "timestamp": "2025-12-09T10:30:00Z",
  "before_hours": 24,
  "after_hours": 0
}
```

**mark_important**
```python
# Mark episode as important
{
  "episode_id": "550e8400-...",
  "importance": 1.0
}
```

**create_episodes_batch**
```python
# Batch create episodes
{
  "episodes": [
    {"content": "Event 1", "outcome": "success"},
    {"content": "Event 2", "outcome": "neutral"}
  ]
}
```

**recall_batch**
```python
# Batch recall with multiple queries
{
  "queries": ["query 1", "query 2"],
  "limit_per_query": 5
}
```

### Semantic Memory Tools

**create_entity**
```python
# Create knowledge entity
{
  "name": "FSRS",
  "entity_type": "CONCEPT",
  "summary": "Free Spaced Repetition Scheduler"
}
```

**create_relation**
```python
# Create relationship between entities
{
  "source_id": "660e8400-...",
  "target_id": "770e8400-...",
  "relation_type": "IMPLEMENTS",
  "weight": 0.5
}
```

**semantic_recall**
```python
# Search entities by content
{
  "query": "memory algorithms",
  "limit": 10
}
```

**spread_activation**
```python
# ACT-R spreading activation
{
  "entity_id": "660e8400-...",
  "depth": 2,
  "threshold": 0.1
}
```

**supersede_fact**
```python
# Update entity with bi-temporal versioning
{
  "entity_id": "660e8400-...",
  "new_summary": "Updated description",
  "new_details": "Extended information"
}
```

**create_entities_batch**
```python
# Batch create entities
{
  "entities": [
    {"name": "Entity 1", "entity_type": "CONCEPT", "summary": "..."},
    {"name": "Entity 2", "entity_type": "PERSON", "summary": "..."}
  ]
}
```

### Procedural Memory Tools

**create_skill**
```python
# Store procedural skill
{
  "name": "pytest-testing",
  "domain": "coding",
  "task": "Write pytest tests",
  "steps": [
    {"order": 1, "action": "Read function", "tool": "read"},
    {"order": 2, "action": "Write tests", "tool": "write"}
  ]
}
```

**recall_skill**
```python
# Search skills by task
{
  "task": "write unit tests",
  "domain": "coding",
  "limit": 5
}
```

**execute_skill**
```python
# Record skill execution
{
  "skill_id": "990e8400-...",
  "success": true
}
```

**deprecate_skill**
```python
# Mark skill as deprecated
{
  "skill_id": "990e8400-...",
  "replacement_id": "aa0e8400-..."
}
```

**build_skill**
```python
# Build skill from recent episodes
{
  "task": "deploy to production",
  "lookback_hours": 24
}
```

**how_to**
```python
# Get step-by-step instructions
{
  "task": "write pytest tests",
  "domain": "coding"
}
```

**create_skills_batch**
```python
# Batch create skills
{
  "skills": [
    {"name": "Skill 1", "domain": "coding", "task": "..."},
    {"name": "Skill 2", "domain": "research", "task": "..."}
  ]
}
```

### System Tools

**consolidate_now**
```python
# Trigger memory consolidation
{
  "deep": false  # true for deep consolidation (HDBSCAN clustering)
}
```

**get_provenance**
```python
# Get memory provenance chain
{
  "memory_id": "550e8400-...",
  "memory_type": "episode"
}
```

**get_session_id**
```python
# Get current session ID
{}
```

**memory_stats**
```python
# Get memory system statistics
{
  "include_details": true
}
```

**apply_hebbian_decay**
```python
# Apply time-based decay to relationships
{
  "decay_rate": 0.95
}
```

**search_all_memories**
```python
# Search across all memory types
{
  "query": "FSRS implementation",
  "limit_per_type": 5
}
```

**get_related_memories**
```python
# Get related memories for an entity
{
  "entity_id": "660e8400-...",
  "depth": 2
}
```

### Bio-Inspired Learning Tools

**bio_encode**
```python
# Encode experience with hippocampal encoding
{
  "content": "Important event",
  "context": {"project": "research"},
  "valence": 0.8
}
```

**bio_eligibility_update**
```python
# Update eligibility traces
{
  "memory_id": "550e8400-...",
  "outcome": "success"
}
```

**bio_eligibility_credit**
```python
# Apply credit assignment
{
  "reward": 1.0,
  "context": "task_completion"
}
```

**bio_eligibility_step**
```python
# Step eligibility trace decay
{
  "decay_rate": 0.95
}
```

**bio_fes_write**
```python
# Write to frontal eye field state
{
  "state_id": "search_mode",
  "value": {"attention": 0.8, "focus": "code"}
}
```

**bio_fes_read**
```python
# Read from frontal eye field state
{
  "state_id": "search_mode"
}
```

**bio_status**
```python
# Get bio-inspired learning status
{}
```

### Tool Categories

**Episodic (6 tools)**:
- create_episode, recall_episodes, query_at_time, mark_important
- create_episodes_batch, recall_batch

**Semantic (6 tools)**:
- create_entity, create_relation, semantic_recall, spread_activation
- supersede_fact, create_entities_batch

**Procedural (7 tools)**:
- create_skill, recall_skill, execute_skill, deprecate_skill
- build_skill, how_to, create_skills_batch

**System (7 tools)**:
- consolidate_now, get_provenance, get_session_id, memory_stats
- apply_hebbian_decay, search_all_memories, get_related_memories

**Bio-Inspired (7 tools)**:
- bio_encode, bio_eligibility_update, bio_eligibility_credit, bio_eligibility_step
- bio_fes_write, bio_fes_read, bio_status

---

## Error Handling

### Error Response Format

All errors return a JSON object with a `detail` field:

```json
{
  "detail": "Error message"
}
```

### Error Sanitization

Errors are sanitized to prevent information leakage:

**Removed**:
- Database connection strings and credentials
- Internal file paths
- API keys and tokens
- Stack traces (logged server-side)

**Example Sanitized Error**:
```json
{
  "detail": "Failed to create episode: database connection error"
}
```

The full error with stack trace is logged server-side for debugging.

### Common Errors

**400 Bad Request**:
```json
{
  "detail": "Validation error: content exceeds maximum length of 50000 characters"
}
```

**404 Not Found**:
```json
{
  "detail": "Episode 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

**429 Too Many Requests**:
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 30

{
  "detail": "Rate limit exceeded. Try again in 30 seconds."
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Failed to create episode: internal error"
}
```

**503 Service Unavailable**:
```json
{
  "detail": "Persistence layer not initialized"
}
```

### Validation Errors

**Invalid Session ID**:
```json
{
  "detail": "Invalid session ID: must be 1-128 characters, alphanumeric with hyphens/underscores"
}
```

**Invalid UUID**:
```json
{
  "detail": "Invalid entity ID format"
}
```

**Invalid Enum Value**:
```json
{
  "detail": "Invalid entity_type: must be one of CONCEPT, PERSON, PLACE, EVENT, OBJECT, SKILL"
}
```

---

## Rate Limiting

### Configuration

**Default Limits**:
- 100 requests per minute per session
- Sliding window implementation
- Thread-safe rate limiting

### Rate Limit Headers

Every response includes rate limit information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1638360000
```

**Fields**:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

### Rate Limit Exceeded

When rate limit is exceeded:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 30
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1638360030

{
  "detail": "Rate limit exceeded. Try again in 30 seconds."
}
```

**Client Handling**:
1. Check `Retry-After` header for wait time
2. Exponential backoff recommended
3. Monitor `X-RateLimit-Remaining` to avoid hitting limit

### Rate Limit Bypass

For testing or administrative operations, rate limits can be temporarily disabled in the configuration.

---

## Best Practices

### Session Management

1. **Use Descriptive Session IDs**: Use meaningful session IDs like `user-123` or `project-abc`
2. **Session Isolation**: Each session has its own memory namespace
3. **Cleanup**: Sessions are automatically cleaned up after inactivity

### Performance Optimization

1. **Batch Operations**: Use batch endpoints (`create_episodes_batch`, `recall_batch`) for multiple operations
2. **Limit Results**: Use appropriate `limit` parameters to avoid overwhelming responses
3. **Filter Early**: Apply filters (`project`, `outcome`, `entity_type`) to reduce result sets
4. **Cache Results**: Cache frequently accessed entities and skills client-side

### Error Handling

1. **Check Status Codes**: Always check HTTP status codes before parsing responses
2. **Retry Logic**: Implement exponential backoff for 429 and 503 errors
3. **Log Errors**: Log error details for debugging
4. **Graceful Degradation**: Handle errors gracefully in user-facing applications

### Memory Management

1. **Mark Important Memories**: Use `mark_important` to prevent decay of critical memories
2. **Regular Consolidation**: Trigger `consolidate_now` periodically to optimize memory
3. **Supersede Facts**: Use `supersede_fact` instead of deleting outdated entities
4. **Deprecate Skills**: Use `deprecate_skill` instead of deleting outdated procedures

### WebSocket Best Practices

1. **Reconnect Logic**: Implement automatic reconnection on disconnect
2. **Ping/Pong**: Send periodic pings to keep connection alive
3. **Channel Selection**: Subscribe to specific channels to reduce traffic
4. **Error Handling**: Handle WebSocket errors gracefully

---

## Examples

### Complete Workflow: Learning a New Skill

```bash
# 1. Create episodes while learning
curl -X POST http://localhost:8765/api/v1/episodes \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: learning-pytest" \
  -d '{
    "content": "Read pytest documentation on fixtures",
    "project": "testing-course",
    "outcome": "success",
    "emotional_valence": 0.7
  }'

# 2. Create knowledge entities
curl -X POST http://localhost:8765/api/v1/entities \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: learning-pytest" \
  -d '{
    "name": "pytest fixtures",
    "entity_type": "CONCEPT",
    "summary": "Reusable test setup and teardown mechanisms"
  }'

# 3. Create a skill
curl -X POST http://localhost:8765/api/v1/skills \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: learning-pytest" \
  -d '{
    "name": "use-pytest-fixtures",
    "domain": "coding",
    "task": "Write pytest tests using fixtures",
    "steps": [
      {"order": 1, "action": "Define fixture with @pytest.fixture"},
      {"order": 2, "action": "Pass fixture as test parameter"},
      {"order": 3, "action": "Use fixture in test assertions"}
    ]
  }'

# 4. Record successful execution
curl -X POST http://localhost:8765/api/v1/skills/{skill_id}/execute \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: learning-pytest" \
  -d '{
    "success": true,
    "notes": "Successfully used fixtures in test suite"
  }'

# 5. Query how-to later
curl "http://localhost:8765/api/v1/skills/how-to/use%20pytest%20fixtures" \
  -H "X-Session-ID: learning-pytest"
```

### Research Workflow

```bash
# 1. Store research findings
curl -X POST http://localhost:8765/api/v1/episodes \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: research-fsrs" \
  -d '{
    "content": "FSRS paper shows 95% accuracy in predicting memory retention",
    "project": "memory-research",
    "outcome": "success",
    "emotional_valence": 0.9
  }'

# 2. Create entities for key concepts
curl -X POST http://localhost:8765/api/v1/entities \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: research-fsrs" \
  -d '{
    "name": "FSRS Algorithm",
    "entity_type": "CONCEPT",
    "summary": "Free Spaced Repetition Scheduler with 95% accuracy"
  }'

# 3. Link concepts
curl -X POST http://localhost:8765/api/v1/entities/relations \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: research-fsrs" \
  -d '{
    "source_id": "{fsrs_id}",
    "target_id": "{memory_retention_id}",
    "relation_type": "RELATED_TO",
    "weight": 0.8
  }'

# 4. Spread activation to find related concepts
curl -X POST http://localhost:8765/api/v1/entities/spread-activation \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: research-fsrs" \
  -d '{
    "entity_id": "{fsrs_id}",
    "depth": 3,
    "threshold": 0.2
  }'
```

---

## Additional Resources

- **Architecture Documentation**: `/mnt/projects/t4d/t4dm/docs/architecture.md`
- **SDK Documentation**: `/mnt/projects/t4d/t4dm/docs/SDK.md`
- **Algorithm Details**: `/mnt/projects/t4d/t4dm/docs/algorithms.md`
- **Testing Guide**: `/mnt/projects/t4d/t4dm/docs/archive/TESTING_GUIDE.md`
- **Hooks System**: `/mnt/projects/t4d/t4dm/docs/hooks_quick_reference.md`

---

**Last Updated**: 2025-12-09 | **Version**: 0.1.0
