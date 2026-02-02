# T4DM REST API Documentation

## Overview

The T4DM REST API provides programmatic access to the tripartite neural memory system:

- **Episodic Memory**: Time-sequenced experiences with FSRS decay
- **Semantic Memory**: Knowledge graph with ACT-R activation and Hebbian learning
- **Procedural Memory**: Skill patterns with execution tracking

## Base URL

```
http://localhost:8765/api/v1
```

## Authentication

Pass session ID via `X-Session-ID` header to isolate memory namespaces:

```bash
curl -H "X-Session-ID: my-session" http://localhost:8765/api/v1/health
```

## Rate Limiting

- 100 requests per minute per session
- Returns `429 Too Many Requests` with `Retry-After` header when exceeded

## Endpoints

### System

#### Health Check
```
GET /api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "0.1.0",
  "session_id": "default"
}
```

#### Memory Statistics
```
GET /api/v1/stats
```

Response:
```json
{
  "session_id": "default",
  "episodic": {"total_episodes": 150},
  "semantic": {"total_entities": 45, "total_relations": 120},
  "procedural": {"total_skills": 12}
}
```

#### Trigger Consolidation
```
POST /api/v1/consolidate
```

Request:
```json
{"deep": false}
```

Response:
```json
{
  "success": true,
  "type": "light",
  "results": {"duplicates_found": 3, "decay_updated": 50}
}
```

---

### Episodic Memory

#### Create Episode
```
POST /api/v1/episodes
```

Request:
```json
{
  "content": "Implemented FSRS decay algorithm for memory retrieval",
  "project": "t4dm",
  "file": "src/memory/episodic.py",
  "tool": "python",
  "outcome": "success",
  "emotional_valence": 0.8
}
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "default",
  "content": "Implemented FSRS decay algorithm...",
  "timestamp": "2024-01-15T10:30:00Z",
  "outcome": "success",
  "emotional_valence": 0.8,
  "context": {"project": "t4dm", "file": "src/memory/episodic.py"},
  "access_count": 1,
  "stability": 1.0,
  "retrievability": 1.0
}
```

#### Get Episode
```
GET /api/v1/episodes/{episode_id}
```

#### List Episodes
```
GET /api/v1/episodes?page=1&page_size=20&project=t4dm&outcome=success
```

#### Recall Episodes (Semantic Search)
```
POST /api/v1/episodes/recall
```

Request:
```json
{
  "query": "memory decay algorithms",
  "limit": 10,
  "min_similarity": 0.5,
  "project": "t4dm"
}
```

Response:
```json
{
  "query": "memory decay algorithms",
  "episodes": [...],
  "scores": [0.92, 0.87, 0.75]
}
```

#### Mark Episode Important
```
POST /api/v1/episodes/{episode_id}/mark-important?importance=1.0
```

#### Delete Episode
```
DELETE /api/v1/episodes/{episode_id}
```

---

### Semantic Memory

#### Create Entity
```
POST /api/v1/entities
```

Request:
```json
{
  "name": "FSRS Algorithm",
  "entity_type": "CONCEPT",
  "summary": "Free Spaced Repetition Scheduler for memory decay",
  "details": "Implements power-law forgetting curve with stability tracking"
}
```

#### Get Entity
```
GET /api/v1/entities/{entity_id}
```

#### List Entities
```
GET /api/v1/entities?entity_type=CONCEPT&limit=50
```

#### Create Relationship
```
POST /api/v1/entities/relations
```

Request:
```json
{
  "source_id": "550e8400-...",
  "target_id": "660e8400-...",
  "relation_type": "IMPLEMENTS",
  "weight": 0.5
}
```

#### Recall Entities (Semantic Search)
```
POST /api/v1/entities/recall
```

Request:
```json
{
  "query": "memory algorithms",
  "limit": 10,
  "entity_types": ["CONCEPT", "TECHNIQUE"]
}
```

#### Spreading Activation
```
POST /api/v1/entities/spread-activation
```

Request:
```json
{
  "entity_id": "550e8400-...",
  "depth": 2,
  "threshold": 0.1
}
```

Response:
```json
{
  "entities": [...],
  "activations": [1.0, 0.8, 0.65, 0.42],
  "paths": [["FSRS"], ["FSRS", "IMPLEMENTS", "Memory"], ...]
}
```

#### Supersede Entity
```
POST /api/v1/entities/{entity_id}/supersede
```

Creates new version and marks old as invalid (bi-temporal versioning).

---

### Procedural Memory

#### Create Skill
```
POST /api/v1/skills
```

Request:
```json
{
  "name": "pytest-test-creation",
  "domain": "coding",
  "task": "Create pytest unit tests for Python functions",
  "steps": [
    {"order": 1, "action": "Identify function under test", "tool": "read"},
    {"order": 2, "action": "Create test file in tests/", "tool": "write"},
    {"order": 3, "action": "Write test cases with assertions"},
    {"order": 4, "action": "Run pytest to verify", "tool": "bash"}
  ],
  "script": "Given a function, create a test file with multiple test cases covering edge cases",
  "trigger_pattern": "write tests for"
}
```

#### Get Skill
```
GET /api/v1/skills/{skill_id}
```

#### List Skills
```
GET /api/v1/skills?domain=coding&include_deprecated=false&limit=50
```

#### Recall Skills (Semantic Search)
```
POST /api/v1/skills/recall
```

Request:
```json
{
  "query": "how to write unit tests",
  "domain": "coding",
  "limit": 5
}
```

#### Record Execution
```
POST /api/v1/skills/{skill_id}/execute
```

Request:
```json
{
  "success": true,
  "duration_ms": 5000,
  "notes": "Successfully created 5 test cases"
}
```

#### Deprecate Skill
```
POST /api/v1/skills/{skill_id}/deprecate?replacement_id=...
```

#### How-To Query
```
GET /api/v1/skills/how-to/write%20pytest%20tests
```

Response:
```json
{
  "query": "write pytest tests",
  "skill": {...},
  "steps": [
    "1. Identify function under test (using read)",
    "2. Create test file in tests/ (using write)",
    "3. Write test cases with assertions",
    "4. Run pytest to verify (using bash)"
  ],
  "confidence": 0.95
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common status codes:
- `400` - Bad Request (invalid parameters)
- `404` - Not Found
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error
- `503` - Service Unavailable (storage connection issues)

### Error Sanitization

Error messages are sanitized to prevent information leakage:
- Database connection strings (Neo4j URIs, credentials) are removed
- Internal file paths are masked
- API keys/tokens are redacted
- Stack traces are logged server-side but not returned to clients

Example sanitized response:
```json
{
  "detail": "Failed to create episode: database connection error"
}
```

The full error with stack trace is logged server-side for debugging.

---

## OpenAPI Schema

Interactive API documentation available at:
- Swagger UI: `http://localhost:8765/docs`
- ReDoc: `http://localhost:8765/redoc`
- OpenAPI JSON: `http://localhost:8765/openapi.json`
