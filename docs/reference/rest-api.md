# REST API Reference

Complete HTTP API documentation.

## Base URL

```
http://localhost:8765/api/v1
```

## Authentication

### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Production | API key for authentication |
| `X-Admin-Key` | Some endpoints | Admin key for sensitive operations |
| `X-Session-ID` | Optional | Session identifier (defaults to configured) |

### Rate Limiting

- **Limit**: 100 requests per 60 seconds per session
- **Headers**: `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## Episodes

### Create Episode

```http
POST /api/v1/episodes
```

**Request Body:**

```json
{
  "content": "Learned about REST APIs",
  "project": "api-project",
  "file": "api.py",
  "tool": "curl",
  "outcome": "SUCCESS",
  "emotional_valence": 0.8
}
```

**Response:** `201 Created`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "default",
  "content": "Learned about REST APIs",
  "timestamp": "2026-01-03T12:00:00Z",
  "stability": 1.0,
  "retrievability": 1.0
}
```

### List Episodes

```http
GET /api/v1/episodes?offset=0&limit=50
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offset` | int | 0 | Pagination offset |
| `limit` | int | 50 | Max results (1-100) |
| `project` | string | - | Filter by project |

**Response:** `200 OK`

```json
{
  "items": [...],
  "total": 150,
  "offset": 0,
  "limit": 50,
  "has_more": true
}
```

### Recall Episodes

```http
POST /api/v1/episodes/recall
```

**Request Body:**

```json
{
  "query": "REST API",
  "limit": 10,
  "min_similarity": 0.7,
  "filters": {
    "project": "api-project"
  }
}
```

**Response:** `200 OK`

```json
{
  "query": "REST API",
  "results": [
    {
      "episode": {...},
      "score": 0.92,
      "components": {
        "similarity": 0.90,
        "recency": 0.85,
        "importance": 0.80
      }
    }
  ]
}
```

### Mark Important

```http
POST /api/v1/episodes/{episode_id}/mark-important
```

**Request Body:**

```json
{
  "importance": 1.0
}
```

## Entities

### Create Entity

```http
POST /api/v1/entities
```

**Request Body:**

```json
{
  "name": "REST API",
  "entity_type": "CONCEPT",
  "summary": "HTTP-based API architecture",
  "details": {
    "methods": ["GET", "POST", "PUT", "DELETE"]
  }
}
```

**Entity Types:** `CONCEPT`, `PERSON`, `PROJECT`, `TOOL`, `TECHNIQUE`, `FACT`

### Create Relationship

```http
POST /api/v1/entities/relations
```

**Request Body:**

```json
{
  "source_id": "...",
  "target_id": "...",
  "relation_type": "RELATED_TO"
}
```

### Spread Activation

```http
POST /api/v1/entities/spread-activation
```

**Request Body:**

```json
{
  "entity_id": "...",
  "depth": 2,
  "decay_factor": 0.5
}
```

**Response:**

```json
{
  "entities": [...],
  "activations": {
    "entity_id_1": 1.0,
    "entity_id_2": 0.5,
    "entity_id_3": 0.25
  },
  "paths": [...]
}
```

## Skills

### Create Skill

```http
POST /api/v1/skills
```

**Request Body:**

```json
{
  "name": "api_request",
  "domain": "CODING",
  "trigger_pattern": "make API request",
  "description": "Make HTTP API requests",
  "script": "curl -X POST ...",
  "steps": [
    {
      "order": 1,
      "action": "Prepare headers",
      "tool": "none"
    },
    {
      "order": 2,
      "action": "Send request",
      "tool": "curl"
    }
  ]
}
```

**Domains:** `CODING`, `RESEARCH`, `TRADING`, `DEVOPS`, `WRITING`

### Record Execution

```http
POST /api/v1/skills/{skill_id}/execute
```

**Request Body:**

```json
{
  "success": true,
  "duration_ms": 1500,
  "output": "Response: 200 OK"
}
```

### How-To Query

```http
GET /api/v1/skills/how-to/{query}
```

**Response:**

```json
{
  "skill": {...},
  "steps": [...],
  "confidence": 0.85
}
```

## System

### Health Check

```http
GET /api/v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2026-01-03T12:00:00Z",
  "version": "0.4.0",
  "session_id": "default"
}
```

### Statistics

```http
GET /api/v1/stats
```

**Response:**

```json
{
  "session_id": "default",
  "episodic": {
    "count": 1500,
    "avg_stability": 0.75
  },
  "semantic": {
    "entity_count": 200,
    "relation_count": 450
  },
  "procedural": {
    "skill_count": 50,
    "avg_success_rate": 0.85
  }
}
```

### Consolidate

```http
POST /api/v1/consolidate
```

**Headers:** Requires `X-Admin-Key`

**Request Body:**

```json
{
  "deep": true
}
```

## WebSocket

### Connect

```
ws://localhost:8765/ws/events
ws://localhost:8765/ws/memory
ws://localhost:8765/ws/learning
ws://localhost:8765/ws/health
```

### Event Types

```json
// Memory events
{"type": "memory_added", "data": {"id": "...", "content_preview": "..."}}
{"type": "memory_promoted", "data": {"id": "...", "target": "semantic"}}

// Learning events
{"type": "dopamine_rpe", "data": {"rpe": 0.3, "context": "..."}}
{"type": "gate_updated", "data": {"layer": "input", "magnitude": 0.02}}

// Health events
{"type": "health_update", "data": {"cpu": 45, "memory": 60}}
```

## Error Responses

### Error Format

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Episode not found",
    "details": {
      "episode_id": "..."
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `UNAUTHORIZED` | 401 | Missing/invalid API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

## OpenAPI

Full OpenAPI specification available at:

- **Swagger UI**: `http://localhost:8765/docs`
- **ReDoc**: `http://localhost:8765/redoc`
- **JSON**: `http://localhost:8765/openapi.json`
