# API Module

**Path**: `t4dm/api/` | **Files**: 13 | **Framework**: FastAPI

Production-grade REST API for the T4DM tripartite memory system.

---

## Quick Start

```python
from t4dm.api import app, main

# Run server
main()  # Starts uvicorn on configured host:port
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
├─────────────────────────────────────────────────────────┤
│  Middleware Stack (applied in reverse order):           │
│  1. ApiKeyAuthMiddleware      - X-API-Key validation    │
│  2. RequestSizeLimitMiddleware - 5MB body limit         │
│  3. SecurityHeadersMiddleware  - XSS/clickjacking       │
│  4. CORSMiddleware             - Origin validation      │
├─────────────────────────────────────────────────────────┤
│  Routes:                                                │
│  /api/v1/episodes/    - Episodic memory CRUD + recall   │
│  /api/v1/entities/    - Semantic entities + relations   │
│  /api/v1/skills/      - Procedural memory + execution   │
│  /api/v1/config/      - Runtime configuration           │
│  /api/v1/persistence/ - WAL + checkpoint status         │
│  /api/v1/viz/         - Visualization + bio mechanisms  │
│  /ws/*                - WebSocket event streaming       │
├─────────────────────────────────────────────────────────┤
│  Lifespan:                                              │
│  startup  → OpenTelemetry, persistence, health metrics  │
│  shutdown → Final checkpoint, service cleanup           │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

| File | Purpose | Key Exports |
|------|---------|-------------|
| `__init__.py` | Public API | `app`, `main` |
| `server.py` | FastAPI factory + lifespan | `app`, `lifespan()`, middlewares |
| `deps.py` | Dependency injection | `get_session_id()`, `get_memory_services()`, `require_api_key()` |
| `errors.py` | Error sanitization | `sanitize_error()`, `create_error_response()` |
| `websocket.py` | Real-time events | `ConnectionManager`, `emit_*()` functions |
| `routes/episodes.py` | Episodic CRUD | 7 endpoints |
| `routes/entities.py` | Semantic CRUD | 10 endpoints |
| `routes/skills.py` | Procedural CRUD | 9 endpoints |
| `routes/config.py` | Configuration | 5 endpoints + presets |
| `routes/system.py` | Health/stats | 6 endpoints |
| `routes/persistence.py` | WAL/checkpoint | 7 endpoints |
| `routes/visualization.py` | 3D graphs + bio | 40+ endpoints |

---

## Authentication

### Headers

| Header | Required | Purpose |
|--------|----------|---------|
| `X-Session-ID` | Optional | Session isolation (defaults to config) |
| `X-API-Key` | Conditional | API authentication (if `api_key_required=true`) |
| `X-Admin-Key` | For admin ops | Config updates, consolidation, presets |

### Security Features

- Constant-time key comparison (timing attack prevention)
- Request size limit (5MB)
- CORS origin validation
- Error message sanitization (no credentials/paths leaked)
- Path traversal protection on doc endpoints

---

## Endpoints Summary

### Episodic Memory (`/api/v1/episodes/`)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/` | Create episode |
| GET | `/` | List with pagination |
| GET | `/{id}` | Get single |
| PUT | `/{id}` | Update |
| DELETE | `/{id}` | Delete |
| POST | `/recall` | Semantic search |
| POST | `/{id}/mark-important` | Boost valence |

### Semantic Memory (`/api/v1/entities/`)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/` | Create entity |
| GET | `/` | List with type filter |
| GET | `/{id}` | Get single |
| PUT | `/{id}` | Update |
| DELETE | `/{id}` | Delete |
| POST | `/relations` | Create relationship |
| POST | `/recall` | Semantic search |
| POST | `/spread-activation` | ACT-R activation |
| POST | `/{id}/supersede` | Bi-temporal version |

### Procedural Memory (`/api/v1/skills/`)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/` | Create skill |
| GET | `/` | List with domain filter |
| GET | `/{id}` | Get single |
| PUT | `/{id}` | Update |
| DELETE | `/{id}` | Delete |
| POST | `/recall` | Task search |
| POST | `/{id}/execute` | Record execution |
| POST | `/{id}/deprecate` | Soft delete |
| GET | `/how-to/{query}` | Natural language lookup |

### WebSocket Channels

| Channel | Events |
|---------|--------|
| `/ws/events` | All events |
| `/ws/memory` | Memory add/promote/remove/consolidate |
| `/ws/learning` | Gate/scorer/trace updates + neuromod |
| `/ws/health` | CPU, memory, checkpoint age |

---

## Configuration Presets

Apply via `POST /api/v1/config/presets/{name}`:

| Preset | Purpose |
|--------|---------|
| `bio-plausible` | CompBio-recommended biological fidelity |
| `performance` | Computational efficiency |
| `conservative` | Memory retention priority |
| `exploration` | Novelty-seeking bias |

---

## Rate Limiting

- Default: 100 requests / 60 seconds per session
- Returns `429 Too Many Requests` with `Retry-After` header

---

## Error Responses

```json
{
  "status": "error",
  "code": 400,
  "detail": "Sanitized error message"
}
```

Sanitized patterns:
- Connection strings with credentials
- API keys, tokens, passwords
- File paths with usernames
- IP addresses with ports

---

## Dependencies

**Internal**:
- `t4dm.core.config` - Settings
- `t4dm.core.services` - Memory services
- `t4dm.observability.tracing` - OpenTelemetry
- `t4dm.persistence` - WAL + checkpoints
- `t4dm.consolidation.service` - Consolidation

**External**:
- FastAPI, Pydantic, Uvicorn
- psutil (health metrics)
- OpenTelemetry SDK

---

## Deployment

```python
# Production
config = {
    "api_key": "your-key",
    "api_key_required": True,
    "admin_api_key": "admin-key",
    "cors_origins": ["https://your-domain.com"],
}
```

Health check: `GET /api/v1/health` (no auth required)
