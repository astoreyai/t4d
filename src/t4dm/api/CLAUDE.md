# API Module
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/api/`

## What
FastAPI-based REST API and WebSocket server for the tripartite memory system. Exposes CRUD operations for episodic, semantic, and procedural memory plus real-time event streaming, visualization endpoints, and demo APIs.

## How
- FastAPI application with lifespan management (startup: OpenTelemetry, persistence, health metrics; shutdown: connection draining, final checkpoint)
- Middleware stack (applied in reverse order): ApiKeyAuth, RateLimit, RequestSizeLimit (5MB), RequestTracking, SecurityHeaders, CORS
- Dependency injection via `deps.py` for session ID, memory services, and API key validation
- WebSocket connection manager with channel-based event streaming (events, memory, learning, health)
- Error sanitization strips credentials, paths, and IPs from responses

## Why
Provides programmatic and real-time access to memory operations. Decouples the memory system from specific clients (CLI, Kymera, external integrations). Production-hardened with security headers, rate limiting, graceful shutdown, and API key authentication.

## Key Files
| File | Purpose |
|------|---------|
| `server.py` | FastAPI app factory, lifespan, middleware stack |
| `deps.py` | Dependency injection (session ID, services, auth) |
| `errors.py` | Error sanitization (no credential/path leaks) |
| `websocket.py` | ConnectionManager, health metrics broadcaster |
| `routes/episodes.py` | Episodic memory CRUD + recall (7 endpoints) |
| `routes/entities.py` | Semantic memory CRUD + spread activation (10 endpoints) |
| `routes/skills.py` | Procedural memory CRUD + execution tracking (9 endpoints) |
| `routes/system.py` | Health check, stats (6 endpoints) |
| `routes/visualization.py` | 3D graph + bio mechanism endpoints (40+) |
| `routes/persistence.py` | WAL + checkpoint status (7 endpoints) |
| `routes/compat.py` | Mem0-compatible REST API endpoints |
| `routes/ws_viz.py` | WebSocket visualization streaming |
| `middleware/rate_limit.py` | Token bucket rate limiting (100 req/min) |

## Data Flow
```
Client Request
    -> Middleware Stack (auth, rate limit, size check, tracking)
    -> Router (episodes/entities/skills/system/viz)
    -> deps.py (get_services, get_session_id)
    -> core.services (memory operations)
    -> Response (sanitized errors)
```

## Integration Points
- **core**: `get_settings()`, `get_services()`, `cleanup_services()`
- **consolidation**: `get_consolidation_service()` for consolidation endpoints
- **observability**: OpenTelemetry tracing, Prometheus metrics
- **persistence**: WAL + checkpoint management via lifespan
- **WebSocket channels**: `/ws/events`, `/ws/memory`, `/ws/learning`, `/ws/health`
