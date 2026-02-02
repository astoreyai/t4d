# Phase 10: Production Readiness - Implementation Plan

**Generated**: 2026-01-04 | **Agent**: Explore | **Status**: PLANNING COMPLETE

---

## Executive Summary

Phase 10 hardens T4DM for production deployment across 5 dimensions:

1. **Module Cleanup** - Consolidate `/bridge` and `/bridges`, remove dead code
2. **API Documentation** - Complete OpenAPI spec, SDK streaming, WebSocket schemas
3. **Deployment** - K8s init containers, graceful shutdown, HPA tuning
4. **Observability** - Prometheus metrics endpoint, trace instrumentation
5. **Performance** - Embedding cache, bulk APIs, pagination

---

## 1. Module Cleanup

### 1.1 Bridge Consolidation

**Current State**:
- `/src/t4dm/bridge/` (singular): `MemoryNCABridge` - NCA field coupling
- `/src/t4dm/bridges/` (plural): Dopamine, Capsule, FF bridges

**Action**: Merge into `/src/t4dm/bridges/`

| Current File | New Location | Notes |
|--------------|--------------|-------|
| `bridge/memory_nca.py` | `bridges/nca_bridge.py` | Rename class to NCABridge |
| `bridge/__init__.py` | DELETE | |
| `tests/bridge/` | `tests/bridges/nca/` | |

**Import Updates** (13 locations):
- `examples/nca_demo.py`
- Any files importing from `t4dm.bridge`

### 1.2 Dead Code Archive

| File | Status | Action |
|------|--------|--------|
| `learning/generative_replay.py` | Skeleton only | Archive to `docs/archive/` |

---

## 2. API Documentation

### 2.1 Missing Endpoint Documentation

| Endpoint | File | Status |
|----------|------|--------|
| `POST /api/v1/checkpoint` | `server.py` | Needs route file |
| `GET /api/v1/checkpoint/status` | `server.py` | Needs route file |
| `/ws` WebSocket | `websocket.py` | Schema undefined |

**Create**: `/src/t4dm/api/routes/persistence.py`

### 2.2 SDK Enhancements

| Feature | Status | Priority |
|---------|--------|----------|
| `stream_recall()` | Missing | HIGH |
| `stream_search()` | Missing | HIGH |
| Bulk import/export | Missing | MEDIUM |
| `async with client.session()` | Missing | MEDIUM |
| Retry decorator | Missing | LOW |

---

## 3. Deployment

### 3.1 Kubernetes Gaps

| Component | Status | Action |
|-----------|--------|--------|
| Init container | Missing | Add for DB migrations |
| Sidecar logging | Missing | Add Fluent Bit |
| Network policy | Missing | Add inter-service rules |
| Pod disruption budget | Missing | Add for graceful updates |
| HPA thresholds | Placeholder | Tune with load testing |

### 3.2 Graceful Shutdown

**Current Issues**:
- No SIGTERM handler for pod eviction
- No connection draining for in-flight requests
- Fixed 30s health check timeout

**Add to `server.py`**:
```python
import signal

def handle_sigterm(signum, frame):
    logger.info("SIGTERM received, initiating graceful shutdown")
    # Drain connections
    # Checkpoint state
    # Exit

signal.signal(signal.SIGTERM, handle_sigterm)
```

---

## 4. Observability

### 4.1 Prometheus Metrics

**Gap**: `/metrics` endpoint NOT registered in API

**Add to `server.py`**:
```python
from t4dm.observability.prometheus import prometheus_router
app.include_router(prometheus_router)
```

### 4.2 Missing Instrumentation

| Area | Status | Action |
|------|--------|--------|
| Request count/latency histograms | Missing | Add middleware |
| Database operation spans | Missing | Instrument Neo4j/Qdrant |
| Memory operation tracing | Missing | Add to episodic/semantic |
| Alert thresholds | Missing | Define SLO/SLI |

---

## 5. Performance

### 5.1 Caching Opportunities

| Operation | Latency | Cache Benefit |
|-----------|---------|---------------|
| Embedding lookups | 50-500ms | HIGH |
| Entity graph traversal | 10-100ms | MEDIUM |
| FSRS calculations | <1ms | LOW |
| Consolidation clustering | 5-60s | MEDIUM |

**Implementation**: Redis or in-memory LRU for embeddings

### 5.2 Bulk API Endpoints

**Add**:
- `POST /api/v1/episodes/batch`
- `GET /api/v1/episodes/search/batch`
- Cursor pagination support

---

## Implementation Roadmap

### Phase 10a: Module Cleanup (5 days)
1. Consolidate `/bridge` → `/bridges/nca_bridge.py` (2d)
2. Update all imports (1d)
3. Test migration (1.5d)
4. Archive dead code (0.5d)

### Phase 10b: API Documentation (6 days)
1. Document Persistence APIs (1.5d)
2. Define WebSocket schema (1d)
3. Add SDK streaming (1.5d)
4. Update OpenAPI (1d)
5. Write integration guide (1d)

### Phase 10c: Deployment Hardening (8 days)
1. Add K8s init container (2d)
2. Implement graceful shutdown (2d)
3. Add PDB, network policy (1.5d)
4. Configure HPA (1d)
5. Pre-deployment checklist (0.5d)
6. Staging test (1d)

### Phase 10d: Observability (5 days)
1. Register `/metrics` endpoint (1d)
2. Add request histograms (1.5d)
3. Instrument database ops (1.5d)
4. Add profiling hooks (1d)

### Phase 10e: Performance (6 days)
1. Implement embedding cache (2d)
2. Add bulk API endpoints (2d)
3. Implement cursor pagination (1d)
4. Optimize Cypher queries (1d)

---

## File Changes Summary

### Source Files (29)
- `/src/t4dm/bridge/` → DELETE (after migration)
- `/src/t4dm/bridges/__init__.py` → Update exports
- `/src/t4dm/api/server.py` → Add prometheus, SIGTERM
- `/src/t4dm/api/routes/persistence.py` → CREATE
- `/src/t4dm/sdk/client.py` → Add batch operations

### Config/Deploy (9)
- `/deploy/kubernetes/*.yaml` → 6 files
- `/docker-compose.yml` → TLS config
- `/Dockerfile` → Labels, optimize

### Tests (9)
- `/tests/bridge/` → Migrate to `/tests/bridges/nca/`
- Add batch endpoint tests
- Add Prometheus metrics tests

---

## Effort Estimate

**Total**: 30 days (6 weeks)

- **Critical Path**: Cleanup → Deploy → Observability (18 days)
- **Parallelizable**: API docs + Performance (8 days)
- **Testing & Review**: 4 days
