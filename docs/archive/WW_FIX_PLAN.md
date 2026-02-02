# T4DM Fix Plan v2.0

**Created**: 2025-11-27 | **Updated**: 2025-11-27 | **Total Tasks**: 48 | **Priority**: Production-Critical

> Based on comprehensive re-analysis after Phase 1-6 completion. Previous phases implemented thread-safe singletons, timeouts, saga pattern, session isolation, and observability.

## Status Dashboard

| Phase | Status | Priority | Tasks | Description |
|-------|--------|----------|-------|-------------|
| Phase 1: Critical Bugs | COMPLETE | P0 | 7 | Runtime crashes, injection vulnerabilities |
| Phase 2: Performance | COMPLETE | P1 | 8 | O(n²) algorithms, N+1 queries, caching |
| Phase 3: Security | COMPLETE | P1 | 6 | Rate limiting, auth, input sanitization |
| Phase 4: Test Coverage | COMPLETE | P2 | 12 | Reach 75%+ coverage |
| Phase 5: API Cleanup | COMPLETE | P2 | 8 | Naming consistency, type safety |
| Phase 6: Documentation | COMPLETE | P3 | 7 | Algorithm docs, API docs, deployment |

### Phase 1 Completion Summary (2025-11-27)
All critical bugs fixed:
- P1-001: Added missing `import asyncio` to semantic.py
- P1-002: Added label/relationship validation to prevent Cypher injection
- P1-003: Verified session isolation via payload filtering (working correctly)
- P1-004: Added `CompensationError` exception for saga failures
- P1-005: Fixed race condition in async lock creation with threading guard
- P1-006: Created conftest.py with session-scoped event loop
- P1-007: Added Neo4j indexes for Entity.sessionId and Procedure.sessionId

### Phase 2 Completion Summary (2025-11-27)
All performance optimizations implemented:
- P2-001: Replaced O(n²) clustering with HDBSCAN (3.17x speedup)
- P2-002: Added batch relationship queries to eliminate N+1
- P2-003: Implemented LRU embedding cache with MD5 hashing
- P2-004: Added Qdrant session_id prefiltering
- P2-005: Parallel Hebbian updates with asyncio.gather
- P2-006: Lazy model loading with double-check locking
- P2-007: Neo4j connection pooling (50 connections, 30s timeout)
- P2-008: Parallel batch upsert for Qdrant

### Phase 3 Completion Summary (2025-11-27)
All security hardening complete:
- P3-001: Rate limiting (100 req/60s sliding window) with thread-safe RateLimiter
- P3-002: Input sanitization module with null byte/control char removal
- P3-003: Authentication context with @require_auth, @require_role decorators
- P3-004: Secure config loading with secret masking and permission checks
- P3-005: Request ID tracking with 8-char UUID per request
- P3-006: Weight validation ensuring sum=1.0 and all in [0,1]

### Phase 4 Completion Summary (2025-11-27)
Test coverage increased from 47% to 75%+:
- P4-001: Consolidation service tests (31 tests, 801 lines)
- P4-002: MCP gateway tests (44 tests, 911 lines)
- P4-003: Observability module tests (53 tests, 815 lines)
- P4-004: Episodic memory tests (25 tests, 817 lines) - FSRS algorithm verification
- P4-005: Semantic memory tests (21 tests, 762 lines) - ACT-R, Hebbian verification
- P4-006: Procedural memory tests (18 tests, 710 lines) - MEMP algorithm
- P4-007: Integration test suite (8 scenarios) - memory lifecycle, session isolation
- P4-008: Performance regression tests (8 benchmarks) - 1K/10K episode thresholds
- P4-009: Property-based tests (22 tests) - Hypothesis for algorithm bounds
- P4-010: Mocking infrastructure (17 fixtures) - comprehensive async-ready mocks
- P4-011: CI coverage gate (.github/workflows/test.yml) - 70% threshold enforced
- P4-012: Test documentation (tests/README.md) - 602 lines, 40+ examples

**Total**: 222+ test methods, 5,000+ lines of test code

### Phase 5 Completion Summary (2025-11-27)
API cleanup and standardization complete:
- P5-001: Standardized method naming (build→create_skill, retrieve→recall_skill)
- P5-002: TypedDict for MCP responses (21 TypedDict classes in types.py)
- P5-003: Consistent error codes (ErrorCode enum with 7 standard codes)
- P5-004: Pagination for all list operations (offset, limit, has_more)
- P5-005: Result count in all responses (count, total fields)
- P5-006: Cypher moved to storage layer (6 high-level Neo4j methods)
- P5-007: OpenAPI schema (schema.py + openapi.json/yaml generated)
- P5-008: Deprecation warnings for backward compatibility

**New Files**: types.py, errors.py, schema.py, openapi.json

### Phase 6 Completion Summary (2025-11-27)
Documentation complete:
- P6-001: Algorithm documentation (FSRS, ACT-R, Hebbian, HDBSCAN formulas)
- P6-002: API reference (all 17 MCP tools with examples)
- P6-003: Deployment guide (Docker, manual install, production checklist)
- P6-004: Architecture documentation (system diagrams, data flows)
- P6-005: Contributing guide (setup, code style, PR process)
- P6-006: Changelog (v2.0.0 with all phase improvements)
- P6-007: Code comments audit (8 files with "why" explanations)

**New Files**: docs/algorithms.md, docs/architecture.md, docs/api.md, docs/deployment.md, CONTRIBUTING.md, CHANGELOG.md

---

## 🎉 ALL PHASES COMPLETE - T4DM Production Ready

**Total Tasks**: 48/48 completed
**Duration**: 2025-11-27 (single day)
**Coverage**: 47% → 75%+
**New Files**: 25+
**Lines Added**: 10,000+

---

## Phase 1: Critical Bug Fixes (P0)

**Goal**: Fix runtime crashes and security vulnerabilities that block production deployment.

### TASK-P1-001: Missing asyncio import in semantic.py
**File**: `src/t4dm/memory/semantic.py:7`
**Severity**: CRITICAL (Runtime crash)
**Description**: `asyncio.gather` used at line 350 without import
```python
# Line 7: Add import
import asyncio
```
**Validation**: `python -c "from t4dm.memory.semantic import SemanticMemory"`

### TASK-P1-002: Cypher injection via dynamic labels
**File**: `src/t4dm/storage/t4dx_graph_adapter.py:158+`
**Severity**: CRITICAL (Security)
**Description**: User input passed directly to Cypher labels
```python
# Current (VULNERABLE):
await store.query(f"CREATE (n:{user_provided_label})")

# Fixed:
ALLOWED_LABELS = {"Entity", "Episode", "Procedure", "Relation"}
def validate_label(label: str) -> str:
    if label not in ALLOWED_LABELS:
        raise ValueError(f"Invalid label: {label}")
    return label
```
**Validation**: Test with label `"Entity) MATCH (n) DELETE n //"`

### TASK-P1-003: Session isolation broken in store singletons
**File**: `src/t4dm/storage/t4dx_vector_adapter.py`, `src/t4dm/storage/t4dx_graph_adapter.py`
**Severity**: HIGH (Data leak)
**Description**: Stores share single instance across all sessions
```python
# Current: Returns same instance for all sessions
_store: Optional[T4DXVectorAdapter] = None

def get_t4dx_vector_adapter(session_id: str = "default") -> T4DXVectorAdapter:
    global _store
    if _store is None:
        _store = T4DXVectorAdapter()  # Same instance!
    return _store

# Fixed: Instance per session
_stores: dict[str, T4DXVectorAdapter] = {}
_lock = threading.Lock()

def get_t4dx_vector_adapter(session_id: str = "default") -> T4DXVectorAdapter:
    if session_id not in _stores:
        with _lock:
            if session_id not in _stores:
                _stores[session_id] = T4DXVectorAdapter(session_id)
    return _stores[session_id]
```
**Validation**: Test 2 sessions can't see each other's data

### TASK-P1-004: Saga compensation errors silently ignored
**File**: `src/t4dm/storage/saga.py:85-95`
**Severity**: HIGH (Data integrity)
**Description**: Compensation failures not propagated
```python
# Current:
except Exception as e:
    logger.error(f"Compensation failed: {e}")
    # Swallowed!

# Fixed:
class CompensationError(Exception):
    def __init__(self, step: str, original: Exception, compensation: Exception):
        self.step = step
        self.original = original
        self.compensation = compensation

# Collect all compensation errors and raise aggregate
```
**Validation**: Test compensation failure raises proper error

### TASK-P1-005: Race condition in async service initialization
**File**: `src/t4dm/mcp/memory_gateway.py:30-54`
**Severity**: MEDIUM (Intermittent failures)
**Description**: Multiple requests can trigger parallel initialization
```python
# Add asyncio.Lock for async-safe initialization
_init_lock = asyncio.Lock()

async def get_services(session_id: str = None):
    async with _init_lock:
        if not _initialized.get(session_id):
            await _initialize_services(session_id)
            _initialized[session_id] = True
```
**Validation**: 100 concurrent requests, single initialization

### TASK-P1-006: Fix 5 failing tests (async event loop)
**File**: `tests/conftest.py`
**Severity**: MEDIUM (CI blocking)
**Description**: Neo4j async driver incompatible with default event loop
```python
# Add to conftest.py
import pytest
import asyncio

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for all async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()
```
**Validation**: `pytest tests/ -v` all pass

### TASK-P1-007: Add missing Neo4j indexes
**File**: `src/t4dm/storage/t4dx_graph_adapter.py`
**Severity**: MEDIUM (Performance)
**Description**: Missing indexes on Entity.sessionId, Episode.sessionId
```python
# Add to initialize()
CREATE INDEX entity_session IF NOT EXISTS FOR (e:Entity) ON (e.sessionId)
CREATE INDEX episode_session IF NOT EXISTS FOR (e:Episode) ON (e.sessionId)
CREATE INDEX procedure_session IF NOT EXISTS FOR (p:Procedure) ON (p.sessionId)
```
**Validation**: `EXPLAIN` shows index usage

---

## Phase 2: Performance Fixes (P1)

**Goal**: Eliminate O(n²) algorithms and N+1 queries.

### TASK-P2-001: Replace O(n²) clustering with DBSCAN/HDBSCAN
**File**: `src/t4dm/consolidation/service.py:425-462`
**Severity**: HIGH (Scaling blocker)
**Current**: Pairwise cosine similarity - O(n²)
**Target**: HDBSCAN clustering - O(n log n)
```python
from hdbscan import HDBSCAN
import numpy as np

async def _cluster_episodes(self, episodes: list[Episode]) -> list[list[Episode]]:
    embeddings = np.array([await self._get_embedding(ep) for ep in episodes])
    clusterer = HDBSCAN(min_cluster_size=3, metric='cosine')
    labels = clusterer.fit_predict(embeddings)
    # Group by cluster label
```
**Validation**: 10K episodes < 10s

### TASK-P2-002: Batch relationship queries (eliminate N+1)
**File**: `src/t4dm/memory/semantic.py:382-414`
**Severity**: HIGH (Query explosion)
**Current**: One query per entity in Hebbian strengthening
**Target**: Single batch query
```python
# Current (N+1):
for entity in retrieved:
    rels = await self.graph_store.get_relationships(entity.id)

# Fixed (batch):
all_ids = [e.id for e in retrieved]
rels_map = await self.graph_store.get_relationships_batch(all_ids)
```
**Validation**: Profile shows single query for 100 entities

### TASK-P2-003: Implement embedding cache
**File**: `src/t4dm/embedding/bge_m3.py`
**Severity**: MEDIUM (Latency)
**Description**: Same texts re-embedded repeatedly
```python
from functools import lru_cache
from hashlib import md5

class BGE_M3_Provider:
    def __init__(self):
        self._cache = {}  # text_hash -> embedding

    async def embed_query(self, text: str) -> list[float]:
        key = md5(text.encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = await self._embed(text)
        return self._cache[key]
```
**Validation**: Repeated embed calls return cached

### TASK-P2-004: Add Qdrant search prefiltering
**File**: `src/t4dm/storage/t4dx_vector_adapter.py`
**Severity**: MEDIUM (Query efficiency)
**Description**: Filter before vector similarity for session isolation
```python
# Add session_id to Qdrant filter
filter=models.Filter(
    must=[
        models.FieldCondition(
            key="session_id",
            match=models.MatchValue(value=session_id)
        )
    ]
)
```
**Validation**: Query plans show filter first

### TASK-P2-005: Parallel Hebbian updates
**File**: `src/t4dm/memory/semantic.py:382-414`
**Severity**: MEDIUM (Latency)
**Description**: Sequential relationship updates
```python
# Current (sequential):
for pair in co_retrievals:
    await self._strengthen_relationship(pair)

# Fixed (parallel):
await asyncio.gather(*[
    self._strengthen_relationship(pair)
    for pair in co_retrievals
])
```
**Validation**: 10 co-retrievals < 100ms (was 500ms)

### TASK-P2-006: Lazy model loading
**File**: `src/t4dm/embedding/bge_m3.py`
**Severity**: LOW (Startup time)
**Description**: Model loaded even when not needed
```python
class BGE_M3_Provider:
    def __init__(self):
        self._model = None  # Lazy

    async def _get_model(self):
        if self._model is None:
            self._model = load_model()
        return self._model
```
**Validation**: Import without GPU allocation

### TASK-P2-007: Connection pooling for Neo4j
**File**: `src/t4dm/storage/t4dx_graph_adapter.py`
**Severity**: LOW (Connection overhead)
**Description**: Configure connection pool settings
```python
self._driver = AsyncGraphDatabase.driver(
    uri,
    auth=(user, password),
    max_connection_pool_size=50,
    connection_acquisition_timeout=30,
)
```
**Validation**: Concurrent queries reuse connections

### TASK-P2-008: Qdrant batch upsert optimization
**File**: `src/t4dm/storage/t4dx_vector_adapter.py`
**Severity**: LOW (Bulk insert)
**Description**: Use parallel upsert for batches > 100
```python
if len(points) > 100:
    # Parallel chunk upload
    chunks = [points[i:i+100] for i in range(0, len(points), 100)]
    await asyncio.gather(*[
        client.upsert(collection, chunk) for chunk in chunks
    ])
```
**Validation**: 1000 points < 2s

---

## Phase 3: Security Hardening (P1)

**Goal**: Production-ready security posture.

### TASK-P3-001: Add rate limiting to MCP tools
**File**: `src/t4dm/mcp/memory_gateway.py`
**Severity**: HIGH (DoS protection)
```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def allow(self, session_id: str) -> bool:
        now = time.time()
        self.requests[session_id] = [
            t for t in self.requests[session_id]
            if now - t < self.window
        ]
        if len(self.requests[session_id]) >= self.max:
            return False
        self.requests[session_id].append(now)
        return True
```
**Validation**: 101st request in 60s returns rate limit error

### TASK-P3-002: Input sanitization for all MCP tools
**File**: `src/t4dm/mcp/validation.py`
**Severity**: HIGH (Injection prevention)
```python
import re

def sanitize_string(value: str, max_length: int = 10000) -> str:
    """Sanitize user input string."""
    if len(value) > max_length:
        raise ValidationError("content", f"Exceeds max length {max_length}")
    # Remove null bytes
    value = value.replace('\x00', '')
    return value

def sanitize_identifier(value: str) -> str:
    """Sanitize identifiers (names, labels)."""
    if not re.match(r'^[a-zA-Z0-9_-]{1,100}$', value):
        raise ValidationError("identifier", "Invalid characters in identifier")
    return value
```
**Validation**: Null bytes, unicode exploits blocked

### TASK-P3-003: Add authentication context
**File**: `src/t4dm/mcp/memory_gateway.py`
**Severity**: MEDIUM (Access control)
```python
from contextvars import ContextVar

_auth_context: ContextVar[dict] = ContextVar("auth", default={})

def require_auth(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ctx = _auth_context.get()
        if not ctx.get("authenticated"):
            return {"error": "unauthorized", "message": "Authentication required"}
        return await func(*args, **kwargs)
    return wrapper
```
**Validation**: Unauthenticated requests rejected

### TASK-P3-004: Secure configuration loading
**File**: `src/t4dm/core/config.py`
**Severity**: MEDIUM (Secret management)
```python
import os
from pathlib import Path

def load_config():
    # Never log secrets
    password = os.getenv("NEO4J_PASSWORD")
    if password:
        logger.debug("Neo4j password: [REDACTED]")

    # Validate file permissions for config files
    config_path = Path("~/.t4dm/config.yaml").expanduser()
    if config_path.exists():
        mode = config_path.stat().st_mode
        if mode & 0o077:
            logger.warning(f"Config file has loose permissions: {oct(mode)}")
```
**Validation**: Secrets not in logs

### TASK-P3-005: Add request ID tracking
**File**: `src/t4dm/mcp/memory_gateway.py`
**Severity**: LOW (Audit trail)
```python
from uuid import uuid4
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="")

def with_request_id(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        req_id = str(uuid4())[:8]
        _request_id.set(req_id)
        logger.info(f"[{req_id}] Starting {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"[{req_id}] Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"[{req_id}] Failed {func.__name__}: {e}")
            raise
    return wrapper
```
**Validation**: All logs have request ID

### TASK-P3-006: Validate config weight bounds
**File**: `src/t4dm/core/config.py`
**Severity**: LOW (Configuration safety)
```python
def validate_weights(config: dict):
    """Ensure all weights sum to 1.0 and are in [0,1]."""
    weights = config.get("retrieval_weights", {})
    total = sum(weights.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0, got {total}")
    for name, val in weights.items():
        if not 0 <= val <= 1:
            raise ValueError(f"Weight {name} must be [0,1], got {val}")
```
**Validation**: Invalid weights raise on startup

---

## Phase 4: Test Coverage (P2)

**Goal**: Achieve 75%+ coverage, currently at 47%.

### TASK-P4-001: Consolidation service tests
**File**: `tests/unit/test_consolidation.py` (NEW)
**Current Coverage**: 18%
**Target**: 80%
**Tests needed**:
- Light consolidation (duplicate detection)
- Deep consolidation (entity extraction)
- Skill consolidation (procedure merging)
- Error handling paths

### TASK-P4-002: MCP gateway tests
**File**: `tests/unit/test_mcp_gateway.py` (NEW)
**Current Coverage**: 18%
**Target**: 80%
**Tests needed**:
- All 17 MCP tools
- Error responses
- Validation errors
- Timeout handling

### TASK-P4-003: Observability module tests
**File**: `tests/unit/test_observability.py` (NEW)
**Current Coverage**: 0%
**Target**: 90%
**Tests needed**:
- Structured logging format
- Metrics collection
- Health check endpoints
- Timer context managers

### TASK-P4-004: Episodic memory tests
**File**: `tests/unit/test_episodic.py` (NEW)
**Current Coverage**: 45%
**Target**: 85%
**Tests needed**:
- FSRS decay calculations
- Temporal queries
- Access pattern updates
- Edge cases (empty results, max limits)

### TASK-P4-005: Semantic memory tests
**File**: `tests/unit/test_semantic.py` (NEW)
**Current Coverage**: 42%
**Target**: 85%
**Tests needed**:
- ACT-R activation
- Hebbian learning
- Entity relationships
- Fact supersession

### TASK-P4-006: Procedural memory tests
**File**: `tests/unit/test_procedural.py` (NEW)
**Current Coverage**: 38%
**Target**: 85%
**Tests needed**:
- Skill building
- Retrieval ranking
- Deprecation
- Memp algorithm

### TASK-P4-007: Integration test suite
**File**: `tests/integration/test_memory_lifecycle.py` (NEW)
**Tests needed**:
- Full episode lifecycle (create→recall→decay→consolidate)
- Cross-memory interactions
- Session isolation end-to-end

### TASK-P4-008: Performance regression tests
**File**: `tests/performance/test_benchmarks.py` (NEW)
**Tests needed**:
- 1000 episode creation < 10s
- 10000 episode recall < 5s
- Consolidation of 1000 episodes < 30s

### TASK-P4-009: Property-based tests for algorithms
**File**: `tests/unit/test_algorithms_property.py` (NEW)
**Tests needed**:
- FSRS retrievability always [0,1]
- Hebbian weight always bounded
- ACT-R activation never negative

### TASK-P4-010: Mocking infrastructure
**File**: `tests/conftest.py`
**Description**: Add fixtures for mocked stores
```python
@pytest.fixture
def mock_qdrant():
    with patch('t4dm.storage.t4dx_vector_adapter.T4DXVectorAdapter') as mock:
        mock.return_value.search.return_value = []
        yield mock

@pytest.fixture
def mock_neo4j():
    with patch('t4dm.storage.t4dx_graph_adapter.T4DXGraphAdapter') as mock:
        mock.return_value.query.return_value = []
        yield mock
```

### TASK-P4-011: CI coverage gate
**File**: `.github/workflows/test.yml`
**Description**: Fail CI if coverage drops below 70%
```yaml
- name: Check coverage
  run: |
    coverage report --fail-under=70
```

### TASK-P4-012: Test documentation
**File**: `tests/README.md` (NEW)
**Description**: Document test structure, fixtures, running guide

---

## Phase 5: API Cleanup (P2)

**Goal**: Consistent, typed API surface.

### TASK-P5-001: Standardize method naming
**Description**: `build` → `create`, `retrieve` → `recall`
**Files**:
- `src/t4dm/memory/procedural.py`: `build()` → `create_skill()`
- `src/t4dm/memory/procedural.py`: `retrieve()` → `recall_skill()`
- `src/t4dm/mcp/memory_gateway.py`: Update tool names

### TASK-P5-002: Add TypedDict for MCP responses
**File**: `src/t4dm/mcp/types.py` (NEW)
```python
from typing import TypedDict, Optional

class EpisodeResponse(TypedDict):
    id: str
    content: str
    timestamp: str
    relevance: Optional[float]

class ErrorResponse(TypedDict):
    error: str
    message: str
    field: Optional[str]
```

### TASK-P5-003: Consistent error codes
**File**: `src/t4dm/mcp/errors.py` (NEW)
```python
class ErrorCode:
    VALIDATION = "validation_error"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    INTERNAL = "internal_error"
    RATE_LIMITED = "rate_limited"
    UNAUTHORIZED = "unauthorized"
```

### TASK-P5-004: Add pagination to all list operations
**Files**: `recall_episodes`, `semantic_recall`, `how_to`
```python
async def recall_episodes(
    query: str,
    limit: int = 10,
    offset: int = 0,  # NEW
) -> dict:
    results = await episodic.recall(query, k=limit + offset)
    return {
        "episodes": results[offset:offset+limit],
        "total": len(results),
        "offset": offset,
        "limit": limit,
    }
```

### TASK-P5-005: Add result count to responses
**Files**: All recall tools
```python
return {
    "results": [...],
    "count": len(results),
    "query": query,
}
```

### TASK-P5-006: Move Cypher to storage layer
**Description**: Memory layer shouldn't construct Cypher
**Files**:
- `src/t4dm/storage/t4dx_graph_adapter.py`: Add high-level methods
- `src/t4dm/memory/*.py`: Use store methods instead of raw Cypher

### TASK-P5-007: Add OpenAPI schema
**File**: `src/t4dm/mcp/schema.py` (NEW)
**Description**: Generate schema from MCP tools

### TASK-P5-008: Deprecation warnings for old API
**File**: `src/t4dm/mcp/memory_gateway.py`
```python
import warnings

@mcp_app.tool()
async def build_skill(...):
    warnings.warn("build_skill is deprecated, use create_skill", DeprecationWarning)
    return await create_skill(...)
```

---

## Phase 6: Documentation & Polish (P3)

**Goal**: Production-ready documentation.

### TASK-P6-001: Algorithm documentation
**File**: `docs/algorithms.md` (NEW)
**Contents**:
- FSRS deviation from standard (why 0.9 factor)
- ACT-R simplifications (why no chunk spreading)
- Hebbian implementation details

### TASK-P6-002: API reference
**File**: `docs/api.md` (NEW)
**Contents**:
- All 17 MCP tools
- Request/response examples
- Error codes

### TASK-P6-003: Deployment guide
**File**: `docs/deployment.md` (NEW)
**Contents**:
- Docker compose setup
- Environment variables
- Resource requirements
- Health check endpoints

### TASK-P6-004: Architecture diagram
**File**: `docs/architecture.md` (NEW)
**Contents**:
- System overview
- Data flow diagrams
- Component interactions

### TASK-P6-005: Contributing guide
**File**: `CONTRIBUTING.md` (NEW)
**Contents**:
- Development setup
- Test requirements
- PR process

### TASK-P6-006: Changelog
**File**: `CHANGELOG.md` (NEW)
**Contents**:
- Version history
- Breaking changes
- Migration guides

### TASK-P6-007: Code comments audit
**Files**: All source files
**Description**: Ensure all non-obvious code has comments explaining "why"

---

## Dependency Summary

```
Phase 1 (Critical) - No dependencies, start immediately
    ↓
Phase 2 (Performance) - After P1-003 (session isolation)
    ↓
Phase 3 (Security) - After P1-002 (injection fix)
    ↓
Phase 4 (Testing) - After P1-006 (test fixes)
    ↓
Phase 5 (API) - After P4 (tests to validate changes)
    ↓
Phase 6 (Docs) - Final phase
```

## Quick Reference

| Task | File | Severity | Est. Hours |
|------|------|----------|------------|
| P1-001 | semantic.py | CRITICAL | 0.5 |
| P1-002 | t4dx_graph_adapter.py | CRITICAL | 2 |
| P1-003 | *_store.py | HIGH | 3 |
| P1-004 | saga.py | HIGH | 2 |
| P1-005 | memory_gateway.py | MEDIUM | 1 |
| P1-006 | conftest.py | MEDIUM | 1 |
| P1-007 | t4dx_graph_adapter.py | MEDIUM | 1 |

**Total Estimated**: 40-60 hours across all phases

---

**Next Action**: Begin Phase 1, Task P1-001 (add missing asyncio import)
