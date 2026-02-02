# T4DM Architecture Review

**Review Date**: 2026-01-07
**Version**: 0.5.0
**Codebase Size**: 242 Python files, ~136,000 LOC
**Test Coverage**: 80% (288 test files, 7,970+ tests passing)

---

## Executive Summary

### Overall Architecture Scores

| Category | Score | Grade |
|----------|-------|-------|
| **Modularity** | 8.5/10 | A |
| **Maintainability** | 7.5/10 | B+ |
| **Scalability** | 7.0/10 | B |
| **Code Quality** | 8.0/10 | A- |
| **Production Readiness** | 7.5/10 | B+ |
| **Overall** | 7.7/10 | B+ |

**Verdict**: Well-architected system with strong separation of concerns, excellent protocol-based abstractions, and robust testing. Production-ready with identified areas for optimization.

---

## 1. Module Structure & Dependencies

### Architecture Overview

```
src/t4dm/
├── core/           # Domain models, protocols, config (foundation layer)
├── storage/        # Neo4j, Qdrant persistence (data layer)
├── memory/         # Episodic, Semantic, Procedural subsystems (domain layer)
├── nca/            # Neural Cellular Automata (26 modules, biological layer)
├── learning/       # Plasticity, dopamine, consolidation (learning layer)
├── api/            # REST API with FastAPI (presentation layer)
├── sdk/            # Python client library (SDK layer)
├── mcp/            # Model Context Protocol server (integration layer)
├── hooks/          # Extensibility system (plugin layer)
├── consolidation/  # Memory consolidation services (background layer)
├── visualization/  # Telemetry and dashboards (observability layer)
└── cli/            # Command-line interface (UI layer)
```

### Dependency Analysis

**Total Modules**: 143 Python modules
**Import Depth**: Core → Services → API (proper layering)
**Circular Dependencies**: None detected ✓

**Most Complex Modules** (by import count):
1. `nca/__init__.py` - 26 imports (aggregation module, expected)
2. `memory/episodic.py` - 25 imports (high coupling - see concerns)
3. `learning/__init__.py` - 23 imports (aggregation module, expected)
4. `visualization/__init__.py` - 19 imports (aggregation module)
5. `api/routes/__init__.py` - 13 imports (router aggregation)

**Strengths**:
- Clear layered architecture with unidirectional dependencies
- Protocol-based abstractions in `core/protocols.py` enable pluggable backends
- No circular dependencies between modules
- Consistent use of factory functions (`get_*`, `create_*`)

**Concerns**:
- `memory/episodic.py` has 25 direct imports - consider splitting
- Bridge modules (`bridges/`, `bridge/`) create coupling between memory and NCA subsystems
- Large `__init__.py` files in `nca/` and `learning/` expose 100+ symbols

**Recommendations**:
1. Split `memory/episodic.py` (3,616 LOC) into smaller components:
   - `episodic_store.py` - Storage operations
   - `episodic_scoring.py` - Retrieval ranking
   - `episodic_decay.py` - FSRS decay logic
2. Create explicit bridge interface in `core/protocols.py` for memory-NCA integration
3. Consider facade pattern for `nca/__init__.py` to reduce surface area

---

## 2. Design Patterns

### Protocol/Interface Usage

**Score**: 9/10 ✓

- Excellent use of `typing.Protocol` for runtime-checkable interfaces
- Core protocols defined in `core/protocols.py`:
  - `EmbeddingProvider` - Pluggable embedding models
  - `VectorStore` - Database abstraction (Qdrant, FAISS, etc.)
  - `GraphStore` - Graph database abstraction (Neo4j)
  - `EpisodicStore`, `SemanticStore`, `ProceduralStore` - Memory subsystem interfaces

**Example**:
```python
@runtime_checkable
class VectorStore(Protocol):
    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        ...
```

This enables:
- Testing with mock implementations
- Easy backend swapping (e.g., Qdrant → Milvus)
- Clear contracts for third-party integrations

### Dependency Injection

**Score**: 7/10

- Custom DI container in `core/container.py` with:
  - Singleton registration (lazy instantiation)
  - Scoped services (per-session isolation)
  - Factory pattern for service creation
- Factory functions use `@lru_cache` for singleton pattern (116 occurrences)
- 785 factory/getter functions (`get_*()`, `create_*()`)

**Strengths**:
- Thread-safe singleton initialization with double-checked locking
- Scoped services support session isolation
- Decorator-based injection: `@inject("service_key")`

**Weaknesses**:
- DI container not consistently used - many modules use module-level globals
- `@lru_cache` singletons harder to reset in tests than container-managed
- No automatic dependency resolution (must manually wire in `configure_production()`)

**Recommendation**:
Standardize on DI container for all stateful services. Migrate from:
```python
@lru_cache(maxsize=1)
def get_t4dx_vector_adapter() -> T4DXVectorAdapter:
    return T4DXVectorAdapter()
```

To:
```python
def get_t4dx_vector_adapter() -> T4DXVectorAdapter:
    return get_container().resolve("t4dx_vector_adapter")
```

### Repository Pattern

**Score**: 8/10 ✓

- Clean separation between domain models (`core/types.py`) and storage (`storage/`)
- Storage abstractions:
  - `T4DXVectorAdapter` - Vector operations
  - `T4DXGraphAdapter` - Graph operations
  - `EpisodicMemory`, `SemanticMemory`, `ProceduralMemory` - Domain repositories

**Strengths**:
- Storage layer hidden behind protocols
- Saga pattern in `storage/saga.py` for distributed transactions
- Circuit breaker pattern in `storage/resilience.py` for fault tolerance

**Weaknesses**:
- Memory subsystems (`memory/*.py`) directly instantiate storage classes
- No repository interface abstraction - uses concrete classes

### Observer/Event Pattern

**Score**: 6/10

- Hooks system in `hooks/` for extensibility:
  - `PreStoreHook`, `PostStoreHook`, `OnRecallHook`
  - Registry pattern for dynamic hook registration
- WebSocket support for real-time events (`api/websocket.py`)

**Weaknesses**:
- Hooks are synchronous - blocks main operation
- No event bus for decoupled communication
- Limited use of async observers

**Recommendation**:
Implement async event system:
```python
class EventBus:
    async def publish(self, event: Event) -> None:
        await asyncio.gather(*[h.handle(event) for h in self._handlers])
```

---

## 3. API Design

### REST API Organization

**Score**: 8/10 ✓

**Route Structure**:
```
/api/v1/
├── episodes/           # Episodic memory CRUD
├── entities/           # Semantic memory entities
├── skills/             # Procedural memory skills
├── config/             # Configuration management
├── health, /stats      # System monitoring
├── viz/                # Visualization data
└── demo/               # Phase 11 interactive demos
    ├── explorer/       # Memory graph explorer
    ├── dream/          # Dream trajectory viewer
    ├── nt/             # Neurotransmitter dashboard
    └── learning/       # Learning trace viewer
```

**Strengths**:
- RESTful resource naming
- Consistent versioning (`/api/v1/`)
- OpenAPI documentation auto-generated
- Health check endpoint for load balancers
- Proper HTTP status codes (404, 429, 503)

**Weaknesses**:
- Demo routes (`/api/v1/demo/*`) should be in separate service
- Large route handlers (200+ LOC in some endpoints)
- Missing rate limiting on non-demo routes
- No pagination links in list responses (HATEOAS)

**Security**:
- API key authentication middleware (`ApiKeyAuthMiddleware`)
- CORS configuration with validated origins
- Security headers (XSS protection, HSTS, CSP)
- Request size limiting (5MB max)
- Cypher injection prevention via whitelisting

**Async/Await Consistency**: ✓ All route handlers are async

### SDK Client Design

**Score**: 9/10 ✓

**Client Classes**:
- `AsyncT4DMClient` - Async httpx client
- `T4DMClient` - Synchronous wrapper

**Strengths**:
- Context manager support (`async with` / `with`)
- Pydantic models for type safety (`sdk/models.py`)
- Custom exceptions (`T4DMError`, `NotFoundError`, `RateLimitError`)
- Session ID propagation via headers
- Timeout configuration

**Example**:
```python
async with AsyncT4DMClient(session_id="user-123") as ww:
    episode = await ww.create_episode("Learned about async/await")
    results = await ww.recall_episodes("async patterns")
```

**Weaknesses**:
- Sync client wraps async client (introduces overhead)
- No retry logic for transient failures
- No connection pooling configuration exposed

### MCP Server Design

**Score**: 7/10

- Model Context Protocol server in `mcp/server.py`
- Tools for memory operations exposed to LLMs
- JSON-RPC over stdio

**Weaknesses**:
- Limited tool set (only basic memory operations)
- No streaming support for long responses
- MCP spec compliance not tested

---

## 4. Code Quality Metrics

### Function/Class Sizes

**Largest Classes** (by method count):
1. `NeuralFieldSolver` - 34 methods (acceptable for PDE solver)
2. `StabilityMonitor` - 32 methods (too large, split into analyzers)
3. `LocusCoeruleus` - 32 methods (biological model complexity)
4. `TelemetryHub` - 30 methods (needs refactoring)

**Largest Functions** (by line count):
1. `create_ww_router` - 429 lines (CRITICAL - split into modules)
2. `__init__` (EpisodicMemory) - 247 lines (constructor doing too much)
3. `reconsolidate` - 165 lines (complex algorithm, acceptable)
4. `__init__` (SleepConsolidator) - 150 lines (refactor needed)

**Code Smells**:
- 232 `print()` statements (should use `logger`)
- 27 bare `except Exception:` handlers (overly broad)
- 0 star imports (good)
- 0 TODO/FIXME markers (technical debt managed externally)

**Recommendations**:
1. **PRIORITY**: Refactor `create_ww_router` (429 lines) into route groups
2. Reduce `EpisodicMemory.__init__` by moving setup to factory functions
3. Replace `print()` with structured logging
4. Replace bare `except Exception:` with specific exception types

### Cyclomatic Complexity

**Note**: `radon` not available in environment, unable to measure. Based on manual review:

**Complex Functions** (visual inspection):
- `EpisodicMemory.recall()` - Multiple filters and scoring paths
- `SemanticMemory.spread_activation()` - Recursive graph traversal
- `SleepConsolidator.consolidate()` - Multi-phase state machine

**Recommendation**: Add complexity linting to CI:
```bash
radon cc src/ww --min B --show-complexity
```

### Documentation Coverage

**Metric**: ~85% of public functions have docstrings

**Strengths**:
- Comprehensive module-level docstrings
- Type hints on all public APIs
- Examples in SDK client docstrings

**Weaknesses**:
- Internal functions often lack docstrings
- No standardized format (Google/NumPy style)
- Complex algorithms lack implementation notes

### Test Coverage Distribution

**Overall**: 80% coverage (good)

**Well-Tested Modules** (>80%):
- `core/` - 90%+ (excellent)
- `storage/` - 85%+ (good)
- `memory/` - 80%+ (good)
- `nca/` - 75%+ (acceptable for research code)

**Under-Tested Modules** (<60%):
- `api/routes/visualization.py` - 34% (Phase 11 demos)
- `api/server.py` - 34% (startup/shutdown)
- `bridges/*` - 30-50% (integration layer)
- `interfaces/*` - Low coverage (optional UI)

**Test Organization**:
- 288 test files across 60+ directories
- Good separation: `unit/`, `integration/`, `e2e/`
- Fixtures in `conftest.py` (5 files)
- Hypothesis property tests present

**Recommendations**:
1. Increase `bridges/` coverage to 70% (integration critical)
2. Add startup/shutdown integration tests
3. Skip demo route coverage (ephemeral code)

---

## 5. Production Readiness

### Error Handling

**Score**: 7/10

**Strengths**:
- Custom exception hierarchy (8 exception classes)
- Circuit breaker pattern for database failures
- Timeout handling on database operations
- Retry logic in consolidation service

**Weaknesses**:
- 27 bare `except Exception:` handlers
- Inconsistent error responses (some return 500, others 400 for same issue)
- Missing exception context propagation

**Example of Good Error Handling**:
```python
async def _with_timeout(self, coro, operation: str):
    try:
        async with asyncio.timeout(self.timeout):
            return await coro
    except TimeoutError:
        logger.error(f"Timeout in operation '{operation}' after {self.timeout}s")
        raise DatabaseTimeoutError(operation, self.timeout)
```

**Recommendations**:
1. Replace bare `except Exception:` with specific types
2. Add error code enumeration for API errors
3. Implement structured error responses:
   ```json
   {
     "error": {
       "code": "MEMORY_NOT_FOUND",
       "message": "Episode not found",
       "details": {"episode_id": "..."}
     }
   }
   ```

### Logging Consistency

**Score**: 8/10 ✓

- 1,384 logging statements (good coverage)
- Structured logging with levels:
  - `logger.debug()` - Trace information
  - `logger.info()` - Normal operations
  - `logger.warning()` - Degraded state
  - `logger.error()` - Failures
  - `logger.critical()` - Security violations

**Strengths**:
- Contextual logging with operation details
- Sensitive data masking (`mask_secret()`)
- Correlation IDs via OpenTelemetry spans

**Weaknesses**:
- 232 `print()` statements (should be `logger.info()`)
- No structured logging format (JSON logging)
- Log levels not configurable per module

**Recommendation**: Enable JSON logging for production:
```python
logging.basicConfig(
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
    handlers=[JSONHandler()]
)
```

### Configuration Management

**Score**: 9/10 ✓

**Excellent Implementation**:
- Pydantic Settings for validation
- Environment variable support (12-factor app)
- Strong password validation (12 chars, 3 character classes)
- Secure defaults
- File permission enforcement

**Example**:
```python
class Settings(BaseSettings):
    neo4j_uri: str = Field(..., description="Neo4j URI")
    neo4j_password: str = Field(..., description="Neo4j password")

    @field_validator("neo4j_password")
    def validate_password(cls, v: str) -> str:
        return validate_password_strength(v, "neo4j_password")
```

**Security Features**:
- Password complexity enforcement
- Weak password blacklist
- Secret masking in logs
- File permission checks (0o600 for secrets)
- CORS origin validation (no wildcards in production)

**Only Weakness**: No centralized config service for distributed deployments

### Security Considerations

**Score**: 8/10 ✓

**Strengths**:
1. **Input Validation**:
   - Cypher injection prevention via whitelisting
   - UUID type validation
   - Session ID sanitization
   - Property name validation

2. **Authentication**:
   - API key middleware
   - Auto-enabled in production
   - Constant-time comparison (`secrets.compare_digest`)

3. **Transport Security**:
   - HSTS headers for HTTPS
   - XSS protection headers
   - Content Security Policy
   - Frame-Options: DENY

4. **Data Protection**:
   - Sensitive field filtering (`PrivacyFilter`)
   - Secret detection (`detect-secrets` in CI)
   - Secure password storage (delegated to databases)

**Weaknesses**:
1. No rate limiting on critical routes (only middleware structure exists)
2. Missing request signing for SDK client
3. No audit logging for admin operations
4. Session tokens not rotated

**Recommendations**:
1. Implement per-endpoint rate limiting
2. Add JWT-based authentication for SDK
3. Create audit log table for compliance

---

## 6. Scalability Concerns

### Stateful vs Stateless Components

**Stateless** (Scalable):
- API server (FastAPI)
- SDK client
- MCP server

**Stateful** (Scaling challenges):
- Memory consolidation service (background worker)
- Persistence manager (checkpoint state)
- WebSocket connections (session affinity needed)
- DI container (per-process state)

**Recommendation**:
- Move consolidation to separate service (Celery worker)
- Use Redis for WebSocket session state
- Implement sticky sessions for WebSocket routes

### Database Access Patterns

**Score**: 7/10

**Strengths**:
- Connection pooling via Neo4j driver
- Async operations throughout
- Circuit breakers prevent cascading failures
- Batch operations for bulk inserts

**Weaknesses**:
- N+1 query problem in spreading activation
- No query result caching
- Missing database read replicas support
- No prepared statement reuse

**Example N+1 Issue**:
```python
# Current: N+1 queries
for entity_id in entity_ids:
    entity = await self.get_entity(entity_id)  # Separate query

# Better: Single batch query
entities = await self.get_entities_batch(entity_ids)
```

**Recommendations**:
1. Add Redis caching layer for hot entities
2. Implement batch query APIs
3. Support read replicas for graph queries

### Caching Strategies

**Score**: 5/10 (Needs Improvement)

**Current State**:
- Embedding cache in `FeatureAligner` (in-memory, per-process)
- No distributed caching
- No cache invalidation strategy
- TTL-based eviction only

**Missing**:
- Redis/Memcached integration
- Cache warming strategies
- Cache stampede prevention
- Hierarchical caching (L1/L2)

**Recommendation**: Implement tiered caching:
```python
# L1: Process cache (100ms TTL)
# L2: Redis cache (1 hour TTL)
# L3: Database
async def get_entity_cached(entity_id: UUID) -> Entity:
    if entity := l1_cache.get(entity_id):
        return entity
    if entity := await redis.get(f"entity:{entity_id}"):
        l1_cache.set(entity_id, entity)
        return entity
    entity = await db.get_entity(entity_id)
    await redis.setex(f"entity:{entity_id}", 3600, entity)
    l1_cache.set(entity_id, entity)
    return entity
```

### Concurrency Handling

**Score**: 8/10 ✓

**Strengths**:
- Full async/await adoption (93 files with async functions)
- Thread-safe singleton initialization
- asyncio.Lock for critical sections
- Graceful shutdown with connection draining

**Example**:
```python
class RequestTrackingMiddleware:
    async def dispatch(self, request: Request, call_next):
        global _active_requests

        if _shutdown_event.is_set():
            return JSONResponse(status_code=503, ...)

        async with _active_requests_lock:
            _active_requests += 1
        try:
            return await call_next(request)
        finally:
            async with _active_requests_lock:
                _active_requests -= 1
```

**Weaknesses**:
- No semaphore limiting for concurrent DB operations
- Missing backpressure handling
- No connection pool exhaustion handling

**Recommendation**: Add connection pool limits:
```python
MAX_CONCURRENT_DB_OPS = 100
db_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DB_OPS)

async def db_operation():
    async with db_semaphore:
        return await db.query(...)
```

---

## 7. Anti-Patterns & Code Smells

### Identified Anti-Patterns

1. **God Object**: `EpisodicMemory` class (3,616 LOC, 25+ responsibilities)
   - **Impact**: Hard to test, maintain, extend
   - **Fix**: Extract scoring, decay, encoding into separate classes

2. **Feature Envy**: Bridge modules access NCA internals extensively
   - **Impact**: High coupling between subsystems
   - **Fix**: Define explicit bridge protocols

3. **Primitive Obsession**: Heavy use of `dict[str, Any]` for complex data
   - **Impact**: No type safety, runtime errors
   - **Fix**: Use Pydantic models consistently

4. **Magic Numbers**: Hardcoded thresholds throughout NCA modules
   - **Impact**: Difficult to tune, unclear meaning
   - **Fix**: Extract to configuration classes

5. **Long Method**: `create_ww_router()` at 429 lines
   - **Impact**: Unmaintainable, untestable
   - **Fix**: Split into route group factories

### Good Practices Observed

1. **Protocol-Based Design**: Excellent abstraction via typing.Protocol
2. **Saga Pattern**: Distributed transaction handling
3. **Circuit Breaker**: Fault tolerance in storage layer
4. **Factory Functions**: Consistent object creation
5. **Type Hints**: Comprehensive type annotations
6. **Async First**: Native async/await, no threadpool hacks
7. **Security**: Input validation, Cypher injection prevention
8. **Observability**: OpenTelemetry integration

---

## 8. Specific Recommendations (Prioritized by Impact)

### Critical (Do Now)

1. **Split `memory/episodic.py`** (3,616 LOC)
   - **Impact**: High - Core memory subsystem maintainability
   - **Effort**: 2-3 days
   - **Files**: Create `episodic_store.py`, `episodic_scoring.py`, `episodic_decay.py`

2. **Refactor `create_ww_router()`** (429 lines)
   - **Impact**: High - Integration endpoint maintainability
   - **Effort**: 1 day
   - **Files**: Split into `integration/routes/*.py`

3. **Replace print() with logger**
   - **Impact**: High - Production logging broken
   - **Effort**: 2-3 hours (automated)
   - **Command**: `ruff --select T201 --fix src/ww`

### High Priority (Next Sprint)

4. **Implement Redis caching layer**
   - **Impact**: High - 10-100x performance improvement for hot paths
   - **Effort**: 3-4 days
   - **Files**: Create `storage/redis_cache.py`, update memory subsystems

5. **Add rate limiting to API routes**
   - **Impact**: High - DoS prevention
   - **Effort**: 1 day
   - **Library**: `slowapi` or custom middleware

6. **Increase bridge test coverage to 70%**
   - **Impact**: Medium-High - Integration reliability
   - **Effort**: 2-3 days
   - **Files**: `tests/bridges/*`

7. **Extract demo routes to separate service**
   - **Impact**: Medium - Better separation of concerns
   - **Effort**: 2 days
   - **Deployment**: Docker Compose service for demos

### Medium Priority (This Quarter)

8. **Implement event bus for async observers**
   - **Impact**: Medium - Better extensibility
   - **Effort**: 3-4 days
   - **Files**: Create `core/event_bus.py`

9. **Add JSON structured logging**
   - **Impact**: Medium - Better log analysis
   - **Effort**: 1 day
   - **Library**: `structlog` or `python-json-logger`

10. **Standardize on DI container**
    - **Impact**: Medium - Better testability
    - **Effort**: 4-5 days
    - **Files**: Migrate all `@lru_cache` singletons

11. **Batch query optimization**
    - **Impact**: Medium - Reduce N+1 queries
    - **Effort**: 2-3 days
    - **Files**: Add batch methods to storage layer

### Low Priority (Nice to Have)

12. **Add HATEOAS links to API responses**
13. **Implement request signing for SDK**
14. **Create audit log table**
15. **Add Prometheus metrics for all routes**
16. **Implement cache warming strategies**

---

## 9. Architecture Evolution Path

### Short-term (1-2 months)
- Refactor large classes/functions
- Add Redis caching
- Improve test coverage to 85%
- Extract demo APIs

### Medium-term (3-6 months)
- Event-driven architecture with message bus
- Microservices split (API, Worker, Demo)
- Read replica support
- Advanced rate limiting

### Long-term (6-12 months)
- Multi-tenancy support
- Distributed tracing (Jaeger)
- GraphQL API option
- Kubernetes-native deployment

---

## 10. Conclusion

T4DM demonstrates **strong architectural fundamentals** with excellent separation of concerns, protocol-based abstractions, and comprehensive testing. The codebase is production-ready with the following caveats:

**Strengths**:
- Clean layered architecture
- Protocol-driven design enables extensibility
- Strong security posture (input validation, authentication)
- Good test coverage (80%)
- Async-first implementation

**Key Improvements Needed**:
1. Refactor large classes (`EpisodicMemory`, `TelemetryHub`)
2. Add caching layer for scalability
3. Improve logging (remove print statements)
4. Increase bridge test coverage
5. Implement rate limiting

**Overall Assessment**: **B+ Architecture** - Solid foundation with clear path to A-grade through targeted refactoring and infrastructure additions.

---

**Reviewed by**: ww-algorithm Agent
**Date**: 2026-01-07
**Next Review**: Q2 2026 (after implementing critical recommendations)
