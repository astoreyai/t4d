# Architecture Refactoring Quick Start

**Full Plan**: [ARCHITECTURE_REFACTORING_PLAN.md](/mnt/projects/ww/docs/ARCHITECTURE_REFACTORING_PLAN.md)

---

## TL;DR

Fix 7 critical architecture issues in 3 phases (5 weeks):

1. **Phase 1** (2 weeks): Split 3,616-line episodic.py into 6 focused modules
2. **Phase 2** (2 weeks): Add Redis caching + rate limiting (5-10x speedup)
3. **Phase 3** (1 week): Convert 232 print() to logger + bridge tests

**Zero breaking changes** - all 8,075 tests pass throughout.

---

## Critical Issues Fixed

| Issue | Current | Target | Impact |
|-------|---------|--------|--------|
| God Object (episodic.py) | 3,616 lines | 400 lines | Maintainability |
| Long Method (config router) | 429 lines | 150 lines | Readability |
| print() statements | 232 | 0 | Observability |
| Caching layer | None | Redis | 5x speedup |
| API rate limiting | None | 100-1000/min | Production ready |
| N+1 queries | O(N) | O(1) batch | 10-100x speedup |
| Bridge test coverage | 5 files | 20+ files | Reliability |

---

## Phase 1: Episodic Decomposition (Week 1-2)

**Goal**: Split God object into 6 service modules

### New Module Structure

```
src/ww/memory/
├── episodic.py              # 400 lines - Facade (backward compat)
├── episodic_storage.py      # 800 lines - Storage operations
├── episodic_retrieval.py    # 1,200 lines - Search & recall
├── episodic_learning.py     # 600 lines - Reconsolidation
├── episodic_fusion.py       # 400 lines - Learned fusion
└── episodic_saga.py         # 400 lines - Transaction coordination
```

### Quick Start

```bash
# 1. Create branch
git checkout -b refactor/phase1-episodic

# 2. Create new modules (templates provided in plan)
touch src/ww/memory/episodic_{storage,retrieval,learning,fusion,saga}.py

# 3. Extract classes (copy from episodic.py lines specified in plan)
# - episodic_fusion.py: Lines 64-474 (LearnedFusionWeights, LearnedReranker)
# - episodic_storage.py: Lines 950-1400, 2300-2500
# - episodic_retrieval.py: Lines 1200-2300
# - episodic_learning.py: Lines 2500-3100
# - episodic_saga.py: Lines 950-1200 (saga coordination)

# 4. Refactor episodic.py to facade (template in plan)

# 5. Run tests (MUST pass all existing tests)
pytest tests/memory/test_episodic*.py -v
pytest tests/ -v  # Full suite

# 6. Add new service tests
pytest tests/memory/test_episodic_{storage,retrieval,learning,integration}.py -v
```

### Success Criteria

- [ ] episodic.py ≤ 400 lines
- [ ] All 8,075 tests pass
- [ ] No API changes (backward compatible)
- [ ] Coverage ≥ 80%

---

## Phase 2: Caching & Performance (Week 3-4)

**Goal**: Add Redis caching layer + rate limiting

### Components

1. **Redis Cache** (`src/ww/storage/redis_cache.py`)
   - Embedding cache: 1h TTL, 70-80% hit rate
   - Search results: 5m TTL
   - Graph relationships: 10m TTL

2. **Rate Limiting** (`src/ww/api/middleware/rate_limit.py`)
   - Authenticated: 1000 req/min
   - Anonymous: 100 req/min
   - Admin: 50 req/min

3. **Batch Queries** (optimize graph traversal)
   - Replace N+1 pattern with `get_relationships_batch()`

### Quick Start

```bash
# 1. Add Redis to docker-compose.yml (template in plan)
docker-compose up -d redis

# 2. Create cache module
cp docs/ARCHITECTURE_REFACTORING_PLAN.md cache_template.py  # Extract code
vim src/ww/storage/redis_cache.py

# 3. Integrate with embedding provider
vim src/ww/embedding/bge_m3.py  # Add cache-aside pattern

# 4. Add rate limiting middleware
vim src/ww/api/middleware/rate_limit.py
vim src/ww/api/server.py  # app.add_middleware(RateLimitMiddleware)

# 5. Optimize graph queries
vim src/ww/memory/episodic_retrieval.py  # Use batch queries

# 6. Test performance
pytest tests/benchmarks/test_refactoring_performance.py -v
```

### Success Criteria

- [ ] Redis operational in docker-compose
- [ ] Embedding cache hit rate 70-80%
- [ ] Graph traversal ≤ 100ms (vs ~2s)
- [ ] Rate limiting active
- [ ] All tests pass

---

## Phase 3: Quality & Observability (Week 5)

**Goal**: Logger conversion + comprehensive bridge tests

### Tasks

1. **Logger Conversion** (232 print statements)
   ```bash
   # Find all prints
   grep -rn "print(" src/ww --include="*.py" | wc -l  # 232

   # Convert (manual or script)
   python scripts/convert_prints_to_logger.py --check  # Dry run
   python scripts/convert_prints_to_logger.py --apply  # Convert

   # Priority files:
   # - src/ww/nca/*.py (17 files)
   # - src/ww/interfaces/*.py (9 files)
   # - src/ww/bridges/*.py (3 files)
   ```

2. **Bridge Tests** (5 → 20+ files)
   ```bash
   # Create new test files (templates in plan)
   cd tests/bridges/
   touch test_{glymphatic,consolidation,hippocampus,vta,adenosine}_bridge.py
   touch test_{pattern_separation,neuromod_orchestra,learned_gate}_bridge.py
   touch test_{buffer_manager,three_factor,end_to_end_learning}.py

   # Run coverage
   pytest tests/bridges/ -v --cov=src/ww/bridges --cov-report=html
   # Target: 85% coverage
   ```

### Success Criteria

- [ ] Zero print() in src/ (except tests)
- [ ] 20+ bridge test files
- [ ] Bridge coverage ≥ 85%
- [ ] Docs updated
- [ ] Code quality score ≥ 9.0/10

---

## Docker Configuration

### Add Redis to docker-compose.yml

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: ww-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    networks:
      - ww-network

volumes:
  redis-data:
```

### Environment Variables (.env)

```bash
# Redis
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=50

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_AUTHENTICATED=1000
RATE_LIMIT_ANONYMOUS=100
```

---

## Testing Checklist

### Before Each Phase

```bash
# Baseline
pytest tests/ -v --cov=src/ww --cov-report=html
coverage report --fail-under=80

# Save benchmark baseline
pytest tests/benchmarks -v --benchmark-save=before
```

### After Each Phase

```bash
# Regression check
pytest tests/ -v --cov=src/ww

# Coverage check (no regression)
coverage report --fail-under=80

# Performance validation
pytest tests/benchmarks -v --benchmark-save=after
pytest-benchmark compare before after
```

### Test Count Validation

```bash
# Ensure tests don't decrease
TEST_COUNT=$(pytest --collect-only 2>/dev/null | grep "tests collected" | awk '{print $1}')
if [ "$TEST_COUNT" -lt 8075 ]; then
  echo "ERROR: Test count decreased!"
  exit 1
fi
```

---

## Rollback Procedures

### Phase 1 Rollback (Episodic)

```bash
# If tests fail or performance regresses > 10%
git revert $(git log --oneline | grep "Phase 1" | awk '{print $1}')
pytest tests/ -v  # Confirm stability
```

### Phase 2 Rollback (Caching)

```bash
# Disable Redis (no code changes)
export REDIS_ENABLED=false

# Or in .env
echo "REDIS_ENABLED=false" >> .env

# Restart API
docker-compose restart api
```

### Phase 3 Rollback (Quality)

```bash
# Revert logger changes if needed
git revert $(git log --oneline | grep "Phase 3" | awk '{print $1}')

# Bridge tests are additive (no rollback needed)
```

---

## Complexity Analysis

### Phase 1: Algorithmic Impact

| Operation | Before | After | Notes |
|-----------|--------|-------|-------|
| Episode storage | O(1) | O(1) | No change (delegation) |
| Recall | O(k log n) | O(k log n) | No change (same logic) |
| Service overhead | - | < 5% | Acceptable for maintainability |

### Phase 2: Performance Improvements

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Embedding (cached) | 50ms | 10ms | **5x** |
| Graph traversal (100 nodes) | 2000ms | 100ms | **20x** |
| Repeat queries | 50ms | 10ms | **5x** |

### Phase 3: Quality Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Code quality | 7.7/10 | 9.0/10 | +1.3 |
| Bridge coverage | ~60% | 85%+ | +25% |
| Observability | print() | logger | Structured |

---

## Parallelization Strategy

### Can Run in Parallel

- **Phase 1 (Episodic)** + **CompBio work** (different modules)
- **Phase 2 (Caching)** + **Hinton work** (different modules)
- **Phase 3 (Quality)** + **Phases 1-2** (cleanup task)

### Must Run Sequential

- Phase 2 depends on Phase 1 (cleaner service interfaces)
- Final release depends on all 3 phases

### Optimal Schedule (2 developers)

**Week 1-2**: Dev1 = Phase 1, Dev2 = Phase 3 (logger)
**Week 3-4**: Dev1 = Phase 2, Dev2 = Phase 3 (bridge tests)
**Week 5**: Both = Phase 3 (docs + validation)

**Total**: 4 weeks (vs 5 sequential)

---

## Key Decisions

### Why Facade Pattern?

- **Backward compatibility**: No API changes
- **Incremental migration**: Internals can be updated gradually
- **Zero risk**: All tests pass without modification

### Why Redis over memcached?

- **Data structures**: Lists, sets for complex caching
- **Persistence**: AOF for durability
- **TTL support**: Per-key expiration
- **Active development**: Better ecosystem

### Why Token Bucket for rate limiting?

- **Smooth traffic**: Better than fixed window
- **Burst tolerance**: Allows short spikes
- **Simple implementation**: No external state needed

---

## Success Metrics Summary

| Phase | Key Metric | Target | Impact |
|-------|------------|--------|--------|
| Phase 1 | episodic.py size | ≤ 400 lines | Maintainability |
| Phase 2 | Cache hit rate | 70-80% | Performance |
| Phase 3 | Bridge coverage | 85%+ | Reliability |
| **Overall** | **Code quality** | **9.0/10** | **Production ready** |

---

## Next Steps

1. **Review Plan**: Read full [ARCHITECTURE_REFACTORING_PLAN.md](/mnt/projects/ww/docs/ARCHITECTURE_REFACTORING_PLAN.md)
2. **Create Branch**: `git checkout -b refactor/phase1-episodic`
3. **Start Phase 1**: Extract episodic modules
4. **Run Tests**: Continuous validation
5. **Iterate**: Phase 2, Phase 3

---

## Questions?

- **Full details**: `/mnt/projects/ww/docs/ARCHITECTURE_REFACTORING_PLAN.md`
- **Architecture review**: `/mnt/projects/ww/docs/CODE_QUALITY_REVIEW.md`
- **Current roadmap**: `/mnt/projects/ww/docs/ROADMAP.md`

---

**Document Version**: 1.0
**Last Updated**: 2026-01-07
**Status**: Ready to Execute
