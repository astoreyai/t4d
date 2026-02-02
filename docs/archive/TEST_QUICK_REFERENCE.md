# Test Coverage Quick Reference

## Key Metrics
- **Overall Coverage**: 77% (3721/4860 statements)
- **Tests Passing**: 1121/1235 (91%)
- **Tests Failing**: 114 (9%)
- **Additional Tests Needed**: 100-150

## One-Minute Fixes

### 1. Fix Password Validation (3 tests)
```bash
# Edit .env file
sed -i 's/NEO4J_PASSWORD=wwpassword/NEO4J_PASSWORD=WwPass123!/g' /mnt/projects/t4d/t4dm/.env

# Verify
grep NEO4J_PASSWORD /mnt/projects/t4d/t4dm/.env
```
**Impact**: Unblocks 40+ indirect tests

### 2. Start Database Services
```bash
# Start services (requires Docker)
cd /mnt/projects/ww
docker-compose up neo4j qdrant

# In another terminal, run tests
pytest tests/test_memory.py -v
```
**Impact**: Unblocks 54 database tests

## Coverage by Priority

### RED - Critical Gaps
| File | Coverage | Tests Failing | Action |
|------|----------|---------------|--------|
| t4dx_graph_adapter.py | 39% | 30+ | Add 25-30 tests |
| mcp/schema.py | 0% | unknown | Add 10+ tests |
| unified.py | 18% | missing | Add 35-40 tests |
| mcp/server.py | 20% | unknown | Add 10+ tests |

### YELLOW - Should Fix
| File | Coverage | Tests Failing | Action |
|------|----------|---------------|--------|
| mcp/tools/episodic.py | 49% | 14 | Add 15-20 tests |
| mcp/tools/procedural.py | 53% | 12 | Add 10-15 tests |
| mcp/tools/semantic.py | 46% | 8 | Add 12-15 tests |
| mcp/tools/system.py | 33% | 12 | Add 15-20 tests |
| t4dx_vector_adapter.py | 62% | 8 | Add 15-20 tests |
| consolidation/service.py | 69% | 12 | Add 10-15 tests |

### GREEN - Good Shape
| File | Coverage | Tests Failing | Action |
|------|----------|---------------|--------|
| episodic.py | 93% | 9 | Minor edge cases |
| semantic.py | 91% | 4 | Minor edge cases |
| procedural.py | 97% | 10 | Minor edge cases |
| gateway.py | 100% | 0 | Already complete |
| tracing.py | 89% | 0 | Already complete |

## Failure Root Causes

### By Error Type
1. **Configuration Error** (3 tests)
   - `neo4j_password` too weak
   - Fix: Update .env

2. **Database Connection Error** (54 tests)
   - Neo4j/Qdrant not running
   - Fix: Start docker-compose

3. **Mock Setup Error** (16 tests)
   - Missing attributes in mock returns
   - Fix: Update conftest.py fixture returns

4. **Import Error** (2 tests)
   - Missing `hypothesis` module for property tests
   - Fix: Already installed, just run tests again

5. **Fixture Error** (39 tests)
   - Settings validation or database access
   - Fix: Fix #1 + #2 above

## Test Categories Status

```
Memory Core              93% ✓ (good)
├── episodic.py         93% ✓
├── semantic.py         91% ✓
├── procedural.py       97% ✓
└── unified.py          18% ✗ CRITICAL

MCP Layer              77% ~ (mixed)
├── gateway.py         100% ✓
├── types.py           100% ✓
├── validation.py       97% ✓
├── errors.py          100% ✓
├── server.py           20% ✗ CRITICAL
├── schema.py            0% ✗ CRITICAL
└── tools/            49% ✗ (episodic, semantic, procedural, system)

Storage Layer          50% ~ (mixed)
├── saga.py             97% ✓
├── t4dx_graph_adapter.py      39% ✗ CRITICAL
└── t4dx_vector_adapter.py     62% ~ (needs work)

Observability          64% ~ (mixed)
├── logging.py         100% ✓
├── metrics.py         100% ✓
├── tracing.py          89% ✓
└── health.py           59% ~ (needs work)

Consolidation          69% ~ (needs work)
└── service.py          69% ~

Embedding              96% ✓ (good)
└── bge_m3.py           96% ✓

Extraction             93% ✓ (good)
└── entity_extractor.py 93% ✓
```

## Files to Create (Priority Order)

1. `tests/storage/test_neo4j_detailed.py` (25-30 tests)
   - Cypher injection tests
   - Complex query coverage
   - Error handling paths

2. `tests/unit/test_unified_memory.py` (35-40 tests)
   - Cross-memory consistency
   - Session isolation
   - Consolidation routing

3. `tests/mcp/test_tools_batch.py` (50+ tests)
   - Batch operations for all 3 memory types
   - Error recovery
   - Partial failure handling

4. `tests/storage/test_qdrant_detailed.py` (15-20 tests)
   - Vector normalization
   - Batch operations
   - Search optimization

5. `tests/chaos/test_resilience.py` (15-20 tests)
   - Connection pool exhaustion
   - Timeout recovery
   - Transaction rollback

## Test Execution Quick Commands

```bash
# Just unit tests (no DB required)
pytest tests/unit/ -v

# Just integration tests (need DB)
pytest tests/integration/ -v --timeout=30

# Just security tests
pytest tests/security/ -v

# Memory tests only
pytest tests/unit/test_episodic.py tests/unit/test_semantic.py tests/unit/test_procedural.py -v

# Show only failures
pytest tests/ -lf -v

# Re-run only last failed
pytest tests/ --lf

# With HTML coverage report
pytest tests/ --cov=src/ww --cov-report=html -v
# Then open htmlcov/index.html
```

## Environment Variables

```bash
# Required for tests
T4DM_TEST_MODE=true
NEO4J_PASSWORD=WwPass123!  # Must be complex (2+ of: upper, lower, digit, special)
NEO4J_HOST=localhost
NEO4J_PORT=7687
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Coverage Goals

| Target | Current | Gap | Tests Needed |
|--------|---------|-----|--------------|
| Overall | 77% | 13% | 500+ statements |
| t4dx_graph_adapter | 39% | 41% | 170+ statements |
| unified.py | 18% | 67% | 80+ statements |
| mcp/tools | 45% | 50% | 180+ statements |
| Overall Target | 85% | - | ~100 new tests |

## Common Test Patterns

### Memory Test Template
```python
@pytest.mark.asyncio
async def test_feature_name(mock_episodic_memory):
    """Test description."""
    # Setup
    memory = mock_episodic_memory

    # Execute
    result = await memory.create(...)

    # Verify
    assert result is not None
```

### Storage Test Template
```python
@pytest.mark.asyncio
async def test_database_feature(mock_t4dx_graph_adapter):
    """Test description."""
    store = mock_t4dx_graph_adapter
    store.query.return_value = [{"id": "test"}]

    result = await store.query("...")
    assert len(result) == 1
```

### Integration Test Template
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow():
    """Test complete workflow."""
    from t4dm.memory.episodic import get_episodic_memory

    memory = get_episodic_memory("session-001")
    await memory.initialize()

    # ... test

    await memory.close()
```

## Debugging Tips

### See what mocks are returning
```python
# In test code
print(mock_t4dx_graph_adapter.query.return_value)
print(mock_t4dx_graph_adapter.query.call_args)
```

### Check what's being tested
```bash
# Show test collection
pytest tests/unit/test_episodic.py --collect-only -q

# Show test paths only
pytest tests/ --collect-only -q | head -20
```

### Profile slow tests
```bash
# Show slowest 10 tests
pytest tests/ --durations=10

# Mark tests as slow if > 1 second
pytest tests/ --benchmark-disable -v --timeout=60
```

### Debug failed test
```bash
# Drop into debugger on failure
pytest tests/unit/test_episodic.py -v --pdb

# Show full diff for assertions
pytest tests/ -v --tb=short

# Show print statements
pytest tests/unit/test_episodic.py -v -s
```

## Next Steps

**Today (1 hour)**:
1. ✅ Run coverage report (done)
2. ✅ Identify gaps (done)
3. Fix password in .env
4. Document findings

**This Week**:
1. Start database services
2. Fix Priority 1 issues
3. Create neo4j_detailed tests
4. Create unified_memory tests

**Target**: Get failing tests <50, coverage to 85%
