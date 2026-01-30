# World Weaver Tests Index

## Quick Navigation

### Location: `/mnt/projects/ww/tests/`

## Test Categories

### 1. Integration Tests
**Path**: `tests/integration/`

Tests complete end-to-end workflows across the tripartite memory system.

| File | Tests | Purpose |
|------|-------|---------|
| `test_memory_lifecycle.py` | 8 | Full system workflows (P4-007) |
| `test_session_isolation.py` | 10+ | Session isolation verification |
| `conftest.py` | N/A | Shared fixtures and mocks |

**Run**:
```bash
pytest tests/integration/ -v
```

**Key Tests**:
1. `test_episode_lifecycle` - Episode creation→recall→decay cycle
2. `test_cross_memory_episode_to_entity_extraction` - Episodic→semantic
3. `test_session_isolation_e2e` - Session isolation verification
4. `test_concurrent_recalls` - Concurrent operation safety
5. `test_partial_failure_recovery` - Error handling with sagas
6. `test_consolidation_workflow` - Multi-step consolidation
7. `test_memory_decay_application` - FSRS decay mechanism
8. `test_multi_session_consolidation` - Cross-session workflows

### 2. Performance Tests (Slow)
**Path**: `tests/performance/`

Performance benchmarks with regression detection thresholds.

| File | Tests | Purpose |
|------|-------|---------|
| `test_benchmarks.py` | 8 | Performance benchmarks (P4-008) |
| `conftest.py` | N/A | Benchmark utilities and fixtures |

**Run**:
```bash
pytest tests/performance/ -m slow -v
```

**Key Benchmarks**:
1. `test_create_1000_episodes` - Throughput: < 10s
2. `test_recall_from_10000_episodes` - Search latency: < 5s
3. `test_consolidate_1000_episodes` - Consolidation: < 30s
4. `test_100_concurrent_operations` - Concurrent ops: < 30s
5. `test_memory_usage_under_load` - Heap usage: < 1GB
6. `test_embedding_generation_performance` - Embedding speed: < 2s
7. `test_vector_search_performance` - Search speed: < 0.1s/query
8. `test_graph_operation_performance` - Graph ops: < 5s

### 3. Unit Tests
**Path**: `tests/unit/`

Component-level testing (existing tests).

**Files**: 15+ test modules covering:
- Memory services (episodic, semantic, procedural)
- Storage backends (Neo4j, Qdrant)
- Consolidation algorithms
- Configuration and validation
- Rate limiting and timeouts

**Run**:
```bash
pytest tests/unit/ -v
```

### 4. Security Tests
**Path**: `tests/security/`

Security and injection testing.

| File | Purpose |
|------|---------|
| `test_injection.py` | SQL/Cypher injection prevention |

**Run**:
```bash
pytest tests/security/ -v
```

## Test Execution Patterns

### All Tests (Recommended for Pre-Commit)
```bash
pytest tests/ -v --cov=src/ww
```

### Fast Tests Only (CI-Friendly)
```bash
pytest tests/ -m "not slow" -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Performance Benchmarks (Long Running)
```bash
pytest tests/performance/ -m slow -v
```

### Specific Test
```bash
pytest tests/integration/test_memory_lifecycle.py::test_episode_lifecycle -v
```

### With Coverage Report
```bash
pytest tests/ -v --cov=src/ww --cov-report=html --cov-report=term-missing
```

## Pytest Markers

Configure in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "benchmark: marks tests as benchmarks",
    "integration: marks tests as integration tests",
    "performance: marks tests as performance tests",
    "security: marks tests as security tests",
    "asyncio: marks tests as async",
]
```

**Usage**:

| Marker | Command |
|--------|---------|
| `@pytest.mark.slow` | `pytest -m slow` (slow tests) |
| `@pytest.mark.benchmark` | `pytest -m benchmark` (benchmarks) |
| `@pytest.mark.integration` | `pytest -m integration` (integration) |
| `@pytest.mark.security` | `pytest -m security` (security) |
| `-m "not slow"` | Skip slow tests |

## Fixtures

### Session Management
- `test_session_id` - Unique test session ID
- `integration_session_id` - Integration test session
- `test_session_a`, `test_session_b` - Isolation test sessions
- `benchmark_session_id` - Performance test session

### Mock Stores
- `mock_embedding_provider` - BGE-M3 embeddings (1024-dim)
- `mock_qdrant_store` - Vector store simulation
- `mock_neo4j_store` - Graph store simulation

### Convenience Fixtures
- `episodic_memory_with_mocks` - Pre-configured EpisodicMemory
- `semantic_memory_with_mocks` - Pre-configured SemanticMemory
- `procedural_memory_with_mocks` - Pre-configured ProceduralMemory

### Performance Utilities
- `benchmark_timer` - Simple timer
- `benchmark_results` - Result collection and reporting
- `memory_tracker` - Heap usage tracking
- `generate_load` - Load data generation
- `async_benchmark` - Async function timing
- `repeat_benchmark` - Statistical sampling

## Files Reference

### Test Files Created (New)

| File | Size | Purpose |
|------|------|---------|
| `tests/integration/test_memory_lifecycle.py` | 20KB | 8 integration tests (P4-007) |
| `tests/integration/conftest.py` | 10KB | Integration test fixtures |
| `tests/performance/test_benchmarks.py` | 20KB | 8 performance benchmarks (P4-008) |
| `tests/performance/conftest.py` | 15KB | Benchmark fixtures and utilities |

### Documentation Files Created (New)

| File | Size | Purpose |
|------|------|---------|
| `TESTING_GUIDE.md` | 15KB | Complete testing documentation |
| `TEST_IMPLEMENTATION_SUMMARY.md` | 11KB | Implementation overview |
| `TESTS_INDEX.md` | This file | Test structure reference |

### Configuration Files Updated

| File | Changes |
|------|---------|
| `pyproject.toml` | Added pytest markers, dev dependencies |

## Performance Thresholds

| Benchmark | Threshold | Rationale |
|-----------|-----------|-----------|
| Create 1000 episodes | < 10s | 10ms/episode (embedding + storage) |
| Recall from 10K | < 5s | O(log n) HNSW search + filtering |
| Consolidate 1000 | < 30s | NLP extraction + clustering |
| 100 concurrent ops | < 30s | Connection pool efficiency |
| Memory for 10K | < 1GB | Embeddings (40MB) + payload + overhead |
| Embedding 100+100 | < 2s | ~20ms per embedding batch |
| Vector search/query | < 0.1s | Fast HNSW with filtering |
| Graph 1000+900 | < 5s | Batch node/rel creation |

## Development Workflow

### Before Committing
```bash
# Run fast tests
pytest tests/ -m "not slow" -v --cov=src/ww

# Check coverage
coverage report --fail-under=80
```

### Before Merging PR
```bash
# Run all tests
pytest tests/ -v --cov=src/ww

# Run benchmarks
pytest tests/performance/ -m slow -v

# Check for regressions
# (Compare benchmark output with baseline)
```

### CI/CD Pipeline
```bash
# Fast path (normal commits)
pytest -m "not slow" --cov -x  # Stop on first failure

# Nightly (full testing)
pytest tests/ --cov
pytest tests/performance/ -m slow  # Long-running benchmarks
```

## Common Commands

### Run Everything
```bash
pytest tests/
```

### Run Only Integration Tests
```bash
pytest tests/integration/
```

### Run Only Performance Tests
```bash
pytest tests/performance/ -m slow
```

### Run Specific Test
```bash
pytest tests/integration/test_memory_lifecycle.py::test_episode_lifecycle -v
```

### Run with Verbose Output
```bash
pytest tests/ -vv -s  # -s shows print statements
```

### Run with Coverage
```bash
pytest tests/ --cov=src/ww --cov-report=html
# Open: htmlcov/index.html
```

### Run Tests in Parallel
```bash
# Install: pip install pytest-xdist
pytest tests/ -n auto  # Use all CPU cores
```

### Run Tests Matching Pattern
```bash
pytest -k "episode" tests/  # Run *episode* tests
pytest -k "not slow" tests/  # Run tests without 'slow' marker
```

## Test Statistics

| Category | Count | Type |
|----------|-------|------|
| Integration Tests | 8 | End-to-end workflows |
| Performance Benchmarks | 8 | Thresholds + timing |
| Session Isolation Tests | 10+ | Existing tests |
| Unit Tests | 15+ | Component testing |
| Security Tests | 3+ | Injection prevention |
| **Total** | **45+** | All categories |

## Environment Setup

### Install Dependencies
```bash
cd /mnt/projects/ww
pip install -e ".[dev]"
```

### Required Packages (Key)
- `pytest>=7.0.0` - Test runner
- `pytest-asyncio>=0.21.0` - Async test support
- `pytest-cov>=4.0.0` - Coverage reporting
- `pytest-benchmark>=4.0.0` - Benchmark utilities
- `psutil>=5.9.0` - Memory tracking

## Troubleshooting

### Tests Timeout
```bash
pytest --timeout=300 tests/  # 5 minute timeout
```

### Mock Import Error
Ensure patches apply before import:
```python
with patch("ww.memory.episodic.get_embedding_provider", return_value=mock):
    # Import and use here
```

### Async Test Fails
Use `AsyncMock` from `unittest.mock`:
```python
from unittest.mock import AsyncMock
mock = AsyncMock()
```

### Session Filter Not Applied
Verify mock's `search()` call includes filter:
```python
search_filter = mock_qdrant_store.search.call_args[1]["filter"]
assert search_filter["session_id"] == expected_session
```

## Resources

- **Testing Guide**: `TESTING_GUIDE.md` - Detailed documentation
- **Implementation Summary**: `TEST_IMPLEMENTATION_SUMMARY.md` - Overview
- **Pytest Docs**: https://docs.pytest.org/
- **AsyncIO**: https://docs.python.org/3/library/asyncio.html
- **unittest.mock**: https://docs.python.org/3/library/unittest.mock.html

## Key Files for Development

### Read These First
1. `TESTING_GUIDE.md` - How to run and use tests
2. `tests/integration/test_memory_lifecycle.py` - Example integration tests
3. `tests/performance/test_benchmarks.py` - Example performance tests

### Reference Files
1. `tests/integration/conftest.py` - Fixture definitions
2. `tests/performance/conftest.py` - Utility definitions
3. `pyproject.toml` - Test configuration

## Summary

World Weaver has enterprise-grade testing with:
- ✓ 8 integration tests for full workflows
- ✓ 8 performance benchmarks with thresholds
- ✓ Pytest markers for flexible execution
- ✓ Complete fixture library
- ✓ Comprehensive documentation
- ✓ Ready for CI/CD integration
