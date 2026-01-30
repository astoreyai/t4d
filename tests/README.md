# World Weaver Test Suite

Comprehensive test infrastructure for the World Weaver tripartite memory system (episodic, semantic, procedural).

## Overview

The test suite is organized into three main categories:

- **Unit Tests**: Fast, isolated tests of individual components (~1-2 seconds each)
- **Integration Tests**: Slower tests with real/mocked databases (~5-10 seconds each)
- **Security Tests**: Validation of input sanitization and injection prevention

Property-based testing validates algorithm invariants across wide input ranges.

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures and configuration
├── unit/                                # Fast unit tests
│   ├── test_algorithms_property.py      # Property-based tests (FSRS, Hebbian, ACT-R)
│   ├── test_batch_queries.py            # Neo4j batch query optimization
│   ├── test_clustering.py               # Semantic clustering tests
│   ├── test_config_security.py          # Configuration validation
│   ├── test_db_timeouts.py              # Database timeout handling
│   ├── test_neo4j_connection_pool.py    # Neo4j connection pooling
│   ├── test_qdrant_optimizations.py     # Vector store optimizations
│   ├── test_rate_limiter.py             # Rate limiting functionality
│   ├── test_saga.py                     # Distributed transaction patterns
│   └── test_validation.py               # Input validation
├── integration/                         # Integration tests with services
│   ├── test_session_isolation.py        # Multi-instance isolation
│   └── test_memory_lifecycle.py         # Full memory lifecycle workflows
├── security/                            # Security and injection tests
│   └── test_injection.py                # Input injection prevention
└── performance/                         # Benchmarks (if needed)
    └── test_benchmarks.py               # Performance baselines
```

## Running Tests

### Run All Tests

```bash
# Verbose output with coverage
pytest tests/ -v --cov=src/ww --cov-report=html

# Quick run (skip slow tests)
pytest tests/ -v -m "not slow"

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v -m integration

# Run only security tests
pytest tests/security/ -v -m security

# Run only property-based tests
pytest tests/unit/test_algorithms_property.py -v
```

### Run Specific Tests

```bash
# Single test file
pytest tests/unit/test_batch_queries.py -v

# Single test class
pytest tests/unit/test_algorithms_property.py::TestFSRSRetrievability -v

# Single test function
pytest tests/unit/test_algorithms_property.py::TestFSRSRetrievability::test_retrievability_bounded_in_unit_interval -v

# With hypothesis verbosity
pytest tests/unit/test_algorithms_property.py -v --hypothesis-verbosity=verbose
```

### Coverage Reports

```bash
# Terminal report
pytest tests/ --cov=src/ww --cov-report=term-missing

# HTML report (opens in browser)
pytest tests/ --cov=src/ww --cov-report=html
open htmlcov/index.html

# Coverage report with annotations
pytest tests/ --cov=src/ww --cov-report=term-missing --cov-fail-under=70

# Per-file breakdown
coverage report -m --skip-covered
```

## Fixtures

### Core Fixtures

#### `test_session_id`
Provides a unique session ID for test isolation.

```python
def test_example(test_session_id):
    assert test_session_id.startswith("test-")
```

#### `event_loop`
Session-scoped event loop for async tests. Uses session scope to maintain Neo4j connection pools.

### Mock Fixtures

#### `mock_qdrant_store`
Mocked Qdrant vector store for unit tests.

```python
@pytest.mark.asyncio
async def test_with_qdrant(mock_qdrant_store):
    # Configure mock for test
    mock_qdrant_store.search.return_value = [{"id": "1", "score": 0.95}]

    # Use mock
    results = await mock_qdrant_store.search(...)

    # Verify calls
    mock_qdrant_store.search.assert_called()
```

Available methods:
- `initialize()`: async
- `add()`: async, returns None
- `search()`: async, returns list of results
- `delete()`: async
- `count()`: async, returns int
- `close()`: async
- `upsert()`: async

#### `mock_neo4j_store`
Mocked Neo4j graph store for unit tests.

```python
@pytest.mark.asyncio
async def test_with_neo4j(mock_neo4j_store):
    mock_neo4j_store.get_node.return_value = {
        "id": "node-1",
        "name": "Test Entity"
    }

    result = await mock_neo4j_store.get_node("node-1")
    assert result["name"] == "Test Entity"
```

Available methods:
- `initialize()`: async
- `query()`: async, returns list
- `create_node()`: async, returns id
- `get_node()`: async, returns node or None
- `delete_node()`: async
- `create_relationship()`: async
- `get_relationships()`: async, returns list
- `get_relationships_batch()`: async, returns dict
- `close()`: async

#### `mock_embedding_provider`
Mocked embedding provider returning 1024-dimensional vectors.

```python
@pytest.mark.asyncio
async def test_with_embeddings(mock_embedding_provider):
    embedding = await mock_embedding_provider.embed_query("test")
    assert len(embedding) == 1024
```

Available methods:
- `embed_query()`: async, returns list[float]
- `embed_documents()`: async, returns list[list[float]]

### Memory Service Fixtures

#### `mock_episodic_memory`
Fully initialized episodic memory service with mocked backends.

```python
@pytest.mark.asyncio
async def test_episodic(mock_episodic_memory):
    # Service is already initialized
    episode = await mock_episodic_memory.create(
        content="Test event",
        outcome="success"
    )
    assert episode is not None
```

#### `mock_semantic_memory`
Fully initialized semantic memory service with mocked backends.

```python
@pytest.mark.asyncio
async def test_semantic(mock_semantic_memory):
    entity = await mock_semantic_memory.create_entity(
        name="Test Concept",
        entity_type="CONCEPT",
        summary="A test concept"
    )
    assert entity.name == "Test Concept"
```

#### `mock_procedural_memory`
Fully initialized procedural memory service with mocked backends.

```python
@pytest.mark.asyncio
async def test_procedural(mock_procedural_memory):
    # Service is ready to use
    pass
```

#### `all_memory_services`
All three memory services with consistent session ID for cross-memory tests.

```python
@pytest.mark.asyncio
async def test_cross_memory(all_memory_services):
    session_id = all_memory_services["session_id"]
    episodic = all_memory_services["episodic"]
    semantic = all_memory_services["semantic"]
    procedural = all_memory_services["procedural"]
```

### Mock Builders (Factories)

#### `mock_search_result`
Factory for creating realistic vector search results.

```python
def test_search(mock_search_result):
    result = mock_search_result(
        id="entity-123",
        score=0.92,
        payload={"content": "Test content"}
    )
    assert result["score"] == 0.92
```

#### `mock_graph_node`
Factory for creating Neo4j node objects.

```python
def test_node(mock_graph_node):
    node = mock_graph_node(
        id="node-1",
        label="Entity",
        properties={"name": "Test"}
    )
    assert node["label"] == "Entity"
```

#### `mock_graph_relationship`
Factory for creating Neo4j relationship objects.

```python
def test_relationship(mock_graph_relationship):
    rel = mock_graph_relationship(
        source_id="node-1",
        target_id="node-2",
        rel_type="RELATES_TO",
        weight=0.7
    )
    assert rel["weight"] == 0.7
```

### Configuration Fixtures

#### `mock_settings`
Provides default test configuration.

```python
def test_with_settings(mock_settings):
    assert mock_settings.fsrs_default_stability == 1.0
    assert mock_settings.hebbian_learning_rate == 0.1
```

#### `patch_settings`
Patches `get_settings()` globally during test.

```python
def test_patched(patch_settings):
    # All code using get_settings() gets mock_settings
    from ww.core.config import get_settings
    settings = get_settings()
    assert settings.session_id == "test-session"
```

## Writing Tests

### Async Tests

Use `@pytest.mark.asyncio` decorator for async functions:

```python
@pytest.mark.asyncio
async def test_async_operation(mock_qdrant_store):
    await mock_qdrant_store.initialize()
    result = await mock_qdrant_store.search([0.1] * 1024)
    assert result == []
```

### Property-Based Tests

Use `hypothesis` for generative testing:

```python
from hypothesis import given, strategies as st

class TestAlgorithm:
    @given(st.floats(min_value=0, max_value=100))
    def test_property(self, value):
        """Property should hold for all valid inputs."""
        result = some_algorithm(value)
        assert 0 <= result <= 1  # Bounded property
```

Available strategies:
- `st.floats()`: floating point numbers
- `st.integers()`: integers
- `st.text()`: strings
- `st.lists()`: lists
- `st.dictionaries()`: dicts
- `st.one_of()`: union types

### Mock Configuration

Configure mocks per-test:

```python
@pytest.mark.asyncio
async def test_with_custom_mock(mock_neo4j_store):
    # Setup
    mock_neo4j_store.get_node.return_value = {"id": "test", "name": "Test"}
    mock_neo4j_store.create_relationship.side_effect = ValueError("Already exists")

    # Test
    node = await mock_neo4j_store.get_node("test")
    assert node["name"] == "Test"

    # Verify
    mock_neo4j_store.get_node.assert_called_with("test")
    assert mock_neo4j_store.get_node.call_count == 1
```

### Testing Error Paths

```python
@pytest.mark.asyncio
async def test_handles_errors(mock_qdrant_store):
    mock_qdrant_store.search.side_effect = ConnectionError("Network error")

    with pytest.raises(ConnectionError):
        await mock_qdrant_store.search([0.1] * 1024)
```

## Algorithm Testing

### FSRS (Spaced Repetition) Tests

Location: `tests/unit/test_algorithms_property.py`

Validates:
- **Bounded [0, 1]**: R(t, S) = (1 + 0.9*t/S)^(-0.5) always returns valid probability
- **Monotonic Decrease**: Retrievability decreases with time
- **Approaches Zero**: As t → ∞, R → 0
- **Higher Stability**: Longer intervals preserve retrievability

Example:
```python
def test_fsrs_retrievability_bounded():
    """Retrievability should always be in [0, 1]."""
    for days in [0, 1, 10, 100, 1000]:
        for stability in [0.1, 1.0, 10.0]:
            r = (1 + 0.9 * days / stability) ** (-0.5)
            assert 0 <= r <= 1
```

### Hebbian Weight Tests

Location: `tests/unit/test_algorithms_property.py`

Validates:
- **Bounded [0, 1]**: w' = w + lr*(1-w) stays in valid range
- **Monotonic Growth**: Weights never decrease
- **Convergence**: Repeated updates approach 1.0
- **Learning Rate Effect**: Higher rates converge faster

Example:
```python
def test_hebbian_convergence():
    """Repeated strengthening should approach 1."""
    w = 0.1
    for _ in range(100):
        w = w + 0.1 * (1.0 - w)
    assert w > 0.95  # Close to 1
```

### ACT-R Activation Tests

Location: `tests/unit/test_algorithms_property.py`

Validates:
- **Logarithmic Sum**: A = ln(Σ t_i^(-d))
- **Frequency Effect**: More accesses increase activation
- **Recency Effect**: Recent accesses weighted more heavily
- **Power-Law Decay**: Older items decay faster

## Test Markers

Use pytest markers to categorize tests:

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Only integration tests
pytest tests/ -m integration

# Only security tests
pytest tests/ -m security

# Combine markers
pytest tests/ -m "integration and not slow"
```

Available markers:
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.security`: Security tests
- `@pytest.mark.asyncio`: Async tests (auto-applied)

## Best Practices

### Test Organization

1. **One assertion per test** (when possible): Easier to debug failures
2. **Descriptive names**: `test_what_it_does_given_what_input()`
3. **Arrange-Act-Assert**: Setup, execute, verify
4. **Use fixtures**: Reduce boilerplate and improve reusability

```python
@pytest.mark.asyncio
async def test_create_entity_with_valid_inputs(mock_semantic_memory):
    # Arrange
    name = "Test Concept"
    entity_type = "CONCEPT"
    summary = "A test concept"

    # Act
    entity = await mock_semantic_memory.create_entity(
        name=name,
        entity_type=entity_type,
        summary=summary
    )

    # Assert
    assert entity.name == name
    assert entity.entity_type.value == entity_type
```

### Mock Best Practices

1. **Isolate dependencies**: Mock external services
2. **Configure return values**: Set up expected behavior
3. **Verify interactions**: Assert mocks were called correctly
4. **Test error paths**: Use `side_effect` for exceptions

```python
@pytest.mark.asyncio
async def test_handles_network_error(mock_qdrant_store):
    # Setup mock to raise error
    mock_qdrant_store.search.side_effect = ConnectionError("Network down")

    # Test error handling
    with pytest.raises(ConnectionError):
        await some_function_using_qdrant()

    # Verify recovery (if applicable)
    mock_qdrant_store.search.assert_called_once()
```

### Property Testing Best Practices

1. **State invariants**: Properties that should always be true
2. **Bounded inputs**: Use reasonable value ranges
3. **Shrinking**: Hypothesis finds minimal failing examples
4. **Seeds**: Use `--hypothesis-seed=N` for reproducibility

```python
from hypothesis import given, strategies as st, assume

@given(st.floats(min_value=0, max_value=10000))
def test_stability_always_positive(days):
    # Assume valid input
    assume(days >= 0)

    # Test invariant
    stability = 1.0
    retrieval = (1 + 0.9 * days / stability) ** (-0.5)

    # Verify property
    assert 0 <= retrieval <= 1  # Bounded
```

## CI/CD Integration

The test suite is integrated into GitHub Actions (see `.github/workflows/test.yml`).

### What Runs Automatically

On every push and pull request:
1. **Unit tests**: All tests except marked `slow`
2. **Property tests**: Hypothesis-based algorithm validation
3. **Integration tests**: Real Neo4j instances
4. **Security tests**: Input validation and injection prevention
5. **Coverage check**: Enforces 70% minimum coverage

### Coverage Requirements

- **Minimum**: 70% of source code
- **Target**: 85%+ for critical paths (memory services, algorithms)
- **Reports**: Available in PR comments and artifacts

### View Coverage Locally

```bash
pytest tests/ --cov=src/ww --cov-report=html
open htmlcov/index.html
```

## Troubleshooting

### Common Issues

**Issue: Tests hang on asyncio operations**
- Cause: Event loop conflicts
- Solution: Use session-scoped event loop (already configured)

**Issue: Neo4j connection refused**
- Cause: Docker service not running
- Solution: For local testing, start docker: `docker-compose up neo4j`

**Issue: Hypothesis examples too slow**
- Cause: Large dataset generation
- Solution: Reduce `max_size` in list/dict strategies

**Issue: Mock not being called**
- Cause: Import paths incorrect
- Solution: Verify patch paths match actual import locations

```python
# Correct
with patch('ww.memory.semantic.get_qdrant_store'):
    ...

# Incorrect (would patch the wrong module)
with patch('qdrant_store.get_qdrant_store'):
    ...
```

## Performance Testing

For benchmarking, use pytest-benchmark:

```bash
pip install pytest-benchmark
```

```python
def test_batch_vs_individual(benchmark):
    def run_batch():
        # Test code
        pass

    result = benchmark(run_batch)
    # Results printed to stdout
```

## Debug Mode

Run tests with verbose output and print statements:

```bash
pytest tests/ -vv -s --tb=short

# Show all variable values on failure
pytest tests/ -vv --showlocals
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Hypothesis for property testing](https://hypothesis.readthedocs.io/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
