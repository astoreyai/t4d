# Testing Guide

Write reliable tests for World Weaver integrations.

## Test Structure

World Weaver's test suite is organized as:

```
tests/
├── unit/           # Fast unit tests (mocked)
├── integration/    # Integration tests (real services)
├── performance/    # Benchmarks
├── security/       # Security tests
├── chaos/          # Resilience tests
├── p4/             # P4 feature tests
└── conftest.py     # Shared fixtures
```

## Quick Start

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v -m integration

# With coverage
pytest tests/ --cov=src/ww --cov-report=html
```

## Writing Unit Tests

### Basic Test

```python
import pytest
from ww.memory.episodic import EpisodicMemory

@pytest.mark.asyncio
async def test_episode_creation():
    memory = EpisodicMemory()

    episode = await memory.create(
        content="Test content",
        session_id="test-session"
    )

    assert episode.id is not None
    assert episode.content == "Test content"
```

### Using Fixtures

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    client = AsyncMock()
    client.search.return_value = [
        MagicMock(id="1", score=0.9, payload={"content": "result"})
    ]
    return client

@pytest.fixture
def mock_neo4j():
    """Mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=None)
    return driver

@pytest.mark.asyncio
async def test_recall_with_mocks(mock_qdrant, mock_neo4j):
    memory = EpisodicMemory(
        qdrant_client=mock_qdrant,
        neo4j_driver=mock_neo4j
    )

    results = await memory.recall("query", limit=5)
    assert len(results) > 0
```

## Integration Tests

### With Real Services

```python
import pytest

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_memory_lifecycle():
    """Test store and recall with real services."""
    from ww import memory

    # Store
    await memory.store(
        "Integration test content",
        importance=0.8
    )

    # Recall
    results = await memory.recall("integration test")
    assert len(results) > 0
    assert "integration" in results[0].content.lower()
```

### Docker Compose for Tests

```yaml
# docker-compose.test.yml
version: '3.8'
services:
  neo4j-test:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/testpassword
    ports:
      - "7688:7687"

  qdrant-test:
    image: qdrant/qdrant
    ports:
      - "6334:6333"
```

```bash
# Start test services
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/ -v -m integration

# Cleanup
docker-compose -f docker-compose.test.yml down
```

## Testing Hooks

```python
import pytest
from ww.hooks.base import HookContext, HookPhase
from uuid import uuid4

@pytest.fixture
def hook_context():
    """Create test hook context."""
    return HookContext(
        hook_id=uuid4(),
        session_id="test-session",
        operation="create",
        phase=HookPhase.PRE,
        module="episodic",
        input_data={"content": "test content"}
    )

@pytest.mark.asyncio
async def test_custom_hook(hook_context):
    from my_hooks import MyCustomHook

    hook = MyCustomHook()

    # Test execution
    result = await hook.execute(hook_context)
    assert result.error is None

    # Test should_execute
    assert hook.should_execute(hook_context) == True
```

## Testing Learning Systems

```python
import pytest
import numpy as np

def test_dopamine_rpe():
    from ww.learning.dopamine import DopamineSystem

    da = DopamineSystem()

    # Positive surprise
    rpe = da.compute_rpe(expected=0.5, actual=0.8)
    assert rpe > 0

    # Negative surprise
    rpe = da.compute_rpe(expected=0.8, actual=0.3)
    assert rpe < 0

    # No surprise
    rpe = da.compute_rpe(expected=0.5, actual=0.5)
    assert abs(rpe) < 0.01
```

## Testing NCA Dynamics

```python
import pytest
import numpy as np

def test_neural_field_stability():
    from ww.nca import NeuralField

    field = NeuralField()

    # Initialize
    field.reset()
    initial_state = field.state.copy()

    # Step multiple times
    for _ in range(100):
        field.step(dt=0.01)

    # Should converge to attractor
    final_state = field.state
    assert np.linalg.norm(final_state - initial_state) > 0.1  # Changed
    assert field.is_stable()  # Converged
```

## Fixtures Reference

### Common Fixtures (conftest.py)

| Fixture | Description |
|---------|-------------|
| `event_loop` | Session-scoped event loop |
| `mock_embedding_service` | Mocked embedding service |
| `mock_qdrant_store` | Mocked Qdrant store |
| `mock_neo4j_store` | Mocked Neo4j store |
| `test_session_id` | Unique session ID per test |
| `sample_episode` | Sample episode object |
| `sample_entity` | Sample entity object |

### Using Fixtures

```python
@pytest.mark.asyncio
async def test_with_fixtures(
    mock_qdrant_store,
    mock_neo4j_store,
    test_session_id,
    sample_episode
):
    # Fixtures are automatically injected
    assert test_session_id.startswith("test-")
    assert sample_episode.content is not None
```

## Best Practices

### 1. Isolate Tests

```python
@pytest.fixture(autouse=True)
async def cleanup(test_session_id):
    """Clean up after each test."""
    yield
    # Cleanup code here
    await cleanup_session(test_session_id)
```

### 2. Use Markers

```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_large_consolidation():
    ...

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_database():
    ...
```

### 3. Parameterize Tests

```python
@pytest.mark.parametrize("content,expected", [
    ("short", False),
    ("a" * 100, True),
    ("normal content", True),
])
@pytest.mark.asyncio
async def test_content_validation(content, expected):
    result = validate_content(content)
    assert result == expected
```

### 4. Test Edge Cases

```python
@pytest.mark.asyncio
async def test_empty_recall():
    """Test recall with no matches."""
    results = await memory.recall("nonexistent query xyz123")
    assert len(results) == 0

@pytest.mark.asyncio
async def test_unicode_content():
    """Test content with unicode characters."""
    await memory.store("日本語テスト 🎉")
    results = await memory.recall("日本語")
    assert len(results) > 0
```

## Coverage Goals

| Module | Target | Current |
|--------|--------|---------|
| `memory/` | 85% | 80% |
| `learning/` | 80% | 75% |
| `nca/` | 90% | 88% |
| `hooks/` | 85% | 85% |
| `api/` | 80% | 82% |
| **Overall** | **80%** | **80%** |
