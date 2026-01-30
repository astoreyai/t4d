"""
Pytest Configuration for World Weaver Tests.

Provides fixtures and configuration for async tests, database mocking,
and test isolation.
"""

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import AsyncGenerator, Any
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest
import pytest_asyncio
from dotenv import load_dotenv

# Load .env file for environment variables (especially for integration tests)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Configure pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )


@pytest.fixture(autouse=True, scope="function")
def reset_circuit_breakers():
    """Reset circuit breakers before each test to ensure test isolation."""
    # Import here to avoid circular imports
    from ww.storage.resilience import _circuit_breakers

    # Clear all circuit breakers before each test
    _circuit_breakers.clear()
    yield
    # Also clear after test for good measure
    _circuit_breakers.clear()


@pytest.fixture(autouse=True, scope="function")
def auto_patch_settings(monkeypatch, request):
    """
    Auto-patch environment settings for unit tests only.

    For integration tests, we use real environment variables from .env file.

    This ensures:
    1. Strong password that passes validation (unit tests only)
    2. Test mode enabled to bypass external connections (unit tests only)
    3. Consistent configuration across all unit tests
    """
    from ww.core.config import get_settings

    # Check if this is an integration test BEFORE clearing caches
    markers = [m.name for m in request.node.iter_markers()]
    is_integration = "integration" in markers

    # Clear Settings cache to ensure fresh settings for each test
    get_settings.cache_clear()

    # For integration tests, DON'T clear gateway services - neo4j driver
    # connections get confused with different event loops if we reinitialize
    # For unit tests, clear everything to ensure isolation
    if not is_integration:
        try:
            from ww.core.services import reset_services
            reset_services()
        except ImportError:
            pass

        # Also clear storage instance caches to prevent test pollution
        try:
            from ww.storage.neo4j_store import _neo4j_instances
            _neo4j_instances.clear()
        except ImportError:
            pass

        try:
            from ww.storage.qdrant_store import _qdrant_instances
            _qdrant_instances.clear()
        except ImportError:
            pass
    else:
        # For integration tests, just reset the rate limiter
        pass  # Rate limiter reset not needed in v0.2.0+

    # Skip env patching for integration tests - they use real credentials
    if is_integration:
        return

    monkeypatch.setenv("WW_NEO4J_PASSWORD", "TestPassword123!")
    monkeypatch.setenv("NEO4J_PASSWORD", "TestPassword123!")
    monkeypatch.setenv("WW_TEST_MODE", "true")
    monkeypatch.setenv("WW_SESSION_ID", "test-session")


@pytest.fixture(scope="session")
def event_loop_policy():
    """Return the event loop policy to use."""
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture(scope="session")
async def session_event_loop():
    """
    Create a session-scoped event loop for integration tests.

    This provides a single event loop for all integration tests to share,
    which is critical for Neo4j's async driver that maintains connection
    pools across requests.
    """
    yield asyncio.get_event_loop()


@pytest.fixture(scope="function")
def test_session_id():
    """Generate a unique test session ID."""
    return f"test-{uuid.uuid4().hex[:8]}"


# ============================================================================
# Infrastructure Availability Checks
# ============================================================================

@pytest.fixture(scope="session")
def qdrant_available():
    """Check if Qdrant is available and healthy (including write capability)."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        import uuid

        client = QdrantClient(url="http://localhost:6333", timeout=5)

        # Try to create/recreate a test collection and write to it
        test_collection = "_ww_health_check_test"
        try:
            # Delete if exists
            try:
                client.delete_collection(test_collection)
            except Exception:
                pass

            # Create test collection
            client.create_collection(
                collection_name=test_collection,
                vectors_config=VectorParams(size=8, distance=Distance.COSINE)
            )

            # Try write operation - this will fail if RocksDB is corrupt
            client.upsert(
                collection_name=test_collection,
                points=[PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[0.1] * 8,
                    payload={"test": True}
                )]
            )

            # Cleanup
            client.delete_collection(test_collection)
            return True

        except Exception:
            # Write failed - RocksDB or storage is corrupt
            try:
                client.delete_collection(test_collection)
            except Exception:
                pass
            return False

    except Exception:
        return False


@pytest.fixture(scope="session")
def neo4j_available():
    """Check if Neo4j is available and healthy."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", os.environ.get("WW_NEO4J_PASSWORD", "neo4j")),
        )
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_integration_if_unavailable(request, qdrant_available, neo4j_available):
    """Skip integration tests when infrastructure is unavailable."""
    markers = [m.name for m in request.node.iter_markers()]
    if "integration" in markers:
        if not qdrant_available:
            pytest.skip("Qdrant not available - skipping integration test")
        if not neo4j_available:
            pytest.skip("Neo4j not available - skipping integration test")


# ============================================================================
# Mock Fixtures for Storage Backends
# ============================================================================

@pytest_asyncio.fixture(scope="function")
async def mock_qdrant_store():
    """
    Mock Qdrant vector store for unit tests.

    Returns a mock that can be configured per-test with reasonable defaults
    for typical vector store operations.

    Mocked methods:
    - initialize: async
    - add: async, returns None
    - search: async, returns empty list by default
    - delete: async, returns None
    - count: async, returns 0
    - close: async
    - upsert: async, returns None
    """
    mock = MagicMock()
    mock.initialize = AsyncMock()
    mock.add = AsyncMock(return_value=None)
    mock.search = AsyncMock(return_value=[])
    mock.delete = AsyncMock(return_value=None)
    mock.count = AsyncMock(return_value=0)
    mock.close = AsyncMock()
    mock.upsert = AsyncMock(return_value=None)
    mock.get = AsyncMock(return_value=[])
    mock.update_payload = AsyncMock(return_value=None)
    mock.scroll = AsyncMock(return_value=([], None))  # Added for pagination tests
    mock.batch_update_payloads = AsyncMock(return_value=1)  # Added for batch access tracking

    return mock


@pytest_asyncio.fixture(scope="function")
async def mock_neo4j_store():
    """
    Mock Neo4j graph store for unit tests.

    Returns a mock that can be configured per-test with reasonable defaults
    for typical graph database operations.

    Mocked methods:
    - initialize: async
    - query: async, returns empty list by default
    - create_node: async, returns test-id
    - get_node: async, returns None by default
    - delete_node: async
    - create_relationship: async
    - get_relationships: async, returns empty list by default
    - get_relationships_batch: async, returns empty dict by default
    - close: async
    """
    mock = MagicMock()
    mock.initialize = AsyncMock()
    mock.query = AsyncMock(return_value=[])
    mock.create_node = AsyncMock(return_value="test-id")
    mock.get_node = AsyncMock(return_value=None)
    mock.update_node = AsyncMock(return_value=None)  # Added for saga tests
    mock.delete_node = AsyncMock()
    mock.create_relationship = AsyncMock()
    mock.get_relationships = AsyncMock(return_value=[])
    mock.get_relationships_batch = AsyncMock(return_value={})
    mock.update_property = AsyncMock(return_value=None)  # Added for saga tests
    mock.close = AsyncMock()

    return mock


@pytest_asyncio.fixture(scope="function")
async def mock_embedding_provider():
    """
    Mock embedding provider for unit tests.

    Returns embeddings of the correct dimension (1024 for BGE-M3).

    Mocked methods:
    - embed_query: async, returns 1024-dim vector
    - embed_documents: async, returns list of 1024-dim vectors
    """
    mock = MagicMock()
    mock.embed_query = AsyncMock(return_value=[0.1] * 1024)
    mock.embed_documents = AsyncMock(return_value=[[0.1] * 1024])

    return mock


# ============================================================================
# Memory Service Fixtures
# ============================================================================

@pytest_asyncio.fixture(scope="function")
async def mock_episodic_memory(mock_qdrant_store, mock_neo4j_store, mock_embedding_provider, test_session_id):
    """
    Provide mocked episodic memory service with all dependencies.

    Useful for testing episodic memory in isolation.
    """
    from ww.memory.episodic import EpisodicMemory

    # Patch the storage getters
    with patch('ww.memory.episodic.get_qdrant_store', return_value=mock_qdrant_store), \
         patch('ww.memory.episodic.get_neo4j_store', return_value=mock_neo4j_store), \
         patch('ww.memory.episodic.get_embedding_provider', return_value=mock_embedding_provider):

        memory = EpisodicMemory(session_id=test_session_id)
        await memory.initialize()
        return memory


@pytest_asyncio.fixture(scope="function")
async def mock_semantic_memory(mock_qdrant_store, mock_neo4j_store, mock_embedding_provider, test_session_id):
    """
    Provide mocked semantic memory service with all dependencies.

    Useful for testing semantic memory in isolation.
    """
    from ww.memory.semantic import SemanticMemory

    with patch('ww.memory.semantic.get_qdrant_store', return_value=mock_qdrant_store), \
         patch('ww.memory.semantic.get_neo4j_store', return_value=mock_neo4j_store), \
         patch('ww.memory.semantic.get_embedding_provider', return_value=mock_embedding_provider):

        memory = SemanticMemory(session_id=test_session_id)
        await memory.initialize()
        return memory


@pytest_asyncio.fixture(scope="function")
async def mock_procedural_memory(mock_qdrant_store, mock_neo4j_store, mock_embedding_provider, test_session_id):
    """
    Provide mocked procedural memory service with all dependencies.

    Useful for testing procedural memory in isolation.
    """
    from ww.memory.procedural import ProceduralMemory

    with patch('ww.memory.procedural.get_qdrant_store', return_value=mock_qdrant_store), \
         patch('ww.memory.procedural.get_neo4j_store', return_value=mock_neo4j_store), \
         patch('ww.memory.procedural.get_embedding_provider', return_value=mock_embedding_provider):

        memory = ProceduralMemory(session_id=test_session_id)
        await memory.initialize()
        return memory


@pytest_asyncio.fixture(scope="function")
async def all_memory_services(mock_episodic_memory, mock_semantic_memory, mock_procedural_memory, test_session_id):
    """
    Provide all three memory services with consistent session ID.

    Useful for testing cross-memory interactions.
    """
    return {
        "episodic": mock_episodic_memory,
        "semantic": mock_semantic_memory,
        "procedural": mock_procedural_memory,
        "session_id": test_session_id,
    }


# ============================================================================
# Mock Response Builders (for realistic test data)
# ============================================================================

@pytest.fixture
def mock_search_result():
    """Factory for creating mock vector search results."""
    def _build(id: str = "test-id", score: float = 0.95, payload: dict = None):
        return {
            "id": id,
            "score": score,
            "payload": payload or {
                "session_id": "test-session",
                "timestamp": datetime.now().isoformat(),
                "content": "Test content",
            }
        }
    return _build


@pytest.fixture
def mock_graph_node():
    """Factory for creating mock Neo4j nodes."""
    def _build(id: str = "node-1", label: str = "Entity", properties: dict = None):
        return {
            "id": id,
            "label": label,
            "properties": properties or {
                "name": "Test Entity",
                "summary": "Test summary",
                "stability": 1.0,
                "accessCount": 1,
            }
        }
    return _build


@pytest.fixture
def mock_graph_relationship():
    """Factory for creating mock Neo4j relationships."""
    def _build(source_id: str = "node-1", target_id: str = "node-2",
               rel_type: str = "RELATES_TO", weight: float = 0.5):
        return {
            "source_id": source_id,
            "target_id": target_id,
            "type": rel_type,
            "weight": weight,
            "coAccessCount": 1,
        }
    return _build


# ============================================================================
# Configuration and Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_settings():
    """Provide mock settings for testing."""
    import os
    from ww.core.config import Settings

    # Set test mode to bypass validation
    os.environ["WW_TEST_MODE"] = "true"

    settings = Settings(
        session_id="test-session",
        fsrs_default_stability=1.0,
        fsrs_target_retention=0.9,
        hebbian_learning_rate=0.1,
        hebbian_initial_weight=0.1,
        actr_decay=0.5,
        actr_threshold=-3.0,
        actr_noise=0.0,
        retrieval_semantic_weight=0.4,
        retrieval_activation_weight=0.35,
        retrieval_retrievability_weight=0.25,
        neo4j_password="TestPass123!",  # Strong password for tests
    )

    return settings


@pytest.fixture
def patch_settings(mock_settings):
    """Patch get_settings() with mock settings."""
    with patch('ww.core.config.get_settings', return_value=mock_settings):
        yield mock_settings


# ============================================================================
# Cleanup and Isolation
# ============================================================================

@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio for anyio tests."""
    return "asyncio"


@pytest_asyncio.fixture(autouse=True, scope="function")
async def cleanup_after_test(test_session_id, request):
    """Clean up any test resources after each test."""
    yield

    # For integration tests, properly close connections to prevent loop issues
    markers = [m.name for m in request.node.iter_markers()]
    if "integration" in markers:
        try:
            from ww.core.services import cleanup_services
            await cleanup_services()
        except Exception as e:
            # Log but don't fail - cleanup is best effort
            import logging
            logging.getLogger(__name__).debug(f"Cleanup error (ignored): {e}")


# ============================================================================
# Test Markers and Configuration
# ============================================================================

@pytest.fixture
def slow_marker():
    """Mark test as slow."""
    def _mark(test_func):
        return pytest.mark.slow(test_func)
    return _mark


def pytest_collection_modifyitems(config, items):
    """Add slow marker to tests with 'slow' in their name."""
    for item in items:
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
