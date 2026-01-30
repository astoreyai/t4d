"""
Pytest configuration for integration tests.

Provides fixtures for setting up real or realistic mock stores
for end-to-end testing of World Weaver memory system.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4


# ============================================================================
# Session Fixtures
# ============================================================================

@pytest.fixture
def integration_session_id():
    """Generate unique session ID for integration test."""
    return f"int-{uuid4().hex[:8]}"


@pytest.fixture
def integration_session_a():
    """Session A for isolation tests."""
    return f"session-a-{uuid4().hex[:6]}"


@pytest.fixture
def integration_session_b():
    """Session B for isolation tests."""
    return f"session-b-{uuid4().hex[:6]}"


# ============================================================================
# Mock Store Fixtures for Integration Tests
# ============================================================================

@pytest_asyncio.fixture
async def integration_embedding_provider():
    """
    Mock embedding provider for integration tests.

    Returns consistent 1024-dim vectors (BGE-M3 standard).
    """
    mock = AsyncMock()
    mock.embed_query = AsyncMock(return_value=[0.1] * 1024)
    mock.embed_documents = AsyncMock(
        side_effect=lambda docs: [[0.1 + i / 1000 for i in range(1024)] for _ in docs]
    )
    return mock


@pytest_asyncio.fixture
async def integration_qdrant_store():
    """
    Mock Qdrant vector store for integration tests.

    Simulates real Qdrant behavior with:
    - Vector storage tracking
    - Payload management
    - Search result simulation
    """
    mock = AsyncMock()

    # Collection names
    mock.episodes_collection = "test_episodes"
    mock.entities_collection = "test_entities"
    mock.procedures_collection = "test_procedures"

    # Storage simulation
    mock._storage = {
        "test_episodes": {},
        "test_entities": {},
        "test_procedures": {},
    }

    async def mock_add(collection, ids, vectors, payloads):
        """Simulate Qdrant add operation."""
        for id_val, vector, payload in zip(ids, vectors, payloads):
            mock._storage[collection][id_val] = {
                "vector": vector,
                "payload": payload,
            }

    async def mock_search(collection, vector, limit, filter=None):
        """Simulate Qdrant search operation."""
        results = []
        for id_val, data in list(mock._storage[collection].items())[:limit]:
            # Check filter
            if filter:
                payload = data["payload"]
                matches = all(payload.get(k) == v for k, v in filter.items())
                if not matches:
                    continue

            # Mock similarity score
            similarity = 0.95 - len(results) * 0.05
            results.append((id_val, similarity, data["payload"]))

        return results

    async def mock_get(collection, ids):
        """Simulate Qdrant get operation."""
        results = []
        for id_val in ids:
            if id_val in mock._storage[collection]:
                data = mock._storage[collection][id_val]
                results.append((id_val, data["payload"]))
        return results

    async def mock_update_payload(collection, id_val, payload):
        """Simulate Qdrant update operation."""
        if id_val in mock._storage[collection]:
            mock._storage[collection][id_val]["payload"].update(payload)

    # Set up mock methods
    mock.initialize = AsyncMock()
    mock.add = AsyncMock(side_effect=mock_add)
    mock.search = AsyncMock(side_effect=mock_search)
    mock.get = AsyncMock(side_effect=mock_get)
    mock.update_payload = AsyncMock(side_effect=mock_update_payload)
    mock.delete = AsyncMock()
    mock.close = AsyncMock()

    return mock


@pytest_asyncio.fixture
async def integration_neo4j_store():
    """
    Mock Neo4j graph store for integration tests.

    Simulates real Neo4j behavior with:
    - Node storage
    - Relationship tracking
    - Property updates
    """
    mock = AsyncMock()

    # Storage simulation
    mock._nodes = {}
    mock._relationships = {}

    async def mock_create_node(label, properties):
        """Simulate Neo4j create_node."""
        node_id = str(uuid4())
        mock._nodes[node_id] = {
            "label": label,
            "properties": properties,
        }
        return node_id

    async def mock_create_relationship(source_id, target_id, rel_type, properties=None):
        """Simulate Neo4j create_relationship."""
        rel_key = f"{source_id}-{rel_type}-{target_id}"
        mock._relationships[rel_key] = {
            "source": source_id,
            "target": target_id,
            "type": rel_type,
            "properties": properties or {},
        }

    async def mock_get_relationships(source_id, rel_type=None):
        """Simulate Neo4j get_relationships."""
        results = []
        for rel_key, rel in mock._relationships.items():
            if rel["source"] == source_id:
                if rel_type is None or rel["type"] == rel_type:
                    results.append(rel)
        return results

    async def mock_update_node(node_id, properties, label):
        """Simulate Neo4j update_node."""
        if node_id in mock._nodes:
            mock._nodes[node_id]["properties"].update(properties)

    # Set up mock methods
    mock.initialize = AsyncMock()
    mock.create_node = AsyncMock(side_effect=mock_create_node)
    mock.create_relationship = AsyncMock(side_effect=mock_create_relationship)
    mock.get_relationships = AsyncMock(side_effect=mock_get_relationships)
    mock.update_node = AsyncMock(side_effect=mock_update_node)
    mock.query = AsyncMock(return_value=[])
    mock.close = AsyncMock()

    return mock


# ============================================================================
# Patched Fixture Combinations
# ============================================================================

@pytest_asyncio.fixture
async def episodic_memory_with_mocks(
    integration_embedding_provider,
    integration_qdrant_store,
    integration_neo4j_store,
    integration_session_id,
):
    """
    Create EpisodicMemory with all dependencies mocked.

    Usage:
        async def test_example(episodic_memory_with_mocks):
            episodic = episodic_memory_with_mocks
            await episodic.initialize()
    """
    from unittest.mock import patch
    from ww.memory.episodic import EpisodicMemory

    with patch(
        "ww.memory.episodic.get_embedding_provider",
        return_value=integration_embedding_provider,
    ):
        with patch(
            "ww.memory.episodic.get_qdrant_store",
            return_value=integration_qdrant_store,
        ):
            with patch(
                "ww.memory.episodic.get_neo4j_store",
                return_value=integration_neo4j_store,
            ):
                return EpisodicMemory(integration_session_id)


@pytest_asyncio.fixture
async def semantic_memory_with_mocks(
    integration_embedding_provider,
    integration_qdrant_store,
    integration_neo4j_store,
    integration_session_id,
):
    """
    Create SemanticMemory with all dependencies mocked.

    Usage:
        async def test_example(semantic_memory_with_mocks):
            semantic = semantic_memory_with_mocks
            await semantic.initialize()
    """
    from unittest.mock import patch
    from ww.memory.semantic import SemanticMemory

    with patch(
        "ww.memory.semantic.get_embedding_provider",
        return_value=integration_embedding_provider,
    ):
        with patch(
            "ww.memory.semantic.get_qdrant_store",
            return_value=integration_qdrant_store,
        ):
            with patch(
                "ww.memory.semantic.get_neo4j_store",
                return_value=integration_neo4j_store,
            ):
                return SemanticMemory(integration_session_id)


@pytest_asyncio.fixture
async def procedural_memory_with_mocks(
    integration_embedding_provider,
    integration_qdrant_store,
    integration_neo4j_store,
    integration_session_id,
):
    """
    Create ProceduralMemory with all dependencies mocked.

    Usage:
        async def test_example(procedural_memory_with_mocks):
            procedural = procedural_memory_with_mocks
            await procedural.initialize()
    """
    from unittest.mock import patch
    from ww.memory.procedural import ProceduralMemory

    with patch(
        "ww.memory.procedural.get_embedding_provider",
        return_value=integration_embedding_provider,
    ):
        with patch(
            "ww.memory.procedural.get_qdrant_store",
            return_value=integration_qdrant_store,
        ):
            with patch(
                "ww.memory.procedural.get_neo4j_store",
                return_value=integration_neo4j_store,
            ):
                return ProceduralMemory(integration_session_id)


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest_asyncio.fixture(autouse=True)
async def cleanup_integration_test():
    """
    Auto-cleanup after each integration test.

    Ensures no lingering connections or resources.
    """
    yield

    # Cleanup code runs after test
    # (Actual store cleanup handled by individual test fixtures)
    pass


@pytest.fixture(scope="function")
def integration_test_timeout():
    """
    Timeout for integration tests (default: 30 seconds).

    Integration tests may be slower than unit tests.
    """
    return 30
