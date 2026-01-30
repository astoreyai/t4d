"""
Unit tests for saga integration into memory services.

Tests verify that create/update/delete operations are atomic across
Qdrant and Neo4j stores, with proper compensation on failure.
"""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from ww.memory.episodic import EpisodicMemory
from ww.memory.semantic import SemanticMemory
from ww.memory.procedural import ProceduralMemory
from ww.storage.saga import SagaState


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_stores():
    """Create mock Qdrant and Neo4j stores."""
    qdrant = AsyncMock()
    qdrant.add = AsyncMock(return_value=None)
    qdrant.delete = AsyncMock(return_value=None)
    qdrant.update_payload = AsyncMock(return_value=None)
    qdrant.episodes_collection = "episodes"
    qdrant.entities_collection = "entities"
    qdrant.procedures_collection = "procedures"

    neo4j = AsyncMock()
    neo4j.create_node = AsyncMock(return_value={"id": "test-node"})
    neo4j.delete_node = AsyncMock(return_value=None)
    neo4j.update_node = AsyncMock(return_value=None)

    return qdrant, neo4j


@pytest.fixture
def mock_embedding():
    """Create mock embedding provider."""
    embedding = AsyncMock()
    embedding.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return embedding


# ============================================================================
# Episodic Memory Saga Tests
# ============================================================================

@pytest.mark.asyncio
async def test_episodic_create_success(mock_stores, mock_embedding):
    """Test successful episodic memory creation with saga."""
    qdrant, neo4j = mock_stores

    with patch("ww.memory.episodic.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.episodic.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding):

        memory = EpisodicMemory(session_id="test-session")
        # Disable FF encoding and gating for predictable test behavior
        # (P7.1 FF bridge modifies valence based on novelty detection)
        memory._ff_encoding_enabled = False
        memory._gating_enabled = False

        episode = await memory.create(
            content="Test episode content",
            outcome="success",
            valence=0.8,
        )

        # Verify both stores were called
        qdrant.add.assert_called_once()
        neo4j.create_node.assert_called_once()

        # Verify episode was created
        assert episode.content == "Test episode content"
        assert episode.outcome.value == "success"
        assert episode.emotional_valence == 0.8


@pytest.mark.asyncio
async def test_episodic_create_neo4j_failure_rollback(mock_stores, mock_embedding):
    """Test episodic creation with Neo4j failure triggers Qdrant rollback."""
    qdrant, neo4j = mock_stores
    neo4j.create_node.side_effect = RuntimeError("Neo4j connection error")

    with patch("ww.memory.episodic.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.episodic.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding):

        memory = EpisodicMemory(session_id="test-session")

        with pytest.raises(Exception):  # Saga raises CompensationError by default
            await memory.create(content="Test content")

        # Qdrant should have been called and then compensated (deleted)
        qdrant.add.assert_called_once()
        qdrant.delete.assert_called_once()


@pytest.mark.asyncio
async def test_episodic_mark_important_success(mock_stores, mock_embedding):
    """Test marking episode important updates both stores atomically."""
    qdrant, neo4j = mock_stores

    # Use a real UUID
    episode_id = uuid4()

    # Mock get to return existing episode
    qdrant.get = AsyncMock(return_value=[
        (str(episode_id), {
            "session_id": "test-session",
            "content": "Test",
            "timestamp": "2025-11-27T12:00:00",
            "ingested_at": "2025-11-27T12:00:00",
            "context": {},
            "outcome": "neutral",
            "emotional_valence": 0.5,
            "access_count": 1,
            "last_accessed": "2025-11-27T12:00:00",
            "stability": 3.0,
        })
    ])

    with patch("ww.memory.episodic.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.episodic.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding):

        memory = EpisodicMemory(session_id="test-session")

        episode = await memory.mark_important(episode_id, new_valence=0.9)

        # Verify both stores were updated
        qdrant.update_payload.assert_called_once()
        neo4j.update_node.assert_called_once()

        # Verify valence was updated
        assert episode.emotional_valence == 0.9


@pytest.mark.asyncio
async def test_episodic_mark_important_rollback(mock_stores, mock_embedding):
    """Test mark important with Neo4j failure rolls back Qdrant update."""
    qdrant, neo4j = mock_stores
    neo4j.update_node.side_effect = RuntimeError("Update failed")

    # Use a real UUID
    episode_id = uuid4()

    # Mock get to return existing episode
    qdrant.get = AsyncMock(return_value=[
        (str(episode_id), {
            "session_id": "test-session",
            "content": "Test",
            "timestamp": "2025-11-27T12:00:00",
            "ingested_at": "2025-11-27T12:00:00",
            "context": {},
            "outcome": "neutral",
            "emotional_valence": 0.5,
            "access_count": 1,
            "last_accessed": "2025-11-27T12:00:00",
            "stability": 3.0,
        })
    ])

    with patch("ww.memory.episodic.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.episodic.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding):

        memory = EpisodicMemory(session_id="test-session")

        with pytest.raises(RuntimeError):
            await memory.mark_important(episode_id, new_valence=0.9)

        # Qdrant should have been updated then compensated (reverted to old value)
        assert qdrant.update_payload.call_count == 2  # Action + compensation


# ============================================================================
# Semantic Memory Saga Tests
# ============================================================================

@pytest.mark.asyncio
async def test_semantic_create_entity_success(mock_stores, mock_embedding):
    """Test successful semantic entity creation with saga."""
    qdrant, neo4j = mock_stores

    with patch("ww.memory.semantic.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.semantic.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.semantic.get_embedding_provider", return_value=mock_embedding):

        memory = SemanticMemory(session_id="test-session")
        entity = await memory.create_entity(
            name="Test Entity",
            entity_type="CONCEPT",
            summary="A test concept",
            details="Detailed description",
        )

        # Verify both stores were called
        qdrant.add.assert_called_once()
        neo4j.create_node.assert_called_once()

        # Verify entity was created
        assert entity.name == "Test Entity"
        assert entity.entity_type.value == "CONCEPT"


@pytest.mark.asyncio
async def test_semantic_create_entity_rollback(mock_stores, mock_embedding):
    """Test entity creation with Neo4j failure triggers rollback."""
    qdrant, neo4j = mock_stores
    neo4j.create_node.side_effect = RuntimeError("Neo4j error")

    with patch("ww.memory.semantic.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.semantic.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.semantic.get_embedding_provider", return_value=mock_embedding):

        memory = SemanticMemory(session_id="test-session")

        with pytest.raises(Exception):
            await memory.create_entity(
                name="Test Entity",
                entity_type="CONCEPT",
                summary="A test concept",
            )

        # Qdrant should have been compensated
        qdrant.add.assert_called_once()
        qdrant.delete.assert_called_once()


@pytest.mark.asyncio
async def test_semantic_supersede_success(mock_stores, mock_embedding):
    """Test superseding entity updates both stores atomically."""
    qdrant, neo4j = mock_stores

    # Use a real UUID
    entity_id = uuid4()

    # Mock get to return existing entity
    qdrant.get = AsyncMock(return_value=[
        (str(entity_id), {
            "session_id": "test-session",
            "name": "Old Entity",
            "entity_type": "CONCEPT",
            "summary": "Old summary",
            "details": None,
            "source": None,
            "stability": 3.0,
            "access_count": 1,
            "last_accessed": "2025-11-27T12:00:00",
            "created_at": "2025-11-27T12:00:00",
            "valid_from": "2025-11-27T12:00:00",
            "valid_to": None,
        })
    ])

    with patch("ww.memory.semantic.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.semantic.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.semantic.get_embedding_provider", return_value=mock_embedding):

        memory = SemanticMemory(session_id="test-session")

        new_entity = await memory.supersede(
            entity_id,
            new_summary="New summary",
            new_details="New details",
        )

        # Verify old entity validity was closed in both stores
        assert neo4j.update_node.call_count >= 1
        assert qdrant.update_payload.call_count >= 1

        # Verify new entity was created
        qdrant.add.assert_called()
        neo4j.create_node.assert_called()


# ============================================================================
# Procedural Memory Saga Tests
# ============================================================================

@pytest.mark.asyncio
async def test_procedural_create_skill_success(mock_stores, mock_embedding):
    """Test successful procedural skill creation with saga."""
    qdrant, neo4j = mock_stores

    with patch("ww.memory.procedural.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.procedural.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.procedural.get_embedding_provider", return_value=mock_embedding):

        memory = ProceduralMemory(session_id="test-session")

        trajectory = [
            {"action": "Read file", "tool": "Read", "parameters": {}, "result": "success"},
            {"action": "Write file", "tool": "Write", "parameters": {}, "result": "success"},
        ]

        procedure = await memory.create_skill(
            trajectory=trajectory,
            outcome_score=0.9,
            domain="coding",
            name="Test Procedure",
        )

        # Verify both stores were called
        qdrant.add.assert_called_once()
        neo4j.create_node.assert_called_once()

        # Verify procedure was created
        assert procedure.name == "Test Procedure"
        assert procedure.domain.value == "coding"
        assert len(procedure.steps) == 2


@pytest.mark.asyncio
async def test_procedural_create_skill_rollback(mock_stores, mock_embedding):
    """Test skill creation with Neo4j failure triggers rollback."""
    qdrant, neo4j = mock_stores
    neo4j.create_node.side_effect = RuntimeError("Neo4j error")

    with patch("ww.memory.procedural.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.procedural.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.procedural.get_embedding_provider", return_value=mock_embedding):

        memory = ProceduralMemory(session_id="test-session")

        trajectory = [
            {"action": "Test action", "tool": "Test", "parameters": {}, "result": "success"},
        ]

        with pytest.raises(Exception):
            await memory.create_skill(
                trajectory=trajectory,
                outcome_score=0.9,
                domain="coding",
            )

        # Qdrant should have been compensated
        qdrant.add.assert_called_once()
        qdrant.delete.assert_called_once()


@pytest.mark.asyncio
async def test_procedural_update_success(mock_stores, mock_embedding):
    """Test updating procedure execution stats atomically."""
    qdrant, neo4j = mock_stores

    # Use a real UUID
    procedure_id = uuid4()

    # Mock get to return existing procedure
    qdrant.get = AsyncMock(return_value=[
        (str(procedure_id), {
            "session_id": "test-session",
            "name": "Test Procedure",
            "domain": "coding",
            "trigger_pattern": "test",
            "steps": [],
            "script": "test script",
            "success_rate": 0.8,
            "execution_count": 5,
            "last_executed": "2025-11-27T12:00:00",
            "version": 1,
            "deprecated": False,
            "consolidated_into": None,
            "created_at": "2025-11-27T12:00:00",
            "created_from": "trajectory",
        })
    ])

    with patch("ww.memory.procedural.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.procedural.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.procedural.get_embedding_provider", return_value=mock_embedding):

        memory = ProceduralMemory(session_id="test-session")

        procedure = await memory.update(procedure_id, success=True)

        # Verify both stores were updated
        qdrant.update_payload.assert_called_once()
        neo4j.update_node.assert_called_once()


@pytest.mark.asyncio
async def test_procedural_update_rollback(mock_stores, mock_embedding):
    """Test procedure update with Neo4j failure rolls back Qdrant."""
    qdrant, neo4j = mock_stores
    neo4j.update_node.side_effect = RuntimeError("Update failed")

    # Use a real UUID
    procedure_id = uuid4()

    # Mock get to return existing procedure
    qdrant.get = AsyncMock(return_value=[
        (str(procedure_id), {
            "session_id": "test-session",
            "name": "Test Procedure",
            "domain": "coding",
            "trigger_pattern": "test",
            "steps": [],
            "script": "test script",
            "success_rate": 0.8,
            "execution_count": 5,
            "last_executed": "2025-11-27T12:00:00",
            "version": 1,
            "deprecated": False,
            "consolidated_into": None,
            "created_at": "2025-11-27T12:00:00",
            "created_from": "trajectory",
        })
    ])

    with patch("ww.memory.procedural.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.procedural.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.procedural.get_embedding_provider", return_value=mock_embedding):

        memory = ProceduralMemory(session_id="test-session")

        with pytest.raises(RuntimeError):
            await memory.update(procedure_id, success=True)

        # Qdrant should have been updated then compensated
        assert qdrant.update_payload.call_count == 2


@pytest.mark.asyncio
async def test_procedural_deprecate_success(mock_stores, mock_embedding):
    """Test deprecating procedure updates both stores atomically."""
    qdrant, neo4j = mock_stores
    from datetime import datetime

    with patch("ww.memory.procedural.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.procedural.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.procedural.get_embedding_provider", return_value=mock_embedding):

        memory = ProceduralMemory(session_id="test-session")
        procedure_id = uuid4()
        now = datetime.now()

        # RACE-006 FIX: Mock get_procedure to return non-deprecated procedure
        qdrant.get.return_value = [
            (
                str(procedure_id),
                {
                    "name": "Test Skill",
                    "domain": "coding",
                    "trigger_pattern": "test",
                    "steps": [],
                    "script": None,
                    "success_rate": 0.8,
                    "execution_count": 5,
                    "last_executed": now.isoformat(),
                    "version": 1,
                    "deprecated": False,  # Not yet deprecated
                    "consolidated_into": None,
                    "created_at": now.isoformat(),
                    "created_from": "trajectory",
                },
            )
        ]

        await memory.deprecate(procedure_id, reason="Outdated")

        # Verify both stores were updated
        qdrant.update_payload.assert_called_once()
        neo4j.update_node.assert_called_once()


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.asyncio
async def test_idempotent_create_handling(mock_stores, mock_embedding):
    """Test handling of duplicate create (vector already exists)."""
    qdrant, neo4j = mock_stores

    # First call succeeds
    qdrant.add.side_effect = [None, RuntimeError("Already exists")]

    with patch("ww.memory.episodic.get_qdrant_store", return_value=qdrant), \
         patch("ww.memory.episodic.get_neo4j_store", return_value=neo4j), \
         patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding):

        memory = EpisodicMemory(session_id="test-session")

        # First create should succeed
        episode1 = await memory.create(content="Test 1")
        assert episode1 is not None

        # Second create should handle duplicate gracefully
        with pytest.raises(Exception):
            await memory.create(content="Test 2")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
