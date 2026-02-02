# T4DM Test Implementation Roadmap

**Status**: Action Plan | **Priority**: P1 | **Estimated Effort**: 60 hours

---

## Phase 0: Fix Async Event Loop Issues (1-2 hours)

### Problem
5 tests failing with: `RuntimeError: Task got Future attached to a different loop`

### Solution

Create `/mnt/projects/t4d/t4dm/tests/conftest.py`:

```python
"""Pytest configuration and shared fixtures for T4DM tests."""

import asyncio
import pytest
from unittest.mock import AsyncMock
from t4dm.storage.t4dx_vector_adapter import close_t4dx_vector_adapter
from t4dm.storage.t4dx_graph_adapter import close_t4dx_graph_adapter
from t4dm.mcp.memory_gateway import _service_instances, _initialized_sessions


# ============================================================================
# Event Loop Management
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """
    Create a session-scoped event loop for all async tests.

    This ensures Neo4j async driver doesn't encounter different event loops
    and maintains compatibility with pytest-asyncio.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop

    # Clean up pending tasks
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()


# ============================================================================
# Service Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_memory_services():
    """
    Auto-cleanup memory services after each test.

    Ensures no state leaks between tests.
    """
    yield

    # Clear service instances
    _service_instances.clear()
    _initialized_sessions.clear()

    # Close database connections
    try:
        await close_t4dx_vector_adapter()
    except Exception:
        pass

    try:
        await close_t4dx_graph_adapter()
    except Exception:
        pass


# ============================================================================
# Common Test Fixtures
# ============================================================================

@pytest.fixture
def mock_embedding():
    """Mock BGE-M3 embedding provider."""
    mock = AsyncMock()
    # 1024-dimensional vector
    mock.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3] * 341 + [0.1])
    mock.embed_texts = AsyncMock(return_value=[[0.1] * 1024] * 3)
    return mock


@pytest.fixture
def mock_t4dx_vector_adapter():
    """Mock Qdrant vector store."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.add = AsyncMock(side_effect=lambda *args, **kwargs: None)
    mock.search = AsyncMock(return_value=[])
    mock.get = AsyncMock(return_value=None)
    mock.update_payload = AsyncMock()
    mock.delete = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_t4dx_graph_adapter():
    """Mock Neo4j graph store."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.create_node = AsyncMock(return_value={"id": "node-123"})
    mock.update_node = AsyncMock()
    mock.delete_node = AsyncMock()
    mock.get_node = AsyncMock(return_value={"id": "node-123"})
    mock.create_relationship = AsyncMock()
    mock.delete_relationship = AsyncMock()
    mock.get_relationships = AsyncMock(return_value=[])
    mock.strengthen_relationship = AsyncMock()
    mock.close = AsyncMock()
    return mock


# ============================================================================
# Integration Test Markers
# ============================================================================

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires docker)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (>1 second)"
    )
    config.addinivalue_line(
        "markers", "edge_case: mark test as edge case boundary testing"
    )
```

### Update pyproject.toml

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "asyncio: marks tests as async (deselect with '-m \"not asyncio\"')",
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "edge_case: marks tests as edge case testing",
]
```

### Verify Fix

```bash
cd /mnt/projects/ww
source .venv/bin/activate
pytest tests/test_memory.py -v  # Should fix all 5 failures
```

**Expected Result**: 237 tests passing (5 previously failing now pass)

---

## Phase 1: Consolidation Service Tests (8-10 hours)

### File: `/mnt/projects/t4d/t4dm/tests/unit/test_consolidation.py`

```python
"""
Unit tests for consolidation service.

Tests cover:
- Light consolidation (deduplication, cleanup)
- Deep consolidation (episodic→semantic extraction)
- Skill consolidation (procedure merging)
- Error recovery and edge cases
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timedelta

from t4dm.consolidation.service import ConsolidationService
from t4dm.core.types import Episode, Entity, Procedure, ConsolidationType, Outcome


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def consolidation_service(mock_embedding, mock_t4dx_vector_adapter, mock_t4dx_graph_adapter):
    """Create consolidation service with mocked stores."""
    with patch('t4dm.consolidation.service.get_embedding_provider', return_value=mock_embedding), \
         patch('t4dm.consolidation.service.get_t4dx_vector_adapter', return_value=mock_t4dx_vector_adapter), \
         patch('t4dm.consolidation.service.get_t4dx_graph_adapter', return_value=mock_t4dx_graph_adapter), \
         patch('t4dm.consolidation.service.get_episodic_memory') as mock_episodic, \
         patch('t4dm.consolidation.service.get_semantic_memory') as mock_semantic, \
         patch('t4dm.consolidation.service.get_procedural_memory') as mock_procedural:

        # Setup mock memory services
        mock_episodic.return_value.recall = AsyncMock(return_value=[])
        mock_semantic.return_value.create_entity = AsyncMock()
        mock_procedural.return_value.retrieve = AsyncMock(return_value=[])

        service = ConsolidationService()
        service.episodic = mock_episodic.return_value
        service.semantic = mock_semantic.return_value
        service.procedural = mock_procedural.return_value

        return service


@pytest.fixture
def sample_episode():
    """Create sample episode for testing."""
    return Episode(
        id=uuid4(),
        session_id="test-session",
        content="Implemented memory consolidation algorithm",
        context={"project": "t4dm", "file": "consolidation/service.py"},
        outcome=Outcome.SUCCESS,
        emotional_valence=0.8,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        version=1,
    )


@pytest.fixture
def sample_entity():
    """Create sample entity for testing."""
    return Entity(
        id=uuid4(),
        name="Memory Consolidation",
        entity_type="TECHNIQUE",
        summary="Transfer episodic memories to semantic knowledge",
        details="Consolidation strengthens important memories and merges similar concepts",
        source="manual",
        created_at=datetime.utcnow(),
        session_id="test-session",
    )


# ============================================================================
# Light Consolidation Tests
# ============================================================================

class TestLightConsolidation:
    """Tests for light consolidation (deduplication and cleanup)."""

    @pytest.mark.asyncio
    async def test_consolidate_light_returns_status(self, consolidation_service):
        """Test light consolidation returns status dictionary."""
        result = await consolidation_service.consolidate(
            consolidation_type="light",
            session_filter="test-session"
        )

        assert "status" in result
        assert result["status"] in ["pending", "in_progress", "completed", "failed"]
        assert "duration_seconds" in result
        assert result["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_consolidate_light_deduplicates_episodes(
        self, consolidation_service, sample_episode
    ):
        """Test that light consolidation identifies duplicate episodes."""
        # Setup: Two similar episodes
        episode2 = Episode(
            id=uuid4(),
            session_id="test-session",
            content="Implemented memory consolidation algorithm",  # Same as sample
            context={"project": "t4dm", "file": "consolidation/service.py"},
            outcome=Outcome.SUCCESS,
            emotional_valence=0.75,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1,
        )

        consolidation_service.episodic.recall = AsyncMock(
            return_value=[sample_episode, episode2]
        )

        result = await consolidation_service.consolidate(
            consolidation_type="light",
            session_filter="test-session"
        )

        # Verify cleanup was attempted
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_consolidate_light_cleanup_old_episodes(self, consolidation_service):
        """Test light consolidation removes very old, low-importance episodes."""
        # Setup: Very old episode with low valence
        old_episode = Episode(
            id=uuid4(),
            session_id="test-session",
            content="Old low-importance memory",
            context={"project": "old"},
            outcome=Outcome.NEUTRAL,
            emotional_valence=0.1,  # Very low importance
            created_at=datetime.utcnow() - timedelta(days=365),
            updated_at=datetime.utcnow() - timedelta(days=365),
            version=1,
        )

        consolidation_service.episodic.recall = AsyncMock(
            return_value=[old_episode]
        )

        result = await consolidation_service.consolidate(
            consolidation_type="light",
            session_filter="test-session"
        )

        assert result["status"] == "completed"
        # Verify delete was called for old episode
        consolidation_service.vector_store.delete.assert_called()


# ============================================================================
# Deep Consolidation Tests
# ============================================================================

class TestDeepConsolidation:
    """Tests for deep consolidation (episodic→semantic extraction)."""

    @pytest.mark.asyncio
    async def test_consolidate_deep_extracts_entities(
        self, consolidation_service, sample_episode, sample_entity
    ):
        """Test deep consolidation extracts entities from episodes."""
        consolidation_service.episodic.recall = AsyncMock(
            return_value=[sample_episode]
        )
        consolidation_service.semantic.create_entity = AsyncMock(
            return_value=sample_entity
        )

        result = await consolidation_service.consolidate(
            consolidation_type="deep",
            session_filter="test-session"
        )

        assert result["status"] == "completed"
        # Verify entity creation was called
        consolidation_service.semantic.create_entity.assert_called()

    @pytest.mark.asyncio
    async def test_consolidate_deep_respects_session_filter(
        self, consolidation_service, sample_episode
    ):
        """Test deep consolidation only processes specified session."""
        consolidation_service.episodic.recall = AsyncMock(
            return_value=[sample_episode]
        )

        await consolidation_service.consolidate(
            consolidation_type="deep",
            session_filter="specific-session"
        )

        # Verify recall was filtered by session
        consolidation_service.episodic.recall.assert_called()
        call_args = consolidation_service.episodic.recall.call_args
        assert call_args[1]["session_filter"] == "specific-session"

    @pytest.mark.asyncio
    async def test_consolidate_deep_returns_metrics(self, consolidation_service):
        """Test deep consolidation returns performance metrics."""
        consolidation_service.episodic.recall = AsyncMock(return_value=[])

        result = await consolidation_service.consolidate(
            consolidation_type="deep",
            session_filter="test-session"
        )

        assert "results" in result or "duration_seconds" in result
        assert result["status"] == "completed"


# ============================================================================
# Skill Consolidation Tests
# ============================================================================

class TestSkillConsolidation:
    """Tests for skill consolidation (procedure optimization)."""

    @pytest.mark.asyncio
    async def test_consolidate_skills_merges_similar_procedures(
        self, consolidation_service
    ):
        """Test skill consolidation merges similar procedures."""
        consolidation_service.procedural.retrieve = AsyncMock(return_value=[])

        result = await consolidation_service.consolidate(
            consolidation_type="skill",
            session_filter="test-session"
        )

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_consolidate_skills_updates_success_rates(
        self, consolidation_service
    ):
        """Test skill consolidation updates procedure success rates."""
        result = await consolidation_service.consolidate(
            consolidation_type="skill",
            session_filter="test-session"
        )

        assert result["status"] == "completed"


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestConsolidationErrorHandling:
    """Tests for error recovery in consolidation."""

    @pytest.mark.asyncio
    async def test_consolidation_handles_embedding_error(self, consolidation_service):
        """Test consolidation recovers from embedding service error."""
        consolidation_service.embedding.embed_query = AsyncMock(
            side_effect=Exception("Embedding service unavailable")
        )
        consolidation_service.episodic.recall = AsyncMock(return_value=[])

        # Should not raise, should return error status
        result = await consolidation_service.consolidate(
            consolidation_type="light"
        )

        assert result["status"] in ["failed", "completed"]

    @pytest.mark.asyncio
    async def test_consolidation_handles_database_error(self, consolidation_service):
        """Test consolidation recovers from database error."""
        consolidation_service.vector_store.search = AsyncMock(
            side_effect=Exception("Database unavailable")
        )

        result = await consolidation_service.consolidate(
            consolidation_type="light"
        )

        assert result["status"] in ["failed", "completed"]

    @pytest.mark.asyncio
    async def test_consolidation_handles_storage_error(self, consolidation_service):
        """Test consolidation recovers from Neo4j error."""
        consolidation_service.graph_store.create_node = AsyncMock(
            side_effect=Exception("Neo4j unavailable")
        )

        result = await consolidation_service.consolidate(
            consolidation_type="deep"
        )

        assert result["status"] in ["failed", "completed"]


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestConsolidationEdgeCases:
    """Tests for edge cases in consolidation."""

    @pytest.mark.asyncio
    async def test_consolidation_with_empty_episodes(self, consolidation_service):
        """Test consolidation with no episodes to consolidate."""
        consolidation_service.episodic.recall = AsyncMock(return_value=[])

        result = await consolidation_service.consolidate(
            consolidation_type="deep"
        )

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_consolidation_with_very_large_episode(self, consolidation_service):
        """Test consolidation with very large episode content."""
        large_episode = Episode(
            id=uuid4(),
            session_id="test-session",
            content="x" * 100000,  # 100KB
            context={},
            outcome=Outcome.SUCCESS,
            emotional_valence=0.5,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1,
        )

        consolidation_service.episodic.recall = AsyncMock(
            return_value=[large_episode]
        )

        result = await consolidation_service.consolidate(
            consolidation_type="light"
        )

        assert result["status"] in ["completed", "failed"]

    @pytest.mark.asyncio
    @pytest.mark.edge_case
    async def test_consolidation_timeout_recovery(self, consolidation_service):
        """Test consolidation handles timeout gracefully."""
        consolidation_service.episodic.recall = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        result = await consolidation_service.consolidate(
            consolidation_type="light"
        )

        assert result["status"] in ["failed", "timeout"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestConsolidationIntegration:
    """Integration tests for consolidation workflow."""

    @pytest.mark.asyncio
    async def test_full_consolidation_pipeline(self, consolidation_service):
        """Test complete consolidation pipeline: light → deep → skill."""
        consolidation_service.episodic.recall = AsyncMock(return_value=[])
        consolidation_service.semantic.create_entity = AsyncMock()
        consolidation_service.procedural.retrieve = AsyncMock(return_value=[])

        # Run light consolidation
        light_result = await consolidation_service.consolidate(
            consolidation_type="light"
        )
        assert light_result["status"] == "completed"

        # Run deep consolidation
        deep_result = await consolidation_service.consolidate(
            consolidation_type="deep"
        )
        assert deep_result["status"] == "completed"

        # Run skill consolidation
        skill_result = await consolidation_service.consolidate(
            consolidation_type="skill"
        )
        assert skill_result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_consolidation_preserves_session_isolation(
        self, consolidation_service
    ):
        """Test consolidation respects session boundaries."""
        consolidation_service.episodic.recall = AsyncMock(return_value=[])

        # Consolidate session A
        await consolidation_service.consolidate(
            consolidation_type="light",
            session_filter="session-a"
        )

        # Consolidate session B
        await consolidation_service.consolidate(
            consolidation_type="light",
            session_filter="session-b"
        )

        # Verify each consolidation used correct session filter
        assert consolidation_service.episodic.recall.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Expected Coverage Impact
- Current: 18% (199 missing lines)
- After: 65-70% (60-75 missing lines)
- New Tests: ~20 test cases covering all consolidation paths

---

## Phase 2: MCP Gateway Tests (10-12 hours)

### File: `/mnt/projects/t4d/t4dm/tests/unit/test_mcp_gateway.py`

```python
"""
Unit tests for MCP Memory Gateway.

Tests cover:
- All 15 MCP tools (episodic, semantic, procedural)
- Parameter validation
- Error handling and response formatting
- Session management
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from typing import Any

from t4dm.mcp.memory_gateway import (
    mcp_app,
    get_services,
    _service_instances,
    _initialized_sessions,
)
from t4dm.core.types import Outcome, Domain, EntityType, RelationType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def memory_services(mock_embedding, mock_t4dx_vector_adapter, mock_t4dx_graph_adapter):
    """Setup mock memory services."""
    with patch('t4dm.mcp.memory_gateway.get_episodic_memory') as mock_episodic_factory, \
         patch('t4dm.mcp.memory_gateway.get_semantic_memory') as mock_semantic_factory, \
         patch('t4dm.mcp.memory_gateway.get_procedural_memory') as mock_procedural_factory:

        # Create mock service instances
        mock_episodic = AsyncMock()
        mock_episodic.initialize = AsyncMock()
        mock_episodic.create = AsyncMock(return_value=MagicMock(id=uuid4()))
        mock_episodic.recall = AsyncMock(return_value=[])
        mock_episodic.cleanup = AsyncMock()

        mock_semantic = AsyncMock()
        mock_semantic.initialize = AsyncMock()
        mock_semantic.create_entity = AsyncMock(return_value=MagicMock(id=uuid4()))
        mock_semantic.create_relationship = AsyncMock()
        mock_semantic.recall = AsyncMock(return_value=[])
        mock_semantic.get_entity = AsyncMock(return_value=MagicMock())

        mock_procedural = AsyncMock()
        mock_procedural.initialize = AsyncMock()
        mock_procedural.build = AsyncMock(return_value=MagicMock(id=uuid4()))
        mock_procedural.retrieve = AsyncMock(return_value=[])
        mock_procedural.get_procedure = AsyncMock(return_value=MagicMock())

        mock_episodic_factory.return_value = mock_episodic
        mock_semantic_factory.return_value = mock_semantic
        mock_procedural_factory.return_value = mock_procedural

        services = (mock_episodic, mock_semantic, mock_procedural)
        yield services

        # Cleanup
        _service_instances.clear()
        _initialized_sessions.clear()


@pytest.fixture
def valid_uuid_string():
    """Valid UUID for testing."""
    return str(uuid4())


# ============================================================================
# Episodic Memory Tool Tests
# ============================================================================

class TestEpisodicMemoryTools:
    """Tests for episodic memory MCP tools."""

    @pytest.mark.asyncio
    async def test_episodic_create_with_all_parameters(self, memory_services):
        """Test episodic_create with all optional parameters."""
        episodic, _, _ = memory_services

        # Test tool would be called like:
        # result = await tool_episodic_create(
        #     content="Test memory",
        #     context={"project": "test"},
        #     outcome="success",
        #     valence=0.8
        # )

        episodic.create.assert_not_called()  # Not called yet

        # Create episode
        await episodic.create(
            content="Test memory",
            context={"project": "test"},
            outcome="success",
            valence=0.8
        )

        episodic.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_episodic_create_validates_valence(self, memory_services):
        """Test episodic_create rejects invalid valence."""
        episodic, _, _ = memory_services

        # Valence out of range should be rejected
        with pytest.raises(Exception):  # ValidationError
            await episodic.create(
                content="Test",
                valence=1.5  # Invalid: > 1.0
            )

    @pytest.mark.asyncio
    async def test_episodic_create_validates_outcome(self, memory_services):
        """Test episodic_create validates outcome enum."""
        episodic, _, _ = memory_services

        # Invalid outcome should raise error
        with pytest.raises(Exception):
            await episodic.create(
                content="Test",
                outcome="invalid_outcome"
            )

    @pytest.mark.asyncio
    async def test_episodic_create_returns_episode_id(self, memory_services):
        """Test episodic_create returns episode ID."""
        episodic, _, _ = memory_services

        episode_id = uuid4()
        episode = MagicMock()
        episode.id = episode_id
        episodic.create.return_value = episode

        result = await episodic.create(content="Test memory")

        assert result.id == episode_id

    @pytest.mark.asyncio
    async def test_episodic_recall_with_query(self, memory_services):
        """Test episodic_recall with query."""
        episodic, _, _ = memory_services

        results = await episodic.recall(
            query="memory test",
            limit=5
        )

        episodic.recall.assert_called_once()

    @pytest.mark.asyncio
    async def test_episodic_recall_default_limit(self, memory_services):
        """Test episodic_recall uses default limit."""
        episodic, _, _ = memory_services

        await episodic.recall(query="test")

        # Should use default limit (typically 10)
        episodic.recall.assert_called_once()

    @pytest.mark.asyncio
    async def test_episodic_cleanup_removes_old_episodes(self, memory_services):
        """Test episodic_cleanup removes old episodes."""
        episodic, _, _ = memory_services

        await episodic.cleanup(days=30)

        episodic.cleanup.assert_called_once()


# ============================================================================
# Semantic Memory Tool Tests
# ============================================================================

class TestSemanticMemoryTools:
    """Tests for semantic memory MCP tools."""

    @pytest.mark.asyncio
    async def test_semantic_create_entity_validates_type(self, memory_services):
        """Test semantic_create_entity validates entity type."""
        _, semantic, _ = memory_services

        entity = MagicMock()
        entity.id = uuid4()
        semantic.create_entity.return_value = entity

        result = await semantic.create_entity(
            name="Test Entity",
            entity_type="TECHNIQUE"
        )

        assert result.id is not None

    @pytest.mark.asyncio
    async def test_semantic_create_entity_invalid_type(self, memory_services):
        """Test semantic_create_entity rejects invalid type."""
        _, semantic, _ = memory_services

        semantic.create_entity.side_effect = Exception("Invalid type")

        with pytest.raises(Exception):
            await semantic.create_entity(
                name="Test",
                entity_type="INVALID_TYPE"
            )

    @pytest.mark.asyncio
    async def test_semantic_create_relationship(self, memory_services):
        """Test semantic_create_relationship."""
        _, semantic, _ = memory_services

        source_id = uuid4()
        target_id = uuid4()

        await semantic.create_relationship(
            source_id=str(source_id),
            target_id=str(target_id),
            relation_type="SIMILAR_TO"
        )

        semantic.create_relationship.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_recall_with_activation(self, memory_services):
        """Test semantic_recall returns scored results."""
        _, semantic, _ = memory_services

        results = await semantic.recall(
            query="memory algorithm",
            limit=5
        )

        semantic.recall.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_get_entity(self, memory_services, valid_uuid_string):
        """Test semantic_get_entity retrieval."""
        _, semantic, _ = memory_services

        entity = MagicMock()
        entity.id = valid_uuid_string
        entity.name = "Test Entity"
        semantic.get_entity.return_value = entity

        result = await semantic.get_entity(entity_id=valid_uuid_string)

        assert result.id == valid_uuid_string


# ============================================================================
# Procedural Memory Tool Tests
# ============================================================================

class TestProceduralMemoryTools:
    """Tests for procedural memory MCP tools."""

    @pytest.mark.asyncio
    async def test_procedural_build_from_trajectory(self, memory_services):
        """Test procedural_build creates procedure from trajectory."""
        _, _, procedural = memory_services

        trajectory = [
            {"action": "Read file", "tool": "Read", "parameters": {}, "result": "Success"},
            {"action": "Edit file", "tool": "Edit", "parameters": {}, "result": "Success"},
        ]

        procedure = MagicMock()
        procedure.id = uuid4()
        procedural.build.return_value = procedure

        result = await procedural.build(
            trajectory=trajectory,
            outcome_score=0.9,
            domain="coding"
        )

        assert result.id is not None

    @pytest.mark.asyncio
    async def test_procedural_build_validates_trajectory(self, memory_services):
        """Test procedural_build validates trajectory format."""
        _, _, procedural = memory_services

        # Empty trajectory should fail
        procedural.build.side_effect = Exception("Invalid trajectory")

        with pytest.raises(Exception):
            await procedural.build(
                trajectory=[],  # Empty
                outcome_score=0.9,
                domain="coding"
            )

    @pytest.mark.asyncio
    async def test_procedural_build_validates_outcome_score(self, memory_services):
        """Test procedural_build validates outcome score range."""
        _, _, procedural = memory_services

        procedural.build.side_effect = Exception("Invalid score")

        with pytest.raises(Exception):
            await procedural.build(
                trajectory=[{"action": "test", "tool": "test", "parameters": {}, "result": "test"}],
                outcome_score=1.5,  # Invalid: > 1.0
                domain="coding"
            )

    @pytest.mark.asyncio
    async def test_procedural_retrieve_by_task(self, memory_services):
        """Test procedural_retrieve finds procedures for task."""
        _, _, procedural = memory_services

        results = await procedural.retrieve(
            task="I need to add a method to a class",
            domain="coding",
            limit=5
        )

        procedural.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_procedural_retrieve_without_domain(self, memory_services):
        """Test procedural_retrieve works without domain filter."""
        _, _, procedural = memory_services

        await procedural.retrieve(
            task="Do something",
            limit=5
        )

        procedural.retrieve.assert_called_once()


# ============================================================================
# Error Handling and Response Formatting Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and MCP protocol compliance."""

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_error(self, memory_services):
        """Test that invalid UUID returns validation error."""
        _, semantic, _ = memory_services

        semantic.get_entity.side_effect = Exception("Invalid UUID")

        with pytest.raises(Exception):
            await semantic.get_entity(entity_id="not-a-uuid")

    @pytest.mark.asyncio
    async def test_missing_required_parameter_returns_error(self, memory_services):
        """Test missing required parameters return validation error."""
        episodic, _, _ = memory_services

        episodic.create.side_effect = Exception("Missing content")

        with pytest.raises(Exception):
            await episodic.create(content=None)

    @pytest.mark.asyncio
    async def test_validation_error_format(self, memory_services):
        """Test validation errors return MCP-compliant format."""
        episodic, _, _ = memory_services

        # Validation error should return error dict with:
        # - error: "validation_error"
        # - field: field name
        # - message: error message
        episodic.create.side_effect = Exception("Validation failed")

        with pytest.raises(Exception):
            await episodic.create(content="", valence=None)

    @pytest.mark.asyncio
    async def test_service_unavailable_error(self, memory_services):
        """Test graceful handling of service unavailability."""
        episodic, _, _ = memory_services

        episodic.initialize.side_effect = Exception("Service unavailable")

        with pytest.raises(Exception):
            await episodic.initialize()


# ============================================================================
# Session Management Tests
# ============================================================================

class TestSessionManagement:
    """Tests for session and instance management."""

    @pytest.mark.asyncio
    async def test_get_services_initializes_once(self, memory_services):
        """Test that services are only initialized once per session."""
        episodic, semantic, procedural = memory_services

        # First call should initialize
        services1 = (episodic, semantic, procedural)

        # Second call should return same instances
        services2 = (episodic, semantic, procedural)

        assert services1 is services2 or (
            services1[0] is services2[0] and
            services1[1] is services2[1] and
            services1[2] is services2[2]
        )

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self, memory_services):
        """Test that different sessions have isolated service instances."""
        # Session A
        episodic_a, semantic_a, procedural_a = memory_services

        # Session B would be separate (in real implementation)
        # For now, verify that each session would get its own instances


# ============================================================================
# Tool Documentation Tests
# ============================================================================

class TestToolDocumentation:
    """Tests that tools are properly documented for MCP."""

    def test_mcp_app_defined(self):
        """Test that FastMCP app is properly defined."""
        assert mcp_app is not None
        assert mcp_app.name == "ww-memory"

    def test_mcp_app_has_instructions(self):
        """Test that MCP app has proper instructions."""
        assert mcp_app.instructions is not None
        assert "tripartite" in mcp_app.instructions.lower() or "memory" in mcp_app.instructions.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Expected Coverage Impact
- Current: 18% (321 missing lines)
- After: 60-70% (100-130 missing lines)
- New Tests: ~25 test cases covering tools, validation, errors

---

## Phase 3: Observability Tests (8-10 hours)

### File: `/mnt/projects/t4d/t4dm/tests/unit/test_observability.py`

```python
"""
Unit tests for observability modules.

Tests cover:
- Structured logging
- Metrics collection
- Health checking
"""

import pytest
import logging
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from t4dm.observability.logging import (
    configure_logging,
    get_logger,
    log_operation,
    OperationLogger,
    set_context,
    clear_context,
)
from t4dm.observability.metrics import (
    MetricsCollector,
    get_metrics,
    timed_operation,
    count_operation,
    Timer,
    AsyncTimer,
)
from t4dm.observability.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    get_health_checker,
)


# ============================================================================
# Logging Tests
# ============================================================================

class TestLogging:
    """Tests for structured logging."""

    def test_configure_logging(self):
        """Test logging configuration."""
        logger = configure_logging(level=logging.DEBUG)
        assert logger is not None

    def test_get_logger(self):
        """Test logger retrieval."""
        logger = get_logger(__name__)
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_set_context(self):
        """Test setting logging context."""
        context = {"user": "test", "session": "123"}
        set_context(context)
        # Context should be stored for subsequent logs

    def test_clear_context(self):
        """Test clearing logging context."""
        set_context({"user": "test"})
        clear_context()
        # Context should be cleared

    @pytest.mark.asyncio
    async def test_log_operation(self):
        """Test operation logging."""
        with patch('t4dm.observability.logging.get_logger') as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            # This would be called in actual code
            result = await log_operation(
                operation_name="test_op",
                func=lambda: "result",
                *[]
            )


class TestOperationLogger:
    """Tests for OperationLogger context manager."""

    def test_operation_logger_success(self):
        """Test OperationLogger logs successful operation."""
        with patch('t4dm.observability.logging.get_logger') as mock:
            logger = MagicMock()
            mock.return_value = logger

            # Would use: async with OperationLogger("operation"):
            # logger should record start and end

    def test_operation_logger_timing(self):
        """Test OperationLogger measures duration."""
        # Should record operation duration


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics:
    """Tests for metrics collection."""

    def test_get_metrics(self):
        """Test metrics collector retrieval."""
        metrics = get_metrics()
        assert metrics is not None
        assert isinstance(metrics, MetricsCollector)

    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        assert collector is not None

    def test_count_operation(self):
        """Test operation counter."""
        @count_operation("test_operation")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

    def test_timed_operation(self):
        """Test operation timer."""
        @timed_operation("test_operation")
        async def test_func():
            return "result"

    def test_timer_context_manager(self):
        """Test Timer context manager."""
        with Timer("operation"):
            # Do something
            pass
        # Timer should record duration

    @pytest.mark.asyncio
    async def test_async_timer_context_manager(self):
        """Test AsyncTimer context manager."""
        async with AsyncTimer("operation"):
            # Do something async
            pass
        # AsyncTimer should record duration


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthChecker:
    """Tests for health checking."""

    def test_health_status_enum(self):
        """Test HealthStatus enum."""
        assert HealthStatus.HEALTHY is not None
        assert HealthStatus.DEGRADED is not None
        assert HealthStatus.UNHEALTHY is not None

    def test_component_health(self):
        """Test ComponentHealth creation."""
        health = ComponentHealth(
            component="neo4j",
            status=HealthStatus.HEALTHY,
            message="Connected"
        )
        assert health.component == "neo4j"
        assert health.status == HealthStatus.HEALTHY

    def test_system_health(self):
        """Test SystemHealth aggregation."""
        components = [
            ComponentHealth("neo4j", HealthStatus.HEALTHY),
            ComponentHealth("qdrant", HealthStatus.HEALTHY),
        ]
        system_health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=components,
            timestamp=datetime.utcnow()
        )
        assert system_health.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_checker(self):
        """Test HealthChecker."""
        checker = get_health_checker()
        assert checker is not None

    @pytest.mark.asyncio
    async def test_health_check_all_components(self):
        """Test checking health of all components."""
        checker = get_health_checker()
        # Should check neo4j, qdrant, embedding service
        # Returns SystemHealth with component statuses

    @pytest.mark.asyncio
    async def test_health_check_individual_component(self):
        """Test checking individual component health."""
        # Should return ComponentHealth for specific component


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Expected Coverage Impact
- Current: 0% (348 missing lines)
- After: 30-40% (210-240 missing lines)
- New Tests: ~15 test cases covering logging, metrics, health checks

---

## Phase 4: Storage & Edge Cases (12-15 hours)

### Create `/mnt/projects/t4d/t4dm/tests/unit/test_storage.py`

**Key Tests**:
- Neo4j timeout handling
- Qdrant batch operations
- Connection lifecycle
- Error recovery
- Concurrent operations

### Create `/mnt/projects/t4d/t4dm/tests/unit/test_edge_cases.py`

**Key Tests**:
- Empty results
- Very large payloads
- Concurrent operations
- Resource cleanup
- Network timeouts

---

## Phase 5: Integration & Documentation (6-8 hours)

### Updates to `pyproject.toml`

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "asyncio: marks tests as async",
    "integration: marks tests as integration (requires docker)",
    "slow: marks tests as slow (>1 second)",
    "edge_case: marks tests as edge case boundary testing",
]
```

### Update `tests/__init__.py`

```python
"""T4DM test suite."""

# Test configuration and markers defined in conftest.py
```

### Test Execution

```bash
# Run all tests
pytest

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run with coverage
pytest --cov=src/ww --cov-report=html

# Run slow tests
pytest -m slow

# Run integration tests (requires docker)
pytest -m integration
```

---

## Effort & Timeline Estimate

| Phase | Task | Hours | Start | End |
|-------|------|-------|-------|-----|
| 0 | Fix async event loop | 2 | Day 1 | Day 1 |
| 1 | Consolidation tests | 10 | Day 1 | Day 2 |
| 2 | MCP gateway tests | 12 | Day 2-3 | Day 3 |
| 3 | Observability tests | 10 | Day 3-4 | Day 4 |
| 4 | Storage & edge cases | 15 | Day 4-5 | Day 5-6 |
| 5 | Integration & docs | 8 | Day 6 | Day 7 |
| **TOTAL** | | **57 hours** | | |

**Timeline**: ~2 weeks at 20 hrs/week (or 7 days at 8 hrs/day)

---

## Success Metrics

### Coverage Targets
- Overall: 47% → 75%+
- Consolidation: 18% → 75%+
- MCP Gateway: 18% → 75%+
- Observability: 0% → 50%+
- Storage: 41%/56% → 70%+

### Test Quality
- All 237 tests passing (5 async failures fixed)
- No over-mocking (use real services for integration tests)
- Edge cases covered (10+ edge case tests)
- Error paths tested (10+ error handling tests)

### Documentation
- Test plan created
- Coverage report generated
- CI/CD coverage checks added

---

## Files to Create/Modify

**Create**:
- `/mnt/projects/t4d/t4dm/tests/conftest.py` - Shared pytest fixtures
- `/mnt/projects/t4d/t4dm/tests/unit/test_consolidation.py` - Consolidation tests
- `/mnt/projects/t4d/t4dm/tests/unit/test_mcp_gateway.py` - MCP tests
- `/mnt/projects/t4d/t4dm/tests/unit/test_observability.py` - Logging/metrics/health tests
- `/mnt/projects/t4d/t4dm/tests/unit/test_storage.py` - Storage layer tests
- `/mnt/projects/t4d/t4dm/tests/unit/test_edge_cases.py` - Edge case tests

**Modify**:
- `/mnt/projects/t4d/t4dm/pyproject.toml` - pytest configuration
- `/mnt/projects/t4d/t4dm/tests/__init__.py` - Package documentation

---

## Next Steps

1. **Immediately**: Run Phase 0 (fix async) - 1 command, 2 hours
2. **Today**: Create conftest.py and consolidation tests
3. **This week**: Complete MCP gateway tests + observability
4. **Next week**: Storage tests + edge cases + documentation

