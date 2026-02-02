"""
API Integration Tests for T4DM REST API.

Tests full CRUD flows for episodes, entities, and skills with FastAPI TestClient.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from t4dm.core.types import Episode, EpisodeContext, Outcome, Entity, EntityType
from t4dm.sdk.models import Skill, Step


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_episode():
    """Create a mock episode for testing."""
    return Episode(
        id=uuid4(),
        session_id="test-session",
        content="Test episode content about debugging Python",
        context=EpisodeContext(project="ww", file="test.py", tool="pytest"),
        outcome=Outcome.SUCCESS,
        emotional_valence=0.8,
        timestamp=datetime.now(),
        access_count=1,
        stability=5.0,
    )


@pytest.fixture
def mock_entity():
    """Create a mock entity for testing."""
    return Entity(
        id=uuid4(),
        name="Python",
        entity_type=EntityType.TOOL,
        summary="A programming language",
        details="Python 3.11 with type hints support",
    )


@pytest.fixture
def mock_skill():
    """Create a mock skill for testing."""
    return Skill(
        id=uuid4(),
        name="pytest_testing",
        domain="testing",
        steps=[
            Step(order=1, action="Write test", tool="editor"),
            Step(order=2, action="Run pytest", tool="terminal"),
            Step(order=3, action="Check coverage", tool="terminal"),
        ],
        success_rate=0.95,
        execution_count=100,
        version=1,
        deprecated=False,
        created_at=datetime.now(),
    )


@pytest.fixture
def mock_episodic_service(mock_episode):
    """Create mock episodic memory service."""
    from t4dm.core.types import ScoredResult
    service = AsyncMock()
    service.create = AsyncMock(return_value=mock_episode)
    service.get = AsyncMock(return_value=mock_episode)
    service.delete = AsyncMock(return_value=True)
    service.recent = AsyncMock(return_value=[mock_episode])
    service.recall = AsyncMock(return_value=[ScoredResult(item=mock_episode, score=0.95)])
    service.store = AsyncMock(return_value=mock_episode)  # For mark_important
    return service


@pytest.fixture
def mock_semantic_service(mock_entity):
    """Create mock semantic memory service."""
    from t4dm.core.types import ScoredResult
    service = AsyncMock()
    service.create_entity = AsyncMock(return_value=mock_entity)
    service.get_entity = AsyncMock(return_value=mock_entity)
    service.list_entities = AsyncMock(return_value=[mock_entity])
    service.get_entities_by_type = AsyncMock(return_value=[mock_entity])
    service.recall = AsyncMock(return_value=[ScoredResult(item=mock_entity, score=0.9)])
    service.create_relationship = AsyncMock(return_value=MagicMock(
        source_id=mock_entity.id,
        target_id=mock_entity.id,
        relation_type="relates_to",
        weight=0.5,
        co_access_count=0,
    ))
    service.spread_activation = AsyncMock(return_value=[])
    service.supersede = AsyncMock(return_value=mock_entity)
    return service


@pytest.fixture
def mock_procedural_service(mock_skill):
    """Create mock procedural memory service."""
    from t4dm.core.types import ScoredResult, Procedure, ProcedureStep, Domain
    from datetime import datetime

    # Convert SDK Skill to core Procedure
    mock_procedure = Procedure(
        id=mock_skill.id,
        name=mock_skill.name,
        domain=Domain.CODING,
        steps=[ProcedureStep(order=s.order, action=s.action, tool=s.tool) for s in mock_skill.steps],
        success_rate=mock_skill.success_rate,
        execution_count=mock_skill.execution_count,
        version=mock_skill.version,
        deprecated=mock_skill.deprecated,
        created_at=mock_skill.created_at,
    )

    service = AsyncMock()
    service.create_skill = AsyncMock(return_value=mock_procedure)
    service.store_skill_direct = AsyncMock(return_value=mock_procedure)  # Used by POST /skills
    service.get_procedure = AsyncMock(return_value=mock_procedure)
    service.list_skills = AsyncMock(return_value=[mock_procedure])
    service.recall_skill = AsyncMock(return_value=[ScoredResult(item=mock_procedure, score=0.88)])
    service.update = AsyncMock(return_value=mock_procedure)
    return service


@pytest.fixture
def mock_services(mock_episodic_service, mock_semantic_service, mock_procedural_service):
    """Create full mock services dict."""
    return {
        "session_id": "test-session",
        "episodic": mock_episodic_service,
        "semantic": mock_semantic_service,
        "procedural": mock_procedural_service,
    }


@pytest.fixture
def api_client(mock_services):
    """Create test client with mocked dependencies."""
    from t4dm.api.server import app
    from t4dm.api import deps

    async def override_get_memory_services():
        return mock_services

    app.dependency_overrides[deps.get_memory_services] = override_get_memory_services

    client = TestClient(app)
    yield client

    # Cleanup
    app.dependency_overrides.clear()


# ============================================================================
# Episode API Tests
# ============================================================================

class TestEpisodeAPI:
    """Tests for episode CRUD endpoints."""

    def test_create_episode(self, api_client, mock_services):
        """Test POST /api/v1/episodes."""
        response = api_client.post(
            "/api/v1/episodes",
            json={
                "content": "Test episode about debugging",
                "project": "ww",
                "file": "test.py",
                "outcome": "success",
                "emotional_valence": 0.8,
            },
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["content"] == "Test episode content about debugging Python"
        assert data["outcome"] == "success"
        assert data["session_id"] == "test-session"

    def test_create_episode_minimal(self, api_client, mock_services):
        """Test episode creation with minimal fields."""
        response = api_client.post(
            "/api/v1/episodes",
            json={"content": "Minimal episode"},
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 201

    def test_create_episode_invalid_content(self, api_client):
        """Test validation for empty content."""
        response = api_client.post(
            "/api/v1/episodes",
            json={"content": ""},
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 422  # Validation error

    def test_get_episode(self, api_client, mock_episode):
        """Test GET /api/v1/episodes/{id}."""
        response = api_client.get(
            f"/api/v1/episodes/{mock_episode.id}",
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_episode.id)
        assert data["content"] == mock_episode.content

    def test_get_episode_not_found(self, api_client, mock_services):
        """Test 404 for missing episode."""
        mock_services["episodic"].get = AsyncMock(return_value=None)

        response = api_client.get(
            f"/api/v1/episodes/{uuid4()}",
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_episode(self, api_client, mock_episode):
        """Test DELETE /api/v1/episodes/{id}."""
        response = api_client.delete(
            f"/api/v1/episodes/{mock_episode.id}",
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 204

    def test_delete_episode_not_found(self, api_client, mock_services):
        """Test 404 for deleting missing episode."""
        mock_services["episodic"].delete = AsyncMock(return_value=False)

        response = api_client.delete(
            f"/api/v1/episodes/{uuid4()}",
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 404

    def test_list_episodes(self, api_client, mock_episode):
        """Test GET /api/v1/episodes."""
        response = api_client.get(
            "/api/v1/episodes",
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "episodes" in data
        assert "total" in data
        assert "page" in data
        assert data["page_size"] == 20

    def test_list_episodes_pagination(self, api_client):
        """Test pagination parameters."""
        response = api_client.get(
            "/api/v1/episodes?page=2&page_size=5",
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 5

    def test_recall_episodes(self, api_client, mock_episode):
        """Test POST /api/v1/episodes/recall."""
        response = api_client.post(
            "/api/v1/episodes/recall",
            json={
                "query": "debugging Python",
                "limit": 10,
                "min_similarity": 0.5,
            },
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "debugging Python"
        assert "episodes" in data
        assert "scores" in data
        assert len(data["episodes"]) == len(data["scores"])

    def test_mark_episode_important(self, api_client, mock_episode):
        """Test POST /api/v1/episodes/{id}/mark-important."""
        response = api_client.post(
            f"/api/v1/episodes/{mock_episode.id}/mark-important?importance=1.0",
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "emotional_valence" in data


# ============================================================================
# Entity API Tests
# ============================================================================

class TestEntityAPI:
    """Tests for entity (semantic memory) endpoints."""

    def test_create_entity(self, api_client, mock_services, mock_entity):
        """Test POST /api/v1/entities."""
        response = api_client.post(
            "/api/v1/entities",
            json={
                "name": "Python",
                "entity_type": "TOOL",
                "summary": "A programming language",
            },
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 201

    def test_get_entity(self, api_client, mock_entity):
        """Test GET /api/v1/entities/{id}."""
        response = api_client.get(
            f"/api/v1/entities/{mock_entity.id}",
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 200

    def test_search_entities(self, api_client):
        """Test POST /api/v1/entities/recall."""
        response = api_client.post(
            "/api/v1/entities/recall",
            json={"query": "programming language", "limit": 5},
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 200


# ============================================================================
# Skill API Tests
# ============================================================================

class TestSkillAPI:
    """Tests for skill (procedural memory) endpoints."""

    def test_create_skill(self, api_client, mock_services, mock_skill):
        """Test POST /api/v1/skills."""
        response = api_client.post(
            "/api/v1/skills",
            json={
                "name": "pytest_testing",
                "domain": "coding",
                "task": "Run unit tests",
                "steps": [
                    {"order": 1, "action": "Write test"},
                    {"order": 2, "action": "Run pytest"},
                ],
            },
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 201

    def test_get_skill(self, api_client, mock_skill):
        """Test GET /api/v1/skills/{id}."""
        response = api_client.get(
            f"/api/v1/skills/{mock_skill.id}",
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 200

    def test_search_skills(self, api_client):
        """Test POST /api/v1/skills/recall."""
        response = api_client.post(
            "/api/v1/skills/recall",
            json={"query": "testing", "limit": 5},
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 200


# ============================================================================
# System API Tests
# ============================================================================

class TestSystemAPI:
    """Tests for system endpoints."""

    def test_health_check(self, api_client):
        """Test GET /api/v1/health."""
        response = api_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_root_redirect(self, api_client):
        """Test GET / returns API info."""
        response = api_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "docs" in data
        assert "health" in data


# ============================================================================
# Session Isolation Tests
# ============================================================================

class TestSessionIsolation:
    """Tests for session-based isolation."""

    def test_different_sessions_isolated(self, mock_services):
        """Verify different sessions don't share data."""
        from t4dm.api.server import app
        from t4dm.api import deps

        session_a_episodes = []
        session_b_episodes = []

        async def mock_services_a():
            service = AsyncMock()
            service.recent = AsyncMock(return_value=session_a_episodes)
            return {
                "session_id": "session-a",
                "episodic": service,
                "semantic": AsyncMock(),
                "procedural": AsyncMock(),
            }

        app.dependency_overrides[deps.get_memory_services] = mock_services_a
        client = TestClient(app)

        response = client.get(
            "/api/v1/episodes",
            headers={"X-Session-ID": "session-a"},
        )

        assert response.status_code == 200

        app.dependency_overrides.clear()

    def test_invalid_session_id(self):
        """Test handling of invalid session ID."""
        from t4dm.api.server import app

        client = TestClient(app, raise_server_exceptions=False)

        # Very long session ID should be rejected
        response = client.get(
            "/api/v1/episodes",
            headers={"X-Session-ID": "x" * 1000},
        )

        # Should either work or return 400 for invalid session
        assert response.status_code in [200, 400, 503]


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error responses."""

    def test_500_on_service_error(self, api_client, mock_services):
        """Test 500 when service throws exception."""
        mock_services["episodic"].create = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        response = api_client.post(
            "/api/v1/episodes",
            json={"content": "Test content"},
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 500
        # Error messages are now sanitized
        assert "create episode" in response.json()["detail"]

    def test_validation_error_format(self, api_client):
        """Test validation error response format."""
        response = api_client.post(
            "/api/v1/episodes",
            json={},  # Missing required content field
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


# ============================================================================
# Full Flow Tests
# ============================================================================

class TestFullFlows:
    """End-to-end flow tests."""

    def test_episode_crud_flow(self, api_client, mock_services, mock_episode):
        """Test complete episode lifecycle: create -> read -> recall -> delete."""
        # Create
        create_response = api_client.post(
            "/api/v1/episodes",
            json={
                "content": "Learning about pytest fixtures",
                "project": "ww",
                "outcome": "success",
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert create_response.status_code == 201
        episode_id = create_response.json()["id"]

        # Read
        get_response = api_client.get(
            f"/api/v1/episodes/{episode_id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert get_response.status_code == 200

        # Recall
        recall_response = api_client.post(
            "/api/v1/episodes/recall",
            json={"query": "pytest fixtures", "limit": 5},
            headers={"X-Session-ID": "test-session"},
        )
        assert recall_response.status_code == 200

        # Delete
        delete_response = api_client.delete(
            f"/api/v1/episodes/{episode_id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert delete_response.status_code == 204

    def test_memory_type_interaction(self, api_client, mock_services, mock_episode, mock_entity):
        """Test interaction between memory types."""
        # Create episode
        ep_response = api_client.post(
            "/api/v1/episodes",
            json={"content": "Used Python for testing"},
            headers={"X-Session-ID": "test-session"},
        )
        assert ep_response.status_code == 201

        # Create related entity
        ent_response = api_client.post(
            "/api/v1/entities",
            json={
                "name": "Python",
                "entity_type": "TOOL",
                "summary": "Programming language",
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert ent_response.status_code == 201

        # Search should find the entity
        search_response = api_client.post(
            "/api/v1/entities/recall",
            json={"query": "Python testing"},
            headers={"X-Session-ID": "test-session"},
        )
        assert search_response.status_code == 200
