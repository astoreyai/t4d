"""
Comprehensive endpoint testing for T4DM API.

Tests ALL endpoint categories with focus on:
- Auth requirements (X-Admin-Key header)
- Input validation
- Error handling
- Response schemas
- Edge cases

Uses proper dependency injection mocking to test without actual DB connections.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from fastapi.testclient import TestClient

from t4dm.api.server import app
from t4dm.api import deps
from t4dm.core.types import (
    Episode, EpisodeContext, Outcome, Entity, EntityType, Relationship, RelationType,
    Procedure, ProcedureStep, Domain, ScoredResult
)


# ============================================================================
# Fixtures - Mock Services (matching existing integration test pattern)
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
def mock_procedure():
    """Create a mock procedure for testing."""
    steps = [
        ProcedureStep(order=1, action="Write test", tool="editor"),
        ProcedureStep(order=2, action="Run pytest", tool="terminal"),
    ]
    proc = Procedure(
        id=uuid4(),
        name="pytest_testing",
        domain=Domain.CODING,
        steps=steps,
        success_rate=0.95,
        execution_count=100,
        version=1,
        deprecated=False,
        created_at=datetime.now(),
    )
    return proc


@pytest.fixture
def mock_episodic_service(mock_episode):
    """Create mock episodic memory service."""
    service = AsyncMock()
    service.create = AsyncMock(return_value=mock_episode)
    service.get = AsyncMock(return_value=mock_episode)
    service.delete = AsyncMock(return_value=True)
    service.recent = AsyncMock(return_value=[mock_episode])
    service.recall = AsyncMock(return_value=[ScoredResult(item=mock_episode, score=0.95)])
    service.store = AsyncMock(return_value=mock_episode)
    service.update = AsyncMock(return_value=mock_episode)
    return service


@pytest.fixture
def mock_semantic_service(mock_entity):
    """Create mock semantic memory service."""
    service = AsyncMock()
    service.create_entity = AsyncMock(return_value=mock_entity)
    service.get_entity = AsyncMock(return_value=mock_entity)
    service.delete_entity = AsyncMock(return_value=True)
    service.update_entity = AsyncMock(return_value=mock_entity)
    service.list_entities = AsyncMock(return_value=[mock_entity])
    service.get_entities_by_type = AsyncMock(return_value=[mock_entity])
    service.recall = AsyncMock(return_value=[ScoredResult(item=mock_entity, score=0.95)])

    rel = Relationship(
        source_id=mock_entity.id,
        target_id=uuid4(),
        relation_type=RelationType.PART_OF,
        weight=0.5,
    )
    service.create_relationship = AsyncMock(return_value=rel)
    service.spread_activation = AsyncMock(return_value=[(mock_entity, 0.9, ["path"])])
    service.supersede = AsyncMock(return_value=mock_entity)

    return service


@pytest.fixture
def mock_procedural_service(mock_procedure):
    """Create mock procedural memory service."""
    service = AsyncMock()
    service.store_skill_direct = AsyncMock(return_value=mock_procedure)
    service.get_procedure = AsyncMock(return_value=mock_procedure)
    service.delete = AsyncMock(return_value=True)
    service.update = AsyncMock(return_value=mock_procedure)
    service.list_skills = AsyncMock(return_value=[mock_procedure])
    service.recall_skill = AsyncMock(return_value=[ScoredResult(item=mock_procedure, score=0.88)])
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
    async def override_get_memory_services():
        return mock_services

    app.dependency_overrides[deps.get_memory_services] = override_get_memory_services

    client = TestClient(app)
    yield client

    # Cleanup
    app.dependency_overrides.clear()


# ============================================================================
# 1. EPISODE ENDPOINTS (/api/v1/episodes)
# ============================================================================

class TestEpisodeEndpoints:
    """Test all episode CRUD and search endpoints."""

    def test_create_episode_success(self, api_client):
        """POST /episodes - Create episode with valid data."""
        response = api_client.post(
            "/api/v1/episodes",
            json={
                "content": "Test episode content",
                "project": "test-project",
                "file": "test.py",
                "tool": "pytest",
                "outcome": "success",
                "emotional_valence": 0.8,
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 201
        assert "id" in response.json()

    def test_create_episode_minimal(self, api_client):
        """POST /episodes - Create with minimal fields."""
        response = api_client.post(
            "/api/v1/episodes",
            json={"content": "Minimal content"},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 201

    def test_create_episode_invalid_content_empty(self, api_client):
        """POST /episodes - Reject empty content."""
        response = api_client.post(
            "/api/v1/episodes",
            json={"content": ""},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_create_episode_invalid_content_too_long(self, api_client):
        """POST /episodes - Reject content exceeding max_length (50KB)."""
        response = api_client.post(
            "/api/v1/episodes",
            json={"content": "x" * 50001},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_create_episode_invalid_emotional_valence_low(self, api_client):
        """POST /episodes - Reject emotional_valence < 0."""
        response = api_client.post(
            "/api/v1/episodes",
            json={"content": "Test", "emotional_valence": -0.1},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_create_episode_invalid_emotional_valence_high(self, api_client):
        """POST /episodes - Reject emotional_valence > 1."""
        response = api_client.post(
            "/api/v1/episodes",
            json={"content": "Test", "emotional_valence": 1.1},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_get_episode_success(self, api_client, mock_episode):
        """GET /episodes/{id} - Retrieve episode by ID."""
        response = api_client.get(
            f"/api/v1/episodes/{mock_episode.id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_episode.id)

    def test_get_episode_not_found(self, api_client, mock_services):
        """GET /episodes/{id} - Return 404 for non-existent episode."""
        mock_services["episodic"].get = AsyncMock(return_value=None)
        fake_id = uuid4()
        response = api_client.get(
            f"/api/v1/episodes/{fake_id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_get_episode_invalid_uuid(self, api_client):
        """GET /episodes/{id} - Reject invalid UUID format."""
        response = api_client.get(
            "/api/v1/episodes/not-a-uuid",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_update_episode_success(self, api_client, mock_episode):
        """PUT /episodes/{id} - Update episode."""
        response = api_client.put(
            f"/api/v1/episodes/{mock_episode.id}",
            json={"content": "Updated content", "emotional_valence": 0.9},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200

    def test_update_episode_not_found(self, api_client, mock_services):
        """PUT /episodes/{id} - Return 404 for non-existent episode."""
        mock_services["episodic"].get = AsyncMock(return_value=None)
        fake_id = uuid4()
        response = api_client.put(
            f"/api/v1/episodes/{fake_id}",
            json={"content": "Updated"},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_delete_episode_success(self, api_client, mock_episode):
        """DELETE /episodes/{id} - Delete episode."""
        response = api_client.delete(
            f"/api/v1/episodes/{mock_episode.id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 204

    def test_delete_episode_not_found(self, api_client, mock_services):
        """DELETE /episodes/{id} - Return 404 for non-existent episode."""
        mock_services["episodic"].delete = AsyncMock(return_value=False)
        fake_id = uuid4()
        response = api_client.delete(
            f"/api/v1/episodes/{fake_id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_list_episodes_default_pagination(self, api_client):
        """GET /episodes - List with default pagination."""
        response = api_client.get(
            "/api/v1/episodes",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "episodes" in data
        assert "total" in data
        assert "page" in data

    def test_list_episodes_invalid_page(self, api_client):
        """GET /episodes - Reject invalid page number."""
        response = api_client.get(
            "/api/v1/episodes?page=0",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_recall_episodes_success(self, api_client):
        """POST /episodes/recall - Search episodes."""
        response = api_client.post(
            "/api/v1/episodes/recall",
            json={"query": "test episode", "limit": 10},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "episodes" in data
        assert "scores" in data

    def test_recall_episodes_invalid_query_empty(self, api_client):
        """POST /episodes/recall - Reject empty query."""
        response = api_client.post(
            "/api/v1/episodes/recall",
            json={"query": ""},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_recall_episodes_invalid_limit(self, api_client):
        """POST /episodes/recall - Reject limit > 100."""
        response = api_client.post(
            "/api/v1/episodes/recall",
            json={"query": "test", "limit": 101},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422


# ============================================================================
# 2. ENTITY ENDPOINTS (/api/v1/entities)
# ============================================================================

class TestEntityEndpoints:
    """Test all entity CRUD and semantic operations."""

    def test_create_entity_success(self, api_client):
        """POST /entities - Create entity with valid data."""
        response = api_client.post(
            "/api/v1/entities",
            json={
                "name": "Test Entity",
                "entity_type": "CONCEPT",
                "summary": "A test concept",
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 201
        assert "id" in response.json()

    def test_create_entity_minimal(self, api_client):
        """POST /entities - Create with minimal fields."""
        response = api_client.post(
            "/api/v1/entities",
            json={
                "name": "Minimal",
                "entity_type": "CONCEPT",
                "summary": "Minimal entity",
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 201

    def test_create_entity_invalid_name_empty(self, api_client):
        """POST /entities - Reject empty name."""
        response = api_client.post(
            "/api/v1/entities",
            json={
                "name": "",
                "entity_type": "CONCEPT",
                "summary": "Test",
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_create_entity_invalid_type(self, api_client):
        """POST /entities - Reject invalid entity type."""
        response = api_client.post(
            "/api/v1/entities",
            json={
                "name": "Test",
                "entity_type": "not_a_valid_type",
                "summary": "Test",
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_get_entity_success(self, api_client, mock_entity):
        """GET /entities/{id} - Retrieve entity by ID."""
        response = api_client.get(
            f"/api/v1/entities/{mock_entity.id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == mock_entity.name

    def test_get_entity_not_found(self, api_client, mock_services):
        """GET /entities/{id} - Return 404 for non-existent entity."""
        mock_services["semantic"].get_entity = AsyncMock(return_value=None)
        fake_id = uuid4()
        response = api_client.get(
            f"/api/v1/entities/{fake_id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_update_entity_success(self, api_client, mock_entity):
        """PUT /entities/{id} - Update entity."""
        response = api_client.put(
            f"/api/v1/entities/{mock_entity.id}",
            json={"name": "Updated Name", "summary": "Updated summary"},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200

    def test_update_entity_not_found(self, api_client, mock_services):
        """PUT /entities/{id} - Return 404 for non-existent entity."""
        mock_services["semantic"].get_entity = AsyncMock(return_value=None)
        fake_id = uuid4()
        response = api_client.put(
            f"/api/v1/entities/{fake_id}",
            json={"name": "Updated"},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_delete_entity_success(self, api_client, mock_entity):
        """DELETE /entities/{id} - Delete entity."""
        response = api_client.delete(
            f"/api/v1/entities/{mock_entity.id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 204

    def test_delete_entity_not_found(self, api_client, mock_services):
        """DELETE /entities/{id} - Return 404 for non-existent entity."""
        mock_services["semantic"].delete_entity = AsyncMock(return_value=False)
        fake_id = uuid4()
        response = api_client.delete(
            f"/api/v1/entities/{fake_id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_list_entities_success(self, api_client):
        """GET /entities - List all entities."""
        response = api_client.get(
            "/api/v1/entities",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "total" in data

    def test_list_entities_filter_by_type(self, api_client):
        """GET /entities - Filter by entity type."""
        response = api_client.get(
            "/api/v1/entities?entity_type=CONCEPT",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200

    def test_recall_entities_success(self, api_client):
        """POST /entities/recall - Semantic search for entities."""
        response = api_client.post(
            "/api/v1/entities/recall",
            json={"query": "test concept", "limit": 10},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        assert "entities" in response.json()

    def test_recall_entities_with_type_filter(self, api_client):
        """POST /entities/recall - Recall with entity type filter."""
        response = api_client.post(
            "/api/v1/entities/recall",
            json={"query": "test", "entity_types": ["CONCEPT"], "limit": 5},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200

    def test_create_relation_success(self, api_client, mock_entity):
        """POST /entities/relations - Create relationship."""
        target_id = uuid4()
        response = api_client.post(
            "/api/v1/entities/relations",
            json={
                "source_id": str(mock_entity.id),
                "target_id": str(target_id),
                "relation_type": "PART_OF",
                "weight": 0.5,
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 201

    def test_create_relation_missing_source(self, api_client, mock_services):
        """POST /entities/relations - Return 404 if source doesn't exist."""
        mock_services["semantic"].get_entity = AsyncMock(side_effect=[None, None])
        target_id = uuid4()
        response = api_client.post(
            "/api/v1/entities/relations",
            json={
                "source_id": str(uuid4()),
                "target_id": str(target_id),
                "relation_type": "PART_OF",
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_spread_activation_success(self, api_client, mock_entity):
        """POST /entities/spread-activation - Perform spreading activation."""
        response = api_client.post(
            "/api/v1/entities/spread-activation",
            json={"entity_id": str(mock_entity.id), "depth": 2, "threshold": 0.1},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "activations" in data

    def test_supersede_entity_success(self, api_client, mock_entity):
        """POST /entities/{id}/supersede - Create new version."""
        response = api_client.post(
            f"/api/v1/entities/{mock_entity.id}/supersede",
            json={
                "name": "New Version",
                "entity_type": "CONCEPT",
                "summary": "Updated concept",
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200


# ============================================================================
# 3. SKILL ENDPOINTS (/api/v1/skills)
# ============================================================================

class TestSkillEndpoints:
    """Test all skill CRUD and execution endpoints."""

    def test_create_skill_success(self, api_client):
        """POST /skills - Create skill with steps."""
        response = api_client.post(
            "/api/v1/skills",
            json={
                "name": "Test Skill",
                "domain": "coding",
                "task": "Write and test code",
                "steps": [
                    {"order": 1, "action": "Write code", "tool": "editor"},
                    {"order": 2, "action": "Run tests", "tool": "pytest"},
                ],
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 201
        assert response.json()["name"]  # Response should contain name

    def test_create_skill_minimal(self, api_client):
        """POST /skills - Create skill with minimal fields."""
        response = api_client.post(
            "/api/v1/skills",
            json={
                "name": "Minimal Skill",
                "domain": "coding",
                "task": "Minimal task",
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 201

    def test_create_skill_invalid_name_empty(self, api_client):
        """POST /skills - Reject empty name."""
        response = api_client.post(
            "/api/v1/skills",
            json={"name": "", "domain": "coding", "task": "Test"},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_create_skill_invalid_domain(self, api_client):
        """POST /skills - Reject invalid domain."""
        response = api_client.post(
            "/api/v1/skills",
            json={"name": "Test", "domain": "invalid_domain", "task": "Test"},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_get_skill_success(self, api_client, mock_procedure):
        """GET /skills/{id} - Retrieve skill by ID."""
        response = api_client.get(
            f"/api/v1/skills/{mock_procedure.id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == mock_procedure.name

    def test_get_skill_not_found(self, api_client, mock_services):
        """GET /skills/{id} - Return 404 for non-existent skill."""
        mock_services["procedural"].get_procedure = AsyncMock(return_value=None)
        fake_id = uuid4()
        response = api_client.get(
            f"/api/v1/skills/{fake_id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_update_skill_success(self, api_client, mock_procedure):
        """PUT /skills/{id} - Update skill."""
        response = api_client.put(
            f"/api/v1/skills/{mock_procedure.id}",
            json={"name": "Updated Skill", "trigger_pattern": "when code needs testing"},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200

    def test_update_skill_not_found(self, api_client, mock_services):
        """PUT /skills/{id} - Return 404 for non-existent skill."""
        mock_services["procedural"].get_procedure = AsyncMock(return_value=None)
        fake_id = uuid4()
        response = api_client.put(
            f"/api/v1/skills/{fake_id}",
            json={"name": "Updated"},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_delete_skill_success(self, api_client, mock_procedure):
        """DELETE /skills/{id} - Delete skill."""
        response = api_client.delete(
            f"/api/v1/skills/{mock_procedure.id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 204

    def test_delete_skill_not_found(self, api_client, mock_services):
        """DELETE /skills/{id} - Return 404 for non-existent skill."""
        mock_services["procedural"].delete = AsyncMock(return_value=False)
        fake_id = uuid4()
        response = api_client.delete(
            f"/api/v1/skills/{fake_id}",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_list_skills_success(self, api_client):
        """GET /skills - List all skills."""
        response = api_client.get(
            "/api/v1/skills",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "skills" in data
        assert "total" in data

    def test_list_skills_filter_domain(self, api_client):
        """GET /skills - Filter by domain."""
        response = api_client.get(
            "/api/v1/skills?domain=coding",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200

    def test_recall_skills_success(self, api_client):
        """POST /skills/recall - Semantic search for skills."""
        response = api_client.post(
            "/api/v1/skills/recall",
            json={"query": "how to test code", "limit": 5},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        assert "skills" in response.json()

    def test_record_execution_success(self, api_client, mock_procedure):
        """POST /skills/{id}/execute - Record skill execution."""
        response = api_client.post(
            f"/api/v1/skills/{mock_procedure.id}/execute",
            json={"success": True, "duration_ms": 5000},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200

    def test_record_execution_not_found(self, api_client, mock_services):
        """POST /skills/{id}/execute - Return 404 for non-existent skill."""
        mock_services["procedural"].get_procedure = AsyncMock(return_value=None)
        fake_id = uuid4()
        response = api_client.post(
            f"/api/v1/skills/{fake_id}/execute",
            json={"success": True},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 404

    def test_deprecate_skill_success(self, api_client, mock_procedure):
        """POST /skills/{id}/deprecate - Deprecate a skill."""
        response = api_client.post(
            f"/api/v1/skills/{mock_procedure.id}/deprecate",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200

    def test_how_to_query_success(self, api_client):
        """GET /skills/how-to/{query} - Natural language procedural query."""
        response = api_client.get(
            "/api/v1/skills/how-to/how%20to%20write%20tests",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "steps" in data


# ============================================================================
# 4. CONFIG ENDPOINTS (/api/v1/config)
# ============================================================================

class TestConfigEndpoints:
    """Test configuration management endpoints."""

    def test_get_config_success(self, api_client):
        """GET /config - Retrieve full system configuration."""
        response = api_client.get(
            "/api/v1/config",
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200

    def test_get_config_no_auth_required(self, api_client):
        """GET /config - No auth required for read access."""
        response = api_client.get("/api/v1/config")
        assert response.status_code == 200

    def test_update_config_requires_admin_key(self, api_client):
        """PUT /config - Require X-Admin-Key header."""
        response = api_client.put(
            "/api/v1/config",
            json={"fsrs": {"defaultStability": 1.5}},
        )
        # Should fail without auth (401, 403, or 422)
        assert response.status_code in [401, 403, 422]

    def test_reset_config_requires_auth(self, api_client):
        """POST /config/reset - Require authentication."""
        response = api_client.post("/api/v1/config/reset")
        assert response.status_code in [401, 403, 422]


# ============================================================================
# 5. INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Test input validation across all endpoints."""

    def test_missing_required_fields(self, api_client):
        """Reject requests with missing required fields."""
        response = api_client.post(
            "/api/v1/episodes",
            json={"project": "test"},  # Missing 'content'
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_type_mismatch(self, api_client):
        """Reject values with wrong type."""
        response = api_client.post(
            "/api/v1/episodes",
            json={
                "content": "Test",
                "emotional_valence": "not a number",  # Should be float
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_enum_validation(self, api_client):
        """Reject invalid enum values."""
        response = api_client.post(
            "/api/v1/entities",
            json={
                "name": "Test",
                "entity_type": "not_a_valid_type",
                "summary": "Test",
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422


# ============================================================================
# 6. ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and response formats."""

    def test_404_response_structure(self, api_client, mock_services):
        """404 responses should have proper structure."""
        mock_services["episodic"].get = AsyncMock(return_value=None)
        fake_id = uuid4()
        response = api_client.get(
            f"/api/v1/episodes/{fake_id}",
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_validation_error_response(self, api_client):
        """Validation errors should include field information."""
        response = api_client.post(
            "/api/v1/episodes",
            json={
                "content": "",  # Invalid: empty
                "emotional_valence": 2.0  # Invalid: > 1.0
            },
            headers={"X-Session-ID": "test-session"},
        )

        assert response.status_code == 422
        assert "detail" in response.json()


# ============================================================================
# 7. RESPONSE SCHEMA TESTS
# ============================================================================

class TestResponseSchemas:
    """Test response schemas match documented models."""

    def test_episode_response_schema(self, api_client, mock_episode):
        """Episode response contains all required fields."""
        response = api_client.get(
            f"/api/v1/episodes/{mock_episode.id}",
            headers={"X-Session-ID": "test-session"},
        )
        data = response.json()

        required_fields = [
            "id", "session_id", "content", "timestamp",
            "outcome", "emotional_valence", "context",
            "access_count", "stability"
        ]

        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_entity_response_schema(self, api_client, mock_entity):
        """Entity response contains all required fields."""
        response = api_client.get(
            f"/api/v1/entities/{mock_entity.id}",
            headers={"X-Session-ID": "test-session"},
        )
        data = response.json()

        required_fields = [
            "id", "name", "entity_type", "summary",
            "stability", "access_count", "created_at"
        ]

        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_skill_response_schema(self, api_client, mock_procedure):
        """Skill response contains all required fields."""
        response = api_client.get(
            f"/api/v1/skills/{mock_procedure.id}",
            headers={"X-Session-ID": "test-session"},
        )
        data = response.json()

        required_fields = [
            "id", "name", "domain", "steps",
            "success_rate", "execution_count",
            "version", "deprecated", "created_at"
        ]

        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# ============================================================================
# 8. EDGE CASES & BOUNDARY TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_max_query_length_boundary(self, api_client):
        """Test max_length boundary for query (10KB)."""
        # Valid: exactly at limit
        response = api_client.post(
            "/api/v1/episodes/recall",
            json={"query": "x" * 10000, "limit": 10},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 200

        # Invalid: exceeds limit
        response = api_client.post(
            "/api/v1/episodes/recall",
            json={"query": "x" * 10001},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 422

    def test_emotional_valence_boundaries(self, api_client):
        """Test emotional valence boundary conditions."""
        # Valid: boundaries
        for valence in [0.0, 0.5, 1.0]:
            response = api_client.post(
                "/api/v1/episodes",
                json={"content": "Test", "emotional_valence": valence},
                headers={"X-Session-ID": "test-session"},
            )
            assert response.status_code == 201

        # Invalid: outside boundaries
        for valence in [-0.1, 1.1]:
            response = api_client.post(
                "/api/v1/episodes",
                json={"content": "Test", "emotional_valence": valence},
                headers={"X-Session-ID": "test-session"},
            )
            assert response.status_code == 422

    def test_unicode_content(self, api_client):
        """Test Unicode content handling."""
        response = api_client.post(
            "/api/v1/episodes",
            json={"content": "Test with emoji 🚀 and Chinese 中文"},
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 201

    def test_special_characters_in_strings(self, api_client):
        """Test special characters in string fields."""
        response = api_client.post(
            "/api/v1/episodes",
            json={
                "content": 'Test with "quotes" and \'apostrophes\' and \\ backslashes'
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert response.status_code == 201


# ============================================================================
# 9. INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test multi-step workflows and interactions."""

    def test_episode_to_entity_workflow(self, api_client):
        """Test creating episode then entity from same session."""
        # Create episode
        ep_response = api_client.post(
            "/api/v1/episodes",
            json={"content": "Learned about Python"},
            headers={"X-Session-ID": "test-session"},
        )
        assert ep_response.status_code == 201

        # Create entity from context
        ent_response = api_client.post(
            "/api/v1/entities",
            json={
                "name": "Python",
                "entity_type": "TOOL",
                "summary": "Programming language",
                "source": ep_response.json()["id"],
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert ent_response.status_code == 201

    def test_skill_execution_tracking_workflow(self, api_client):
        """Test creating skill and recording executions."""
        # Create skill
        skill_response = api_client.post(
            "/api/v1/skills",
            json={
                "name": "Test Skill",
                "domain": "coding",
                "task": "Write tests",
                "steps": [{"order": 1, "action": "Write test"}],
            },
            headers={"X-Session-ID": "test-session"},
        )
        assert skill_response.status_code == 201
        skill_id = skill_response.json()["id"]

        # Record execution
        exec_response = api_client.post(
            f"/api/v1/skills/{skill_id}/execute",
            json={"success": True, "duration_ms": 5000},
            headers={"X-Session-ID": "test-session"},
        )
        assert exec_response.status_code == 200
