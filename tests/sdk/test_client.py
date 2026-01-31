"""Tests for World Weaver SDK client."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx

from t4dm.sdk.client import (
    AsyncWorldWeaverClient,
    WorldWeaverError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
)
from t4dm.sdk.models import Episode, Entity, Skill, Step, EpisodeContext


class TestWorldWeaverError:
    """Tests for SDK exceptions."""

    def test_base_error(self):
        """Base error has message and status."""
        err = WorldWeaverError("test error", status_code=500, response={"error": "fail"})
        assert str(err) == "test error"
        assert err.status_code == 500
        assert err.response == {"error": "fail"}

    def test_connection_error(self):
        """Connection error is a WorldWeaverError."""
        err = ConnectionError("connection failed")
        assert isinstance(err, WorldWeaverError)

    def test_not_found_error(self):
        """NotFoundError is a WorldWeaverError."""
        err = NotFoundError("resource not found", status_code=404)
        assert err.status_code == 404

    def test_rate_limit_error(self):
        """RateLimitError has retry_after."""
        err = RateLimitError("too many requests", retry_after=60)
        assert err.status_code == 429
        assert err.retry_after == 60


class TestAsyncWorldWeaverClient:
    """Tests for AsyncWorldWeaverClient class."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return AsyncWorldWeaverClient(
            base_url="http://localhost:8765",
            session_id="test-session",
            timeout=10.0,
        )

    def test_initialization(self, client):
        """Client initializes with correct settings."""
        assert client.base_url == "http://localhost:8765"
        assert client.session_id == "test-session"
        assert client.timeout == 10.0
        assert client._client is None

    def test_initialization_strips_trailing_slash(self):
        """Base URL trailing slash is stripped."""
        client = AsyncWorldWeaverClient(base_url="http://localhost:8765/")
        assert client.base_url == "http://localhost:8765"

    @pytest.mark.asyncio
    async def test_connect(self, client):
        """Connect initializes httpx client."""
        await client.connect()
        assert client._client is not None
        assert isinstance(client._client, httpx.AsyncClient)
        await client.close()

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Close cleans up httpx client."""
        await client.connect()
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Async context manager works."""
        async with AsyncWorldWeaverClient() as client:
            assert client._client is not None
        assert client._client is None

    def test_get_client_raises_when_not_connected(self, client):
        """Get client raises when not connected."""
        with pytest.raises(WorldWeaverError, match="not connected"):
            client._get_client()


class TestAsyncClientEpisodes:
    """Tests for episode-related client methods."""

    @pytest.fixture
    def mock_response(self):
        """Create mock response."""
        response = MagicMock()
        response.status_code = 200
        response.content = b'{"test": "data"}'
        return response

    def _make_episode_response(self, content="test content"):
        """Helper to create full episode response."""
        return {
            "id": str(uuid4()),
            "session_id": "test-session",
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "outcome": "neutral",
            "emotional_valence": 0.5,
            "access_count": 0,
            "stability": 0.5,
            "context": {"project": None, "file": None, "tool": None},
        }

    @pytest.mark.asyncio
    async def test_create_episode(self):
        """Create episode makes correct request."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = self._make_episode_response("test content")
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        episode = await client.create_episode("test content")
        assert episode.content == "test content"

    @pytest.mark.asyncio
    async def test_get_episode(self):
        """Get episode by ID."""
        episode_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = self._make_episode_response("retrieved episode")
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        episode = await client.get_episode(episode_id)
        assert episode.content == "retrieved episode"

    @pytest.mark.asyncio
    async def test_list_episodes(self):
        """List episodes with pagination."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "episodes": [self._make_episode_response("ep1")],
            "total": 1,
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        episodes, total = await client.list_episodes()
        assert len(episodes) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_recall_episodes(self):
        """Recall episodes by query."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "query": "search term",
            "episodes": [],
            "scores": [],
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        result = await client.recall_episodes("search term")
        assert result.query == "search term"


class TestAsyncClientEntities:
    """Tests for entity-related client methods."""

    @pytest.mark.asyncio
    async def test_create_entity(self):
        """Create entity makes correct request."""
        entity_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "id": str(entity_id),
            "name": "Python",
            "entity_type": "technology",
            "summary": "A programming language",
            "details": None,
            "source": None,
            "stability": 0.5,
            "access_count": 0,
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        entity = await client.create_entity("Python", "technology", "A programming language")
        assert entity.name == "Python"
        assert entity.entity_type == "technology"


class TestAsyncClientErrorHandling:
    """Tests for client error handling."""

    @pytest.mark.asyncio
    async def test_404_raises_not_found(self):
        """404 response raises NotFoundError."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.content = b'{"error": "not found"}'
        mock_response.json.return_value = {"error": "not found"}
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        with pytest.raises(NotFoundError):
            await client.get_episode(uuid4())

    @pytest.mark.asyncio
    async def test_429_raises_rate_limit(self):
        """429 response raises RateLimitError."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.content = b'{}'
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        with pytest.raises(RateLimitError) as exc:
            await client.health()
        assert exc.value.retry_after == 60

    @pytest.mark.asyncio
    async def test_500_raises_generic_error(self):
        """500 response raises WorldWeaverError."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.content = b'{"error": "internal"}'
        mock_response.json.return_value = {"error": "internal"}
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        with pytest.raises(WorldWeaverError) as exc:
            await client.health()
        assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_connect_error_raises_connection_error(self):
        """Connection error raises ConnectionError."""
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        with pytest.raises(ConnectionError):
            await client.health()

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self):
        """Timeout raises WorldWeaverError."""
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        with pytest.raises(WorldWeaverError, match="timed out"):
            await client.health()


class TestAsyncClientHealthAndStats:
    """Tests for health and stats methods."""

    @pytest.mark.asyncio
    async def test_health(self):
        """Health check returns status."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "session_id": "test",
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        health = await client.health()
        assert health.status == "healthy"

    @pytest.mark.asyncio
    async def test_stats(self):
        """Stats returns memory statistics."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "session_id": "test-session",
            "episodic": {"count": 100},
            "semantic": {"count": 50},
            "procedural": {"count": 10},
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        stats = await client.stats()
        assert stats.episodic["count"] == 100

    @pytest.mark.asyncio
    async def test_consolidate(self):
        """Consolidate triggers memory consolidation."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "status": "completed",
            "consolidated": 50,
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        result = await client.consolidate(deep=True)
        assert result["status"] == "completed"


class TestAsyncClientSkills:
    """Tests for skill-related client methods."""

    def _make_skill_response(self, name="test-skill"):
        """Helper to create skill response."""
        return {
            "id": str(uuid4()),
            "name": name,
            "domain": "testing",
            "trigger_pattern": None,
            "steps": [{"order": 1, "action": "test action", "tool": None, "parameters": {}, "expected_outcome": None}],
            "script": None,
            "success_rate": 0.95,
            "execution_count": 10,
            "last_executed": datetime.utcnow().isoformat(),
            "version": 1,
            "deprecated": False,
            "created_at": datetime.utcnow().isoformat(),
        }

    @pytest.mark.asyncio
    async def test_create_skill(self):
        """Create skill makes correct request."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = self._make_skill_response("my-skill")
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        skill = await client.create_skill("my-skill", "testing", "test task")
        assert skill.name == "my-skill"
        assert skill.domain == "testing"

    @pytest.mark.asyncio
    async def test_get_skill(self):
        """Get skill by ID."""
        skill_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = self._make_skill_response("retrieved-skill")
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        skill = await client.get_skill(skill_id)
        assert skill.name == "retrieved-skill"

    @pytest.mark.asyncio
    async def test_list_skills(self):
        """List skills with filtering."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "skills": [self._make_skill_response("skill1"), self._make_skill_response("skill2")],
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        skills = await client.list_skills(domain="testing")
        assert len(skills) == 2

    @pytest.mark.asyncio
    async def test_recall_skills(self):
        """Recall skills by query."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "skills": [self._make_skill_response("matched-skill")],
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        skills = await client.recall_skills("test query", domain="testing")
        assert len(skills) == 1
        assert skills[0].name == "matched-skill"

    @pytest.mark.asyncio
    async def test_record_execution(self):
        """Record skill execution."""
        skill_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        resp = self._make_skill_response("executed-skill")
        resp["execution_count"] = 11
        mock_response.json.return_value = resp
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        skill = await client.record_execution(skill_id, success=True, duration_ms=100)
        assert skill.execution_count == 11

    @pytest.mark.asyncio
    async def test_deprecate_skill(self):
        """Deprecate a skill."""
        skill_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        resp = self._make_skill_response("deprecated-skill")
        resp["deprecated"] = True
        mock_response.json.return_value = resp
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        skill = await client.deprecate_skill(skill_id, replacement_id=uuid4())
        assert skill.deprecated is True

    @pytest.mark.asyncio
    async def test_how_to(self):
        """How-to query returns skill and steps."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "skill": self._make_skill_response("howto-skill"),
            "steps": ["Step 1: Do this", "Step 2: Do that"],
            "confidence": 0.85,
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        skill, steps, confidence = await client.how_to("how to test", domain="testing")
        assert skill.name == "howto-skill"
        assert len(steps) == 2
        assert confidence == 0.85

    @pytest.mark.asyncio
    async def test_how_to_no_skill_found(self):
        """How-to returns None when no skill matches."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "skill": None,
            "steps": [],
            "confidence": 0.0,
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        skill, steps, confidence = await client.how_to("unknown task")
        assert skill is None
        assert steps == []
        assert confidence == 0.0


class TestAsyncClientMoreEntities:
    """Additional tests for entity-related client methods."""

    def _make_entity_response(self, name="test-entity"):
        """Helper to create entity response."""
        return {
            "id": str(uuid4()),
            "name": name,
            "entity_type": "concept",
            "summary": f"A {name}",
            "details": None,
            "source": None,
            "stability": 0.5,
            "access_count": 0,
            "created_at": datetime.utcnow().isoformat(),
        }

    @pytest.mark.asyncio
    async def test_get_entity(self):
        """Get entity by ID."""
        entity_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = self._make_entity_response("retrieved-entity")
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        entity = await client.get_entity(entity_id)
        assert entity.name == "retrieved-entity"

    @pytest.mark.asyncio
    async def test_list_entities(self):
        """List entities with type filtering."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "entities": [self._make_entity_response("entity1"), self._make_entity_response("entity2")],
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        entities = await client.list_entities(entity_type="concept", limit=50)
        assert len(entities) == 2

    @pytest.mark.asyncio
    async def test_create_relation(self):
        """Create relationship between entities."""
        source_id = uuid4()
        target_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "source_id": str(source_id),
            "target_id": str(target_id),
            "relation_type": "related_to",
            "weight": 0.5,
            "co_access_count": 0,
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        rel = await client.create_relation(source_id, target_id, "related_to", weight=0.5)
        assert rel.relation_type == "related_to"
        assert rel.weight == 0.5

    @pytest.mark.asyncio
    async def test_recall_entities(self):
        """Recall entities by query."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "entities": [self._make_entity_response("found-entity")],
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        entities = await client.recall_entities("test", limit=10, entity_types=["concept"])
        assert len(entities) == 1
        assert entities[0].name == "found-entity"

    @pytest.mark.asyncio
    async def test_spread_activation(self):
        """Spread activation from entity."""
        entity_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "entities": [self._make_entity_response("activated-entity")],
            "activations": [0.9],
            "paths": [["node1", "node2"]],
        }
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        result = await client.spread_activation(entity_id, depth=2, threshold=0.1)
        assert len(result.entities) == 1
        assert result.activations[0] == 0.9

    @pytest.mark.asyncio
    async def test_supersede_entity(self):
        """Supersede an entity with updated info."""
        entity_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = self._make_entity_response("superseded-entity")
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        entity = await client.supersede_entity(entity_id, "superseded-entity", "concept", "Updated summary", details="More details")
        assert entity.name == "superseded-entity"


class TestAsyncClientMoreEpisodes:
    """Additional tests for episode-related client methods."""

    def _make_episode_response(self, content="test content"):
        """Helper to create episode response."""
        return {
            "id": str(uuid4()),
            "session_id": "test-session",
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "outcome": "neutral",
            "emotional_valence": 0.5,
            "access_count": 0,
            "stability": 0.5,
            "context": {"project": None, "file": None, "tool": None},
        }

    @pytest.mark.asyncio
    async def test_delete_episode(self):
        """Delete an episode."""
        episode_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b''
        mock_response.json.return_value = {}
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        await client.delete_episode(episode_id)
        mock_client.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_important(self):
        """Mark an episode as important."""
        episode_id = uuid4()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = self._make_episode_response("important episode")
        mock_client.request = AsyncMock(return_value=mock_response)

        client = AsyncWorldWeaverClient()
        client._client = mock_client

        episode = await client.mark_important(episode_id, importance=0.9)
        assert episode.content == "important episode"


class TestWorldWeaverSyncClient:
    """Tests for synchronous WorldWeaverClient."""

    from t4dm.sdk.client import WorldWeaverClient

    def test_initialization(self):
        """Sync client initializes correctly."""
        client = self.WorldWeaverClient(
            base_url="http://localhost:8765",
            session_id="test-session",
            timeout=15.0,
        )
        assert client.base_url == "http://localhost:8765"
        assert client.session_id == "test-session"
        assert client.timeout == 15.0
        assert client._client is None

    def test_connect(self):
        """Connect initializes httpx sync client."""
        client = self.WorldWeaverClient()
        client.connect()
        assert client._client is not None
        client.close()

    def test_close(self):
        """Close cleans up httpx client."""
        client = self.WorldWeaverClient()
        client.connect()
        client.close()
        assert client._client is None

    def test_context_manager(self):
        """Context manager works for sync client."""
        with self.WorldWeaverClient() as client:
            assert client._client is not None
        assert client._client is None

    def test_get_client_raises_when_not_connected(self):
        """Get client raises when not connected."""
        from t4dm.sdk.client import WorldWeaverError
        client = self.WorldWeaverClient()
        with pytest.raises(WorldWeaverError, match="not connected"):
            client._get_client()


class TestWorldWeaverSyncClientMethods:
    """Tests for sync client API methods."""

    from t4dm.sdk.client import WorldWeaverClient

    def _make_episode_response(self, content="test content"):
        """Helper to create episode response."""
        return {
            "id": str(uuid4()),
            "session_id": "test-session",
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "outcome": "neutral",
            "emotional_valence": 0.5,
            "access_count": 0,
            "stability": 0.5,
            "context": {"project": None, "file": None, "tool": None},
        }

    def _make_entity_response(self, name="test-entity"):
        """Helper to create entity response."""
        return {
            "id": str(uuid4()),
            "name": name,
            "entity_type": "concept",
            "summary": f"A {name}",
            "details": None,
            "source": None,
            "stability": 0.5,
            "access_count": 0,
            "created_at": datetime.utcnow().isoformat(),
        }

    def _make_skill_response(self, name="test-skill"):
        """Helper to create skill response."""
        return {
            "id": str(uuid4()),
            "name": name,
            "domain": "testing",
            "trigger_pattern": None,
            "steps": [],
            "script": None,
            "success_rate": 0.95,
            "execution_count": 10,
            "last_executed": None,
            "version": 1,
            "deprecated": False,
            "created_at": datetime.utcnow().isoformat(),
        }

    def test_health(self):
        """Sync health check."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
        }

        with patch.object(client._client, 'request', return_value=mock_response):
            health = client.health()
            assert health.status == "healthy"

        client.close()

    def test_create_episode(self):
        """Sync create episode."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = self._make_episode_response("new episode")

        with patch.object(client._client, 'request', return_value=mock_response):
            episode = client.create_episode("new episode", project="test")
            assert episode.content == "new episode"

        client.close()

    def test_recall_episodes(self):
        """Sync recall episodes."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "query": "test",
            "episodes": [self._make_episode_response("found")],
            "scores": [0.9],
        }

        with patch.object(client._client, 'request', return_value=mock_response):
            result = client.recall_episodes("test", limit=10)
            assert result.query == "test"
            assert len(result.episodes) == 1

        client.close()

    def test_create_entity(self):
        """Sync create entity."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = self._make_entity_response("new-entity")

        with patch.object(client._client, 'request', return_value=mock_response):
            entity = client.create_entity("new-entity", "concept", "A concept")
            assert entity.name == "new-entity"

        client.close()

    def test_recall_entities(self):
        """Sync recall entities."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "entities": [self._make_entity_response("found-entity")],
        }

        with patch.object(client._client, 'request', return_value=mock_response):
            entities = client.recall_entities("test")
            assert len(entities) == 1
            assert entities[0].name == "found-entity"

        client.close()

    def test_create_skill(self):
        """Sync create skill."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = self._make_skill_response("new-skill")

        with patch.object(client._client, 'request', return_value=mock_response):
            skill = client.create_skill("new-skill", "testing", "test task")
            assert skill.name == "new-skill"

        client.close()

    def test_how_to(self):
        """Sync how-to query."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "skill": self._make_skill_response("howto-skill"),
            "steps": ["Step 1"],
            "confidence": 0.8,
        }

        with patch.object(client._client, 'request', return_value=mock_response):
            skill, steps, confidence = client.how_to("test query")
            assert skill.name == "howto-skill"
            assert len(steps) == 1
            assert confidence == 0.8

        client.close()

    def test_how_to_no_skill(self):
        """Sync how-to with no skill found."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{}'
        mock_response.json.return_value = {
            "skill": None,
            "steps": [],
            "confidence": 0.0,
        }

        with patch.object(client._client, 'request', return_value=mock_response):
            skill, steps, confidence = client.how_to("unknown")
            assert skill is None
            assert steps == []
            assert confidence == 0.0

        client.close()


class TestSyncClientErrorHandling:
    """Tests for sync client error handling."""

    from t4dm.sdk.client import WorldWeaverClient

    def test_404_raises_not_found(self):
        """404 raises NotFoundError in sync client."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.content = b'{"error": "not found"}'
        mock_response.json.return_value = {"error": "not found"}

        with patch.object(client._client, 'request', return_value=mock_response):
            with pytest.raises(NotFoundError):
                client.health()

        client.close()

    def test_429_raises_rate_limit(self):
        """429 raises RateLimitError in sync client."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "30"}
        mock_response.content = b'{}'

        with patch.object(client._client, 'request', return_value=mock_response):
            with pytest.raises(RateLimitError) as exc:
                client.health()
            assert exc.value.retry_after == 30

        client.close()

    def test_500_raises_error(self):
        """500 raises WorldWeaverError in sync client."""
        client = self.WorldWeaverClient()
        client.connect()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.content = b'{"error": "internal"}'
        mock_response.json.return_value = {"error": "internal"}

        with patch.object(client._client, 'request', return_value=mock_response):
            with pytest.raises(WorldWeaverError) as exc:
                client.health()
            assert exc.value.status_code == 500

        client.close()

    def test_connect_error(self):
        """Connection error raises ConnectionError in sync client."""
        client = self.WorldWeaverClient()
        client.connect()

        with patch.object(client._client, 'request', side_effect=httpx.ConnectError("refused")):
            with pytest.raises(ConnectionError):
                client.health()

        client.close()

    def test_timeout_error(self):
        """Timeout raises WorldWeaverError in sync client."""
        client = self.WorldWeaverClient()
        client.connect()

        with patch.object(client._client, 'request', side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(WorldWeaverError, match="timed out"):
                client.health()

        client.close()
