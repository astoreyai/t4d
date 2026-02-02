"""
Unit tests for T4DM SDK.

Tests SDK models, async client, sync client, and error handling.
"""

import pytest
import json
from datetime import datetime
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from t4dm.sdk import (
    WorldWeaverClient,
    AsyncWorldWeaverClient,
    Episode,
    Entity,
    Skill,
    RecallResult,
    ActivationResult,
)
from t4dm.sdk.models import (
    EpisodeContext,
    Relationship,
    Step,
    HealthStatus,
    MemoryStats,
)
from t4dm.sdk.client import (
    WorldWeaverError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
)


class TestSDKModels:
    """Tests for SDK data models."""

    def test_episode_context_creation(self):
        ctx = EpisodeContext(project="test_project", file="test.py", tool="edit")
        assert ctx.project == "test_project"
        assert ctx.file == "test.py"
        assert ctx.tool == "edit"

    def test_episode_context_defaults(self):
        ctx = EpisodeContext()
        assert ctx.project is None
        assert ctx.file is None
        assert ctx.tool is None

    def test_episode_creation(self):
        ep_id = uuid4()
        now = datetime.now()
        ctx = EpisodeContext(project="my_proj")
        episode = Episode(
            id=ep_id,
            session_id="session_123",
            content="Test content",
            timestamp=now,
            outcome="success",
            emotional_valence=0.8,
            context=ctx,
            access_count=3,
            stability=0.9,
        )
        assert episode.id == ep_id
        assert episode.session_id == "session_123"
        assert episode.content == "Test content"
        assert episode.outcome == "success"
        assert episode.emotional_valence == 0.8
        assert episode.context.project == "my_proj"

    def test_episode_with_retrievability(self):
        episode = Episode(
            id=uuid4(),
            session_id="sess",
            content="content",
            timestamp=datetime.now(),
            outcome="neutral",
            emotional_valence=0.5,
            context=EpisodeContext(),
            access_count=1,
            stability=0.5,
            retrievability=0.75,
        )
        assert episode.retrievability == 0.75

    def test_entity_creation(self):
        entity = Entity(
            id=uuid4(),
            name="Python",
            entity_type="language",
            summary="A programming language",
            details="Python is versatile",
            source="documentation",
            stability=0.95,
            access_count=10,
            created_at=datetime.now(),
        )
        assert entity.name == "Python"
        assert entity.entity_type == "language"
        assert entity.details == "Python is versatile"

    def test_entity_minimal(self):
        entity = Entity(
            id=uuid4(),
            name="Entity",
            entity_type="concept",
            summary="A summary",
            stability=0.5,
            access_count=0,
            created_at=datetime.now(),
        )
        assert entity.details is None
        assert entity.source is None

    def test_relationship_creation(self):
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type="related_to",
            weight=0.8,
            co_access_count=5,
        )
        assert rel.relation_type == "related_to"
        assert rel.weight == 0.8

    def test_step_creation(self):
        step = Step(
            order=1,
            action="Run pytest",
            tool="bash",
            parameters={"command": "pytest"},
            expected_outcome="Tests pass",
        )
        assert step.order == 1
        assert step.tool == "bash"
        assert step.parameters["command"] == "pytest"

    def test_step_defaults(self):
        step = Step(order=1, action="Do something")
        assert step.tool is None
        assert step.parameters == {}
        assert step.expected_outcome is None

    def test_skill_creation(self):
        skill = Skill(
            id=uuid4(),
            name="run_tests",
            domain="development",
            trigger_pattern="run.*tests",
            steps=[Step(order=1, action="pytest")],
            script=None,
            success_rate=0.9,
            execution_count=10,
            last_executed=datetime.now(),
            version=2,
            deprecated=False,
            created_at=datetime.now(),
        )
        assert skill.name == "run_tests"
        assert skill.success_rate == 0.9
        assert len(skill.steps) == 1

    def test_skill_deprecated(self):
        skill = Skill(
            id=uuid4(),
            name="old_skill",
            domain="legacy",
            steps=[],
            success_rate=0.5,
            execution_count=100,
            version=1,
            deprecated=True,
            created_at=datetime.now(),
        )
        assert skill.deprecated is True

    def test_recall_result_creation(self):
        episodes = [
            Episode(
                id=uuid4(),
                session_id="sess",
                content="test",
                timestamp=datetime.now(),
                outcome="neutral",
                emotional_valence=0.5,
                context=EpisodeContext(),
                access_count=1,
                stability=0.5,
            )
        ]
        result = RecallResult(query="test query", episodes=episodes, scores=[0.9])
        assert result.query == "test query"
        assert len(result.episodes) == 1
        assert result.scores == [0.9]

    def test_activation_result_creation(self):
        entities = [
            Entity(
                id=uuid4(),
                name="Test",
                entity_type="concept",
                summary="Test entity",
                stability=0.5,
                access_count=0,
                created_at=datetime.now(),
            )
        ]
        result = ActivationResult(
            entities=entities,
            activations=[1.0, 0.8, 0.5],
            paths=[["a", "b"], ["a", "c"]],
        )
        assert len(result.entities) == 1
        assert result.activations == [1.0, 0.8, 0.5]

    def test_health_status_creation(self):
        status = HealthStatus(
            status="healthy",
            timestamp="2024-01-01T00:00:00",
            version="1.0.0",
            session_id="sess_123",
        )
        assert status.status == "healthy"
        assert status.version == "1.0.0"

    def test_memory_stats_creation(self):
        stats = MemoryStats(
            session_id="sess",
            episodic={"count": 100},
            semantic={"entity_count": 50},
            procedural={"skill_count": 25},
        )
        assert stats.episodic["count"] == 100
        assert stats.semantic["entity_count"] == 50


class TestSDKExceptions:
    """Tests for SDK exception classes."""

    def test_world_weaver_error(self):
        error = WorldWeaverError("Test error", status_code=500, response={"error": "detail"})
        assert str(error) == "Test error"
        assert error.status_code == 500
        assert error.response == {"error": "detail"}

    def test_connection_error(self):
        error = ConnectionError("Cannot connect")
        assert isinstance(error, WorldWeaverError)
        assert str(error) == "Cannot connect"

    def test_not_found_error(self):
        error = NotFoundError("Resource not found", status_code=404)
        assert isinstance(error, WorldWeaverError)
        assert error.status_code == 404

    def test_rate_limit_error(self):
        error = RateLimitError("Too many requests", retry_after=60)
        assert error.status_code == 429
        assert error.retry_after == 60


class TestAsyncWorldWeaverClient:
    """Tests for AsyncWorldWeaverClient."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock HTTP response."""
        def _make_response(data: dict, status_code: int = 200):
            response = MagicMock()
            response.status_code = status_code
            response.content = json.dumps(data).encode() if data else b""
            response.json.return_value = data
            response.headers = {}
            return response
        return _make_response

    @pytest.fixture
    def client(self):
        return AsyncWorldWeaverClient(
            base_url="http://localhost:8765",
            session_id="test_session",
            timeout=10.0,
        )

    def test_client_creation(self, client):
        assert client.base_url == "http://localhost:8765"
        assert client.session_id == "test_session"
        assert client.timeout == 10.0
        assert client._client is None

    def test_client_creation_strips_trailing_slash(self):
        client = AsyncWorldWeaverClient(base_url="http://localhost:8765/")
        assert client.base_url == "http://localhost:8765"

    @pytest.mark.asyncio
    async def test_connect_creates_client(self, client):
        await client.connect()
        assert client._client is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_close_removes_client(self, client):
        await client.connect()
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncWorldWeaverClient() as client:
            assert client._client is not None
        assert client._client is None

    @pytest.mark.asyncio
    async def test_get_client_raises_if_not_connected(self, client):
        with pytest.raises(WorldWeaverError) as exc:
            client._get_client()
        assert "not connected" in str(exc.value)

    @pytest.mark.asyncio
    async def test_request_handles_404(self, client, mock_response):
        await client.connect()

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response({"error": "Not found"}, 404)

            with pytest.raises(NotFoundError) as exc:
                await client._request("GET", "/missing")

            assert exc.value.status_code == 404

        await client.close()

    @pytest.mark.asyncio
    async def test_request_handles_429(self, client, mock_response):
        await client.connect()

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            response = mock_response(None, 429)
            response.headers = {"Retry-After": "30"}
            mock.return_value = response

            with pytest.raises(RateLimitError) as exc:
                await client._request("GET", "/rate_limited")

            assert exc.value.retry_after == 30

        await client.close()

    @pytest.mark.asyncio
    async def test_request_handles_500(self, client, mock_response):
        await client.connect()

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response({"error": "Internal error"}, 500)

            with pytest.raises(WorldWeaverError) as exc:
                await client._request("GET", "/error")

            assert exc.value.status_code == 500

        await client.close()

    @pytest.mark.asyncio
    async def test_request_handles_connect_error(self, client):
        await client.connect()

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(ConnectionError) as exc:
                await client._request("GET", "/health")

            assert "Failed to connect" in str(exc.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_request_handles_timeout(self, client):
        await client.connect()

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(WorldWeaverError) as exc:
                await client._request("GET", "/slow")

            assert "timed out" in str(exc.value)

        await client.close()

    @pytest.mark.asyncio
    async def test_health_returns_status(self, client, mock_response):
        await client.connect()

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response({
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00",
                "version": "1.0.0",
            })

            result = await client.health()
            assert result.status == "healthy"

        await client.close()

    @pytest.mark.asyncio
    async def test_create_episode(self, client, mock_response):
        await client.connect()

        ep_id = str(uuid4())
        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response({
                "id": ep_id,
                "session_id": "test_session",
                "content": "Test content",
                "timestamp": "2024-01-01T00:00:00",
                "outcome": "success",
                "emotional_valence": 0.8,
                "context": {"project": "test"},
                "access_count": 0,
                "stability": 0.5,
            })

            result = await client.create_episode(
                content="Test content",
                project="test",
                outcome="success",
                emotional_valence=0.8,
            )

            assert result.content == "Test content"
            assert result.outcome == "success"
            assert result.context.project == "test"

        await client.close()

    @pytest.mark.asyncio
    async def test_recall_episodes(self, client, mock_response):
        await client.connect()

        ep_id = str(uuid4())
        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response({
                "query": "test query",
                "episodes": [{
                    "id": ep_id,
                    "session_id": "sess",
                    "content": "Matched content",
                    "timestamp": "2024-01-01T00:00:00",
                    "outcome": "neutral",
                    "emotional_valence": 0.5,
                    "context": {},
                    "access_count": 1,
                    "stability": 0.6,
                }],
                "scores": [0.9],
            })

            result = await client.recall_episodes("test query", limit=5)

            assert result.query == "test query"
            assert len(result.episodes) == 1
            assert result.scores[0] == 0.9

        await client.close()

    @pytest.mark.asyncio
    async def test_create_entity(self, client, mock_response):
        await client.connect()

        entity_id = str(uuid4())
        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response({
                "id": entity_id,
                "name": "Python",
                "entity_type": "language",
                "summary": "A programming language",
                "stability": 0.5,
                "access_count": 0,
                "created_at": "2024-01-01T00:00:00",
            })

            result = await client.create_entity(
                name="Python",
                entity_type="language",
                summary="A programming language",
            )

            assert result.name == "Python"
            assert result.entity_type == "language"

        await client.close()

    @pytest.mark.asyncio
    async def test_spread_activation(self, client, mock_response):
        await client.connect()

        entity_id = str(uuid4())
        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response({
                "entities": [{
                    "id": str(uuid4()),
                    "name": "Related",
                    "entity_type": "concept",
                    "summary": "Related entity",
                    "stability": 0.5,
                    "access_count": 0,
                    "created_at": "2024-01-01T00:00:00",
                }],
                "activations": [0.8, 0.5],
                "paths": [["a", "b"]],
            })

            result = await client.spread_activation(UUID(entity_id), depth=2)

            assert len(result.entities) == 1
            assert result.activations == [0.8, 0.5]

        await client.close()

    @pytest.mark.asyncio
    async def test_create_skill(self, client, mock_response):
        await client.connect()

        skill_id = str(uuid4())
        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response({
                "id": skill_id,
                "name": "run_tests",
                "domain": "development",
                "trigger_pattern": None,
                "steps": [{"order": 1, "action": "pytest"}],
                "script": None,
                "success_rate": 1.0,
                "execution_count": 0,
                "last_executed": None,
                "version": 1,
                "deprecated": False,
                "created_at": "2024-01-01T00:00:00",
            })

            result = await client.create_skill(
                name="run_tests",
                domain="development",
                task="Run test suite",
                steps=[{"order": 1, "action": "pytest"}],
            )

            assert result.name == "run_tests"
            assert len(result.steps) == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_how_to(self, client, mock_response):
        await client.connect()

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response({
                "skill": {
                    "id": str(uuid4()),
                    "name": "test_skill",
                    "domain": "dev",
                    "trigger_pattern": None,
                    "steps": [{"order": 1, "action": "do it"}],
                    "script": None,
                    "success_rate": 0.9,
                    "execution_count": 10,
                    "last_executed": None,
                    "version": 1,
                    "deprecated": False,
                    "created_at": "2024-01-01T00:00:00",
                },
                "steps": ["Step 1", "Step 2"],
                "confidence": 0.85,
            })

            skill, steps, confidence = await client.how_to("run tests")

            assert skill is not None
            assert skill.name == "test_skill"
            assert steps == ["Step 1", "Step 2"]
            assert confidence == 0.85

        await client.close()


class TestSyncWorldWeaverClient:
    """Tests for synchronous WorldWeaverClient."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock HTTP response."""
        def _make_response(data: dict, status_code: int = 200):
            response = MagicMock()
            response.status_code = status_code
            response.content = json.dumps(data).encode() if data else b""
            response.json.return_value = data
            response.headers = {}
            return response
        return _make_response

    @pytest.fixture
    def client(self):
        return WorldWeaverClient(
            base_url="http://localhost:8765",
            session_id="test_session",
        )

    def test_client_creation(self, client):
        assert client.base_url == "http://localhost:8765"
        assert client.session_id == "test_session"

    def test_connect_creates_client(self, client):
        client.connect()
        assert client._client is not None
        client.close()

    def test_context_manager(self):
        with WorldWeaverClient() as client:
            assert client._client is not None
        assert client._client is None

    def test_get_client_raises_if_not_connected(self, client):
        with pytest.raises(WorldWeaverError) as exc:
            client._get_client()
        assert "not connected" in str(exc.value)

    def test_request_handles_404(self, client, mock_response):
        client.connect()

        with patch.object(client._client, "request") as mock:
            mock.return_value = mock_response({"error": "Not found"}, 404)

            with pytest.raises(NotFoundError):
                client._request("GET", "/missing")

        client.close()

    def test_request_handles_429(self, client, mock_response):
        client.connect()

        with patch.object(client._client, "request") as mock:
            response = mock_response(None, 429)
            response.headers = {"Retry-After": "60"}
            mock.return_value = response

            with pytest.raises(RateLimitError) as exc:
                client._request("GET", "/rate_limited")

            assert exc.value.retry_after == 60

        client.close()

    def test_health(self, client, mock_response):
        client.connect()

        with patch.object(client._client, "request") as mock:
            mock.return_value = mock_response({
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00",
                "version": "1.0.0",
            })

            result = client.health()
            assert result.status == "healthy"

        client.close()

    def test_create_episode(self, client, mock_response):
        client.connect()

        ep_id = str(uuid4())
        with patch.object(client._client, "request") as mock:
            mock.return_value = mock_response({
                "id": ep_id,
                "session_id": "test",
                "content": "Test",
                "timestamp": "2024-01-01T00:00:00",
                "outcome": "success",
                "emotional_valence": 0.7,
                "context": {},
                "access_count": 0,
                "stability": 0.5,
            })

            result = client.create_episode("Test", outcome="success")
            assert result.outcome == "success"

        client.close()

    def test_recall_episodes(self, client, mock_response):
        client.connect()

        with patch.object(client._client, "request") as mock:
            mock.return_value = mock_response({
                "query": "test",
                "episodes": [],
                "scores": [],
            })

            result = client.recall_episodes("test")
            assert result.query == "test"
            assert result.episodes == []

        client.close()

    def test_create_entity(self, client, mock_response):
        client.connect()

        with patch.object(client._client, "request") as mock:
            mock.return_value = mock_response({
                "id": str(uuid4()),
                "name": "Test",
                "entity_type": "concept",
                "summary": "A test entity",
                "stability": 0.5,
                "access_count": 0,
                "created_at": "2024-01-01T00:00:00",
            })

            result = client.create_entity("Test", "concept", "A test entity")
            assert result.name == "Test"

        client.close()

    def test_recall_entities(self, client, mock_response):
        client.connect()

        with patch.object(client._client, "request") as mock:
            mock.return_value = mock_response({"entities": []})

            result = client.recall_entities("test")
            assert result == []

        client.close()

    def test_create_skill(self, client, mock_response):
        client.connect()

        with patch.object(client._client, "request") as mock:
            mock.return_value = mock_response({
                "id": str(uuid4()),
                "name": "test_skill",
                "domain": "dev",
                "trigger_pattern": None,
                "steps": [],
                "script": None,
                "success_rate": 1.0,
                "execution_count": 0,
                "last_executed": None,
                "version": 1,
                "deprecated": False,
                "created_at": "2024-01-01T00:00:00",
            })

            result = client.create_skill("test_skill", "dev", "task")
            assert result.name == "test_skill"

        client.close()

    def test_how_to_no_skill(self, client, mock_response):
        client.connect()

        with patch.object(client._client, "request") as mock:
            mock.return_value = mock_response({
                "skill": None,
                "steps": [],
                "confidence": 0.0,
            })

            skill, steps, confidence = client.how_to("unknown task")
            assert skill is None
            assert steps == []
            assert confidence == 0.0

        client.close()


class TestClientSessionHeaders:
    """Tests for session ID header handling."""

    def test_async_client_sets_session_header(self):
        client = AsyncWorldWeaverClient(session_id="my_session")
        # Headers are set on connect, so we can't check until then
        assert client.session_id == "my_session"

    def test_sync_client_sets_session_header(self):
        client = WorldWeaverClient(session_id="sync_session")
        assert client.session_id == "sync_session"

    @pytest.mark.asyncio
    async def test_async_client_no_session(self):
        client = AsyncWorldWeaverClient()
        await client.connect()
        # Should work without session ID
        assert "X-Session-ID" not in client._client.headers
        await client.close()

    def test_sync_client_no_session(self):
        client = WorldWeaverClient()
        client.connect()
        assert "X-Session-ID" not in client._client.headers
        client.close()


class TestClientTimeout:
    """Tests for timeout configuration."""

    def test_async_client_default_timeout(self):
        client = AsyncWorldWeaverClient()
        assert client.timeout == 30.0

    def test_async_client_custom_timeout(self):
        client = AsyncWorldWeaverClient(timeout=60.0)
        assert client.timeout == 60.0

    def test_sync_client_default_timeout(self):
        client = WorldWeaverClient()
        assert client.timeout == 30.0

    def test_sync_client_custom_timeout(self):
        client = WorldWeaverClient(timeout=120.0)
        assert client.timeout == 120.0
