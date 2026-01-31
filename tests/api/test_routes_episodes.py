"""
Tests for episode API routes.

Tests request/response models and endpoint behavior.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from pydantic import ValidationError

from t4dm.api.routes.episodes import (
    EpisodeCreate,
    EpisodeResponse,
    EpisodeList,
    EpisodeUpdate,
    RecallRequest,
    RecallResponse,
)
from t4dm.core.types import Outcome, EpisodeContext


# =============================================================================
# Test EpisodeCreate Model
# =============================================================================


class TestEpisodeCreate:
    """Tests for EpisodeCreate request model."""

    def test_minimal_create(self):
        """Test minimal episode creation."""
        req = EpisodeCreate(content="Test content")
        assert req.content == "Test content"
        assert req.project is None
        assert req.file is None
        assert req.tool is None
        assert req.outcome == Outcome.NEUTRAL
        assert req.emotional_valence == 0.5
        assert req.timestamp is None

    def test_full_create(self):
        """Test episode creation with all fields."""
        now = datetime.now()
        req = EpisodeCreate(
            content="Full content",
            project="world-weaver",
            file="src/test.py",
            tool="pytest",
            outcome=Outcome.SUCCESS,
            emotional_valence=0.9,
            timestamp=now,
        )
        assert req.content == "Full content"
        assert req.project == "world-weaver"
        assert req.file == "src/test.py"
        assert req.tool == "pytest"
        assert req.outcome == Outcome.SUCCESS
        assert req.emotional_valence == 0.9
        assert req.timestamp == now

    def test_content_min_length(self):
        """Test content must be non-empty."""
        with pytest.raises(ValidationError):
            EpisodeCreate(content="")

    def test_content_max_length(self):
        """Test content max length (50KB)."""
        # Should work with content just under limit
        long_content = "a" * 50000
        req = EpisodeCreate(content=long_content)
        assert len(req.content) == 50000

        # Should fail with content over limit
        with pytest.raises(ValidationError):
            EpisodeCreate(content="a" * 50001)

    def test_valence_bounds(self):
        """Test emotional valence must be in [0, 1]."""
        # Valid bounds
        EpisodeCreate(content="test", emotional_valence=0.0)
        EpisodeCreate(content="test", emotional_valence=1.0)

        # Invalid bounds
        with pytest.raises(ValidationError):
            EpisodeCreate(content="test", emotional_valence=-0.1)
        with pytest.raises(ValidationError):
            EpisodeCreate(content="test", emotional_valence=1.1)

    def test_project_max_length(self):
        """Test project field max length."""
        # Should work with project at limit
        EpisodeCreate(content="test", project="a" * 500)

        # Should fail with project over limit
        with pytest.raises(ValidationError):
            EpisodeCreate(content="test", project="a" * 501)

    def test_file_max_length(self):
        """Test file field max length."""
        EpisodeCreate(content="test", file="a" * 1000)
        with pytest.raises(ValidationError):
            EpisodeCreate(content="test", file="a" * 1001)

    def test_tool_max_length(self):
        """Test tool field max length."""
        EpisodeCreate(content="test", tool="a" * 200)
        with pytest.raises(ValidationError):
            EpisodeCreate(content="test", tool="a" * 201)


# =============================================================================
# Test EpisodeResponse Model
# =============================================================================


class TestEpisodeResponse:
    """Tests for EpisodeResponse model."""

    def test_full_response(self):
        """Test full episode response."""
        ep_id = uuid4()
        now = datetime.now()
        resp = EpisodeResponse(
            id=ep_id,
            session_id="session-123",
            content="Test content",
            timestamp=now,
            outcome=Outcome.SUCCESS,
            emotional_valence=0.8,
            context=EpisodeContext(project="ww"),
            access_count=5,
            stability=2.0,
            retrievability=0.95,
        )
        assert resp.id == ep_id
        assert resp.session_id == "session-123"
        assert resp.outcome == Outcome.SUCCESS
        assert resp.access_count == 5
        assert resp.stability == 2.0
        assert resp.retrievability == 0.95

    def test_optional_retrievability(self):
        """Test retrievability is optional."""
        resp = EpisodeResponse(
            id=uuid4(),
            session_id="test",
            content="Test",
            timestamp=datetime.now(),
            outcome=Outcome.NEUTRAL,
            emotional_valence=0.5,
            context=EpisodeContext(),
            access_count=1,
            stability=1.0,
        )
        assert resp.retrievability is None


# =============================================================================
# Test EpisodeList Model
# =============================================================================


class TestEpisodeList:
    """Tests for EpisodeList model."""

    def test_empty_list(self):
        """Test empty episode list."""
        resp = EpisodeList(
            episodes=[],
            total=0,
            page=1,
            page_size=10,
        )
        assert resp.episodes == []
        assert resp.total == 0

    def test_with_episodes(self):
        """Test list with episodes."""
        episodes = [
            EpisodeResponse(
                id=uuid4(),
                session_id="test",
                content="Episode 1",
                timestamp=datetime.now(),
                outcome=Outcome.SUCCESS,
                emotional_valence=0.5,
                context=EpisodeContext(),
                access_count=1,
                stability=1.0,
            ),
            EpisodeResponse(
                id=uuid4(),
                session_id="test",
                content="Episode 2",
                timestamp=datetime.now(),
                outcome=Outcome.FAILURE,
                emotional_valence=0.3,
                context=EpisodeContext(),
                access_count=2,
                stability=1.5,
            ),
        ]
        resp = EpisodeList(
            episodes=episodes,
            total=10,
            page=1,
            page_size=2,
        )
        assert len(resp.episodes) == 2
        assert resp.total == 10
        assert resp.page == 1
        assert resp.page_size == 2


# =============================================================================
# Test EpisodeUpdate Model
# =============================================================================


class TestEpisodeUpdate:
    """Tests for EpisodeUpdate model."""

    def test_empty_update(self):
        """Test update with no fields."""
        update = EpisodeUpdate()
        assert update.content is None
        assert update.emotional_valence is None
        assert update.outcome is None
        assert update.project is None

    def test_partial_update(self):
        """Test partial update."""
        update = EpisodeUpdate(
            content="Updated content",
            emotional_valence=0.9,
        )
        assert update.content == "Updated content"
        assert update.emotional_valence == 0.9
        assert update.outcome is None

    def test_full_update(self):
        """Test update with all fields."""
        update = EpisodeUpdate(
            content="Full update",
            emotional_valence=0.7,
            outcome=Outcome.SUCCESS,
            project="new-project",
            file="new-file.py",
            tool="new-tool",
        )
        assert update.content == "Full update"
        assert update.emotional_valence == 0.7
        assert update.outcome == Outcome.SUCCESS
        assert update.project == "new-project"
        assert update.file == "new-file.py"
        assert update.tool == "new-tool"

    def test_content_max_length(self):
        """Test content max length validation."""
        with pytest.raises(ValidationError):
            EpisodeUpdate(content="a" * 50001)


# =============================================================================
# Test RecallRequest Model
# =============================================================================


class TestRecallRequest:
    """Tests for RecallRequest model."""

    def test_minimal_request(self):
        """Test minimal recall request."""
        req = RecallRequest(query="test query")
        assert req.query == "test query"
        assert req.limit == 10
        assert req.min_similarity == 0.5
        assert req.project is None
        assert req.outcome is None

    def test_full_request(self):
        """Test full recall request."""
        req = RecallRequest(
            query="complex query",
            limit=50,
            min_similarity=0.8,
            project="world-weaver",
            outcome=Outcome.SUCCESS,
        )
        assert req.query == "complex query"
        assert req.limit == 50
        assert req.min_similarity == 0.8
        assert req.project == "world-weaver"
        assert req.outcome == Outcome.SUCCESS

    def test_query_min_length(self):
        """Test query must be non-empty."""
        with pytest.raises(ValidationError):
            RecallRequest(query="")

    def test_query_max_length(self):
        """Test query max length (10KB)."""
        # Should work with query at limit
        RecallRequest(query="a" * 10000)

        # Should fail with query over limit
        with pytest.raises(ValidationError):
            RecallRequest(query="a" * 10001)

    def test_limit_bounds(self):
        """Test limit must be in [1, 100]."""
        RecallRequest(query="test", limit=1)
        RecallRequest(query="test", limit=100)

        with pytest.raises(ValidationError):
            RecallRequest(query="test", limit=0)
        with pytest.raises(ValidationError):
            RecallRequest(query="test", limit=101)

    def test_min_similarity_bounds(self):
        """Test min_similarity must be in [0, 1]."""
        RecallRequest(query="test", min_similarity=0.0)
        RecallRequest(query="test", min_similarity=1.0)

        with pytest.raises(ValidationError):
            RecallRequest(query="test", min_similarity=-0.1)
        with pytest.raises(ValidationError):
            RecallRequest(query="test", min_similarity=1.1)


# =============================================================================
# Test RecallResponse Model
# =============================================================================


class TestRecallResponse:
    """Tests for RecallResponse model."""

    def test_empty_results(self):
        """Test empty recall results."""
        resp = RecallResponse(
            query="test query",
            episodes=[],
            scores=[],
        )
        assert resp.query == "test query"
        assert resp.episodes == []
        assert resp.scores == []

    def test_with_results(self):
        """Test recall results with episodes."""
        episodes = [
            EpisodeResponse(
                id=uuid4(),
                session_id="test",
                content="Match 1",
                timestamp=datetime.now(),
                outcome=Outcome.SUCCESS,
                emotional_valence=0.8,
                context=EpisodeContext(),
                access_count=3,
                stability=2.0,
                retrievability=0.9,
            ),
            EpisodeResponse(
                id=uuid4(),
                session_id="test",
                content="Match 2",
                timestamp=datetime.now(),
                outcome=Outcome.NEUTRAL,
                emotional_valence=0.5,
                context=EpisodeContext(),
                access_count=1,
                stability=1.0,
                retrievability=0.7,
            ),
        ]
        resp = RecallResponse(
            query="search term",
            episodes=episodes,
            scores=[0.95, 0.82],
        )
        assert resp.query == "search term"
        assert len(resp.episodes) == 2
        assert len(resp.scores) == 2
        assert resp.scores[0] == 0.95
