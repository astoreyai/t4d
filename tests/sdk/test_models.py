"""Tests for T4DM SDK models."""

import pytest
from datetime import datetime
from uuid import uuid4

from t4dm.sdk.models import (
    Episode,
    Entity,
    Skill,
    Step,
    Relationship,
    RecallResult,
    ActivationResult,
    HealthStatus,
    MemoryStats,
    EpisodeContext,
)


class TestEpisodeContext:
    """Tests for EpisodeContext model."""

    def test_default_values(self):
        """EpisodeContext has correct defaults."""
        ctx = EpisodeContext()
        assert ctx.project is None
        assert ctx.file is None
        assert ctx.tool is None

    def test_with_values(self):
        """EpisodeContext accepts values."""
        ctx = EpisodeContext(project="myproj", file="main.py", tool="edit")
        assert ctx.project == "myproj"
        assert ctx.file == "main.py"
        assert ctx.tool == "edit"


class TestEpisode:
    """Tests for Episode model."""

    def test_required_fields(self):
        """Episode requires all required fields."""
        ep = Episode(
            id=uuid4(),
            session_id="test-session",
            content="Test content",
            timestamp=datetime.utcnow(),
            outcome="success",
            emotional_valence=0.8,
            context=EpisodeContext(),
            access_count=0,
            stability=0.5,
        )
        assert ep.content == "Test content"
        assert ep.outcome == "success"

    def test_optional_retrievability(self):
        """Episode retrievability is optional."""
        ep = Episode(
            id=uuid4(),
            session_id="test",
            content="Test",
            timestamp=datetime.utcnow(),
            outcome="neutral",
            emotional_valence=0.5,
            context=EpisodeContext(),
            access_count=0,
            stability=0.5,
        )
        assert ep.retrievability is None

    def test_with_retrievability(self):
        """Episode can have retrievability."""
        ep = Episode(
            id=uuid4(),
            session_id="test",
            content="Test",
            timestamp=datetime.utcnow(),
            outcome="neutral",
            emotional_valence=0.5,
            context=EpisodeContext(),
            access_count=5,
            stability=0.8,
            retrievability=0.9,
        )
        assert ep.retrievability == 0.9


class TestEntity:
    """Tests for Entity model."""

    def test_required_fields(self):
        """Entity requires all required fields."""
        entity = Entity(
            id=uuid4(),
            name="Python",
            entity_type="technology",
            summary="Programming language",
            stability=0.5,
            access_count=10,
            created_at=datetime.utcnow(),
        )
        assert entity.name == "Python"
        assert entity.entity_type == "technology"

    def test_optional_fields(self):
        """Entity optional fields default to None."""
        entity = Entity(
            id=uuid4(),
            name="Test",
            entity_type="concept",
            summary="A concept",
            stability=0.5,
            access_count=0,
            created_at=datetime.utcnow(),
        )
        assert entity.details is None
        assert entity.source is None


class TestRelationship:
    """Tests for Relationship model."""

    def test_relationship_fields(self):
        """Relationship has correct fields."""
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type="uses",
            weight=0.8,
            co_access_count=5,
        )
        assert rel.relation_type == "uses"
        assert rel.weight == 0.8


class TestStep:
    """Tests for Step model."""

    def test_step_required_fields(self):
        """Step requires order and action."""
        step = Step(order=1, action="click button")
        assert step.order == 1
        assert step.action == "click button"

    def test_step_defaults(self):
        """Step has correct defaults."""
        step = Step(order=1, action="test")
        assert step.tool is None
        assert step.parameters == {}
        assert step.expected_outcome is None

    def test_step_with_all_fields(self):
        """Step with all fields."""
        step = Step(
            order=2,
            action="run command",
            tool="bash",
            parameters={"cmd": "ls"},
            expected_outcome="file list",
        )
        assert step.tool == "bash"
        assert step.parameters["cmd"] == "ls"


class TestSkill:
    """Tests for Skill model."""

    def test_skill_required_fields(self):
        """Skill requires all required fields."""
        skill = Skill(
            id=uuid4(),
            name="git-commit",
            domain="git",
            steps=[Step(order=1, action="stage changes")],
            success_rate=0.95,
            execution_count=100,
            version=1,
            deprecated=False,
            created_at=datetime.utcnow(),
        )
        assert skill.name == "git-commit"
        assert skill.success_rate == 0.95

    def test_skill_optional_fields(self):
        """Skill optional fields."""
        skill = Skill(
            id=uuid4(),
            name="test",
            domain="testing",
            steps=[],
            success_rate=1.0,
            execution_count=0,
            version=1,
            deprecated=False,
            created_at=datetime.utcnow(),
        )
        assert skill.trigger_pattern is None
        assert skill.script is None
        assert skill.last_executed is None


class TestRecallResult:
    """Tests for RecallResult model."""

    def test_recall_result(self):
        """RecallResult contains query and results."""
        result = RecallResult(
            query="python",
            episodes=[],
            scores=[],
        )
        assert result.query == "python"
        assert result.episodes == []


class TestActivationResult:
    """Tests for ActivationResult model."""

    def test_activation_result(self):
        """ActivationResult contains entities and activations."""
        result = ActivationResult(
            entities=[],
            activations=[],
            paths=[],
        )
        assert result.entities == []
        assert result.activations == []


class TestHealthStatus:
    """Tests for HealthStatus model."""

    def test_health_status(self):
        """HealthStatus has required fields."""
        status = HealthStatus(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
        )
        assert status.status == "healthy"
        assert status.session_id is None


class TestMemoryStats:
    """Tests for MemoryStats model."""

    def test_memory_stats(self):
        """MemoryStats has required fields."""
        stats = MemoryStats(
            session_id="test-session",
            episodic={"count": 100, "avg_stability": 0.7},
            semantic={"count": 50},
            procedural={"count": 10},
        )
        assert stats.session_id == "test-session"
        assert stats.episodic["count"] == 100
