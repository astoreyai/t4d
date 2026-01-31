"""
Tests for core data types.

Tests Episode, Entity, Procedure, Relationship, and related types.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from t4dm.core.types import (
    Outcome,
    EntityType,
    RelationType,
    Domain,
    ConsolidationType,
    EpisodeContext,
    ProcedureStep,
    Episode,
    Entity,
    Relationship,
    Procedure,
    ConsolidationEvent,
    EpisodeQuery,
    EntityQuery,
    ProcedureQuery,
    ScoredResult,
)


# =============================================================================
# Test Enums
# =============================================================================


class TestOutcome:
    """Tests for Outcome enum."""

    def test_outcome_values(self):
        """Test outcome enum values."""
        assert Outcome.SUCCESS.value == "success"
        assert Outcome.FAILURE.value == "failure"
        assert Outcome.PARTIAL.value == "partial"
        assert Outcome.NEUTRAL.value == "neutral"

    def test_outcome_from_string(self):
        """Test creating outcome from string."""
        assert Outcome("success") == Outcome.SUCCESS
        assert Outcome("failure") == Outcome.FAILURE


class TestEntityType:
    """Tests for EntityType enum."""

    def test_entity_type_values(self):
        """Test entity type values."""
        assert EntityType.CONCEPT.value == "CONCEPT"
        assert EntityType.PERSON.value == "PERSON"
        assert EntityType.PROJECT.value == "PROJECT"
        assert EntityType.TOOL.value == "TOOL"
        assert EntityType.TECHNIQUE.value == "TECHNIQUE"
        assert EntityType.FACT.value == "FACT"


class TestRelationType:
    """Tests for RelationType enum."""

    def test_relation_type_values(self):
        """Test relation type values."""
        assert RelationType.USES.value == "USES"
        assert RelationType.PRODUCES.value == "PRODUCES"
        assert RelationType.REQUIRES.value == "REQUIRES"
        assert RelationType.CAUSES.value == "CAUSES"


class TestDomain:
    """Tests for Domain enum."""

    def test_domain_values(self):
        """Test domain values."""
        assert Domain.CODING.value == "coding"
        assert Domain.RESEARCH.value == "research"
        assert Domain.TRADING.value == "trading"


class TestConsolidationType:
    """Tests for ConsolidationType enum."""

    def test_consolidation_type_values(self):
        """Test consolidation type values."""
        assert ConsolidationType.EPISODIC_TO_SEMANTIC.value == "episodic_to_semantic"
        assert ConsolidationType.SKILL_MERGE.value == "skill_merge"
        assert ConsolidationType.PATTERN_EXTRACT.value == "pattern_extract"


# =============================================================================
# Test Context Models
# =============================================================================


class TestEpisodeContext:
    """Tests for EpisodeContext model."""

    def test_default_context(self):
        """Test default context creation."""
        ctx = EpisodeContext()
        assert ctx.project is None
        assert ctx.file is None
        assert ctx.tool is None
        assert ctx.cwd is None
        assert ctx.git_branch is None

    def test_context_with_values(self):
        """Test context with values."""
        ctx = EpisodeContext(
            project="world-weaver",
            file="src/ww/core/types.py",
            tool="pytest",
            cwd="/home/user/ww",
            git_branch="main",
        )
        assert ctx.project == "world-weaver"
        assert ctx.file == "src/ww/core/types.py"
        assert ctx.tool == "pytest"

    def test_context_extra_allowed(self):
        """Test extra fields are allowed."""
        ctx = EpisodeContext(
            project="ww",
            custom_field="custom_value",
        )
        assert ctx.project == "ww"
        assert ctx.custom_field == "custom_value"


class TestProcedureStep:
    """Tests for ProcedureStep model."""

    def test_valid_step(self):
        """Test creating valid step."""
        step = ProcedureStep(
            order=1,
            action="Run pytest",
            tool="pytest",
            parameters={"verbose": True},
            expected_outcome="All tests pass",
        )
        assert step.order == 1
        assert step.action == "Run pytest"
        assert step.tool == "pytest"
        assert step.parameters["verbose"] is True

    def test_minimal_step(self):
        """Test minimal step creation."""
        step = ProcedureStep(order=1, action="Do something")
        assert step.order == 1
        assert step.action == "Do something"
        assert step.tool is None
        assert step.parameters == {}

    def test_order_validation(self):
        """Test order must be >= 1."""
        with pytest.raises(ValueError):
            ProcedureStep(order=0, action="Invalid")

    def test_action_min_length(self):
        """Test action must be non-empty."""
        with pytest.raises(ValueError):
            ProcedureStep(order=1, action="")


# =============================================================================
# Test Episode
# =============================================================================


class TestEpisode:
    """Tests for Episode model."""

    def test_default_episode(self):
        """Test default episode creation."""
        ep = Episode(session_id="test-session", content="Test content")
        assert ep.session_id == "test-session"
        assert ep.content == "Test content"
        assert isinstance(ep.id, UUID)
        assert ep.embedding is None
        assert ep.outcome == Outcome.NEUTRAL
        assert ep.emotional_valence == 0.5
        assert ep.access_count == 1
        assert ep.stability == 1.0

    def test_episode_with_all_fields(self):
        """Test episode with all fields."""
        now = datetime.now()
        ep = Episode(
            session_id="session-123",
            content="Full content",
            embedding=[0.1] * 1024,
            timestamp=now,
            context=EpisodeContext(project="ww"),
            outcome=Outcome.SUCCESS,
            emotional_valence=0.8,
            stability=2.5,
        )
        assert ep.content == "Full content"
        assert len(ep.embedding) == 1024
        assert ep.context.project == "ww"
        assert ep.outcome == Outcome.SUCCESS
        assert ep.emotional_valence == 0.8

    def test_retrievability_fresh(self):
        """Test retrievability for fresh episode."""
        ep = Episode(session_id="test", content="Fresh")
        # Just created, should be near 1.0
        r = ep.retrievability()
        assert r > 0.99

    def test_retrievability_decay(self):
        """Test retrievability decays over time."""
        ep = Episode(
            session_id="test",
            content="Old memory",
            last_accessed=datetime.now() - timedelta(days=7),
            stability=1.0,
        )
        r = ep.retrievability()
        # 7 days with stability=1.0: R = (1 + 0.9 * 7)^(-0.5) ≈ 0.37
        assert r < 0.5
        assert r > 0.3

    def test_retrievability_with_high_stability(self):
        """Test retrievability with high stability decays slower."""
        ep = Episode(
            session_id="test",
            content="Stable memory",
            last_accessed=datetime.now() - timedelta(days=7),
            stability=7.0,  # 7x more stable
        )
        r = ep.retrievability()
        # 7 days with stability=7.0: R = (1 + 0.9 * 1)^(-0.5) ≈ 0.73
        assert r > 0.7

    def test_valence_validation(self):
        """Test emotional valence must be in [0, 1]."""
        with pytest.raises(ValueError):
            Episode(session_id="test", content="x", emotional_valence=1.5)
        with pytest.raises(ValueError):
            Episode(session_id="test", content="x", emotional_valence=-0.1)


# =============================================================================
# Test Entity
# =============================================================================


class TestEntity:
    """Tests for Entity model."""

    def test_default_entity(self):
        """Test default entity creation."""
        entity = Entity(
            name="Python",
            entity_type=EntityType.TOOL,
            summary="Programming language",
        )
        assert entity.name == "Python"
        assert entity.entity_type == EntityType.TOOL
        assert entity.summary == "Programming language"
        assert isinstance(entity.id, UUID)
        assert entity.embedding is None
        assert entity.stability == 1.0

    def test_entity_is_valid_current(self):
        """Test entity is_valid for current entity."""
        entity = Entity(
            name="Test",
            entity_type=EntityType.CONCEPT,
            summary="Test entity",
        )
        assert entity.is_valid() is True

    def test_entity_is_valid_expired(self):
        """Test entity is_valid for expired entity."""
        past = datetime.now() - timedelta(days=10)
        entity = Entity(
            name="Test",
            entity_type=EntityType.CONCEPT,
            summary="Test entity",
            valid_from=past - timedelta(days=5),
            valid_to=past,
        )
        assert entity.is_valid() is False

    def test_entity_is_valid_future(self):
        """Test entity is_valid for future entity."""
        future = datetime.now() + timedelta(days=10)
        entity = Entity(
            name="Test",
            entity_type=EntityType.CONCEPT,
            summary="Test entity",
            valid_from=future,
        )
        assert entity.is_valid() is False

    def test_entity_is_valid_at_specific_time(self):
        """Test entity is_valid at specific time."""
        base = datetime(2024, 1, 1)
        entity = Entity(
            name="Test",
            entity_type=EntityType.CONCEPT,
            summary="Test",
            valid_from=datetime(2024, 1, 1),
            valid_to=datetime(2024, 12, 31),
        )
        assert entity.is_valid(at_time=datetime(2024, 6, 1)) is True
        assert entity.is_valid(at_time=datetime(2023, 12, 31)) is False
        assert entity.is_valid(at_time=datetime(2025, 1, 1)) is False


# =============================================================================
# Test Relationship
# =============================================================================


class TestRelationship:
    """Tests for Relationship model."""

    def test_default_relationship(self):
        """Test default relationship creation."""
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
        )
        assert rel.weight == 0.1
        assert rel.co_access_count == 1

    def test_strengthen(self):
        """Test Hebbian strengthening."""
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
            weight=0.5,
        )
        old_weight = rel.weight
        new_weight = rel.strengthen(learning_rate=0.1)

        # w' = 0.5 + 0.1 * (1 - 0.5) = 0.5 + 0.05 = 0.55
        assert new_weight == pytest.approx(0.55)
        assert rel.weight == pytest.approx(0.55)
        assert rel.co_access_count == 2

    def test_strengthen_bounded(self):
        """Test strengthening is bounded at 1.0."""
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
            weight=0.99,
        )
        # After strengthening: 0.99 + 0.1 * (1 - 0.99) = 0.99 + 0.001 = 0.991
        new_weight = rel.strengthen(learning_rate=0.1)
        assert new_weight <= 1.0

    def test_strengthen_multiple_times(self):
        """Test multiple strengthening calls."""
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
            weight=0.1,
        )
        for _ in range(10):
            rel.strengthen(learning_rate=0.1)

        assert rel.co_access_count == 11
        assert rel.weight > 0.6
        assert rel.weight <= 1.0


# =============================================================================
# Test Procedure
# =============================================================================


class TestProcedure:
    """Tests for Procedure model."""

    def test_default_procedure(self):
        """Test default procedure creation."""
        proc = Procedure(
            name="run-tests",
            domain=Domain.CODING,
        )
        assert proc.name == "run-tests"
        assert proc.domain == Domain.CODING
        assert proc.steps == []
        assert proc.success_rate == 1.0
        assert proc.execution_count == 1
        assert proc.version == 1
        assert proc.deprecated is False

    def test_update_success_rate_success(self):
        """Test updating success rate with success."""
        proc = Procedure(
            name="test",
            domain=Domain.CODING,
            success_rate=0.8,
            execution_count=10,
        )
        new_rate = proc.update_success_rate(success=True)
        # (0.8 * 10 + 1) / 11 = 9 / 11 ≈ 0.818
        assert new_rate == pytest.approx(9 / 11)
        assert proc.execution_count == 11

    def test_update_success_rate_failure(self):
        """Test updating success rate with failure."""
        proc = Procedure(
            name="test",
            domain=Domain.CODING,
            success_rate=0.8,
            execution_count=10,
        )
        new_rate = proc.update_success_rate(success=False)
        # (0.8 * 10) / 11 = 8 / 11 ≈ 0.727
        assert new_rate == pytest.approx(8 / 11)
        assert proc.execution_count == 11

    def test_should_deprecate_true(self):
        """Test should_deprecate returns True for failing procedure."""
        proc = Procedure(
            name="failing",
            domain=Domain.CODING,
            success_rate=0.2,
            execution_count=15,
        )
        assert proc.should_deprecate(min_executions=10, min_success=0.3) is True

    def test_should_deprecate_false_not_enough_executions(self):
        """Test should_deprecate returns False without enough executions."""
        proc = Procedure(
            name="new",
            domain=Domain.CODING,
            success_rate=0.2,
            execution_count=5,
        )
        assert proc.should_deprecate(min_executions=10, min_success=0.3) is False

    def test_should_deprecate_false_high_success(self):
        """Test should_deprecate returns False with high success rate."""
        proc = Procedure(
            name="good",
            domain=Domain.CODING,
            success_rate=0.9,
            execution_count=100,
        )
        assert proc.should_deprecate(min_executions=10, min_success=0.3) is False


# =============================================================================
# Test ConsolidationEvent
# =============================================================================


class TestConsolidationEvent:
    """Tests for ConsolidationEvent model."""

    def test_event_creation(self):
        """Test consolidation event creation."""
        source_ids = [uuid4(), uuid4()]
        target_id = uuid4()
        event = ConsolidationEvent(
            event_type=ConsolidationType.EPISODIC_TO_SEMANTIC,
            source_ids=source_ids,
            target_id=target_id,
            confidence=0.85,
            pattern_strength=3,
            metadata={"key": "value"},
        )
        assert event.event_type == ConsolidationType.EPISODIC_TO_SEMANTIC
        assert len(event.source_ids) == 2
        assert event.target_id == target_id
        assert event.confidence == 0.85
        assert event.pattern_strength == 3


# =============================================================================
# Test Query Types
# =============================================================================


class TestEpisodeQuery:
    """Tests for EpisodeQuery model."""

    def test_minimal_query(self):
        """Test minimal query."""
        q = EpisodeQuery(query="test query")
        assert q.query == "test query"
        assert q.limit == 10
        assert q.session_filter is None
        assert q.time_start is None

    def test_full_query(self):
        """Test full query with all fields."""
        now = datetime.now()
        q = EpisodeQuery(
            query="search term",
            limit=50,
            session_filter="session-123",
            time_start=now - timedelta(days=7),
            time_end=now,
        )
        assert q.query == "search term"
        assert q.limit == 50
        assert q.session_filter == "session-123"


class TestEntityQuery:
    """Tests for EntityQuery model."""

    def test_minimal_query(self):
        """Test minimal query."""
        q = EntityQuery(query="Python")
        assert q.query == "Python"
        assert q.context_entities == []
        assert q.limit == 10
        assert q.include_spreading is True

    def test_query_with_context(self):
        """Test query with context entities."""
        q = EntityQuery(
            query="test",
            context_entities=["Python", "testing"],
            include_spreading=False,
        )
        assert len(q.context_entities) == 2
        assert q.include_spreading is False


class TestProcedureQuery:
    """Tests for ProcedureQuery model."""

    def test_minimal_query(self):
        """Test minimal query."""
        q = ProcedureQuery(task="run tests")
        assert q.task == "run tests"
        assert q.domain is None
        assert q.limit == 5

    def test_query_with_domain(self):
        """Test query with domain filter."""
        q = ProcedureQuery(
            task="deploy",
            domain=Domain.DEVOPS,
            limit=10,
        )
        assert q.domain == Domain.DEVOPS
        assert q.limit == 10


# =============================================================================
# Test ScoredResult
# =============================================================================


class TestScoredResult:
    """Tests for ScoredResult generic model."""

    def test_scored_result_with_episode(self):
        """Test scored result with Episode."""
        ep = Episode(session_id="test", content="Test content")
        result = ScoredResult(item=ep, score=0.95)
        assert result.item == ep
        assert result.score == 0.95
        assert result.components == {}

    def test_scored_result_with_entity(self):
        """Test scored result with Entity."""
        entity = Entity(
            name="Python",
            entity_type=EntityType.TOOL,
            summary="Language",
        )
        result = ScoredResult(
            item=entity,
            score=0.85,
            components={"semantic": 0.8, "graph": 0.1},
        )
        assert result.item.name == "Python"
        assert result.score == 0.85
        assert result.components["semantic"] == 0.8

    def test_scored_result_with_procedure(self):
        """Test scored result with Procedure."""
        proc = Procedure(name="test", domain=Domain.CODING)
        result = ScoredResult(item=proc, score=0.75)
        assert result.item.name == "test"
        assert result.score == 0.75

    def test_score_validation(self):
        """Test score must be in [0, 1]."""
        with pytest.raises(ValueError):
            ScoredResult(item="test", score=1.5)
        with pytest.raises(ValueError):
            ScoredResult(item="test", score=-0.1)
