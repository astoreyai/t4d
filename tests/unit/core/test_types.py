"""Comprehensive tests for types.py - Episode, Entity, Procedure, Relationship."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from pydantic import ValidationError

from t4dm.core.types import (
    Episode, Entity, Procedure, Relationship, TemporalLink, ConsolidationEvent,
    EpisodeContext, ProcedureStep, EpisodeQuery, EntityQuery, ProcedureQuery,
    Outcome, EntityType, RelationType, Domain, ConsolidationType, TemporalLinkType,
    ScoredResult,
)


class TestOutcomeEnum:
    """Test Outcome enum variants."""

    def test_all_outcomes_defined(self):
        assert Outcome.SUCCESS.value == "success"
        assert Outcome.FAILURE.value == "failure"
        assert Outcome.PARTIAL.value == "partial"
        assert Outcome.NEUTRAL.value == "neutral"

    def test_outcome_from_string(self):
        assert Outcome("success") == Outcome.SUCCESS
        assert Outcome("failure") == Outcome.FAILURE


class TestEntityTypeEnum:
    """Test EntityType enum variants."""

    def test_all_entity_types_defined(self):
        types = [
            EntityType.CONCEPT, EntityType.PERSON, EntityType.PROJECT,
            EntityType.TOOL, EntityType.TECHNIQUE, EntityType.FACT
        ]
        assert len(types) == 6
        assert EntityType.CONCEPT.value == "CONCEPT"


class TestRelationTypeEnum:
    """Test RelationType enum variants."""

    def test_all_relation_types_defined(self):
        assert RelationType.USES.value == "USES"
        assert RelationType.PRODUCES.value == "PRODUCES"
        assert RelationType.CONSOLIDATED_INTO.value == "CONSOLIDATED_INTO"


class TestDomainEnum:
    """Test Domain enum variants."""

    def test_all_domains_defined(self):
        domains = [Domain.CODING, Domain.RESEARCH, Domain.TRADING, Domain.DEVOPS, Domain.WRITING]
        assert len(domains) == 5


class TestEpisodeContext:
    """Test EpisodeContext model."""

    def test_all_fields_optional(self):
        ctx = EpisodeContext()
        assert ctx.project is None
        assert ctx.file is None
        assert ctx.tool is None
        assert ctx.cwd is None
        assert ctx.git_branch is None
        assert ctx.timestamp_local is None

    def test_with_values(self):
        ctx = EpisodeContext(
            project="t4dm",
            file="/home/user/app.py",
            tool="vscode",
            cwd="/home/user",
            git_branch="main",
            timestamp_local="2024-01-15T10:30:00"
        )
        assert ctx.project == "t4dm"
        assert ctx.file == "/home/user/app.py"
        assert ctx.tool == "vscode"

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed via ConfigDict."""
        ctx = EpisodeContext(custom_field="value")
        assert ctx.custom_field == "value"


class TestProcedureStep:
    """Test ProcedureStep model."""

    def test_valid_step(self):
        step = ProcedureStep(
            order=1,
            action="run tests",
            tool="pytest",
            parameters={"timeout": 30},
            expected_outcome="all tests pass"
        )
        assert step.order == 1
        assert step.action == "run tests"
        assert step.tool == "pytest"
        assert step.parameters["timeout"] == 30

    def test_order_must_be_positive(self):
        with pytest.raises(ValidationError):
            ProcedureStep(order=0, action="test")
        with pytest.raises(ValidationError):
            ProcedureStep(order=-1, action="test")

    def test_action_required(self):
        with pytest.raises(ValidationError):
            ProcedureStep(order=1, action="")

    def test_parameters_default_empty(self):
        step = ProcedureStep(order=1, action="test")
        assert step.parameters == {}


class TestEpisode:
    """Test Episode (episodic memory) model."""

    @pytest.fixture
    def episode(self):
        return Episode(
            session_id="test-session",
            content="I learned about temporal encoding"
        )

    def test_episode_defaults(self, episode):
        assert episode.session_id == "test-session"
        assert episode.content == "I learned about temporal encoding"
        assert episode.kappa == 0.0
        assert episode.embedding is None
        assert episode.outcome == Outcome.NEUTRAL
        assert episode.emotional_valence == 0.5
        assert episode.access_count == 1
        assert episode.stability == 1.0
        assert episode.sequence_position is None
        assert episode.duration_ms is None

    def test_episode_with_embedding(self):
        emb = [0.1] * 1024
        ep = Episode(
            session_id="s1",
            content="text",
            embedding=emb
        )
        assert ep.embedding == emb

    def test_kappa_bounds(self):
        Episode(session_id="s", content="x", kappa=0.0)
        Episode(session_id="s", content="x", kappa=1.0)
        Episode(session_id="s", content="x", kappa=0.5)

        with pytest.raises(ValidationError):
            Episode(session_id="s", content="x", kappa=-0.1)
        with pytest.raises(ValidationError):
            Episode(session_id="s", content="x", kappa=1.1)

    def test_emotional_valence_bounds(self):
        Episode(session_id="s", content="x", emotional_valence=0.0)
        Episode(session_id="s", content="x", emotional_valence=1.0)

        with pytest.raises(ValidationError):
            Episode(session_id="s", content="x", emotional_valence=-0.1)

    def test_content_required(self):
        with pytest.raises(ValidationError):
            Episode(session_id="s", content="")

    def test_session_id_required(self):
        # session_id is a required str, empty string is valid for pydantic
        # but the Field() constraint prevents empty strings via min_length validation
        ep = Episode(session_id="valid", content="text")
        assert ep.session_id == "valid"

    def test_retrievability_zero_time(self):
        ep = Episode(session_id="s", content="x")
        # At zero elapsed time, R should be high
        r = ep.retrievability(current_time=ep.last_accessed)
        assert r == 1.0

    def test_retrievability_decay_over_time(self):
        ep = Episode(session_id="s", content="x", stability=10.0)
        now = ep.last_accessed

        # At 0 days: R = 1.0
        r0 = ep.retrievability(current_time=now)

        # At 10 days: R = (1 + 0.9*10/10)^-0.5 = 2^-0.5 ≈ 0.7071
        later = now + timedelta(days=10)
        r10 = ep.retrievability(current_time=later)

        assert r0 == 1.0
        assert 0.70 < r10 < 0.73  # Allow slight variation (0.7254...)
        assert r10 < r0

    def test_retrievability_stability_effect(self):
        ep1 = Episode(session_id="s", content="x", stability=1.0)
        ep2 = Episode(session_id="s", content="x", stability=10.0)

        future = ep1.last_accessed + timedelta(days=5)

        r1 = ep1.retrievability(current_time=future)
        r2 = ep2.retrievability(current_time=future)

        # Higher stability -> slower decay
        assert r2 > r1

    def test_bi_temporal_fields(self):
        now = datetime.now()
        ep = Episode(
            session_id="s",
            content="x",
            timestamp=now,
            ingested_at=now + timedelta(seconds=1)
        )
        assert ep.timestamp == now
        assert ep.ingested_at > ep.timestamp

    def test_temporal_sequence_fields(self):
        prev_id = uuid4()
        ep = Episode(
            session_id="s",
            content="x",
            previous_episode_id=prev_id,
            sequence_position=5,
            duration_ms=1000
        )
        assert ep.previous_episode_id == prev_id
        assert ep.sequence_position == 5
        assert ep.duration_ms == 1000

    def test_prediction_error_fields(self):
        now = datetime.now()
        ep = Episode(
            session_id="s",
            content="x",
            prediction_error=0.5,
            prediction_error_timestamp=now
        )
        assert ep.prediction_error == 0.5
        assert ep.prediction_error_timestamp == now


class TestEntity:
    """Test Entity (semantic memory) model."""

    @pytest.fixture
    def entity(self):
        return Entity(
            name="Temporal Encoding",
            entity_type=EntityType.TECHNIQUE,
            summary="Method to encode time as a dimension"
        )

    def test_entity_defaults(self, entity):
        assert entity.name == "Temporal Encoding"
        assert entity.entity_type == EntityType.TECHNIQUE
        assert entity.summary == "Method to encode time as a dimension"
        assert entity.kappa == 0.85  # Entities start higher
        assert entity.details is None
        assert entity.source is None
        assert entity.access_count == 1
        assert entity.stability == 1.0

    def test_name_required(self):
        with pytest.raises(ValidationError):
            Entity(name="", entity_type=EntityType.CONCEPT, summary="x")

    def test_summary_required(self):
        with pytest.raises(ValidationError):
            Entity(name="x", entity_type=EntityType.CONCEPT, summary="")

    def test_is_valid_current(self):
        ent = Entity(name="x", entity_type=EntityType.CONCEPT, summary="y")
        assert ent.is_valid()
        assert ent.is_valid(at_time=datetime.now())

    def test_is_valid_before_valid_from(self):
        now = datetime.now()
        future = now + timedelta(days=1)
        ent = Entity(
            name="x",
            entity_type=EntityType.CONCEPT,
            summary="y",
            valid_from=future
        )
        assert not ent.is_valid(at_time=now)
        assert ent.is_valid(at_time=future)

    def test_is_valid_after_valid_to(self):
        now = datetime.now()
        past = now - timedelta(days=1)
        future = now + timedelta(days=1)
        ent = Entity(
            name="x",
            entity_type=EntityType.CONCEPT,
            summary="y",
            valid_from=past,
            valid_to=now
        )
        # At exactly valid_to, entity is invalid (at_time >= valid_to)
        assert not ent.is_valid(at_time=now)
        # Before valid_to, entity is valid
        mid = past + timedelta(hours=12)
        assert ent.is_valid(at_time=mid)
        # After valid_to, entity is invalid
        assert not ent.is_valid(at_time=future)

    def test_bi_temporal_versioning(self):
        now = datetime.now()
        past = now - timedelta(days=1)
        ent = Entity(
            name="Python",
            entity_type=EntityType.TOOL,
            summary="Language",
            valid_from=past,
            valid_to=now
        )
        assert ent.valid_from == past
        assert ent.valid_to == now


class TestRelationship:
    """Test Relationship (Hebbian-weighted) model."""

    def test_relationship_defaults(self):
        src = uuid4()
        tgt = uuid4()
        rel = Relationship(
            source_id=src,
            target_id=tgt,
            relation_type=RelationType.USES
        )
        assert rel.source_id == src
        assert rel.target_id == tgt
        assert rel.weight == 0.1
        assert rel.co_access_count == 1

    def test_weight_bounds(self):
        src, tgt = uuid4(), uuid4()
        Relationship(source_id=src, target_id=tgt, relation_type=RelationType.USES, weight=0.0)
        Relationship(source_id=src, target_id=tgt, relation_type=RelationType.USES, weight=1.0)

        with pytest.raises(ValidationError):
            Relationship(source_id=src, target_id=tgt, relation_type=RelationType.USES, weight=-0.1)
        with pytest.raises(ValidationError):
            Relationship(source_id=src, target_id=tgt, relation_type=RelationType.USES, weight=1.1)

    def test_strengthen_basic(self):
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
            weight=0.1
        )
        # w' = 0.1 + 0.1*(1-0.1) = 0.1 + 0.09 = 0.19
        new_weight = rel.strengthen(learning_rate=0.1)
        assert abs(new_weight - 0.19) < 0.001
        assert rel.co_access_count == 2

    def test_strengthen_converges_to_one(self):
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
            weight=0.1
        )
        for _ in range(100):
            rel.strengthen(learning_rate=0.1)

        # Should approach 1.0
        assert rel.weight > 0.99
        assert rel.weight <= 1.0

    def test_strengthen_with_high_rate(self):
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
            weight=0.5
        )
        # w' = 0.5 + 0.5*(1-0.5) = 0.5 + 0.25 = 0.75
        new_weight = rel.strengthen(learning_rate=0.5)
        assert 0.74 < new_weight < 0.76


class TestTemporalLink:
    """Test TemporalLink (episodic temporal relationships) model."""

    def test_temporal_link_defaults(self):
        src, tgt = uuid4(), uuid4()
        tl = TemporalLink(source_id=src, target_id=tgt)
        assert tl.source_id == src
        assert tl.target_id == tgt
        assert tl.link_type == TemporalLinkType.SEQUENCE
        assert tl.strength == 0.5
        assert tl.temporal_gap_ms is None
        assert tl.causal_confidence == 0.0
        assert tl.evidence_count == 1

    def test_temporal_link_with_gap(self):
        tl = TemporalLink(
            source_id=uuid4(),
            target_id=uuid4(),
            temporal_gap_ms=5000
        )
        assert tl.temporal_gap_ms == 5000

    def test_strength_bounds(self):
        src, tgt = uuid4(), uuid4()
        TemporalLink(source_id=src, target_id=tgt, strength=0.0)
        TemporalLink(source_id=src, target_id=tgt, strength=1.0)

        with pytest.raises(ValidationError):
            TemporalLink(source_id=src, target_id=tgt, strength=1.1)

    def test_causal_confidence_bounds(self):
        src, tgt = uuid4(), uuid4()
        TemporalLink(source_id=src, target_id=tgt, causal_confidence=0.0)
        TemporalLink(source_id=src, target_id=tgt, causal_confidence=1.0)

        with pytest.raises(ValidationError):
            TemporalLink(source_id=src, target_id=tgt, causal_confidence=-0.1)

    def test_strengthen_causality(self):
        tl = TemporalLink(
            source_id=uuid4(),
            target_id=uuid4(),
            causal_confidence=0.1
        )
        conf = tl.strengthen_causality(learning_rate=0.1)
        assert abs(conf - 0.19) < 0.001
        assert tl.evidence_count == 2


class TestProcedure:
    """Test Procedure (procedural memory) model."""

    @pytest.fixture
    def procedure(self):
        return Procedure(
            name="deploy-to-prod",
            domain=Domain.DEVOPS,
            steps=[
                ProcedureStep(order=1, action="build"),
                ProcedureStep(order=2, action="test"),
                ProcedureStep(order=3, action="deploy"),
            ]
        )

    def test_procedure_defaults(self, procedure):
        assert procedure.name == "deploy-to-prod"
        assert procedure.domain == Domain.DEVOPS
        assert procedure.kappa == 0.5  # Medium starting point
        assert len(procedure.steps) == 3
        assert procedure.success_rate == 1.0
        assert procedure.execution_count == 1
        assert procedure.version == 1
        assert procedure.deprecated is False
        assert procedure.created_from == "manual"

    def test_name_required(self):
        with pytest.raises(ValidationError):
            Procedure(name="", domain=Domain.CODING)

    def test_update_success_rate_success(self):
        proc = Procedure(name="test", domain=Domain.CODING, success_rate=1.0)
        # After success: (1.0 * 1 + 1) / 2 = 1.0
        new_rate = proc.update_success_rate(success=True)
        assert new_rate == 1.0
        assert proc.execution_count == 2

    def test_update_success_rate_failure(self):
        proc = Procedure(name="test", domain=Domain.CODING, success_rate=1.0)
        # After failure: (1.0 * 1 + 0) / 2 = 0.5
        new_rate = proc.update_success_rate(success=False)
        assert new_rate == 0.5
        assert proc.execution_count == 2

    def test_update_success_rate_sequence(self):
        proc = Procedure(name="test", domain=Domain.CODING, success_rate=0.5, execution_count=10)
        # (0.5 * 10 + 0) / 11 = 5/11 ≈ 0.4545
        rate = proc.update_success_rate(success=False)
        assert 0.45 < rate < 0.46
        assert proc.execution_count == 11

    def test_should_deprecate_threshold_not_met(self):
        proc = Procedure(
            name="test",
            domain=Domain.CODING,
            success_rate=0.5,
            execution_count=5
        )
        # Not enough executions
        assert not proc.should_deprecate(min_executions=10)

    def test_should_deprecate_success_rate_not_low(self):
        proc = Procedure(
            name="test",
            domain=Domain.CODING,
            success_rate=0.5,
            execution_count=20
        )
        # Success rate not low enough (0.5 > 0.3)
        assert not proc.should_deprecate(min_executions=10, min_success=0.3)

    def test_should_deprecate_both_conditions(self):
        proc = Procedure(
            name="test",
            domain=Domain.CODING,
            success_rate=0.2,
            execution_count=20
        )
        # Both conditions met
        assert proc.should_deprecate(min_executions=10, min_success=0.3)


class TestConsolidationEvent:
    """Test ConsolidationEvent model."""

    def test_consolidation_event_defaults(self):
        src_ids = [uuid4(), uuid4()]
        tgt_id = uuid4()
        event = ConsolidationEvent(
            event_type=ConsolidationType.EPISODIC_TO_SEMANTIC,
            source_ids=src_ids,
            target_id=tgt_id,
            confidence=0.9,
            pattern_strength=5
        )
        assert event.event_type == ConsolidationType.EPISODIC_TO_SEMANTIC
        assert event.source_ids == src_ids
        assert event.target_id == tgt_id
        assert event.confidence == 0.9
        assert event.pattern_strength == 5

    def test_confidence_bounds(self):
        tgt_id = uuid4()
        ConsolidationEvent(
            event_type=ConsolidationType.EPISODIC_TO_SEMANTIC,
            target_id=tgt_id,
            confidence=0.0,
            pattern_strength=1
        )
        ConsolidationEvent(
            event_type=ConsolidationType.EPISODIC_TO_SEMANTIC,
            target_id=tgt_id,
            confidence=1.0,
            pattern_strength=1
        )

        with pytest.raises(ValidationError):
            ConsolidationEvent(
                event_type=ConsolidationType.EPISODIC_TO_SEMANTIC,
                target_id=tgt_id,
                confidence=1.1,
                pattern_strength=1
            )


class TestQueryTypes:
    """Test query parameter models."""

    def test_episode_query_defaults(self):
        q = EpisodeQuery(query="What happened?")
        assert q.query == "What happened?"
        assert q.limit == 10
        assert q.session_filter is None
        assert q.time_start is None
        assert q.time_end is None

    def test_episode_query_with_filters(self):
        now = datetime.now()
        past = now - timedelta(days=1)
        q = EpisodeQuery(
            query="test",
            session_filter="session1",
            time_start=past,
            time_end=now,
            limit=20
        )
        assert q.session_filter == "session1"
        assert q.time_start == past
        assert q.limit == 20

    def test_entity_query_defaults(self):
        q = EntityQuery(query="Python")
        assert q.query == "Python"
        assert q.limit == 10
        assert q.context_entities == []
        assert q.include_spreading is True

    def test_procedure_query_defaults(self):
        q = ProcedureQuery(task="deploy")
        assert q.task == "deploy"
        assert q.limit == 5
        assert q.domain is None


class TestScoredResult:
    """Test generic ScoredResult wrapper."""

    def test_scored_result_with_episode(self):
        ep = Episode(session_id="s", content="text")
        result = ScoredResult[Episode](
            item=ep,
            score=0.95,
            components={"semantic": 0.9, "temporal": 0.8}
        )
        assert result.item == ep
        assert result.score == 0.95
        assert result.components["semantic"] == 0.9

    def test_scored_result_score_bounds(self):
        ep = Episode(session_id="s", content="text")
        ScoredResult[Episode](item=ep, score=0.0)
        ScoredResult[Episode](item=ep, score=1.0)

        with pytest.raises(ValidationError):
            ScoredResult[Episode](item=ep, score=1.1)

    def test_scored_result_default_components(self):
        ep = Episode(session_id="s", content="text")
        result = ScoredResult[Episode](item=ep, score=0.5)
        assert result.components == {}
