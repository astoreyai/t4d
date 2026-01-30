"""
Tests for shared serialization utilities.

Ensures consistent conversion between domain objects and storage formats.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from ww.core.serialization import (
    DateTimeSerializer,
    EpisodeSerializer,
    EntitySerializer,
    ProcedureSerializer,
    get_serializer,
    register_serializer,
    Serializer,
)
from ww.core.types import (
    Episode,
    EpisodeContext,
    Outcome,
    Entity,
    EntityType,
    Procedure,
    Domain,
    ProcedureStep,
)


class TestDateTimeSerializer:
    """Test datetime serialization utilities."""

    def test_to_iso_with_datetime(self):
        """Test datetime to ISO string conversion."""
        dt = datetime(2025, 1, 15, 10, 30, 45)
        result = DateTimeSerializer.to_iso(dt)
        assert result == "2025-01-15T10:30:45"

    def test_to_iso_with_none(self):
        """Test None handling in to_iso."""
        result = DateTimeSerializer.to_iso(None)
        assert result is None

    def test_from_iso_with_string(self):
        """Test ISO string to datetime conversion."""
        iso_str = "2025-01-15T10:30:45"
        result = DateTimeSerializer.from_iso(iso_str)
        assert result == datetime(2025, 1, 15, 10, 30, 45)

    def test_from_iso_with_none(self):
        """Test None handling in from_iso."""
        result = DateTimeSerializer.from_iso(None)
        assert result is None

    def test_roundtrip(self):
        """Test datetime serialization roundtrip."""
        original = datetime(2025, 1, 15, 10, 30, 45)
        iso_str = DateTimeSerializer.to_iso(original)
        restored = DateTimeSerializer.from_iso(iso_str)
        assert restored == original

    def test_roundtrip_with_timezone(self):
        """Test datetime with timezone roundtrip."""
        original = datetime(2025, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        iso_str = DateTimeSerializer.to_iso(original)
        restored = DateTimeSerializer.from_iso(iso_str)
        assert restored == original


class TestEpisodeSerializer:
    """Test Episode serialization."""

    @pytest.fixture
    def serializer(self):
        """Episode serializer instance."""
        return EpisodeSerializer()

    @pytest.fixture
    def episode(self):
        """Sample episode for testing."""
        return Episode(
            id=uuid4(),
            session_id="test-session",
            content="User asked about pytest, Claude explained fixtures.",
            timestamp=datetime(2025, 1, 15, 10, 30, 45),
            ingested_at=datetime(2025, 1, 15, 10, 30, 50),
            context=EpisodeContext(
                project="world-weaver",
                file="tests/test_serialization.py",
                tool="pytest",
            ),
            outcome=Outcome.SUCCESS,
            emotional_valence=0.8,
            access_count=5,
            last_accessed=datetime(2025, 1, 15, 11, 0, 0),
            stability=2.5,
        )

    def test_to_payload(self, serializer, episode):
        """Test episode to payload conversion."""
        payload = serializer.to_payload(episode, "test-session")

        assert payload["session_id"] == "test-session"
        assert payload["content"] == episode.content
        assert payload["timestamp"] == "2025-01-15T10:30:45"
        assert payload["ingested_at"] == "2025-01-15T10:30:50"
        assert payload["context"]["project"] == "world-weaver"
        assert payload["outcome"] == "success"
        assert payload["emotional_valence"] == 0.8
        assert payload["access_count"] == 5
        assert payload["stability"] == 2.5

    def test_from_payload(self, serializer, episode):
        """Test payload to episode conversion."""
        payload = serializer.to_payload(episode, "test-session")
        restored = serializer.from_payload(str(episode.id), payload)

        assert restored.id == episode.id
        assert restored.session_id == episode.session_id
        assert restored.content == episode.content
        assert restored.timestamp == episode.timestamp
        assert restored.ingested_at == episode.ingested_at
        assert restored.context.project == episode.context.project
        assert restored.outcome == episode.outcome
        assert restored.emotional_valence == episode.emotional_valence
        assert restored.access_count == episode.access_count
        assert restored.stability == episode.stability
        assert restored.embedding is None

    def test_roundtrip(self, serializer, episode):
        """Test episode serialization roundtrip."""
        payload = serializer.to_payload(episode, "test-session")
        restored = serializer.from_payload(str(episode.id), payload)

        # Verify all critical fields match (excluding embedding which is always None in payload)
        assert restored.id == episode.id
        assert restored.content == episode.content
        assert restored.timestamp == episode.timestamp
        assert restored.outcome == episode.outcome
        assert restored.stability == episode.stability

    def test_to_graph_props(self, serializer, episode):
        """Test episode to graph properties conversion."""
        props = serializer.to_graph_props(episode, "test-session")

        assert props["id"] == str(episode.id)
        assert props["sessionId"] == "test-session"
        assert props["content"] == episode.content  # Not truncated (< 500 chars)
        assert props["timestamp"] == "2025-01-15T10:30:45"
        assert props["outcome"] == "success"
        assert props["emotionalValence"] == 0.8
        assert props["stability"] == 2.5

    def test_graph_props_truncation(self, serializer):
        """Test content truncation in graph properties."""
        long_content = "a" * 1000  # 1000 chars
        episode = Episode(
            session_id="test",
            content=long_content,
        )

        props = serializer.to_graph_props(episode, "test")
        assert len(props["content"]) == 500
        assert props["content"] == "a" * 500

    def test_from_payload_minimal(self, serializer):
        """Test from_payload with minimal data."""
        minimal_payload = {
            "session_id": "test",
            "content": "minimal episode",
        }

        restored = serializer.from_payload("00000000-0000-0000-0000-000000000001", minimal_payload)

        assert restored.session_id == "test"
        assert restored.content == "minimal episode"
        assert restored.outcome == Outcome.NEUTRAL  # Default
        assert restored.emotional_valence == 0.0  # Default
        assert restored.access_count == 1  # Default (minimum)
        assert restored.stability == 1.0  # Default
        # Datetime fields should be populated with defaults
        assert restored.timestamp is not None
        assert restored.ingested_at is not None
        assert restored.last_accessed is not None


class TestEntitySerializer:
    """Test Entity serialization."""

    @pytest.fixture
    def serializer(self):
        """Entity serializer instance."""
        return EntitySerializer()

    @pytest.fixture
    def entity(self):
        """Sample entity for testing."""
        return Entity(
            id=uuid4(),
            name="pytest",
            entity_type=EntityType.TOOL,
            summary="Python testing framework",
            details="Fixture-based testing framework with parametrization support",
            source="episode-123",
            stability=3.5,
            access_count=12,
            last_accessed=datetime(2025, 1, 15, 11, 0, 0),
            created_at=datetime(2025, 1, 10, 9, 0, 0),
            valid_from=datetime(2025, 1, 10, 9, 0, 0),
            valid_to=None,
        )

    def test_to_payload(self, serializer, entity):
        """Test entity to payload conversion."""
        payload = serializer.to_payload(entity, "test-session")

        assert payload["session_id"] == "test-session"
        assert payload["name"] == "pytest"
        assert payload["entity_type"] == "TOOL"
        assert payload["summary"] == "Python testing framework"
        assert payload["details"] == entity.details
        assert payload["source"] == "episode-123"
        assert payload["stability"] == 3.5
        assert payload["access_count"] == 12

    def test_from_payload(self, serializer, entity):
        """Test payload to entity conversion."""
        payload = serializer.to_payload(entity, "test-session")
        restored = serializer.from_payload(str(entity.id), payload)

        assert restored.id == entity.id
        assert restored.name == entity.name
        assert restored.entity_type == entity.entity_type
        assert restored.summary == entity.summary
        assert restored.details == entity.details
        assert restored.source == entity.source
        assert restored.stability == entity.stability
        assert restored.access_count == entity.access_count
        assert restored.embedding is None

    def test_roundtrip(self, serializer, entity):
        """Test entity serialization roundtrip."""
        payload = serializer.to_payload(entity, "test-session")
        restored = serializer.from_payload(str(entity.id), payload)

        assert restored.id == entity.id
        assert restored.name == entity.name
        assert restored.entity_type == entity.entity_type
        assert restored.valid_from == entity.valid_from
        assert restored.valid_to == entity.valid_to

    def test_to_graph_props(self, serializer, entity):
        """Test entity to graph properties conversion."""
        props = serializer.to_graph_props(entity, "test-session")

        assert props["id"] == str(entity.id)
        assert props["sessionId"] == "test-session"
        assert props["name"] == "pytest"
        assert props["entityType"] == "TOOL"
        assert props["summary"] == "Python testing framework"
        assert props["stability"] == 3.5
        assert props["validTo"] == ""  # None becomes empty string

    def test_graph_props_truncation(self, serializer):
        """Test summary/details truncation in graph properties."""
        long_summary = "s" * 1000
        long_details = "d" * 1000
        entity = Entity(
            name="test",
            entity_type=EntityType.CONCEPT,
            summary=long_summary,
            details=long_details,
        )

        props = serializer.to_graph_props(entity, "test")
        assert len(props["summary"]) == 500
        assert len(props["details"]) == 500

    def test_from_payload_minimal(self, serializer):
        """Test from_payload with minimal data."""
        minimal_payload = {
            "name": "test-entity",
            "summary": "test summary",
        }

        restored = serializer.from_payload("00000000-0000-0000-0000-000000000001", minimal_payload)

        assert restored.name == "test-entity"
        assert restored.summary == "test summary"
        assert restored.entity_type == EntityType.CONCEPT  # Default
        assert restored.stability == 1.0  # Default
        assert restored.access_count == 1  # Default (minimum)
        # Datetime fields should be populated with defaults
        assert restored.last_accessed is not None
        assert restored.created_at is not None
        assert restored.valid_from is not None


class TestProcedureSerializer:
    """Test Procedure serialization."""

    @pytest.fixture
    def serializer(self):
        """Procedure serializer instance."""
        return ProcedureSerializer()

    @pytest.fixture
    def procedure(self):
        """Sample procedure for testing."""
        return Procedure(
            id=uuid4(),
            name="Run pytest with coverage",
            domain=Domain.CODING,
            trigger_pattern="run tests",
            steps=[
                ProcedureStep(
                    order=1,
                    action="pytest tests/ -v",
                    tool="pytest",
                    parameters={"verbose": True},
                    expected_outcome="All tests pass",
                ),
                ProcedureStep(
                    order=2,
                    action="pytest --cov=src tests/",
                    tool="pytest",
                    parameters={"coverage": True},
                    expected_outcome="Coverage > 90%",
                ),
            ],
            script="PROCEDURE: Run tests with coverage\nSTEPS:\n  1. pytest tests/ -v\n  2. pytest --cov=src tests/",
            success_rate=0.95,
            execution_count=20,
            last_executed=datetime(2025, 1, 15, 11, 0, 0),
            version=2,
            deprecated=False,
            consolidated_into=None,
            created_at=datetime(2025, 1, 10, 9, 0, 0),
            created_from="trajectory",
        )

    def test_to_payload(self, serializer, procedure):
        """Test procedure to payload conversion."""
        payload = serializer.to_payload(procedure, "test-session")

        assert payload["session_id"] == "test-session"
        assert payload["name"] == "Run pytest with coverage"
        assert payload["domain"] == "coding"
        assert payload["trigger_pattern"] == "run tests"
        assert len(payload["steps"]) == 2
        assert payload["steps"][0]["action"] == "pytest tests/ -v"
        assert payload["script"] == procedure.script
        assert payload["success_rate"] == 0.95
        assert payload["execution_count"] == 20
        assert payload["version"] == 2
        assert payload["deprecated"] is False
        assert payload["created_from"] == "trajectory"

    def test_from_payload(self, serializer, procedure):
        """Test payload to procedure conversion."""
        payload = serializer.to_payload(procedure, "test-session")
        restored = serializer.from_payload(str(procedure.id), payload)

        assert restored.id == procedure.id
        assert restored.name == procedure.name
        assert restored.domain == procedure.domain
        assert restored.trigger_pattern == procedure.trigger_pattern
        assert len(restored.steps) == 2
        assert restored.steps[0].action == "pytest tests/ -v"
        assert restored.script == procedure.script
        assert restored.success_rate == procedure.success_rate
        assert restored.execution_count == procedure.execution_count
        assert restored.embedding is None

    def test_roundtrip(self, serializer, procedure):
        """Test procedure serialization roundtrip."""
        payload = serializer.to_payload(procedure, "test-session")
        restored = serializer.from_payload(str(procedure.id), payload)

        assert restored.id == procedure.id
        assert restored.name == procedure.name
        assert restored.domain == procedure.domain
        assert len(restored.steps) == len(procedure.steps)
        assert restored.success_rate == procedure.success_rate
        assert restored.version == procedure.version

    def test_to_graph_props(self, serializer, procedure):
        """Test procedure to graph properties conversion."""
        props = serializer.to_graph_props(procedure, "test-session")

        assert props["id"] == str(procedure.id)
        assert props["sessionId"] == "test-session"
        assert props["name"] == "Run pytest with coverage"
        assert props["domain"] == "coding"
        assert props["triggerPattern"] == "run tests"
        assert props["stepCount"] == 2
        assert props["successRate"] == 0.95
        assert props["executionCount"] == 20
        assert props["version"] == 2
        assert props["deprecated"] is False

    def test_graph_props_script_truncation(self, serializer):
        """Test script truncation in graph properties."""
        long_script = "s" * 1000
        procedure = Procedure(
            name="test",
            domain=Domain.CODING,
            script=long_script,
        )

        props = serializer.to_graph_props(procedure, "test")
        assert len(props["script"]) == 500

    def test_from_payload_minimal(self, serializer):
        """Test from_payload with minimal data."""
        minimal_payload = {
            "name": "test-procedure",
        }

        restored = serializer.from_payload("00000000-0000-0000-0000-000000000001", minimal_payload)

        assert restored.name == "test-procedure"
        assert restored.domain == Domain.CODING  # Default
        assert restored.success_rate == 1.0  # Default
        assert restored.execution_count == 1  # Default (minimum)
        assert restored.version == 1  # Default
        assert restored.deprecated is False  # Default
        assert restored.created_from == "manual"  # Default
        # Datetime fields should be populated with defaults
        assert restored.created_at is not None


class TestSerializerFactory:
    """Test serializer factory functions."""

    def test_get_serializer_episode(self):
        """Test getting episode serializer."""
        serializer = get_serializer("episode")
        assert isinstance(serializer, EpisodeSerializer)

    def test_get_serializer_entity(self):
        """Test getting entity serializer."""
        serializer = get_serializer("entity")
        assert isinstance(serializer, EntitySerializer)

    def test_get_serializer_procedure(self):
        """Test getting procedure serializer."""
        serializer = get_serializer("procedure")
        assert isinstance(serializer, ProcedureSerializer)

    def test_get_serializer_invalid(self):
        """Test getting invalid serializer raises error."""
        with pytest.raises(ValueError, match="No serializer registered for type: invalid"):
            get_serializer("invalid")

    def test_register_custom_serializer(self):
        """Test registering custom serializer."""
        class CustomSerializer(Serializer):
            def to_payload(self, obj, session_id):
                return {"custom": "payload"}

            def from_payload(self, id_str, payload):
                return {"custom": "object"}

            def to_graph_props(self, obj, session_id):
                return {"custom": "props"}

        custom = CustomSerializer()
        register_serializer("custom", custom)

        retrieved = get_serializer("custom")
        assert retrieved is custom
        assert retrieved.to_payload(None, "test") == {"custom": "payload"}

    def test_get_serializer_caches_instances(self):
        """Test serializer instances are cached."""
        serializer1 = get_serializer("episode")
        serializer2 = get_serializer("episode")
        assert serializer1 is serializer2  # Same instance


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_episode_with_none_datetime_fields(self):
        """Test episode serialization with None datetime fields in payload."""
        serializer = EpisodeSerializer()

        # Create payload with None datetime fields
        payload = {
            "session_id": "test",
            "content": "test",
            "timestamp": None,
            "ingested_at": None,
            "last_accessed": None,
            "context": {},
            "outcome": "neutral",
            "emotional_valence": 0.5,
            "access_count": 1,
            "stability": 1.0,
        }

        # Restore - should handle None by using defaults
        restored = serializer.from_payload("00000000-0000-0000-0000-000000000001", payload)
        assert restored.last_accessed is not None  # Default to now
        assert restored.timestamp is not None  # Default to now
        assert restored.ingested_at is not None  # Default to now

    def test_entity_with_empty_strings(self):
        """Test entity serialization with empty optional strings."""
        entity = Entity(
            name="test",
            entity_type=EntityType.CONCEPT,
            summary="test",
            details="",  # Empty string
            source="",  # Empty string
        )

        serializer = EntitySerializer()
        payload = serializer.to_payload(entity, "test")

        assert payload["details"] == ""
        assert payload["source"] == ""

        restored = serializer.from_payload(str(entity.id), payload)
        assert restored.details == ""
        assert restored.source == ""

    def test_procedure_with_empty_steps(self):
        """Test procedure serialization with empty steps list."""
        procedure = Procedure(
            name="test",
            domain=Domain.CODING,
            steps=[],  # Empty steps
        )

        serializer = ProcedureSerializer()
        payload = serializer.to_payload(procedure, "test")

        assert payload["steps"] == []

        restored = serializer.from_payload(str(procedure.id), payload)
        assert restored.steps == []

        props = serializer.to_graph_props(procedure, "test")
        assert props["stepCount"] == 0
