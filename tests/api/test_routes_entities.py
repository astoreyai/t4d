"""Tests for entities API routes."""

import pytest
from datetime import datetime
from uuid import uuid4

from ww.api.routes.entities import (
    EntityCreate,
    EntityResponse,
    EntityList,
    RelationCreate,
    RelationResponse,
    SemanticRecallRequest,
    SpreadActivationRequest,
    ActivationResponse,
)
from ww.core.types import EntityType, RelationType


class TestEntityCreate:
    """Tests for EntityCreate model."""

    def test_basic_entity(self):
        """Create basic entity."""
        entity = EntityCreate(
            name="Test Entity",
            entity_type=EntityType.CONCEPT,
            summary="A test concept",
        )
        assert entity.name == "Test Entity"
        assert entity.entity_type == EntityType.CONCEPT
        assert entity.summary == "A test concept"
        assert entity.details is None
        assert entity.source is None

    def test_entity_with_all_fields(self):
        """Create entity with all fields."""
        entity = EntityCreate(
            name="Project X",
            entity_type=EntityType.PROJECT,
            summary="Our main project",
            details="Extended description of Project X with more context",
            source="episode-123",
        )
        assert entity.details is not None
        assert entity.source == "episode-123"

    def test_entity_types(self):
        """Test all entity types can be used."""
        for entity_type in EntityType:
            entity = EntityCreate(
                name="Test",
                entity_type=entity_type,
                summary="A test",
            )
            assert entity.entity_type == entity_type

    def test_empty_name(self):
        """Name cannot be empty."""
        with pytest.raises(ValueError):
            EntityCreate(
                name="",
                entity_type=EntityType.CONCEPT,
                summary="Test",
            )

    def test_empty_summary(self):
        """Summary cannot be empty."""
        with pytest.raises(ValueError):
            EntityCreate(
                name="Test",
                entity_type=EntityType.CONCEPT,
                summary="",
            )


class TestEntityResponse:
    """Tests for EntityResponse model."""

    def test_basic_response(self):
        """Create basic entity response."""
        entity_id = uuid4()
        now = datetime.now()
        response = EntityResponse(
            id=entity_id,
            name="Test",
            entity_type=EntityType.CONCEPT,
            summary="A test",
            details=None,
            source=None,
            stability=0.5,
            access_count=0,
            created_at=now,
            valid_from=now,
            valid_to=None,
        )
        assert response.id == entity_id
        assert response.stability == 0.5
        assert response.access_count == 0

    def test_response_with_all_fields(self):
        """Create response with all fields."""
        entity_id = uuid4()
        now = datetime.now()
        response = EntityResponse(
            id=entity_id,
            name="Entity",
            entity_type=EntityType.PERSON,
            summary="A person",
            details="More about this person",
            source="episode-456",
            stability=0.8,
            access_count=42,
            created_at=now,
            valid_from=now,
            valid_to=now,
        )
        assert response.details is not None
        assert response.source == "episode-456"
        assert response.access_count == 42
        assert response.valid_to is not None


class TestEntityList:
    """Tests for EntityList model."""

    def test_empty_list(self):
        """Create empty entity list."""
        entity_list = EntityList(entities=[], total=0)
        assert len(entity_list.entities) == 0
        assert entity_list.total == 0

    def test_list_with_entities(self):
        """Create list with entities."""
        now = datetime.now()
        entities = [
            EntityResponse(
                id=uuid4(),
                name=f"Entity {i}",
                entity_type=EntityType.CONCEPT,
                summary=f"Concept {i}",
                details=None,
                source=None,
                stability=0.5,
                access_count=i,
                created_at=now,
                valid_from=now,
                valid_to=None,
            )
            for i in range(5)
        ]
        entity_list = EntityList(entities=entities, total=5)
        assert len(entity_list.entities) == 5
        assert entity_list.total == 5


class TestRelationCreate:
    """Tests for RelationCreate model."""

    def test_basic_relation(self):
        """Create basic relationship."""
        source_id = uuid4()
        target_id = uuid4()
        relation = RelationCreate(
            source_id=source_id,
            target_id=target_id,
            relation_type=RelationType.USES,
        )
        assert relation.source_id == source_id
        assert relation.target_id == target_id
        assert relation.relation_type == RelationType.USES
        assert relation.weight == 0.1  # default

    def test_relation_with_weight(self):
        """Create relationship with custom weight."""
        relation = RelationCreate(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.PRODUCES,
            weight=0.8,
        )
        assert relation.weight == 0.8

    def test_weight_bounds(self):
        """Weight has valid bounds."""
        # Valid min
        relation = RelationCreate(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
            weight=0.0,
        )
        assert relation.weight == 0.0

        # Valid max
        relation = RelationCreate(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
            weight=1.0,
        )
        assert relation.weight == 1.0

        # Below min
        with pytest.raises(ValueError):
            RelationCreate(
                source_id=uuid4(),
                target_id=uuid4(),
                relation_type=RelationType.USES,
                weight=-0.1,
            )

        # Above max
        with pytest.raises(ValueError):
            RelationCreate(
                source_id=uuid4(),
                target_id=uuid4(),
                relation_type=RelationType.USES,
                weight=1.5,
            )

    def test_all_relation_types(self):
        """Test all relation types can be used."""
        for rel_type in RelationType:
            relation = RelationCreate(
                source_id=uuid4(),
                target_id=uuid4(),
                relation_type=rel_type,
            )
            assert relation.relation_type == rel_type


class TestRelationResponse:
    """Tests for RelationResponse model."""

    def test_basic_response(self):
        """Create basic relation response."""
        source_id = uuid4()
        target_id = uuid4()
        response = RelationResponse(
            source_id=source_id,
            target_id=target_id,
            relation_type=RelationType.SIMILAR_TO,
            weight=0.5,
            co_access_count=10,
        )
        assert response.source_id == source_id
        assert response.weight == 0.5
        assert response.co_access_count == 10


class TestSemanticRecallRequest:
    """Tests for SemanticRecallRequest model."""

    def test_basic_request(self):
        """Create basic recall request."""
        request = SemanticRecallRequest(query="test query")
        assert request.query == "test query"
        assert request.limit == 10  # default
        assert request.entity_types is None

    def test_request_with_filters(self):
        """Create request with filters."""
        request = SemanticRecallRequest(
            query="find concepts",
            limit=25,
            entity_types=[EntityType.CONCEPT, EntityType.PROJECT],
        )
        assert request.limit == 25
        assert len(request.entity_types) == 2

    def test_empty_query(self):
        """Query cannot be empty."""
        with pytest.raises(ValueError):
            SemanticRecallRequest(query="")

    def test_limit_bounds(self):
        """Limit has valid bounds."""
        # Valid min
        request = SemanticRecallRequest(query="test", limit=1)
        assert request.limit == 1

        # Valid max
        request = SemanticRecallRequest(query="test", limit=100)
        assert request.limit == 100

        # Below min
        with pytest.raises(ValueError):
            SemanticRecallRequest(query="test", limit=0)

        # Above max
        with pytest.raises(ValueError):
            SemanticRecallRequest(query="test", limit=101)


class TestSpreadActivationRequest:
    """Tests for SpreadActivationRequest model."""

    def test_basic_request(self):
        """Create basic spread activation request."""
        entity_id = uuid4()
        request = SpreadActivationRequest(entity_id=entity_id)
        assert request.entity_id == entity_id
        assert request.depth == 2  # default
        assert request.threshold == 0.1  # default

    def test_request_with_params(self):
        """Create request with custom params."""
        request = SpreadActivationRequest(
            entity_id=uuid4(),
            depth=4,
            threshold=0.5,
        )
        assert request.depth == 4
        assert request.threshold == 0.5

    def test_depth_bounds(self):
        """Depth has valid bounds."""
        entity_id = uuid4()

        # Valid min
        request = SpreadActivationRequest(entity_id=entity_id, depth=1)
        assert request.depth == 1

        # Valid max
        request = SpreadActivationRequest(entity_id=entity_id, depth=5)
        assert request.depth == 5

        # Below min
        with pytest.raises(ValueError):
            SpreadActivationRequest(entity_id=entity_id, depth=0)

        # Above max
        with pytest.raises(ValueError):
            SpreadActivationRequest(entity_id=entity_id, depth=6)

    def test_threshold_bounds(self):
        """Threshold has valid bounds."""
        entity_id = uuid4()

        # Valid min
        request = SpreadActivationRequest(entity_id=entity_id, threshold=0.0)
        assert request.threshold == 0.0

        # Valid max
        request = SpreadActivationRequest(entity_id=entity_id, threshold=1.0)
        assert request.threshold == 1.0

        # Below min
        with pytest.raises(ValueError):
            SpreadActivationRequest(entity_id=entity_id, threshold=-0.1)

        # Above max
        with pytest.raises(ValueError):
            SpreadActivationRequest(entity_id=entity_id, threshold=1.1)


class TestActivationResponse:
    """Tests for ActivationResponse model."""

    def test_empty_response(self):
        """Create empty activation response."""
        response = ActivationResponse(
            entities=[],
            activations=[],
            paths=[],
        )
        assert len(response.entities) == 0
        assert len(response.activations) == 0
        assert len(response.paths) == 0

    def test_response_with_data(self):
        """Create response with data."""
        now = datetime.now()
        entities = [
            EntityResponse(
                id=uuid4(),
                name=f"Entity {i}",
                entity_type=EntityType.CONCEPT,
                summary="Test",
                details=None,
                source=None,
                stability=0.5,
                access_count=0,
                created_at=now,
                valid_from=now,
                valid_to=None,
            )
            for i in range(3)
        ]
        activations = [1.0, 0.7, 0.5]
        paths = [
            ["start", "entity1"],
            ["start", "entity1", "entity2"],
            ["start", "entity1", "entity2", "entity3"],
        ]

        response = ActivationResponse(
            entities=entities,
            activations=activations,
            paths=paths,
        )
        assert len(response.entities) == 3
        assert len(response.activations) == 3
        assert len(response.paths) == 3
        assert response.activations[0] == 1.0


class TestEntityTypeEnum:
    """Tests for EntityType enum values."""

    def test_all_entity_types(self):
        """Test all entity type values."""
        types = list(EntityType)
        assert len(types) > 0

        # Common types should exist
        assert EntityType.CONCEPT in types
        assert EntityType.PERSON in types
        assert EntityType.PROJECT in types

    def test_entity_type_values(self):
        """Test entity type string values."""
        assert EntityType.CONCEPT.value == "CONCEPT"
        assert EntityType.PERSON.value == "PERSON"


class TestRelationTypeEnum:
    """Tests for RelationType enum values."""

    def test_all_relation_types(self):
        """Test all relation type values."""
        types = list(RelationType)
        assert len(types) > 0

        # Common types should exist
        assert RelationType.USES in types
        assert RelationType.PRODUCES in types
        assert RelationType.SIMILAR_TO in types

    def test_relation_type_values(self):
        """Test relation type string values."""
        assert RelationType.USES.value == "USES"
        assert RelationType.SIMILAR_TO.value == "SIMILAR_TO"
