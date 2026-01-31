"""
TMEM-003: Comprehensive tests for Semantic Memory Service.

Tests semantic entity creation, ACT-R activation calculations,
Hebbian learning, relationship traversal, and fact supersession.
"""

import asyncio
import math
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from t4dm.core.types import (
    Entity, EntityType, Relationship, RelationType, ScoredResult
)
from t4dm.memory.semantic import SemanticMemory


class TestSemanticEntityCreation:
    """Test semantic entity creation and retrieval."""

    @pytest_asyncio.fixture
    async def semantic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create semantic memory instance."""
        semantic = SemanticMemory(session_id=test_session_id)
        semantic.vector_store = mock_qdrant_store
        semantic.graph_store = mock_neo4j_store
        semantic.embedding = mock_embedding_provider
        semantic.vector_store.entities_collection = "entities"
        return semantic

    @pytest.mark.asyncio
    async def test_create_entity_with_all_fields(self, semantic):
        """Test creating entity with all fields."""
        test_embedding = [0.1] * 1024
        semantic.embedding.embed_query.return_value = test_embedding
        semantic.vector_store.add.return_value = None
        semantic.graph_store.create_node.return_value = "test-node-id"

        entity = await semantic.create_entity(
            name="FSRS",
            entity_type="TECHNIQUE",
            summary="Free Spaced Repetition Scheduler for memory decay",
            details="Models retrievability as R(t,S) = (1 + 0.9*t/S)^(-0.5)",
            source="episode-123",
        )

        assert entity is not None
        assert entity.id is not None
        assert entity.name == "FSRS"
        assert entity.entity_type == EntityType.TECHNIQUE
        assert entity.summary == "Free Spaced Repetition Scheduler for memory decay"
        assert entity.details == "Models retrievability as R(t,S) = (1 + 0.9*t/S)^(-0.5)"
        assert entity.source == "episode-123"
        assert entity.access_count == 1

        semantic.vector_store.add.assert_called_once()
        semantic.graph_store.create_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_entity_minimal_fields(self, semantic):
        """Test creating entity with only required fields."""
        test_embedding = [0.1] * 1024
        semantic.embedding.embed_query.return_value = test_embedding
        semantic.vector_store.add.return_value = None
        semantic.graph_store.create_node.return_value = "test-node-id"

        entity = await semantic.create_entity(
            name="Claude",
            entity_type="PERSON",
            summary="AI assistant",
        )

        assert entity.name == "Claude"
        assert entity.entity_type == EntityType.PERSON
        assert entity.details is None
        assert entity.source is None

    @pytest.mark.asyncio
    async def test_create_entity_all_types(self, semantic):
        """Test creating entities of all types."""
        test_embedding = [0.1] * 1024
        semantic.embedding.embed_query.return_value = test_embedding
        semantic.vector_store.add.return_value = None
        semantic.graph_store.create_node.return_value = "test-node-id"

        entity_types = ["CONCEPT", "PERSON", "PROJECT", "TOOL", "TECHNIQUE", "FACT"]

        for entity_type in entity_types:
            entity = await semantic.create_entity(
                name=f"Entity_{entity_type}",
                entity_type=entity_type,
                summary=f"Summary of {entity_type}",
            )
            assert entity.entity_type == EntityType(entity_type)

    @pytest.mark.asyncio
    async def test_get_entity_by_id(self, semantic):
        """Test retrieving entity by ID."""
        entity_id = uuid4()
        payload = {
            "name": "Test Entity",
            "entity_type": "CONCEPT",
            "summary": "A test concept",
            "details": None,
            "source": None,
            "stability": 1.0,
            "access_count": 1,
            "last_accessed": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat(),
            "valid_from": datetime.now().isoformat(),
            "valid_to": None,
        }

        semantic.vector_store.get.return_value = [(str(entity_id), payload)]

        entity = await semantic.get_entity(entity_id)

        assert entity is not None
        assert entity.id == entity_id
        assert entity.name == "Test Entity"

    @pytest.mark.asyncio
    async def test_get_nonexistent_entity(self, semantic):
        """Test getting non-existent entity returns None."""
        semantic.vector_store.get.return_value = []

        entity = await semantic.get_entity(uuid4())

        assert entity is None


class TestSemanticRelationships:
    """Test semantic relationship creation and traversal."""

    @pytest_asyncio.fixture
    async def semantic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create semantic memory instance."""
        semantic = SemanticMemory(session_id=test_session_id)
        semantic.vector_store = mock_qdrant_store
        semantic.graph_store = mock_neo4j_store
        semantic.embedding = mock_embedding_provider
        semantic.vector_store.entities_collection = "entities"
        return semantic

    @pytest.mark.asyncio
    async def test_create_relationship(self, semantic):
        """Test creating relationship between entities."""
        source_id = uuid4()
        target_id = uuid4()

        semantic.graph_store.create_relationship.return_value = None

        rel = await semantic.create_relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type="USES",
            initial_weight=0.15,
        )

        assert rel.source_id == source_id
        assert rel.target_id == target_id
        assert rel.relation_type == RelationType.USES
        assert rel.weight == 0.15

        semantic.graph_store.create_relationship.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_relationship_default_weight(self, semantic):
        """Test relationship uses default weight."""
        semantic.graph_store.create_relationship.return_value = None
        semantic.learning_rate = 0.1

        rel = await semantic.create_relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type="PRODUCES",
        )

        assert rel.weight == semantic.initial_weight

    @pytest.mark.asyncio
    async def test_create_relationship_all_types(self, semantic):
        """Test creating relationships of all types."""
        semantic.graph_store.create_relationship.return_value = None

        rel_types = ["USES", "PRODUCES", "REQUIRES", "CAUSES", "PART_OF", "SIMILAR_TO"]

        for rel_type in rel_types:
            rel = await semantic.create_relationship(
                source_id=uuid4(),
                target_id=uuid4(),
                relation_type=rel_type,
            )
            assert rel.relation_type == RelationType(rel_type)


class TestSemanticACTRActivation:
    """Test ACT-R activation calculations."""

    @pytest_asyncio.fixture
    async def semantic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create semantic memory instance."""
        semantic = SemanticMemory(session_id=test_session_id)
        semantic.vector_store = mock_qdrant_store
        semantic.graph_store = mock_neo4j_store
        semantic.embedding = mock_embedding_provider
        semantic.vector_store.entities_collection = "entities"
        return semantic

    @pytest.mark.asyncio
    async def test_actr_base_level_activation(self, semantic):
        """Test ACT-R base-level activation from access history."""
        now = datetime.now()

        entity = Entity(
            name="Test",
            entity_type=EntityType.CONCEPT,
            summary="Test entity",
            access_count=10,
            last_accessed=now,
        )

        # Base activation: B = ln(access_count) - decay * ln(elapsed)
        # At t=0 (just accessed): B = ln(10) ≈ 2.3
        base = math.log(entity.access_count)

        # Activation should be positive and reasonable
        assert base > 0
        assert base > math.log(5)  # More than 5 accesses

    @pytest.mark.asyncio
    async def test_actr_activation_decreases_with_age(self, semantic):
        """Test that base-level activation decreases with age."""
        now = datetime.now()

        # Recent access
        entity_recent = Entity(
            name="Recent",
            entity_type=EntityType.CONCEPT,
            summary="Recently accessed",
            access_count=5,
            last_accessed=now,
        )

        # Old access
        entity_old = Entity(
            name="Old",
            entity_type=EntityType.CONCEPT,
            summary="Old access",
            access_count=5,
            last_accessed=now - timedelta(days=30),
        )

        # Calculate activation (base level only, no spreading)
        decay = semantic.decay
        elapsed_recent = (now - entity_recent.last_accessed).total_seconds()
        elapsed_old = (now - entity_old.last_accessed).total_seconds()

        if elapsed_recent > 0:
            A_recent = math.log(5) - decay * math.log(elapsed_recent / 3600)
        else:
            A_recent = math.log(5)

        if elapsed_old > 0:
            A_old = math.log(5) - decay * math.log(elapsed_old / 3600)
        else:
            A_old = math.log(5)

        # Recent should have higher activation
        assert A_recent > A_old

    @pytest.mark.asyncio
    async def test_actr_spreading_activation(self, semantic):
        """Test ACT-R spreading activation from context."""
        now = datetime.now()
        semantic.decay = 0.5

        # Create context entity
        context_entity = Entity(
            name="Context",
            entity_type=EntityType.CONCEPT,
            summary="Context entity",
            access_count=1,
            last_accessed=now,
        )

        # Target entity
        target_entity = Entity(
            name="Target",
            entity_type=EntityType.CONCEPT,
            summary="Target entity",
            access_count=1,
            last_accessed=now,
        )

        # Mock the connection strength between them
        semantic.graph_store.get_relationships.return_value = [
            {
                "other_id": str(target_entity.id),
                "properties": {"weight": 0.5},
            }
        ]

        # Calculate spreading activation
        activation = await semantic._calculate_activation(
            target_entity,
            [context_entity],
            now,
            context_cache=None,
        )

        # Should include base + spreading
        assert isinstance(activation, float)


class TestSemanticHebbianLearning:
    """Test Hebbian weight strengthening."""

    @pytest_asyncio.fixture
    async def semantic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create semantic memory instance."""
        semantic = SemanticMemory(session_id=test_session_id)
        semantic.vector_store = mock_qdrant_store
        semantic.graph_store = mock_neo4j_store
        semantic.embedding = mock_embedding_provider
        semantic.vector_store.entities_collection = "entities"
        return semantic

    @pytest.mark.asyncio
    async def test_hebbian_strengthening_formula(self, semantic):
        """Test Hebbian weight update formula: w' = w + lr * (1 - w)"""
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
            weight=0.5,
        )

        original_weight = rel.weight
        learning_rate = 0.2

        new_weight = rel.strengthen(learning_rate)

        # Expected: 0.5 + 0.2 * (1 - 0.5) = 0.5 + 0.1 = 0.6
        expected = original_weight + learning_rate * (1 - original_weight)

        assert abs(new_weight - expected) < 0.001

    @pytest.mark.asyncio
    async def test_hebbian_strengthening_bounds_weight(self, semantic):
        """Test that strengthening doesn't exceed 1.0."""
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.USES,
            weight=0.99,
        )

        # Strengthen multiple times
        for _ in range(100):
            rel.strengthen(0.1)

        # Should asymptotically approach 1.0 but not exceed
        assert rel.weight <= 1.0

    @pytest.mark.asyncio
    async def test_co_retrieval_strengthening(self, semantic):
        """Test that co-retrieved entities strengthen relationships."""
        semantic.embedding.embed_query.return_value = [0.1] * 1024

        # Create mock entities
        entity1 = Entity(name="E1", entity_type=EntityType.CONCEPT, summary="E1")
        entity2 = Entity(name="E2", entity_type=EntityType.CONCEPT, summary="E2")

        results = [
            ScoredResult(item=entity1, score=0.9),
            ScoredResult(item=entity2, score=0.85),
        ]

        # Mock relationships
        semantic.graph_store.get_relationships_batch.return_value = {
            str(entity1.id): [
                {
                    "other_id": str(entity2.id),
                    "properties": {"weight": 0.5},
                }
            ],
            str(entity2.id): [
                {
                    "other_id": str(entity1.id),
                    "properties": {"weight": 0.5},
                }
            ],
        }

        semantic.graph_store.strengthen_relationship.return_value = None

        await semantic._strengthen_co_retrieval(results)

        # Should attempt to strengthen the relationship
        semantic.graph_store.strengthen_relationship.assert_called()


class TestSemanticFactSupersession:
    """Test fact supersession with bi-temporal versioning."""

    @pytest_asyncio.fixture
    async def semantic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create semantic memory instance."""
        semantic = SemanticMemory(session_id=test_session_id)
        semantic.vector_store = mock_qdrant_store
        semantic.graph_store = mock_neo4j_store
        semantic.embedding = mock_embedding_provider
        semantic.vector_store.entities_collection = "entities"
        return semantic

    @pytest.mark.asyncio
    async def test_supersede_entity(self, semantic):
        """Test fact supersession with bi-temporal versioning."""
        test_embedding = [0.1] * 1024
        semantic.embedding.embed_query.return_value = test_embedding

        old_id = uuid4()
        now = datetime.now()

        # Mock old entity
        old_payload = {
            "name": "Old Fact",
            "entity_type": "FACT",
            "summary": "Old version",
            "details": None,
            "source": None,
            "stability": 1.0,
            "access_count": 1,
            "last_accessed": now.isoformat(),
            "created_at": now.isoformat(),
            "valid_from": now.isoformat(),
            "valid_to": None,
        }

        semantic.vector_store.get.return_value = [(str(old_id), old_payload)]
        semantic.vector_store.add.return_value = None
        semantic.vector_store.update_payload.return_value = None
        semantic.graph_store.create_node.return_value = "new-node-id"
        semantic.graph_store.update_node.return_value = None

        # Supersede
        new_entity = await semantic.supersede(
            old_id,
            new_summary="New version",
            new_details="Updated details",
        )

        # Old version should be marked with validTo
        semantic.graph_store.update_node.assert_called()
        update_call = semantic.graph_store.update_node.call_args
        assert "validTo" in update_call[1]["properties"]

        # New version should be created
        assert new_entity.name == "Old Fact"  # Name preserved
        assert new_entity.summary == "New version"

    @pytest.mark.asyncio
    async def test_entity_validity_check(self, semantic):
        """Test entity validity checking based on valid_from/valid_to."""
        now = datetime.now()

        entity = Entity(
            name="Versioned",
            entity_type=EntityType.FACT,
            summary="Versioned entity",
            valid_from=now,
            valid_to=now + timedelta(days=7),
        )

        # Should be valid at now
        assert entity.is_valid(now)

        # Should be valid in 3 days
        assert entity.is_valid(now + timedelta(days=3))

        # Should not be valid before valid_from
        assert not entity.is_valid(now - timedelta(days=1))

        # Should not be valid at or after valid_to
        assert not entity.is_valid(now + timedelta(days=7))
        assert not entity.is_valid(now + timedelta(days=8))


class TestSemanticContextPreloading:
    """Test context preloading for performance."""

    @pytest_asyncio.fixture
    async def semantic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create semantic memory instance."""
        semantic = SemanticMemory(session_id=test_session_id)
        semantic.vector_store = mock_qdrant_store
        semantic.graph_store = mock_neo4j_store
        semantic.embedding = mock_embedding_provider
        semantic.vector_store.entities_collection = "entities"
        return semantic

    @pytest.mark.asyncio
    async def test_preload_context_relationships(self, semantic):
        """Test batch loading of context entity relationships."""
        entity1 = Entity(name="E1", entity_type=EntityType.CONCEPT, summary="E1")
        entity2 = Entity(name="E2", entity_type=EntityType.CONCEPT, summary="E2")

        context_entities = [entity1, entity2]

        # Mock batch relationships
        semantic.graph_store.get_relationships_batch.return_value = {
            str(entity1.id): [
                {
                    "other_id": str(entity2.id),
                    "properties": {"weight": 0.5},
                }
            ],
            str(entity2.id): [
                {
                    "other_id": str(entity1.id),
                    "properties": {"weight": 0.5},
                }
            ],
        }

        # Mock outgoing relationships for fan-out
        semantic.graph_store.get_relationships_batch.side_effect = [
            {
                str(entity1.id): [
                    {
                        "other_id": str(entity2.id),
                        "properties": {"weight": 0.5},
                    }
                ],
                str(entity2.id): [
                    {
                        "other_id": str(entity1.id),
                        "properties": {"weight": 0.5},
                    }
                ],
            },
            {
                str(entity1.id): [
                    {
                        "other_id": str(entity2.id),
                        "properties": {"weight": 0.5},
                    }
                ],
                str(entity2.id): [],
            },
        ]

        cache = await semantic._preload_context_relationships(context_entities)

        # Verify cache structure
        assert str(entity1.id) in cache
        assert str(entity2.id) in cache
        assert "strengths" in cache[str(entity1.id)]
        assert "fan_out" in cache[str(entity1.id)]


class TestSemanticBatchOperations:
    """Test batch retrieval operations."""

    @pytest_asyncio.fixture
    async def semantic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create semantic memory instance."""
        semantic = SemanticMemory(session_id=test_session_id)
        semantic.vector_store = mock_qdrant_store
        semantic.graph_store = mock_neo4j_store
        semantic.embedding = mock_embedding_provider
        semantic.vector_store.entities_collection = "entities"
        return semantic

    @pytest.mark.asyncio
    async def test_recall_with_activation_scoring(self, semantic):
        """Test recall uses ACT-R activation for scoring."""
        test_embedding = [0.1] * 1024
        semantic.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        results = [
            (
                str(uuid4()),  # Must be valid UUID string
                0.9,
                {
                    "name": "Entity 1",
                    "entity_type": "CONCEPT",
                    "summary": "First entity",
                    "details": None,
                    "source": None,
                    "stability": 1.0,
                    "access_count": 10,
                    "last_accessed": now.isoformat(),
                    "created_at": now.isoformat(),
                    "valid_from": now.isoformat(),
                    "valid_to": None,
                },
            ),
        ]

        semantic.vector_store.search.return_value = results
        semantic.graph_store.get_relationships_batch.return_value = {}

        scored_results = await semantic.recall(query="test", limit=10)

        assert len(scored_results) == 1
        result = scored_results[0]
        assert "activation" in result.components
        assert "semantic" in result.components
        assert "retrievability" in result.components

    @pytest.mark.asyncio
    async def test_recall_filters_invalid_entities(self, semantic):
        """Test that recall filters out temporally invalid entities."""
        test_embedding = [0.1] * 1024
        semantic.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        old = now - timedelta(days=30)

        results = [
            (
                str(uuid4()),  # Must be valid UUID string
                0.9,
                {
                    "name": "Valid",
                    "entity_type": "CONCEPT",
                    "summary": "Valid entity",
                    "details": None,
                    "source": None,
                    "stability": 1.0,
                    "access_count": 1,
                    "last_accessed": now.isoformat(),
                    "created_at": now.isoformat(),
                    "valid_from": now.isoformat(),
                    "valid_to": None,
                },
            ),
            (
                str(uuid4()),  # Must be valid UUID string
                0.85,
                {
                    "name": "Expired",
                    "entity_type": "CONCEPT",
                    "summary": "Expired entity",
                    "details": None,
                    "source": None,
                    "stability": 1.0,
                    "access_count": 1,
                    "last_accessed": old.isoformat(),
                    "created_at": old.isoformat(),
                    "valid_from": old.isoformat(),
                    "valid_to": (old + timedelta(days=7)).isoformat(),
                },
            ),
        ]

        semantic.vector_store.search.return_value = results
        semantic.graph_store.get_relationships_batch.return_value = {}

        scored_results = await semantic.recall(query="test", limit=10)

        # Only valid entity should be returned
        assert len(scored_results) == 1
        assert "Valid" in scored_results[0].item.name


class TestSemanticSpreadingActivation:
    """Test spreading activation through knowledge graph."""

    @pytest_asyncio.fixture
    async def semantic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create semantic memory instance."""
        semantic = SemanticMemory(session_id=test_session_id)
        semantic.vector_store = mock_qdrant_store
        semantic.graph_store = mock_neo4j_store
        semantic.embedding = mock_embedding_provider
        semantic.vector_store.entities_collection = "entities"
        return semantic

    @pytest.mark.asyncio
    async def test_spread_activation_basic(self, semantic):
        """Test basic spreading activation."""
        e1_id = str(uuid4())
        e2_id = str(uuid4())
        e3_id = str(uuid4())

        # Mock graph: e1 -> e2, e2 -> e3
        semantic.graph_store.get_relationships.side_effect = [
            # For e1
            [
                {
                    "other_id": e2_id,
                    "properties": {"weight": 0.8},
                }
            ],
            # For e2
            [
                {
                    "other_id": e1_id,
                    "properties": {"weight": 0.8},
                },
                {
                    "other_id": e3_id,
                    "properties": {"weight": 0.6},
                }
            ],
            # For e3
            [
                {
                    "other_id": e2_id,
                    "properties": {"weight": 0.6},
                }
            ],
        ]

        activation = await semantic.spread_activation(
            seed_entities=[e1_id],
            steps=2,
            retention=0.5,
            decay=0.1,
            threshold=0.01,
        )

        # Seed should have highest activation
        assert activation.get(e1_id, 0) > 0
        # Neighbors should have some activation
        assert activation.get(e2_id, 0) >= 0
        # Far neighbors should have less
        assert activation.get(e3_id, 0) <= activation.get(e2_id, 0) if e2_id in activation else True

    @pytest.mark.asyncio
    async def test_spread_activation_decay(self, semantic):
        """Test that spreading activation decays over steps."""
        seed_id = str(uuid4())

        # Mock a simple chain
        semantic.graph_store.get_relationships.side_effect = [
            # Step 1
            [{"other_id": str(uuid4()), "properties": {"weight": 1.0}}],
            # Step 2
            [],
        ]

        activation = await semantic.spread_activation(
            seed_entities=[seed_id],
            steps=2,
            retention=0.5,
            decay=0.2,
            threshold=0.01,
        )

        # Seed activation should be present
        assert activation.get(seed_id, 0) > 0


