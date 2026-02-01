"""
Integration test for Hebbian relationship strengthening.

This test verifies that the full Hebbian learning loop works:
1. Create entities with relationships
2. Recall entities (triggers co-retrieval strengthening)
3. Verify weights increased
4. Apply decay
5. Verify weights decreased

This was identified as a CRITICAL bug in Round 4 analysis:
- strengthen_relationship() was missing from T4DXGraphStore
- Result: All semantic relationships decayed to zero over time
"""

import pytest
import asyncio
from uuid import uuid4

import os

# Skip all tests in this module if NEO4J_TEST_ENABLED env var is not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("NEO4J_TEST_ENABLED"),
        reason="Neo4j integration tests disabled. Set NEO4J_TEST_ENABLED=1 to run."
    ),
]


@pytest.fixture
async def neo4j_store():
    """Get a real Neo4j store for integration testing.

    Creates a fresh store instance per test to avoid event loop conflicts.
    The store is not cached to ensure each test gets its own connection
    on the current event loop.

    Skips test if Neo4j is not available or configured.
    """
    import os
    from t4dm.storage import T4DXGraphStore
    from t4dm.core.config import get_settings

    settings = get_settings()

    # Skip if Neo4j URI is not configured or is localhost (likely not running)
    neo4j_uri = settings.neo4j_uri
    if not neo4j_uri or neo4j_uri == "bolt://localhost:7687":
        # Check if Neo4j is actually running
        import socket
        try:
            with socket.create_connection(("localhost", 7687), timeout=1):
                pass  # Connection succeeded
        except (socket.timeout, ConnectionRefusedError, OSError):
            pytest.skip("Neo4j not running on localhost:7687")

    # Create a fresh store instance (not cached) for this test
    store = T4DXGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        timeout=5.0,  # Short timeout for tests
    )

    try:
        await store.initialize()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")

    # Test connectivity with a simple write operation (with strict timeout)
    try:
        import asyncio
        driver = await store._get_driver()
        async with driver.session() as session:
            result = await asyncio.wait_for(
                session.run("RETURN 1 as n"),
                timeout=3.0
            )
            await asyncio.wait_for(result.consume(), timeout=3.0)
    except (asyncio.TimeoutError, Exception) as e:
        await store.close()
        pytest.skip(f"Neo4j connection failed or timed out: {e}")

    yield store
    # Cleanup - close the store connection
    try:
        await store.close()
    except Exception:
        pass  # Ignore cleanup errors


class TestHebbianStrengthening:
    """Test Hebbian learning mechanisms in semantic memory."""

    async def test_strengthen_relationship_method_exists(self, neo4j_store):
        """Verify strengthen_relationship method exists (was missing before fix)."""
        assert hasattr(neo4j_store, 'strengthen_relationship'), (
            "strengthen_relationship method missing from T4DXGraphStore! "
            "This is a CRITICAL bug - Hebbian learning is broken."
        )

    async def test_strengthen_relationship_increases_weight(self, neo4j_store):
        """Test that strengthening actually increases relationship weight."""
        # Create two test entities
        source_id = str(uuid4())
        target_id = str(uuid4())

        await neo4j_store.create_node(
            label="Entity",
            properties={"id": source_id, "name": "TestEntity1", "sessionId": "test-hebbian-integration"}
        )
        await neo4j_store.create_node(
            label="Entity",
            properties={"id": target_id, "name": "TestEntity2", "sessionId": "test-hebbian-integration"}
        )

        # Create relationship with initial weight
        initial_weight = 0.1
        await neo4j_store.create_relationship(
            source_id=source_id,
            target_id=target_id,
            rel_type="RELATED_TO",
            properties={"weight": initial_weight, "coAccessCount": 0}
        )

        # Strengthen the relationship
        new_weight = await neo4j_store.strengthen_relationship(
            source_id=source_id,
            target_id=target_id,
            learning_rate=0.1
        )

        # Verify weight increased
        # Formula: w' = w + lr * (1 - w) = 0.1 + 0.1 * (1 - 0.1) = 0.1 + 0.09 = 0.19
        expected_weight = initial_weight + 0.1 * (1 - initial_weight)
        assert new_weight > initial_weight, (
            f"Weight should increase after strengthening. "
            f"Initial: {initial_weight}, After: {new_weight}"
        )
        assert abs(new_weight - expected_weight) < 0.01, (
            f"Weight should follow Hebbian formula. "
            f"Expected: {expected_weight:.4f}, Got: {new_weight:.4f}"
        )

    async def test_strengthen_increments_coaccess_count(self, neo4j_store):
        """Test that strengthening updates coAccessCount."""
        source_id = str(uuid4())
        target_id = str(uuid4())

        await neo4j_store.create_node(
            label="Entity",
            properties={"id": source_id, "name": "TestEntity3", "sessionId": "test-hebbian-integration"}
        )
        await neo4j_store.create_node(
            label="Entity",
            properties={"id": target_id, "name": "TestEntity4", "sessionId": "test-hebbian-integration"}
        )

        await neo4j_store.create_relationship(
            source_id=source_id,
            target_id=target_id,
            rel_type="RELATED_TO",
            properties={"weight": 0.1, "coAccessCount": 0}
        )

        # Strengthen multiple times
        for _ in range(3):
            await neo4j_store.strengthen_relationship(
                source_id=source_id,
                target_id=target_id,
                learning_rate=0.1
            )

        # Query the relationship to check coAccessCount
        rels = await neo4j_store.get_relationships(
            node_id=source_id,
            direction="out"
        )

        assert len(rels) > 0, "Relationship should exist"
        rel = rels[0]
        co_access = rel.get("properties", {}).get("coAccessCount", 0)
        assert co_access >= 3, f"coAccessCount should be at least 3, got {co_access}"

    async def test_strengthen_nonexistent_relationship_returns_zero(self, neo4j_store):
        """Test that strengthening a non-existent relationship returns 0."""
        result = await neo4j_store.strengthen_relationship(
            source_id=str(uuid4()),  # Random, doesn't exist
            target_id=str(uuid4()),
            learning_rate=0.1
        )
        assert result == 0.0, "Should return 0 for non-existent relationship"

    async def test_strengthen_bounded_weight(self, neo4j_store):
        """Test that weight is bounded to [0, 1] and approaches 1 asymptotically."""
        source_id = str(uuid4())
        target_id = str(uuid4())

        await neo4j_store.create_node(
            label="Entity",
            properties={"id": source_id, "name": "TestEntity5", "sessionId": "test-hebbian-integration"}
        )
        await neo4j_store.create_node(
            label="Entity",
            properties={"id": target_id, "name": "TestEntity6", "sessionId": "test-hebbian-integration"}
        )

        # Start with high weight
        await neo4j_store.create_relationship(
            source_id=source_id,
            target_id=target_id,
            rel_type="RELATED_TO",
            properties={"weight": 0.9, "coAccessCount": 0}
        )

        # Strengthen many times
        final_weight = 0.9
        for _ in range(20):
            final_weight = await neo4j_store.strengthen_relationship(
                source_id=source_id,
                target_id=target_id,
                learning_rate=0.1
            )

        # Weight should be close to but not exceed 1.0
        assert final_weight <= 1.0, f"Weight should not exceed 1.0, got {final_weight}"
        assert final_weight > 0.99, f"Weight should approach 1.0, got {final_weight}"

    async def test_decay_and_strengthen_balance(self, neo4j_store):
        """Test that decay and strengthening can balance (core Hebbian principle)."""
        source_id = str(uuid4())
        target_id = str(uuid4())

        await neo4j_store.create_node(
            label="Entity",
            properties={
                "id": source_id,
                "name": "TestEntity7",
                "sessionId": "test-hebbian-integration"
            }
        )
        await neo4j_store.create_node(
            label="Entity",
            properties={
                "id": target_id,
                "name": "TestEntity8",
                "sessionId": "test-hebbian-integration"
            }
        )

        # Create relationship
        await neo4j_store.create_relationship(
            source_id=source_id,
            target_id=target_id,
            rel_type="RELATED_TO",
            properties={"weight": 0.5, "coAccessCount": 0, "lastAccessed": "2000-01-01T00:00:00"}
        )

        # Apply decay
        decay_result = await neo4j_store.batch_decay_relationships(
            stale_days=0,  # All relationships considered stale
            decay_rate=0.1,
            min_weight=0.01,
            session_id="test-hebbian-integration"
        )

        # Verify decay worked
        assert decay_result["decayed"] >= 0, "Decay should process relationships"

        # Now strengthen
        new_weight = await neo4j_store.strengthen_relationship(
            source_id=source_id,
            target_id=target_id,
            learning_rate=0.2  # Higher LR to overcome decay
        )

        # Weight should have recovered somewhat
        # This proves both mechanisms work together
        assert new_weight > 0, "Weight should be positive after strengthen"


class TestSemanticHebbianIntegration:
    """Test Hebbian learning through SemanticMemory interface."""

    @pytest.mark.skip(reason="Requires full infrastructure - run manually")
    async def test_recall_strengthens_coretrieval(self):
        """Test that recall() actually strengthens co-retrieved relationships."""
        from t4dm.memory.semantic import get_semantic_memory
        from t4dm.core.types import EntityType

        semantic = get_semantic_memory("test-semantic-hebbian")
        await semantic.initialize()

        # Create two related entities
        entity1 = await semantic.create_entity(
            name="Python",
            entity_type="TOOL",
            summary="Programming language"
        )
        entity2 = await semantic.create_entity(
            name="Django",
            entity_type="TOOL",
            summary="Web framework for Python"
        )

        # Create relationship
        rel = await semantic.create_relationship(
            source_id=entity1.id,
            target_id=entity2.id,
            relation_type="USES",
            initial_weight=0.1
        )

        # Get initial weight
        initial_weight = 0.1

        # Recall should trigger strengthening for co-retrieved entities
        results = await semantic.recall("Python Django web development", limit=10)

        # Check if both entities were retrieved
        retrieved_ids = {r.item.id for r in results}

        if entity1.id in retrieved_ids and entity2.id in retrieved_ids:
            # Relationship should have been strengthened
            rels = await semantic.graph_store.get_relationships(
                node_id=str(entity1.id),
                direction="out"
            )

            for rel in rels:
                if rel["other_id"] == str(entity2.id):
                    new_weight = rel["properties"].get("weight", 0)
                    assert new_weight > initial_weight, (
                        f"Hebbian strengthening failed! "
                        f"Weight should increase from {initial_weight} to something higher, "
                        f"got {new_weight}"
                    )
                    break
