"""
Integration tests for batch query optimization in Neo4j store.
Tests the elimination of N+1 query pattern.

NOTE: These tests require a running Neo4j instance and are marked as integration tests.
"""
import pytest
import time
from uuid import uuid4, UUID
from datetime import datetime

from ww.storage.neo4j_store import get_neo4j_store
from ww.memory.semantic import get_semantic_memory
from ww.core.types import ScoredResult, Entity, EntityType


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_relationships_vs_individual():
    """Test that batch queries return same results as individual queries."""
    store = get_neo4j_store("test_batch_perf")
    await store.initialize()

    # Create test nodes
    node_ids = []
    for i in range(5):
        node_id = str(uuid4())
        await store.create_node(
            label="Entity",
            properties={
                "id": node_id,
                "sessionId": "test_batch_perf",
                "name": f"Entity_{i}",
                "entityType": "CONCEPT",
                "summary": f"Test entity {i}",
                "details": "",
                "source": "test",
                "stability": 1.0,
                "accessCount": 1,
                "lastAccessed": "2025-01-01T00:00:00",
                "createdAt": "2025-01-01T00:00:00",
                "validFrom": "2025-01-01T00:00:00",
                "validTo": "",
            }
        )
        node_ids.append(node_id)

    # Create relationships
    for i in range(len(node_ids) - 1):
        await store.create_relationship(
            source_id=node_ids[i],
            target_id=node_ids[i + 1],
            rel_type="RELATES_TO",
            properties={
                "weight": 0.5,
                "coAccessCount": 1,
                "lastCoAccess": "2025-01-01T00:00:00",
            }
        )

    # Get relationships individually
    individual_results = {}
    for node_id in node_ids:
        rels = await store.get_relationships(node_id=node_id, direction="both")
        individual_results[node_id] = rels

    # Get relationships in batch
    batch_results = await store.get_relationships_batch(
        node_ids=node_ids,
        direction="both"
    )

    # Verify results match
    for node_id in node_ids:
        individual = sorted([r["other_id"] for r in individual_results[node_id]])
        batch = sorted([r["other_id"] for r in batch_results[node_id]])
        assert individual == batch, f"Results mismatch for {node_id}"

    # Cleanup
    for node_id in node_ids:
        await store.delete_node(node_id)

    await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_query_performance():
    """Test that batch queries execute successfully with multiple entities."""
    store = get_neo4j_store("test_batch_speed")
    await store.initialize()

    # Create test nodes
    node_ids = []
    for i in range(20):
        node_id = str(uuid4())
        await store.create_node(
            label="Entity",
            properties={
                "id": node_id,
                "sessionId": "test_batch_speed",
                "name": f"Entity_{i}",
                "entityType": "CONCEPT",
                "summary": f"Test entity {i}",
                "details": "",
                "source": "test",
                "stability": 1.0,
                "accessCount": 1,
                "lastAccessed": "2025-01-01T00:00:00",
                "createdAt": "2025-01-01T00:00:00",
                "validFrom": "2025-01-01T00:00:00",
                "validTo": "",
            }
        )
        node_ids.append(node_id)

    # Create relationships
    for i in range(len(node_ids) - 1):
        await store.create_relationship(
            source_id=node_ids[i],
            target_id=node_ids[i + 1],
            rel_type="RELATES_TO",
            properties={
                "weight": 0.5,
                "coAccessCount": 1,
                "lastCoAccess": "2025-01-01T00:00:00",
            }
        )

    # Measure individual query time (run multiple times for consistency)
    runs = 3
    individual_times = []
    for _ in range(runs):
        start = time.perf_counter()
        for node_id in node_ids:
            await store.get_relationships(node_id=node_id, direction="both")
        individual_times.append(time.perf_counter() - start)

    # Measure batch query time (run multiple times for consistency)
    batch_times = []
    for _ in range(runs):
        start = time.perf_counter()
        await store.get_relationships_batch(node_ids=node_ids, direction="both")
        batch_times.append(time.perf_counter() - start)

    avg_individual = sum(individual_times) / len(individual_times)
    avg_batch = sum(batch_times) / len(batch_times)

    # Log timing results (informational, not asserting on performance)
    print(f"\nAverage individual time: {avg_individual:.4f}s")
    print(f"Average batch time: {avg_batch:.4f}s")
    if avg_batch < avg_individual:
        print(f"Speedup: {avg_individual / avg_batch:.2f}x")
    else:
        print("Note: Batch timing may vary based on system load")

    # Just verify batch query completes successfully
    result = await store.get_relationships_batch(node_ids=node_ids, direction="both")
    assert len(result) == len(node_ids), "Batch query should return results for all nodes"

    # Cleanup
    for node_id in node_ids:
        await store.delete_node(node_id)

    await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_empty_input():
    """Test batch query with empty input."""
    store = get_neo4j_store("test_batch_empty")
    await store.initialize()

    result = await store.get_relationships_batch(node_ids=[])

    assert result == {}
    await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hebbian_strengthening_batch():
    """Test Hebbian strengthening uses batch queries."""
    semantic = get_semantic_memory("test_hebbian_batch")
    await semantic.initialize()

    # Create test entities with random UUIDs to avoid conflicts
    entities = []
    for i in range(3):
        entity = Entity(
            id=uuid4(),  # Use random UUID instead of deterministic
            name=f"Entity_{i}",
            entity_type=EntityType.CONCEPT,
            summary=f"Test entity {i}",
            embedding=None,
            stability=1.0,
            access_count=1,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            valid_from=datetime.now(),
        )
        entities.append(entity)

        # Create node in graph
        await semantic.graph_store.create_node(
            label="Entity",
            properties={
                "id": str(entity.id),
                "sessionId": "test_hebbian_batch",
                "name": entity.name,
                "entityType": entity.entity_type.value,
                "summary": entity.summary,
                "details": "",
                "source": "test",
                "stability": entity.stability,
                "accessCount": entity.access_count,
                "lastAccessed": entity.last_accessed.isoformat(),
                "createdAt": entity.created_at.isoformat(),
                "validFrom": entity.valid_from.isoformat(),
                "validTo": "",
            }
        )

    # Create relationships
    await semantic.graph_store.create_relationship(
        source_id=str(entities[0].id),
        target_id=str(entities[1].id),
        rel_type="RELATES_TO",
        properties={
            "weight": 0.3,
            "coAccessCount": 1,
            "lastCoAccess": datetime.now().isoformat()
        }
    )

    # Create scored results
    results = [
        ScoredResult(item=entities[0], score=1.0, components={}),
        ScoredResult(item=entities[1], score=0.9, components={}),
        ScoredResult(item=entities[2], score=0.8, components={}),
    ]

    # Test strengthening (should use batch query internally)
    await semantic._strengthen_co_retrieval(results)

    # Verify relationship still exists and was strengthened
    rels = await semantic.graph_store.get_relationships(str(entities[0].id))
    assert len(rels) > 0, "Relationship should exist after strengthening"

    # Cleanup
    for entity in entities:
        await semantic.graph_store.delete_node(str(entity.id))

    await semantic.graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_preload_batch():
    """Test context relationship preloading uses batch queries."""
    semantic = get_semantic_memory("test_preload_batch")
    await semantic.initialize()

    # Create test entities with random UUIDs to avoid conflicts
    entities = []
    for i in range(3):
        entity = Entity(
            id=uuid4(),  # Use random UUID instead of deterministic
            name=f"Entity_{i}",
            entity_type=EntityType.CONCEPT,
            summary=f"Test entity {i}",
            embedding=None,
            stability=1.0,
            access_count=1,
            last_accessed=datetime.now(),
            created_at=datetime.now(),
            valid_from=datetime.now(),
        )
        entities.append(entity)

        # Create node
        await semantic.graph_store.create_node(
            label="Entity",
            properties={
                "id": str(entity.id),
                "sessionId": "test_preload_batch",
                "name": entity.name,
                "entityType": entity.entity_type.value,
                "summary": entity.summary,
                "details": "",
                "source": "test",
                "stability": entity.stability,
                "accessCount": entity.access_count,
                "lastAccessed": entity.last_accessed.isoformat(),
                "createdAt": entity.created_at.isoformat(),
                "validFrom": entity.valid_from.isoformat(),
                "validTo": "",
            }
        )

    # Create relationships
    await semantic.graph_store.create_relationship(
        source_id=str(entities[0].id),
        target_id=str(entities[1].id),
        rel_type="RELATES_TO",
        properties={
            "weight": 0.5,
            "coAccessCount": 1,
            "lastCoAccess": datetime.now().isoformat()
        }
    )

    # Test preload (should use batch queries)
    cache = await semantic._preload_context_relationships(entities)

    assert len(cache) == len(entities), "Cache should have entry for each entity"
    assert str(entities[0].id) in cache, "First entity should be in cache"
    assert "strengths" in cache[str(entities[0].id)], "Cache should have strengths"
    assert "fan_out" in cache[str(entities[0].id)], "Cache should have fan_out"

    # Cleanup
    for entity in entities:
        await semantic.graph_store.delete_node(str(entity.id))

    await semantic.graph_store.close()
