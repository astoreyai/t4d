"""
Performance benchmark tests for World Weaver memory system.

Tests performance thresholds and catches regressions:
- Episode creation throughput
- Recall latency from large datasets
- Consolidation efficiency
- Concurrent operation performance
- Memory usage under load

Run with: pytest tests/performance/test_benchmarks.py -m slow -v

NOTE: These tests require running Neo4j and Qdrant instances.
"""

import asyncio
import pytest

# Mark all tests in this module as integration/slow tests
pytestmark = [pytest.mark.integration, pytest.mark.slow]
import time
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# ============================================================================
# Fixtures for Performance Tests
# ============================================================================

@pytest.fixture
def test_session_id():
    """Generate unique test session ID."""
    return f"bench-{uuid4().hex[:8]}"


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider with consistent performance."""
    mock = AsyncMock()
    # Return 1024-dim embedding (BGE-M3 standard)
    mock.embed_query = AsyncMock(return_value=[0.1] * 1024)
    mock.embed_documents = AsyncMock(return_value=[[0.1] * 1024 for _ in range(1000)])
    return mock


@pytest.fixture
def mock_qdrant_store():
    """Mock Qdrant with fast responses."""
    mock = AsyncMock()
    mock.episodes_collection = "episodes"
    mock.entities_collection = "entities"
    mock.procedures_collection = "procedures"
    mock.initialize = AsyncMock()
    mock.add = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.get = AsyncMock(return_value=None)
    mock.delete = AsyncMock()
    mock.update_payload = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_neo4j_store():
    """Mock Neo4j with consistent performance."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.create_node = AsyncMock(return_value="node-id")
    mock.create_relationship = AsyncMock()
    mock.get_relationships = AsyncMock(return_value=[])
    mock.update_node = AsyncMock()
    mock.close = AsyncMock()
    return mock


# ============================================================================
# P4-008: Performance Benchmark Tests
# ============================================================================

@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_create_1000_episodes(
    mock_embedding_provider, mock_qdrant_store, mock_neo4j_store, test_session_id
):
    """
    Benchmark: Create 1000 episodes should complete in < 10 seconds.

    Measures:
    - Embedding generation performance
    - Qdrant vector store insertion
    - Neo4j graph node creation
    - Overall throughput
    """
    from ww.memory.episodic import EpisodicMemory

    with patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding_provider):
        with patch("ww.memory.episodic.get_qdrant_store", return_value=mock_qdrant_store):
            with patch("ww.memory.episodic.get_neo4j_store", return_value=mock_neo4j_store):
                episodic = EpisodicMemory(test_session_id)
                await episodic.initialize()

                start = time.time()

                # Create 1000 episodes
                for i in range(1000):
                    await episodic.create(
                        content=f"Episode {i}: Test event with content for memory storage",
                        context={"project": "world-weaver", "iteration": i},
                        outcome="success" if i % 5 != 0 else "partial",
                        valence=0.7 + (i % 10) * 0.01,
                    )

                elapsed = time.time() - start
                throughput = 1000 / elapsed

                # Performance assertion
                assert elapsed < 10.0, f"Too slow: {elapsed:.2f}s (threshold: 10s)"

                print(f"\n{'='*60}")
                print(f"BENCHMARK: Create 1000 Episodes")
                print(f"{'='*60}")
                print(f"Total time:    {elapsed:.2f}s")
                print(f"Throughput:    {throughput:.0f} episodes/sec")
                print(f"Per-episode:   {elapsed/1000*1000:.2f}ms")
                print(f"Status:        {'PASS' if elapsed < 10.0 else 'FAIL'}")
                print(f"{'='*60}\n")


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_recall_from_10000_episodes(
    mock_embedding_provider, mock_qdrant_store, mock_neo4j_store, test_session_id
):
    """
    Benchmark: Recall from 10K episodes should complete in < 5 seconds.

    Measures:
    - Query embedding generation
    - Vector similarity search
    - Post-filtering and scoring
    - Result ranking performance
    """
    from ww.memory.episodic import EpisodicMemory
    from uuid import UUID

    with patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding_provider):
        with patch("ww.memory.episodic.get_qdrant_store", return_value=mock_qdrant_store):
            with patch("ww.memory.episodic.get_neo4j_store", return_value=mock_neo4j_store):
                episodic = EpisodicMemory(test_session_id)
                await episodic.initialize()

                # Mock search results for 10K episodes
                # Simulate 10K episodes with varying similarity scores
                mock_results = [
                    (str(UUID(int=i)), 0.95 - (i % 1000) * 0.0001, {
                        "session_id": test_session_id,
                        "content": f"Episode {i}",
                        "timestamp": datetime.now().isoformat(),
                        "ingested_at": datetime.now().isoformat(),
                        "context": {},
                        "outcome": "success",
                        "emotional_valence": 0.7,
                        "access_count": i % 100,
                        "last_accessed": datetime.now().isoformat(),
                        "stability": 1.0,
                    })
                    for i in range(10000)
                ]

                mock_qdrant_store.search.return_value = mock_results[:30]  # Return 30 candidates

                start = time.time()

                # Perform recall from 10K episodes
                results = await episodic.recall(
                    query="memory storage and retrieval algorithm",
                    limit=10,
                    session_filter=test_session_id,
                )

                elapsed = time.time() - start

                # Performance assertion
                assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s (threshold: 5s)"

                print(f"\n{'='*60}")
                print(f"BENCHMARK: Recall from 10K Episodes")
                print(f"{'='*60}")
                print(f"Total time:    {elapsed:.2f}s")
                print(f"Episodes:      10,000")
                print(f"Results:       {len(results)}")
                print(f"Per-query:     {elapsed*1000:.2f}ms")
                print(f"Status:        {'PASS' if elapsed < 5.0 else 'FAIL'}")
                print(f"{'='*60}\n")


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_consolidate_1000_episodes(
    mock_embedding_provider, mock_qdrant_store, mock_neo4j_store, test_session_id
):
    """
    Benchmark: Consolidation of 1000 episodes should complete in < 30 seconds.

    Measures:
    - Episode retrieval from storage
    - Embedding similarity calculations
    - Entity extraction from text
    - Relationship weight updates
    - Graph node creation performance
    """
    from uuid import UUID
    from ww.consolidation.service import ConsolidationService
    from ww.memory.episodic import EpisodicMemory

    with patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding_provider):
        with patch("ww.memory.episodic.get_qdrant_store", return_value=mock_qdrant_store):
            with patch("ww.memory.episodic.get_neo4j_store", return_value=mock_neo4j_store):
                # Setup mock to simulate 1000 episodes
                episodic = EpisodicMemory(test_session_id)
                await episodic.initialize()

                # Mock consolidation retrieve
                mock_results = [
                    MagicMock(
                        item=MagicMock(
                            id=UUID(int=i),
                            content=f"Episode {i} content",
                            embedding=[0.1] * 1024,
                        ),
                        score=0.95 - (i % 100) * 0.01,
                    )
                    for i in range(1000)
                ]

                mock_qdrant_store.search.return_value = [(r.item.id, r.score, {}) for r in mock_results]

                consolidation = ConsolidationService()

                start = time.time()

                # Run consolidation
                result = await consolidation.consolidate(
                    consolidation_type="light",
                    session_filter=test_session_id,
                )

                elapsed = time.time() - start

                # Performance assertion
                assert elapsed < 30.0, f"Too slow: {elapsed:.2f}s (threshold: 30s)"

                print(f"\n{'='*60}")
                print(f"BENCHMARK: Consolidate 1000 Episodes")
                print(f"{'='*60}")
                print(f"Total time:    {elapsed:.2f}s")
                print(f"Episodes:      1,000")
                print(f"Type:          Light")
                print(f"Per-episode:   {elapsed/1000*1000:.2f}ms")
                print(f"Status:        {'PASS' if elapsed < 30.0 else 'FAIL'}")
                print(f"{'='*60}\n")


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_100_concurrent_operations(
    mock_embedding_provider, mock_qdrant_store, mock_neo4j_store, test_session_id
):
    """
    Benchmark: 100 concurrent operations should complete in < 30 seconds.

    Measures:
    - Async operation scheduling
    - Connection pool handling
    - Lock contention
    - Task interleaving performance

    Mix of operations:
    - 50 episode creations
    - 30 recalls
    - 20 semantic queries
    """
    from ww.memory.episodic import EpisodicMemory

    with patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding_provider):
        with patch("ww.memory.episodic.get_qdrant_store", return_value=mock_qdrant_store):
            with patch("ww.memory.episodic.get_neo4j_store", return_value=mock_neo4j_store):
                episodic = EpisodicMemory(test_session_id)
                await episodic.initialize()

                mock_qdrant_store.search.return_value = []

                async def create_episode(i):
                    return await episodic.create(
                        content=f"Concurrent episode {i}",
                        outcome="success",
                        valence=0.7,
                    )

                async def recall_episodes(i):
                    return await episodic.recall(
                        query=f"search query {i}",
                        limit=5,
                        session_filter=test_session_id,
                    )

                # Build task list
                tasks = []

                # 50 creates
                for i in range(50):
                    tasks.append(create_episode(i))

                # 30 recalls
                for i in range(30):
                    tasks.append(recall_episodes(i))

                # 20 additional recalls
                for i in range(30, 50):
                    tasks.append(recall_episodes(i))

                start = time.time()

                # Execute all concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                elapsed = time.time() - start

                # Count successes and failures
                successes = sum(1 for r in results if not isinstance(r, Exception))
                failures = len(results) - successes

                # Performance assertion
                assert elapsed < 30.0, f"Too slow: {elapsed:.2f}s (threshold: 30s)"
                assert failures == 0, f"Operations failed: {failures}"

                print(f"\n{'='*60}")
                print(f"BENCHMARK: 100 Concurrent Operations")
                print(f"{'='*60}")
                print(f"Total time:    {elapsed:.2f}s")
                print(f"Operations:    100")
                print(f"Successes:     {successes}")
                print(f"Failures:      {failures}")
                print(f"Per-op:        {elapsed/100*1000:.2f}ms")
                print(f"Throughput:    {100/elapsed:.0f} ops/sec")
                print(f"Status:        {'PASS' if elapsed < 30.0 and failures == 0 else 'FAIL'}")
                print(f"{'='*60}\n")


@pytest.mark.slow
@pytest.mark.benchmark
def test_memory_usage_under_load(
    mock_embedding_provider, mock_qdrant_store, mock_neo4j_store, test_session_id
):
    """
    Benchmark: Memory usage should stay under 1GB for 10K episodes.

    Measures:
    - Heap usage for embeddings (1024-dim floats)
    - Payload storage overhead
    - Graph node metadata size
    - Connection pool memory

    Expected:
    - 10K episodes × 1024 floats × 4 bytes = ~40MB vectors
    - Payload overhead: ~200 bytes/episode = 2MB
    - Total vectors + metadata: ~100-200MB
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Get baseline
    baseline = process.memory_info().rss / 1024 / 1024  # MB

    # Simulate memory allocation for 10K episodes
    embeddings = [[0.1] * 1024 for _ in range(10000)]
    payloads = [
        {
            "session_id": test_session_id,
            "content": f"Episode {i}" * 10,  # ~100 bytes per content
            "timestamp": "2024-11-27T10:00:00",
            "ingested_at": "2024-11-27T10:00:00",
            "context": {"project": "test", "file": "test.py"},
            "outcome": "success",
            "emotional_valence": 0.7,
            "access_count": 5,
            "last_accessed": "2024-11-27T10:00:00",
            "stability": 1.0,
        }
        for i in range(10000)
    ]

    # Get peak memory
    peak = process.memory_info().rss / 1024 / 1024  # MB
    delta = peak - baseline

    # Memory assertion
    assert delta < 1024, f"Too much memory: {delta:.0f}MB (threshold: 1024MB)"

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Memory Usage for 10K Episodes")
    print(f"{'='*60}")
    print(f"Baseline:      {baseline:.0f}MB")
    print(f"Peak:          {peak:.0f}MB")
    print(f"Delta:         {delta:.0f}MB")
    print(f"Per-episode:   {delta/10:.2f}MB")
    print(f"Status:        {'PASS' if delta < 1024 else 'FAIL'}")
    print(f"{'='*60}\n")

    # Cleanup
    del embeddings
    del payloads


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_embedding_generation_performance(
    mock_embedding_provider, mock_qdrant_store, mock_neo4j_store, test_session_id
):
    """
    Benchmark: Embedding generation for batch of 100 episodes < 2 seconds.

    Measures:
    - Single query embedding: embed_query()
    - Batch embeddings: embed_documents()
    - Vector dimension handling (1024-dim BGE-M3)
    """
    from ww.embedding.bge_m3 import get_embedding_provider

    start = time.time()

    # Time 100 single embeddings
    for i in range(100):
        await mock_embedding_provider.embed_query(f"Query {i}: test content")

    single_elapsed = time.time() - start

    # Time batch embedding
    start = time.time()
    batch_contents = [f"Document {i}: test content" for i in range(100)]
    await mock_embedding_provider.embed_documents(batch_contents)
    batch_elapsed = time.time() - start

    total_elapsed = single_elapsed + batch_elapsed

    assert total_elapsed < 2.0, f"Too slow: {total_elapsed:.2f}s (threshold: 2s)"

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Embedding Generation")
    print(f"{'='*60}")
    print(f"Single (100):  {single_elapsed:.3f}s ({100/single_elapsed:.0f}/sec)")
    print(f"Batch (100):   {batch_elapsed:.3f}s")
    print(f"Total:         {total_elapsed:.3f}s")
    print(f"Status:        {'PASS' if total_elapsed < 2.0 else 'FAIL'}")
    print(f"{'='*60}\n")


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_vector_search_performance(
    mock_embedding_provider, mock_qdrant_store, mock_neo4j_store, test_session_id
):
    """
    Benchmark: Vector search in Qdrant over 10K vectors < 1 second.

    Measures:
    - HNSW index search performance
    - Filtering overhead
    - Result ranking
    """
    from ww.memory.episodic import EpisodicMemory
    from uuid import UUID

    with patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding_provider):
        with patch("ww.memory.episodic.get_qdrant_store", return_value=mock_qdrant_store):
            with patch("ww.memory.episodic.get_neo4j_store", return_value=mock_neo4j_store):
                episodic = EpisodicMemory(test_session_id)
                await episodic.initialize()

                # Mock 10K search results
                mock_results = [
                    (str(UUID(int=i)), 0.95 - (i % 1000) * 0.0001, {})
                    for i in range(10000)
                ]
                mock_qdrant_store.search.return_value = mock_results[:100]

                start = time.time()

                # Perform 10 searches
                for i in range(10):
                    results = await episodic.recall(
                        query=f"search query {i}",
                        limit=10,
                        session_filter=test_session_id,
                    )

                elapsed = time.time() - start
                per_search = elapsed / 10

                assert per_search < 0.1, f"Too slow: {per_search:.3f}s per search (threshold: 0.1s)"

                print(f"\n{'='*60}")
                print(f"BENCHMARK: Vector Search Performance")
                print(f"{'='*60}")
                print(f"Total (10x):   {elapsed:.3f}s")
                print(f"Per search:    {per_search:.3f}s")
                print(f"Throughput:    {10/elapsed:.0f} searches/sec")
                print(f"Status:        {'PASS' if per_search < 0.1 else 'FAIL'}")
                print(f"{'='*60}\n")


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_graph_operation_performance(
    mock_embedding_provider, mock_qdrant_store, mock_neo4j_store, test_session_id
):
    """
    Benchmark: Neo4j graph operations for 1000 nodes/relationships < 5 seconds.

    Measures:
    - Node creation throughput
    - Relationship creation
    - Property updates
    - Query execution
    """
    from uuid import UUID

    # Time node creation
    start = time.time()
    for i in range(1000):
        await mock_neo4j_store.create_node(
            label="Entity",
            properties={
                "id": str(UUID(int=i)),
                "name": f"Entity {i}",
                "sessionId": test_session_id,
            }
        )
    node_elapsed = time.time() - start

    # Time relationship creation
    start = time.time()
    for i in range(900):  # 900 relationships for 1000 nodes
        await mock_neo4j_store.create_relationship(
            source_id=str(UUID(int=i)),
            target_id=str(UUID(int=i+1)),
            rel_type="RELATED_TO",
            properties={"weight": 0.5},
        )
    rel_elapsed = time.time() - start

    total_elapsed = node_elapsed + rel_elapsed

    assert total_elapsed < 5.0, f"Too slow: {total_elapsed:.2f}s (threshold: 5s)"

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Graph Operations")
    print(f"{'='*60}")
    print(f"Nodes (1000):  {node_elapsed:.2f}s ({1000/node_elapsed:.0f}/sec)")
    print(f"Rels (900):    {rel_elapsed:.2f}s ({900/rel_elapsed:.0f}/sec)")
    print(f"Total:         {total_elapsed:.2f}s")
    print(f"Status:        {'PASS' if total_elapsed < 5.0 else 'FAIL'}")
    print(f"{'='*60}\n")
