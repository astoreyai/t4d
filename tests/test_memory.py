"""
TMEM-001: Test episodic memory creation and retrieval.

Tests the core memory functionality of World Weaver.

NOTE: These tests require running Neo4j and Qdrant instances.
"""

import asyncio
import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration
from datetime import datetime


@pytest.mark.asyncio
async def test_episodic_memory_create():
    """Test creating and retrieving episodic memories."""
    from ww.memory.episodic import get_episodic_memory

    # Initialize episodic memory with test session
    episodic = get_episodic_memory("test-session-001")
    await episodic.initialize()

    # Create an episode
    episode = await episodic.create(
        content="Implemented the tripartite memory system for World Weaver. "
                "Created episodic, semantic, and procedural memory services.",
        context={
            "project": "world-weaver",
            "file": "src/ww/memory/episodic.py",
            "tool": "Write",
        },
        outcome="success",
        valence=0.8,
    )

    assert episode is not None
    assert episode.id is not None
    assert episode.session_id == "test-session-001"
    assert episode.content.startswith("Implemented")
    assert episode.outcome.value == "success"
    assert episode.emotional_valence == 0.8

    print(f"Created episode: {episode.id}")
    print(f"Retrievability: {episode.retrievability():.4f}")

    return episode


@pytest.mark.asyncio
async def test_episodic_memory_recall():
    """Test recalling episodic memories by semantic similarity."""
    from ww.memory.episodic import get_episodic_memory

    episodic = get_episodic_memory("test-session-001")
    await episodic.initialize()

    # Recall episodes about memory systems
    results = await episodic.recall(
        query="memory system implementation",
        limit=5,
    )

    assert isinstance(results, list)

    print(f"Found {len(results)} episodes")
    for r in results:
        print(f"  Score: {r.score:.4f} - {r.item.content[:60]}...")
        print(f"    Components: {r.components}")

    return results


@pytest.mark.asyncio
async def test_semantic_memory_create():
    """Test creating semantic entities and relationships."""
    from ww.memory.semantic import get_semantic_memory

    semantic = get_semantic_memory()
    await semantic.initialize()

    # Create a concept entity
    entity = await semantic.create_entity(
        name="FSRS",
        entity_type="TECHNIQUE",
        summary="Free Spaced Repetition Scheduler algorithm for memory decay",
        details="FSRS models memory retrievability as R(t,S) = (1 + 0.9*t/S)^(-0.5)",
        source="manual",
    )

    assert entity is not None
    assert entity.name == "FSRS"
    assert entity.entity_type.value == "TECHNIQUE"

    print(f"Created entity: {entity.id} - {entity.name}")

    return entity


@pytest.mark.asyncio
async def test_semantic_recall_with_activation():
    """Test ACT-R activation-based retrieval."""
    from ww.memory.semantic import get_semantic_memory

    semantic = get_semantic_memory()
    await semantic.initialize()

    results = await semantic.recall(
        query="memory decay algorithms",
        limit=5,
    )

    print(f"Found {len(results)} entities")
    for r in results:
        print(f"  Score: {r.score:.4f} - {r.item.name}")
        print(f"    Components: {r.components}")

    return results


@pytest.mark.asyncio
async def test_procedural_memory_build():
    """Test building procedural memories from trajectories."""
    from ww.memory.procedural import get_procedural_memory

    procedural = get_procedural_memory()
    await procedural.initialize()

    # Build a skill from a successful trajectory
    trajectory = [
        {
            "action": "Read existing code file",
            "tool": "Read",
            "parameters": {"file_path": "/src/module.py"},
            "result": "Got file contents",
        },
        {
            "action": "Edit function to add parameter",
            "tool": "Edit",
            "parameters": {"old_string": "def foo():", "new_string": "def foo(x):"},
            "result": "Function updated",
        },
        {
            "action": "Run tests to verify",
            "tool": "Bash",
            "parameters": {"command": "pytest"},
            "result": "All tests passed",
        },
    ]

    procedure = await procedural.create_skill(
        trajectory=trajectory,
        outcome_score=0.9,
        domain="coding",
        trigger_pattern="add parameter to function",
    )

    assert procedure is not None
    assert procedure.domain.value == "coding"
    assert len(procedure.steps) == 3

    print(f"Built procedure: {procedure.id} - {procedure.name}")
    print(f"Steps: {len(procedure.steps)}")
    print(f"Script:\n{procedure.script}")

    return procedure


@pytest.mark.asyncio
async def test_procedural_retrieve():
    """Test retrieving procedures by task similarity."""
    from ww.memory.procedural import get_procedural_memory

    procedural = get_procedural_memory()
    await procedural.initialize()

    results = await procedural.recall_skill(
        task="I need to add a new argument to an existing function",
        domain="coding",
        limit=3,
    )

    print(f"Found {len(results)} procedures")
    for r in results:
        print(f"  Score: {r.score:.4f} - {r.item.name}")
        print(f"    Success rate: {r.item.success_rate}")

    return results


async def run_all_tests():
    """Run all memory tests."""
    print("=" * 60)
    print("TMEM-001: World Weaver Memory System Tests")
    print("=" * 60)

    print("\n1. Testing episodic memory creation...")
    episode = await test_episodic_memory_create()

    print("\n2. Testing episodic memory recall...")
    await test_episodic_memory_recall()

    print("\n3. Testing semantic memory creation...")
    entity = await test_semantic_memory_create()

    print("\n4. Testing semantic recall with ACT-R activation...")
    await test_semantic_recall_with_activation()

    print("\n5. Testing procedural memory build...")
    procedure = await test_procedural_memory_build()

    print("\n6. Testing procedural memory retrieval...")
    await test_procedural_retrieve()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
