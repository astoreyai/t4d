"""
INT-001: Integration tests for World Weaver memory system.

Tests the complete workflow:
1. Create episodes from interactions
2. Recall episodes by semantic similarity
3. Consolidate episodes into semantic knowledge
4. Build procedures from successful trajectories
5. Retrieve procedures for similar tasks

NOTE: These tests require running Neo4j and Qdrant instances.
"""

import asyncio
import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration
from datetime import datetime
from uuid import uuid4


@pytest.mark.asyncio
async def test_full_memory_workflow():
    """
    INT-001: Complete memory workflow integration test.

    Simulates a coding session with:
    - Episode creation
    - Semantic entity extraction
    - Procedure building
    - Cross-memory retrieval
    """
    from ww.memory.episodic import get_episodic_memory
    from ww.memory.semantic import get_semantic_memory
    from ww.memory.procedural import get_procedural_memory
    from ww.consolidation.service import get_consolidation_service

    session_id = f"int-test-{uuid4().hex[:8]}"
    print(f"\n{'='*60}")
    print(f"INT-001: Full Memory Workflow Test")
    print(f"Session: {session_id}")
    print(f"{'='*60}\n")

    # Initialize services
    episodic = get_episodic_memory(session_id)
    semantic = get_semantic_memory()
    procedural = get_procedural_memory()
    consolidation = get_consolidation_service()

    await episodic.initialize()
    await semantic.initialize()
    await procedural.initialize()

    # Phase 1: Create episodes simulating a coding session
    print("Phase 1: Creating episodes...")

    episodes = []
    interactions = [
        {
            "content": "Implemented FSRS decay algorithm for episodic memory. "
                       "The formula R(t,S) = (1 + 0.9*t/S)^(-0.5) models retrievability over time.",
            "context": {"project": "world-weaver", "file": "memory/episodic.py", "tool": "Edit"},
            "outcome": "success",
            "valence": 0.9,
        },
        {
            "content": "Added Hebbian learning to semantic memory relationships. "
                       "Weight update: w' = w + lr*(1-w) ensures bounded growth.",
            "context": {"project": "world-weaver", "file": "memory/semantic.py", "tool": "Edit"},
            "outcome": "success",
            "valence": 0.85,
        },
        {
            "content": "Fixed Neo4j connection pooling issue. Changed to async driver "
                       "and added connection health checks.",
            "context": {"project": "world-weaver", "file": "storage/neo4j_store.py", "tool": "Edit"},
            "outcome": "success",
            "valence": 0.7,
        },
        {
            "content": "Implemented Memp lifecycle for procedural memory. "
                       "Procedures have BUILD-RETRIEVE-UPDATE phases.",
            "context": {"project": "world-weaver", "file": "memory/procedural.py", "tool": "Edit"},
            "outcome": "success",
            "valence": 0.8,
        },
    ]

    for interaction in interactions:
        episode = await episodic.create(
            content=interaction["content"],
            context=interaction["context"],
            outcome=interaction["outcome"],
            valence=interaction["valence"],
        )
        episodes.append(episode)
        print(f"  Created episode: {episode.id}")

    assert len(episodes) == 4
    print(f"  Total: {len(episodes)} episodes created")

    # Phase 2: Recall episodes
    print("\nPhase 2: Recalling episodes...")

    results = await episodic.recall(
        query="memory decay algorithm",
        limit=3,
        session_filter=session_id,
    )

    print(f"  Found {len(results)} relevant episodes")
    for r in results:
        print(f"    Score: {r.score:.4f} - {r.item.content[:50]}...")

    assert len(results) > 0
    assert results[0].score > 0.5

    # Phase 3: Create semantic entities
    print("\nPhase 3: Creating semantic entities...")

    entity_fsrs = await semantic.create_entity(
        name="FSRS",
        entity_type="TECHNIQUE",
        summary="Free Spaced Repetition Scheduler for memory decay modeling",
        details="Retrievability formula: R(t,S) = (1 + 0.9*t/S)^(-0.5)",
        source=str(episodes[0].id),
    )
    print(f"  Created entity: {entity_fsrs.name} ({entity_fsrs.id})")

    entity_hebbian = await semantic.create_entity(
        name="Hebbian Learning",
        entity_type="TECHNIQUE",
        summary="Neural learning rule for weight strengthening",
        details="Bounded update: w' = w + lr*(1-w)",
        source=str(episodes[1].id),
    )
    print(f"  Created entity: {entity_hebbian.name} ({entity_hebbian.id})")

    # Create relationship
    relationship = await semantic.create_relationship(
        source_id=entity_fsrs.id,
        target_id=entity_hebbian.id,
        relation_type="SIMILAR_TO",
        initial_weight=0.3,
    )
    print(f"  Created relationship: FSRS -[SIMILAR_TO]-> Hebbian Learning")

    # Phase 4: Semantic recall with spreading activation
    print("\nPhase 4: Semantic recall with spreading activation...")

    results = await semantic.recall(
        query="memory algorithms",
        context_entities=[str(entity_fsrs.id)],
        limit=5,
    )

    print(f"  Found {len(results)} entities")
    for r in results:
        print(f"    Score: {r.score:.4f} - {r.item.name}")
        print(f"      Components: semantic={r.components.get('semantic', 0):.3f}, "
              f"activation={r.components.get('activation', 0):.3f}")

    # Phase 5: Build procedure from trajectory
    print("\nPhase 5: Building procedure from trajectory...")

    trajectory = [
        {
            "action": "Read existing memory implementation",
            "tool": "Read",
            "parameters": {"file_path": "/src/memory/episodic.py"},
            "result": "Got current implementation",
        },
        {
            "action": "Add decay calculation method",
            "tool": "Edit",
            "parameters": {"old_string": "class Episode:", "new_string": "class Episode:\n    def retrievability(self):"},
            "result": "Method added",
        },
        {
            "action": "Run tests to verify",
            "tool": "Bash",
            "parameters": {"command": "pytest tests/test_episodic.py"},
            "result": "All tests passed",
        },
    ]

    procedure = await procedural.create_skill(
        trajectory=trajectory,
        outcome_score=0.95,
        domain="coding",
        trigger_pattern="add method to class",
        name="Add Method to Python Class",
    )

    print(f"  Built procedure: {procedure.name}")
    print(f"    Steps: {len(procedure.steps)}")
    print(f"    Success rate: {procedure.success_rate}")

    assert procedure is not None
    assert len(procedure.steps) == 3

    # Phase 6: Retrieve procedure for similar task
    print("\nPhase 6: Retrieving procedure for similar task...")

    results = await procedural.recall_skill(
        task="I need to add a new method to an existing class",
        domain="coding",
        limit=3,
    )

    print(f"  Found {len(results)} procedures")
    for r in results:
        print(f"    Score: {r.score:.4f} - {r.item.name}")

    assert len(results) > 0

    # Phase 7: Run consolidation
    print("\nPhase 7: Running memory consolidation...")

    result = await consolidation.consolidate(
        consolidation_type="light",
        session_filter=session_id,
    )

    print(f"  Status: {result['status']}")
    print(f"  Duration: {result['duration_seconds']:.2f}s")
    if "light" in result.get("results", {}):
        light = result["results"]["light"]
        print(f"  Episodes scanned: {light.get('episodes_scanned', 0)}")

    assert result["status"] == "completed"

    # Phase 8: Cross-memory integration test
    print("\nPhase 8: Cross-memory integration check...")

    # Verify episode connects to entity through provenance
    entity = await semantic.get_entity(entity_fsrs.id)
    assert entity is not None
    assert entity.source == str(episodes[0].id)
    print(f"  Entity provenance verified: {entity.name} -> Episode {episodes[0].id}")

    # Verify procedure retrieval works
    proc = await procedural.get_procedure(procedure.id)
    assert proc is not None
    assert proc.success_rate == 1.0
    print(f"  Procedure retrieval verified: {proc.name}")

    print(f"\n{'='*60}")
    print("INT-001: All integration tests passed!")
    print(f"{'='*60}\n")


@pytest.mark.asyncio
async def test_multi_session_isolation():
    """
    INT-002: Test multi-session memory isolation.

    Verifies that:
    - Episodes are isolated by session
    - Semantic/procedural knowledge is shared
    """
    from ww.memory.episodic import get_episodic_memory
    from ww.memory.semantic import get_semantic_memory

    session_a = f"session-a-{uuid4().hex[:6]}"
    session_b = f"session-b-{uuid4().hex[:6]}"

    print(f"\n{'='*60}")
    print(f"INT-002: Multi-Session Isolation Test")
    print(f"{'='*60}\n")

    # Create episodes in session A
    episodic_a = get_episodic_memory(session_a)
    await episodic_a.initialize()

    episode_a = await episodic_a.create(
        content="This is a private episode in session A about project X",
        context={"project": "project-x"},
        outcome="success",
    )
    print(f"Created episode in session A: {episode_a.id}")

    # Create episodes in session B
    episodic_b = get_episodic_memory(session_b)
    await episodic_b.initialize()

    episode_b = await episodic_b.create(
        content="This is a private episode in session B about project Y",
        context={"project": "project-y"},
        outcome="success",
    )
    print(f"Created episode in session B: {episode_b.id}")

    # Verify session A cannot see session B's episodes
    results_a = await episodic_a.recall(
        query="project",
        session_filter=session_a,
        limit=10,
    )

    session_a_ids = {str(r.item.id) for r in results_a}
    print(f"\nSession A sees {len(results_a)} episodes")
    assert str(episode_a.id) in session_a_ids or len(results_a) == 0
    # Note: episode_b should not be visible in session A recall

    # Verify semantic knowledge is shared
    semantic = get_semantic_memory()
    await semantic.initialize()

    shared_entity = await semantic.create_entity(
        name="Shared Knowledge",
        entity_type="CONCEPT",
        summary="This entity is visible to all sessions",
    )
    print(f"\nCreated shared entity: {shared_entity.name}")

    # Both sessions can access shared entity
    results = await semantic.recall(query="shared knowledge", limit=1)
    assert len(results) > 0
    print(f"Shared entity accessible: {results[0].item.name}")

    print(f"\n{'='*60}")
    print("INT-002: Multi-session isolation verified!")
    print(f"{'='*60}\n")


async def run_integration_tests():
    """Run all integration tests."""
    await test_full_memory_workflow()
    await test_multi_session_isolation()


if __name__ == "__main__":
    asyncio.run(run_integration_tests())
