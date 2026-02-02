"""
Integration tests for session isolation in T4DM memory services.

Tests that session_id isolation is properly enforced across:
- Episodic Memory (stored in Qdrant payloads)
- Semantic Memory (stored in Qdrant payloads)
- Procedural Memory (stored in Qdrant payloads)
- Neo4j graph relationships (sessionId property)

Each memory type must:
1. Store session_id in payloads/properties
2. Filter by session_id during recall/retrieve
3. Respect session_filter parameter overrides
4. Default to "default" session if no filter returns all data
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import UUID, uuid4
from datetime import datetime

from t4dm.memory.episodic import EpisodicMemory
from t4dm.memory.semantic import SemanticMemory
from t4dm.memory.procedural import ProceduralMemory
from t4dm.core.types import (
    Episode, Entity, EntityType, Relationship, RelationType,
    Procedure, Domain, ProcedureStep, ScoredResult, Outcome, EpisodeContext,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_embedding():
    """Mock embedding provider."""
    mock = AsyncMock()
    mock.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3] * 341 + [0.1])  # 1024-dim
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock Qdrant store."""
    mock = AsyncMock()
    mock.episodes_collection = "test_episodes"
    mock.entities_collection = "test_entities"
    mock.procedures_collection = "test_procedures"
    mock.initialize = AsyncMock()
    mock.add = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.get = AsyncMock(return_value=None)
    mock.update_payload = AsyncMock()
    return mock


@pytest.fixture
def mock_graph_store():
    """Mock Neo4j store."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.create_node = AsyncMock()
    mock.create_relationship = AsyncMock()
    mock.update_node = AsyncMock()
    mock.get_relationships = AsyncMock(return_value=[])
    mock.strengthen_relationship = AsyncMock()
    return mock


# ============================================================================
# Episodic Memory Session Isolation Tests
# ============================================================================

class TestEpisodicSessionIsolation:
    """Test episodic memory session isolation."""

    @pytest.mark.asyncio
    async def test_episodic_create_stores_session_id(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that create() stores session_id in both vector and graph stores."""
        with patch("t4dm.memory.episodic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.episodic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.episodic.get_graph_store", return_value=mock_graph_store):
                    episodic = EpisodicMemory("session-a")
                    await episodic.initialize()

                    # Create episode
                    episode = await episodic.create(
                        content="Test episode in session A",
                        context={"project": "test"},
                        outcome="success",
                        valence=0.7,
                    )

                    # Verify add() was called with correct payload
                    mock_vector_store.add.assert_called_once()
                    call_args = mock_vector_store.add.call_args
                    payloads = call_args.kwargs["payloads"]

                    # Check session_id in Qdrant payload
                    assert len(payloads) == 1
                    assert payloads[0]["session_id"] == "session-a"
                    assert payloads[0]["content"] == "Test episode in session A"

                    # Verify create_node() was called with correct properties
                    mock_graph_store.create_node.assert_called_once()
                    node_call_args = mock_graph_store.create_node.call_args
                    props = node_call_args.kwargs["properties"]

                    # Check sessionId in Neo4j properties
                    assert props["sessionId"] == "session-a"

    @pytest.mark.asyncio
    async def test_episodic_recall_filters_by_current_session(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that recall() filters by current session when session_filter is None."""
        with patch("t4dm.memory.episodic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.episodic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.episodic.get_graph_store", return_value=mock_graph_store):
                    # Setup mock search results
                    mock_vector_store.search.return_value = []

                    episodic = EpisodicMemory("session-a")
                    await episodic.initialize()

                    # Recall without explicit session_filter
                    results = await episodic.recall("test query")

                    # Verify search was called with session filter
                    mock_vector_store.search.assert_called_once()
                    search_call = mock_vector_store.search.call_args
                    search_filter = search_call.kwargs["filter"]

                    # Should filter by current session
                    assert search_filter is not None
                    assert search_filter["session_id"] == "session-a"

    @pytest.mark.asyncio
    async def test_episodic_recall_respects_explicit_session_filter(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that recall() respects explicit session_filter parameter."""
        with patch("t4dm.memory.episodic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.episodic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.episodic.get_graph_store", return_value=mock_graph_store):
                    # Setup mock search results
                    mock_vector_store.search.return_value = []

                    episodic = EpisodicMemory("session-a")
                    await episodic.initialize()

                    # Recall with explicit session filter (different session)
                    results = await episodic.recall(
                        "test query",
                        session_filter="session-b"
                    )

                    # Verify search was called with explicit session filter
                    search_call = mock_vector_store.search.call_args
                    search_filter = search_call.kwargs["filter"]

                    # Should use explicit session filter, not current session
                    assert search_filter is not None
                    assert search_filter["session_id"] == "session-b"

    @pytest.mark.asyncio
    async def test_episodic_recall_default_session_no_filter(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that recall() with 'default' session doesn't apply session filter."""
        with patch("t4dm.memory.episodic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.episodic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.episodic.get_graph_store", return_value=mock_graph_store):
                    # Setup mock search results
                    mock_vector_store.search.return_value = []

                    episodic = EpisodicMemory("default")
                    await episodic.initialize()

                    # Recall without session_filter
                    results = await episodic.recall("test query")

                    # Verify search was called without session filter
                    search_call = mock_vector_store.search.call_args
                    search_filter = search_call.kwargs["filter"]

                    # Should pass None (no filter) for 'default' session
                    assert search_filter is None

    @pytest.mark.asyncio
    async def test_episodic_session_isolation_cross_contamination(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that session A and B don't see each other's data."""
        with patch("t4dm.memory.episodic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.episodic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.episodic.get_graph_store", return_value=mock_graph_store):
                    # Create episodes in session A
                    episodic_a = EpisodicMemory("session-a")
                    await episodic_a.initialize()
                    episode_a = await episodic_a.create("Content from A")

                    # Create episodes in session B
                    episodic_b = EpisodicMemory("session-b")
                    await episodic_b.initialize()
                    episode_b = await episodic_b.create("Content from B")

                    # Verify payloads are different
                    calls = mock_vector_store.add.call_args_list
                    assert len(calls) == 2

                    # First call (session A)
                    payload_a = calls[0].kwargs["payloads"][0]
                    assert payload_a["session_id"] == "session-a"
                    assert "Content from A" in payload_a["content"]

                    # Second call (session B)
                    payload_b = calls[1].kwargs["payloads"][0]
                    assert payload_b["session_id"] == "session-b"
                    assert "Content from B" in payload_b["content"]

    @pytest.mark.asyncio
    async def test_episodic_recall_returns_scored_results(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that recall returns ScoredResult objects with proper structure."""
        with patch("t4dm.memory.episodic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.episodic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.episodic.get_graph_store", return_value=mock_graph_store):
                    # Setup mock search to return an episode
                    test_id = str(uuid4())
                    test_payload = {
                        "session_id": "session-a",
                        "content": "Test content",
                        "timestamp": datetime.now().isoformat(),
                        "ingested_at": datetime.now().isoformat(),
                        "context": {},
                        "outcome": "success",
                        "emotional_valence": 0.8,
                        "access_count": 1,  # Must be >= 1
                        "last_accessed": datetime.now().isoformat(),
                        "stability": 1.0,
                    }
                    mock_vector_store.search.return_value = [
                        (test_id, 0.85, test_payload)
                    ]
                    mock_vector_store.update_payload = AsyncMock()

                    episodic = EpisodicMemory("session-a")
                    await episodic.initialize()

                    results = await episodic.recall("test query", limit=5)

                    assert len(results) == 1
                    assert isinstance(results[0], ScoredResult)
                    assert isinstance(results[0].item, Episode)
                    assert results[0].score > 0
                    assert "semantic" in results[0].components


# ============================================================================
# Semantic Memory Session Isolation Tests
# ============================================================================

class TestSemanticSessionIsolation:
    """Test semantic memory session isolation."""

    @pytest.mark.asyncio
    async def test_semantic_create_entity_stores_session_id(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that create_entity() stores session_id in payload and properties."""
        with patch("t4dm.memory.semantic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.semantic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.semantic.get_graph_store", return_value=mock_graph_store):
                    semantic = SemanticMemory("session-a")
                    await semantic.initialize()

                    entity = await semantic.create_entity(
                        name="TestConcept",
                        entity_type="CONCEPT",
                        summary="A test concept",
                    )

                    # Verify add() was called with session_id in payload
                    mock_vector_store.add.assert_called_once()
                    call_args = mock_vector_store.add.call_args
                    payloads = call_args.kwargs["payloads"]

                    assert len(payloads) == 1
                    assert payloads[0]["session_id"] == "session-a"

                    # Verify create_node() was called with sessionId in properties
                    mock_graph_store.create_node.assert_called_once()
                    node_call_args = mock_graph_store.create_node.call_args
                    props = node_call_args.kwargs["properties"]

                    assert props["sessionId"] == "session-a"

    @pytest.mark.asyncio
    async def test_semantic_recall_filters_by_current_session(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that recall() filters by current session by default."""
        with patch("t4dm.memory.semantic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.semantic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.semantic.get_graph_store", return_value=mock_graph_store):
                    mock_vector_store.search.return_value = []
                    mock_graph_store.get_relationships = AsyncMock(return_value=[])

                    semantic = SemanticMemory("session-b")
                    await semantic.initialize()

                    results = await semantic.recall("test query")

                    # Verify search filters by session
                    search_call = mock_vector_store.search.call_args
                    search_filter = search_call.kwargs["filter"]

                    assert search_filter is not None
                    assert search_filter["session_id"] == "session-b"

    @pytest.mark.asyncio
    async def test_semantic_recall_respects_explicit_session_filter(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that recall() respects explicit session_filter parameter."""
        with patch("t4dm.memory.semantic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.semantic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.semantic.get_graph_store", return_value=mock_graph_store):
                    mock_vector_store.search.return_value = []
                    mock_graph_store.get_relationships = AsyncMock(return_value=[])

                    semantic = SemanticMemory("session-a")
                    await semantic.initialize()

                    results = await semantic.recall(
                        "test query",
                        session_filter="session-c"
                    )

                    # Verify search uses explicit filter
                    search_call = mock_vector_store.search.call_args
                    search_filter = search_call.kwargs["filter"]

                    assert search_filter is not None
                    assert search_filter["session_id"] == "session-c"

    @pytest.mark.asyncio
    async def test_semantic_recall_default_session_no_filter(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that recall() with 'default' session doesn't apply session filter."""
        with patch("t4dm.memory.semantic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.semantic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.semantic.get_graph_store", return_value=mock_graph_store):
                    mock_vector_store.search.return_value = []
                    mock_graph_store.get_relationships = AsyncMock(return_value=[])

                    semantic = SemanticMemory("default")
                    await semantic.initialize()

                    results = await semantic.recall("test query")

                    search_call = mock_vector_store.search.call_args
                    search_filter = search_call.kwargs["filter"]

                    # 'default' session should not filter by session_id
                    assert search_filter is None

    @pytest.mark.asyncio
    async def test_semantic_create_relationship_respects_session(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that relationships are created with session context."""
        with patch("t4dm.memory.semantic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.semantic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.semantic.get_graph_store", return_value=mock_graph_store):
                    semantic = SemanticMemory("session-x")
                    await semantic.initialize()

                    source_id = uuid4()
                    target_id = uuid4()

                    rel = await semantic.create_relationship(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type="USES",
                    )

                    # Verify relationship was created
                    mock_graph_store.create_relationship.assert_called_once()
                    rel_call = mock_graph_store.create_relationship.call_args

                    assert rel_call.kwargs["source_id"] == str(source_id)
                    assert rel_call.kwargs["target_id"] == str(target_id)

    @pytest.mark.asyncio
    async def test_semantic_session_isolation_entities(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that entities from different sessions are isolated."""
        with patch("t4dm.memory.semantic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.semantic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.semantic.get_graph_store", return_value=mock_graph_store):
                    # Create entity in session-1
                    semantic_1 = SemanticMemory("session-1")
                    await semantic_1.initialize()
                    entity_1 = await semantic_1.create_entity(
                        name="Concept1",
                        entity_type="CONCEPT",
                        summary="First concept",
                    )

                    # Create entity in session-2
                    semantic_2 = SemanticMemory("session-2")
                    await semantic_2.initialize()
                    entity_2 = await semantic_2.create_entity(
                        name="Concept2",
                        entity_type="CONCEPT",
                        summary="Second concept",
                    )

                    # Verify payloads have different session_ids
                    calls = mock_vector_store.add.call_args_list
                    assert len(calls) == 2

                    payload_1 = calls[0].kwargs["payloads"][0]
                    assert payload_1["session_id"] == "session-1"

                    payload_2 = calls[1].kwargs["payloads"][0]
                    assert payload_2["session_id"] == "session-2"


# ============================================================================
# Procedural Memory Session Isolation Tests
# ============================================================================

class TestProceduralSessionIsolation:
    """Test procedural memory session isolation."""

    @pytest.mark.asyncio
    async def test_procedural_build_stores_session_id(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that build() stores session_id in payload and properties."""
        with patch("t4dm.memory.procedural.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.procedural.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.procedural.get_graph_store", return_value=mock_graph_store):
                    procedural = ProceduralMemory("session-dev")
                    await procedural.initialize()

                    trajectory = [
                        {"action": "Analyze task", "tool": "read"},
                        {"action": "Write solution", "tool": "write"},
                    ]

                    procedure = await procedural.create_skill(
                        trajectory=trajectory,
                        outcome_score=0.9,
                        domain="coding",
                    )

                    # Verify add() was called with session_id in payload
                    mock_vector_store.add.assert_called_once()
                    call_args = mock_vector_store.add.call_args
                    payloads = call_args.kwargs["payloads"]

                    assert len(payloads) == 1
                    assert payloads[0]["session_id"] == "session-dev"

                    # Verify create_node() was called with sessionId
                    mock_graph_store.create_node.assert_called_once()
                    node_call_args = mock_graph_store.create_node.call_args
                    props = node_call_args.kwargs["properties"]

                    assert props["sessionId"] == "session-dev"

    @pytest.mark.asyncio
    async def test_procedural_retrieve_filters_by_current_session(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that retrieve() filters by current session by default."""
        with patch("t4dm.memory.procedural.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.procedural.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.procedural.get_graph_store", return_value=mock_graph_store):
                    mock_vector_store.search.return_value = []

                    procedural = ProceduralMemory("session-prod")
                    await procedural.initialize()

                    results = await procedural.recall_skill("Find procedures for task")

                    # Verify search filters by session
                    search_call = mock_vector_store.search.call_args
                    search_filter = search_call.kwargs["filter"]

                    assert search_filter is not None
                    assert search_filter["session_id"] == "session-prod"
                    assert search_filter["deprecated"] is False

    @pytest.mark.asyncio
    async def test_procedural_retrieve_respects_explicit_session_filter(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that retrieve() respects explicit session_filter parameter."""
        with patch("t4dm.memory.procedural.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.procedural.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.procedural.get_graph_store", return_value=mock_graph_store):
                    mock_vector_store.search.return_value = []

                    procedural = ProceduralMemory("session-current")
                    await procedural.initialize()

                    results = await procedural.recall_skill(
                        "Find procedures",
                        session_filter="session-other"
                    )

                    search_call = mock_vector_store.search.call_args
                    search_filter = search_call.kwargs["filter"]

                    assert search_filter is not None
                    assert search_filter["session_id"] == "session-other"

    @pytest.mark.asyncio
    async def test_procedural_retrieve_default_session_no_filter(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that retrieve() with 'default' session doesn't apply session filter."""
        with patch("t4dm.memory.procedural.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.procedural.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.procedural.get_graph_store", return_value=mock_graph_store):
                    mock_vector_store.search.return_value = []

                    procedural = ProceduralMemory("default")
                    await procedural.initialize()

                    results = await procedural.recall_skill("Find procedures")

                    search_call = mock_vector_store.search.call_args
                    search_filter = search_call.kwargs["filter"]

                    # 'default' session should not filter by session_id
                    assert search_filter is not None
                    # But should still filter deprecated
                    assert search_filter.get("deprecated") is False
                    assert "session_id" not in search_filter

    @pytest.mark.asyncio
    async def test_procedural_session_isolation_procedures(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that procedures from different sessions are isolated."""
        with patch("t4dm.memory.procedural.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.procedural.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.procedural.get_graph_store", return_value=mock_graph_store):
                    trajectory = [{"action": "Step 1"}]

                    # Create in session-research
                    procedural_research = ProceduralMemory("session-research")
                    await procedural_research.initialize()
                    proc_research = await procedural_research.create_skill(
                        trajectory=trajectory,
                        outcome_score=0.8,
                        domain="research",
                    )

                    # Create in session-trading
                    procedural_trading = ProceduralMemory("session-trading")
                    await procedural_trading.initialize()
                    proc_trading = await procedural_trading.create_skill(
                        trajectory=trajectory,
                        outcome_score=0.85,
                        domain="trading",
                    )

                    # Verify payloads have different session_ids
                    calls = mock_vector_store.add.call_args_list
                    assert len(calls) == 2

                    payload_research = calls[0].kwargs["payloads"][0]
                    assert payload_research["session_id"] == "session-research"
                    assert payload_research["domain"] == "research"

                    payload_trading = calls[1].kwargs["payloads"][0]
                    assert payload_trading["session_id"] == "session-trading"
                    assert payload_trading["domain"] == "trading"

    @pytest.mark.asyncio
    async def test_procedural_domain_filter_combined_with_session(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that domain filter is combined with session filter."""
        with patch("t4dm.memory.procedural.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.procedural.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.procedural.get_graph_store", return_value=mock_graph_store):
                    mock_vector_store.search.return_value = []

                    procedural = ProceduralMemory("session-dev")
                    await procedural.initialize()

                    results = await procedural.recall_skill(
                        "Find coding procedures",
                        domain="coding"
                    )

                    search_call = mock_vector_store.search.call_args
                    search_filter = search_call.kwargs["filter"]

                    # Should have both session and domain filters
                    assert search_filter is not None
                    assert search_filter["session_id"] == "session-dev"
                    assert search_filter["domain"] == "coding"
                    assert search_filter["deprecated"] is False


# ============================================================================
# Cross-Session Integration Tests
# ============================================================================

class TestCrossSessionIntegration:
    """Integration tests verifying isolation across multiple sessions simultaneously."""

    @pytest.mark.asyncio
    async def test_three_concurrent_sessions_episodic(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that three concurrent sessions don't interfere."""
        with patch("t4dm.memory.episodic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.episodic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.episodic.get_graph_store", return_value=mock_graph_store):
                    # Create three instances
                    episodic_a = EpisodicMemory("session-a")
                    episodic_b = EpisodicMemory("session-b")
                    episodic_c = EpisodicMemory("session-c")

                    await asyncio.gather(
                        episodic_a.initialize(),
                        episodic_b.initialize(),
                        episodic_c.initialize(),
                    )

                    # Create episodes concurrently
                    await asyncio.gather(
                        episodic_a.create("Content A1"),
                        episodic_b.create("Content B1"),
                        episodic_c.create("Content C1"),
                    )

                    # Verify correct session_ids in all calls
                    calls = mock_vector_store.add.call_args_list
                    assert len(calls) == 3

                    sessions_seen = set()
                    for call in calls:
                        payload = call.kwargs["payloads"][0]
                        sessions_seen.add(payload["session_id"])

                    assert sessions_seen == {"session-a", "session-b", "session-c"}

    @pytest.mark.asyncio
    async def test_three_concurrent_sessions_semantic(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test semantic memory with three concurrent sessions."""
        with patch("t4dm.memory.semantic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.semantic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.semantic.get_graph_store", return_value=mock_graph_store):
                    semantic_a = SemanticMemory("session-1")
                    semantic_b = SemanticMemory("session-2")
                    semantic_c = SemanticMemory("session-3")

                    await asyncio.gather(
                        semantic_a.initialize(),
                        semantic_b.initialize(),
                        semantic_c.initialize(),
                    )

                    await asyncio.gather(
                        semantic_a.create_entity("Entity1", "CONCEPT", "Summary1"),
                        semantic_b.create_entity("Entity2", "CONCEPT", "Summary2"),
                        semantic_c.create_entity("Entity3", "CONCEPT", "Summary3"),
                    )

                    calls = mock_vector_store.add.call_args_list
                    assert len(calls) == 3

                    sessions = {
                        calls[0].kwargs["payloads"][0]["session_id"],
                        calls[1].kwargs["payloads"][0]["session_id"],
                        calls[2].kwargs["payloads"][0]["session_id"],
                    }
                    assert sessions == {"session-1", "session-2", "session-3"}

    @pytest.mark.asyncio
    async def test_three_concurrent_sessions_procedural(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test procedural memory with three concurrent sessions."""
        with patch("t4dm.memory.procedural.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.procedural.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.procedural.get_graph_store", return_value=mock_graph_store):
                    procedural_x = ProceduralMemory("session-x")
                    procedural_y = ProceduralMemory("session-y")
                    procedural_z = ProceduralMemory("session-z")

                    await asyncio.gather(
                        procedural_x.initialize(),
                        procedural_y.initialize(),
                        procedural_z.initialize(),
                    )

                    trajectory = [{"action": "Step 1"}]

                    await asyncio.gather(
                        procedural_x.build(trajectory, 0.9, "coding"),
                        procedural_y.build(trajectory, 0.85, "research"),
                        procedural_z.build(trajectory, 0.8, "trading"),
                    )

                    calls = mock_vector_store.add.call_args_list
                    assert len(calls) == 3

                    sessions = {
                        calls[0].kwargs["payloads"][0]["session_id"],
                        calls[1].kwargs["payloads"][0]["session_id"],
                        calls[2].kwargs["payloads"][0]["session_id"],
                    }
                    assert sessions == {"session-x", "session-y", "session-z"}

    @pytest.mark.asyncio
    async def test_all_three_memory_types_respect_session_filter(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Test that all three memory types respect session_filter override (mocked)."""
        with patch("t4dm.memory.episodic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.episodic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.episodic.get_graph_store", return_value=mock_graph_store):
                    with patch("t4dm.memory.semantic.get_embedding_provider", return_value=mock_embedding):
                        with patch("t4dm.memory.semantic.get_vector_store", return_value=mock_vector_store):
                            with patch("t4dm.memory.semantic.get_graph_store", return_value=mock_graph_store):
                                with patch("t4dm.memory.procedural.get_embedding_provider", return_value=mock_embedding):
                                    with patch("t4dm.memory.procedural.get_vector_store", return_value=mock_vector_store):
                                        with patch("t4dm.memory.procedural.get_graph_store", return_value=mock_graph_store):
                                            mock_vector_store.search.return_value = []
                                            mock_graph_store.get_relationships = AsyncMock(return_value=[])

                                            episodic = EpisodicMemory("current-session")
                                            semantic = SemanticMemory("current-session")
                                            procedural = ProceduralMemory("current-session")

                                            await asyncio.gather(
                                                episodic.initialize(),
                                                semantic.initialize(),
                                                procedural.initialize(),
                                            )

                                            # All query with override to "target-session"
                                            await asyncio.gather(
                                                episodic.recall("query", session_filter="target-session"),
                                                semantic.recall("query", session_filter="target-session"),
                                                procedural.recall_skill("query", session_filter="target-session"),
                                            )

                                            # Verify all search calls used the target session
                                            calls = mock_vector_store.search.call_args_list
                                            assert len(calls) == 3

                                            for call in calls:
                                                search_filter = call.kwargs["filter"]
                                                assert search_filter is not None
                                                assert search_filter["session_id"] == "target-session"


# ============================================================================
# Payload/Property Structure Tests
# ============================================================================

class TestPayloadStructure:
    """Verify correct structure of stored payloads and properties."""

    @pytest.mark.asyncio
    async def test_episodic_payload_complete_structure(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Verify episodic payload has all required fields."""
        with patch("t4dm.memory.episodic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.episodic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.episodic.get_graph_store", return_value=mock_graph_store):
                    episodic = EpisodicMemory("test-session")
                    await episodic.initialize()

                    await episodic.create(
                        content="Test content",
                        context={"project": "test"},
                        outcome="success",
                        valence=0.7,
                    )

                    payload = mock_vector_store.add.call_args.kwargs["payloads"][0]

                    # Verify all required fields
                    required_fields = [
                        "session_id", "content", "timestamp", "ingested_at",
                        "context", "outcome", "emotional_valence", "access_count",
                        "last_accessed", "stability"
                    ]
                    for field in required_fields:
                        assert field in payload, f"Missing field: {field}"

                    # Verify types
                    assert isinstance(payload["session_id"], str)
                    assert isinstance(payload["content"], str)
                    assert isinstance(payload["emotional_valence"], float)

    @pytest.mark.asyncio
    async def test_semantic_payload_complete_structure(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Verify semantic payload has all required fields."""
        with patch("t4dm.memory.semantic.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.semantic.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.semantic.get_graph_store", return_value=mock_graph_store):
                    semantic = SemanticMemory("test-session")
                    await semantic.initialize()

                    await semantic.create_entity(
                        name="TestEntity",
                        entity_type="CONCEPT",
                        summary="Test summary",
                        details="Test details",
                    )

                    payload = mock_vector_store.add.call_args.kwargs["payloads"][0]

                    required_fields = [
                        "session_id", "name", "entity_type", "summary",
                        "details", "source", "stability", "access_count",
                        "last_accessed", "created_at", "valid_from", "valid_to"
                    ]
                    for field in required_fields:
                        assert field in payload, f"Missing field: {field}"

                    assert payload["session_id"] == "test-session"
                    assert payload["name"] == "TestEntity"

    @pytest.mark.asyncio
    async def test_procedural_payload_complete_structure(
        self, mock_embedding, mock_vector_store, mock_graph_store
    ):
        """Verify procedural payload has all required fields."""
        with patch("t4dm.memory.procedural.get_embedding_provider", return_value=mock_embedding):
            with patch("t4dm.memory.procedural.get_vector_store", return_value=mock_vector_store):
                with patch("t4dm.memory.procedural.get_graph_store", return_value=mock_graph_store):
                    procedural = ProceduralMemory("test-session")
                    await procedural.initialize()

                    trajectory = [
                        {"action": "Step 1", "tool": "tool1"},
                        {"action": "Step 2", "tool": "tool2"},
                    ]
                    await procedural.create_skill(
                        trajectory=trajectory,
                        outcome_score=0.9,
                        domain="coding",
                    )

                    payload = mock_vector_store.add.call_args.kwargs["payloads"][0]

                    required_fields = [
                        "session_id", "name", "domain", "trigger_pattern",
                        "steps", "script", "success_rate", "execution_count",
                        "last_executed", "version", "deprecated", "created_at",
                        "created_from"
                    ]
                    for field in required_fields:
                        assert field in payload, f"Missing field: {field}"

                    assert payload["session_id"] == "test-session"
                    assert payload["domain"] == "coding"
                    assert payload["deprecated"] is False
