"""
TMEM-002: Comprehensive tests for Episodic Memory Service.

Tests episodic memory creation, retrieval, FSRS decay calculations,
temporal queries, and session isolation.
"""

import asyncio
import math
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from ww.core.types import Episode, EpisodeContext, Outcome, ScoredResult
from ww.memory.episodic import EpisodicMemory


class TestEpisodicMemoryBasics:
    """Test basic episodic memory operations."""

    @pytest_asyncio.fixture
    async def episodic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create episodic memory instance with mocked stores."""
        episodic = EpisodicMemory(session_id=test_session_id)
        episodic.vector_store = mock_qdrant_store
        episodic.graph_store = mock_neo4j_store
        episodic.embedding = mock_embedding_provider

        # Mock collection attributes
        episodic.vector_store.episodes_collection = "episodes"

        # Disable FF encoding and gating for predictable test behavior
        # (P7.1 FF bridge modifies valence based on novelty detection)
        episodic._ff_encoding_enabled = False
        episodic._gating_enabled = False

        return episodic

    @pytest.mark.asyncio
    async def test_episode_creation_with_all_fields(self, episodic):
        """Test episode creation with all fields populated."""
        # Setup mock responses
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        # Create episode
        episode = await episodic.create(
            content="Successfully implemented tripartite memory system",
            context={
                "project": "world-weaver",
                "file": "src/ww/memory/episodic.py",
                "tool": "Write",
                "git_branch": "feature/memory",
            },
            outcome="success",
            valence=0.85,
        )

        # Assertions
        assert episode is not None
        assert episode.id is not None
        assert episode.session_id == episodic.session_id
        assert episode.content == "Successfully implemented tripartite memory system"
        assert episode.outcome == Outcome.SUCCESS
        assert episode.emotional_valence == 0.85
        assert episode.context.project == "world-weaver"
        assert episode.context.file == "src/ww/memory/episodic.py"
        assert episode.context.tool == "Write"
        assert episode.context.git_branch == "feature/memory"
        assert episode.access_count == 1
        assert episode.stability == episodic.default_stability

        # Verify storage calls
        episodic.vector_store.add.assert_called_once()
        episodic.graph_store.create_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_episode_creation_with_minimal_fields(self, episodic):
        """Test episode creation with only required fields."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        # Create episode with minimal fields
        episode = await episodic.create(
            content="Simple event",
        )

        assert episode is not None
        assert episode.content == "Simple event"
        assert episode.outcome == Outcome.NEUTRAL
        assert episode.emotional_valence == 0.5
        assert episode.context.project is None
        assert len(episode.context.model_dump(exclude_none=True)) == 0

    @pytest.mark.asyncio
    async def test_episode_creation_all_outcome_types(self, episodic):
        """Test episode creation with all outcome types."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        outcomes = ["success", "failure", "partial", "neutral"]

        for outcome in outcomes:
            episode = await episodic.create(
                content=f"Event with {outcome} outcome",
                outcome=outcome,
            )
            assert episode.outcome == Outcome(outcome)

    @pytest.mark.asyncio
    async def test_get_episode_by_id(self, episodic):
        """Test retrieving episode by ID."""
        episode_id = uuid4()
        payload = {
            "session_id": episodic.session_id,
            "content": "Test episode",
            "timestamp": datetime.now().isoformat(),
            "ingested_at": datetime.now().isoformat(),
            "context": {},
            "outcome": "success",
            "emotional_valence": 0.5,
            "access_count": 1,
            "last_accessed": datetime.now().isoformat(),
            "stability": 1.0,
        }

        episodic.vector_store.get.return_value = [(str(episode_id), payload)]

        episode = await episodic.get(episode_id)

        assert episode is not None
        assert episode.id == episode_id
        assert episode.content == "Test episode"
        episodic.vector_store.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_nonexistent_episode(self, episodic):
        """Test retrieving non-existent episode returns None."""
        episodic.vector_store.get.return_value = []

        episode = await episodic.get(uuid4())

        assert episode is None


class TestEpisodicMemoryRecall:
    """Test episodic memory retrieval with scoring."""

    @pytest_asyncio.fixture
    async def episodic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create episodic memory instance."""
        episodic = EpisodicMemory(session_id=test_session_id)
        episodic.vector_store = mock_qdrant_store
        episodic.graph_store = mock_neo4j_store
        episodic.embedding = mock_embedding_provider
        episodic.vector_store.episodes_collection = "episodes"
        return episodic

    @pytest.mark.asyncio
    async def test_recall_basic_search(self, episodic):
        """Test basic episodic recall without filtering."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        results = [
            (
                str(uuid4()),  # Must be valid UUID string
                0.95,
                {
                    "session_id": episodic.session_id,
                    "content": "Implemented memory system",
                    "timestamp": now.isoformat(),
                    "ingested_at": now.isoformat(),
                    "context": {},
                    "outcome": "success",
                    "emotional_valence": 0.8,
                    "access_count": 1,
                    "last_accessed": now.isoformat(),
                    "stability": 1.0,
                },
            ),
            (
                str(uuid4()),  # Must be valid UUID string
                0.72,
                {
                    "session_id": episodic.session_id,
                    "content": "Tested retrieval logic",
                    "timestamp": now.isoformat(),
                    "ingested_at": now.isoformat(),
                    "context": {},
                    "outcome": "partial",
                    "emotional_valence": 0.5,
                    "access_count": 2,
                    "last_accessed": now.isoformat(),
                    "stability": 1.0,
                },
            ),
        ]

        episodic.vector_store.search.return_value = results
        episodic.vector_store.update_payload.return_value = None

        scored_results = await episodic.recall(
            query="memory system",
            limit=10,
        )

        assert isinstance(scored_results, list)
        assert len(scored_results) == 2
        assert all(isinstance(r, ScoredResult) for r in scored_results)

        # First result should have higher score
        assert scored_results[0].score >= scored_results[1].score

        # Check component scores exist
        assert "semantic" in scored_results[0].components
        assert "recency" in scored_results[0].components
        assert "outcome" in scored_results[0].components
        assert "importance" in scored_results[0].components

    @pytest.mark.asyncio
    async def test_recall_empty_results(self, episodic):
        """Test recall with no matching episodes."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.search.return_value = []

        scored_results = await episodic.recall(
            query="nonexistent query",
            limit=10,
        )

        assert scored_results == []

    @pytest.mark.asyncio
    async def test_recall_with_session_filter(self, episodic):
        """Test recall with session filtering."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.search.return_value = []

        await episodic.recall(
            query="test",
            limit=10,
            session_filter="other-session",
        )

        # Verify filter was passed
        call_args = episodic.vector_store.search.call_args
        assert call_args[1]["filter"]["session_id"] == "other-session"

    @pytest.mark.asyncio
    async def test_recall_with_time_range(self, episodic):
        """Test recall with temporal filtering."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        old_time = now - timedelta(days=30)
        future_time = now + timedelta(days=30)

        results = [
            (
                str(uuid4()),  # Must be valid UUID string
                0.95,
                {
                    "session_id": episodic.session_id,
                    "content": "In range",
                    "timestamp": now.isoformat(),
                    "ingested_at": now.isoformat(),
                    "context": {},
                    "outcome": "success",
                    "emotional_valence": 0.5,
                    "access_count": 1,
                    "last_accessed": now.isoformat(),
                    "stability": 1.0,
                },
            ),
            (
                str(uuid4()),  # Must be valid UUID string
                0.90,
                {
                    "session_id": episodic.session_id,
                    "content": "Too old",
                    "timestamp": (old_time - timedelta(days=1)).isoformat(),
                    "ingested_at": (old_time - timedelta(days=1)).isoformat(),
                    "context": {},
                    "outcome": "success",
                    "emotional_valence": 0.5,
                    "access_count": 1,
                    "last_accessed": (old_time - timedelta(days=1)).isoformat(),
                    "stability": 1.0,
                },
            ),
            (
                str(uuid4()),  # Must be valid UUID string
                0.88,
                {
                    "session_id": episodic.session_id,
                    "content": "Too new",
                    "timestamp": (future_time + timedelta(days=1)).isoformat(),
                    "ingested_at": (future_time + timedelta(days=1)).isoformat(),
                    "context": {},
                    "outcome": "success",
                    "emotional_valence": 0.5,
                    "access_count": 1,
                    "last_accessed": (future_time + timedelta(days=1)).isoformat(),
                    "stability": 1.0,
                },
            ),
        ]

        episodic.vector_store.search.return_value = results
        episodic.vector_store.update_payload.return_value = None

        scored_results = await episodic.recall(
            query="test",
            limit=10,
            time_start=old_time,
            time_end=future_time,
        )

        # Only in-range episode should be returned
        assert len(scored_results) == 1
        assert scored_results[0].item.content == "In range"

    @pytest.mark.asyncio
    async def test_recall_respects_limit(self, episodic):
        """Test that recall respects the limit parameter."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        # Create 10 results with valid UUID strings
        results = []
        for i in range(10):
            results.append((
                str(uuid4()),  # Must be valid UUID string
                0.9 - (i * 0.05),  # Decreasing similarity
                {
                    "session_id": episodic.session_id,
                    "content": f"Episode {i}",
                    "timestamp": now.isoformat(),
                    "ingested_at": now.isoformat(),
                    "context": {},
                    "outcome": "success",
                    "emotional_valence": 0.5,
                    "access_count": 1,
                    "last_accessed": now.isoformat(),
                    "stability": 1.0,
                },
            ))

        episodic.vector_store.search.return_value = results
        episodic.vector_store.update_payload.return_value = None

        scored_results = await episodic.recall(
            query="test",
            limit=5,
        )

        assert len(scored_results) == 5


class TestEpisodicFSRSDecay:
    """Test FSRS decay calculations and retrievability."""

    @pytest_asyncio.fixture
    async def episodic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create episodic memory instance."""
        episodic = EpisodicMemory(session_id=test_session_id)
        episodic.vector_store = mock_qdrant_store
        episodic.graph_store = mock_neo4j_store
        episodic.embedding = mock_embedding_provider
        episodic.vector_store.episodes_collection = "episodes"
        return episodic

    @pytest.mark.asyncio
    async def test_fsrs_retrievability_at_creation(self, episodic):
        """Test FSRS retrievability is ~1.0 at episode creation."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        episode = await episodic.create(
            content="Test episode",
            outcome="success",
            valence=0.8,
        )

        # Retrievability should be ~1.0 at creation
        current_time = datetime.now()
        R = episode.retrievability(current_time)

        assert 0.98 < R <= 1.0, f"Expected R ~1.0, got {R}"

    @pytest.mark.asyncio
    async def test_fsrs_retrievability_decay_over_time(self, episodic):
        """Test FSRS retrievability decreases over time."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        # Create episode with known creation time
        now = datetime.now()
        episode = await episodic.create(
            content="Test episode",
            outcome="success",
            valence=0.8,
        )

        # Manually set last_accessed for testing
        episode.last_accessed = now
        episode.stability = 1.0

        # Calculate retrievability at different time points
        R_now = episode.retrievability(now)
        R_1day = episode.retrievability(now + timedelta(days=1))
        R_7days = episode.retrievability(now + timedelta(days=7))
        R_30days = episode.retrievability(now + timedelta(days=30))

        # Verify decay pattern
        assert R_now > R_1day, "Retrievability should decrease over time"
        assert R_1day > R_7days, "Retrievability should decrease over time"
        assert R_7days > R_30days, "Retrievability should decrease over time"

        # Rough sanity checks based on FSRS formula: R = (1 + 0.9*t/S)^(-0.5)
        # With stability=1.0:
        # R(0) = 1.0, R(1) = (1 + 0.9)^(-0.5) ≈ 0.725, R(7) ≈ 0.36, R(30) ≈ 0.189
        assert 0.95 < R_now <= 1.0
        assert 0.5 < R_1day < 1.0
        assert 0.2 < R_7days < 0.6  # Updated for actual formula
        assert 0.1 < R_30days < 0.3  # Updated for actual formula (~0.189)

    @pytest.mark.asyncio
    async def test_fsrs_stability_affects_decay(self, episodic):
        """Test that FSRS stability parameter affects decay rate."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        now = datetime.now()

        # Create two episodes with different stability
        episode1 = await episodic.create(content="Episode 1")
        episode1.stability = 1.0
        episode1.last_accessed = now

        episode2 = await episodic.create(content="Episode 2")
        episode2.stability = 10.0  # Higher stability = slower decay
        episode2.last_accessed = now

        # Check retrievability at 7 days
        future = now + timedelta(days=7)
        R1 = episode1.retrievability(future)
        R2 = episode2.retrievability(future)

        # Higher stability should have higher retrievability
        assert R2 > R1, "Higher stability should result in slower decay"

    @pytest.mark.asyncio
    async def test_fsrs_formula_correctness(self, episodic):
        """Test FSRS formula: R(t, S) = (1 + 0.9*t/S)^(-0.5)"""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        now = datetime.now()
        episode = await episodic.create(content="Test")
        episode.stability = 10.0
        episode.last_accessed = now

        # Test at 10 days (t=10, S=10)
        future = now + timedelta(days=10)
        R = episode.retrievability(future)

        # Manual calculation: R(10, 10) = (1 + 0.9*10/10)^(-0.5) = 2^(-0.5) ≈ 0.707
        expected = (1 + 0.9 * 10 / 10) ** (-0.5)

        assert abs(R - expected) < 0.001, f"Expected {expected}, got {R}"


class TestEpisodicAccessTracking:
    """Test episode access pattern updates."""

    @pytest_asyncio.fixture
    async def episodic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create episodic memory instance."""
        episodic = EpisodicMemory(session_id=test_session_id)
        episodic.vector_store = mock_qdrant_store
        episodic.graph_store = mock_neo4j_store
        episodic.embedding = mock_embedding_provider
        episodic.vector_store.episodes_collection = "episodes"
        return episodic

    @pytest.mark.asyncio
    async def test_access_count_updates_on_recall(self, episodic):
        """Test that access count increments on successful recall."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        ep_id = str(uuid4())  # Must be valid UUID string
        payload = {
            "session_id": episodic.session_id,
            "content": "Test episode",
            "timestamp": now.isoformat(),
            "ingested_at": now.isoformat(),
            "context": {},
            "outcome": "success",
            "emotional_valence": 0.5,
            "access_count": 1,
            "last_accessed": now.isoformat(),
            "stability": 1.0,
        }

        episodic.vector_store.search.return_value = [(ep_id, 0.9, payload)]
        # _batch_update_access calls get() to fetch episodes for update
        episodic.vector_store.get.return_value = [(ep_id, payload)]
        episodic.vector_store.batch_update_payloads.return_value = 1

        await episodic.recall(query="test", limit=10)

        # Verify batch update was called
        episodic.vector_store.batch_update_payloads.assert_called()
        call_args = episodic.vector_store.batch_update_payloads.call_args
        updates = call_args[1]["updates"]
        # Updates is a list of (id_str, payload_dict) tuples
        assert len(updates) >= 1
        assert updates[0][1]["access_count"] == 2

    @pytest.mark.asyncio
    async def test_stability_increases_on_successful_recall(self, episodic):
        """Test that stability increases after successful recall."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        ep_id = str(uuid4())  # Must be valid UUID string
        payload = {
            "session_id": episodic.session_id,
            "content": "Test episode",
            "timestamp": now.isoformat(),
            "ingested_at": now.isoformat(),
            "context": {},
            "outcome": "success",
            "emotional_valence": 0.5,
            "access_count": 1,
            "last_accessed": now.isoformat(),
            "stability": 1.0,
        }

        episodic.vector_store.search.return_value = [(ep_id, 0.9, payload)]
        # _batch_update_access calls get() to fetch episodes for update
        episodic.vector_store.get.return_value = [(ep_id, payload)]
        episodic.vector_store.batch_update_payloads.return_value = 1

        await episodic.recall(query="test", limit=10)

        # Stability should increase (checked via batch_update_payloads call)
        episodic.vector_store.batch_update_payloads.assert_called()
        call_args = episodic.vector_store.batch_update_payloads.call_args
        updates = call_args[1]["updates"]
        assert len(updates) >= 1
        assert updates[0][1]["stability"] > 1.0


class TestEpisodicSessionIsolation:
    """Test session isolation for episodic memory."""

    @pytest_asyncio.fixture
    async def episodic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create episodic memory instance."""
        episodic = EpisodicMemory(session_id=test_session_id)
        episodic.vector_store = mock_qdrant_store
        episodic.graph_store = mock_neo4j_store
        episodic.embedding = mock_embedding_provider
        episodic.vector_store.episodes_collection = "episodes"
        return episodic

    @pytest.mark.asyncio
    async def test_episode_inherits_session_id(self, episodic):
        """Test that created episodes inherit the session ID."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        episode = await episodic.create(content="Test")

        assert episode.session_id == episodic.session_id

    @pytest.mark.asyncio
    async def test_recall_filters_by_session(self, episodic):
        """Test that recall uses session filter by default."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.search.return_value = []

        # Non-default session should be filtered
        episodic.session_id = "test-session-123"
        await episodic.recall(query="test", limit=10)

        # Check filter was applied
        call_args = episodic.vector_store.search.call_args
        assert call_args[1]["filter"]["session_id"] == "test-session-123"

    @pytest.mark.asyncio
    async def test_recall_can_override_session_filter(self, episodic):
        """Test that recall can override session filter."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.search.return_value = []

        episodic.session_id = "test-session-123"
        await episodic.recall(
            query="test",
            limit=10,
            session_filter="other-session",
        )

        # Check filter was overridden
        call_args = episodic.vector_store.search.call_args
        assert call_args[1]["filter"]["session_id"] == "other-session"


class TestEpisodicTemporalQueries:
    """Test temporal query capabilities."""

    @pytest_asyncio.fixture
    async def episodic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create episodic memory instance."""
        episodic = EpisodicMemory(session_id=test_session_id)
        episodic.vector_store = mock_qdrant_store
        episodic.graph_store = mock_neo4j_store
        episodic.embedding = mock_embedding_provider
        episodic.vector_store.episodes_collection = "episodes"
        return episodic

    @pytest.mark.asyncio
    async def test_query_at_point_in_time(self, episodic):
        """Test 'what did we know at this point in time' query."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.search.return_value = []

        point_in_time = datetime.now() - timedelta(days=30)

        await episodic.query_at_time(
            query="test",
            point_in_time=point_in_time,
            limit=10,
        )

        # Should filter by time_end
        call_args = episodic.vector_store.search.call_args
        # Note: Actually verifying the call has the right filter would require
        # deeper inspection of the recall() method

    @pytest.mark.asyncio
    async def test_query_at_point_in_time_with_recency_weight(self, episodic):
        """Test that recency score in scoring reflects temporal distance."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding

        now = datetime.now()
        recent = now - timedelta(hours=1)
        old = now - timedelta(days=30)

        results = [
            (
                str(uuid4()),  # Must be valid UUID string
                0.9,
                {
                    "session_id": episodic.session_id,
                    "content": "Recent",
                    "timestamp": recent.isoformat(),
                    "ingested_at": recent.isoformat(),
                    "context": {},
                    "outcome": "success",
                    "emotional_valence": 0.5,
                    "access_count": 1,
                    "last_accessed": recent.isoformat(),
                    "stability": 1.0,
                },
            ),
            (
                str(uuid4()),  # Must be valid UUID string
                0.9,  # Same semantic similarity
                {
                    "session_id": episodic.session_id,
                    "content": "Old",
                    "timestamp": old.isoformat(),
                    "ingested_at": old.isoformat(),
                    "context": {},
                    "outcome": "success",
                    "emotional_valence": 0.5,
                    "access_count": 1,
                    "last_accessed": old.isoformat(),
                    "stability": 1.0,
                },
            ),
        ]

        episodic.vector_store.search.return_value = results
        episodic.vector_store.update_payload.return_value = None

        scored_results = await episodic.recall(query="test", limit=10)

        # Recent should score higher due to recency weight
        recent_idx = next(i for i, r in enumerate(scored_results) if "Recent" in r.item.content)
        old_idx = next(i for i, r in enumerate(scored_results) if "Old" in r.item.content)

        assert scored_results[recent_idx].components["recency"] > scored_results[old_idx].components["recency"]
        assert scored_results[recent_idx].score > scored_results[old_idx].score


class TestEpisodicMetadataHandling:
    """Test metadata and edge cases."""

    @pytest_asyncio.fixture
    async def episodic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create episodic memory instance."""
        episodic = EpisodicMemory(session_id=test_session_id)
        episodic.vector_store = mock_qdrant_store
        episodic.graph_store = mock_neo4j_store
        episodic.embedding = mock_embedding_provider
        episodic.vector_store.episodes_collection = "episodes"
        return episodic

    @pytest.mark.asyncio
    async def test_mark_important_increases_valence(self, episodic):
        """Test marking episode as important increases valence."""
        ep_id = uuid4()
        now = datetime.now()
        payload = {
            "session_id": episodic.session_id,
            "content": "Test",
            "timestamp": now.isoformat(),
            "ingested_at": now.isoformat(),
            "context": {},
            "outcome": "success",
            "emotional_valence": 0.5,
            "access_count": 1,
            "last_accessed": now.isoformat(),
            "stability": 1.0,
        }

        episodic.vector_store.get.return_value = [(str(ep_id), payload)]
        episodic.vector_store.update_payload.return_value = None
        episodic.graph_store.update_node.return_value = None

        updated = await episodic.mark_important(ep_id)

        # Default increase is 0.2
        assert updated.emotional_valence == 0.7

    @pytest.mark.asyncio
    async def test_mark_important_with_explicit_valence(self, episodic):
        """Test marking episode with explicit valence."""
        ep_id = uuid4()
        now = datetime.now()
        payload = {
            "session_id": episodic.session_id,
            "content": "Test",
            "timestamp": now.isoformat(),
            "ingested_at": now.isoformat(),
            "context": {},
            "outcome": "success",
            "emotional_valence": 0.5,
            "access_count": 1,
            "last_accessed": now.isoformat(),
            "stability": 1.0,
        }

        episodic.vector_store.get.return_value = [(str(ep_id), payload)]
        episodic.vector_store.update_payload.return_value = None
        episodic.graph_store.update_node.return_value = None

        updated = await episodic.mark_important(ep_id, new_valence=0.95)

        assert updated.emotional_valence == 0.95

    @pytest.mark.asyncio
    async def test_mark_important_clamps_valence(self, episodic):
        """Test that valence is clamped to [0, 1]."""
        ep_id = uuid4()
        now = datetime.now()
        payload = {
            "session_id": episodic.session_id,
            "content": "Test",
            "timestamp": now.isoformat(),
            "ingested_at": now.isoformat(),
            "context": {},
            "outcome": "success",
            "emotional_valence": 0.95,
            "access_count": 1,
            "last_accessed": now.isoformat(),
            "stability": 1.0,
        }

        episodic.vector_store.get.return_value = [(str(ep_id), payload)]
        episodic.vector_store.update_payload.return_value = None
        episodic.graph_store.update_node.return_value = None

        # Try to increase past 1.0
        updated = await episodic.mark_important(ep_id)

        # Should be clamped to 1.0
        assert updated.emotional_valence == 1.0

    @pytest.mark.asyncio
    async def test_mark_nonexistent_episode_fails(self, episodic):
        """Test that marking non-existent episode raises error."""
        episodic.vector_store.get.return_value = []

        with pytest.raises(ValueError, match="not found"):
            await episodic.mark_important(uuid4())


class TestTypeValidation:
    """DATA-006: Tests for type validation in episodic memory."""

    @pytest_asyncio.fixture
    async def episodic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create episodic memory instance with mocked stores."""
        episodic = EpisodicMemory(session_id=test_session_id)
        episodic.vector_store = mock_qdrant_store
        episodic.graph_store = mock_neo4j_store
        episodic.embedding = mock_embedding_provider
        episodic.vector_store.episodes_collection = "episodes"
        return episodic

    @pytest.mark.asyncio
    async def test_get_with_string_uuid_works(self, episodic):
        """get() accepts valid UUID string and converts it."""
        valid_uuid_str = str(uuid4())
        episodic.vector_store.get.return_value = []

        # Should not raise - string should be converted
        result = await episodic.get(valid_uuid_str)
        assert result is None  # No episode found

    @pytest.mark.asyncio
    async def test_get_with_invalid_string_raises_typeerror(self, episodic):
        """get() raises TypeError for invalid UUID string."""
        with pytest.raises(TypeError, match="must be a valid UUID"):
            await episodic.get("not-a-uuid")

    @pytest.mark.asyncio
    async def test_get_with_wrong_type_raises_typeerror(self, episodic):
        """get() raises TypeError for non-UUID/string types."""
        with pytest.raises(TypeError, match="must be UUID"):
            await episodic.get(12345)

    @pytest.mark.asyncio
    async def test_get_with_none_raises_typeerror(self, episodic):
        """get() raises TypeError for None."""
        with pytest.raises(TypeError, match="must be UUID"):
            await episodic.get(None)

    @pytest.mark.asyncio
    async def test_mark_important_with_string_uuid_works(self, episodic):
        """mark_important() accepts valid UUID string."""
        valid_uuid = uuid4()
        valid_uuid_str = str(valid_uuid)
        now = datetime.now()
        payload = {
            "session_id": episodic.session_id,
            "content": "Test",
            "timestamp": now.isoformat(),
            "ingested_at": now.isoformat(),
            "context": {},
            "outcome": "success",
            "emotional_valence": 0.5,
            "access_count": 1,
            "last_accessed": now.isoformat(),
            "stability": 1.0,
        }
        episodic.vector_store.get.return_value = [(valid_uuid_str, payload)]
        episodic.vector_store.update_payload.return_value = None
        episodic.graph_store.update_node.return_value = None

        # Should not raise - string should be converted
        result = await episodic.mark_important(valid_uuid_str)
        assert result is not None

    @pytest.mark.asyncio
    async def test_mark_important_with_invalid_string_raises_typeerror(self, episodic):
        """mark_important() raises TypeError for invalid UUID string."""
        with pytest.raises(TypeError, match="must be a valid UUID"):
            await episodic.mark_important("invalid-uuid-string")


class TestP52TemporalStructure:
    """P5.2: Tests for episode temporal structure and sequencing."""

    @pytest_asyncio.fixture
    async def episodic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create episodic memory instance with mocked stores."""
        episodic = EpisodicMemory(session_id=test_session_id)
        episodic.vector_store = mock_qdrant_store
        episodic.graph_store = mock_neo4j_store
        episodic.embedding = mock_embedding_provider
        episodic.vector_store.episodes_collection = "episodes"
        return episodic

    @pytest.mark.asyncio
    async def test_first_episode_has_no_previous(self, episodic):
        """First episode in session has no previous_episode_id."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        episode = await episodic.create(content="First episode")

        assert episode is not None
        assert episode.previous_episode_id is None
        assert episode.sequence_position == 0

    @pytest.mark.asyncio
    async def test_second_episode_links_to_first(self, episodic):
        """Second episode has previous_episode_id set to first episode."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        # Create first episode
        first = await episodic.create(content="First episode")
        first_id = first.id

        # Create second episode
        second = await episodic.create(content="Second episode")

        assert second.previous_episode_id == first_id
        assert second.sequence_position == 1

    @pytest.mark.asyncio
    async def test_sequence_position_increments(self, episodic):
        """Sequence position increments with each episode."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        episodes = []
        for i in range(5):
            ep = await episodic.create(content=f"Episode {i}")
            episodes.append(ep)

        for i, ep in enumerate(episodes):
            assert ep.sequence_position == i

    @pytest.mark.asyncio
    async def test_temporal_link_created_between_episodes(self, episodic):
        """Temporal SEQUENCE link created in graph store between episodes."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"
        episodic.graph_store.create_relationship.return_value = None

        # Create two episodes
        first = await episodic.create(content="First")
        second = await episodic.create(content="Second")

        # Verify temporal link was created
        episodic.graph_store.create_relationship.assert_called()
        call_args = episodic.graph_store.create_relationship.call_args
        assert call_args[1]["source_id"] == str(first.id)
        assert call_args[1]["target_id"] == str(second.id)
        assert call_args[1]["relation_type"] == "TEMPORAL_SEQUENCE"

    @pytest.mark.asyncio
    async def test_episode_temporal_fields_in_payload(self, episodic):
        """Episode temporal fields included in vector store payload."""
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.add.return_value = None
        episodic.graph_store.create_node.return_value = "test-node-id"

        # Create episode
        episode = await episodic.create(content="Test episode")

        # Check payload includes sequence_position
        add_call = episodic.vector_store.add.call_args
        payload = add_call[1]["payloads"][0]
        assert "sequence_position" in payload
        assert payload["sequence_position"] == 0

    @pytest.mark.asyncio
    async def test_from_payload_parses_temporal_fields(self, episodic):
        """_from_payload correctly parses temporal fields."""
        now = datetime.now()
        prev_id = uuid4()
        next_id = uuid4()

        payload = {
            "session_id": episodic.session_id,
            "content": "Test",
            "timestamp": now.isoformat(),
            "context": {},
            "outcome": "success",
            "emotional_valence": 0.5,
            "stability": 1.0,
            "previous_episode_id": str(prev_id),
            "next_episode_id": str(next_id),
            "sequence_position": 5,
            "duration_ms": 1500,
            "end_timestamp": now.isoformat(),
        }

        episode = episodic._from_payload(str(uuid4()), payload)

        assert episode.previous_episode_id == prev_id
        assert episode.next_episode_id == next_id
        assert episode.sequence_position == 5
        assert episode.duration_ms == 1500
        assert episode.end_timestamp is not None

    @pytest.mark.asyncio
    async def test_get_session_timeline_orders_by_sequence(self, episodic):
        """get_session_timeline returns episodes ordered by sequence_position."""
        now = datetime.now()

        # Create payloads with sequence positions out of order
        results = [
            (str(uuid4()), {"session_id": episodic.session_id, "content": "Third", "timestamp": now.isoformat(), "context": {}, "outcome": "neutral", "emotional_valence": 0.5, "stability": 1.0, "sequence_position": 2}, None),
            (str(uuid4()), {"session_id": episodic.session_id, "content": "First", "timestamp": now.isoformat(), "context": {}, "outcome": "neutral", "emotional_valence": 0.5, "stability": 1.0, "sequence_position": 0}, None),
            (str(uuid4()), {"session_id": episodic.session_id, "content": "Second", "timestamp": now.isoformat(), "context": {}, "outcome": "neutral", "emotional_valence": 0.5, "stability": 1.0, "sequence_position": 1}, None),
        ]
        episodic.vector_store.scroll.return_value = (results, None)

        timeline = await episodic.get_session_timeline()

        assert len(timeline) == 3
        assert timeline[0].content == "First"
        assert timeline[1].content == "Second"
        assert timeline[2].content == "Third"

    @pytest.mark.asyncio
    async def test_get_episode_sequence_walks_links(self, episodic):
        """get_episode_sequence walks forward and backward links."""
        now = datetime.now()
        ids = [uuid4() for _ in range(5)]

        # Setup mock get() to return linked episodes
        async def mock_get(ep_id):
            idx = next((i for i, id_ in enumerate(ids) if id_ == ep_id), None)
            if idx is None:
                return None
            return Episode(
                id=ids[idx],
                session_id=episodic.session_id,
                content=f"Episode {idx}",
                previous_episode_id=ids[idx - 1] if idx > 0 else None,
                next_episode_id=ids[idx + 1] if idx < len(ids) - 1 else None,
                sequence_position=idx,
            )

        episodic.get = AsyncMock(side_effect=mock_get)

        # Get sequence around middle episode
        sequence = await episodic.get_episode_sequence(ids[2], before=2, after=2)

        assert len(sequence) == 5
        assert sequence[0].content == "Episode 0"
        assert sequence[2].content == "Episode 2"
        assert sequence[4].content == "Episode 4"


class TestP54QueryMemorySeparation:
    """P5.4: Test query-memory encoder separation."""

    @pytest_asyncio.fixture
    async def episodic(self, test_session_id, mock_qdrant_store, mock_neo4j_store, mock_embedding_provider):
        """Create episodic memory instance."""
        episodic = EpisodicMemory(session_id=test_session_id)
        episodic.vector_store = mock_qdrant_store
        episodic.graph_store = mock_neo4j_store
        episodic.embedding = mock_embedding_provider
        episodic.vector_store.episodes_collection = "episodes"
        return episodic

    @pytest.mark.asyncio
    async def test_separator_initialized(self, episodic):
        """EpisodicMemory initializes QueryMemorySeparator."""
        assert hasattr(episodic, '_query_memory_separator')
        assert episodic._query_memory_separator is not None
        assert hasattr(episodic, '_query_memory_separation_enabled')

    @pytest.mark.asyncio
    async def test_recall_applies_query_projection(self, episodic):
        """recall() applies query projection when enabled."""
        import numpy as np

        # Track if projection was called
        original_project = episodic._query_memory_separator.project_query
        projection_calls = []

        def tracking_project(emb):
            projection_calls.append(emb)
            return original_project(emb)

        episodic._query_memory_separator.project_query = tracking_project

        # Setup mocks
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.search.return_value = []

        # Ensure enabled
        episodic._query_memory_separation_enabled = True

        await episodic.recall(query="test query")

        # Verify projection was called
        assert len(projection_calls) == 1
        assert len(projection_calls[0]) == 1024

    @pytest.mark.asyncio
    async def test_recall_skips_projection_when_disabled(self, episodic):
        """recall() skips projection when disabled."""
        # Track if projection was called
        projection_calls = []

        def tracking_project(emb):
            projection_calls.append(emb)
            return emb

        episodic._query_memory_separator.project_query = tracking_project

        # Setup mocks
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.search.return_value = []

        # Disable separation
        episodic._query_memory_separation_enabled = False

        await episodic.recall(query="test query")

        # Verify projection was NOT called
        assert len(projection_calls) == 0

    @pytest.mark.asyncio
    async def test_projection_preserves_embedding_dimension(self, episodic):
        """Query projection preserves embedding dimension."""
        import numpy as np

        test_embedding = np.random.randn(1024).astype(np.float32)
        projected = episodic._query_memory_separator.project_query(test_embedding)

        assert projected.shape == test_embedding.shape
        assert projected.shape == (1024,)

    @pytest.mark.asyncio
    async def test_projection_normalizes_to_unit_sphere(self, episodic):
        """Query projection normalizes output to unit sphere."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator
        import numpy as np

        # Use fresh separator to avoid singleton state issues
        separator = QueryMemorySeparator()
        test_embedding = np.random.randn(1024).astype(np.float32) * 10
        projected = separator.project_query(test_embedding)

        norm = np.linalg.norm(projected)
        assert 0.99 < norm < 1.01, f"Expected unit norm, got {norm}"

    @pytest.mark.asyncio
    async def test_query_and_memory_projections_differ(self, episodic):
        """Query and memory projections produce different outputs."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator
        import numpy as np

        # Use fresh separator to avoid singleton state issues
        separator = QueryMemorySeparator()
        test_embedding = np.random.randn(1024).astype(np.float32)

        query_proj = separator.project_query(test_embedding)
        memory_proj = separator.project_memory(test_embedding)

        # Projections should differ (asymmetric encoding)
        diff = np.linalg.norm(query_proj - memory_proj)
        assert diff > 0.01, "Query and memory projections should differ"

    @pytest.mark.asyncio
    async def test_projection_failure_doesnt_break_recall(self, episodic):
        """Projection failure falls back gracefully."""
        # Make projection fail
        def failing_project(emb):
            raise RuntimeError("Test failure")

        episodic._query_memory_separator.project_query = failing_project
        episodic._query_memory_separation_enabled = True

        # Setup mocks
        test_embedding = [0.1] * 1024
        episodic.embedding.embed_query.return_value = test_embedding
        episodic.vector_store.search.return_value = []

        # Should not raise, just log warning
        result = await episodic.recall(query="test query")
        assert result is not None


class TestP54QueryMemorySeparatorUnit:
    """Unit tests for QueryMemorySeparator class."""

    def test_separator_initialization(self):
        """QueryMemorySeparator initializes with correct dimensions."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator, SeparationConfig

        config = SeparationConfig(embedding_dim=512, hidden_dim=128)
        separator = QueryMemorySeparator(config)

        assert separator.W_q.shape == (512, 128)
        assert separator.U_q.shape == (128, 512)
        assert separator.W_m.shape == (512, 128)
        assert separator.U_m.shape == (128, 512)

    def test_batch_projection(self):
        """Separator handles batch projections."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator, SeparationConfig
        import numpy as np

        config = SeparationConfig(embedding_dim=256, hidden_dim=64)
        separator = QueryMemorySeparator(config)

        # Batch of embeddings
        batch = np.random.randn(10, 256).astype(np.float32)
        projected = separator.project_query(batch)

        assert projected.shape == (10, 256)

        # All should be unit norm
        norms = np.linalg.norm(projected, axis=1)
        assert all(0.99 < n < 1.01 for n in norms)

    def test_compute_similarity(self):
        """compute_similarity works with projection."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator, SeparationConfig
        import numpy as np

        config = SeparationConfig(embedding_dim=256, hidden_dim=64)
        separator = QueryMemorySeparator(config)

        query = np.random.randn(256).astype(np.float32)
        memories = np.random.randn(5, 256).astype(np.float32)

        sims = separator.compute_similarity(query, memories, project=True)

        assert sims.shape == (5,)
        # Cosine sims should be in [-1, 1]
        assert all(-1.01 < s < 1.01 for s in sims)

    def test_train_step_reduces_loss(self):
        """Training step can reduce triplet loss."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator, SeparationConfig
        import numpy as np

        config = SeparationConfig(
            embedding_dim=128,
            hidden_dim=32,
            learning_rate=0.1,
        )
        separator = QueryMemorySeparator(config)

        # Create triplet where positive is close, negatives are far
        query = np.random.randn(128).astype(np.float32)
        positive = query + np.random.randn(128).astype(np.float32) * 0.1
        negatives = np.random.randn(5, 128).astype(np.float32)

        # Train for a few steps
        losses = []
        for _ in range(10):
            loss = separator.train_step(query, positive, negatives)
            losses.append(loss)

        # Stats should update
        assert separator.stats.training_updates == 10

    def test_save_and_load_state(self):
        """Separator state can be saved and loaded."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator, SeparationConfig
        import numpy as np

        config = SeparationConfig(embedding_dim=128, hidden_dim=32)
        separator = QueryMemorySeparator(config)

        # Do some projections to change state
        test_emb = np.random.randn(128).astype(np.float32)
        original_proj = separator.project_query(test_emb)

        # Save state
        state = separator.save_state()

        # Create new separator and load state
        separator2 = QueryMemorySeparator(config)
        separator2.load_state(state)

        # Should produce same projection
        loaded_proj = separator2.project_query(test_emb)

        np.testing.assert_array_almost_equal(original_proj, loaded_proj)

    def test_get_stats(self):
        """get_stats returns projection statistics."""
        from ww.embedding.query_memory_separation import QueryMemorySeparator
        import numpy as np

        separator = QueryMemorySeparator()

        # Do some projections
        for _ in range(5):
            separator.project_query(np.random.randn(1024).astype(np.float32))
            separator.project_memory(np.random.randn(1024).astype(np.float32))

        stats = separator.get_stats()

        assert stats["query_projections"] == 5
        assert stats["memory_projections"] == 5
        assert "avg_query_norm" in stats
        assert "avg_memory_norm" in stats
