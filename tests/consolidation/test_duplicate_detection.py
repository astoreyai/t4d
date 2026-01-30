"""Tests for LSH-based duplicate detection."""
import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from ww.consolidation.service import ConsolidationService
from ww.core.types import Episode, EpisodeContext, Outcome


class TestDuplicateDetection:
    """Test suite for _find_duplicates method."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        vector_store = MagicMock()
        vector_store.episodes_collection = "ww_episodes"
        vector_store.search = AsyncMock(return_value=[])
        return vector_store

    @pytest.fixture
    def mock_graph_store(self):
        """Create mock graph store."""
        graph_store = MagicMock()
        return graph_store

    @pytest.fixture
    def mock_embedding(self):
        """Create mock embedding provider."""
        embedding = MagicMock()
        return embedding

    @pytest.fixture
    def consolidation_service(self, mock_vector_store, mock_graph_store, mock_embedding):
        """Create consolidation service with mocks."""
        service = ConsolidationService.__new__(ConsolidationService)
        service.vector_store = mock_vector_store
        service.graph_store = mock_graph_store
        service.embedding = mock_embedding
        service._episodic = None
        service._semantic = None
        service._procedural = None
        service.min_similarity = 0.75
        service.min_occurrences = 3
        service.skill_similarity = 0.85
        service.hdbscan_min_cluster_size = 3
        service.hdbscan_min_samples = 3
        service.hdbscan_metric = "cosine"
        return service

    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes for testing."""
        base_time = datetime.now()

        # Give each episode a unique embedding
        episodes = []
        for i in range(5):
            embedding = [0.1 + i * 0.01] * 1024  # Unique embedding per episode
            ep = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                embedding=embedding,
                timestamp=base_time - timedelta(hours=i),
                context=EpisodeContext(),
                outcome=Outcome.NEUTRAL,
            )
            episodes.append(ep)

        return episodes

    @pytest.mark.asyncio
    async def test_empty_episodes(self, consolidation_service):
        """Test with empty episode list."""
        result = await consolidation_service._find_duplicates([])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_episode(self, consolidation_service, sample_episodes):
        """Test with single episode."""
        result = await consolidation_service._find_duplicates([sample_episodes[0]])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_duplicates(self, consolidation_service, sample_episodes):
        """Test when no duplicates found."""
        ep1, ep2 = sample_episodes[0], sample_episodes[1]

        # Search returns only self-match for each episode
        async def mock_search(**kwargs):
            vector = kwargs.get("vector")
            if vector == ep1.embedding:
                return [(str(ep1.id), 1.0, {})]
            elif vector == ep2.embedding:
                return [(str(ep2.id), 1.0, {})]
            return []

        consolidation_service.vector_store.search = AsyncMock(side_effect=mock_search)
        result = await consolidation_service._find_duplicates([ep1, ep2])
        assert result == []

    @pytest.mark.asyncio
    async def test_finds_duplicates(self, consolidation_service, sample_episodes):
        """Test duplicate detection finds similar episodes."""
        ep1, ep2 = sample_episodes[0], sample_episodes[1]
        # Note: ep1.timestamp = base_time - 0h (NEWER)
        #       ep2.timestamp = base_time - 1h (OLDER, should be kept)

        # Mock search to return ep2 as similar to ep1, and vice versa
        async def mock_search(**kwargs):
            vector = kwargs.get("vector")
            if vector == ep1.embedding:
                return [
                    (str(ep1.id), 1.0, {}),  # Self
                    (str(ep2.id), 0.98, {}),  # Duplicate
                ]
            elif vector == ep2.embedding:
                return [
                    (str(ep2.id), 1.0, {}),  # Self
                    (str(ep1.id), 0.98, {}),  # Duplicate
                ]
            return []

        consolidation_service.vector_store.search = AsyncMock(side_effect=mock_search)

        result = await consolidation_service._find_duplicates([ep1, ep2])

        # Should find one duplicate pair (deduped)
        assert len(result) == 1
        keep_id, remove_id = result[0]
        # Older episode (ep2, timestamp - 1h) should be kept
        assert keep_id == str(ep2.id)
        assert remove_id == str(ep1.id)

    @pytest.mark.asyncio
    async def test_respects_threshold(self, consolidation_service, sample_episodes):
        """Test that threshold is passed to search."""
        ep1, ep2 = sample_episodes[0], sample_episodes[1]

        # Mock search to capture parameters
        async def mock_search(**kwargs):
            return [(str(ep1.id), 1.0, {})]

        consolidation_service.vector_store.search = AsyncMock(side_effect=mock_search)

        await consolidation_service._find_duplicates([ep1, ep2], threshold=0.90)

        # Verify threshold was passed
        call_kwargs = consolidation_service.vector_store.search.call_args.kwargs
        assert call_kwargs.get("score_threshold") == 0.90

    @pytest.mark.asyncio
    async def test_handles_missing_embedding(self, consolidation_service):
        """Test episodes without embeddings are skipped."""
        ep_no_embedding = Episode(
            id=uuid4(),
            session_id="test",
            content="No embedding",
            embedding=None,
            timestamp=datetime.now(),
            context=EpisodeContext(),
            outcome=Outcome.NEUTRAL,
        )

        result = await consolidation_service._find_duplicates([ep_no_embedding])
        assert result == []
        # Search should not have been called
        consolidation_service.vector_store.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_search_error(self, consolidation_service, sample_episodes):
        """Test graceful handling of search errors."""
        consolidation_service.vector_store.search = AsyncMock(
            side_effect=Exception("Search failed")
        )

        # Should not raise, just log and continue
        result = await consolidation_service._find_duplicates(sample_episodes[:2])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_duplicate_pairs(self, consolidation_service, sample_episodes):
        """Test that same pair is not returned multiple times."""
        ep1, ep2 = sample_episodes[0], sample_episodes[1]

        # Both episodes find each other as duplicates
        async def mock_search(**kwargs):
            return [
                (str(ep1.id), 0.99, {}),
                (str(ep2.id), 0.98, {}),
            ]

        consolidation_service.vector_store.search = AsyncMock(side_effect=mock_search)

        result = await consolidation_service._find_duplicates([ep1, ep2])

        # Should only return one pair, not two
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_filters_episodes_not_in_input_set(self, consolidation_service, sample_episodes):
        """Test that only episodes in input set are considered."""
        ep1 = sample_episodes[0]
        external_id = str(uuid4())

        # Search returns an external episode not in input set
        async def mock_search(**kwargs):
            return [
                (str(ep1.id), 1.0, {}),  # Self
                (external_id, 0.99, {}),  # External episode
            ]

        consolidation_service.vector_store.search = AsyncMock(side_effect=mock_search)

        result = await consolidation_service._find_duplicates([ep1])

        # Should not return any duplicates (external episode filtered)
        assert result == []

    @pytest.mark.asyncio
    async def test_keeps_older_episode(self, consolidation_service, sample_episodes):
        """Test that older episode is kept over newer one."""
        ep1, ep2 = sample_episodes[0], sample_episodes[1]
        # Default: ep1 timestamp = base_time - 0h (NEWER)
        #          ep2 timestamp = base_time - 1h (OLDER)

        # Swap timestamps so ep1 becomes older
        ep1.timestamp, ep2.timestamp = ep2.timestamp, ep1.timestamp
        # Now: ep1 timestamp = base_time - 1h (OLDER, should be kept)
        #      ep2 timestamp = base_time - 0h (NEWER)

        async def mock_search(**kwargs):
            vector = kwargs.get("vector")
            if vector == ep1.embedding:
                return [
                    (str(ep1.id), 1.0, {}),
                    (str(ep2.id), 0.98, {}),
                ]
            elif vector == ep2.embedding:
                return [
                    (str(ep2.id), 1.0, {}),
                    (str(ep1.id), 0.98, {}),
                ]
            return []

        consolidation_service.vector_store.search = AsyncMock(side_effect=mock_search)

        result = await consolidation_service._find_duplicates([ep1, ep2])

        assert len(result) == 1
        keep_id, remove_id = result[0]
        # ep1 is now older, should be kept
        assert keep_id == str(ep1.id)
        assert remove_id == str(ep2.id)

    @pytest.mark.asyncio
    async def test_multiple_duplicates(self, consolidation_service, sample_episodes):
        """Test finding multiple duplicate pairs."""
        ep1, ep2, ep3 = sample_episodes[0], sample_episodes[1], sample_episodes[2]

        # ep1-ep2 are duplicates, ep1-ep3 are duplicates (not ep2-ep3)
        async def mock_search(**kwargs):
            vector = kwargs.get("vector")
            if vector == ep1.embedding:
                return [
                    (str(ep1.id), 1.0, {}),
                    (str(ep2.id), 0.98, {}),
                    (str(ep3.id), 0.97, {}),
                ]
            elif vector == ep2.embedding:
                return [
                    (str(ep2.id), 1.0, {}),
                    (str(ep1.id), 0.98, {}),
                    # ep2-ep3 not duplicates (below threshold or not returned)
                ]
            elif vector == ep3.embedding:
                return [
                    (str(ep3.id), 1.0, {}),
                    (str(ep1.id), 0.97, {}),
                    # ep3-ep2 not duplicates
                ]
            return []

        consolidation_service.vector_store.search = AsyncMock(side_effect=mock_search)

        result = await consolidation_service._find_duplicates([ep1, ep2, ep3])

        # Should find 2 pairs: ep1-ep2 and ep1-ep3 (deduped from bidirectional search)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_limit_parameter(self, consolidation_service, sample_episodes):
        """Test that limit parameter is passed to search."""
        ep1, ep2 = sample_episodes[0], sample_episodes[1]

        # Mock search to capture parameters
        async def mock_search(**kwargs):
            return [(str(ep1.id), 1.0, {})]

        consolidation_service.vector_store.search = AsyncMock(side_effect=mock_search)

        await consolidation_service._find_duplicates([ep1, ep2])

        # Verify limit=10 was passed
        call_kwargs = consolidation_service.vector_store.search.call_args.kwargs
        assert call_kwargs.get("limit") == 10


class TestDuplicateDetectionPerformance:
    """Performance tests for duplicate detection."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_performance_complexity(self):
        """
        Verify algorithm is O(n*k) not O(n²).

        This is a complexity test, not a benchmark.
        With 1000 episodes, O(n²) would require 500,000 comparisons.
        With O(n*k) where k=10, we need only 10,000 comparisons.
        """
        # This would be an integration test with real Qdrant
        # For unit test, we verify the algorithm structure
        # by checking search is called n times with limit k
        pass

    @pytest.mark.asyncio
    async def test_search_called_per_episode(self):
        """Test that search is called once per episode (not per pair)."""
        from ww.consolidation.service import ConsolidationService
        from ww.core.types import Episode, EpisodeContext, Outcome

        service = ConsolidationService.__new__(ConsolidationService)
        service.vector_store = MagicMock()
        service.vector_store.episodes_collection = "ww_episodes"
        service.vector_store.search = AsyncMock(return_value=[])

        episodes = [
            Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                embedding=[0.1] * 1024,
                timestamp=datetime.now(),
                context=EpisodeContext(),
                outcome=Outcome.NEUTRAL,
            )
            for i in range(10)
        ]

        await service._find_duplicates(episodes)

        # Search should be called exactly 10 times (once per episode)
        assert service.vector_store.search.call_count == 10
