"""
P10-002: Tests for cursor-based pagination in episodic memory.

Tests time-based windowed retrieval with pagination for consolidation operations.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from ww.core.types import Episode, EpisodeContext, Outcome


class TestEpisodicPagination:
    """Test paginated time-based retrieval."""

    @pytest.fixture
    def episodic(self, test_session_id):
        """Create episodic memory instance with fully mocked stores."""
        # Create mocked stores
        mock_qdrant = MagicMock()
        mock_qdrant.episodes_collection = "episodes"
        mock_qdrant.scroll = AsyncMock(return_value=([], None))
        mock_qdrant.count = AsyncMock(return_value=0)

        mock_neo4j = MagicMock()
        mock_embedding = MagicMock()

        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.session_id = test_session_id
        mock_settings.episodic_weight_semantic = 0.4
        mock_settings.episodic_weight_recency = 0.25
        mock_settings.episodic_weight_outcome = 0.2
        mock_settings.episodic_weight_importance = 0.15
        mock_settings.fsrs_default_stability = 2.5
        mock_settings.fsrs_decay_factor = 0.69
        mock_settings.fsrs_recency_decay = 0.1
        mock_settings.ff_encoder_enabled = False  # Phase 5: Disable for unit tests
        mock_settings.capsule_layer_enabled = False  # Phase 6: Disable for unit tests
        mock_settings.capsule_retrieval_enabled = False  # Phase 6: Disable for unit tests
        mock_settings.embedding_dimension = 1024

        # Patch all dependencies
        with patch("ww.memory.episodic.get_settings", return_value=mock_settings):
            with patch("ww.memory.episodic.get_qdrant_store", return_value=mock_qdrant):
                with patch("ww.memory.episodic.get_neo4j_store", return_value=mock_neo4j):
                    with patch("ww.memory.episodic.get_embedding_provider", return_value=mock_embedding):
                        with patch("ww.memory.episodic.get_ff_encoder", return_value=None):
                            from ww.memory.episodic import EpisodicMemory
                            episodic = EpisodicMemory(session_id=test_session_id)
                            yield episodic

    @pytest.mark.asyncio
    async def test_recall_by_timerange_single_page(self, episodic):
        """Test time-based recall with single page of results."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()

        # Mock scroll response with 10 episodes
        mock_episodes = []
        for i in range(10):
            episode_id = str(uuid4())
            payload = {
                "session_id": episodic.session_id,
                "content": f"Episode {i}",
                "timestamp": (start_time + timedelta(hours=i)).isoformat(),
                "ingested_at": (start_time + timedelta(hours=i)).isoformat(),
                "context": {},
                "outcome": "neutral",
                "emotional_valence": 0.5,
                "access_count": 1,
                "last_accessed": start_time.isoformat(),
                "stability": 2.5,
            }
            mock_episodes.append((episode_id, payload, None))

        episodic.vector_store.scroll.return_value = (mock_episodes, None)

        # Call recall_by_timerange
        episodes, next_cursor = await episodic.recall_by_timerange(
            start_time=start_time,
            end_time=end_time,
            page_size=100,
        )

        # Assertions
        assert len(episodes) == 10
        assert next_cursor is None
        assert all(isinstance(ep, Episode) for ep in episodes)

        # Verify scroll was called correctly
        episodic.vector_store.scroll.assert_called_once()
        call_args = episodic.vector_store.scroll.call_args
        assert call_args.kwargs["collection"] == "episodes"
        assert call_args.kwargs["limit"] == 100
        assert call_args.kwargs["offset"] == 0
        assert call_args.kwargs["with_payload"] is True
        assert call_args.kwargs["with_vectors"] is False

    @pytest.mark.asyncio
    async def test_recall_by_timerange_multiple_pages(self, episodic):
        """Test time-based recall with multiple pages."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()

        # Mock first page (100 episodes)
        mock_page1 = []
        for i in range(100):
            episode_id = str(uuid4())
            payload = {
                "session_id": episodic.session_id,
                "content": f"Episode {i}",
                "timestamp": (start_time + timedelta(minutes=i)).isoformat(),
                "ingested_at": (start_time + timedelta(minutes=i)).isoformat(),
                "context": {},
                "outcome": "neutral",
                "emotional_valence": 0.5,
                "access_count": 1,
                "last_accessed": start_time.isoformat(),
                "stability": 2.5,
            }
            mock_page1.append((episode_id, payload, None))

        # Mock second page (50 episodes)
        mock_page2 = []
        for i in range(100, 150):
            episode_id = str(uuid4())
            payload = {
                "session_id": episodic.session_id,
                "content": f"Episode {i}",
                "timestamp": (start_time + timedelta(minutes=i)).isoformat(),
                "ingested_at": (start_time + timedelta(minutes=i)).isoformat(),
                "context": {},
                "outcome": "neutral",
                "emotional_valence": 0.5,
                "access_count": 1,
                "last_accessed": start_time.isoformat(),
                "stability": 2.5,
            }
            mock_page2.append((episode_id, payload, None))

        # Setup scroll to return different results based on offset
        def scroll_side_effect(**kwargs):
            offset = kwargs.get("offset", 0)
            if offset == 0:
                return (mock_page1, 100)
            elif offset == 100:
                return (mock_page2, None)
            else:
                return ([], None)

        episodic.vector_store.scroll.side_effect = scroll_side_effect

        # Fetch first page
        episodes_page1, cursor1 = await episodic.recall_by_timerange(
            start_time=start_time,
            end_time=end_time,
            page_size=100,
        )

        assert len(episodes_page1) == 100
        assert cursor1 == "100"

        # Fetch second page
        episodes_page2, cursor2 = await episodic.recall_by_timerange(
            start_time=start_time,
            end_time=end_time,
            page_size=100,
            cursor=cursor1,
        )

        assert len(episodes_page2) == 50
        assert cursor2 is None

    @pytest.mark.asyncio
    async def test_recall_by_timerange_with_session_filter(self, episodic):
        """Test time-based recall with session filter."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()

        episodic.vector_store.scroll.return_value = ([], None)

        await episodic.recall_by_timerange(
            start_time=start_time,
            end_time=end_time,
            page_size=100,
            session_filter="specific-session",
        )

        # Verify filter includes session_id
        call_args = episodic.vector_store.scroll.call_args
        scroll_filter = call_args.kwargs["scroll_filter"]
        assert "session_id" in scroll_filter
        assert scroll_filter["session_id"] == "specific-session"

    @pytest.mark.asyncio
    async def test_recall_by_timerange_caps_page_size(self, episodic):
        """Test that page size is capped at 500."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()

        episodic.vector_store.scroll.return_value = ([], None)

        await episodic.recall_by_timerange(
            start_time=start_time,
            end_time=end_time,
            page_size=1000,  # Request more than cap
        )

        # Verify limit is capped
        call_args = episodic.vector_store.scroll.call_args
        assert call_args.kwargs["limit"] == 500

    @pytest.mark.asyncio
    async def test_recall_by_timerange_invalid_cursor(self, episodic):
        """Test handling of invalid cursor."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()

        episodic.vector_store.scroll.return_value = ([], None)

        # Should not raise, should default to offset 0
        await episodic.recall_by_timerange(
            start_time=start_time,
            end_time=end_time,
            page_size=100,
            cursor="invalid-cursor",
        )

        # Verify offset defaults to 0
        call_args = episodic.vector_store.scroll.call_args
        assert call_args.kwargs["offset"] == 0

    @pytest.mark.asyncio
    async def test_count_by_timerange(self, episodic):
        """Test counting episodes in time range."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()

        episodic.vector_store.count.return_value = 42

        count = await episodic.count_by_timerange(
            start_time=start_time,
            end_time=end_time,
        )

        assert count == 42

        # Verify count was called with correct filter
        call_args = episodic.vector_store.count.call_args
        assert call_args.kwargs["collection"] == "episodes"
        count_filter = call_args.kwargs["count_filter"]
        assert "timestamp" in count_filter
        assert "gte" in count_filter["timestamp"]
        assert "lte" in count_filter["timestamp"]

    @pytest.mark.asyncio
    async def test_count_by_timerange_with_session_filter(self, episodic):
        """Test counting episodes with session filter."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()

        episodic.vector_store.count.return_value = 10

        count = await episodic.count_by_timerange(
            start_time=start_time,
            end_time=end_time,
            session_filter="specific-session",
        )

        # Verify filter includes session_id
        call_args = episodic.vector_store.count.call_args
        count_filter = call_args.kwargs["count_filter"]
        assert "session_id" in count_filter
        assert count_filter["session_id"] == "specific-session"

    @pytest.mark.asyncio
    async def test_recall_by_timerange_error_handling(self, episodic):
        """Test error handling in time-based recall."""
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()

        episodic.vector_store.scroll.side_effect = Exception("Database error")

        with pytest.raises(Exception) as exc_info:
            await episodic.recall_by_timerange(
                start_time=start_time,
                end_time=end_time,
            )

        assert "Database error" in str(exc_info.value)


class TestConsolidationPagination:
    """Test pagination in consolidation service."""

    @pytest.fixture
    def consolidation_service(self):
        """Create consolidation service with mocked dependencies."""
        # Mock hdbscan import
        import sys
        hdbscan_mock = MagicMock()
        sys.modules['hdbscan'] = hdbscan_mock

        from ww.consolidation.service import ConsolidationService

        # Mock all the get_* functions
        with patch("ww.consolidation.service.get_settings") as mock_settings:
            with patch("ww.consolidation.service.get_embedding_provider") as mock_emb:
                with patch("ww.consolidation.service.get_qdrant_store") as mock_qdrant:
                    with patch("ww.consolidation.service.get_neo4j_store") as mock_neo4j:
                        mock_settings.return_value.consolidation_min_similarity = 0.75
                        mock_settings.return_value.consolidation_min_occurrences = 3
                        mock_settings.return_value.consolidation_skill_similarity = 0.85
                        mock_settings.return_value.hdbscan_min_cluster_size = 3
                        mock_settings.return_value.hdbscan_min_samples = None
                        mock_settings.return_value.hdbscan_metric = "cosine"

                        mock_emb.return_value = MagicMock()
                        mock_qdrant_inst = MagicMock()
                        mock_qdrant_inst.episodes_collection = "episodes"
                        mock_qdrant.return_value = mock_qdrant_inst
                        mock_neo4j.return_value = MagicMock()

                        service = ConsolidationService()

                        # Mock services
                        service._episodic = AsyncMock()
                        service._semantic = AsyncMock()
                        service._procedural = AsyncMock()

                        yield service

    @pytest.mark.asyncio
    async def test_consolidate_light_uses_pagination(self, consolidation_service):
        """Test that light consolidation uses paginated retrieval."""
        # Mock count
        consolidation_service._episodic.count_by_timerange.return_value = 250

        # Mock paginated results
        def recall_side_effect(**kwargs):
            cursor = kwargs.get("cursor")
            if cursor is None:
                # First page: 200 episodes
                episodes = [
                    Episode(
                        session_id="test",
                        content=f"Episode {i}",
                        embedding=[0.1] * 1024,
                        context=EpisodeContext(),
                        outcome=Outcome.NEUTRAL,
                    )
                    for i in range(200)
                ]
                return (episodes, "200")
            elif cursor == "200":
                # Second page: 50 episodes
                episodes = [
                    Episode(
                        session_id="test",
                        content=f"Episode {i}",
                        embedding=[0.1] * 1024,
                        context=EpisodeContext(),
                        outcome=Outcome.NEUTRAL,
                    )
                    for i in range(200, 250)
                ]
                return (episodes, None)
            else:
                return ([], None)

        consolidation_service._episodic.recall_by_timerange.side_effect = recall_side_effect

        # Mock duplicate detection
        consolidation_service._find_duplicates = AsyncMock(return_value=[])

        # Run consolidation
        result = await consolidation_service._consolidate_light()

        # Verify pagination was used
        assert consolidation_service._episodic.count_by_timerange.called
        assert consolidation_service._episodic.recall_by_timerange.call_count == 2
        assert result["episodes_scanned"] == 250
        assert result["pages_loaded"] == 2

    @pytest.mark.asyncio
    async def test_consolidate_light_respects_limit(self, consolidation_service):
        """Test that light consolidation respects safety limit."""
        # Mock count to be very large
        consolidation_service._episodic.count_by_timerange.return_value = 20000

        # Mock paginated results to always return full pages
        def recall_side_effect(**kwargs):
            cursor = kwargs.get("cursor")
            offset = int(cursor) if cursor else 0

            if offset >= 10000:
                return ([], None)

            episodes = [
                Episode(
                    session_id="test",
                    content=f"Episode {i}",
                    embedding=[0.1] * 1024,
                    context=EpisodeContext(),
                    outcome=Outcome.NEUTRAL,
                )
                for i in range(offset, offset + 200)
            ]
            return (episodes, str(offset + 200))

        consolidation_service._episodic.recall_by_timerange.side_effect = recall_side_effect
        consolidation_service._find_duplicates = AsyncMock(return_value=[])

        # Run consolidation
        result = await consolidation_service._consolidate_light()

        # Verify limit was respected
        assert result["episodes_scanned"] == 10000
        assert result["pages_loaded"] == 50  # 10000 / 200

    @pytest.mark.asyncio
    async def test_consolidate_deep_uses_pagination(self, consolidation_service):
        """Test that deep consolidation uses paginated retrieval."""
        # Mock count
        consolidation_service._episodic.count_by_timerange.return_value = 400

        # Mock paginated results
        def recall_side_effect(**kwargs):
            cursor = kwargs.get("cursor")
            if cursor is None:
                episodes = [
                    Episode(
                        session_id="test",
                        content=f"Episode {i}",
                        embedding=[0.1] * 1024,
                        context=EpisodeContext(),
                        outcome=Outcome.NEUTRAL,
                    )
                    for i in range(200)
                ]
                return (episodes, "200")
            elif cursor == "200":
                episodes = [
                    Episode(
                        session_id="test",
                        content=f"Episode {i}",
                        embedding=[0.1] * 1024,
                        context=EpisodeContext(),
                        outcome=Outcome.NEUTRAL,
                    )
                    for i in range(200, 400)
                ]
                return (episodes, None)
            else:
                return ([], None)

        consolidation_service._episodic.recall_by_timerange.side_effect = recall_side_effect
        consolidation_service._cluster_episodes = AsyncMock(return_value=[])

        # Set min_occurrences
        consolidation_service.min_occurrences = 3

        # Run consolidation
        result = await consolidation_service._consolidate_deep()

        # Verify pagination was used
        assert consolidation_service._episodic.count_by_timerange.called
        assert consolidation_service._episodic.recall_by_timerange.call_count == 2
        assert result["pages_loaded"] == 2
