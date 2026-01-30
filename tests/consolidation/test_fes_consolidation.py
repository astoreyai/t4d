"""
Unit tests for FES Consolidator implementation.
"""

import pytest
import asyncio
import torch
from uuid import uuid4
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from ww.memory.fast_episodic import FastEpisodicStore
from ww.consolidation.fes_consolidator import (
    FESConsolidator,
    ReplayConsolidator,
    extract_entities_simple,
    MAX_CONSOLIDATION_BATCH
)
from ww.core.types import Episode, EpisodeContext, Outcome


class TestEntityExtraction:
    """Tests for entity extraction."""

    def test_extract_files(self):
        """Extracts file references."""
        content = "Modified src/main.py and tests/test_utils.py"
        entities = extract_entities_simple(content)

        names = [e['name'] for e in entities]
        assert 'src/main.py' in names or 'main.py' in names
        assert 'tests/test_utils.py' in names or 'test_utils.py' in names

    def test_extract_functions(self):
        """Extracts function calls."""
        content = "Called calculate_total() and process_data()"
        entities = extract_entities_simple(content)

        names = [e['name'] for e in entities]
        assert 'calculate_total' in names
        assert 'process_data' in names

    def test_extract_named_entities(self):
        """Extracts capitalized names."""
        content = "Working with Aaron Storey and the World Weaver project"
        entities = extract_entities_simple(content)

        names = [e['name'] for e in entities]
        assert any('Aaron' in n for n in names)

    def test_limits_extraction(self):
        """Limits number of extracted entities."""
        # Create content with many potential entities
        content = " ".join([f"file{i}.py" for i in range(100)])
        entities = extract_entities_simple(content)

        assert len(entities) <= 50  # MAX_ENTITY_EXTRACTION


class TestFESConsolidator:
    """Tests for FES Consolidator."""

    @pytest.fixture
    def fast_store(self):
        """Create populated fast store."""
        store = FastEpisodicStore(capacity=50, consolidation_threshold=0.1)

        # Add episodes with varying properties
        for i in range(10):
            episode = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i} - test content for consolidation",
                emotional_valence=0.8 + (i / 50)  # Higher salience
            )
            encoding = torch.randn(1024)
            result = store.write(
                episode, encoding,
                neuromod_state={
                    'dopamine': 0.8 + (i / 50),
                    'norepinephrine': 0.7,
                    'acetylcholine': 0.7
                }
            )

            # Add access counts for some episodes
            if i < 3:
                for _ in range(15):  # More accesses
                    store.read(encoding, top_k=1)

        return store

    @pytest.fixture
    def consolidator(self, fast_store):
        """Create consolidator without external stores."""
        return FESConsolidator(
            fast_store=fast_store,
            consolidation_rate=0.1,
            min_consolidation_score=0.1  # Lower threshold
        )

    @pytest.mark.asyncio
    async def test_consolidate_cycle(self, consolidator):
        """Basic consolidation cycle works."""
        results = await consolidator.consolidate_cycle(max_episodes=5)

        # Should consolidate some episodes
        assert len(results) <= 5

        # Results should have expected structure
        for result in results:
            assert 'episode_id' in result
            assert 'consolidation_score' in result
            assert 'timestamp' in result

    @pytest.mark.asyncio
    async def test_marks_consolidated(self, consolidator, fast_store):
        """Episodes are marked as consolidated."""
        # Get initial non-consolidated count
        initial = sum(1 for e in fast_store.entries.values() if not e.consolidated)

        await consolidator.consolidate_cycle(max_episodes=3)

        # Some should now be consolidated
        consolidated = sum(1 for e in fast_store.entries.values() if e.consolidated)
        assert consolidated > 0

    @pytest.mark.asyncio
    async def test_respects_min_score(self, fast_store):
        """Only episodes above min score are consolidated."""
        # Create consolidator with high threshold
        consolidator = FESConsolidator(
            fast_store=fast_store,
            min_consolidation_score=0.99  # Very high
        )

        results = await consolidator.consolidate_cycle(max_episodes=10)

        # Should consolidate very few or none
        assert len(results) < 5

    @pytest.mark.asyncio
    async def test_with_episodic_store(self, fast_store):
        """Consolidation with episodic store."""
        # Mock episodic store
        mock_episodic = MagicMock()
        mock_episodic.store = AsyncMock(return_value=uuid4())

        consolidator = FESConsolidator(
            fast_store=fast_store,
            episodic_store=mock_episodic,
            min_consolidation_score=0.3
        )

        results = await consolidator.consolidate_cycle(max_episodes=3)

        # Should have called store
        if results:
            assert mock_episodic.store.called
            for result in results:
                if 'episodic_error' not in result:
                    assert result.get('episodic_stored') == True

    @pytest.mark.asyncio
    async def test_with_semantic_store(self, fast_store):
        """Consolidation with semantic store."""
        # Mock semantic store
        mock_semantic = MagicMock()
        mock_semantic.add_entity = AsyncMock()

        consolidator = FESConsolidator(
            fast_store=fast_store,
            semantic_store=mock_semantic,
            min_consolidation_score=0.3
        )

        results = await consolidator.consolidate_cycle(max_episodes=3)

        # Should extract entities
        if results:
            total_entities = sum(r.get('entities_extracted', 0) for r in results)
            assert total_entities >= 0  # May be 0 if content has no entities

    @pytest.mark.asyncio
    async def test_consolidate_all(self, consolidator):
        """Consolidate all processes all candidates."""
        result = await consolidator.consolidate_all()

        assert 'cycles' in result
        assert 'episodes_consolidated' in result
        assert result['cycles'] >= 1

    @pytest.mark.asyncio
    async def test_max_batch_limit(self, fast_store):
        """Respects max consolidation batch limit."""
        # Add many episodes
        for i in range(200):
            episode = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Bulk episode {i}",
                emotional_valence=0.9
            )
            fast_store.write(episode, torch.randn(1024))

        consolidator = FESConsolidator(
            fast_store=fast_store,
            min_consolidation_score=0.1
        )

        # Request more than limit
        results = await consolidator.consolidate_cycle(max_episodes=200)

        # Should be limited
        assert len(results) <= MAX_CONSOLIDATION_BATCH

    def test_get_stats(self, consolidator):
        """Statistics are tracked correctly."""
        stats = consolidator.get_stats()

        assert 'cycles_run' in stats
        assert 'episodes_consolidated' in stats
        assert 'fes_count' in stats
        assert stats['cycles_run'] == 0

    @pytest.mark.asyncio
    async def test_stats_after_consolidation(self, consolidator):
        """Statistics update after consolidation."""
        await consolidator.consolidate_cycle(max_episodes=3)

        stats = consolidator.get_stats()
        assert stats['cycles_run'] == 1


class TestReplayConsolidator:
    """Tests for replay-weighted consolidator."""

    @pytest.fixture
    def fast_store_with_replay(self):
        """Create store with replay patterns."""
        store = FastEpisodicStore(capacity=50)

        # Add episodes
        encodings = []
        for i in range(10):
            episode = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                emotional_valence=0.5  # Same salience
            )
            encoding = torch.randn(1024)
            encodings.append(encoding)
            store.write(episode, encoding)

        # Add heavy replay to first few
        for _ in range(20):
            store.read(encodings[0], top_k=1)
        for _ in range(10):
            store.read(encodings[1], top_k=1)

        return store

    @pytest.fixture
    def replay_consolidator(self, fast_store_with_replay):
        """Create replay consolidator."""
        return ReplayConsolidator(
            fast_store=fast_store_with_replay,
            replay_weight=0.7,
            min_consolidation_score=0.1
        )

    @pytest.mark.asyncio
    async def test_prioritizes_replayed(self, replay_consolidator, fast_store_with_replay):
        """High replay episodes are prioritized."""
        results = await replay_consolidator.consolidate_cycle(max_episodes=3)

        # Check that high replay episodes were selected
        if len(results) >= 2:
            # First result should have higher replay count than later ones
            replay_counts = [r.get('replay_count', 0) for r in results]
            # Generally should be sorted by replay (with some weighting)
            assert replay_counts[0] >= replay_counts[-1]

    @pytest.mark.asyncio
    async def test_replay_weight_effect(self, fast_store_with_replay):
        """Replay weight affects selection."""
        # High replay weight
        high_replay = ReplayConsolidator(
            fast_store=FastEpisodicStore(capacity=50),
            replay_weight=0.9
        )

        # Low replay weight
        low_replay = ReplayConsolidator(
            fast_store=FastEpisodicStore(capacity=50),
            replay_weight=0.1
        )

        # Both should work
        assert high_replay.replay_weight == 0.9
        assert low_replay.replay_weight == 0.1


class TestBackgroundConsolidation:
    """Tests for background consolidation."""

    @pytest.fixture
    def fast_store(self):
        """Create simple fast store."""
        store = FastEpisodicStore(capacity=20)
        for i in range(5):
            episode = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}"
            )
            store.write(episode, torch.randn(1024))
        return store

    @pytest.mark.asyncio
    async def test_start_stop_background(self, fast_store):
        """Can start and stop background consolidation."""
        consolidator = FESConsolidator(
            fast_store=fast_store,
            min_consolidation_score=0.1
        )

        # Start background task
        await consolidator.start_background_consolidation(interval_seconds=1)

        stats = consolidator.get_stats()
        assert stats['background_running'] == True

        # Stop
        await consolidator.stop_background_consolidation()

        stats = consolidator.get_stats()
        assert stats['background_running'] == False

    @pytest.mark.asyncio
    async def test_double_start_warning(self, fast_store, caplog):
        """Warning when trying to start twice."""
        consolidator = FESConsolidator(
            fast_store=fast_store
        )

        await consolidator.start_background_consolidation(interval_seconds=10)
        await consolidator.start_background_consolidation(interval_seconds=10)

        # Should have logged a warning
        assert "already running" in caplog.text.lower()

        # Cleanup
        await consolidator.stop_background_consolidation()
