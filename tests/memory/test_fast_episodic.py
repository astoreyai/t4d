"""
Unit tests for Fast Episodic Store implementation.
"""

import pytest
import torch
import time
from uuid import uuid4
from datetime import datetime

from ww.memory.fast_episodic import FastEpisodicStore, FastEpisodicConfig, MAX_CAPACITY
from ww.core.types import Episode, EpisodeContext, Outcome


class TestFastEpisodicStore:
    """Tests for Fast Episodic Store."""

    @pytest.fixture
    def store(self):
        """Create default FES."""
        return FastEpisodicStore(
            capacity=100,
            learning_rate=0.1,
            consolidation_threshold=0.5
        )

    @pytest.fixture
    def sample_episode(self):
        """Create sample episode."""
        return Episode(
            id=uuid4(),
            session_id="test-session",
            content="Test episode content for testing",
            embedding=[0.1] * 1024,
            emotional_valence=0.7,
            outcome=Outcome.SUCCESS
        )

    def test_initialization(self, store):
        """Store initializes correctly."""
        assert store.capacity == 100
        assert store.learning_rate == 0.1
        assert store.count == 0

    def test_security_validation(self):
        """Security limits are enforced."""
        # Capacity limit
        with pytest.raises(ValueError, match="MAX_CAPACITY"):
            FastEpisodicStore(capacity=MAX_CAPACITY + 1)

        # Invalid capacity
        with pytest.raises(ValueError, match="positive"):
            FastEpisodicStore(capacity=0)

        # Invalid learning rate
        with pytest.raises(ValueError, match="learning_rate"):
            FastEpisodicStore(learning_rate=2.0)

    def test_write_episode(self, store, sample_episode):
        """Writing episode works correctly."""
        encoding = torch.randn(1024)

        result = store.write(sample_episode, encoding)

        assert result["stored"] == True
        assert "episode_id" in result
        assert 0 <= result["salience"] <= 1
        assert store.count == 1

    def test_one_shot_learning(self, store):
        """Single exposure creates retrievable memory."""
        # Create and store episode
        episode = Episode(
            id=uuid4(),
            session_id="test",
            content="One-shot learning test",
            embedding=[1.0] * 512 + [0.0] * 512,  # Distinctive pattern
            emotional_valence=0.8
        )
        encoding = torch.tensor([1.0] * 512 + [0.0] * 512)

        store.write(episode, encoding)

        # Should be retrievable immediately with similar cue
        cue = torch.tensor([0.9] * 512 + [0.1] * 512)
        results = store.read(cue, top_k=1)

        assert len(results) == 1
        retrieved_episode, similarity = results[0]
        assert retrieved_episode.content == "One-shot learning test"
        assert similarity > 0.8

    def test_write_multiple_episodes(self, store):
        """Multiple episodes can be stored."""
        for i in range(10):
            episode = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                emotional_valence=0.5
            )
            encoding = torch.randn(1024)
            store.write(episode, encoding)

        assert store.count == 10

    def test_read_top_k(self, store):
        """Read returns top-k most similar episodes."""
        # Store diverse episodes
        encodings = []
        for i in range(5):
            episode = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}"
            )
            encoding = torch.randn(1024)
            encodings.append(encoding)
            store.write(episode, encoding)

        # Query with first encoding
        results = store.read(encodings[0], top_k=3)

        assert len(results) == 3
        # First result should be most similar
        assert results[0][1] >= results[1][1]
        assert results[1][1] >= results[2][1]

    def test_read_empty_store(self, store):
        """Reading from empty store returns empty list."""
        cue = torch.randn(1024)
        results = store.read(cue, top_k=5)

        assert results == []

    def test_access_count_tracking(self, store, sample_episode):
        """Access counts are tracked correctly."""
        encoding = torch.randn(1024)
        result = store.write(sample_episode, encoding)
        episode_id = result["episode_id"]

        # Initial access count
        assert store.access_counts[episode_id] == 0

        # Read multiple times
        for _ in range(3):
            store.read(encoding, top_k=1)

        # Access count should increase
        assert store.access_counts[episode_id] == 3

    def test_capacity_management(self):
        """Capacity is enforced through eviction."""
        store = FastEpisodicStore(capacity=5)

        # Fill to capacity
        for i in range(5):
            episode = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                emotional_valence=0.5
            )
            store.write(episode, torch.randn(1024))

        assert store.count == 5

        # Add one more - should trigger eviction
        extra = Episode(
            id=uuid4(),
            session_id="test",
            content="Extra episode",
            emotional_valence=0.9  # High salience
        )
        store.write(extra, torch.randn(1024))

        # Still at capacity
        assert store.count == 5

    def test_salience_based_eviction(self):
        """Low salience episodes are evicted first."""
        store = FastEpisodicStore(capacity=3)

        # Add low salience episode
        low_sal = Episode(
            id=uuid4(),
            session_id="test",
            content="Low salience",
            emotional_valence=0.1
        )
        store.write(
            low_sal,
            torch.randn(1024),
            neuromod_state={'dopamine': 0.1, 'norepinephrine': 0.1, 'acetylcholine': 0.1}
        )

        # Add high salience episodes
        for i in range(2):
            high_sal = Episode(
                id=uuid4(),
                session_id="test",
                content=f"High salience {i}",
                emotional_valence=0.9
            )
            store.write(
                high_sal,
                torch.randn(1024),
                neuromod_state={'dopamine': 0.9, 'norepinephrine': 0.9, 'acetylcholine': 0.9}
            )

        # Add one more to trigger eviction
        trigger = Episode(
            id=uuid4(),
            session_id="test",
            content="Trigger",
            emotional_valence=0.8
        )
        store.write(
            trigger,
            torch.randn(1024),
            neuromod_state={'dopamine': 0.8, 'norepinephrine': 0.8, 'acetylcholine': 0.8}
        )

        # Low salience should have been evicted
        contents = [e.episode.content for e in store.entries.values()]
        assert "Low salience" not in contents

    def test_consolidation_candidates(self, store):
        """Consolidation candidates are correctly identified."""
        # Store episodes with varying properties
        for i in range(5):
            episode = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                emotional_valence=0.9 if i == 0 else 0.3  # First has high salience
            )
            encoding = torch.randn(1024)
            result = store.write(
                episode, encoding,
                neuromod_state={'dopamine': 0.9 if i == 0 else 0.2, 'norepinephrine': 0.5, 'acetylcholine': 0.5}
            )

            # Access first episode multiple times
            if i == 0:
                episode_id = result["episode_id"]
                for _ in range(10):
                    store.read(encoding, top_k=1)

        candidates = store.get_consolidation_candidates()

        # High-salience, high-access episode should be top candidate
        if candidates:
            top_candidate = candidates[0]
            assert top_candidate[1].content == "Episode 0"

    def test_mark_consolidated(self, store, sample_episode):
        """Mark consolidated works correctly."""
        encoding = torch.randn(1024)
        result = store.write(sample_episode, encoding)
        episode_id = result["episode_id"]

        # Mark as consolidated
        assert store.mark_consolidated(episode_id) == True
        assert store.entries[episode_id].consolidated == True

        # Non-existent ID
        assert store.mark_consolidated("nonexistent") == False

    def test_remove_episode(self, store, sample_episode):
        """Episode removal works correctly."""
        encoding = torch.randn(1024)
        result = store.write(sample_episode, encoding)
        episode_id = result["episode_id"]

        assert store.count == 1
        assert store.remove(episode_id) == True
        assert store.count == 0

        # Can't remove twice
        assert store.remove(episode_id) == False

    def test_clear(self, store):
        """Clear removes all entries."""
        for i in range(5):
            episode = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}"
            )
            store.write(episode, torch.randn(1024))

        assert store.count == 5
        store.clear()
        assert store.count == 0

    def test_get_stats(self, store, sample_episode):
        """Statistics are computed correctly."""
        encoding = torch.randn(1024)
        store.write(sample_episode, encoding)
        store.read(encoding, top_k=1)

        stats = store.get_stats()

        assert stats["count"] == 1
        assert stats["capacity"] == 100
        assert stats["total_writes"] == 1
        assert stats["total_reads"] == 1
        assert "average_salience" in stats

    def test_neuromod_salience_computation(self, store):
        """Neuromodulator-based salience is computed correctly."""
        episode = Episode(
            id=uuid4(),
            session_id="test",
            content="Test",
            emotional_valence=0.5
        )

        # High neuromod state -> high salience
        result_high = store.write(
            episode, torch.randn(1024),
            neuromod_state={'dopamine': 1.0, 'norepinephrine': 1.0, 'acetylcholine': 1.0}
        )

        store.clear()

        # Low neuromod state -> low salience
        result_low = store.write(
            episode, torch.randn(1024),
            neuromod_state={'dopamine': 0.0, 'norepinephrine': 0.0, 'acetylcholine': 0.0}
        )

        assert result_high["salience"] > result_low["salience"]

    def test_consolidated_not_evicted(self):
        """Consolidated episodes are protected from eviction."""
        store = FastEpisodicStore(capacity=2)

        # Store and consolidate first episode
        first = Episode(
            id=uuid4(),
            session_id="test",
            content="First - consolidated",
            emotional_valence=0.1  # Low salience
        )
        result1 = store.write(first, torch.randn(1024))
        store.mark_consolidated(result1["episode_id"])

        # Store second episode
        second = Episode(
            id=uuid4(),
            session_id="test",
            content="Second",
            emotional_valence=0.5
        )
        store.write(second, torch.randn(1024))

        # Add third to trigger eviction
        third = Episode(
            id=uuid4(),
            session_id="test",
            content="Third",
            emotional_valence=0.9
        )
        store.write(third, torch.randn(1024))

        # First (consolidated) should still be there
        contents = [e.episode.content for e in store.entries.values()]
        assert "First - consolidated" in contents

    def test_learning_rate_difference(self):
        """FES learning rate is 100x faster than semantic default."""
        # This is a conceptual test - FES uses 0.1 vs semantic's 0.001
        fes = FastEpisodicStore(learning_rate=0.1)

        # FES should be able to store immediately without training
        episode = Episode(
            id=uuid4(),
            session_id="test",
            content="Instant learning"
        )
        encoding = torch.randn(1024)

        result = fes.write(episode, encoding)
        assert result["stored"] == True

        # Immediately retrievable
        results = fes.read(encoding, top_k=1)
        assert len(results) == 1
        assert results[0][1] > 0.99  # Near-perfect match

    def test_interference_resistance(self, store):
        """Similar patterns can be stored and retrieved distinctly."""
        # Create two similar but distinct episodes
        base_encoding = torch.randn(1024)

        episode1 = Episode(
            id=uuid4(),
            session_id="test",
            content="Episode A - unique content"
        )
        encoding1 = base_encoding + 0.1 * torch.randn(1024)
        store.write(episode1, encoding1)

        episode2 = Episode(
            id=uuid4(),
            session_id="test",
            content="Episode B - different content"
        )
        encoding2 = base_encoding + 0.1 * torch.randn(1024)
        store.write(episode2, encoding2)

        # Query with first encoding
        results1 = store.read(encoding1, top_k=1)
        assert results1[0][0].content == "Episode A - unique content"

        # Query with second encoding
        results2 = store.read(encoding2, top_k=1)
        assert results2[0][0].content == "Episode B - different content"


class TestFastEpisodicConfig:
    """Tests for FES configuration."""

    def test_default_config(self):
        """Default config has expected values."""
        config = FastEpisodicConfig()

        assert config.capacity == 10000
        assert config.learning_rate == 0.1
        assert config.eviction_strategy == 'lru_salience'
        assert config.consolidation_threshold == 0.7
        assert 'da' in config.salience_weights

    def test_config_creates_store(self):
        """Config can be used to create store."""
        config = FastEpisodicConfig(capacity=500, learning_rate=0.2)

        store = FastEpisodicStore(
            capacity=config.capacity,
            learning_rate=config.learning_rate,
            eviction_strategy=config.eviction_strategy,
            consolidation_threshold=config.consolidation_threshold,
            salience_weights=config.salience_weights
        )

        assert store.capacity == 500
        assert store.learning_rate == 0.2
