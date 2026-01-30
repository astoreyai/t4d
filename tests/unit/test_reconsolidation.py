"""
Unit tests for World Weaver memory reconsolidation module.

Tests the ReconsolidationEngine and related utilities.
"""

import pytest
import numpy as np
from uuid import uuid4
from datetime import datetime, timedelta

from ww.learning.reconsolidation import (
    ReconsolidationEngine,
    ReconsolidationUpdate,
    reconsolidate,
)


class TestReconsolidationUpdate:
    """Tests for ReconsolidationUpdate dataclass."""

    def test_creation(self):
        update = ReconsolidationUpdate(
            memory_id=uuid4(),
            query_embedding=np.random.randn(128),
            original_embedding=np.random.randn(128),
            updated_embedding=np.random.randn(128),
            outcome_score=0.8,
            advantage=0.3,
            learning_rate=0.01
        )
        assert update.outcome_score == 0.8
        assert update.advantage == 0.3

    def test_update_magnitude(self):
        original = np.zeros(128)
        updated = np.ones(128) * 0.1
        update = ReconsolidationUpdate(
            memory_id=uuid4(),
            query_embedding=np.random.randn(128),
            original_embedding=original,
            updated_embedding=updated,
            outcome_score=0.8,
            advantage=0.3,
            learning_rate=0.01
        )
        expected = np.linalg.norm(updated - original)
        assert abs(update.update_magnitude - expected) < 1e-6


class TestReconsolidationEngine:
    """Tests for ReconsolidationEngine."""

    @pytest.fixture
    def engine(self):
        return ReconsolidationEngine(
            base_learning_rate=0.01,
            max_update_magnitude=0.1,
            baseline=0.5,
            cooldown_hours=0.0  # Disable cooldown for tests
        )

    @pytest.fixture
    def random_embedding(self):
        emb = np.random.randn(128)
        return emb / np.linalg.norm(emb)

    def test_creation_default(self):
        engine = ReconsolidationEngine()
        assert engine.base_learning_rate == 0.01
        assert engine.max_update_magnitude == 0.1
        assert engine.baseline == 0.5

    def test_creation_custom(self):
        engine = ReconsolidationEngine(
            base_learning_rate=0.05,
            max_update_magnitude=0.2,
            baseline=0.6,
            cooldown_hours=2.0
        )
        assert engine.base_learning_rate == 0.05
        assert engine.max_update_magnitude == 0.2
        assert engine.baseline == 0.6
        assert engine.cooldown_hours == 2.0

    def test_compute_advantage_positive(self, engine):
        advantage = engine.compute_advantage(0.8)
        assert abs(advantage - 0.3) < 1e-10  # 0.8 - 0.5

    def test_compute_advantage_negative(self, engine):
        advantage = engine.compute_advantage(0.2)
        assert advantage == -0.3  # 0.2 - 0.5

    def test_compute_advantage_neutral(self, engine):
        advantage = engine.compute_advantage(0.5)
        assert advantage == 0.0

    def test_should_update_first_time(self, engine):
        mem_id = uuid4()
        assert engine.should_update(mem_id) is True

    def test_should_update_within_lability_window(self):
        """BIO-MAJOR-001: Memory should be updatable within lability window."""
        engine = ReconsolidationEngine(lability_window_hours=1.0)
        mem_id = uuid4()

        # Trigger lability (simulate retrieval)
        engine.trigger_lability(mem_id)

        # Should update (within lability window)
        assert engine.should_update(mem_id) is True

    def test_should_not_update_after_lability_window(self):
        """BIO-MAJOR-001: Memory should NOT be updatable after lability window closes."""
        engine = ReconsolidationEngine(lability_window_hours=1.0)
        mem_id = uuid4()

        # Simulate retrieval 2 hours ago (outside lability window)
        engine._last_retrieval[str(mem_id)] = datetime.now() - timedelta(hours=2)

        # Should NOT update (lability window has closed, memory restabilized)
        assert engine.should_update(mem_id) is False

    def test_reconsolidate_positive_outcome(self, engine, random_embedding):
        mem_id = uuid4()
        query_emb = np.random.randn(128)
        query_emb = query_emb / np.linalg.norm(query_emb)
        memory_emb = random_embedding

        new_emb = engine.reconsolidate(
            memory_id=mem_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.9
        )

        assert new_emb is not None
        # Should move toward query
        old_dist = np.linalg.norm(query_emb - memory_emb)
        new_dist = np.linalg.norm(query_emb - new_emb)
        assert new_dist < old_dist

    def test_reconsolidate_negative_outcome(self, engine, random_embedding):
        mem_id = uuid4()
        query_emb = np.random.randn(128)
        query_emb = query_emb / np.linalg.norm(query_emb)
        memory_emb = random_embedding

        new_emb = engine.reconsolidate(
            memory_id=mem_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.1
        )

        assert new_emb is not None
        # Should move away from query
        old_dist = np.linalg.norm(query_emb - memory_emb)
        new_dist = np.linalg.norm(query_emb - new_emb)
        assert new_dist > old_dist

    def test_reconsolidate_neutral_skips(self, engine, random_embedding):
        mem_id = uuid4()
        query_emb = random_embedding

        new_emb = engine.reconsolidate(
            memory_id=mem_id,
            memory_embedding=random_embedding,
            query_embedding=query_emb,
            outcome_score=0.5  # Neutral
        )

        # Should skip update
        assert new_emb is None

    def test_reconsolidate_normalized_output(self, engine, random_embedding):
        mem_id = uuid4()
        query_emb = np.random.randn(128)

        new_emb = engine.reconsolidate(
            memory_id=mem_id,
            memory_embedding=random_embedding,
            query_embedding=query_emb,
            outcome_score=0.9
        )

        # Output should be normalized
        assert abs(np.linalg.norm(new_emb) - 1.0) < 1e-6

    def test_reconsolidate_max_update_magnitude(self, engine, random_embedding):
        mem_id = uuid4()
        # Very different query
        query_emb = -random_embedding

        # Use high learning rate
        new_emb = engine.reconsolidate(
            memory_id=mem_id,
            memory_embedding=random_embedding,
            query_embedding=query_emb,
            outcome_score=1.0,
            learning_rate=1.0  # Very high
        )

        # Update should be clipped
        update_magnitude = np.linalg.norm(new_emb - random_embedding)
        assert update_magnitude <= engine.max_update_magnitude + 0.01

    def test_reconsolidate_records_history(self, engine, random_embedding):
        mem_id = uuid4()
        query_emb = np.random.randn(128)

        engine.reconsolidate(
            memory_id=mem_id,
            memory_embedding=random_embedding,
            query_embedding=query_emb,
            outcome_score=0.8
        )

        history = engine.get_update_history()
        assert len(history) == 1
        assert history[0].memory_id == mem_id

    def test_reconsolidate_same_embedding_skips(self, engine, random_embedding):
        mem_id = uuid4()

        # Query is same as memory
        new_emb = engine.reconsolidate(
            memory_id=mem_id,
            memory_embedding=random_embedding,
            query_embedding=random_embedding.copy(),
            outcome_score=0.9
        )

        # Should skip (no direction to update)
        assert new_emb is None

    def test_batch_reconsolidate(self, engine):
        memories = [
            (uuid4(), np.random.randn(128) / np.sqrt(128))
            for _ in range(3)
        ]
        query_emb = np.random.randn(128)

        updates = engine.batch_reconsolidate(
            memories=memories,
            query_embedding=query_emb,
            outcome_score=0.8
        )

        assert len(updates) == 3
        for mem_id, new_emb in updates.items():
            assert new_emb is not None
            assert abs(np.linalg.norm(new_emb) - 1.0) < 0.01

    def test_batch_reconsolidate_per_memory_rewards(self, engine):
        mem1 = uuid4()
        mem2 = uuid4()
        memories = [
            (mem1, np.random.randn(128) / np.sqrt(128)),
            (mem2, np.random.randn(128) / np.sqrt(128)),
        ]
        query_emb = np.random.randn(128)

        updates = engine.batch_reconsolidate(
            memories=memories,
            query_embedding=query_emb,
            outcome_score=0.5,  # Neutral overall
            per_memory_rewards={
                str(mem1): 0.9,  # mem1 was helpful
                str(mem2): 0.1,  # mem2 was not
            }
        )

        # Both should update (different per-memory rewards)
        assert len(updates) == 2

    def test_get_update_history_filtered(self, engine, random_embedding):
        mem1 = uuid4()
        mem2 = uuid4()

        for mem_id in [mem1, mem2, mem1, mem2]:
            engine.reconsolidate(
                memory_id=mem_id,
                memory_embedding=random_embedding,
                query_embedding=np.random.randn(128),
                outcome_score=0.8
            )

        history = engine.get_update_history(memory_id=mem1)
        assert len(history) == 2
        assert all(u.memory_id == mem1 for u in history)

    def test_get_update_history_limited(self, engine, random_embedding):
        for _ in range(10):
            engine.reconsolidate(
                memory_id=uuid4(),
                memory_embedding=random_embedding,
                query_embedding=np.random.randn(128),
                outcome_score=0.8
            )

        history = engine.get_update_history(limit=5)
        assert len(history) == 5

    def test_get_stats_empty(self, engine):
        stats = engine.get_stats()
        assert stats["total_updates"] == 0
        assert stats["positive_updates"] == 0
        assert stats["negative_updates"] == 0

    def test_get_stats(self, engine, random_embedding):
        # Positive updates
        for _ in range(3):
            engine.reconsolidate(
                memory_id=uuid4(),
                memory_embedding=random_embedding,
                query_embedding=np.random.randn(128),
                outcome_score=0.8
            )

        # Negative updates
        for _ in range(2):
            engine.reconsolidate(
                memory_id=uuid4(),
                memory_embedding=random_embedding,
                query_embedding=np.random.randn(128),
                outcome_score=0.2
            )

        stats = engine.get_stats()
        assert stats["total_updates"] == 5
        assert stats["positive_updates"] == 3
        assert stats["negative_updates"] == 2
        assert stats["avg_magnitude"] > 0

    def test_clear_history(self, engine, random_embedding):
        engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=random_embedding,
            query_embedding=np.random.randn(128),
            outcome_score=0.8
        )

        engine.clear_history()

        assert len(engine.get_update_history()) == 0
        assert engine.get_stats()["total_updates"] == 0


class TestReconsolidateFunction:
    """Tests for stateless reconsolidate function."""

    def test_positive_outcome_moves_toward_query(self):
        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        new_emb = reconsolidate(
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.8,  # Positive
            learning_rate=0.1
        )

        # Should move toward query (positive y component)
        assert new_emb[1] > 0

    def test_negative_outcome_moves_away_from_query(self):
        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        new_emb = reconsolidate(
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.2,  # Negative
            learning_rate=0.1
        )

        # Should move away from query (negative y component)
        assert new_emb[1] < 0

    def test_normalized_output(self):
        memory_emb = np.random.randn(128)
        query_emb = np.random.randn(128)

        new_emb = reconsolidate(
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.8
        )

        assert abs(np.linalg.norm(new_emb) - 1.0) < 1e-6

    def test_same_embedding_returns_copy(self):
        emb = np.array([1.0, 0.0, 0.0])

        new_emb = reconsolidate(
            memory_embedding=emb,
            query_embedding=emb.copy(),
            outcome_score=0.8
        )

        assert np.allclose(new_emb, emb)

    def test_learning_rate_scales_update(self):
        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        # Small learning rate
        new_emb_small = reconsolidate(
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.8,
            learning_rate=0.01
        )

        # Large learning rate
        new_emb_large = reconsolidate(
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.8,
            learning_rate=0.1
        )

        # Larger LR should cause larger update
        small_update = np.linalg.norm(new_emb_small - memory_emb)
        large_update = np.linalg.norm(new_emb_large - memory_emb)
        assert large_update > small_update
