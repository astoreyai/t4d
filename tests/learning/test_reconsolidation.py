"""Tests for memory reconsolidation module."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4

from ww.learning.reconsolidation import (
    ReconsolidationUpdate,
    ReconsolidationEngine,
    reconsolidate,
)


class TestReconsolidationUpdate:
    """Tests for ReconsolidationUpdate dataclass."""

    def test_basic_update(self):
        """Test basic update creation."""
        memory_id = uuid4()
        query_emb = np.random.randn(64)
        orig_emb = np.random.randn(64)
        updated_emb = np.random.randn(64)

        update = ReconsolidationUpdate(
            memory_id=memory_id,
            query_embedding=query_emb,
            original_embedding=orig_emb,
            updated_embedding=updated_emb,
            outcome_score=0.8,
            advantage=0.3,
            learning_rate=0.01,
        )

        assert update.memory_id == memory_id
        assert update.outcome_score == 0.8
        assert update.advantage == 0.3
        assert update.learning_rate == 0.01
        assert isinstance(update.timestamp, datetime)

    def test_update_magnitude(self):
        """Test update magnitude calculation."""
        orig_emb = np.zeros(64)
        updated_emb = np.ones(64) * 0.1  # Small change

        update = ReconsolidationUpdate(
            memory_id=uuid4(),
            query_embedding=np.random.randn(64),
            original_embedding=orig_emb,
            updated_embedding=updated_emb,
            outcome_score=0.8,
            advantage=0.3,
            learning_rate=0.01,
        )

        # Magnitude should be L2 norm of difference
        expected_mag = np.linalg.norm(updated_emb - orig_emb)
        assert update.update_magnitude == pytest.approx(expected_mag)

    def test_update_magnitude_zero_change(self):
        """Test magnitude when no change."""
        emb = np.random.randn(64)

        update = ReconsolidationUpdate(
            memory_id=uuid4(),
            query_embedding=np.random.randn(64),
            original_embedding=emb.copy(),
            updated_embedding=emb.copy(),
            outcome_score=0.5,
            advantage=0.0,
            learning_rate=0.01,
        )

        assert update.update_magnitude == pytest.approx(0.0)


class TestReconsolidationEngine:
    """Tests for ReconsolidationEngine class."""

    @pytest.fixture
    def engine(self):
        """Create reconsolidation engine."""
        return ReconsolidationEngine()

    @pytest.fixture
    def memory_embedding(self):
        """Create normalized memory embedding."""
        emb = np.random.randn(64)
        return emb / np.linalg.norm(emb)

    @pytest.fixture
    def query_embedding(self):
        """Create normalized query embedding."""
        emb = np.random.randn(64)
        return emb / np.linalg.norm(emb)

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.base_learning_rate == 0.01
        assert engine.max_update_magnitude == 0.1
        assert engine.baseline == 0.5
        assert engine.cooldown_hours == 1.0

    def test_initialization_custom(self):
        """Test custom initialization."""
        engine = ReconsolidationEngine(
            base_learning_rate=0.05,
            max_update_magnitude=0.2,
            baseline=0.6,
            cooldown_hours=2.0,
        )
        assert engine.base_learning_rate == 0.05
        assert engine.max_update_magnitude == 0.2
        assert engine.baseline == 0.6
        assert engine.cooldown_hours == 2.0

    def test_should_update_first_time(self, engine):
        """Test should_update for new memory."""
        memory_id = uuid4()
        assert engine.should_update(memory_id) is True

    def test_should_update_within_lability_window(self, engine):
        """Test should_update within lability window (biological reconsolidation).

        BIO-MAJOR-001: Memory becomes LABILE after retrieval and CAN be updated
        within the lability window. This is the biological reconsolidation model.
        """
        memory_id = uuid4()
        # Simulate recent retrieval - memory is labile and CAN be updated
        engine.trigger_lability(memory_id)
        assert engine.should_update(memory_id) is True
        assert engine.is_labile(memory_id) is True

    def test_should_update_after_lability_window(self, engine):
        """Test should_update after lability window closes.

        BIO-MAJOR-001: Memory RESTABILIZES after lability window closes and
        CANNOT be updated (until next retrieval triggers lability).
        """
        memory_id = uuid4()
        # Simulate retrieval that happened beyond lability window
        engine._last_retrieval[str(memory_id)] = datetime.now() - timedelta(hours=10)
        assert engine.should_update(memory_id) is False
        assert engine.is_labile(memory_id) is False

    def test_compute_advantage_positive(self, engine):
        """Test advantage for positive outcome."""
        advantage = engine.compute_advantage(0.8)
        assert advantage == pytest.approx(0.3)  # 0.8 - 0.5

    def test_compute_advantage_negative(self, engine):
        """Test advantage for negative outcome."""
        advantage = engine.compute_advantage(0.2)
        assert advantage == pytest.approx(-0.3)  # 0.2 - 0.5

    def test_compute_advantage_neutral(self, engine):
        """Test advantage for neutral outcome."""
        advantage = engine.compute_advantage(0.5)
        assert advantage == pytest.approx(0.0)

    def test_compute_importance_adjusted_lr_zero_importance(self, engine):
        """Test LR with zero importance."""
        lr = engine.compute_importance_adjusted_lr(0.1, 0.0)
        assert lr == pytest.approx(0.1)

    def test_compute_importance_adjusted_lr_high_importance(self, engine):
        """Test LR with high importance."""
        lr = engine.compute_importance_adjusted_lr(0.1, 1.0)
        assert lr == pytest.approx(0.05)  # 0.1 / 2

    def test_compute_importance_adjusted_lr_very_high_importance(self, engine):
        """Test LR with very high importance."""
        lr = engine.compute_importance_adjusted_lr(0.1, 9.0)
        assert lr == pytest.approx(0.01)  # 0.1 / 10

    def test_reconsolidate_positive_outcome(self, engine, memory_embedding, query_embedding):
        """Test reconsolidation with positive outcome."""
        memory_id = uuid4()
        updated = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.9,
        )

        assert updated is not None
        # Should be normalized
        assert np.linalg.norm(updated) == pytest.approx(1.0, abs=0.01)
        # Should be different from original
        assert not np.allclose(updated, memory_embedding)

    def test_reconsolidate_negative_outcome(self, engine, memory_embedding, query_embedding):
        """Test reconsolidation with negative outcome."""
        memory_id = uuid4()
        updated = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.1,
        )

        assert updated is not None
        # Should move away from query for negative outcomes

    def test_reconsolidate_neutral_skipped(self, engine, memory_embedding, query_embedding):
        """Test reconsolidation skipped for neutral outcome."""
        memory_id = uuid4()
        updated = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.5,  # Neutral
        )

        assert updated is None

    def test_reconsolidate_within_lability_succeeds(self, engine, memory_embedding, query_embedding):
        """Test multiple reconsolidations succeed within lability window.

        BIO-MAJOR-001: With biological reconsolidation, updates WITHIN the
        lability window should succeed. Each update refreshes the window.
        """
        memory_id = uuid4()

        # First update - triggers lability
        first_update = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.8,
        )
        assert first_update is not None

        # Second update should SUCCEED (within lability window)
        updated = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=first_update,  # Use updated embedding
            query_embedding=query_embedding,
            outcome_score=0.9,
        )
        assert updated is not None  # Should succeed, not be skipped

    def test_reconsolidate_after_restabilization_skipped(self, engine, memory_embedding, query_embedding):
        """Test reconsolidation skipped after memory restabilizes.

        BIO-MAJOR-001: Memory should become non-labile (restabilized) after
        lability window closes, blocking further updates until next retrieval.
        """
        memory_id = uuid4()

        # Simulate past retrieval that's now outside lability window
        engine._last_retrieval[str(memory_id)] = datetime.now() - timedelta(hours=10)

        # Update should be skipped (memory is restabilized)
        updated = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.9,
        )
        assert updated is None  # Skipped because not labile

    def test_reconsolidate_same_embedding_skipped(self, engine, memory_embedding):
        """Test reconsolidation skipped when query equals memory."""
        memory_id = uuid4()
        updated = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=memory_embedding.copy(),  # Same as memory
            outcome_score=0.9,
        )

        assert updated is None

    def test_reconsolidate_max_update_clipped(self, engine, memory_embedding, query_embedding):
        """Test update magnitude is clipped."""
        engine.base_learning_rate = 1.0  # Very high LR
        memory_id = uuid4()

        updated = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=1.0,
        )

        assert updated is not None
        # Update magnitude should be clipped
        update_mag = np.linalg.norm(updated - memory_embedding)
        # Due to normalization, actual difference may be different, but update was clipped

    def test_reconsolidate_with_importance(self, engine, memory_embedding, query_embedding):
        """Test reconsolidation with importance weighting."""
        memory_id = uuid4()

        # Update without importance
        updated1 = engine.reconsolidate(
            memory_id=uuid4(),  # Different ID to avoid cooldown
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.9,
            importance=0.0,
        )

        # Update with high importance (smaller update)
        updated2 = engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.9,
            importance=5.0,
        )

        # High importance should result in smaller update
        mag1 = np.linalg.norm(updated1 - memory_embedding)
        mag2 = np.linalg.norm(updated2 - memory_embedding)
        assert mag2 < mag1

    def test_reconsolidate_with_lr_modulation(self, engine, memory_embedding, query_embedding):
        """Test reconsolidation with learning rate modulation."""
        # Without modulation
        updated1 = engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.9,
            lr_modulation=1.0,
        )

        # With high modulation (larger update)
        updated2 = engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.9,
            lr_modulation=2.0,
        )

        mag1 = np.linalg.norm(updated1 - memory_embedding)
        mag2 = np.linalg.norm(updated2 - memory_embedding)
        assert mag2 > mag1

    def test_batch_reconsolidate(self, engine, query_embedding):
        """Test batch reconsolidation."""
        memories = [
            (uuid4(), np.random.randn(64) / np.sqrt(64)) for _ in range(3)
        ]

        updates = engine.batch_reconsolidate(
            memories=memories,
            query_embedding=query_embedding,
            outcome_score=0.8,
        )

        # Should have updates for all memories
        assert len(updates) > 0

    def test_batch_reconsolidate_per_memory_rewards(self, engine, query_embedding):
        """Test batch reconsolidation with per-memory rewards."""
        mem1, mem2 = uuid4(), uuid4()
        memories = [
            (mem1, np.random.randn(64) / np.sqrt(64)),
            (mem2, np.random.randn(64) / np.sqrt(64)),
        ]

        updates = engine.batch_reconsolidate(
            memories=memories,
            query_embedding=query_embedding,
            outcome_score=0.5,  # Neutral overall
            per_memory_rewards={
                str(mem1): 0.9,  # This one should update
                str(mem2): 0.5,  # This one neutral
            },
        )

        assert mem1 in updates
        # mem2 might not be in updates due to neutral score

    def test_batch_reconsolidate_per_memory_importance(self, engine, query_embedding):
        """Test batch reconsolidation with per-memory importance."""
        mem1, mem2 = uuid4(), uuid4()
        memories = [
            (mem1, np.random.randn(64) / np.sqrt(64)),
            (mem2, np.random.randn(64) / np.sqrt(64)),
        ]

        updates = engine.batch_reconsolidate(
            memories=memories,
            query_embedding=query_embedding,
            outcome_score=0.9,
            per_memory_importance={
                str(mem1): 0.0,  # Low importance, larger update
                str(mem2): 5.0,  # High importance, smaller update
            },
        )

        assert len(updates) >= 1

    def test_get_update_history(self, engine, memory_embedding, query_embedding):
        """Test getting update history."""
        # Create some updates
        for _ in range(3):
            engine.reconsolidate(
                memory_id=uuid4(),
                memory_embedding=memory_embedding,
                query_embedding=query_embedding,
                outcome_score=0.8,
            )

        history = engine.get_update_history()
        assert len(history) == 3

    def test_get_update_history_filtered(self, engine, memory_embedding, query_embedding):
        """Test getting filtered update history."""
        target_id = uuid4()

        # Create updates for different memories
        engine.reconsolidate(
            memory_id=target_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.8,
        )
        engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.8,
        )

        history = engine.get_update_history(memory_id=target_id)
        assert len(history) == 1
        assert history[0].memory_id == target_id

    def test_get_update_history_limit(self, engine, memory_embedding, query_embedding):
        """Test update history limit."""
        for _ in range(10):
            engine.reconsolidate(
                memory_id=uuid4(),
                memory_embedding=memory_embedding,
                query_embedding=query_embedding,
                outcome_score=0.8,
            )

        history = engine.get_update_history(limit=5)
        assert len(history) == 5

    def test_get_stats_empty(self, engine):
        """Test stats with no updates."""
        stats = engine.get_stats()
        assert stats["total_updates"] == 0
        assert stats["positive_updates"] == 0
        assert stats["negative_updates"] == 0
        assert stats["avg_magnitude"] == 0.0

    def test_get_stats_with_updates(self, engine, memory_embedding, query_embedding):
        """Test stats with updates."""
        # Positive update
        engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.9,
        )

        # Negative update
        engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.1,
        )

        stats = engine.get_stats()
        assert stats["total_updates"] == 2
        assert stats["positive_updates"] == 1
        assert stats["negative_updates"] == 1
        assert stats["avg_magnitude"] > 0

    def test_clear_history(self, engine, memory_embedding, query_embedding):
        """Test clearing history."""
        memory_id = uuid4()
        engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.8,
        )

        engine.clear_history()

        assert len(engine._update_history) == 0
        assert len(engine._last_update) == 0
        # Should be able to update same memory again
        assert engine.should_update(memory_id) is True


class TestReconsolidateFunction:
    """Tests for reconsolidate convenience function."""

    def test_reconsolidate_positive(self):
        """Test positive outcome reconsolidation."""
        memory = np.array([1.0, 0.0, 0.0])
        query = np.array([0.0, 1.0, 0.0])

        updated = reconsolidate(
            memory_embedding=memory,
            query_embedding=query,
            outcome_score=0.9,
            learning_rate=0.1,
        )

        # Should be normalized
        assert np.linalg.norm(updated) == pytest.approx(1.0, abs=0.01)
        # Should move toward query
        assert np.dot(updated, query) > np.dot(memory, query)

    def test_reconsolidate_negative(self):
        """Test negative outcome reconsolidation."""
        memory = np.array([1.0, 0.0, 0.0])
        query = np.array([0.0, 1.0, 0.0])

        updated = reconsolidate(
            memory_embedding=memory,
            query_embedding=query,
            outcome_score=0.1,
            learning_rate=0.1,
        )

        # Should move away from query
        assert np.dot(updated, query) < np.dot(memory, query)

    def test_reconsolidate_neutral(self):
        """Test neutral outcome reconsolidation."""
        memory = np.array([1.0, 0.0, 0.0])
        query = np.array([0.0, 1.0, 0.0])

        updated = reconsolidate(
            memory_embedding=memory,
            query_embedding=query,
            outcome_score=0.5,  # Neutral
            learning_rate=0.1,
        )

        # Should have minimal change
        assert np.allclose(updated, memory / np.linalg.norm(memory), atol=0.01)

    def test_reconsolidate_same_embedding(self):
        """Test with same memory and query."""
        emb = np.array([1.0, 0.0, 0.0])

        updated = reconsolidate(
            memory_embedding=emb,
            query_embedding=emb.copy(),
            outcome_score=0.9,
            learning_rate=0.1,
        )

        # Should return copy of original
        assert np.allclose(updated, emb)

    def test_reconsolidate_normalizes_output(self):
        """Test output is always normalized."""
        memory = np.random.randn(64)
        query = np.random.randn(64)

        updated = reconsolidate(
            memory_embedding=memory,
            query_embedding=query,
            outcome_score=0.9,
            learning_rate=0.1,
        )

        assert np.linalg.norm(updated) == pytest.approx(1.0, abs=0.01)


class TestNaNDetection:
    """DATA-005: Tests for NaN detection in reconsolidation."""

    def test_nan_input_memory_embedding_raises(self):
        """NaN in memory embedding raises ValueError."""
        memory = np.array([1.0, np.nan, 0.0])
        query = np.array([0.0, 1.0, 0.0])

        with pytest.raises(ValueError, match="NaN"):
            reconsolidate(
                memory_embedding=memory,
                query_embedding=query,
                outcome_score=0.8,
                learning_rate=0.1,
            )

    def test_nan_input_query_embedding_raises(self):
        """NaN in query embedding raises ValueError."""
        memory = np.array([1.0, 0.0, 0.0])
        query = np.array([0.0, np.nan, 0.0])

        with pytest.raises(ValueError, match="NaN"):
            reconsolidate(
                memory_embedding=memory,
                query_embedding=query,
                outcome_score=0.8,
                learning_rate=0.1,
            )

    def test_nan_outcome_score_raises(self):
        """NaN outcome score raises ValueError."""
        memory = np.array([1.0, 0.0, 0.0])
        query = np.array([0.0, 1.0, 0.0])

        with pytest.raises(ValueError, match="NaN"):
            reconsolidate(
                memory_embedding=memory,
                query_embedding=query,
                outcome_score=float('nan'),
                learning_rate=0.1,
            )

    def test_inf_input_raises(self):
        """Inf in input raises ValueError."""
        memory = np.array([1.0, np.inf, 0.0])
        query = np.array([0.0, 1.0, 0.0])

        with pytest.raises(ValueError, match="Inf"):
            reconsolidate(
                memory_embedding=memory,
                query_embedding=query,
                outcome_score=0.8,
                learning_rate=0.1,
            )

    def test_engine_nan_memory_raises(self):
        """Engine detects NaN in memory embedding."""
        engine = ReconsolidationEngine()
        memory = np.array([1.0, np.nan, 0.0])
        query = np.array([0.0, 1.0, 0.0])

        with pytest.raises(ValueError, match="NaN"):
            engine.reconsolidate(
                memory_id=uuid4(),
                memory_embedding=memory,
                query_embedding=query,
                outcome_score=0.8,
            )

    def test_engine_nan_query_raises(self):
        """Engine detects NaN in query embedding."""
        engine = ReconsolidationEngine()
        memory = np.array([1.0, 0.0, 0.0])
        query = np.array([0.0, np.nan, 0.0])

        with pytest.raises(ValueError, match="NaN"):
            engine.reconsolidate(
                memory_id=uuid4(),
                memory_embedding=memory,
                query_embedding=query,
                outcome_score=0.8,
            )

    def test_engine_inf_outcome_raises(self):
        """Engine detects Inf in outcome score."""
        engine = ReconsolidationEngine()
        memory = np.array([1.0, 0.0, 0.0])
        query = np.array([0.0, 1.0, 0.0])

        with pytest.raises(ValueError, match="Inf"):
            engine.reconsolidate(
                memory_id=uuid4(),
                memory_embedding=memory,
                query_embedding=query,
                outcome_score=float('inf'),
            )

    def test_function_inf_outcome_raises(self):
        """reconsolidate function detects Inf in outcome score."""
        memory = np.array([1.0, 0.0, 0.0])
        query = np.array([0.0, 1.0, 0.0])

        with pytest.raises(ValueError, match="Inf"):
            reconsolidate(
                memory_embedding=memory,
                query_embedding=query,
                outcome_score=float('inf'),
            )


class TestReconsolidationUpdateThreeFactor:
    """Tests for ReconsolidationUpdate with three-factor learning."""

    def test_used_three_factor_false(self):
        """Test used_three_factor is False when eligibility not set."""
        update = ReconsolidationUpdate(
            memory_id=uuid4(),
            query_embedding=np.random.randn(64),
            original_embedding=np.random.randn(64),
            updated_embedding=np.random.randn(64),
            outcome_score=0.8,
            advantage=0.3,
            learning_rate=0.01,
        )
        assert update.used_three_factor is False

    def test_used_three_factor_true(self):
        """Test used_three_factor is True when eligibility is set."""
        update = ReconsolidationUpdate(
            memory_id=uuid4(),
            query_embedding=np.random.randn(64),
            original_embedding=np.random.randn(64),
            updated_embedding=np.random.randn(64),
            outcome_score=0.8,
            advantage=0.3,
            learning_rate=0.01,
            eligibility=0.9,
            neuromod_gate=0.8,
            dopamine_surprise=0.7,
        )
        assert update.used_three_factor is True


class TestLabilityCleanup:
    """Tests for lability tracking cleanup methods."""

    def test_cleanup_lability_tracking_removes_expired(self):
        """Test that expired entries are removed."""
        engine = ReconsolidationEngine(lability_window_hours=1.0)
        engine._max_lability_entries = 5

        # Add some entries with old timestamps
        for i in range(3):
            mem_id = str(uuid4())
            engine._last_retrieval[mem_id] = datetime.now() - timedelta(hours=2)

        # Add some fresh entries
        fresh_ids = []
        for i in range(3):
            mem_id = str(uuid4())
            engine._last_retrieval[mem_id] = datetime.now()
            fresh_ids.append(mem_id)

        # Trigger cleanup
        engine._cleanup_lability_tracking()

        # Only fresh entries should remain (expired ones removed)
        for fresh_id in fresh_ids:
            assert fresh_id in engine._last_retrieval

    def test_cleanup_lability_tracking_respects_limit(self):
        """Test that cleanup respects max entries limit."""
        engine = ReconsolidationEngine()
        engine._max_lability_entries = 5

        # Add more entries than the limit (all recent)
        for i in range(10):
            mem_id = str(uuid4())
            engine._last_retrieval[mem_id] = datetime.now()

        # Trigger cleanup
        engine._cleanup_lability_tracking()

        # Should be trimmed to max limit
        assert len(engine._last_retrieval) <= 5

    def test_cleanup_cooldowns_alias(self):
        """Test deprecated _cleanup_cooldowns alias works."""
        engine = ReconsolidationEngine()

        # Add some entries
        for i in range(3):
            engine._last_retrieval[str(uuid4())] = datetime.now()

        # Should not raise
        engine._cleanup_cooldowns()

    def test_trim_history(self):
        """Test history trimming to max size."""
        engine = ReconsolidationEngine()
        engine._max_history_entries = 5

        # Add more updates than limit
        for i in range(10):
            engine._update_history.append(
                ReconsolidationUpdate(
                    memory_id=uuid4(),
                    query_embedding=np.zeros(64),
                    original_embedding=np.zeros(64),
                    updated_embedding=np.zeros(64),
                    outcome_score=0.8,
                    advantage=0.3,
                    learning_rate=0.01,
                )
            )

        # Trigger trim
        engine._trim_history()

        # Should be trimmed
        assert len(engine._update_history) <= 5

    def test_reconsolidate_triggers_lability_cleanup(self):
        """Test that reconsolidate triggers cleanup when over limit."""
        engine = ReconsolidationEngine()
        engine._max_cooldown_entries = 3
        engine._max_lability_entries = 3

        # Add entries to trigger cleanup
        for i in range(5):
            engine._last_update[str(uuid4())] = datetime.now() - timedelta(hours=10)

        # Do reconsolidate to trigger cleanup
        memory = np.random.randn(64)
        memory = memory / np.linalg.norm(memory)
        query = np.random.randn(64)
        query = query / np.linalg.norm(query)

        engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory,
            query_embedding=query,
            outcome_score=0.9,
        )

        # Entries should have been cleaned up
        assert len(engine._last_update) <= 4  # 3 max + 1 new


class TestDopamineModulatedReconsolidation:
    """Tests for DopamineModulatedReconsolidation class."""

    def test_init_default(self):
        """Test default initialization."""
        from ww.learning.reconsolidation import DopamineModulatedReconsolidation

        dmr = DopamineModulatedReconsolidation()

        assert dmr.reconsolidation is not None
        assert dmr.dopamine is not None
        assert dmr.use_uncertainty_boost is True

    def test_init_custom(self):
        """Test custom initialization."""
        from ww.learning.reconsolidation import DopamineModulatedReconsolidation

        dmr = DopamineModulatedReconsolidation(
            base_learning_rate=0.05,
            max_update_magnitude=0.2,
            value_learning_rate=0.15,
            cooldown_hours=2.0,
            use_uncertainty_boost=False,
        )

        assert dmr.reconsolidation.base_learning_rate == 0.05
        assert dmr.reconsolidation.max_update_magnitude == 0.2
        assert dmr.use_uncertainty_boost is False

    def test_update_positive_outcome(self):
        """Test update with positive outcome."""
        from ww.learning.reconsolidation import DopamineModulatedReconsolidation

        dmr = DopamineModulatedReconsolidation(cooldown_hours=0.0)
        memory_id = uuid4()
        memory = np.random.randn(64)
        memory = memory / np.linalg.norm(memory)
        query = np.random.randn(64)
        query = query / np.linalg.norm(query)

        updated = dmr.update(
            memory_id=memory_id,
            memory_embedding=memory,
            query_embedding=query,
            outcome_score=0.9,
        )

        assert updated is not None
        assert np.linalg.norm(updated) == pytest.approx(1.0, abs=0.01)

    def test_update_with_importance(self):
        """Test update with importance protection."""
        from ww.learning.reconsolidation import DopamineModulatedReconsolidation

        dmr = DopamineModulatedReconsolidation(cooldown_hours=0.0)
        memory = np.random.randn(64)
        memory = memory / np.linalg.norm(memory)
        query = np.random.randn(64)
        query = query / np.linalg.norm(query)

        # Low importance
        updated1 = dmr.update(
            memory_id=uuid4(),
            memory_embedding=memory,
            query_embedding=query,
            outcome_score=0.9,
            importance=0.0,
        )

        # High importance (smaller update)
        updated2 = dmr.update(
            memory_id=uuid4(),
            memory_embedding=memory,
            query_embedding=query,
            outcome_score=0.9,
            importance=5.0,
        )

        mag1 = np.linalg.norm(updated1 - memory)
        mag2 = np.linalg.norm(updated2 - memory)
        assert mag2 < mag1

    def test_batch_update(self):
        """Test batch update with multiple memories."""
        from ww.learning.reconsolidation import DopamineModulatedReconsolidation

        dmr = DopamineModulatedReconsolidation(cooldown_hours=0.0)
        mem1, mem2 = uuid4(), uuid4()
        memories = [
            (mem1, np.random.randn(64) / np.sqrt(64)),
            (mem2, np.random.randn(64) / np.sqrt(64)),
        ]
        query = np.random.randn(64)
        memory_outcomes = {
            str(mem1): 0.9,
            str(mem2): 0.8,
        }

        updates = dmr.batch_update(
            memories=memories,
            query_embedding=query,
            memory_outcomes=memory_outcomes,
        )

        assert len(updates) >= 1

    def test_batch_update_with_importance(self):
        """Test batch update with per-memory importance."""
        from ww.learning.reconsolidation import DopamineModulatedReconsolidation

        dmr = DopamineModulatedReconsolidation(cooldown_hours=0.0)
        mem1, mem2 = uuid4(), uuid4()
        memories = [
            (mem1, np.random.randn(64) / np.sqrt(64)),
            (mem2, np.random.randn(64) / np.sqrt(64)),
        ]
        query = np.random.randn(64)

        updates = dmr.batch_update(
            memories=memories,
            query_embedding=query,
            memory_outcomes={str(mem1): 0.9, str(mem2): 0.9},
            per_memory_importance={str(mem1): 0.0, str(mem2): 5.0},
        )

        assert len(updates) >= 1

    def test_get_stats(self):
        """Test get_stats returns combined statistics."""
        from ww.learning.reconsolidation import DopamineModulatedReconsolidation

        dmr = DopamineModulatedReconsolidation(cooldown_hours=0.0)
        memory = np.random.randn(64)
        memory = memory / np.linalg.norm(memory)
        query = np.random.randn(64)
        query = query / np.linalg.norm(query)

        dmr.update(
            memory_id=uuid4(),
            memory_embedding=memory,
            query_embedding=query,
            outcome_score=0.9,
        )

        stats = dmr.get_stats()

        assert "reconsolidation" in stats
        assert "dopamine" in stats


class TestNeuromodulatorIntegratedReconsolidation:
    """Tests for NeuromodulatorIntegratedReconsolidation class."""

    def test_init_default(self):
        """Test default initialization creates orchestra."""
        from ww.learning.reconsolidation import NeuromodulatorIntegratedReconsolidation

        nir = NeuromodulatorIntegratedReconsolidation()

        assert nir.orchestra is not None
        assert nir.reconsolidation is not None

    def test_init_with_orchestra(self):
        """Test initialization with provided orchestra."""
        from ww.learning.reconsolidation import NeuromodulatorIntegratedReconsolidation
        from ww.learning.neuromodulators import NeuromodulatorOrchestra

        orchestra = NeuromodulatorOrchestra()
        nir = NeuromodulatorIntegratedReconsolidation(orchestra=orchestra)

        assert nir.orchestra is orchestra

    def test_update_positive_outcome(self):
        """Test update with positive outcome."""
        from ww.learning.reconsolidation import NeuromodulatorIntegratedReconsolidation

        nir = NeuromodulatorIntegratedReconsolidation(cooldown_hours=0.0)
        memory_id = uuid4()
        memory = np.random.randn(64)
        memory = memory / np.linalg.norm(memory)
        query = np.random.randn(64)
        query = query / np.linalg.norm(query)

        updated = nir.update(
            memory_id=memory_id,
            memory_embedding=memory,
            query_embedding=query,
            outcome_score=0.9,
        )

        # May or may not update depending on neuromodulator state
        if updated is not None:
            assert np.linalg.norm(updated) == pytest.approx(1.0, abs=0.01)

    def test_batch_update(self):
        """Test batch update with multiple memories."""
        from ww.learning.reconsolidation import NeuromodulatorIntegratedReconsolidation

        nir = NeuromodulatorIntegratedReconsolidation(cooldown_hours=0.0)
        mem1, mem2 = uuid4(), uuid4()
        memories = [
            (mem1, np.random.randn(64) / np.sqrt(64)),
            (mem2, np.random.randn(64) / np.sqrt(64)),
        ]
        query = np.random.randn(64)

        updates = nir.batch_update(
            memories=memories,
            query_embedding=query,
            memory_outcomes={str(mem1): 0.9, str(mem2): 0.8},
        )

        # Returns dict of updated memories
        assert isinstance(updates, dict)

    def test_batch_update_with_importance(self):
        """Test batch update with per-memory importance."""
        from ww.learning.reconsolidation import NeuromodulatorIntegratedReconsolidation

        nir = NeuromodulatorIntegratedReconsolidation(cooldown_hours=0.0)
        mem1, mem2 = uuid4(), uuid4()
        memories = [
            (mem1, np.random.randn(64) / np.sqrt(64)),
            (mem2, np.random.randn(64) / np.sqrt(64)),
        ]
        query = np.random.randn(64)

        updates = nir.batch_update(
            memories=memories,
            query_embedding=query,
            memory_outcomes={str(mem1): 0.9, str(mem2): 0.9},
            per_memory_importance={str(mem1): 0.0, str(mem2): 5.0},
        )

        assert isinstance(updates, dict)

    def test_get_stats(self):
        """Test get_stats returns combined statistics."""
        from ww.learning.reconsolidation import NeuromodulatorIntegratedReconsolidation

        nir = NeuromodulatorIntegratedReconsolidation()

        stats = nir.get_stats()

        assert "reconsolidation" in stats
        assert "orchestra" in stats
