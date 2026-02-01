"""
Tests for reconsolidation integration in episodic memory.

Verifies that:
1. ReconsolidationEngine computes updates correctly
2. EpisodicMemory.apply_reconsolidation works end-to-end
3. Embeddings actually change in the vector store
"""

import pytest
import numpy as np
import torch
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from t4dm.learning.reconsolidation import ReconsolidationEngine, reconsolidate
from t4dm.memory.episodic import EpisodicMemory


class TestReconsolidationEngine:
    """Unit tests for ReconsolidationEngine."""

    def test_positive_outcome_pulls_toward_query(self):
        """Positive outcome should pull memory embedding toward query."""
        engine = ReconsolidationEngine(base_learning_rate=0.1)

        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        result = engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=1.0  # Very positive
        )

        assert result is not None
        # Should have moved toward query direction
        # Original was [1,0,0], query is [0,1,0], so y component should increase
        assert result[1] > memory_emb[1]

    def test_negative_outcome_pushes_away_from_query(self):
        """Negative outcome should push memory embedding away from query."""
        engine = ReconsolidationEngine(base_learning_rate=0.1)

        memory_emb = np.array([0.5, 0.5, 0.0])
        memory_emb = memory_emb / np.linalg.norm(memory_emb)  # Normalize
        query_emb = np.array([1.0, 0.0, 0.0])

        result = engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.0  # Very negative
        )

        assert result is not None
        # Should have moved away from query direction
        # Since advantage is negative, embedding should move opposite to (query - memory)
        # This means x component should decrease (moving away from [1,0,0])
        assert result[0] < memory_emb[0]

    def test_neutral_outcome_no_update(self):
        """Neutral outcome (0.5) should not update embedding."""
        engine = ReconsolidationEngine(base_learning_rate=0.1)

        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        result = engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.5  # Neutral
        )

        # Should return None for neutral outcomes
        assert result is None

    def test_lability_allows_updates_within_window(self):
        """BIO-MAJOR-001: Memory should be updatable within lability window."""
        engine = ReconsolidationEngine(
            base_learning_rate=0.1,
            lability_window_hours=1.0
        )

        memory_id = uuid4()
        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        # First update should work (triggers lability)
        result1 = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=1.0
        )
        assert result1 is not None

        # Second update should ALSO work (within lability window)
        # BIO-MAJOR-001: Changed from cooldown (blocks) to lability (allows)
        result2 = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=result1,
            query_embedding=query_emb,
            outcome_score=1.0
        )
        assert result2 is not None  # Now allowed within lability window

    def test_max_update_magnitude_clipping(self):
        """Update magnitude should be clipped to max_update_magnitude."""
        engine = ReconsolidationEngine(
            base_learning_rate=1.0,  # Very high
            max_update_magnitude=0.01  # Very low
        )

        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        result = engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=1.0
        )

        assert result is not None
        # Update magnitude should be roughly max_update_magnitude
        update_magnitude = np.linalg.norm(result - memory_emb)
        assert update_magnitude <= 0.02  # Allow some tolerance due to normalization

    def test_batch_reconsolidate(self):
        """Batch reconsolidation should update multiple memories."""
        engine = ReconsolidationEngine(base_learning_rate=0.1)

        memories = [
            (uuid4(), np.array([1.0, 0.0, 0.0])),
            (uuid4(), np.array([0.0, 1.0, 0.0])),
            (uuid4(), np.array([0.0, 0.0, 1.0])),
        ]
        query_emb = np.array([0.5, 0.5, 0.0])
        query_emb = query_emb / np.linalg.norm(query_emb)

        updates = engine.batch_reconsolidate(
            memories=memories,
            query_embedding=query_emb,
            outcome_score=0.8  # Positive
        )

        assert len(updates) == 3
        for mem_id, new_emb in updates.items():
            assert isinstance(new_emb, np.ndarray)
            assert np.abs(np.linalg.norm(new_emb) - 1.0) < 0.01  # Should be normalized

    def test_stats_tracking(self):
        """Engine should track reconsolidation statistics."""
        engine = ReconsolidationEngine(base_learning_rate=0.1)

        # Perform some updates
        for i in range(5):
            engine.reconsolidate(
                memory_id=uuid4(),
                memory_embedding=np.random.randn(3),
                query_embedding=np.random.randn(3),
                outcome_score=0.8 if i < 3 else 0.2
            )

        stats = engine.get_stats()
        assert stats["total_updates"] == 5
        assert stats["positive_updates"] == 3
        assert stats["negative_updates"] == 2

    def test_importance_weighted_lr_adjustment(self):
        """Importance should reduce learning rate."""
        engine = ReconsolidationEngine(base_learning_rate=0.1)

        # Zero importance = full learning rate
        lr_0 = engine.compute_importance_adjusted_lr(0.1, importance=0.0)
        assert abs(lr_0 - 0.1) < 1e-6

        # Importance 1 = half learning rate
        lr_1 = engine.compute_importance_adjusted_lr(0.1, importance=1.0)
        assert abs(lr_1 - 0.05) < 1e-6

        # Importance 9 = 1/10 learning rate
        lr_9 = engine.compute_importance_adjusted_lr(0.1, importance=9.0)
        assert abs(lr_9 - 0.01) < 1e-6

    def test_importance_reduces_update_magnitude(self):
        """Higher importance should result in smaller updates."""
        engine = ReconsolidationEngine(base_learning_rate=0.1)

        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        # Update with no importance
        result_no_importance = engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_emb.copy(),
            query_embedding=query_emb,
            outcome_score=1.0,
            importance=0.0
        )

        # Update with high importance (different memory ID to avoid cooldown)
        result_high_importance = engine.reconsolidate(
            memory_id=uuid4(),
            memory_embedding=memory_emb.copy(),
            query_embedding=query_emb,
            outcome_score=1.0,
            importance=9.0  # 1/10 learning rate
        )

        assert result_no_importance is not None
        assert result_high_importance is not None

        # Calculate update magnitudes
        mag_no_importance = np.linalg.norm(result_no_importance - memory_emb)
        mag_high_importance = np.linalg.norm(result_high_importance - memory_emb)

        # High importance should have smaller update
        assert mag_high_importance < mag_no_importance
        # Should be roughly 1/10
        assert mag_high_importance < 0.2 * mag_no_importance

    def test_batch_reconsolidate_with_importance(self):
        """Batch reconsolidation should respect per-memory importance."""
        engine = ReconsolidationEngine(base_learning_rate=0.1)

        mem1_id = uuid4()
        mem2_id = uuid4()

        mem1_emb = np.array([1.0, 0.0, 0.0])
        mem2_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        memories = [
            (mem1_id, mem1_emb.copy()),
            (mem2_id, mem2_emb.copy()),
        ]

        # Give mem2 high importance
        importance = {
            str(mem1_id): 0.0,   # Full update
            str(mem2_id): 9.0,   # Reduced update
        }

        updates = engine.batch_reconsolidate(
            memories=memories,
            query_embedding=query_emb,
            outcome_score=1.0,
            per_memory_importance=importance
        )

        assert mem1_id in updates
        assert mem2_id in updates

        # Check update magnitudes
        mag1 = np.linalg.norm(updates[mem1_id] - mem1_emb)
        mag2 = np.linalg.norm(updates[mem2_id] - mem2_emb)

        # mem2 should have much smaller update due to importance
        assert mag2 < mag1
        assert mag2 < 0.2 * mag1


class TestSimpleReconsolidate:
    """Test the convenience reconsolidate function."""

    def test_reconsolidate_function(self):
        """Simple reconsolidate function should work."""
        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        result = reconsolidate(
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=1.0,
            learning_rate=0.1
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        # Should be normalized
        assert np.abs(np.linalg.norm(result) - 1.0) < 0.01


class TestEpisodicMemoryReconsolidation:
    """Integration tests for EpisodicMemory.apply_reconsolidation."""

    @pytest.fixture
    def mock_episodic_memory(self):
        """Create a mock episodic memory with mocked backends."""
        # Pre-generate UUIDs for consistent testing
        self.test_uuid1 = uuid4()
        self.test_uuid2 = uuid4()

        with patch('t4dm.memory.episodic.get_settings') as mock_settings, \
             patch('t4dm.memory.episodic.get_embedding_provider') as mock_embedding, \
             patch('t4dm.memory.episodic.get_vector_store') as mock_qdrant, \
             patch('t4dm.memory.episodic.get_graph_store') as mock_neo4j, \
             patch('t4dm.memory.episodic.get_ff_encoder', return_value=None):

            # Configure settings
            settings = MagicMock()
            settings.session_id = "test-session"
            settings.episodic_weight_semantic = 0.4
            settings.episodic_weight_recency = 0.25
            settings.episodic_weight_outcome = 0.2
            settings.episodic_weight_importance = 0.15
            settings.fsrs_default_stability = 1.0
            settings.fsrs_decay_factor = 0.9
            settings.fsrs_recency_decay = 0.1
            settings.embedding_hybrid_enabled = False
            settings.ff_encoder_enabled = False  # Phase 5: Disable for unit tests
            settings.capsule_layer_enabled = False  # Phase 6: Disable for unit tests
            settings.capsule_retrieval_enabled = False  # Phase 6: Disable for unit tests
            settings.embedding_dimension = 1024
            mock_settings.return_value = settings

            # Configure embedding provider - use distinct query embedding
            # to ensure reconsolidation has something to update toward
            query_emb = np.zeros(1024)
            query_emb[0] = 1.0  # Unit vector in first dimension
            embedding = MagicMock()
            embedding.embed_query = AsyncMock(return_value=query_emb.tolist())
            mock_embedding.return_value = embedding

            # Configure vector store with embeddings different from query
            # so reconsolidation will produce updates
            mem1_emb = np.zeros(1024)
            mem1_emb[1] = 1.0  # Unit vector in second dimension
            mem2_emb = np.zeros(1024)
            mem2_emb[2] = 1.0  # Unit vector in third dimension

            qdrant = MagicMock()
            qdrant.episodes_collection = "episodes"
            # Use proper UUID strings for mock return values
            qdrant.get_with_vectors = AsyncMock(return_value=[
                (str(self.test_uuid1), {"content": "test"}, mem1_emb.tolist()),
                (str(self.test_uuid2), {"content": "test2"}, mem2_emb.tolist()),
            ])
            qdrant.batch_update_vectors = AsyncMock(return_value=2)
            mock_qdrant.return_value = qdrant

            # Configure graph store
            neo4j = MagicMock()
            mock_neo4j.return_value = neo4j

            memory = EpisodicMemory(session_id="test")
            memory._test_uuid1 = self.test_uuid1
            memory._test_uuid2 = self.test_uuid2
            yield memory

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Phase 5 FFEncoder changes affect reconsolidation behavior - TODO investigate")
    async def test_apply_reconsolidation_positive(self, mock_episodic_memory):
        """Test reconsolidation with positive outcome."""
        memory = mock_episodic_memory

        # Use the UUIDs that the mock is expecting
        result = await memory.apply_reconsolidation(
            episode_ids=[memory._test_uuid1, memory._test_uuid2],
            query="test query",
            outcome_score=0.9
        )

        # Should have reconsolidated 2 episodes
        assert result == 2

        # Verify vector store was called to update
        memory.vector_store.batch_update_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_reconsolidation_neutral_no_update(self, mock_episodic_memory):
        """Test that neutral outcomes don't trigger updates."""
        memory = mock_episodic_memory

        result = await memory.apply_reconsolidation(
            episode_ids=[uuid4()],
            query="test query",
            outcome_score=0.5  # Neutral
        )

        # Should not update anything (neutral outcome)
        assert result == 0

    @pytest.mark.asyncio
    async def test_apply_reconsolidation_disabled(self, mock_episodic_memory):
        """Test that reconsolidation can be disabled."""
        memory = mock_episodic_memory
        memory._reconsolidation_enabled = False

        result = await memory.apply_reconsolidation(
            episode_ids=[uuid4()],
            query="test query",
            outcome_score=1.0
        )

        # Should return 0 when disabled
        assert result == 0

    @pytest.mark.asyncio
    async def test_apply_reconsolidation_empty_ids(self, mock_episodic_memory):
        """Test that empty ID list returns 0."""
        memory = mock_episodic_memory

        result = await memory.apply_reconsolidation(
            episode_ids=[],
            query="test query",
            outcome_score=1.0
        )

        assert result == 0

    def test_get_reconsolidation_stats(self, mock_episodic_memory):
        """Test stats retrieval."""
        memory = mock_episodic_memory

        stats = memory.get_reconsolidation_stats()

        assert "total_updates" in stats
        assert "positive_updates" in stats
        assert "negative_updates" in stats


class TestLearnedFusionEnabled:
    """Test that learned fusion is enabled by default."""

    def test_learned_fusion_default_enabled(self):
        """NeuroSymbolicReasoner should have learned fusion enabled by default."""
        from t4dm.learning.neuro_symbolic import NeuroSymbolicReasoner

        reasoner = NeuroSymbolicReasoner()

        assert reasoner.use_learned_fusion is True
        assert reasoner.learned_fusion is not None


class TestDopamineRPE:
    """Test dopamine-like reward prediction error system."""

    def test_compute_rpe_positive_surprise(self):
        """Better than expected should give positive RPE."""
        from t4dm.learning.dopamine import DopamineSystem

        dopamine = DopamineSystem(default_expected=0.5)
        mem_id = uuid4()

        rpe = dopamine.compute_rpe(mem_id, actual_outcome=0.9)

        assert rpe.rpe > 0  # Positive surprise
        assert rpe.expected == 0.5
        assert rpe.actual == 0.9
        assert rpe.is_positive_surprise

    def test_compute_rpe_negative_surprise(self):
        """Worse than expected should give negative RPE."""
        from t4dm.learning.dopamine import DopamineSystem

        dopamine = DopamineSystem(default_expected=0.5)
        mem_id = uuid4()

        rpe = dopamine.compute_rpe(mem_id, actual_outcome=0.1)

        assert rpe.rpe < 0  # Negative surprise
        assert rpe.is_negative_surprise

    def test_value_estimate_updates(self):
        """Value estimates should update toward actual outcomes."""
        from t4dm.learning.dopamine import DopamineSystem

        dopamine = DopamineSystem(default_expected=0.5, value_learning_rate=0.2)
        mem_id = uuid4()

        # Initial expectation is 0.5
        assert dopamine.get_expected_value(mem_id) == 0.5

        # After high outcome, expectation should increase
        dopamine.update_expectations(mem_id, actual_outcome=1.0)
        new_expected = dopamine.get_expected_value(mem_id)
        assert new_expected > 0.5

        # After many high outcomes, should approach 1.0
        for _ in range(20):
            dopamine.update_expectations(mem_id, actual_outcome=1.0)
        final_expected = dopamine.get_expected_value(mem_id)
        assert final_expected > 0.9

    def test_surprise_modulates_learning_rate(self):
        """Higher surprise should increase learning rate."""
        from t4dm.learning.dopamine import DopamineSystem

        dopamine = DopamineSystem(default_expected=0.5)

        # Small surprise (outcome near expected)
        small_rpe = dopamine.compute_rpe(uuid4(), actual_outcome=0.55)
        small_lr = dopamine.modulate_learning_rate(0.1, small_rpe, use_uncertainty=False)

        # Large surprise (outcome far from expected)
        large_rpe = dopamine.compute_rpe(uuid4(), actual_outcome=1.0)
        large_lr = dopamine.modulate_learning_rate(0.1, large_rpe, use_uncertainty=False)

        # Large surprise should have higher LR
        assert large_lr > small_lr

    def test_rpe_for_fusion_training(self):
        """RPE should be converted to [0,1] range for ranking loss."""
        from t4dm.learning.dopamine import DopamineSystem

        dopamine = DopamineSystem(default_expected=0.5)

        outcomes = {
            str(uuid4()): 0.9,  # Positive surprise
            str(uuid4()): 0.5,  # No surprise
            str(uuid4()): 0.1,  # Negative surprise
        }

        rpe_targets = dopamine.get_rpe_for_fusion_training(outcomes)

        # All should be in [0, 1]
        for v in rpe_targets.values():
            assert 0.0 <= v <= 1.0

        # Get keys in order
        keys = list(outcomes.keys())
        # Positive surprise (0.9) should rank highest
        assert rpe_targets[keys[0]] > rpe_targets[keys[1]]
        # Negative surprise (0.1) should rank lowest
        assert rpe_targets[keys[2]] < rpe_targets[keys[1]]


class TestLearnedFusionTraining:
    """Test that LearnedFusion can be trained with gradient flow."""

    def test_train_fusion_step_updates_weights(self):
        """Training step should update fusion network weights."""
        from t4dm.learning.neuro_symbolic import NeuroSymbolicReasoner

        reasoner = NeuroSymbolicReasoner()

        # Capture initial weights
        initial_weights = [
            p.clone().detach()
            for p in reasoner.learned_fusion.parameters()
        ]

        # Create training data
        query_emb = np.random.randn(1024).astype(np.float32)
        neural = {"m1": 0.9, "m2": 0.5, "m3": 0.1}
        symbolic = {"m1": 0.3, "m2": 0.7, "m3": 0.4}
        recency = {"m1": 0.8, "m2": 0.6, "m3": 0.9}
        outcome = {"m1": 0.7, "m2": 0.5, "m3": 0.3}
        # Target rewards: m2 should rank highest (contradicts neural)
        targets = {"m1": 0.3, "m2": 0.9, "m3": 0.1}

        # Train for several steps
        for _ in range(10):
            loss = reasoner.train_fusion_step(
                query_emb, neural, symbolic, recency, outcome, targets
            )
            assert loss > 0  # Loss should be positive

        # Check weights changed
        final_weights = list(reasoner.learned_fusion.parameters())
        weights_changed = False
        for initial, final in zip(initial_weights, final_weights):
            if not torch.allclose(initial, final, atol=1e-6):
                weights_changed = True
                break

        assert weights_changed, "Fusion weights should update during training"

    def test_train_fusion_step_reduces_loss(self):
        """Training should reduce loss over time."""
        from t4dm.learning.neuro_symbolic import NeuroSymbolicReasoner

        reasoner = NeuroSymbolicReasoner()

        query_emb = np.random.randn(1024).astype(np.float32)
        neural = {"m1": 0.9, "m2": 0.5, "m3": 0.1, "m4": 0.3}
        symbolic = {"m1": 0.3, "m2": 0.7, "m3": 0.4, "m4": 0.6}
        recency = {"m1": 0.8, "m2": 0.6, "m3": 0.9, "m4": 0.4}
        outcome = {"m1": 0.7, "m2": 0.5, "m3": 0.3, "m4": 0.8}
        targets = {"m1": 0.2, "m2": 0.8, "m3": 0.1, "m4": 0.9}

        # Collect losses
        losses = []
        for _ in range(50):
            loss = reasoner.train_fusion_step(
                query_emb, neural, symbolic, recency, outcome, targets
            )
            losses.append(loss)

        # Loss should generally decrease
        first_10_avg = sum(losses[:10]) / 10
        last_10_avg = sum(losses[-10:]) / 10
        assert last_10_avg < first_10_avg, "Loss should decrease with training"

    def test_train_fusion_stats_tracking(self):
        """Training stats should be tracked correctly."""
        from t4dm.learning.neuro_symbolic import NeuroSymbolicReasoner

        reasoner = NeuroSymbolicReasoner()

        query_emb = np.random.randn(1024).astype(np.float32)
        neural = {"m1": 0.9, "m2": 0.5}
        symbolic = {"m1": 0.3, "m2": 0.7}
        recency = {"m1": 0.8, "m2": 0.6}
        outcome = {"m1": 0.7, "m2": 0.5}
        targets = {"m1": 0.3, "m2": 0.9}

        # Initial stats
        stats = reasoner.get_fusion_training_stats()
        assert stats["enabled"] is True
        assert stats["train_steps"] == 0

        # Train and check
        for _ in range(5):
            reasoner.train_fusion_step(
                query_emb, neural, symbolic, recency, outcome, targets
            )

        stats = reasoner.get_fusion_training_stats()
        assert stats["train_steps"] == 5
        assert stats["avg_loss"] > 0
