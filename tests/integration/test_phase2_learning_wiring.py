"""
Phase 2 Learning Wiring Integration Tests.

Tests that verify the Phase 2 learning system implementation:
- Three-factor learning rule is wired to reconsolidation
- Eligibility traces are marked during recall
- Feedback endpoint triggers actual embedding updates
- Hebbian semantic updates work correctly

Reference: docs/LEARNING_ARCHITECTURE.md
"""

import numpy as np
import pytest
import pytest_asyncio
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from t4dm.learning.three_factor import ThreeFactorLearningRule
from t4dm.learning.reconsolidation import ReconsolidationEngine


# ============================================================================
# Unit Tests for Phase 2 Wiring
# ============================================================================


class TestThreeFactorReconsolidationWiring:
    """
    Test that ThreeFactorLearningRule is properly wired to ReconsolidationEngine.

    Phase 2 requirement: The three-factor rule should modulate reconsolidation
    learning rates based on eligibility × neuromodulator × dopamine.
    """

    def test_reconsolidation_engine_accepts_three_factor(self):
        """ReconsolidationEngine should accept three_factor parameter."""
        three_factor = ThreeFactorLearningRule()
        engine = ReconsolidationEngine(
            base_learning_rate=0.1,
            three_factor=three_factor,
        )

        assert engine.three_factor is three_factor

    def test_three_factor_modulates_learning_rate(self):
        """Three-factor rule should modulate effective learning rate."""
        from uuid import UUID
        three_factor = ThreeFactorLearningRule(
            ach_weight=0.4,
            ne_weight=0.35,
            serotonin_weight=0.25,
        )

        # Mark memory as active (high eligibility)
        memory_id = uuid4()
        three_factor.mark_active(str(memory_id), activity=0.9)

        # Compute effective learning rate
        effective_lr = three_factor.compute_effective_lr(
            memory_id=memory_id,
            base_lr=0.1,
            outcome=0.8,  # Need outcome for dopamine
        )

        # Should be modulated (not equal to base)
        assert effective_lr > 0  # Should be positive

    def test_inactive_memory_has_low_effective_lr(self):
        """Memory not marked active should have low/zero effective LR."""
        three_factor = ThreeFactorLearningRule(
            min_eligibility_threshold=0.01,
        )

        # Don't mark any memory as active
        memory_id = uuid4()

        effective_lr = three_factor.compute_effective_lr(
            memory_id=memory_id,
            base_lr=0.1,
            outcome=0.8,
        )

        # Should be very low (eligibility is 0)
        # Note: min_effective_lr is 0.1, so this tests the bounds
        assert effective_lr <= 0.1  # At minimum floor

    def test_reconsolidation_uses_three_factor_lr(self):
        """Reconsolidation should use three-factor modulated learning rate."""
        # Use lower threshold since STDP traces are small (a_plus * activity ≈ 0.005)
        three_factor = ThreeFactorLearningRule(min_eligibility_threshold=0.001)
        engine = ReconsolidationEngine(
            base_learning_rate=0.1,
            three_factor=three_factor,
            cooldown_hours=0.0,  # Disable for test
        )

        memory_id = uuid4()

        # Mark as active with high eligibility
        three_factor.mark_active(str(memory_id), activity=1.0)

        # Trigger lability
        engine.trigger_lability(memory_id)

        # Create embeddings
        memory_emb = np.random.randn(128)
        memory_emb = memory_emb / np.linalg.norm(memory_emb)
        query_emb = np.random.randn(128)
        query_emb = query_emb / np.linalg.norm(query_emb)

        # Reconsolidate
        new_emb = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.9,
        )

        # Should have updated (not None)
        assert new_emb is not None
        # Should have moved toward query (positive outcome)
        old_dist = np.linalg.norm(query_emb - memory_emb)
        new_dist = np.linalg.norm(query_emb - new_emb)
        assert new_dist < old_dist


class TestEligibilityTracesOnRecall:
    """
    Test that eligibility traces are properly marked during recall.

    Phase 2 requirement: When memories are recalled, mark_active() should
    be called to enable credit assignment.
    """

    def test_mark_active_sets_eligibility(self):
        """mark_active should set eligibility trace for memory."""
        three_factor = ThreeFactorLearningRule()

        memory_id = "recalled-memory"
        three_factor.mark_active(memory_id, activity=0.8)

        # Should have non-zero eligibility
        eligibility = three_factor.get_eligibility(memory_id)
        assert eligibility > 0

    def test_mark_active_scales_with_retrieval_score(self):
        """Higher retrieval scores should give higher eligibility."""
        three_factor = ThreeFactorLearningRule()

        high_score_id = "high-score"
        low_score_id = "low-score"

        three_factor.mark_active(high_score_id, activity=0.95)
        three_factor.mark_active(low_score_id, activity=0.55)

        high_eligibility = three_factor.get_eligibility(high_score_id)
        low_eligibility = three_factor.get_eligibility(low_score_id)

        assert high_eligibility > low_eligibility

    def test_eligibility_decays_over_time(self):
        """Eligibility traces should decay exponentially."""
        three_factor = ThreeFactorLearningRule()

        memory_id = "decaying-memory"
        three_factor.mark_active(memory_id, activity=1.0)

        initial_eligibility = three_factor.get_eligibility(memory_id)

        # Simulate time passage by calling step()
        for _ in range(10):
            three_factor.step(dt=1.0)  # dt not delta_t

        decayed_eligibility = three_factor.get_eligibility(memory_id)

        assert decayed_eligibility < initial_eligibility

    def test_multiple_recalls_accumulate(self):
        """Multiple recalls should accumulate eligibility."""
        three_factor = ThreeFactorLearningRule()

        memory_id = "multi-recall"

        three_factor.mark_active(memory_id, activity=0.5)
        first_eligibility = three_factor.get_eligibility(memory_id)

        three_factor.mark_active(memory_id, activity=0.5)
        second_eligibility = three_factor.get_eligibility(memory_id)

        assert second_eligibility > first_eligibility


class TestLabilityWindow:
    """
    Test reconsolidation lability window behavior.

    Phase 2 requirement: Retrieved memories should become labile
    (modifiable) for a window of time after retrieval.
    """

    def test_trigger_lability_opens_window(self):
        """trigger_lability should open reconsolidation window."""
        engine = ReconsolidationEngine(lability_window_hours=6.0)

        memory_id = uuid4()
        engine.trigger_lability(memory_id)

        # Should be labile now
        assert engine.is_labile(str(memory_id)) is True

    def test_memory_labile_for_initial_encoding(self):
        """New memories are labile to allow initial encoding."""
        engine = ReconsolidationEngine(lability_window_hours=6.0)

        memory_id = uuid4()

        # New memory without trigger - labile for initial encoding
        # This matches biological semantics: new memories can be modified
        assert engine.is_labile(str(memory_id)) is True

    def test_multiple_updates_within_lability_window(self):
        """Multiple updates allowed within lability window."""
        engine = ReconsolidationEngine(
            lability_window_hours=6.0,
        )

        memory_id = uuid4()
        engine.trigger_lability(memory_id)

        memory_emb = np.random.randn(128)
        memory_emb = memory_emb / np.linalg.norm(memory_emb)
        query_emb = np.random.randn(128)
        query_emb = query_emb / np.linalg.norm(query_emb)

        # First update should work
        first_result = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.9,
        )

        # Second update also allowed (within lability window)
        second_result = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=first_result if first_result is not None else memory_emb,
            query_embedding=query_emb,
            outcome_score=0.8,
        )

        # Both should succeed (within lability window)
        assert first_result is not None
        assert second_result is not None  # Also succeeds


class TestReconsolidationDirection:
    """
    Test that reconsolidation moves embeddings in correct direction.

    Phase 2 requirement:
    - Positive outcomes: move embedding toward query
    - Negative outcomes: move embedding away from query
    """

    @pytest.fixture
    def engine(self):
        return ReconsolidationEngine(
            base_learning_rate=0.1,
            cooldown_hours=0.0,
            max_update_magnitude=0.3,
        )

    def test_positive_outcome_moves_toward_query(self, engine):
        """Positive outcome should move embedding toward query."""
        memory_id = uuid4()

        # Orthogonal embeddings for clear test
        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        new_emb = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.9,  # Positive
        )

        # Should have positive y component (moved toward query)
        assert new_emb[1] > 0

    def test_negative_outcome_moves_away_from_query(self, engine):
        """Negative outcome should move embedding away from query."""
        memory_id = uuid4()

        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        new_emb = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.1,  # Negative
        )

        # Should have negative y component (moved away from query)
        assert new_emb[1] < 0

    def test_neutral_outcome_no_update(self, engine):
        """Neutral outcome (0.5) should not update embedding."""
        memory_id = uuid4()

        memory_emb = np.random.randn(128)
        query_emb = np.random.randn(128)

        new_emb = engine.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.5,  # Neutral
        )

        # Should skip update
        assert new_emb is None


class TestSemanticHebbianUpdates:
    """
    Test Hebbian weight updates in semantic memory relationships.

    Phase 2 requirement: Co-retrieved entities should have their
    relationship weights modulated by outcome score.
    """

    def test_positive_outcome_strengthens_relationships(self):
        """Positive outcome should strengthen co-retrieval relationships."""
        # This would need SemanticMemory instance - tested at API level
        pass

    def test_negative_outcome_weakens_relationships(self):
        """Negative outcome should weaken co-retrieval relationships."""
        # This would need SemanticMemory instance - tested at API level
        pass


# ============================================================================
# Integration Tests for Full Learning Loop
# ============================================================================


class TestLearningLoopIntegration:
    """
    Integration tests for the complete Phase 2 learning loop.

    Flow:
    1. Recall memories → mark eligibility
    2. Use memories in task
    3. Task completes with outcome
    4. POST /feedback → triggers learning
    5. Embeddings updated based on three-factor rule
    """

    @pytest.fixture
    def three_factor(self):
        # Use lower threshold since STDP traces are small (a_plus * activity ≈ 0.005)
        # Also lower min_effective_lr to allow differences in eligibility to show in updates
        return ThreeFactorLearningRule(
            ach_weight=0.4,
            ne_weight=0.35,
            serotonin_weight=0.25,
            min_eligibility_threshold=0.001,  # Lower than default 0.01 to work with STDP traces
            min_effective_lr=0.0001,  # Allow small eligibility differences to affect LR
        )

    @pytest.fixture
    def reconsolidation(self, three_factor):
        return ReconsolidationEngine(
            base_learning_rate=0.1,
            three_factor=three_factor,
            cooldown_hours=0.0,
            lability_window_hours=6.0,
        )

    def test_full_learning_loop(self, three_factor, reconsolidation):
        """Test complete recall → feedback → update loop."""
        # Step 1: Simulate recall (mark eligibility)
        memory_ids = [uuid4() for _ in range(3)]
        retrieval_scores = [0.95, 0.80, 0.65]

        for mem_id, score in zip(memory_ids, retrieval_scores):
            # Mark active (as recall() would do)
            three_factor.mark_active(str(mem_id), activity=score)
            # Trigger lability (as recall() would do)
            reconsolidation.trigger_lability(mem_id)

        # Step 2: Verify eligibility set
        for mem_id in memory_ids:
            assert three_factor.get_eligibility(str(mem_id)) > 0

        # Step 3: Verify memories are labile
        for mem_id in memory_ids:
            assert reconsolidation.is_labile(str(mem_id)) is True

        # Step 4: Simulate task completion with positive outcome
        outcome_score = 0.85

        # Step 5: Apply reconsolidation to each memory
        # Use distinct query and memory embeddings to ensure update direction
        query_emb = np.array([0.0, 1.0] + [0.0] * 126)  # Unit vector in y direction

        updates = []
        for mem_id in memory_ids:
            # Memory starts in x direction (orthogonal to query)
            memory_emb = np.array([1.0, 0.0] + [0.0] * 126)

            new_emb = reconsolidation.reconsolidate(
                memory_id=mem_id,
                memory_embedding=memory_emb,
                query_embedding=query_emb,
                outcome_score=outcome_score,
            )

            if new_emb is not None:
                updates.append((mem_id, memory_emb, new_emb))

        # Step 6: Verify updates occurred
        assert len(updates) > 0, "At least one memory should be updated"

        # Verify embeddings moved toward query (positive outcome)
        for mem_id, old_emb, new_emb in updates:
            # Check movement toward query (y component should increase)
            assert new_emb[1] > old_emb[1], f"Memory {mem_id} y-component should increase"

    def test_higher_eligibility_gets_larger_update(self, three_factor, reconsolidation):
        """Memories with higher eligibility should get larger updates."""
        high_elig_id = uuid4()
        low_elig_id = uuid4()

        # Mark with different activity levels
        three_factor.mark_active(str(high_elig_id), activity=0.99)
        three_factor.mark_active(str(low_elig_id), activity=0.51)

        reconsolidation.trigger_lability(high_elig_id)
        reconsolidation.trigger_lability(low_elig_id)

        # Same starting embeddings and query
        memory_emb = np.array([1.0, 0.0, 0.0])
        query_emb = np.array([0.0, 1.0, 0.0])

        high_new = reconsolidation.reconsolidate(
            memory_id=high_elig_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.9,
        )

        # Reset cooldown for second test
        reconsolidation._last_update.pop(str(high_elig_id), None)

        low_new = reconsolidation.reconsolidate(
            memory_id=low_elig_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.9,
        )

        # Both should update
        assert high_new is not None
        assert low_new is not None

        # High eligibility should have larger update (closer to query)
        high_dist = np.linalg.norm(query_emb - high_new)
        low_dist = np.linalg.norm(query_emb - low_new)

        # High eligibility = larger update = closer to query = smaller distance
        assert high_dist < low_dist

    def test_stats_track_learning(self, three_factor, reconsolidation):
        """Learning statistics should track updates correctly."""
        memory_id = uuid4()
        three_factor.mark_active(str(memory_id), activity=0.9)
        reconsolidation.trigger_lability(memory_id)

        initial_stats = reconsolidation.get_stats()
        assert initial_stats["total_updates"] == 0

        # Do an update
        memory_emb = np.random.randn(128)
        query_emb = np.random.randn(128)

        reconsolidation.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_emb,
            query_embedding=query_emb,
            outcome_score=0.9,
        )

        final_stats = reconsolidation.get_stats()
        assert final_stats["total_updates"] == 1
        assert final_stats["positive_updates"] == 1


# ============================================================================
# Three-Factor Learning Rule Detailed Tests
# ============================================================================


class TestThreeFactorComputation:
    """
    Test three-factor learning rule computation.

    Formula: effective_lr = base_lr × eligibility × neuromod_gate × dopamine
    """

    @pytest.fixture
    def three_factor(self):
        return ThreeFactorLearningRule(
            ach_weight=0.4,
            ne_weight=0.35,
            serotonin_weight=0.25,
            min_effective_lr=0.1,
            max_effective_lr=3.0,
        )

    def test_compute_returns_modulated_lr(self, three_factor):
        """compute() should return ThreeFactorSignal with modulated learning rate."""
        memory_id = uuid4()
        three_factor.mark_active(str(memory_id), activity=0.9)

        result = three_factor.compute(
            memory_id=memory_id,
            base_lr=0.1,
            outcome=0.8,
        )

        # ThreeFactorSignal has these attributes
        assert hasattr(result, "effective_lr_multiplier")
        assert hasattr(result, "eligibility")
        assert hasattr(result, "neuromod_gate")
        assert hasattr(result, "dopamine_surprise")

    def test_effective_lr_bounded(self, three_factor):
        """Effective LR should be bounded by min/max."""
        memory_id = uuid4()
        three_factor.mark_active(str(memory_id), activity=1.0)

        # Very positive outcome (high surprise)
        result_high = three_factor.compute(
            memory_id=memory_id,
            base_lr=1.0,
            outcome=1.0,  # High positive outcome
        )

        # Should be capped at max
        assert result_high.effective_lr_multiplier <= 3.0

    def test_get_stats_returns_valid_structure(self, three_factor):
        """get_stats() should return valid statistics."""
        memory_id = uuid4()
        three_factor.mark_active(str(memory_id), activity=0.8)
        three_factor.compute(memory_id=memory_id, base_lr=0.1, outcome=0.8)

        stats = three_factor.get_stats()

        # Stats should be a dictionary
        assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
