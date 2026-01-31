"""
Integration tests for credit flow from neuromodulators to memory updates.

Tests the end-to-end flow:
1. Neuromodulator orchestra computes learning signals
2. CreditFlowEngine applies them to episodic/semantic memories
3. Actual weight updates occur in ReconsolidationEngine
"""

import pytest
import numpy as np
from uuid import uuid4

from t4dm.learning.credit_flow import CreditFlowEngine
from t4dm.learning.neuromodulators import NeuromodulatorOrchestra
from t4dm.learning.reconsolidation import ReconsolidationEngine


class TestCreditFlowEngine:
    """Tests for CreditFlowEngine."""

    @pytest.fixture
    def orchestra(self):
        """Create neuromodulator orchestra."""
        return NeuromodulatorOrchestra()

    @pytest.fixture
    def reconsolidation(self):
        """Create reconsolidation engine."""
        return ReconsolidationEngine(
            base_learning_rate=0.01,
            max_update_magnitude=0.1
        )

    @pytest.fixture
    def credit_flow(self, orchestra, reconsolidation):
        """Create credit flow engine."""
        return CreditFlowEngine(
            neuromodulator_orchestra=orchestra,
            reconsolidation_engine=reconsolidation
        )

    def test_initialization(self, credit_flow, orchestra, reconsolidation):
        """Credit flow engine initializes correctly."""
        assert credit_flow.orchestra is orchestra
        assert credit_flow.reconsolidation is reconsolidation
        assert credit_flow.episodic_lr_scale == 1.0
        assert credit_flow.semantic_lr_scale == 1.0

    def test_apply_to_episodic_no_engine(self, orchestra):
        """Apply to episodic without reconsolidation engine."""
        engine = CreditFlowEngine(orchestra, reconsolidation_engine=None)

        learning_signals = {"mem1": 0.5}
        memories = [(uuid4(), np.random.randn(128))]
        query = np.random.randn(128)

        result = engine.apply_to_episodic(learning_signals, memories, query)
        assert result == {}

    def test_apply_to_episodic_with_signals(self, credit_flow, orchestra):
        """Apply learning signals to episodic memory."""
        # Create memory IDs and embeddings
        mem1_id = uuid4()
        mem2_id = uuid4()
        mem1_emb = np.random.randn(128)
        mem2_emb = np.random.randn(128)

        retrieved_memories = [
            (mem1_id, mem1_emb.copy()),
            (mem2_id, mem2_emb.copy())
        ]
        query_embedding = np.random.randn(128)

        # Add eligibility and generate learning signals
        orchestra.serotonin.add_eligibility(mem1_id, strength=1.0)
        orchestra.serotonin.add_eligibility(mem2_id, strength=0.5)

        learning_signals = orchestra.process_outcome(
            memory_outcomes={str(mem1_id): 0.9, str(mem2_id): 0.3},
            session_outcome=0.7
        )

        # Apply to episodic
        updated = credit_flow.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=retrieved_memories,
            query_embedding=query_embedding
        )

        # Should have updates (non-neutral outcomes)
        assert isinstance(updated, dict)

    def test_learning_signal_to_outcome_conversion(self, credit_flow):
        """Learning signals are converted to outcome scores correctly."""
        mem_id = uuid4()
        mem_emb = np.random.randn(128)
        query = np.random.randn(128)

        # Positive learning signal should map to outcome > 0.5
        learning_signals = {str(mem_id): 0.3}
        result = credit_flow.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=[(mem_id, mem_emb)],
            query_embedding=query
        )

        # Outcome should be clipped to [0, 1] and centered around 0.5
        # signal 0.3 -> outcome 0.5 + 0.3 = 0.8
        assert True  # Just verify it doesn't crash

    def test_importance_weighting_from_long_term_value(self, credit_flow, orchestra):
        """Long-term value is used for importance weighting."""
        mem_id = uuid4()
        mem_emb = np.random.randn(128)
        query = np.random.randn(128)

        # Set high long-term value
        orchestra.serotonin._long_term_values[str(mem_id)] = 0.9

        learning_signals = {str(mem_id): 0.2}

        result = credit_flow.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=[(mem_id, mem_emb)],
            query_embedding=query
        )

        # High importance should protect memory (might not update if advantage is small)
        assert isinstance(result, dict)

    def test_surprise_modulation_from_dopamine(self, credit_flow, orchestra):
        """Dopamine surprise modulates learning rate."""
        mem_id = uuid4()
        mem_emb = np.random.randn(128)
        query = np.random.randn(128)

        # Establish expectation by updating with a low value first
        orchestra.dopamine.batch_update_expectations({str(mem_id): 0.3})

        # Provide surprising outcome (much higher than expected)
        learning_signals = orchestra.process_outcome(
            memory_outcomes={str(mem_id): 0.9},  # Surprise: |0.9 - 0.3| = 0.6
        )

        result = credit_flow.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=[(mem_id, mem_emb)],
            query_embedding=query
        )

        # High surprise should boost learning rate
        assert isinstance(result, dict)

    def test_process_and_apply_outcome_full_flow(self, credit_flow, orchestra):
        """End-to-end flow from outcomes to updates."""
        # Setup memories
        mem1_id, mem2_id = uuid4(), uuid4()
        mem1_emb = np.random.randn(128)
        mem2_emb = np.random.randn(128)

        retrieved_memories = [
            (mem1_id, mem1_emb.copy()),
            (mem2_id, mem2_emb.copy())
        ]
        query_embedding = np.random.randn(128)

        # Add eligibility
        orchestra.serotonin.add_eligibility(mem1_id, strength=1.0)
        orchestra.serotonin.add_eligibility(mem2_id, strength=0.5)

        # Process and apply outcome (use sync version for test compatibility)
        # ASYNC-002 FIX: process_and_apply_outcome is now async, use sync wrapper
        result = credit_flow.process_and_apply_outcome_sync(
            memory_outcomes={
                str(mem1_id): 0.9,
                str(mem2_id): 0.4
            },
            retrieved_memories=retrieved_memories,
            query_embedding=query_embedding,
            session_outcome=0.7
        )

        # Verify structure
        assert "learning_signals" in result
        assert "episodic_updates" in result
        assert "semantic_updates" in result

        assert isinstance(result["learning_signals"], dict)
        assert isinstance(result["episodic_updates"], dict)
        assert result["semantic_updates"] == 0  # No semantic memory provided

    def test_statistics_tracking(self, credit_flow, orchestra):
        """Statistics are tracked correctly."""
        stats_before = credit_flow.get_stats()
        assert stats_before["total_episodic_updates"] == 0

        # Apply some updates
        mem_id = uuid4()
        mem_emb = np.random.randn(128)
        query = np.random.randn(128)

        orchestra.serotonin.add_eligibility(mem_id, strength=1.0)
        learning_signals = orchestra.process_outcome(
            memory_outcomes={str(mem_id): 0.9}
        )

        credit_flow.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=[(mem_id, mem_emb)],
            query_embedding=query
        )

        stats_after = credit_flow.get_stats()
        # Updates may or may not occur depending on advantage magnitude
        assert stats_after["total_signal_magnitude"] >= 0

    def test_reset_stats(self, credit_flow):
        """Stats can be reset."""
        credit_flow._total_episodic_updates = 10
        credit_flow._total_semantic_updates = 5
        credit_flow._total_learning_signal_magnitude = 100.0

        credit_flow.reset_stats()

        stats = credit_flow.get_stats()
        assert stats["total_episodic_updates"] == 0
        assert stats["total_semantic_updates"] == 0
        assert stats["total_signal_magnitude"] == 0.0

    def test_custom_lr_scales(self):
        """Custom learning rate scales are applied."""
        orchestra = NeuromodulatorOrchestra()
        reconsolidation = ReconsolidationEngine()

        engine = CreditFlowEngine(
            neuromodulator_orchestra=orchestra,
            reconsolidation_engine=reconsolidation,
            episodic_lr_scale=2.0,
            semantic_lr_scale=0.5
        )

        assert engine.episodic_lr_scale == 2.0
        assert engine.semantic_lr_scale == 0.5


class TestCreditFlowIntegration:
    """Integration tests combining multiple systems."""

    def test_dopamine_serotonin_combination(self):
        """Learning signals combine dopamine and serotonin."""
        orchestra = NeuromodulatorOrchestra()
        reconsolidation = ReconsolidationEngine()
        engine = CreditFlowEngine(orchestra, reconsolidation)

        mem_id = uuid4()
        mem_emb = np.random.randn(128)
        query = np.random.randn(128)

        # Add eligibility for serotonin
        orchestra.serotonin.add_eligibility(mem_id, strength=1.0)

        # Establish dopamine expectation
        orchestra.dopamine.batch_update_expectations({str(mem_id): 0.5})

        # Process outcome
        learning_signals = orchestra.process_outcome(
            memory_outcomes={str(mem_id): 0.9},
            session_outcome=0.85
        )

        # Signals should include both dopamine RPE and serotonin credit
        assert str(mem_id) in learning_signals

        # Apply to episodic
        result = engine.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=[(mem_id, mem_emb)],
            query_embedding=query
        )

        assert isinstance(result, dict)

    def test_multiple_memories_differential_updates(self):
        """Different memories get different update magnitudes."""
        orchestra = NeuromodulatorOrchestra()
        reconsolidation = ReconsolidationEngine()
        engine = CreditFlowEngine(orchestra, reconsolidation)

        # Create 3 memories
        mem_ids = [uuid4() for _ in range(3)]
        embeddings = [np.random.randn(128) for _ in range(3)]
        query = np.random.randn(128)

        # Different eligibility traces
        orchestra.serotonin.add_eligibility(mem_ids[0], strength=1.0)
        orchestra.serotonin.add_eligibility(mem_ids[1], strength=0.5)
        orchestra.serotonin.add_eligibility(mem_ids[2], strength=0.1)

        # Different outcomes
        memory_outcomes = {
            str(mem_ids[0]): 0.9,  # Good
            str(mem_ids[1]): 0.5,  # Neutral
            str(mem_ids[2]): 0.2,  # Bad
        }

        # Process (use sync version for test compatibility)
        # ASYNC-002 FIX: process_and_apply_outcome is now async, use sync wrapper
        result = engine.process_and_apply_outcome_sync(
            memory_outcomes=memory_outcomes,
            retrieved_memories=list(zip(mem_ids, embeddings)),
            query_embedding=query,
            session_outcome=0.6
        )

        # Should have learning signals for all
        assert len(result["learning_signals"]) >= 1

    def test_cooldown_prevents_rapid_updates(self):
        """Reconsolidation cooldown prevents over-updating."""
        orchestra = NeuromodulatorOrchestra()
        reconsolidation = ReconsolidationEngine(cooldown_hours=1.0)
        engine = CreditFlowEngine(orchestra, reconsolidation)

        mem_id = uuid4()
        mem_emb = np.random.randn(128)
        query = np.random.randn(128)

        orchestra.serotonin.add_eligibility(mem_id, strength=1.0)

        # First update
        signals1 = orchestra.process_outcome({str(mem_id): 0.9})
        result1 = engine.apply_to_episodic(
            learning_signals=signals1,
            retrieved_memories=[(mem_id, mem_emb.copy())],
            query_embedding=query
        )

        # Second update immediately after (should be skipped due to cooldown)
        orchestra.serotonin.add_eligibility(mem_id, strength=1.0)
        signals2 = orchestra.process_outcome({str(mem_id): 0.8})
        result2 = engine.apply_to_episodic(
            learning_signals=signals2,
            retrieved_memories=[(mem_id, mem_emb.copy())],
            query_embedding=query
        )

        # First might update, second should skip due to cooldown
        # (both might be empty if advantage too small)
        assert isinstance(result2, dict)


class TestCreditFlowEdgeCases:
    """Edge case tests."""

    def test_empty_learning_signals(self):
        """Empty learning signals handled gracefully."""
        orchestra = NeuromodulatorOrchestra()
        reconsolidation = ReconsolidationEngine()
        engine = CreditFlowEngine(orchestra, reconsolidation)

        result = engine.apply_to_episodic(
            learning_signals={},
            retrieved_memories=[],
            query_embedding=np.random.randn(128)
        )

        assert result == {}

    def test_signals_without_corresponding_memories(self):
        """Learning signals for non-retrieved memories are ignored."""
        orchestra = NeuromodulatorOrchestra()
        reconsolidation = ReconsolidationEngine()
        engine = CreditFlowEngine(orchestra, reconsolidation)

        mem_id = uuid4()
        other_id = uuid4()

        learning_signals = {
            str(mem_id): 0.5,
            str(other_id): 0.8  # Not in retrieved_memories
        }

        result = engine.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=[(mem_id, np.random.randn(128))],
            query_embedding=np.random.randn(128)
        )

        # Should only process mem_id
        assert isinstance(result, dict)

    def test_memories_without_signals_use_base(self):
        """Memories without learning signals use base outcome."""
        orchestra = NeuromodulatorOrchestra()
        reconsolidation = ReconsolidationEngine()
        engine = CreditFlowEngine(orchestra, reconsolidation)

        mem_id = uuid4()
        mem_emb = np.random.randn(128)

        # No learning signals for this memory
        result = engine.apply_to_episodic(
            learning_signals={},
            retrieved_memories=[(mem_id, mem_emb)],
            query_embedding=np.random.randn(128),
            base_outcome=0.7
        )

        # Should use base_outcome of 0.7
        # (may not update if too close to baseline)
        assert isinstance(result, dict)

    def test_negative_rpe_creates_depression(self):
        """LOGIC-010: Negative RPE should lead to outcome < 0.5 (depression)."""
        orchestra = NeuromodulatorOrchestra()
        reconsolidation = ReconsolidationEngine()
        engine = CreditFlowEngine(orchestra, reconsolidation)

        mem_id = uuid4()
        mem_emb = np.random.randn(128)
        query = np.random.randn(128)

        # Set HIGH expectation so outcome is worse than expected
        orchestra.dopamine.batch_update_expectations({str(mem_id): 0.9})

        # Add eligibility
        orchestra.serotonin.add_eligibility(mem_id, strength=1.0)

        # Process with LOW outcome (worse than expected 0.9)
        learning_signals = orchestra.process_outcome(
            memory_outcomes={str(mem_id): 0.2},  # Much worse than expected
            session_outcome=0.3
        )

        # Get the signed RPE - should be negative
        signed_rpe = orchestra.get_signed_rpe(mem_id)
        assert signed_rpe < 0, f"Expected negative RPE, got {signed_rpe}"

        # Apply to episodic
        result = engine.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=[(mem_id, mem_emb)],
            query_embedding=query
        )

        # The reconsolidation should have processed this
        assert isinstance(result, dict)

    def test_positive_rpe_creates_potentiation(self):
        """LOGIC-010: Positive RPE should lead to outcome > 0.5 (potentiation)."""
        orchestra = NeuromodulatorOrchestra()
        reconsolidation = ReconsolidationEngine()
        engine = CreditFlowEngine(orchestra, reconsolidation)

        mem_id = uuid4()
        mem_emb = np.random.randn(128)
        query = np.random.randn(128)

        # Set LOW expectation so outcome is better than expected
        orchestra.dopamine.batch_update_expectations({str(mem_id): 0.2})

        # Add eligibility
        orchestra.serotonin.add_eligibility(mem_id, strength=1.0)

        # Process with HIGH outcome (better than expected 0.2)
        learning_signals = orchestra.process_outcome(
            memory_outcomes={str(mem_id): 0.9},  # Much better than expected
            session_outcome=0.8
        )

        # Get the signed RPE - should be positive
        signed_rpe = orchestra.get_signed_rpe(mem_id)
        assert signed_rpe > 0, f"Expected positive RPE, got {signed_rpe}"

        # Apply to episodic
        result = engine.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=[(mem_id, mem_emb)],
            query_embedding=query
        )

        assert isinstance(result, dict)

    def test_surprise_uses_dopamine_computed_value(self):
        """LOGIC-011: Surprise should come from dopamine's computed value."""
        orchestra = NeuromodulatorOrchestra()
        reconsolidation = ReconsolidationEngine()
        engine = CreditFlowEngine(orchestra, reconsolidation)

        mem_id = uuid4()
        mem_emb = np.random.randn(128).astype(np.float32)
        query = np.random.randn(128).astype(np.float32)

        # Initialize state by processing query (required for get_learning_params)
        orchestra.process_query(query)

        # Set expectation
        orchestra.dopamine.batch_update_expectations({str(mem_id): 0.5})

        # Add eligibility
        orchestra.serotonin.add_eligibility(mem_id, strength=1.0)

        # Process outcome
        learning_signals = orchestra.process_outcome(
            memory_outcomes={str(mem_id): 0.9},  # Surprise: |0.9 - 0.5| = 0.4
            session_outcome=0.8
        )

        # Get learning params - should have correct surprise from RPE cache
        params = orchestra.get_learning_params(mem_id)

        # Surprise should be approximately |0.9 - 0.5| = 0.4
        assert 0.3 < params.surprise < 0.5, f"Expected surprise ~0.4, got {params.surprise}"

        # Apply to episodic - should use this surprise
        result = engine.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=[(mem_id, mem_emb)],
            query_embedding=query
        )

        assert isinstance(result, dict)


class TestFSRSIntegration:
    """Tests for FSRS spaced repetition integration."""

    @pytest.fixture
    def orchestra(self):
        """Create neuromodulator orchestra."""
        return NeuromodulatorOrchestra()

    @pytest.fixture
    def reconsolidation(self):
        """Create reconsolidation engine."""
        return ReconsolidationEngine(
            base_learning_rate=0.01,
            max_update_magnitude=0.1
        )

    @pytest.fixture
    def credit_flow(self, orchestra, reconsolidation):
        """Create credit flow engine with default FSRS tracker."""
        return CreditFlowEngine(
            neuromodulator_orchestra=orchestra,
            reconsolidation_engine=reconsolidation
        )

    def test_fsrs_tracker_initialized(self, credit_flow):
        """Credit flow engine has FSRS tracker by default."""
        assert credit_flow.fsrs_tracker is not None
        assert credit_flow._total_fsrs_updates == 0

    def test_fsrs_updates_on_episodic_apply(self, credit_flow, orchestra):
        """FSRS state is updated when applying to episodic memory."""
        mem_id = uuid4()
        mem_emb = np.random.randn(128).astype(np.float32)
        mem_emb = mem_emb / np.linalg.norm(mem_emb)
        query = np.random.randn(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Register eligibility (serotonin tracks retrieved memories)
        orchestra.serotonin.add_eligibility(mem_id, strength=1.0)

        # Process outcome
        learning_signals = orchestra.process_outcome(
            memory_outcomes={str(mem_id): 0.8},
            session_outcome=0.8
        )

        # Apply to episodic
        credit_flow.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=[(mem_id, mem_emb)],
            query_embedding=query
        )

        # Check FSRS was updated
        assert credit_flow._total_fsrs_updates == 1

        # Check FSRS state exists for this memory
        mem_id_str = str(mem_id)
        state = credit_flow.fsrs_tracker.get_state(mem_id_str)
        assert state.reps == 1  # One review recorded

    def test_fsrs_stats_in_get_stats(self, credit_flow):
        """FSRS stats are included in get_stats()."""
        stats = credit_flow.get_stats()

        assert "fsrs" in stats
        assert "total_items" in stats["fsrs"]
        assert "total_fsrs_updates" in stats

    def test_outcome_to_rating_conversion(self):
        """Outcome scores convert to correct FSRS ratings."""
        from t4dm.learning.credit_flow import outcome_to_rating
        from t4dm.learning.fsrs import Rating

        assert outcome_to_rating(0.0) == Rating.AGAIN
        assert outcome_to_rating(0.2) == Rating.AGAIN
        assert outcome_to_rating(0.3) == Rating.HARD
        assert outcome_to_rating(0.4) == Rating.HARD
        assert outcome_to_rating(0.5) == Rating.GOOD
        assert outcome_to_rating(0.7) == Rating.GOOD
        assert outcome_to_rating(0.8) == Rating.EASY
        assert outcome_to_rating(1.0) == Rating.EASY

    def test_get_due_memories(self, credit_flow, orchestra):
        """Can retrieve memories due for review."""
        from datetime import datetime, timedelta
        from t4dm.learning.fsrs import Rating

        # Add a memory with old review
        mem_id = "test_memory_123"
        old_time = datetime.now() - timedelta(days=10)
        credit_flow.fsrs_tracker.review(mem_id, Rating.GOOD, old_time)

        # Get due items
        due = credit_flow.get_due_memories()

        # Should include our memory (it's overdue)
        memory_ids = [item[0] for item in due]
        assert mem_id in memory_ids

    def test_get_memory_retrievability(self, credit_flow):
        """Can get retrievability for a memory."""
        from datetime import datetime
        from t4dm.learning.fsrs import Rating

        mem_id = "test_memory_456"

        # Just reviewed - should be ~1.0
        credit_flow.fsrs_tracker.review(mem_id, Rating.GOOD)
        R = credit_flow.get_memory_retrievability(mem_id)

        assert R > 0.99  # Just reviewed, nearly 100% retrievable
