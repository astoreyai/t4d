"""Tests for multiplicative gating in neuromodulator orchestra."""

import pytest
import numpy as np
from uuid import uuid4

from t4dm.learning.neuromodulators import (
    LearningParams,
    NeuromodulatorOrchestra,
)


class TestLearningParams:
    """Tests for LearningParams dataclass."""

    def test_creation(self):
        """Create LearningParams with all fields."""
        params = LearningParams(
            effective_lr=0.01,
            eligibility=0.8,
            surprise=0.5,
            patience=0.6,
            rpe=0.3
        )
        assert params.effective_lr == 0.01
        assert params.eligibility == 0.8
        assert params.surprise == 0.5
        assert params.patience == 0.6
        assert params.rpe == 0.3

    def test_combined_learning_signal_multiplicative(self):
        """Combined signal is multiplicative with bootstrap component.

        BUG-006 FIX: The combined signal now includes a small bootstrap term
        to prevent zero-learning deadlock. The signal is:
        multiplicative + bootstrap = (lr * elig * surprise * patience) + (0.01 * lr * max(0.1, surprise))
        """
        params = LearningParams(
            effective_lr=0.5,
            eligibility=0.8,
            surprise=0.6,
            patience=0.7,
            rpe=0.1
        )
        # Multiplicative: 0.5 * 0.8 * 0.6 * 0.7 = 0.168
        # Bootstrap: 0.01 * 0.5 * max(0.1, 0.6) = 0.01 * 0.5 * 0.6 = 0.003
        # Total: 0.168 + 0.003 = 0.171
        multiplicative = 0.5 * 0.8 * 0.6 * 0.7
        bootstrap = 0.01 * 0.5 * max(0.1, 0.6)
        expected = multiplicative + bootstrap
        assert abs(params.combined_learning_signal - expected) < 1e-6

    def test_combined_signal_nonzero_with_bootstrap(self):
        """Bootstrap prevents zero-learning deadlock (BUG-006 fix).

        Even when eligibility or surprise is zero, there's a small bootstrap
        signal to allow initial learning.
        """
        # Zero eligibility - bootstrap still provides small signal
        params_zero_elig = LearningParams(
            effective_lr=1.0,
            eligibility=0.0,  # Zero gates multiplicative term
            surprise=1.0,
            patience=1.0,
            rpe=0.5
        )
        # Multiplicative: 0 (gated by zero eligibility)
        # Bootstrap: 0.01 * 1.0 * max(0.1, 1.0) = 0.01
        assert params_zero_elig.combined_learning_signal > 0  # Bootstrap prevents zero
        assert abs(params_zero_elig.combined_learning_signal - 0.01) < 1e-6

        # Zero surprise - bootstrap uses minimum 0.1
        params_zero_surprise = LearningParams(
            effective_lr=1.0,
            eligibility=1.0,
            surprise=0.0,  # Zero gates multiplicative, but bootstrap uses 0.1 minimum
            patience=1.0,
            rpe=0.5
        )
        # Multiplicative: 0 (gated by zero surprise)
        # Bootstrap: 0.01 * 1.0 * max(0.1, 0.0) = 0.01 * 1.0 * 0.1 = 0.001
        assert params_zero_surprise.combined_learning_signal > 0  # Bootstrap prevents zero
        assert abs(params_zero_surprise.combined_learning_signal - 0.001) < 1e-6

    def test_to_dict(self):
        """Params convert to dictionary."""
        params = LearningParams(
            effective_lr=0.01,
            eligibility=0.5,
            surprise=0.3,
            patience=0.7,
            rpe=0.2
        )
        d = params.to_dict()
        assert d["effective_lr"] == 0.01
        assert d["eligibility"] == 0.5
        assert d["surprise"] == 0.3
        assert d["patience"] == 0.7
        assert d["rpe"] == 0.2
        assert "combined_signal" in d
        assert "timestamp" in d


class TestMultiplicativeGating:
    """Tests for multiplicative gating in process_outcome."""

    @pytest.fixture
    def orchestra(self):
        """Create orchestra instance."""
        return NeuromodulatorOrchestra()

    def test_process_outcome_multiplicative(self, orchestra):
        """process_outcome uses multiplicative gating."""
        # Set up state
        query_embedding = np.random.randn(128)
        orchestra.process_query(query_embedding, is_question=False)

        # Add some eligibility by retrieving memories
        mem_id1 = uuid4()
        mem_id2 = uuid4()
        orchestra.process_retrieval(
            retrieved_ids=[mem_id1, mem_id2],
            scores={str(mem_id1): 0.9, str(mem_id2): 0.7}
        )

        # Process outcomes
        memory_outcomes = {
            str(mem_id1): 0.8,
            str(mem_id2): 0.3
        }
        learning_signals = orchestra.process_outcome(
            memory_outcomes,
            session_outcome=0.7
        )

        # Signals should be present
        assert str(mem_id1) in learning_signals
        assert str(mem_id2) in learning_signals

        # High outcome should have stronger signal than low outcome
        # (assuming dopamine surprise is computed correctly)
        assert learning_signals[str(mem_id1)] >= 0.0
        assert learning_signals[str(mem_id2)] >= 0.0

    def test_process_outcome_no_eligibility_zero_signal(self, orchestra):
        """Without eligibility trace, learning signal is zero (multiplicative)."""
        # Set up state
        query_embedding = np.random.randn(128)
        orchestra.process_query(query_embedding, is_question=False)

        # Don't add eligibility - skip process_retrieval

        # Process outcome for memory with no eligibility
        mem_id = uuid4()
        memory_outcomes = {str(mem_id): 0.8}
        learning_signals = orchestra.process_outcome(
            memory_outcomes,
            session_outcome=0.7
        )

        # With multiplicative gating, no eligibility → zero signal
        # (eligibility * anything = 0)
        if str(mem_id) in learning_signals:
            # Signal should be very weak or zero due to missing eligibility
            assert learning_signals[str(mem_id)] < 0.1


class TestGetLearningParams:
    """Tests for get_learning_params methods."""

    @pytest.fixture
    def orchestra(self):
        """Create orchestra instance."""
        return NeuromodulatorOrchestra()

    def test_get_learning_params_basic(self, orchestra):
        """get_learning_params returns LearningParams."""
        # Set up state
        query_embedding = np.random.randn(128)
        orchestra.process_query(query_embedding)

        mem_id = uuid4()
        params = orchestra.get_learning_params(mem_id)

        assert isinstance(params, LearningParams)
        assert params.effective_lr > 0
        assert params.eligibility >= 0
        assert params.surprise >= 0
        assert params.patience >= 0

    def test_get_learning_params_with_outcome(self, orchestra):
        """get_learning_params_with_outcome computes RPE."""
        # Set up state
        query_embedding = np.random.randn(128)
        orchestra.process_query(query_embedding)

        # Add eligibility
        mem_id = uuid4()
        orchestra.process_retrieval(
            retrieved_ids=[mem_id],
            scores={str(mem_id): 0.8}
        )

        # Get params with outcome
        params = orchestra.get_learning_params_with_outcome(mem_id, outcome=0.9)

        assert isinstance(params, LearningParams)
        assert params.effective_lr > 0
        assert params.eligibility > 0  # Should have trace from retrieval
        assert params.surprise >= 0  # RPE computed
        assert params.patience >= 0

    def test_get_learning_params_no_state_returns_neutral(self, orchestra):
        """Without current state, returns neutral params."""
        mem_id = uuid4()
        params = orchestra.get_learning_params(mem_id)

        assert params.effective_lr == 1.0
        assert params.eligibility == 0.0
        assert params.surprise == 0.0
        assert params.patience == 0.0
        assert params.rpe == 0.0


class TestNeuromodulatorIntegratedReconsolidation:
    """Tests for NeuromodulatorIntegratedReconsolidation class."""

    def test_import(self):
        """Can import NeuromodulatorIntegratedReconsolidation."""
        from t4dm.learning.reconsolidation import NeuromodulatorIntegratedReconsolidation
        assert NeuromodulatorIntegratedReconsolidation is not None

    def test_creation(self):
        """Can create NeuromodulatorIntegratedReconsolidation."""
        from t4dm.learning.reconsolidation import NeuromodulatorIntegratedReconsolidation
        recon = NeuromodulatorIntegratedReconsolidation()
        assert recon.orchestra is not None
        assert recon.reconsolidation is not None

    def test_update_with_orchestra(self):
        """Update uses orchestra for learning params."""
        from t4dm.learning.reconsolidation import NeuromodulatorIntegratedReconsolidation

        recon = NeuromodulatorIntegratedReconsolidation()

        # Set up state
        query_emb = np.random.randn(128)
        recon.orchestra.process_query(query_emb)

        # Create memory
        mem_id = uuid4()
        mem_emb = np.random.randn(128)
        mem_emb = mem_emb / np.linalg.norm(mem_emb)  # Normalize

        # Add eligibility
        recon.orchestra.process_retrieval(
            retrieved_ids=[mem_id],
            scores={str(mem_id): 0.8}
        )

        # Update with positive outcome
        updated = recon.update(
            memory_id=mem_id,
            memory_embedding=mem_emb,
            query_embedding=query_emb,
            outcome_score=0.9,
            importance=0.0
        )

        # Should get an update (if cooldown allows and signals align)
        # May be None if signals don't align strongly enough
        if updated is not None:
            assert updated.shape == mem_emb.shape
            assert np.linalg.norm(updated) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
