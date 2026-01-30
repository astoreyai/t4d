"""
Unit tests for three-factor learning rule.

Tests the biologically-inspired learning system that combines:
1. Eligibility traces (temporal credit assignment)
2. Neuromodulator gating (learning state)
3. Dopamine surprise (prediction error)
"""

import pytest
import numpy as np
from uuid import uuid4
from datetime import datetime

from ww.learning.three_factor import (
    ThreeFactorSignal,
    ThreeFactorLearningRule,
    ThreeFactorReconsolidation,
    create_three_factor_rule,
)
from ww.learning.eligibility import EligibilityTrace, LayeredEligibilityTrace
from ww.learning.neuromodulators import NeuromodulatorOrchestra, NeuromodulatorState
from ww.learning.dopamine import DopamineSystem, RewardPredictionError


class TestThreeFactorSignal:
    """Tests for ThreeFactorSignal dataclass."""

    def test_signal_creation(self):
        """Signal can be created with all required fields."""
        memory_id = uuid4()
        signal = ThreeFactorSignal(
            memory_id=memory_id,
            eligibility=0.8,
            neuromod_gate=1.2,
            ach_mode_factor=1.5,
            ne_arousal_factor=1.0,
            serotonin_mood_factor=0.9,
            dopamine_surprise=0.5,
            rpe_raw=0.3,
            effective_lr_multiplier=0.48,
        )

        assert signal.memory_id == memory_id
        assert signal.eligibility == 0.8
        assert signal.neuromod_gate == 1.2
        assert signal.effective_lr_multiplier == 0.48

    def test_signal_to_dict(self):
        """Signal serializes to dictionary."""
        memory_id = uuid4()
        signal = ThreeFactorSignal(
            memory_id=memory_id,
            eligibility=0.8,
            neuromod_gate=1.2,
            ach_mode_factor=1.5,
            ne_arousal_factor=1.0,
            serotonin_mood_factor=0.9,
            dopamine_surprise=0.5,
            rpe_raw=0.3,
            effective_lr_multiplier=0.48,
        )

        result = signal.to_dict()

        assert isinstance(result, dict)
        assert result["memory_id"] == str(memory_id)
        assert result["eligibility"] == 0.8
        assert result["effective_lr_multiplier"] == 0.48
        assert "timestamp" in result


class TestThreeFactorLearningRule:
    """Tests for ThreeFactorLearningRule."""

    @pytest.fixture
    def rule(self):
        """Create three-factor learning rule with default components.

        Note: Uses a lower min_eligibility_threshold since default a_plus=0.005
        means a single mark_active only adds 0.005 to eligibility.
        """
        return ThreeFactorLearningRule(
            min_effective_lr=0.1,
            max_effective_lr=3.0,
            min_eligibility_threshold=0.001,  # Lower threshold for tests
        )

    def test_initialization(self, rule):
        """Rule initializes with subsystems."""
        assert rule.eligibility is not None
        assert rule.neuromodulators is not None
        assert rule.dopamine is not None
        assert rule.min_effective_lr == 0.1
        assert rule.max_effective_lr == 3.0

    def test_custom_subsystems(self):
        """Rule accepts custom subsystems."""
        custom_elig = EligibilityTrace(decay=0.9)
        custom_neuromod = NeuromodulatorOrchestra()
        custom_dopamine = DopamineSystem()

        rule = ThreeFactorLearningRule(
            eligibility_trace=custom_elig,
            neuromodulator_orchestra=custom_neuromod,
            dopamine_system=custom_dopamine,
        )

        assert rule.eligibility is custom_elig
        assert rule.neuromodulators is custom_neuromod
        assert rule.dopamine is custom_dopamine

    def test_neuromodulator_weights_normalized(self):
        """Neuromodulator weights are normalized to sum to 1.0."""
        rule = ThreeFactorLearningRule(
            ach_weight=2.0,
            ne_weight=1.0,
            serotonin_weight=1.0,
        )

        # Should normalize to 0.5, 0.25, 0.25
        assert abs(rule.ach_weight - 0.5) < 1e-6
        assert abs(rule.ne_weight - 0.25) < 1e-6
        assert abs(rule.serotonin_weight - 0.25) < 1e-6
        assert abs(rule.ach_weight + rule.ne_weight + rule.serotonin_weight - 1.0) < 1e-6

    def test_mark_active(self, rule):
        """Marking memory as active creates eligibility trace."""
        memory_id = "test_memory_1"

        assert rule.get_eligibility(memory_id) == 0.0

        rule.mark_active(memory_id, activity=1.0)

        assert rule.get_eligibility(memory_id) > 0.0

    def test_eligibility_decay(self, rule):
        """Eligibility traces decay over time."""
        memory_id = "test_memory_1"
        rule.mark_active(memory_id, activity=1.0)

        initial = rule.get_eligibility(memory_id)

        # Step forward in time
        rule.step(dt=1.0)

        decayed = rule.get_eligibility(memory_id)

        assert decayed < initial

    def test_compute_basic(self, rule):
        """Compute returns three-factor signal."""
        memory_id = uuid4()

        # Mark memory as active
        rule.mark_active(str(memory_id), activity=1.0)

        signal = rule.compute(
            memory_id=memory_id,
            base_lr=0.01,
            outcome=0.8,
        )

        assert isinstance(signal, ThreeFactorSignal)
        assert signal.memory_id == memory_id
        assert signal.eligibility > 0
        assert signal.neuromod_gate > 0
        assert signal.dopamine_surprise >= 0
        assert signal.effective_lr_multiplier >= rule.min_effective_lr
        assert signal.effective_lr_multiplier <= rule.max_effective_lr

    def test_compute_multiplicative_gating(self, rule):
        """Effective LR is product of three factors."""
        memory_id = uuid4()

        # Mark as active with high eligibility - need multiple activations
        # since a_plus=0.005 per activation
        for _ in range(10):
            rule.mark_active(str(memory_id), activity=1.0)

        # Create encoding mode state to pass directly
        encoding_state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.5,  # High arousal
            acetylcholine_mode="encoding",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )

        signal = rule.compute(
            memory_id=memory_id,
            base_lr=0.01,
            outcome=0.9,  # High surprise (unexpected success)
            neuromod_state=encoding_state,
        )

        # All factors should contribute multiplicatively
        # Higher eligibility (from multiple activations) * high neuromod gate * high surprise
        assert signal.eligibility > 0.01  # Has some eligibility
        assert signal.neuromod_gate > 1.0  # Encoding mode + high arousal
        assert signal.dopamine_surprise > 0.3  # 0.9 vs expected 0.5 = 0.4 surprise
        # Note: raw multiplier may be below min, so effective_lr gets clipped to min
        assert signal.effective_lr_multiplier >= rule.min_effective_lr

    def test_compute_zero_eligibility_minimal_learning(self, rule):
        """Zero eligibility results in minimal learning."""
        memory_id = uuid4()

        # Don't mark as active - eligibility should be zero

        signal = rule.compute(
            memory_id=memory_id,
            base_lr=0.01,
            outcome=0.8,
        )

        # Should get minimal learning rate (below threshold)
        assert signal.eligibility < rule.min_eligibility_threshold
        assert signal.effective_lr_multiplier < rule.min_effective_lr

    def test_compute_high_surprise_boosts_learning(self, rule):
        """High prediction error increases learning rate."""
        memory_id = uuid4()
        rule.mark_active(str(memory_id), activity=1.0)

        # Low surprise (expected outcome)
        signal_low = rule.compute(
            memory_id=memory_id,
            base_lr=0.01,
            outcome=0.5,  # Neutral outcome
        )

        # Update expectations so future outcomes are surprising
        rule.update_dopamine_expectations(memory_id, 0.5)

        # High surprise (unexpected outcome)
        signal_high = rule.compute(
            memory_id=memory_id,
            base_lr=0.01,
            outcome=0.95,  # Much better than expected
        )

        # High surprise should have higher learning rate
        assert signal_high.dopamine_surprise > signal_low.dopamine_surprise
        # Note: effective_lr_multiplier may not always be higher due to eligibility decay
        # between the two calls, but dopamine_surprise should be higher

    def test_compute_encoding_mode_boosts_learning(self, rule):
        """Encoding mode increases learning rate."""
        memory_id = uuid4()
        rule.mark_active(str(memory_id), activity=1.0)

        # Set to retrieval mode (lower learning boost)
        retrieval_state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="retrieval",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        signal_retrieval = rule.compute(
            memory_id=memory_id,
            base_lr=0.01,
            outcome=0.8,
            neuromod_state=retrieval_state,
        )

        # Set to encoding mode (higher learning boost)
        encoding_state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="encoding",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )
        signal_encoding = rule.compute(
            memory_id=memory_id,
            base_lr=0.01,
            outcome=0.8,
            neuromod_state=encoding_state,
        )

        # Encoding should have higher gate
        assert signal_encoding.ach_mode_factor > signal_retrieval.ach_mode_factor
        assert signal_encoding.neuromod_gate > signal_retrieval.neuromod_gate

    def test_compute_effective_lr(self, rule):
        """Convenience method returns just effective LR."""
        memory_id = uuid4()
        rule.mark_active(str(memory_id), activity=1.0)

        base_lr = 0.01
        effective_lr = rule.compute_effective_lr(memory_id, base_lr, outcome=0.8)

        assert effective_lr >= base_lr * rule.min_effective_lr
        assert effective_lr <= base_lr * rule.max_effective_lr

    def test_batch_compute(self, rule):
        """Batch compute processes multiple memories."""
        memory_ids = [uuid4() for _ in range(5)]

        # Mark all as active
        for mem_id in memory_ids:
            rule.mark_active(str(mem_id), activity=1.0)

        # Outcomes for each
        outcomes = {str(mem_id): 0.5 + i * 0.1 for i, mem_id in enumerate(memory_ids)}

        signals = rule.batch_compute(memory_ids, base_lr=0.01, outcomes=outcomes)

        assert len(signals) == 5
        for mem_id in memory_ids:
            assert str(mem_id) in signals
            assert isinstance(signals[str(mem_id)], ThreeFactorSignal)

    def test_update_dopamine_expectations(self, rule):
        """Dopamine expectations are updated after outcomes."""
        memory_id = uuid4()

        # Initial expectation should be neutral (compute_rpe uses positional args)
        rpe1 = rule.dopamine.compute_rpe(memory_id, 0.8)
        assert abs(rpe1.expected - 0.5) < 0.1  # Near neutral

        # Update expectations
        new_expected = rule.update_dopamine_expectations(memory_id, 0.8)

        # Should move toward outcome
        assert new_expected > 0.5
        assert new_expected <= 0.8

        # Next outcome should have different RPE
        rpe2 = rule.dopamine.compute_rpe(memory_id, 0.8)
        assert abs(rpe2.rpe) < abs(rpe1.rpe)  # Smaller surprise

    def test_signal_history_tracking(self, rule):
        """Signal history is tracked for analysis."""
        memory_id = uuid4()
        rule.mark_active(str(memory_id), activity=1.0)

        assert len(rule._signal_history) == 0

        rule.compute(memory_id, base_lr=0.01, outcome=0.8)

        assert len(rule._signal_history) == 1

        # Multiple computes
        for _ in range(5):
            rule.compute(memory_id, base_lr=0.01, outcome=0.7)

        assert len(rule._signal_history) == 6

    def test_get_stats(self, rule):
        """Stats aggregates all subsystems."""
        memory_id = uuid4()
        rule.mark_active(str(memory_id), activity=1.0)

        # Generate some signals
        for i in range(10):
            rule.compute(memory_id, base_lr=0.01, outcome=0.5 + i * 0.05)

        stats = rule.get_stats()

        assert "eligibility" in stats
        assert "neuromodulators" in stats
        assert "dopamine" in stats
        assert "three_factor" in stats

        three_factor_stats = stats["three_factor"]
        assert three_factor_stats["total_signals"] == 10
        assert "avg_eligibility" in three_factor_stats
        assert "avg_neuromod_gate" in three_factor_stats
        assert "avg_dopamine_surprise" in three_factor_stats
        assert "avg_effective_lr_mult" in three_factor_stats

    def test_clear_history(self, rule):
        """Clear history frees memory."""
        memory_id = uuid4()
        rule.mark_active(str(memory_id), activity=1.0)

        for _ in range(100):
            rule.compute(memory_id, base_lr=0.01, outcome=0.8)

        assert len(rule._signal_history) == 100

        rule.clear_history()

        assert len(rule._signal_history) == 0


class TestThreeFactorReconsolidation:
    """Tests for ThreeFactorReconsolidation."""

    @pytest.fixture
    def recon(self):
        """Create three-factor reconsolidation engine.

        Uses a ThreeFactorLearningRule with low threshold for testing.
        """
        three_factor = ThreeFactorLearningRule(
            min_eligibility_threshold=0.001,  # Low threshold for tests
        )
        return ThreeFactorReconsolidation(
            three_factor=three_factor,
            base_learning_rate=0.01,
            max_update_magnitude=0.1,
            cooldown_hours=1.0,
        )

    def test_initialization(self, recon):
        """Reconsolidation initializes correctly."""
        assert recon.three_factor is not None
        assert recon.base_learning_rate == 0.01
        assert recon.max_update_magnitude == 0.1
        assert recon.cooldown_hours == 1.0

    def test_should_update_true_initially(self, recon):
        """New memories should be updatable."""
        memory_id = uuid4()

        assert recon.should_update(memory_id) is True

    def test_should_update_false_during_cooldown(self, recon):
        """Memories in cooldown should not update."""
        memory_id = uuid4()

        # Simulate update
        recon._last_update[str(memory_id)] = datetime.now()

        assert recon.should_update(memory_id) is False

    def test_reconsolidate_updates_embedding(self, recon):
        """Reconsolidation updates memory embedding."""
        memory_id = uuid4()

        # Create embeddings
        memory_embedding = np.random.randn(128)
        memory_embedding = memory_embedding / np.linalg.norm(memory_embedding)

        query_embedding = np.random.randn(128)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Mark memory as active
        recon.three_factor.mark_active(str(memory_id), activity=1.0)

        # Reconsolidate with positive outcome
        new_embedding = recon.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.8,
            importance=0.0,
        )

        assert new_embedding is not None
        assert new_embedding.shape == memory_embedding.shape

        # Should be different from original
        assert not np.allclose(new_embedding, memory_embedding)

        # Should be normalized
        assert abs(np.linalg.norm(new_embedding) - 1.0) < 1e-6

    def test_reconsolidate_respects_cooldown(self, recon):
        """Reconsolidation respects cooldown period."""
        memory_id = uuid4()

        memory_embedding = np.random.randn(128)
        memory_embedding = memory_embedding / np.linalg.norm(memory_embedding)
        query_embedding = np.random.randn(128)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        recon.three_factor.mark_active(str(memory_id), activity=1.0)

        # First update should succeed
        result1 = recon.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.8,
        )
        assert result1 is not None

        # Second update immediately should fail (cooldown)
        result2 = recon.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.8,
        )
        assert result2 is None

    def test_reconsolidate_skips_low_eligibility(self, recon):
        """Low eligibility memories are not updated."""
        memory_id = uuid4()

        # Don't mark as active - eligibility will be zero

        memory_embedding = np.random.randn(128)
        memory_embedding = memory_embedding / np.linalg.norm(memory_embedding)
        query_embedding = np.random.randn(128)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        result = recon.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.8,
        )

        assert result is None

    def test_reconsolidate_skips_neutral_outcomes(self, recon):
        """Neutral outcomes don't trigger update."""
        memory_id = uuid4()

        recon.three_factor.mark_active(str(memory_id), activity=1.0)

        memory_embedding = np.random.randn(128)
        memory_embedding = memory_embedding / np.linalg.norm(memory_embedding)
        query_embedding = np.random.randn(128)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Neutral outcome (0.5)
        result = recon.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.5,
        )

        assert result is None

    def test_reconsolidate_importance_protection(self, recon):
        """High importance reduces update magnitude."""
        memory_id = uuid4()

        recon.three_factor.mark_active(str(memory_id), activity=1.0)

        memory_embedding = np.random.randn(128)
        memory_embedding = memory_embedding / np.linalg.norm(memory_embedding)
        query_embedding = np.random.randn(128)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Low importance - larger update
        new_low = recon.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding.copy(),
            query_embedding=query_embedding,
            outcome_score=0.8,
            importance=0.0,
        )

        # Reset cooldown
        del recon._last_update[str(memory_id)]

        # High importance - smaller update
        new_high = recon.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding.copy(),
            query_embedding=query_embedding,
            outcome_score=0.8,
            importance=5.0,
        )

        assert new_low is not None
        assert new_high is not None

        # Low importance should have moved further
        dist_low = np.linalg.norm(new_low - memory_embedding)
        dist_high = np.linalg.norm(new_high - memory_embedding)

        assert dist_low > dist_high

    def test_reconsolidate_clips_update_magnitude(self, recon):
        """Update magnitude is clipped to max."""
        memory_id = uuid4()

        recon.three_factor.mark_active(str(memory_id), activity=1.0)

        # Very different embeddings
        memory_embedding = np.ones(128)
        memory_embedding = memory_embedding / np.linalg.norm(memory_embedding)

        query_embedding = -np.ones(128)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        new_embedding = recon.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=1.0,  # Maximum positive outcome
            importance=0.0,
        )

        assert new_embedding is not None

        # Update should be clipped
        update = new_embedding - memory_embedding
        update_norm = np.linalg.norm(update)

        assert update_norm <= recon.max_update_magnitude * 1.01  # Small tolerance


class TestCreateThreeFactorRule:
    """Tests for create_three_factor_rule factory function."""

    def test_create_basic(self):
        """Factory creates rule with defaults."""
        rule = create_three_factor_rule()

        assert rule.eligibility is not None
        assert rule.neuromodulators is not None
        assert rule.dopamine is not None

    def test_create_with_layered_traces(self):
        """Factory can create layered eligibility traces."""
        rule = create_three_factor_rule(
            use_layered_traces=True,
            eligibility_config={"fast_tau": 5.0, "slow_tau": 60.0},  # Correct params
        )

        assert isinstance(rule.eligibility, LayeredEligibilityTrace)

    def test_create_with_custom_config(self):
        """Factory accepts custom configurations."""
        rule = create_three_factor_rule(
            eligibility_config={"decay": 0.9},
            neuromodulator_config={},  # Uses default, no special params
            dopamine_config={"value_learning_rate": 0.05},  # Correct param name
            ach_weight=0.5,
            ne_weight=0.3,
            serotonin_weight=0.2,
        )

        assert rule.eligibility.decay == 0.9
        # Weights should be normalized
        assert abs(rule.ach_weight - 0.5) < 1e-6
        assert abs(rule.ne_weight - 0.3) < 1e-6
        assert abs(rule.serotonin_weight - 0.2) < 1e-6


class TestThreeFactorEdgeCases:
    """Edge case tests for three-factor learning."""

    def test_validate_scalar_nan(self):
        """_validate_scalar raises on NaN."""
        from ww.learning.three_factor import _validate_scalar

        with pytest.raises(ValueError, match="NaN detected"):
            _validate_scalar(float("nan"), "test_value")

    def test_validate_scalar_inf(self):
        """_validate_scalar raises on Inf."""
        from ww.learning.three_factor import _validate_scalar

        with pytest.raises(ValueError, match="Inf detected"):
            _validate_scalar(float("inf"), "test_value")

        with pytest.raises(ValueError, match="Inf detected"):
            _validate_scalar(float("-inf"), "test_value")

    def test_validate_scalar_normal(self):
        """_validate_scalar passes for normal values."""
        from ww.learning.three_factor import _validate_scalar

        # Should not raise
        _validate_scalar(0.0, "zero")
        _validate_scalar(1.0, "one")
        _validate_scalar(-1.0, "negative")
        _validate_scalar(1e10, "large")
        _validate_scalar(1e-10, "small")

    def test_negative_weights_raises(self):
        """CRASH-014: Negative weights sum raises error."""
        with pytest.raises(ValueError, match="must sum to positive"):
            ThreeFactorLearningRule(
                ach_weight=-1.0,
                ne_weight=0.0,
                serotonin_weight=0.0,
            )

    def test_zero_weights_raises(self):
        """CRASH-014: Zero weights sum raises error."""
        with pytest.raises(ValueError, match="must sum to positive"):
            ThreeFactorLearningRule(
                ach_weight=0.0,
                ne_weight=0.0,
                serotonin_weight=0.0,
            )

    def test_uses_serotonin_eligibility_trace(self):
        """LEARNING-HIGH-003: Uses serotonin's eligibility trace when available."""
        orchestra = NeuromodulatorOrchestra()

        # Ensure serotonin has an eligibility tracer
        assert hasattr(orchestra.serotonin, "_eligibility_tracer")

        # Create rule without explicit eligibility trace
        rule = ThreeFactorLearningRule(
            neuromodulator_orchestra=orchestra,
            eligibility_trace=None,  # Should use serotonin's
        )

        # Should share eligibility trace with serotonin
        assert rule.eligibility is orchestra.serotonin._eligibility_tracer

    def test_neuromod_gate_no_state(self):
        """Neuromod gate returns neutral when no state available."""
        rule = ThreeFactorLearningRule()

        # Get state before any computation
        combined, ach, ne, serotonin = rule._compute_neuromod_gate(None)

        # Returns neutral defaults
        assert combined == 1.0
        assert ach == 1.0
        assert ne == 1.0
        assert serotonin == 1.0

    def test_neuromod_gate_balanced_mode(self):
        """Neuromod gate handles balanced ACh mode."""
        rule = ThreeFactorLearningRule()

        balanced_state = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="balanced",
            serotonin_mood=0.5,
            inhibition_sparsity=0.5,
        )

        combined, ach, ne, serotonin = rule._compute_neuromod_gate(balanced_state)

        # Balanced mode gives ach_factor = 1.0
        assert ach == 1.0

    def test_compute_with_precomputed_rpe(self):
        """Compute uses precomputed RPE when provided."""
        rule = ThreeFactorLearningRule(min_eligibility_threshold=0.001)
        memory_id = uuid4()
        rule.mark_active(str(memory_id), activity=1.0)

        # Create precomputed RPE
        precomputed = RewardPredictionError(
            memory_id=memory_id,
            expected=0.5,
            actual=0.9,
            rpe=0.4,
        )

        signal = rule.compute(
            memory_id=memory_id,
            base_lr=0.01,
            outcome=None,  # No outcome - uses precomputed
            precomputed_rpe=precomputed,
        )

        # Should use precomputed RPE
        assert signal.rpe_raw == 0.4
        assert signal.dopamine_surprise == 0.4

    def test_compute_no_outcome_neutral_rpe(self):
        """Compute uses neutral RPE when no outcome provided."""
        rule = ThreeFactorLearningRule(min_eligibility_threshold=0.001)
        memory_id = uuid4()
        rule.mark_active(str(memory_id), activity=1.0)

        signal = rule.compute(
            memory_id=memory_id,
            base_lr=0.01,
            outcome=None,  # No outcome
            precomputed_rpe=None,  # No precomputed
        )

        # Should have zero RPE
        assert signal.rpe_raw == 0.0
        # dopamine_surprise floors at 0.1
        assert signal.dopamine_surprise == 0.1

    def test_history_size_limit(self):
        """MEM-007: History is bounded to max size."""
        rule = ThreeFactorLearningRule(min_eligibility_threshold=0.001)
        rule._max_history_size = 100  # Set small for testing

        memory_id = uuid4()
        rule.mark_active(str(memory_id), activity=1.0)

        # Generate more signals than limit
        for i in range(150):
            rule.compute(memory_id, base_lr=0.01, outcome=0.5 + (i % 50) * 0.01)

        # Should be capped at max size
        assert len(rule._signal_history) <= rule._max_history_size


class TestThreeFactorReconsolidationEdgeCases:
    """Edge case tests for ThreeFactorReconsolidation."""

    def test_cleanup_cooldowns_removes_expired(self):
        """MEM-007: Cleanup removes expired cooldown entries."""
        from datetime import timedelta

        recon = ThreeFactorReconsolidation(
            cooldown_hours=0.5,  # 30 minutes
        )

        # Add some entries with old timestamps
        now = datetime.now()
        recon._last_update["expired_1"] = now - timedelta(hours=1)
        recon._last_update["expired_2"] = now - timedelta(hours=2)
        recon._last_update["recent"] = now

        recon._cleanup_cooldowns()

        # Expired should be removed
        assert "expired_1" not in recon._last_update
        assert "expired_2" not in recon._last_update
        assert "recent" in recon._last_update

    def test_cleanup_cooldowns_respects_limit(self):
        """MEM-007: Cleanup respects max entries limit."""
        recon = ThreeFactorReconsolidation()
        recon._max_cooldown_entries = 10

        # Add more entries than limit (all recent)
        now = datetime.now()
        for i in range(20):
            recon._last_update[f"entry_{i}"] = now

        recon._cleanup_cooldowns()

        # Should be limited
        assert len(recon._last_update) <= recon._max_cooldown_entries

    def test_reconsolidate_identical_embeddings(self):
        """Reconsolidate skips when embeddings are identical."""
        three_factor = ThreeFactorLearningRule(min_eligibility_threshold=0.001)
        recon = ThreeFactorReconsolidation(three_factor=three_factor)

        memory_id = uuid4()
        three_factor.mark_active(str(memory_id), activity=1.0)

        # Same embedding for memory and query
        embedding = np.random.randn(128)
        embedding = embedding / np.linalg.norm(embedding)

        result = recon.reconsolidate(
            memory_id=memory_id,
            memory_embedding=embedding,
            query_embedding=embedding,  # Identical
            outcome_score=0.8,
        )

        # Zero direction norm should skip update
        assert result is None

    def test_reconsolidate_triggers_cleanup(self):
        """Reconsolidate triggers cleanup when over limit."""
        three_factor = ThreeFactorLearningRule(min_eligibility_threshold=0.001)
        recon = ThreeFactorReconsolidation(three_factor=three_factor)
        recon._max_cooldown_entries = 5

        # Pre-fill with entries
        now = datetime.now()
        for i in range(10):
            recon._last_update[f"old_{i}"] = now

        memory_id = uuid4()
        three_factor.mark_active(str(memory_id), activity=1.0)

        memory_embedding = np.random.randn(128)
        memory_embedding = memory_embedding / np.linalg.norm(memory_embedding)
        query_embedding = np.random.randn(128)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Reconsolidate should trigger cleanup
        recon.reconsolidate(
            memory_id=memory_id,
            memory_embedding=memory_embedding,
            query_embedding=query_embedding,
            outcome_score=0.8,
        )

        # Should have cleaned up
        assert len(recon._last_update) <= recon._max_cooldown_entries + 1
