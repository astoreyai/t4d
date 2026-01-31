"""Tests for dopamine reward prediction error module."""
import pytest
import numpy as np
from uuid import uuid4

from t4dm.learning.dopamine import (
    LearnedValueEstimator,
    DopamineSystem,
    RewardPredictionError,
    compute_rpe,
)


class TestLearnedValueEstimator:
    """Tests for LearnedValueEstimator neural network."""

    def test_init_creates_weights(self):
        """Test initialization creates weight matrices."""
        estimator = LearnedValueEstimator(embedding_dim=128, hidden_dim=32)

        assert estimator.W1.shape == (128, 32)
        assert estimator.b1.shape == (32,)
        assert estimator.W2.shape == (32, 1)
        assert estimator.b2.shape == (1,)

    def test_init_default_dimensions(self):
        """Test default dimensions are 1024 -> 256 -> 1."""
        estimator = LearnedValueEstimator()

        assert estimator.embedding_dim == 1024
        assert estimator.hidden_dim == 256
        assert estimator.W1.shape == (1024, 256)

    def test_estimate_returns_value_in_range(self):
        """Test estimate returns value between 0 and 1."""
        estimator = LearnedValueEstimator(embedding_dim=64, hidden_dim=16)
        embedding = np.random.randn(64)

        value = estimator.estimate(embedding)

        assert 0.0 <= value <= 1.0

    def test_estimate_pads_short_embedding(self):
        """Test estimate pads embeddings shorter than expected."""
        estimator = LearnedValueEstimator(embedding_dim=64, hidden_dim=16)
        short_embedding = np.random.randn(32)  # Half the expected size

        value = estimator.estimate(short_embedding)

        assert 0.0 <= value <= 1.0

    def test_estimate_truncates_long_embedding(self):
        """Test estimate truncates embeddings longer than expected."""
        estimator = LearnedValueEstimator(embedding_dim=64, hidden_dim=16)
        long_embedding = np.random.randn(128)  # Double the expected size

        value = estimator.estimate(long_embedding)

        assert 0.0 <= value <= 1.0

    def test_estimate_caches_activations(self):
        """Test estimate caches intermediate activations for backprop."""
        estimator = LearnedValueEstimator(embedding_dim=64, hidden_dim=16)
        embedding = np.random.randn(64)

        estimator.estimate(embedding)

        assert estimator._last_embedding is not None
        assert estimator._last_hidden is not None
        assert estimator._last_output is not None

    def test_update_modifies_weights(self):
        """Test update changes weights based on TD error."""
        estimator = LearnedValueEstimator(embedding_dim=64, hidden_dim=16)
        embedding = np.random.randn(64)

        # Get initial weights
        W1_before = estimator.W1.copy()

        # Estimate and update with error
        estimator.estimate(embedding)
        estimator.update(embedding, td_error=0.5)

        # Weights should have changed
        assert not np.allclose(estimator.W1, W1_before)

    def test_update_increments_count(self):
        """Test update increments update count."""
        estimator = LearnedValueEstimator(embedding_dim=64, hidden_dim=16)
        embedding = np.random.randn(64)

        assert estimator._update_count == 0

        estimator.estimate(embedding)
        estimator.update(embedding, td_error=0.1)

        assert estimator._update_count == 1

    def test_update_accumulates_td_error(self):
        """Test update accumulates total TD error."""
        estimator = LearnedValueEstimator(embedding_dim=64, hidden_dim=16)
        embedding = np.random.randn(64)

        estimator.estimate(embedding)
        estimator.update(embedding, td_error=0.3)

        estimator.estimate(embedding)
        estimator.update(embedding, td_error=0.2)

        assert estimator._total_td_error == pytest.approx(0.5, abs=0.01)

    def test_get_stats_returns_metrics(self):
        """Test get_stats returns update statistics."""
        estimator = LearnedValueEstimator(embedding_dim=64, hidden_dim=16)
        embedding = np.random.randn(64)

        estimator.estimate(embedding)
        estimator.update(embedding, td_error=0.5)

        stats = estimator.get_stats()

        assert "update_count" in stats
        assert "avg_td_error" in stats
        assert stats["update_count"] == 1

    def test_save_and_load_state(self):
        """Test save_state and load_state preserve weights."""
        estimator1 = LearnedValueEstimator(embedding_dim=64, hidden_dim=16)
        embedding = np.random.randn(64)

        # Train a bit
        estimator1.estimate(embedding)
        estimator1.update(embedding, td_error=0.5)

        # Save state
        state = estimator1.save_state()

        # Create new estimator and load
        estimator2 = LearnedValueEstimator(embedding_dim=64, hidden_dim=16)
        estimator2.load_state(state)

        # Should produce same output
        val1 = estimator1.estimate(embedding)
        val2 = estimator2.estimate(embedding)
        assert val1 == pytest.approx(val2, abs=0.001)


class TestDopamineSystem:
    """Tests for DopamineSystem reward prediction error system."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        system = DopamineSystem()

        assert system.default_expected == 0.5
        assert system.value_learning_rate == 0.1

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        system = DopamineSystem(
            default_expected=0.3,
            value_learning_rate=0.2,
            surprise_threshold=0.1,
        )

        assert system.default_expected == 0.3
        assert system.value_learning_rate == 0.2
        assert system.surprise_threshold == 0.1

    def test_get_expected_value_default(self):
        """Test get_expected_value returns default for new memory."""
        system = DopamineSystem(default_expected=0.6)
        memory_id = uuid4()

        value = system.get_expected_value(memory_id)

        assert value == 0.6

    def test_compute_rpe_returns_rpe_object(self):
        """Test compute_rpe returns RewardPredictionError."""
        system = DopamineSystem()
        memory_id = uuid4()

        rpe = system.compute_rpe(memory_id=memory_id, actual_outcome=0.8)

        assert isinstance(rpe, RewardPredictionError)
        assert rpe.memory_id == memory_id

    def test_rpe_positive_surprise(self):
        """Test positive RPE when outcome exceeds expectation."""
        system = DopamineSystem(default_expected=0.5)
        memory_id = uuid4()

        rpe = system.compute_rpe(memory_id=memory_id, actual_outcome=1.0)

        # RPE should be positive (surprise)
        assert rpe.rpe > 0
        assert rpe.is_positive_surprise

    def test_rpe_negative_surprise(self):
        """Test negative RPE when outcome below expectation."""
        system = DopamineSystem(default_expected=0.5)
        memory_id = uuid4()

        rpe = system.compute_rpe(memory_id=memory_id, actual_outcome=0.0)

        # RPE should be negative (disappointment)
        assert rpe.rpe < 0
        assert rpe.is_negative_surprise

    def test_update_expectations_changes_value(self):
        """Test update_expectations modifies value estimate."""
        system = DopamineSystem(default_expected=0.5, value_learning_rate=0.5)
        memory_id = uuid4()

        initial = system.get_expected_value(memory_id)
        system.update_expectations(memory_id, actual_outcome=1.0)
        updated = system.get_expected_value(memory_id)

        assert updated != initial
        assert updated > initial  # Moved toward 1.0

    def test_batch_compute_rpe(self):
        """Test computing RPE for multiple memories."""
        system = DopamineSystem()

        outcomes = {
            str(uuid4()): 0.7,
            str(uuid4()): 0.3,
            str(uuid4()): 0.9,
        }

        rpes = system.batch_compute_rpe(outcomes)

        assert len(rpes) == 3
        assert all(isinstance(r, RewardPredictionError) for r in rpes.values())

    def test_get_uncertainty_decreases_with_observations(self):
        """Test uncertainty decreases as we observe more outcomes."""
        system = DopamineSystem()
        memory_id = uuid4()

        initial_uncertainty = system.get_uncertainty(memory_id)

        # Observe several outcomes
        for _ in range(5):
            system.update_expectations(memory_id, actual_outcome=0.7)

        final_uncertainty = system.get_uncertainty(memory_id)

        assert final_uncertainty < initial_uncertainty

    def test_modulate_learning_rate_scales_with_surprise(self):
        """Test modulated learning rate increases with |RPE|."""
        system = DopamineSystem()
        memory_id = uuid4()

        # High surprise
        rpe_high = RewardPredictionError(
            memory_id=memory_id,
            expected=0.5,
            actual=1.0,
            rpe=0.5,
        )

        # Low surprise
        rpe_low = RewardPredictionError(
            memory_id=memory_id,
            expected=0.5,
            actual=0.55,
            rpe=0.05,
        )

        lr_high = system.modulate_learning_rate(0.1, rpe_high)
        lr_low = system.modulate_learning_rate(0.1, rpe_low)

        assert lr_high > lr_low

    def test_get_stats_returns_statistics(self):
        """Test get_stats returns system statistics."""
        system = DopamineSystem()
        memory_id = uuid4()

        system.compute_rpe(memory_id, actual_outcome=0.8)
        stats = system.get_stats()

        assert "total_signals" in stats
        assert "positive_surprises" in stats
        assert "memories_tracked" in stats

    def test_learned_mode_uses_network(self):
        """Test learned mode creates value network."""
        system = DopamineSystem(use_learned_values=True, embedding_dim=64)

        assert system._value_network is not None
        assert system.use_learned_values

    def test_learned_mode_estimate_from_embedding(self):
        """Test learned mode estimates value from embedding."""
        system = DopamineSystem(use_learned_values=True, embedding_dim=64)
        memory_id = uuid4()
        embedding = np.random.randn(64)

        value = system.get_expected_value(memory_id, embedding=embedding)

        assert 0.0 <= value <= 1.0

    def test_save_and_load_state(self):
        """Test save_state and load_state preserve system."""
        system1 = DopamineSystem(default_expected=0.6)
        memory_id = uuid4()

        system1.update_expectations(memory_id, actual_outcome=0.9)
        state = system1.save_state()

        system2 = DopamineSystem()
        system2.load_state(state)

        assert system2.default_expected == 0.6
        assert system2.get_expected_value(memory_id) == system1.get_expected_value(memory_id)

    def test_reset_expectations_clears_values(self):
        """Test reset_expectations clears all learned values."""
        system = DopamineSystem()
        memory_id = uuid4()

        system.update_expectations(memory_id, actual_outcome=0.9)
        system.reset_expectations()

        # Should return default again
        value = system.get_expected_value(memory_id)
        assert value == system.default_expected


class TestTDLambda:
    """Tests for TD(λ) eligibility traces."""

    def test_mark_memory_active_sets_trace(self):
        """Test mark_memory_active sets eligibility to 1.0."""
        system = DopamineSystem()
        memory_id = uuid4()

        system.mark_memory_active(memory_id)

        assert system.get_eligibility_trace(memory_id) == 1.0

    def test_decay_eligibility_traces(self):
        """Test traces decay over time."""
        system = DopamineSystem(td_lambda=0.9, discount_gamma=0.95)
        memory_id = uuid4()

        system.mark_memory_active(memory_id)
        initial = system.get_eligibility_trace(memory_id)

        system.decay_eligibility_traces()
        after_decay = system.get_eligibility_trace(memory_id)

        assert after_decay < initial
        assert after_decay == pytest.approx(0.9 * 0.95, abs=0.01)

    def test_clear_eligibility_traces(self):
        """Test clearing all eligibility traces."""
        system = DopamineSystem()
        memory_id = uuid4()

        system.mark_memory_active(memory_id)
        system.clear_eligibility_traces()

        assert system.get_eligibility_trace(memory_id) == 0.0

    def test_update_with_td_lambda_updates_eligible_memories(self):
        """Test TD(λ) update modifies eligible memories."""
        system = DopamineSystem()
        memory_id = uuid4()

        # Mark memory active
        system.mark_memory_active(memory_id)
        system.update_expectations(memory_id, actual_outcome=0.5)

        initial_value = system.get_expected_value(memory_id)

        # Apply TD error
        updated = system.update_with_td_lambda(td_error=0.3)

        assert str(memory_id) in updated
        assert updated[str(memory_id)] != initial_value

    def test_process_reward_with_traces(self):
        """Test full reward processing with traces."""
        system = DopamineSystem()
        current_id = uuid4()
        next_id = uuid4()

        td_error, updated = system.process_reward_with_traces(
            reward=1.0,
            current_memory_id=current_id,
            next_memory_id=next_id,
        )

        assert isinstance(td_error, float)
        assert isinstance(updated, dict)


class TestRewardPredictionError:
    """Tests for RewardPredictionError dataclass."""

    def test_rpe_creation(self):
        """Test RPE can be created with required fields."""
        rpe = RewardPredictionError(
            memory_id=uuid4(),
            expected=0.5,
            actual=0.8,
            rpe=0.3,
        )

        assert rpe.rpe == 0.3
        assert rpe.expected == 0.5
        assert rpe.actual == 0.8

    def test_is_positive_surprise(self):
        """Test is_positive_surprise property."""
        rpe = RewardPredictionError(
            memory_id=uuid4(),
            expected=0.5,
            actual=0.9,
            rpe=0.4,
        )
        assert rpe.is_positive_surprise

    def test_is_negative_surprise(self):
        """Test is_negative_surprise property."""
        rpe = RewardPredictionError(
            memory_id=uuid4(),
            expected=0.5,
            actual=0.1,
            rpe=-0.4,
        )
        assert rpe.is_negative_surprise

    def test_surprise_magnitude(self):
        """Test surprise_magnitude property."""
        rpe = RewardPredictionError(
            memory_id=uuid4(),
            expected=0.5,
            actual=0.1,
            rpe=-0.4,
        )
        assert rpe.surprise_magnitude == 0.4


class TestComputeRPE:
    """Tests for compute_rpe convenience function."""

    def test_compute_rpe_basic(self):
        """Test basic RPE computation."""
        rpe = compute_rpe(actual=0.8, expected=0.5)
        assert rpe == pytest.approx(0.3, abs=0.001)

    def test_compute_rpe_negative(self):
        """Test negative RPE computation."""
        rpe = compute_rpe(actual=0.2, expected=0.5)
        assert rpe == pytest.approx(-0.3, abs=0.001)

    def test_compute_rpe_default_expected(self):
        """Test RPE with default expected value."""
        rpe = compute_rpe(actual=0.7)  # default expected=0.5
        assert rpe == pytest.approx(0.2, abs=0.001)
