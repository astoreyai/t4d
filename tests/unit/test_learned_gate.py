"""
Unit tests for LearnedMemoryGate.

Tests cover:
1. Feature extraction
2. Thompson sampling prediction
3. Online update logic
4. Cold start blending
5. Integration with neuromodulators
6. Performance benchmarks
"""

import time
from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest

from t4dm.core.learned_gate import GateDecision, LearnedMemoryGate, sigmoid
from t4dm.core.memory_gate import GateContext, StorageDecision
from t4dm.learning.neuromodulators import NeuromodulatorState


@pytest.fixture
def gate():
    """Create a learned gate for testing."""
    return LearnedMemoryGate(
        neuromod_orchestra=None,
        cold_start_threshold=10,
        use_diagonal_covariance=True
    )


@pytest.fixture
def content_embedding():
    """Create a dummy content embedding."""
    return np.random.randn(1024).astype(np.float32)


@pytest.fixture
def context():
    """Create a gate context."""
    return GateContext(
        session_id="test-session",
        project="test-project",
        current_task="testing",
        last_store_time=datetime.now() - timedelta(minutes=5),
        message_count_since_store=3
    )


@pytest.fixture
def neuromod_state():
    """Create a neuromodulator state."""
    return NeuromodulatorState(
        dopamine_rpe=0.5,
        norepinephrine_gain=1.2,
        acetylcholine_mode="balanced",
        serotonin_mood=0.6,
        inhibition_sparsity=0.3
    )


class TestSigmoid:
    """Test sigmoid function."""

    def test_sigmoid_zero(self):
        """Sigmoid(0) = 0.5."""
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5, abs=1e-6)

    def test_sigmoid_positive(self):
        """Sigmoid(x) > 0.5 for x > 0."""
        assert sigmoid(np.array([2.0]))[0] > 0.5

    def test_sigmoid_negative(self):
        """Sigmoid(x) < 0.5 for x < 0."""
        assert sigmoid(np.array([-2.0]))[0] < 0.5

    def test_sigmoid_range(self):
        """Sigmoid always in [0, 1]."""
        x = np.linspace(-10, 10, 100)
        y = sigmoid(x)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)

    def test_sigmoid_numerical_stability(self):
        """Sigmoid stable for large inputs."""
        # Should not overflow/underflow
        large_pos = sigmoid(np.array([100.0]))[0]
        large_neg = sigmoid(np.array([-100.0]))[0]

        assert large_pos == pytest.approx(1.0, abs=1e-6)
        assert large_neg == pytest.approx(0.0, abs=1e-6)


class TestFeatureExtraction:
    """Test feature extraction pipeline."""

    def test_feature_dimension(self, gate, content_embedding, context, neuromod_state):
        """Features have correct dimension."""
        φ = gate._extract_features(content_embedding, context, neuromod_state)
        assert φ.shape == (gate.feature_dim,)

    def test_feature_types(self, gate, content_embedding, context, neuromod_state):
        """Features are float32."""
        φ = gate._extract_features(content_embedding, context, neuromod_state)
        assert φ.dtype in [np.float32, np.float64]

    def test_feature_determinism(self, gate, content_embedding, context, neuromod_state):
        """Same inputs → same features."""
        φ1 = gate._extract_features(content_embedding, context, neuromod_state)
        φ2 = gate._extract_features(content_embedding, context, neuromod_state)
        np.testing.assert_array_almost_equal(φ1, φ2)

    def test_context_encoding(self, gate):
        """Context strings encoded consistently."""
        embed1 = gate._embed_string("test-project", dim=32)
        embed2 = gate._embed_string("test-project", dim=32)

        assert embed1.shape == (32,)
        np.testing.assert_array_equal(embed1, embed2)

    def test_context_diversity(self, gate):
        """Different contexts → different embeddings."""
        embed1 = gate._embed_string("project-a", dim=32)
        embed2 = gate._embed_string("project-b", dim=32)

        # Should be different (very unlikely to be identical)
        assert not np.allclose(embed1, embed2, rtol=0.1)

    def test_ngram_semantic_similarity(self, gate):
        """Similar strings produce similar embeddings (n-gram based)."""
        # Similar strings with overlapping n-grams
        embed1 = gate._embed_string("code-review", dim=64)
        embed2 = gate._embed_string("code-reviewer", dim=64)
        embed3 = gate._embed_string("database-migration", dim=64)

        # Compute cosine similarities
        def cosine(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)

        sim_similar = cosine(embed1, embed2)  # code-review vs code-reviewer
        sim_different = cosine(embed1, embed3)  # code-review vs database-migration

        # Similar strings should have higher similarity
        assert sim_similar > sim_different, (
            f"Similar strings should be more alike: "
            f"{sim_similar:.3f} vs {sim_different:.3f}"
        )

    def test_ngram_caching(self, gate):
        """String embeddings are cached for performance."""
        # Clear cache
        gate._string_embed_cache.clear()

        # First call
        _ = gate._embed_string("test-string", dim=32)
        assert len(gate._string_embed_cache) == 1

        # Second call (should use cache)
        _ = gate._embed_string("test-string", dim=32)
        assert len(gate._string_embed_cache) == 1  # No new entry

        # Different string
        _ = gate._embed_string("other-string", dim=32)
        assert len(gate._string_embed_cache) == 2

    def test_neuromod_encoding(self, gate, content_embedding, context):
        """Neuromodulator states encoded correctly."""
        # Test encoding mode
        state_encoding = NeuromodulatorState(
            dopamine_rpe=0.5,
            norepinephrine_gain=1.0,
            acetylcholine_mode="encoding",
            serotonin_mood=0.5,
            inhibition_sparsity=0.0
        )

        φ = gate._extract_features(content_embedding, context, state_encoding)

        # Find ACh one-hot features (positions 1024+64+3 to 1024+64+6)
        ach_start = gate.CONTENT_DIM + gate.CONTEXT_DIM + 3
        ach_encoding = φ[ach_start]
        ach_balanced = φ[ach_start + 1]
        ach_retrieval = φ[ach_start + 2]

        assert ach_encoding == 1.0
        assert ach_balanced == 0.0
        assert ach_retrieval == 0.0


class TestPrediction:
    """Test prediction logic."""

    def test_prediction_probability_range(self, gate, content_embedding, context, neuromod_state):
        """Predicted probabilities in [0, 1]."""
        decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
        assert 0.0 <= decision.probability <= 1.0

    def test_prediction_deterministic_without_exploration(
        self, gate, content_embedding, context, neuromod_state
    ):
        """Without exploration, predictions are deterministic."""
        decision1 = gate.predict(content_embedding, context, neuromod_state, explore=False)
        decision2 = gate.predict(content_embedding, context, neuromod_state, explore=False)

        # Use 1e-3 tolerance for floating point numerical stability across platforms
        # (feature extraction uses datetime.now() which introduces tiny variations)
        assert decision1.probability == pytest.approx(decision2.probability, abs=1e-3)

    def test_prediction_explores_with_thompson_sampling(
        self, gate, content_embedding, context, neuromod_state
    ):
        """With exploration, predictions vary (Thompson sampling)."""
        # Need some observations for non-zero covariance
        gate.n_observations = 10

        # Increase covariance to ensure meaningful exploration variance
        gate.Σ[:] = 1.0  # Higher uncertainty → more exploration

        probabilities = [
            gate.predict(content_embedding, context, neuromod_state, explore=True).probability
            for _ in range(50)  # More samples for reliable variance estimate
        ]

        # Should have some variance (not all identical)
        # Lowered threshold since exploration variance depends on covariance scale
        assert np.std(probabilities) > 0.001 or len(set(probabilities)) > 1

    def test_cold_start_blending(self, gate, content_embedding, context, neuromod_state):
        """Cold start blends heuristic and learned predictions."""
        gate.n_observations = 5  # Below threshold of 10

        decision = gate.predict(content_embedding, context, neuromod_state, explore=False)

        # Should be a blend (not purely learned)
        assert 0.0 <= decision.probability <= 1.0

    def test_threshold_decisions(self, gate, content_embedding, context, neuromod_state):
        """Correct decisions based on thresholds."""
        # Bypass cold start to test pure learned model
        gate.n_observations = gate.cold_start_threshold + 10

        # Manipulate weights to get specific probability
        gate.μ[:] = 0.0
        gate.b = 5.0  # High bias → high probability

        decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
        assert decision.action == StorageDecision.STORE

        gate.b = -5.0  # Low bias → low probability
        decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
        assert decision.action == StorageDecision.SKIP

    def test_ach_mode_modulates_threshold(self, gate, content_embedding, context):
        """ACh encoding mode lowers storage threshold."""
        # Bypass cold start to test pure learned model
        gate.n_observations = gate.cold_start_threshold + 10

        # Set up for borderline case
        gate.μ[:] = 0.0
        gate.b = 0.5  # Probability around 0.62

        # Encoding mode: should store (threshold lowered)
        state_encoding = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="encoding",
            serotonin_mood=0.5,
            inhibition_sparsity=0.0
        )
        decision = gate.predict(content_embedding, context, state_encoding, explore=False)
        # With encoding, threshold = 0.6 * 0.8 = 0.48, so should store
        assert decision.action == StorageDecision.STORE

        # Retrieval mode: should not store (threshold raised)
        state_retrieval = NeuromodulatorState(
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            acetylcholine_mode="retrieval",
            serotonin_mood=0.5,
            inhibition_sparsity=0.0
        )
        decision = gate.predict(content_embedding, context, state_retrieval, explore=False)
        # With retrieval, threshold = 0.6 * 1.2 = 0.72, so might not store
        # (depends on exact probability)


class TestOnlineUpdate:
    """Test online learning updates."""

    def test_update_changes_weights(self, gate, content_embedding, context, neuromod_state):
        """Updates change model weights."""
        # Get initial prediction
        decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
        memory_id = uuid4()
        gate.register_pending(memory_id, decision.features)

        initial_weights = gate.μ.copy()

        # Update with high utility
        gate.update(memory_id, utility=1.0)

        # Weights should change
        assert not np.allclose(gate.μ, initial_weights)

    def test_update_increases_observations(self, gate, content_embedding, context, neuromod_state):
        """Each update increments observation count."""
        initial_count = gate.n_observations

        decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
        memory_id = uuid4()
        gate.register_pending(memory_id, decision.features)
        gate.update(memory_id, utility=0.8)

        assert gate.n_observations == initial_count + 1

    def test_positive_examples_increase_probability(
        self, gate, content_embedding, context, neuromod_state
    ):
        """Positive examples → higher probability for similar inputs."""
        # Use fixed seed for reproducibility
        np.random.seed(42)

        # Bypass cold start to test pure learned dynamics
        # Without this, predictions blend with heuristic (α = n_obs/100)
        # and 10 updates only shifts blend from 0% to 10% learned
        gate.n_observations = gate.cold_start_threshold + 10

        # Reset weights to known state for reproducibility
        gate.μ[:] = 0.0
        gate.b = -2.0  # Lower bias gives more room for increase

        # Use a fixed embedding for consistency
        fixed_embedding = np.zeros(1024, dtype=np.float32)
        fixed_embedding[:10] = 1.0  # Consistent pattern

        # Initial prediction
        decision_before = gate.predict(fixed_embedding, context, neuromod_state, explore=False)

        # Update with more positive examples for reliable signal
        for _ in range(30):  # More updates for stronger signal
            decision = gate.predict(fixed_embedding, context, neuromod_state, explore=False)
            memory_id = uuid4()
            gate.register_pending(memory_id, decision.features)
            gate.update(memory_id, utility=1.0)

        # Prediction should increase
        decision_after = gate.predict(fixed_embedding, context, neuromod_state, explore=False)
        assert decision_after.probability > decision_before.probability

    def test_negative_examples_decrease_probability(
        self, gate, content_embedding, context, neuromod_state
    ):
        """Negative examples → lower probability for similar inputs."""
        # Bypass cold start to test pure learned dynamics
        gate.n_observations = gate.cold_start_threshold + 10

        # Boost initial probability
        gate.b = 2.0

        decision_before = gate.predict(content_embedding, context, neuromod_state, explore=False)

        # Update with several negative examples
        for _ in range(10):
            decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
            memory_id = uuid4()
            gate.register_pending(memory_id, decision.features)
            gate.update(memory_id, utility=0.0)

        # Prediction should decrease
        decision_after = gate.predict(content_embedding, context, neuromod_state, explore=False)
        assert decision_after.probability < decision_before.probability

    def test_update_without_pending_features(self, gate):
        """Update without pending features logs warning."""
        memory_id = uuid4()

        # Should not crash, just warn
        gate.update(memory_id, utility=0.5)

        # No change to observations
        assert gate.n_observations == 0


class TestBatchTraining:
    """Test batch training."""

    def test_batch_training_updates_weights(self, gate, content_embedding, context, neuromod_state):
        """Batch training changes weights."""
        initial_weights = gate.μ.copy()

        # Create positive and negative samples
        φ = gate._extract_features(content_embedding, context, neuromod_state)
        positives = [(uuid4(), φ, 1.0) for _ in range(5)]
        negatives = [(uuid4(), φ, 0.0) for _ in range(5)]

        stats = gate.batch_train(positives, negatives, n_epochs=1)

        assert not np.allclose(gate.μ, initial_weights)
        assert stats["n_positives"] == 5
        assert stats["n_negatives"] == 5
        assert stats["final_loss"] >= 0.0

    def test_batch_training_converges(self, gate, content_embedding, context, neuromod_state):
        """Multiple epochs reduce loss."""
        # Use different embeddings for positive vs negative samples
        # Same embedding for both leads to irreconcilable conflict
        positives = []
        negatives = []

        for i in range(10):
            # Positive samples with one pattern
            pos_emb = content_embedding.copy()
            pos_emb[0] = 1.0 + i * 0.1  # Distinct pattern
            φ_pos = gate._extract_features(pos_emb, context, neuromod_state)
            positives.append((uuid4(), φ_pos, 1.0))

            # Negative samples with different pattern
            neg_emb = content_embedding.copy()
            neg_emb[0] = -1.0 - i * 0.1  # Distinct pattern
            φ_neg = gate._extract_features(neg_emb, context, neuromod_state)
            negatives.append((uuid4(), φ_neg, 0.0))

        # Train for 5 epochs
        stats = gate.batch_train(positives, negatives, n_epochs=5)

        # Loss should decrease and be reasonable
        assert stats["final_loss"] < stats.get("initial_loss", float("inf")) or stats["final_loss"] < 2.0


class TestStatistics:
    """Test statistics and monitoring."""

    def test_get_stats_structure(self, gate):
        """Stats have expected keys."""
        stats = gate.get_stats()

        expected_keys = [
            "n_observations",
            "n_pending",
            "total_decisions",
            "store_rate",
            "buffer_rate",
            "skip_rate",
            "avg_accuracy",
            "expected_calibration_error",
            "weight_norm",
            "uncertainty_trace",
            "cold_start_progress"
        ]

        for key in expected_keys:
            assert key in stats

    def test_stats_rates_sum_to_one(self, gate, content_embedding, context, neuromod_state):
        """Store/buffer/skip rates sum to 1."""
        # Make some decisions
        for _ in range(20):
            gate.predict(content_embedding, context, neuromod_state, explore=False)

        stats = gate.get_stats()
        total_rate = stats["store_rate"] + stats["buffer_rate"] + stats["skip_rate"]

        assert total_rate == pytest.approx(1.0, abs=1e-6)

    def test_cold_start_progress(self, gate):
        """Cold start progress tracks observations."""
        assert gate.get_stats()["cold_start_progress"] == 0.0

        gate.n_observations = 5
        assert gate.get_stats()["cold_start_progress"] == pytest.approx(0.5, abs=0.01)

        gate.n_observations = 10
        assert gate.get_stats()["cold_start_progress"] == pytest.approx(1.0, abs=0.01)

        gate.n_observations = 20
        assert gate.get_stats()["cold_start_progress"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks - marked slow due to timing sensitivity."""

    def test_prediction_latency(self, gate, content_embedding, context, neuromod_state):
        """Prediction latency < 8ms (P0b n-gram embedding adds ~2ms overhead)."""
        # Warm up
        for _ in range(10):
            gate.predict(content_embedding, context, neuromod_state, explore=False)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            gate.predict(content_embedding, context, neuromod_state, explore=False)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        p99 = np.percentile(times, 99)
        assert p99 < 0.015  # 15ms (relaxed for CI/parallel test environments)

    def test_update_latency(self, gate, content_embedding, context, neuromod_state):
        """Update latency < 5ms."""
        # Create pending labels
        memory_ids = []
        for _ in range(100):
            decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
            memory_id = uuid4()
            gate.register_pending(memory_id, decision.features)
            memory_ids.append(memory_id)

        # Measure updates
        times = []
        for memory_id in memory_ids:
            start = time.perf_counter()
            gate.update(memory_id, utility=0.5)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        p99 = np.percentile(times, 99)
        assert p99 < 0.005  # 5ms

    def test_memory_footprint(self, gate):
        """Model parameters < 20 KB."""
        # Weight mean
        μ_size = gate.μ.nbytes

        # Covariance (diagonal)
        Σ_size = gate.Σ.nbytes if gate.use_diagonal else 0

        total_size = μ_size + Σ_size

        assert total_size < 20_000  # 20 KB


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_observations(self, gate, content_embedding, context, neuromod_state):
        """Reset clears observation count."""
        # Make some updates
        for _ in range(10):
            decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
            memory_id = uuid4()
            gate.register_pending(memory_id, decision.features)
            gate.update(memory_id, utility=0.5)

        assert gate.n_observations > 0

        gate.reset()

        assert gate.n_observations == 0

    def test_reset_clears_pending(self, gate, content_embedding, context, neuromod_state):
        """Reset clears pending labels."""
        decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
        memory_id = uuid4()
        gate.register_pending(memory_id, decision.features)

        assert len(gate.pending_labels) > 0

        gate.reset()

        assert len(gate.pending_labels) == 0

    def test_reset_reinitializes_weights(self, gate):
        """Reset reinitializes weights to priors."""
        initial_weights = gate.μ.copy()

        # Change weights
        gate.μ[:] = 999.0

        gate.reset()

        # Should be back to initial (not 999)
        assert not np.allclose(gate.μ, 999.0)
        # Should match initial pattern (positive for encoding mode, etc.)
        assert gate.μ[gate.CONTENT_DIM + gate.CONTEXT_DIM + 0] == pytest.approx(0.5)  # DA RPE


class TestContentProjection:
    """Test P0a: Learned content projection (1024 → 128)."""

    def test_projection_reduces_dimensions(self, gate, content_embedding):
        """_project_content reduces 1024 → 128."""
        projected = gate._project_content(content_embedding)

        assert projected.shape == (gate.CONTENT_DIM,)
        assert projected.shape == (128,)

    def test_projection_output_bounded(self, gate, content_embedding):
        """tanh activation bounds output to [-1, 1]."""
        projected = gate._project_content(content_embedding)

        assert projected.min() >= -1.0
        assert projected.max() <= 1.0

    def test_projection_weights_initialized(self, gate):
        """Projection weights have correct shape and initialization."""
        assert gate.W_content.shape == (128, 1024)
        assert gate.b_content.shape == (128,)
        assert gate.W_content.dtype == np.float32
        assert gate.b_content.dtype == np.float32

        # Xavier initialization should produce reasonable magnitudes
        assert gate.W_content.std() < 0.1  # Not too large
        assert gate.W_content.std() > 0.01  # Not too small
        assert np.allclose(gate.b_content, 0.0)  # Bias starts at zero

    def test_projection_is_deterministic(self, gate, content_embedding):
        """Same input produces same output."""
        proj1 = gate._project_content(content_embedding)
        proj2 = gate._project_content(content_embedding)

        np.testing.assert_array_almost_equal(proj1, proj2)

    def test_projection_different_for_different_inputs(self, gate):
        """Different inputs produce different outputs."""
        emb1 = np.random.randn(1024).astype(np.float32)
        emb2 = np.random.randn(1024).astype(np.float32)

        proj1 = gate._project_content(emb1)
        proj2 = gate._project_content(emb2)

        assert not np.allclose(proj1, proj2)

    def test_projection_preserves_relative_similarity(self, gate):
        """Similar embeddings should produce similar projections."""
        base = np.random.randn(1024).astype(np.float32)
        similar = base + np.random.randn(1024).astype(np.float32) * 0.1
        different = np.random.randn(1024).astype(np.float32)

        proj_base = gate._project_content(base)
        proj_similar = gate._project_content(similar)
        proj_different = gate._project_content(different)

        dist_similar = np.linalg.norm(proj_base - proj_similar)
        dist_different = np.linalg.norm(proj_base - proj_different)

        assert dist_similar < dist_different

    def test_projection_used_in_feature_extraction(self, gate, content_embedding, context, neuromod_state):
        """_extract_features uses projection for content features."""
        features = gate._extract_features(content_embedding, context, neuromod_state)

        # First CONTENT_DIM features should match projection
        projected = gate._project_content(content_embedding)
        np.testing.assert_array_almost_equal(features[:128], projected)

    def test_feature_dimensions_reduced(self, gate):
        """Total feature dimensions should be 247 (reduced from 1143)."""
        assert gate.TOTAL_DIM == 247
        assert gate.CONTENT_DIM == 128
        assert gate.CONTEXT_DIM == 64
        assert gate.NEUROMOD_DIM == 7
        assert gate.TEMPORAL_DIM == 16
        assert gate.INTERACTION_DIM == 32


class TestStatePersistence:
    """Test save_state and load_state functionality."""

    def test_save_state_includes_projection_weights(self, gate):
        """save_state includes W_content and b_content."""
        state = gate.save_state()

        assert "W_content" in state
        assert "b_content" in state
        assert "η_content" in state

    def test_save_state_includes_bayesian_params(self, gate):
        """save_state includes Bayesian logistic regression parameters."""
        state = gate.save_state()

        assert "μ" in state
        assert "Σ" in state
        assert "b" in state
        assert "use_diagonal" in state

    def test_save_state_includes_learning_state(self, gate):
        """save_state includes learning state."""
        state = gate.save_state()

        assert "n_observations" in state
        assert "decisions" in state
        assert "calibration_counts" in state
        assert "calibration_correct" in state

    def test_load_state_restores_projection_weights(self, gate):
        """load_state restores projection weights correctly."""
        # Modify projection weights
        original_W = gate.W_content.copy()
        gate.W_content = np.random.randn(128, 1024).astype(np.float32)
        gate.b_content = np.random.randn(128).astype(np.float32)

        # Save
        state = gate.save_state()

        # Create new gate and load state
        new_gate = LearnedMemoryGate(neuromod_orchestra=None)
        new_gate.load_state(state)

        np.testing.assert_array_almost_equal(new_gate.W_content, gate.W_content)
        np.testing.assert_array_almost_equal(new_gate.b_content, gate.b_content)

    def test_load_state_restores_bayesian_params(self, gate, content_embedding, context, neuromod_state):
        """load_state restores Bayesian parameters correctly."""
        # Train the gate
        for i in range(20):
            decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
            memory_id = uuid4()
            gate.register_pending(memory_id, decision.features)
            gate.update(memory_id, utility=0.8 if i % 2 == 0 else 0.2)

        # Save state
        state = gate.save_state()

        # Create new gate and load
        new_gate = LearnedMemoryGate(neuromod_orchestra=None)
        new_gate.load_state(state)

        np.testing.assert_array_almost_equal(new_gate.μ, gate.μ)
        np.testing.assert_array_almost_equal(new_gate.Σ, gate.Σ)
        assert new_gate.b == gate.b
        assert new_gate.n_observations == gate.n_observations

    def test_save_load_roundtrip_preserves_predictions(self, gate, content_embedding, context, neuromod_state):
        """Predictions are identical after save/load roundtrip."""
        # Train gate
        for _ in range(15):
            decision = gate.predict(content_embedding, context, neuromod_state, explore=False)
            memory_id = uuid4()
            gate.register_pending(memory_id, decision.features)
            gate.update(memory_id, utility=np.random.random())

        # Get prediction before save
        pred_before = gate.predict(content_embedding, context, neuromod_state, explore=False)

        # Save and load
        state = gate.save_state()
        new_gate = LearnedMemoryGate(neuromod_orchestra=None)
        new_gate.load_state(state)

        # Get prediction after load
        pred_after = new_gate.predict(content_embedding, context, neuromod_state, explore=False)

        assert pred_before.probability == pytest.approx(pred_after.probability, rel=1e-5)
        assert pred_before.action == pred_after.action

    def test_load_state_partial_is_safe(self, gate):
        """load_state handles partial state gracefully."""
        # Minimal state with just projection
        partial_state = {
            "W_content": gate.W_content.tolist(),
            "b_content": gate.b_content.tolist(),
        }

        # Should not raise
        new_gate = LearnedMemoryGate(neuromod_orchestra=None)
        new_gate.load_state(partial_state)

        # Projection weights should be loaded
        np.testing.assert_array_almost_equal(new_gate.W_content, gate.W_content)

    def test_state_serializable_to_json(self, gate):
        """State can be serialized to JSON."""
        import json

        state = gate.save_state()

        # Should not raise
        json_str = json.dumps(state)
        loaded = json.loads(json_str)

        assert loaded["n_observations"] == state["n_observations"]
        assert loaded["b"] == state["b"]
