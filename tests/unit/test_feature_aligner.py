"""
Tests for FeatureAligner joint gating-retrieval optimization.

Phase 3: Validates feature alignment and joint loss computation.
"""

import pytest
import numpy as np

from t4dm.memory.feature_aligner import (
    FeatureAligner,
    AlignmentResult,
    JointLossWeights,
)


class TestJointLossWeights:
    """Test JointLossWeights dataclass."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = JointLossWeights()

        assert weights.gate == 1.0
        assert weights.retrieval == 0.5
        assert weights.consistency == 0.3
        assert weights.diversity == 0.1

    def test_custom_weights(self):
        """Test custom weight values."""
        weights = JointLossWeights(
            gate=0.8,
            retrieval=0.4,
            consistency=0.5,
            diversity=0.2,
        )

        assert weights.gate == 0.8
        assert weights.retrieval == 0.4


class TestAlignmentResult:
    """Test AlignmentResult dataclass."""

    def test_initialization(self):
        """Test result initializes correctly."""
        projected = np.array([0.5, 0.6, 0.7, 0.8])

        result = AlignmentResult(
            projected_features=projected,
            consistency_loss=0.15,
            diversity_loss=0.2,
            joint_loss=0.35,
            gate_prediction=0.75,
            retrieval_score=0.65,
        )

        assert result.consistency_loss == 0.15
        assert result.diversity_loss == 0.2
        assert result.gate_prediction == 0.75
        assert result.retrieval_score == 0.65


class TestFeatureAlignerBasic:
    """Test FeatureAligner basic functionality."""

    @pytest.fixture
    def aligner(self):
        """Create aligner instance."""
        return FeatureAligner(
            gate_dim=247,
            retrieval_dim=4,
            hidden_dim=32,
            learning_rate=0.01,
        )

    def test_initialization(self, aligner):
        """Test aligner initializes correctly."""
        assert aligner.gate_dim == 247
        assert aligner.retrieval_dim == 4
        assert aligner.hidden_dim == 32
        assert aligner.n_updates == 0

        # Weights should be initialized
        assert aligner.W1.shape == (32, 247)
        assert aligner.W2.shape == (4, 32)
        assert aligner.b1.shape == (32,)
        assert aligner.b2.shape == (4,)

    def test_project_basic(self, aligner):
        """Test basic projection."""
        gate_features = np.random.randn(247).astype(np.float32)
        projected = aligner.project(gate_features)

        assert projected.shape == (4,)
        # Output should be in [0, 1] (sigmoid)
        assert np.all(projected >= 0)
        assert np.all(projected <= 1)

    def test_project_handles_shorter_input(self, aligner):
        """Shorter input should be padded."""
        gate_features = np.random.randn(100).astype(np.float32)
        projected = aligner.project(gate_features)

        assert projected.shape == (4,)

    def test_project_handles_longer_input(self, aligner):
        """Longer input should be truncated."""
        gate_features = np.random.randn(500).astype(np.float32)
        projected = aligner.project(gate_features)

        assert projected.shape == (4,)


class TestConsistencyLoss:
    """Test consistency loss computation."""

    @pytest.fixture
    def aligner(self):
        """Create aligner instance."""
        return FeatureAligner(gate_dim=247, retrieval_dim=4)

    def test_consistency_loss_range(self, aligner):
        """Consistency loss should be non-negative."""
        gate_features = np.random.randn(247).astype(np.float32)
        retrieval_scores = {
            "semantic": 0.8,
            "recency": 0.6,
            "outcome": 0.5,
            "importance": 0.7,
        }

        loss = aligner.compute_consistency_loss(gate_features, retrieval_scores)

        assert loss >= 0

    def test_consistency_loss_with_defaults(self, aligner):
        """Should handle missing retrieval scores."""
        gate_features = np.random.randn(247).astype(np.float32)
        retrieval_scores = {"semantic": 0.9}  # Only one score

        loss = aligner.compute_consistency_loss(gate_features, retrieval_scores)

        assert loss >= 0
        assert np.isfinite(loss)


class TestDiversityLoss:
    """Test diversity loss computation."""

    @pytest.fixture
    def aligner(self):
        """Create aligner instance."""
        return FeatureAligner()

    def test_diversity_loss_cold_start(self, aligner):
        """Cold start should return prior."""
        loss = aligner.compute_diversity_loss(0.5)

        # Not enough history
        assert loss == 0.5

    def test_diversity_loss_balanced(self, aligner):
        """Balanced decisions should have low diversity loss."""
        # Simulate balanced decisions
        for _ in range(50):
            aligner.compute_diversity_loss(0.3)
            aligner.compute_diversity_loss(0.7)

        # Final loss with balanced history
        loss = aligner.compute_diversity_loss(0.5)

        # Balanced = low diversity loss (high entropy)
        assert loss < 0.5

    def test_diversity_loss_collapsed(self, aligner):
        """Collapsed decisions should have high diversity loss."""
        # Simulate always-store decisions
        for _ in range(100):
            aligner.compute_diversity_loss(0.95)

        # Final loss with collapsed history
        loss = aligner.compute_diversity_loss(0.95)

        # Collapsed = high diversity loss (low entropy)
        assert loss > 0.5

    def test_diversity_loss_history_limit(self, aligner):
        """History should be limited."""
        for i in range(200):
            aligner.compute_diversity_loss(0.5)

        assert len(aligner._recent_decisions) <= aligner._max_history


class TestJointLoss:
    """Test joint loss computation."""

    @pytest.fixture
    def aligner(self):
        """Create aligner with specific weights."""
        weights = JointLossWeights(
            gate=1.0,
            retrieval=0.5,
            consistency=0.3,
            diversity=0.1,
        )
        return FeatureAligner(loss_weights=weights)

    def test_joint_loss_computation(self, aligner):
        """Test joint loss is weighted sum."""
        joint = aligner.compute_joint_loss(
            gate_loss=0.4,
            retrieval_loss=0.3,
            consistency_loss=0.2,
            diversity_loss=0.1,
        )

        expected = 1.0 * 0.4 + 0.5 * 0.3 + 0.3 * 0.2 + 0.1 * 0.1
        assert np.isclose(joint, expected)

    def test_joint_loss_zero_components(self, aligner):
        """Zero losses should give zero joint loss."""
        joint = aligner.compute_joint_loss(0.0, 0.0, 0.0, 0.0)
        assert joint == 0.0


class TestAlignment:
    """Test full alignment computation."""

    @pytest.fixture
    def aligner(self):
        """Create aligner instance."""
        return FeatureAligner()

    def test_align_returns_result(self, aligner):
        """Align should return complete result."""
        gate_features = np.random.randn(247).astype(np.float32)
        retrieval_scores = {
            "semantic": 0.8,
            "recency": 0.6,
            "outcome": 0.5,
            "importance": 0.7,
        }

        result = aligner.align(
            gate_features=gate_features,
            gate_prediction=0.75,
            retrieval_scores=retrieval_scores,
        )

        assert isinstance(result, AlignmentResult)
        assert result.projected_features.shape == (4,)
        assert result.gate_prediction == 0.75
        assert result.consistency_loss >= 0
        assert result.diversity_loss >= 0

    def test_align_with_losses(self, aligner):
        """Align should include provided losses in joint loss."""
        gate_features = np.random.randn(247).astype(np.float32)
        retrieval_scores = {
            "semantic": 0.8,
            "recency": 0.6,
            "outcome": 0.5,
            "importance": 0.7,
        }

        result_no_loss = aligner.align(
            gate_features=gate_features,
            gate_prediction=0.75,
            retrieval_scores=retrieval_scores,
            gate_loss=0.0,
            retrieval_loss=0.0,
        )

        result_with_loss = aligner.align(
            gate_features=gate_features,
            gate_prediction=0.75,
            retrieval_scores=retrieval_scores,
            gate_loss=0.3,
            retrieval_loss=0.2,
        )

        # Joint loss should be higher when gate/retrieval losses are added
        assert result_with_loss.joint_loss > result_no_loss.joint_loss


class TestTraining:
    """Test training functionality."""

    @pytest.fixture
    def aligner(self):
        """Create aligner instance."""
        return FeatureAligner(learning_rate=0.01)

    def test_update_changes_weights(self, aligner):
        """Update should modify weights."""
        W1_before = aligner.W1.copy()

        gate_features = np.random.randn(247).astype(np.float32)
        retrieval_scores = {
            "semantic": 0.9,
            "recency": 0.8,
            "outcome": 0.7,
            "importance": 0.6,
        }

        aligner.update(gate_features, retrieval_scores)

        assert not np.allclose(W1_before, aligner.W1)

    def test_update_increments_counter(self, aligner):
        """Update should increment counter."""
        gate_features = np.random.randn(247).astype(np.float32)
        retrieval_scores = {"semantic": 0.8, "recency": 0.6}

        assert aligner.n_updates == 0
        aligner.update(gate_features, retrieval_scores)
        assert aligner.n_updates == 1

    def test_update_returns_loss(self, aligner):
        """Update should return consistency loss."""
        gate_features = np.random.randn(247).astype(np.float32)
        retrieval_scores = {
            "semantic": 0.8,
            "recency": 0.6,
            "outcome": 0.5,
            "importance": 0.7,
        }

        loss = aligner.update(gate_features, retrieval_scores)

        assert loss >= 0
        assert np.isfinite(loss)

    def test_training_reduces_loss(self, aligner):
        """Repeated training should reduce loss on same input."""
        np.random.seed(42)
        gate_features = np.random.randn(247).astype(np.float32)
        retrieval_scores = {
            "semantic": 0.8,
            "recency": 0.6,
            "outcome": 0.5,
            "importance": 0.7,
        }

        initial_loss = aligner.update(gate_features, retrieval_scores)

        for _ in range(50):
            aligner.update(gate_features, retrieval_scores)

        final_loss = aligner.update(gate_features, retrieval_scores)

        # Loss should decrease with training
        assert final_loss < initial_loss


class TestNeuromodLearningRate:
    """Test neuromodulator learning rate adjustment."""

    @pytest.fixture
    def aligner(self):
        """Create aligner instance."""
        return FeatureAligner()

    def test_baseline_lr(self, aligner):
        """Baseline should return close to base LR."""
        lr = aligner.get_neuromod_learning_rate(
            base_lr=0.01,
            dopamine_rpe=0.0,
            norepinephrine_gain=1.0,
            ach_mode="retrieval",
        )

        assert np.isclose(lr, 0.01, atol=0.001)

    def test_surprise_boost(self, aligner):
        """High |RPE| should increase LR."""
        baseline_lr = aligner.get_neuromod_learning_rate(0.01, 0.0, 1.0, "retrieval")
        surprise_lr = aligner.get_neuromod_learning_rate(0.01, 0.8, 1.0, "retrieval")

        assert surprise_lr > baseline_lr

    def test_arousal_boost(self, aligner):
        """High NE should increase LR."""
        baseline_lr = aligner.get_neuromod_learning_rate(0.01, 0.0, 1.0, "retrieval")
        arousal_lr = aligner.get_neuromod_learning_rate(0.01, 0.0, 1.5, "retrieval")

        assert arousal_lr > baseline_lr

    def test_encoding_boost(self, aligner):
        """Encoding mode should increase LR."""
        retrieval_lr = aligner.get_neuromod_learning_rate(0.01, 0.0, 1.0, "retrieval")
        encoding_lr = aligner.get_neuromod_learning_rate(0.01, 0.0, 1.0, "encoding")

        assert encoding_lr > retrieval_lr

    def test_lr_clipping(self, aligner):
        """LR should be clipped to reasonable range."""
        # Very high modulation
        lr = aligner.get_neuromod_learning_rate(0.01, 1.0, 2.0, "encoding")

        # Should be clipped
        assert lr <= 0.01 * 3.0
        assert lr >= 0.01 * 0.3


class TestStatistics:
    """Test statistics tracking."""

    @pytest.fixture
    def aligner(self):
        """Create aligner instance."""
        return FeatureAligner()

    def test_get_statistics(self, aligner):
        """Test statistics retrieval."""
        stats = aligner.get_statistics()

        assert "n_updates" in stats
        assert "running_consistency_loss" in stats
        assert "running_diversity_loss" in stats
        assert "recent_decision_mean" in stats
        assert "loss_weights" in stats

    def test_statistics_update(self, aligner):
        """Statistics should update with training."""
        gate_features = np.random.randn(247).astype(np.float32)
        retrieval_scores = {"semantic": 0.8}

        aligner.update(gate_features, retrieval_scores)

        stats = aligner.get_statistics()
        assert stats["n_updates"] == 1
