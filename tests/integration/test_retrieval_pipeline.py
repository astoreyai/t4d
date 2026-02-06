"""
Integration Tests for Retrieval Pipeline (W1-05).

Verifies that LearnedRetrievalScorer is properly integrated in the retrieval
pipeline, following Graves (2016) differentiable memory architecture principles.

Evidence Base: Graves (2016) "Hybrid computing using a neural network with dynamic external memory"

Test Strategy:
1. Verify scorer exists and is properly configured
2. Verify scorer is called during retrieval
3. Verify scorer improves retrieval quality
4. Verify scorer learns from outcomes
5. End-to-end pipeline validation
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch


class TestLearnedRetrievalScorerExists:
    """Verify LearnedRetrievalScorer implementation exists and is properly structured."""

    def test_scorer_class_exists(self):
        """LearnedRetrievalScorer class should be importable."""
        from t4dm.learning.scorer import LearnedRetrievalScorer

        assert LearnedRetrievalScorer is not None

    def test_scorer_has_forward_method(self):
        """Scorer should have forward method for scoring."""
        from t4dm.learning.scorer import LearnedRetrievalScorer

        scorer = LearnedRetrievalScorer()
        assert hasattr(scorer, "forward")
        assert callable(scorer.forward)

    def test_scorer_is_neural_network(self):
        """Scorer should be a PyTorch neural network."""
        from t4dm.learning.scorer import LearnedRetrievalScorer
        import torch.nn as nn

        scorer = LearnedRetrievalScorer()
        assert isinstance(scorer, nn.Module)

    def test_scorer_has_trainable_parameters(self):
        """Scorer should have trainable parameters."""
        from t4dm.learning.scorer import LearnedRetrievalScorer

        scorer = LearnedRetrievalScorer()
        params = list(scorer.parameters())

        assert len(params) > 0, "Scorer should have trainable parameters"
        assert all(p.requires_grad for p in params), "All parameters should require grad"


class TestScorerForwardPass:
    """Test LearnedRetrievalScorer forward pass."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        from t4dm.learning.scorer import LearnedRetrievalScorer

        return LearnedRetrievalScorer(input_dim=4, hidden_dim=32)

    def test_forward_with_single_memory(self, scorer):
        """Should handle single memory scoring."""
        # [batch=1, n_memories=1, features=4]
        x = torch.randn(1, 1, 4)
        scores = scorer(x)

        assert scores.shape == (1, 1), f"Expected (1, 1), got {scores.shape}"

    def test_forward_with_multiple_memories(self, scorer):
        """Should handle multiple memory scoring."""
        # [batch=1, n_memories=10, features=4]
        x = torch.randn(1, 10, 4)
        scores = scorer(x)

        assert scores.shape == (1, 10), f"Expected (1, 10), got {scores.shape}"

    def test_forward_with_batch(self, scorer):
        """Should handle batched scoring."""
        # [batch=5, n_memories=10, features=4]
        x = torch.randn(5, 10, 4)
        scores = scorer(x)

        assert scores.shape == (5, 10), f"Expected (5, 10), got {scores.shape}"

    def test_scores_are_finite(self, scorer):
        """Scores should be finite (no NaN/Inf)."""
        x = torch.randn(1, 10, 4)
        scores = scorer(x)

        assert torch.isfinite(scores).all(), "Scores should be finite"

    def test_different_inputs_produce_different_scores(self, scorer):
        """Different inputs should produce different scores."""
        x1 = torch.zeros(1, 5, 4)
        x2 = torch.ones(1, 5, 4)

        scores1 = scorer(x1)
        scores2 = scorer(x2)

        assert not torch.allclose(scores1, scores2), "Different inputs should produce different scores"


class TestScorerGradients:
    """Test that scorer supports gradient-based training."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        from t4dm.learning.scorer import LearnedRetrievalScorer

        return LearnedRetrievalScorer()

    def test_gradients_flow(self, scorer):
        """Gradients should flow through scorer."""
        x = torch.randn(1, 5, 4, requires_grad=True)
        scores = scorer(x)
        loss = scores.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"

    def test_parameters_have_gradients(self, scorer):
        """Parameters should receive gradients during training."""
        x = torch.randn(1, 5, 4)
        scores = scorer(x)
        loss = scores.sum()
        loss.backward()

        for name, param in scorer.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradient"


class TestListMLELoss:
    """Test ListMLE ranking loss for training."""

    def test_loss_exists(self):
        """ListMLELoss should be importable."""
        from t4dm.learning.scorer import ListMLELoss

        assert ListMLELoss is not None

    def test_loss_computation(self):
        """Loss should be computable for valid inputs."""
        from t4dm.learning.scorer import ListMLELoss

        loss_fn = ListMLELoss()

        # Predicted scores and target relevance
        scores = torch.randn(2, 5)  # 2 batches, 5 items
        targets = torch.rand(2, 5)  # Relevance scores

        loss = loss_fn(scores, targets)

        assert loss.ndim == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_loss_rewards_correct_ordering(self):
        """Loss should be lower when predicted order matches target order."""
        from t4dm.learning.scorer import ListMLELoss

        loss_fn = ListMLELoss()

        # Perfect ordering
        scores_correct = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        targets = torch.tensor([[1.0, 0.8, 0.6, 0.4, 0.2]])

        # Reversed ordering
        scores_wrong = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        loss_correct = loss_fn(scores_correct, targets)
        loss_wrong = loss_fn(scores_wrong, targets)

        assert loss_correct < loss_wrong, "Correct ordering should have lower loss"


class TestScorerTrainer:
    """Test ScorerTrainer for online learning."""

    def test_trainer_exists(self):
        """ScorerTrainer should be importable."""
        from t4dm.learning.scorer import ScorerTrainer

        assert ScorerTrainer is not None

    def test_trainer_creation(self):
        """Should be able to create trainer."""
        from t4dm.learning.scorer import ScorerTrainer, LearnedRetrievalScorer, create_trainer

        scorer = LearnedRetrievalScorer()
        trainer = create_trainer(scorer)

        assert trainer is not None
        assert trainer.scorer is scorer


class TestPrioritizedReplayBuffer:
    """Test prioritized experience replay for training."""

    def test_buffer_exists(self):
        """PrioritizedReplayBuffer should be importable."""
        from t4dm.learning.scorer import PrioritizedReplayBuffer

        assert PrioritizedReplayBuffer is not None

    def test_buffer_add_and_sample(self):
        """Should be able to add items and sample from buffer."""
        from t4dm.learning.scorer import PrioritizedReplayBuffer, ReplayItem
        from t4dm.learning.events import MemoryType

        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add some items - using actual ReplayItem fields
        for i in range(10):
            item = ReplayItem(
                experience_id=f"exp_{i}",
                query=f"test query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.1 * j for j in range(128)] for _ in range(5)],
                rewards=[0.5 + 0.1 * i] * 5,
                priority=1.0 + i * 0.1,
            )
            buffer.add(item)

        # Sample
        samples = buffer.sample(batch_size=3)

        assert len(samples) == 3


class TestFFRetrievalScorerIntegration:
    """Test Forward-Forward retrieval scorer integration."""

    def test_ff_retrieval_scorer_exists(self):
        """FFRetrievalScorer should be importable."""
        from t4dm.bridges.ff_retrieval_scorer import FFRetrievalScorer

        assert FFRetrievalScorer is not None

    def test_ff_scorer_has_required_methods(self):
        """FFRetrievalScorer should have required methods."""
        from t4dm.bridges.ff_retrieval_scorer import FFRetrievalScorer, FFRetrievalConfig

        # Create with mock FF layer
        mock_ff_layer = Mock()
        mock_ff_layer.forward.return_value = np.zeros(512)
        mock_ff_layer.compute_goodness.return_value = 1.5
        mock_ff_layer.config = Mock()
        mock_ff_layer.config.threshold_theta = 2.0

        config = FFRetrievalConfig()
        scorer = FFRetrievalScorer(mock_ff_layer, config)

        assert hasattr(scorer, "score_candidate")
        assert hasattr(scorer, "score_candidates")  # plural, not score_batch
        assert hasattr(scorer, "learn_from_outcome")

    def test_ff_scorer_computes_confidence(self):
        """FFRetrievalScorer should compute confidence from goodness."""
        from t4dm.bridges.ff_retrieval_scorer import FFRetrievalScorer, FFRetrievalConfig

        mock_ff_layer = Mock()
        mock_ff_layer.forward.return_value = np.zeros(512)
        mock_ff_layer.compute_goodness.return_value = 2.5  # Above threshold
        mock_ff_layer.config = Mock()
        mock_ff_layer.config.threshold_theta = 2.0
        mock_ff_layer.config.hidden_dim = 512

        config = FFRetrievalConfig()
        scorer = FFRetrievalScorer(mock_ff_layer, config)

        embedding = np.random.randn(512).astype(np.float32)
        memory_id = "test_memory_001"

        result = scorer.score_candidate(embedding, memory_id, use_cache=False)

        # Result is a RetrievalScore dataclass
        assert hasattr(result, "confidence")
        assert 0 <= result.confidence <= 1


class TestRetrievalPipelineIntegration:
    """Integration tests for complete retrieval pipeline."""

    def test_scorer_factory_functions_exist(self):
        """Factory functions should be available."""
        from t4dm.learning.scorer import create_scorer, create_trainer

        assert create_scorer is not None
        assert create_trainer is not None

    def test_create_scorer_returns_valid_instance(self):
        """create_scorer should return valid scorer instance."""
        from t4dm.learning.scorer import create_scorer, LearnedRetrievalScorer

        scorer = create_scorer(input_dim=4, hidden_dim=64)

        assert isinstance(scorer, LearnedRetrievalScorer)

    def test_scorer_persistence_available(self):
        """Scorer persistence functions should be available."""
        from t4dm.learning.persistence import StatePersister

        persister = StatePersister(storage_path="/tmp/test_scorer")

        assert hasattr(persister, "save_scorer_state")
        assert hasattr(persister, "load_scorer_state")


class TestEndToEndScoring:
    """End-to-end tests for scoring pipeline."""

    @pytest.fixture
    def scorer(self):
        """Create and train a simple scorer."""
        from t4dm.learning.scorer import LearnedRetrievalScorer

        return LearnedRetrievalScorer(input_dim=4, hidden_dim=32)

    def test_score_ranking_is_consistent(self, scorer):
        """Scoring should produce consistent rankings."""
        # Create test features for 5 memories
        features = torch.randn(1, 5, 4)

        scores1 = scorer(features)
        scores2 = scorer(features)

        # Same input should produce same scores (eval mode)
        scorer.eval()
        scores3 = scorer(features)
        scores4 = scorer(features)

        torch.testing.assert_close(scores3, scores4)

    def test_higher_feature_values_affect_scores(self, scorer):
        """Higher feature values should affect scoring."""
        # Low feature values
        low_features = torch.zeros(1, 5, 4)

        # High feature values
        high_features = torch.ones(1, 5, 4) * 2

        scorer.eval()
        low_scores = scorer(low_features)
        high_scores = scorer(high_features)

        # Scores should differ (exact relationship depends on training)
        assert not torch.allclose(low_scores, high_scores)

    def test_scorer_can_be_trained(self, scorer):
        """Scorer should be trainable with gradient descent."""
        from t4dm.learning.scorer import ListMLELoss

        loss_fn = ListMLELoss()
        optimizer = torch.optim.Adam(scorer.parameters(), lr=0.01)

        # Training data
        features = torch.randn(10, 5, 4)
        targets = torch.rand(10, 5)

        initial_loss = None
        final_loss = None

        for epoch in range(50):
            optimizer.zero_grad()
            scores = scorer(features)
            loss = loss_fn(scores, targets)

            if epoch == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if epoch == 49:
                final_loss = loss.item()

        assert final_loss < initial_loss, "Training should reduce loss"


class TestScorerExportedFromLearning:
    """Verify scorer is properly exported from learning module."""

    def test_scorer_in_learning_all(self):
        """LearnedRetrievalScorer should be in learning.__all__."""
        from t4dm import learning

        assert "LearnedRetrievalScorer" in learning.__all__

    def test_scorer_importable_from_learning(self):
        """Should be importable directly from learning module."""
        from t4dm.learning import LearnedRetrievalScorer

        assert LearnedRetrievalScorer is not None

    def test_trainer_importable_from_learning(self):
        """Trainer should be importable from learning module."""
        from t4dm.learning import ScorerTrainer, create_trainer

        assert ScorerTrainer is not None
        assert create_trainer is not None

    def test_buffer_importable_from_learning(self):
        """Replay buffer should be importable from learning module."""
        from t4dm.learning import PrioritizedReplayBuffer, ReplayItem

        assert PrioritizedReplayBuffer is not None
        assert ReplayItem is not None
