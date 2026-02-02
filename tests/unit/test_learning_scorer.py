"""
Unit tests for T4DM neural retrieval scorer.

Tests LearnedRetrievalScorer, PrioritizedReplayBuffer, ListMLELoss, and ScorerTrainer.
"""

import pytest
import torch
from uuid import uuid4

from t4dm.learning.scorer import (
    LearnedRetrievalScorer,
    PrioritizedReplayBuffer,
    ReplayItem,
    ListMLELoss,
    ScorerTrainer,
    TrainerConfig,
    create_scorer,
    create_trainer,
)
from t4dm.learning.events import MemoryType, Experience


class TestLearnedRetrievalScorer:
    """Tests for LearnedRetrievalScorer neural network."""

    def test_creation_default(self):
        scorer = LearnedRetrievalScorer()
        assert scorer.input_dim == 4
        assert scorer.hidden_dim == 32
        assert scorer.residual_proj is not None

    def test_creation_custom_dims(self):
        scorer = LearnedRetrievalScorer(input_dim=6, hidden_dim=64)
        assert scorer.input_dim == 6
        assert scorer.hidden_dim == 64

    def test_creation_same_dims_no_projection(self):
        scorer = LearnedRetrievalScorer(input_dim=32, hidden_dim=32)
        assert scorer.residual_proj is None

    def test_forward_single_memory(self):
        scorer = LearnedRetrievalScorer()
        # [batch=1, n_memories=1, input_dim=4]
        x = torch.randn(1, 1, 4)
        output = scorer(x)

        assert output.shape == (1, 1)

    def test_forward_multiple_memories(self):
        scorer = LearnedRetrievalScorer()
        # [batch=2, n_memories=5, input_dim=4]
        x = torch.randn(2, 5, 4)
        output = scorer(x)

        assert output.shape == (2, 5)

    def test_forward_gradient_flow(self):
        scorer = LearnedRetrievalScorer()
        x = torch.randn(1, 3, 4, requires_grad=True)
        output = scorer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_score_memories_empty(self):
        scorer = LearnedRetrievalScorer()
        scores = scorer.score_memories([])
        assert scores == []

    def test_score_memories_single(self):
        scorer = LearnedRetrievalScorer()
        component_vectors = [[0.9, 0.5, 0.7, 0.3]]
        scores = scorer.score_memories(component_vectors)

        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_score_memories_multiple(self):
        scorer = LearnedRetrievalScorer()
        component_vectors = [
            [0.9, 0.5, 0.7, 0.3],
            [0.7, 0.6, 0.5, 0.4],
            [0.5, 0.8, 0.3, 0.6],
        ]
        scores = scorer.score_memories(component_vectors)

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_parameter_count(self):
        scorer = LearnedRetrievalScorer(input_dim=4, hidden_dim=32)
        params = sum(p.numel() for p in scorer.parameters())

        # Expected: fc1 + fc2 + fc_out + norms + residual_proj
        # fc1: 4*32 + 32 = 160
        # fc2: 32*32 + 32 = 1056
        # fc_out: 32*1 + 1 = 33
        # norm1: 32*2 = 64
        # norm2: 32*2 = 64
        # residual_proj: 4*32 + 32 = 160
        assert params > 1000  # Reasonable size

    def test_dropout_training_vs_eval(self):
        scorer = LearnedRetrievalScorer(dropout=0.5)
        x = torch.randn(1, 5, 4)

        scorer.train()
        outputs_train = []
        for _ in range(10):
            outputs_train.append(scorer(x).detach())

        scorer.eval()
        outputs_eval = []
        for _ in range(10):
            outputs_eval.append(scorer(x).detach())

        # Training outputs should vary (dropout), eval should be consistent
        eval_var = torch.stack(outputs_eval).var()
        assert eval_var < 0.05  # All eval outputs nearly same (numerical precision)


class TestReplayItem:
    """Tests for ReplayItem dataclass."""

    def test_creation(self):
        item = ReplayItem(
            experience_id="exp123",
            query="test query",
            memory_type=MemoryType.EPISODIC,
            component_vectors=[[0.9, 0.5, 0.7, 0.3]],
            rewards=[0.8],
            priority=1.5,
        )

        assert item.experience_id == "exp123"
        assert item.query == "test query"
        assert item.priority == 1.5


class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer."""

    def test_creation(self):
        buffer = PrioritizedReplayBuffer(capacity=100)
        assert len(buffer) == 0
        assert buffer.capacity == 100

    def test_add_single(self):
        buffer = PrioritizedReplayBuffer()
        item = ReplayItem(
            experience_id="1",
            query="test",
            memory_type=MemoryType.EPISODIC,
            component_vectors=[[0.5] * 4],
            rewards=[0.5],
        )
        buffer.add(item)
        assert len(buffer) == 1

    def test_add_multiple(self):
        buffer = PrioritizedReplayBuffer()
        for i in range(10):
            item = ReplayItem(
                experience_id=str(i),
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5] * 4],
                rewards=[0.5],
            )
            buffer.add(item)
        assert len(buffer) == 10

    def test_capacity_limit(self):
        buffer = PrioritizedReplayBuffer(capacity=5)
        for i in range(10):
            item = ReplayItem(
                experience_id=str(i),
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5] * 4],
                rewards=[0.5],
            )
            buffer.add(item)
        assert len(buffer) == 5

    def test_sample_empty(self):
        buffer = PrioritizedReplayBuffer()
        items, indices, weights = buffer.sample(5)
        assert items == []
        assert indices == []
        assert weights == []

    def test_sample_basic(self):
        buffer = PrioritizedReplayBuffer()
        for i in range(20):
            item = ReplayItem(
                experience_id=str(i),
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5] * 4],
                rewards=[0.5],
            )
            buffer.add(item)

        items, indices, weights = buffer.sample(5)
        assert len(items) == 5
        assert len(indices) == 5
        assert len(weights) == 5
        assert all(0 < w <= 1 for w in weights)

    def test_sample_no_replacement(self):
        buffer = PrioritizedReplayBuffer()
        for i in range(10):
            item = ReplayItem(
                experience_id=str(i),
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5] * 4],
                rewards=[0.5],
            )
            buffer.add(item)

        _, indices, _ = buffer.sample(10)
        # All indices should be unique
        assert len(set(indices)) == 10

    def test_update_priorities(self):
        buffer = PrioritizedReplayBuffer()
        for i in range(5):
            item = ReplayItem(
                experience_id=str(i),
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5] * 4],
                rewards=[0.5],
            )
            buffer.add(item)

        old_max = buffer.max_priority
        buffer.update_priorities([0, 1], [10.0, 20.0])

        assert buffer.priorities[0] > old_max
        assert buffer.priorities[1] > buffer.priorities[0]

    def test_add_from_experience(self):
        buffer = PrioritizedReplayBuffer()
        exp = Experience(
            query="test query",
            memory_type=MemoryType.SEMANTIC,
            retrieved_ids=[uuid4()],
            retrieval_scores=[0.9],
            component_vectors=[[0.9, 0.5, 0.7, 0.3]],
            outcome_score=0.8,
            per_memory_rewards={"mem1": 0.75},
            priority=2.0,
        )

        buffer.add_from_experience(exp)
        assert len(buffer) == 1


class TestListMLELoss:
    """Tests for ListMLELoss ranking loss."""

    def test_creation(self):
        loss_fn = ListMLELoss()
        assert loss_fn.temperature == 1.0
        assert loss_fn.eps == 1e-10

    def test_creation_custom_temp(self):
        loss_fn = ListMLELoss(temperature=0.5)
        assert loss_fn.temperature == 0.5

    def test_forward_basic(self):
        loss_fn = ListMLELoss()
        scores = torch.tensor([[3.0, 1.0, 2.0]])
        relevance = torch.tensor([[1.0, 0.0, 0.5]])

        loss = loss_fn(scores, relevance)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_forward_perfect_ranking(self):
        loss_fn = ListMLELoss()
        # Scores perfectly match relevance order
        scores = torch.tensor([[3.0, 2.0, 1.0]])
        relevance = torch.tensor([[3.0, 2.0, 1.0]])

        loss = loss_fn(scores, relevance)
        assert loss >= 0
        assert loss < 1.0  # Should be low

    def test_forward_inverted_ranking(self):
        loss_fn = ListMLELoss()
        # Scores inverted from relevance
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        relevance = torch.tensor([[3.0, 2.0, 1.0]])

        loss = loss_fn(scores, relevance)
        # Should be higher than perfect ranking
        assert loss > 0.5

    def test_forward_with_mask(self):
        loss_fn = ListMLELoss()
        scores = torch.tensor([[3.0, 1.0, 2.0, 0.0]])
        relevance = torch.tensor([[1.0, 0.0, 0.5, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        loss = loss_fn(scores, relevance, mask)
        assert loss >= 0

    def test_forward_batch(self):
        loss_fn = ListMLELoss()
        scores = torch.randn(4, 5)
        relevance = torch.rand(4, 5)

        loss = loss_fn(scores, relevance)
        assert loss.ndim == 0

    def test_gradient_flow(self):
        loss_fn = ListMLELoss()
        scores = torch.randn(2, 3, requires_grad=True)
        relevance = torch.rand(2, 3)

        loss = loss_fn(scores, relevance)
        loss.backward()

        assert scores.grad is not None


class TestTrainerConfig:
    """Tests for TrainerConfig dataclass."""

    def test_defaults(self):
        config = TrainerConfig()
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.batch_size == 32
        assert config.grad_clip == 1.0

    def test_custom_values(self):
        config = TrainerConfig(
            learning_rate=1e-3,
            batch_size=64,
            epochs_per_update=5,
        )
        assert config.learning_rate == 1e-3
        assert config.batch_size == 64


class TestScorerTrainer:
    """Tests for ScorerTrainer."""

    def test_creation(self):
        scorer = LearnedRetrievalScorer()
        config = TrainerConfig()
        trainer = ScorerTrainer(scorer, config)

        assert trainer.step == 0
        assert trainer.total_loss == 0.0

    def test_train_step_empty_buffer(self):
        scorer = LearnedRetrievalScorer()
        config = TrainerConfig(batch_size=4)
        trainer = ScorerTrainer(scorer, config)

        loss = trainer.train_step()
        assert loss is None

    def test_train_step_with_data(self):
        scorer = LearnedRetrievalScorer()
        config = TrainerConfig(batch_size=2)
        trainer = ScorerTrainer(scorer, config)

        # Add enough data
        for i in range(10):
            exp = Experience(
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5, 0.5, 0.5, 0.5] for _ in range(3)],
                outcome_score=0.7,
                per_memory_rewards={f"m{j}": 0.6 for j in range(3)},
            )
            trainer.add_experience(exp)

        loss = trainer.train_step()
        assert loss is not None
        assert loss >= 0
        assert trainer.step == 1

    def test_train_epoch(self):
        scorer = LearnedRetrievalScorer()
        config = TrainerConfig(batch_size=4)
        trainer = ScorerTrainer(scorer, config)

        # Add data
        for i in range(20):
            exp = Experience(
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5] * 4 for _ in range(3)],
                outcome_score=0.7,
                per_memory_rewards={f"m{j}": 0.6 for j in range(3)},
            )
            trainer.add_experience(exp)

        avg_loss = trainer.train_epoch()
        assert avg_loss >= 0

    def test_avg_loss(self):
        scorer = LearnedRetrievalScorer()
        config = TrainerConfig(batch_size=2, device='cpu')
        trainer = ScorerTrainer(scorer, config)

        # Initial avg_loss
        assert trainer.avg_loss == 0.0

        # Add data and train - need multiple memories per experience
        for i in range(10):
            exp = Experience(
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5 + j*0.1] * 4 for j in range(3)],
                outcome_score=0.7,
                per_memory_rewards={f"m{j}": 0.6 for j in range(3)},
            )
            trainer.add_experience(exp)

        trainer.train_step()
        assert trainer.avg_loss >= 0

    def test_checkpoint_save_load(self, tmp_path):
        scorer = LearnedRetrievalScorer()
        config = TrainerConfig(batch_size=2, device='cpu')
        trainer = ScorerTrainer(scorer, config)

        # Train a bit - need multiple memories per experience
        for i in range(10):
            exp = Experience(
                query=f"query {i}",
                memory_type=MemoryType.EPISODIC,
                component_vectors=[[0.5 + j*0.1] * 4 for j in range(3)],
                outcome_score=0.7,
                per_memory_rewards={f"m{j}": 0.6 for j in range(3)},
            )
            trainer.add_experience(exp)
        trainer.train_step()

        # Save
        ckpt_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        # Load into new trainer
        scorer2 = LearnedRetrievalScorer()
        trainer2 = ScorerTrainer(scorer2, config)
        trainer2.load_checkpoint(ckpt_path)

        assert trainer2.step == trainer.step


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_scorer_default(self):
        scorer = create_scorer()
        assert isinstance(scorer, LearnedRetrievalScorer)
        assert scorer.input_dim == 4

    def test_create_scorer_custom(self):
        scorer = create_scorer(input_dim=8, hidden_dim=64)
        assert scorer.input_dim == 8
        assert scorer.hidden_dim == 64

    def test_create_trainer_default(self):
        trainer = create_trainer()
        assert isinstance(trainer, ScorerTrainer)
        assert isinstance(trainer.scorer, LearnedRetrievalScorer)

    def test_create_trainer_custom_scorer(self):
        scorer = create_scorer(input_dim=6)
        trainer = create_trainer(scorer=scorer)
        assert trainer.scorer.input_dim == 6
