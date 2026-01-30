"""
Tests for VAE Training System.

Tests the VAEReplayTrainer and related components for Phase 5.
"""

import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from ww.learning.vae_training import (
    VAETrainingConfig,
    TrainingStats,
    VAEReplayTrainer,
)


# Mock classes
@dataclass
class MockEpisode:
    """Mock episode for testing."""
    id: str = None
    embedding: np.ndarray = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid4())
        if self.embedding is None:
            self.embedding = np.random.randn(1024).astype(np.float32)


class MockEpisodicMemory:
    """Mock episodic memory for testing."""

    def __init__(self, n_episodes: int = 100):
        self.episodes = [MockEpisode() for _ in range(n_episodes)]
        self.get_recent_calls = []

    async def get_recent(self, hours: int = 24, limit: int = 100):
        self.get_recent_calls.append({"hours": hours, "limit": limit})
        return self.episodes[:limit]


class MockVAEGenerator:
    """Mock VAE generator for testing."""

    def __init__(self, embedding_dim: int = 1024, latent_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.train_step_calls = []
        self.generate_calls = []
        self._loss_sequence = iter(np.linspace(1.0, 0.1, 100))

    def train_step(self, batch: np.ndarray) -> float:
        """Mock training step."""
        self.train_step_calls.append(batch)
        try:
            return next(self._loss_sequence)
        except StopIteration:
            return 0.1

    def generate(self, n_samples: int, temperature: float = 1.0) -> list[np.ndarray]:
        """Generate mock samples."""
        self.generate_calls.append({"n_samples": n_samples, "temperature": temperature})
        return [np.random.randn(self.embedding_dim).astype(np.float32) for _ in range(n_samples)]

    def get_statistics(self) -> dict:
        return {
            "n_training_steps": len(self.train_step_calls),
            "embedding_dim": self.embedding_dim,
            "latent_dim": self.latent_dim,
        }


class TestVAETrainingConfig:
    """Tests for VAETrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VAETrainingConfig()
        assert config.buffer_size == 10000
        assert config.batch_size == 32
        assert config.epochs_per_training == 10
        assert config.min_samples_for_training == 32

    def test_custom_config(self):
        """Test custom configuration."""
        config = VAETrainingConfig(
            buffer_size=5000,
            batch_size=64,
            epochs_per_training=20,
        )
        assert config.buffer_size == 5000
        assert config.batch_size == 64
        assert config.epochs_per_training == 20


class TestTrainingStats:
    """Tests for TrainingStats."""

    def test_default_stats(self):
        """Test default stats values."""
        stats = TrainingStats()
        assert stats.epochs_completed == 0
        assert stats.total_batches == 0
        assert stats.final_loss == float('inf')

    def test_stats_to_dict(self):
        """Test stats serialization."""
        stats = TrainingStats(
            epochs_completed=5,
            total_batches=100,
            final_loss=0.5,
            best_loss=0.4,
            mean_loss=0.55,
        )
        d = stats.to_dict()
        assert d["epochs_completed"] == 5
        assert d["total_batches"] == 100
        assert d["final_loss"] == 0.5
        assert "timestamp" in d


class TestVAEReplayTrainer:
    """Tests for VAEReplayTrainer."""

    @pytest.fixture
    def vae(self):
        return MockVAEGenerator()

    @pytest.fixture
    def memory(self):
        return MockEpisodicMemory(n_episodes=100)

    @pytest.fixture
    def trainer(self, vae, memory):
        return VAEReplayTrainer(vae, memory)

    def test_initialization(self, vae, memory):
        """Test trainer initialization."""
        trainer = VAEReplayTrainer(vae, memory)
        assert trainer.vae == vae
        assert trainer.memory == memory
        assert trainer.get_buffer_size() == 0

    def test_initialization_without_memory(self, vae):
        """Test trainer initialization without memory."""
        trainer = VAEReplayTrainer(vae, None)
        assert trainer.memory is None

    @pytest.mark.asyncio
    async def test_collect_wake_samples(self, trainer, memory):
        """Test wake sample collection."""
        collected = await trainer.collect_wake_samples(n_samples=50)
        assert collected == 50
        assert trainer.get_buffer_size() == 50
        assert len(memory.get_recent_calls) == 1

    @pytest.mark.asyncio
    async def test_collect_wake_samples_no_memory(self, vae):
        """Test collection with no memory returns 0."""
        trainer = VAEReplayTrainer(vae, None)
        collected = await trainer.collect_wake_samples()
        assert collected == 0

    @pytest.mark.asyncio
    async def test_collect_wake_samples_default_params(self, trainer):
        """Test collection uses config defaults."""
        await trainer.collect_wake_samples()
        # Should use default limit from config
        assert trainer.get_buffer_size() == trainer.config.default_collection_limit

    def test_add_sample(self, trainer):
        """Test direct sample addition."""
        emb = np.random.randn(1024)
        trainer.add_sample(emb)
        assert trainer.get_buffer_size() == 1

    def test_add_samples(self, trainer):
        """Test adding multiple samples."""
        embeddings = [np.random.randn(1024) for _ in range(10)]
        added = trainer.add_samples(embeddings)
        assert added == 10
        assert trainer.get_buffer_size() == 10

    def test_train_vae_insufficient_samples(self, trainer):
        """Test training with insufficient samples."""
        # Add fewer than minimum
        trainer.add_samples([np.random.randn(1024) for _ in range(5)])
        stats = trainer.train_vae()
        assert stats.epochs_completed == 0

    def test_train_vae_success(self, trainer, vae):
        """Test successful VAE training."""
        # Add enough samples
        trainer.add_samples([np.random.randn(1024) for _ in range(100)])
        stats = trainer.train_vae(epochs=3, batch_size=16)

        assert stats.epochs_completed == 3
        assert stats.total_batches > 0
        assert stats.samples_trained == 100
        assert len(vae.train_step_calls) > 0

    def test_train_vae_early_stopping(self, trainer, vae):
        """Test early stopping during training."""
        # Make loss not improve
        vae._loss_sequence = iter([1.0] * 100)

        trainer.add_samples([np.random.randn(1024) for _ in range(100)])
        # Use very short patience
        trainer.config.early_stopping_patience = 2
        stats = trainer.train_vae(epochs=20)

        assert stats.stopped_early
        assert stats.epochs_completed < 20

    def test_generate_for_replay(self, trainer, vae):
        """Test synthetic sample generation."""
        samples = trainer.generate_for_replay(n_samples=20, temperature=0.8)
        assert len(samples) == 20
        assert len(vae.generate_calls) == 1
        assert vae.generate_calls[0]["temperature"] == 0.8

    def test_interleave_with_real(self, trainer):
        """Test memory interleaving."""
        synthetic = [np.random.randn(1024) for _ in range(30)]
        real = [np.random.randn(1024) for _ in range(20)]

        combined = trainer.interleave_with_real(synthetic, real, interleave_ratio=0.5)

        # Should have mix of both
        assert len(combined) <= len(synthetic) + len(real)

    def test_clear_buffer(self, trainer):
        """Test buffer clearing."""
        trainer.add_samples([np.random.randn(1024) for _ in range(50)])
        assert trainer.get_buffer_size() == 50

        trainer.clear_buffer()
        assert trainer.get_buffer_size() == 0

    def test_get_training_history(self, trainer):
        """Test training history retrieval."""
        trainer.add_samples([np.random.randn(1024) for _ in range(100)])
        trainer.train_vae(epochs=2)
        trainer.train_vae(epochs=2)

        history = trainer.get_training_history()
        assert len(history) == 2

    def test_get_statistics(self, trainer, vae):
        """Test statistics retrieval."""
        trainer.add_samples([np.random.randn(1024) for _ in range(50)])
        stats = trainer.get_statistics()

        assert stats["training_count"] == 0
        assert "history_length" in stats


class TestVAEReplayTrainerBuffer:
    """Tests for buffer management."""

    def test_buffer_max_size(self):
        """Test buffer respects max size."""
        vae = MockVAEGenerator()
        config = VAETrainingConfig(buffer_size=100)
        trainer = VAEReplayTrainer(vae, None, config)

        # Add more than buffer size
        trainer.add_samples([np.random.randn(1024) for _ in range(150)])

        # Should be capped at buffer_size
        assert trainer.get_buffer_size() == 100

    def test_buffer_fifo(self):
        """Test buffer uses FIFO eviction."""
        vae = MockVAEGenerator()
        config = VAETrainingConfig(buffer_size=10)
        trainer = VAEReplayTrainer(vae, None, config)

        # Add samples with identifiable pattern
        for i in range(15):
            emb = np.full(1024, i, dtype=np.float32)
            trainer.add_sample(emb)

        # Oldest should be evicted
        buffer_list = list(trainer._training_buffer)
        assert buffer_list[0][0] == 5  # First remaining should be 5


class TestIntegration:
    """Integration tests for VAE training."""

    @pytest.mark.asyncio
    async def test_full_training_cycle(self):
        """Test complete wake-collect-train-generate cycle."""
        vae = MockVAEGenerator()
        memory = MockEpisodicMemory(n_episodes=200)
        trainer = VAEReplayTrainer(vae, memory)

        # 1. Collect wake samples
        collected = await trainer.collect_wake_samples(n_samples=100)
        assert collected == 100

        # 2. Train VAE
        stats = trainer.train_vae(epochs=3)
        assert stats.epochs_completed == 3
        assert stats.samples_trained == 100

        # 3. Generate synthetic memories
        synthetic = trainer.generate_for_replay(n_samples=50)
        assert len(synthetic) == 50

        # 4. Interleave with real
        real = [np.random.randn(1024) for _ in range(30)]
        combined = trainer.interleave_with_real(synthetic, real)
        assert len(combined) > 0

    @pytest.mark.asyncio
    async def test_incremental_training(self):
        """Test incremental training over multiple sessions."""
        vae = MockVAEGenerator()
        memory = MockEpisodicMemory(n_episodes=500)
        trainer = VAEReplayTrainer(vae, memory)

        # Multiple collection and training cycles
        for _ in range(3):
            await trainer.collect_wake_samples(n_samples=50)
            trainer.train_vae(epochs=2)

        history = trainer.get_training_history()
        assert len(history) == 3

    def test_memory_efficiency(self):
        """Test memory stays bounded."""
        vae = MockVAEGenerator()
        config = VAETrainingConfig(buffer_size=1000)
        trainer = VAEReplayTrainer(vae, None, config)

        # Add many samples
        for _ in range(50):
            trainer.add_samples([np.random.randn(1024) for _ in range(100)])

        # Buffer should be bounded
        assert trainer.get_buffer_size() <= 1000
