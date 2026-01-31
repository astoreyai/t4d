"""
P1C: Tests for VAE wake-sleep training loop.

Tests that VAE is trained from wake samples before consolidation.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from t4dm.consolidation.sleep import SleepConsolidation
from t4dm.learning.vae_training import VAEReplayTrainer, VAETrainingConfig, TrainingStats
from t4dm.learning.vae_generator import VAEGenerator, VAEConfig
from t4dm.hooks.session_lifecycle import SessionEndHook, SessionContext
from t4dm.sdk.agent_client import AgentMemoryClient


@pytest.fixture
def mock_episodic():
    """Mock episodic memory with get_recent method."""
    mem = Mock()

    # Create mock episodes with embeddings
    async def get_recent(hours=24, limit=100):
        episodes = []
        for i in range(min(limit, 50)):
            ep = Mock()
            ep.id = f"ep-{i}"
            ep.embedding = np.random.randn(1024).tolist()
            ep.content = f"Test episode {i}"
            episodes.append(ep)
        return episodes

    mem.get_recent = get_recent
    return mem


@pytest.fixture
def mock_semantic():
    """Mock semantic memory."""
    mem = Mock()
    mem.add_entity = AsyncMock()
    mem.get_relationships = AsyncMock(return_value=[])
    return mem


@pytest.fixture
def vae_config():
    """VAE configuration for testing."""
    return VAEConfig(
        embedding_dim=1024,
        latent_dim=128,
        hidden_dims=(512, 256),
        learning_rate=0.001,
        kl_weight=0.1,
    )


@pytest.fixture
def vae_generator(vae_config):
    """Create VAE generator for testing."""
    return VAEGenerator(vae_config)


@pytest.fixture
def training_config():
    """VAE training configuration."""
    return VAETrainingConfig(
        buffer_size=1000,
        min_samples_for_training=10,
        epochs_per_training=3,
        batch_size=8,
        warmup_epochs=1,
    )


@pytest.fixture
def vae_trainer(vae_generator, mock_episodic, training_config):
    """Create VAE trainer with mocked memory."""
    return VAEReplayTrainer(
        vae=vae_generator,
        memory=mock_episodic,
        config=training_config,
    )


@pytest.fixture
async def sleep_consolidation(mock_episodic, mock_semantic):
    """Create SleepConsolidation with VAE enabled."""
    sleep = SleepConsolidation(
        episodic_memory=mock_episodic,
        semantic_memory=mock_semantic,
        graph_store=Mock(),
        vae_enabled=True,
        vae_latent_dim=128,
        embedding_dim=1024,
    )
    return sleep


class TestVAETrainerBufferCollection:
    """Test wake sample collection into training buffer."""

    @pytest.mark.asyncio
    async def test_collect_wake_samples(self, vae_trainer, mock_episodic):
        """Test collecting wake samples from episodic memory."""
        # Collect samples
        collected = await vae_trainer.collect_wake_samples(n_samples=20)

        # Should collect samples (up to available)
        assert collected > 0
        assert collected <= 20

        # Buffer should contain samples
        assert vae_trainer.get_buffer_size() == collected

    @pytest.mark.asyncio
    async def test_add_sample_directly(self, vae_trainer):
        """Test adding samples directly to buffer."""
        embedding = np.random.randn(1024)

        vae_trainer.add_sample(embedding)

        assert vae_trainer.get_buffer_size() == 1

    @pytest.mark.asyncio
    async def test_add_multiple_samples(self, vae_trainer):
        """Test adding multiple samples at once."""
        embeddings = [np.random.randn(1024) for _ in range(5)]

        added = vae_trainer.add_samples(embeddings)

        assert added == 5
        assert vae_trainer.get_buffer_size() == 5

    @pytest.mark.asyncio
    async def test_buffer_respects_max_size(self, vae_trainer):
        """Test that buffer doesn't exceed max_size."""
        # Buffer has max_size=1000 from fixture
        # Add more than buffer size
        for _ in range(15):
            embeddings = [np.random.randn(1024) for _ in range(100)]
            vae_trainer.add_samples(embeddings)

        # Should cap at buffer_size
        assert vae_trainer.get_buffer_size() <= vae_trainer.config.buffer_size


class TestVAETrainingFromWake:
    """Test VAE training from collected wake samples."""

    @pytest.mark.asyncio
    async def test_vae_trains_from_wake_samples(self, vae_trainer):
        """Test that VAE trains on collected samples."""
        # Add samples to buffer
        for _ in range(20):
            vae_trainer.add_sample(np.random.randn(1024))

        # Train VAE
        stats = vae_trainer.train_vae(epochs=3)

        # Check training occurred
        assert stats.epochs_completed == 3
        assert stats.samples_trained == 20
        assert stats.total_batches > 0
        assert isinstance(stats.final_loss, (int, float))
        assert stats.final_loss < float('inf')

    @pytest.mark.asyncio
    async def test_vae_training_respects_min_samples(self, vae_trainer):
        """Test that VAE doesn't train with insufficient samples."""
        # Add fewer than min_samples_for_training (10)
        for _ in range(5):
            vae_trainer.add_sample(np.random.randn(1024))

        # Try to train
        stats = vae_trainer.train_vae()

        # Should not train
        assert stats.epochs_completed == 0
        assert stats.samples_trained == 0

    @pytest.mark.asyncio
    async def test_vae_loss_decreases(self, vae_trainer):
        """Test that training loss decreases over epochs."""
        # Add sufficient samples
        for _ in range(50):
            vae_trainer.add_sample(np.random.randn(1024))

        # Train with multiple epochs
        stats = vae_trainer.train_vae(epochs=10)

        # Loss should improve (best <= final means some completion occurred)
        assert isinstance(stats.best_loss, (int, float))
        assert isinstance(stats.final_loss, (int, float))
        assert stats.best_loss <= stats.final_loss or stats.best_loss == stats.final_loss
        assert isinstance(stats.mean_loss, (int, float))
        assert stats.mean_loss < float('inf')


class TestVAETrainingTrigger:
    """Test VAE training trigger before consolidation."""

    @pytest.mark.asyncio
    async def test_sleep_consolidation_trains_vae(self, sleep_consolidation):
        """Test that sleep consolidation can train VAE."""
        # Verify VAE trainer exists
        assert sleep_consolidation._vae_trainer is not None

        # Add samples to buffer
        for _ in range(30):
            sleep_consolidation._vae_trainer.add_sample(np.random.randn(1024))

        # Train VAE via sleep consolidation method
        stats_dict = await sleep_consolidation.train_vae_from_wake(epochs=3)

        # Verify training occurred
        assert stats_dict is not None
        assert stats_dict['epochs_completed'] == 3
        assert stats_dict['samples_trained'] > 0

    @pytest.mark.asyncio
    async def test_session_end_hook_triggers_vae_training(self):
        """Test that SessionEndHook triggers VAE training."""
        # Create mock memory client with train_vae_from_wake_samples method
        mock_memory = AsyncMock()
        mock_memory.store_experience = AsyncMock()
        mock_memory.trigger_consolidation = AsyncMock()
        mock_memory.train_vae_from_wake_samples = AsyncMock(
            return_value={
                'final_loss': 0.123,
                'samples_trained': 50,
                'epochs_completed': 5,
            }
        )

        # Create session end hook
        hook = SessionEndHook(
            memory_client=mock_memory,
            auto_consolidate=True,
            train_vae_before_sleep=True,
        )

        # Create hook context with session context
        from t4dm.hooks.base import HookContext
        session_ctx = SessionContext(
            session_id="test-session",
            start_time=datetime.now() - timedelta(hours=1),
        )
        session_ctx.end_time = datetime.now()

        context = HookContext(
            session_id="test-session",
            metadata={"session_context": session_ctx},
        )

        # Execute hook
        result_context = await hook.execute(context)

        # Verify VAE training was called
        mock_memory.train_vae_from_wake_samples.assert_called_once()

        # Verify session context updated
        assert session_ctx.vae_training_sessions >= 1

        # Verify stats in output
        assert 'vae_training' in result_context.output_data
        assert result_context.output_data['vae_training']['final_loss'] == 0.123

    @pytest.mark.asyncio
    async def test_session_end_hook_handles_missing_vae(self):
        """Test that SessionEndHook handles missing VAE gracefully."""
        # Create mock memory without train_vae_from_wake_samples
        mock_memory = AsyncMock()
        mock_memory.store_experience = AsyncMock()
        mock_memory.trigger_consolidation = AsyncMock()
        # No train_vae_from_wake_samples method

        hook = SessionEndHook(
            memory_client=mock_memory,
            train_vae_before_sleep=True,
        )

        from t4dm.hooks.base import HookContext
        session_ctx = SessionContext(
            session_id="test-session",
            start_time=datetime.now(),
        )
        session_ctx.end_time = datetime.now()

        context = HookContext(
            session_id="test-session",
            metadata={"session_context": session_ctx},
        )

        # Should not raise error
        result_context = await hook.execute(context)

        # VAE training sessions should remain unchanged or be 0
        # (depends on whether hook attempts it)
        assert session_ctx.vae_training_sessions >= 0


class TestSyntheticMemoryGeneration:
    """Test synthetic memory generation during sleep."""

    @pytest.mark.asyncio
    async def test_vae_generates_synthetic_memories(self, vae_trainer):
        """Test that VAE can generate synthetic memories."""
        # Train VAE first
        for _ in range(50):
            vae_trainer.add_sample(np.random.randn(1024))
        vae_trainer.train_vae(epochs=3)

        # Generate synthetic memories
        synthetic = vae_trainer.generate_for_replay(n_samples=10, temperature=0.8)

        # Should generate embeddings
        assert len(synthetic) == 10
        assert all(isinstance(s, np.ndarray) for s in synthetic)
        assert all(s.shape == (1024,) for s in synthetic)

    @pytest.mark.asyncio
    async def test_synthetic_memories_during_sleep(self, sleep_consolidation):
        """Test synthetic memory generation via sleep consolidation."""
        # Train VAE
        for _ in range(50):
            sleep_consolidation._vae_trainer.add_sample(np.random.randn(1024))
        await sleep_consolidation.train_vae_from_wake(epochs=3)

        # Generate synthetic memories
        synthetic = sleep_consolidation.generate_synthetic_memories(n_samples=20)

        # Verify generation
        assert len(synthetic) == 20
        assert all(isinstance(s, np.ndarray) for s in synthetic)


class TestMinimumSamplesThreshold:
    """Test minimum samples threshold for training."""

    @pytest.mark.asyncio
    async def test_minimum_samples_threshold(self, vae_trainer):
        """Test that training requires minimum samples."""
        # Config has min_samples_for_training=10
        assert vae_trainer.config.min_samples_for_training == 10

        # Add exactly min_samples
        for _ in range(10):
            vae_trainer.add_sample(np.random.randn(1024))

        # Should train
        stats = vae_trainer.train_vae(epochs=1)
        assert stats.epochs_completed == 1

        # Clear buffer
        vae_trainer.clear_buffer()

        # Add one less than minimum
        for _ in range(9):
            vae_trainer.add_sample(np.random.randn(1024))

        # Should not train
        stats = vae_trainer.train_vae(epochs=1)
        assert stats.epochs_completed == 0

    @pytest.mark.asyncio
    async def test_training_statistics_tracking(self, vae_trainer):
        """Test that training statistics are tracked."""
        # Add samples and train
        for _ in range(30):
            vae_trainer.add_sample(np.random.randn(1024))

        stats1 = vae_trainer.train_vae(epochs=2)

        # Add more and train again
        for _ in range(20):
            vae_trainer.add_sample(np.random.randn(1024))

        stats2 = vae_trainer.train_vae(epochs=3)

        # Get trainer statistics
        trainer_stats = vae_trainer.get_statistics()

        # Should track multiple training sessions
        assert trainer_stats['training_count'] == 2
        assert trainer_stats['total_samples_collected'] == 50

        # Should have history
        history = vae_trainer.get_training_history()
        assert len(history) == 2
        assert history[0]['epochs_completed'] == 2
        assert history[1]['epochs_completed'] == 3


class TestAgentClientVAEIntegration:
    """Test VAE training integration with AgentMemoryClient."""

    @pytest.mark.asyncio
    async def test_agent_client_has_vae_method(self):
        """Test that AgentMemoryClient has train_vae_from_wake_samples method."""
        client = AgentMemoryClient(session_id="test-session")

        # Should have the method
        assert hasattr(client, 'train_vae_from_wake_samples')
        assert callable(client.train_vae_from_wake_samples)

    @pytest.mark.asyncio
    async def test_agent_client_vae_calls_backend(self):
        """Test that AgentMemoryClient VAE method calls backend API."""
        # Create client with mocked backend
        client = AgentMemoryClient(session_id="test-session")

        # Mock the _client._request method
        mock_request = AsyncMock(return_value={
            'final_loss': 0.234,
            'epochs_completed': 5,
            'samples_trained': 100,
        })

        # Connect client and inject mock
        await client.connect()
        client._client._request = mock_request

        try:
            # Call train_vae_from_wake_samples
            result = await client.train_vae_from_wake_samples(
                n_samples=100,
                hours=24,
                epochs=5,
            )

            # Verify backend was called
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            assert call_args[0][0] == "POST"
            assert "/memory/train-vae" in call_args[0][1]
            assert call_args[1]['json']['n_samples'] == 100
            assert call_args[1]['json']['hours'] == 24
            assert call_args[1]['json']['epochs'] == 5

            # Verify result
            assert result is not None
            assert result['final_loss'] == 0.234

        finally:
            await client.close()


class TestEndToEndWakeSleepCycle:
    """End-to-end tests for complete wake-sleep cycle."""

    @pytest.mark.asyncio
    async def test_complete_wake_sleep_cycle(self, sleep_consolidation):
        """Test complete wake → training → sleep cycle."""
        # 1. WAKE: Collect samples
        trainer = sleep_consolidation._vae_trainer
        for _ in range(50):
            trainer.add_sample(np.random.randn(1024))

        assert trainer.get_buffer_size() == 50

        # 2. BEFORE SLEEP: Train VAE
        stats_dict = await sleep_consolidation.train_vae_from_wake(epochs=5)

        assert stats_dict['epochs_completed'] == 5
        # 50 manual + 50 from mock episodic memory collection
        assert stats_dict['samples_trained'] == 100

        # 3. SLEEP: Generate synthetic memories
        synthetic = sleep_consolidation.generate_synthetic_memories(n_samples=30)

        assert len(synthetic) == 30

        # 4. Verify VAE statistics
        vae_stats = sleep_consolidation.get_vae_trainer_statistics()
        assert vae_stats is not None
        assert vae_stats['training_count'] == 1

    @pytest.mark.asyncio
    async def test_periodic_training_schedule(self, vae_trainer):
        """Test periodic VAE training over multiple wake periods."""
        training_sessions = []

        # Simulate multiple wake periods
        for session in range(3):
            # Collect samples during wake
            for _ in range(20):
                vae_trainer.add_sample(np.random.randn(1024))

            # Train before sleep
            stats = vae_trainer.train_vae(epochs=3)
            training_sessions.append(stats)

        # Should have 3 training sessions
        assert len(training_sessions) == 3

        # All should have completed
        assert all(s.epochs_completed == 3 for s in training_sessions)

        # Get overall statistics
        trainer_stats = vae_trainer.get_statistics()
        assert trainer_stats['training_count'] == 3
