"""
Tests for GenerativeReplaySystem (wake-sleep algorithm).

Tests the generator training and confidence computation fixes.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock

from ww.learning.generative_replay import (
    GenerativeReplaySystem,
    GenerativeReplayConfig,
    GeneratedSample,
    ReplayPhase,
    ReplayStats,
    create_generative_replay,
)


class TestGenerativeReplayConfig:
    """Tests for GenerativeReplayConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = GenerativeReplayConfig()
        assert config.wake_learning_rate == 0.01
        assert config.sleep_learning_rate == 0.001
        assert config.generation_temperature == 1.0
        assert config.n_sleep_samples == 100
        assert config.interleave_ratio == 0.5

    def test_custom_values(self):
        """Test custom configuration."""
        config = GenerativeReplayConfig(
            wake_learning_rate=0.05,
            sleep_learning_rate=0.005,
            n_sleep_samples=50,
        )
        assert config.wake_learning_rate == 0.05
        assert config.sleep_learning_rate == 0.005
        assert config.n_sleep_samples == 50


class TestGeneratedSample:
    """Tests for GeneratedSample dataclass."""

    def test_defaults(self):
        """Test default sample values."""
        sample = GeneratedSample()
        assert sample.source_phase == ReplayPhase.SLEEP
        assert sample.temperature == 1.0
        assert sample.confidence == 0.5
        assert sample.embedding.shape == (1024,)

    def test_custom_embedding(self):
        """Test sample with custom embedding."""
        emb = np.random.randn(512)
        sample = GeneratedSample(embedding=emb, confidence=0.9)
        assert sample.embedding.shape == (512,)
        assert sample.confidence == 0.9


class TestReplayStats:
    """Tests for ReplayStats."""

    def test_creation(self):
        """Test stats creation."""
        stats = ReplayStats(
            phase=ReplayPhase.WAKE,
            n_samples_processed=100,
            mean_reconstruction_loss=0.05,
        )
        assert stats.phase == ReplayPhase.WAKE
        assert stats.n_samples_processed == 100
        assert stats.mean_reconstruction_loss == 0.05


class TestGenerativeReplaySystemInit:
    """Tests for GenerativeReplaySystem initialization."""

    def test_default_init(self):
        """Test default initialization."""
        system = GenerativeReplaySystem()
        assert system.current_phase == ReplayPhase.IDLE
        assert system.config.wake_learning_rate == 0.01

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = GenerativeReplayConfig(n_sleep_samples=50)
        system = GenerativeReplaySystem(config=config)
        assert system.config.n_sleep_samples == 50

    def test_with_generator(self):
        """Test initialization with generator."""
        mock_gen = MagicMock()
        system = GenerativeReplaySystem(generator=mock_gen)
        assert system._generator is mock_gen


class TestWakePhase:
    """Tests for wake phase processing."""

    @pytest.mark.asyncio
    async def test_process_wake_stores_history(self):
        """Test that wake phase stores embeddings in history."""
        system = GenerativeReplaySystem()
        embeddings = [np.random.randn(64) for _ in range(10)]

        stats = await system.process_wake(embeddings)

        assert stats.phase == ReplayPhase.WAKE
        assert stats.n_samples_processed == 10
        assert len(system._wake_history) == 10

    @pytest.mark.asyncio
    async def test_process_wake_limits_history(self):
        """Test that wake history is limited."""
        system = GenerativeReplaySystem()

        # Add many embeddings
        for _ in range(150):
            embeddings = [np.random.randn(64) for _ in range(100)]
            await system.process_wake(embeddings)

        # History should be capped at 10000
        assert len(system._wake_history) <= 10000

    @pytest.mark.asyncio
    async def test_process_wake_with_generator_training(self):
        """Test wake phase calls generator.train_step when available."""
        mock_gen = MagicMock()
        mock_gen.train_step = MagicMock(return_value={
            'recon_loss': 0.05,
            'kl_loss': 0.01,
            'total_loss': 0.06,
        })

        system = GenerativeReplaySystem(generator=mock_gen)
        embeddings = [np.random.randn(64) for _ in range(5)]

        stats = await system.process_wake(embeddings)

        mock_gen.train_step.assert_called_once_with(embeddings)
        assert stats.mean_reconstruction_loss == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_process_wake_encode_decode_fallback(self):
        """Test wake phase uses encode/decode when train_step not available."""
        mock_gen = MagicMock(spec=['encode', 'decode'])
        mock_gen.encode = MagicMock(return_value=[np.random.randn(32) for _ in range(5)])
        mock_gen.decode = MagicMock(return_value=[np.random.randn(64) for _ in range(5)])

        system = GenerativeReplaySystem(generator=mock_gen)
        embeddings = [np.random.randn(64) for _ in range(5)]

        stats = await system.process_wake(embeddings)

        mock_gen.encode.assert_called_once()
        mock_gen.decode.assert_called_once()
        assert stats.mean_reconstruction_loss >= 0


class TestSleepPhase:
    """Tests for sleep phase processing."""

    @pytest.mark.asyncio
    async def test_run_sleep_phase_stub_mode(self):
        """Test sleep phase without generator (stub mode)."""
        system = GenerativeReplaySystem()

        # First add some wake history
        embeddings = [np.random.randn(64) for _ in range(50)]
        await system.process_wake(embeddings)

        # Run sleep phase
        stats = await system.run_sleep_phase(n_samples=20)

        assert stats.phase == ReplayPhase.SLEEP
        assert stats.n_samples_processed <= 20
        assert len(system._generated_samples) > 0

    @pytest.mark.asyncio
    async def test_run_sleep_phase_with_generator(self):
        """Test sleep phase with actual generator."""
        mock_gen = MagicMock()
        mock_gen.generate = MagicMock(return_value=[np.random.randn(64) for _ in range(10)])
        mock_gen.encode = MagicMock(return_value=[np.random.randn(32) for _ in range(10)])
        mock_gen.decode = MagicMock(return_value=[np.random.randn(64) for _ in range(10)])

        system = GenerativeReplaySystem(generator=mock_gen)

        stats = await system.run_sleep_phase(n_samples=10, temperature=0.8)

        mock_gen.generate.assert_called_once_with(10, 0.8)
        assert stats.n_samples_processed == 10

    @pytest.mark.asyncio
    async def test_sleep_phase_confidence_computation(self):
        """Test that confidence is computed based on reconstruction error."""
        # Create a mock generator that returns similar embeddings (low error)
        original_embs = [np.ones(64) * 0.5 for _ in range(5)]

        mock_gen = MagicMock()
        mock_gen.generate = MagicMock(return_value=original_embs)
        # Return very similar reconstructions (low error -> high confidence)
        mock_gen.encode = MagicMock(return_value=[np.random.randn(32) for _ in range(5)])
        mock_gen.decode = MagicMock(return_value=[np.ones(64) * 0.51 for _ in range(5)])

        system = GenerativeReplaySystem(generator=mock_gen)

        stats = await system.run_sleep_phase(n_samples=5)

        # With low reconstruction error, confidence should be high
        assert stats.mean_confidence > 0.5

    @pytest.mark.asyncio
    async def test_sleep_phase_callback(self):
        """Test that on_sample_generated callback is called."""
        callback_samples = []
        def on_sample(sample):
            callback_samples.append(sample)

        system = GenerativeReplaySystem(on_sample_generated=on_sample)

        # Add wake history
        await system.process_wake([np.random.randn(64) for _ in range(10)])

        # Run sleep
        await system.run_sleep_phase(n_samples=5)

        assert len(callback_samples) > 0
        assert all(isinstance(s, GeneratedSample) for s in callback_samples)


class TestInterleavedBatches:
    """Tests for interleaved batch generation."""

    @pytest.mark.asyncio
    async def test_get_interleaved_batch(self):
        """Test interleaved batch mixing old and new samples."""
        system = GenerativeReplaySystem()

        # Generate some samples in sleep phase
        await system.process_wake([np.random.randn(64) for _ in range(20)])
        await system.run_sleep_phase(n_samples=50)

        # Get interleaved batch
        new_embeddings = [np.random.randn(64) for _ in range(20)]
        batch = system.get_interleaved_batch(batch_size=10, new_embeddings=new_embeddings)

        # Batch should contain embeddings
        assert len(batch) <= 10
        assert all(isinstance(e, np.ndarray) for e in batch)


class TestStatistics:
    """Tests for statistics tracking."""

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        system = GenerativeReplaySystem()

        # Process wake
        await system.process_wake([np.random.randn(64) for _ in range(10)])

        # Process sleep
        await system.run_sleep_phase(n_samples=5)

        stats = system.get_statistics()

        assert stats["total_wake_samples"] == 10
        assert stats["wake_history_size"] == 10
        assert stats["total_generations"] == 1
        assert stats["current_phase"] == "idle"

    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Test clearing history."""
        system = GenerativeReplaySystem()

        await system.process_wake([np.random.randn(64) for _ in range(10)])
        await system.run_sleep_phase(n_samples=5)

        system.clear_history()

        assert len(system._wake_history) == 0
        assert len(system._generated_samples) == 0


class TestFactoryFunction:
    """Tests for create_generative_replay factory."""

    def test_create_default(self):
        """Test factory with defaults."""
        system = create_generative_replay()
        assert isinstance(system, GenerativeReplaySystem)

    def test_create_with_config(self):
        """Test factory with custom config."""
        config = GenerativeReplayConfig(n_sleep_samples=25)
        system = create_generative_replay(config=config)
        assert system.config.n_sleep_samples == 25
