"""
Tests for adapter training system (Phase 2C).

Covers:
- TrainingConfig dataclass
- TrainingStats dataclass
- EmbeddingCache LRU cache
- AdapterTrainer orchestration
- ContinuousTrainer background loop
- Factory functions
"""

from __future__ import annotations

import asyncio
import numpy as np
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import uuid4

from ww.encoding.adapter_training import (
    TrainingConfig,
    TrainingStats,
    EmbeddingCache,
    AdapterTrainer,
    ContinuousTrainer,
    create_adapter_trainer,
    create_continuous_trainer,
)
from ww.learning.retrieval_feedback import RetrievalFeedback


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return np.random.randn(1024).astype(np.float32)


@pytest.fixture
def sample_embeddings():
    """Create multiple sample embeddings."""
    return {
        "query_1": np.random.randn(1024).astype(np.float32),
        "result_1": np.random.randn(1024).astype(np.float32),
        "result_2": np.random.randn(1024).astype(np.float32),
        "result_3": np.random.randn(1024).astype(np.float32),
    }


@pytest.fixture
def mock_adapter():
    """Create a mock OnlineEmbeddingAdapter."""
    adapter = MagicMock()
    adapter.train_step = MagicMock(return_value=0.5)
    adapter.get_stats = MagicMock(
        return_value={
            "step_count": 10,
            "mean_positive_sim": 0.8,
            "mean_negative_sim": 0.3,
        }
    )
    return adapter


@pytest.fixture
def mock_feedback_collector():
    """Create a mock RetrievalFeedbackCollector."""
    collector = MagicMock()
    collector.get_training_batch = MagicMock(return_value=[])
    collector.get_statistics = MagicMock(
        return_value={
            "persisted_feedbacks": 50,
            "total_retrievals": 100,
        }
    )
    return collector


@pytest.fixture
def mock_embedder():
    """Create a mock EmbeddingAdapter."""
    return MagicMock()


@pytest.fixture
def training_config():
    """Create a test training config."""
    return TrainingConfig(
        batch_size=16,
        min_confidence=0.5,
        relevance_threshold=0.6,
        max_batches_per_epoch=5,
        feedback_lookback_hours=24,
        embedding_cache_size=100,
        warmup_batches=2,
        early_stop_patience=3,
        early_stop_min_delta=0.001,
    )


@pytest.fixture
def adapter_trainer(mock_adapter, mock_feedback_collector, mock_embedder, training_config):
    """Create an AdapterTrainer instance."""
    return AdapterTrainer(
        adapter=mock_adapter,
        feedback=mock_feedback_collector,
        embedder=mock_embedder,
        config=training_config,
    )


# ============================================================================
# TrainingConfig Tests
# ============================================================================


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_initialization(self):
        """Test TrainingConfig with default values."""
        config = TrainingConfig()
        assert config.batch_size == 32
        assert config.min_confidence == 0.5
        assert config.relevance_threshold == 0.5
        assert config.max_batches_per_epoch == 100
        assert config.feedback_lookback_hours == 24
        assert config.embedding_cache_size == 10000
        assert config.warmup_batches == 5
        assert config.early_stop_patience == 10
        assert config.early_stop_min_delta == 0.001

    def test_custom_initialization(self, training_config):
        """Test TrainingConfig with custom values."""
        assert training_config.batch_size == 16
        assert training_config.min_confidence == 0.5
        assert training_config.relevance_threshold == 0.6
        assert training_config.max_batches_per_epoch == 5

    def test_config_fields(self):
        """Test all TrainingConfig fields are accessible."""
        config = TrainingConfig(
            batch_size=64,
            min_confidence=0.7,
            relevance_threshold=0.8,
        )
        assert config.batch_size == 64
        assert config.min_confidence == 0.7
        assert config.relevance_threshold == 0.8


# ============================================================================
# TrainingStats Tests
# ============================================================================


class TestTrainingStats:
    """Test TrainingStats dataclass."""

    def test_default_initialization(self):
        """Test TrainingStats with default values."""
        stats = TrainingStats()
        assert stats.epoch == 0
        assert stats.batches_trained == 0
        assert stats.total_samples == 0
        assert stats.mean_loss == 0.0
        assert stats.final_loss == 0.0
        assert stats.best_loss == float("inf")
        assert stats.positive_samples == 0
        assert stats.negative_samples == 0
        assert stats.training_time_seconds == 0.0
        assert stats.stopped_early is False
        assert isinstance(stats.timestamp, datetime)

    def test_custom_initialization(self):
        """Test TrainingStats with custom values."""
        stats = TrainingStats(
            epoch=1,
            batches_trained=5,
            total_samples=100,
            mean_loss=0.2,
            final_loss=0.15,
            best_loss=0.1,
            positive_samples=50,
            negative_samples=50,
            training_time_seconds=10.5,
            stopped_early=True,
        )
        assert stats.epoch == 1
        assert stats.batches_trained == 5
        assert stats.total_samples == 100
        assert stats.mean_loss == 0.2
        assert stats.final_loss == 0.15
        assert stats.best_loss == 0.1
        assert stats.stopped_early is True

    def test_to_dict(self):
        """Test TrainingStats.to_dict() serialization."""
        stats = TrainingStats(
            epoch=1,
            batches_trained=3,
            total_samples=50,
            mean_loss=0.25,
            final_loss=0.2,
            best_loss=0.15,
            positive_samples=30,
            negative_samples=20,
            training_time_seconds=5.0,
            stopped_early=False,
        )
        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert stats_dict["epoch"] == 1
        assert stats_dict["batches_trained"] == 3
        assert stats_dict["total_samples"] == 50
        assert stats_dict["mean_loss"] == 0.25
        assert stats_dict["final_loss"] == 0.2
        assert stats_dict["best_loss"] == 0.15
        assert stats_dict["positive_samples"] == 30
        assert stats_dict["negative_samples"] == 20
        assert stats_dict["training_time_seconds"] == 5.0
        assert stats_dict["stopped_early"] is False
        assert isinstance(stats_dict["timestamp"], str)

    def test_to_dict_iso_format_timestamp(self):
        """Test that timestamp is ISO formatted in to_dict."""
        now = datetime.now()
        stats = TrainingStats(timestamp=now)
        stats_dict = stats.to_dict()
        assert stats_dict["timestamp"] == now.isoformat()


# ============================================================================
# EmbeddingCache Tests
# ============================================================================


class TestEmbeddingCache:
    """Test EmbeddingCache LRU cache."""

    def test_initialization(self):
        """Test EmbeddingCache initialization."""
        cache = EmbeddingCache(max_size=100)
        assert cache.max_size == 100
        assert len(cache) == 0

    def test_custom_max_size(self):
        """Test cache with custom max size."""
        cache = EmbeddingCache(max_size=50)
        assert cache.max_size == 50

    def test_put_and_get(self, sample_embedding):
        """Test basic put and get operations."""
        cache = EmbeddingCache()
        cache.put("embedding_1", sample_embedding)

        cached = cache.get("embedding_1")
        assert cached is not None
        np.testing.assert_array_equal(cached, sample_embedding)

    def test_get_nonexistent(self):
        """Test getting non-existent entry returns None."""
        cache = EmbeddingCache()
        assert cache.get("nonexistent") is None

    def test_cache_hit_updates_access_time(self, sample_embedding):
        """Test that accessing item updates its timestamp."""
        cache = EmbeddingCache()
        cache.put("emb_1", sample_embedding)

        # Get the cached item and check timestamp
        cached1 = cache.get("emb_1")
        time1 = cache._cache["emb_1"][1]

        # Wait a tiny bit and get again
        import time
        time.sleep(0.01)
        cached2 = cache.get("emb_1")
        time2 = cache._cache["emb_1"][1]

        # Second timestamp should be later
        assert time2 > time1

    def test_lru_eviction(self, sample_embedding):
        """Test that least recently used entry is evicted."""
        cache = EmbeddingCache(max_size=3)

        # Fill cache with 3 items
        emb1 = sample_embedding
        emb2 = sample_embedding * 2
        emb3 = sample_embedding * 3

        cache.put("emb_1", emb1)
        cache.put("emb_2", emb2)
        cache.put("emb_3", emb3)

        assert len(cache) == 3

        # Access emb_1 to make it recently used
        cache.get("emb_1")

        # Add new item, emb_2 should be evicted (least recently used)
        emb4 = sample_embedding * 4
        cache.put("emb_4", emb4)

        assert len(cache) == 3
        assert cache.get("emb_1") is not None
        assert cache.get("emb_2") is None  # Evicted
        assert cache.get("emb_3") is not None
        assert cache.get("emb_4") is not None

    def test_clear(self, sample_embedding):
        """Test clearing the cache."""
        cache = EmbeddingCache()
        cache.put("emb_1", sample_embedding)
        cache.put("emb_2", sample_embedding * 2)

        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        assert cache.get("emb_1") is None

    def test_len(self, sample_embedding):
        """Test cache length."""
        cache = EmbeddingCache()
        assert len(cache) == 0

        cache.put("emb_1", sample_embedding)
        assert len(cache) == 1

        cache.put("emb_2", sample_embedding)
        assert len(cache) == 2

    def test_embedding_copy_on_put(self, sample_embedding):
        """Test that embeddings are copied when cached."""
        cache = EmbeddingCache()
        original = sample_embedding.copy()
        cache.put("emb_1", sample_embedding)

        # Modify original
        sample_embedding[0] = 999.0

        # Cached version should be unchanged
        cached = cache.get("emb_1")
        assert cached[0] != 999.0
        np.testing.assert_array_equal(cached, original)


# ============================================================================
# AdapterTrainer Tests
# ============================================================================


class TestAdapterTrainer:
    """Test AdapterTrainer main class."""

    def test_initialization(self, adapter_trainer, mock_adapter, mock_feedback_collector, mock_embedder, training_config):
        """Test AdapterTrainer initialization."""
        assert adapter_trainer.adapter is mock_adapter
        assert adapter_trainer.feedback is mock_feedback_collector
        assert adapter_trainer.embedder is mock_embedder
        assert adapter_trainer.config is training_config
        assert isinstance(adapter_trainer._cache, EmbeddingCache)
        assert adapter_trainer._epoch_count == 0
        assert adapter_trainer._total_batches == 0
        assert len(adapter_trainer._training_history) == 0

    def test_initialization_with_default_config(self, mock_adapter, mock_feedback_collector, mock_embedder):
        """Test AdapterTrainer uses default config if none provided."""
        trainer = AdapterTrainer(
            adapter=mock_adapter,
            feedback=mock_feedback_collector,
            embedder=mock_embedder,
        )
        assert isinstance(trainer.config, TrainingConfig)
        assert trainer.config.batch_size == 32

    def test_group_by_query(self, adapter_trainer):
        """Test _group_by_query groups feedback by query ID."""
        feedbacks = [
            RetrievalFeedback(query_id="q1", result_id="r1", relevance=0.8),
            RetrievalFeedback(query_id="q1", result_id="r2", relevance=0.7),
            RetrievalFeedback(query_id="q2", result_id="r3", relevance=0.9),
            RetrievalFeedback(query_id="q2", result_id="r4", relevance=0.6),
        ]

        grouped = adapter_trainer._group_by_query(feedbacks)

        assert len(grouped) == 2
        assert "q1" in grouped
        assert "q2" in grouped
        assert len(grouped["q1"]) == 2
        assert len(grouped["q2"]) == 2

    def test_split_by_relevance(self, adapter_trainer):
        """Test _split_by_relevance splits feedback into positive/negative."""
        feedbacks = [
            RetrievalFeedback(query_id="q1", result_id="r1", relevance=0.8),  # positive
            RetrievalFeedback(query_id="q1", result_id="r2", relevance=0.7),  # positive
            RetrievalFeedback(query_id="q1", result_id="r3", relevance=0.5),  # on threshold
            RetrievalFeedback(query_id="q1", result_id="r4", relevance=0.4),  # negative
        ]

        positives, negatives = adapter_trainer._split_by_relevance(feedbacks)

        # Config threshold is 0.6
        assert len(positives) == 2  # 0.8, 0.7
        assert len(negatives) == 2  # 0.5, 0.4

    def test_cache_embeddings(self, adapter_trainer, sample_embeddings):
        """Test cache_embeddings pre-caches embeddings."""
        adapter_trainer.cache_embeddings(
            query_id="query_1",
            query_embedding=sample_embeddings["query_1"],
            result_embeddings={
                "result_1": sample_embeddings["result_1"],
                "result_2": sample_embeddings["result_2"],
            },
        )

        # Verify caching
        assert adapter_trainer._cache.get("query:query_1") is not None
        assert adapter_trainer._cache.get("result:result_1") is not None
        assert adapter_trainer._cache.get("result:result_2") is not None

    def test_reset_cache(self, adapter_trainer, sample_embedding):
        """Test reset_cache clears the embedding cache."""
        adapter_trainer._cache.put("emb_1", sample_embedding)
        assert len(adapter_trainer._cache) > 0

        adapter_trainer.reset_cache()
        assert len(adapter_trainer._cache) == 0

    def test_get_training_history(self, adapter_trainer):
        """Test get_training_history returns training history."""
        # Add some dummy stats
        stats1 = TrainingStats(epoch=1, batches_trained=5)
        stats2 = TrainingStats(epoch=2, batches_trained=3)
        adapter_trainer._training_history = [stats1, stats2]

        history = adapter_trainer.get_training_history()

        assert len(history) == 2
        assert all(isinstance(h, dict) for h in history)
        assert history[0]["epoch"] == 1
        assert history[1]["epoch"] == 2

    def test_get_stats(self, adapter_trainer, mock_adapter):
        """Test get_stats returns trainer statistics."""
        adapter_trainer._epoch_count = 5
        adapter_trainer._total_batches = 20

        stats = adapter_trainer.get_stats()

        assert stats["epoch_count"] == 5
        assert stats["total_batches"] == 20
        assert "cache_size" in stats
        assert "adapter_stats" in stats
        assert "config" in stats
        assert stats["config"]["batch_size"] == 16

    @pytest.mark.asyncio
    async def test_train_epoch_no_feedback(self, adapter_trainer, mock_feedback_collector):
        """Test train_epoch with no feedback available."""
        mock_feedback_collector.get_training_batch.return_value = []

        stats = await adapter_trainer.train_epoch()

        assert stats.epoch == 1
        assert stats.batches_trained == 0
        assert stats.total_samples == 0
        assert stats.mean_loss == 0.0
        assert stats.training_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_train_epoch_increments_epoch_count(self, adapter_trainer, mock_feedback_collector):
        """Test that train_epoch increments epoch counter."""
        mock_feedback_collector.get_training_batch.return_value = []

        assert adapter_trainer._epoch_count == 0

        await adapter_trainer.train_epoch()
        assert adapter_trainer._epoch_count == 1

        await adapter_trainer.train_epoch()
        assert adapter_trainer._epoch_count == 2

    @pytest.mark.asyncio
    async def test_train_epoch_with_feedback(self, adapter_trainer, mock_feedback_collector, mock_adapter, sample_embeddings):
        """Test train_epoch processes feedback correctly."""
        feedbacks = [
            RetrievalFeedback(query_id="q1", result_id="r1", relevance=0.8, clicked=True),
            RetrievalFeedback(query_id="q1", result_id="r2", relevance=0.4, clicked=False),
        ]
        # Return feedbacks once, then empty for subsequent calls
        mock_feedback_collector.get_training_batch.side_effect = [feedbacks, []]
        mock_adapter.train_step.return_value = 0.3

        # Cache embeddings to enable training
        adapter_trainer.cache_embeddings(
            query_id="q1",
            query_embedding=sample_embeddings["query_1"],
            result_embeddings={
                "r1": sample_embeddings["result_1"],
                "r2": sample_embeddings["result_2"],
            },
        )

        stats = await adapter_trainer.train_epoch()

        assert stats.epoch == 1
        assert stats.batches_trained == 1
        assert stats.total_samples == 2
        assert stats.positive_samples == 1
        assert stats.negative_samples == 1

    @pytest.mark.asyncio
    async def test_train_epoch_early_stopping(self, adapter_trainer, mock_feedback_collector, mock_adapter, sample_embeddings):
        """Test early stopping when patience exceeded."""
        # Setup feedback with degrading loss
        feedbacks1 = [
            RetrievalFeedback(query_id="q1", result_id="r1", relevance=0.8, clicked=True),
        ]
        feedbacks2 = [
            RetrievalFeedback(query_id="q2", result_id="r2", relevance=0.8, clicked=True),
        ]

        # Return feedbacks on each batch call, then empty
        mock_feedback_collector.get_training_batch.side_effect = [
            feedbacks1,
            feedbacks2,
            feedbacks1,  # batch 3
            feedbacks2,  # batch 4
            feedbacks1,  # batch 5
            [],  # No more feedback
        ]

        # Loss values: 1.0, 0.9, 0.95, 0.96, 0.97 (no improvement after batch 3)
        mock_adapter.train_step.side_effect = [1.0, 0.9, 0.95, 0.96, 0.97]

        # Cache embeddings
        adapter_trainer.cache_embeddings(
            query_id="q1",
            query_embedding=sample_embeddings["query_1"],
            result_embeddings={"r1": sample_embeddings["result_1"]},
        )
        adapter_trainer.cache_embeddings(
            query_id="q2",
            query_embedding=sample_embeddings["query_1"],
            result_embeddings={"r2": sample_embeddings["result_2"]},
        )

        stats = await adapter_trainer.train_epoch()

        # Should stop early due to patience threshold
        assert stats.stopped_early

    @pytest.mark.asyncio
    async def test_train_epoch_respects_max_batches(self, adapter_trainer, mock_feedback_collector):
        """Test that train_epoch respects max_batches_per_epoch."""
        feedbacks = [
            RetrievalFeedback(query_id=f"q{i}", result_id=f"r{i}", relevance=0.8)
            for i in range(10)
        ]

        # Return 1 feedback per batch call
        mock_feedback_collector.get_training_batch.side_effect = [
            [fb] for fb in feedbacks
        ] + [[]]  # Then no more

        stats = await adapter_trainer.train_epoch(max_batches=3)

        # Should stop at 3 batches even though more feedback available
        assert stats.batches_trained <= 3

    def test_train_on_feedback_empty(self, adapter_trainer):
        """Test train_on_feedback with empty feedback."""
        loss = adapter_trainer.train_on_feedback([], {}, {})
        assert loss == 0.0

    def test_train_on_feedback_sync(self, adapter_trainer, mock_adapter, sample_embeddings):
        """Test synchronous train_on_feedback."""
        feedbacks = [
            RetrievalFeedback(query_id="q1", result_id="r1", relevance=0.8),
            RetrievalFeedback(query_id="q1", result_id="r2", relevance=0.4),
        ]

        mock_adapter.train_step.return_value = 0.5

        loss = adapter_trainer.train_on_feedback(
            feedbacks=feedbacks,
            query_embeddings={"q1": sample_embeddings["query_1"]},
            result_embeddings={
                "r1": sample_embeddings["result_1"],
                "r2": sample_embeddings["result_2"],
            },
        )

        assert loss == 0.5
        assert adapter_trainer._total_batches == 1

    def test_train_on_feedback_missing_embeddings(self, adapter_trainer, mock_adapter, sample_embeddings):
        """Test train_on_feedback skips when embeddings missing."""
        feedbacks = [
            RetrievalFeedback(query_id="q1", result_id="r1", relevance=0.8),
        ]

        # Query missing from embeddings
        loss = adapter_trainer.train_on_feedback(
            feedbacks=feedbacks,
            query_embeddings={},
            result_embeddings=sample_embeddings,
        )

        assert loss == 0.0
        mock_adapter.train_step.assert_not_called()


# ============================================================================
# ContinuousTrainer Tests
# ============================================================================


class TestContinuousTrainer:
    """Test ContinuousTrainer background training."""

    def test_initialization(self, adapter_trainer):
        """Test ContinuousTrainer initialization."""
        trainer = ContinuousTrainer(
            trainer=adapter_trainer,
            interval_seconds=60.0,
            min_feedback_count=20,
        )

        assert trainer.trainer is adapter_trainer
        assert trainer.interval_seconds == 60.0
        assert trainer.min_feedback_count == 20
        assert trainer._running is False
        assert trainer._task is None

    def test_initialization_defaults(self, adapter_trainer):
        """Test ContinuousTrainer with default values."""
        trainer = ContinuousTrainer(trainer=adapter_trainer)

        assert trainer.interval_seconds == 300.0
        assert trainer.min_feedback_count == 10

    @pytest.mark.asyncio
    async def test_is_running_property(self, adapter_trainer):
        """Test is_running property."""
        trainer = ContinuousTrainer(trainer=adapter_trainer)

        assert trainer.is_running is False
        trainer._running = True
        assert trainer.is_running is True

    @pytest.mark.asyncio
    async def test_start_creates_task(self, adapter_trainer):
        """Test start creates background task."""
        trainer = ContinuousTrainer(trainer=adapter_trainer, interval_seconds=0.1)

        assert trainer._task is None
        await trainer.start()

        assert trainer._running is True
        assert trainer._task is not None

        # Cleanup
        await trainer.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, adapter_trainer):
        """Test start is idempotent (calling twice doesn't create new task)."""
        trainer = ContinuousTrainer(trainer=adapter_trainer, interval_seconds=10.0)

        await trainer.start()
        task1 = trainer._task

        await trainer.start()
        task2 = trainer._task

        # Should be the same task
        assert task1 is task2

        await trainer.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, adapter_trainer):
        """Test stop cancels background task."""
        trainer = ContinuousTrainer(trainer=adapter_trainer, interval_seconds=0.1)

        await trainer.start()
        assert trainer._running is True

        await trainer.stop()
        assert trainer._running is False

    @pytest.mark.asyncio
    async def test_stop_handles_cancelled_error(self, adapter_trainer):
        """Test stop handles CancelledError gracefully."""
        trainer = ContinuousTrainer(trainer=adapter_trainer)
        await trainer.start()

        # Should not raise
        await trainer.stop()

    @pytest.mark.asyncio
    async def test_training_loop_respects_interval(self, adapter_trainer, mock_feedback_collector):
        """Test training loop respects interval timing."""
        mock_feedback_collector.get_statistics.return_value = {"persisted_feedbacks": 50}
        mock_feedback_collector.get_training_batch.return_value = []

        trainer = ContinuousTrainer(
            trainer=adapter_trainer,
            interval_seconds=0.1,
            min_feedback_count=5,
        )

        await trainer.start()

        # Wait for at least one iteration
        await asyncio.sleep(0.15)

        await trainer.stop()

        # At least one training should have run
        assert adapter_trainer._epoch_count >= 1

    @pytest.mark.asyncio
    async def test_training_loop_skips_insufficient_feedback(self, adapter_trainer, mock_feedback_collector):
        """Test training loop skips when feedback below threshold."""
        mock_feedback_collector.get_statistics.return_value = {"persisted_feedbacks": 2}

        trainer = ContinuousTrainer(
            trainer=adapter_trainer,
            interval_seconds=0.05,
            min_feedback_count=10,
        )

        await trainer.start()
        await asyncio.sleep(0.1)
        await trainer.stop()

        # Should not have trained
        assert adapter_trainer._epoch_count == 0

    @pytest.mark.asyncio
    async def test_training_loop_handles_exceptions(self, adapter_trainer, mock_feedback_collector):
        """Test training loop continues after exceptions."""
        # Make train_epoch raise exception once, then work
        async def train_epoch_side_effect():
            if adapter_trainer._epoch_count == 0:
                raise ValueError("Test error")
            return TrainingStats()

        mock_feedback_collector.get_statistics.return_value = {"persisted_feedbacks": 50}
        adapter_trainer.train_epoch = AsyncMock(side_effect=train_epoch_side_effect)

        trainer = ContinuousTrainer(
            trainer=adapter_trainer,
            interval_seconds=0.05,
            min_feedback_count=5,
        )

        await trainer.start()
        await asyncio.sleep(0.2)
        await trainer.stop()

        # Loop should have continued despite error
        assert adapter_trainer.train_epoch.call_count >= 1

    @pytest.mark.asyncio
    async def test_get_stats(self, adapter_trainer):
        """Test get_stats returns continuous trainer stats."""
        trainer = ContinuousTrainer(
            trainer=adapter_trainer,
            interval_seconds=60.0,
            min_feedback_count=15,
        )

        stats = trainer.get_stats()

        assert isinstance(stats, dict)
        assert stats["running"] is False
        assert stats["interval_seconds"] == 60.0
        assert stats["min_feedback_count"] == 15
        assert stats["last_train_time"] is None
        assert "trainer_stats" in stats

    @pytest.mark.asyncio
    async def test_get_stats_with_training(self, adapter_trainer, mock_feedback_collector):
        """Test get_stats after training has run."""
        mock_feedback_collector.get_statistics.return_value = {"persisted_feedbacks": 50}
        mock_feedback_collector.get_training_batch.return_value = []

        trainer = ContinuousTrainer(
            trainer=adapter_trainer,
            interval_seconds=0.05,
            min_feedback_count=5,
        )

        await trainer.start()
        await asyncio.sleep(0.15)
        await trainer.stop()

        stats = trainer.get_stats()

        assert stats["running"] is False
        assert stats["last_train_time"] is not None


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_adapter_trainer(self, mock_adapter, mock_feedback_collector, mock_embedder):
        """Test create_adapter_trainer factory."""
        trainer = create_adapter_trainer(
            adapter=mock_adapter,
            feedback=mock_feedback_collector,
            embedder=mock_embedder,
            batch_size=64,
            relevance_threshold=0.7,
        )

        assert isinstance(trainer, AdapterTrainer)
        assert trainer.config.batch_size == 64
        assert trainer.config.relevance_threshold == 0.7

    def test_create_adapter_trainer_kwargs(self, mock_adapter, mock_feedback_collector, mock_embedder):
        """Test create_adapter_trainer with additional kwargs."""
        trainer = create_adapter_trainer(
            adapter=mock_adapter,
            feedback=mock_feedback_collector,
            embedder=mock_embedder,
            batch_size=32,
            relevance_threshold=0.5,
            warmup_batches=10,
            early_stop_patience=20,
        )

        assert trainer.config.warmup_batches == 10
        assert trainer.config.early_stop_patience == 20

    def test_create_continuous_trainer(self, adapter_trainer):
        """Test create_continuous_trainer factory."""
        trainer = create_continuous_trainer(
            trainer=adapter_trainer,
            interval_minutes=10.0,
            min_feedback=25,
        )

        assert isinstance(trainer, ContinuousTrainer)
        assert trainer.interval_seconds == 600.0  # 10 minutes * 60
        assert trainer.min_feedback_count == 25

    def test_create_continuous_trainer_defaults(self, adapter_trainer):
        """Test create_continuous_trainer with defaults."""
        trainer = create_continuous_trainer(trainer=adapter_trainer)

        assert trainer.interval_seconds == 300.0  # 5 minutes default
        assert trainer.min_feedback_count == 10


# ============================================================================
# Integration Tests
# ============================================================================


class TestAdapterTrainerIntegration:
    """Integration tests for full training pipeline."""

    @pytest.mark.asyncio
    async def test_full_training_pipeline(self, mock_adapter, mock_feedback_collector, mock_embedder, training_config, sample_embeddings):
        """Test full training pipeline from setup to stats."""
        # Create trainer
        trainer = AdapterTrainer(
            adapter=mock_adapter,
            feedback=mock_feedback_collector,
            embedder=mock_embedder,
            config=training_config,
        )

        # Cache embeddings
        trainer.cache_embeddings(
            query_id="q1",
            query_embedding=sample_embeddings["query_1"],
            result_embeddings={
                "r1": sample_embeddings["result_1"],
                "r2": sample_embeddings["result_2"],
            },
        )

        # Setup feedback - return once, then empty
        feedbacks = [
            RetrievalFeedback(query_id="q1", result_id="r1", relevance=0.8, clicked=True),
            RetrievalFeedback(query_id="q1", result_id="r2", relevance=0.4, clicked=False),
        ]
        mock_feedback_collector.get_training_batch.side_effect = [feedbacks, []]
        mock_adapter.train_step.return_value = 0.25

        # Train epoch
        stats = await trainer.train_epoch()

        # Verify results
        assert stats.epoch == 1
        assert stats.batches_trained == 1
        assert stats.total_samples == 2
        assert stats.mean_loss == 0.25
        assert stats.training_time_seconds >= 0

        # Check history
        history = trainer.get_training_history()
        assert len(history) == 1

        # Check stats
        stats_dict = trainer.get_stats()
        assert stats_dict["epoch_count"] == 1
        assert stats_dict["total_batches"] == 1

    @pytest.mark.asyncio
    async def test_multiple_training_epochs(self, adapter_trainer, mock_feedback_collector, mock_adapter, sample_embeddings):
        """Test multiple training epochs."""
        # Setup cache
        adapter_trainer.cache_embeddings(
            query_id="q1",
            query_embedding=sample_embeddings["query_1"],
            result_embeddings={"r1": sample_embeddings["result_1"]},
        )

        feedbacks = [
            RetrievalFeedback(query_id="q1", result_id="r1", relevance=0.8),
        ]

        # Train multiple epochs
        for epoch in range(3):
            mock_feedback_collector.get_training_batch.side_effect = [feedbacks, []]
            mock_adapter.train_step.return_value = 0.5 - (epoch * 0.1)

            stats = await adapter_trainer.train_epoch()

            assert stats.epoch == epoch + 1

        # Check history
        history = adapter_trainer.get_training_history()
        assert len(history) == 3

        # Verify epoch numbers
        for idx, h in enumerate(history):
            assert h["epoch"] == idx + 1

    def test_sync_and_async_training(self, adapter_trainer, mock_adapter, sample_embeddings):
        """Test both sync and async training methods."""
        feedbacks = [
            RetrievalFeedback(query_id="q1", result_id="r1", relevance=0.8),
            RetrievalFeedback(query_id="q1", result_id="r2", relevance=0.4),
        ]

        mock_adapter.train_step.return_value = 0.3

        # Sync training
        loss = adapter_trainer.train_on_feedback(
            feedbacks=feedbacks,
            query_embeddings={"q1": sample_embeddings["query_1"]},
            result_embeddings={
                "r1": sample_embeddings["result_1"],
                "r2": sample_embeddings["result_2"],
            },
        )

        assert loss == 0.3
        assert adapter_trainer._total_batches == 1
